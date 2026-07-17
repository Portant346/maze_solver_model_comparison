"""Clasificador de inferencia para el robot (lado WSL2/Ubuntu).

Pipeline TWO-STAGE (desde nav-v13):
  - Detector GEOMETRICO de verde (green_bbox): si hay verde con area >=
    --arrow-area-thresh, recorta el bbox y lo pasa al CLASIFICADOR ESPECIALIZADO
    de flechas (`cnn_arrows`, 3 clases). El arrow ve la flecha 'en grande'.
  - Si no hay verde suficiente, responde el clasificador GENERAL (`cnn_unificado`,
    10 clases). La cruz es roja y la diana azul, asi que ninguna activa el gate
    de verde -> el main las clasifica como antes.

Asi se ataca el cuello de botella en vivo (ARROW_LEFT/RIGHT con ~0.5 de confianza
por la baja resolucion de la flecha en pasillos): el arrow las ve siempre llenando
el frame. El contrato externo de SignClassifier.predict() no cambia: devuelve
(cls, conf, probs10), con las probs10 reconstruidas a partir de las 3 probs del
arrow cuando responde el. navigate.py no necesita ningun cambio.

El preprocesado replica EXACTAMENTE el de entrenamiento (Resize -> [0,1] ->
normalizacion ImageNet) con cv2/numpy para no depender de torchvision en WSL.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Reutiliza la definicion de la CNN y la mascara verde del lado de Modelos
SRC_DIR = Path(__file__).resolve().parent.parent / "Modelos" / "src"
sys.path.insert(0, str(SRC_DIR))
from models import build_model  # noqa: E402
from vision_utils import green_bbox  # noqa: E402

# Orden de clases del MAIN = orden alfabetico de ImageFolder. Debe coincidir
# con el del entrenamiento de cnn_unificado.
CLASSES = ["ARROW_LEFT", "ARROW_RIGHT", "ARROW_UP", "CROSS", "FREE_PATH",
           "GOAL", "UNKNOWN", "WALL", "WALL_LEFT", "WALL_RIGHT"]

# Clases del ARROW (alfabetico tambien); deben mapearse a sus indices en CLASSES.
ARROW_CLASSES = ["ARROW_LEFT", "ARROW_RIGHT", "ARROW_UP"]
ARROW_TO_MAIN_IDX = [CLASSES.index(c) for c in ARROW_CLASSES]

MAIN_IMG_H, MAIN_IMG_W = 100, 160
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_RUN = Path(__file__).resolve().parent.parent / "Modelos" / "runs" / "cnn_unificado"
DEFAULT_ARROW_RUN = Path(__file__).resolve().parent.parent / "Modelos" / "runs" / "cnn_arrows"

# Gate: area minima del bbox verde (fraccion del frame) para invocar el arrow.
# Por debajo de esto el verde es ruido o muy lejano y se confia en el main.
DEFAULT_ARROW_AREA_THRESH = 0.003   # ~0.3% del frame (= 100/200/160 ~ 35x35 px en cam 320x200)


def center_crop(rgb, frac):
    """Recorta el recuadro central (frac del ancho y del alto). frac>=1 -> sin recorte."""
    if frac >= 1.0:
        return rgb
    h, w = rgb.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return rgb[y0:y0 + ch, x0:x0 + cw]


def central_band(rgb, w_frac):
    """Recorta la FRANJA CENTRAL en ancho (mantiene TODO el alto). w_frac>=1 -> sin recorte.

    A diferencia de center_crop (que recorta ancho Y alto), esto solo quita los
    LATERALES, dejando la franja vertical central. Sirve para que el robot NO vea las
    senales muy a los extremos (izquierda/derecha) del campo de vision, que desviaban la
    trayectoria al detectarse demasiado bien. Se aplica al frame ENTERO antes del gate de
    verde, asi tampoco el specialist de flechas ve una flecha lateral."""
    if w_frac >= 1.0:
        return rgb
    h, w = rgb.shape[:2]
    cw = int(w * w_frac)
    x0 = (w - cw) // 2
    return rgb[:, x0:x0 + cw]


def preprocess(rgb, img_h, img_w):
    """rgb HxWx3 uint8 -> tensor (1,3,img_h,img_w) normalizado ImageNet."""
    img = cv2.resize(rgb, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)


def _load_model(run_dir, num_classes, device):
    """Construye el modelo segun el config.json del run y carga sus pesos.

    Pasa img_h/img_w ademas de base_channels/dropout: la CNN los ignora, pero
    mlp/rnn/transformer dimensionan sus capas con el tamano de entrada. Asi se
    puede usar cualquier familia (no solo la CNN) como clasificador principal.
    """
    cfg = json.loads((run_dir / "config.json").read_text())
    model = build_model(
        cfg.get("model", "cnn"), num_classes=num_classes,
        base_channels=cfg.get("base_channels", 32),
        dropout=cfg.get("dropout", 0.3),
        img_h=cfg.get("img_h", MAIN_IMG_H), img_w=cfg.get("img_w", MAIN_IMG_W),
        # En inferencia SIEMPRE cargamos un state_dict propio, asi que los modelos
        # preentrenados (vgg_tl/resnet_tl) no deben descargar pesos ImageNet al
        # construirse: pretrained=False evita la descarga (y la necesidad de red).
        # El resto de modelos lo ignoran via **kwargs. torchvision sigue siendo
        # necesario en el entorno solo para definir la arquitectura vgg11_bn/resnet18.
        pretrained=False,
    ).to(device)
    state = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


class SignClassifier:
    """Two-stage: arrow specialist con gate por color verde + main para el resto.

    Si arrow_run=None o no existe, cae al modo single-stage (solo main), util
    para A/B testing o para comparar con la version anterior.
    """

    def __init__(self, run_dir=DEFAULT_RUN, arrow_run=DEFAULT_ARROW_RUN,
                 device="cpu", crop_frac=1.0, view_w_frac=1.0,
                 arrow_area_thresh=DEFAULT_ARROW_AREA_THRESH,
                 arrow_bbox_margin=0.15):
        self.device = torch.device(device)
        self.crop_frac = crop_frac
        # Franja central en ancho aplicada al frame ENTERO (gate + main): recorta los
        # laterales para no ver senales a los extremos (ver navigate.py --view-w-frac).
        self.view_w_frac = view_w_frac
        self.arrow_area_thresh = arrow_area_thresh
        self.arrow_bbox_margin = arrow_bbox_margin

        run_dir = Path(run_dir)
        self.main, _ = _load_model(run_dir, num_classes=len(CLASSES), device=self.device)

        # Arrow es opcional: si no esta entrenado, funcionamos en single-stage.
        self.arrow = None
        self.arrow_img_h = self.arrow_img_w = None
        if arrow_run is not None:
            arrow_run = Path(arrow_run)
            if (arrow_run / "best_model.pt").is_file():
                self.arrow, acfg = _load_model(
                    arrow_run, num_classes=len(ARROW_CLASSES), device=self.device,
                )
                self.arrow_img_h = acfg.get("img_h", 96)
                self.arrow_img_w = acfg.get("img_w", 96)

    @torch.no_grad()
    def _predict_main(self, rgb):
        rgb = center_crop(rgb, self.crop_frac)
        x = preprocess(rgb, MAIN_IMG_H, MAIN_IMG_W).to(self.device)
        probs = F.softmax(self.main(x), dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        return CLASSES[idx], float(probs[idx]), probs

    @torch.no_grad()
    def _predict_arrow(self, rgb_crop):
        x = preprocess(rgb_crop, self.arrow_img_h, self.arrow_img_w).to(self.device)
        probs3 = F.softmax(self.arrow(x), dim=1)[0].cpu().numpy()
        # Reconstruir un vector de 10 clases con ceros y las 3 probs en su sitio
        probs10 = np.zeros(len(CLASSES), dtype=np.float32)
        for i, mi in enumerate(ARROW_TO_MAIN_IDX):
            probs10[mi] = probs3[i]
        idx3 = int(probs3.argmax())
        return ARROW_CLASSES[idx3], float(probs3[idx3]), probs10

    @torch.no_grad()
    def predict(self, rgb):
        """Devuelve (clase, confianza, vector_de_probabilidades_10).

        Pipeline: gate por verde -> arrow (3 clases) si hay flecha; main (10) si no.
        Si el arrow no esta cargado, siempre responde el main.
        """
        # Acota la VISION a la franja central en ancho ANTES del gate: ni el gate de
        # verde ni el main veran las senales laterales (a los extremos del frame).
        rgb = central_band(rgb, self.view_w_frac)
        if self.arrow is not None:
            # El bbox se calcula sobre el frame COMPLETO (sin crop_frac), porque
            # crop_frac es un filtro del main pensado para ignorar señales
            # perifericas; el arrow gate es independiente.
            bbox = green_bbox(rgb, margin=self.arrow_bbox_margin)
            if bbox is not None and bbox[4] >= self.arrow_area_thresh:
                x0, y0, x1, y1, _ = bbox
                return self._predict_arrow(rgb[y0:y1, x0:x1])
        return self._predict_main(rgb)


if __name__ == "__main__":
    # Prueba rapida: clasifica una imagen del dataset pasada como argumento.
    #   python3 cnn_infer.py <ruta_imagen>
    clf = SignClassifier()
    stage = "two-stage (main + arrow)" if clf.arrow is not None else "single-stage (main solo)"
    print(f"SignClassifier listo en modo {stage}")
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cls, conf, _ = clf.predict(rgb)
        print(f"{path} -> {cls} ({conf:.3f})")
