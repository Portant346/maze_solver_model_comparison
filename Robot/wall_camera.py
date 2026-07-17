"""Detector de PARED por CAMARA (pixeles negros) -- modulo intercambiable.

Es el detector GEOMETRICO historico del TFG (nav-v6..v25): una pared cercana sale
casi NEGRA en la camara, asi que se cuenta la fraccion de pixeles oscuros por franjas
verticales (izq/centro/der). Aqui vive la logica que antes estaba suelta en navigate.py
(`wall_open_side`, `make_wall_debug`); navigate.py la importa de aqui sin cambios de
comportamiento. Es el BASELINE validado, conservado como ablacion frente al LIDAR
(`--wall-source camera`).

Define ademas el CONTRATO COMUN `WallReading` que consumen tanto este detector como
el de LIDAR (`wall_lidar.py`), de modo que `Policy.decide()` no distingue la fuente:
    open_side    : 'left'|'right'|None  -> lado mas despejado de una pared de frente
    center_blocked: bool                -> hay pared de frente
    sides        : (l, c, r)            -> magnitud por franja (negro en camara; el log la guarda)
    center_dark  : float|None           -> proxy de bloqueo central (negro central en camara)
"""
from collections import namedtuple

import numpy as np
try:                                  # cv2 opcional: solo para la ventana de debug (--show)
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False


# Contrato comun camara/LIDAR. `sides` y `center_dark` cambian de UNIDAD segun la fuente
# (fraccion de negro en camara; metros/proxy en LIDAR) pero el SENTIDO que usa la politica
# es el mismo: open_side y center_blocked. El resto es informativo (log/debug).
WallReading = namedtuple("WallReading", ["open_side", "center_blocked", "sides", "center_dark"])


def wall_open_side(rgb, dark_thresh=50.0, center_block=0.5, side_margin=0.05, lower_frac=1.0):
    """Detector GEOMETRICO de direccion de pared (al margen del clasificador).

    Divide el frame en 3 franjas verticales (izq/centro/der) y mide la fraccion
    de pixeles OSCUROS en cada una (una pared cercana sale casi NEGRA en la
    camara; el suelo/lejania es gris). Si el centro esta bloqueado (fraccion
    oscura > center_block) hay pared de frente: devuelve entonces el lado MAS
    DESPEJADO (el de menos negro) para girar hacia el.

    lower_frac (0..1, nav-v21): fraccion INFERIOR del alto que se analiza (1.0 = todo
    el alto). Mirar solo la parte BAJA mide el espacio NAVEGABLE CERCANO al robot (suelo
    vs pared inmediata a ras de suelo) e IGNORA la parte alta (paredes lejanas, fondo de
    los pasillos perpendiculares, techo) que en los cruces ensuciaba la lectura: una pared
    al frente repartia negro por las 3 franjas (|dl-dr|~0 -> "no se") o una pared al fondo
    del corredor perpendicular se leia como bloqueo, y el robot no encaraba el pasillo.

    Devuelve (lado|None, (dl, dc, dr)). lado=None si el centro no esta bloqueado
    o si ambos lados estan parecidos (|dl-dr| < side_margin). Prioridad inferior
    a las señales: solo se usa para elegir el giro de una pared SIN señal.
    """
    h, w = rgb.shape[:2]
    y0 = int(h * (1.0 - lower_frac)) if lower_frac < 1.0 else 0   # filas a analizar: [y0, h)
    gray = rgb[y0:].mean(axis=2)
    dark = gray < dark_thresh
    t = w // 3
    dl = float(dark[:, :t].mean())
    dc = float(dark[:, t:2 * t].mean())
    dr = float(dark[:, 2 * t:].mean())
    if dc < center_block or abs(dl - dr) < side_margin:
        return None, (dl, dc, dr)
    return ("left" if dl < dr else "right"), (dl, dc, dr)


def make_wall_debug(rgb, dark_thresh, lower_frac, view_w_frac, dark, open_side,
                    center_blocked, cls, conf, state, scale=3):
    """Render BGR para la ventana de debug (--show): SOLO lo que usa el DETECTOR DE PARED.

    Muestra UNICAMENTE la franja INFERIOR analizada (recortada, sin la parte alta que el
    detector ignora): tinta de ROJO los pixeles "negros" (gray < dark_thresh) = lo que cuenta
    como pared cercana, y dibuja las 2 lineas que separan las 3 franjas izq/centro/der.
    Texto: L/C/R de negro, lado abierto, bloqueo de frente, clase/conf y estado.
    """
    h, w = rgb.shape[:2]
    y0 = int(h * (1.0 - lower_frac)) if lower_frac < 1.0 else 0
    band = rgb[y0:]                              # SOLO la franja inferior (lo que usa el detector)
    bgr = cv2.cvtColor(band, cv2.COLOR_RGB2BGR)
    mask = band.mean(axis=2) < dark_thresh       # pixeles que cuentan como pared
    over = bgr.copy()
    over[mask] = (0, 0, 255)
    bgr = cv2.addWeighted(over, 0.45, bgr, 0.55, 0)
    bh = bgr.shape[0]
    t = w // 3
    cv2.line(bgr, (t, 0), (t, bh), (0, 255, 255), 1)        # franjas izq|centro
    cv2.line(bgr, (2 * t, 0), (2 * t, bh), (0, 255, 255), 1)  # centro|der
    bgr = cv2.resize(bgr, (w * scale, bh * scale), interpolation=cv2.INTER_NEAREST)
    dl, dc, dr = dark if dark is not None else (0.0, 0.0, 0.0)
    txt = [f"negro  L={dl:.2f}  C={dc:.2f}  R={dr:.2f}",
           f"lado abierto={open_side}   pared de frente={center_blocked}",
           f"{cls} {conf:.2f}  [{state}]"]
    y = 20
    for ln in txt:                               # texto con borde negro para que se lea siempre
        cv2.putText(bgr, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(bgr, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return bgr


class CameraWallDetector:
    """Adaptador del detector de camara al contrato comun `WallReading`.

    Envuelve `wall_open_side` para que navigate.py pueda intercambiarlo con
    `LidarWallDetector` sin tocar la logica de la politica. Es 100% el baseline:
    misma lectura de pixeles negros, mismos parametros.
    """
    source = "camera"
    provides_alignment = False     # la camara NO encarrila (centrado geometrico danino, nav-v20)
    provides_map = False

    def __init__(self, dark_thresh=50.0, center_block=0.5, side_margin=0.05,
                 lower_frac=1.0):
        self.dark_thresh = dark_thresh
        self.center_block = center_block
        self.side_margin = side_margin
        self.lower_frac = lower_frac

    def read(self, rgb):
        """Devuelve un WallReading a partir del frame RGB de la camara."""
        open_side, dark = wall_open_side(
            rgb, self.dark_thresh, self.center_block, self.side_margin, self.lower_frac)
        center_blocked = dark[1] >= self.center_block
        return WallReading(open_side=open_side, center_blocked=center_blocked,
                           sides=dark, center_dark=dark[1])
