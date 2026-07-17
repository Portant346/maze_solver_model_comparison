"""Mapa SEMANTICO de la navegacion: pinta el path del robot segun la señal detectada.

Idea (2026-06-27): a medida que el LIDAR mapea, colorear la trayectoria en funcion de lo que
el clasificador ve en cada instante, con la SATURACION proporcional a la confianza. Da un dato
visual para comparar modelos: donde "ve" cada uno cada cosa y, sobre todo, donde se inventa un
GOAL (el fallo de rnn/transformer con carteles vistos por detras = casi todo blanco).

Solo se colorean las 5 clases de SEÑAL (el resto de frames -percepcion- queda en el gris del
mapa base). Color base por clase (la confianza dessatura hacia blanco):

    CROSS         -> rojo
    GOAL          -> azul
    ARROW_UP      -> verde
    ARROW_RIGHT   -> turquesa
    ARROW_LEFT    -> pistacho

El dato YA esta en los logs: `logs/nav_<ts>.csv` (cls, conf, pos_x, pos_y por frame) y la
rejilla de ocupacion en `logs/lidar_map_<ts>.npz`. Por eso esto es POST-PROCESO: regenera los
mapas de runs ya hechas sin re-ejecutar Gazebo. La misma paleta la usa navigate.py --show-semantic
para el modo en vivo (capa OccupancyMap.mark_semantic).

Uso:
    python Robot/semantic_map.py --logs-dir logs            # todas las runs de nav_comparison.csv
    python Robot/semantic_map.py --csv logs/nav_X.csv --npz logs/lidar_map_Y.npz --out m.png
"""
import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

from wall_lidar import OccupancyMap

# Color BASE por clase, en BGR (formato de cv2). Solo las 5 clases de señal.
CLASS_COLORS = {
    "CROSS":       (0, 0, 255),       # rojo
    "GOAL":        (255, 0, 0),       # azul
    "ARROW_UP":    (0, 200, 0),       # verde
    "ARROW_RIGHT": (208, 224, 64),    # turquesa
    "ARROW_LEFT":  (114, 197, 147),   # pistacho (verde claro amarillento)
}

# Orden canonico de modelos para los mosaicos (rejilla 2x2).
MODEL_ORDER = ["cnn", "rnn", "transformer", "mlp"]


def semantic_color(cls, conf):
    """(clase, confianza) -> color BGR, o None si la clase no es de señal.

    La confianza CRUDA (0..1) controla la saturacion: conf=1 -> color base saturado;
    conf->0 -> tiende a BLANCO (palido). Asi un modelo poco confiado (p. ej. el MLP, que
    vive en ~0.5-0.7) sale visiblemente palido frente a la CNN. Monotono en conf.
    """
    base = CLASS_COLORS.get(cls)
    if base is None:
        return None
    c = float(np.clip(conf, 0.0, 1.0))
    # mezcla lineal hacia blanco: out = base*c + 255*(1-c) (conserva el tono dominante)
    out = tuple(int(round(b * c + 255.0 * (1.0 - c))) for b in base)
    return out


# ----------------------------- emparejado csv <-> npz --------------------------------------

_TS_RE = re.compile(r"(\d{8}_\d{6})")


def _ts_of(path):
    """Extrae el timestamp YYYYMMDD_HHMMSS del nombre como string ordenable, o None."""
    m = _TS_RE.search(Path(path).name)
    return m.group(1) if m else None


def pair_npz_for_csv(csv_path, npz_paths):
    """Empareja un nav_<ts>.csv con el lidar_map_<ts>.npz mas adecuado.

    El csv lleva el ts de INICIO de la run y el npz el de FIN (se guarda al terminar), asi que
    se elige el npz de timestamp INMEDIATAMENTE POSTERIOR; si no hay posterior, el mas cercano.
    """
    cts = _ts_of(csv_path)
    cand = [(p, _ts_of(p)) for p in npz_paths]
    cand = [(p, t) for p, t in cand if t is not None]
    if not cand or cts is None:
        return None
    after = [(t, p) for p, t in cand if t >= cts]
    if after:
        return min(after)[1]               # el posterior mas cercano
    return min(cand, key=lambda pt: abs(int(pt[1].replace("_", "")) - int(cts.replace("_", ""))))[0]


# ----------------------------- render del mapa semantico -----------------------------------

def _read_track(csv_path):
    """Lee (pos_x, pos_y, cls, conf) de las filas con pose y clase de señal validas."""
    pts = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            cls = (row.get("cls") or "").strip()
            if cls not in CLASS_COLORS:
                continue
            px, py, cf = row.get("pos_x"), row.get("pos_y"), row.get("conf")
            if not px or not py:
                continue
            try:
                pts.append((float(px), float(py), cls, float(cf or 0.0)))
            except ValueError:
                continue
    return pts


def _paint_cell(img, r, c, color, thick=1):
    """Pinta un bloque (2*thick+1) alrededor de (r,c) con color BGR, recortado a la imagen."""
    n = img.shape[0]
    r0, r1 = max(0, r - thick), min(n, r + thick + 1)
    c0, c1 = max(0, c - thick), min(n, c + thick + 1)
    img[r0:r1, c0:c1] = color


def _crop_to_content(img, omap, pad=12):
    """Recorta al bounding box de lo observado (libre|pared|traj) + margen, para que se vea."""
    occupied = (omap.free > 0) | (omap.occ > 0) | omap.traj
    rows = np.where(occupied.any(axis=1))[0]
    cols = np.where(occupied.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return img
    r0, r1 = max(0, rows[0] - pad), min(img.shape[0], rows[-1] + pad + 1)
    c0, c1 = max(0, cols[0] - pad), min(img.shape[1], cols[-1] + pad + 1)
    return img[r0:r1, c0:c1]


def render_semantic_map(csv_path, npz_path, thick=1, crop=True):
    """Devuelve una imagen BGR: mapa de ocupacion + trayectoria coloreada por clase/confianza."""
    omap = OccupancyMap.from_npz(npz_path)
    img = omap.render()                       # base: pared negro / libre blanco / desconocido gris
    for px, py, cls, conf in _read_track(csv_path):
        color = semantic_color(cls, conf)
        if color is None:
            continue
        r, c = omap._to_cells(np.array([px]), np.array([py]))
        if r.size:
            _paint_cell(img, int(r[0]), int(c[0]), color, thick=thick)
    if crop:
        img = _crop_to_content(img, omap)
    return img


# ----------------------------- escritura PNG / mosaico -------------------------------------

def write_png(img, path):
    """Guarda BGR como PNG (cv2) o PPM (fallback sin cv2). Devuelve la ruta escrita."""
    if HAS_CV2:
        cv2.imwrite(str(path), img)
        return str(path)
    ppm = str(path).rsplit(".", 1)[0] + ".ppm"
    rgb = img[:, :, ::-1]
    h, w = img.shape[:2]
    with open(ppm, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(np.ascontiguousarray(rgb).tobytes())
    return ppm


# Orden y etiqueta corta de cada color para la leyenda (reusa CLASS_COLORS como fuente de color).
LEGEND_ITEMS = [
    ("CROSS", "cruz"), ("GOAL", "goal"), ("ARROW_UP", "arriba"),
    ("ARROW_RIGHT", "derecha"), ("ARROW_LEFT", "izquierda"),
]


def draw_legend(img):
    """Añade abajo una franja con los 5 colores y su significado + nota de saturacion.

    Devuelve una imagen nueva (más alta). Sin cv2, devuelve la imagen tal cual (no se puede rotular).
    """
    if not HAS_CV2:
        return img
    w = img.shape[1]
    strip_h = 30
    strip = np.full((strip_h, w, 3), 245, dtype=np.uint8)        # fondo claro
    n = len(LEGEND_ITEMS)
    cell = max(1, w // n)
    sw = 16                                                       # lado del swatch
    for i, (cls, name) in enumerate(LEGEND_ITEMS):
        x0 = i * cell + 4
        y0 = (strip_h - sw) // 2
        color = CLASS_COLORS[cls]
        cv2.rectangle(strip, (x0, y0), (x0 + sw, y0 + sw), color, -1)
        cv2.rectangle(strip, (x0, y0), (x0 + sw, y0 + sw), (60, 60, 60), 1)
        cv2.putText(strip, name, (x0 + sw + 4, strip_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    note = np.full((18, w, 3), 230, dtype=np.uint8)
    cv2.putText(note, "saturacion = confianza  (palido = baja confianza)", (5, 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1, cv2.LINE_AA)
    return np.vstack([img, strip, note])


def _label_tile(img, text):
    """Pone una barra de titulo arriba (cv2). Sin cv2, devuelve la imagen tal cual."""
    if not HAS_CV2:
        return img
    bar = np.full((22, img.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(bar, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _pad_to(img, h, w):
    """Centra img en un lienzo (h,w,3) gris oscuro."""
    out = np.full((h, w, 3), 20, dtype=np.uint8)
    ih, iw = img.shape[:2]
    y, x = (h - ih) // 2, (w - iw) // 2
    out[y:y + ih, x:x + iw] = img
    return out


def make_mosaic(tiles_by_model, world):
    """Compone los 4 modelos del mismo mundo en una rejilla 2x2 (cnn/rnn/transformer/mlp)."""
    tiles = []
    for model in MODEL_ORDER:
        img = tiles_by_model.get(model)
        if img is None:
            img = np.full((80, 80, 3), 20, dtype=np.uint8)
        tiles.append(_label_tile(img, f"{model}  w{world}"))
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    tiles = [_pad_to(t, h, w) for t in tiles]
    top = np.hstack([tiles[0], tiles[1]])
    bot = np.hstack([tiles[2], tiles[3]])
    return np.vstack([top, bot])


# ----------------------------- driver sobre los logs ---------------------------------------

def _parse_label(label):
    """'cnn_lidar_w3' -> ('cnn', '3'); tolera otros formatos -> (modelo, None)."""
    m = re.match(r"([a-zA-Z]+).*?_w(\d+)$", label or "")
    if m:
        return m.group(1).lower(), m.group(2)
    return (label or "").lower(), None


def process_logs(logs_dir, comparison_csv=None, thick=1, no_crop=False, legend=True):
    logs_dir = Path(logs_dir)
    comp = Path(comparison_csv) if comparison_csv else logs_dir / "nav_comparison.csv"
    npz_paths = sorted(str(p) for p in logs_dir.glob("lidar_map_*.npz"))
    if not comp.exists():
        print(f"No existe {comp}; nada que procesar.")
        return []
    written = []
    mosaics = {}            # world -> {model: img}
    with open(comp, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        log = (row.get("log") or "").strip()
        label = (row.get("label") or "").strip()
        csv_path = logs_dir / log
        if not csv_path.exists():
            print(f"(salto {label}: no existe {csv_path})")
            continue
        npz = pair_npz_for_csv(str(csv_path), npz_paths)
        if npz is None:
            print(f"(salto {label}: sin .npz emparejable)")
            continue
        img = render_semantic_map(str(csv_path), npz, thick=thick, crop=not no_crop)
        out = logs_dir / f"semantic_map_{label or csv_path.stem}.png"
        # el PNG por run lleva leyenda; el `img` crudo (sin leyenda) va al mosaico (1 leyenda comun)
        written.append(write_png(draw_legend(img) if legend else img, out))
        model, world = _parse_label(label)
        if world is not None:
            mosaics.setdefault(world, {})[model] = img
        print(f"  {label:24s} <- {Path(npz).name}  -> {out.name}")
    for world, tiles in sorted(mosaics.items()):
        mos = make_mosaic(tiles, world)
        if legend:
            mos = draw_legend(mos)
        out = logs_dir / f"semantic_mosaic_w{world}.png"
        written.append(write_png(mos, out))
        print(f"  mosaico mundo {world} -> {out.name}")
    return written


def main():
    ap = argparse.ArgumentParser(description="Mapa semantico de navegacion (path coloreado por señal/confianza)")
    ap.add_argument("--logs-dir", default="logs", help="Carpeta de logs (nav_*.csv + lidar_map_*.npz + nav_comparison.csv)")
    ap.add_argument("--comparison-csv", default=None, help="CSV de comparacion (def: <logs-dir>/nav_comparison.csv)")
    ap.add_argument("--csv", default=None, help="Modo puntual: un nav_<ts>.csv concreto")
    ap.add_argument("--npz", default=None, help="Modo puntual: su lidar_map_<ts>.npz")
    ap.add_argument("--out", default=None, help="Modo puntual: PNG de salida")
    ap.add_argument("--thick", type=int, default=1, help="Grosor del rastro en celdas (radio)")
    ap.add_argument("--no-crop", action="store_true", help="No recortar al area observada")
    ap.add_argument("--no-legend", action="store_true", help="No añadir la leyenda de colores a las imagenes")
    args = ap.parse_args()

    if args.csv and args.npz:
        img = render_semantic_map(args.csv, args.npz, thick=args.thick, crop=not args.no_crop)
        if not args.no_legend:
            img = draw_legend(img)
        out = args.out or (os.path.splitext(args.csv)[0] + "_semantic.png")
        print("Escrito:", write_png(img, out))
        return
    written = process_logs(args.logs_dir, args.comparison_csv, thick=args.thick,
                           no_crop=args.no_crop, legend=not args.no_legend)
    print(f"\n{len(written)} imagen(es) escritas en {args.logs_dir}.")


if __name__ == "__main__":
    main()
