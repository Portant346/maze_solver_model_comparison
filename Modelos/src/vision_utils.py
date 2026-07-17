"""Utilidades de vision compartidas entre training, inferencia y navegacion.

Una sola fuente de verdad para 'que es verde' / 'que es azul' en el proyecto.
La mascara verde aisla las 3 clases de flecha (ARROW_LEFT/RIGHT/UP); la cruz
es roja y la diana es azul, asi que el verde sirve de GATE perfecto para
activar el clasificador de flechas en el pipeline two-stage.
"""
import numpy as np


# Mascara verde: misma heuristica que arrow_angle() en navigate.py (un solo
# contrato para que entrenamiento e inferencia 'vean' el mismo verde).
GREEN_MIN = 80         # canal G minimo
GREEN_MARGIN = 20      # G - R y G - B minimos (para descartar gris/blanco)
GREEN_MIN_PX = 30      # pixeles minimos para considerar que hay verde


def green_mask(rgb):
    """Mascara booleana HxW: True donde el pixel es verde."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    return (g > GREEN_MIN) & (g - r > GREEN_MARGIN) & (g - b > GREEN_MARGIN)


def green_bbox(rgb, margin=0.15, min_px=GREEN_MIN_PX, square=True):
    """Bounding box del verde en `rgb`, con margen relativo opcional.

    Devuelve (x0, y0, x1, y1, area_frac) en coordenadas de pixel del frame
    original, o None si no hay verde suficiente (< min_px pixeles).

    - margin: fraccion del lado del bbox a anadir como margen (15% por lado
      por defecto). Sirve para no cortar la punta de la flecha y conservar
      algo de contexto. El bbox se recorta a los limites de la imagen.
    - square: si True, expande el bbox al lado mayor para que sea cuadrado
      (evita distorsion al hacer resize al tamano de entrada de la CNN).
    - area_frac: fraccion del frame original que ocupa el bbox final. Se usa
      en inferencia como gate (si es muy bajo, no merece la pena recortar).
    """
    h, w = rgb.shape[:2]
    mask = green_mask(rgb)
    ys, xs = np.nonzero(mask)
    if xs.size < min_px:
        return None

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    bw, bh = x1 - x0, y1 - y0
    mx = int(round(bw * margin))
    my = int(round(bh * margin))
    x0, x1 = x0 - mx, x1 + mx
    y0, y1 = y0 - my, y1 + my

    if square:
        # Expandir el lado menor al tamano del mayor (centrado en el bbox).
        side = max(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        x0, x1 = cx - side // 2, cx - side // 2 + side
        y0, y1 = cy - side // 2, cy - side // 2 + side

    # Recortar a los limites del frame.
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None

    area_frac = ((x1 - x0) * (y1 - y0)) / float(h * w)
    return x0, y0, x1, y1, area_frac
