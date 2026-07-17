"""Tests OFFLINE del mapa semantico (semantic_map.py + OccupancyMap.from_npz). No necesita
Gazebo ni cv2: trabaja sobre datos sinteticos en archivos temporales. Comprueba la paleta
(color por clase, saturacion por confianza), el round-trip from_npz, el emparejado csv<->npz y
el render de un rastro coloreado. Ejecutar:
    python Robot/test_semantic_map.py
"""
import csv
import tempfile
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wall_lidar import OccupancyMap   # noqa: E402
import semantic_map as sm             # noqa: E402

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        passed += 1; print(f"  ok   {name}")
    else:
        failed += 1; print(f"  FAIL {name}")


print("== T1: paleta semantic_color (clase -> tono, confianza -> saturacion) ==")
check("FREE_PATH no es señal -> None", sm.semantic_color("FREE_PATH", 0.9) is None)
check("WALL no es señal -> None", sm.semantic_color("WALL", 0.5) is None)
check("UNKNOWN no es señal -> None", sm.semantic_color("UNKNOWN", 1.0) is None)
check("GOAL conf=1 -> azul base (255,0,0) BGR", sm.semantic_color("GOAL", 1.0) == (255, 0, 0))
check("CROSS conf=1 -> rojo base (0,0,255) BGR", sm.semantic_color("CROSS", 1.0) == (0, 0, 255))
# tono dominante conservado a cualquier confianza
gB, gG, gR = sm.semantic_color("GOAL", 0.4)
check("GOAL: canal azul (B) sigue siendo el maximo", gB >= gG and gB >= gR)
rB, rG, rR = sm.semantic_color("CROSS", 0.4)
check("CROSS: canal rojo (R) sigue siendo el maximo", rR >= rB and rR >= rG)
uB, uG, uR = sm.semantic_color("ARROW_UP", 0.4)
check("ARROW_UP: canal verde (G) sigue siendo el maximo", uG >= uB and uG >= uR)
# monotonia: mas confianza -> mas saturado (mas lejos del blanco)
def dist_white(c):
    return sum((255 - v) ** 2 for v in c)
check("GOAL: conf alta mas saturada que conf baja",
      dist_white(sm.semantic_color("GOAL", 0.9)) > dist_white(sm.semantic_color("GOAL", 0.5)))
check("conf=0 -> blanco (sin saturacion)", sm.semantic_color("GOAL", 0.0) == (255, 255, 255))
check("las 5 clases de señal tienen color", all(sm.semantic_color(k, 1.0) is not None for k in
      ("CROSS", "GOAL", "ARROW_UP", "ARROW_RIGHT", "ARROW_LEFT")))

print("== T2: OccupancyMap.from_npz round-trip ==")
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    m0 = OccupancyMap(res=0.1, size_m=4.0)
    m0.origin = (1.5, -2.0)
    m0.traj[m0.n // 2, m0.n // 2] = True
    m0.free[10, 12] = 3
    m0.occ[5, 6] = 2
    npz = d / "lidar_map_20260101_120000.npz"
    m0.save(d / "m.png", npz)
    m1 = OccupancyMap.from_npz(npz)
    check("res se preserva", abs(m1.res - m0.res) < 1e-9)
    check("n (lado de la rejilla) coincide", m1.n == m0.n)
    check("origin se preserva", m1.origin == m0.origin)
    check("matriz free se preserva", int(m1.free[10, 12]) == 3)
    check("matriz occ se preserva", int(m1.occ[5, 6]) == 2)
    # _to_cells coincide tras el round-trip (misma geometria de celdas)
    r0, c0 = m0._to_cells(np.array([1.5]), np.array([-2.0]))
    r1, c1 = m1._to_cells(np.array([1.5]), np.array([-2.0]))
    check("_to_cells igual tras from_npz", r0.tolist() == r1.tolist() and c0.tolist() == c1.tolist())

print("== T3: emparejado csv <-> npz (elige el .npz posterior mas cercano) ==")
csvp = "logs/nav_20260101_120000.csv"
npzs = ["logs/lidar_map_20260101_115900.npz",   # anterior
        "logs/lidar_map_20260101_120030.npz",   # posterior cercano (correcto)
        "logs/lidar_map_20260101_130000.npz"]   # posterior lejano
check("empareja con el posterior mas cercano",
      sm.pair_npz_for_csv(csvp, npzs) == "logs/lidar_map_20260101_120030.npz")
check("sin posterior -> el mas cercano (anterior)",
      sm.pair_npz_for_csv("logs/nav_20260101_120000.csv",
                          ["logs/lidar_map_20260101_115959.npz"]) == "logs/lidar_map_20260101_115959.npz")
check("sin npz -> None", sm.pair_npz_for_csv(csvp, []) is None)
check("_parse_label extrae modelo y mundo", sm._parse_label("cnn_lidar_w3") == ("cnn", "3"))

print("== T4: render de un rastro coloreado sobre el mapa base ==")
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    m = OccupancyMap(res=0.1, size_m=4.0)
    m.origin = (0.0, 0.0)
    npz = d / "lidar_map_20260101_120000.npz"
    m.save(d / "m.png", npz)
    cx = m.n // 2                       # (0,0) cae en la celda central
    csvf = d / "nav_20260101_120000.csv"
    header = ["t", "frame", "cls", "conf", "state", "lin", "ang", "yaw_deg",
              "pos_x", "pos_y", "arrow_ang", "open_side", "dark_l", "dark_c", "dark_r",
              "goal_off", "goal_area", "note"]
    with open(csvf, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        # GOAL con conf alta en el centro (0,0); un FREE_PATH (no debe pintar) desplazado
        w.writerow(["0.1", 1, "GOAL", "1.000", "FOLLOW", "0", "0", "0",
                    "0.000", "0.000", "", "", "0", "0", "0", "", "0.0", ""])
        w.writerow(["0.2", 2, "FREE_PATH", "0.900", "FOLLOW", "0", "0", "0",
                    "0.300", "0.000", "", "", "0", "0", "0", "", "0.0", ""])
    img = sm.render_semantic_map(str(csvf), str(npz), thick=0, crop=False)
    px = tuple(int(v) for v in img[cx, cx])
    check("celda central (GOAL conf=1) pintada de azul (255,0,0)", px == (255, 0, 0))
    # la celda del FREE_PATH (a +0.3 m en x -> +3 celdas en columna) NO se pinta de color de señal
    fp = tuple(int(v) for v in img[cx, cx + 3])
    check("celda de FREE_PATH no coloreada (queda gris del mapa)", fp != (255, 0, 0))

print("== T5: leyenda (draw_legend) ==")
if sm.HAS_CV2:
    base = np.full((40, 400, 3), 245, dtype=np.uint8)
    leg = sm.draw_legend(base)
    check("la leyenda hace la imagen MAS ALTA", leg.shape[0] > base.shape[0])
    check("la leyenda conserva el ancho", leg.shape[1] == base.shape[1])
    strip = leg[base.shape[0]:]                      # la franja añadida abajo
    def has_color(img, bgr):
        return bool(np.any(np.all(img == np.array(bgr, dtype=np.uint8), axis=-1)))
    check("la franja contiene los 5 colores de la paleta",
          all(has_color(strip, sm.CLASS_COLORS[k]) for k, _ in sm.LEGEND_ITEMS))
else:
    print("  (cv2 no disponible: draw_legend devuelve la imagen tal cual)")
    check("sin cv2, draw_legend no rompe", sm.draw_legend(np.zeros((4, 4, 3), np.uint8)).shape[0] == 4)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
