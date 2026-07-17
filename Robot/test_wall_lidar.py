"""Tests OFFLINE del detector de pared por LIDAR (nav-v26). No necesita Gazebo: trabaja sobre
escaneos SINTETICOS (corredores/cruces) construidos a mano. Comprueba el contrato comun
WallReading (open_side/center_blocked/sides), el ENCARRILADO al pasillo (corridor_correction) y
el mapa de ocupacion (OccupancyMap). Ejecutar:
    python Robot/test_wall_lidar.py
"""
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wall_camera import CameraWallDetector, WallReading   # noqa: E402
from wall_lidar import LidarWallDetector, OccupancyMap, make_scan   # noqa: E402

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        passed += 1; print(f"  ok   {name}")
    else:
        failed += 1; print(f"  FAIL {name}")


def build_scan(d_left=None, d_right=None, d_front=None, d_back=None,
               yaw=0.0, n=360, range_max=15.0):
    """Escaneo de un corredor: paredes a distancia PERPENDICULAR d_* (m) del robot, que esta
    girado `yaw` rad respecto al eje del pasillo. Un beam a angulo-robot a apunta en el mundo a
    yaw+a; la distancia a una pared recta es d/sin o d/cos segun el lado. Sin pared -> range_max."""
    alpha = np.linspace(-math.pi, math.pi, n, endpoint=False)
    beta = yaw + alpha
    s, c = np.sin(beta), np.cos(beta)
    r = np.full(n, np.inf)
    eps = 1e-3

    def add(mask, dist, denom):
        cand = np.full(n, np.inf)
        cand[mask] = dist / denom[mask]
        cand[cand <= 0] = np.inf
        return np.minimum(r, cand)

    if d_left is not None:                       # pared izquierda: mundo +y, beams con sin>0
        r = add(s > eps, d_left, s)
    if d_right is not None:                      # pared derecha: mundo -y, beams con sin<0
        r = add(s < -eps, -d_right, s)
    if d_front is not None:                      # pared de frente: mundo +x, beams con cos>0
        r = add(c > eps, d_front, c)
    if d_back is not None:                        # pared trasera: mundo -x, beams con cos<0
        r = add(c < -eps, -d_back, c)
    r = np.where(np.isfinite(r) & (r <= range_max), r, range_max)
    return make_scan(r, -math.pi, 2 * math.pi / n, range_max=range_max)


det = LidarWallDetector()   # parametros por defecto (block 1.5, side_margin 0.3, wall_present 2.5)

print("== T1: contrato WallReading y normalizacion de angulos ==")
sc = build_scan(d_left=1.0, d_right=1.0)
check("make_scan normaliza angulos a [-pi, pi]", float(np.abs(sc.angles).max()) <= math.pi + 1e-6)
rd = det.read(sc)
check("read() devuelve un WallReading", isinstance(rd, WallReading))
check("pasillo recto sin frente -> NO center_blocked", rd.center_blocked is False)
check("pasillo recto -> open_side None", rd.open_side is None)
check("sides = (izq, frente, der) en metros (frente ~ libre)", rd.sides[1] >= det.block_dist)

print("== T2: pared de frente -> open_side al lado MAS DESPEJADO ==")
# frente a 1.0 m (<1.5 bloqueo), izquierda ABIERTA (sin pared), derecha a 0.8 m -> abrir IZQUIERDA
sc = build_scan(d_front=1.0, d_right=0.8)
rd = det.read(sc)
check("frente cercano -> center_blocked True", rd.center_blocked is True)
check("mas hueco a la izq -> open_side 'left'", rd.open_side == "left")
# espejo: pared a la izquierda, derecha abierta -> abrir DERECHA
rd2 = det.read(build_scan(d_front=1.0, d_left=0.8))
check("espejo: mas hueco a la der -> open_side 'right'", rd2.open_side == "right")
# lados parecidos -> no decide lado (None) aunque haya frente
rd3 = det.read(build_scan(d_front=1.0, d_left=1.0, d_right=1.0))
check("lados parecidos -> open_side None (bajo el margen)", rd3.open_side is None)

print("== T3: encarrilado LATERAL (centrar en el pasillo) ==")
corr_center = det.corridor_correction(build_scan(d_left=1.0, d_right=1.0))
check("centrado y paralelo -> correccion ~0", abs(corr_center) < 0.05)
corr_right = det.corridor_correction(build_scan(d_left=1.5, d_right=0.5))   # pegado a la DERECHA
check("pegado a la derecha -> corrige a la IZQUIERDA (corr>0)", corr_right > 0.05)
corr_left = det.corridor_correction(build_scan(d_left=0.5, d_right=1.5))    # pegado a la IZQUIERDA
check("pegado a la izquierda -> corrige a la DERECHA (corr<0)", corr_left < -0.05)
check("correccion saturada a +-align_max", abs(corr_right) <= det.align_max + 1e-9)

print("== T4: encarrilado de ORIENTACION (quedar paralelo) ==")
corr_ccw = det.corridor_correction(build_scan(d_left=1.0, d_right=1.0, yaw=math.radians(10)))
check("robot girado a IZQ (yaw+10) -> corrige a la DERECHA (corr<0)", corr_ccw < -0.02)
corr_cw = det.corridor_correction(build_scan(d_left=1.0, d_right=1.0, yaw=math.radians(-10)))
check("robot girado a DER (yaw-10) -> corrige a la IZQUIERDA (corr>0)", corr_cw > 0.02)

print("== T5: en un CRUCE no se encarrila (sigue recto) ==")
# solo pared derecha (izquierda abierta = boca de cruce): sin las DOS paredes, sin termino lateral
corr_cross = det.corridor_correction(build_scan(d_right=1.0))   # robot paralelo a la pared der
check("una sola pared (cruce) y paralelo -> correccion ~0 (recto)", abs(corr_cross) < 0.05)
corr_open = det.corridor_correction(build_scan())               # area abierta, sin paredes
check("sin paredes -> correccion 0", corr_open == 0.0)

print("== T6: align_gain=0 desactiva el encarrilado (solo deteccion) ==")
det0 = LidarWallDetector(align_gain=0.0)
check("align_gain 0 -> correccion 0", det0.corridor_correction(build_scan(d_left=1.5, d_right=0.5)) == 0.0)

print("== T7: simetria de contrato camara/LIDAR ==")
cam = CameraWallDetector()
frame = np.zeros((20, 30, 3), dtype=np.uint8)   # todo negro: pared de frente
cam_rd = cam.read(frame)
check("camara tambien devuelve WallReading", isinstance(cam_rd, WallReading))
check("ambas fuentes exponen los mismos campos",
      cam_rd._fields == det.read(build_scan(d_front=1.0, d_right=0.8))._fields)

print("== T8: OccupancyMap (impactos en marco de odometria) ==")
omap = OccupancyMap(res=0.1, size_m=20.0)
# robot en (0,0) yaw 0, pared SOLO de frente a 2.0 m -> impacto esperado en mundo (2,0)
omap.update(build_scan(d_front=2.0), (0.0, 0.0), 0.0)
cx = omap.n // 2
col_front = cx + int(round(2.0 / omap.res))
check("hay impactos registrados", int(omap.occ.sum()) > 0)
check("impacto de la pared frontal en la celda (2 m al frente)",
      omap.occ[cx - 1:cx + 2, col_front - 1:col_front + 2].sum() > 0)
check("hay espacio LIBRE marcado a lo largo del rayo", int(omap.free.sum()) > 0)
check("la trayectoria incluye la celda del robot", bool(omap.traj[cx, cx]))
# sin pose -> no rompe ni cuenta
omap.update(build_scan(d_front=2.0), (None, None), None)
check("update sin odometria no añade actualizaciones", omap.updates == 1)

print("== T8b: integrate_walls=False (gate de giro) marca traj pero NO acumula paredes ==")
gmap = OccupancyMap(res=0.1, size_m=20.0)
gcx = gmap.n // 2
# girando: solo trayectoria, sin integrar paredes
gmap.update(build_scan(d_front=2.0), (0.0, 0.0), 0.0, integrate_walls=False)
check("girando: la trayectoria SI se marca", bool(gmap.traj[gcx, gcx]))
check("girando: NO se acumulan impactos de pared", int(gmap.occ.sum()) == 0)
check("girando: NO se acumula espacio libre", int(gmap.free.sum()) == 0)
check("girando: update cuenta igual (no se salta del todo)", gmap.updates == 1)
# recto (default True): ahora si entran las paredes
gmap.update(build_scan(d_front=2.0), (0.0, 0.0), 0.0)
check("recto (integrate_walls=True por defecto): SI hay impactos", int(gmap.occ.sum()) > 0)
# anti-smear: un escaneo a OTRO rumbo con integrate_walls=False NO añade paredes nuevas
occ_before = int(gmap.occ.sum())
gmap.update(build_scan(d_front=2.0, yaw=1.2), (0.0, 0.0), 1.2, integrate_walls=False)
check("escaneo girado con integrate_walls=False -> NO duplica paredes (mapa fijo)",
      int(gmap.occ.sum()) == occ_before)

print("== T8c: render exige wall_min_hits impactos para pintar pared ==")
rmap = OccupancyMap(res=0.1, size_m=20.0, wall_min_hits=2)
rmap.origin = (0.0, 0.0)
DARK = np.array([40, 40, 40])
rmap.occ[10, 10] = 1                                  # celda A: 1 impacto (< wall_min_hits)
rmap.occ[10, 14] = 2                                  # celda B: 2 impactos (>= wall_min_hits)
img = rmap.render()
check("1 impacto (< wall_min_hits) -> la celda NO se pinta de pared",
      not bool((img[10, 10] == DARK).all()))
check("2 impactos (>= wall_min_hits) -> la celda SI se pinta de pared",
      bool((img[10, 14] == DARK).all()))

print("== T9: OccupancyMap.save genera ficheros ==")
out = Path(__file__).resolve().parent / "_test_lidar_map"
written = omap.save(out.with_suffix(".png"), out.with_suffix(".npz"))
check("save escribe al menos 2 ficheros (npz + imagen)", len(written) >= 2)
check("todos los ficheros existen", all(Path(w).is_file() for w in written))
for w in written:                               # limpieza
    try:
        Path(w).unlink()
    except OSError:
        pass

print(f"\n{passed} ok, {failed} FAIL")
sys.exit(1 if failed else 0)
