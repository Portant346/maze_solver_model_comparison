"""Tests OFFLINE de nav-v25: rumbo anclado a la rejilla de 8 direcciones (45 grados) con el yaw
absoluto de la odometria. No necesita Gazebo (stubea gz/cnn_infer). Ejecutar:
    python Robot/test_nav_grid.py
"""
import sys
import math
import types
from pathlib import Path


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m


_stub("gz")
_stub("gz.transport13", Node=object)
_stub("gz.msgs10")
_stub("gz.msgs10.image_pb2", Image=object)
_stub("gz.msgs10.twist_pb2", Twist=object)
_stub("gz.msgs10.odometry_pb2", Odometry=object)
_stub("cnn_infer", SignClassifier=object, DEFAULT_ARROW_RUN="", DEFAULT_RUN="")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import navigate  # noqa: E402

Policy = navigate.Policy
R = math.radians
D = math.degrees

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        passed += 1; print(f"  ok   {name}")
    else:
        failed += 1; print(f"  FAIL {name}")


def mk(anchor=0.0, **kw):
    defaults = dict(forward=0.3, turn=0.6, steer=0.3, arrow_conf_thresh=0.95,
                    sign_conf_thresh=0.55, pending_lost_frames=3,
                    heading_grid=True, grid_deg=45.0, heading_tol_deg=6.0,
                    heading_hold_gain=1.0, heading_hold_deadzone_deg=3.0,
                    heading_recalib_gain=0.2, heading_recalib_patience=3,
                    heading_recalib_max_deg=20.0, turn_ramp_deg=25.0)
    defaults.update(kw)
    p = Policy(**defaults)
    p.heading_anchor = anchor   # fija la rejilla en 0/45/90... (evita captura del 1er yaw)
    return p

TURN = Policy.TURN

print("== T1: snap / residuo a la rejilla de 45 ==")
p = mk()
check("snap 50° -> 45°", abs(D(p._grid_snap(R(50))) - 45) < 1e-6)
check("snap 95° -> 90°", abs(D(p._grid_snap(R(95))) - 90) < 1e-6)
check("snap 20° -> 0°", abs(D(p._grid_snap(R(20)))) < 1e-6)
check("snap 350° -> 0° (wrap)", abs(D(p._grid_snap(R(350)))) < 1e-6)
check("residuo 50° -> -5°", abs(D(p._grid_residual(R(50))) - (-5)) < 1e-6)
p2 = mk(anchor=R(10))   # rejilla anclada a 10° -> 10/55/100...
check("anclada a 10°: snap 52° -> 55°", abs(D(p2._grid_snap(R(52))) - 55) < 1e-6)

print("== T2: giro a rumbo ABSOLUTO corrige la deriva (no acumula) ==")
# robot desviado 5° (a 95° en vez de 90°); flecha IZQ -> debe aterrizar en 180° (rejilla), no en 185°.
# (recalib desactivada aqui para aislar el snapping; en sim el heading-hold mantiene el yaw en rejilla)
p = mk(heading_recalib_patience=10**9)
p.decide("ARROW_LEFT", 0.99, 100.0, yaw=R(95), arrow_ang=-80.0)   # commit pending, _last_yaw=95
p.decide("FREE_PATH", 0.9, 100.1, yaw=R(95)); p.decide("FREE_PATH", 0.9, 100.2, yaw=R(95))
lin, ang, _, note = p.decide("FREE_PATH", 0.9, 100.3, yaw=R(95))  # 3er frame perdido -> dispara giro
check("giro de flecha disparado (state TURN)", p.state == TURN)
check("rumbo objetivo snappeado a 180° (no 185°)", abs(D(p.turn_target_yaw) - 180) < 1)
check("gira a la IZQUIERDA (ang>0) hacia el objetivo", ang > 0)
# alimentar yaw subiendo hacia 180 -> completa al llegar (no a 185)
p.decide("FREE_PATH", 0.9, 100.4, yaw=R(150))
_, _, _, _ = p.decide("FREE_PATH", 0.9, 100.5, yaw=R(177))   # err=3° < tol 6° -> completa
check("completa al alcanzar ~180° (vuelve a FOLLOW)", p.state == Policy.FOLLOW)

print("== T3: CRUZ = 180 a rumbo absoluto ==")
p = mk(heading_recalib_patience=10**9)
p.decide("FREE_PATH", 0.9, 100.0, yaw=R(90))     # _last_yaw=90
_, ang, _, _ = p.decide("CROSS", 0.90, 100.1, yaw=R(90))
check("CRUZ -> objetivo a ~180° del rumbo actual", abs(abs(D(navigate.ang_diff(p.turn_target_yaw, R(90)))) - 180) < 1.0)
check("CRUZ -> arranca girando (state TURN, ang!=0)", p.state == TURN and ang != 0.0)

print("== T4: ANTI-ACUMULACION (el snap absorbe el sobre-giro de cada giro) ==")
# Simula N giros de 90° con un OVERSHOOT fijo de 4° por giro (delay/RTF). El objetivo de cada giro
# se calcula snappeando (yaw_actual_con_overshoot + 90) -> debe caer SIEMPRE en la rejilla (k*45).
p = mk()
actual = 0.0
ok_grid = True
for i in range(12):
    target = p._grid_snap(actual + R(90))
    # ¿el objetivo está sobre la rejilla (múltiplo de 45° desde el ancla 0)?
    res_target = D(p._grid_residual(target))
    if abs(res_target) > 0.5:
        ok_grid = False
    actual = target + R(4.0)        # el robot sobre-gira 4° (plant)
check("12 giros con overshoot 4°: cada objetivo cae EXACTO en la rejilla (no deriva)", ok_grid)
check("rumbo final snappeado sigue en la rejilla (residuo pequeño)", abs(D(p._grid_residual(actual))) <= 4.001)

print("== T5: heading-hold (mantener el rumbo canonico en recto) ==")
p = mk()
check("residuo dentro de zona muerta (2°) -> 0", p._heading_hold(R(2)) == 0.0)
hh = p._heading_hold(R(8))     # yaw 8°, rejilla 0 -> debe girar NEGATIVO (bajar yaw a 0)
check("yaw 8° -> corrige hacia 0 (ang<0)", hh < 0 and abs(hh) <= p.steer + 1e-9)
hh2 = p._heading_hold(R(-8))   # yaw -8° -> girar POSITIVO (subir a 0)
check("yaw -8° -> corrige hacia 0 (ang>0)", hh2 > 0)
check("sin odom (yaw None) -> 0", p._heading_hold(None) == 0.0)
check("--no-heading-grid -> 0", mk(heading_grid=False)._heading_hold(R(8)) == 0.0)

print("== T6: auto-recalibracion del ancla ==")
p = mk()   # patience=3, gain=0.2; ancla 0, robot recto consistente a 4°
for i in range(6):
    p._maybe_recalibrate(R(4), "FREE_PATH", 0.9)
check("tras frames rectos a 4°, el ancla se acerca al pasillo (anchor>0)", D(p.heading_anchor) > 0)
check("el ancla no se pasa del rumbo real (<=4°)", 0 < D(p.heading_anchor) <= 4.001)
# residuo en la frontera (~22°, > max 20°) -> NO recalibra (ambiguo entre dos rumbos de rejilla)
p = mk()
for i in range(6):
    p._maybe_recalibrate(R(22), "FREE_PATH", 0.9)   # snap(22°)=0 -> residuo -22° (>max 20°)
check("residuo ~22° (>max, ambiguo) -> NO recalibra (ancla intacta)", abs(D(p.heading_anchor)) < 1e-9)
# clase no-recta resetea el contador
p = mk(); p._maybe_recalibrate(R(4), "FREE_PATH", 0.9); p._maybe_recalibrate(R(4), "WALL", 0.9)
check("clase no-recta resetea straight_count", p.straight_count == 0)

print("== T7: fallback sin rejilla/odom = comportamiento nav-v24 (giro relativo) ==")
p = mk(heading_grid=False)
p.decide("ARROW_RIGHT", 0.99, 100.0, yaw=R(90), arrow_ang=80.0)
p.decide("FREE_PATH", 0.9, 100.1, yaw=R(90)); p.decide("FREE_PATH", 0.9, 100.2, yaw=R(90))
p.decide("FREE_PATH", 0.9, 100.3, yaw=R(90))   # dispara giro
check("--no-heading-grid: turn_target_yaw es None (giro relativo)", p.turn_target_yaw is None and p.state == TURN)
# sin yaw: aunque heading_grid on, cae al relativo
p = mk()
p.decide("ARROW_RIGHT", 0.99, 100.0, yaw=None, arrow_ang=80.0)
p.decide("FREE_PATH", 0.9, 100.1, yaw=None); p.decide("FREE_PATH", 0.9, 100.2, yaw=None)
p.decide("FREE_PATH", 0.9, 100.3, yaw=None)
check("yaw None: turn_target_yaw es None (fallback)", p.turn_target_yaw is None)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
