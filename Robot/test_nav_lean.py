"""Tests OFFLINE de la navegacion LEAN (nav-v24): dirigida por el clasificador, geometria SOLO
para elegir el lado de una pared YA clasificada como WALL. Sin centrado/curva, sin taper, sin
latches, sin giro perceptual. No necesita Gazebo (stubea gz/cnn_infer). Ejecutar:
    python Robot/test_nav_lean.py
"""
import sys
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

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        passed += 1; print(f"  ok   {name}")
    else:
        failed += 1; print(f"  FAIL {name}")


def mk(**kw):
    # yaw=None -> los giros se completan por tiempo (deterministico para el test)
    return Policy(forward=0.3, turn=0.6, steer=0.3, search=0.4,
                  conf_thresh=0.4, sign_conf_thresh=0.55, arrow_conf_thresh=0.95,
                  wall_probe_s=2.0, pending_lost_frames=3, **kw)

TURN = navigate.Policy.TURN
PROBE = navigate.Policy.WALL_PROBE

print("== T1: RECTO en cruce (sin centrado/nivelado) ==")
p = mk()
lin, ang, _, _ = p.decide("FREE_PATH", 0.9, 100.0, dark=(0.8, 0.0, 0.05), open_side="left", center_blocked=False)
check("FREE_PATH centro limpio + lado abierto -> recto (ang==0, lin>0)", ang == 0.0 and lin > 0)
lin, ang, _, _ = p.decide("ARROW_UP", 0.9, 100.0, dark=(0.7, 0.0, 0.1), open_side="right")
check("ARROW_UP -> recto a velocidad plena (ang==0, lin==forward)", ang == 0.0 and abs(lin - p.forward) < 1e-9)

print("== T2: FLECHA (acerca recto + gira al perder de vista, direccion correcta) ==")
p = mk()
lin, ang, _, _ = p.decide("ARROW_RIGHT", 0.99, 100.0, arrow_ang=80.0)
check("ARROW_RIGHT confiable -> pending right, recto (ang==0)", p.pending == "right" and ang == 0.0 and lin > 0)
# pierde de vista x3 -> gira la flecha (derecha => turn_dir -1, ang<0)
p.decide("FREE_PATH", 0.9, 100.1); p.decide("FREE_PATH", 0.9, 100.2)
lin, ang, _, note = p.decide("FREE_PATH", 0.9, 100.3)
check("flecha der perdida x3 -> giro DERECHA (turn_dir<0, ang<0)", p.state == TURN and p.turn_dir == -1 and ang < 0)
check("nota del giro = flecha-der@perdida", "flecha-der@perdida" in note)
# simetrico izquierda
p = mk(); p.decide("ARROW_LEFT", 0.99, 100.0, arrow_ang=-80.0)
p.decide("FREE_PATH", 0.9, 100.1); p.decide("FREE_PATH", 0.9, 100.2)
_, ang, _, _ = p.decide("FREE_PATH", 0.9, 100.3)
check("flecha izq perdida x3 -> giro IZQUIERDA (turn_dir>0, ang>0)", p.turn_dir == +1 and ang > 0)
# via rapida flecha@pared (cls=="WALL")
p = mk(); p.decide("ARROW_LEFT", 0.99, 100.0, arrow_ang=-80.0)
lin, ang, _, note = p.decide("WALL", 0.9, 100.1)
check("flecha pendiente + WALL -> giro de FLECHA @pared (no maniobra de pared)", "flecha-izq@pared" in note and lin == 0.0)

print("== T3: CRUZ = 180 simple (sin rompe-bucles) ==")
p = mk()
lin, ang, _, note = p.decide("CROSS", 0.90, 100.0)
check("CRUZ confiable -> 180 (ang>0, lin==0)", ang > 0 and lin == 0.0 and p.turn_target_deg == p.cross_deg and "CRUZ" in note)
p = mk()
lin, ang, _, note = p.decide("CROSS", 0.50, 100.0)
check("CRUZ dudosa -> recto sin fijar (ang==0, lin>0)", ang == 0.0 and lin > 0 and "dudosa" in note)
# 2a cruz tras un 180 anterior -> OTRA VEZ 180 (no 90 lateral)
p = mk(); p.decide("CROSS", 0.90, 100.0)
p.state = Policy.FOLLOW; p.ignore_signs_until = 0.0     # simula fin de giro + enfriamiento pasado
_, _, _, note = p.decide("CROSS", 0.90, 120.0)
check("2a CRUZ -> de nuevo 180 (sin salida lateral 90)", "CRUZ" in note and "lateral" not in note and p.turn_target_deg == p.cross_deg)

print("== T4: WALL -> WALL_PROBE -> gira al lado despejado por GEOMETRIA ==")
p = mk()
lin, ang, _, note = p.decide("WALL", 0.9, 100.0, open_side="left", dark=(0.1, 0.8, 0.9))
check("WALL -> entra a WALL_PROBE (sondeo, lin>0, ang==0)", p.state == PROBE and lin > 0 and ang == 0.0)
lin, ang, _, note = p.decide("WALL", 0.9, 100.0 + p.wall_probe_s + 0.1, open_side="left", dark=(0.1, 0.8, 0.9))
check("tras el sondeo, sin señal -> gira al lado despejado IZQ (turn_dir>0, ang>0)", p.turn_dir == +1 and ang > 0 and "left" in note)
p = mk(); p.decide("WALL", 0.9, 100.0, open_side="right", dark=(0.9, 0.8, 0.1))
_, ang, _, note = p.decide("WALL", 0.9, 100.0 + p.wall_probe_s + 0.1, open_side="right", dark=(0.9, 0.8, 0.1))
check("WALL open_side=right -> gira DERECHA (turn_dir<0, ang<0)", p.turn_dir == -1 and ang < 0)

print("== T5: PRIORIDAD GOAL > flecha > cruz > pared ==")
# GOAL > flecha
p = mk(); p.pending = "right"
p.decide("GOAL", 0.9, 100.0, goal_off=0.0, goal_area=0.02)
check("GOAL homing anula la intencion de flecha (pending None)", p.pending is None)
# flecha > cruz
p = mk(); p.pending = "left"
_, _, _, note = p.decide("CROSS", 0.99, 100.0)
check("flecha pendiente > CRUZ (cuenta perdido, NO 180)", p.state != TURN and "pendiente" in note)
# flecha > pared
p = mk(); p.pending = "right"
_, _, _, note = p.decide("WALL", 0.9, 100.0)
check("flecha pendiente > PARED (gira la flecha @pared)", "flecha-der@pared" in note)
# racha GOAL -> done
p = mk(); done = False
for i in range(5):
    _, _, done, _ = p.decide("GOAL", 0.9, 100.0 + i)
check("5x GOAL>0.85 -> laberinto resuelto (done)", done is True)

print("== T6 (CLAVE): NINGUNA pared dispara giro sin la clase WALL ==")
for c in ("FREE_PATH", "ARROW_UP"):
    p = mk()
    lin, ang, _, _ = p.decide(c, 0.9, 100.0, dark=(0.1, 0.9, 0.9), center_blocked=True, open_side="left")
    check(f"{c} con centro 90% negro + center_blocked -> NO gira (ang==0, lin>0, FOLLOW)",
          ang == 0.0 and lin > 0 and p.state != TURN)
# SIN señal fiable (lost_go_straight=True por defecto):
# (a) centro LIBRE -> sigue RECTO (no se para a buscar); el LIDAR encarrila el pasillo.
p = mk()
lin, ang, _, _ = p.decide("UNKNOWN", 0.2, 100.0, dark=(0.1, 0.0, 0.1), center_blocked=False)
check("sin señal + centro libre -> sigue RECTO (lin>0, FOLLOW, no busca)",
      lin > 0 and p.state != TURN)
# (b) PARED de frente -> gira al lado despejado (callejon/giro forzado sin señal)
p = mk()
lin, ang, _, _ = p.decide("UNKNOWN", 0.2, 100.0, dark=(0.1, 0.9, 0.9), center_blocked=True, open_side="left")
check("sin señal + PARED de frente -> gira al lado despejado (TURN)", p.state == TURN)
# (c) ABLACION --no-lost-go-straight: vuelve al comportamiento clasico (avanza despacio, no gira)
p = mk(lost_go_straight=False)
lin, ang, _, _ = p.decide("UNKNOWN", 0.2, 100.0, dark=(0.1, 0.9, 0.9), center_blocked=True)
check("ablacion (no-lost-go-straight): UNKNOWN + centro bloqueado -> avanza despacio (no gira)",
      p.state != TURN and ang == 0.0)
# control positivo: con clase WALL SI reacciona
p = mk()
_, _, _, _ = p.decide("WALL", 0.9, 100.0, dark=(0.1, 0.9, 0.9), center_blocked=True, open_side="left")
check("control: cls=WALL SI entra a WALL_PROBE", p.state == PROBE)

print("== T7: regresiones ==")
p = mk()
check("_search_dir: izq mas negro -> derecha (-1)", p._search_dir(None, (0.9, 0.0, 0.1)) == -1)
check("_search_dir: der mas negro -> izquierda (+1)", p._search_dir(None, (0.1, 0.0, 0.9)) == +1)
check("_arrow_turn_deg: vector concuerda -> magnitud del vector", abs(p._arrow_turn_deg("right", 60.0) - 60.0) < 1e-9)
check("_arrow_turn_deg: vector contradice -> nominal", abs(p._arrow_turn_deg("right", -60.0) - p.arrow_deg) < 1e-9)
check("_arrow_turn_deg: sin vector -> nominal", abs(p._arrow_turn_deg("left", None) - p.arrow_deg) < 1e-9)
# TURN interrumpible por CRUZ SOLO sobre giro de PARED (no de flecha)
p = mk(); p.decide("WALL", 0.9, 100.0, open_side="left", dark=(0.1, 0.8, 0.9))
p.decide("WALL", 0.9, 100.0 + p.wall_probe_s + 0.1, open_side="left", dark=(0.1, 0.8, 0.9))  # -> TURN [PARED]
_, _, _, note = p.decide("CROSS", 0.90, 100.0 + p.wall_probe_s + 0.2)
check("CRUZ confiable interrumpe un giro de PARED -> 180", "interrumpe" in note)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
