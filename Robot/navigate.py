"""Controlador de navegacion del robot en Gazebo (Harmonic).

Bucle cerrado en tiempo real:
    camara (/camera/image_raw) -> CNN -> maquina de estados -> /cmd_vel

Politica REACTIVA de lazo cerrado: cada decision fija una velocidad (lineal y
angular) y el robot la mantiene hasta la siguiente. Como solo hace falta
re-evaluar con cada frame, los giros se resuelven solos (gira hasta que la
vista de delante este despejada), lo que tolera bien la baja tasa de la camara.

Convencion de signo (igual que move_robot.py): angular.z > 0 = girar IZQUIERDA,
angular.z < 0 = girar DERECHA.

Uso (en WSL2, con un mundo ya lanzado con `gz sim ...`):
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 navigate.py
    # o usar el lanzador run_navigate.sh que ya exporta la variable.
"""
import argparse
import csv
import math
import os
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

# Necesario ANTES de importar gz.msgs (conflicto con protobuf nuevo en ~/.local)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
try:                                  # cv2 opcional: solo para la ventana de debug (--show)
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.twist_pb2 import Twist
try:                                  # odometria opcional (giro por angulo real)
    from gz.msgs10.odometry_pb2 import Odometry
    HAS_ODOM = True
except Exception:                     # si el binding no esta, se cae a giro por tiempo
    Odometry = None
    HAS_ODOM = False
try:                                  # LIDAR opcional (--wall-source lidar)
    from gz.msgs10.laserscan_pb2 import LaserScan
    HAS_LIDAR = True
except Exception:                     # si el binding no esta, solo queda la fuente camara
    LaserScan = None
    HAS_LIDAR = False

from cnn_infer import SignClassifier, DEFAULT_ARROW_RUN
# Detectores de pared INTERCAMBIABLES (mismo contrato WallReading): pixeles negros
# (camara, baseline) vs distancias 360 (LIDAR, encarrila + mapea). Ver wall_camera.py / wall_lidar.py
from wall_camera import CameraWallDetector, make_wall_debug
from wall_lidar import LidarWallDetector, OccupancyMap, make_scan
from semantic_map import semantic_color


def yaw_from_quat(q):
    """Yaw (rad) a partir de un cuaternion gz (q.w, q.x, q.y, q.z)."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def ang_diff(a, b):
    """Diferencia angular mas corta a-b en (-pi, pi]."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


def arrow_angle(rgb, min_green=80, green_margin=20, min_px=30):
    """Angulo (grados) al que apunta la flecha VERDE respecto a 'recto arriba'.

    Negativo = apunta a la IZQUIERDA, positivo = DERECHA, ~0 = recto. None si no
    hay verde suficiente. Segmenta el verde (la flecha resalta sobre el gris),
    saca el eje mayor por PCA y resuelve la PUNTA como el lado de la franja
    transversal mas ANCHA (la cabeza de la flecha). Validado offline sobre el
    dataset: ~94% de acierto de signo en IZQ, ~85% en DER (por eso en navegacion
    se cruza con la clase de la CNN: la direccion la manda la CNN, el vector solo
    aporta la MAGNITUD del giro cuando concuerda).
    """
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    mask = (g > min_green) & (g - r > green_margin) & (g - b > green_margin)
    ys, xs = np.nonzero(mask)
    if xs.size < min_px:
        return None
    x0 = xs.astype(np.float64) - xs.mean()
    y0 = ys.astype(np.float64) - ys.mean()
    evals, evecs = np.linalg.eigh(np.cov(np.stack([x0, y0])))
    u = evecs[:, int(np.argmax(evals))]
    ux, uy = float(u[0]), float(u[1])
    p = x0 * ux + y0 * uy                 # proyeccion sobre el eje mayor
    q = -x0 * uy + y0 * ux                # perpendicular
    nb = 10
    edges = np.linspace(p.min(), p.max(), nb + 1)
    widths = []
    for i in range(nb):
        sel = (p >= edges[i]) & (p < edges[i + 1])
        widths.append(q[sel].std() if sel.sum() > 2 else 0.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sign = 1.0 if centers[int(np.argmax(widths))] > 0 else -1.0   # punta = franja mas ancha
    tx, ty = sign * ux, sign * uy
    return math.degrees(math.atan2(tx, -ty))   # imagen: arriba=-y; >0 der, <0 izq


def goal_in_view(rgb, min_blue=60, blue_margin=25, min_px=20):
    """Localiza la DIANA (blanco AZUL) en el frame por su color.

    Devuelve (offset_frac, area_frac): offset_frac en [-1,1] del centroide azul
    respecto al centro (neg=izquierda, pos=derecha) y area_frac = azul/total.
    Devuelve (None, 0.0) si no hay azul suficiente. El area sirve de proxy de
    cercania (mas grande = mas cerca). Validado offline: la diana siempre tiene
    azul (~2-3.5% a distancia de navegacion); CROSS/flechas no tienen azul.
    """
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    mask = (b > min_blue) & (b - r > blue_margin) & (b - g > blue_margin)
    n = int(mask.sum())
    h, w, _ = rgb.shape
    if n < min_px:
        return None, 0.0
    cx = float(np.nonzero(mask)[1].mean())
    return (cx - w / 2.0) / (w / 2.0), n / float(h * w)


class Policy:
    """Maquina de estados para navegar leyendo señales.

    PRIORIDAD de maniobras: GOAL > FLECHA > CRUZ > PARED. Una intencion de flecha
    pendiente NO la pisa una cruz ni una pared (ni en FOLLOW ni durante el giro);
    una cruz solo manda sobre una pared. La GOAL (diana) manda sobre todo.

    Dos mecanismos clave (ademas de esa prioridad):

    1) INTENCION DIFERIDA (solo FLECHAS): al ver una flecha NO gira al instante
       (giraria demasiado pronto, a una distancia que depende de cuando la
       detecto y del delay del topic). Memoriza la direccion (self.pending) y
       sigue ACERCANDOSE al cartel; ejecuta el giro en cuanto lo PIERDE DE VISTA
       (la clase deja de ser una flecha durante pending_lost_frames frames
       seguidos), o antes si se topa con una pared (via rapida). Asi el giro
       ocurre junto al cartel sin depender de que haya una pared detras: una
       flecha "flotando" sin pared tambien acaba girando (antes, al disparar solo
       con WALL, se perdia). La CRUZ (media vuelta) NO se difiere: como solo
       invierte el sentido, gira 180 en cuanto se ve con confianza (la salida
       esta detras, no hace falta llegar a una pared).

    2) GIRO CON ALINEACION (POR ODOMETRIA): gira AL MENOS el angulo NOMINAL
       (objetivo) y luego sigue hasta que el camino de delante queda despejado
       (FREE_PATH/ARROW_UP), con tope en objetivo+turn_tol_deg. NUNCA gira de
       menos (flecha >=90, cruz >=180) pero corrige overshoot parando al alinear.
       Ventana solo HACIA ARRIBA: [objetivo, objetivo+turn_tol_deg]. El angulo
       girado se MIDE con la ODOMETRIA del robot (yaw real acumulado), asi que es
       independiente del RTF de Gazebo y de la inercia. Si no hay odometria, cae
       a giro por TIEMPO (t = angulo / vel_angular * turn_scale; asume RTF~1).

    Tras un giro hay ENFRIAMIENTO (--post-turn-s) que ignora señales un momento
    para alejarse del cartel y no re-dispararlo. WALL_PROBE: si llega a una
    pared SIN intencion previa, avanza despacio buscando una señal antes de
    girar por defecto.

    3) GATING DE DIRECCIONALES: una flecha/cruz solo COMPROMETE la maniobra si
       supera un umbral de confianza MAYOR que el de actuar. Hay DOS umbrales:
         - sign_conf_thresh (0.55): aplica a la CRUZ (la atiende el main; sigue
           teniendo confianza moderada en vivo).
         - arrow_conf_thresh (0.95): aplica a las FLECHAS, atendidas por el
           specialist (cnn_arrows). Como el specialist responde con conf ~1.0
           cuando ve una flecha de verdad, podemos exigir mucho mas alto y
           rechazar cualquier respuesta dudosa sin riesgo de perder maniobras.

    4) GIRO INTERRUMPIBLE (cross_interrupt_conf): una CRUZ muy confiable vista a
       mitad de un giro reconvierte la maniobra a media vuelta (180) desde ese
       punto. Antes una cruz vista durante un giro de pared se perdia.

    Estados: FOLLOW | WALL_PROBE | TURN.
    """
    FOLLOW, TURN, WALL_PROBE = "FOLLOW", "TURN", "WALL_PROBE"

    def __init__(self, forward=0.3, turn=0.6, steer=0.3, search=0.4, conf_thresh=0.4,
                 sign_conf_thresh=0.55, arrow_conf_thresh=0.95,
                 cross_interrupt_conf=0.85, pending_lost_frames=3,
                 arrow_deg=90.0, cross_deg=180.0, wall_deg=90.0, turn_tol_deg=20.0,
                 post_turn_s=1.5, turn_scale=1.0, unknown_patience=4,
                 wall_probe_s=2.0, wall_probe_speed=0.3,
                 arrow_min_deg=25.0, arrow_max_deg=135.0,
                 goal_reach_frac=0.08, goal_center_tol=0.30,
                 goal_confirm_cooldown=6.0,
                 goal_streak_conf=0.85, goal_streak_count=5,
                 turn_stop_on_open=True, turn_open_thresh=0.10,
                 turn_open_drop=0.20, turn_min_open_deg=20.0,
                 search_default=-1,
                 heading_grid=True, grid_deg=45.0, heading_tol_deg=6.0,
                 heading_hold_gain=1.0, heading_hold_deadzone_deg=3.0,
                 heading_recalib_gain=0.1, heading_recalib_patience=5,
                 heading_recalib_max_deg=20.0, turn_ramp_deg=25.0,
                 lost_go_straight=True):
        self.forward = forward
        self.turn = turn
        self.steer = steer
        self.search = search
        self.lost_go_straight = lost_go_straight   # sin señal -> seguir RECTO (LIDAR) en vez de girar a "buscar"
        self.conf_thresh = conf_thresh
        self.sign_conf_thresh = sign_conf_thresh        # umbral (mayor) para COMPROMETER la CRUZ
        self.arrow_conf_thresh = arrow_conf_thresh      # umbral (alto) para COMPROMETER una FLECHA (specialist)
        self.cross_interrupt_conf = cross_interrupt_conf  # conf minima para que una CRUZ interrumpa un giro
        self.pending_lost_frames = pending_lost_frames    # frames sin ver el cartel para girar la flecha pendiente
        self.arrow_deg = arrow_deg
        self.cross_deg = cross_deg
        self.wall_deg = wall_deg
        self.turn_tol_deg = turn_tol_deg          # margen EXTRA por encima del objetivo (ventana [obj, obj+tol])
        self.turn_scale = turn_scale
        self.post_turn_s = post_turn_s
        self.unknown_patience = unknown_patience
        self.wall_probe_s = wall_probe_s          # tiempo de sondeo ante una pared
        self.wall_probe_speed = wall_probe_speed  # fraccion de forward al sondear
        self.arrow_min_deg = arrow_min_deg        # clamp inferior del giro de flecha por vector
        self.arrow_max_deg = arrow_max_deg        # clamp superior del giro de flecha por vector
        self.goal_reach_frac = goal_reach_frac    # area azul (frac) para dar la META por alcanzada
        self.goal_center_tol = goal_center_tol    # |offset| max de la diana para considerarla centrada
        self.goal_confirm_cooldown = goal_confirm_cooldown  # s que se ignora la diana tras un rechazo del usuario
        self.goal_reject_until = 0.0              # hasta este t la META se ignora (falso positivo confirmado por el usuario)
        self.goal_streak_conf = goal_streak_conf  # conf minima de la clase GOAL para sumar a la racha de finalizacion
        self.goal_streak_count = goal_streak_count  # GOAL consecutivos (>goal_streak_conf) para dar el laberinto por resuelto
        self.goal_streak = 0                      # contador de GOAL consecutivos por encima del umbral
        self.turn_stop_on_open = turn_stop_on_open  # parar el giro al revelarse un pasillo nuevo
        self.turn_open_thresh = turn_open_thresh    # negro centro <= esto = frente despejado
        self.turn_open_drop = turn_open_drop        # caida de negro centro vs el inicio del giro
        self.turn_min_open_deg = turn_min_open_deg  # giro minimo antes de permitir parar por apertura
        # nav-v24 (revert a lean): el algoritmo de pared vuelve a ser MINIMO -> la geometria
        # (wall_open_side) solo elige el LADO de una pared YA clasificada como WALL; NO dispara
        # evitacion por su cuenta. Se quitaron: giro perceptual (v18), centrado/curva _center_steer
        # (v20/v23), taper de velocidad (v23b), latch de cruz/rompe-bucles (v20/v23e), disparo
        # geometrico (v22). El clasificador vuelve a mandar (prioridad GOAL>flecha>cruz>pared).
        self.last_cross_t = -1e9                    # instante del ultimo giro de cruz (solo para log)
        self.search_default = search_default        # sentido (+1 izq/-1 der) SOLO si no hay geometria ni open_side
        self.wall_side_margin = 0.15                # |dl-dr| minimo para que la geometria decida el lado abierto
        # nav-v25: rumbo anclado a la rejilla de 8 direcciones (45 grados) con el yaw absoluto de la
        # odometria. Cada giro aterriza en un rumbo CANONICO (corrige la deriva en vez de acumularla) y
        # en recto se mantiene ese rumbo (heading-hold). Requiere odometria; si no hay yaw, cae al
        # fallback nav-v24 (giro relativo/tiempo) y el heading-hold se desactiva solo.
        self.heading_grid = heading_grid            # activar el anclaje a la rejilla
        self.grid_step = math.radians(grid_deg)     # paso de la rejilla (rad); 45 grados = 8 direcciones
        self.heading_tol = math.radians(heading_tol_deg)  # tolerancia para dar por terminado el giro al rumbo objetivo
        self.heading_hold_gain = heading_hold_gain  # ganancia del heading-hold (residuo rad -> angular)
        self.heading_hold_deadzone = math.radians(heading_hold_deadzone_deg)  # residuo por debajo = no corrige
        self.heading_recalib_gain = heading_recalib_gain  # cuanto se acerca el ancla al residuo en cada frame recto
        self.heading_recalib_patience = heading_recalib_patience  # frames rectos confiables seguidos antes de recalibrar
        self.heading_recalib_max = math.radians(heading_recalib_max_deg)  # residuo maximo para recalibrar (si mayor, no es deriva)
        self.turn_ramp = math.radians(turn_ramp_deg)  # banda de rampa proporcional del giro cerca del objetivo
        self.heading_anchor = None                  # yaw de referencia de la rejilla (se captura al 1er yaw)
        self.turn_target_yaw = None                 # rumbo ABSOLUTO objetivo del giro en curso (None = sin grid)
        self.straight_count = 0                     # frames rectos confiables seguidos (para recalibrar el ancla)
        self._last_yaw = None                       # ultimo yaw visto en decide (para _start_turn)

        self.pending = None           # 'left'|'right' -> maniobra a ejecutar en el cruce
        self.pending_angle = arrow_deg # grados a girar de la flecha pendiente (del vector, o nominal)
        self.last_arrow = None
        self.state = self.FOLLOW
        self.turn_dir = -1            # +1 izquierda, -1 derecha
        self.turn_start = 0.0
        self.turn_min_t = 0.0         # tiempo minimo de giro (fallback sin odometria)
        self.turn_max_t = 0.0         # tope de giro por tiempo (fallback / safety)
        self.turn_target_deg = 0.0    # angulo objetivo del giro (para odometria)
        self.turn_max_deg = 0.0       # tope de angulo del giro (objetivo + tol)
        self.turn_accum_deg = 0.0     # angulo REAL acumulado (por odometria) en el giro
        self.turn_last_yaw = None     # ultimo yaw visto (para acumular)
        self.turn_use_odom = False    # True si en este giro hubo lecturas de odometria
        self.turn_start_darkc = None  # negro central al empezar el giro (para detectar apertura)
        self.turn_trigger = ""        # que disparo el giro (para el log)
        self.ignore_signs_until = 0.0
        self.unknown_count = 0
        self.search_latch = 0         # sentido de busqueda FIJADO al entrar (anti-vaiven); 0 = no buscando
        self.lost_count = 0           # frames seguidos SIN ver el cartel con intencion pendiente
        self.probe_start = 0.0        # inicio del sondeo de pared
        self.probe_open_side = None   # lado mas despejado visto durante el sondeo (geometria)

    def _t(self, deg):
        return math.radians(max(0.0, deg)) / max(self.turn, 1e-6) * self.turn_scale

    def _grid_ready(self, yaw):
        """True si el anclaje a la rejilla esta operativo (activo, con odom y ancla)."""
        return self.heading_grid and yaw is not None and self.heading_anchor is not None

    def _grid_snap(self, yaw):
        """Rumbo CANONICO de la rejilla mas cercano a yaw (multiplo de grid_step desde el ancla),
        normalizado a (-pi, pi]."""
        k = round(ang_diff(yaw, self.heading_anchor) / self.grid_step)
        snapped = self.heading_anchor + k * self.grid_step
        return math.atan2(math.sin(snapped), math.cos(snapped))

    def _grid_residual(self, yaw):
        """Error CON SIGNO del yaw al rumbo de rejilla mas cercano (rad). >0 = hay que girar +."""
        return ang_diff(self._grid_snap(yaw), yaw)

    def _heading_hold(self, yaw):
        """nav-v25: correccion angular para MANTENER el rumbo canonico en recto (heading-hold). Solo
        corrige el residuo al multiplo de 45 mas cercano -> mantiene el rumbo del pasillo sin desviarse
        en los cruces (una apertura lateral no cambia el yaw). 0 si no hay rejilla/odom."""
        if not self._grid_ready(yaw):
            return 0.0
        res = self._grid_residual(yaw)
        if abs(res) < self.heading_hold_deadzone:
            return 0.0
        return max(-self.steer, min(self.steer, self.heading_hold_gain * res))

    def _maybe_recalibrate(self, yaw, cls, conf):
        """nav-v25: en tramos rectos largos y CONFIABLES (FREE_PATH/ARROW_UP), el rumbo real del robot
        ES una direccion de la rejilla -> acerca suavemente el ancla al residuo (low-pass) para bloquear
        la rejilla a los pasillos reales (robusto a un spawn algo torcido o a deriva lenta de la odom)."""
        if not self._grid_ready(yaw):
            return
        if cls in ("FREE_PATH", "ARROW_UP") and conf >= self.conf_thresh:
            self.straight_count += 1
        else:
            self.straight_count = 0
            return
        if self.straight_count >= self.heading_recalib_patience:
            res = self._grid_residual(yaw)                 # = rumbo_rejilla - yaw
            if abs(res) < self.heading_recalib_max:        # residuo pequeño = deriva, no un giro real
                # mueve el ANCLA HACIA el yaw (la rejilla sigue al pasillo real): anchor += (yaw - snap)
                nuevo = self.heading_anchor - self.heading_recalib_gain * res
                self.heading_anchor = math.atan2(math.sin(nuevo), math.cos(nuevo))

    def _arrow_turn_deg(self, cnn_dir, arrow_ang):
        """Grados a girar para una flecha. La DIRECCION la manda la CNN (cnn_dir);
        el VECTOR (arrow_ang) solo aporta la MAGNITUD si concuerda en signo y cae
        en [arrow_min_deg, arrow_max_deg]. Si no, usa el nominal arrow_deg."""
        if arrow_ang is None:
            return self.arrow_deg
        vec_left = arrow_ang < 0
        if (cnn_dir == "left") != vec_left:        # el vector contradice a la CNN
            return self.arrow_deg
        return min(max(abs(arrow_ang), self.arrow_min_deg), self.arrow_max_deg)

    def _search_dir(self, open_side, dark):
        """Sentido (+1 izq / -1 der) cuando se PIERDE la señal y hay que buscar girando.
        Prefiere GEOMETRÍA (lado MENOS negro = más abierto) si la diferencia supera el margen;
        si no, usa open_side; y solo si no hay nada, el defecto configurable search_default."""
        if dark is not None:
            dl, _dc, dr = dark
            if abs(dl - dr) >= self.wall_side_margin:        # un lado claramente más abierto
                return -1 if dl > dr else +1                  # izq más negro -> buscar a la DERECHA
        if open_side == "left":
            return +1
        if open_side == "right":
            return -1
        return self.search_default

    def _wall_maneuver(self, open_side, now):
        """nav-v24 (lean): ante una PARED clasificada (cls=='WALL') SIN intencion de flecha, entra a
        WALL_PROBE: avanza despacio buscando una señal del cruce y, si no aparece, gira por ANGULO
        (nav-v9, con alineacion) hacia el lado que la GEOMETRIA marca despejado (wall_open_side). La
        geometria SOLO elige el lado (el clasificador colapsa WALL_LEFT/RIGHT->WALL); no dispara nada."""
        self.state = self.WALL_PROBE
        self.probe_start = now
        self.probe_open_side = open_side          # hint inicial (mas distancia = mas fiable)
        return self.forward * self.wall_probe_speed, 0.0, False, \
            "PARED -> sondeo (busca señal antes de girar)"

    def _cross_turn(self, now, trigger="CRUZ"):
        """Media vuelta de CRUZ (la salida esta detras). nav-v24: 180 simple (sin rompe-bucles ni
        latch); el enfriamiento post_turn_s evita re-disparar con la misma cruz en cuadro."""
        self.last_arrow = None
        self.last_cross_t = now
        return self._start_turn(+1, self.cross_deg, now, trigger)

    def _start_turn(self, direction, target_deg, now, trigger):
        """Inicia un giro. nav-v25: si hay rejilla+odom, el giro apunta a un rumbo ABSOLUTO de la rejilla
        (turn_target_yaw = snap del destino previsto = yaw + direction*target_deg) -> corrige la deriva.
        Si no, fallback nav-v24: giro RELATIVO por angulo (odom) o por tiempo, con alineacion nav-v12."""
        self.state = self.TURN
        self.turn_dir = direction
        self.turn_start = now
        self.turn_min_t = self._t(target_deg)
        self.turn_max_t = self._t(target_deg + self.turn_tol_deg)
        self.turn_target_deg = target_deg
        self.turn_max_deg = target_deg + self.turn_tol_deg
        self.turn_accum_deg = 0.0
        self.turn_last_yaw = None
        self.turn_use_odom = False
        self.turn_start_darkc = None
        self.turn_trigger = trigger
        self.pending = None           # intencion consumida
        self.lost_count = 0
        self.straight_count = 0       # nav-v25: un giro rompe el tramo recto (recalibracion)
        lado = "izq" if direction > 0 else "der"
        if self._grid_ready(self._last_yaw):
            desired = self._last_yaw + direction * math.radians(target_deg)
            self.turn_target_yaw = self._grid_snap(desired)
            tgt = f"-> rumbo rejilla {math.degrees(self.turn_target_yaw):+.0f}°"
        else:
            self.turn_target_yaw = None      # sin rejilla -> giro relativo/tiempo (nav-v24)
            tgt = f"~{target_deg:.0f}° (relativo)"
        return 0.0, direction * self.turn, False, f"INICIO giro {lado} {tgt} [{trigger}]"

    def reject_goal(self, now):
        """El usuario confirmo por terminal que la META detectada era FALSA.

        Ignora la diana durante goal_confirm_cooldown segundos para que el robot
        se aleje del falso positivo sin volver a pararse (ni hacer homing) con la
        misma imagen."""
        self.goal_reject_until = now + self.goal_confirm_cooldown
        self.goal_streak = 0          # la racha que disparo el (falso) fin se descarta

    def decide(self, cls, conf, now, open_side=None, yaw=None, arrow_ang=None,
               goal_off=None, goal_area=0.0, center_blocked=False, center_dark=None,
               dark=None):
        """Devuelve (linear_x, angular_z, done, nota).

        open_side ('left'|'right'|None): lado despejado segun el detector
        geometrico de pared (wall_open_side); solo se usa para elegir el giro de
        una pared SIN señal (las señales tienen prioridad).
        yaw (rad|None): orientacion actual del robot (odometria); si esta, los
        giros se completan por ANGULO REAL medido (independiente del RTF).
        arrow_ang (grados|None): angulo al que apunta la flecha verde (arrow_angle);
        da la MAGNITUD del giro de flecha (la direccion la manda la CNN).
        goal_off (frac|None), goal_area (frac): posicion y tamaño de la DIANA azul
        (goal_in_view). center_blocked (bool): pared negra de frente (geometria; solo para el log).
        dark ((dl,dc,dr)|None): fraccion de negro por franja izq/centro/der (solo para el log)."""
        # FINALIZACION POR RACHA DE GOAL (nav-v16): el laberinto se da por RESUELTO
        # cuando la clase GOAL se detecta con conf > goal_streak_conf en goal_streak_count
        # frames CONSECUTIVOS (cada uno ya es un voto sobre --vote frames). Cualquier OTRA
        # clase, o un GOAL con conf <= umbral, RESETEA el contador. Sustituye al disparo por
        # area de un solo frame: exigir una racha sostenida es un guardia robusto contra
        # falsos positivos (un GOAL espurio aislado no termina la run) y es automatico, lo
        # que permite medir limpiamente el TIEMPO de resolucion de cada ejecucion.
        # nav-v25: yaw absoluto para la rejilla. Captura el ancla al primer yaw (el robot nace alineado
        # a un pasillo) y guarda el ultimo yaw (lo usa _start_turn para fijar el rumbo objetivo).
        if yaw is not None:
            self._last_yaw = yaw
            if self.heading_grid and self.heading_anchor is None:
                self.heading_anchor = yaw

        if cls == "GOAL" and conf > self.goal_streak_conf and now >= self.goal_reject_until:
            self.goal_streak += 1
        else:
            self.goal_streak = 0
        if self.goal_streak >= self.goal_streak_count:
            return 0.0, 0.0, True, (
                f"META confirmada: GOAL x{self.goal_streak} seguidos "
                f"(conf>{self.goal_streak_conf:.2f}) -> laberinto RESUELTO")

        # Tras un rechazo del usuario (reject_goal), se ignora la diana un rato:
        # ni para por META ni hace homing, asi se aleja del falso positivo.
        if now < self.goal_reject_until:
            goal_off, goal_area = None, 0.0

        # --- Estado TURN: gira el minimo y luego hasta despejar (tope maximo) ---
        if self.state == self.TURN:
            # Giro INTERRUMPIBLE: una CRUZ muy confiable vista a mitad de un giro de
            # PARED lo redefine a media vuelta (180) desde aqui (cruz > pared). El
            # guardia "PARED in trigger" hace que NO interrumpa un giro de FLECHA
            # (flecha > cruz) ni otra cruz (no re-dispara con la misma cruz en cuadro).
            if (cls == "CROSS" and conf >= self.cross_interrupt_conf
                    and "PARED" in self.turn_trigger):
                return self._start_turn(self.turn_dir, self.cross_deg, now,
                                        "CRUZ (interrumpe giro de pared)")
            elapsed = now - self.turn_start
            # Acumula el angulo REAL girado por odometria (si hay yaw)
            if yaw is not None:
                if self.turn_last_yaw is not None:
                    self.turn_accum_deg += abs(math.degrees(ang_diff(yaw, self.turn_last_yaw)))
                self.turn_last_yaw = yaw
                self.turn_use_odom = True
            if self.turn_target_yaw is not None and yaw is not None:
                # nav-v25: GIRO A RUMBO ABSOLUTO de la rejilla. Lazo cerrado sobre el yaw -> el delay no
                # acumula: se gira hasta ALCANZAR el rumbo objetivo y el angular baja proporcionalmente
                # cerca de el (amortigua el sobre-giro). El sentido lo da el signo del error (auto-corrige
                # un sobre-giro). Cap de tiempo de seguridad por si la odom se congela.
                err = ang_diff(self.turn_target_yaw, yaw)          # rad, con signo
                cap = elapsed >= self.turn_max_t * 8
                if abs(err) <= self.heading_tol or cap:
                    self.state = self.FOLLOW
                    self.ignore_signs_until = now + self.post_turn_s
                else:
                    frac = max(0.25, min(1.0, abs(err) / self.turn_ramp))
                    w = (1.0 if err > 0 else -1.0) * self.turn * frac
                    lado = "izq" if err > 0 else "der"
                    return 0.0, w, False, (
                        f"girando {lado} -> {math.degrees(self.turn_target_yaw):+.0f}° "
                        f"(err {math.degrees(err):+.0f}°) [{self.turn_trigger}] percibe:{cls}")
            else:
                # FALLBACK nav-v24 (sin rejilla/odom): giro RELATIVO por angulo acumulado o por tiempo,
                # con parada por APERTURA DE PASILLO (nav-v12) y alineacion al despejarse el frente.
                opened = False
                if (self.turn_stop_on_open and center_dark is not None
                        and "CRUZ" not in self.turn_trigger):
                    if self.turn_start_darkc is None:
                        self.turn_start_darkc = center_dark
                    opened = (center_dark <= self.turn_open_thresh
                              and center_dark <= self.turn_start_darkc - self.turn_open_drop)
                clear = cls in ("FREE_PATH", "ARROW_UP") and conf >= self.conf_thresh
                if self.turn_use_odom:
                    turned = self.turn_accum_deg
                    open_stop = opened and turned >= self.turn_min_open_deg
                    completado = (open_stop
                                  or (turned >= self.turn_target_deg and clear)
                                  or turned >= self.turn_max_deg
                                  or elapsed >= self.turn_max_t * 8)
                    prog = f"{turned:.0f}/{self.turn_target_deg:.0f}°" + (" ABRE" if open_stop else "")
                else:
                    open_stop = opened and elapsed >= self._t(self.turn_min_open_deg)
                    completado = (open_stop or (elapsed >= self.turn_min_t and clear)
                                  or (elapsed >= self.turn_max_t))
                    prog = f"{elapsed:.1f}s(sin odom)"
                if not completado:
                    lado = "izq" if self.turn_dir > 0 else "der"
                    dc = f" dc={center_dark:.2f}" if center_dark is not None else ""
                    return 0.0, self.turn_dir * self.turn, False, \
                        f"girando {lado} {prog}{dc} [{self.turn_trigger}] percibe:{cls}"
                self.state = self.FOLLOW
                self.ignore_signs_until = now + self.post_turn_s

        # --- Estado WALL_PROBE: pared al frente, busca señal antes de girar ---
        if self.state == self.WALL_PROBE:
            elapsed = now - self.probe_start
            if open_side is not None:          # recuerda el ultimo lado despejado visto
                self.probe_open_side = open_side
            # Prioridad SEÑAL > PARED: una direccional CONFIABLE manda sobre el
            # giro de pared (umbral mayor para no comprometer con ruido ~0.5).
            if conf >= self.sign_conf_thresh and cls == "CROSS":
                self.last_arrow = None
                self.last_cross_t = now
                return self._start_turn(+1, self.cross_deg, now, "CRUZ (sondeo)")
            if conf >= self.arrow_conf_thresh and cls == "ARROW_LEFT":
                self.last_arrow = "left"
                return self._start_turn(+1, self._arrow_turn_deg("left", arrow_ang), now, "flecha-izq (sondeo)")
            if conf >= self.arrow_conf_thresh and cls == "ARROW_RIGHT":
                self.last_arrow = "right"
                return self._start_turn(-1, self._arrow_turn_deg("right", arrow_ang), now, "flecha-der (sondeo)")
            if conf >= self.conf_thresh and cls in ("ARROW_UP", "FREE_PATH"):
                self.state = self.FOLLOW          # se abrio camino, no era callejon
            elif elapsed >= self.wall_probe_s:
                self.state = self.FOLLOW          # sin señal -> elige lado por GEOMETRIA, o por defecto
                if self.probe_open_side in ("left", "right"):
                    side = self.probe_open_side
                    d = +1 if side == "left" else -1
                    return self._start_turn(d, self.wall_deg, now, f"PARED -> lado despejado: {side}")
                d = +1 if self.last_arrow == "left" else -1
                return self._start_turn(d, self.wall_deg, now, "PARED (sin señal, por defecto)")
            else:
                return self.forward * self.wall_probe_speed, 0.0, False, \
                    f"PARED -> sondeo {elapsed:.1f}/{self.wall_probe_s:.1f}s (busca señal)"

        # --- Estado FOLLOW ---
        signs_on = now >= self.ignore_signs_until
        self._maybe_recalibrate(yaw, cls, conf)   # nav-v25: re-ancla la rejilla en tramos rectos confiables
        hh = self._heading_hold(yaw)              # nav-v25: correccion para mantener el rumbo canonico en recto

        # DIANA a la vista (nav-v11): dirigirse hacia ella. Maxima prioridad en FOLLOW.
        # NO para (eso lo hace 'META alcanzada' por cercania). Si la diana esta a un lado,
        # gira hacia ella SIN avanzar (no embiste una pared lateral); si esta centrada,
        # avanza, salvo que haya una pared negra de por medio y la diana aun lejos -> esquiva.
        if cls == "GOAL" and conf >= self.conf_thresh and goal_off is not None:
            self.pending = None                       # la diana manda sobre cualquier intencion
            ang = max(-self.turn, min(self.turn, -goal_off * self.turn))
            if abs(goal_off) > self.goal_center_tol:
                return 0.0, ang, False, f"META a un lado -> giro hacia ella (off {goal_off:+.2f})"
            if center_blocked and goal_area < self.goal_reach_frac:
                d = +1 if open_side == "left" else -1
                return self.forward * 0.4, d * self.turn, False, \
                    f"META pero PARED de por medio -> esquivo ({'izq' if d > 0 else 'der'})"
            return self.forward, ang, False, f"META centrada -> avanzo (area {goal_area*100:.1f}%)"

        # INTENCION DE FLECHA pendiente (nav-v7): el robot se ACERCA mientras VE el
        # cartel y gira en cuanto lo PIERDE DE VISTA (la clase deja de ser una flecha
        # durante pending_lost_frames frames seguidos). Va antes de la rama de baja
        # confianza para contar tambien cuando el cartel sale de cuadro y la conf cae.
        # Asi el giro no depende de toparse con una pared (una flecha "flotando" gira).
        if self.pending in ("left", "right"):
            d = +1 if self.pending == "left" else -1
            lado = "izq" if self.pending == "left" else "der"
            # VIA RAPIDA (nav-v7): pared de frente (cls=="WALL") -> gira YA la FLECHA (el cruce esta
            # aqui), no la maniobra de pared (flecha > pared).
            if cls == "WALL" and conf >= self.conf_thresh:
                return self._start_turn(d, self.pending_angle, now, f"flecha-{lado}@pared")
            if cls in ("ARROW_LEFT", "ARROW_RIGHT"):
                self.lost_count = 0                       # sigue viendo el cartel -> acercarse RECTO
                return self.forward, hh, False, f"flecha {lado.upper()} pendiente -> me acerco"
            # Cualquier OTRA clase (incluida la CRUZ): el cartel ya no se ve. La FLECHA
            # tiene prioridad sobre la cruz, asi que NO se deja que la cruz dispare su
            # media vuelta: se cuentan los frames perdidos y, al llegar al umbral, se
            # GIRA la flecha; mientras tanto se sigue avanzando hacia el cruce.
            self.lost_count += 1
            if self.lost_count >= self.pending_lost_frames:
                return self._start_turn(d, self.pending_angle, now, f"flecha-{lado}@perdida")
            return self.forward, hh, False, (
                f"flecha {lado.upper()} pendiente, cartel perdido "
                f"{self.lost_count}/{self.pending_lost_frames} (ignoro {cls}) -> avanzo")

        # nav-v24 (lean): NO hay disparo geometrico de pared. La pared se atiende SOLO via la clase
        # WALL del clasificador (mas abajo); la geometria (open_side) solo elige el LADO del giro. Asi
        # el robot NO reacciona a una pared antes que a una señal (prioridad GOAL>flecha>cruz>pared).

        # Baja confianza o UNKNOWN: avanza despacio; si persiste, busca girando.
        # EXCEPCION (nav-v19): una CRUZ dudosa NO dispara la busqueda. La cruz marca un
        # sitio que NO queremos pisar (la salida esta detras; la maniobra correcta es la
        # media vuelta cuando se confirma). Ponerse a "buscar" ahi pierde tiempo y puede
        # arrastrar al robot hacia el cruce. Se deja caer a la rama CROSS de mas abajo:
        # avanza sin comprometerse y, si la conf sube, hace la media vuelta para alejarse.
        if (conf < self.conf_thresh or cls == "UNKNOWN") and cls != "CROSS":
            self.unknown_count += 1
            # SIN señal fiable para decidir el cruce. Dos comportamientos:
            if self.lost_go_straight:
                # nav (con LIDAR): NO hace falta girar a "buscar" -> los pasillos se siguen RECTOS y en un
                # CRUCE sin informacion se sigue RECTO (el encarrilado del LIDAR centra). Solo si hay PARED
                # de frente (center_blocked) se gira hacia el lado despejado (callejon/giro forzado sin señal).
                if center_blocked:
                    if self.search_latch == 0:
                        self.search_latch = self._search_dir(open_side, dark)
                    d = self.search_latch
                    return self._start_turn(d, self.wall_deg, now,
                                            f"sin señal + PARED -> giro al lado despejado ({'izq' if d > 0 else 'der'})")
                return self.forward, hh, False, f"sin señal ({cls} {conf:.2f}) -> sigo RECTO"
            # CLASICO (camara, ablacion --no-lost-go-straight): avanza despacio y, si persiste, BUSCA girando
            # hacia el lado mas abierto (geometria/open_side, _search_dir; sentido fijado al entrar, anti-vaiven).
            if self.unknown_count >= self.unknown_patience:
                if self.search_latch == 0:
                    self.search_latch = self._search_dir(open_side, dark)
                d = self.search_latch
                lado = "izq" if d > 0 else "der"
                return 0.0, d * self.search, False, f"sin señal x{self.unknown_count} -> buscar ({lado})"
            return self.forward * 0.4, hh, False, f"{cls} ({conf:.2f}) -> avanza despacio"
        self.unknown_count = 0
        self.search_latch = 0                     # salio de la incertidumbre -> resetea el barrido

        # Señales -> fijan INTENCION y se sigue AVANZANDO hasta el cruce (pared).
        # Solo se COMPROMETE la intencion si la confianza supera su umbral
        # (sign_conf_thresh para la CRUZ, arrow_conf_thresh para las FLECHAS); con
        # menos, se avanza igual pero sin tocar pending. Esto rechaza respuestas
        # dudosas; con el specialist de flechas (cnn_arrows) la conf real llega a
        # ~1.0 cuando hay flecha, asi que el umbral alto no pierde maniobras.
        if cls == "CROSS" and signs_on:
            if conf >= self.sign_conf_thresh:
                # CRUZ = media vuelta 180 en el sitio (la salida esta detras).
                return self._cross_turn(now)
            return self.forward, hh, False, f"CRUZ dudosa ({conf:.2f}) -> avanzo, sin fijar"
        if cls == "ARROW_LEFT" and signs_on:
            if conf >= self.arrow_conf_thresh:
                self.pending = "left"; self.last_arrow = "left"; self.lost_count = 0
                self.pending_angle = self._arrow_turn_deg("left", arrow_ang)
                return self.forward, hh, False, f"flecha IZQ vista (giro~{self.pending_angle:.0f}°) -> me acerco"
            return self.forward, hh, False, f"flecha IZQ dudosa ({conf:.2f}) -> avanzo, sin fijar"
        if cls == "ARROW_RIGHT" and signs_on:
            if conf >= self.arrow_conf_thresh:
                self.pending = "right"; self.last_arrow = "right"; self.lost_count = 0
                self.pending_angle = self._arrow_turn_deg("right", arrow_ang)
                return self.forward, hh, False, f"flecha DER vista (giro~{self.pending_angle:.0f}°) -> me acerco"
            return self.forward, hh, False, f"flecha DER dudosa ({conf:.2f}) -> avanzo, sin fijar"
        if cls == "ARROW_UP":
            self.pending = None; self.lost_count = 0  # recto explicito anula intencion
            return self.forward, hh, False, "recto"
        if cls in ("FREE_PATH", "CROSS", "ARROW_LEFT", "ARROW_RIGHT"):
            # RECTO con heading-hold (nav-v25): mantiene el rumbo canonico del pasillo (odom), no nivela
            # por vision -> no se mete en el perpendicular en los cruces.
            return self.forward, hh, False, "recto" + ("" if signs_on else " (enfriamiento)")
        if cls == "WALL":
            # Pared al frente SIN intencion de flecha (la via rapida flecha@pared se resuelve antes,
            # en el bloque de intencion pendiente: flecha > pared). La CRUZ gira en el sitio al verla.
            # nav-v24: entra a WALL_PROBE y gira por ANGULO hacia el lado despejado (geometria).
            return self._wall_maneuver(open_side, now)
        if cls == "WALL_LEFT":
            return self.forward * 0.6, -self.steer, False, "pared izq -> corrige der"
        if cls == "WALL_RIGHT":
            return self.forward * 0.6, self.steer, False, "pared der -> corrige izq"
        return self.forward * 0.4, 0.0, False, "otro -> avanza despacio"


class NavController:
    def __init__(self, args):
        self.args = args
        # Specialist de flechas (two-stage):
        #   --single-stage        -> None (ablacion: el main clasifica todo)
        #   --arrow-run <carpeta> -> ese specialist (para emparejar cada main con el
        #                            arrow de SU MISMA familia: comparacion en igualdad)
        #   por defecto           -> cnn_arrows (sistema desplegado)
        if getattr(args, "single_stage", False):
            arrow_run = None
        elif getattr(args, "arrow_run", None):
            arrow_run = args.arrow_run
        else:
            arrow_run = DEFAULT_ARROW_RUN
        self.clf = SignClassifier(run_dir=args.run_dir, arrow_run=arrow_run,
                                  crop_frac=args.crop_frac, view_w_frac=args.view_w_frac)
        self.policy = Policy(forward=args.forward, turn=args.turn, steer=args.steer,
                             search=args.search, conf_thresh=args.conf_thresh,
                             sign_conf_thresh=args.sign_conf_thresh,
                             arrow_conf_thresh=args.arrow_conf_thresh,
                             cross_interrupt_conf=args.cross_interrupt_conf,
                             pending_lost_frames=args.pending_lost_frames,
                             arrow_deg=args.arrow_deg, cross_deg=args.cross_deg,
                             wall_deg=args.wall_deg, turn_tol_deg=args.turn_tol_deg,
                             post_turn_s=args.post_turn_s,
                             turn_scale=args.turn_scale,
                             wall_probe_s=args.wall_probe_s,
                             wall_probe_speed=args.wall_probe_speed,
                             arrow_min_deg=args.arrow_min_deg,
                             arrow_max_deg=args.arrow_max_deg,
                             goal_reach_frac=args.goal_reach_frac,
                             goal_center_tol=args.goal_center_tol,
                             goal_confirm_cooldown=args.goal_confirm_cooldown,
                             goal_streak_conf=args.goal_streak_conf,
                             goal_streak_count=args.goal_streak_count,
                             turn_stop_on_open=args.turn_stop_on_open,
                             turn_open_thresh=args.turn_open_thresh,
                             turn_open_drop=args.turn_open_drop,
                             turn_min_open_deg=args.turn_min_open_deg,
                             search_default=args.search_default,
                             heading_grid=args.heading_grid, grid_deg=args.grid_deg,
                             heading_tol_deg=args.heading_tol_deg,
                             heading_hold_gain=args.heading_hold_gain,
                             heading_hold_deadzone_deg=args.heading_hold_deadzone_deg,
                             heading_recalib_gain=args.heading_recalib_gain,
                             heading_recalib_patience=args.heading_recalib_patience,
                             heading_recalib_max_deg=args.heading_recalib_max_deg,
                             turn_ramp_deg=args.turn_ramp_deg,
                             lost_go_straight=args.lost_go_straight)
        self.node = Node()
        self.pub = self.node.advertise(args.cmd_topic, Twist)
        self.lock = threading.Lock()
        self.latest = None          # ultimo frame RGB
        self.frame_id = 0           # contador de frames recibidos
        self.latest_yaw = None      # ultimo yaw (rad) por odometria
        self.latest_pos = (None, None)  # ultima posicion (x,y) por odometria
        self.latest_scan = None     # ultimo Scan del LIDAR (si --wall-source lidar)
        self.scan_id = 0            # contador de escaneos LIDAR recibidos
        # Pose CONTEMPORANEA al ultimo escaneo (la odom va a 50 Hz; al llegar el escaneo se
        # captura el yaw/pos del instante). Asi el mapa integra el escaneo con su pose de <=20 ms
        # en vez de la del frame de camara (~100 ms despues) -> sin smear rotacional en los giros.
        self.latest_scan_yaw = None
        self.latest_scan_pos = (None, None)
        # gate de rotacion del mapa: yaw/tiempo de la ultima integracion, para medir la velocidad
        # angular real y NO integrar paredes mientras el robot gira (rumbo inestable -> smear).
        self._map_prev_yaw = None
        self._map_prev_t = None
        self.votes = deque(maxlen=args.vote)
        self.last_cmd = (0.0, 0.0)  # ultimo (lin, ang) publicado
        self.node.subscribe(Image, args.image_topic, self._on_image)
        self.use_odom = args.odom_turn and HAS_ODOM
        if self.use_odom:
            self.node.subscribe(Odometry, args.odom_topic, self._on_odom)

        # nav-v26: FUENTE de deteccion de pared INTERCAMBIABLE -> camara (pixeles negros,
        # baseline) o LIDAR (distancias 360, ENCARRILA el robot en el pasillo + mapea). Ambas
        # cumplen el mismo contrato WallReading, asi que Policy.decide() NO cambia y la jerarquia
        # GOAL>FLECHA>CRUZ>LIDAR queda intacta. Si se pide LIDAR pero falta el binding, cae a camara.
        self.use_lidar = (args.wall_source == "lidar") and HAS_LIDAR
        if args.wall_source == "lidar" and not HAS_LIDAR:
            print("(--wall-source lidar: binding LaserScan no disponible -> uso camara)")
        if self.use_lidar:
            self.detector = LidarWallDetector(
                block_dist=args.lidar_block_dist, side_margin=args.lidar_side_margin,
                front_half_deg=args.lidar_front_half_deg, side_half_deg=args.lidar_side_half_deg,
                wall_present_dist=args.lidar_wall_present, align_gain=args.lidar_align_gain,
                align_max=args.lidar_align_max)
            self.node.subscribe(LaserScan, args.lidar_topic, self._on_scan)
        else:
            self.detector = CameraWallDetector(
                dark_thresh=args.wall_dark_thresh, center_block=args.wall_center_block,
                lower_frac=args.wall_lower_frac)
        # Se calcula lectura de pared si hay LIDAR (siempre) o si la camara la tiene activada
        self.wall_active = self.use_lidar or args.wall_dir
        # Mapa de ocupacion POR EJECUCION (solo LIDAR): apoyo visual, no influye en el control
        self.omap = OccupancyMap(res=args.map_res, size_m=args.map_size) if self.use_lidar else None
        self._show_map_on = bool(args.show_map) and self.use_lidar and HAS_CV2
        if args.show_map and self.use_lidar and not HAS_CV2:
            print("(--show-map: cv2 no disponible -> sin ventana de mapa; igual se guarda el PNG al terminar)")
        # Mapa SEMANTICO en vivo (--show-semantic): pinta el path por clase/confianza. Opt-in,
        # APAGADO por defecto -> las runs de comparacion no pagan overhead ni cambian de timing.
        self._show_semantic_on = bool(getattr(args, "show_semantic", False)) and self.use_lidar and HAS_CV2
        if getattr(args, "show_semantic", False) and self.use_lidar and not HAS_CV2:
            print("(--show-semantic: cv2 no disponible -> sin ventana del mapa semantico)")

        crop = "sin recorte" if args.crop_frac >= 1.0 else f"recorte central {args.crop_frac:.0%}"
        if args.view_w_frac < 1.0:
            crop += f" | franja central {args.view_w_frac:.0%} del ancho (ignora laterales)"
        if not args.odom_turn:
            odom = "giro por tiempo"
        elif self.use_odom:
            odom = f"giro por odometria ({args.odom_topic})"
        else:
            odom = "giro por tiempo (sin binding de odometria)"
        if self.use_lidar:
            wall = (f"pared: LIDAR {args.lidar_topic} (bloqueo<{args.lidar_block_dist:.1f}m) "
                    "+ encarrilado al pasillo + mapa por ejecucion")
        elif args.wall_dir:
            wall = ("pared: camara, franja inferior {:.0%} del alto".format(args.wall_lower_frac)
                    if args.wall_lower_frac < 1.0 else "pared: camara, todo el alto")
        else:
            wall = "pared: detector geometrico OFF"
        print(f"Suscrito a {args.image_topic}, publicando en {args.cmd_topic} | {crop} | {odom} | {wall}")
        if args.heading_grid:
            giros = f"rejilla {args.grid_deg:.0f}° (rumbo absoluto por odometria); flecha snapea a la rejilla"
        else:
            giros = (f"relativo; flecha {args.arrow_deg:.0f}-{args.arrow_deg + args.turn_tol_deg:.0f}°"
                     + (" + encarrilado LIDAR al pasillo" if self.use_lidar else ""))
        print(f"Giros: {giros}")

        # Ventana de debug en tiempo real (--show): visualiza lo que ve el detector de pared
        self._show_on = bool(args.show)
        if args.show and not HAS_CV2:
            print("(--show: cv2 no disponible -> sin ventana de debug)")
            self._show_on = False
        elif self._show_on:
            print("Ventana de debug de paredes ACTIVA (--show). Pulsa 'q' en la ventana para cerrarla.")

        # Log CSV de cada decision (vision + odometria + estado) para analisis posterior
        self.t_start = time.time()
        self.log_file = None
        self.log_writer = None
        self.log_path = None
        if args.log:
            log_dir = Path(args.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            path = log_dir / f"nav_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.log_path = path
            self.log_file = open(path, "w", newline="")
            self.log_writer = csv.writer(self.log_file)
            self.log_writer.writerow([
                "t", "frame", "cls", "conf", "state", "lin", "ang",
                "yaw_deg", "pos_x", "pos_y", "arrow_ang",
                "open_side", "dark_l", "dark_c", "dark_r",
                "goal_off", "goal_area", "note",
            ])
            print(f"Log de navegacion -> {path}")

    def _on_image(self, msg):
        """Callback de la camara (corre en hilo de gz). Decodifica y guarda el frame."""
        w, h = msg.width, msg.height
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        if buf.size == h * w * 3:                  # RGB_INT8
            rgb = buf.reshape(h, w, 3)
            with self.lock:
                self.latest = rgb.copy()
                self.frame_id += 1

    def _on_odom(self, msg):
        """Callback de odometria (hilo de gz). Guarda el yaw y la posicion del robot."""
        yaw = yaw_from_quat(msg.pose.orientation)
        p = msg.pose.position
        with self.lock:
            self.latest_yaw = yaw
            self.latest_pos = (float(p.x), float(p.y))

    def _on_scan(self, msg):
        """Callback del LIDAR (hilo de gz). Decodifica el LaserScan a un Scan (rangos+angulos)."""
        scan = make_scan(list(msg.ranges), msg.angle_min, msg.angle_step,
                         range_max=msg.range_max if msg.range_max > 0 else None)
        with self.lock:
            self.latest_scan = scan
            # snapshot de la pose del instante del escaneo (yaw/pos mas recientes de la odom 50Hz)
            self.latest_scan_yaw = self.latest_yaw
            self.latest_scan_pos = self.latest_pos
            self.scan_id += 1

    def _publish(self, lin, ang):
        msg = Twist()
        msg.linear.x = float(lin)
        msg.angular.z = float(ang)
        self.pub.publish(msg)

    def _log_row(self, t0, fid, cls, conf, lin, ang, yaw, pos,
                 arrow_ang, open_side, dark, goal_off, goal_area, note):
        """Escribe una fila del CSV: vision + odometria + estado de cada decision."""
        if self.log_writer is None:
            return
        darks = [f"{dark[0]:.3f}", f"{dark[1]:.3f}", f"{dark[2]:.3f}"] if dark is not None else ["", "", ""]
        self.log_writer.writerow([
            f"{t0 - self.t_start:.3f}", fid, cls, f"{conf:.3f}", self.policy.state,
            f"{lin:.3f}", f"{ang:.3f}",
            f"{math.degrees(yaw):.2f}" if yaw is not None else "",
            f"{pos[0]:.3f}" if pos[0] is not None else "",
            f"{pos[1]:.3f}" if pos[1] is not None else "",
            f"{arrow_ang:.1f}" if arrow_ang is not None else "",
            open_side or "", darks[0], darks[1], darks[2],
            f"{goal_off:.3f}" if goal_off is not None else "",
            f"{goal_area:.4f}", note,
        ])
        self.log_file.flush()

    def _vote(self, cls, conf):
        """Voto mayoritario sobre los ultimos `vote` frames (estabiliza ruido).

        Devuelve (clase_votada, conf_votada) donde conf_votada es la confianza
        MEDIA de los frames que coinciden con la clase ganadora; asi la confianza
        que ve la politica corresponde de verdad a la clase elegida (necesario
        para el gating de direccionales). Con --vote 1 equivale a (cls, conf).
        """
        self.votes.append((cls, conf))
        counts = Counter(c for c, _ in self.votes)
        voted = counts.most_common(1)[0][0]
        confs = [cf for c, cf in self.votes if c == voted]
        return voted, sum(confs) / len(confs)

    def _confirm_goal(self, cls, conf, goal_area):
        """Pregunta por terminal si la META detectada es real. Devuelve True si el
        usuario confirma que se ha terminado; False si era un falso positivo.

        Por seguridad ante falsos positivos, SOLO una respuesta afirmativa explicita
        (s/si/y/yes) termina la run; cualquier otra cosa (incluido Enter) la reanuda.
        Si no hay terminal interactiva (stdin cerrado), se da por alcanzada para no
        colgar una ejecucion desatendida."""
        prompt = (f"\n>>> META detectada ({cls} conf={conf:.2f}, area {goal_area*100:.1f}%). "
                  "¿Se ha alcanzado realmente la diana? [s/N] ")
        try:
            ans = input(prompt).strip().lower()
        except EOFError:
            print("(sin terminal interactiva -> se da por alcanzada)", flush=True)
            return True
        return ans in ("s", "si", "sí", "y", "yes")

    def run(self):
        period = 1.0 / self.args.rate
        last_proc = 0  # ultimo frame_id procesado
        print(f"Control a {self.args.rate} Hz. Ctrl+C para parar.\n")
        try:
            while True:
                t0 = time.time()
                with self.lock:
                    fid = self.frame_id
                    frame = self.latest
                    yaw = self.latest_yaw
                    pos = self.latest_pos
                    scan = self.latest_scan
                    scan_yaw = self.latest_scan_yaw    # pose CONTEMPORANEA al escaneo (para el mapa)
                    scan_pos = self.latest_scan_pos

                if frame is None:
                    time.sleep(period)
                    continue

                if fid != last_proc:
                    # Frame nuevo -> inferir, votar y decidir
                    last_proc = fid
                    cls, conf, _ = self.clf.predict(frame)
                    voted, conf = self._vote(cls, conf)
                    # Lectura de PARED por la fuente activa (camara=frame, LIDAR=scan); mismo contrato
                    open_side, dark, center_blocked, center_dark = (None, None, False, None)
                    if self.wall_active:
                        if self.use_lidar:
                            reading = self.detector.read(scan) if scan is not None else None
                        else:
                            reading = self.detector.read(frame)
                        if reading is not None:        # (LIDAR aun sin primer escaneo -> reading None)
                            open_side, center_blocked = reading.open_side, reading.center_blocked
                            dark, center_dark = reading.sides, reading.center_dark
                    arrow_ang = None
                    if self.args.arrow_vector and voted in ("ARROW_LEFT", "ARROW_RIGHT"):
                        arrow_ang = arrow_angle(frame)
                    goal_off, goal_area = (None, 0.0)
                    if self.args.goal_homing:
                        goal_off, goal_area = goal_in_view(frame)
                    lin, ang, done, note = self.policy.decide(
                        voted, conf, t0, open_side=open_side, yaw=yaw, arrow_ang=arrow_ang,
                        goal_off=goal_off, goal_area=goal_area, center_blocked=center_blocked,
                        center_dark=center_dark, dark=dark)
                    # nav-v26: ENCARRILADO por LIDAR (prioridad MAS BAJA). Solo cuando la politica va
                    # RECTO en FOLLOW (FREE_PATH/ARROW_UP, sin flecha pendiente, sin pared de frente, sin
                    # homing de meta a la vista) se suma una correccion SUAVE para centrar/enfilar el robot
                    # en el pasillo. Respeta GOAL>FLECHA>CRUZ>LIDAR: con maniobra superior NO es recto en
                    # FOLLOW -> no corrige. En cruces (una pared desaparece) la correccion da ~0 -> recto.
                    if (self.use_lidar and scan is not None and self.detector.provides_alignment
                            and not done and self.policy.state == Policy.FOLLOW
                            and self.policy.pending is None and not center_blocked
                            and voted in ("FREE_PATH", "ARROW_UP")
                            and (goal_off is None or goal_area < self.policy.goal_reach_frac)):
                        align = self.detector.corridor_correction(scan, yaw)
                        if abs(align) > 1e-3:
                            ang = max(-self.policy.steer, min(self.policy.steer, ang + align))
                            note += f" | encarril {align:+.2f}"
                    self.last_cmd = (lin, ang)
                    if not self.args.dry_run:
                        self._publish(lin, ang)
                    if self.omap is not None and scan is not None:
                        # mapa por ejecucion (no afecta al control). Se integra con la pose
                        # CONTEMPORANEA al escaneo (no la del frame). Ademas, con --map-turn-gate solo se
                        # acumulan PAREDES cuando el RUMBO ES ESTABLE: se mide la velocidad angular real
                        # (entre escaneos consecutivos) y si supera --map-gate-degps el escaneo no integra
                        # paredes (giro/correccion -> saldrian abanicadas). El rastro se marca siempre.
                        integrate = True
                        if self.args.map_turn_gate and scan_yaw is not None:
                            if self._map_prev_yaw is not None and self._map_prev_t is not None:
                                dt = max(1e-3, t0 - self._map_prev_t)
                                omega = abs(math.degrees(ang_diff(scan_yaw, self._map_prev_yaw))) / dt
                                integrate = omega <= self.args.map_gate_degps
                            self._map_prev_yaw = scan_yaw   # CADA escaneo (mide la TASA, no el acumulado)
                            self._map_prev_t = t0
                        self.omap.update(scan, scan_pos, scan_yaw, integrate_walls=integrate)
                    if self._show_semantic_on and self.omap is not None:
                        # capa semantica (solo con el flag): pinta la pose con el color de la clase
                        self.omap.mark_semantic(pos, semantic_color(voted, conf))
                    if dark is not None and voted in ("WALL", "WALL_LEFT", "WALL_RIGHT"):
                        unit = "dist LCR(m)" if self.use_lidar else "negro LCR"
                        note += f" | {unit}={dark[0]:.2f}/{dark[1]:.2f}/{dark[2]:.2f}"
                    if voted == "GOAL":
                        note += f" | racha GOAL {self.policy.goal_streak}/{self.policy.goal_streak_count}"
                    print(f"{voted:11s} conf={conf:.2f} | v={lin:+.2f} w={ang:+.2f} | {note}",
                          flush=True)
                    self._log_row(t0, fid, voted, conf, lin, ang, yaw, pos,
                                  arrow_ang, open_side, dark, goal_off, goal_area, note)
                    if self._show_on and not self.use_lidar:   # debug de pixeles negros (solo camara)
                        try:
                            dbg = make_wall_debug(
                                frame, self.args.wall_dark_thresh, self.args.wall_lower_frac,
                                self.args.view_w_frac, dark, open_side, center_blocked,
                                voted, conf, self.policy.state, scale=self.args.show_scale)
                            cv2.imshow("robot - paredes (debug)", dbg)
                            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                                print("\n>>> Ventana de debug cerrada ('q'). Sigo navegando sin ventana.")
                                cv2.destroyAllWindows()
                                self._show_on = False
                        except Exception as e:
                            print(f"(--show: no se pudo mostrar la ventana: {e}; desactivado)")
                            self._show_on = False
                    if self._show_map_on and self.omap is not None:   # mapa LIDAR en vivo (--show-map)
                        try:
                            cv2.imshow("robot - mapa LIDAR", self.omap.render())
                            cv2.waitKey(1)
                        except Exception as e:
                            print(f"(--show-map: no se pudo mostrar el mapa: {e}; desactivado)")
                            self._show_map_on = False
                    if self._show_semantic_on and self.omap is not None:   # mapa SEMANTICO en vivo
                        try:
                            cv2.imshow("robot - mapa semantico", self.omap.render_semantic())
                            cv2.waitKey(1)
                        except Exception as e:
                            print(f"(--show-semantic: no se pudo mostrar el mapa: {e}; desactivado)")
                            self._show_semantic_on = False
                    if done:
                        self._publish(0.0, 0.0)   # detiene el robot mientras se confirma
                        if self.args.goal_confirm and not self._confirm_goal(voted, conf, goal_area):
                            # Falso positivo: ignora la diana un rato y sigue navegando
                            self.policy.reject_goal(time.time())
                            print(">>> Meta DESCARTADA. Reanudando navegacion "
                                  f"(ignoro la diana {self.policy.goal_confirm_cooldown:.0f}s).\n",
                                  flush=True)
                            continue
                        print("\n>>> Laberinto resuelto. Robot detenido.")
                        break
                else:
                    # Sin frame nuevo: re-publica el ultimo comando (sigue moviendose)
                    if not self.args.dry_run:
                        self._publish(*self.last_cmd)

                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
        except KeyboardInterrupt:
            print("\nInterrumpido. Deteniendo el robot.")
        finally:
            self._publish(0.0, 0.0)
            if (self._show_on or self._show_map_on or self._show_semantic_on) and HAS_CV2:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            if self.log_file is not None:
                self.log_file.close()
                print("Log guardado.")
            # Mapa de ocupacion del LIDAR de ESTA ejecucion (apoyo visual): PNG + .npz
            if self.omap is not None and self.omap.updates > 0:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_dir = Path(self.args.log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                png = log_dir / f"lidar_map_{ts}.png"
                npz = log_dir / f"lidar_map_{ts}.npz"
                try:
                    written = self.omap.save(png, npz)
                    print(f"Mapa LIDAR guardado -> {', '.join(written)}")
                except Exception as e:
                    print(f"(mapa LIDAR: fallo al guardar: {e})")
            if self.args.analyze:
                self._run_analysis()

    def _run_analysis(self):
        """Si se paso --analyze, ejecuta analyze_log.py sobre el log de ESTA run
        al terminar (analiza la ruta exacta que se acaba de escribir, no 'el mas
        reciente'). Sin --analyze no se ejecuta nada. Reenvia --analyze-label
        como --label para etiquetar la run en la tabla de comparacion."""
        if self.log_path is None:
            print("(--analyze ignorado: no hay log que analizar, ¿--no-log?)")
            return
        script = Path(__file__).resolve().parent.parent / "Modelos" / "src" / "analyze_log.py"
        if not script.is_file():
            print(f"(--analyze: no encuentro {script})")
            return
        cmd = [sys.executable, str(script), "--log", str(self.log_path)]
        if self.args.analyze_label:
            cmd += ["--label", self.args.analyze_label]
        print(f"\n>>> Analizando el log: {' '.join(cmd)}\n", flush=True)
        try:
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"(--analyze: fallo al ejecutar analyze_log.py: {e})")


DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def parse_args():
    p = argparse.ArgumentParser(description="Navegacion del robot por clasificacion de imagenes")
    p.add_argument("--run-dir", default=None, help="Carpeta del modelo (def: Modelos/runs/cnn_improved)")
    p.add_argument("--single-stage", action="store_true",
                   help="Desactiva el specialist de flechas (cnn_arrows): el modelo de --run-dir clasifica TODAS las clases. Ablacion para medir cuanto aporta el specialist")
    p.add_argument("--arrow-run", default=None,
                   help="Carpeta del specialist de flechas (def: cnn_arrows). Para comparar familias en igualdad, empareja con el arrow de la misma familia, p.ej. --run-dir Modelos/runs/rnn_unificado --arrow-run Modelos/runs/rnn_arrows")
    p.add_argument("--image-topic", default="/camera/image_raw")
    p.add_argument("--cmd-topic", default="/cmd_vel")
    p.add_argument("--crop-frac", type=float, default=1.0,
                   help="Recorte central del frame en vivo (p.ej. 0.6). 1.0 = sin recorte")
    p.add_argument("--view-w-frac", type=float, default=1.0,
                   help="Franja central en ANCHO que ve el robot (recorta los laterales, "
                        "mantiene el alto). Se aplica al frame entero ANTES del gate de verde, "
                        "asi ni el main ni el specialist ven senales a los extremos. 1.0 = sin recorte")
    p.add_argument("--rate", type=float, default=10.0, help="Hz del bucle de control (camara a 10 Hz)")
    p.add_argument("--vote", type=int, default=3,
                   help="Voto mayoritario sobre N frames distintos (estabiliza la decision)")
    p.add_argument("--conf-thresh", type=float, default=0.4,
                   help="Umbral de confianza para ACTUAR (avanzar/corregir)")
    p.add_argument("--sign-conf-thresh", type=float, default=0.55,
                   help="Umbral (mayor) para COMPROMETER la CRUZ (atendida por el main classifier)")
    p.add_argument("--arrow-conf-thresh", type=float, default=0.95,
                   help="Umbral (alto) para COMPROMETER una FLECHA. Con el specialist (cnn_arrows) la conf en vivo llega a ~1.0, asi que puede subirse mucho para rechazar respuestas dudosas sin perder maniobras")
    p.add_argument("--cross-interrupt-conf", type=float, default=0.85,
                   help="Conf minima para que una CRUZ interrumpa un giro en curso y lo reconvierta a 180")
    p.add_argument("--pending-lost-frames", type=int, default=3,
                   help="Frames seguidos sin ver el cartel para girar una flecha pendiente (al perderla de vista)")
    p.add_argument("--wall-source", choices=("camera", "lidar"), default="lidar",
                   help="nav-v26: FUENTE de deteccion de pared. 'lidar' (def) = distancias 360, encarrila el robot en el pasillo y mapea por ejecucion. 'camera' = pixeles negros (baseline nav-v6..v25, ablacion). Misma jerarquia GOAL>FLECHA>CRUZ>pared en ambos")
    p.add_argument("--wall-dir", action=argparse.BooleanOptionalAction, default=True,
                   help="(camara) Detector GEOMETRICO de direccion de pared (negro por franjas) para elegir el giro sin señal. Ignorado con --wall-source lidar")
    p.add_argument("--wall-dark-thresh", type=float, default=50.0,
                   help="(camara) Brillo (0-255) por debajo del cual un pixel cuenta como pared/negro")
    p.add_argument("--wall-center-block", type=float, default=0.5,
                   help="(camara) Fraccion de negro en la franja central para considerar pared de frente")
    p.add_argument("--wall-lower-frac", type=float, default=0.5,
                   help="(camara) nav-v21: fraccion INFERIOR del alto del frame que usa el detector de pared (1.0 = todo el alto). Mirar solo la parte baja mide el espacio navegable CERCANO (suelo vs pared inmediata) e ignora paredes lejanas/techo de la parte alta que confundian los cruces. Solo afecta al detector geometrico de pared, no a los clasificadores (eso es --view-w-frac)")
    # --- LIDAR (--wall-source lidar): deteccion de pared por distancias + encarrilado + mapa ---
    p.add_argument("--lidar-topic", default="/lidar", help="(lidar) Topic del LaserScan 2D (gz.msgs.LaserScan)")
    p.add_argument("--lidar-block-dist", type=float, default=1.5,
                   help="(lidar) Distancia frontal (m) por debajo de la cual hay PARED de frente (center_blocked). Calibrado al robot (~1 m de largo) y al corredor de 4 m; subir si frena tarde, bajar si frena demasiado lejos")
    p.add_argument("--lidar-side-margin", type=float, default=0.3,
                   help="(lidar) Diferencia (m) entre hueco izq/der para decidir el lado abierto de una pared")
    p.add_argument("--lidar-front-half-deg", type=float, default=30.0,
                   help="(lidar) Semiancho (grados) del sector FRONTAL para medir el bloqueo de frente")
    p.add_argument("--lidar-side-half-deg", type=float, default=45.0,
                   help="(lidar) Semiancho (grados) de los sectores laterales (centrados en +-90) para medir las paredes")
    p.add_argument("--lidar-wall-present", type=float, default=4.0,
                   help="(lidar) Distancia (m) por debajo de la cual se considera que ESA pared existe (para encarrilar). Calibrado al corredor de 4 m (paredes a ~2 m del centro; ~3.8 m si el robot se pega a una). En un cruce las paredes se alejan (>4 m) -> no se encarrila -> recto")
    p.add_argument("--lidar-align-gain", type=float, default=1.0,
                   help="(lidar) Escala global del ENCARRILADO al pasillo (0 = desactiva el encarrilado, solo deteccion de pared)")
    p.add_argument("--lidar-align-max", type=float, default=0.4,
                   help="(lidar) Saturacion (rad/s) de la correccion de encarrilado")
    p.add_argument("--map-res", type=float, default=0.1,
                   help="(lidar) Resolucion del mapa de ocupacion en metros/celda")
    p.add_argument("--map-size", type=float, default=40.0,
                   help="(lidar) Lado del mapa de ocupacion en metros (rejilla cuadrada centrada en el spawn)")
    p.add_argument("--show-map", action=argparse.BooleanOptionalAction, default=False,
                   help="(lidar) Ventana en vivo del mapa de ocupacion mientras navega (requiere cv2 y entorno grafico)")
    p.add_argument("--show-semantic", action=argparse.BooleanOptionalAction, default=False,
                   help="(lidar) Ventana en vivo del MAPA SEMANTICO: path coloreado por clase detectada "
                        "(cruz=rojo, goal=azul, up=verde, der=turquesa, izq=pistacho), saturacion=confianza. "
                        "Opt-in; apagado no afecta a las runs. El mismo mapa se regenera por post-proceso con semantic_map.py")
    p.add_argument("--map-turn-gate", action=argparse.BooleanOptionalAction, default=True,
                   help="(lidar) Acumular paredes en el mapa SOLO con el rumbo estable: si la velocidad angular "
                        "real supera --map-gate-degps, el escaneo no integra paredes (giro/correccion -> saldrian "
                        "abanicadas). El rastro del robot se sigue marcando. `--no-map-turn-gate` = integrar siempre.")
    p.add_argument("--map-gate-degps", type=float, default=6.0,
                   help="(lidar) Umbral de velocidad angular (grados/s) por debajo del cual SI se integran paredes "
                        "en el mapa. Mas bajo = mapa mas FIJO (menos cobertura en las esquinas). Def 6.")
    p.add_argument("--odom-turn", action=argparse.BooleanOptionalAction, default=True,
                   help="Completar los giros por ANGULO REAL medido con odometria (robusto al RTF); si no, por tiempo")
    p.add_argument("--odom-topic", default="/model/vehicle_blue/odometry",
                   help="Topic de odometria del robot (gz.msgs.Odometry)")
    p.add_argument("--heading-grid", action=argparse.BooleanOptionalAction, default=None,
                   help="nav-v25: anclar el rumbo del robot a la rejilla de 8 direcciones (45 grados) con el yaw ABSOLUTO de la odometria. Cada giro aterriza en un rumbo CANONICO (corrige la deriva en vez de acumularla) y en recto se mantiene ese rumbo (heading-hold). nav-v26: el DEFAULT depende de --wall-source -> ON con 'camera' (la rejilla alinea), OFF con 'lidar' (el ENCARRILADO del LIDAR alinea al pasillo, no hace falta rejilla). --heading-grid/--no-heading-grid lo fuerza")
    p.add_argument("--grid-deg", type=float, default=45.0,
                   help="nav-v25: paso de la rejilla de rumbos en grados (45 = 8 direcciones N/NE/E/SE/S/SO/O/NO)")
    p.add_argument("--heading-tol-deg", type=float, default=6.0,
                   help="nav-v25: tolerancia (grados) para dar por terminado un giro al alcanzar el rumbo objetivo de la rejilla")
    p.add_argument("--heading-hold-gain", type=float, default=1.0,
                   help="nav-v25: ganancia del heading-hold (residuo al rumbo de rejilla en rad -> correccion angular, limitada por --steer). Mantiene el rumbo del pasillo en recto")
    p.add_argument("--heading-hold-deadzone-deg", type=float, default=3.0,
                   help="nav-v25: residuo (grados) por debajo del cual el heading-hold NO corrige (evita micro-oscilacion cuando ya esta alineado)")
    p.add_argument("--heading-recalib-gain", type=float, default=0.1,
                   help="nav-v25: cuanto se acerca el ANCLA de la rejilla al residuo en cada frame recto confiable (low-pass; bloquea la rejilla a los pasillos reales)")
    p.add_argument("--heading-recalib-patience", type=int, default=5,
                   help="nav-v25: frames rectos (FREE_PATH/ARROW_UP) confiables seguidos antes de empezar a recalibrar el ancla")
    p.add_argument("--heading-recalib-max-deg", type=float, default=20.0,
                   help="nav-v25: residuo maximo (grados) para recalibrar el ancla; por encima se asume que NO es deriva (es un giro real) y no se toca")
    p.add_argument("--turn-ramp-deg", type=float, default=25.0,
                   help="nav-v25: banda (grados) de rampa proporcional del giro cerca del rumbo objetivo: el angular baja al acercarse para amortiguar el sobre-giro por el delay de Gazebo")
    p.add_argument("--goal-homing", action=argparse.BooleanOptionalAction, default=True,
                   help="Dirigirse hacia la DIANA (azul) en vez de parar al verla; para solo al llegar (cerca y centrada)")
    p.add_argument("--goal-reach-frac", type=float, default=0.08,
                   help="Fraccion de azul en el frame para dar la META por alcanzada (mas alto = hay que acercarse mas)")
    p.add_argument("--goal-center-tol", type=float, default=0.30,
                   help="|offset| maximo de la diana para considerarla centrada (parar/avanzar)")
    p.add_argument("--goal-confirm", action=argparse.BooleanOptionalAction, default=False,
                   help="Al dar el laberinto por resuelto, PREGUNTAR por terminal si de verdad se ha alcanzado (opt-in). Desde nav-v16 el fin lo decide la RACHA de GOAL (--goal-streak-count), que ya es robusta, asi que por defecto se para en automatico (off)")
    p.add_argument("--goal-confirm-cooldown", type=float, default=6.0,
                   help="Segundos que se ignora la diana tras rechazar una META (para alejarse del falso positivo sin re-disparar)")
    p.add_argument("--goal-streak-conf", type=float, default=0.85,
                   help="Conf minima de la clase GOAL para sumar al contador de finalizacion (la racha)")
    p.add_argument("--goal-streak-count", type=int, default=5,
                   help="Detecciones GOAL CONSECUTIVAS (>--goal-streak-conf) para dar el laberinto por resuelto y parar")
    p.add_argument("--log", action=argparse.BooleanOptionalAction, default=True,
                   help="Guardar un CSV por ejecucion (vision + odometria + estado de cada decision)")
    p.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR),
                   help="Carpeta de los logs (def: TFG_v2/logs)")
    p.add_argument("--analyze", action="store_true",
                   help="Al TERMINAR la run, ejecuta Modelos/src/analyze_log.py sobre el log de esta ejecucion. Sin este flag no se analiza nada al acabar.")
    p.add_argument("--analyze-label", default=None,
                   help="Etiqueta (familia) que se pasa a analyze_log.py como --label cuando se usa --analyze (p.ej. cnn, mlp). Registra la run en la tabla de comparacion.")
    p.add_argument("--forward", type=float, default=0.3,
                   help="Velocidad lineal al avanzar (m/s).")
    p.add_argument("--turn", type=float, default=0.6)
    p.add_argument("--steer", type=float, default=0.3)
    p.add_argument("--search", type=float, default=0.4)
    p.add_argument("--lost-go-straight", action=argparse.BooleanOptionalAction, default=True,
                   help="Sin señal fiable para decidir un cruce, SEGUIR RECTO (el LIDAR sigue el pasillo) en vez de "
                        "pararse a girar buscando una señal. Solo gira si hay PARED de frente (center_blocked). "
                        "`--no-lost-go-straight` = comportamiento clasico de cámara (girar para buscar).")
    p.add_argument("--search-default", type=int, choices=(-1, 1), default=-1,
                   help="nav-v22: sentido de busqueda/rompe-bucles SOLO cuando no hay geometria (ni dark ni open_side). +1=izquierda, -1=derecha. Por defecto -1 para quitar el viejo sesgo fijo a la izquierda")
    p.add_argument("--arrow-deg", type=float, default=None,
                   help="Grados a girar con una flecha (NOMINAL/fallback si no se usa el vector). nav-v26: DEFAULT segun --wall-source -> 90 con 'camera' (la rejilla snapea), 85 con 'lidar' (con la ventana --turn-tol-deg el giro cae en ~85-95 y luego el ENCARRILADO del LIDAR alinea al pasillo)")
    p.add_argument("--arrow-vector", action=argparse.BooleanOptionalAction, default=None,
                   help="Estimar el angulo de giro del VECTOR de la flecha verde (magnitud adaptativa); si no, usa --arrow-deg fijo. nav-v26: DEFAULT ON con 'camera', OFF con 'lidar' (giro de flecha nominal CONSISTENTE ~85-95, sin depender del vector; el LIDAR alinea despues)")
    p.add_argument("--arrow-min-deg", type=float, default=25.0,
                   help="Clamp inferior del giro de flecha por vector")
    p.add_argument("--arrow-max-deg", type=float, default=135.0,
                   help="Clamp superior del giro de flecha por vector")
    p.add_argument("--cross-deg", type=float, default=180.0, help="Grados a girar con una cruz")
    p.add_argument("--wall-deg", type=float, default=90.0, help="Grados a girar al toparse una pared")
    p.add_argument("--turn-tol-deg", type=float, default=None,
                   help="Margen EXTRA por encima del objetivo: gira de obj a obj+tol (nunca menos), completa al despejar. nav-v26: DEFAULT 20 con 'camera', 10 con 'lidar' (con --arrow-deg 85 -> flecha en 85-95)")
    p.add_argument("--turn-stop-on-open", action=argparse.BooleanOptionalAction, default=True,
                   help="Parar el giro al revelarse un pasillo despejado por delante (giro adaptativo: 45 en vez de 90)")
    p.add_argument("--turn-open-thresh", type=float, default=0.10,
                   help="Negro del centro por debajo del cual el frente se considera despejado (para parar el giro; mas bajo = mas alineado)")
    p.add_argument("--turn-open-drop", type=float, default=0.20,
                   help="Caida minima del negro central (vs inicio del giro) para considerar que se ABRE un pasillo nuevo")
    p.add_argument("--turn-min-open-deg", type=float, default=None,
                   help="Giro minimo antes de permitir parar por apertura de pasillo (evita parar al instante). nav-v26: DEFAULT 20 con 'camera', 85 con 'lidar' (la flecha gira al menos ~85 antes de poder cerrar el giro al abrirse el pasillo)")
    p.add_argument("--post-turn-s", type=float, default=1.5,
                   help="Enfriamiento tras girar: ignora señales N s para alejarse del cartel")
    p.add_argument("--wall-probe-s", type=float, default=2.0,
                   help="Ante una pared, avanza despacio buscando señal N s antes de girar")
    p.add_argument("--wall-probe-speed", type=float, default=0.3,
                   help="Fraccion de la velocidad de avance durante el sondeo de pared")
    p.add_argument("--turn-scale", type=float, default=1.0,
                   help="Factor de calibracion de la duracion de giro (si gira de mas/menos)")
    p.add_argument("--show", action="store_true",
                   help="Abre una VENTANA en tiempo real con lo que ve el DETECTOR DE PARED: franja inferior analizada, pixeles negros (pared) en rojo, las 3 franjas izq/centro/der y sus valores, lado abierto y bloqueo de frente. Requiere cv2 y entorno grafico (WSLg). Pulsa 'q' en la ventana para cerrarla")
    p.add_argument("--show-scale", type=int, default=3,
                   help="Factor de ampliacion de la ventana de --show (la camara es 320x200)")
    p.add_argument("--dry-run", action="store_true", help="No publica cmd_vel, solo imprime decisiones")
    args = p.parse_args()
    # nav-v26: el MODO DE GIRO depende de la fuente de pared (defaults que dejaste en None se
    # resuelven aqui; pasar el flag explicitamente lo fuerza en cualquiera de los dos modos):
    #   - camera: rejilla nav-v25 ON -> los giros aterrizan en multiplos de --grid-deg (la rejilla
    #             alinea el rumbo); flecha por VECTOR (magnitud adaptativa) que la rejilla snapea.
    #   - lidar : rejilla OFF -> giro RELATIVO; el robot no se alinea por rejilla sino que el
    #             ENCARRILADO del LIDAR lo enfila al pasillo al que apunta la flecha TRAS girar. Por eso
    #             la flecha gira ~85-95 (nominal 85, ventana 10) de forma CONSISTENTE (sin vector) y no
    #             cierra el giro antes de ~85 (turn-min-open-deg 85).
    lidar = (args.wall_source == "lidar")
    if args.heading_grid is None:
        args.heading_grid = not lidar
    if args.arrow_vector is None:
        args.arrow_vector = not lidar
    if args.arrow_deg is None:
        args.arrow_deg = 85.0 if lidar else 90.0
    if args.turn_tol_deg is None:
        args.turn_tol_deg = 10.0 if lidar else 20.0
    if args.turn_min_open_deg is None:
        args.turn_min_open_deg = 85.0 if lidar else 20.0
    if args.run_dir is None:
        from cnn_infer import DEFAULT_RUN
        args.run_dir = str(DEFAULT_RUN)
    else:
        args.run_dir = _resolve_run_dir(args.run_dir)
    if args.arrow_run is not None:
        args.arrow_run = _resolve_run_dir(args.arrow_run)
    return args


def _resolve_run_dir(run_dir):
    """run_navigate.sh hace `cd Robot/`, asi que una ruta relativa se resuelve
    respecto a Robot/. Si ahi no existe, se prueba relativa a la raiz del repo
    (TFG_v2/), que es como se documentan las rutas (p.ej. Modelos/runs/...)."""
    rd = Path(run_dir)
    if not rd.is_absolute() and not (rd / "config.json").is_file():
        cand = Path(__file__).resolve().parent.parent / run_dir
        if (cand / "config.json").is_file():
            return str(cand)
    return run_dir


if __name__ == "__main__":
    NavController(parse_args()).run()
