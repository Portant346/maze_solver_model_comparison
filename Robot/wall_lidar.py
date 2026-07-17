"""Detector de PARED por LIDAR 2D -- modulo intercambiable (alternativa a la camara).

Motivacion (nav-v26): los pixeles negros de la camara (`wall_camera.py`) son fragiles
(dependen de iluminacion/viewpoint, alcance = el de la camara, confunden cruces con
paredes lejanas). Un LIDAR 2D ve 360 grados AL INSTANTE, mide DISTANCIAS reales y mucho
mas lejos -> permite ENCARRILAR el robot en el pasillo sin rotar. Al ir bien enfilado, la
camara lee mejor las señales y en los cruces el robot sigue recto hasta que una señal mande.

Este modulo expone el MISMO contrato `WallReading` que el de camara, de modo que
`Policy.decide()` no distingue la fuente (jerarquia INTACTA: GOAL > FLECHA > CRUZ > LIDAR;
el LIDAR ocupa el hueco de baja prioridad que antes ocupaban los pixeles negros). Ademas
aporta:
  - corridor_correction(): correccion angular SUAVE para centrar/enfilar el robot en el
    pasillo (lo que la camara no hacia). Solo actua con DOS paredes presentes (en un cruce
    una pared desaparece -> no corrige -> recto). Se aplica en navigate.py SOLO cuando no
    hay maniobra de señal/meta activa (prioridad mas baja).
  - OccupancyMap: mapa de ocupacion 2D POR EJECUCION (marco de odometria) que se guarda al
    terminar (PNG + .npy) como APOYO VISUAL. NO influye en el control ni persiste entre runs.

El detector trabaja sobre un `Scan` ya decodificado (rangos + angulos en numpy), no sobre el
mensaje de Gazebo, para poder testearse OFFLINE sin gz (igual que test_nav_*.py).
"""
import math
from collections import namedtuple

import numpy as np
try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

from wall_camera import WallReading


# Escaneo LIDAR ya decodificado y normalizado. angles en RADIANES, marco del robot:
# 0 = FRENTE (+x), positivo = ANTIHORARIO (izquierda), [-pi, pi]. ranges en metros.
Scan = namedtuple("Scan", ["ranges", "angles", "range_max"])


def make_scan(ranges, angle_min, angle_step, range_max=None):
    """Construye un `Scan` desde arrays crudos (lo que llega del LaserScan de gz).

    Normaliza los angulos a (-pi, pi] (0 = frente) para que la sectorizacion sea
    independiente de si el sensor barre [-pi,pi] o [0,2pi]. range_max se infiere del
    rango maximo finito si no se da.
    """
    ranges = np.asarray(ranges, dtype=np.float64)
    n = ranges.shape[0]
    angles = angle_min + np.arange(n) * angle_step
    angles = np.arctan2(np.sin(angles), np.cos(angles))   # -> (-pi, pi]
    if range_max is None:
        finite = ranges[np.isfinite(ranges)]
        range_max = float(finite.max()) if finite.size else 10.0
    return Scan(ranges=ranges, angles=angles, range_max=float(range_max))


class LidarWallDetector:
    """Detector de pared por LIDAR con el contrato comun `WallReading` + encarrilado.

    Sectoriza el escaneo en FRENTE / IZQUIERDA / DERECHA y mide la DISTANCIA libre
    (clearance) de cada sector. center_blocked si el frente esta mas cerca que
    `block_dist`; open_side = lado con mas hueco (si la diferencia supera `side_margin`).
    Esto es el espejo geometrico del detector de camara (alli "mas despejado = menos
    negro"; aqui "mas despejado = mas distancia").
    """
    source = "lidar"
    provides_alignment = True      # el LIDAR SI encarrila (lo que la camara no hacia)
    provides_map = True

    def __init__(self, block_dist=1.5, side_margin=0.3, front_half_deg=30.0,
                 side_half_deg=45.0, wall_present_dist=2.5,
                 align_gain=1.0, align_lat=0.4, align_head=0.8, align_max=0.4):
        self.block_dist = block_dist                 # frente mas cerca que esto -> pared de frente
        self.side_margin = side_margin               # |izq-der| (m) minimo para decidir el lado abierto
        self.front_half = math.radians(front_half_deg)   # semiancho del sector frontal
        self.side_half = math.radians(side_half_deg)      # semiancho de los sectores laterales (centrados en +-90)
        self.wall_present_dist = wall_present_dist    # una pared "existe" si su clearance < esto
        self.align_gain = align_gain                  # escala global del encarrilado (--lidar-align-gain)
        self.align_lat = align_lat                    # peso del termino LATERAL (centrado)
        self.align_head = align_head                  # peso del termino de ORIENTACION (paralelo)
        self.align_max = align_max                    # saturacion de la correccion (rad/s)

    # -- helpers de sector -------------------------------------------------
    def _sector(self, scan, center):
        """(clearance_min, angulos, rangos_saneados) del sector centrado en `center`.

        Rangos invalidos (0, inf, NaN) o sin retorno se tratan como range_max
        (= sin obstaculo en esa direccion)."""
        lo, hi = center - self.side_half, center + self.side_half
        # diferencia angular a center para seleccionar el sector (robusto al wrap)
        d = np.arctan2(np.sin(scan.angles - center), np.cos(scan.angles - center))
        sel = np.abs(d) <= self.side_half
        if not sel.any():
            return scan.range_max, np.array([]), np.array([])
        a = scan.angles[sel]
        r = scan.ranges[sel].astype(np.float64).copy()
        bad = ~np.isfinite(r) | (r <= 0.0)
        r[bad] = scan.range_max
        r = np.clip(r, 0.0, scan.range_max)
        return float(r.min()), a, r

    def _front_min(self, scan):
        d = np.arctan2(np.sin(scan.angles), np.cos(scan.angles))   # dist a 0 (frente)
        sel = np.abs(d) <= self.front_half
        if not sel.any():
            return scan.range_max
        r = scan.ranges[sel].astype(np.float64).copy()
        bad = ~np.isfinite(r) | (r <= 0.0)
        r[bad] = scan.range_max
        return float(np.clip(r, 0.0, scan.range_max).min())

    # -- contrato comun ----------------------------------------------------
    def read(self, scan):
        """Devuelve un WallReading desde el escaneo (mismo contrato que la camara)."""
        front = self._front_min(scan)
        left, _, _ = self._sector(scan, math.pi / 2.0)
        right, _, _ = self._sector(scan, -math.pi / 2.0)
        center_blocked = front < self.block_dist
        open_side = None
        if center_blocked and abs(left - right) >= self.side_margin:
            open_side = "left" if left > right else "right"
        # center_dark: proxy en [0,1] (1 = pegado) para el fallback de giro sin odometria
        ref = max(self.block_dist * 2.0, 1e-6)
        center_dark = float(np.clip(1.0 - front / ref, 0.0, 1.0))
        # sides en METROS (izq, centro=frente, der): el log las guarda tal cual (distancias)
        return WallReading(open_side=open_side, center_blocked=center_blocked,
                           sides=(left, front, right), center_dark=center_dark)

    # -- encarrilado al pasillo -------------------------------------------
    def corridor_correction(self, scan, yaw=None):
        """Correccion angular SUAVE (rad/s) para centrar y enfilar el robot en el pasillo.

        Dos terminos, ambos en marco del robot (no necesitan odometria):
          - LATERAL: (clearance_izq - clearance_der). Si hay mas hueco a un lado, el robot
            esta pegado al otro -> corrige hacia el lado con mas hueco. SOLO con las DOS
            paredes presentes (en un cruce una pared desaparece -> sin termino lateral ->
            no se mete en el cruce; el recto lo mantiene el heading-hold de odometria).
          - ORIENTACION: el angulo del punto MAS CERCANO de cada pared esta a +-90 grados
            cuando el robot va paralelo; su desviacion mide el error de rumbo. Corrige para
            volver a quedar paralelo. Usa las paredes presentes (1 o 2).
        Sin paredes (area abierta) -> 0. Saturada a +-align_max. yaw se acepta por
        compatibilidad de firma pero no se usa (la correccion es puramente geometrica)."""
        lc, la, lr = self._sector(scan, math.pi / 2.0)     # pared izquierda
        rc, ra, rr = self._sector(scan, -math.pi / 2.0)    # pared derecha
        left_wall = lc < self.wall_present_dist
        right_wall = rc < self.wall_present_dist

        heading_terms = []
        if left_wall and la.size:
            perp = la[int(np.argmin(lr))]                  # angulo del punto mas cercano izq
            heading_terms.append(perp - math.pi / 2.0)     # 0 si paralelo; = -theta (theta=giro CCW)
        if right_wall and ra.size:
            perp = ra[int(np.argmin(rr))]                  # angulo del punto mas cercano der
            heading_terms.append(perp + math.pi / 2.0)     # 0 si paralelo; = -theta
        heading = float(np.mean(heading_terms)) if heading_terms else 0.0
        # steer de orientacion = +k * mean(perp_err): perp_err = -theta -> corrige theta a 0
        head_steer = self.align_head * heading

        lat_steer = 0.0
        if left_wall and right_wall:
            # mas hueco a la izq (lc>rc) -> robot pegado a la der -> girar IZQ (+)
            lat_steer = self.align_lat * (lc - rc) / max(self.wall_present_dist, 1e-6)

        corr = self.align_gain * (lat_steer + head_steer)
        return float(np.clip(corr, -self.align_max, self.align_max))


class OccupancyMap:
    """Mapa de ocupacion 2D POR EJECUCION en marco de odometria (apoyo VISUAL).

    Acumula los impactos del LIDAR en coordenadas del mundo usando (x, y, yaw) de la
    odometria, marca el espacio LIBRE a lo largo de cada rayo y la TRAYECTORIA del robot.
    Parte VACIO en cada ejecucion (no carga nada previo) y NO interviene en el control:
    solo se guarda al terminar (PNG + .npy) para analizar la run. El origen se fija en la
    primera posicion conocida del robot (queda centrado en la rejilla).
    """
    def __init__(self, res=0.1, size_m=40.0, free_samples=24, wall_min_hits=2):
        self.res = res                                # metros por celda
        self.n = int(round(size_m / res))             # celdas por lado (rejilla cuadrada)
        if self.n % 2 == 0:
            self.n += 1                               # impar -> hay celda central exacta
        self.size_m = self.n * res
        self.free_samples = free_samples              # puntos muestreados por rayo (espacio libre)
        self.wall_min_hits = wall_min_hits            # impactos minimos para pintar una celda como pared
                                                      # (descarta celdas "manchadas" vistas muy pocas veces)
        self.occ = np.zeros((self.n, self.n), dtype=np.uint32)   # conteo de impactos (pared)
        self.free = np.zeros((self.n, self.n), dtype=np.uint32)  # conteo de "visto libre"
        self.traj = np.zeros((self.n, self.n), dtype=bool)       # celdas por las que paso el robot
        self.origin = None                            # (x,y) world de la celda central
        self.updates = 0

    def _to_cells(self, xs, ys):
        """Mundo (x,y) -> indices (fila, col) validos dentro de la rejilla."""
        ox, oy = self.origin
        cx = self.n // 2
        col = np.round((xs - ox) / self.res).astype(np.int64) + cx
        row = cx - np.round((ys - oy) / self.res).astype(np.int64)   # y hacia ARRIBA
        ok = (row >= 0) & (row < self.n) & (col >= 0) & (col < self.n)
        return row[ok], col[ok]

    def update(self, scan, pos, yaw, integrate_walls=True):
        """Integra un escaneo desde la pose (pos=(x,y), yaw). Silencioso si no hay pose.

        Con `integrate_walls=False` solo marca la TRAYECTORIA (y fija el origen) pero NO acumula
        free/occ: se usa MIENTRAS EL ROBOT GIRA, cuando el escaneo y el yaw no están bien alineados
        y las paredes saldrían abanicadas (smear). El rastro del robot sí se sigue marcando.
        """
        if pos is None or pos[0] is None or yaw is None:
            return
        x, y = float(pos[0]), float(pos[1])
        if self.origin is None:
            self.origin = (x, y)
        self.updates += 1
        # trayectoria (se marca SIEMPRE, también girando -> path continuo)
        r0, c0 = self._to_cells(np.array([x]), np.array([y]))
        if r0.size:
            self.traj[r0, c0] = True
        if not integrate_walls:
            return

        ang = yaw + scan.angles
        r = scan.ranges.astype(np.float64)
        finite = np.isfinite(r) & (r > 0.0)
        hit = finite & (r < scan.range_max * 0.999)      # retorno real = obstaculo
        # espacio LIBRE: muestrea a lo largo de cada rayo hasta el impacto (o range_max)
        reff = np.where(finite, np.minimum(r, scan.range_max), scan.range_max)
        ts = np.linspace(0.0, 1.0, self.free_samples, endpoint=False)[1:]   # sin el origen
        fx = x + np.outer(reff, ts) * np.cos(ang)[:, None]
        fy = y + np.outer(reff, ts) * np.sin(ang)[:, None]
        fr, fc = self._to_cells(fx.ravel(), fy.ravel())
        if fr.size:
            np.add.at(self.free, (fr, fc), 1)
        # impactos (pared)
        if hit.any():
            hx = x + r[hit] * np.cos(ang[hit])
            hy = y + r[hit] * np.sin(ang[hit])
            hr, hc = self._to_cells(hx, hy)
            if hr.size:
                np.add.at(self.occ, (hr, hc), 1)

    def render(self):
        """Imagen BGR del mapa: paredes NEGRO, libre BLANCO, desconocido GRIS, trayectoria ROJO."""
        img = np.full((self.n, self.n, 3), 128, dtype=np.uint8)      # gris = desconocido
        seen = self.free > 0
        img[seen] = (235, 235, 235)                                  # libre = casi blanco
        min_hits = max(1, int(getattr(self, "wall_min_hits", 1)))
        wall = self.occ >= np.maximum(min_hits, self.free // 2)      # mas impactos que "libre" -> pared
        wall &= self.occ >= min_hits                                 # y vista al menos min_hits veces
        img[wall] = (40, 40, 40)                                     # pared = casi negro
        img[self.traj] = (0, 0, 255)                                 # trayectoria = rojo (BGR)
        return img

    # ---- apoyo para el MAPA SEMANTICO (capa de color por clase detectada) -------------
    # No interviene en el control; se rellena solo si navigate.py corre con --show-semantic
    # (o en post-proceso). Perezoso: las matrices se crean al primer mark_semantic().
    def mark_semantic(self, pos, color):
        """Pinta la celda de la pose actual con un color BGR (capa semantica en vivo).

        `color` = (B,G,R) o None (clase sin color -> no se pinta). Crea la capa la 1a vez.
        """
        if pos is None or pos[0] is None or color is None or self.origin is None:
            return
        if getattr(self, "sem", None) is None:
            self.sem = np.zeros((self.n, self.n, 3), dtype=np.uint8)
            self.sem_set = np.zeros((self.n, self.n), dtype=bool)
        r, c = self._to_cells(np.array([float(pos[0])]), np.array([float(pos[1])]))
        if r.size:
            self.sem[r, c] = np.asarray(color, dtype=np.uint8)
            self.sem_set[r, c] = True

    def render_semantic(self):
        """render() base con la capa semantica encima de la trayectoria (si existe)."""
        img = self.render()
        if getattr(self, "sem", None) is not None and self.sem_set.any():
            img[self.sem_set] = self.sem[self.sem_set]
        return img

    @classmethod
    def from_npz(cls, path):
        """Reconstruye un OccupancyMap (solo lectura) desde un .npz guardado por save().

        Para POST-PROCESO: reutiliza render()/_to_cells sin re-ejecutar la navegacion. No
        recalcula nada del control; solo recoloca las matrices occ/free/traj y el origen.
        """
        data = np.load(str(path), allow_pickle=False)
        res = float(data["res"])
        size_m = float(data["size_m"])
        m = cls(res=res, size_m=size_m)
        m.occ = data["occ"].astype(np.uint32)
        m.free = data["free"].astype(np.uint32)
        m.traj = data["traj"].astype(bool)
        # n debe cuadrar con las matrices guardadas (size_m ya venia ajustado a impar*res)
        if m.occ.shape[0] != m.n:
            m.n = int(m.occ.shape[0])
            m.size_m = m.n * res
        org = np.asarray(data["origin"], dtype=float).ravel()
        m.origin = (float(org[0]), float(org[1]))
        m.updates = int(m.traj.sum())
        return m

    def save(self, png_path, npy_path=None):
        """Guarda el mapa: PNG (visual) + .npy (rejilla cruda occ/free/traj). Devuelve rutas escritas."""
        written = []
        if npy_path is not None:
            np.savez(str(npy_path), occ=self.occ, free=self.free, traj=self.traj,
                     res=self.res, size_m=self.size_m,
                     origin=np.array(self.origin if self.origin else (0.0, 0.0)))
            written.append(str(npy_path))
        img = self.render()
        if HAS_CV2:
            cv2.imwrite(str(png_path), img)
            written.append(str(png_path))
        else:                                                        # sin cv2: PPM minimo (sin dependencias)
            ppm = str(png_path).rsplit(".", 1)[0] + ".ppm"
            rgb = img[:, :, ::-1]
            with open(ppm, "wb") as f:
                f.write(f"P6\n{self.n} {self.n}\n255\n".encode())
                f.write(rgb.tobytes())
            written.append(ppm)
        return written
