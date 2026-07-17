"""Analiza un log de navegacion (logs/nav_*.csv) y resume el comportamiento del
clasificador a lo largo de la ejecucion.

Sirve de **metrica objetiva para comparar modelos** (cnn_unificado vs futuros
MLP/RNN/transformer): cuanto se 'comprometio' la red, cuanto ruido genero,
estabilidad por chunks consecutivos, etc.

Que hace:
  1) Segmenta el log en CHUNKS por clase consecutiva (FREE_PATH x20, ARROW_RIGHT x15,
     etc.). Por cada chunk reporta n_frames, duracion, conf min/max/mean/std y la
     fraccion de frames que superan el umbral 'correcto' para esa clase.
  2) Detecta CHUNKS DE RUIDO del clasificador: chunks cortos (<= noise_max_len)
     flanqueados por dos chunks largos (>= noise_min_neighbor) de la MISMA otra
     clase y con conf media >= su umbral. Es el patron tipico de un fallo
     transitorio (p.ej. 15-FREE_PATH 2-ARROW_LEFT 7-FREE_PATH).
  3) Reporta un resumen global: total frames, duracion, frames por clase,
     fraccion 'correcta' por clase, numero de chunks por clase, fraccion del
     log que se identifica como ruido. Por cada SENAL: confianza media/min/max
     (sobre frames crudos) — una conf_max alta en senales que no deberian
     comprometerse (p.ej. flechas en el borde) explica desvios de rumbo. Y el
     conteo de SALIDAS registradas (detecciones de GOAL: episodios y frames +
     conf_max), que delata un GOAL espurio (falsa diana que termina la run antes).
  4) Metricas de RESOLUCION y EFICIENCIA de ordenes (motion_metrics): si la run
     resolvio el laberinto y en que TIEMPO (nav-v16), distancia recorrida por
     odometria, desplazamiento neto/directez, reparto del tiempo por tipo de orden
     (avance/giro/correccion/parado) y numero de giros por disparador.
  5) REGISTRA cada run analizada en un CSV maestro de comparacion (logs/
     nav_comparison.csv): una fila por run, etiquetada con --label (la familia).
     Asi la comparacion entre familias queda guardada y se imprime con --compare.

Umbral por clase (alineado con navigate.py nav-v13):
  - ARROW_LEFT/ARROW_RIGHT: 0.95 (specialist cnn_arrows, conf en vivo ~1.0)
  - resto (CROSS, GOAL, FREE_PATH, ARROW_UP, WALL*, UNKNOWN): 0.55

Uso (desde Modelos/src):
    python analyze_log.py --label cnn               # analiza el log mas reciente y lo registra como 'cnn'
    python analyze_log.py --log ../../logs/nav_20260526_185003.csv --label mlp
    python analyze_log.py --no-json                 # solo consola, sin JSON
    python analyze_log.py --noise-max-len 3 --noise-min-neighbor 4
    python analyze_log.py --compare                 # imprime la tabla maestra de comparacion y sale

Salida: tabla + resumen + metricas de resolucion/eficiencia en consola; JSON con todo
en logs/<misma>.analysis.json; y una fila por run en logs/nav_comparison.csv (--label).
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = SRC_DIR.parent.parent / "logs"
DEFAULT_COMPARISON_CSV = DEFAULT_LOG_DIR / "nav_comparison.csv"
# CSV en formato LARGO: una fila por (run, senal) -> conteo + conf media/min/max.
# Permite comparar CADA senal entre modelos (filtrar por clase y ver todas las labels).
DEFAULT_SIGNAL_CSV = DEFAULT_LOG_DIR / "nav_signal_stats.csv"

ARROW_LR = {"ARROW_LEFT", "ARROW_RIGHT"}
ARROW_CONF_THRESH = 0.95   # = navigate.py --arrow-conf-thresh (specialist)
SIGN_CONF_THRESH = 0.55    # = navigate.py --sign-conf-thresh (resto)


def conf_threshold_for(cls):
    """Umbral por clase: alto para flechas L/R, moderado para el resto."""
    return ARROW_CONF_THRESH if cls in ARROW_LR else SIGN_CONF_THRESH


# ---------- Lectura del CSV ----------

def read_log(path):
    """Lee el CSV y devuelve lista de filas como dicts (todos los campos como str)."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float(s):
    """Float robusto a campos vacios/no numericos -> None."""
    try:
        return float(s) if s not in (None, "", "None") else None
    except ValueError:
        return None


# ---------- Segmentacion en chunks ----------

def build_chunks(rows):
    """Agrupa filas consecutivas con la misma 'cls' en un chunk.

    Devuelve lista de dicts con stats por chunk:
      idx, cls, n, t_start, t_end, dur_s, conf_min/max/mean/std,
      correct_frac, thresh, frame_start, frame_end, pos_start, pos_end.
    """
    chunks = []
    if not rows:
        return chunks
    current = None
    for row in rows:
        cls = row.get("cls", "") or ""
        if cls == "":
            continue
        if current is None or cls != current["cls"]:
            if current is not None:
                chunks.append(_finalize_chunk(current))
            current = {"cls": cls, "rows": [row]}
        else:
            current["rows"].append(row)
    if current is not None:
        chunks.append(_finalize_chunk(current))
    for i, c in enumerate(chunks):
        c["idx"] = i
    return chunks


def _finalize_chunk(buf):
    """Calcula las estadisticas de un chunk a partir de sus filas crudas."""
    rows = buf["rows"]
    cls = buf["cls"]
    confs = [_to_float(r.get("conf")) for r in rows]
    confs = [c for c in confs if c is not None]
    n = len(rows)

    t0 = _to_float(rows[0].get("t"))
    t1 = _to_float(rows[-1].get("t"))
    dur = (t1 - t0) if (t0 is not None and t1 is not None) else None

    thresh = conf_threshold_for(cls)
    correct = sum(1 for c in confs if c >= thresh)
    correct_frac = correct / n if n else 0.0

    return {
        "cls": cls,
        "n": n,
        "frame_start": int(_to_float(rows[0].get("frame")) or 0),
        "frame_end": int(_to_float(rows[-1].get("frame")) or 0),
        "t_start": t0,
        "t_end": t1,
        "dur_s": dur,
        "conf_min": min(confs) if confs else None,
        "conf_max": max(confs) if confs else None,
        "conf_mean": statistics.fmean(confs) if confs else None,
        "conf_std": statistics.pstdev(confs) if len(confs) > 1 else 0.0,
        "thresh": thresh,
        "correct_frac": correct_frac,
        "pos_start": (_to_float(rows[0].get("pos_x")), _to_float(rows[0].get("pos_y"))),
        "pos_end": (_to_float(rows[-1].get("pos_x")), _to_float(rows[-1].get("pos_y"))),
    }


# ---------- Deteccion de ruido del clasificador ----------

def find_noise_chunks(chunks, max_len=2, min_neighbor=5):
    """Marca como ruido los chunks cortos rodeados por dos chunks largos de la
    MISMA clase, ambos con conf media >= su umbral.

    Patron clasico:  [..., A:15:conf>=thrA, X:2:?, A:7:conf>=thrA, ...]
    El chunk X de 2 frames es muy probablemente un fallo del clasificador.

    Devuelve lista de chunks (referencias a los originales) con campo 'noise'=True.
    """
    noisy = []
    for i in range(1, len(chunks) - 1):
        c = chunks[i]
        prev_c, next_c = chunks[i - 1], chunks[i + 1]
        if c["n"] > max_len:
            continue
        if prev_c["cls"] != next_c["cls"] or prev_c["cls"] == c["cls"]:
            continue
        if prev_c["n"] < min_neighbor or next_c["n"] < min_neighbor:
            continue
        neighbor_thresh = conf_threshold_for(prev_c["cls"])
        if (prev_c["conf_mean"] or 0.0) < neighbor_thresh:
            continue
        if (next_c["conf_mean"] or 0.0) < neighbor_thresh:
            continue
        c["noise"] = True
        c["noise_context"] = {
            "prev_cls": prev_c["cls"], "prev_n": prev_c["n"],
            "next_cls": next_c["cls"], "next_n": next_c["n"],
        }
        noisy.append(c)
    return noisy


# ---------- Resumen global ----------

def per_class_conf_stats(rows):
    """Confianza por SEÑAL (clase) a partir de las filas crudas: n, media, min y max.

    Mas preciso que reconstruir desde las medias de chunk: usa TODAS las confianzas
    de cada clase. Sirve para ver como de confiada esta cada red con cada senal — en
    particular, una conf_max alta en una senal que no deberia comprometerse (p.ej.
    flechas vistas en el borde de la imagen) explica desvios de rumbo."""
    raw = {}
    for r in rows:
        cls = r.get("cls") or ""
        if not cls:
            continue
        c = _to_float(r.get("conf"))
        if c is None:
            continue
        raw.setdefault(cls, []).append(c)
    out = {}
    for cls, confs in raw.items():
        out[cls] = {
            "conf_n": len(confs),
            "conf_mean": statistics.fmean(confs) if confs else None,
            "conf_min": min(confs) if confs else None,
            "conf_max": max(confs) if confs else None,
            "conf_std": statistics.pstdev(confs) if len(confs) > 1 else 0.0,
        }
    return out


def summarize(chunks, noisy, rows=None):
    """Resumen global del log: totales, por clase, ruido.

    `rows` (opcional) habilita la confianza media/min/max por senal calculada sobre
    las filas crudas (mas precisa) y el conteo de SALIDAS (detecciones de GOAL)."""
    total_frames = sum(c["n"] for c in chunks)
    noisy_frames = sum(c["n"] for c in noisy)

    by_class = {}
    for c in chunks:
        d = by_class.setdefault(c["cls"], {
            "frames": 0, "chunks": 0, "correct": 0,
            "confs": [], "thresh": c["thresh"],
        })
        d["frames"] += c["n"]
        d["chunks"] += 1
        d["correct"] += round(c["correct_frac"] * c["n"])
        # Para el conf medio ponderado a nivel clase reconstruimos a grano grueso:
        if c["conf_mean"] is not None:
            d["confs"].append((c["conf_mean"], c["n"]))

    raw_conf = per_class_conf_stats(rows) if rows is not None else {}

    by_class_out = {}
    for cls, d in by_class.items():
        weighted = (sum(m * n for m, n in d["confs"]) / d["frames"]) if d["frames"] else None
        rc = raw_conf.get(cls, {})
        by_class_out[cls] = {
            "frames": d["frames"],
            "frac": d["frames"] / total_frames if total_frames else 0.0,
            "chunks": d["chunks"],
            "correct_frac": d["correct"] / d["frames"] if d["frames"] else 0.0,
            "conf_mean_weighted": weighted,
            # Confianza por senal (media/min/max) sobre frames crudos (mas precisa):
            "conf_mean": rc.get("conf_mean", weighted),
            "conf_min": rc.get("conf_min"),
            "conf_max": rc.get("conf_max"),
            "conf_std": rc.get("conf_std"),
            "thresh": d["thresh"],
        }

    t_start = min((c["t_start"] for c in chunks if c["t_start"] is not None), default=None)
    t_end = max((c["t_end"] for c in chunks if c["t_end"] is not None), default=None)
    duration = (t_end - t_start) if (t_start is not None and t_end is not None) else None

    total_correct = sum(round(v["correct_frac"] * v["frames"]) for v in by_class_out.values())

    # SALIDAS registradas = detecciones de la diana/salida (clase GOAL): cuantos frames
    # y cuantos episodios (chunks) se clasifico GOAL. Un GOAL temprano/espurio (p.ej. una
    # flecha leida como diana) dispara el fin por racha y termina la run antes de tiempo.
    goal = by_class_out.get("GOAL", {})
    goal_frames = goal.get("frames", 0)
    goal_chunks = goal.get("chunks", 0)
    goal_conf_max = goal.get("conf_max")

    return {
        "total_frames": total_frames,
        "duration_s": duration,
        "fps_effective": (total_frames / duration) if duration else None,
        "n_chunks": len(chunks),
        "n_noise_chunks": len(noisy),
        "noisy_frames": noisy_frames,
        "noisy_frac": noisy_frames / total_frames if total_frames else 0.0,
        "correct_frac_global": total_correct / total_frames if total_frames else 0.0,
        "goal_frames": goal_frames,
        "goal_chunks": goal_chunks,
        "goal_conf_max": goal_conf_max,
        "by_class": by_class_out,
    }


# ---------- Metricas de resolucion, movimiento y eficiencia de ordenes ----------

def motion_metrics(rows):
    """Metricas de RESOLUCION y EFICIENCIA de ordenes a partir de las filas crudas.

    Habilitado por nav-v16 (fin determinista por racha de GOAL): permite medir el
    TIEMPO de resolucion y como de eficientes son las ordenes de cada familia.

      - solved / solve_time_s: si la run dio el laberinto por resuelto (nota
        'RESUELTO' de nav-v16, o 'META alcanzada' del esquema anterior) y en que t.
      - distance_m / net_disp_m / path_directness: longitud del recorrido por
        odometria, desplazamiento neto inicio->fin y su cociente (1.0 = linea recta;
        bajo = mucho dar vueltas). proxy de eficiencia espacial.
      - reparto del TIEMPO por tipo de orden (ponderado por dt entre frames):
        avance recto, giro en el sitio, avance+correccion, parado. progress_frac =
        fraccion de tiempo con avance lineal (orden 'util' que progresa).
      - n_turns y desglose por disparador (flecha / cruz / pared).
      - reparto de FRAMES por estado (FOLLOW/TURN/WALL_PROBE).
    """
    EPS = 1e-3
    ts = [_to_float(r.get("t")) for r in rows]
    t_first = next((t for t in ts if t is not None), None)
    t_last = next((t for t in reversed(ts) if t is not None), None)
    total_time = (t_last - t_first) if (t_first is not None and t_last is not None) else None

    # --- resolucion ---
    solved, solve_time = False, None
    for r in rows:
        note = r.get("note") or ""
        if "RESUELTO" in note or "META alcanzada" in note:
            solved, solve_time = True, _to_float(r.get("t"))
            break

    # --- distancia recorrida por odometria, desplazamiento neto y directez ---
    pts = [(_to_float(r.get("pos_x")), _to_float(r.get("pos_y"))) for r in rows]
    pts = [(x, y) for x, y in pts if x is not None and y is not None]
    dist = sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(pts, pts[1:]))
    net = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) if len(pts) >= 2 else 0.0
    directness = (net / dist) if dist > 1e-6 else None

    # --- reparto del tiempo por tipo de orden (ponderado por dt) ---
    t_fwd = t_turn = t_steer = t_stop = 0.0
    for r0, r1 in zip(rows, rows[1:]):
        t0, t1 = _to_float(r0.get("t")), _to_float(r1.get("t"))
        if t0 is None or t1 is None or t1 <= t0:
            continue
        dt = t1 - t0
        lin = abs(_to_float(r0.get("lin")) or 0.0)
        ang = abs(_to_float(r0.get("ang")) or 0.0)
        if lin > EPS and ang > EPS:
            t_steer += dt          # avance + correccion (homing / esquiva / steer)
        elif ang > EPS:
            t_turn += dt           # giro en el sitio
        elif lin > EPS:
            t_fwd += dt            # avance recto
        else:
            t_stop += dt           # parado
    t_motion = t_fwd + t_turn + t_steer + t_stop
    frac = (lambda x: (x / t_motion) if t_motion > 1e-6 else 0.0)

    # --- maniobras (giros) por disparador: filas cuya nota empieza por 'INICIO giro' ---
    triggers = []
    for r in rows:
        note = r.get("note") or ""
        if note.startswith("INICIO giro") and "[" in note and "]" in note:
            triggers.append(note[note.find("[") + 1:note.find("]")])
    turns_arrow = sum(1 for t in triggers if "flecha" in t.lower())
    turns_cross = sum(1 for t in triggers if "cruz" in t.lower())
    turns_wall = sum(1 for t in triggers if "pared" in t.lower())

    # --- reparto de frames por estado ---
    states = [r.get("state") or "" for r in rows]
    by_state = {s: states.count(s) for s in sorted(set(states)) if s}

    return {
        "solved": solved,
        "solve_time_s": solve_time,
        "total_time_s": total_time,
        "distance_m": dist,
        "net_disp_m": net,
        "path_directness": directness,
        "n_turns": len(triggers),
        "turns_arrow": turns_arrow,
        "turns_cross": turns_cross,
        "turns_wall": turns_wall,
        "time_forward_s": t_fwd,
        "time_turn_s": t_turn,
        "time_steer_s": t_steer,
        "time_stop_s": t_stop,
        "forward_frac": frac(t_fwd),
        "turn_frac": frac(t_turn),
        "steer_frac": frac(t_steer),
        "stop_frac": frac(t_stop),
        "progress_frac": frac(t_fwd + t_steer),
        "avg_speed_mps": (dist / t_motion) if t_motion > 1e-6 else None,
        "frames_by_state": by_state,
    }


# ---------- Registro de comparacion entre runs (CSV maestro) ----------

COMPARE_FIELDS = [
    "log", "label", "solved", "solve_time_s", "total_time_s",
    "distance_m", "net_disp_m", "path_directness", "avg_speed_mps",
    "n_turns", "turns_arrow", "turns_cross", "turns_wall",
    "forward_frac", "turn_frac", "steer_frac", "stop_frac", "progress_frac",
    "correct_frac_global", "noisy_frac", "total_frames",
    # Salidas (detecciones de GOAL): episodios, frames y conf maxima de GOAL.
    # goal_chunks>1 o un solved muy rapido apunta a un GOAL espurio (falsa diana).
    "goal_chunks", "goal_frames", "goal_conf_max",
]


def _fmt_cell(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:.4f}"
    return "" if v is None else str(v)


def update_comparison_csv(path, row):
    """Inserta/actualiza (clave = 'log') una fila-resumen en el CSV de comparacion.

    Asi cada run analizada queda REGISTRADA en una tabla maestra; re-analizar el
    mismo log reemplaza su fila (no duplica). Una fila por run = una familia."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.is_file():
        with open(path, newline="") as f:
            existing = [r for r in csv.DictReader(f) if r.get("log") != row["log"]]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COMPARE_FIELDS)
        w.writeheader()
        for r in existing:
            w.writerow({k: r.get(k, "") for k in COMPARE_FIELDS})
        w.writerow({k: _fmt_cell(row.get(k)) for k in COMPARE_FIELDS})


# ---------- Registro y comparacion POR SEÑAL (formato largo) ----------

SIGNAL_STATS_FIELDS = [
    "log", "label", "class", "frames", "chunks", "correct_frac",
    "conf_mean", "conf_min", "conf_max",
]


def update_signal_stats_csv(path, log_name, label, by_class):
    """Registra, en formato LARGO, UNA fila por (run, senal) con el conteo (frames y
    chunks) y la confianza media/min/max de esa senal en esa run. Re-analizar el mismo
    log reemplaza sus filas (clave = 'log'). Asi se puede comparar CADA senal entre
    modelos: filtra por 'class' y mira todas las 'label'."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.is_file():
        with open(path, newline="") as f:
            existing = [r for r in csv.DictReader(f) if r.get("log") != log_name]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SIGNAL_STATS_FIELDS)
        w.writeheader()
        for r in existing:
            w.writerow({k: r.get(k, "") for k in SIGNAL_STATS_FIELDS})
        for cls in sorted(by_class):
            d = by_class[cls]
            w.writerow({
                "log": log_name, "label": label, "class": cls,
                "frames": d["frames"], "chunks": d["chunks"],
                "correct_frac": _fmt_cell(d["correct_frac"]),
                "conf_mean": _fmt_cell(d.get("conf_mean")),
                "conf_min": _fmt_cell(d.get("conf_min")),
                "conf_max": _fmt_cell(d.get("conf_max")),
            })


def print_signal_comparison(path, only_class=None):
    """Imprime, por SEÑAL, el conteo y la confianza (media/min/max) de cada modelo,
    para comparar todas las senales entre modelos. Con only_class, solo esa senal."""
    if not path.is_file():
        print(f"(aun no hay stats por senal en {path})")
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("(stats por senal vacios)")
        return

    def gf(r, k):
        try:
            return float(r.get(k) or 0)
        except ValueError:
            return 0.0

    classes = sorted({r["class"] for r in rows if r.get("class")})
    if only_class:
        classes = [c for c in classes if c == only_class]
        if not classes:
            print(f"(la senal '{only_class}' no aparece en {path})")
            return

    print(f"\nCOMPARACION POR SENAL  ({path})")
    for cls in classes:
        crows = sorted((r for r in rows if r.get("class") == cls),
                       key=lambda r: r.get("label", ""))
        print(f"\n=== {cls} ===")
        print(f"  {'label':18s} {'detec':>6s} {'episod':>6s} {'c_med':>6s} "
              f"{'c_min':>6s} {'c_max':>6s} {'%correct':>9s}")
        for r in crows:
            print(f"  {r.get('label',''):18s} {int(gf(r,'frames')):6d} "
                  f"{int(gf(r,'chunks')):6d} {gf(r,'conf_mean'):6.3f} "
                  f"{gf(r,'conf_min'):6.3f} {gf(r,'conf_max'):6.3f} "
                  f"{gf(r,'correct_frac')*100:8.1f}%")


def print_comparison(path):
    """Imprime la tabla maestra de comparacion (todas las runs registradas)."""
    if not path.is_file():
        print(f"(aun no hay tabla de comparacion en {path})")
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("(tabla de comparacion vacia)")
        return

    def gf(r, k):
        try:
            return float(r.get(k) or 0)
        except ValueError:
            return 0.0

    print(f"\nTABLA DE COMPARACION  ({path})")
    print(f"  {'label':18s} {'res':>3s} {'t_solve':>8s} {'t_tot':>7s} {'dist_m':>7s} "
          f"{'direc':>5s} {'giros':>5s} {'avance%':>7s} {'%corr':>6s} {'ruido%':>7s} {'salidas':>7s}")
    for r in sorted(rows, key=lambda r: r.get("label", "")):
        res = "si" if r.get("solved") in ("1", "True", "true") else "no"
        gch = int(gf(r, "goal_chunks"))
        # marca el GOAL espurio: varias salidas, o un solved sospechosamente rapido
        sus = "!" if (gch > 1 or (res == "si" and 0 < gf(r, "solve_time_s") < 20)) else " "
        print(f"  {r.get('label',''):18s} {res:>3s} {gf(r,'solve_time_s'):8.1f} "
              f"{gf(r,'total_time_s'):7.1f} {gf(r,'distance_m'):7.2f} "
              f"{gf(r,'path_directness'):5.2f} {int(gf(r,'n_turns')):5d} "
              f"{gf(r,'forward_frac')*100:6.0f}% {gf(r,'correct_frac_global')*100:5.1f}% "
              f"{gf(r,'noisy_frac')*100:6.1f}% {gch:6d}{sus}")


# ---------- Salida por consola ----------

def print_report(log_path, chunks, noisy, summary, motion):
    print(f"\nLOG: {log_path}")
    print(f"  Frames totales: {summary['total_frames']}  |  Duracion: "
          f"{summary['duration_s']:.1f}s  |  fps_efectivo: "
          f"{summary['fps_effective']:.1f}")
    print(f"  Chunks: {summary['n_chunks']}  |  Chunks de ruido: {summary['n_noise_chunks']} "
          f"({summary['noisy_frac']*100:.1f}% de frames)  |  %correct global: "
          f"{summary['correct_frac_global']*100:.1f}%")
    gcm = (f"{summary['goal_conf_max']:.2f}" if summary.get("goal_conf_max") is not None else "-")
    aviso = "  <-- revisar: posible GOAL espurio" if (
        summary.get("goal_chunks", 0) > 1 or
        (motion.get("solved") and (motion.get("solve_time_s") or 0) < 20)) else ""
    print(f"  Salidas (GOAL) registradas: {summary['goal_chunks']} episodio(s) / "
          f"{summary['goal_frames']} frames  |  conf_max GOAL: {gcm}{aviso}")

    m = motion
    print("\n  Resolucion y eficiencia de ordenes:")
    estado = "RESUELTO" if m["solved"] else "NO resuelto"
    tsolve = f"{m['solve_time_s']:.1f}s" if m["solve_time_s"] is not None else "-"
    ttot = f"{m['total_time_s']:.1f}s" if m["total_time_s"] is not None else "-"
    print(f"    Estado: {estado}    t_resolucion: {tsolve}    t_total: {ttot}")
    dirc = f"{m['path_directness']:.2f}" if m["path_directness"] is not None else "-"
    spd = f"{m['avg_speed_mps']:.3f}" if m["avg_speed_mps"] is not None else "-"
    print(f"    Distancia: {m['distance_m']:.2f} m    desplaz_neto: {m['net_disp_m']:.2f} m"
          f"    directez: {dirc}    vel_media: {spd} m/s")
    print(f"    Giros: {m['n_turns']}  (flecha {m['turns_arrow']}, cruz {m['turns_cross']}, "
          f"pared {m['turns_wall']})")
    print(f"    Tiempo por orden:  avance {m['forward_frac']*100:.0f}%   "
          f"giro {m['turn_frac']*100:.0f}%   avance+correc {m['steer_frac']*100:.0f}%   "
          f"parado {m['stop_frac']*100:.0f}%    | progreso util {m['progress_frac']*100:.0f}%")

    print("\n  Por clase (confianza media/min/max de cada senal):")
    print(f"    {'clase':12s}  {'frames':>6s} {'%':>6s}  {'chunks':>6s}  "
          f"{'c_med':>6s} {'c_min':>6s} {'c_max':>6s}  {'%correct':>9s}  thr")
    for cls in sorted(summary["by_class"]):
        d = summary["by_class"][cls]
        cm = f"{d['conf_mean']:.3f}" if d.get("conf_mean") is not None else "   -  "
        ci = f"{d['conf_min']:.3f}" if d.get("conf_min") is not None else "   -  "
        cM = f"{d['conf_max']:.3f}" if d.get("conf_max") is not None else "   -  "
        print(f"    {cls:12s}  {d['frames']:6d} {d['frac']*100:5.1f}% "
              f"{d['chunks']:6d}  {cm:>6s} {ci:>6s} {cM:>6s}  {d['correct_frac']*100:8.1f}%  "
              f"{d['thresh']:.2f}")

    print("\n  Chunks (en orden):")
    print(f"    {'#':>3s}  {'clase':12s}  {'n':>3s}  {'dur_s':>5s}  "
          f"{'min':>5s}  {'mean':>5s}  {'max':>5s}  {'%ok':>5s}  ruido")
    for c in chunks:
        cm = f"{c['conf_mean']:.2f}" if c["conf_mean"] is not None else "  -  "
        ci = f"{c['conf_min']:.2f}" if c["conf_min"] is not None else "  -  "
        cM = f"{c['conf_max']:.2f}" if c["conf_max"] is not None else "  -  "
        dur = f"{c['dur_s']:.1f}" if c["dur_s"] is not None else "  -  "
        flag = "RUIDO" if c.get("noise") else ""
        print(f"    {c['idx']:3d}  {c['cls']:12s}  {c['n']:3d}  {dur:>5s}  "
              f"{ci:>5s}  {cm:>5s}  {cM:>5s}  {c['correct_frac']*100:4.0f}%  {flag}")

    if noisy:
        print("\n  Posibles errores del clasificador (chunks marcados RUIDO):")
        for c in noisy:
            ctx = c["noise_context"]
            print(f"    chunk #{c['idx']}: {c['cls']} x{c['n']} (conf "
                  f"{c['conf_min']:.2f}-{c['conf_max']:.2f}) entre "
                  f"{ctx['prev_cls']} x{ctx['prev_n']} y "
                  f"{ctx['next_cls']} x{ctx['next_n']}")


# ---------- CLI ----------

def find_latest_log(log_dir):
    cands = sorted(log_dir.glob("nav_*.csv"))
    return cands[-1] if cands else None


def parse_args():
    p = argparse.ArgumentParser(description="Analiza un log de navegacion.")
    p.add_argument("--log", default=None,
                   help="Ruta al CSV (def: el mas reciente en TFG_v2/logs/)")
    p.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR),
                   help=f"Carpeta de logs si --log no se da (def: {DEFAULT_LOG_DIR})")
    p.add_argument("--noise-max-len", type=int, default=2,
                   help="Longitud maxima para que un chunk se considere candidato a ruido (def 2)")
    p.add_argument("--noise-min-neighbor", type=int, default=5,
                   help="Longitud minima de los dos chunks vecinos para confirmar ruido (def 5)")
    p.add_argument("--no-json", action="store_true",
                   help="No escribir el JSON (.analysis.json) al lado del CSV")
    p.add_argument("--label", default=None,
                   help="Etiqueta de esta run en la tabla de comparacion (p.ej. la familia: cnn/mlp/rnn/transformer). Por defecto, el nombre del log")
    p.add_argument("--compare-csv", default=str(DEFAULT_COMPARISON_CSV),
                   help=f"CSV maestro donde se registra cada run analizada (def: {DEFAULT_COMPARISON_CSV})")
    p.add_argument("--no-compare-csv", action="store_true",
                   help="No registrar esta run en los CSV de comparacion (maestro y por senal)")
    p.add_argument("--compare", action="store_true",
                   help="Solo IMPRIMIR la tabla maestra de comparacion (todas las runs registradas) y salir")
    p.add_argument("--signal-csv", default=str(DEFAULT_SIGNAL_CSV),
                   help=f"CSV (formato largo) de stats por senal: una fila por (run, senal) (def: {DEFAULT_SIGNAL_CSV})")
    p.add_argument("--no-signal-csv", action="store_true",
                   help="No registrar las stats por senal de esta run")
    p.add_argument("--compare-signals", action="store_true",
                   help="Solo IMPRIMIR la comparacion POR SEÑAL entre modelos (conteo + conf media/min/max) y salir")
    p.add_argument("--signal-class", default=None,
                   help="Con --compare-signals, filtra a una sola senal (p.ej. ARROW_LEFT)")
    return p.parse_args()


def main():
    args = parse_args()
    compare_csv = Path(args.compare_csv)
    signal_csv = Path(args.signal_csv)

    # Modo comparacion: imprimir la tabla maestra y salir (no analiza ningun log)
    if args.compare:
        print_comparison(compare_csv)
        return

    # Modo comparacion POR SEÑAL: imprimir conteo+conf de cada senal por modelo y salir
    if args.compare_signals:
        print_signal_comparison(signal_csv, only_class=args.signal_class)
        return

    log_path = Path(args.log) if args.log else find_latest_log(Path(args.log_dir))
    if log_path is None or not log_path.is_file():
        print(f"No se encontro ningun log (intentado: {log_path})")
        return

    rows = read_log(log_path)
    chunks = build_chunks(rows)
    noisy = find_noise_chunks(chunks, args.noise_max_len, args.noise_min_neighbor)
    summary = summarize(chunks, noisy, rows)
    motion = motion_metrics(rows)

    print_report(log_path, chunks, noisy, summary, motion)

    if not args.no_json:
        out = log_path.with_suffix(".analysis.json")
        # Limpia tuplas (no serializables como tales en JSON estricto)
        clean_chunks = [{**c, "pos_start": list(c["pos_start"]),
                         "pos_end": list(c["pos_end"])} for c in chunks]
        payload = {
            "log": str(log_path),
            "label": args.label or log_path.stem,
            "params": {
                "noise_max_len": args.noise_max_len,
                "noise_min_neighbor": args.noise_min_neighbor,
                "arrow_conf_thresh": ARROW_CONF_THRESH,
                "sign_conf_thresh": SIGN_CONF_THRESH,
            },
            "summary": summary,
            "motion": motion,
            "chunks": clean_chunks,
            "noise_chunk_idx": [c["idx"] for c in noisy],
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nJSON guardado en: {out}")

    # Registrar esta run en el CSV maestro de comparacion (una fila por run/familia)
    if not args.no_compare_csv:
        row = {
            "log": log_path.name,
            "label": args.label or log_path.stem,
            "correct_frac_global": summary["correct_frac_global"],
            "noisy_frac": summary["noisy_frac"],
            "total_frames": summary["total_frames"],
            "goal_chunks": summary["goal_chunks"],
            "goal_frames": summary["goal_frames"],
            "goal_conf_max": summary["goal_conf_max"],
            **{k: motion.get(k) for k in (
                "solved", "solve_time_s", "total_time_s", "distance_m", "net_disp_m",
                "path_directness", "avg_speed_mps", "n_turns", "turns_arrow",
                "turns_cross", "turns_wall", "forward_frac", "turn_frac",
                "steer_frac", "stop_frac", "progress_frac")},
        }
        update_comparison_csv(compare_csv, row)
        print(f"Run registrada en la tabla de comparacion (label='{row['label']}'): {compare_csv}")

    # Stats POR SEÑAL (formato largo): una fila por (run, senal) con conteo + conf.
    if not args.no_signal_csv:
        update_signal_stats_csv(signal_csv, log_path.name, args.label or log_path.stem,
                                summary["by_class"])
        print(f"Stats por senal registradas (label='{args.label or log_path.stem}'): {signal_csv}")

    if not args.no_compare_csv:
        print_comparison(compare_csv)


if __name__ == "__main__":
    main()
