#!/usr/bin/env python3
"""Genera las figuras COMBINADAS nuevas de la memoria a partir de los CSV de
resultados ya existentes en el repositorio. No reentrena ni reejecuta nada:
solo lee CSV y dibuja.

Salida (en memoria/img/):
  1. clf_familias_barras.png    -> test_acc + F1 de las 4 familias (clasificacion).
  2. clf_todas_arquitecturas.png-> todas las arquitecturas ordenadas por test_acc,
                                   con num_params anotado (coste-rendimiento).
  3. nav_resumen.png            -> mundos resueltos y correct_frac por modelo.
  4. clf_vs_nav.png             -> dispersion accuracy de clasificacion vs. calidad
                                   de navegacion (la figura clave de la discusion).

Uso:
    python memoria/scripts/gen_figuras_memoria.py
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Rutas (relativas a la raiz del repo TFG_v2) ---------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNS = os.path.join(ROOT, "Modelos", "runs")
LOGS = os.path.join(ROOT, "logs")
OUT = os.path.join(ROOT, "memoria", "img")
os.makedirs(OUT, exist_ok=True)

CMP_FAMILIAS = os.path.join(RUNS, "compare_20260531_113952", "summary.csv")
CMP_CNN = os.path.join(RUNS, "compare_20260610_132232", "summary.csv")
CMP_TL = os.path.join(RUNS, "compare_20260610_135553", "summary.csv")
NAV = os.path.join(LOGS, "nav_comparison.csv")

plt.rcParams.update({"font.size": 11, "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.3})


def read_summary(path):
    """Lee un summary.csv de compare_models.py -> lista de dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_nav(path):
    """Lee nav_comparison.csv -> dict modelo -> {world: row}. Si un (modelo,mundo)
    aparece repetido, se queda con la ultima ejecucion."""
    by_model = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            label = row["label"]           # p.ej. cnn_w3
            model, _, world = label.rpartition("_")
            by_model[model][world] = row
    return by_model


# --- Figura 1: barras clasificacion (4 familias) ---------------------------
def fig_familias():
    rows = read_summary(CMP_FAMILIAS)
    rows.sort(key=lambda r: float(r["test_acc"]), reverse=True)
    names = [r["model"].upper() for r in rows]
    acc = [float(r["test_acc"]) for r in rows]
    f1 = [float(r["test_f1_macro"]) for r in rows]

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar(x - w / 2, acc, w, label="Exactitud (test)", color="#3b6fb0")
    b2 = ax.bar(x + w / 2, f1, w, label="F1 macro (test)", color="#c0682a")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Puntuación")
    ax.set_title("Clasificación de 10 clases: comparativa de familias")
    ax.legend(loc="lower left")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "clf_familias_barras.png")
    fig.savefig(p); plt.close(fig); print("escrito", p)


# --- Figura 2: todas las arquitecturas por test_acc + num_params -----------
def fig_todas():
    rows = read_summary(CMP_FAMILIAS) + read_summary(CMP_CNN) + read_summary(CMP_TL)
    # Quitar el cnn duplicado (aparece en familias y en cnn-desde-cero): dejar el de familias.
    seen, uniq = set(), []
    for r in rows:
        if r["model"] in seen:
            continue
        seen.add(r["model"]); uniq.append(r)
    uniq.sort(key=lambda r: float(r["test_acc"]), reverse=True)

    names = [r["model"] for r in uniq]
    acc = [float(r["test_acc"]) for r in uniq]
    params = [int(r["num_params"]) for r in uniq]
    tl = {"vgg_tl", "resnet_tl"}
    colors = ["#7a3ab0" if n in tl else "#3b6fb0" for n in names]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(names))
    bars = ax.bar(x, acc, color=colors)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.08); ax.set_ylabel("Exactitud (test)")
    ax.set_title("Todas las arquitecturas ordenadas por exactitud "
                 "(morado = transfer learning)")
    for b, p_ in zip(bars, params):
        m = p_ / 1e6
        txt = f"{m:.1f}M" if m >= 1 else f"{p_/1e3:.0f}K"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                txt, ha="center", va="bottom", fontsize=8, color="#444")
    ax.text(0.99, 0.02, "Etiqueta sobre la barra = número de parámetros",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="#666")
    fig.tight_layout()
    p = os.path.join(OUT, "clf_todas_arquitecturas.png")
    fig.savefig(p); plt.close(fig); print("escrito", p)


# --- Figura 3: resumen de navegacion ---------------------------------------
# Orden de presentacion: las 4 familias primero, luego desde-cero, luego TL.
NAV_ORDER = ["cnn", "rnn", "transformer", "mlp",
             "fcn", "unet", "yolo", "vgg", "resnet", "resnet_tl", "vgg_tl"]


def nav_aggregate():
    nav = read_nav(NAV)
    agg = {}
    for m in NAV_ORDER:
        if m not in nav:
            continue
        rows = list(nav[m].values())
        solved = sum(int(r["solved"]) for r in rows)
        n = len(rows)
        cf = np.mean([float(r["correct_frac_global"]) for r in rows])
        agg[m] = {"solved": solved, "n": n, "correct": cf}
    return agg


def fig_nav():
    agg = nav_aggregate()
    models = [m for m in NAV_ORDER if m in agg]
    solved = [agg[m]["solved"] for m in models]
    ntot = [agg[m]["n"] for m in models]
    correct = [agg[m]["correct"] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(models))
    b = ax1.bar(x, solved, color="#3b8a52")
    ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=30, ha="right")
    ax1.set_ylim(0, 5.5); ax1.set_ylabel("Mundos resueltos")
    ax1.set_title("Mundos resueltos (de 5)")
    for bi, s, nt in zip(b, solved, ntot):
        ax1.text(bi.get_x() + bi.get_width() / 2, bi.get_height() + 0.05,
                 f"{s}/{nt}", ha="center", va="bottom", fontsize=9)

    b2 = ax2.bar(x, correct, color="#b0413b")
    ax2.set_xticks(x); ax2.set_xticklabels(models, rotation=30, ha="right")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Fracción de clasificación correcta (media)")
    ax2.set_title("Calidad de percepción en vivo")
    for bi in b2:
        ax2.text(bi.get_x() + bi.get_width() / 2, bi.get_height() + 0.01,
                 f"{bi.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Navegación en simulación: 11 modelos × 5 mundos")
    fig.tight_layout()
    p = os.path.join(OUT, "nav_resumen.png")
    fig.savefig(p); plt.close(fig); print("escrito", p)


# --- Figura 4: clasificacion vs. navegacion --------------------------------
def fig_clf_vs_nav():
    # test_acc de clasificacion (10 clases) por modelo, de las tres comparativas.
    clf = {}
    for path in (CMP_FAMILIAS, CMP_CNN, CMP_TL):
        for r in read_summary(path):
            clf.setdefault(r["model"], float(r["test_acc"]))
    agg = nav_aggregate()

    xs, ys, labels = [], [], []
    for m in NAV_ORDER:
        if m in clf and m in agg:
            xs.append(clf[m]); ys.append(agg[m]["correct"]); labels.append(m)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(xs, ys, s=60, color="#3b6fb0", zorder=3)
    for x, y, l in zip(xs, ys, labels):
        ax.annotate(l, (x, y), textcoords="offset points", xytext=(6, 4),
                    fontsize=9)
    ax.set_xlabel("Exactitud en clasificación (test, 10 clases)")
    ax.set_ylabel("Calidad de percepción en navegación\n"
                  "(fracción correcta media en vivo)")
    ax.set_title("Clasificar bien no garantiza navegar bien")
    ax.set_xlim(0.6, 0.95); ax.set_ylim(0, 1.05)
    fig.tight_layout()
    p = os.path.join(OUT, "clf_vs_nav.png")
    fig.savefig(p); plt.close(fig); print("escrito", p)


if __name__ == "__main__":
    fig_familias()
    fig_todas()
    fig_nav()
    fig_clf_vs_nav()
    print("Figuras generadas en", OUT)
