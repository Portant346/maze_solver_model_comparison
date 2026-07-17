"""Barrido de hiperparametros + graficas comparativas.

Entrena el modelo con cada combinacion de la rejilla (grid), recoge las
estadisticas de cada entrenamiento y produce:
    runs/sweep_<ts>/summary.csv        una fila por configuracion
    runs/sweep_<ts>/compare_val_acc.png  curvas val_acc superpuestas
    runs/sweep_<ts>/compare_val_loss.png curvas val_loss superpuestas
    runs/sweep_<ts>/ranking_val_acc.png  barras del mejor val_acc por config
    runs/sweep_<ts>/ranking_test_acc.png barras del test_acc por config

Edita GRID abajo para elegir que hiperparametros combinar. El numero de
entrenamientos es el producto de las longitudes de cada lista.

Uso (desde Modelos/src):
    python sweep.py --epochs 20
"""
import argparse
import csv
import itertools
from datetime import datetime
from pathlib import Path

import data as data_mod
import plots
from train import run_training

SRC_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SRC_DIR.parent / "dataset"
DEFAULT_RUNS = SRC_DIR.parent / "runs"

# --- Rejilla de hiperparametros a combinar (editable) ---
# Barrido reducido (6 configs) para comparar la sensibilidad a hiperparametros
# DENTRO de cada tipo de modelo sin disparar el coste en CPU (se corre por cada
# modelo: cnn/mlp/rnn/transformer). Ejes:
#   lr      -> principal sospechoso de la oscilacion de val_acc
#   dropout -> cuanta regularizacion ayuda con pocos datos
# El optimizador y el scheduler se fijan en base_cfg (adamw + cosine) para que
# las configuraciones sean comparables entre si. 3 x 2 = 6 configuraciones.
GRID = {
    "lr": [1e-3, 3e-4, 1e-4],
    "dropout": [0.3, 0.5],
}


def short(name, value):
    """Etiqueta corta y legible para identificar la config en las graficas."""
    abbr = {"lr": "lr", "batch_size": "bs", "optimizer": "opt",
            "dropout": "do", "base_channels": "ch"}
    return f"{abbr.get(name, name)}{value}"


def parse_args():
    p = argparse.ArgumentParser(description="Barrido de hiperparametros")
    p.add_argument("--model", default="cnn")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA))
    p.add_argument("--out-dir", default=str(DEFAULT_RUNS))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def base_cfg(args, sweep_dir):
    return {
        "model": args.model, "data_dir": args.data_dir, "out_dir": str(sweep_dir),
        "epochs": args.epochs, "batch_size": 32, "lr": 1e-3, "optimizer": "adamw",
        "weight_decay": 1e-4, "dropout": 0.3, "base_channels": 32,
        "img_h": data_mod.DEFAULT_IMG_H, "img_w": data_mod.DEFAULT_IMG_W,
        "augment": True, "class_weights": True, "flip_swap": False,
        "scheduler": "cosine", "seed": args.seed,
    }


def main():
    args = parse_args()
    sweep_dir = Path(args.out_dir) / (
        f"sweep_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)

    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"Barrido: {len(combos)} configuraciones x {args.epochs} epocas\n")

    runs, summary = [], []
    for i, values in enumerate(combos, 1):
        cfg = base_cfg(args, sweep_dir)
        label_parts = []
        for k, v in zip(keys, values):
            cfg[k] = v
            label_parts.append(short(k, v))
        run_name = f"{i:02d}_" + "_".join(label_parts)
        cfg["run_name"] = run_name

        print(f"=== [{i}/{len(combos)}] {run_name} ===")
        history, metrics = run_training(cfg, verbose=True)
        runs.append({"name": run_name, "history": history})
        summary.append(metrics)
        print()

    # --- Resumen CSV (una fila por configuracion) ---
    cols = ["run_name", "lr", "batch_size", "optimizer", "dropout", "base_channels",
            "best_val_acc", "best_epoch", "test_acc", "test_f1_macro",
            "inference_ms_per_img", "num_params"]
    with open(sweep_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(summary, key=lambda r: r["best_val_acc"], reverse=True))

    # --- Graficas comparativas ---
    plots.plot_sweep_curves(runs, sweep_dir / "compare_val_acc.png", metric="val_acc")
    plots.plot_sweep_curves(runs, sweep_dir / "compare_val_loss.png", metric="val_loss")
    plots.plot_sweep_bars([{"name": m["run_name"], "best_val_acc": m["best_val_acc"]}
                           for m in summary], sweep_dir / "ranking_val_acc.png",
                          metric="best_val_acc")
    plots.plot_sweep_bars([{"name": m["run_name"], "test_acc": m["test_acc"]}
                           for m in summary], sweep_dir / "ranking_test_acc.png",
                          metric="test_acc")

    best = max(summary, key=lambda r: r["best_val_acc"])
    print(f"Mejor configuracion: {best['run_name']} "
          f"(val_acc={best['best_val_acc']:.3f}, test_acc={best['test_acc']:.3f})")
    print(f"Resultados del barrido en: {sweep_dir}")


if __name__ == "__main__":
    main()
