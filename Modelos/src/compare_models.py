"""Comparativa entre TIPOS de modelo (cnn / mlp / rnn / transformer).

Entrena cada arquitectura del registro con la MISMA receta (mismos datos,
epocas, lr, optimizador, dropout, scheduler...) para que la comparacion sea
apples-to-apples y mida la diferencia que aporta la *familia* de modelo, no el
ajuste de hiperparametros. Para comparar configuraciones de hiperparametros
DENTRO de un modelo, usar sweep.py --model <nombre>.

Produce en runs/compare_<ts>/:
    <modelo>/                 salida completa de cada entrenamiento (igual que train.py)
    summary.csv               una fila por modelo (test_acc, F1, params, ms/img...)
    compare_val_acc.png       curvas val_acc superpuestas de los modelos
    compare_val_loss.png      curvas val_loss superpuestas
    ranking_test_acc.png      barras de test_acc por modelo
    ranking_test_f1.png       barras de F1 macro por modelo
    ranking_num_params.png    barras de nº de parametros por modelo
    ranking_infer_ms.png      barras de tiempo de inferencia por modelo

Uso (desde Modelos/src), receta = la de cnn_unificado:
    python compare_models.py --data-dir ../dataset_unificado --epochs 30 \
           --lr 3e-4 --optimizer adamw --dropout 0.3 --scheduler cosine
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import data as data_mod
import plots
from models import MODEL_REGISTRY
from train import run_training

SRC_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SRC_DIR.parent / "dataset_unificado"
DEFAULT_RUNS = SRC_DIR.parent / "runs"


def parse_args():
    p = argparse.ArgumentParser(description="Comparativa entre tipos de modelo")
    p.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY),
                   help=f"Modelos a comparar (def: todos). Disponibles: {list(MODEL_REGISTRY)}")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA))
    p.add_argument("--out-dir", default=str(DEFAULT_RUNS))
    # Receta comun (por defecto, la de cnn_unificado)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--optimizer", default="adamw", choices=["adam", "adamw", "sgd"])
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--img-h", type=int, default=data_mod.DEFAULT_IMG_H)
    p.add_argument("--img-w", type=int, default=data_mod.DEFAULT_IMG_W)
    p.add_argument("--scheduler", default="cosine", choices=["none", "cosine"])
    p.add_argument("--flip-swap", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def cfg_for(args, model, compare_dir):
    return {
        "model": model, "data_dir": args.data_dir, "out_dir": str(compare_dir),
        "run_name": model, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "optimizer": args.optimizer, "weight_decay": args.weight_decay,
        "dropout": args.dropout, "base_channels": args.base_channels,
        "img_h": args.img_h, "img_w": args.img_w,
        "augment": not args.no_augment, "class_weights": not args.no_class_weights,
        "flip_swap": args.flip_swap, "scheduler": args.scheduler, "seed": args.seed,
    }


def main():
    args = parse_args()
    compare_dir = Path(args.out_dir) / f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    compare_dir.mkdir(parents=True, exist_ok=True)

    print(f"Comparativa de {len(args.models)} modelos x {args.epochs} epocas")
    print(f"Receta: lr={args.lr} opt={args.optimizer} dropout={args.dropout} "
          f"scheduler={args.scheduler} flip_swap={args.flip_swap} "
          f"base_channels={args.base_channels}")
    print(f"Datos: {args.data_dir}\nSalida: {compare_dir}\n")

    runs, summary = [], []
    for i, model in enumerate(args.models, 1):
        print(f"=== [{i}/{len(args.models)}] modelo={model} ===")
        history, metrics = run_training(cfg_for(args, model, compare_dir), verbose=True)
        runs.append({"name": model, "history": history})
        summary.append(metrics)
        print()

    # --- Resumen CSV (una fila por modelo) ---
    cols = ["model", "run_name", "num_params", "best_val_acc", "best_epoch",
            "test_acc", "test_f1_macro", "inference_ms_per_img",
            "epochs", "lr", "optimizer", "dropout", "base_channels"]
    rows = sorted(summary, key=lambda r: r["test_acc"], reverse=True)
    with open(compare_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # --- Graficas comparativas (reutilizan plots.py) ---
    plots.plot_sweep_curves(runs, compare_dir / "compare_val_acc.png", metric="val_acc")
    plots.plot_sweep_curves(runs, compare_dir / "compare_val_loss.png", metric="val_loss")
    for metric, fname in [("test_acc", "ranking_test_acc.png"),
                          ("test_f1_macro", "ranking_test_f1.png"),
                          ("num_params", "ranking_num_params.png"),
                          ("inference_ms_per_img", "ranking_infer_ms.png")]:
        plots.plot_sweep_bars(
            [{"name": m["model"], metric: m[metric]} for m in summary],
            compare_dir / fname, metric=metric,
        )

    # --- Ranking por consola ---
    print("Ranking por test_acc:")
    print(f"  {'modelo':12s} {'test_acc':>9s} {'F1_macro':>9s} {'params':>12s} {'ms/img':>8s}")
    for r in rows:
        print(f"  {r['model']:12s} {r['test_acc']:>9.3f} {r['test_f1_macro']:>9.3f} "
              f"{r['num_params']:>12,} {r['inference_ms_per_img']:>8.2f}")
    print(f"\nResultados de la comparativa en: {compare_dir}")


if __name__ == "__main__":
    main()
