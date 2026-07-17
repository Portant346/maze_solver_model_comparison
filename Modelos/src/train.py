"""Entrenar UNA configuracion y guardar todas sus estadisticas.

Uso (desde la carpeta Modelos/src):
    python train.py --model cnn --epochs 30 --lr 1e-3 --batch-size 32

Salidas en Modelos/runs/<run-name>/:
    config.json               hiperparametros usados
    history.csv               loss/accuracy por epoca (train y val)
    curves.png                curvas de loss y accuracy
    confusion_matrix.png      matriz de confusion en test
    classification_report.txt precision/recall/F1 por clase (test)
    metrics.json             resumen (mejor val_acc, test_acc, t. inferencia)
    best_model.pt             pesos del mejor modelo (por val_acc)
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score

import data as data_mod
import engine
import plots
from models import build_model, count_parameters

SRC_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SRC_DIR.parent / "dataset"
DEFAULT_RUNS = SRC_DIR.parent / "runs"


def parse_args():
    p = argparse.ArgumentParser(description="Entrenar un modelo de clasificacion")
    p.add_argument("--model", default="cnn")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA))
    p.add_argument("--out-dir", default=str(DEFAULT_RUNS))
    p.add_argument("--run-name", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "sgd"])
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--img-h", type=int, default=data_mod.DEFAULT_IMG_H)
    p.add_argument("--img-w", type=int, default=data_mod.DEFAULT_IMG_W)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--flip-swap", action="store_true",
                   help="Aumento por flip horizontal con intercambio de etiqueta (clases L/R)")
    p.add_argument("--scheduler", default="none", choices=["none", "cosine"],
                   help="Scheduler de learning rate")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_training(cfg, verbose=True):
    """Ejecuta un entrenamiento completo descrito por el dict `cfg`.

    Devuelve (history, metrics) y persiste todo en out-dir/run-name.
    Reutilizable desde sweep.py.
    """
    engine.set_seed(cfg["seed"])
    device = engine.get_device()

    run_name = cfg.get("run_name") or (
        f"{cfg['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = Path(cfg["out_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    loaders = data_mod.create_dataloaders(
        cfg["data_dir"], img_h=cfg["img_h"], img_w=cfg["img_w"],
        batch_size=cfg["batch_size"], augment=cfg["augment"],
        flip_swap=cfg.get("flip_swap", False),
    )
    classes = loaders["classes"]
    num_classes = len(classes)

    model = build_model(
        cfg["model"], num_classes=num_classes,
        base_channels=cfg["base_channels"], dropout=cfg["dropout"],
        img_h=cfg["img_h"], img_w=cfg["img_w"],
    ).to(device)

    weight = loaders["class_weights"].to(device) if cfg["class_weights"] else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = engine.build_optimizer(
        cfg["optimizer"], model.parameters(), lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = None
    if cfg.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"]
        )

    if verbose:
        print(f"[{run_name}] device={device} | clases={num_classes} | "
              f"params={count_parameters(model):,}")

    ckpt = run_dir / "best_model.pt"
    history, best = engine.train_model(
        model, loaders, criterion, optimizer, device,
        epochs=cfg["epochs"], checkpoint_path=ckpt, scheduler=scheduler,
        verbose=verbose,
    )

    # --- Evaluacion final en test con el mejor modelo ---
    model.load_state_dict(torch.load(ckpt, map_location=device))
    test_loss, test_acc, y_true, y_pred = engine.evaluate(
        model, loaders["test"], criterion, device
    )
    test_f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Tiempo de inferencia por imagen (relevante para tiempo real en el robot)
    infer_ms = measure_inference_time(model, device, cfg["img_h"], cfg["img_w"])

    # --- Persistir estadisticas ---
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    with open(run_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    report = classification_report(y_true, y_pred, target_names=classes, digits=3,
                                   zero_division=0)
    (run_dir / "classification_report.txt").write_text(report)

    metrics = {
        "run_name": run_name,
        "model": cfg["model"],
        "best_val_acc": best["val_acc"],
        "best_epoch": best["epoch"],
        "test_acc": test_acc,
        "test_loss": test_loss,
        "test_f1_macro": test_f1_macro,
        "inference_ms_per_img": infer_ms,
        "num_params": count_parameters(model),
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
        "batch_size": cfg["batch_size"],
        "optimizer": cfg["optimizer"],
        "dropout": cfg["dropout"],
        "base_channels": cfg["base_channels"],
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plots.plot_training_curves(history, run_dir / "curves.png", title=run_name)
    plots.plot_confusion_matrix(y_true, y_pred, classes, run_dir / "confusion_matrix.png")

    if verbose:
        print(f"[{run_name}] best_val_acc={best['val_acc']:.3f} "
              f"test_acc={test_acc:.3f} test_f1={test_f1_macro:.3f} "
              f"infer={infer_ms:.1f} ms/img")
        print(f"  -> resultados en {run_dir}")

    return history, metrics


@torch.no_grad()
def measure_inference_time(model, device, img_h, img_w, n=50):
    model.eval()
    x = torch.randn(1, 3, img_h, img_w, device=device)
    for _ in range(5):  # calentamiento
        model(x)
    t0 = time.time()
    for _ in range(n):
        model(x)
    return (time.time() - t0) / n * 1000.0


def cfg_from_args(args):
    return {
        "model": args.model,
        "data_dir": args.data_dir,
        "out_dir": args.out_dir,
        "run_name": args.run_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "base_channels": args.base_channels,
        "img_h": args.img_h,
        "img_w": args.img_w,
        "augment": not args.no_augment,
        "class_weights": not args.no_class_weights,
        "flip_swap": args.flip_swap,
        "scheduler": args.scheduler,
        "seed": args.seed,
    }


if __name__ == "__main__":
    run_training(cfg_from_args(parse_args()))
