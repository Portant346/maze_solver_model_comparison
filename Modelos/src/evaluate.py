"""Evaluar un checkpoint ya entrenado sobre el conjunto de test.

Recalcula accuracy, F1 macro, reporte por clase, matriz de confusion y
tiempo de inferencia. Util para revisar un modelo sin reentrenar.

Uso (desde Modelos/src):
    python evaluate.py --run-dir ../runs/<nombre_del_run>
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score

import data as data_mod
import engine
import plots
from models import build_model
from train import measure_inference_time

SRC_DIR = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Evaluar un checkpoint en test")
    p.add_argument("--run-dir", required=True, help="Carpeta del run con config.json y best_model.pt")
    p.add_argument("--data-dir", default=str(SRC_DIR.parent / "dataset"))
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    device = engine.get_device()

    loaders = data_mod.create_dataloaders(
        args.data_dir, img_h=cfg["img_h"], img_w=cfg["img_w"],
        batch_size=cfg["batch_size"], augment=False,
    )
    classes = loaders["classes"]

    model = build_model(cfg["model"], num_classes=len(classes),
                        base_channels=cfg["base_channels"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred = engine.evaluate(model, loaders["test"], criterion, device)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    infer_ms = measure_inference_time(model, device, cfg["img_h"], cfg["img_w"])

    print(f"test_acc={test_acc:.3f} | test_loss={test_loss:.3f} | "
          f"f1_macro={f1:.3f} | infer={infer_ms:.1f} ms/img\n")
    print(classification_report(y_true, y_pred, target_names=classes, digits=3,
                                zero_division=0))

    plots.plot_confusion_matrix(y_true, y_pred, classes, run_dir / "confusion_matrix_eval.png")
    print(f"Matriz de confusion guardada en {run_dir / 'confusion_matrix_eval.png'}")


if __name__ == "__main__":
    main()
