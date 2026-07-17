"""Motor de entrenamiento y evaluacion.

Independiente del tipo de modelo: registra estadisticas por epoca
(loss y accuracy de train y val), guarda el mejor checkpoint segun val_acc
y devuelve el historial completo para graficarlo y compararlo despues.
"""
import random
import time

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(name, params, lr, weight_decay=0.0, momentum=0.9):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"Optimizador desconocido: {name!r}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Devuelve (loss, accuracy, y_true, y_pred) sobre todo el loader."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return running_loss / total, correct / total, y_true, y_pred


def train_model(model, loaders, criterion, optimizer, device, epochs,
                checkpoint_path=None, scheduler=None, verbose=True):
    """Entrena varias epocas y devuelve (history, best).

    history: lista de dicts por epoca con loss/accuracy de train y val.
    best: dict con la mejor val_acc y la epoca en que se alcanzo.
    """
    history = []
    best = {"val_acc": -1.0, "epoch": -1, "val_loss": float("inf")}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(model, loaders["val"], criterion, device)
        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr,
            "time_s": epoch_time,
        })

        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "val_loss": val_loss, "epoch": epoch}
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)

        if verbose:
            print(f"  epoch {epoch:3d}/{epochs} | "
                  f"train_loss {train_loss:.4f} acc {train_acc:.3f} | "
                  f"val_loss {val_loss:.4f} acc {val_acc:.3f} | {epoch_time:.1f}s")

    return history, best
