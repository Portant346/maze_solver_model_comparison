"""Graficas de estadisticas: curvas de entrenamiento, matriz de confusion
y comparativa entre configuraciones del barrido de hiperparametros.
"""
import matplotlib
matplotlib.use("Agg")  # backend sin ventana, solo guarda a fichero
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_training_curves(history, out_path, title="Entrenamiento"):
    """Curvas de loss y accuracy (train vs val) frente a la epoca."""
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax1.set_xlabel("epoca"); ax1.set_ylabel("loss"); ax1.set_title("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax2.plot(epochs, [h["val_acc"] for h in history], label="val")
    ax2.set_xlabel("epoca"); ax2.set_ylabel("accuracy"); ax2.set_title("Accuracy")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, out_path, normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        with np.errstate(all="ignore"):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("prediccion"); ax.set_ylabel("real")
    ax.set_title("Matriz de confusion" + (" (normalizada)" if normalize else ""))

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_norm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if val > (0.5 if normalize else cm.max() / 2) else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_sweep_curves(runs, out_path, metric="val_acc"):
    """Superpone la curva `metric` de todas las configuraciones del sweep.

    runs: lista de dicts {"name": str, "history": [..]}
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    for run in runs:
        epochs = [h["epoch"] for h in run["history"]]
        ax.plot(epochs, [h[metric] for h in run["history"]], label=run["name"], alpha=0.8)
    ax.set_xlabel("epoca"); ax.set_ylabel(metric)
    ax.set_title(f"Comparativa de configuraciones ({metric})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_sweep_bars(summary, out_path, metric="best_val_acc"):
    """Barras del `metric` final por configuracion, ordenadas de mejor a peor.

    summary: lista de dicts con al menos "name" y la clave `metric`.
    """
    items = sorted(summary, key=lambda r: r[metric], reverse=True)
    names = [r["name"] for r in items]
    values = [r[metric] for r in items]

    fig, ax = plt.subplots(figsize=(11, max(4, 0.4 * len(names))))
    ypos = np.arange(len(names))
    ax.barh(ypos, values, color="steelblue")
    ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(metric); ax.set_title(f"{metric} por configuracion")
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
