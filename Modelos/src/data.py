"""Carga del dataset y preparacion de DataLoaders.

El dataset tiene 10 clases en train/val/test (ImageFolder):
  ARROW_LEFT, ARROW_RIGHT, ARROW_UP, CROSS, FREE_PATH,
  GOAL, UNKNOWN, WALL, WALL_LEFT, WALL_RIGHT

Sobre el volteo horizontal:
  - NO se usa RandomHorizontalFlip "ciego" como aumento, porque cambiaria el
    significado de las clases direccionales (ARROW_LEFT/RIGHT, WALL_LEFT/RIGHT).
  - SI se ofrece (flip_swap=True) un aumento que voltea horizontalmente Y
    reasigna la etiqueta cuando corresponde: ARROW_LEFT<->ARROW_RIGHT y
    WALL_LEFT<->WALL_RIGHT; el resto de clases conservan su etiqueta (son
    simetricas o no direccionales en el eje horizontal). Esto duplica el train
    y equilibra las clases izquierda/derecha, que son las mas debiles.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# Normalizacion estandar de ImageNet (util tambien para transfer learning futuro)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Resolucion de la camara del robot: 320x200 (ancho x alto), relacion 16:10.
# Reescalamos manteniendo esa relacion para no deformar las flechas.
DEFAULT_IMG_H = 100
DEFAULT_IMG_W = 160

# Pares de clases que intercambian etiqueta al voltear horizontalmente.
HFLIP_SWAPS = [("ARROW_LEFT", "ARROW_RIGHT"), ("WALL_LEFT", "WALL_RIGHT")]


def build_transforms(img_h=DEFAULT_IMG_H, img_w=DEFAULT_IMG_W, train=False, augment=True):
    """Transformaciones para train o para val/test (sin flip horizontal aqui)."""
    base = [transforms.Resize((img_h, img_w))]
    if train and augment:
        # Aumentos seguros para clases direccionales (sin flip horizontal):
        base += [
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ]
    base += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(base)


def build_swap_map(class_names):
    """idx_clase -> idx_clase tras voltear horizontalmente (identidad si no aplica)."""
    name_to_idx = {c: i for i, c in enumerate(class_names)}
    swap = {i: i for i in range(len(class_names))}
    for a, b in HFLIP_SWAPS:
        if a in name_to_idx and b in name_to_idx:
            swap[name_to_idx[a]] = name_to_idx[b]
            swap[name_to_idx[b]] = name_to_idx[a]
    return swap


class FlipSwapDataset(Dataset):
    """Duplica el dataset: cada muestra aparece como original y como volteada
    horizontalmente con la etiqueta reasignada segun build_swap_map.

    El flip se aplica sobre la imagen PIL ANTES del resto de transforms.
    """

    def __init__(self, root, transform, swap_map, classes):
        self.base = datasets.ImageFolder(root, transform=None)  # devuelve (PIL, label)
        self.transform = transform
        self.swap_map = swap_map
        self.classes = classes
        self.n = len(self.base)
        # Etiquetas efectivas de las 2N entradas (para pesos de clase / stats)
        self.labels = [lbl for _, lbl in self.base.samples] + \
                      [swap_map[lbl] for _, lbl in self.base.samples]

    def __len__(self):
        return 2 * self.n

    def __getitem__(self, idx):
        if idx < self.n:
            img, label = self.base[idx]
        else:
            img, label = self.base[idx - self.n]
            img = ImageOps.mirror(img)        # volteo horizontal
            label = self.swap_map[label]      # reasignacion de etiqueta
        return self.transform(img), label


def create_datasets(data_dir, img_h=DEFAULT_IMG_H, img_w=DEFAULT_IMG_W,
                    augment=True, flip_swap=False):
    data_dir = Path(data_dir)
    train_tf = build_transforms(img_h, img_w, train=True, augment=augment)
    eval_tf = build_transforms(img_h, img_w, train=False)

    # Necesitamos los nombres de clase para construir el mapa de swap
    classes = datasets.ImageFolder(data_dir / "train").classes
    if flip_swap:
        train_ds = FlipSwapDataset(data_dir / "train", train_tf,
                                   build_swap_map(classes), classes)
    else:
        train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)

    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)
    return train_ds, val_ds, test_ds, classes


def compute_class_weights(train_ds, num_classes):
    """Pesos inversamente proporcionales a la frecuencia de cada clase.

    Funciona tanto con ImageFolder (.samples) como con FlipSwapDataset (.labels).
    """
    if hasattr(train_ds, "labels"):
        labels = train_ds.labels
    else:
        labels = [lbl for _, lbl in train_ds.samples]
    counts = np.zeros(num_classes, dtype=np.float64)
    for lbl in labels:
        counts[lbl] += 1
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(data_dir, img_h=DEFAULT_IMG_H, img_w=DEFAULT_IMG_W,
                       batch_size=32, augment=True, flip_swap=False, num_workers=0):
    """Crea los DataLoaders y devuelve tambien nombres de clase y pesos.

    num_workers=0 por defecto: en Windows los workers de multiprocessing dan
    problemas si el script no esta protegido con if __name__ == '__main__'.
    """
    train_ds, val_ds, test_ds, classes = create_datasets(
        data_dir, img_h, img_w, augment, flip_swap
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    class_weights = compute_class_weights(train_ds, len(classes))
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "classes": classes,
        "class_weights": class_weights,
    }
