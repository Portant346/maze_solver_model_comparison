"""Mezcla los frames capturados en vivo (Modelos/dataset_live/<CLASE>/) dentro
del dataset principal (train/val/test) con un reparto configurable.

Los ficheros capturados empiezan por 'live_', asi que quedan identificados y se
pueden quitar luego si hace falta. Se COPIAN (dataset_live se conserva como
respaldo).

Uso (desde Modelos/src):
    python merge_live_data.py            # reparte 70/15/15
    python merge_live_data.py --dry-run  # solo muestra que haria
"""
import argparse
import random
import shutil
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SRC_DIR.parent / "dataset"
DEFAULT_LIVE = SRC_DIR.parent / "dataset_live"

CLASSES = ["ARROW_LEFT", "ARROW_RIGHT", "ARROW_UP", "CROSS", "FREE_PATH",
           "GOAL", "UNKNOWN", "WALL", "WALL_LEFT", "WALL_RIGHT"]


def parse_args():
    p = argparse.ArgumentParser(description="Mezcla dataset_live en el dataset")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--live-dir", default=str(DEFAULT_LIVE))
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.15)  # test = resto
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    dataset, live = Path(args.dataset), Path(args.live_dir)
    totals = {"train": 0, "val": 0, "test": 0}

    for cls in CLASSES:
        cls_dir = live / cls
        if not cls_dir.is_dir():
            continue
        imgs = sorted(cls_dir.glob("*.png"))
        if not imgs:
            continue
        rng.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        splits = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train + n_val],
            "test": imgs[n_train + n_val:],
        }
        print(f"{cls:11s} {n:3d} -> train {len(splits['train'])}, "
              f"val {len(splits['val'])}, test {len(splits['test'])}")
        for split, files in splits.items():
            dest = dataset / split / cls
            if not args.dry_run:
                dest.mkdir(parents=True, exist_ok=True)
                for f in files:
                    shutil.copy2(f, dest / f.name)
            totals[split] += len(files)

    print(f"\nTotal a añadir -> train {totals['train']}, val {totals['val']}, "
          f"test {totals['test']}" + ("  (DRY-RUN, nada copiado)" if args.dry_run else ""))
    if not args.dry_run and sum(totals.values()):
        print("Hecho. Reentrena con:  python train.py --epochs 50 --lr 3e-4 "
              "--optimizer adamw --dropout 0.3 --flip-swap --scheduler cosine "
              "--run-name cnn_v2")


if __name__ == "__main__":
    main()
