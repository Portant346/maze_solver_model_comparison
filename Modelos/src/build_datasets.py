"""Construye datasets para la comparativa 'solo-nuevo' vs 'unificado'.

Fuentes:
  - dataset/   : 10 clases (ingles), 320x200, ya con split train/val/test.
  - imagenes/  : 6 clases (español), 180x100, plano, con aumentos en el nombre
                 (flipH_, flipV_, rot180_...). Etiquetas ya correctas.

Salidas:
  - dataset_nuevo/      : 6 clases (ingles), SOLO con imagenes/.
  - dataset_unificado/  : 10 clases = copia de dataset/ + imagenes/ mapeadas.

El split es AGRUPADO por imagen base: todas las variantes aumentadas de una
misma foto (image123, flipH_image123, rot180_image123...) van al MISMO split,
para que no haya fugas (casi-duplicados repartidos entre train y test).

Los ficheros basura '*.png:Zone.Identifier' se ignoran solos (glob '*.png').

Uso (desde Modelos/src):
    python build_datasets.py            # construye ambos (reparto 70/15/15)
    python build_datasets.py --dry-run
"""
import argparse
import re
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
IMAGENES = ROOT / "imagenes"
OUT_NUEVO = ROOT / "dataset_nuevo"
OUT_UNIF = ROOT / "dataset_unificado"

ES2EN = {
    "Flecha_Izq": "ARROW_LEFT", "Flecha_Der": "ARROW_RIGHT", "Flecha_arriba": "ARROW_UP",
    "Cruz": "CROSS", "Meta": "GOAL", "Adelante": "FREE_PATH",
}
BASE_RE = re.compile(r"image\d+")


def base_id(path):
    m = BASE_RE.search(path.name)
    return m.group(0) if m else path.stem


def grouped_split(files, train, val, rng):
    groups = {}
    for f in files:
        groups.setdefault(base_id(f), []).append(f)
    keys = list(groups)
    rng.shuffle(keys)
    n = len(keys)
    n_tr, n_va = int(n * train), int(n * val)
    sel = {"train": keys[:n_tr], "val": keys[n_tr:n_tr + n_va], "test": keys[n_tr + n_va:]}
    return {s: [f for k in ks for f in groups[k]] for s, ks in sel.items()}


def copy_into(splits, out_dir, cls, dry):
    counts = {}
    for split, files in splits.items():
        counts[split] = len(files)
        if not dry:
            dest = out_dir / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dest / f.name)
    return counts


def main():
    p = argparse.ArgumentParser(description="Construye dataset_nuevo y dataset_unificado")
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.15)  # test = resto
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only", choices=["nuevo", "unificado"], default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    rng = random.Random(args.seed)

    # --- dataset_nuevo: 6 clases (ingles), solo imagenes/ ---
    if args.only in (None, "nuevo"):
        print("=== dataset_nuevo (6 clases, solo imagenes/) ===")
        if OUT_NUEVO.exists() and not args.dry_run:
            shutil.rmtree(OUT_NUEVO)
        for es, en in ES2EN.items():
            files = sorted((IMAGENES / es).glob("*.png"))
            sp = grouped_split(files, args.train, args.val, rng)
            c = copy_into(sp, OUT_NUEVO, en, args.dry_run)
            print(f"  {en:11s} {len(files):4d} -> tr {c['train']}, val {c['val']}, test {c['test']}")

    # --- dataset_unificado: copia dataset/ (10 clases) + imagenes/ mapeadas ---
    if args.only in (None, "unificado"):
        print("\n=== dataset_unificado (10 clases = dataset + imagenes) ===")
        if not args.dry_run:
            if OUT_UNIF.exists():
                shutil.rmtree(OUT_UNIF)
            shutil.copytree(DATASET, OUT_UNIF)  # base: las 10 clases con su split
        for es, en in ES2EN.items():
            files = sorted((IMAGENES / es).glob("*.png"))
            sp = grouped_split(files, args.train, args.val, rng)
            c = copy_into(sp, OUT_UNIF, en, args.dry_run)
            print(f"  +{en:11s} {len(files):4d} -> tr {c['train']}, val {c['val']}, test {c['test']}")

    if args.dry_run:
        print("\n(DRY-RUN: no se ha copiado nada)")
    else:
        print(f"\nHecho. Entrenar p.ej.:")
        print("  python train.py --data-dir ../dataset_nuevo --run-name cnn_nuevo "
              "--epochs 40 --lr 3e-4 --optimizer adamw --dropout 0.3 --scheduler cosine")
        print("  python train.py --data-dir ../dataset_unificado --run-name cnn_unificado "
              "--epochs 40 --lr 3e-4 --optimizer adamw --dropout 0.3 --scheduler cosine")


if __name__ == "__main__":
    main()
