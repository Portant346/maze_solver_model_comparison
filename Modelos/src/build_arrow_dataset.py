"""Construye `dataset_arrows/` recortando cada flecha al bbox de su verde.

Lee `dataset_unificado/{train,val,test}/{ARROW_LEFT,ARROW_RIGHT,ARROW_UP}/`
y guarda en `dataset_arrows/<split>/<clase>/` las mismas imagenes recortadas
al bounding box del verde (con margen, cuadrado por defecto). Las imagenes
sin verde detectable se descartan y se cuentan para diagnostico.

Es el primer paso del pipeline two-stage: el clasificador de flechas se
entrenara sobre estos crops, en los que la flecha ya llena la imagen (no
unos pocos pixeles en una esquina).

Uso (desde Modelos/src):
    python build_arrow_dataset.py
    python build_arrow_dataset.py --margin 0.20 --no-square --in dataset --out dataset_arrows2
"""
import argparse
from pathlib import Path

import cv2

from vision_utils import green_bbox


ARROW_CLASSES = ("ARROW_LEFT", "ARROW_RIGHT", "ARROW_UP")
SPLITS = ("train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser(description="Construye dataset_arrows recortando al verde")
    src_dir = Path(__file__).resolve().parent.parent
    p.add_argument("--in", dest="src", default=str(src_dir / "dataset_unificado"),
                   help="Dataset fuente (por defecto: Modelos/dataset_unificado)")
    p.add_argument("--out", default=str(src_dir / "dataset_arrows"),
                   help="Dataset destino (por defecto: Modelos/dataset_arrows)")
    p.add_argument("--margin", type=float, default=0.15,
                   help="Margen relativo alrededor del bbox del verde (def 0.15)")
    p.add_argument("--no-square", action="store_true",
                   help="No forzar bbox cuadrado (por defecto SI se hace, evita distorsion)")
    p.add_argument("--min-px", type=int, default=30,
                   help="Pixeles verdes minimos para conservar la imagen (def 30)")
    return p.parse_args()


def process_image(src_path, dst_path, margin, square, min_px):
    """Devuelve True si la imagen tenia verde y se guardo, False si se descarto."""
    bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return False
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bbox = green_bbox(rgb, margin=margin, min_px=min_px, square=square)
    if bbox is None:
        return False
    x0, y0, x1, y1, _ = bbox
    crop_rgb = rgb[y0:y1, x0:x1]
    if crop_rgb.size == 0:
        return False
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), crop_bgr)
    return True


def main():
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.out)
    square = not args.no_square

    print(f"Fuente:  {src_root}")
    print(f"Destino: {dst_root}")
    print(f"Parametros: margin={args.margin}, square={square}, min_px={args.min_px}")
    print()

    grand_kept = 0
    grand_drop = 0
    for split in SPLITS:
        for cls in ARROW_CLASSES:
            src_dir = src_root / split / cls
            dst_dir = dst_root / split / cls
            if not src_dir.is_dir():
                print(f"  [WARN] no existe {src_dir}")
                continue
            kept = drop = 0
            for img_path in sorted(src_dir.glob("*.png")):
                dst_path = dst_dir / img_path.name
                if process_image(img_path, dst_path, args.margin, square, args.min_px):
                    kept += 1
                else:
                    drop += 1
            total = kept + drop
            pct = 100.0 * drop / total if total else 0.0
            print(f"  {split:5s}/{cls:11s}  kept={kept:4d}  drop={drop:3d}  ({pct:.1f}% sin verde)")
            grand_kept += kept
            grand_drop += drop

    total = grand_kept + grand_drop
    pct = 100.0 * grand_drop / total if total else 0.0
    print()
    print(f"TOTAL  kept={grand_kept}  drop={grand_drop}  ({pct:.1f}% sin verde)")


if __name__ == "__main__":
    main()
