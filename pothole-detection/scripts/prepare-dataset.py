"""Builds the labeled pothole / no-pothole image folders under data/.

Source: "Cracks and Potholes in Road Images Dataset" (Passos et al., 2020),
2235 road-surface photographs captured by Brazil's National Department of
Transport Infrastructure, each shipped with a binary pothole mask. We derive
a binary classification label per image directly from that mask: an image
counts as "pothole" if its pothole mask has any annotated pixels, and
"no_pothole" otherwise (the image may still show a crack, just no pothole).

Usage:
    python scripts/prepare_dataset.py
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_URL = "https://github.com/biankatpas/Cracks-and-Potholes-in-Road-Images-Dataset.git"
POTHOLE_PIXEL_THRESHOLD = 50  # minimum annotated pixels to count as a pothole


def clone_source(tmp_dir: Path) -> Path:
    dataset_dir = tmp_dir / "repo" / "Dataset"
    if dataset_dir.exists():
        return dataset_dir
    print(f"Cloning source dataset into {tmp_dir} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(tmp_dir / "repo")],
        check=True,
    )
    return dataset_dir


def label_from_mask(mask_path: Path) -> int:
    mask = np.array(Image.open(mask_path).convert("L"))
    return int((mask > 10).sum() > POTHOLE_PIXEL_THRESHOLD)


def build_dataset(dataset_dir: Path, out_dir: Path) -> None:
    pothole_dir = out_dir / "pothole"
    no_pothole_dir = out_dir / "no_pothole"
    pothole_dir.mkdir(parents=True, exist_ok=True)
    no_pothole_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    n_pothole = n_no_pothole = n_skipped = 0

    for i, sample_dir in enumerate(sample_dirs):
        raw = next(sample_dir.glob("*_RAW.jpg"), None)
        mask = next(sample_dir.glob("*_POTHOLE.png"), None)
        if raw is None or mask is None:
            n_skipped += 1
            continue

        label = label_from_mask(mask)
        dest = (pothole_dir if label else no_pothole_dir) / f"{sample_dir.name}.jpg"
        shutil.copyfile(raw, dest)
        if label:
            n_pothole += 1
        else:
            n_no_pothole += 1

        if (i + 1) % 400 == 0:
            print(f"  processed {i + 1}/{len(sample_dirs)}")

    print(f"Done. pothole={n_pothole} no_pothole={n_no_pothole} skipped={n_skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="data", help="output directory (default: data/)"
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="keep the cloned source repo instead of deleting it after use",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    if any(out_dir.glob("*/*.jpg")):
        print(f"{out_dir} already looks populated, skipping. Delete it to rebuild.")
        sys.exit(0)

    tmp_dir = Path("/tmp/cnn_street_source")
    dataset_dir = clone_source(tmp_dir)
    build_dataset(dataset_dir, out_dir)

    if not args.keep_source:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()