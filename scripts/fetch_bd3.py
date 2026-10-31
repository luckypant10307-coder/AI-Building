"""Fetch the full BD3 dataset from Kaggle and report what landed.

Needs a Kaggle API token, which only you can create:
    kaggle.com -> avatar -> Settings -> API -> "Create New Token"
    save the downloaded kaggle.json to ~/.kaggle/kaggle.json

Then:
    pip install kaggle
    python scripts/fetch_bd3.py

It downloads, extracts, finds the folder that actually holds the class
directories, and prints the exact training command for it.
"""

import argparse
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

DATASET = "praveenkottari/bd3-dataset-for-building-defect-detection"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# BD3 ships an augmented copy (14,000 images) derived from the original 3,965.
# Training on both puts transformed copies of a test image into training.
AUGMENT_HINTS = ("augment", "aug_", "-aug", "augmented")


def token_present():
    """Any of the three credential shapes the Kaggle client accepts."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True                                     # newer standalone token
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True                                     # classic username + key
    config = Path(os.environ.get("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle")
    return (config / "access_token").exists() or (config / "kaggle.json").exists()


def download(dest):
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download",
           "-d", DATASET, "-p", str(dest), "--unzip"]
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit("kaggle download failed -- see the message above")


def class_dirs(root):
    """Folders holding images, with their counts, deepest-first."""
    counts = Counter()
    for path in Path(root).rglob("*"):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            counts[path.parent] += 1
    return counts


def pick_root(counts):
    """The parent whose children look like the original class folders."""
    by_parent = Counter()
    for folder, n in counts.items():
        by_parent[folder.parent] += n
    candidates = [
        (parent, total) for parent, total in by_parent.items()
        if not any(h in str(parent).lower() for h in AUGMENT_HINTS)
    ]
    if not candidates:
        candidates = list(by_parent.items())
    # the original set is ~3,965; prefer the biggest non-augmented grouping
    return max(candidates, key=lambda item: item[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data/bd3_full")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    dest = Path(args.dest)

    if not args.skip_download:
        if not token_present():
            raise SystemExit(
                "No Kaggle token found.\n"
                "  1. kaggle.com -> avatar -> Settings -> API -> Create New Token\n"
                f"  2. save kaggle.json to {Path.home() / '.kaggle' / 'kaggle.json'}\n"
                "  3. pip install kaggle\n"
                "Then run this again.")
        print(f"downloading {DATASET} -> {dest}")
        download(dest)

    counts = class_dirs(dest)
    if not counts:
        raise SystemExit(f"no images found under {dest}")

    root, total = pick_root(counts)
    print(f"\nimage folders found: {len(counts)}")
    for folder, n in sorted(counts.items()):
        mark = "  (augmented, excluded)" if any(
            h in str(folder).lower() for h in AUGMENT_HINTS) else ""
        print(f"  {n:>6}  {folder}{mark}")

    print(f"\nclass-folder root : {root}")
    print(f"images under it   : {total}")
    if total < 3000:
        print("  warning: BD3's original set is about 3,965 images. This looks "
              "smaller -- check the folder above is the right one.")
    print("\nNow run:")
    print(f"  python scripts/train_defect_classifier.py --data \"{root}\" --epochs 12")
    print(f"  python scripts/train_defect_classifier.py --data \"{root}\" "
          "--epochs 12 --split random")
    print("\nReport both splits. The gap between them is leakage, not skill.")


if __name__ == "__main__":
    main()
