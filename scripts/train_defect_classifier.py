"""Fine-tune the BD3 building-defect classifier (component 7).

This is the only model in the project trained on real photographs.

Two split modes, and the difference between them is the point:

  --split grouped  (default) holds out the highest-numbered files in each class.
                   BD3 filenames are sequential per class (cls00_001, cls00_002,
                   ...), and consecutive numbers tend to be the same wall shot
                   from slightly different angles. Keeping a block together
                   approximates a by-wall split.
  --split random   stratified random, which is what the BD3 paper does.

Report both. If random scores much higher than grouped, the gap is near-
duplicate photographs of one wall sitting in train and test at once, not skill.
That is the image-domain version of splitting by seed rather than by row.

Usage:
    python scripts/train_defect_classifier.py --data data/bd3_raw --epochs 12
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from smart_property.vision import (                       # noqa: E402
    DAMP_CLASSES, build_model, eval_transforms, normalise, train_transforms)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Below this many test images, any score is noise. Say so rather than print it.
MIN_MEANINGFUL_TEST = 50
# BD3 also ships 14,000 augmented images derived from the original 3,965. Mixing
# them in puts rotated and recoloured copies of test images into training, which
# is why the paper's augmented scores (0.99 F1) are so much higher than its
# original ones (0.83). Skipped unless explicitly asked for.
AUGMENT_HINTS = ("augment", "aug_", "-aug")


class ImageList(Dataset):
    def __init__(self, items, classes, tf):
        self.items, self.classes, self.tf = items, classes, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), label


def discover(root, allow_augmented=False):
    """Every image under root, grouped by its folder name.

    Augmented copies are skipped by default: they are derived from the
    originals, so letting them in means a transformed version of a test image
    can sit in the training set.
    """
    by_class = defaultdict(list)
    skipped = 0
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES or path.parent == Path(root):
            continue
        if not allow_augmented and any(h in str(path).lower() for h in AUGMENT_HINTS):
            skipped += 1
            continue
        by_class[path.parent.name].append(path)
    if not by_class:
        raise SystemExit(f"no images found under {root}")
    if skipped:
        print(f"skipped {skipped} augmented images (pass --allow-augmented to "
              "include them, but then the test set is no longer clean)")
    return dict(sorted(by_class.items()))


def split(by_class, classes, mode, seed, val_frac=0.2, test_frac=0.2):
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for name, paths in by_class.items():
        label = classes.index(name)
        paths = sorted(paths)                 # filename order == capture order
        if mode == "random":
            paths = list(rng.permutation(paths))
        n = len(paths)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        # grouped: the held-out blocks are the END of the sequence, kept contiguous
        test += [(p, label) for p in paths[n - n_test:]]
        val += [(p, label) for p in paths[n - n_test - n_val:n - n_test]]
        train += [(p, label) for p in paths[:n - n_test - n_val]]
    return train, val, test


def run_epoch(model, loader, criterion, optimiser=None, device="cpu"):
    training = optimiser is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    preds_all, labels_all = [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            preds = logits.argmax(1)
            total_loss += float(loss) * len(labels)
            correct += int((preds == labels).sum())
            seen += len(labels)
            preds_all += preds.cpu().tolist()
            labels_all += labels.cpu().tolist()
    return total_loss / max(seen, 1), correct / max(seen, 1), preds_all, labels_all


def damp_scores(preds, labels, classes):
    """Collapse to the binary question the system actually asks of a photo."""
    damp = {i for i, c in enumerate(classes) if normalise(c) in DAMP_CLASSES}
    p = np.array([x in damp for x in preds])
    y = np.array([x in damp for x in labels])
    tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "support": int(y.sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/bd3_raw")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=6,
                        help="image-loading processes; 0 loads in-line and is much slower")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["grouped", "random"], default="grouped")
    parser.add_argument("--allow-augmented", action="store_true",
                        help="include BD3's augmented copies (contaminates the test set)")
    parser.add_argument("--out", default="models/defect_resnet18.pt")
    parser.add_argument("--report", default="models/defect_report.json")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the per-epoch checkpoint beside --out")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    by_class = discover(args.data, allow_augmented=args.allow_augmented)
    classes = list(by_class)
    train_items, val_items, test_items = split(
        by_class, classes, args.split, args.seed)

    print(f"data   : {args.data}")
    print(f"classes: {len(classes)}  {classes}")
    print(f"images : {sum(len(v) for v in by_class.values())} "
          f"(train {len(train_items)}, val {len(val_items)}, test {len(test_items)})")
    print(f"split  : {args.split}   device: {device}\n")

    # Decoding 512x512 JPEGs is not free. With num_workers=0 it happens in the
    # main process between batches, so the CPU sits idle waiting on images
    # instead of training -- measured at roughly half throughput on this set.
    common = dict(num_workers=args.workers,
                  persistent_workers=args.workers > 0,
                  prefetch_factor=4 if args.workers > 0 else None)
    loaders = {
        "train": DataLoader(ImageList(train_items, classes, train_transforms(args.size)),
                            batch_size=args.batch, shuffle=True, **common),
        "val": DataLoader(ImageList(val_items, classes, eval_transforms(args.size)),
                          batch_size=args.batch, **common),
        "test": DataLoader(ImageList(test_items, classes, eval_transforms(args.size)),
                           batch_size=args.batch, **common),
    }

    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Checkpoint every epoch so an interrupted run resumes instead of restarting.
    # CPU training here is ~20 minutes; losing it to a reboot is avoidable.
    ckpt_path = Path(str(args.out) + ".ckpt")
    best_val, best_state, start_epoch = -1.0, None, 1
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimiser.load_state_dict(ckpt["optimiser"])
        best_val, best_state = ckpt["best_val"], ckpt["best_state"]
        start_epoch = ckpt["epoch"] + 1
        print(f"resumed from {ckpt_path} at epoch {start_epoch} "
              f"(best val so far {best_val:.2f})\n")

    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, tr_acc, *_ = run_epoch(model, loaders["train"], criterion,
                                        optimiser, device)
        va_loss, va_acc, *_ = run_epoch(model, loaders["val"], criterion,
                                        None, device)
        flag = ""
        if va_acc > best_val:
            best_val, flag = va_acc, "  *"
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch:>2}  train loss {tr_loss:.3f} acc {tr_acc:.2f}   "
              f"val loss {va_loss:.3f} acc {va_acc:.2f}{flag}")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(),
                    "optimiser": optimiser.state_dict(), "epoch": epoch,
                    "best_val": best_val, "best_state": best_state,
                    "classes": classes, "size": args.size}, ckpt_path)

    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_acc, preds, labels = run_epoch(model, loaders["test"], criterion,
                                           None, device)
    damp = damp_scores(preds, labels, classes)

    print(f"\ntest accuracy ({len(test_items)} images): {test_acc:.3f}")
    print(f"damp question (algae or stain vs rest): "
          f"precision {damp['precision']:.2f}  recall {damp['recall']:.2f}  "
          f"f1 {damp['f1']:.2f}  (support {damp['support']})")

    try:
        from sklearn.metrics import classification_report
        present = sorted(set(labels) | set(preds))
        print("\n" + classification_report(
            labels, preds, labels=present,
            target_names=[classes[i] for i in present], zero_division=0))
    except Exception as exc:
        print(f"(per-class report unavailable: {exc})")

    smoke = len(test_items) < MIN_MEANINGFUL_TEST
    if smoke:
        print("=" * 70)
        print(f"SMOKE TEST ONLY -- {len(test_items)} test images is far too few "
              "for these numbers to mean anything.")
        print("The pipeline is verified to run on real photographs. It has not")
        print("been shown to work. Fetch the full BD3 set (3,965 images) and")
        print("rerun before quoting any figure from this script.")
        print("=" * 70)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes,
                "size": args.size}, args.out)
    report = {
        "data": str(args.data), "split": args.split, "classes": classes,
        "counts": {"train": len(train_items), "val": len(val_items),
                   "test": len(test_items)},
        "test_accuracy": test_acc, "best_val_accuracy": best_val,
        "damp_binary": damp,
        "smoke_test": smoke,
        "real_photographs": True,
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"\nsaved model to {args.out} and report to {args.report}")


if __name__ == "__main__":
    main()
