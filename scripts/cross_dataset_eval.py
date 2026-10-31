"""Test the BD3 classifier on a dataset it has never seen: house-wall-damage.

Held-out accuracy within BD3 only says the model generalises across photographs
of the same buildings, taken by the same person, with the same phone, in the
same week. That is the weakest kind of generalisation and it is the one almost
every paper reports.

This is the stronger test. house-wall-damage (Roboflow, CC BY 4.0, 3,449
images) was collected by someone else entirely, and shares four classes with
BD3: algae, peeling, spalling and crack. A model that holds up here is doing
something about walls rather than about one photographer's camera.

Expect a drop. The size of the drop is the finding -- it is the honest estimate
of what deployment on someone else's building would look like.

    python scripts/cross_dataset_eval.py --model models/bd3_grouped.pt
"""

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from smart_property.vision import DefectClassifier, normalise   # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# BD3 label -> the house-wall-damage label meaning the same thing.
# BD3 splits cracks by severity and house-wall-damage does not, so both
# collapse to one crack class. "plain" and "stain" have no counterpart.
TO_SHARED = {
    "algae": "algae",
    "peeling": "peeling",
    "spalling": "spalling",
    "major_crack": "wall_crack",
    "minor_crack": "wall_crack",
    "major crack": "wall_crack",
    "minor crack": "wall_crack",
}
# house-wall-damage classes BD3 cannot express, so they are not scored.
UNSCORABLE = {"electrical_fault"}


def collect(root):
    for folder in sorted(Path(root).iterdir()):
        if not folder.is_dir() or normalise(folder.name) in UNSCORABLE:
            continue
        for image in sorted(folder.iterdir()):
            if image.suffix.lower() in IMAGE_SUFFIXES:
                yield image, normalise(folder.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/bd3_grouped.pt")
    parser.add_argument("--data", default="data/hwd_raw/test")
    parser.add_argument("--report", default="models/cross_dataset_report.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    clf = DefectClassifier.load(args.model)
    mapped = {c: TO_SHARED.get(normalise(c)) for c in clf.classes}
    scorable = sorted({v for v in mapped.values() if v})
    print(f"model classes : {clf.classes}")
    print(f"shared classes: {scorable}")
    print(f"unscored      : BD3 'plain'/'stain' have no counterpart; "
          f"house-wall-damage {sorted(UNSCORABLE)} cannot be predicted\n")

    items = list(collect(args.data))
    if args.limit:
        items = items[:args.limit]
    if not items:
        raise SystemExit(f"no images under {args.data}")

    correct = 0
    abstained = 0
    per_class = Counter()
    per_class_right = Counter()
    confusion = Counter()

    for path, truth in items:
        prediction = clf.predict(path)
        shared = mapped.get(prediction["top_class"])
        per_class[truth] += 1
        if shared is None:
            # the model's best guess was 'plain' or 'stain': it has no way to
            # say this image's true class, so count it, do not hide it
            abstained += 1
            confusion[(truth, f"<{prediction['top_class']}>")] += 1
            continue
        confusion[(truth, shared)] += 1
        if shared == truth:
            correct += 1
            per_class_right[truth] += 1

    total = len(items)
    print(f"{'class':<16}{'n':>6}{'correct':>9}{'recall':>9}")
    for name in sorted(per_class):
        n = per_class[name]
        print(f"{name:<16}{n:>6}{per_class_right[name]:>9}"
              f"{per_class_right[name] / n:>9.3f}")
    print(f"{'overall':<16}{total:>6}{correct:>9}{correct / total:>9.3f}")
    print(f"\nunmappable predictions ('plain'/'stain'): {abstained} "
          f"({abstained / total:.1%}) -- counted as wrong above")

    print("\nmost common confusions:")
    for (truth, got), n in confusion.most_common(8):
        if truth != got:
            print(f"  {truth:<14} predicted as {got:<16} {n}")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({
        "model": args.model, "data": args.data, "images": total,
        "accuracy": correct / total, "abstained": abstained,
        "per_class": {k: {"n": per_class[k], "correct": per_class_right[k]}
                      for k in per_class},
        "note": "cross-dataset: trained on BD3, tested on house-wall-damage",
    }, indent=2))
    print(f"\nsaved {args.report}")
    print("\nThis is the only number in the project measured on photographs "
          "from\na source the model was not trained on.")


if __name__ == "__main__":
    main()
