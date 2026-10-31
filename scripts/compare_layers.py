"""Head-to-head: the rule engine (layer 1) against the sequence model (layer 3).

The question layer 3 has to answer is not "is it accurate" -- both layers score
well once a fault is old. It is "does it know sooner", because the whole product
claim is finding a problem before anyone can see it.

So this truncates each simulated run at a fixed number of hours after the fault
begins and asks both layers to name it, on seeds the sequence model never
trained on.

Both figures come from the simulator, so neither absolute number is evidence
about real flats. The *comparison* is more robust than either side of it: both
layers face identical ground truth. It is still not a clean contest -- the CNN
trained on this simulator, while the rules were hand-written against the same
physics -- so read the mechanism, not just the gap.

Usage:
    python scripts/compare_layers.py --model models/sequence_cnn.pt
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from smart_property.config import EXTERNAL_WALL, FAULT_KINDS, ROOMS   # noqa: E402
from smart_property.fusion import MIN_CONFIDENCE                      # noqa: E402
from smart_property.rules import classify_room                        # noqa: E402
from smart_property.sequence import SequenceClassifier                # noqa: E402
from smart_property.simulator import simulate                         # noqa: E402

RAIN_FED = [r for r in ROOMS if EXTERNAL_WALL[r]]


def room_for(kind, seed):
    pool = RAIN_FED if kind == "roof_seepage" else ROOMS
    return pool[seed % len(pool)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/sequence_cnn.pt")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(32, 40)),
                        help="seeds held out of sequence-model training")
    parser.add_argument("--hours", type=int, nargs="+",
                        default=[6, 12, 24, 48, 96])
    parser.add_argument("--days", type=int, default=25)
    parser.add_argument("--start-day", type=float, default=10.0)
    args = parser.parse_args()

    sequence = SequenceClassifier.load(args.model)
    print(f"held-out seeds: {args.seeds}\n")
    print(f"{'hours since onset':<20}{'rules (layer 1)':>17}"
          f"{'1D-CNN (layer 3)':>19}{'cases':>7}")

    for hours in args.hours:
        rules_ok = cnn_ok = total = 0
        for seed in args.seeds:
            for kind in FAULT_KINDS:
                room = room_for(kind, seed)
                df = simulate(seed=seed, days=args.days, fault_kind=kind,
                              room=room, start_day=args.start_day)
                cut = (df["ts"].min() + pd.Timedelta(days=args.start_day)
                       + pd.Timedelta(hours=hours))
                window = df[(df["room"] == room) & (df["ts"] <= cut)]

                try:
                    best = classify_room(window)["best"]
                    rules_ok += (best["kind"] == kind
                                 and best["confidence"] >= MIN_CONFIDENCE)
                except ValueError:
                    pass                       # not enough history yet

                pred = sequence.predict(window)
                cnn_ok += bool(pred and pred["most_likely_fault"] == kind
                               and pred["by_label"]["none"] < 0.5)
                total += 1

        print(f"{hours:<20}{rules_ok / total:>17.3f}"
              f"{cnn_ok / total:>19.3f}{total:>7}")

    print("\nBoth columns are simulated. The gap in the first 24 hours is the "
          "argument for layer 3;\nthe mechanism behind it is that the rule "
          "engine gates on the wall being measurably\nwetter, which takes days, "
          "while flow and pipe noise step up within hours.")


if __name__ == "__main__":
    main()
