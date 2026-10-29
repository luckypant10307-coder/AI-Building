"""Export labelled feature windows for the learned fault classifier (component 4).

One row per room per 24-hour window, with the evidence measures as features and
the injected fault as the label. Train on one set of seeds, test on seeds the
model has never seen -- the split is by seed, not by row, because rows from the
same simulated flat are not independent.

    python scripts/export_dataset.py --seeds 40 --out data/windows.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from smart_property.config import EXTERNAL_WALL, FAULT_KINDS, ROOMS   # noqa: E402
from smart_property.rules import room_evidence                        # noqa: E402
from smart_property.simulator import simulate                         # noqa: E402

RAIN_FED = [r for r in ROOMS if EXTERNAL_WALL[r]]


def windows_for_run(df, fault_kind, fault_room, seed, days, baseline_days=7):
    """Evidence measures at each day's end, once the baseline week is past."""
    rows = []
    start = df["ts"].min()
    for room in ROOMS:
        room_df = df[df["room"] == room]
        for day in range(baseline_days + 2, days):
            cut = start + pd.Timedelta(days=day)
            window = room_df[room_df["ts"] <= cut]
            try:
                measures = room_evidence(window, baseline_days=baseline_days)
            except ValueError:
                continue
            rows.append({
                "seed": seed, "room": room, "day": day,
                **measures,
                "label": fault_kind if room == fault_room else "none",
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--days", type=int, default=25)
    parser.add_argument("--out", default="data/windows.csv")
    args = parser.parse_args()

    rows = []
    cases = [(None, None)] + [(k, None) for k in FAULT_KINDS]
    for seed in range(args.seeds):
        for kind, _ in cases:
            pool = RAIN_FED if kind == "roof_seepage" else ROOMS
            room = pool[seed % len(pool)] if kind else None
            df = simulate(seed=seed, days=args.days, fault_kind=kind,
                          room=room, start_day=10.0)
            rows += windows_for_run(df, kind, room, seed, args.days)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)
    print(f"wrote {len(frame)} windows to {out}")
    print(frame["label"].value_counts().to_string())
    print("\nSplit by seed, never by row: rows from one flat are not independent.")


if __name__ == "__main__":
    main()
