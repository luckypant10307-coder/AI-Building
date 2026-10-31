"""Train layer 3, the sequence model, on simulated sensor windows.

READ THIS BEFORE QUOTING ANY NUMBER THIS SCRIPT PRINTS.

The model is trained and tested on the project's own simulator. The simulator
generates faults from hand-written rules; a convolutional net has more than
enough capacity to learn those rules exactly. A high score here means the model
reproduces the simulator, not that it detects leaks in a real flat. It is a
coherence check on the architecture, not evidence.

What is still worth measuring, and what this script reports:

  * split by SEED, never by window -- windows from one simulated flat overlap
    heavily and are not independent
  * detection latency: accuracy bucketed by hours since the fault began, which
    says how early the shape of the data gives the fault away
  * (the head-to-head against layer 1, the rule engine, lives in
    scripts/compare_layers.py and runs on these same held-out seeds)

Usage:
    python scripts/train_sequence_model.py --seeds 40 --epochs 15
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from smart_property.config import (                       # noqa: E402
    EXTERNAL_WALL, FAULT_KINDS, ROOMS, SAMPLES_PER_DAY)
from smart_property.sequence import (                     # noqa: E402
    CHANNELS, LABELS, STEPS_PER_HOUR, SequenceCNN, baseline_stats,
    extract_window)
from smart_property.simulator import simulate             # noqa: E402

RAIN_FED = [r for r in ROOMS if EXTERNAL_WALL[r]]


def room_for(kind, seed):
    pool = RAIN_FED if kind == "roof_seepage" else ROOMS
    return pool[seed % len(pool)]


def build_split(seeds, days, window_hours, stride_hours, baseline_days,
                start_day):
    """Windows for a set of seeds. Returns X, y, and hours-since-onset."""
    windows, labels, onsets = [], [], []
    length = window_hours * STEPS_PER_HOUR
    stride = stride_hours * STEPS_PER_HOUR
    first_end = max(baseline_days * SAMPLES_PER_DAY, length)

    for seed in seeds:
        for kind in [None] + list(FAULT_KINDS):
            fault_room = room_for(kind, seed) if kind else None
            df = simulate(seed=seed, days=days, fault_kind=kind,
                          room=fault_room, start_day=start_day)
            for room in ROOMS:
                room_df = df[df["room"] == room].sort_values("ts")
                room_df = room_df.reset_index(drop=True)
                mean, std = baseline_stats(room_df, baseline_days)
                for end in range(first_end, len(room_df) + 1, stride):
                    window = extract_window(room_df, end, mean, std, window_hours)
                    if window is None:
                        continue
                    end_day = end / SAMPLES_PER_DAY
                    active = kind is not None and room == fault_room \
                        and end_day > start_day
                    windows.append(window)
                    labels.append(LABELS.index(kind) if active else 0)
                    onsets.append((end_day - start_day) * 24.0 if active else -1.0)

    X = torch.from_numpy(np.stack(windows))
    y = torch.tensor(labels, dtype=torch.long)
    return X, y, np.array(onsets, dtype=np.float32)


def evaluate(model, X, y, onset, batch=256, device="cpu"):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            chunk = X[i:i + batch].to(device)
            preds.append(model(chunk).argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    truth = y.numpy()

    overall = float((preds == truth).mean())
    faulty = truth != 0
    healthy = ~faulty
    return {
        "accuracy": overall,
        "false_alarm_rate": float((preds[healthy] != 0).mean()) if healthy.any() else 0.0,
        "fault_recall": float((preds[faulty] == truth[faulty]).mean()) if faulty.any() else 0.0,
        "detected_any_fault": float((preds[faulty] != 0).mean()) if faulty.any() else 0.0,
        "n_healthy": int(healthy.sum()),
        "n_faulty": int(faulty.sum()),
    }, preds, truth


def latency_table(preds, truth, onset):
    """Does the right answer arrive early or only once damage is obvious?"""
    rows = []
    for low, high in [(0, 12), (12, 24), (24, 48), (48, 96), (96, 1e9)]:
        mask = (onset >= low) & (onset < high) & (truth != 0)
        if not mask.any():
            continue
        rows.append({
            "hours_since_onset": f"{low}-{high:g}" if high < 1e9 else f"{low}+",
            "windows": int(mask.sum()),
            "exact_kind": float((preds[mask] == truth[mask]).mean()),
            "any_fault": float((preds[mask] != 0).mean()),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--days", type=int, default=25)
    parser.add_argument("--start-day", type=float, default=10.0)
    parser.add_argument("--window-hours", type=int, default=48)
    parser.add_argument("--stride-hours", type=int, default=6)
    parser.add_argument("--baseline-days", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="models/sequence_cnn.pt")
    parser.add_argument("--report", default="models/sequence_report.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Split by seed. Windows from one simulated flat overlap by design, so a
    # window-level split would put near-copies in train and test at once.
    all_seeds = list(range(args.seeds))
    n_test = max(2, args.seeds // 5)
    test_seeds = all_seeds[-n_test:]
    val_seeds = all_seeds[-2 * n_test:-n_test]
    train_seeds = all_seeds[:-2 * n_test]
    print(f"seeds  : train {len(train_seeds)}, val {len(val_seeds)}, "
          f"test {len(test_seeds)}  (split by seed, never by window)")

    common = dict(days=args.days, window_hours=args.window_hours,
                  stride_hours=args.stride_hours,
                  baseline_days=args.baseline_days, start_day=args.start_day)
    print("building windows ...")
    Xtr, ytr, _ = build_split(train_seeds, **common)
    Xva, yva, ova = build_split(val_seeds, **common)
    Xte, yte, ote = build_split(test_seeds, **common)
    print(f"windows: train {len(Xtr)}, val {len(Xva)}, test {len(Xte)}  "
          f"({len(CHANNELS)} channels x {Xtr.shape[-1]} steps "
          f"= {args.window_hours} h)\n")

    counts = np.bincount(ytr.numpy(), minlength=len(LABELS))
    print("train label counts: " + ", ".join(
        f"{LABELS[i]} {counts[i]}" for i in range(len(LABELS))) + "\n")

    # healthy windows massively outnumber faulty ones; weight the loss
    weights = torch.tensor(
        [len(ytr) / (len(LABELS) * max(c, 1)) for c in counts], dtype=torch.float32)

    model = SequenceCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch,
                        shuffle=True)

    best_val, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            total += float(loss) * len(yb)
        va, _, _ = evaluate(model, Xva, yva, ova, device=device)
        score = va["fault_recall"] - va["false_alarm_rate"]
        flag = ""
        if score > best_val:
            best_val, flag = score, "  *"
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch:>2}  loss {total / len(ytr):.4f}   "
              f"val recall {va['fault_recall']:.3f}  "
              f"false alarms {va['false_alarm_rate']:.3f}{flag}")

    if best_state:
        model.load_state_dict(best_state)

    test, preds, truth = evaluate(model, Xte, yte, ote, device=device)
    print(f"\nheld-out seeds: {test_seeds}")
    print(f"  windows            : {test['n_healthy']} healthy, "
          f"{test['n_faulty']} faulty")
    print(f"  exact fault kind   : {test['fault_recall']:.3f}")
    print(f"  any fault detected : {test['detected_any_fault']:.3f}")
    print(f"  false alarm rate   : {test['false_alarm_rate']:.3f}")

    rows = latency_table(preds, truth, ote)
    print("\n  how early is it right?")
    print(f"  {'hours since onset':<20}{'windows':>9}{'exact kind':>12}{'any fault':>11}")
    for row in rows:
        print(f"  {row['hours_since_onset']:<20}{row['windows']:>9}"
              f"{row['exact_kind']:>12.3f}{row['any_fault']:>11.3f}")

    print("\n" + "=" * 72)
    print("These numbers come from the simulator, which generates faults from")
    print("hand-written rules this model has enough capacity to memorise.")
    print("They show the architecture is coherent. They are not evidence that")
    print("it detects leaks in a real flat.")
    print("=" * 72)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "labels": LABELS,
                "window_hours": args.window_hours,
                "baseline_days": args.baseline_days}, args.out)
    Path(args.report).write_text(json.dumps({
        "test": test, "latency": rows, "test_seeds": test_seeds,
        "window_hours": args.window_hours, "channels": CHANNELS,
        "trained_on": "simulator", "real_data": False,
    }, indent=2))
    print(f"\nsaved model to {args.out} and report to {args.report}")


if __name__ == "__main__":
    main()
