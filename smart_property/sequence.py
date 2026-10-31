"""Layer 3: a sequence model over raw sensor windows.

The rule engine (layer 1) reads hand-written physics evidence and the Isolation
Forest (layer 2) scores how unusual an hour looks. Neither reads the *shape* of
the last few days. This does: a 1D-CNN over the last 24-72 hours of readings,
returning a probability per fault kind.

Normalisation matters more than the architecture here. Every channel is
z-scored against that room's own baseline week, so the model sees "how far from
this room's normal" rather than absolute degrees and litres. Without it the
model learns which room it is looking at instead of what is wrong with it.
"""

import numpy as np
import torch
import torch.nn as nn

from .config import FAULT_KINDS, SAMPLE_MINUTES

CHANNELS = ["wall_moisture", "wall_surface_rh", "air_temp", "air_rh",
            "wall_temp", "flow_lpm", "pipe_db", "rain_mm_h"]

LABELS = ["none"] + list(FAULT_KINDS)
WINDOW_HOURS = 48
STEPS_PER_HOUR = 60 // SAMPLE_MINUTES


def baseline_stats(room_df, baseline_days=7):
    """Mean and spread of each channel over the room's first quiet week."""
    frame = room_df.set_index("ts").sort_index()
    cutoff = frame.index.min() + np.timedelta64(baseline_days, "D")
    base = frame[frame.index < cutoff][CHANNELS]
    mean = base.mean().to_numpy(dtype=np.float32)
    std = base.std().to_numpy(dtype=np.float32)
    # rain can be flat-zero across a dry baseline; never divide by ~0
    std = np.where(std < 1e-3, 1.0, std).astype(np.float32)
    return mean, std


def extract_window(room_df, end_index, mean, std, window_hours=WINDOW_HOURS):
    """One normalised (channels x timesteps) window ending at end_index."""
    length = window_hours * STEPS_PER_HOUR
    start = end_index - length
    if start < 0:
        return None
    block = room_df[CHANNELS].to_numpy(dtype=np.float32)[start:end_index]
    if len(block) != length:
        return None
    return ((block - mean) / std).T.astype(np.float32)   # channels first


class SequenceCNN(nn.Module):
    """Small dilated 1D-CNN. Depth is for receptive field, not capacity.

    Kept deliberately small: the training data is simulated, and a larger model
    would simply memorise the simulator's generative rules faster.
    """

    def __init__(self, in_channels=len(CHANNELS), num_classes=len(LABELS),
                 width=32, dropout=0.3):
        super().__init__()

        def block(cin, cout, dilation):
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=5, padding=2 * dilation,
                          dilation=dilation),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, width, 1),
            block(width, width * 2, 2),
            block(width * 2, width * 2, 4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(width * 2, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


class SequenceClassifier:
    """A trained sequence model, loaded for inference on one room."""

    def __init__(self, model, labels, window_hours, baseline_days=7, device="cpu"):
        self.model = model.eval().to(device)
        self.labels = labels
        self.window_hours = window_hours
        self.baseline_days = baseline_days
        self.device = device

    @classmethod
    def load(cls, path, device="cpu"):
        blob = torch.load(path, map_location=device, weights_only=False)
        model = SequenceCNN(num_classes=len(blob["labels"]))
        model.load_state_dict(blob["state_dict"])
        return cls(model, blob["labels"], blob.get("window_hours", WINDOW_HOURS),
                   blob.get("baseline_days", 7), device)

    @torch.no_grad()
    def predict(self, room_df):
        """Fault probabilities from this room's most recent window."""
        frame = room_df.sort_values("ts").reset_index(drop=True)
        mean, std = baseline_stats(frame, self.baseline_days)
        window = extract_window(frame, len(frame), mean, std, self.window_hours)
        if window is None:
            return None                       # not enough history yet
        batch = torch.from_numpy(window).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(batch), dim=1)[0].cpu().numpy()
        by_label = {name: float(probs[i]) for i, name in enumerate(self.labels)}
        fault = max((k for k in by_label if k != "none"), key=by_label.get)
        return {
            "by_label": by_label,
            "fault_probability": float(1.0 - by_label.get("none", 0.0)),
            "most_likely_fault": fault,
            "most_likely_prob": by_label[fault],
        }
