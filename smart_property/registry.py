"""Which components are actually available on this machine.

Every model in this system is optional. The sensor side (components 1, 2, 3
and fusion) needs nothing but numpy, pandas and scikit-learn and always runs.
Everything else switches on if its weights are present and its library is
installed, and stays quietly off otherwise.

That is deliberate. A prototype that refuses to start because a 43 MB
checkpoint is missing is not demonstrable, and silently pretending a component
ran when it did not is worse. `describe()` reports exactly what is live, so the
dashboard and the CLI can say so rather than implying the whole architecture is
running.

Loading is lazy and cached: nothing is read from disk until something asks.
"""

import os
from functools import lru_cache
from pathlib import Path

SEQUENCE_WEIGHTS = Path("models/sequence_cnn.pt")
DEFECT_WEIGHTS = Path("models/bd3_grouped.pt")
CRACK_WEIGHTS = Path("models/yolov8n-crack-seg-best.pt")


@lru_cache(maxsize=1)
def sequence_model():
    """Component 4: 1D-CNN second opinion on the fault kind."""
    if not SEQUENCE_WEIGHTS.exists():
        return None
    try:
        from .sequence import SequenceClassifier
        return SequenceClassifier.load(SEQUENCE_WEIGHTS)
    except Exception:
        return None


@lru_cache(maxsize=1)
def defect_model():
    """Component 7: damp, mould and surface-defect classifier."""
    if not DEFECT_WEIGHTS.exists():
        return None
    try:
        from .vision import DefectClassifier
        return DefectClassifier.load(DEFECT_WEIGHTS)
    except Exception:
        return None


@lru_cache(maxsize=1)
def segmenter():
    """Component 5: crack segmentation. Needs ultralytics as well as weights."""
    if not CRACK_WEIGHTS.exists():
        return None
    try:
        from .segmentation import CrackSegmenter
        return CrackSegmenter(CRACK_WEIGHTS)
    except Exception:
        return None


def advisor(language="en"):
    """Component 9: Gemini rewriting, when a key is configured."""
    try:
        from .llm import build_advisor
        return build_advisor(language)
    except Exception:
        return None


def components(language="en"):
    """Keyword arguments for analyse_flat, with whatever is available."""
    return {
        "sequence_model": sequence_model(),
        "defect_model": defect_model(),
        "segmenter": segmenter(),
        "llm": advisor(language),
    }


def describe():
    """One row per component: what it is, whether it is live, and why not."""
    gemini_key = bool(os.environ.get("GEMINI_API_KEY"))
    rows = [
        ("1", "Anomaly detector", "Isolation Forest", True, ""),
        ("2", "Fault rules", "building physics", True, ""),
        ("3", "Damage forecaster", "trend (Chronos-2 opt-in)", True, ""),
        ("4", "Sequence model", "1D-CNN, 48 h",
         sequence_model() is not None,
         "" if SEQUENCE_WEIGHTS.exists() else f"missing {SEQUENCE_WEIGHTS}"),
        ("5", "Crack segmentation", "YOLO on Crack-seg",
         segmenter() is not None,
         "" if CRACK_WEIGHTS.exists() else f"missing {CRACK_WEIGHTS}"),
        ("6", "Crack width", "OpenCV + ArUco", True, "needs a marker in frame"),
        ("7", "Damp classifier", "ResNet18 on BD3",
         defect_model() is not None,
         "" if DEFECT_WEIGHTS.exists() else f"missing {DEFECT_WEIGHTS}"),
        ("8", "Acoustic classifier", "PANNs CNN14", False, "not built"),
        ("9", "Repair advisor", "Gemini, template fallback", True,
         "" if gemini_key else "GEMINI_API_KEY unset, using templates"),
        ("-", "Alert fusion", "logic", True, ""),
    ]
    return [{"id": i, "name": n, "how": h, "live": bool(l), "note": note}
            for i, n, h, l, note in rows]
