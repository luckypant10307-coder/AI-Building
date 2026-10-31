"""Validate component 6 against lines of known width.

The plan says: photograph a printed card with lines of known width and report
the measurement error in mm. This does that in software -- it renders the card,
warps it as a phone held at an angle would see it, adds blur and sensor noise,
and measures it back.

Ground truth here is genuinely known: the line was drawn N millimetres wide.
Nothing about physics is being simulated, so unlike the sensor benchmarks this
is not circular. What it does NOT test is finding a crack on a real wall --
that is component 5. This tests the measuring chain: marker detection,
perspective rectification, skeleton and distance transform.

Two modes separate those concerns:
    --mask truth      give the measurer the exact drawn line  (measurement error)
    --mask classical  let segment_classical find it           (+ segmentation error)

Usage:
    python scripts/validate_crack_width.py --trials 5
"""

import argparse
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from smart_property.crack import measure          # noqa: E402

MARKER_MM = 50.0
WIDTHS_MM = [0.2, 0.5, 1.0, 2.0, 5.0]


def render_card(width_mm, px_per_mm=12.0, marker_mm=MARKER_MM, rng=None,
                warp=True, blur=True, noise=True):
    """A white card carrying an ArUco marker and one dark line of known width."""
    rng = rng or np.random.default_rng(0)
    height, width = int(140 * px_per_mm), int(190 * px_per_mm)
    card = np.full((height, width), 238, np.uint8)

    side = int(marker_mm * px_per_mm)
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 7, side)
    mx, my = int(12 * px_per_mm), int(12 * px_per_mm)
    card[my:my + side, mx:mx + side] = marker

    # the line, drawn to an exact pixel width derived from its mm width
    line_px = max(int(round(width_mm * px_per_mm)), 1)
    y = int(100 * px_per_mm)
    x0, x1 = int(20 * px_per_mm), int(170 * px_per_mm)
    truth = np.zeros_like(card)
    cv2.line(truth, (x0, y), (x1, y), 255, line_px)
    card[truth > 0] = 38

    card = cv2.cvtColor(card, cv2.COLOR_GRAY2BGR)

    if warp:
        # a phone held a little off-square
        jitter = rng.uniform(-0.035, 0.035, size=(4, 2)) * [width, height]
        src = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
        matrix = cv2.getPerspectiveTransform(src, (src + jitter).astype(np.float32))
        card = cv2.warpPerspective(card, matrix, (width, height),
                                   borderValue=(238, 238, 238))
        truth = cv2.warpPerspective(truth, matrix, (width, height),
                                    flags=cv2.INTER_NEAREST)
    if blur:
        card = cv2.GaussianBlur(card, (3, 3), 0.8)
    if noise:
        card = np.clip(card.astype(np.int16)
                       + rng.normal(0, 3, card.shape).astype(np.int16),
                       0, 255).astype(np.uint8)
    return card, truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--mask", choices=["truth", "classical", "both"],
                        default="both")
    parser.add_argument("--px-per-mm", type=float, default=12.0)
    args = parser.parse_args()

    modes = ["truth", "classical"] if args.mask == "both" else [args.mask]
    print(f"Synthetic test card, {args.trials} trials per width, "
          f"rendered at {args.px_per_mm:g} px/mm, warped + blurred + noised.\n")

    for mode in modes:
        label = ("measurement chain only (exact mask given)" if mode == "truth"
                 else "whole chain (classical stand-in finds the line)")
        print(f"--- mask: {mode} -- {label}")
        print(f"{'true width':>11}{'measured':>11}{'error':>9}{'rel err':>9}"
              f"{'found':>7}")
        rows = []
        for width_mm in WIDTHS_MM:
            measured, found = [], 0
            for trial in range(args.trials):
                rng = np.random.default_rng(1000 * trial + int(width_mm * 10))
                card, truth = render_card(width_mm, args.px_per_mm, rng=rng)
                result = measure(card, truth if mode == "truth" else None,
                                 marker_mm=MARKER_MM)
                if "error" in result:
                    continue
                found += 1
                measured.append(result["median_mm"])
            if not measured:
                print(f"{width_mm:>10.1f}{'--':>11}{'--':>9}{'--':>9}"
                      f"{found:>4}/{args.trials}")
                continue
            mean = float(np.mean(measured))
            err = mean - width_mm
            rows.append(abs(err))
            print(f"{width_mm:>10.1f}{mean:>11.2f}{err:>+9.2f}"
                  f"{err / width_mm:>+8.0%}{found:>4}/{args.trials}")
        if rows:
            print(f"{'mean abs error':>11}{'':>11}{np.mean(rows):>+9.2f} mm\n")

    print("Ground truth is the drawn line width, so this is a real measurement\n"
          "error, not a simulated one. It does not show that cracks can be found\n"
          "on a real wall -- that is component 5.")


if __name__ == "__main__":
    main()
