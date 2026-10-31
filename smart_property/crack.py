"""Crack width and growth (component 6). OpenCV only, no learned model.

Takes a crack mask and a photograph containing a printed ArUco marker of known
size, and reports the crack's width in millimetres. Two photographs of the same
wall taken on different visits can then be compared, because the marker fixes
the scale and the viewing angle in both.

The marker is what makes this measurement rather than guesswork. Without it a
crack is only ever so many pixels wide, which depends on how close the phone
was held.

Component 5 (segmentation) is not built yet, so `segment_classical` is a
stand-in that finds dark linear structures. It is a ruler, not a detector: give
it a photo with no crack and it will still return something. Replace it with
the YOLO mask as soon as component 5 exists.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize

DEFAULT_DICT = cv2.aruco.DICT_4X4_50


@dataclass
class Scale:
    """Millimetres per pixel, derived from a marker of known physical size."""
    mm_per_px: float
    marker_px: float
    corners: np.ndarray

    @property
    def px_per_mm(self):
        return 1.0 / self.mm_per_px


def find_scale(image, marker_mm=50.0, dictionary=DEFAULT_DICT):
    """Locate an ArUco marker and convert pixels to millimetres."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(dictionary),
        cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(grey)
    if ids is None or len(corners) == 0:
        return None
    quad = corners[0].reshape(4, 2)
    # mean of the four edges is steadier than any single one under perspective
    sides = [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
    marker_px = float(np.mean(sides))
    return Scale(marker_mm / marker_px, marker_px, quad)


def rectification(image, scale, px_per_mm=4.0, marker_mm=50.0):
    """The warp that makes the marker square and the scale fixed.

    Returns the transform and output size rather than a warped image, so the
    photograph and any mask drawn on it go through exactly the same mapping.
    Two visits photographed from slightly different angles become comparable,
    which is what makes growth meaningful rather than an artefact of where the
    photographer stood.
    """
    side = marker_mm * px_per_mm
    target = np.array([[0, 0], [side, 0], [side, side], [0, side]],
                      dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(scale.corners.astype(np.float32), target)
    height, width = image.shape[:2]
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]],
                       dtype=np.float32).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    offset = np.array([[1, 0, -moved[:, 0].min()],
                       [0, 1, -moved[:, 1].min()], [0, 0, 1]])
    size = (int(np.ceil(np.ptp(moved[:, 0]))), int(np.ceil(np.ptp(moved[:, 1]))))
    return offset @ matrix, size, 1.0 / px_per_mm


def rectify(image, scale, px_per_mm=4.0, marker_mm=50.0):
    matrix, size, mm_per_px = rectification(image, scale, px_per_mm, marker_mm)
    return cv2.warpPerspective(image, matrix, size), mm_per_px


def segment_classical(image, marker_quad=None, block=None, c=8, min_area=40,
                      px_per_mm=None, max_crack_mm=8.0):
    """Stand-in crack mask: dark, thin, elongated structures.

    Not a detector. It assumes a crack is present and finds the most
    crack-shaped thing in the frame.

    The threshold neighbourhood must be wider than the crack. An adaptive
    threshold compares each pixel to its own surroundings, so if the block is
    narrower than the dark band, the middle of the band looks like its own
    background and only the two edges survive -- which silently reports a 5 mm
    crack as 0.4 mm. The block is therefore sized from the image scale to span
    `max_crack_mm`, not left at a fixed pixel count.
    """
    if block is None:
        block = int(round(max_crack_mm * px_per_mm)) if px_per_mm else 31
        block = max(block | 1, 3)          # adaptiveThreshold needs odd >= 3
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    grey = cv2.bilateralFilter(grey, 7, 50, 50)
    mask = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block, c)
    if marker_quad is not None:
        # the marker is black-and-white and would dominate every measurement
        cv2.fillConvexPoly(mask, marker_quad.astype(np.int32), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if area < min_area:
            continue
        # elongated: a crack is much longer than it is wide
        if max(w, h) < 3 * max(min(w, h), 1):
            continue
        keep[labels == i] = 255
    return keep


def crack_width(mask, mm_per_px):
    """Width along the crack, in millimetres.

    The distance transform gives each pixel's distance to the nearest
    background pixel; on the crack's skeleton that distance is half the local
    width, so twice it is the width at that point.
    """
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return None
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    skeleton = skeletonize(binary.astype(bool))
    widths_px = 2.0 * distance[skeleton]
    if widths_px.size == 0:
        return None
    widths_mm = widths_px * mm_per_px
    return {
        "max_mm": float(widths_mm.max()),
        "median_mm": float(np.median(widths_mm)),
        "p90_mm": float(np.percentile(widths_mm, 90)),
        "length_mm": float(skeleton.sum() * mm_per_px),
        "points": int(widths_mm.size),
    }


def measure(image, mask=None, marker_mm=50.0, px_per_mm=24.0):
    """Full path: find the marker, rectify, segment if needed, measure.

    `px_per_mm` sets the rectified resolution and therefore the smallest width
    that can be resolved: at 4 px/mm one pixel is 0.25 mm and the distance
    transform carries a fixed overestimate of roughly +0.4 mm. 16 px/mm puts
    the quantisation floor near 0.06 mm, well under a hairline crack.
    """
    scale = find_scale(image, marker_mm)
    if scale is None:
        return {"error": "no ArUco marker found; cannot convert pixels to mm"}
    matrix, size, mm_per_px = rectification(image, scale, px_per_mm, marker_mm)
    flat = cv2.warpPerspective(image, matrix, size)

    if mask is None:
        flat_scale = find_scale(flat, marker_mm)
        quad = flat_scale.corners if flat_scale else None
        flat_mask = segment_classical(flat, quad, px_per_mm=px_per_mm)
        source = "classical stand-in (component 5 not built)"
    else:
        # the mask was drawn on the original frame; it has to take the same warp
        flat_mask = cv2.warpPerspective((mask > 0).astype(np.uint8) * 255,
                                        matrix, size, flags=cv2.INTER_NEAREST)
        source = "supplied mask"
    widths = crack_width(flat_mask, mm_per_px)
    if widths is None:
        return {"error": "no crack pixels found", "mask_source": source}
    trusted = mask is not None
    out = {**widths, "mm_per_px": mm_per_px, "mask_source": source,
           "marker_px": scale.marker_px, "width_trustworthy": trusted}
    if not trusted:
        # measured against known line widths: the chain itself is good to
        # +0.08 mm, but the stand-in segmentation is off by as much as 3 mm
        out["warning"] = ("width came from the classical stand-in, measured at "
                          "up to 3 mm error on known line widths; treat as "
                          "presence-only until component 5 supplies the mask")
    return out


def growth(before, after):
    """Widening between two visits, in millimetres."""
    if "max_mm" not in before or "max_mm" not in after:
        return None
    delta = after["max_mm"] - before["max_mm"]
    return {
        "before_max_mm": before["max_mm"],
        "after_max_mm": after["max_mm"],
        "growth_mm": delta,
        "growing": delta > 0.2,      # below this is measurement noise, see validation
    }
