"""
Improved placement analysis for Stage 7 overlays.

Two upgrades over the original analyze_frame_for_overlay():

1. TRACKS THE FACE ACROSS THE EVENT'S FULL DURATION, not just its start
   frame. The original sampled one frame at event["start"] and placed
   the overlay based on that single moment — if the speaker turns or
   drifts during the 2-4s the overlay is on screen, the box can end up
   overlapping the face by the end even though it was clear at the
   start. This version samples several frames across [start, end],
   detects the face in each with MediaPipe (more robust than Haar,
   especially off-angle), and uses the UNION of all detected boxes as
   the "keep clear of this" region — so placement is safe for the
   whole time the overlay is visible, not just its first frame.

2. PICKS THE VISUALLY QUIETEST SAFE ZONE, not just the first one that
   fits. The original tried "above" then fell back to "side_left" or
   "side_right" in a fixed order regardless of what's actually in
   those regions. This version scores every safe candidate region by
   local visual busyness (Laplacian edge-density — high-detail
   backgrounds like bookshelves/text/patterns score worse than plain
   walls or soft bokeh) and picks the lowest-busyness option, so the
   overlay tends to land somewhere it won't get visually lost or
   fight with background detail.
"""
import cv2
import numpy as np

# NOTE: this environment's mediapipe build (0.10.33) only ships the new
# Tasks API, which needs a .tflite model downloaded from a Google
# storage host this sandbox can't reach. Falls back to Haar cascade
# (same detector the original stage7 used) so this is testable here.
# On your RunPod/Colab box, with real internet access, swap this for
# either MediaPipe's Tasks API face detector (better off-angle/small-
# face recall) or `pip install "mediapipe<0.10.9"` for the older
# solutions.face_detection API — the rest of this module (tracking
# across the event window, busyness scoring) doesn't care which
# detector fills in detect_face_bbox().
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_face_bbox(frame_rgb: np.ndarray):
    """Returns (x, y, w, h) in pixels for the largest detected face, or None."""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return (int(fx), int(fy), int(fw), int(fh))


def track_face_union(get_frame_fn, start: float, end: float, num_samples: int = 5):
    """Samples num_samples evenly-spaced frames across [start, end],
    detects the face in each, and returns the UNION bounding box
    (pixel coords) covering every detection — i.e. the full region the
    face occupies at any point during the overlay's on-screen time.
    Returns None if no face was found in any sample."""
    if num_samples < 2:
        times = [start]
    else:
        times = [start + i * (end - start) / (num_samples - 1) for i in range(num_samples)]

    boxes = []
    last_frame = None
    for t in times:
        frame = get_frame_fn(min(t, end - 0.001) if end > start else start)
        last_frame = frame
        bbox = detect_face_bbox(frame)
        if bbox:
            boxes.append(bbox)

    if not boxes:
        return None, last_frame

    xs0 = [b[0] for b in boxes]
    ys0 = [b[1] for b in boxes]
    xs1 = [b[0] + b[2] for b in boxes]
    ys1 = [b[1] + b[3] for b in boxes]
    union = (min(xs0), min(ys0), max(xs1) - min(xs0), max(ys1) - min(ys0))
    return union, last_frame


def region_busyness(frame_rgb: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    """Scores how visually 'busy' a region is via Laplacian variance
    (a standard edge-density / focus measure) — higher = more detail/
    texture/edges in that area, which will visually compete with an
    overlay and its text. Lower is a better placement candidate."""
    h, w = frame_rgb.shape[:2]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return float("inf")
    region = cv2.cvtColor(frame_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(region, cv2.CV_64F).var())


def analyze_region_for_overlay(get_frame_fn, start: float, end: float,
                                box_width_px: int, box_height_px: int,
                                num_track_samples: int = 5) -> dict:
    """
    Video-aware replacement for analyze_frame_for_overlay(). Returns
    the same shape of dict, but placement decisions are informed by
    the face's movement across the whole event window and by which
    candidate region is visually quietest, not just the first that fits.
    """
    face_union, ref_frame = track_face_union(get_frame_fn, start, end, num_track_samples)
    h, w = ref_frame.shape[:2]

    HEAD_CLEARANCE_FRAC = 0.14
    SIDE_MARGIN_PX = 20
    SAFE_MARGIN_PX = 16  # keep overlays off the extreme frame edges

    candidates = []  # (placement, x_frac, y_frac, x0, y0, x1, y1)

    if face_union is None:
        candidates.append(("default", 0.5, 0.06,
                            w // 2 - box_width_px // 2, int(0.06 * h),
                            w // 2 + box_width_px // 2, int(0.06 * h) + box_height_px))
    else:
        fx, fy, fw, fh = face_union
        face_top_frac = fy / h
        box_margin_frac_v = (box_height_px + 30) / h
        space_above = face_top_frac
        if space_above >= HEAD_CLEARANCE_FRAC + box_margin_frac_v:
            y_frac = max(0.03, face_top_frac - HEAD_CLEARANCE_FRAC - box_margin_frac_v)
            x0 = w // 2 - box_width_px // 2
            candidates.append(("above", 0.5, y_frac, x0, int(y_frac * h),
                                x0 + box_width_px, int(y_frac * h) + box_height_px))

        space_left_px = fx
        space_right_px = w - (fx + fw)
        y_frac_side = max(0.05, min(0.85, (fy + fh / 2) / h - (box_height_px / 2) / h))
        if space_left_px >= box_width_px + SIDE_MARGIN_PX:
            x0 = SAFE_MARGIN_PX
            candidates.append(("side_left", (space_left_px / 2) / w, y_frac_side,
                                x0, int(y_frac_side * h), x0 + box_width_px, int(y_frac_side * h) + box_height_px))
        if space_right_px >= box_width_px + SIDE_MARGIN_PX:
            x0 = w - SAFE_MARGIN_PX - box_width_px
            candidates.append(("side_right", 1.0 - (space_right_px / 2) / w, y_frac_side,
                                x0, int(y_frac_side * h), x0 + box_width_px, int(y_frac_side * h) + box_height_px))

    if not candidates:
        return {"y_frac": 0.03, "x_frac": 0.5, "placement": "none",
                "is_light_bg": False, "face_found": face_union is not None, "safe_to_place": False}

    # Score every safe candidate by busyness on the reference frame; pick the quietest.
    scored = [(c, region_busyness(ref_frame, c[3], c[4], c[5], c[6])) for c in candidates]
    scored.sort(key=lambda pair: pair[1])
    best, best_score = scored[0]
    placement, x_frac, y_frac, x0, y0, x1, y1 = best

    sample = ref_frame[max(0, y0):max(0, y0) + 100, max(0, x0):max(0, x0) + 100]
    if sample.size == 0:
        is_light_bg = False
    else:
        luminance = (0.299 * sample[..., 0] + 0.587 * sample[..., 1] + 0.114 * sample[..., 2]).mean()
        is_light_bg = luminance > 150

    return {"y_frac": y_frac, "x_frac": x_frac, "placement": placement,
            "is_light_bg": is_light_bg, "face_found": face_union is not None,
            "safe_to_place": True, "busyness_score": round(best_score, 1),
            "candidates_considered": len(candidates)}
