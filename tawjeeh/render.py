"""Draw parking spots and counters on a frame."""

from __future__ import annotations

import cv2
import numpy as np

from .occupancy import FREE, OCCUPIED, UNKNOWN

# OpenCV colors in BGR order
COLORS = {
    FREE: (60, 200, 60),        # green = available
    OCCUPIED: (40, 40, 230),    # red = occupied
    UNKNOWN: (0, 200, 255),     # yellow/orange = uncertain
}


def draw_spots(frame, spots, states, H_ref_to_frame, show_ids=True):
    warped = []
    for spot in spots:
        pts = cv2.perspectiveTransform(spot.polygon.reshape(-1, 1, 2).astype(np.float32), H_ref_to_frame)
        warped.append(pts.reshape(-1, 2).round().astype(np.int32))
    # Light fill for available spots (one layer for all spots, for faster drawing)
    free_polys = [p for p, st in zip(warped, states) if st == FREE]
    if free_polys:
        overlay = frame.copy()
        cv2.fillPoly(overlay, free_polys, COLORS[FREE])
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)
    for spot, st, pts_i in zip(spots, states, warped):
        color = COLORS[int(st)]
        short = min(np.linalg.norm(pts_i[0] - pts_i[1]), np.linalg.norm(pts_i[1] - pts_i[2]))
        thick = 1 if short < 30 else 2
        cv2.polylines(frame, [pts_i], True, color, thick, cv2.LINE_AA)
        if show_ids:
            c = pts_i.mean(0).astype(int)
            scale = float(np.clip(short / 70, 0.28, 0.6))
            (tw, th), _ = cv2.getTextSize(spot.id, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            org = (int(c[0] - tw / 2), int(c[1] + th / 2))
            cv2.putText(frame, spot.id, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, spot.id, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_panel(frame, counts, title="Tawjeeh - Smart Parking", extra=None):
    h, w = frame.shape[:2]
    s = max(0.5, w / 1920 * 0.9)          # font size scales with video resolution
    lh = int(32 * s / 0.6)
    lines = [
        (title, (255, 255, 255)),
        (f"Available : {counts['free']}", COLORS[FREE]),
        (f"Occupied  : {counts['occupied']}", COLORS[OCCUPIED]),
    ]
    if counts.get("unknown"):
        lines.append((f"Uncertain : {counts['unknown']}", COLORS[UNKNOWN]))
    lines.append((f"Total spots: {counts['total']}", (230, 230, 230)))
    if extra:
        lines.append((extra, (180, 180, 180)))
    pw = int(330 * s / 0.6)
    ph = lh * len(lines) + 12
    x0, y0 = 10, 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, dst=frame)
    for i, (txt, col) in enumerate(lines):
        y = y0 + lh * (i + 1) - 4
        cv2.putText(frame, txt, (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, s * (0.85 if i == 0 else 1.0),
                    col, 2 if i else 1, cv2.LINE_AA)
    return frame
