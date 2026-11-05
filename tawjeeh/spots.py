"""
Parking spot definitions: saving/loading and automatic detection.

How automatic detection works:
  1) Detect cars in several frames spread across the video, and map their positions to the reference frame.
  2) A parked car appears in the same place in most frames, while a moving car's position changes.
     Cluster nearby detections and accept only places where a car appeared repeatedly.
  3) Arrange these places into rows (spots in a row are adjacent and share the same direction).
  4) A gap in a row of one spot or more between two cars may be empty spots or a driving lane.
     It is accepted as an "inferred" spot only if a parked car was seen there in at least two frames;
     otherwise it is saved as a "candidate" place that is not counted until the user accepts it manually.

Limitations of this method (important):
  - A spot that stays empty for the whole video and sits at the end of a row (not between two cars) will not be detected automatically.
  - If the model never detects a particular car, its spot appears only if it is inferred from a gap.
  - That is why the manual calibration tool (calibrate.py) exists, for adding/removing spots and saving them.
"""

from __future__ import annotations

import json
import string
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass(eq=False)   # compared by identity (the polygon is a numpy array)
class Spot:
    id: str
    polygon: np.ndarray          # (N,2) in reference-frame coordinates
    source: str = "detected"     # detected | inferred | manual

    def to_dict(self):
        return {"id": self.id, "source": self.source,
                "polygon": [[round(float(x), 1), round(float(y), 1)] for x, y in self.polygon]}

    @staticmethod
    def from_dict(d):
        return Spot(d["id"], np.array(d["polygon"], np.float32), d.get("source", "manual"))


@dataclass
class CameraConfig:
    video: str
    frame_size: tuple
    reference_frame_index: int = 0
    reference_image: str = ""
    source: str = "auto"
    stabilize: bool = True
    detector: dict = field(default_factory=dict)
    occupancy: dict = field(default_factory=dict)
    spots: list = field(default_factory=list)
    candidates: list = field(default_factory=list)   # possible unconfirmed spots (not counted)
    notes: list = field(default_factory=list)

    def save(self, path: Path):
        d = {k: v for k, v in self.__dict__.items() if k not in ("spots", "candidates")}
        d["frame_size"] = list(self.frame_size)
        d["spots"] = [s.to_dict() for s in self.spots]
        d["candidates"] = [s.to_dict() for s in self.candidates]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "CameraConfig":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        spots = [Spot.from_dict(s) for s in d.pop("spots", [])]
        cands = [Spot.from_dict(s) for s in d.pop("candidates", [])]
        d["frame_size"] = tuple(d["frame_size"])
        cfg = CameraConfig(**d)
        cfg.spots = spots
        cfg.candidates = cands
        return cfg


# ---------------------------------------------------------------------- #
# Geometry helpers for rotated rectangles
# ---------------------------------------------------------------------- #
def rect_params(poly: np.ndarray):
    """Returns (cx, cy, short, long, theta), where theta is the long axis direction in degrees [0,180)."""
    (cx, cy), (w, h), ang = cv2.minAreaRect(poly.astype(np.float32))
    if w >= h:
        long_, short, theta = w, h, ang
    else:
        long_, short, theta = h, w, ang + 90
    return cx, cy, short, long_, theta % 180


def rect_polygon(cx, cy, short, long_, theta):
    return cv2.boxPoints(((cx, cy), (long_, short), theta)).astype(np.float32)


def _angle_mean(thetas):
    """Mean of angles with a 180-degree period."""
    t = np.radians(np.asarray(thetas) * 2)
    return (np.degrees(np.arctan2(np.sin(t).mean(), np.cos(t).mean())) / 2) % 180


def _angle_diff(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


# ---------------------------------------------------------------------- #
# Automatic detection
# ---------------------------------------------------------------------- #
def discover_spots(det_per_frame: list[list[np.ndarray]], min_presence: float = 0.25,
                   fill_gaps: bool = True, max_gap_spots: int = 4, min_row_len: int = 3):
    """
    det_per_frame: for each sample frame, a list of car polygons (in reference-frame coordinates).
    Returns (spots, candidates, report)
    """
    n_frames = len(det_per_frame)
    all_rects = [rect_params(p) for frame in det_per_frame for p in frame]
    frame_ids = [i for i, frame in enumerate(det_per_frame) for _ in frame]
    if not all_rects:
        return [], [], {"error": "No cars were detected in the selected frames"}
    R = np.array(all_rects, np.float32)
    med_short = float(np.median(R[:, 2]))

    # 1) Cluster nearby detections (radius scales with each car's size, to support angled perspective)
    clusters = []  # each item: dict(members)
    centers = np.zeros((0, 2), np.float32)
    sizes = np.zeros(0, np.float32)
    for idx in np.argsort(-R[:, 3]):  # stable ordering
        p = R[idx, :2]
        if len(centers):
            d = np.linalg.norm(centers - p, axis=1)
            j = int(d.argmin())
            if d[j] < 0.45 * max(R[idx, 2], sizes[j]):
                c = clusters[j]
                c["members"].append(idx)
                centers[j] = R[c["members"], :2].mean(0)
                sizes[j] = np.median(R[c["members"], 2])
                continue
        clusters.append({"members": [idx]})
        centers = np.vstack([centers, p])
        sizes = np.append(sizes, R[idx, 2])

    # 2) Accept places where a car appeared in enough frames
    min_hits = max(2, int(np.ceil(min_presence * n_frames)))
    cand = []
    for c in clusters:
        frames_hit = {frame_ids[m] for m in c["members"]}
        if len(frames_hit) < min_hits:
            continue
        M = R[c["members"]]
        cand.append(dict(cx=float(np.median(M[:, 0])), cy=float(np.median(M[:, 1])),
                         short=float(np.median(M[:, 2])), long=float(np.median(M[:, 3])),
                         theta=float(_angle_mean(M[:, 4])), hits=len(frames_hit), source="detected"))
    rejected_moving = len(clusters) - len(cand)

    # Remove duplicates: two overlapping places (over 30% of the smaller) -> keep the one seen most often
    cand.sort(key=lambda c: -c["hits"])
    kept = []
    for c in cand:
        pc = rect_polygon(c["cx"], c["cy"], c["short"], c["long"], c["theta"])
        ac = c["short"] * c["long"]
        dup = False
        for k in kept:
            if np.hypot(c["cx"] - k["cx"], c["cy"] - k["cy"]) > (c["long"] + k["long"]) / 2:
                continue
            inter, _ = cv2.intersectConvexConvex(pc, k["poly"])
            if inter > 0.3 * min(ac, k["short"] * k["long"]):
                dup = True
                break
        if not dup:
            c["poly"] = pc
            kept.append(c)
    cand = kept

    # 3) Form rows
    rows = _group_rows(cand)

    # Exclude isolated places (not part of a row) since they are usually cars in driving lanes
    isolated = sum(len(r) for r in rows if len(r) < min_row_len)
    rows = [r for r in rows if len(r) >= min_row_len]

    # 4) Fill gaps within rows
    #    A gap may be empty spots or a driving lane (like the arrows in Parking.mp4).
    #    So an inferred spot is accepted only if a car was seen inside it in at least two frames (real evidence).
    #    Places with no evidence are saved as "candidates" and not counted; they can be accepted manually in calibrate.py.
    candidates = []
    if fill_gaps:
        for row in rows:
            new = _fill_row_gaps(row, max_gap_spots)
            for c in new:
                ev = _gap_evidence(c, R, frame_ids)
                c["hits"] = ev
                if ev >= 2:
                    row.append(c)
                else:
                    candidates.append(c)
            _order_along_row(row)

    spots = _label_rows(rows)
    cand_spots = [Spot(f"?{i + 1}", rect_polygon(c["cx"], c["cy"], c["short"] * 1.08, c["long"] * 1.05, c["theta"]),
                       "candidate") for i, c in enumerate(candidates)]
    report = {
        "sampled_frames": n_frames,
        "raw_detections": len(all_rects),
        "median_car_width_px": round(med_short, 1),
        "rejected_as_moving_or_rare": rejected_moving,
        "rejected_isolated": isolated,
        "rows": len(rows),
        "spots_detected": sum(1 for s in spots if s.source == "detected"),
        "spots_inferred_from_gaps": sum(1 for s in spots if s.source == "inferred"),
        "unverified_gap_candidates": len(cand_spots),
    }
    return spots, cand_spots, report


def _gap_evidence(c, R, frame_ids):
    """Number of frames where a car appeared inside this place with the same direction as the spots."""
    d = np.hypot(R[:, 0] - c["cx"], R[:, 1] - c["cy"])
    near = np.where(d < 0.5 * c["short"])[0]
    frames = {frame_ids[i] for i in near if _angle_diff(R[i, 4], c["theta"]) < 25}
    return len(frames)


def _row_axes(c):
    t = np.radians(c["theta"])
    u = np.array([np.cos(t), np.sin(t)])     # car length direction
    v = np.array([-np.sin(t), np.cos(t)])    # row direction (cars side by side)
    return u, v


def _group_rows(cand):
    n = len(cand)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        ui, vi = _row_axes(cand[i])
        for j in range(i + 1, n):
            if _angle_diff(cand[i]["theta"], cand[j]["theta"]) > 20:
                continue
            d = np.array([cand[j]["cx"] - cand[i]["cx"], cand[j]["cy"] - cand[i]["cy"]])
            along_car = abs(d @ ui)
            along_row = abs(d @ vi)
            L = max(cand[i]["long"], cand[j]["long"])
            S = max(cand[i]["short"], cand[j]["short"])
            if along_car < 0.3 * L and along_row < 4.5 * S:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(cand[i])
    return list(groups.values())


def _order_along_row(row):
    v = _row_axes({"theta": _angle_mean([c["theta"] for c in row])})[1]
    # Order from left to right / top to bottom
    if abs(v[0]) >= abs(v[1]):
        v = v if v[0] > 0 else -v
    else:
        v = v if v[1] > 0 else -v
    row.sort(key=lambda c: c["cx"] * v[0] + c["cy"] * v[1])
    return row


def _fill_row_gaps(row, max_gap_spots):
    if len(row) < 2:
        return []
    _order_along_row(row)
    gaps = [np.hypot(b["cx"] - a["cx"], b["cy"] - a["cy"]) for a, b in zip(row, row[1:])]
    pitch = float(np.median(gaps))
    # The distance between two adjacent spots is at least one car width
    pitch = max(pitch, 1.05 * float(np.median([c["short"] for c in row])))
    new = []
    for a, b, g in zip(list(row), row[1:], gaps):
        k = int(round(g / pitch))
        if k < 2 or k - 1 > max_gap_spots or g < 1.6 * pitch:
            continue
        for m in range(1, k):
            f = m / k
            new.append(dict(cx=a["cx"] + f * (b["cx"] - a["cx"]), cy=a["cy"] + f * (b["cy"] - a["cy"]),
                            short=(a["short"] + b["short"]) / 2, long=(a["long"] + b["long"]) / 2,
                            theta=float(_angle_mean([a["theta"], b["theta"]])), hits=0, source="inferred"))
    return new


def _row_letter(i):
    letters = string.ascii_uppercase
    return letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26]


def _label_rows(rows):
    rows = [_order_along_row(r) for r in rows]
    rows.sort(key=lambda r: (round(np.mean([c["cy"] for c in r]) / 40), np.mean([c["cx"] for c in r])))
    spots = []
    for ri, row in enumerate(rows):
        for k, c in enumerate(row, 1):
            # Expand the polygon slightly so it represents the spot, not just the car
            poly = rect_polygon(c["cx"], c["cy"], c["short"] * 1.08, c["long"] * 1.05, c["theta"])
            spots.append(Spot(f"{_row_letter(ri)}{k}", poly, c["source"]))
    return spots


def relabel(spots: list[Spot]) -> list[Spot]:
    """Renumbers the spots (after manual editing) using the same row logic."""
    cand = []
    for s in spots:
        cx, cy, sh, lo, th = rect_params(s.polygon)
        cand.append(dict(cx=cx, cy=cy, short=sh, long=lo, theta=th, spot=s))
    rows = _group_rows(cand)
    rows = [_order_along_row(r) for r in rows]
    rows.sort(key=lambda r: (round(np.mean([c["cy"] for c in r]) / 40), np.mean([c["cx"] for c in r])))
    out = []
    for ri, row in enumerate(rows):
        for k, c in enumerate(row, 1):
            s = c["spot"]
            out.append(Spot(f"{_row_letter(ri)}{k}", s.polygon, s.source))
    return out
