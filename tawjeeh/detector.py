"""
Vehicle detector using YOLO models in ONNX format.

- Runs on CPU only and does not need PyTorch or Ultralytics.
- Uses onnxruntime if installed (faster), otherwise falls back to OpenCV's cv2.dnn.
- Supports two model types:
    1) yolo11n.onnx      : general COCO model (car / bus / truck) - suited to angled cameras.
    2) yolo11n-obb.onnx  : DOTA aerial model (small/large vehicle) with oriented boxes - suited to top-down footage.
- Supports tiling: splits the frame into overlapping tiles and enlarges them, because cars in
  aerial footage are very small and many are lost if the whole frame is shrunk to the model's size.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort  # optional
except Exception:  # noqa: BLE001
    ort = None


# Class names treated as a "car" for each model type
COCO_VEHICLE_NAMES = {"car", "truck", "bus"}
DOTA_VEHICLE_NAMES = {"small vehicle", "large vehicle"}


@dataclass
class Detection:
    """A single detected car."""
    polygon: np.ndarray   # (4, 2) box corners (possibly rotated) in frame coordinates
    center: tuple          # (x, y)
    conf: float
    label: str

    @property
    def area(self) -> float:
        return float(cv2.contourArea(self.polygon.astype(np.float32)))


class VehicleDetector:
    def __init__(self, model_path: str | Path, conf: float = 0.25, iou: float = 0.45,
                 tile_size: int | None = None, tile_overlap: float = 0.2,
                 imgsz: int | None = None):
        """
        model_path : path to the ONNX file.
        conf       : minimum confidence to accept a detection.
        iou        : threshold for merging duplicate detections (NMS).
        tile_size  : size of each tile in original-frame pixels (None = whole frame).
        imgsz      : model input size (read from the model if fixed).
        """
        self.model_path = str(model_path)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.conf = conf
        self.iou = iou
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        meta_names = None
        fixed_size = None
        if ort is not None:
            self.backend = "onnxruntime"
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            if isinstance(inp.shape[2], int):
                fixed_size = inp.shape[2]
            names_str = self.session.get_modelmeta().custom_metadata_map.get("names")
            if names_str:
                meta_names = ast.literal_eval(names_str)
        else:
            self.backend = "opencv-dnn"
            self.net = cv2.dnn.readNetFromONNX(self.model_path)

        self.is_obb = "obb" in Path(self.model_path).name.lower()
        if meta_names is None:
            meta_names = _default_names(self.is_obb)
        self.names = meta_names
        wanted = DOTA_VEHICLE_NAMES if self.is_obb else COCO_VEHICLE_NAMES
        self.vehicle_ids = [i for i, n in self.names.items() if n in wanted]
        self.imgsz = imgsz or fixed_size or (1024 if self.is_obb else 640)

    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        tiles = self._make_tiles(w, h)
        boxes, scores, labels = [], [], []
        for (x0, y0, x1, y1) in tiles:
            crop = frame[y0:y1, x0:x1]
            b, s, l = self._infer(crop)
            if len(b):
                b[:, 0] += x0
                b[:, 1] += y0
                # Ignore detections cut off at a tile edge (they appear whole in the neighboring tile)
                keep = self._not_on_inner_edge(b, (x0, y0, x1, y1), w, h)
                boxes.append(b[keep]); scores.append(s[keep]); labels += [l[i] for i in np.where(keep)[0]]
        if not boxes:
            return []
        boxes = np.concatenate(boxes); scores = np.concatenate(scores)
        keep = self._nms(boxes, scores)
        dets = []
        for i in keep:
            cx, cy, bw, bh, ang = boxes[i]
            poly = cv2.boxPoints(((cx, cy), (bw, bh), np.degrees(ang)))
            dets.append(Detection(poly, (float(cx), float(cy)), float(scores[i]), labels[i]))
        return dets

    # ------------------------------------------------------------------ #
    def _make_tiles(self, w, h):
        if not self.tile_size or (w <= self.tile_size and h <= self.tile_size):
            return [(0, 0, w, h)]
        t = self.tile_size
        step = int(t * (1 - self.tile_overlap))
        xs = list(range(0, max(w - t, 0) + 1, step))
        ys = list(range(0, max(h - t, 0) + 1, step))
        if xs[-1] + t < w: xs.append(w - t)
        if ys[-1] + t < h: ys.append(h - t)
        return [(x, y, min(x + t, w), min(y + t, h)) for y in ys for x in xs]

    def _not_on_inner_edge(self, b, tile, W, H, margin=3):
        x0, y0, x1, y1 = tile
        half = np.abs(b[:, 2:4]).max(axis=1) / 2
        left = (b[:, 0] - half < x0 + margin) & (x0 > 0)
        right = (b[:, 0] + half > x1 - margin) & (x1 < W)
        top = (b[:, 1] - half < y0 + margin) & (y0 > 0)
        bot = (b[:, 1] + half > y1 - margin) & (y1 < H)
        return ~(left | right | top | bot)

    def _infer(self, img):
        """Returns an (N,5) array: cx,cy,w,h,angle(rad) in input-image coordinates."""
        s = self.imgsz
        h, w = img.shape[:2]
        r = min(s / w, s / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        canvas = np.full((s, s, 3), 114, np.uint8)
        px, py = (s - nw) // 2, (s - nh) // 2
        canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (s, s), swapRB=True)

        if self.backend == "onnxruntime":
            out = self.session.run(None, {self.input_name: blob})[0]
        else:
            self.net.setInput(blob)
            out = self.net.forward()
        out = out[0].T  # (anchors, C)

        nc = len(self.names)
        cls_scores = out[:, 4:4 + nc]
        vid = self.vehicle_ids
        sub = cls_scores[:, vid]
        best = sub.argmax(1)
        conf = sub[np.arange(len(sub)), best]
        m = conf >= self.conf
        if not m.any():
            return np.zeros((0, 5), np.float32), np.zeros(0), []
        rows = out[m]
        conf = conf[m]
        lab = [self.names[vid[k]] for k in best[m]]
        b = rows[:, :4].copy()
        ang = rows[:, 4 + nc] if self.is_obb else np.zeros(len(rows))
        b[:, 0] = (b[:, 0] - px) / r
        b[:, 1] = (b[:, 1] - py) / r
        b[:, 2] /= r
        b[:, 3] /= r
        return np.column_stack([b, ang]).astype(np.float32), conf.astype(np.float32), lab

    def _nms(self, boxes, scores):
        if self.is_obb:
            rr = [((float(b[0]), float(b[1])), (float(b[2]), float(b[3])), float(np.degrees(b[4]))) for b in boxes]
            idx = cv2.dnn.NMSBoxesRotated(rr, scores.tolist(), self.conf, self.iou)
        else:
            xywh = [[float(b[0] - b[2] / 2), float(b[1] - b[3] / 2), float(b[2]), float(b[3])] for b in boxes]
            idx = cv2.dnn.NMSBoxes(xywh, scores.tolist(), self.conf, self.iou)
        return np.array(idx).reshape(-1).tolist()


def _default_names(is_obb):
    if is_obb:
        dota = ["plane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court",
                "ground track field", "harbor", "bridge", "large vehicle", "small vehicle", "helicopter",
                "roundabout", "soccer ball field", "swimming pool"]
        return dict(enumerate(dota))
    coco = ("person bicycle car motorcycle airplane bus train truck").split()
    names = {i: f"class{i}" for i in range(80)}
    names.update(dict(enumerate(coco)))
    return names
