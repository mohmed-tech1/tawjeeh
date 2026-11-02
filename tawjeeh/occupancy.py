"""
تحديد حالة كل موقف في كل إطار + تثبيت الحالة عبر الزمن.

الحالة الخام لكل إطار (raw):
  OCCUPIED : سيارة مكتشفة تغطي جزءًا كافيًا من الموقف.
  FREE     : لا توجد سيارة مكتشفة، والمنطقة تبدو أرضية فارغة.
  UNKNOWN  : (أ) تداخل جزئي غير حاسم أو سيارة مجاورة تغطي جزءًا كبيرًا من الموقف، أو (ب) لا يوجد كشف لكن المنطقة تبدو غير فارغة
             (ربما سيارة لم يكتشفها النموذج)، أو (ج) الموقف خارج الصورة حاليًا.
  -> لا نصنّف الموقف "متاحًا" إلا إذا كان الدليل واضحًا.

التثبيت الزمني (smoothing):
  لكل موقف نافذة لآخر N حالات خام. تتغير الحالة المعروضة فقط إذا كانت نسبة حالة معينة
  داخل النافذة أكبر من vote_ratio. هذا يمنع الوميض بين الأحمر والأخضر.

منع احتساب السيارة الواحدة في أكثر من موقف:
  كل سيارة تُسند إلى موقف واحد فقط: الموقف الذي تقع داخله أكبر نسبة من مساحتها.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

FREE, OCCUPIED, UNKNOWN = 0, 1, 2
STATE_NAMES = {FREE: "free", OCCUPIED: "occupied", UNKNOWN: "unknown"}


@dataclass
class OccupancyParams:
    occupied_overlap: float = 0.35   # نسبة مساحة السيارة داخل الموقف لاعتباره مشغولًا
    partial_overlap: float = 0.15    # أقل من occupied وأكبر من هذه = غير مؤكد
    blocked_coverage: float = 0.25   # سيارة (مُسندة لموقف مجاور) تغطي هذه النسبة من الموقف = غير مؤكد
    busy_threshold: float = 0.10     # كثافة الحواف داخل موقف بلا كشف: فوقها = غير مؤكد
    window_seconds: float = 0.5      # طول نافذة التثبيت الزمني
    vote_ratio: float = 0.6          # نسبة الأصوات المطلوبة لتغيير الحالة
    keep_ratio: float = 0.3          # أقل نسبة للإبقاء على الحالة السابقة
    use_appearance_check: bool = True

    @staticmethod
    def from_dict(d):
        p = OccupancyParams()
        for k, v in (d or {}).items():
            if hasattr(p, k):
                setattr(p, k, type(getattr(p, k))(v))
        return p


class OccupancyTracker:
    def __init__(self, spots, fps: float, params: OccupancyParams | None = None):
        self.spots = spots
        self.p = params or OccupancyParams()
        self.polys = [s.polygon.astype(np.float32) for s in spots]
        self.areas = np.array([cv2.contourArea(p) for p in self.polys])
        self.centers = np.array([p.mean(0) for p in self.polys])
        self.radius = np.array([np.linalg.norm(p - p.mean(0), axis=1).max() for p in self.polys])
        n_win = max(3, int(round(self.p.window_seconds * fps)))
        self.history = [deque(maxlen=n_win) for _ in spots]
        self.state = np.full(len(spots), UNKNOWN, np.int8)
        self.last_raw = np.full(len(spots), UNKNOWN, np.int8)
        self.last_busy = np.zeros(len(spots))

    # ------------------------------------------------------------------ #
    def raw_states(self, det_polys_ref, frame=None, H_ref_to_frame=None):
        """det_polys_ref: مضلعات السيارات بإحداثيات الإطار المرجعي."""
        n = len(self.spots)
        best = np.zeros(n)             # أكبر نسبة من مساحة سيارة مُسندة لهذا الموقف
        covered = np.zeros(n)          # أكبر نسبة من مساحة الموقف تغطيها أي سيارة (حتى غير المُسندة له)
        for dp in det_polys_ref:
            dp = dp.astype(np.float32)
            d_area = cv2.contourArea(dp)
            if d_area <= 1:
                continue
            c = dp.mean(0)
            near = np.where(np.linalg.norm(self.centers - c, axis=1) < self.radius + np.linalg.norm(dp - c, axis=1).max())[0]
            best_j, best_r = -1, 0.0
            for j in near:
                inter, _ = cv2.intersectConvexConvex(self.polys[j], dp)
                covered[j] = max(covered[j], inter / max(self.areas[j], 1.0))
                r = inter / d_area
                if r > best_r:
                    best_j, best_r = j, r
            if best_j >= 0:
                best[best_j] = max(best[best_j], best_r)  # السيارة تُحسب لموقف واحد فقط

        raw = np.full(n, FREE, np.int8)
        raw[best >= self.p.partial_overlap] = UNKNOWN
        # سيارة متجاورة تغطي جزءًا كبيرًا من الموقف (متوقفة على الحد) -> لا نعتبره متاحًا
        raw[covered >= self.p.blocked_coverage] = UNKNOWN
        raw[best >= self.p.occupied_overlap] = OCCUPIED

        if frame is not None and H_ref_to_frame is not None:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 150)
            for j in range(n):
                pts = cv2.perspectiveTransform(self.polys[j].reshape(-1, 1, 2), H_ref_to_frame).reshape(-1, 2)
                if pts[:, 0].min() < 0 or pts[:, 1].min() < 0 or pts[:, 0].max() >= w or pts[:, 1].max() >= h:
                    raw[j] = UNKNOWN          # الموقف خارج الصورة
                    continue
                if raw[j] == FREE and self.p.use_appearance_check:
                    busy = _edge_density(edges, pts)
                    self.last_busy[j] = busy
                    if busy > self.p.busy_threshold:
                        raw[j] = UNKNOWN
        return raw

    def update(self, raw):
        self.last_raw = raw
        for j, r in enumerate(raw):
            hist = self.history[j]
            hist.append(int(r))
            cnt = np.bincount(np.fromiter(hist, np.int8), minlength=3)
            ratios = cnt / len(hist)
            cur = self.state[j]
            if len(hist) < 3:                       # بداية الفيديو: نأخذ الحالة مباشرة
                self.state[j] = r
            elif ratios[OCCUPIED] >= self.p.vote_ratio:
                self.state[j] = OCCUPIED
            elif ratios[FREE] >= self.p.vote_ratio:
                self.state[j] = FREE
            elif ratios[cur] >= self.p.keep_ratio:
                pass                                # إبقاء الحالة السابقة (hysteresis)
            else:
                self.state[j] = UNKNOWN
        return self.state

    def counts(self):
        return {
            "free": int((self.state == FREE).sum()),
            "occupied": int((self.state == OCCUPIED).sum()),
            "unknown": int((self.state == UNKNOWN).sum()),
            "total": len(self.spots),
        }


def _edge_density(edges, pts):
    """نسبة بكسلات الحواف في الجزء الداخلي من الموقف (نتجاهل خطوط التخطيط على الأطراف)."""
    c = pts.mean(0)
    inner = (c + 0.6 * (pts - c)).astype(np.int32)
    x, y, w, h = cv2.boundingRect(inner)
    if w < 3 or h < 3:
        return 0.0
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, inner - [x, y], 255)
    roi = edges[y:y + h, x:x + w]
    m = mask > 0
    return float((roi[m] > 0).mean()) if m.any() else 0.0
