"""
تثبيت الصورة (Frame alignment).

إحداثيات المواقف تُحفظ على "إطار مرجعي" واحد لكل كاميرا. إذا اهتزت الكاميرا أو تحركت
قليلًا (مثل تصوير الدرون في Parking.mp4)، نحسب لكل إطار مصفوفة تحويل (Homography)
تربطه بالإطار المرجعي، فتبقى المواقف في أماكنها الصحيحة على الصورة.

للكاميرات الثابتة تمامًا تكون المصفوفة قريبة من المصفوفة الأحادية ولا يتغير شيء.
"""

from __future__ import annotations

import cv2
import numpy as np


class FrameAligner:
    def __init__(self, reference_frame: np.ndarray, work_width: int = 960, n_features: int = 4000):
        self.scale = min(1.0, work_width / reference_frame.shape[1])
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ref_kp, self.ref_des = self._features(reference_frame)
        # مشهد بلا تفاصيل كافية (أرضية متجانسة): لا يمكن التثبيت -> نفترض كاميرا ثابتة
        self.usable = self.ref_des is not None and len(self.ref_kp) >= 80
        self.last_H = np.eye(3)          # آخر تحويل ناجح (إطار حالي -> مرجعي)
        self.last_inliers = 0

    def _features(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.scale != 1.0:
            g = cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return self.orb.detectAndCompute(g, None)

    def match_score(self, frame) -> tuple[np.ndarray | None, int]:
        """يحسب التحويل من هذا الإطار إلى المرجعي. يرجع (H, عدد النقاط المتطابقة)."""
        kp, des = self._features(frame)
        if des is None or self.ref_des is None or len(kp) < 10:
            return None, 0
        pairs = self.matcher.knnMatch(des, self.ref_des, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < 12:
            return None, len(good)
        src = np.float32([kp[m.queryIdx].pt for m in good]) / self.scale
        dst = np.float32([self.ref_kp[m.trainIdx].pt for m in good]) / self.scale
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            return None, 0
        inl = int(mask.sum())
        # رفض التحويلات غير المنطقية (تكبير/تصغير شديد)
        det = np.linalg.det(H[:2, :2])
        if not (0.5 < det < 2.0) or inl < 12:
            return None, inl
        return H, inl

    def align(self, frame) -> tuple[np.ndarray, bool]:
        """يرجع (H إطار->مرجعي، هل نجح التثبيت في هذا الإطار)."""
        if not self.usable:
            return np.eye(3), True
        H, inl = self.match_score(frame)
        self.last_inliers = inl
        if H is None:
            return self.last_H, False
        self.last_H = H
        return H, True


def warp_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)
