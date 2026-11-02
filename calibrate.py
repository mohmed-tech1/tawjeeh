"""
أداة معايرة المواقف (مراجعة/تعديل يدوي) - تُستخدم مرة واحدة لكل كاميرا.

التشغيل:
    .\\.venv\\Scripts\\python.exe calibrate.py Parking.mp4            # يفتح المواقف المكتشفة تلقائيًا للمراجعة
    .\\.venv\\Scripts\\python.exe calibrate.py cam2.mp4 --empty       # يبدأ بدون مواقف (تحديد يدوي كامل)
    .\\.venv\\Scripts\\python.exe calibrate.py cam2.mp4 --empty --frame 120   # استخدام إطار آخر كمرجع

التحكم في النافذة:
    زر الفأرة الأيسر  : إضافة نقطة. كل 4 نقاط = موقف جديد (بالترتيب حول الموقف).
                        النقر على مكان رمادي "?" يقبله كموقف.
    زر الفأرة الأيمن  : حذف الموقف الذي تحت المؤشر.
    D : وضع النسخ - كل نقرة يسار تضع نسخة من آخر موقف في مكان النقرة (أسرع لصف كامل).
    C : قبول كل الأماكن الرمادية المرشحة.
    Z : تراجع عن آخر نقطة / آخر موقف مضاف.
    S : حفظ (ويعيد ترقيم المواقف A1, A2 ...).
    Q أو Esc : خروج.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from tawjeeh.pipeline import (CONFIGS_DIR, auto_calibrate, config_path_for, read_frame,
                              save_config_with_reference, video_info)
from tawjeeh.spots import CameraConfig, Spot, relabel
from tawjeeh.occupancy import OccupancyParams

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


class SpotEditor:
    """منطق التعديل (بدون واجهة) حتى يمكن اختباره."""

    def __init__(self, cfg: CameraConfig, ref_frame):
        self.cfg = cfg
        self.ref = ref_frame
        self.points: list[tuple[float, float]] = []
        self.stamp_mode = False
        self.history: list[tuple[str, Spot]] = []
        self.dirty = False

    @staticmethod
    def _inside(poly, x, y):
        return cv2.pointPolygonTest(poly.astype(np.float32), (float(x), float(y)), False) >= 0

    def left_click(self, x, y):
        if not self.points:
            for i, c in enumerate(self.cfg.candidates):          # قبول مكان مرشح
                if self._inside(c.polygon, x, y):
                    spot = Spot("new", c.polygon.copy(), "manual")
                    self.cfg.candidates.pop(i)
                    self._add(spot)
                    return "accepted-candidate"
            if self.stamp_mode and self.cfg.spots:
                tmpl = self.cfg.spots[-1].polygon
                poly = tmpl - tmpl.mean(0) + np.array([x, y], np.float32)
                self._add(Spot("new", poly.astype(np.float32), "manual"))
                return "stamped"
        self.points.append((x, y))
        if len(self.points) == 4:
            poly = np.array(self.points, np.float32)
            # ترتيب النقاط حول المركز لتجنب المضلع المتقاطع
            c = poly.mean(0)
            order = np.argsort(np.arctan2(poly[:, 1] - c[1], poly[:, 0] - c[0]))
            self.points = []
            self._add(Spot("new", poly[order], "manual"))
            return "added"
        return "point"

    def right_click(self, x, y):
        for i in range(len(self.cfg.spots) - 1, -1, -1):
            if self._inside(self.cfg.spots[i].polygon, x, y):
                s = self.cfg.spots.pop(i)
                self.history.append(("deleted", s))
                self.dirty = True
                return s.id
        return None

    def accept_all_candidates(self):
        n = len(self.cfg.candidates)
        for c in self.cfg.candidates:
            self._add(Spot("new", c.polygon.copy(), "manual"))
        self.cfg.candidates = []
        return n

    def undo(self):
        if self.points:
            self.points.pop()
            return "point"
        if self.history:
            action, s = self.history.pop()
            if action == "added":
                self.cfg.spots.remove(s)
            else:
                self.cfg.spots.append(s)
            self.dirty = True
            return action
        return None

    def _add(self, spot):
        self.cfg.spots.append(spot)
        self.history.append(("added", spot))
        self.dirty = True

    def save(self):
        self.cfg.spots = relabel(self.cfg.spots)
        if any(s.source == "manual" for s in self.cfg.spots) or self.cfg.source == "auto-failed":
            self.cfg.source = "manual" if all(s.source == "manual" for s in self.cfg.spots) else "auto+manual"
        self.cfg.notes = [n for n in self.cfg.notes if "غير موثوق" not in n]
        save_config_with_reference(self.cfg, self.ref)
        self.dirty = False

    def render(self):
        img = self.ref.copy()
        for s in self.cfg.candidates:
            cv2.polylines(img, [s.polygon.astype(np.int32)], True, (170, 170, 170), 1, cv2.LINE_AA)
        for s in self.cfg.spots:
            col = (255, 0, 255) if s.source == "manual" else (60, 200, 60)
            cv2.polylines(img, [s.polygon.astype(np.int32)], True, col, 1, cv2.LINE_AA)
        for p in self.points:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1)
        if len(self.points) > 1:
            cv2.polylines(img, [np.array(self.points, np.int32)], False, (0, 255, 255), 1)
        status = (f"spots={len(self.cfg.spots)}  candidates={len(self.cfg.candidates)}  "
                  f"stamp={'ON' if self.stamp_mode else 'off'}  {'*unsaved*' if self.dirty else 'saved'}")
        help_ = "L-click: point(4=spot)/accept '?'  R-click: delete  D: stamp  C: accept all '?'  Z: undo  S: save  Q: quit"
        for i, t in enumerate((status, help_)):
            cv2.putText(img, t, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, t, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img


def run_gui(editor: SpotEditor, max_width=1600):
    h, w = editor.ref.shape[:2]
    scale = min(1.0, max_width / w)
    win = "Tawjeeh - Calibrate spots"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, int(w * scale), int(h * scale))

    def on_mouse(event, x, y, flags, param):
        # النافذة قابلة لتغيير الحجم؛ OpenCV يعيد الإحداثيات بمقياس الصورة الأصلية
        if event == cv2.EVENT_LBUTTONDOWN:
            editor.left_click(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            editor.right_click(x, y)

    cv2.setMouseCallback(win, on_mouse)
    while True:
        cv2.imshow(win, editor.render())
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            if editor.dirty:
                print("توجد تغييرات غير محفوظة: اضغط S للحفظ أو Q مرة أخرى للخروج بدون حفظ.")
                editor.dirty = False
                continue
            break
        if k == ord("s"):
            editor.save()
            print(f"تم الحفظ: {len(editor.cfg.spots)} موقف -> configs\\{Path(editor.cfg.video).stem}.json")
        elif k == ord("z"):
            editor.undo()
        elif k == ord("d"):
            editor.stamp_mode = not editor.stamp_mode
        elif k == ord("c"):
            print(f"تم قبول {editor.accept_all_candidates()} مكان مرشح")
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Tawjeeh - spot calibration tool")
    ap.add_argument("video")
    ap.add_argument("--empty", action="store_true", help="البدء بدون مواقف (تحديد يدوي كامل)")
    ap.add_argument("--frame", type=int, default=0, help="رقم الإطار المرجعي (مع --empty فقط)")
    args = ap.parse_args()

    from main import resolve_video
    video = resolve_video(args.video)
    cfg_path = config_path_for(video)

    if args.empty:
        fps, n, w, h = video_info(video)
        ref = read_frame(video, args.frame)
        from tawjeeh.pipeline import choose_detector
        cfg = CameraConfig(video=video.name, frame_size=(w, h), reference_frame_index=args.frame,
                           source="manual", stabilize=True, detector=choose_detector(ref),
                           occupancy=OccupancyParams().__dict__)
    elif cfg_path.exists():
        cfg = CameraConfig.load(cfg_path)
        ref = cv2.imread(str(CONFIGS_DIR / cfg.reference_image))
    else:
        print("لا توجد إعدادات لهذا الفيديو؛ تشغيل الاكتشاف التلقائي أولًا...")
        cfg = auto_calibrate(video)
        ref = cv2.imread(str(CONFIGS_DIR / cfg.reference_image))

    run_gui(SpotEditor(cfg, ref))


if __name__ == "__main__":
    main()
