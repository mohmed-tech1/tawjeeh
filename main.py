"""
Tawjeeh - نظام المواقف الذكية
ملف التشغيل الرئيسي.

أمثلة (من داخل مجلد المشروع على Windows):
    .\\.venv\\Scripts\\python.exe main.py                     # يعالج كل الفيديوهات في المجلد و videos\\
    .\\.venv\\Scripts\\python.exe main.py Parking.mp4         # فيديو واحد
    .\\.venv\\Scripts\\python.exe main.py Parking.mp4 --no-show
    .\\.venv\\Scripts\\python.exe main.py cam2.mp4 --config configs\\Parking.json   # نفس الكاميرا/المشهد
    .\\.venv\\Scripts\\python.exe main.py Parking.mp4 --recalibrate               # إعادة الاكتشاف التلقائي

لكل فيديو:
  - إذا لم توجد إعدادات مواقف له في configs\\ ، يتم اكتشاف المواقف تلقائيًا وحفظها.
  - إذا كان الاكتشاف التلقائي غير موثوق، يتوقف ويطلب المعايرة اليدوية عبر calibrate.py.
  - الناتج في output\\ : <اسم>_result.mp4 و <اسم>_counts.csv و <اسم>_summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tawjeeh.pipeline import (ROOT, VIDEO_EXTS, auto_calibrate, check_config_matches,
                              config_path_for, process_video)
from tawjeeh.spots import CameraConfig

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def find_videos() -> list[Path]:
    vids = []
    for folder in (ROOT, ROOT / "videos"):
        if folder.is_dir():
            vids += [p for p in sorted(folder.iterdir()) if p.suffix.lower() in VIDEO_EXTS]
    return vids


def resolve_video(arg: str) -> Path:
    p = Path(arg)
    for cand in (p, ROOT / p, ROOT / "videos" / p):
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(f"الفيديو غير موجود: {arg}")


def main():
    ap = argparse.ArgumentParser(description="Tawjeeh smart parking detection")
    ap.add_argument("videos", nargs="*", help="فيديو أو أكثر (بدون قيمة = كل الفيديوهات)")
    ap.add_argument("--config", help="استخدام ملف إعدادات محدد (لفيديو آخر من نفس الكاميرا)")
    ap.add_argument("--recalibrate", action="store_true", help="إعادة اكتشاف المواقف تلقائيًا")
    ap.add_argument("--no-show", action="store_true", help="عدم عرض نافذة الفيديو أثناء المعالجة")
    ap.add_argument("--no-ids", action="store_true", help="إخفاء أرقام المواقف")
    ap.add_argument("--detect-every", type=int, default=1, help="تشغيل الكشف كل N إطار (أسرع على الأجهزة الضعيفة)")
    ap.add_argument("--max-frames", type=int, default=None, help="معالجة أول N إطار فقط (للتجربة)")
    args = ap.parse_args()

    videos = [resolve_video(v) for v in args.videos] if args.videos else find_videos()
    if not videos:
        print("لم يتم العثور على فيديوهات. ضع ملفات MP4 في مجلد المشروع أو في مجلد videos\\")
        return 1

    results = []
    for video in videos:
        print("=" * 70)
        cfg_path = Path(args.config) if args.config else config_path_for(video)
        if cfg_path.exists() and not args.recalibrate:
            cfg = CameraConfig.load(cfg_path)
            print(f"[إعدادات] {cfg_path.name}: {len(cfg.spots)} موقف (المصدر: {cfg.source})")
            if not check_config_matches(cfg, video):
                print(f"  ✗ تم تخطي {video.name}: إعدادات المواقف لا تطابق هذه الكاميرا.\n"
                      f"    شغّل: python main.py \"{video.name}\" --recalibrate   أو   python calibrate.py \"{video.name}\"")
                continue
        else:
            cfg = auto_calibrate(video)
            print(f"  - حُفظت الإعدادات في configs\\{video.stem}.json (راجع configs\\{video.stem}_spots_preview.jpg)")
        if cfg.source == "auto-failed" or not cfg.spots:
            print(f"  ✗ لا يمكن الاعتماد على الاكتشاف التلقائي لـ {video.name}.\n"
                  f"    حدّد المواقف يدويًا مرة واحدة: python calibrate.py \"{video.name}\"")
            continue
        summary = process_video(video, cfg, show=not args.no_show, detect_every=max(1, args.detect_every),
                                max_frames=args.max_frames, show_ids=not args.no_ids)
        results.append(summary)

    if results:
        print("=" * 70)
        print("ملخص لكل فيديو (الحالة في آخر إطار):")
        for s in results:
            c = s["final_counts"]
            print(f"  {s['video']:<30} متاح={c['free']:<4} مشغول={c['occupied']:<4} "
                  f"غير مؤكد={c['unknown']:<4} الإجمالي={c['total']}  -> output\\{s['output_video']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
