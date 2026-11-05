"""
Tawjeeh - Smart Parking System
Main entry point.

Examples (from inside the project folder on Windows):
    .\\.venv\\Scripts\\python.exe main.py                     # processes every video in the folder and the videos folder
    .\\.venv\\Scripts\\python.exe main.py Parking.mp4         # one video
    .\\.venv\\Scripts\\python.exe main.py Parking.mp4 --no-show
    .\\.venv\\Scripts\\python.exe main.py cam2.mp4 --config configs\\Parking.json   # same camera/scene
    .\\.venv\\Scripts\\python.exe main.py Parking.mp4 --recalibrate               # re-run automatic detection

For each video:
  - If no spot settings exist for it in configs\\ , spots are detected automatically and saved.
  - If automatic detection is unreliable, it stops and asks for manual calibration with calibrate.py.
  - Output in output\\ : <name>_result.mp4, <name>_counts.csv and <name>_summary.json
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
    raise FileNotFoundError(f"Video not found: {arg}")


def main():
    ap = argparse.ArgumentParser(description="Tawjeeh smart parking detection")
    ap.add_argument("videos", nargs="*", help="One or more videos (none = all videos)")
    ap.add_argument("--config", help="Use a specific settings file (for another video from the same camera)")
    ap.add_argument("--recalibrate", action="store_true", help="Re-run automatic spot detection")
    ap.add_argument("--no-show", action="store_true", help="Do not show the video window during processing")
    ap.add_argument("--no-ids", action="store_true", help="Hide spot numbers")
    ap.add_argument("--detect-every", type=int, default=1, help="Run detection every N frames (faster on weak machines)")
    ap.add_argument("--max-frames", type=int, default=None, help="Process only the first N frames (for testing)")
    args = ap.parse_args()

    videos = [resolve_video(v) for v in args.videos] if args.videos else find_videos()
    if not videos:
        print("No videos found. Put MP4 files in the project folder or in the videos\\ folder")
        return 1

    results = []
    for video in videos:
        print("=" * 70)
        cfg_path = Path(args.config) if args.config else config_path_for(video)
        if cfg_path.exists() and not args.recalibrate:
            cfg = CameraConfig.load(cfg_path)
            print(f"[Settings] {cfg_path.name}: {len(cfg.spots)} spots (source: {cfg.source})")
            if not check_config_matches(cfg, video):
                print(f"  ✗ Skipped {video.name}: the spot settings do not match this camera.\n"
                      f"    Run: python main.py \"{video.name}\" --recalibrate   or   python calibrate.py \"{video.name}\"")
                continue
        else:
            cfg = auto_calibrate(video)
            print(f"  - Settings saved to configs\\{video.stem}.json (review configs\\{video.stem}_spots_preview.jpg)")
        if cfg.source == "auto-failed" or not cfg.spots:
            print(f"  ✗ Automatic detection is not reliable for {video.name}.\n"
                  f"    Mark the spots manually once: python calibrate.py \"{video.name}\"")
            continue
        summary = process_video(video, cfg, show=not args.no_show, detect_every=max(1, args.detect_every),
                                max_frames=args.max_frames, show_ids=not args.no_ids)
        results.append(summary)

    if results:
        print("=" * 70)
        print("Summary per video (status in the last frame):")
        for s in results:
            c = s["final_counts"]
            print(f"  {s['video']:<30} available={c['free']:<4} occupied={c['occupied']:<4} "
                  f"uncertain={c['unknown']:<4} total={c['total']}  -> output\\{s['output_video']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
