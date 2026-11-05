"""
Full processing pipeline:
  - auto_calibrate : automatically detect the spots for a video/camera and save the settings.
  - process_video  : process the video and output an MP4 video plus a statistics file.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .detector import VehicleDetector
from .occupancy import OccupancyParams, OccupancyTracker, STATE_NAMES
from .render import draw_panel, draw_spots
from .spots import CameraConfig, discover_spots
from .stabilizer import FrameAligner, warp_points

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CONFIGS_DIR = ROOT / "configs"
OUTPUT_DIR = ROOT / "output"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

MODEL_AERIAL = "yolo11n-obb.onnx"   # top-down cameras (drone/ceiling)
MODEL_GROUND = "yolo11n.onnx"       # angled / ground-level cameras


def config_path_for(video: Path) -> Path:
    return CONFIGS_DIR / f"{video.stem}.json"


def read_frame(video: Path, index: int):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {index} from {video}")
    return frame


def video_info(video: Path):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n, w, h


def make_detector(det_cfg: dict) -> VehicleDetector:
    return VehicleDetector(MODELS_DIR / det_cfg["model"], conf=det_cfg.get("conf", 0.25),
                           iou=det_cfg.get("iou", 0.45), tile_size=det_cfg.get("tile_size"))


def choose_detector(frame, log=print) -> dict:
    """Tries the aerial model and the regular model, and picks the one that detects more cars in this scene."""
    best = None
    for model in (MODEL_AERIAL, MODEL_GROUND):
        if not (MODELS_DIR / model).exists():
            continue
        cfg = {"model": model, "conf": 0.25, "iou": 0.45, "tile_size": None}
        n = len(make_detector(cfg).detect(frame))
        log(f"  - Model {model}: {n} cars in the reference frame")
        if best is None or n > best[1]:
            best = (cfg, n)
    if best is None:
        raise FileNotFoundError(f"No model found in {MODELS_DIR}")
    return best[0]


# ---------------------------------------------------------------------- #
def auto_calibrate(video: Path, n_samples: int = 40, ref_index: int = 0, log=print) -> CameraConfig:
    fps, n_frames, w, h = video_info(video)
    log(f"[Auto-calibration] {video.name}: {w}x{h}, {fps:.1f} fps, {n_frames} frames")
    ref = read_frame(video, ref_index)
    det_cfg = choose_detector(ref, log)
    detector = make_detector(det_cfg)
    aligner = FrameAligner(ref)

    idxs = np.linspace(0, max(n_frames - 1, 0), min(n_samples, max(n_frames, 1))).astype(int)
    cap = cv2.VideoCapture(str(video))
    samples, failed_align, total_shift = [], 0, []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        H, ok_align = aligner.align(frame)
        if not ok_align and i != ref_index:
            failed_align += 1
            continue
        total_shift.append(float(np.abs(warp_points([[w / 2, h / 2]], H) - [w / 2, h / 2]).max()))
        dets = detector.detect(frame)
        samples.append([warp_points(d.polygon, H) for d in dets])
    cap.release()
    log(f"  - Sample frames: {len(samples)} (alignment failed on {failed_align})")

    spots, candidates, report = discover_spots(samples)
    report["camera_motion_px_max"] = round(max(total_shift) if total_shift else 0.0, 1)
    cfg = CameraConfig(video=video.name, frame_size=(w, h), reference_frame_index=ref_index,
                       source="auto", stabilize=True, detector=det_cfg,
                       occupancy=OccupancyParams().__dict__, spots=spots, candidates=candidates)

    # Assess the reliability of the automatic detection
    notes = []
    if len(spots) < 4:
        notes.append("Automatic detection is unreliable: too few spots found. Use calibrate.py to mark them manually.")
        cfg.source = "auto-failed"
    if candidates:
        notes.append(f"{len(candidates)} possible spots (gaps with no evidence of a car) were not counted; "
                     "review them in the preview image (gray) and accept the correct ones with calibrate.py.")
    if failed_align > len(idxs) / 3:
        notes.append("The camera moves too much and alignment failed on many frames.")
    cfg.notes = notes
    log("  - Report: " + json.dumps(report, ensure_ascii=False))
    for nt in notes:
        log("  ! " + nt)
    save_config_with_reference(cfg, ref)
    return cfg


def save_config_with_reference(cfg: CameraConfig, ref_frame):
    stem = Path(cfg.video).stem
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    ref_path = CONFIGS_DIR / f"{stem}_reference.jpg"
    cv2.imwrite(str(ref_path), ref_frame)
    cfg.reference_image = ref_path.name
    cfg.save(CONFIGS_DIR / f"{stem}.json")
    # Spot preview image for visual verification
    prev = ref_frame.copy()
    for s in cfg.spots:
        col = {"detected": (60, 200, 60), "inferred": (255, 160, 0), "manual": (255, 0, 255)}.get(s.source, (200, 200, 200))
        cv2.polylines(prev, [s.polygon.astype(np.int32)], True, col, 1, cv2.LINE_AA)
        c = s.polygon.mean(0).astype(int)
        cv2.putText(prev, s.id, (int(c[0]) - 8, int(c[1]) + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)
    for s in cfg.candidates:
        cv2.polylines(prev, [s.polygon.astype(np.int32)], True, (170, 170, 170), 1, cv2.LINE_AA)
        c = s.polygon.mean(0).astype(int)
        cv2.putText(prev, "?", (int(c[0]) - 4, int(c[1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(prev, "green=detected  blue=inferred(gap)  magenta=manual  gray ?=unverified candidate", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(CONFIGS_DIR / f"{stem}_spots_preview.jpg"), prev)


def check_config_matches(cfg: CameraConfig, video: Path, log=print) -> bool:
    """Checks that the spot settings belong to this same camera (never use one camera's spots on another)."""
    fps, n, w, h = video_info(video)
    if (w, h) != tuple(cfg.frame_size):
        log(f"  ! Video resolution {w}x{h} differs from the settings {cfg.frame_size}.")
        return False
    ref_path = CONFIGS_DIR / cfg.reference_image
    if not ref_path.exists():
        return True
    ref = cv2.imread(str(ref_path))
    frame = read_frame(video, min(cfg.reference_frame_index, max(n - 1, 0)))
    aligner = FrameAligner(ref)
    if not aligner.usable:
        return True        # scene lacks enough detail to compare; matching resolution is enough
    H, inliers = aligner.match_score(frame)
    if H is None or inliers < 40:
        log(f"  ! Scene does not match the settings reference image (matched points: {inliers}).")
        return False
    # Same place but a different angle? Only slight shake/shift is allowed (8% of the image diagonal)
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    shift = np.linalg.norm(warp_points(corners, H) - corners, axis=1).max()
    if shift > 0.08 * np.hypot(w, h):
        log(f"  ! Camera angle differs from the settings (shift of {shift:.0f} pixels).")
        return False
    return True


# ---------------------------------------------------------------------- #
def open_writer(path: Path, fps, size):
    # mp4v works with OpenCV on Windows without extra libraries and opens in VLC and Windows Media Player
    for fourcc in ("mp4v", "avc1"):
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if wr.isOpened():
            return wr, fourcc
        wr.release()
    raise RuntimeError("Could not create the output video file (no supported MP4 codec in OpenCV)")


def process_video(video: Path, cfg: CameraConfig, show: bool = True, detect_every: int = 1,
                  max_frames: int | None = None, show_ids: bool = True, log=print):
    fps, n_frames, w, h = video_info(video)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{video.stem}_result.mp4"
    csv_path = OUTPUT_DIR / f"{video.stem}_counts.csv"

    detector = make_detector(cfg.detector)
    params = OccupancyParams.from_dict(cfg.occupancy)
    tracker = OccupancyTracker(cfg.spots, fps / max(detect_every, 1), params)
    ref = cv2.imread(str(CONFIGS_DIR / cfg.reference_image)) if cfg.reference_image else None
    aligner = FrameAligner(ref) if (cfg.stabilize and ref is not None) else None

    writer, codec = open_writer(out_path, fps, (w, h))
    log(f"[Processing] {video.name} -> {out_path.name} (codec {codec}, model {cfg.detector['model']}, "
        f"{len(cfg.spots)} spots, engine {detector.backend})")

    cap = cv2.VideoCapture(str(video))
    H = np.eye(3)
    align_fail_streak = 0
    rows, idx, t0 = [], 0, time.time()
    win = f"Tawjeeh - {video.name}"
    det_count = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and idx >= max_frames):
            break
        if aligner is not None:
            H_new, ok_align = aligner.align(frame)
            align_fail_streak = 0 if ok_align else align_fail_streak + 1
            H = H_new
        H_inv = np.linalg.inv(H)

        if idx % detect_every == 0:
            dets = detector.detect(frame)
            det_count = len(dets)
            det_ref = [warp_points(d.polygon, H) for d in dets]
            raw = tracker.raw_states(det_ref, frame, H_inv)
            if align_fail_streak > fps:      # alignment lost for more than a second
                raw[:] = 2
            tracker.update(raw)

        counts = tracker.counts()
        vis = draw_spots(frame.copy(), cfg.spots, tracker.state, H_inv, show_ids=show_ids)
        draw_panel(vis, counts, extra=f"t={idx / fps:5.1f}s  cars detected={det_count}")
        writer.write(vis)

        if idx % max(1, int(round(fps / 2))) == 0:
            rows.append({"frame": idx, "time_s": round(idx / fps, 2), **counts, "cars_detected": det_count})
        if show:
            try:
                disp = vis if w <= 1600 else cv2.resize(vis, None, fx=1600 / w, fy=1600 / w)
                cv2.imshow(win, disp)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    log("  - Display stopped by the user.")
                    break
            except cv2.error:
                show = False
                log("  - Cannot open a display window in this environment; continuing to save without display.")
        idx += 1
        if idx % 100 == 0:
            el = time.time() - t0
            log(f"  {idx}/{n_frames} frames ({idx / el:.1f} frames/s) available={counts['free']} occupied={counts['occupied']} uncertain={counts['unknown']}")
    cap.release()
    writer.release()
    if show:
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["frame"])
        wr.writeheader()
        wr.writerows(rows)
    final = tracker.counts()
    summary = {
        "video": video.name, "output_video": out_path.name, "frames_processed": idx,
        "seconds": round(time.time() - t0, 1), "final_counts": final,
        "final_spot_states": {s.id: STATE_NAMES[int(st)] for s, st in zip(cfg.spots, tracker.state)},
    }
    (OUTPUT_DIR / f"{video.stem}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"  ✓ Done: {idx} frames in {summary['seconds']} s. Final: available={final['free']} occupied={final['occupied']} "
        f"uncertain={final['unknown']} of {final['total']}")
    return summary
