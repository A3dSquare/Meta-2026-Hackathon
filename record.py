"""Record gameplay for labeling (YOLO detector) -- small mp4, streamed to disk.

    python record.py --level deadlocked              # 120 s at 5 fps -> recs/deadlocked.mp4
    python record.py --level deadlocked --seconds 60 --fps 8

Start it, click into Geometry Dash, and play (or let the bot play) the level.
Frames are written as they arrive (1280x720), so memory use stays flat. A
2-minute recording is ~600 frames / ~30-60 MB. Ctrl+C stops early and keeps
what was recorded.

Legacy mode (the old reward-replay format, full-res .npz) is still there:
    python record.py --out recs/level1 --seconds 30
It keeps every frame in RAM -- only use it for short clips.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def record_mp4(level: str, seconds: float, fps: float, width: int = 1280, out_dir: str = "recs") -> str:
    import cv2
    from capture import GDCapture
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{level}.mp4")
    k = 1
    while os.path.exists(path):                      # never overwrite a recording
        path = os.path.join(out_dir, f"{level}_{k}.mp4")
        k += 1
    cap = GDCapture()
    writer = None
    n = 0
    try:
        cap.start()
        for _ in range(10):
            cap.read()
        print(f"recording {seconds:.0f}s at {fps} fps -> {path}  (click into GD now; Ctrl+C to stop)")
        t0 = time.perf_counter()
        nxt = t0
        while time.perf_counter() - t0 < seconds:
            cap.read()                               # keeps the capture fresh at 60 Hz
            now = time.perf_counter()
            if now < nxt:
                continue
            nxt += 1.0 / fps
            full = cap.last_fullres
            if full is None:
                continue
            full = np.asarray(full)[:, :, :3]
            h = int(round(full.shape[0] * width / full.shape[1]))
            fr = cv2.resize(full, (width, h), interpolation=cv2.INTER_AREA)
            if writer is None:
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, h))
            writer.write(np.ascontiguousarray(fr))
            n += 1
            if n % max(1, int(fps * 10)) == 0:
                print(f"  {n} frames ({now - t0:.0f}s)")
    except KeyboardInterrupt:
        print("stopped early")
    finally:
        cap.stop()
        if writer is not None:
            writer.release()
    print(f"saved {n} frames -> {path}")
    print(f"next: python label_boxes.py --rec {path}")
    return path


def record_legacy(out: str, seconds: float, frame_hz: float) -> int:
    from capture import GDCapture
    os.makedirs(out, exist_ok=True)
    cap = GDCapture()
    fulls, full_ts, obss, obs_ts = [], [], [], []
    try:
        cap.start()
        for _ in range(10):
            cap.read()
        t0 = time.perf_counter()
        next_full = t0
        while time.perf_counter() - t0 < seconds:
            obs = cap.read()
            now = time.perf_counter()
            obss.append(obs.copy())
            obs_ts.append(now - t0)
            if now >= next_full:
                full = cap.last_fullres
                if full is not None:
                    fulls.append(full.copy())
                    full_ts.append(now - t0)
                next_full += 1.0 / frame_hz
    finally:
        cap.stop()
    np.savez_compressed(os.path.join(out, "frames.npz"),
                        frames=np.stack(fulls) if fulls else np.zeros((0, 1, 1, 3), np.uint8),
                        ts=np.asarray(full_ts))
    np.savez_compressed(os.path.join(out, "obs.npz"),
                        obs=np.stack(obss) if obss else np.zeros((0, 4, 180, 320, 3), np.uint8),
                        ts=np.asarray(obs_ts))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"seconds": seconds, "frame_hz": frame_hz,
                   "n_full": len(fulls), "n_obs": len(obss)}, f, indent=2)
    print(f"saved {len(fulls)} full + {len(obss)} obs -> {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="record gameplay (mp4 for labeling)")
    p.add_argument("--level", help="level name -> recs/<level>.mp4")
    p.add_argument("--seconds", type=float, default=120.0)
    p.add_argument("--fps", type=float, default=5.0, help="frames per second saved")
    p.add_argument("--out", help="LEGACY: output dir for the old .npz format")
    p.add_argument("--frame_hz", type=float, default=10.0, help="LEGACY: full-res save rate")
    a = p.parse_args()
    if a.out:
        return record_legacy(a.out, a.seconds, a.frame_hz)
    if not a.level:
        p.error("give --level NAME (e.g. --level deadlocked)")
    record_mp4(a.level, a.seconds, a.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
