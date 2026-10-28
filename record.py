"""PHASE 1 record: save full-res frames + timestamps for a play session.

Usage:
  python record.py --out recs/level1 --seconds 180
  (then play 3 levels: different bar colors, bright/dark/flashing bgs)

Writes <out>/meta.json + <out>/frames.npz (uint8 BGR full-res, 10Hz to
keep files small) + <out>/obs.npz (320x180 4-stack, 60Hz ticks for
ProgressReward replay). Timestamps are perf_counter walls for 1s labels.
No detector runs here: pure capture, so recording never biases the eval.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description="record a labeled play session")
    p.add_argument("--out", required=True, help="output dir, e.g. recs/level1")
    p.add_argument("--seconds", type=float, default=180.0)
    p.add_argument("--frame_hz", type=float, default=10.0, help="full-res save rate")
    args = p.parse_args()

    from capture import GDCapture
    os.makedirs(args.out, exist_ok=True)
    cap = GDCapture()
    fulls: list[np.ndarray] = []
    full_ts: list[float] = []
    obss: list[np.ndarray] = []
    obs_ts: list[float] = []
    try:
        cap.start()
        for _ in range(10):
            cap.read()
        t0 = time.perf_counter()
        next_full = t0
        while time.perf_counter() - t0 < args.seconds:
            obs = cap.read()
            now = time.perf_counter()
            obss.append(obs.copy())
            obs_ts.append(now - t0)
            if now >= next_full:
                full = cap.last_fullres
                if full is not None:
                    fulls.append(full.copy())
                    full_ts.append(now - t0)
                next_full += 1.0 / args.frame_hz
    finally:
        cap.stop()
    np.savez_compressed(os.path.join(args.out, "frames.npz"),
                        frames=np.stack(fulls) if fulls else np.zeros((0, 1, 1, 3), np.uint8),
                        ts=np.asarray(full_ts))
    np.savez_compressed(os.path.join(args.out, "obs.npz"),
                        obs=np.stack(obss) if obss else np.zeros((0, 4, 180, 320, 3), np.uint8),
                        ts=np.asarray(obs_ts))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"seconds": args.seconds, "frame_hz": args.frame_hz,
                   "n_full": len(fulls), "n_obs": len(obss)}, f, indent=2)
    with open(os.path.join(args.out, "labels.json"), "w") as f:
        json.dump({"deaths": [], "pct": {},
                   "help": "deaths: list of tick indices (60Hz obs clock). "
                           "pct: {second:int 0..100} at 1s intervals. "
                           "Use label.py to fill these fast."}, f, indent=2)
    print(f"saved {len(fulls)} full + {len(obss)} obs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
