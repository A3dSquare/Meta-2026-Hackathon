"""General-data rotation: same brain, many levels (NOT a macro machine).

python train_general.py --levels stereo jumper finger --rotate-min 10

Rotation does NOT swap levels for you (GD has no API): it starts
preview_boxes.py, and when the timer runs out it stops it and prints
"SWITCH TO <level>". Open that level in GD (Auto-Retry ON), then press
ENTER here. The policy only sees entity boxes, so rotation changes WHICH
boxes appear -- forcing reactive jumping over layout memory.

Every run resumes ./checkpoints/preview_boxes.pt (the previous version
started each level from a blank brain because it never passed --model).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

CKPT = "./checkpoints/preview_boxes.pt"


def main() -> int:
    p = argparse.ArgumentParser(description="multi-level rotation trainer")
    p.add_argument("--levels", nargs="+", required=True,
                   help="level names in rotation order, e.g. stereo jumper finger")
    p.add_argument("--rotate-min", type=float, default=10.0)
    p.add_argument("--rounds", type=int, default=3, help="times through the level list")
    p.add_argument("--eps", type=float, default=0.05,
                   help="exploration when starting a level with an existing brain")
    args = p.parse_args()
    per = int(args.rotate_min * 60)
    for rnd in range(args.rounds):
        for lv in args.levels:
            print(f"\n=== round {rnd + 1}: LEVEL {lv} ({args.rotate_min} min) ===")
            input(f"Open '{lv}' in GD (Auto-Retry ON), click into the game window, then press ENTER here... ")
            cmd = [sys.executable, "preview_boxes.py"]
            if os.path.exists(CKPT):
                cmd += ["--model", CKPT, "--eps", str(args.eps)]
            try:
                subprocess.run(cmd, timeout=per)
            except subprocess.TimeoutExpired:
                print("time up -> SWITCH TO NEXT LEVEL (brain is saved after every update)")
            except KeyboardInterrupt:
                print(f"stopped; brain saved in {CKPT}")
                return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
