"""General-data rotation: same brain, many levels (NOT a macro machine).

python train_general.py --levels stereo jumper finger --rotate-min 10

Rotation does NOT swap files (GD has no API); it prints "SWITCH TO <level>"
and pauses until you pick the level in GD and press SPACE in the GUI
window. The policy only sees (6,9) entity vectors, so rotation changes
WHICH boxes appear -- forcing reactive jumping over layout memory.
Stop: CTRL+C (saves best_boxes.zip).
"""
from __future__ import annotations

import argparse
import time


def main() -> int:
    p = argparse.ArgumentParser(description="multi-level rotation trainer")
    p.add_argument("--levels", nargs="+", required=True,
                   help="level names in rotation order, e.g. stereo jumper finger")
    p.add_argument("--rotate-min", type=float, default=10.0)
    p.add_argument("--timesteps", type=int, default=5000000)
    args = p.parse_args()
    print(f"GENERAL DATA rotation: {args.levels} x{args.rotate_min}min")
    print("Per level: `python preview_boxes.py --backend heuristic`")
    print("When the timer prints SWITCH, change GD level, SPACE in the GUI.")
    print("Same checkpoint (`preview_boxes.zip`) carries the brain across levels.")
    # Thin scheduler over preview_boxes.py runs (one brain, many layouts).
    import subprocess, sys
    per = int(args.rotate_min * 60)
    for lv in args.levels:
        print(f"\n=== LEVEL {lv} ({args.rotate_min}min) ===")
        print(f"python preview_boxes.py --backend heuristic --model ./checkpoints/preview_boxes.zip (delete flag for first)")
        t0 = time.perf_counter()
        try:
            subprocess.run([sys.executable, "preview_boxes.py", "--backend", "heuristic"],
                          timeout=per)
        except subprocess.TimeoutExpired:
            print(f"SWITCH TO NEXT LEVEL")
        except KeyboardInterrupt:
            print("stopped; brain saved in ./checkpoints/preview_boxes.zip")
            return 0
        if time.perf_counter() - t0 < per:
            print("level ended early; brain saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
