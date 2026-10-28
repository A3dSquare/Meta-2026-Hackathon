"""PHASE 2 template capture: dump candidate digit cells during play.

Usage: python pct_capture.py --out ./candidates --frames 600
Plays through the live feed, runs the same mask+contour segmentation as
PctReader, saves each cell (20x28) to <out>/cell_<tick>_<i>.png. The user
then picks one clean example per glyph and copies to
./templates/d0.png..d9.png + pct.png. No guessing: templates come from
the user's own screen/font/resolution.
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="./candidates")
    p.add_argument("--frames", type=int, default=600)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from capture import GDCapture
    from pct import PctConfig, PctReader
    cfg = PctConfig.load()
    cap = GDCapture()
    n = 0
    try:
        cap.start()
        for _ in range(10):
            cap.read()
        for t in range(args.frames):
            cap.read()
            full = cap.last_fullres
            if full is None:
                continue
            rdr = PctReader(cfg)
            crop = rdr._crop(np.asarray(full))
            if crop is None:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask = (((hsv[:, :, 1] <= cfg.sat_max)
                     & (hsv[:, :, 2] >= cfg.val_min)).astype(np.uint8) * 255)
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            ch = mask.shape[0]
            boxes = []
            for c in cnts:
                x, y, w, hh = cv2.boundingRect(c)
                if hh >= ch * cfg.min_h_frac and cv2.contourArea(c) >= cfg.min_area:
                    boxes.append((x, y, w, hh))
            boxes.sort()
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            for i, (x, y, w, hh) in enumerate(boxes):
                cell = gray[y:y + hh, x:x + w]
                cell_n = cv2.resize(cell, (20, 28), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(args.out, f"cell_{t:05d}_{i}.png"), cell_n)
                n += 1
    finally:
        cap.stop()
    print(f"saved {n} candidate cells -> {args.out}/ (pick d0..d9 + pct into ./templates/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
