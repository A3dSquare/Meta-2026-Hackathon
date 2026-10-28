"""Spike-detector benchmark on synthetic GD frames (no game needed).

    python test_detect.py            # new detector
    python test_detect.py --old PATH # also score an old detect.py for comparison
    python test_detect.py --dump     # write test_detect_*.png overlays

Reports recall per hazard type (regular / small / mini / ceiling spikes,
blocks) and false-positive spikes per frame.
"""
from __future__ import annotations

import argparse
import importlib.util
import time

import cv2
import numpy as np

import synth_frames as sf


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _hit(gt, e, pad=0.25) -> bool:
    gx0, gy0, gx1, gy1 = gt.x0, gt.y0, gt.x1, gt.y1
    gw, gh = gx1 - gx0, gy1 - gy0
    ex0, ex1 = e.x - e.w / 2, e.x + e.w / 2
    ey0, ey1 = e.y - e.h / 2, e.y + e.h / 2
    gcx, gcy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
    in_det = ex0 - 0.004 <= gcx <= ex1 + 0.004 and ey0 - 0.006 <= gcy <= ey1 + 0.006
    in_gt = gx0 - pad * gw <= e.x <= gx1 + pad * gw and gy0 - pad * gh <= e.y <= gy1 + pad * gh
    return in_gt and (in_det or (e.w < 2.5 * gw))


def score(mod, n_frames: int, use_hires: bool, dump: bool = False, tag: str = "", hard: bool = False):
    kinds = ["regular", "small", "mini", "ceiling", "block"]
    tot = {k: 0 for k in kinds}
    hit = {k: 0 for k in kinds}
    fp = 0
    ms = []
    for seed in range(n_frames):
        full, gts = sf.make_frame(seed, hard)
        rgb320, bgr640 = sf.downscale(full)
        if hasattr(mod, "_PL_LAST"):
            mod._PL_LAST[0] = None
            mod._PL_HOLD[0] = 0
        det = mod.HeuristicDetector()
        t0 = time.perf_counter()
        if use_hires:
            det(rgb320, bgr640)
            ents = det(rgb320, bgr640)   # 2nd frame: temporal confirmation
        else:
            det(rgb320)
            ents = det(rgb320)
        ms.append((time.perf_counter() - t0) * 500.0)
        spikes = [e for e in ents.items if int(e.cls) == 0]
        walls = [e for e in ents.items if int(e.cls) == 6]
        for g in gts:
            tot[g.kind] += 1
            pool = walls if g.kind == "block" else spikes
            if any(_hit(g, e) for e in pool):
                hit[g.kind] += 1
        spike_gts = [g for g in gts if g.kind != "block"]
        for e in spikes:
            if not any(_hit(g, e, pad=0.5) for g in spike_gts):
                fp += 1
        if dump and seed < 6:
            vis = cv2.resize(full, (1280, 720), interpolation=cv2.INTER_AREA)
            for g in gts:
                cv2.rectangle(vis, (int(g.x0 * 1280), int(g.y0 * 720)), (int(g.x1 * 1280), int(g.y1 * 720)), (0, 255, 0), 1)
            for e in spikes + walls:
                c = (0, 0, 255) if e.cls == 0 else (200, 200, 200)
                cv2.rectangle(vis, (int((e.x - e.w / 2) * 1280), int((e.y - e.h / 2) * 720)),
                              (int((e.x + e.w / 2) * 1280), int((e.y + e.h / 2) * 720)), c, 2)
            cv2.imwrite(f"test_detect_{tag}{seed}.png", vis)
    return tot, hit, fp, float(np.mean(ms))


def report(name, tot, hit, fp, ms, n):
    print(f"\n== {name}  ({n} frames, {ms:.1f} ms/frame)")
    for k in tot:
        if tot[k]:
            print(f"  {k:8s} recall {hit[k]:4d}/{tot[k]:<4d} = {100.0 * hit[k] / tot[k]:5.1f}%")
    print(f"  false-positive spikes: {fp}  ({fp / n:.2f}/frame)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--old", default=None, help="path to an old detect.py to compare")
    p.add_argument("--dump", action="store_true")
    p.add_argument("--hard", action="store_true", help="random level colours, glow, particles")
    a = p.parse_args()
    if a.old:
        old = _load(a.old, "detect_old")
        report("OLD detector (320x180)", *score(old, a.frames, False, a.dump, "old", a.hard), a.frames)
    new = _load("detect.py", "detect_new")
    report("NEW detector (320x180 only)", *score(new, a.frames, False, hard=a.hard), a.frames)
    report("NEW detector (640x360 hires, what training uses)", *score(new, a.frames, True, a.dump, "new", a.hard), a.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
