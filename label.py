"""Frame-indexed label viewer: SPACE=death, digits+ENTER=pct, FAST stepping.

Usage: python label.py --rec recs/jumper.mp4
Keys: A/D or arrows = step 1 REAL frame (cached, instant), W/S = jump 10,
SPACE = death at real frame idx, 0-9 = type 1-3 digit percent, ENTER =
confirm pct for this frame, BACKSPACE = cancel digits, X = delete pct
label, N = next hard frame, ENTER (no digits) = save, Q = save+quit.

Speed fix (no scoring change): sequential forward decode with a 128-frame
ring cache — A/D stepping is a dict lookup (~0ms), seeks only on jumps or
direction reversals. Reader overlay comes from the cached .trace.npz when
present (no live template matching in the UI thread); falls back to live
PctReader only when no trace exists.
Format: {"deaths": [frame_idx...], "pct": {"<frame_idx>": int},
"stride": 1, "fps": 60.0}.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np


class FastMP4:
    """Sequential decode + ring cache. Instant A/D stepping, seek only on jumps."""

    def __init__(self, path: str, window: int = 128):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 60.0
        self.window = window
        self._cache: dict[int, np.ndarray] = {}
        self._pos = -1  # last decoded frame idx
        self._dir = 1

    def _decode_to(self, target: int) -> bool:
        target = max(0, min(self.n - 1, target))
        if target in self._cache:
            return True
        # sequential forward: cheap. seek only on jump/backwards.
        if target < self._pos or target > self._pos + 1:
            # jump: seek to nearest, then decode forward
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self._pos = target - 1
        while self._pos < target:
            ok, fr = self.cap.read()
            if not ok:
                return False
            self._pos += 1
            self._cache[self._pos] = fr
            if len(self._cache) > self.window:
                old = min(self._cache.keys())
                self._cache.pop(old, None)
        return target in self._cache

    def get(self, fi: int) -> np.ndarray | None:
        fi = max(0, min(self.n - 1, fi))
        if not self._decode_to(fi):
            return None
        return self._cache[fi].copy()

    def close(self):
        self.cap.release()


def _migrate_old_labels(lab: dict, fps: float) -> tuple[dict, bool]:
    pct = lab.get("pct", {})
    if not pct:
        return lab, False
    keys = list(pct.keys())
    if all(k.isdigit() and int(k) < 10000 and len(str(k)) <= 3 for k in keys):
        new_pct = {str(int(int(k) * fps)): int(v) for k, v in pct.items()}
        lab["pct"] = new_pct
        lab["stride"] = 1
        lab["fps"] = fps
        return lab, True
    return lab, False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rec", required=True, help="mp4 file, e.g. recs/jumper.mp4")
    args = p.parse_args()
    if not args.rec.lower().endswith(".mp4"):
        print("mp4 only in this version: python label.py --rec recs/jumper.mp4")
        return 2
    lab_path = os.path.splitext(args.rec)[0] + ".labels.json"
    trace_path = os.path.splitext(args.rec)[0] + ".trace.npz"
    vid = FastMP4(args.rec)
    lab: dict = {"deaths": [], "pct": {}, "stride": 1, "fps": vid.fps}
    if os.path.exists(lab_path):
        try:
            lab = json.load(open(lab_path))
        except Exception:
            pass
    lab.setdefault("deaths", [])
    lab.setdefault("pct", {})
    lab, migrated = _migrate_old_labels(lab, vid.fps)
    if migrated:
        print(f"migrated old per-second pct to frame idx in {lab_path}")
        json.dump(lab, open(lab_path, "w"), indent=2)
    deaths: list[int] = sorted(lab.get("deaths", []))
    pct: dict[str, int] = {str(k): int(v) for k, v in lab.get("pct", {}).items()}

    # reader overlay: trace cache first (instant), live PctReader fallback
    trace = None
    if os.path.exists(trace_path):
        try:
            trace = np.load(trace_path)
            print(f"overlay from trace {trace_path} (no live matching)")
        except Exception:
            trace = None
    rdr = None
    if trace is None:
        try:
            from pct import PctReader
            rdr = PctReader()
            rdr.load_templates()
        except Exception:
            rdr = None
    reader_cache: dict[int, tuple] = {}

    def reader_out(fi: int):
        if fi in reader_cache:
            return reader_cache[fi]
        if trace is not None:
            try:
                pv = trace["pct"][fi]
                v = None if pv != pv else float(pv)
                out = (v, bool(trace["fresh"][fi]), int(trace["age"][fi]))
                reader_cache[fi] = out
                return out
            except Exception:
                pass
        if rdr is None:
            return (None, False, 999)
        fr = vid.get(fi)
        if fr is None:
            return (None, False, 999)
        out = rdr.read(fr)
        reader_cache[fi] = out
        if len(reader_cache) > 256:
            reader_cache.pop(next(iter(reader_cache)))
        return out

    def is_hard(fi: int) -> bool:
        v, fresh, age = reader_out(fi)
        key = str(fi)
        if v is None:
            return True
        if key in pct and abs(float(v) * 100 - pct[key]) > 3:
            return True
        return False

    i = 0
    digits = ""
    win = "label: A/D 1f W/S 10f SPACE death digits+ENTER pct BKSP cancel X del N hard Q quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        vis = vid.get(i)
        if vis is None:
            i = max(0, min(vid.n - 1, i))
            continue
        vis = vis.copy()
        v, fresh, age = reader_out(i)
        lbl = pct.get(str(i), "?")
        cv2.putText(vis, f"frame {i}/{vid.n} deaths={len(deaths)} pct[{i}]={lbl} "
                    f"reader={'-' if v is None else round(float(v)*100,1)}"
                    f"{'F' if fresh else 'S'}{age} digits:{digits or '-'}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow(win, vis)
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), ord("Q")):
            break
        elif k in (ord("d"), ord("D"), 83):
            i = min(vid.n - 1, i + 1)
        elif k in (ord("a"), ord("A"), 81):
            i = max(0, i - 1)
        elif k in (ord("w"), ord("W"), 82):
            i = min(vid.n - 1, i + 10)
        elif k in (ord("s"), ord("S"), 84):
            i = max(0, i - 10)
        elif k == 32:  # SPACE
            if i not in deaths:
                deaths.append(i)
                deaths.sort()
            print(f"death @ real frame {i}")
        elif ord("0") <= k <= ord("9"):
            if len(digits) < 3:
                digits += chr(k)
        elif k in (8, 127):  # BACKSPACE
            digits = digits[:-1]
        elif k in (ord("x"), ord("X")):
            if str(i) in pct:
                del pct[str(i)]
                print(f"deleted pct @{i}")
        elif k in (ord("n"), ord("N")):
            j = i + 1
            while j < vid.n and not is_hard(j):
                j += 1
                if j % 500 == 0:
                    print(f"scanning hard... {j}/{vid.n}", end="\r")
            if j < vid.n:
                i = j
                print(f"hard frame {i}")
            else:
                print("no more hard frames")
            continue
        elif k in (13, 10):  # ENTER
            if digits != "":
                v_ = int(digits)
                if 0 <= v_ <= 100:
                    pct[str(i)] = v_
                    print(f"pct @{i} = {v_}")
                digits = ""
            else:
                lab_out = {"deaths": sorted(deaths), "pct": pct, "stride": 1, "fps": vid.fps}
                json.dump(lab_out, open(lab_path, "w"), indent=2)
                print(f"saved {lab_path}")
            continue
    lab_out = {"deaths": sorted(deaths), "pct": pct, "stride": 1, "fps": vid.fps}
    json.dump(lab_out, open(lab_path, "w"), indent=2)
    print(f"saved {lab_path}: {len(deaths)} deaths, {len(pct)} pct labels")
    vid.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
