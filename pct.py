"""PHASE 2 -- text-first percent reader: fixed ROI + template matching.

Pipeline (every 60Hz tick, ROI is tiny so cost is <<2ms):
  1. Fixed ROI from config file (pct_roi.json, fractions of full-res),
     written once by `python pct.py --calibrate`. NO per-frame auto-locate.
  2. White-glyph mask (HSV low saturation + high value), dilate 3x3 once
     so the % sign's two dots merge, connected components left->right,
     height >= 50% of ROI + min area filter.
  3. Resize each cell to 20x28, cv2.matchTemplate vs d0..d9 + pct from
     ./templates/. ANY cell below match_thresh (0.6) poisons the WHOLE
     read -> return (None, False, age+1). Wrong digit is worse than no read.
  4. Temporal gate: accept only if 0..100 AND (within 3.0 of last accepted
     OR confirmed by 2 consecutive identical reads). Isolated jumps
     (17 -> 71 -> 18) return stale (last accepted, False).
  5. Fresh/stale: read() returns (value, fresh, age_ticks). fresh=True only
     on an accepted decode this tick. Downstream (gd.py) uses fresh, never
     a silently-held cache. tick() is 60Hz: re-reads every frame.
  6. Coast bug fixed: coast/unlock measured in TICKS (30f = 0.5s at 60Hz),
     never in calls to a 1Hz function. No time-based staleness anywhere.

Calibration:
  `python pct.py --calibrate` grabs a full-res frame, opens an OpenCV
  window: drag a box around the xx% text (wide enough for 100%), ENTER
  saves to pct_roi.json, Q quits without saving.
Template capture:
  `python pct_capture.py` saves candidate digit cells during play; the
  user picks one clean example per glyph into ./templates/d0.png..d9,pct.
Bar-end (_bar_end_frac) is DELETED: eval will show it adds nothing once
text works. If text MAE stays bad, the numbers will say so and the next
step is a small digit CNN, not more heuristics.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("opencv is required: pip install opencv-python") from e

ROI_PATH = "./pct_roi.json"


@dataclass
class PctConfig:
    """Fixed ROI fractions of the FULL-RES frame, loaded from pct_roi.json."""

    x0: float = 0.725
    y0: float = 0.0
    x1: float = 0.795
    y1: float = 0.055
    sat_max: int = 80
    val_min: int = 170
    min_h_frac: float = 0.35
    min_area: int = 60
    match_thresh: float = 0.60
    step_tol: float = 3.0
    confirm_n: int = 2
    template_dir: str = "./templates"
    roi_path: str = ROI_PATH

    @classmethod
    def load(cls, path: str = ROI_PATH) -> "PctConfig":
        cfg = cls()
        if os.path.exists(path):
            try:
                d = json.load(open(path))
                for k in ("x0", "y0", "x1", "y1"):
                    if k in d:
                        setattr(cfg, k, float(d[k]))
            except Exception:
                pass
        return cfg

    def save(self, path: str = ROI_PATH) -> None:
        json.dump({"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1},
                  open(path, "w"), indent=2)


class PctReader:
    """60Hz text-first reader: read() -> (value|None, fresh, age_ticks)."""

    CELL_W, CELL_H = 20, 28

    def __init__(self, config: PctConfig | None = None) -> None:
        self.cfg = config or PctConfig.load()
        self._templates: dict[str, np.ndarray] = {}
        self._cnn = None  # lazy torch TinyDigitCNN (same arch as train_digit_cnn)
        self._cnn_ok: bool = False
        self._last_accept: Optional[float] = None
        self._age: int = 0  # ticks since last accepted read
        self._pending_val: Optional[float] = None
        self._pending_n: int = 0
        self._last_raw: Optional[float] = None
        self._reads = 0
        self._hits = 0
        self._last_ms: float = 0.0

    @property
    def last(self) -> Optional[float]:
        return self._last_accept

    @property
    def health(self) -> dict:
        return {"last": self._last_accept, "raw": self._last_raw,
                "age": self._age, "reads": self._reads, "hits": self._hits,
                "hit_rate": self._hits / max(1, self._reads),
                "n_templates": len(self._templates),
                "cnn": self._cnn_ok}

    def load_cnn(self, path: str = "./digit_cnn.pt") -> bool:
        """DISABLED per eval: CNN regressed MAE 0.83->6.62 (overconfident on
        jumper font). Templates-only wins. Kept as no-op for API compat;
        always returns False so _classify_cell uses templates."""
        self._cnn = None
        self._cnn_ok = False
        return False

    def _classify_cell(self, cell_n: np.ndarray) -> Optional[str]:
        """Template matching (CNN disabled: regressed eval, see load_cnn)."""
        best, best_s = None, self.cfg.match_thresh
        for name, tmpl in self._templates.items():
            rr = cv2.matchTemplate(cell_n, tmpl, cv2.TM_CCOEFF_NORMED)
            s = float(rr.max())
            if s > best_s:
                best, best_s = name, s
        if best is None:
            return None
        return "%" if best == "pct" else best[1]

    def load_templates(self) -> int:
        self._templates.clear()
        for name in [f"d{i}" for i in range(10)] + ["pct"]:
            for ext in (".png", ".bmp", ".pgm"):
                pth = os.path.join(self.cfg.template_dir, name + ext)
                if os.path.exists(pth):
                    img = cv2.imread(pth, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        if img.shape != (self.CELL_H, self.CELL_W):
                            img = cv2.resize(img, (self.CELL_W, self.CELL_H),
                                             interpolation=cv2.INTER_AREA)
                        self._templates[name] = img
                    break
        return len(self._templates)

    def _crop(self, full_bgr: np.ndarray) -> np.ndarray | None:
        h, w = full_bgr.shape[:2]
        x0, y0 = int(w * self.cfg.x0), int(h * self.cfg.y0)
        x1, y1 = int(w * self.cfg.x1), int(h * self.cfg.y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return full_bgr[y0:y1, x0:x1]

    def _decode(self, full_bgr: np.ndarray) -> Optional[float]:
        """Raw template decode, no temporal logic. None = unreadable."""
        crop = self._crop(np.asarray(full_bgr))
        if crop is None or crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = (((hsv[:, :, 1] <= self.cfg.sat_max)
                 & (hsv[:, :, 2] >= self.cfg.val_min)).astype(np.uint8) * 255)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        ch = mask.shape[0]
        boxes = []
        for c in cnts:
            x, y, w, hh = cv2.boundingRect(c)
            if hh >= ch * self.cfg.min_h_frac and cv2.contourArea(c) >= self.cfg.min_area:
                boxes.append((x, y, w, hh))
        if not boxes:
            return None
        # SPLIT wide merged blobs: double digits (e.g. "35") dilate into one
        # contour. A box wider than 1.6x its height is two glyphs side by
        # side -- split at the darkest vertical seam. Fixes double-digit
        # misses without touching single digits.
        split: list[tuple[int, int, int, int]] = []
        gray0 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        for (x, y, w, hh) in boxes:
            if w > int(hh * 1.6) and w >= 8:
                col = gray0[y:y + hh, x:x + w].mean(axis=0)
                lo, hi = int(w * 0.3), max(int(w * 0.3) + 1, int(w * 0.7))
                seam = lo + int(np.argmin(col[lo:hi]))
                split.append((x, y, seam, hh))
                split.append((x + seam, y, w - seam, hh))
            else:
                split.append((x, y, w, hh))
        boxes = [b for b in split if b[2] >= 3 and b[3] >= 4]
        boxes.sort()
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        chars: list[str] = []
        for (x, y, w, hh) in boxes:
            cell = gray[y:y + hh, x:x + w]
            if cell.shape[0] < 4 or cell.shape[1] < 4:
                return None
            cell_n = cv2.resize(cell, (self.CELL_W, self.CELL_H),
                                interpolation=cv2.INTER_AREA)
            c = self._classify_cell(cell_n)
            if c is None:
                return None  # ANY cell below threshold poisons the read
            chars.append(c)
        num = "".join(c for c in chars if c.isdigit())
        if not num:
            return None
        try:
            v = min(100, max(0, int(num))) / 100.0
        except ValueError:
            return None
        return v

    def read(self, full_bgr: np.ndarray | None) -> tuple[Optional[float], bool, int]:
        """60Hz read -> (accepted_value|None, fresh, age_ticks).

        fresh=True only when THIS tick decoded + passed the temporal gate.
        Otherwise returns (last_accepted, False, age). age counts ticks
        since the last accepted read (tick-based coast, never time-based).
        """
        t0 = time.perf_counter()
        self._reads += 1
        raw = self._decode(full_bgr) if full_bgr is not None else None
        self._last_raw = raw
        self._last_ms = (time.perf_counter() - t0) * 1000.0
        if raw is not None and 0.0 <= raw <= 1.0:
            # ZERO OVERRIDE: a clean 0% decode is never a template glitch
            # (glitches read as wrong digits / None, not exact zero). Bypass
            # the jump gate so the 8%-to-0% reset edge is fresh THIS tick,
            # not parked in pending for a confirm that may never come live.
            if raw <= 1e-9 and self._last_accept is not None and self._last_accept > 1e-9:
                self._last_accept = raw
                self._age = 0
                self._pending_val, self._pending_n = None, 0
                self._hits += 1
                return raw, True, 0
            if self._last_accept is None or abs(raw - self._last_accept) <= self.cfg.step_tol / 100.0:
                self._last_accept = raw
                self._age = 0
                self._pending_val, self._pending_n = None, 0
                self._hits += 1
                return raw, True, 0
            if self._pending_val is not None and abs(raw - self._pending_val) < 1e-9:
                self._pending_n += 1
            else:
                self._pending_val, self._pending_n = raw, 1
            if self._pending_n >= self.cfg.confirm_n:
                self._last_accept = raw
                self._age = 0
                self._pending_val, self._pending_n = None, 0
                self._hits += 1
                return raw, True, 0
        self._age += 1
        return self._last_accept, False, self._age

    def tick(self, full_bgr: np.ndarray | None) -> tuple[Optional[float], bool, int]:
        """60Hz alias for read(). No 1Hz cadence: ROI is tiny, cost <<2ms."""
        return self.read(full_bgr)

    def calibrate_dump(self, full_bgr: np.ndarray | None) -> str:
        """Save crop + white mask for ROI debugging (non-interactive)."""
        path = "./pct_calib.png"
        blank = np.zeros((120, 480, 3), dtype=np.uint8)
        if full_bgr is None:
            cv2.imwrite(path, blank)
            return path
        crop = self._crop(np.asarray(full_bgr))
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = (((hsv[:, :, 1] <= self.cfg.sat_max)
                 & (hsv[:, :, 2] >= self.cfg.val_min)).astype(np.uint8) * 255)
        ch, cw = crop.shape[:2]
        canvas = np.zeros((ch + 8, cw * 2 + 8, 3), dtype=np.uint8)
        canvas[:ch, :cw] = crop
        canvas[:ch, cw + 8:, 0] = cv2.resize(mask, (cw, ch))
        canvas[:ch, cw + 8:, 1] = canvas[:ch, cw + 8:, 0]
        canvas[:ch, cw + 8:, 2] = canvas[:ch, cw + 8:, 0]
        cv2.imwrite(path, canvas)
        return path


def _interactive_calibrate() -> int:
    """Drag a box around the xx% text; ENTER saves ROI to pct_roi.json."""
    from capture import GDCapture
    cap = GDCapture()
    try:
        cap.start()
        for _ in range(10):
            cap.read()
        full = cap.last_fullres
        if full is None:
            print("no frame")
            return 1
        img = full.copy()
        r = cv2.selectROI("drag around xx% text (wide enough for 100%), ENTER=save Q=cancel",
                          img, showCrosshair=True)
        cv2.destroyAllWindows()
        x, y, w, h = (int(v) for v in r)
        if w < 8 or h < 8:
            print("cancelled")
            return 1
        H, W = img.shape[:2]
        cfg = PctConfig.load()
        cfg.x0, cfg.y0, cfg.x1, cfg.y1 = x / W, y / H, (x + w) / W, (y + h) / H
        cfg.save()
        print(f"saved ROI x0={cfg.x0:.4f} y0={cfg.y0:.4f} x1={cfg.x1:.4f} y1={cfg.y1:.4f} -> pct_roi.json")
        return 0
    finally:
        cap.stop()


def main() -> int:
    p = argparse.ArgumentParser(description="pct text-first calibration")
    p.add_argument("--calibrate", action="store_true",
                   help="interactive ROI drag, saves pct_roi.json")
    p.add_argument("--dump", action="store_true",
                   help="non-interactive crop+mask dump to pct_calib.png")
    args = p.parse_args()
    if args.calibrate:
        return _interactive_calibrate()
    if args.dump:
        from capture import GDCapture
        cap = GDCapture()
        try:
            cap.start()
            for _ in range(10):
                cap.read()
            full = cap.last_fullres
            path = PctReader().calibrate_dump(full.copy() if full is not None else None)
            print(f"dump={path}")
            return 0
        finally:
            cap.stop()
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
