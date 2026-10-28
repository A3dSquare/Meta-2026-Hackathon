"""Entity detector for the cross-level box policy (pixels -> boxes, no SAM).

Pipeline:
  320x180 RGB (+ optional 640x360 BGR "hires") --detect--> Entities
  -> env_boxes.encode_obs() -> PPO MlpPolicy -> SPACE.

Classes (level-agnostic, same on every GD level):
  0 spike (any hazard triangle: regular, small, mini, ceiling, rows of them)
  1 orb-yellow, 2 orb-blue, 3 orb-pink, 4 pad, 5 portal, 6 wall/block,
  7 player

HAZARDS (spikes + blocks) -- v2, profile based:
  The old detector asked OpenCV for "a contour with exactly 3 corners and an
  aspect ratio of 0.5-1.8" on the 320x180 frame. That misses:
    * small spikes (twice as wide as tall -> aspect ~2.2, rejected),
    * mini spikes (6-10 px at 320x180, eroded away by the 2x2 opening),
    * double/triple spikes (touching triangles merge into one contour that
      is neither a triangle nor in the aspect range),
    * spikes standing on blocks (merge with the block's black body).
  v2 runs at 640x360 when the full-res frame is available and, for every
  dark (and every bright-outline) blob, walks its TOP edge column by column.
  A spike is a peak in that edge profile: a short apex with a slope on BOTH
  sides that rises at most a few px per column. That rule is size-free (mini,
  small and regular spikes all pass), finds each peak of a merged double /
  triple row, finds spikes standing on blocks (peak on a flat top), and
  rejects blocks (flat top + vertical sides) and slopes (one vertical side).
  The same walk on the BOTTOM edge finds ceiling spikes. Blobs that are mostly
  solid rectangles become class 6 (wall) so the policy can see platforms.
  A small tracker confirms hazards over 2 frames and coasts through a single
  missed frame, so boxes neither flash nor flicker.

Backends:
  --backend heuristic (default, zero deps, ~3-5 ms CPU per decision)
  --backend yolo (opt-in, see detect_yolo.py)

Usage:
  python detect.py --probe recs/stereo.mp4   # timing + per-frame hazard counts
  python detect.py --probe recs/stereo.mp4 --show   # watch the boxes
  python detect.py                             # one live frame from the game
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

CLASSES = ["spike", "orb_y", "orb_b", "orb_p", "pad", "portal", "wall", "player"]
CLS_ID = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

# GD orb hues (OpenCV H 0-179): yellow ~25, blue ~105, pink ~140.
ORB_HUES = {"orb_y": (20, 32), "orb_b": (95, 115), "orb_p": (130, 155)}

# One GD block is ~1/10.7 of the screen height (320pt view, 30pt blocks).
BLOCK_FRAC_H = 30.0 / 320.0

# Player-tracker state lives at MODULE level so every detector in the
# process (GUI probe + env) agrees on the cube position.
_PL_LAST: list = [None]  # [(x, y)] last accepted cube center
_PL_HOLD: list = [0]     # [ticks] held-through-miss counter


@dataclass
class Entity:
    cls: int
    x: float  # center x, 0..1
    y: float  # center y, 0..1
    w: float  # width 0..1
    h: float  # height 0..1
    conf: float = 1.0


@dataclass
class Entities:
    items: list = field(default_factory=list)
    # median x-displacement of tracked hazards since the previous call
    # (normalized width; ~-0.017 per 2 frames at 1x speed, ~0 when the camera
    # is frozen by a death). None when nothing could be matched.
    scroll: float | None = None

    def player(self):
        for e in self.items:
            if int(e.cls) == CLS_ID["player"]:
                return e
        return None

    def player_pos(self) -> tuple[float, float]:
        """Player cube center; (0.28, 0.70) when the cube is not found."""
        p = self.player()
        if p is not None:
            return (float(p.x), float(p.y))
        return (0.28, 0.70)

    def nearest(self, k: int = 6, hold: bool = False) -> np.ndarray:
        """LEGACY (6,9) encoding kept for old scripts. New code: env_boxes.encode_obs."""
        px, py = self.player_pos()
        ahead = [e for e in self.items if int(e.cls) != 7 and e.x >= px - 0.02]
        ahead.sort(key=lambda e: e.x)
        out = np.zeros((k, 9), dtype=np.float32)
        prev_dx = 0.0
        for i, e in enumerate(ahead[:k]):
            dx = e.x - px
            out[i] = (dx, e.y - py, e.w, e.h, float(e.cls) / 7.0,
                      dx - (e.w / 2 + 0.02), prev_dx - dx, 1.0 if hold else 0.0, py)
            prev_dx = dx
        return out


# --------------------------------------------------------------------------- #
# Hazard geometry: peaks in a blob's top / bottom edge profile.
# --------------------------------------------------------------------------- #
def _runs(prof: np.ndarray, a: int, b: int):
    """Compress prof[a:b] into runs of equal value: list of [value, start, end]."""
    runs = []
    s = a
    for i in range(a + 1, b + 1):
        if i == b or prof[i] != prof[s]:
            runs.append([int(prof[s]), s, i - 1])
            s = i
    return runs


def profile_peaks(prof: np.ndarray, valid: np.ndarray, *, max_step: int, min_h: int,
                  min_w: int, max_flat: int) -> list:
    """Find spike apexes in an edge profile.

    prof[c] = y of the blob's top edge in column c (smaller = higher on
    screen). Returns tuples (left_col, right_col, apex_y, base_y).

    A peak is a run (the apex, at most a few px wide) whose neighbours on both
    sides are lower; from it we walk outward while the edge keeps descending
    by <= max_step px per column and no flat stretch is wider than max_flat.
    Blocks fail (their sides are vertical: one step >> max_step), ramps fail
    (one side is vertical), flat tops fail (no apex).
    """
    out = []
    n = len(prof)
    c = 0
    while c < n:
        if not valid[c]:
            c += 1
            continue
        a = c
        while c < n and valid[c]:
            c += 1
        b = c  # segment [a, b)
        if b - a < min_w:
            continue
        runs = _runs(prof, a, b)
        for k in range(1, len(runs) - 1):
            v, s, e = runs[k]
            if not (runs[k - 1][0] > v and runs[k + 1][0] > v):
                continue
            apex_w = e - s + 1
            # walk left
            j = k - 1
            last_l = k
            while j >= 0:
                step = runs[j][0] - runs[last_l][0]
                if step <= 0 or step > max_step:
                    break
                last_l = j
                if runs[j][2] - runs[j][1] + 1 > max_flat:
                    break  # reached a platform: that run is the base
                j -= 1
            # walk right
            j = k + 1
            last_r = k
            while j < len(runs):
                step = runs[j][0] - runs[last_r][0]
                if step <= 0 or step > max_step:
                    break
                last_r = j
                if runs[j][2] - runs[j][1] + 1 > max_flat:
                    break
                j += 1
            if last_l == k or last_r == k:
                continue
            left = runs[last_l][2] if runs[last_l][2] - runs[last_l][1] + 1 > max_flat else runs[last_l][1]
            right = runs[last_r][1] if runs[last_r][2] - runs[last_r][1] + 1 > max_flat else runs[last_r][2]
            base_l, base_r = runs[last_l][0], runs[last_r][0]
            h = min(base_l, base_r) - v
            w = right - left + 1
            if h < min_h or w < min_w:
                continue
            if not (0.22 <= h / w <= 1.7):
                continue
            if apex_w > max(3, 0.4 * w):
                continue
            lw, rw = s - left, right - e
            if min(lw, rw) < max(1, 0.18 * w):
                continue
            out.append((left, right, v, max(base_l, base_r)))
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(1e-9, ua)


def detect_hazards(bgr: np.ndarray, player_box=None, force_bright: bool = False) -> list:
    """Spikes + blocks from one BGR frame (any resolution, 640x360 preferred).

    Returns list of (cls, x0, y0, x1, y1, conf) in normalized coordinates.
    player_box (normalized x0,y0,x1,y1) masks the cube out (a rotated cube
    is a diamond, and a diamond's top edge looks exactly like a spike).
    """
    import cv2

    H, W = bgr.shape[:2]
    blk = BLOCK_FRAC_H * H                         # block size in px
    min_h = max(3, int(round(0.22 * blk)))         # mini spikes are ~0.45 block tall
    min_w = max(4, int(round(0.30 * blk)))         # ... and ~0.55 block wide
    max_step = max(3, int(round(0.20 * blk)))      # steepest spike side ~2px/col @640
    max_flat = max(3, int(round(0.20 * blk)))
    ui_cut = int(0.065 * H)                        # progress bar / % text strip
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V, S = hsv[:, :, 2], hsv[:, :, 1]
    med_v = float(np.median(V[::4, ::4]))
    dark_v = int(min(75, 0.55 * med_v))
    dark = (V < dark_v).astype(np.uint8)
    bright = ((V >= 185) & (S <= 75)).astype(np.uint8)
    dark[:ui_cut] = 0
    bright[:ui_cut] = 0
    if player_box is not None:
        px0, py0, px1, py1 = player_box
        pad_x, pad_y = 0.25 * blk / W, 0.25 * blk / H
        x0 = max(0, int((px0 - pad_x) * W)); x1 = min(W, int((px1 + pad_x) * W) + 1)
        y0 = max(0, int((py0 - pad_y) * H)); y1 = min(H, int((py1 + pad_y) * H) + 1)
        dark[y0:y1, x0:x1] = 0
        bright[y0:y1, x0:x1] = 0

    found: list = []  # (cls, x0, y0, x1, y1, conf) in px
    kw = dict(max_step=max_step, min_h=min_h, min_w=min_w, max_flat=max_flat)

    def _scan(mask, solid_check: bool, conf: float):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        walls = []
        for i in range(1, n):
            x, y, bw, bh, area = (int(v) for v in stats[i])
            if area < max(6, min_h * min_w // 2) or bw * bh > 0.30 * W * H:
                continue
            sub = lab[y:y + bh, x:x + bw] == i
            colv = sub.any(axis=0)
            top = np.where(colv, sub.argmax(axis=0), 10**6) + y
            bot = np.where(colv, bh - 1 - sub[::-1].argmax(axis=0), -1) + y
            spikes_here = []
            for flip, prof in ((False, top), (True, (H - 1) - bot)):
                for (l, r, apex, base) in profile_peaks(prof.astype(np.int64), colv, **kw):
                    if flip:
                        ya, yb = (H - 1) - base, (H - 1) - apex  # hanging spike
                    else:
                        ya, yb = apex, base
                    if (ya + yb) / 2 > 0.94 * H:
                        continue  # bottom strip: ground texture, never a hazard you meet
                    reg = sub[ya - y:yb - y + 1, l:r + 1]
                    if reg.size == 0:
                        continue
                    if solid_check:
                        # pixels between edge and base must be filled (hollow
                        # rings = cube outline / text glyph outlines)
                        if flip:
                            edge = bot[l:r + 1] - y
                            filled = sum(int(reg[: max(0, edge[c] - (ya - y)) + 1, c].sum()) for c in range(r - l + 1))
                            tot = sum(max(0, edge[c] - (ya - y)) + 1 for c in range(r - l + 1))
                        else:
                            edge = top[l:r + 1] - y
                            filled = sum(int(reg[max(0, edge[c] - (ya - y)):, c].sum()) for c in range(r - l + 1))
                            tot = sum(reg.shape[0] - max(0, edge[c] - (ya - y)) for c in range(r - l + 1))
                        if tot <= 0 or filled / tot < 0.70:
                            continue
                    else:
                        # bright outline: the inside of a real spike is dark
                        cx0 = l + (r - l) // 4; cx1 = r - (r - l) // 4 + 1
                        cy0 = ya + (yb - ya) // 3; cy1 = yb - 1
                        inner = V[cy0:cy1, cx0:cx1]
                        if inner.size == 0 or float(inner.mean()) > max(95, 0.75 * med_v):
                            continue
                    spikes_here.append((0, x + l, ya, x + r + 1, yb + 1, conf))
            found.extend(spikes_here)
            if solid_check and bw >= 0.4 * blk and bh >= 0.4 * blk:
                sp_area = sum((s[3] - s[1]) * (s[4] - s[2]) * 0.5 for s in spikes_here)
                if area / float(bw * bh) >= 0.55 and sp_area < 0.5 * area:
                    wy0 = y
                    tops = [s[4] for s in spikes_here if s[2] <= y + 1]
                    if tops:  # spikes standing on the block: wall starts at their base
                        wy0 = max(tops)
                    if y + bh - wy0 >= 0.3 * blk:
                        walls.append((6, x, wy0, x + bw, y + bh, 0.8))
        found.extend(walls)

    _scan(dark, True, 0.9)
    if med_v < 120 or force_bright:
        # dark level: black spike bodies blend into the background, so find
        # them by their bright outline instead
        _scan(bright, False, 0.75)

    # de-duplicate (bright-outline path finds the same spikes again)
    out: list = []
    for d in sorted(found, key=lambda t: -t[5]):
        box = (d[1] / W, d[2] / H, d[3] / W, d[4] / H)
        if any(o[0] == d[0] and _iou(box, o[1:5]) > 0.3 for o in out):
            continue
        out.append((d[0], *box, d[5]))
    return out


class _HazardTracker:
    """2-frame confirmation + 2-frame coasting for hazard boxes.

    A real hazard scrolls left smoothly; a particle / UI flash lives one
    frame. Confirmed tracks keep being emitted (moved by their last scroll
    step) through up to two missed frames, so a box doesn't vanish for one
    frame right before the jump decision.
    """

    def __init__(self) -> None:
        self.tracks: list = []  # dict(cls,x0,y0,x1,y1,conf,hits,miss,vx)
        self.last_scroll: float | None = None

    def reset(self) -> None:
        self.tracks = []
        self.last_scroll = None

    def update(self, dets: list) -> list:
        used = set()
        steps = []
        for t in self.tracks:
            best, bj = None, -1
            cx_t = (t["x0"] + t["x1"]) / 2
            cy_t = (t["y0"] + t["y1"]) / 2
            for j, d in enumerate(dets):
                if j in used or d[0] != t["cls"]:
                    continue
                cx, cy = (d[1] + d[3]) / 2, (d[2] + d[4]) / 2
                dxm = cx - cx_t
                if not (-0.10 * (t["miss"] + 1) <= dxm <= 0.015) or abs(cy - cy_t) > 0.06:
                    continue
                cost = abs(dxm - t["vx"]) + abs(cy - cy_t)
                if best is None or cost < best:
                    best, bj = cost, j
            if bj >= 0:
                d = dets[bj]
                used.add(bj)
                vx = ((d[1] + d[3]) / 2 - cx_t) / (t["miss"] + 1)
                if t["hits"] >= 2 or t["miss"] == 0:
                    steps.append(vx)
                t.update(x0=d[1], y0=d[2], x1=d[3], y1=d[4], conf=d[5], vx=vx,
                         hits=t["hits"] + 1, miss=0)
            else:
                t["miss"] += 1
        self.last_scroll = float(np.median(steps)) if steps else None
        if steps:
            gvx = float(np.median(steps))
            for t in self.tracks:
                if t["miss"] > 0:  # coast with the scene scroll
                    t["x0"] += gvx; t["x1"] += gvx; t["vx"] = gvx
        self.tracks = [t for t in self.tracks if t["miss"] <= 2 and t["x1"] > -0.05]
        for j, d in enumerate(dets):
            if j not in used:
                self.tracks.append(dict(cls=d[0], x0=d[1], y0=d[2], x1=d[3], y1=d[4],
                                        conf=d[5], hits=1, miss=0, vx=-0.01))
        out = []
        for t in self.tracks:
            if t["hits"] >= 2:
                out.append(Entity(int(t["cls"]), (t["x0"] + t["x1"]) / 2, (t["y0"] + t["y1"]) / 2,
                                  t["x1"] - t["x0"], t["y1"] - t["y0"],
                                  t["conf"] * (0.7 if t["miss"] else 1.0)))
        return out


class HeuristicDetector:
    """Hazards from the 640x360 frame (or the 320x180 one), the rest from 320x180."""

    def __init__(self, temporal: bool = True) -> None:
        import cv2 as _cv2
        self._cv2 = _cv2
        self.temporal = temporal
        self._tracker = _HazardTracker()

    def reset(self) -> None:
        """New attempt: forget hazard tracks (the screen jumped back to 0%)."""
        self._tracker.reset()

    def _player(self, hsv, w, h) -> Entity | None:
        cv2 = self._cv2
        mask_p = cv2.inRange(hsv, np.array([55, 100, 100]), np.array([95, 255, 255]))
        cnts_p, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in cnts_p:
            area = cv2.contourArea(c)
            if not (15 <= area <= 800):
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            if not (0.15 <= cx <= 0.50 and cy >= 0.30):
                continue
            if not (0.5 <= bw / max(1, bh) <= 3.0):
                continue
            if _PL_LAST[0] is not None:
                if abs(cx - _PL_LAST[0][0]) > 0.08 or abs(cy - _PL_LAST[0][1]) > 0.15:
                    continue
            d = abs(cx - _PL_LAST[0][0]) + abs(cy - _PL_LAST[0][1]) if _PL_LAST[0] else -area
            cands.append((d, Entity(CLS_ID["player"], cx, cy, max(bw / w, 0.045), max(bh / h, 0.08), 0.9)))
        if cands:
            cands.sort(key=lambda t: t[0])
            pl = cands[0][1]
            _PL_LAST[0] = (pl.x, pl.y)
            _PL_HOLD[0] = 0
            return pl
        if _PL_LAST[0] is not None:
            _PL_HOLD[0] += 1
            if _PL_HOLD[0] <= 8:
                return Entity(CLS_ID["player"], _PL_LAST[0][0], _PL_LAST[0][1], 0.045, 0.08, 0.5)
            _PL_LAST[0] = None
        return None

    def __call__(self, rgb320: np.ndarray, hires_bgr: np.ndarray | None = None) -> Entities:
        cv2 = self._cv2
        img = np.asarray(rgb320)
        if img.ndim == 4:
            img = img[-1]
        h, w = img.shape[:2]
        bgr = np.ascontiguousarray(img[:, :, ::-1])
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ents: list[Entity] = []
        pl = self._player(hsv, w, h)
        if pl is not None:
            ents.append(pl)
        pbox = None
        if pl is not None and pl.conf > 0.6:
            pbox = (pl.x - pl.w / 2, pl.y - pl.h / 2, pl.x + pl.w / 2, pl.y + pl.h / 2)
        src = hires_bgr if hires_bgr is not None else bgr
        dets = detect_hazards(np.asarray(src)[:, :, :3], player_box=pbox)
        scroll = None
        if self.temporal:
            ents.extend(self._tracker.update(dets))
            scroll = self._tracker.last_scroll
        else:
            ents.extend(Entity(int(d[0]), (d[1] + d[3]) / 2, (d[2] + d[4]) / 2,
                               d[3] - d[1], d[4] - d[2], d[5]) for d in dets)
        # orbs / pads / portals: saturated hue blobs (unchanged from v1)
        for cls_name, (lo, hi) in ORB_HUES.items():
            mask = cv2.inRange(hsv, np.array([lo, 90, 90]), np.array([hi, 255, 255]))
            mask[: int(0.07 * h)] = 0
            cnts2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts2:
                area = cv2.contourArea(c)
                if not (20 <= area <= 3000):
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                ents.append(Entity(CLS_ID[cls_name], (x + bw / 2) / w, (y + bh / 2) / h,
                                   bw / w, bh / h, 0.8))
        mask_g = cv2.inRange(hsv, np.array([35, 90, 90]), np.array([85, 255, 255]))
        cnts3, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts3:
            area = cv2.contourArea(c)
            if not (40 <= area <= 2500):
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            if pl is not None and abs(cx - pl.x) < 1.5 * pl.w and abs(cy - pl.y) < 1.5 * pl.h:
                continue
            if y / h > 0.55 and bw > 1.5 * bh:  # pads are flat strips
                ents.append(Entity(CLS_ID["pad"], cx, cy, bw / w, bh / h, 0.7))
        return Entities(ents, scroll)


def make_detector(backend: str = "heuristic"):
    if backend == "yolo":
        from detect_yolo import YoloDetector
        return YoloDetector()
    return HeuristicDetector()


def draw(bgr: np.ndarray, ents: Entities) -> np.ndarray:
    """Debug overlay (BGR in, BGR out)."""
    import cv2
    out = bgr.copy()
    H, W = out.shape[:2]
    col = {0: (0, 0, 255), 1: (0, 255, 255), 2: (255, 128, 0), 3: (255, 0, 255),
           4: (0, 165, 255), 5: (255, 0, 255), 6: (160, 160, 160), 7: (0, 255, 0)}
    for e in ents.items:
        x0, y0 = int((e.x - e.w / 2) * W), int((e.y - e.h / 2) * H)
        x1, y1 = int((e.x + e.w / 2) * W), int((e.y + e.h / 2) * H)
        cv2.rectangle(out, (x0, y0), (x1, y1), col.get(int(e.cls), (255, 255, 255)), 2)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="entity detector probe")
    p.add_argument("--probe", default=None, help="mp4 to scan (default: one live frame)")
    p.add_argument("--backend", default="heuristic", choices=("heuristic", "yolo"))
    p.add_argument("--show", action="store_true", help="show the boxes while probing")
    args = p.parse_args()
    det = make_detector(args.backend)
    import cv2
    if args.probe:
        cap = cv2.VideoCapture(args.probe)
        n = 0
        det_t = 0.0
        n_sp = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            small = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (320, 180), interpolation=cv2.INTER_AREA)
            hi = cv2.resize(fr, (640, 360), interpolation=cv2.INTER_AREA)
            t0 = time.perf_counter()
            ents = det(small, hi) if args.backend == "heuristic" else det(small)
            det_t += (time.perf_counter() - t0) * 1000.0
            n += 1
            n_sp.append(sum(1 for e in ents.items if e.cls == 0))
            if args.show:
                cv2.imshow("detect", draw(hi, ents))
                if (cv2.waitKey(16) & 0xFF) in (27, ord("q")):
                    break
        cap.release()
        print(f"frames={n} mean={det_t / max(1, n):.2f}ms backend={args.backend} "
              f"spikes/frame mean={np.mean(n_sp) if n_sp else 0:.2f} frames_with_spikes={sum(1 for v in n_sp if v)}")
        return 0
    from capture import GDCapture
    cap = GDCapture()
    try:
        cap.start()
        for _ in range(10):
            cap.read()
        obs = cap.read()
        full = cap.last_fullres
        hi = cv2.resize(np.asarray(full)[:, :, :3], (640, 360), interpolation=cv2.INTER_AREA) if full is not None else None
        det(obs, hi)
        ents = det(cap.read(), hi)
        print(f"live entities={len(ents.items)} " +
              ", ".join(f"{CLASSES[e.cls]}@({e.x:.2f},{e.y:.2f})" for e in ents.items[:12]))
        if hi is not None:
            cv2.imwrite("detect_probe.png", draw(hi, ents))
            print("wrote detect_probe.png")
    finally:
        cap.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
