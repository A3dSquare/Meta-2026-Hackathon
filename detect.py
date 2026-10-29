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

ORBS / PADS / PORTALS -- colour + shape (detect_specials):
  bright, saturated sprites whose hue differs from the level background (or
  that are much brighter than it), sorted by shape in block units: tall rings
  = portals, round blobs = orbs (class by hue), flat strips = pads. Each keeps
  its mean hue as Entity.hue so the policy can tell e.g. a ship portal from a
  gravity portal. Speed portals are not classified -- speed is measured
  directly from how fast hazards scroll (Entities.scroll).

PLAYER: blob of the icon colour (HeuristicDetector(icon_hue=...), default
green/cyan) in the left half of the screen, any height, tracked frame to frame.

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
_PL_SIZE: list = [(0.045, 0.08)]  # [(w, h)] last accepted icon box size
from collections import deque as _dq
_PL_XS = _dq(maxlen=120)   # recent icon screen x (the camera keeps it at one spot)
_PL_MISS: list = [0]     # frames in a row with no icon found


@dataclass
class Entity:
    cls: int
    x: float  # center x, 0..1
    y: float  # center y, 0..1
    w: float  # width 0..1
    h: float  # height 0..1
    conf: float = 1.0
    hue: float = -1.0  # mean OpenCV hue / 180 for coloured objects (orbs/pads/portals), -1 = n/a
    pt: int = -1       # spikes: which way the tip points (0 up, 1 down, 2 left, 3 right, -1 unknown)


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


def detect_hazards(bgr: np.ndarray, player_box=None, force_bright: bool = False, hsv=None, ref_hsv=None) -> list:
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
    if hsv is None:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V, S = hsv[:, :, 2], hsv[:, :, 1]
    R = hsv if ref_hsv is None else ref_hsv          # (thresholds from the whole frame)
    med_v = float(np.median(R[::4, ::4, 2]))
    dark_v = int(min(75, 0.55 * med_v))
    dark = (V < dark_v).astype(np.uint8)
    # outlines: white, or tinted by the level colour (e.g. lavender on purple) --
    # so "bright" = clearly brighter AND clearly less saturated than the background
    med_s = float(np.median(R[::4, ::4, 1]))
    bright = (((V >= 185) & (S <= 75))
              | ((V >= max(215.0, med_v + 35)) & (S <= min(170.0, 0.75 * med_s)))).astype(np.uint8)
    dark[:ui_cut] = 0
    bright[:ui_cut] = 0
    # the ground line is one long bright row: remove it, or every pillar and
    # spike standing on the ground merges into one screen-wide outline blob
    rows = bright.mean(axis=1) > 0.7
    line_rows = []
    for yy in np.nonzero(rows)[0]:
        if yy > ui_cut and yy < H - 2 and (not line_rows or yy - line_rows[-1] > 2):
            line_rows.append(int(yy))
        bright[max(0, yy - 1):yy + 2] = 0
    # ... but such a row IS a surface: the ground, a corridor's floor or its ceiling (the
    # whole-level test: a corridor floor vanished with it and the cube "fell through").
    # Which side is solid? The side that looks like a block (texture or dark fill).
    line_walls = []
    if line_rows:
        import cv2 as _cv
        _edges = _cv.Canny(_cv.cvtColor(bgr, _cv.COLOR_BGR2GRAY), 40, 120)
        _bg = float(np.percentile(R[::4, ::4, 2], 75))
        band = max(3, int(0.6 * blk))
        def _solidish(a, b):
            a, b = max(ui_cut, a), min(H, b)
            if b - a < 2:
                return 0.0
            er = (_edges[a:b] > 0).mean(axis=1) > 0.05
            dr = V[a:b].mean(axis=1) < 0.7 * _bg
            return float((er | dr).mean())
        bounds = [ui_cut] + line_rows + [H]
        for k, yy in enumerate(line_rows):
            below, above = _solidish(yy + 3, yy + 3 + band), _solidish(yy - 2 - band, yy - 2)
            if below >= 0.5 and below > above + 0.2:
                line_walls.append((0, yy, W, bounds[k + 2]))          # a floor: solid down to the next line
            elif above >= 0.5 and above > below + 0.2:
                line_walls.append((0, bounds[k], W, yy + 1))           # a ceiling: solid up to the last one
            elif yy > 0.5 * H:
                line_walls.append((0, yy, W, bounds[k + 2]))          # can't tell, low: the ground
    if player_box is not None:
        px0, py0, px1, py1 = player_box
        # (0.5 / 0.35: the ship's / UFO's body is much bigger than the coloured core the
        # icon box covers, and its dark parts were found as a spike ON the icon)
        pad_x, pad_y = 0.5 * blk / W, 0.35 * blk / H
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
            # (huge = background -- but a big platform's OUTLINE is huge too, and sparse:
            # Stereo Madness's grid platforms were dropped, so the cube "fell through" them)
            if area < max(6, min_h * min_w // 2) or (bw * bh > 0.30 * W * H
                                                     and (solid_check or area > 0.25 * bw * bh)):
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
                        # bright outline: the inside of a real spike is dark, at
                        # least near the tip (GD spikes fade from black at the tip
                        # to the level colour at the base)
                        if flip:
                            cy0 = yb - (yb - ya) * 2 // 3; cy1 = yb - (yb - ya) // 4
                        else:
                            cy0 = ya + (yb - ya) // 4; cy1 = ya + (yb - ya) * 2 // 3
                        cx0 = x + l + (r - l) * 3 // 8; cx1 = x + r - (r - l) * 3 // 8 + 1
                        inner = V[cy0:max(cy0 + 1, cy1), cx0:max(cx0 + 1, cx1)]
                        if inner.size == 0 or float(inner.mean()) > 0.75 * med_v:
                            continue
                    spikes_here.append((0, x + l, ya, x + r + 1, yb + 1, conf, 1 if flip else 0))
            if solid_check and bw >= 1.0 * blk and bh <= 0.8 * blk and area / float(bw * bh) < 0.75:
                # a row of ground thorns: one wide, short, jagged black band
                tv = top[colv]
                if tv.size >= 4:
                    peaks = int(np.sum((tv[1:-1] < tv[:-2]) & (tv[1:-1] <= tv[2:])))
                    if peaks >= 3 and (y + bh / 2) < 0.94 * H:
                        spikes_here.append((0, x, y, x + bw, y + bh, 0.8, 0))
            if not solid_check and max(bw, bh) >= min_w:
                # spikes pointing left / right (on the sides of pillars): the same
                # peak search on the left / right edge of the outline
                rowv = sub.any(axis=1)
                left = np.where(rowv, sub.argmax(axis=1), 10**6) + x
                right = np.where(rowv, bw - 1 - sub[:, ::-1].argmax(axis=1), -1) + x
                for flip, prof in ((False, left), (True, (W - 1) - right)):
                    for (t0, t1, apex, base) in profile_peaks(prof.astype(np.int64), rowv, **kw):
                        xa, xb = ((W - 1) - base, (W - 1) - apex) if flip else (apex, base)
                        ya0, ya1 = y + t0, y + t1
                        mid = (ya0 + ya1) // 2
                        cx0 = xa + (xb - xa) // 4 if not flip else xa + (xb - xa) // 3
                        cx1 = xa + (xb - xa) * 2 // 3 if not flip else xb - (xb - xa) // 4
                        inner = V[max(0, mid - 1):mid + 2, cx0:max(cx0 + 1, cx1)]
                        if inner.size and float(inner.mean()) <= 0.75 * med_v:
                            spikes_here.append((0, xa, ya0, xb + 1, ya1 + 1, 0.75, 3 if flip else 2))
            found.extend(spikes_here)
            if not solid_check and bw >= 0.5 * blk and bh >= 1.2 * blk:
                # pillar: two long vertical outline edges (its top may carry a spike,
                # so there's no flat top to find)
                colsum = sub.sum(axis=0)
                cols = np.nonzero(colsum >= max(1.0 * blk, 0.45 * bh))[0]
                groups = []      # long vertical edges, grouped (an edge is a few px wide)
                for c in cols:
                    if groups and c - groups[-1][-1] <= 3:
                        groups[-1].append(int(c))
                    else:
                        groups.append([int(c)])
                for ga, gb in zip(groups, groups[1:]):
                    c0, c1 = ga[0], gb[-1]
                    if not (0.5 * blk <= c1 - c0 <= 2.5 * blk):
                        continue
                    ys_a = np.nonzero(sub[:, ga[len(ga) // 2]])[0]
                    ys_b = np.nonzero(sub[:, gb[len(gb) // 2]])[0]
                    ytop = int(max(ys_a[0], ys_b[0]))
                    ybot = int(min(ys_a[-1], ys_b[-1]))
                    if ybot - ytop < 1.0 * blk:
                        continue
                    band = sub[max(0, ytop - int(0.4 * blk)):ytop + int(0.3 * blk) + 1, c0:c1 + 1]
                    if band.size and band.any(axis=0).mean() >= 0.6:   # a top edge joins them
                        walls.append((6, x + c0, y + ytop, x + c1 + 1, y + ybot + 1, 0.8))
            if not solid_check and bw >= 0.5 * blk and bh >= 0.35 * blk:
                # outlined block / platform: a flat bright top edge with the
                # outline going down from it (GD blocks have a white border;
                # their fill is often a black-to-colour gradient or a grid)
                # split the outline into flat-topped stretches: a pillar joined to
                # the ground line, or a platform with a spike on it, still gives
                # one wall per flat top
                tops_rel = np.where(colv, top - y, -10**6)
                c0 = 0
                ncol = len(tops_rel)
                while c0 < ncol:
                    if tops_rel[c0] < 0:
                        c0 += 1
                        continue
                    c1 = c0
                    while c1 + 1 < ncol and tops_rel[c1 + 1] >= 0 and abs(int(tops_rel[c1 + 1]) - int(tops_rel[c0])) <= 2:
                        c1 += 1
                    if c1 - c0 + 1 >= 0.5 * blk:
                        t0 = int(np.median(tops_rel[c0:c1 + 1]))
                        # ... and by where the outline ENDS below: an L-shaped platform (a bar
                        # with a pillar under one end) was one big box, filling the empty corner
                        # under the bar with "solid". A column whose outline stops at the top
                        # edge (the platform's inside isn't outlined) goes as deep as the rest.
                        seg = np.arange(c0, c1 + 1)
                        dmax = int(bot.max())          # deepest point of the whole outline
                        eb = np.where(bot[c0:c1 + 1] - (y + t0) >= 0.35 * blk, bot[c0:c1 + 1], dmax)
                        runs = []                      # [start, end, bottom]
                        for c, e in zip(seg, eb):
                            if runs and abs(int(e) - runs[-1][2]) <= 0.5 * blk:
                                runs[-1][1] = c
                                runs[-1][2] = max(runs[-1][2], int(e))
                            else:
                                runs.append([c, c, int(e)])
                        merged_runs = []               # slivers (< 1 block) join a neighbour
                        for r_ in runs:
                            if merged_runs and (r_[1] - r_[0] + 1 < blk or merged_runs[-1][1] - merged_runs[-1][0] + 1 < blk):
                                m_ = merged_runs[-1]
                                m_[1], m_[2] = r_[1], max(m_[2], r_[2])
                            else:
                                merged_runs.append(list(r_))
                        for d0, d1, eb_ in merged_runs:
                            depth = int(eb_ - (y + t0))
                            if depth >= 0.35 * blk:
                                walls.append((6, x + d0, y + t0, x + d1 + 1, y + t0 + depth + 1, 0.85))
                            elif d1 - d0 + 1 >= 1.5 * blk:
                                # a long top edge on its own (its sides are off screen or
                                # hidden behind the icon): a platform top -- 1 block deep
                                walls.append((6, x + d0, y + t0, x + d1 + 1, y + t0 + int(blk) + 1, 0.7))
                    c0 = c1 + 1
                # blocks hanging from above the screen (no top edge on screen): the same
                # from their flat BOTTOM edge up to where the outline starts
                if y <= ui_cut + 2:
                    bots_rel = np.where(colv, bot - y, -10**6)
                    c0 = 0
                    while c0 < ncol:
                        if bots_rel[c0] < 0:
                            c0 += 1
                            continue
                        c1 = c0
                        while c1 + 1 < ncol and bots_rel[c1 + 1] >= 0 and abs(int(bots_rel[c1 + 1]) - int(bots_rel[c0])) <= 2:
                            c1 += 1
                        if c1 - c0 + 1 >= 0.5 * blk:
                            b0 = int(np.median(bots_rel[c0:c1 + 1]))
                            # only a simple block (nothing of it further down): a structure that
                            # also outlines a corridor below made a box over the corridor
                            if b0 >= 0.35 * blk and bh - b0 <= 0.3 * blk:
                                walls.append((6, x + c0, y, x + c1 + 1, y + b0 + 1, 0.8))
                        c0 = c1 + 1
            if solid_check and bw >= 0.4 * blk and bh >= 0.4 * blk:
                sp_area = sum((s[3] - s[1]) * (s[4] - s[2]) * 0.5 for s in spikes_here)
                if area / float(bw * bh) >= 0.55 and sp_area < 0.5 * area:
                    wy0 = y
                    tops = [s[4] for s in spikes_here if s[2] <= y + 1]
                    if tops:  # spikes standing on the block: wall starts at their base
                        wy0 = max(tops)
                    if y + bh - wy0 >= 0.3 * blk and not (y + bh >= H - 2 and y > 0.85 * H):
                        walls.append((6, x, wy0, x + bw, y + bh, 0.8))
        found.extend(walls)

    _scan(dark, True, 0.9)
    # white outlines: full spike shapes (the black fill often only covers the
    # tip) and outlined blocks/platforms
    _scan(bright, False, 0.75)
    found = _trim_empty_walls(found, bgr, V, blk, player_box)
    out = _merge_hazards(found, W, H, blk)
    for (lx0, ly0, lx1, ly1) in line_walls:          # screen-wide surfaces (kept out of the
        out.append((6, lx0 / W, ly0 / H, lx1 / W, ly1 / H, 0.8, -1))   # merging: not pillars)
    return out


def _trim_empty_walls(found: list, bgr, V, blk: float, player_box=None) -> list:
    """Cut plain-background bands out of block boxes. A box is made from outline pieces
    (a top edge, a bottom edge, side lines), and stray lines -- a portal's glow, a thin
    decoration line, a corridor boundary -- stretched boxes over open space the player
    flies / runs through (whole-level test: a ship corridor 'filled' with solid).
    A row inside a real block has texture (grid lines, outlines) or a fill clearly
    darker than the background; a band >= 0.7 block of neither is open space."""
    import cv2
    out = []
    H, W = V.shape[:2]
    bg = float(np.percentile(V[::4, ::4], 75))       # the (bright) background panels
    gray = None
    for f in found:
        if f[0] != 6:
            out.append(f)
            continue
        x0, y0, x1, y1 = int(f[1]), int(f[2]), int(f[3]), int(f[4])
        if y1 - y0 < 0.7 * blk or x1 - x0 < 4:
            out.append(f)
            continue
        if gray is None:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 40, 120)
            if player_box is not None:     # the icon and its trail aren't the block's texture
                px0, py0, px1, py1 = player_box
                edges[max(0, int(py0 * H - blk)):int(py1 * H + blk),
                      max(0, int(px0 * W - 4 * blk)):int(px1 * W + blk)] = 0
        ix0, ix1 = min(W, x0 + 3), max(0, x1 - 3)
        if ix1 - ix0 < 3:
            out.append(f)
            continue
        e_rows = (edges[y0:y1, ix0:ix1] > 0).mean(axis=1)
        d_rows = V[y0:y1, ix0:ix1].astype(np.float32).mean(axis=1) < 0.7 * bg     # dark fill
        k = max(1, int(0.3 * blk))
        solid = np.convolve((e_rows > 0.05) | d_rows, np.ones(k) / k, mode="same") > 0.3
        need = int(0.7 * blk)
        r = 0
        n = len(solid)
        segs = []                      # solid stretches split by long empty bands
        start = 0
        while r < n:
            if not solid[r]:
                r2 = r
                while r2 < n and not solid[r2]:
                    r2 += 1
                if r2 - r >= need:
                    segs.append((start, r))
                    start = r2
                r = r2
            else:
                r += 1
        segs.append((start, n))
        if float(solid.mean()) < 0.25:
            continue                   # almost all open space: not a block at all
        if len(segs) == 1:
            out.append(f)
            continue
        for a, b in segs:
            if b - a >= 0.35 * blk:
                out.append((6, x0, y0 + a, x1, y0 + b) + tuple(f[5:]))
    return out


def _merge_hazards(found: list, W: int, H: int, blk: float) -> list:
    """Same object seen twice (dark fill + bright outline) -> one box (the union);
    rows of touching spikes / ground thorns -> one strip; walls nested in a
    bigger wall -> dropped. Fewer, truer boxes = the 8 policy slots go to
    things that matter."""
    def contains(a, b, tol=0.15):   # centre of b inside a (with tolerance)
        cx, cy = (b[1] + b[3]) / 2, (b[2] + b[4]) / 2
        tx, ty = min(tol * (a[3] - a[1]), 0.15 * blk), min(tol * (a[4] - a[2]), 0.15 * blk)
        return a[1] - tx <= cx <= a[3] + tx and a[2] - ty <= cy <= a[4] + ty
    out = []
    for cls in (0, 6):
        boxes = [list(f) for f in found if f[0] == cls]
        boxes.sort(key=lambda b: -(b[3] - b[1]) * (b[4] - b[2]))
        merged: list = []
        for b in boxes:
            hit = next((m for m in merged if contains(m, b) or contains(b, m)), None)
            if hit is None:
                merged.append(b)
            elif cls == 0:      # union of fill + outline views of one spike
                hit[1], hit[2] = min(hit[1], b[1]), min(hit[2], b[2])
                hit[3], hit[4] = max(hit[3], b[3]), max(hit[4], b[4])
                hit[5] = max(hit[5], b[5])
        if cls == 6:            # one platform split by a spike standing on it -> rejoin
            merged.sort(key=lambda b: b[1])
            joined: list = []
            for b in merged:
                # (same bottom too -- not an L -- unless one piece is a lone top edge whose
                # depth is a guess; those are often cut by the icon's mask: gap up to 1.6)
                guess = min(b[5], joined[-1][5]) <= 0.71 if joined else False
                if (joined and abs(b[2] - joined[-1][2]) <= 0.15 * blk
                        and b[1] - joined[-1][3] <= (1.6 if guess else 1.2) * blk
                        and (guess or abs(b[4] - joined[-1][4]) <= 0.5 * blk)):
                    j = joined[-1]
                    j[3], j[4] = max(j[3], b[3]), max(j[4], b[4])
                    continue
                joined.append(b)
            stacked: list = []   # pieces of one pillar split by the icon mask -> rejoin
            for b in sorted(joined, key=lambda b: b[2]):
                hit = None
                for s in stacked:
                    ov = min(s[3], b[3]) - max(s[1], b[1])
                    # (similar widths only: a narrow piece under a wide platform -- a portal's
                    # glow -- extended the whole platform down over the corridor)
                    if ov >= 0.7 * max(s[3] - s[1], b[3] - b[1]) and b[2] - s[4] <= 0.4 * blk:
                        hit = s
                        break
                if hit is not None:
                    hit[1], hit[3] = min(hit[1], b[1]), max(hit[3], b[3])
                    hit[4] = max(hit[4], b[4])
                else:
                    stacked.append(b)
            merged = stacked
        if cls == 0:            # strips: same base line, (nearly) touching
            merged.sort(key=lambda b: b[1])
            strips: list = []
            for b in merged:
                if strips:
                    s = strips[-1]
                    small = (b[4] - b[2]) < 0.6 * blk and (s[4] - s[2]) < 0.6 * blk
                    gap = b[1] - s[3]
                    same_base = (abs(b[4] - s[4]) <= 0.25 * blk and abs(b[2] - s[2]) <= 0.3 * blk
                                 and abs((b[4] - b[2]) - (s[4] - s[2])) <= 0.3 * blk)
                    if same_base and gap <= (1.0 if small else 0.35) * blk:
                        s[2], s[3], s[4] = min(s[2], b[2]), max(s[3], b[3]), max(s[4], b[4])
                        s[5] = max(s[5], b[5])
                        continue
                strips.append(b)
            merged = strips
        for b in merged:
            out.append((b[0], b[1] / W, b[2] / H, b[3] / W, b[4] / H, b[5], b[6] if len(b) > 6 else -1))
    return out


def _hue_dist(h, ref):
    d = np.abs(h.astype(np.int16) - int(ref))
    return np.minimum(d, 180 - d)


def _orb_class(hue: float) -> int:
    """Nearest of the three orb classes by hue (others keep their hue feature)."""
    cands = {CLS_ID["orb_y"]: 27, CLS_ID["orb_b"]: 105, CLS_ID["orb_p"]: 150}
    return min(cands, key=lambda c: min(abs(hue - cands[c]), 180 - abs(hue - cands[c])))


PAD_S, PAD_V, PAD_DH = 150, 230, 8
PAD_STRIP_S = 999      # a pad found by its strip under a glow dome (off: on real levels it also found decoration)


def detect_specials(bgr: np.ndarray, player_box=None, hsv=None, solids=None, ref_hsv=None) -> list:
    """Orbs, pads and portals from one BGR frame -> [(cls, x0, y0, x1, y1, conf, hue)].

    GD draws these as bright, saturated sprites whose colour encodes the type
    (yellow/blue/pink orbs and pads, mode/gravity/speed portals). Colour alone
    is not enough -- the level's own background is often just as saturated --
    so a pixel only counts if its hue differs from the background's typical
    hue, or it is clearly brighter than the background. Blobs are then sorted
    by SHAPE, in block units:
      portal: tall (>= 1.7 blocks, height >= 1.6 x width)
      orb:    round, 0.45 .. 1.5 blocks
      pad:    flat strip, height <= 0.5 block, width >= 1.8 x height
    Everything else (decoration) is dropped.
    """
    import cv2

    H, W = bgr.shape[:2]
    blk = BLOCK_FRAC_H * H
    if hsv is None:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hh, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sub = (hsv if ref_hsv is None else ref_hsv)[::4, ::4].reshape(-1, 3)   # (background from the whole frame)
    sat = sub[sub[:, 1] > 60]
    if len(sat) > 50:
        # dominant hue of the saturated background (circular mode via histogram)
        hist = np.bincount(sat[:, 0].astype(np.int64), minlength=180)
        hist = np.convolve(np.concatenate([hist[-5:], hist, hist[:5]]), np.ones(11), "valid")
        bg_h = int(np.argmax(hist))
    else:
        bg_h = -1
    bg_v = float(np.median(sub[:, 2]))
    colour = (S >= 100) & (V >= 150)
    if bg_h >= 0:
        colour &= _hue_dist(Hh, bg_h) >= 14
    glow = (S >= 70) & (V.astype(np.int16) >= min(250, bg_v + 50))
    pad_x0 = pad_x1 = pad_y0 = pad_y1 = None
    if player_box is not None:
        px0, py0, px1, py1 = player_box
        # icons are bigger than their coloured core (wings, robot legs, ship)
        pad_x0 = max(0, int((px0 - 0.6 * blk / W) * W)); pad_x1 = min(W, int((px1 + 0.6 * blk / W) * W) + 1)
        pad_y0 = max(0, int((py0 - 0.6 * blk / H) * H)); pad_y1 = min(H, int((py1 + 0.6 * blk / H) * H) + 1)
    out = []
    # two separate masks: a bright background patch in the "glow" mask must not
    # swallow an orb that the "different hue" mask already isolates cleanly
    for which, raw in (("colour", colour), ("glow", glow)):
        mask = raw.astype(np.uint8)
        mask[: int(0.065 * H)] = 0
        if pad_x0 is not None:
            mask[pad_y0:pad_y1, pad_x0:pad_x1] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in sorted(range(1, n), key=lambda k: -int(stats[k][4])):
            x, y, bw, bh, area = (int(v) for v in stats[i])
            wb, hb = bw / blk, bh / blk
            if area < 0.008 * blk * blk or wb > 5 or hb > 5:
                continue
            m = lab[y:y + bh, x:x + bw] == i
            hue_vals = Hh[y:y + bh, x:x + bw][m]
            ang = hue_vals.astype(np.float32) * (2 * np.pi / 180)
            hue = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2) % 180
            box = (x / W, y / H, (x + bw) / W, (y + bh) / H)
            det = None
            # (1.25 x, was 1.6 x: the particles a portal gives off made Stereo Madness's
            # ship portals too wide to count)
            if hb >= 1.7 and hb >= 1.25 * wb and wb >= 0.3:
                det = (CLS_ID["portal"], *box, 0.8, hue / 180.0)
            elif (which == "colour" and min(wb, hb) >= 0.1 and max(wb, hb) < 0.45 and min(wb, hb) < 0.3
                  and 0.5 <= wb / max(hb, 1e-6) <= 2.0 and m[bh // 2, bw // 2]):
                # small orb core (e.g. a used/pulsing yellow orb) inside a bright ring
                r = (bw + bh) / 4.0
                cx_, cy_ = x + bw / 2.0, y + bh / 2.0
                R = int(round(2.0 * r))
                yy, xx = np.ogrid[-R:R + 1, -R:R + 1]
                ann = (xx * xx + yy * yy <= R * R) & (xx * xx + yy * yy > (1.2 * r) ** 2)
                y0_, x0_ = int(cy_) - R, int(cx_) - R
                if y0_ >= 0 and x0_ >= 0 and y0_ + 2 * R + 1 <= H and x0_ + 2 * R + 1 <= W:
                    ring = V[y0_:y0_ + 2 * R + 1, x0_:x0_ + 2 * R + 1][ann]
                    if ring.size and float((ring >= 235).mean()) >= 0.3:
                        h2 = 0.4 * blk
                        det = (_orb_class(hue), (cx_ - h2) / W, (cy_ - h2) / H, (cx_ + h2) / W, (cy_ + h2) / H, 0.7, hue / 180.0)
            elif 0.3 <= wb <= 1.5 and 0.3 <= hb <= 1.5 and 0.7 <= wb / max(hb, 1e-6) <= 1.4 and m[bh // 2, bw // 2]:
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    c = max(cnts, key=cv2.contourArea)
                    per = cv2.arcLength(c, True)
                    circ = 4 * np.pi * cv2.contourArea(c) / max(1e-6, per * per)
                    # round, not square: decoration squares (Stereo Madness's pink ones)
                    # have circularity ~0.79 and filled corners
                    k_ = max(1, int(0.12 * min(bw, bh)))
                    corners = [m[:k_, :k_].mean(), m[:k_, -k_:].mean(), m[-k_:, :k_].mean(), m[-k_:, -k_:].mean()]
                    if circ >= 0.6 and area >= 0.45 * bw * bh and float(np.mean(corners)) < 0.5:
                        det = (_orb_class(hue), *box, 0.8, hue / 180.0)
            elif hb <= 1.8 and 0.45 <= wb <= 2.2:
                # pad: a flat strip ON a surface (ground / block top, or under a ceiling for
                # an upside-down pad). Its glow dome often merges with it into one taller
                # blob (that blob used to fail "wide and flat"), so look for the strip at
                # the blob's bottom (or top) edge. Must touch a surface edge: a ship's
                # flame, trails and particles float in the air.
                strip = _pad_strip(m, blk)
                if strip is not None:
                    sx0, sy0, sx1, sy1 = strip
                    X0, Y0, X1, Y1 = x + sx0, y + sy0, x + sx1, y + sy1
                    ms = m[sy0:sy1, sx0:sx1]
                    s_mean = float(S[Y0:Y1, X0:X1][ms].mean()) if ms.any() else 0.0
                    # a pad is a solid saturated colour (the pale glow under a slab's outline
                    # isn't), and the "glow" mask only counts for pads standing on something
                    flat = which == "colour" and hb <= 1.0 and wb >= 1.25 * hb   # the old rule: a plain flat blob
                    if (flat or s_mean >= PAD_STRIP_S) and _pad_supported(X0 / W, Y0 / H, X1 / W, Y1 / H, solids, V, blk,
                                                        hang_ok=(which == "colour")):
                        det = (CLS_ID["pad"], X0 / W, Y0 / H, X1 / W, Y1 / H,
                               0.7 if which == "colour" else 0.6, hue / 180.0)
            if det is not None and det[0] in (1, 2, 3):
                cxd, cyd = (det[1] + det[3]) / 2 * W, (det[2] + det[4]) / 2 * H
                if any(o[0] in (1, 2, 3) and abs((o[1] + o[3]) / 2 * W - cxd) < 0.7 * blk
                       and abs((o[2] + o[4]) / 2 * H - cyd) < 0.7 * blk for o in out):
                    det = None   # particles around an orb already found
            if (det is not None and det[0] != CLS_ID["portal"] and player_box is not None
                    and (det[1] + det[3]) / 2 < (player_box[0] + player_box[2]) / 2 - 0.3 * blk / W):
                det = None       # behind the icon: its own trail / particles (and past anyway)
            if det is not None and not any(_iou(det[1:5], o[1:5]) > 0.3 for o in out):
                out.append(det)
    # pads, a third way: a pad's strip is a solid, very bright, saturated colour -- brighter
    # than the background even when its hue is close to it (pink pad, purple level), and
    # brighter than its own translucent glow dome, so here it stands alone.
    pm = ((S >= PAD_S) & (V >= PAD_V) & (V.astype(np.int16) >= int(bg_v) + 25)).astype(np.uint8)
    if bg_h >= 0:
        pm &= (_hue_dist(Hh, bg_h) >= PAD_DH).astype(np.uint8)
    pm[: int(0.065 * H)] = 0
    if pad_x0 is not None:
        pm[pad_y0:pad_y1, pad_x0:pad_x1] = 0
    # a block's glowing outline is bright and saturated too, and a pad standing on it would
    # merge with it: take out long horizontal runs first (a pad is at most ~1.6 blocks)
    L = max(3, int(1.8 * blk))
    pm &= ~cv2.morphologyEx(pm, cv2.MORPH_OPEN, np.ones((1, L), np.uint8)).astype(bool) * np.uint8(1)
    # ...and tall bright things (background panels, portals' sides): a pad is <= 0.4 blocks
    Lv = max(3, int(0.6 * blk))
    pm &= ~cv2.morphologyEx(pm, cv2.MORPH_OPEN, np.ones((Lv, 1), np.uint8)).astype(bool) * np.uint8(1)
    pm = cv2.morphologyEx(pm, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))   # leftover slivers
    n, lab, stats, _ = cv2.connectedComponentsWithStats(pm, connectivity=8)
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        wb, hb = bw / blk, bh / blk
        if not (0.5 <= wb <= 1.6 and 0.08 <= hb <= 0.4 and wb >= 2.0 * hb and area >= 0.45 * bw * bh):
            continue
        mid = lab[y + bh // 2, x:x + bw] == i
        if float(mid.mean()) < 0.8 or float((lab[y:y + bh, x + bw // 2] == i).mean()) < 0.7:
            continue                         # hollow: a slab's outline, not a solid pad
        X0, Y0, X1, Y1 = x / W, y / H, (x + bw) / W, (y + bh) / H
        if any(_iou((X0, Y0, X1, Y1), o[1:5]) > 0.1 or (o[1] <= (X0 + X1) / 2 <= o[3] and o[2] <= (Y0 + Y1) / 2 <= o[4])
               for o in out):
            continue
        if player_box is not None and (X0 + X1) / 2 < (player_box[0] + player_box[2]) / 2 - 0.3 * blk / W:
            continue
        if not _pad_supported(X0, Y0, X1, Y1, solids, V, blk):
            continue
        hv = Hh[y:y + bh, x:x + bw][lab[y:y + bh, x:x + bw] == i].astype(np.float32) * (2 * np.pi / 180)
        hue = (np.degrees(np.arctan2(np.sin(hv).mean(), np.cos(hv).mean())) / 2) % 180
        out.append((CLS_ID["pad"], X0, Y0, X1, Y1, 0.65, hue / 180.0))
    return out


def _pad_strip(m: np.ndarray, blk: float):
    """(x0, y0, x1, y1) in blob pixels of a flat strip along the blob's bottom or top edge
    (a pad, with or without its glow dome), else None."""
    rows = m.sum(axis=1)
    if rows.size == 0:
        return None
    for order in (range(len(rows) - 1, -1, -1), range(len(rows))):   # bottom pad, then top
        order = list(order)
        edge = next((r for r in order if rows[r] >= 0.45 * blk), None)
        if edge is None or abs(order.index(edge)) > 0.12 * blk:
            continue                              # the wide part isn't at the edge
        wmax = rows[edge]
        taken = []
        for r in order[order.index(edge):]:
            if rows[r] < 0.55 * wmax:
                break
            taken.append(r)
            wmax = max(wmax, rows[r])
        hh = len(taken)
        if not taken or hh > 0.45 * blk or hh < 0.08 * blk:
            continue
        cols = np.nonzero(m[min(taken):max(taken) + 1].any(axis=0))[0]
        ww = cols.max() - cols.min() + 1
        if ww < 0.45 * blk or ww > 2.2 * blk or ww < 1.8 * hh:
            continue
        # the strip must be the widest part: above it the dome narrows (or nothing at all)
        rest = [rows[r] for r in order[order.index(edge) + hh:]]
        if rest and max(rest) > 1.1 * ww:
            continue
        return int(cols.min()), int(min(taken)), int(cols.max() + 1), int(max(taken) + 1)
    return None


def _pad_supported(x0, y0, x1, y1, solids, V, blk, hang_ok: bool = True) -> bool:
    """Pad on a block / the ground (or under one, upside down)? The detector's solid boxes
    (normalised x0, y0, x1, y1) first, then the brightness edge (the ground line isn't
    always a solid box) -- an edge only for a pad standing on it."""
    H, W = V.shape
    if solids:
        tol = 0.3 * blk / H
        for a0, b0, a1, b1 in solids:
            ov = min(x1, a1) - max(x0, a0)
            if ov < 0.5 * (x1 - x0):
                continue
            if abs(b0 - y1) <= tol and b1 > y1:        # sits on its top
                return True
            if hang_ok and abs(b1 - y0) <= tol and b0 < y0:   # hangs under its bottom
                return True
    return _on_surface(V, int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H), blk, below_only=True)


def _on_surface(V: np.ndarray, x0: int, y0: int, x1: int, y1: int, blk: float,
                below_only: bool = False) -> bool:
    """Does the strip x0..x1 sit on a horizontal edge (just below it) or hang under one
    (just above it)? An edge = a big brightness step between neighbouring rows, across
    most of the columns, a bit wider than the strip itself."""
    H, W = V.shape
    w = x1 - x0
    xa, xb = max(0, int(x0 - 0.3 * w)), min(W, int(x1 + 0.3 * w))
    reach = max(2, int(0.2 * blk))
    for ya, yb in ((y1 - 1, y1 + reach),) + (() if below_only else ((y0 - reach, y0 + 1),)):
        ya, yb = max(0, ya), min(H - 1, yb)
        if yb - ya < 2 or xb - xa < 4:
            continue
        band = V[ya:yb + 1, xa:xb].astype(np.int16)
        step = np.abs(np.diff(band, axis=0)).max(axis=0)
        # left and right of the strip (the pad itself covers the middle)
        side = np.concatenate([step[: max(1, int(x0 - xa))], step[max(0, int(x1 - xa)):]])
        if side.size >= 2 and float((side >= 15).mean()) >= 0.6:
            return True
    return False


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
                         pt=d[6] if len(d) > 6 and d[6] >= 0 else t.get("pt", -1),
                         hits=t["hits"] + 1, miss=0)
            else:
                t["miss"] += 1
        self.last_scroll = float(np.median(steps)) if steps else None
        if steps:
            gvx = float(np.median(steps))
            for t in self.tracks:
                if t["miss"] > 0:  # coast with the scene scroll
                    t["x0"] += gvx; t["x1"] += gvx; t["vx"] = gvx
        # a hazard seen in 3+ frames keeps coasting with the scroll through 4 missed frames
        # (was 2 for all): one frame without the spike made the predicted path "blip" safe
        self.tracks = [t for t in self.tracks if t["miss"] <= (4 if t["hits"] >= 3 else 2) and t["x1"] > -0.05]
        for j, d in enumerate(dets):
            if j not in used:
                self.tracks.append(dict(cls=d[0], x0=d[1], y0=d[2], x1=d[3], y1=d[4],
                                        conf=d[5], hits=1, miss=0, vx=-0.01,
                                        pt=d[6] if len(d) > 6 else -1))
        out = []
        for t in self.tracks:
            if t["hits"] >= 2:
                out.append(Entity(int(t["cls"]), (t["x0"] + t["x1"]) / 2, (t["y0"] + t["y1"]) / 2,
                                  t["x1"] - t["x0"], t["y1"] - t["y0"],
                                  t["conf"] * (0.7 if t["miss"] else 1.0), pt=int(t.get("pt", -1))))
        return out


class HeuristicDetector:
    """Hazards from the 640x360 frame (or the 320x180 one), the rest from 320x180."""

    def __init__(self, temporal: bool = True, icon_hue: tuple = (55, 95), extra=None,
                 extra_every: int = 1, specials_every: int = 2) -> None:
        import cv2 as _cv2
        self._cv2 = _cv2
        self.temporal = temporal
        self.icon_hue = icon_hue   # OpenCV hue range of your icon's main colour
        self.behind_cut = 0.2      # detect only right of this (fraction of the width): behind the icon
        self._tracker = _HazardTracker()
        # extra: a learned detector (detect_yolo.YoloDetector) whose boxes are ADDED
        # where the built-in rules found nothing. extra_every > 1 runs it every N
        # frames and slides its last boxes with the scroll in between (for CPUs).
        self.extra = extra
        self.extra_every = max(1, int(extra_every))
        self._extra_cache: list = []
        self._extra_n = 0
        # orbs / pads / portals: found every `specials_every` frames; in between, the
        # last ones slide with the scroll (they move with the level). Saves ~3 ms/frame.
        self.specials_every = max(1, int(specials_every))
        self._spec_cache: list = []
        self._spec_n = 0

    def reset(self) -> None:
        """New attempt: forget hazard tracks (the screen jumped back to 0%)."""
        self._tracker.reset()
        self._spec_cache = []
        self._spec_n = 0
        self._extra_cache = []
        self._extra_n = 0

    def _extra_dets(self, src, pbox) -> list:
        n, self._extra_n = self._extra_n, self._extra_n + 1
        if n % self.extra_every == 0 or not self._extra_cache:
            try:
                out = self.extra.detect(src)
            except Exception as e:
                if not getattr(self, "_extra_err", False):
                    self._extra_err = True
                    print(f"[yolo] detection failed, continuing without it: {e}")
                out = []
            if pbox is not None:   # the icon itself is never a hazard
                out = [o for o in out if not _overlaps(o, (0, *pbox))]
            self._extra_cache = out
            return out
        sc = self._tracker.last_scroll or 0.0   # slide cached boxes with the scroll
        self._extra_cache = [(c, x0 + sc, y0, x1 + sc, y1, p) for (c, x0, y0, x1, y1, p) in self._extra_cache]
        return self._extra_cache

    def _player(self, hsv, w, h) -> Entity | None:
        """Icon = blob of the icon colour in the left half, tracked frame to frame.

        Runs on whatever resolution it is given (640x360 preferred: the icon's
        black outline survives there and keeps it apart from a same-coloured
        background). Sizes are in block units so the gates are resolution-free.
        """
        cv2 = self._cv2
        blk2 = (BLOCK_FRAC_H * h) ** 2
        mask_p = cv2.inRange(hsv, np.array([self.icon_hue[0], 100, 100]), np.array([self.icon_hue[1], 255, 255]))
        sub = hsv[::4, ::4].reshape(-1, 3)
        sat = sub[sub[:, 1] > 60]
        if len(sat) > 50:
            hist = np.bincount(sat[:, 0].astype(np.int64), minlength=180)
            bg_h = int(np.argmax(hist))
            lo, hi = self.icon_hue
            if lo - 8 <= bg_h <= hi + 8:
                # background has the icon's colour: keep pixels clearly brighter
                # than it, or clearly a different hue
                bg_v = float(np.median(sub[:, 2]))
                same = _hue_dist(hsv[:, :, 0], bg_h) < 10
                mask_p[same | (hsv[:, :, 2] < min(250, bg_v + 35)) & ~(_hue_dist(hsv[:, :, 0], bg_h) >= 14)] = 0
        cnts_p, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        anchor = float(np.median(_PL_XS)) if len(_PL_XS) >= 20 else None
        for c in cnts_p:
            area = cv2.contourArea(c) / blk2
            if not (0.03 <= area <= 4.0):      # wave/mini icons are small, ships wide
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            # whole height: ships fly high, upside-down cubes run on the ceiling
            # GD's camera keeps the icon at a fixed x (~0.37 of the screen)
            # (bottom 14% excluded: practice mode's green checkpoint buttons sit there)
            if not (0.25 <= cx <= 0.48 and 0.07 <= cy <= 0.86):
                continue
            # the camera keeps the icon at ONE screen x: once that's known, a green thing
            # elsewhere (menu buttons, a practice checkpoint diamond scrolling past) isn't it
            if anchor is not None and abs(cx - anchor) > 0.05:
                continue
            if not (0.3 <= bw / max(1, bh) <= 3.5):
                continue
            if _PL_LAST[0] is not None:
                if abs(cx - _PL_LAST[0][0]) > 0.08 or abs(cy - _PL_LAST[0][1]) > 0.15:
                    continue
            d = (abs(cx - _PL_LAST[0][0]) + abs(cy - _PL_LAST[0][1]) if _PL_LAST[0]
                 else 3 * abs(cx - 0.37) - 0.002 * area)
            if anchor is not None:
                d += 3 * abs(cx - anchor)
            cands.append((d, Entity(CLS_ID["player"], cx, cy, max(bw / w, 0.03), max(bh / h, 0.05), 0.9)))
        if cands:
            cands.sort(key=lambda t: t[0])
            pl = cands[0][1]
            _PL_XS.append(pl.x)
            _PL_MISS[0] = 0
            _PL_LAST[0] = (pl.x, pl.y)
            _PL_SIZE[0] = (pl.w, pl.h)
            _PL_HOLD[0] = 0
            return pl
        _PL_MISS[0] += 1
        if _PL_MISS[0] >= 90:        # ~3 s with nothing at the usual spot: forget it (new level / window)
            _PL_XS.clear()
        if _PL_LAST[0] is not None:
            _PL_HOLD[0] += 1
            if _PL_HOLD[0] <= 8:
                pw, ph = _PL_SIZE[0]
                return Entity(CLS_ID["player"], _PL_LAST[0][0], _PL_LAST[0][1], pw, ph, 0.5)
            _PL_LAST[0] = None
        return None

    def __call__(self, rgb320: np.ndarray, hires_bgr: np.ndarray | None = None) -> Entities:
        cv2 = self._cv2
        img = np.asarray(rgb320)
        if img.ndim == 4:
            img = img[-1]
        h, w = img.shape[:2]
        bgr = np.ascontiguousarray(img[:, :, ::-1])
        ents: list[Entity] = []
        src = hires_bgr if hires_bgr is not None else bgr
        src = np.ascontiguousarray(np.asarray(src)[:, :, :3])
        hsv_src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)     # once per frame, shared below
        pl = self._player(hsv_src, src.shape[1], src.shape[0])
        if pl is not None:
            ents.append(pl)
        pbox = None
        if pl is not None and pl.conf > 0.6:
            pbox = (pl.x - pl.w / 2, pl.y - pl.h / 2, pl.x + pl.w / 2, pl.y + pl.h / 2)
        # coloured sprites near the icon's last spot are the icon (or its trail)
        sbox = (pl.x - pl.w / 2, pl.y - pl.h / 2, pl.x + pl.w / 2, pl.y + pl.h / 2) if pl is not None else None
        # Everything left of the icon is behind it (the blue line only looks ahead): detect
        # on the rest of the frame only -- ~20% fewer pixels for both detection passes.
        Wf = src.shape[1]
        cx0 = int(self.behind_cut * Wf) if pl is not None and pl.x - pl.w / 2 > self.behind_cut + 0.03 else 0
        fx = (lambda b: b) if cx0 == 0 else (lambda b: (cx0 + b * (Wf - cx0)) / Wf)
        tx = (lambda b: b) if cx0 == 0 else (lambda b: (b * Wf - cx0) / (Wf - cx0))
        src_c = src if cx0 == 0 else np.ascontiguousarray(src[:, cx0:])
        hsv_c = hsv_src if cx0 == 0 else np.ascontiguousarray(hsv_src[:, cx0:])
        pbox_c = None if pbox is None else (tx(pbox[0]), pbox[1], tx(pbox[2]), pbox[3])
        sbox_c = None if sbox is None else (tx(sbox[0]), sbox[1], tx(sbox[2]), sbox[3])
        dets = [(d[0], fx(d[1]), d[2], fx(d[3]), d[4]) + tuple(d[5:])
                for d in detect_hazards(src_c, player_box=pbox_c, hsv=hsv_c, ref_hsv=hsv_src)]
        self._spec_n += 1
        if self.specials_every <= 1 or self._spec_n % self.specials_every == 1 or self._spec_n == 1:
            solids = [(tx(d[1]), d[2], tx(d[3]), d[4]) for d in dets if int(d[0]) == CLS_ID["wall"]]
            specials = [(c, fx(a), y0, fx(b), y1) + tuple(r)
                        for c, a, y0, b, y1, *r in detect_specials(src_c, player_box=sbox_c, hsv=hsv_c, solids=solids,
                                                                   ref_hsv=hsv_src)]
        else:
            sc = float(getattr(self._tracker, "last_scroll", None) or 0.0) if self.temporal else 0.0
            specials = [(c, x0 + sc, y0, x1 + sc, y1, cf, hu) for c, x0, y0, x1, y1, cf, hu in self._spec_cache
                        if x1 + sc > 0.0]
        self._spec_cache = specials
        dets = [h for h in dets if not any(_iou(h[1:5], s[1:5]) > 0.3 for s in specials)]
        # a portal's dark middle looked like a spike INSIDE the portal (whole-level test)
        dets = [h for h in dets if not (h[0] == 0 and any(
            s[0] == CLS_ID["portal"] and s[1] <= (h[1] + h[3]) / 2 <= s[3] and s[2] <= (h[2] + h[4]) / 2 <= s[4]
            for s in specials))]
        if self.extra is not None:
            ydets = self._extra_dets(src, pbox)
            hsv_e = None
            for yd in ydets:
                c = int(yd[0])
                from gd_objects import BASE as _BASE, ICON_CLASSES as _IC
                if c in _IC:
                    continue             # icon classes: this (hybrid) mode keeps the built-in icon finder
                c = _BASE[c] if c < len(_BASE) else c
                yd = (c,) + tuple(yd[1:])
                if c in (0, 6):
                    if not any(_overlaps(yd, h) for h in dets):
                        dets.append(tuple(yd[:6]))
                else:
                    if not any(_overlaps(yd, s) for s in specials):
                        if hsv_e is None:
                            hsv_e = hsv_src
                        Hs, Ws = src.shape[:2]
                        reg = hsv_e[int(yd[2] * Hs):int(yd[4] * Hs) + 1, int(yd[1] * Ws):int(yd[3] * Ws) + 1].reshape(-1, 3)
                        reg = reg[reg[:, 1] > 80]
                        hue = float(np.median(reg[:, 0])) / 180.0 if len(reg) else -1.0
                        specials.append((c, yd[1], yd[2], yd[3], yd[4], yd[5], hue))
        scroll = None
        if self.temporal:
            ents.extend(self._tracker.update(dets))
            scroll = self._tracker.last_scroll
        else:
            ents.extend(Entity(int(d[0]), (d[1] + d[3]) / 2, (d[2] + d[4]) / 2,
                               d[3] - d[1], d[4] - d[2], d[5], pt=int(d[6]) if len(d) > 6 else -1) for d in dets)
        # orbs / pads / portals: colour + shape (see detect_specials)
        for s in specials:
            ents.append(Entity(int(s[0]), (s[1] + s[3]) / 2, (s[2] + s[4]) / 2,
                               s[3] - s[1], s[4] - s[2], s[5], s[6]))
        return Entities(ents, scroll)


def _overlaps(a, b) -> bool:
    """Same object? (IoU, or one box's centre inside the other)"""
    if _iou(a[1:5], b[1:5]) > 0.3:
        return True
    for p, q in ((a, b), (b, a)):
        cx, cy = (p[1] + p[3]) / 2, (p[2] + p[4]) / 2
        if q[1] <= cx <= q[3] and q[2] <= cy <= q[4]:
            return True
    return False


def make_detector(backend: str = "heuristic", icon_hue: tuple = (55, 95)):
    """heuristic = built-in rules; yolo = built-in rules + YOLO for what they miss;
    yolo-only = everything from YOLO, icon + its mode included (detect_yolo.YoloOnlyDetector)."""
    if backend == "yolo-only":
        from detect_yolo import YoloOnlyDetector
        try:
            import torch
            if not torch.cuda.is_available():
                print("[yolo-only] no CUDA GPU found: YOLO on the CPU is slow (~30+ ms a frame). "
                      "Install the CUDA build of PyTorch for your GPU.")
        except Exception:
            pass
        return YoloOnlyDetector(icon_hue=icon_hue)
    if backend == "yolo":
        from detect_yolo import YoloDetector
        y = YoloDetector()
        every = 1
        try:
            import torch
            every = 1 if torch.cuda.is_available() else 3
        except Exception:
            every = 3
        if every > 1:
            print("YOLO on CPU: running it every 3rd frame (a CUDA GPU lets it run every frame)")
        return HeuristicDetector(icon_hue=icon_hue, extra=y, extra_every=every)
    return HeuristicDetector(icon_hue=icon_hue)


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
