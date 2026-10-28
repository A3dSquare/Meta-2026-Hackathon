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


@dataclass
class Entity:
    cls: int
    x: float  # center x, 0..1
    y: float  # center y, 0..1
    w: float  # width 0..1
    h: float  # height 0..1
    conf: float = 1.0
    hue: float = -1.0  # mean OpenCV hue / 180 for coloured objects (orbs/pads/portals), -1 = n/a


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
    # outlines: white, or tinted by the level colour (e.g. lavender on purple) --
    # so "bright" = clearly brighter AND clearly less saturated than the background
    med_s = float(np.median(S[::4, ::4]))
    bright = (((V >= 185) & (S <= 75))
              | ((V >= max(215.0, med_v + 35)) & (S <= min(170.0, 0.75 * med_s)))).astype(np.uint8)
    dark[:ui_cut] = 0
    bright[:ui_cut] = 0
    # the ground line is one long bright row: remove it, or every pillar and
    # spike standing on the ground merges into one screen-wide outline blob
    rows = bright.mean(axis=1) > 0.7
    for yy in np.nonzero(rows)[0]:
        bright[max(0, yy - 1):yy + 2] = 0
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
                    spikes_here.append((0, x + l, ya, x + r + 1, yb + 1, conf))
            if solid_check and bw >= 1.0 * blk and bh <= 0.8 * blk and area / float(bw * bh) < 0.75:
                # a row of ground thorns: one wide, short, jagged black band
                tv = top[colv]
                if tv.size >= 4:
                    peaks = int(np.sum((tv[1:-1] < tv[:-2]) & (tv[1:-1] <= tv[2:])))
                    if peaks >= 3 and (y + bh / 2) < 0.94 * H:
                        spikes_here.append((0, x, y, x + bw, y + bh, 0.8))
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
                            spikes_here.append((0, xa, ya0, xb + 1, ya1 + 1, 0.75))
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
                        depth = int(bot[c0:c1 + 1].max() - (y + t0))
                        if depth >= 0.35 * blk:
                            walls.append((6, x + c0, y + t0, x + c1 + 1, y + t0 + depth + 1, 0.85))
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
    return _merge_hazards(found, W, H, blk)


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
                if joined and abs(b[2] - joined[-1][2]) <= 0.15 * blk and b[1] - joined[-1][3] <= 1.2 * blk:
                    j = joined[-1]
                    j[3], j[4] = max(j[3], b[3]), max(j[4], b[4])
                    continue
                joined.append(b)
            stacked: list = []   # pieces of one pillar split by the icon mask -> rejoin
            for b in sorted(joined, key=lambda b: b[2]):
                hit = None
                for s in stacked:
                    ov = min(s[3], b[3]) - max(s[1], b[1])
                    if ov >= 0.7 * min(s[3] - s[1], b[3] - b[1]) and b[2] - s[4] <= 0.4 * blk:
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
            out.append((b[0], b[1] / W, b[2] / H, b[3] / W, b[4] / H, b[5]))
    return out


def _hue_dist(h, ref):
    d = np.abs(h.astype(np.int16) - int(ref))
    return np.minimum(d, 180 - d)


def _orb_class(hue: float) -> int:
    """Nearest of the three orb classes by hue (others keep their hue feature)."""
    cands = {CLS_ID["orb_y"]: 27, CLS_ID["orb_b"]: 105, CLS_ID["orb_p"]: 150}
    return min(cands, key=lambda c: min(abs(hue - cands[c]), 180 - abs(hue - cands[c])))


def detect_specials(bgr: np.ndarray, player_box=None) -> list:
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
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hh, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sub = hsv[::4, ::4].reshape(-1, 3)
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
            if hb >= 1.7 and hb >= 1.6 * wb and wb >= 0.3:
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
                    if circ >= 0.6 and area >= 0.45 * bw * bh:
                        det = (_orb_class(hue), *box, 0.8, hue / 180.0)
            elif which == "colour" and hb <= 1.0 and 0.45 <= wb <= 2.0 and wb >= 1.25 * hb:
                # pad: flat colour strip, often under a big glow dome
                det = (CLS_ID["pad"], *box, 0.7, hue / 180.0)
            if det is not None and det[0] in (1, 2, 3):
                cxd, cyd = (det[1] + det[3]) / 2 * W, (det[2] + det[4]) / 2 * H
                if any(o[0] in (1, 2, 3) and abs((o[1] + o[3]) / 2 * W - cxd) < 0.7 * blk
                       and abs((o[2] + o[4]) / 2 * H - cyd) < 0.7 * blk for o in out):
                    det = None   # particles around an orb already found
            if det is not None and not any(_iou(det[1:5], o[1:5]) > 0.3 for o in out):
                out.append(det)
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

    def __init__(self, temporal: bool = True, icon_hue: tuple = (55, 95), extra=None,
                 extra_every: int = 1) -> None:
        import cv2 as _cv2
        self._cv2 = _cv2
        self.temporal = temporal
        self.icon_hue = icon_hue   # OpenCV hue range of your icon's main colour
        self._tracker = _HazardTracker()
        # extra: a learned detector (detect_yolo.YoloDetector) whose boxes are ADDED
        # where the built-in rules found nothing. extra_every > 1 runs it every N
        # frames and slides its last boxes with the scroll in between (for CPUs).
        self.extra = extra
        self.extra_every = max(1, int(extra_every))
        self._extra_cache: list = []
        self._extra_n = 0

    def reset(self) -> None:
        """New attempt: forget hazard tracks (the screen jumped back to 0%)."""
        self._tracker.reset()
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
        for c in cnts_p:
            area = cv2.contourArea(c) / blk2
            if not (0.03 <= area <= 4.0):      # wave/mini icons are small, ships wide
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            # whole height: ships fly high, upside-down cubes run on the ceiling
            # GD's camera keeps the icon at a fixed x (~0.37 of the screen)
            if not (0.25 <= cx <= 0.48 and 0.07 <= cy <= 0.97):
                continue
            if not (0.3 <= bw / max(1, bh) <= 3.5):
                continue
            if _PL_LAST[0] is not None:
                if abs(cx - _PL_LAST[0][0]) > 0.08 or abs(cy - _PL_LAST[0][1]) > 0.15:
                    continue
            d = (abs(cx - _PL_LAST[0][0]) + abs(cy - _PL_LAST[0][1]) if _PL_LAST[0]
                 else 3 * abs(cx - 0.37) - 0.002 * area)
            cands.append((d, Entity(CLS_ID["player"], cx, cy, max(bw / w, 0.03), max(bh / h, 0.05), 0.9)))
        if cands:
            cands.sort(key=lambda t: t[0])
            pl = cands[0][1]
            _PL_LAST[0] = (pl.x, pl.y)
            _PL_SIZE[0] = (pl.w, pl.h)
            _PL_HOLD[0] = 0
            return pl
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
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ents: list[Entity] = []
        src = hires_bgr if hires_bgr is not None else bgr
        src = np.ascontiguousarray(np.asarray(src)[:, :, :3])
        hsv_src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV) if hires_bgr is not None else hsv
        pl = self._player(hsv_src, src.shape[1], src.shape[0])
        if pl is not None:
            ents.append(pl)
        pbox = None
        if pl is not None and pl.conf > 0.6:
            pbox = (pl.x - pl.w / 2, pl.y - pl.h / 2, pl.x + pl.w / 2, pl.y + pl.h / 2)
        # coloured sprites near the icon's last spot are the icon (or its trail)
        sbox = (pl.x - pl.w / 2, pl.y - pl.h / 2, pl.x + pl.w / 2, pl.y + pl.h / 2) if pl is not None else None
        dets = detect_hazards(src, player_box=pbox)
        specials = detect_specials(src, player_box=sbox)
        dets = [h for h in dets if not any(_iou(h[1:5], s[1:5]) > 0.3 for s in specials)]
        if self.extra is not None:
            ydets = self._extra_dets(src, pbox)
            hsv_e = None
            for yd in ydets:
                c = int(yd[0])
                if c in (0, 6):
                    if not any(_overlaps(yd, h) for h in dets):
                        dets.append(tuple(yd[:6]))
                else:
                    if not any(_overlaps(yd, s) for s in specials):
                        if hsv_e is None:
                            hsv_e = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
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
                               d[3] - d[1], d[4] - d[2], d[5]) for d in dets)
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
    """heuristic = built-in rules; yolo = built-in rules + YOLO for what they miss."""
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
