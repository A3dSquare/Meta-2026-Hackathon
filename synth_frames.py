"""Synthetic Geometry-Dash-like frames for testing the spike detector offline.

Renders 2560x1440 BGR frames that mimic the default GD look: a tiled blue
background, a blue ground with a light line on top, black-filled hazards
with a white outline (regular / small / mini spikes, ceiling spikes,
double and triple spike rows), outlined black blocks with spikes sitting on
them, and a green-cyan player cube (sometimes rotated mid-jump).

It is NOT a substitute for real gameplay footage -- it exists so detector
changes can be measured (recall per spike size, false positives) without the
game running:  python test_detect.py
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np

W, H = 2560, 1440
BLOCK = 135            # px per GD block at 1440p (320pt tall view / 30pt per block)
GROUND_Y = int(0.75 * H)
PLAYER_X = int(0.28 * W)

# (width, height) in blocks, measured from the in-game sprites
SPIKE_SIZES = {
    "regular": (1.0, 0.95),
    "small": (1.0, 0.45),
    "mini": (0.55, 0.45),
}


@dataclass
class GT:
    kind: str           # regular|small|mini|ceiling|block
    x0: float
    y0: float
    x1: float
    y1: float           # normalized 0..1 box


def _bg(rng: random.Random, hard: bool = False) -> np.ndarray:
    hue_shift = rng.randint(-12, 12)
    base = np.array([200 + hue_shift, 90 + hue_shift, 30], np.int32)  # BGR blue
    if hard:  # any level colour, including fairly dark ones
        hsv = np.uint8([[[rng.randint(0, 179), rng.randint(120, 255), rng.randint(110, 230)]]])
        base = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.int32)
    img = np.zeros((H, W, 3), np.int32)
    img[:] = base
    tile = BLOCK * 4
    off = rng.randint(0, tile)
    for ty in range(0, H, tile):
        for tx in range(-tile, W, tile):
            if ((tx + off) // tile + ty // tile) % 2:
                img[ty:ty + tile, max(0, tx + off):tx + off + tile] += np.array([18, 10, 4])
    # vertical brightness gradient like the default background
    grad = np.linspace(1.08, 0.82, H)[:, None, None]
    img = (img * grad).clip(0, 255)
    gy = GROUND_Y
    gcol = np.array([150 + hue_shift, 60, 20]) if not hard else base * rng.uniform(0.35, 0.8)
    img[gy:, :] = gcol * np.linspace(1.0, 0.7, H - gy)[:, None, None]
    img = img.astype(np.uint8)
    cv2.line(img, (0, gy), (W, gy), (255, 240, 230), 5)
    for gx in range(-(off % BLOCK), W, BLOCK * 2):  # ground tile seams
        cv2.line(img, (gx, gy + 6), (gx, H), (120, 45, 15), 3)
    return img


def _spike(img, cx, base_y, wb, hb, down=False):
    w, h = wb * BLOCK, hb * BLOCK
    if down:
        pts = np.array([[cx - w / 2, base_y], [cx + w / 2, base_y], [cx, base_y + h]])
    else:
        pts = np.array([[cx - w / 2, base_y], [cx + w / 2, base_y], [cx, base_y - h]])
    pts = pts.round().astype(np.int32)
    cv2.fillPoly(img, [pts], (8, 8, 8), lineType=cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (250, 250, 250), 5, lineType=cv2.LINE_AA)
    xs, ys = pts[:, 0], pts[:, 1]
    return xs.min(), ys.min(), xs.max(), ys.max()


def _block(img, x0, y0, nw=1, nh=1):
    x1, y1 = x0 + nw * BLOCK, y0 + nh * BLOCK
    cv2.rectangle(img, (x0, y0), (x1, y1), (10, 10, 10), -1)
    for i in range(nw):
        for j in range(nh):
            bx, by = x0 + i * BLOCK, y0 + j * BLOCK
            cv2.rectangle(img, (bx + 3, by + 3), (bx + BLOCK - 3, by + BLOCK - 3), (235, 235, 235), 5)
    return x0, y0, x1, y1


def _player(img, rng, y_center):
    s = BLOCK
    ang = rng.choice([0, 0, 0, 20, 45, 70])
    rect = ((PLAYER_X, y_center), (s * 0.95, s * 0.95), ang)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(img, [box], (0, 0, 0))
    inner = cv2.boxPoints(((PLAYER_X, y_center), (s * 0.72, s * 0.72), ang)).astype(np.int32)
    cv2.fillPoly(img, [inner], (200, 255, 60))   # green-cyan (BGR)
    core = cv2.boxPoints(((PLAYER_X, y_center), (s * 0.3, s * 0.3), ang)).astype(np.int32)
    cv2.fillPoly(img, [core], (40, 200, 255))


def _attempt_text(img, rng):
    cv2.putText(img, f"Attempt {rng.randint(1, 400)}", (rng.randint(600, 1400), int(0.42 * H)),
                cv2.FONT_HERSHEY_DUPLEX, 4.0, (255, 255, 255), 9, cv2.LINE_AA)


def make_frame(seed: int, hard: bool = False) -> tuple[np.ndarray, list[GT]]:
    """One 2560x1440 BGR frame + ground-truth hazard boxes (normalized)."""
    rng = random.Random(seed)
    img = _bg(rng, hard)
    gts: list[GT] = []
    x = PLAYER_X + int(BLOCK * rng.uniform(1.5, 3.0))
    if rng.random() < 0.3:
        _attempt_text(img, rng)
    while x < W - BLOCK * 2:
        r = rng.random()
        if r < 0.40:  # single / double / triple ground spikes of one size
            kind = rng.choice(["regular", "regular", "small", "small", "mini"])
            wb, hb = SPIKE_SIZES[kind]
            n = rng.choice([1, 1, 2, 3])
            for k in range(n):
                cx = x + wb * BLOCK * (k + 0.5)
                b = _spike(img, cx, GROUND_Y, wb, hb)
                gts.append(GT(kind, b[0] / W, b[1] / H, b[2] / W, b[3] / H))
            x += int(n * wb * BLOCK)
        elif r < 0.65:  # block platform, maybe spikes on top
            nw, nh = rng.choice([(1, 1), (2, 1), (3, 1), (1, 2)])
            y0 = GROUND_Y - nh * BLOCK
            b = _block(img, x, y0, nw, nh)
            gts.append(GT("block", b[0] / W, b[1] / H, b[2] / W, b[3] / H))
            if rng.random() < 0.5:
                kind = rng.choice(["regular", "small"])
                wb, hb = SPIKE_SIZES[kind]
                k = rng.randrange(nw)
                bs = _spike(img, x + BLOCK * (k + 0.5), y0, wb, hb)
                gts.append(GT(kind, bs[0] / W, bs[1] / H, bs[2] / W, bs[3] / H))
            x += nw * BLOCK
        elif r < 0.75:  # floating block with a ceiling spike hanging under it
            y0 = GROUND_Y - BLOCK * rng.choice([3, 4])
            b = _block(img, x, y0, 1, 1)
            gts.append(GT("block", b[0] / W, b[1] / H, b[2] / W, b[3] / H))
            bs = _spike(img, x + BLOCK / 2, y0 + BLOCK, 1.0, 0.95, down=True)
            gts.append(GT("ceiling", bs[0] / W, bs[1] / H, bs[2] / W, bs[3] / H))
            x += BLOCK
        x += int(BLOCK * rng.uniform(1.0, 4.0))
    if hard:
        # glow: blurred copy of the bright outlines added back (GD's object glow)
        bright = cv2.inRange(img, (200, 200, 200), (255, 255, 255))
        glow = cv2.GaussianBlur(bright, (0, 0), 9).astype(np.float32) * 0.45
        img = np.clip(img.astype(np.float32) + glow[:, :, None], 0, 255).astype(np.uint8)
        for _ in range(rng.randint(0, 40)):  # particles / pulse sparks
            px, py = rng.randint(0, W - 1), rng.randint(int(0.1 * H), GROUND_Y)
            cv2.circle(img, (px, py), rng.randint(4, 14), (255, 255, 255) if rng.random() < .5 else (0, 0, 0), -1)
    _player(img, rng, GROUND_Y - BLOCK // 2 - (0 if rng.random() < 0.6 else rng.randint(40, 220)))
    noise = np.random.default_rng(seed).normal(0, 3, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, gts


def downscale(full_bgr: np.ndarray):
    """Exactly what the live pipeline produces: 320x180 RGB + 640x360 BGR."""
    rgb320 = cv2.cvtColor(cv2.resize(full_bgr, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    bgr640 = cv2.resize(full_bgr, (640, 360), interpolation=cv2.INTER_AREA)
    return rgb320, bgr640
