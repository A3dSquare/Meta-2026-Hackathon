"""Entity detector for the REAL cross-level path (NOT SAM, NOT pixels).

Pipeline:
  full-res BGR --detect--> entities -> EntityBoxEnv (16 nearest entities
  as (dx, dy, w, h, class) vectors) -> PPO MlpPolicy -> SPACE.

Classes (level-agnostic, same on every GD level):
  0 spike (any color static triangle), 1 orb-yellow, 2 orb-blue,
  3 orb-pink, 4 pad, 5 portal, 6 wall/block edge.

Two backends:
  --backend heuristic (default, zero deps): white-triangle contour test
  for spikes + HSV hue bands for orbs/pads/portals. Runs ~1ms CPU at
  320x180. Good enough for Stereo/Jumper/Finger; ships TODAY.
  --backend yolo (opt-in): ultralytics YOLOv8n trained on boxes/ labels
  (see label_boxes.py). Needs `pip install ultralytics torch`. Heavier
  (~10-20ms CPU, GPU recommended) but learns new entity art.

Usage:
  python detect.py --probe recs/stereo.mp4   # dump entities per frame
  python detect.py --backend heuristic        # live camera probe
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

CLASSES = ["spike", "orb_y", "orb_b", "orb_p", "pad", "portal", "wall", "player"]
CLS_ID = {c: i for i, c in enumerate(CLASSES)}

# GD orb hues (OpenCV H 0-179): yellow ~25, blue ~105, pink ~140.
ORB_HUES = {"orb_y": (20, 32), "orb_b": (95, 115), "orb_p": (130, 155)}


# Player-tracker state. It lives at MODULE level, not on the detector,
# because preview_boxes.py constructs a fresh HeuristicDetector per probe --
# and EntityBoxEnv owns its own instance too. Module scope = one tracker
# shared by every detector user in the process (GUI + env agree on px).
_PL_LAST: list = [None]  # [(x, y)] last accepted cube center
_PL_HOLD: list = [0]     # [ticks] held-through-miss counter


@dataclass
class Entity:
    cls: int
    x: float  # center x, 0..1 of 320x180
    y: float  # center y, 0..1
    w: float  # width 0..1
    h: float  # height 0..1
    conf: float = 1.0


@dataclass
class Entities:
    items: list = field(default_factory=list)

    def player_pos(self) -> tuple[float, float]:
        """Player cube center (x, 0.30 default, y, 0.62 default)."""
        for e in self.items:
            if int(e.cls) == 7:
                return (float(e.x), float(e.y))
        return (0.30, 0.62)

    def nearest(self, k: int = 6, hold: bool = False) -> np.ndarray:
        """(k,9) float32: dx,dy,w,h,cls/7,edge,closing,hold,py -- nearest 6.

        cls is divided by 7 (8 classes now) so every feature sits Box(-2,2).
        dx/dy/edge are PLAYER-RELATIVE (detected cube, not fixed 0.30):
        ship sections and wave parts move the player, so fixed-origin
        math lies. py = player y (tells the net cube vs ship height).
        """
        px, py = self.player_pos()  # detected cube, not assumption
        ahead = [e for e in self.items if int(e.cls) != 7 and e.x >= px - 0.02]
        ahead.sort(key=lambda e: e.x)
        out = np.zeros((k, 9), dtype=np.float32)
        prev_dx = 0.0
        for i, e in enumerate(ahead[:k]):
            dx = round((e.x - px) * 64) / 64  # 1/64 grid (~5px @320): fine enough to time a jump
            edge = round((dx - (e.w / 2 + 0.02)) * 64) / 64
            closing = round((prev_dx - dx) * 32) / 32
            prev_dx = dx
            out[i] = (dx, round((e.y - py) * 32) / 32, e.w, e.h,
                      float(e.cls) / 7.0, edge, closing, 1.0 if hold else 0.0,
                      round(py * 32) / 32)
        return out


class HeuristicDetector:
    """~1ms CPU: contours + hue bands on the 320x180 policy frame."""

    def __init__(self) -> None:
        import cv2 as _cv2
        self._cv2 = _cv2
        # Temporal spike tracks: a real spike scrolls through the frame for
        # dozens of ticks, a particle/UI flash lives 1 tick. Only spikes seen
        # on 2+ consecutive ticks are emitted (1-tick latency, kills flashes).
        self._sp_tracks: list = []

    def __call__(self, rgb320: np.ndarray) -> Entities:
        cv2 = self._cv2
        img = np.asarray(rgb320)
        if img.ndim == 4:
            img = img[-1]
        h, w = img.shape[:2]
        bgr = img[:, :, ::-1].copy()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ents: list[Entity] = []
        # spikes: bright, low-saturation triangles (white on any bg)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        # GROUND-BAND MASK: everything at y >= 0.80 is floor/decoration --
        # triangles there are ground trim, never spikes (real spikes sit ON
        # the floor, centers ~0.55-0.75). Blank it in both spike masks so a
        # wide flat ground feature can never pass any triangle test.
        _inv_ground = np.full((h, w), 255, np.uint8)
        _inv_ground[int(0.80 * h):, :] = 0
        th = cv2.bitwise_and(th, _inv_ground)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if not (30 <= area <= 4000):
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) == 3:
                x, y, bw, bh = cv2.boundingRect(c)
                # Real GD spikes are ~as tall as wide (a long flat triangle
                # is ground/ceiling trim, not a spike) and sit on the floor
                # line (center y > 0.35). Without these gates the bright
                # ground edges at the screen corners passed as spikes.
                if not (0.5 <= bw / max(1, bh) <= 1.8):
                    continue
                if (y + bh / 2) / h < 0.35:
                    continue
                ents.append(Entity(0, (x + bw / 2) / w, (y + bh / 2) / h,
                                   bw / w, bh / h, 0.9))
        # spikes (path 2): DARK triangles. GD spikes are black-filled; on the
        # 320x180 frame the thin white outline often falls under the white
        # threshold, but the black body survives. Triangle test = convex hull
        # has 3 vertices and the blob fills ~half its bounding box (a block
        # fills ~100%, so blocks/ground are rejected by fill ratio).
        mask_d = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 70]))
        mask_d = cv2.morphologyEx(mask_d, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask_d = cv2.bitwise_and(mask_d, _inv_ground)
        cnts_d, _ = cv2.findContours(mask_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_d:
            area = cv2.contourArea(c)
            if not (12 <= area <= 1500):
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 4 or bh < 4 or not (0.5 <= bw / bh <= 1.8):
                continue
            if not (0.3 <= area / float(bw * bh) <= 0.68):
                continue
            if (y + bh / 2) / h < 0.35:  # same floor-line gate as bright path
                continue
            hull = cv2.convexHull(c)
            approx = cv2.approxPolyDP(hull, 0.08 * cv2.arcLength(hull, True), True)
            if len(approx) != 3:
                continue
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            if any(e.cls == 0 and abs(e.x - cx) < 0.02 and abs(e.y - cy) < 0.04 for e in ents):
                continue  # already found by the white-outline path
            ents.append(Entity(0, cx, cy, bw / w, bh / h, 0.85))
        # PLAYER cube: saturated square blob in the left-center band.
        # Cube/ship/robot are all saturated color squares ~14-22px; spikes
        # are triangles (already taken), orbs are round (hue bands below).
        # Ship sections fly higher, so y is free -- the policy reads it.
        # Measured on Stereo frame 0: the cube is a 13x7 saturated
        # green-cyan blob (area ~25, NOT 120+) sitting low-center. The old
        # 120-1200 gate + 0.6-1.7 aspect gate rejected every real cube.
        # Keep it tight in SPACE (x band + low-center y) instead of size.
        # TRACKED: the cube scrolls ~0.006/frame and jumps smoothly, so the
        # nearest in-gate blob to the last position wins; blobs far from it
        # (green-ish level art / pads snapping in) are rejected. A greenish
        # false blob used to steal px for a tick and move EVERY spike dx at
        # once -- the obs/teleport noise the policy could never model.
        mask_p = cv2.inRange(hsv, np.array([55, 100, 100]), np.array([95, 255, 255]))
        cnts_p, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _cands: list = []
        for c in cnts_p:
            area = cv2.contourArea(c)
            if not (15 <= area <= 800):
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            if not (0.15 <= cx <= 0.50 and cy >= 0.45):
                continue
            aspect = bw / max(1, bh)
            if not (0.5 <= aspect <= 3.0):
                continue
            if _PL_LAST[0] is not None:
                if abs(cx - _PL_LAST[0][0]) > 0.08 or abs(cy - _PL_LAST[0][1]) > 0.12:
                    continue  # teleport: not the cube, it's level art
            _cands.append((abs(cx - _PL_LAST[0][0]) if _PL_LAST[0] else 0.0,
                           Entity(CLS_ID["player"], cx, cy, bw / w, bh / h, 0.9)))
        if _cands:
            _cands.sort(key=lambda t: t[0])
            _pl = _cands[0][1]
            ents.append(_pl)
            _PL_LAST[0] = (_pl.x, _pl.y)
            _PL_HOLD[0] = 0
        elif _PL_LAST[0] is not None:
            # No in-gate blob this tick (death flash / occlusion): hold the
            # last cube position for a few ticks so px/py stay stable
            # instead of snapping to the (0.30, 0.62) default mid-jump.
            _PL_HOLD[0] += 1
            if _PL_HOLD[0] <= 8:
                ents.append(Entity(CLS_ID["player"], _PL_LAST[0][0],
                                   _PL_LAST[0][1], 0.05, 0.08, 0.5))
            else:
                _PL_LAST[0] = None
        # orbs/pads/portals: saturated hue blobs
        for cls_name, (lo, hi) in ORB_HUES.items():
            mask = cv2.inRange(hsv, np.array([lo, 90, 90]), np.array([hi, 255, 255]))
            cnts2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts2:
                area = cv2.contourArea(c)
                if not (20 <= area <= 3000):
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                ents.append(Entity(CLS_ID[cls_name], (x + bw / 2) / w, (y + bh / 2) / h,
                                   bw / w, bh / h, 0.8))
        # pads: green rectangles low on screen (heuristic fallback)
        mask_g = cv2.inRange(hsv, np.array([35, 90, 90]), np.array([85, 255, 255]))
        cnts3, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts3:
            area = cv2.contourArea(c)
            if not (40 <= area <= 2500):
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            _pl = next((e for e in ents if e.cls == CLS_ID["player"]), None)
            if _pl is not None and abs((x + bw / 2) / w - _pl.x) < _pl.w and abs((y + bh / 2) / h - _pl.y) < _pl.h:
                continue
            if y / h > 0.55:
                ents.append(Entity(CLS_ID["pad"], (x + bw / 2) / w, (y + bh / 2) / h,
                                    bw / w, bh / h, 0.7))
        # TEMPORAL CONFIRM (the flicker fix): bottom-corner flashes come from
        # particles/UI edges that exist for exactly 1 tick, while a real spike
        # scrolls left smoothly for dozens of ticks. A spike is only emitted
        # if a track near its (slightly scrolled) position was also seen last
        # tick. Cost: spikes appear 1 tick (~16ms) later; still ~2s of warning.
        _now = [e for e in ents if int(e.cls) == 0]
        _confirmed: list = []
        for e in _now:
            if any(abs((e.x - t[0]) - 0.006) < 0.03 and abs(e.y - t[1]) < 0.05
                   for t in self._sp_tracks):
                _confirmed.append(e)
        self._sp_tracks = [(e.x, e.y) for e in _now]
        ents = [e for e in ents if int(e.cls) != 0] + _confirmed
        return Entities(ents)


def make_detector(backend: str = "heuristic"):
    if backend == "yolo":
        from detect_yolo import YoloDetector
        return YoloDetector()
    return HeuristicDetector()


def main() -> int:
    p = argparse.ArgumentParser(description="entity detector probe")
    p.add_argument("--probe", default=None, help="mp4 to scan (default: live camera)")
    p.add_argument("--backend", default="heuristic", choices=("heuristic", "yolo"))
    p.add_argument("--dump", default="./detect_probe.png")
    args = p.parse_args()
    det = make_detector(args.backend)
    if args.probe:
        import cv2
        cap = cv2.VideoCapture(args.probe)
        n = det_t = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            small = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (320, 180))
            t0 = time.perf_counter()
            ents = det(small)
            det_t += (time.perf_counter() - t0) * 1000.0
            n += 1
        cap.release()
        print(f"frames={n} mean={det_t / max(1, n):.2f}ms backend={args.backend}")
        return 0
    from capture import GDCapture
    cap = GDCapture()
    try:
        cap.start()
        for _ in range(10):
            cap.read()
        obs = cap.read()
        ents = det(obs)
        print(f"live entities={len(ents.items)} " +
              ", ".join(f"{CLASSES[e.cls]}@({e.x:.2f},{e.y:.2f})" for e in ents.items[:8]))
    finally:
        cap.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())