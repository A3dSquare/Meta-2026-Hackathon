"""Hazards you teach it: pictures of objects that kill (or that only LOOK deadly).

The built-in detector finds spikes and blocks by rules (black triangle with an outline,
...). Some levels' deadly things don't fit those rules -- saws, monsters, coloured
spikes. This adds them by LOOK: any picture in

    hazards/            objects that kill       (you add them: --grab hazard)
    hazards/learned/    objects that killed it  (it adds them itself, see below)
    hazards/safe/       harmless things the detector calls hazards (--grab safe)

is searched for in every other frame ahead of the icon (template matching). A match of
a hazard picture becomes a hazard box -- the predicted path (cyan line), the safety
shield and the brain all treat it like a spike. A match of a safe picture removes the
detector's hazard boxes there.

Learning from deaths: after each death it looks at the crash moment. If the detector
had NOTHING deadly at the icon (no spike, no wall it ran into) and the predicted path
called it safe, whatever was right in front of the icon probably killed it. That
picture is kept as a candidate (hazards/candidates/); when a second unexplained death
happens in front of something that looks the same, it becomes a learned hazard. Wrong
one? Delete the file in hazards/learned/ (or move it into hazards/safe/).

Grab a picture yourself (GD running; pause it with the object on screen):

    python hazards.py --grab hazard      # drag a box around the deadly object, ENTER
    python hazards.py --grab safe        # ... around a harmless one
    python hazards.py --list
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

BLOCK_FRAC_H = 30.0 / 320.0            # one block as a fraction of the frame height
FRAME_W, FRAME_H = 640, 360            # pictures are kept at this frame size
MATCH = 0.78                           # similarity needed for a match (0..1)
SAME = 0.70                            # two death pictures this similar = the same object
MAX_TEMPLATES = 8
MAX_HITS = 6                           # more matches than this in one frame = background, not an object


def _gray(bgr):
    import cv2
    g = cv2.cvtColor(np.ascontiguousarray(bgr[:, :, :3]), cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (3, 3), 0)


class HazardMemory:
    def __init__(self, root: str = "hazards", every: int = 2) -> None:
        self.root = root
        self.every = max(1, int(every))
        self._n = 0
        self._cache: list = []           # last frame's matches (normalised boxes, kind)
        self._load_t = 0.0
        self.templates: list = []        # (kind, name, gray)
        self.flood: dict = {}            # name -> frames where it matched everywhere
        self._last_msg = ""
        os.makedirs(os.path.join(root, "learned"), exist_ok=True)
        os.makedirs(os.path.join(root, "safe"), exist_ok=True)
        os.makedirs(os.path.join(root, "candidates"), exist_ok=True)
        self.meta_path = os.path.join(root, "candidates", "memory.json")
        try:
            with open(self.meta_path) as f:
                self.meta = json.load(f)
        except Exception:
            self.meta = {"candidates": []}
        self.reload()

    # -- pictures ------------------------------------------------------------
    def reload(self) -> None:
        """(Re)read the picture folders (cheap; done every ~10 s so deleting a file works live)."""
        import cv2
        out = []
        for kind, d in (("hazard", self.root), ("hazard", os.path.join(self.root, "learned")),
                        ("safe", os.path.join(self.root, "safe"))):
            try:
                names = sorted(n for n in os.listdir(d) if n.lower().endswith(".png"))
            except OSError:
                continue
            for n in names:
                img = cv2.imread(os.path.join(d, n))
                if img is None or min(img.shape[:2]) < 6:
                    continue
                out.append((kind, os.path.join(d, n), _gray(img)))
        self.templates = out[:MAX_TEMPLATES * 2]
        self._load_t = time.time()

    @property
    def active(self) -> bool:
        return bool(self.templates)

    # -- every frame -----------------------------------------------------------
    def apply(self, ents, hires_bgr, scroll: float | None = None):
        """Add matches of hazard pictures to ents, drop detector hazards on safe pictures."""
        if time.time() - self._load_t > 10.0:
            self.reload()
        if not self.templates or hires_bgr is None:
            return ents
        from detect import CLS_ID, Entity
        self._n += 1
        if self._n % self.every == 0 or not self._cache:
            self._cache = self._match(hires_bgr, ents)
        elif scroll:
            self._cache = [(k, x0 + scroll, y0, x1 + scroll, y1) for k, x0, y0, x1, y1 in self._cache
                           if x1 + scroll > 0]
        hz = [(x0, y0, x1, y1) for k, x0, y0, x1, y1 in self._cache if k == "hazard"]
        safe = [(x0, y0, x1, y1) for k, x0, y0, x1, y1 in self._cache if k == "safe"]
        items = []
        for e in ents.items:
            if int(e.cls) == CLS_ID["spike"] and any(x0 <= e.x <= x1 and y0 <= e.y <= y1 for x0, y0, x1, y1 in safe):
                continue
            items.append(e)
        for x0, y0, x1, y1 in hz:
            if any(int(e.cls) == 0 and abs(e.x - (x0 + x1) / 2) < (x1 - x0) / 2
                   and abs(e.y - (y0 + y1) / 2) < (y1 - y0) / 2 for e in items):
                continue                 # the detector has it already
            items.append(Entity(CLS_ID["spike"], (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0.8, pt=-1))
        ents.items = items
        return ents

    def _match(self, hires_bgr, ents) -> list:
        import cv2
        img = np.asarray(hires_bgr)
        H, W = img.shape[:2]
        px = ents.player_pos()[0]
        x_lo = max(0, int((px - 0.06) * W))               # ahead of the icon (and a bit behind)
        g = _gray(img[:, x_lo:])
        out = []
        for kind, name, t in self.templates:
            th, tw = t.shape[:2]
            if th >= g.shape[0] or tw >= g.shape[1]:
                continue
            r = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.nonzero(r >= MATCH)
            if len(ys) == 0:
                continue
            # non-maximum suppression: one box per object
            order = np.argsort(-r[ys, xs])
            picked = []
            for i in order:
                x, y = int(xs[i]), int(ys[i])
                if all(abs(x - a) > tw * 0.6 or abs(y - b) > th * 0.6 for a, b in picked):
                    picked.append((x, y))
                if len(picked) > MAX_HITS:
                    break
            if len(picked) > MAX_HITS:          # matches everywhere: it's background
                self.flood[name] = self.flood.get(name, 0) + 1
                if self.flood[name] == 30:
                    print(f"[hazards] {name} matches all over the screen (background, not an object) -- "
                          f"ignoring it. Delete that picture.")
                continue
            if self.flood.get(name, 0) >= 30:
                continue
            for x, y in picked:
                out.append((kind, (x_lo + x) / W, y / H, (x_lo + x + tw) / W, (y + th) / H))
        return out

    # -- after a death -----------------------------------------------------------
    def on_death(self, buf: list, lag: int) -> None:
        """buf: the env's death-snapshot records (oldest first); lag: steps from the crash
        to the death signal. Learns from crashes the detector can't explain."""
        try:
            self._on_death(buf, lag)
        except Exception as ex:          # never let this break training
            print(f"[hazards] couldn't learn from this death: {ex!r}")

    def _on_death(self, buf: list, lag: int) -> None:
        import cv2
        if not buf:
            return
        crash = max(0, len(buf) - 1 - int(lag))
        i = max(0, crash - 1)            # just before the contact
        rec = buf[i]
        pp = rec.get("player_pos")
        if pp is None or not rec.get("player"):
            return
        px, py = pp
        bw, bh = BLOCK_FRAC_H * FRAME_H / FRAME_W, BLOCK_FRAC_H
        # explained? a known hazard touching the icon, or a block it ran into
        for b in rec["boxes"]:
            c, x, y, w, h = b[:5]
            if c == 0 and abs(x - px) < w / 2 + 0.9 * bw and abs(y - py) < h / 2 + 0.9 * bh:
                return
            if c == 6 and 0 <= (x - w / 2) - px < 1.0 * bw and abs(y - py) < h / 2 + 0.4 * bh:
                return
        # only if the predicted path thought it was fine (else it was a known risk)
        sr = rec.get("search") or {}
        if sr and max(sr.get("1", 1.0), sr.get("00", 1.0), sr.get("01", 1.0)) < 0.9:
            return
        frame = rec.get("frame")
        if frame is None:
            return
        H, W = frame.shape[:2]
        x0, x1 = int((px + 0.4 * bw) * W), int((px + 1.9 * bw) * W)
        y0, y1 = int((py - 0.9 * bh) * H), int((py + 0.9 * bh) * H)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return
        patch = np.ascontiguousarray(frame[y0:y1, x0:x1])
        if (W, H) != (FRAME_W, FRAME_H):
            patch = cv2.resize(patch, (max(6, int(patch.shape[1] * FRAME_W / W)),
                                       max(6, int(patch.shape[0] * FRAME_H / H))))
        # trim to the object itself: the box around its edges
        g = _gray(patch)
        ed = cv2.Canny(g, 40, 120)
        ed[(ed > 0).mean(axis=1) > 0.7] = 0      # long horizontal lines (the ground's edge)
        ed[:, (ed > 0).mean(axis=0) > 0.85] = 0  # and vertical ones (a wall's side)
        ys_, xs_ = np.nonzero(ed)
        blk_px = BLOCK_FRAC_H * FRAME_H
        if len(ys_) < 10:
            return                       # empty background: nothing to learn (a timing death)
        tx0, tx1, ty0, ty1 = max(0, xs_.min() - 2), min(g.shape[1], xs_.max() + 3), max(0, ys_.min() - 2), min(g.shape[0], ys_.max() + 3)
        if (tx1 - tx0) < 0.25 * blk_px or (ty1 - ty0) < 0.25 * blk_px:
            return                       # a line (the ground's edge...), not an object
        patch, g = patch[ty0:ty1, tx0:tx1], g[ty0:ty1, tx0:tx1]
        if float(g.std()) < 12.0:
            return
        # something that repeats all over its own frame (ground texture, background tiles)
        # can't be "the thing that killed it"
        fg = _gray(cv2.resize(frame, (FRAME_W, FRAME_H)) if (W, H) != (FRAME_W, FRAME_H) else frame)
        r = cv2.matchTemplate(fg, g, cv2.TM_CCOEFF_NORMED)
        hits = 0
        for _ in range(MAX_HITS + 2):
            _mn, mx, _l, loc = cv2.minMaxLoc(r)
            if mx < MATCH:
                break
            hits += 1
            x_, y_ = loc
            r[max(0, y_ - g.shape[0] // 2):y_ + g.shape[0] // 2 + 1, max(0, x_ - g.shape[1] // 2):x_ + g.shape[1] // 2 + 1] = -1
        if hits > 3:
            return
        cdir = os.path.join(self.root, "candidates")
        best, best_c = 0.0, None
        for cand in self.meta["candidates"]:
            img = cv2.imread(os.path.join(cdir, cand["file"]))
            if img is None:
                continue
            cg = _gray(cv2.resize(img, (g.shape[1], g.shape[0])))
            s = float(cv2.matchTemplate(g, cg, cv2.TM_CCOEFF_NORMED).max())
            if s > best:
                best, best_c = s, cand
        if best_c is not None and best >= SAME:
            best_c["count"] += 1
            if best_c["count"] >= 2 and not best_c.get("promoted"):
                best_c["promoted"] = True
                dst = os.path.join(self.root, "learned", best_c["file"])
                cv2.imwrite(dst, cv2.imread(os.path.join(cdir, best_c["file"])))
                print(f"[hazards] learned a new hazard from {best_c['count']} unexplained deaths: {dst} "
                      f"(wrong? delete it, or move it into {os.path.join(self.root, 'safe')})")
                self.reload()
        else:
            name = f"death_{int(time.time())}.png"
            cv2.imwrite(os.path.join(cdir, name), patch)
            self.meta["candidates"].append({"file": name, "count": 1})
            self.meta["candidates"] = self.meta["candidates"][-40:]
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f, indent=1)


# --------------------------------------------------------------------------- #
# Command line: grab a picture from the running game / list what it knows
# --------------------------------------------------------------------------- #
def _grab(kind: str, root: str) -> None:
    import cv2
    from capture import GDCapture
    cap = GDCapture().start()
    try:
        frame = None
        for _ in range(30):
            cap.read()
            frame = getattr(cap, "last_hires", None)
            if frame is not None:
                break
            time.sleep(0.03)
        if frame is None:
            full = getattr(cap, "last_fullres", None)
            frame = None if full is None else cv2.resize(np.asarray(full)[:, :, :3], (FRAME_W, FRAME_H))
        if frame is None:
            raise SystemExit("couldn't capture the game screen")
        frame = np.ascontiguousarray(frame)
    finally:
        try:
            cap.stop()
        except Exception:
            pass
    big = cv2.resize(frame, (FRAME_W * 2, FRAME_H * 2), interpolation=cv2.INTER_NEAREST)
    print(f"Drag a box TIGHTLY around one {'deadly' if kind == 'hazard' else 'harmless'} object, "
          f"then press ENTER (C = cancel).")
    x, y, w, h = cv2.selectROI(f"mark a {kind}", big, showCrosshair=False)
    cv2.destroyAllWindows()
    if w < 6 or h < 6:
        print("cancelled")
        return
    crop = frame[y // 2:(y + h) // 2, x // 2:(x + w) // 2]
    d = root if kind == "hazard" else os.path.join(root, "safe")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{kind}_{int(time.time())}.png")
    cv2.imwrite(path, crop)
    print(f"saved {path} -- training picks it up within ~10 s (no restart needed)")


def main() -> int:
    p = argparse.ArgumentParser(description="teach the detector objects by picture")
    p.add_argument("--grab", choices=("hazard", "safe"), help="capture the game and mark an object")
    p.add_argument("--list", action="store_true")
    p.add_argument("--root", default="hazards")
    a = p.parse_args()
    if a.grab:
        _grab(a.grab, a.root)
    m = HazardMemory(a.root)
    if a.list or not a.grab:
        for kind, name, t in m.templates:
            print(f"  {kind:6s} {name}  ({t.shape[1]}x{t.shape[0]} px)")
        n = len(m.meta["candidates"])
        print(f"{len(m.templates)} pictures in use, {n} death candidates in {os.path.join(a.root, 'candidates')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
