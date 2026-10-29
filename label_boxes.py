"""Box labeler for the YOLO detector.

    python label_boxes.py --rec recs/deadlocked.mp4
    python label_boxes.py --rec recs/deadlocked.mp4 --step 2     # visit every 2nd frame

Each frame opens PRE-FILLED with what the current detector finds. Fix it:
delete wrong boxes, draw the missing ones, then press D for the next frame.
Only frames you move on from with D/ENTER (or edit) are used for training,
so frames you skip past with W/S don't count.

MOUSE
  left-drag ............ draw a box of the current class
  right-click a box .... delete it
KEYS
  ICON: one box around the WHOLE icon, in the class of the mode it's in:
       7 cube   8 ship   9 ball   U ufo   V wave   R robot   P spider   N swing
  TYPES (press the key again to step to the next type; with the mouse over a box it
  changes that box): O every orb   K every pad   T mode portals
       G gravity / size / mirror / dual / teleport portals   E speed portals
       (pre-filled as a guess -- check it; I adds it on frames that don't have one)
  0 hazard (anything that kills you: spikes, saws, monsters, slime, deadly terrain)
  1 orb yellow   2 orb blue   3 orb pink/other   4 pad   5 portal   6 solid (blocks you can land on)
       (with the mouse over a box, a number key changes THAT box's class)
  D / ENTER ...... frame is done -> next frame (+step)     A ..... previous frame
  W / S .......... jump back / forward 30 frames (not marked done)
  X .............. delete the box under the mouse         C ..... clear all boxes on this frame
  F .............. re-run the pre-fill on this frame        H ..... show / hide this help
  Q / ESC ........ save and quit (it also autosaves every 10 frames)

Writes recs/<name>.boxes.json (coordinates are 0..1 fractions of the frame).
Aim for 40-60 done frames spread over the level, ~150-300 boxes in total.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from gd_objects import GROUP, GROUPS, NAMES as CLASSES

_GCOL = {"base": None, "orb": (0, 230, 255), "pad": (0, 165, 255), "icon": (0, 255, 0),
         "portal": (200, 0, 200), "mode_portal": (200, 0, 200), "other_portal": (255, 120, 200), "speed": (255, 255, 0)}
COLORS = [(0, 0, 255) if n == "hazard" else (200, 200, 200) if n == "solid" else _GCOL[GROUP[i]]
          for i, n in enumerate(CLASSES)]
# direct keys
ICON_KEYS = {ord("7"): 7, ord("8"): 8, ord("9"): 9, ord("u"): 10, ord("v"): 11, ord("r"): 12, ord("p"): 13,
             ord("n"): 14}
# group keys: press again to step through that group's types (over a box: changes the box)
GROUP_KEYS = {ord("o"): "orb", ord("k"): "pad", ord("t"): "mode_portal",
              ord("g"): "other_portal", ord("e"): "speed"}


def next_in_group(group: str, current: int) -> int:
    """The class after `current` in a group (the first one if current isn't in it)."""
    g = GROUPS[group]
    if group == "pad":
        g = [i for i in g if CLASSES[i] != "pad"] + [CLASSES.index("pad")]   # exact types first
    if group == "orb":
        order = ["orb_y", "orb_p", "orb_red", "orb_b", "orb_green", "orb_black", "orb_dash_green",
                 "orb_dash_pink", "orb_spider", "orb_teleport", "orb_toggle"]
        g = [CLASSES.index(n) for n in order]
    return g[(g.index(current) + 1) % len(g)] if current in g else g[0]


class Labels:
    """The label file: boxes + which frames are finished. Pure logic (testable)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.boxes: list[dict] = []
        self.done: set[int] = set()
        self.prefilled: set[int] = set()
        if os.path.exists(path):
            try:
                d = json.load(open(path))
                if isinstance(d, list):          # v1 file: plain box list
                    self.boxes = d
                    self.done = {int(b["fi"]) for b in d}
                else:
                    self.boxes = d.get("boxes", [])
                    self.done = set(int(f) for f in d.get("done", []))
                    self.prefilled = set(int(f) for f in d.get("prefilled", []))
            except Exception as e:
                print(f"could not read {path} ({e}); starting empty")

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"version": 2, "classes": CLASSES, "done": sorted(self.done),
                       "prefilled": sorted(self.prefilled), "boxes": self.boxes}, f, indent=1)
        os.replace(tmp, self.path)

    def on(self, fi: int) -> list[dict]:
        return [b for b in self.boxes if b["fi"] == fi]

    def add(self, fi: int, x0, y0, x1, y1, cls: int) -> None:
        x0, x1 = sorted((float(x0), float(x1)))
        y0, y1 = sorted((float(y0), float(y1)))
        if (x1 - x0) < 0.004 or (y1 - y0) < 0.004:
            return
        self.boxes.append(dict(fi=int(fi), x0=round(x0, 4), y0=round(y0, 4),
                               x1=round(x1, 4), y1=round(y1, 4), cls=int(cls)))

    def at(self, fi: int, x: float, y: float):
        """Smallest box on frame fi containing (x, y)."""
        hits = [b for b in self.on(fi) if b["x0"] <= x <= b["x1"] and b["y0"] <= y <= b["y1"]]
        return min(hits, key=lambda b: (b["x1"] - b["x0"]) * (b["y1"] - b["y0"])) if hits else None

    def delete_at(self, fi: int, x: float, y: float) -> bool:
        b = self.at(fi, x, y)
        if b is not None:
            self.boxes.remove(b)
            return True
        return False

    def clear(self, fi: int) -> None:
        self.boxes = [b for b in self.boxes if b["fi"] != fi]

    def prefill(self, fi: int, frame_bgr: np.ndarray, detector) -> int:
        """Propose boxes from the heuristic detector (classes map 1:1)."""
        import cv2
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        hi = cv2.resize(frame_bgr, (640, 360), interpolation=cv2.INTER_AREA)
        ents = detector(rgb, hi)
        n = 0
        for e in ents.items:
            c = int(e.cls)
            if 0 <= c <= 6:
                self.add(fi, e.x - e.w / 2, e.y - e.h / 2, e.x + e.w / 2, e.y + e.h / 2, guess_type(c, e))
                n += 1
        n += self.add_icon(fi, ents)
        self.prefilled.add(fi)
        return n

    def add_icon(self, fi: int, ents) -> int:
        """The icon box from the built-in finder, grown to the whole icon, as a GUESSED mode
        (wide = ship, else cube). Check it: fix the box / press the right mode key over it."""
        if any(GROUP[b["cls"]] == "icon" for b in self.on(fi)):
            return 0
        pl = ents.player()
        if pl is None:
            return 0
        # the finder only sees the icon's main colour: the whole icon is a bit bigger
        w, h = max(pl.w * 1.25, 0.04), max(pl.h * 1.25, 0.07)
        cls = 8 if w / max(h, 1e-6) * (720 / 1280) > 1.3 else 7
        self.add(fi, pl.x - w / 2, pl.y - h / 2, pl.x + w / 2, pl.y + h / 2, cls)
        return 1


def guess_type(c: int, e) -> int:
    """Pre-fill: a pad / portal's exact type guessed from its colour (check it!)."""
    from gd_objects import pad_kind_from_hue
    from gd_physics import portal_effect
    hue = float(getattr(e, "hue", -1.0))
    if c == 4:
        k = pad_kind_from_hue(hue)
        return CLASSES.index(k) if k else c
    if c == 5:
        fx = portal_effect(hue)
        if fx and fx[0] == "mode" and f"portal_{fx[1]}" in CLASSES:
            return CLASSES.index(f"portal_{fx[1]}")
        if fx and fx[0] == "g":
            return CLASSES.index("portal_grav_flip" if fx[1] == -1 else "portal_grav_normal")
    return c


class Video:
    """Random access to an mp4 with a small cache (sequential stepping is cheap)."""

    def __init__(self, path: str) -> None:
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(path)
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.pos = -1
        self.cache: dict[int, np.ndarray] = {}

    def get(self, fi: int):
        fi = max(0, min(self.n - 1, fi))
        if fi in self.cache:
            return self.cache[fi]
        if fi != self.pos + 1:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = self.cap.read()
        if not ok:
            return None
        self.pos = fi
        self.cache[fi] = fr
        if len(self.cache) > 64:
            self.cache.pop(min(self.cache))
        return fr


def main() -> int:
    p = argparse.ArgumentParser(description="label boxes for the YOLO detector")
    p.add_argument("--rec", required=True, help="recs/<level>.mp4 from record.py")
    p.add_argument("--step", type=int, default=3, help="frames to advance on D")
    p.add_argument("--no-prefill", action="store_true")
    p.add_argument("--icon-hue", type=int, nargs=2, default=(55, 95), metavar=("LO", "HI"))
    a = p.parse_args()
    import cv2
    from detect import HeuristicDetector

    vid = Video(a.rec)
    if vid.n == 0:
        print(f"can't read {a.rec}")
        return 1
    lab = Labels(os.path.splitext(a.rec)[0] + ".boxes.json")
    det = None if a.no_prefill else HeuristicDetector(temporal=False, icon_hue=tuple(a.icon_hue))
    fi = max(lab.done) + a.step if lab.done else 0
    fi = min(fi, vid.n - 1)
    st = dict(cls=0, drag=None, mouse=(0.0, 0.0), help=True, W=1280, H=720)
    win = "label boxes  (H = help)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(ev, x, y, flags, _):
        W, H = st["W"], st["H"]
        st["mouse"] = (x / W, y / H)
        if ev == cv2.EVENT_LBUTTONDOWN:
            st["drag"] = (x / W, y / H, x / W, y / H)
        elif ev == cv2.EVENT_MOUSEMOVE and st["drag"] is not None:
            d = st["drag"]
            st["drag"] = (d[0], d[1], x / W, y / H)
        elif ev == cv2.EVENT_LBUTTONUP and st["drag"] is not None:
            d = st["drag"]
            lab.add(fi_ref[0], d[0], d[1], x / W, y / H, st["cls"])
            lab.done.add(fi_ref[0])
            st["drag"] = None
        elif ev == cv2.EVENT_RBUTTONDOWN:
            if lab.delete_at(fi_ref[0], x / W, y / H):
                lab.done.add(fi_ref[0])

    fi_ref = [fi]
    cv2.setMouseCallback(win, on_mouse)
    since_save = 0
    print(__doc__)
    while True:
        fi_ref[0] = fi
        fr = vid.get(fi)
        if fr is None:
            break
        if det is not None and fi not in lab.prefilled and fi not in lab.done and not lab.on(fi):
            lab.prefill(fi, fr, det)
        H, W = fr.shape[:2]
        st["W"], st["H"] = W, H
        disp = fr.copy()
        for b in lab.on(fi):
            c = COLORS[b["cls"]]
            p0, p1 = (int(b["x0"] * W), int(b["y0"] * H)), (int(b["x1"] * W), int(b["y1"] * H))
            cv2.rectangle(disp, p0, p1, c, 2)
            cv2.putText(disp, CLASSES[b["cls"]], (p0[0], max(12, p0[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        if st["drag"] is not None:
            d = st["drag"]
            cv2.rectangle(disp, (int(d[0] * W), int(d[1] * H)), (int(d[2] * W), int(d[3] * H)), COLORS[st["cls"]], 1)
        n_done = len(lab.done)
        n_boxes = sum(1 for b in lab.boxes if b["fi"] in lab.done)
        no_icon = not any(GROUP[b["cls"]] == "icon" for b in lab.on(fi))
        top = (f"frame {fi + 1}/{vid.n} {'DONE' if fi in lab.done else ''}{'  NO ICON BOX (I)' if no_icon else ''}"
               f"   drawing: {st['cls']} {CLASSES[st['cls']]}"
               f"   done frames {n_done}  boxes {n_boxes}")
        cv2.rectangle(disp, (0, 0), (W, 26), (0, 0, 0), -1)
        cv2.putText(disp, top, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if st["help"]:
            lines = ["drag = box   right-click = delete   class key (over a box: change it)",
                     "D/ENTER = done+next   A = back   W/S = -/+30   X = delete under mouse",
                     "C = clear frame   F = re-fill   I = add the icon box   H = hide help   Q = save+quit",
                     "0 spike/deadly   6 block/solid   1 yellow orb  2 blue orb  3 pink orb",
                     "O orbs  K pads  T mode portals  G gravity/size/mirror/dual/teleport portals  E speed portals",
                     "   (press again = next type; over a box = change that box)",
                     "ICON (one box round the whole icon): 7 cube 8 ship 9 ball U ufo V wave R robot P spider N swing"]
            for k, ln in enumerate(lines):
                cv2.putText(disp, ln, (8, H - 12 - 20 * (len(lines) - 1 - k)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, disp)
        k = cv2.waitKey(30) & 0xFF
        if k == 255:
            continue
        mx, my = st["mouse"]
        if k in (ord("q"), 27):
            break
        elif k in (ord("d"), 13):
            lab.done.add(fi)
            fi = min(vid.n - 1, fi + a.step)
            since_save += 1
        elif k == ord("a"):
            fi = max(0, fi - a.step)
        elif k == ord("w"):
            fi = max(0, fi - 30)
        elif k == ord("s"):
            fi = min(vid.n - 1, fi + 30)
        elif k == ord("x"):
            if lab.delete_at(fi, mx, my):
                lab.done.add(fi)
        elif k == ord("c"):
            lab.clear(fi)
            lab.done.add(fi)
        elif k == ord("f") and det is not None:
            lab.clear(fi)
            lab.prefill(fi, fr, det)
        elif k == ord("h"):
            st["help"] = not st["help"]
        elif k == ord("i") and det is not None:
            ents_ = det(cv2.cvtColor(cv2.resize(fr, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB),
                        cv2.resize(fr, (640, 360), interpolation=cv2.INTER_AREA))
            if lab.add_icon(fi, ents_):
                lab.done.add(fi)
        elif k in GROUP_KEYS:
            b = lab.at(fi, mx, my)
            if b is not None:
                b["cls"] = next_in_group(GROUP_KEYS[k], b["cls"])
                lab.done.add(fi)
            else:
                st["cls"] = next_in_group(GROUP_KEYS[k], st["cls"])
        elif ord("0") <= k <= ord("6") or k in ICON_KEYS:
            c = ICON_KEYS.get(k, k - ord("0"))
            b = lab.at(fi, mx, my)
            if b is not None:
                b["cls"] = c
                lab.done.add(fi)
            else:
                st["cls"] = c
        if since_save >= 10:
            lab.save()
            since_save = 0
    lab.save()
    n_boxes = sum(1 for b in lab.boxes if b["fi"] in lab.done)
    print(f"saved {len(lab.done)} done frames, {n_boxes} boxes -> {lab.path}")
    print("next: python detect_yolo.py --train")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
