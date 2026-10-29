"""Route memory: per-level practice that the brain never trains on.

A GD level is the same every attempt: an obstacle is always the same time after the
attempt starts. When the bot keeps dying at the same moment, this replays the best run
it has had on that level up to just before that spot and tries a different press
timing there -- one change per attempt -- like a player practising a hard part.

What it remembers, per level (routes/<level id>.json, kept forever):
  - what the level looks like at the start (to recognise it -- no level names needed)
  - the best run so far: its presses, and how far it got
  - every spot it got stuck at: how often it died there, every change it tried there
    (so a later session tries something NEW), and the change that got it through

Recognising the level: during the first second of an attempt it takes a few small
pictures of the screen (the level's start: blocks, spikes, colours; the icon and the
"Attempt N" text are left out) and compares them with every level it has seen. Close
enough = that level; else it's a new level. Coming back weeks later works the same.
Rename a level by editing "name" in its file (only for you to read).

Kept apart from the brain:
  - it only takes over on every OTHER attempt, and only once stuck at a spot (the other
    attempts are the brain + shield playing and learning normally)
  - attempts it controlled are NOT given to the brain to learn from, and don't count
    for the brain's "best" (best_boxes.pt)
  - --no-route turns it off.
"""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np

PRINT_TIMES = (0.3, 0.6, 0.9)   # seconds into an attempt where the level's look is sampled
PRINT_W, PRINT_H = 48, 27


def level_print(frame_bgr: np.ndarray) -> np.ndarray:
    """A small colour picture of the level, without the parts that change between
    attempts or players: the top (progress bar, "Attempt N"), the icon's column."""
    import cv2
    f = np.asarray(frame_bgr)[:, :, :3]
    th = cv2.resize(np.ascontiguousarray(f), (PRINT_W, PRINT_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    m = np.ones((PRINT_H, PRINT_W), bool)
    m[: int(0.33 * PRINT_H)] = False                       # top third: % bar, "Attempt N" text
    m[:, int(0.26 * PRINT_W):int(0.50 * PRINT_W)] = False  # the icon (skins differ)
    th[~m] = 0
    return th


def print_distance(a, b) -> float:
    """Mean colour difference (0..255) between two level pictures (masked parts are 0 in both)."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    n = float((a.sum(axis=2) > 0).sum() + 1)
    return float(np.abs(a - b).sum() / (3 * n))


def _merge(iv):
    out = []
    for s, e in sorted(iv):
        if e <= s:
            continue
        if out and s <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


class RouteMemory:
    WINDOW = 1.0      # seconds before the stuck spot where changes are tried
    TOL = 0.25        # deaths this close (s) count as "the same spot"
    AFTER = 0.35      # keep following the change this long past the spot
    STUCK = 4         # deaths at a spot before it starts practising it (the brain gets the first tries)
    MATCH = 6.0       # picture difference below this = the same level (tested: the same level
                      # 1-2 frames off ~0.4; another level ~27; the same level 10 s in ~11)
    SHIFTS = (-1, 1, -2, 2, -3, 3, -4, 4, -6, 6)

    def __init__(self, folder: str = "routes", dt: float = 1 / 30, seed: int = 0, verbose: bool = True,
                 fixed_level: str | None = None, play: bool = False) -> None:
        # play: --play (beat the level, no training): EVERY attempt follows the best run, and
        # the stuck spot is practised from the first death there
        self.play = play
        self.folder = folder
        self.dt = dt
        self.rng = random.Random(seed)
        self.verbose = verbose
        self.fixed_level = fixed_level        # tests: skip recognising the level
        os.makedirs(folder, exist_ok=True)
        self.levels = self._load_all()
        self.level = None                     # the current level's record (dict)
        self._prints: list = []
        self._recognised = False
        self._cur: list = []                  # this attempt's presses: [start, end]
        self._down = None
        self.plan = None                      # (replay_until, window_end, press intervals, change)
        self.controlled = False               # this attempt's moves came from here (not the brain)
        self._parity = 0
        self.attempt_was_route = False

    # -- storage -------------------------------------------------------------------
    def _load_all(self) -> dict:
        out = {}
        for fn in os.listdir(self.folder):
            if fn.endswith(".json"):
                try:
                    d = json.load(open(os.path.join(self.folder, fn)))
                    d["prints"] = [np.asarray(p, np.float32) for p in d.get("prints", [])]
                    for sp in d.get("spots", {}).values():
                        sp["drift"] = 0            # a new session: timing may be steadier now
                    out[d["id"]] = d
                except Exception as ex:
                    print(f"[route] couldn't read {fn}: {ex!r}")
        if out and self.verbose:
            print(f"[route] remembers {len(out)} level(s): " + ", ".join(
                f"{d.get('name') or d['id']} ({d.get('best_T', 0):.1f} s best)" for d in out.values()))
        return out

    def _save(self, lv: dict) -> None:
        d = dict(lv)
        d["prints"] = [np.round(p, 1).tolist() for p in lv["prints"]]
        path = os.path.join(self.folder, f"{lv['id']}.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, path)

    def _new_level(self, prints) -> dict:
        lid = "level_" + time.strftime("%Y%m%d_%H%M%S")
        lv = {"id": lid, "name": "", "created": time.time(), "last_seen": time.time(), "attempts": 0,
              "prints": list(prints), "best": [], "best_T": 0.0, "spots": {}}
        self.levels[lid] = lv
        if self.verbose:
            print(f"[route] a level it hasn't seen before -- remembering it as {lid} "
                  f"(rename it: \"name\" in {os.path.join(self.folder, lid + '.json')})")
        return lv

    def _recognise(self) -> dict:
        if self.fixed_level:
            return self.levels.get(self.fixed_level) or self._new_fixed()
        best, best_d = None, 1e9
        for lv in self.levels.values():
            ps = lv.get("prints") or []
            if len(ps) != len(self._prints):
                continue
            d = float(np.median([print_distance(a, b) for a, b in zip(ps, self._prints)]))
            if d < best_d:
                best, best_d = lv, d
        if best is not None and best_d < self.MATCH:
            if self.verbose and best is not self.level:
                print(f"[route] recognised {best.get('name') or best['id']} (difference {best_d:.1f}); "
                      f"best run so far {best['best_T']:.1f} s")
            return best
        return self._new_level(self._prints)

    def _new_fixed(self) -> dict:
        lv = {"id": self.fixed_level, "name": self.fixed_level, "created": time.time(), "last_seen": time.time(),
              "attempts": 0, "prints": [], "best": [], "best_T": 0.0, "spots": {}}
        self.levels[self.fixed_level] = lv
        return lv

    # -- one attempt -----------------------------------------------------------------
    def start_attempt(self) -> None:
        """New attempt. Practising needs the level known from the START, so it goes by the
        level of the last attempt (and checks it during the first second)."""
        self._cur, self._down, self._prints = [], None, []
        self._recognised = self.fixed_level is not None
        if self.fixed_level and self.level is None:
            self.level = self._recognise()
        self.plan, self.controlled = None, False
        lv = self.level
        if lv is None or not lv["best"]:
            return
        self._parity ^= 1
        key, sp = self._stuck_spot(lv)
        w0 = max(0.0, lv["best_T"] - self.WINDOW)
        if self.play:
            # every attempt: follow the best run; at the spot it died last time try the next
            # change (the brain gets the first go at a new spot); replays that keep drifting
            # leave the last second to the brain
            sp = sp or lv["spots"].setdefault(key, {"deaths": 0, "tried": [], "worked": None, "drift": 0})
            if sp.get("drift", 0) >= 2:
                # replaying the best run keeps dying BEFORE the spot: the saved run doesn't hold
                # (its timing is off) -- the brain plays the whole attempt instead of a replay
                # that dies at the same early spot every time
                if self.verbose and not getattr(self, "_told_drift", False):
                    self._told_drift = True
                    print(f"[play] the saved best run keeps dying before {lv['best_T']:.1f} s -- "
                          f"the brain plays on its own until it beats that run")
                return
            else:
                base = [iv for iv in lv["best"] if iv[1] > w0 and iv[0] < lv["best_T"] + self.AFTER]
                change, iv = self._next_change(sp, base, w0, lv["best_T"])
            self.plan = (w0, lv["best_T"] + self.AFTER, iv, change)
            if self.verbose:
                print(f"[play] following the best run to {w0:.2f} s, then: {change}")
            return
        if sp is None or not self._parity or sp.get("drift", 0) >= 4:
            return                                  # the brain plays this one
        base = [iv for iv in lv["best"] if iv[1] > w0 and iv[0] < lv["best_T"] + self.AFTER]
        change, iv = self._next_change(sp, base, w0, lv["best_T"])
        self.plan = (w0, lv["best_T"] + self.AFTER, iv, change)
        if self.verbose:
            print(f"[route] {lv.get('name') or lv['id']}: stuck at {lv['best_T']:.2f} s ({sp['deaths']} deaths) -- "
                  f"replaying the best run, then trying: {change}")

    def _stuck_spot(self, lv):
        key = f"{lv['best_T']:.1f}"
        sp = lv["spots"].get(key)
        if sp is None or sp.get("deaths", 0) < self.STUCK:
            return key, None
        return key, sp

    def _next_change(self, sp, base, w0: float, T: float):
        """The first change not tried at this spot yet (remembered across sessions)."""
        tried = set(sp.get("tried", []))
        dt = self.dt
        cands = []
        if self.play:
            cands.append(("the brain", None))        # a new spot: the brain's own go first
        latest = sorted(base, key=lambda iv: -iv[0])[:2]
        for k, (s, e) in enumerate(latest):
            rest = [iv for iv in base if iv != [s, e]]
            for o in self.SHIFTS:
                cands.append((f"press {k + 1} from the end {abs(o)} step(s) {'earlier' if o < 0 else 'later'}",
                              rest + [[max(w0, s + o * dt), max(w0, e + o * dt)]]))
            cands.append((f"press {k + 1} from the end left out", rest))
            for x in (2, 4, 6):
                cands.append((f"press {k + 1} from the end held {x} steps longer", rest + [[s, e + x * dt]]))
        for j in range(2, int(self.WINDOW / dt), 2):
            t = T - j * dt
            cands.append((f"an extra tap {j} steps before the spot", base + [[t, t + 2 * dt]]))
        for desc, iv in cands:
            if desc not in tried:
                return desc, (_merge(iv) if iv is not None else None)
        iv, t = [], w0                                   # out of ideas: random presses
        while t < T:
            if self.rng.random() < 0.15:
                iv.append([t, t + self.rng.choice((1, 2, 3)) * dt])
            t += dt
        return f"random presses #{len(tried)}", _merge(iv)

    def action(self, t: float, a: int, frame=None) -> tuple[int, bool]:
        """(action to send, came from here?) at attempt time t, given the brain + shield's a."""
        if not self._recognised and frame is not None and len(self._prints) < len(PRINT_TIMES):
            if t >= PRINT_TIMES[len(self._prints)]:
                self._prints.append(level_print(frame))
                if len(self._prints) == len(PRINT_TIMES):
                    lv = self._recognise()
                    self._recognised = True
                    if lv is not self.level:
                        if self.plan is not None and self.verbose:
                            print("[route] a different level than last attempt: handing back to the brain")
                        self.plan = None                    # the plan was for another level
                        self.level = lv
                    lv["last_seen"] = time.time()
        self.phase = None
        if self.plan is None:
            return a, False
        w0, w1, iv, _ = self.plan
        if t < w0:
            out = int(any(s <= t < e for s, e in self.level["best"]))
            self.phase = "replay"
        elif t < w1 and iv is not None:
            out = int(any(s <= t < e for s, e in iv))
            self.phase = "try"
        else:
            self.controlled = self.controlled or self.play
            self.phase = None
            return a, False                 # the brain (past the spot, or its turn at it)
        self.controlled = True
        return out, True

    def record(self, t: float, a: int) -> None:
        if a and self._down is None:
            self._down = t
        elif not a and self._down is not None:
            self._cur.append([self._down, t])
            self._down = None

    def end_attempt(self, t_crash: float, complete: bool = False, clock_ok: bool = True) -> None:
        """The attempt ended (death at t_crash, or the level completed). clock_ok=False: this
        attempt's clock didn't start at a respawn (the first one after starting the bot), so
        its times may be off -- it may recognise the level but never becomes the best run
        (live: a best run with shifted times made every replay die early)."""
        if self._down is not None:
            self._cur.append([self._down, t_crash])
            self._down = None
        self.attempt_was_route = self.controlled
        lv = self.level
        if lv is None:
            if self._prints and len(self._prints) == len(PRINT_TIMES):
                self.level = lv = self._recognise()
            else:
                return                           # died before the level could be recognised
        if not clock_ok:
            self.plan = None
            return
        lv["attempts"] = lv.get("attempts", 0) + 1
        plan = self.plan
        T = lv["best_T"]
        key = f"{T:.1f}"
        sp = lv["spots"].setdefault(key, {"deaths": 0, "tried": [], "worked": None, "drift": 0}) if T > 0 else None
        if complete or t_crash > T + self.TOL:
            if plan is not None and sp is not None:
                sp["worked"] = plan[3]
                if self.verbose:
                    print(f"[route] got past {T:.2f} s with: {plan[3]}")
            if self.verbose and T > 0:
                print(f"[route] new best on {lv.get('name') or lv['id']}: {t_crash:.2f} s (was {T:.2f} s)"
                      + ("  LEVEL COMPLETE" if complete else ""))
            lv["best"] = _merge([iv[:] for iv in self._cur if iv[0] <= t_crash + 0.5])
            lv["best_T"] = float(t_crash)
            lv["spots"].setdefault(f"{t_crash:.1f}", {"deaths": 1, "tried": [], "worked": None, "drift": 0})
        elif sp is not None and abs(t_crash - T) <= self.TOL:
            sp["deaths"] += 1
            if plan is not None and plan[3] not in sp["tried"]:
                sp["tried"].append(plan[3])
        elif plan is not None and sp is not None:
            if t_crash < plan[0] - 0.1:
                sp["drift"] = sp.get("drift", 0) + 1      # the replay didn't hold
                if sp["drift"] == 4 and self.verbose:
                    print(f"[route] replaying the best run keeps dying before {T:.1f} s (timing not "
                          f"repeatable enough) -- leaving that spot to the brain")
            elif plan[3] not in sp["tried"]:
                sp["tried"].append(plan[3])              # died inside the window: worse
        self.plan = None
        self._save(lv)
