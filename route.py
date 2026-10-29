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

import bisect
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


THUMB_W, THUMB_H = 64, 36


def level_thumb(frame_bgr: np.ndarray) -> np.ndarray:
    """A small grey picture of the level (top and the icon's column blanked) used to line a
    replay up with where the level really is."""
    import cv2
    f = np.asarray(frame_bgr)[:, :, :3]
    g = cv2.cvtColor(cv2.resize(np.ascontiguousarray(f), (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY)
    g[: int(0.33 * THUMB_H)] = 0
    g[:, int(0.26 * THUMB_W):int(0.50 * THUMB_W)] = 0
    return g


def print_distance(a, b) -> float:
    """Mean colour difference (0..255) between two level pictures (masked parts are 0 in both)."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    n = float((a.sum(axis=2) > 0).sum() + 1)
    return float(np.abs(a - b).sum() / (3 * n))


def pressed_at(iv, t: float, dt: float) -> int:
    """Is SPACE down at the decision at time t, replaying presses iv? Each decision covers
    [t - dt/2, t + dt/2): a press counts on every decision whose window it overlaps, so a
    one-decision tap is never lost when the decisions don't fall exactly where they did
    when it was recorded (they used to be checked at the instant t only -- a few ms off and
    a tap vanished, the replay died)."""
    h = 0.5 * dt
    for s, e in iv:
        if s <= t < e:
            return 1                       # the press covers this decision
        if e - s < dt and t - h <= s < t + h:
            return 1                       # a tap no decision may fall inside: the nearest one
    return 0


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
    LEAD = 3.0        # --frontier: the brain takes over this many seconds before the best run's end
    MATCH = 6.0       # picture difference below this = the same level (tested: the same level
                      # 1-2 frames off ~0.4; another level ~27; the same level 10 s in ~11)
    SHIFTS = (-1, 1, -2, 2, -3, 3, -4, 4, -6, 6)

    def __init__(self, folder: str = "routes", dt: float = 1 / 30, seed: int = 0, verbose: bool = True,
                 fixed_level: str | None = None, play: bool = False, frontier: bool = False) -> None:
        # play: --play (beat the level, no training): EVERY attempt follows the best run, and
        # the stuck spot is practised from the first death there
        self.play = play
        # frontier: --frontier (train where it's stuck): every OTHER attempt replays the best
        # run to LEAD seconds before its end, then the brain plays -- and learns -- from there
        self.frontier = frontier and not play
        self.frontier_attempt = False
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
        # lining replays up with the level: the attempt clock starts when the respawn is
        # DETECTED, and that detection jitters by up to ~0.5 s (live death delays 19..46
        # steps) -- so a replay by clock alone pressed early / late and died at the start
        self.offset = 0.0               # level time = attempt clock + offset
        self._offs: list = []
        self._thumbs: list = []         # this attempt: (level time, picture)
        self._dec: list = []
        self._ref = None                # the best run's pictures: (times, pictures), its level id
        self.align_ok = 0

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

    def _save_thumbs(self, lv: dict, t_end: float) -> None:
        """The new best run's pictures: what later replays line themselves up with."""
        th = [(t, p) for t, p in self._thumbs if t <= t_end - 0.1]   # (not the crash / explosion)
        if len(th) < 5:
            return
        th.sort(key=lambda q: q[0])
        path = os.path.join(self.folder, f"{lv['id']}_frames.npz")
        try:
            np.savez_compressed(path, t=np.array([q[0] for q in th], np.float32),
                                th=np.stack([q[1] for q in th]).astype(np.uint8))
            self._ref = None                # reload on next use
        except Exception as ex:
            print(f"[route] couldn't save {path}: {ex!r}")

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
        self.offset, self._offs, self._thumbs = 0.0, [], []
        self._dec = []                      # this attempt: (level time, action) per decision
        self._recognised = self.fixed_level is not None
        if self.fixed_level and self.level is None:
            self.level = self._recognise()
        self.plan, self.controlled = None, False
        lv = self.level
        if lv is None or not lv["best"]:
            return
        self._parity ^= 1
        self.frontier_attempt = False
        key, sp = self._stuck_spot(lv)
        w0 = max(0.0, lv["best_T"] - self.WINDOW)
        if self.frontier:
            spf = lv["spots"].get(key) or {}
            if self._parity and lv["best_T"] > self.LEAD + 1.0 and spf.get("drift", 0) < 2:
                f0 = lv["best_T"] - self.LEAD
                self.plan = (f0, f0, None, "frontier")
                self.frontier_attempt = True
                if self.verbose:
                    print(f"[frontier] replaying the best run to {f0:.1f} s, the brain learns from there "
                          f"(best so far {lv['best_T']:.1f} s)")
            return                           # (the other attempts: the brain from the start)
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

    def _load_ref(self):
        lv = self.level
        if lv is None:
            return None
        if self._ref is not None and self._ref[2] == lv["id"]:
            return self._ref
        path = os.path.join(self.folder, f"{lv['id']}_frames.npz")
        self._ref = None
        if os.path.exists(path):
            try:
                with np.load(path) as d:
                    self._ref = (d["t"].astype(np.float32), d["th"].astype(np.float32), lv["id"])
            except Exception as ex:
                print(f"[route] couldn't read {path}: {ex!r}")
        return self._ref

    def _align(self, t_raw: float, th: np.ndarray) -> None:
        """Where is the level really? The best run's picture closest to this one says."""
        ref = self._load_ref()
        if ref is None or t_raw < 0.12:
            return
        T, P, _ = ref
        guess = t_raw + self.offset
        lo, hi = np.searchsorted(T, guess - 0.5), np.searchsorted(T, guess + 0.5)
        if hi - lo < 3:
            return
        m = th > 0
        if m.sum() < 50:
            return
        d = np.abs(P[lo:hi][:, m] - th[m].astype(np.float32)).mean(axis=1)
        k = int(np.argmin(d))
        if d[k] > 25.0 or (np.median(d) - d[k]) < 2.0:
            return                          # no clear match (a flash, a menu...): keep the old offset
        tk = float(T[lo + k])
        if 0 < k < len(d) - 1:              # between two pictures: where the minimum really is
            a_, b_, c_ = d[k - 1], d[k], d[k + 1]
            den = a_ - 2 * b_ + c_
            if den > 1e-6:
                frac = 0.5 * (a_ - c_) / den
                tk += float(np.clip(frac, -0.5, 0.5)) * float(T[lo + k + 1] - T[lo + k - 1]) / 2.0
        self._offs.append(tk - t_raw)
        self._offs = self._offs[-7:]
        self.offset = float(np.median(self._offs))
        self.align_ok += 1

    def action(self, t: float, a: int, frame=None) -> tuple[int, bool]:
        """(action to send, came from here?) at attempt clock t, given the brain + shield's a."""
        t_raw = t
        if frame is not None and self.level is not None:
            th = level_thumb(frame)
            self._align(t_raw, th)
            self._thumbs.append((t_raw + self.offset, th))
        t = t_raw + self.offset             # level time
        if not self._recognised and frame is not None and len(self._prints) < len(PRINT_TIMES):
            if t_raw >= PRINT_TIMES[len(self._prints)]:
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
            bd = self.level.get("best_dec")
            if bd:                           # decision by decision: the nearest recorded one
                i = bisect.bisect_left(self._bd_t(bd), t)
                cand = [j for j in (i - 1, i) if 0 <= j < len(bd)]
                j = min(cand, key=lambda j: abs(bd[j][0] - t))
                out = int(bd[j][1])
            else:
                out = pressed_at(self.level["best"], t, self.dt)
            self.phase = "replay"
        elif t < w1 and iv is not None:
            out = pressed_at(iv, t, self.dt)
            self.phase = "try"
        else:
            self.controlled = self.controlled or self.play
            self.phase = None
            return a, False                 # the brain (past the spot, or its turn at it)
        self.controlled = True
        return out, True

    def _bd_t(self, bd):
        if getattr(self, "_bd_cache", (None, None))[0] is not bd:
            self._bd_cache = (bd, [q[0] for q in bd])
        return self._bd_cache[1]

    def record(self, t: float, a: int) -> None:
        t = t + self.offset                 # (attempt clock -> level time)
        self._dec.append((round(float(t), 4), int(a)))
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
        t_crash = t_crash + self.offset     # (attempt clock -> level time)
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
            lv["best_dec"] = [list(q) for q in self._dec if q[0] <= t_crash + 0.5]
            lv["best_T"] = float(t_crash)
            self._save_thumbs(lv, t_crash)
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
