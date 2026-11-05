"""Landmark-triggered route memory: key presses to WHERE THE LEVEL IS, not to the clock.

A Geometry Dash level is fixed in space. A press that clears a spike is "press when this spike
is N blocks ahead", not "press at 2.84 s". route.py replays by clock (aligned by pictures);
the clock is the least reliable thing the bot has: respawn detection jitters, decisions
alternate between 33 and 50 ms, input delay is ~66 ms with jitter. What the detector sees
(hazards, blocks, orbs, pads, portals) does not shift with any of that.

What it keeps, per level (routes/<level id>/landmarks.json, in route.py's folder for the level):
  map     the level's landmarks in one-dimensional level coordinates (blocks): each hazard /
          platform / orb / pad / portal cluster's near edge x, width, height, kind. Built by
          CHAINING spacings seen in the same frame, so it does not depend on a clock.
  events  the best run's presses: for every press and release, which landmark was next ahead
          (anchor), its distance in blocks from the icon at the moment the decision was made,
          the speed, the input delay and the icon's height (a diagnostic).
  times   level position -> level time of the best run (route.py's windows are in seconds;
          this is what turns the landmark position into route.py's level time).
  stats   per press: which distance offsets were tried and how often the replay survived
          the next few blocks with each (so it settles on the offset that works).

Replay: every decision it (1) finds the landmarks in view and lines them up with the map
(sequence matching, so a row of identical spikes is told apart by its spacing -- and when it
can't be sure it does NOT lock and the old clock replay carries on), (2) for the next press
asks "how far is the anchor right now?" and sends the press on the decision that is closest to
the recorded distance. The recorded distance was taken at the decision, so the input delay is
already in it; only a different speed changes the delay's share, and that is corrected.
The planner still vetoes (trainer.replay_shield) -- nothing here bypasses it.

No clocks, no env: the core takes plain boxes (blocks, relative to the icon) so it is testable.
`env_adapter` turns EntityBoxEnv's detections into those boxes.
"""
from __future__ import annotations

import bisect as _bi
import json
import math
import os
import time as _time
from dataclasses import dataclass

try:
    import numpy as _np
    np = _np
except Exception:  # pragma: no cover
    import numpy as np


def _plain(o):
    """json.dump fallback: numpy scalars / arrays -> plain python."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

SPEED_1X = 10.38          # blocks / s
REACH = 11.0              # landmarks farther ahead than this are ignored (clipped by the screen edge)
BEHIND = -3.0             # ... and ones this far behind the icon
CLUSTER_GAP = 0.45        # hazards closer than this form ONE landmark (a triple spike)
MIN_OBS = 4               # frames a new landmark must be seen before it enters the map
LOCK_TOL = 0.45           # blocks: a landmark counts as "the same" when acquiring a position
TRACK_TOL = 0.7           # ... when following it frame to frame
ACQ_WINDOW = 12.0         # blocks around the expected position searched at an attempt's start (the respawn
                          # is detected up to ~1 s late: 10 blocks). Wide ON PURPOSE: identical spikes inside
                          # the window make the lock refuse; a narrow window would pick the wrong neighbour
LOST_STEPS = 45           # decisions without a single match before the lock is dropped
ANCHOR_MAX = 10.0         # the anchor is at most this far ahead of the icon
LATE_BLOCKS = 1.0         # a press this much already behind us when we arrive is skipped
BLAME_BLOCKS = 7.0        # a death this soon after a press blames that press's offset
OFFSET_UNIT = 0.17        # blocks (about half a decision at 1x speed)
OFFSET_KS = (0, 1, -1, 2, -2, 3, -3)
GHOST = False             # --ghost: follow the remembered PATH (icon height along the level), not its button times
GHOST_MODES = ("cube",)   # modes the planner's arcs are trusted for (ship etc. keep the press replay)
GHOST_XS = tuple(np.arange(1.0, 7.01, 0.5))   # blocks ahead the predicted arcs are compared with the ghost
GHOST_ON_PATH = 0.0       # blocks: closer than this to the ghost = leave it to the replay (0 = ghost always decides;
                          # the sim test of 0.35 -- ghost only to recover -- was much worse)
HEIGHT_GUARD = 1.2        # blocks: a remembered press made this much HIGHER than the icon is now
                          # (it was mid-air there -- launched by a pad, say) isn't replayed
GHOST_ZONE = 4.0          # blocks: the ghost learns its timing per stretch of this length
GHOST_SHIFT = 0.17        # blocks per timing step it tries (about half a decision at 1x speed)
GHOST_OFF_PATH = 2.0      # blocks: further than this from the ghost's height = it can't guide, replay instead
GHOST_HIT = 4.0           # score penalty for an arc that crashes (the shield checks safety on top)
GHOST_TIE = 0.03          # press only when pressing tracks the ghost clearly better than not
GHOST_STEP = 0.25         # blocks between stored ghost samples
PRECISE_PRESS = False
PLAN_FIRST = {"release": 0, "hold": 1, "tap now": 1, "tap in 1": 0, "tap in 2": 0, "tap in 3": 0,
              "tap in 4": 0, "tap in 6": 0, "hold 3 then release": 1, "release 3 then hold": 0}     # --precise-press: replayed presses go out at the exact moment, not the next decision
PRECISE_REACH = 1.25      # ... fire on the decision before the press point, if it's within this many decisions
STEP_KEEP = 24            # points of the level position -> time table: kept per attempt >= this far apart
KINDS = {0: "hazard", 1: "orb", 2: "orb", 3: "orb", 4: "pad", 5: "portal", 6: "solid"}


@dataclass
class Box:
    x0: float
    x1: float
    bot: float
    top: float
    kind: str

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.top - self.bot


def cluster_boxes(boxes, gap: float = CLUSTER_GAP) -> list:
    """Hazards (and solids) that nearly touch are one landmark: three spikes in a row, whether
    the detector gives three boxes or one merged box, are one 3-wide landmark."""
    out: list = []
    for b in sorted(boxes, key=lambda b: b.x0):
        c = out[-1] if out else None
        if (c is not None and c.kind == b.kind and b.kind in ("hazard", "solid")
                and b.x0 - c.x1 <= gap and abs(b.bot - c.bot) < 1.0):
            c.x1 = max(c.x1, b.x1)
            c.bot = min(c.bot, b.bot)
            c.top = max(c.top, b.top)
        else:
            out.append(Box(b.x0, b.x1, b.bot, b.top, b.kind))
    return out


@dataclass(eq=False)
class Entry:
    """A landmark in the level map. x = near (left) edge in level blocks."""
    x: float
    w: float
    h: float
    kind: str
    n: int = MIN_OBS
    committed: bool = True

    def compat(self, d: Box) -> bool:
        if d.kind != self.kind:
            return False
        if abs(d.h - self.h) > (1.2 if self.kind == "solid" else 0.6):
            return False
        # widths can be cut by the screen's edge or a dropped spike: only reject clear mismatches
        return self.w * 0.3 - 0.1 <= d.w <= self.w + 1.0

    def observe(self, d: Box, pred_x0: float) -> float:
        """The landmark's near edge as seen now. Normally its left edge; if the cluster lost
        its leftmost spike to a detection dropout, its right edge says where the near edge is."""
        e0, e1 = d.x0, d.x1 - self.w
        return e0 if abs(e0 - pred_x0) <= abs(e1 - pred_x0) else e1


class Tracker:
    """Where is the icon in the level map? Matches the landmarks in view to the map each
    decision; extends the map (provisionally) with landmarks it hasn't seen."""

    GAIN = 0.8

    def __init__(self) -> None:
        self.begin([])

    def begin(self, entries, keep_s: bool = False, origin: float = 0.0) -> None:
        self.pool = sorted((Entry(float(e["x"]), float(e["w"]), float(e["h"]), e["kind"],
                                  int(e.get("n", MIN_OBS)), True) for e in entries), key=lambda e: e.x)
        self._xs = [e.x for e in self.pool]
        if not keep_s:
            # icon position in level blocks. The map's 0 is wherever dead reckoning started on
            # the attempt that BUILT it (often not the respawn: the bot's first attempt isn't
            # started by one). `origin` = where a respawn really is in map blocks (learned).
            self.s = float(origin)
        self.s0 = float(self.s)                   # where dead reckoning started this attempt
        self.first_lock = None                    # (correction, blocks dead-reckoned) of the 1st lock
        self.max_jump = None                      # blocks: most a lock may move the clock's position
        self.has_map = bool(self.pool)
        self.locked = not self.has_map            # no map yet: this attempt BUILDS it (dead reckoning)
        self.ever_locked = self.locked
        self.lock_frame = 0 if self.locked else None
        self.lost = 0
        self.frames = 0
        self.matches: list = []                   # (entry, observed near-edge x0) this decision
        self.match_x0: dict = {}
        self.step_blocks = SPEED_1X / 30.0        # blocks per decision (EMA of the measured scroll)
        self.window = ACQ_WINDOW

    # -- helpers ---------------------------------------------------------------------
    def entry_near(self, x: float, tol: float = 0.8):
        xs = getattr(self, "_xs", None)
        if xs is None or len(xs) != len(self.pool):
            xs = [e.x for e in self.pool]
            self._xs = xs
        i = _bi.bisect_left(xs, x)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.pool) and abs(xs[j] - x) <= tol:
                e = self.pool[j]
                if best is None or abs(e.x - x) < abs(best.x - x):
                    best = e
        return best

    def _slice(self, lo: float, hi: float) -> list:
        xs = getattr(self, "_xs", None)
        if xs is None or len(xs) != len(self.pool):
            xs = [e.x for e in self.pool]
            self._xs = xs
        return self.pool[_bi.bisect_left(xs, lo):_bi.bisect_right(xs, hi)]

    # -- one decision ------------------------------------------------------------------
    def update(self, boxes, scroll, reach: float = REACH) -> None:
        dets = [d for d in cluster_boxes(boxes) if BEHIND <= d.x0 <= reach]
        self.frames += 1
        if scroll is not None:
            self.s += scroll
            if scroll > 0.02:
                self.step_blocks = 0.9 * self.step_blocks + 0.1 * scroll
        self.matches, self.match_x0 = [], {}
        if not self.locked:
            self._acquire(dets, reach)
        if self.locked:
            self._track(dets, reach)

    def _score(self, sc: float, dets, cands):
        used, total, res = set(), 0.0, []
        # A box can only match a landmark whose near edge (or far edge) is within LOCK_TOL of
        # it: look those up by position instead of trying every landmark in view (same order,
        # same result; this scoring was most of a 4-37 ms decision on a 1100-landmark map)
        cxs = [e.x for e in cands]
        wmax = max((e.w for e in cands), default=0.0)
        for d in dets:
            best = None
            lo = _bi.bisect_left(cxs, sc + min(d.x0, d.x1 - wmax) - LOCK_TOL - 1e-6)
            hi = _bi.bisect_right(cxs, sc + max(d.x0, d.x1) + LOCK_TOL + 1e-6)
            for e in cands[lo:hi]:
                if id(e) in used or not e.compat(d):
                    continue
                x0 = e.observe(d, e.x - sc)
                r = x0 - (e.x - sc)
                if abs(r) <= LOCK_TOL and (best is None or abs(r) < abs(best[0])):
                    best = (r, e, x0)
            if best is not None:
                used.add(id(best[1]))
                # a width that disagrees (a triple seen as one spike: a dropout) still matches,
                # but counts for half, so a lone spike doesn't tie with a triple elsewhere
                same_w = abs(d.w - best[1].w) <= 0.5
                total += (1.0 - abs(best[0]) / LOCK_TOL) * (1.0 if same_w else 0.5)
                res.append((best[1], best[2]))
        return total, res

    def _acquire(self, dets, reach: float) -> None:
        """No lock: find the position that lines the landmarks in view up with the map. Only
        locks when the answer is clear -- a row of identical spikes that could be any of
        several stays unlocked (the old clock replay carries on)."""
        if not dets or not self.pool:
            return
        W = self.window
        if self.max_jump is not None:
            # the clock bounds where we can be: look-alikes outside that don't compete
            W = min(W, self.max_jump + 0.03 * abs(self.s - self.s0))
        cands_s: list = []
        for d in dets:
            # only landmarks that could put us within W of where we think we are (was: every
            # landmark in the map for every box -- 1101 landmarks x ~25 boxes = 4-15 ms per
            # decision while unlocked, which pushed decisions from 2 frames to 3; same result)
            for e in self._slice(self.s + d.x0 - W - 1e-6, self.s + d.x0 + W + 1e-6):
                if e.compat(d):
                    sc = e.x - d.x0
                    if abs(sc - self.s) <= W:
                        cands_s.append(sc)
        if not cands_s:
            return
        scored = []
        for sc in sorted(set(round(c, 2) for c in cands_s)):
            near = self._slice(sc + BEHIND - 1, sc + reach + 1)
            total, res = self._score(sc, dets, near)
            if total > 0:
                scored.append((total, sc, res))
        if not scored:
            return
        scored.sort(key=lambda q: -q[0])
        best_t, best_s, best_res = scored[0]
        second = next((q[0] for q in scored[1:] if abs(q[1] - best_s) > 1.5), 0.0)
        _need = 1.4 if second > 0.4 * best_t else 0.7
        if best_t >= _need and second <= 0.6 * best_t:
            _s_new = float(np.median([e.x - x0 for e, x0 in best_res]))
            if self.max_jump is not None:
                lim = self.max_jump + 0.03 * abs(self.s - self.s0)
                if abs(_s_new - self.s) > lim:
                    # the clock says we can't be there: a look-alike row of spikes (live 015528:
                    # locks 2.2 and 3.1 blocks off on the first frame of a respawn, replay died at 1%)
                    self.rejected_locks = getattr(self, "rejected_locks", 0) + 1
                    return
            _s_before = self.s
            self.s = _s_new
            if self.first_lock is None:
                self.first_lock = (self.s - _s_before, _s_before - self.s0)
            self.locked = self.ever_locked = True
            self.lock_frame = self.frames
            self.lost = 0
            self.window = min(self.window, ACQ_WINDOW)  # re-locked: next dropout re-acquires nearby

    def _track(self, dets, reach: float) -> None:
        s = self.s
        near = self._slice(s + BEHIND - 4.0, s + reach + 4.0)
        used: set = set()
        pairs, unmatched = [], []
        # nearest-first so a close, good match wins its landmark
        for d in sorted(dets, key=lambda d: d.x0):
            best = None
            for e in near:
                if id(e) in used or not e.compat(d):
                    continue
                pred = e.x - s
                x0 = e.observe(d, pred)
                r = x0 - pred
                if abs(r) <= TRACK_TOL and (best is None or abs(r) < abs(best[0])):
                    best = (r, e, x0, d)
            if best is None:
                unmatched.append(d)
            else:
                used.add(id(best[1]))
                pairs.append(best)
        dirty = False
        if pairs:
            s_meas = float(np.median([e.x - x0 for _, e, x0, _ in pairs]))
            self.s = s + self.GAIN * (s_meas - s)
            self.lost = 0
            for _, e, x0, d in pairs:
                self.matches.append((e, x0))
                self.match_x0[id(e)] = x0
                if not e.committed:                      # a provisional landmark: refine it
                    e.n += 1
                    self._grew_map = True
                    e.x += (self.s + x0 - e.x) / min(e.n, 6)
                    # (grows fast -- the first sighting may be cut by the screen's edge -- shrinks slowly:
                    #  a plain max() over noisy frames inflates the width)
                    e.w += (0.3 if d.w > e.w else 0.05) * (d.w - e.w)
                    dirty = True
        else:
            self.lost += 1
            if self.has_map and self.lost > LOST_STEPS:
                self.locked = False                      # lost it: re-acquire near where we think we are
                self.window = 5.0 + 0.1 * self.lost
        # landmarks the map hasn't got: provisional entries (only while the position is trusted)
        # (building a map from nothing: there is nothing to match yet, the position is dead reckoning)
        if self.locked and (self.lost < 10 or not self.has_map):
            for d in unmatched:
                x = self.s + d.x0
                if self.entry_near(x, 0.8) is not None and any(
                        e.kind == d.kind and abs(e.x - x) < 0.8 for e in
                        self._slice(x - 0.8, x + 0.8)):
                    continue
                e = Entry(x, d.w, d.h, d.kind, 1, False)
                self._grew_map = True
                self.pool.append(e)
                self.matches.append((e, d.x0))
                self.match_x0[id(e)] = d.x0
                dirty = True
        if dirty:
            self.pool.sort(key=lambda e: e.x)
            self._xs = [e.x for e in self.pool]

    def pick_anchor(self):
        """(entry, distance) of the landmark next ahead (else the nearest just behind)."""
        ahead = [(x0, e) for e, x0 in self.matches if x0 >= -0.3]
        if ahead:
            x0, e = min(ahead, key=lambda q: q[0])
            if x0 <= ANCHOR_MAX:
                return e, x0
        behind = [(x0, e) for e, x0 in self.matches if x0 < -0.3]
        if behind:
            x0, e = max(behind, key=lambda q: q[0])
            return e, x0
        return None, None


def env_adapter(env):
    """EntityBoxEnv -> (boxes, scroll blocks, speed blocks/s, icon height, reach) or None when
    the icon isn't on screen. Boxes are the RAW detections (not the shrunken hitboxes), so the
    memory doesn't change when the hitbox model does."""
    import sys as _sys
    _eb = _sys.modules.get("env_boxes")
    if _eb is None:
        from env_boxes import BLOCK_H, BLOCK_W, SPEED_1X as ENV_SPEED
    else:
        BLOCK_H, BLOCK_W, ENV_SPEED = _eb.BLOCK_H, _eb.BLOCK_W, _eb.SPEED_1X
    ents = getattr(env, "last_ents", None)
    if ents is None or ents.player() is None:
        return None
    px, py = ents.player_pos()
    boxes = []
    for e in ents.items:
        kind = KINDS.get(int(e.cls))
        if kind is None:
            continue
        boxes.append(Box((e.x - e.w / 2 - px) / BLOCK_W, (e.x + e.w / 2 - px) / BLOCK_W,
                         (py - (e.y + e.h / 2)) / BLOCK_H, (py - (e.y - e.h / 2)) / BLOCK_H, kind))
    speed = float(getattr(env, "speed", None) or ENV_SPEED)
    sc = getattr(env, "last_shift", None)
    if getattr(env, "_death_frozen", False) or getattr(env, "_last_moved", True) is False:
        scroll = 0.0
    elif sc is not None and -0.5 < sc < -0.0015:
        scroll = -sc / BLOCK_W                              # measured: screen fraction -> blocks
    else:
        scroll = speed * float(env._motion_dt())            # no measurement this step: speed x time
    return boxes, scroll, speed, (0.75 - py) / BLOCK_H, (1.0 - px) / BLOCK_W - 1.5


ORIGIN_MIN = 2            # respawn-origin samples needed before replaying by dead reckoning (no lock)
ORIGIN_MAX_DR = 15.0      # blocks: only a lock this close to the respawn says where the respawn is
ORIGIN_WINDOW = 3.0       # blocks searched around the expected position once the respawn is known
                          # (the full ACQ_WINDOW reached the NEXT spike+pillar of a repeating level:
                          #  live 013804 learned respawns from -1.6 to +10.8 blocks -- wrong locks)
LOCK_MAX_JUMP = 1.2       # blocks: once the respawn is known, a lock may correct the clock's position
                          # by at most this (+3% of the distance travelled) -- start jitter is ~0.3
MAP_VERSION = 2           # v1 maps were built from per-step scroll readings (dropouts, jumps): discarded
AGE_MAX = 0.25            # s: observation ages beyond this are glitches (menus, pauses), not latency
LAT_CLIP = 0.8            # blocks: most a latency correction may move a press


class LandmarkRoute:
    def __init__(self, folder: str = "routes", dt: float = 1 / 30, verbose: bool = True,
                 delay_s: float = 0.066, adapter=env_adapter) -> None:
        self.folder, self.dt, self.verbose, self.delay_s, self.adapter = folder, dt, verbose, delay_s, adapter
        os.makedirs(folder, exist_ok=True)
        self.tracker = Tracker()
        self.level_id = None
        self.entries: list = []        # the map (dicts)
        self.best_events: list = []
        self.times: list = []          # (level position, level time) of the best run
        self.stats: dict = {}
        self.taboos: list = []         # [(position, bad_action)] fatal moves to NOT repeat (death-spot memory)
        self.ghost: list = []          # [(position, icon height)] along the remembered successful path
        self.ghost_log: list = []      # this attempt's (position, icon height), locked decisions only
        self.ghost_report = dict(decided=0, pressed=0, off_path=0, uncovered=0)
        self.ghost_stats: dict = {}    # zone -> {timing step k | "r" (leave it to the replay): [ok, died]}
        self._ghost_arm: dict = {}     # this attempt: zone -> the arm it plays there
        self.best_s = 0.0
        self.speed = SPEED_1X
        self.last_report: dict = {}
        self.origins: list = []        # learned: where a respawn is in map blocks (one per attempt)
        self.clock_ok = False          # set by the trainer: did this attempt start at a respawn?
        self._att_clock_ok = False
        self._obs_ts = None            # perf_counter time the current observation's picture was shown
        self._ages: list = []          # recent action ages at send (s): the typical latency
        self._send_lag = 0.0           # s between replay_action() and the key actually being sent
        self._replay_age = None
        self._reset_attempt()

    # -- storage -----------------------------------------------------------------------
    def _path(self, lid: str) -> str:
        """routes/<level id>/landmarks.json (moved there from the old flat routes/<id>_landmarks.json)."""
        d = os.path.join(self.folder, lid)
        new = os.path.join(d, "landmarks.json")
        old = os.path.join(self.folder, f"{lid}_landmarks.json")
        if not os.path.exists(new) and os.path.exists(old):
            try:
                os.makedirs(d, exist_ok=True)
                os.replace(old, new)
            except OSError:
                return old
        return new

    def set_level(self, lid) -> None:
        if lid == self.level_id:
            return
        self._level_t_cache = None
        first = self.level_id is None
        self.level_id = lid
        self.entries, self.best_events, self.times, self.stats, self.best_s = [], [], [], {}, 0.0
        self.ghost = []
        self.ghost_stats = {}
        self.frontier_visits = 0
        self.origins = []
        self.origins2 = []
        if lid is not None and not os.path.exists(self._path(lid)) and os.path.exists(self._path(lid) + ".tmp"):
            try:
                os.replace(self._path(lid) + ".tmp", self._path(lid))
            except Exception:
                pass
        if lid is not None and os.path.exists(self._path(lid)):
            try:
                with open(self._path(lid)) as _f:
                    d = json.load(_f)
                if int(d.get("version", 1)) < MAP_VERSION:
                    print(f"[landmarks] {lid}: map built with the old scroll-summing position (v{d.get('version', 1)}) "
                          f"-- its positions don't agree run to run, starting the map over")
                    d = {}
                self.entries = d.get("entries", [])
                self.best_events = d.get("events", [])
                self.times = [tuple(p) for p in d.get("times", [])]
                self.stats = d.get("stats", {})
                self.taboos = [[float(t[0]), int(t[1])] for t in d.get("taboos", [])]
                self.best_s = float(d.get("best_s", 0.0))
                self.ghost = [(float(a), float(b)) for a, b in d.get("ghost", [])]
                self.ghost_stats = dict(d.get("ghost_stats", {}) or {})
                self.frontier_visits = int(d.get("frontier_visits", 0) or 0)
                self.origins = [float(o) for o in d.get("origins", [])][-25:]
                self.origins2 = [float(o) for o in d.get("origins2", [])][-25:]
                if self.verbose:
                    print(f"[landmarks] {lid}: {len(self.entries)} landmarks, {len(self.best_events)} presses "
                          f"remembered (best run {self.best_s:.0f} blocks)"
                          + (f", respawn at {self.origin():+.1f} blocks" if self.origins else ""))
            except Exception as ex:
                print(f"[landmarks] couldn't read {self._path(lid)}: {ex!r}")
        if self.entries or not first:
            # a different level than the pool was built for: start the pool over
            # (a new level always restarts at position 0 -- never inherit s).
            self._begin_tracker()
            self._reset_attempt(keep_offsets=False)
            self._pick_offsets()

    def _save(self) -> None:
        if self.level_id is None:
            return
        d = dict(version=MAP_VERSION, entries=self.entries, events=self.best_events, times=self.times,
                 stats=self.stats, best_s=self.best_s, taboos=self.taboos, origins=self.origins,
                 ghost=[[round(float(a), 3), round(float(b), 3)] for a, b in (self.ghost or [])],
                 ghost_stats=self.ghost_stats,
                 frontier_visits=int(getattr(self, "frontier_visits", 0) or 0),
                 origins2=getattr(self, "origins2", []))
        os.makedirs(os.path.dirname(self._path(self.level_id)), exist_ok=True)
        tmp = self._path(self.level_id) + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(d, f, default=_plain)      # detections hand us numpy scalars; plain json can't write them
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except OSError:
                    pass
            try:
                import shutil as _sh
                _dst = self._path(self.level_id)
                if os.path.exists(_dst):
                    _sh.copyfile(_dst, _dst + ".bak")
            except OSError:
                pass
            os.replace(tmp, self._path(self.level_id))
        except BaseException:
            try:
                os.remove(tmp)                       # never leave a half-written .tmp behind
            except OSError:
                pass
            raise

    # -- attempt bookkeeping -----------------------------------------------------------
    def _reset_attempt(self, keep_offsets: bool = False) -> None:
        self._grew_map = False         # did this attempt learn new landmark geometry?
        self.events: list = []         # this attempt's presses (recorded from what was actually sent)
        self._open = None
        self._snap = None
        self.prev_a = 0
        self.hist: list = []           # (attempt clock, position)
        self.tlog: list = []           # (position, level time)
        self.i = 0
        self.holding = False
        self.hold_n = 0
        self.fired: list = []
        self.skipped = 0
        self.in_charge = False
        self._probe_skipped: list = []  # presses left out (probe) this attempt
        self._brain_handled: list = []  # unreliable presses handed to the brain this attempt
        self.pre_lock_events = 0
        self.ghost_log = []
        self.ghost_report = dict(decided=0, pressed=0, off_path=0, uncovered=0)
        self._ghost_arm = {}
        self.report = dict(direct=0, dead=0, fired=0, skipped=0, press_height_err=[], press_dist_err=[])
        if not keep_offsets:
            self.offsets: list = []

    def _pick_offsets(self) -> None:
        self._level_t_cache = None
        self.offsets = [self._pick_offset(i) for i in range(len(self.best_events))]
        # per-press hold variation + leave-out probe: a press blamed several
        # times is mistimed in a way a shift may not fix (too long/short, or
        # shouldn't be there at all). Order: shifts -> hold length -> skip.
        # State persists in stats ("h<steps>", "skip") across sessions.
        self.hold_over: dict = {}
        self.skip_idx: set = set()
        for i in range(len(self.best_events)):
            st = self.stats.get(str(i), {})
            _shift_bad = sum(int(v[1]) for k, v in st.items()
                             if str(k).lstrip("-").isdigit() and isinstance(v, (list, tuple)) and len(v) == 2)
            _shift_ok = sum(int(v[0]) for k, v in st.items()
                            if str(k).lstrip("-").isdigit() and isinstance(v, (list, tuple)) and len(v) == 2)
            # probe (other hold lengths, then leaving it out) only a press that FAILS OFTEN: live
            # (064500) presses that worked 1021 of 1024 times had 3 blamed deaths, went to the
            # leave-out probe and were never sent again -- 77 of 215 presses, a 22% wall
            if _shift_bad < 3 or _shift_bad < 0.3 * (_shift_ok + _shift_bad):
                continue
            _holds = {k: v for k, v in st.items() if isinstance(k, str) and k.startswith("h")}
            _tried_holds = set(_holds)
            _hok = sum(int(v[0]) for v in _holds.values() if isinstance(v, (list, tuple)) and len(v) == 2)
            _sk = st.get("skip", [0, 0])
            _skb = int(_sk[1]) if isinstance(_sk, (list, tuple)) and len(_sk) == 2 else 0
            _sko = int(_sk[0]) if isinstance(_sk, (list, tuple)) and len(_sk) == 2 else 0
            if _hok == 0 and len(_tried_holds) < 2 and _skb == 0 and _sko == 0:
                # shifts exhausted, holds untried: vary the hold length first
                _cands = [1, 3] if int(self.best_events[i].get("steps", 2)) > 2 else [3, 5]
                _done = {k for k in _tried_holds}
                for _h in _cands:
                    if f"h{_h}" not in _done:
                        self.hold_over[i] = _h
                        break
            elif _skb < 2 and _sko == 0:
                # shifts + holds all blamed: probe leaving it out (once each)
                self.skip_idx.add(i)

    def _unreliable(self, i: int) -> bool:
        """Hand this press to the brain + shield? Its replay keeps failing (>= 30% of >= 6 tries
        over all timings) and the brain hasn't done worse there yet. Live (161215): press 44 of
        the 89% run failed 16 of 37 replays at ~19 s while the brain wanted to press right; the
        replay's hold-off overrode it -- 17 of 49 attempts died there."""
        st = self.stats.get(str(i), {})
        ok = bad = 0
        for k, v in st.items():
            if k != "brain" and isinstance(v, (list, tuple)) and len(v) == 2:
                ok += int(v[0]); bad += int(v[1])
        if ok + bad < 6 or bad < 0.3 * (ok + bad):
            return False
        bo, bb = st.get("brain", [0, 0])
        if bo + bb < 4:
            return True                      # give the brain a few tries
        return bb / (bo + bb) < bad / (ok + bad)

    def _pick_offset(self, i: int) -> float:
        """The timing shift to replay press i with: the one with the best record. A press that
        keeps failing (>= 10% of >= 20 tries) gives UNTRIED shifts an optimistic prior (90%), so
        it tries them one by one until each proves worse. Before, an untried shift scored 50%: a
        press working 82% of the time was never shifted -- live (172553) press 54, the 19 s wall,
        148 ok / 32 deaths, only ever replayed at its original timing."""
        st = self.stats.get(str(i), {})
        tot_ok = tot_bad = 0
        for k in OFFSET_KS:
            ok, bad = st.get(str(k), (0, 0))
            tot_ok += int(ok); tot_bad += int(bad)
        explore = (tot_ok + tot_bad) >= 20 and tot_bad >= 0.10 * (tot_ok + tot_bad)
        a, b = (1.8, 0.2) if explore else (1.0, 1.0)
        best, best_score = 0, -1.0
        for k in OFFSET_KS:
            ok, bad = st.get(str(k), (0, 0))
            score = (ok + a) / (ok + bad + a + b) - 0.003 * abs(k)
            if score > best_score:
                best, best_score = k, score
        return best * OFFSET_UNIT

    # -- where a respawn is in the map ------------------------------------------------------
    def origin(self) -> float:
        """Map position of the respawn, learned from RELIABLE first locks only: at least 5 blocks
        in (first-frame locks after a respawn were often wrong -- live 020035 they fed back and
        the 'respawn' crept to ~2 blocks), the median of the last 15, never beyond +-3 blocks.
        Live (163601) 9 of 12 first locks ~20 blocks in corrected +0.7..+1.7 blocks the same way:
        the map's respawn sits ~1 block from 0, so every press before the lock came late and
        lock jumps > 1 block skipped presses. 0 until 3 samples."""
        # DISABLED: live (165848) the learned value ran away to the +3 cap again within a few
        # attempts (locks lean forward ~0.2-0.3 and each attempt starts from the last estimate)
        # and the median didn't move. origins2 is still recorded, as a diagnostic only.
        return 0.0

    def origin_known(self) -> bool:
        return bool(self.entries) or len(self.origins) >= ORIGIN_MIN

    def _begin_tracker(self) -> None:
        """Start an attempt's tracking at the respawn; once its place in the map is known, only
        look for the lock near it (a repeating level has look-alike spots a few blocks on)."""
        self.tracker.begin(self.entries, origin=self.origin())
        if self.origin_known():
            self.tracker.window = ORIGIN_WINDOW
            self.tracker.max_jump = LOCK_MAX_JUMP

    def _learn_origin(self) -> None:
        """The first lock of an attempt that started at a respawn says where the respawn is in
        map blocks: dead reckoning started at s0 there, so the respawn = s0 + the lock's jump.
        Live (010907): the map was built on the bot's first attempt, which began 5.6 blocks past
        the respawn; every later attempt dead-reckoned from 0 = 5.6 blocks off, so unlocked
        replay pressed ~0.5 s early and died at 1-4% while the brain alone had reached 60%."""
        fl = getattr(self.tracker, "first_lock", None)
        if not self._att_clock_ok or fl is None or not self.tracker.has_map:
            return
        corr, dr = fl
        if not (0.0 <= dr <= 40.0) or abs(corr) > ACQ_WINDOW:
            return
        self.origins.append(round(float(corr), 3))     # diagnostic: how far the first lock moved us
        self.origins = self.origins[-25:]
        if 5.0 <= dr <= 40.0:
            # where the respawn is in the map, as this (reliable) lock says
            self.origins2 = (list(getattr(self, "origins2", [])) + [round(float(self.tracker.s0 + corr), 3)])[-25:]

    # -- latency ---------------------------------------------------------------------------
    def _age_now(self):
        """How old the current observation will be when this decision's key is sent (s)."""
        if self._obs_ts is None:
            return None
        a = _time.perf_counter() - float(self._obs_ts)
        if not (0.0 <= a <= AGE_MAX):
            return None
        self._replay_age = a
        return a + self._send_lag

    def _age_ref(self):
        return float(np.median(self._ages)) if len(self._ages) >= 5 else None

    def _lat(self, v_now: float, v_rec: float, age_rec, age_now) -> float:
        """Blocks to add to a recorded distance so the key goes out at the same PLACE: the
        distance was read off a picture that was age old when the key was sent, and that age
        jitters 30..100 ms live (decisions take 33 or 50 ms) -- at 10.4 blocks/s up to 0.7 block."""
        if age_now is None:
            return 0.0
        if age_rec is None:
            age_rec = self._age_ref()
        if age_rec is None:
            return 0.0
        return float(np.clip(v_now * age_now - v_rec * float(age_rec), -LAT_CLIP, LAT_CLIP))

    def begin_attempt(self, lid) -> None:
        self.set_level(lid)
        self._att_clock_ok = bool(self.clock_ok)
        self._begin_tracker()
        self._reset_attempt()
        self._pick_offsets()

    # -- one decision: observe, then (maybe) replay, then record -------------------------
    def observe_env(self, env, t_raw: float) -> None:
        got = self.adapter(env)
        _ts = getattr(env, "_obs_ts", None)
        self._obs_ts = float(_ts) if isinstance(_ts, (int, float)) else None
        if got is None:
            self._snap = None
            return
        boxes, scroll, speed, y, reach = got
        self.observe(boxes, scroll, speed, y, t_raw, reach)

    def observe(self, boxes, scroll, speed, y, t_raw: float, reach: float = REACH,
                map_reach: float = 8.0) -> None:
        # a new attempt restarts the attempt clock: the old position / anchor must not leak in
        # (begin_attempt is only called explicitly in tests; live the trainer just keeps observing).
        _prev = getattr(self, "t_raw", None)
        if _prev is None:
            self._att_clock_ok = bool(self.clock_ok)     # the very first attempt seen
        if _prev is not None and float(t_raw) < float(_prev) - 0.5:
            self._att_clock_ok = bool(self.clock_ok)
            self._begin_tracker()
            self._reset_attempt()
            self._pick_offsets()
        # detections arrive as numpy scalars (float32): plain floats keep the matching exact and the json writable
        boxes = [Box(float(b.x0), float(b.x1), float(b.bot), float(b.top), b.kind) for b in boxes]
        scroll, speed, y = (None if scroll is None else float(scroll)), (None if speed is None else float(speed)), float(y)
        # Dead reckoning by the ATTEMPT CLOCK (captured game frames since the respawn) x speed,
        # not by summing each step's scroll reading: live (013804) those read 6.9..15.6 blocks/s
        # step to step, dropped to 0 on "not moving" glitches and averaged 8.5 against a real
        # ~10.4 -- 2 s in, the position was blocks off and runs disagreed by up to 9 blocks.
        # The clock is exact to a frame (the respawn is detected 31-33 steps after the crash).
        _pt = getattr(self, "t_raw", None)
        if _pt is not None and scroll is not None:
            _dt = float(t_raw) - float(_pt)
            if 0.0 < _dt <= 0.25:
                scroll = (speed or self.speed or SPEED_1X) * _dt
            elif _dt <= 0.0:
                scroll = 0.0
        tr = self.tracker
        tr.update(boxes, scroll, min(reach, map_reach) if map_reach else reach)
        try:
            tr._map_reach = float(min(reach, map_reach) if map_reach else reach)
        except Exception:
            pass
        self.speed = float(speed) if speed else self.speed
        self.y, self.t_raw = float(y), float(t_raw)
        # Dead reckoning always runs (scroll accumulates even unlocked): the crash
        # position must exist for the FIRST commit, which happens before any lock.
        self.hist.append((self.t_raw, tr.s))
        e, d = tr.pick_anchor()
        # what the decision about to be made is based on (record() runs AFTER the step)
        self._snap = dict(entry=e, d=d, s=tr.s, y=self.y, v=self.speed, locked=tr.locked)

    @property
    def locked(self) -> bool:
        return self.tracker.locked and self.tracker.has_map

    def ready(self) -> bool:
        return bool(self.best_events) and self.locked

    def ready_pos(self) -> bool:
        """Position-only replay: presses exist and we have a position estimate, even
        unlocked. Dead reckoning drifts, but it still beats the brain-less void --
        and it is what bootstraps the first lock (replay the area, lock, refine)."""
        return bool(self.best_events) and self._snap is not None

    def level_time(self):
        """Level time (route.py's clock) at the icon's position, from the best run."""
        if not self.locked or len(self.times) < 4 or self._snap is None:
            return None
        key = id(self.times)
        cache = getattr(self, "_level_t_cache", None)
        if cache is None or cache[0] != key:
            cache = (key, [p[0] for p in self.times], [p[1] for p in self.times])
            self._level_t_cache = cache
        _, P, TT = cache
        s = self.tracker.s
        if s < P[0] - 1.0:
            return None
        if s > P[-1]:
            return TT[-1] + (s - P[-1]) / max(3.0, self.speed)
        return float(np.interp(s, P, TT))

    # -- replay --------------------------------------------------------------------------
    def _dist(self, ax):
        e = self.tracker.entry_near(ax, 0.8)
        if e is None:
            return None
        x0 = self.tracker.match_x0.get(id(e))
        if x0 is not None:
            self.report["direct"] += 1
            return x0
        self.report["dead"] += 1                 # not seen this decision: where the map says it is
        return e.x - self.tracker.s

    def replay_action(self, handover_s: float = 0.0):
        """1 / 0 for this decision, or None when it can't say (nothing remembered, past
        the remembered presses, or inside the hand-over before the best end) --
        the caller then falls back to the brain."""
        self.in_charge = False
        self.press_at = None
        if not self.best_events or self._snap is None or self.i >= len(self.best_events):
            return None
        if handover_s > 0 and self.best_s and self.best_s > 5.0:
            # stop ~handover_s before the best run's end: the brain plays the
            # ending live, so it can get FURTHER instead of replaying into the crash.
            try:
                lead = float(self.speed or SPEED_1X) * float(handover_s)
                if self.tracker.s >= float(self.best_s) - lead:
                    self.holding = False
                    return None
            except Exception:
                pass
        tr, ev = self.tracker, self.best_events
        if not self.locked and tr.has_map and not self.origin_known():
            # no lock and no idea where the respawn is in the map: dead reckoning may be blocks
            # off -- let the brain play until the landmarks lock (or the respawn is learned)
            return None
        half = 0.5 * tr.step_blocks
        age_now = self._age_now()
        _skip = getattr(self, "skip_idx", None) or set()
        _hold = getattr(self, "hold_over", None) or {}
        while self.i < len(ev):
            if not self.holding and self._unreliable(self.i):
                e_ = ev[self.i]
                if e_.get("ax") is not None and self.locked:
                    D_ = self._dist(e_["ax"]); tgt_ = e_["d"]
                else:
                    D_, tgt_ = e_["s"] - tr.s, 0.0
                if D_ is not None and D_ < tgt_ - LATE_BLOCKS:
                    if self.i not in self._brain_handled:
                        self._brain_handled.append(self.i)   # scored at the attempt's end
                    self.i += 1              # the brain had this one: on to the next press
                    continue
                if D_ is not None and D_ <= tgt_ + 2.0:
                    self.holding = False
                    return None              # within 2 blocks of it: the brain + shield decide
            if self.i in _skip:
                if self.i not in self._probe_skipped:
                    self._probe_skipped.append(self.i)   # scored at the attempt's end
                self.i += 1                  # leave-out probe: is this press the killer?
                self.skipped += 1
                continue
            e, off = ev[self.i], (self.offsets[self.i] if self.i < len(self.offsets) else 0.0)
            v_rec, delay = e.get("v") or self.speed, e.get("delay") or self.delay_s
            corr = float(np.clip((self.speed - v_rec) * delay, -0.6, 0.6))   # other speed, other delay share
            if self.holding:
                rel = e.get("rel") or {}
                self.hold_n += 1
                D = self._dist(rel["ax"]) if rel.get("ax") is not None else None
                if D is not None:
                    lat = self._lat(self.speed, v_rec, rel.get("age"), age_now)
                    done = D <= rel["d"] + corr - off + lat + half
                else:
                    _n = int(_hold.get(self.i, rel.get("steps", e.get("steps", 1))))
                    done = self.hold_n >= _n
                self.in_charge = True
                if done:
                    self.holding = False
                    self.i += 1
                    return 0
                return 1
            lat = self._lat(self.speed, v_rec, e.get("age"), age_now)
            if e.get("ax") is not None and self.locked:
                D, target = self._dist(e["ax"]), e["d"] + corr - off + lat
            else:                                # no landmark in view, or no lock: by position
                D, target = e["s"] - tr.s, -off + lat
            self.in_charge = True
            if D is None:
                return 0                         # its landmark isn't in view yet
            if D < target - LATE_BLOCKS:
                self.i += 1                      # already past it (lock found late / long dropout)
                self.skipped += 1
                continue
            # --precise-press: a press used to go out at the first decision within half a
            # decision of its point -- anywhere in +-half (50 ms decisions: +-25 ms, +-0.26 block),
            # plus 0-17 ms waiting for the next capture slot. Live (213820) the 19 s triple-spike
            # press 54 failed 19% at its best timing and every timing shift scored worse: the
            # spread, not the mean, was wrong. Instead, fire on the decision BEFORE the point and
            # tell the env how long to wait (env_boxes.step sends the key at that moment).
            reach = PRECISE_REACH * tr.step_blocks if PRECISE_PRESS else half
            if (D <= target + reach and getattr(self, "cur_mode", None) == "cube" and e.get("y") is not None
                    and float(e["y"]) - float(self.y) > HEIGHT_GUARD):
                # The best run pressed here while it was high in the air (live 041703: press 47,
                # y 2.83 -- already launched by the yellow pad). Down on the ground now, the same
                # press is a jump OVER the pad into the pillar: 0 of 9 tries survived it. A
                # different situation than the one remembered: leave it out, the brain and the
                # shield decide (the shield still presses if waiting would die).
                self.report["height_skips"] = self.report.get("height_skips", 0) + 1
                self.i += 1
                self.skipped += 1
                continue
            if D <= target + reach:
                self.holding, self.hold_n = True, 0
                err = None if self._snap is None else self.y - e.get("y", self.y)
                r = float(D - target)
                if PRECISE_PRESS:
                    wait = max(0.0, r) / max(1.0, float(self.speed or SPEED_1X))
                    self.press_at = _time.perf_counter() + self._send_lag + wait
                    self.report.setdefault("press_wait_ms", []).append(round(1000.0 * wait, 1))
                self.fired.append(dict(i=self.i, s=tr.s, k=round(off / OFFSET_UNIT), r=round(r, 3)))
                self.report["fired"] += 1
                self.report["press_dist_err"].append(round(r, 3))
                if err is not None:
                    self.report["press_height_err"].append(round(float(err), 3))
                return 1
            return 0
        return None

    # -- record what was actually sent --------------------------------------------------------
    def record(self, a: int, t_level: float, age=None, wait_extra: float = 0.0) -> None:
        """age: how old the observation was when this key went out (env._action_observation_age_s).
        wait_extra: a --precise-press key went out this much later than a normal one would have
        (s) -- only the remembered press gets it, so a new best run records where it REALLY pressed."""
        a = int(a)
        sn = self._snap
        try:
            age = float(age) if age is not None and 0.0 <= float(age) <= AGE_MAX else None
        except (TypeError, ValueError):
            age = None
        if age is not None:
            self._ages.append(age)
            del self._ages[:-41]
            if self._replay_age is not None:
                self._send_lag = 0.9 * self._send_lag + 0.1 * float(np.clip(age - self._replay_age, 0.0, 0.05))
        self._replay_age = None
        if sn is None:
            self.prev_a = a
            return
        tr = self.tracker
        if sn.get("locked"):
            gl = self.ghost_log
            if not gl or sn["s"] - gl[-1][0] >= GHOST_STEP:
                gl.append((float(sn["s"]), float(sn["y"])))
        # Position-vs-time table builds even unlocked (dead reckoning): the first commit
        # needs it, and replay by position does not require a lock to record where it was.
        if not self.tlog or sn["s"] - self.tlog[-1][0] >= 0.25:
            self.tlog.append((sn["s"], float(t_level)))
        if a and not self.prev_a:
            if not tr.locked:
                self.pre_lock_events += 1
            try:
                ev_age = None if age is None else float(np.clip(age + float(wait_extra or 0.0), 0.0, AGE_MAX))
            except (TypeError, ValueError):
                ev_age = age
            self._open = dict(ax=None, d=None, s=sn["s"], y=sn["y"], v=sn["v"], delay=self.delay_s,
                              steps=1, _e=sn["entry"], _d=sn["d"], t=float(t_level), age=ev_age)
            self.events.append(self._open)
        elif a and self._open is not None:
            self._open["steps"] += 1
        elif not a and self.prev_a and self._open is not None:
            self._open["_rel"] = (sn["entry"], sn["d"])
            self._open["rel_steps"] = self._open["steps"]
            self._open["rel_age"] = age
            self._open = None
        self.prev_a = a

    # -- attempt end ---------------------------------------------------------------------------
    def _crash_position(self, t_crash_raw: float):
        if not self.hist:
            return None
        T = [h[0] for h in self.hist]
        return float(np.interp(t_crash_raw, T, [h[1] for h in self.hist]))

    def note_death_move(self, s: float, a: int, tol: float = 1.0) -> None:
        """Remember that action `a` at position `s` died: future hand-over
        play avoids it there. Taboos persist in the level json, are keyed by
        position (nearest within tol wins, replaced), and capped to the most
        recent 32 so one bad memory can't steer the run forever."""
        try:
            s, a = float(s), int(a)
        except Exception:
            return
        for i, (ps, pa) in enumerate(self.taboos):
            try:
                if abs(float(ps) - s) <= tol:
                    self.taboos[i] = [s, a]
                    break
            except Exception:
                continue
        else:
            self.taboos.append([s, a])
        self.taboos = self.taboos[-32:]
        try:
            self._save()
        except Exception:
            pass

    def taboo_action(self, s: float, tol: float = 1.0):
        """The action known to kill at position `s`, or None."""
        try:
            s = float(s)
        except Exception:
            return None
        for ps, pa in self.taboos:
            try:
                if abs(float(ps) - s) <= tol:
                    return int(pa)
            except Exception:
                continue
        return None

    def end_attempt(self, t_crash_raw: float, best: bool, complete: bool = False, lag_steps: int = 0) -> dict:
        s_crash = self._crash_position(t_crash_raw)
        if not self.tracker.has_map and not self._att_clock_ok:
            # this attempt would START the map, but it didn't begin at a respawn (the bot's first
            # attempt starts wherever the level was): the map's 0 would not be the respawn, and
            # on a repeating level nothing can tell later attempts where they are in it. Wait
            # for an attempt that starts at a respawn -- its 0 IS the respawn.
            self.tracker.begin([])
            rep = dict(self.report, skipped=self.skipped, locked=False, s_crash=None, committed=False,
                       commit_reason="the first attempt didn't start at a respawn: the map starts next attempt")
            self.last_report = rep
            return rep
        _building = not self.tracker.has_map
        _n_orig = len(self.origins)
        if _building and not self.origins:
            self.origins = [0.0] * ORIGIN_MIN      # map started at a respawn: that's 0, by construction
        self._learn_origin()
        if len(self.origins) != _n_orig and self.verbose and len(self.origins) == ORIGIN_MIN:
            print(f"[landmarks] the respawn is at {self.origin():+.2f} blocks in this level's map "
                  f"-- replay by position before the lock is now lined up with it")
        rep = dict(self.report, skipped=self.skipped, locked=self.tracker.ever_locked,
                   s_crash=None if s_crash is None else round(s_crash, 2), committed=False)
        # which remembered press does a death blame? Probes (skip / hold-over)
        # are scored under their own keys so a fix that works is kept.
        _skip = set(getattr(self, "skip_idx", None) or set())
        _hold = dict(getattr(self, "hold_over", None) or {})
        # score the leave-out probes: a left-out press is never "fired", so before this its probe
        # was never scored and the press stayed left out for good. Died within BLAME_BLOCKS after
        # it = leaving it out was bad (after 2 bad ones the press is replayed again); got well past = ok
        for _i in list(getattr(self, "_brain_handled", []) or []):
            if _i >= len(self.best_events):
                continue
            _si = float(self.best_events[_i].get("s", 0.0))
            row = self.stats.setdefault(str(_i), {}).setdefault("brain", [0, 0])
            if complete or s_crash is None or s_crash - _si > BLAME_BLOCKS:
                row[0] += 1
            elif s_crash >= _si - 0.5:
                row[1] += 1
        if getattr(self, "_brain_handled", None) and not self.fired:
            self._save()
        for _i in list(getattr(self, "_probe_skipped", []) or []):
            if _i >= len(self.best_events):
                continue
            _si = float(self.best_events[_i].get("s", 0.0))
            row = self.stats.setdefault(str(_i), {}).setdefault("skip", [0, 0])
            if complete or s_crash is None or s_crash - _si > BLAME_BLOCKS:
                row[0] += 1
            elif s_crash >= _si - 0.5:
                row[1] += 1
        if self.fired and not complete and s_crash is not None and self.best_events:
            last = self.fired[-1]
            _lag_extra = float(getattr(self, "_last_lag_s", 0.0) or 0.0) * float(self.speed or SPEED_1X)
            blamed = -0.5 <= s_crash - last["s"] <= BLAME_BLOCKS + _lag_extra
            for f in self.fired:
                _key = "skip" if f["i"] in _skip else (f"h{_hold.get(f['i'])}" if f["i"] in _hold else str(f["k"]))
                row = self.stats.setdefault(str(f["i"]), {}).setdefault(_key, [0, 0])
                row[1 if (blamed and f is last) else 0] += 1
            rep["blamed"] = self.fired[-1]["i"] if blamed else None
            rep["probe"] = {str(i): ("skip" if i in _skip else f"h{_hold[i]}") for i in (set(_skip) | set(_hold))}
        elif self.fired:
            for f in self.fired:
                self.stats.setdefault(str(f["i"]), {}).setdefault(str(f["k"]), [0, 0])[0] += 1
        # A failed replay blames a press offset -- re-pick offsets NOW so the very
        # next attempt replays with the corrected timing (press later/earlier),
        # instead of repeating the same mistimed jumps until a new best saves.
        if not best and not complete and rep.get("blamed") is not None and self.best_events:
            try:
                self._pick_offsets()
                self._save()
            except Exception:
                pass
        if lag_steps:
            try:
                self._last_lag_s = float(lag_steps) * float(getattr(self, "dt", 1 / 30))
            except Exception:
                pass
        try:
            self._score_ghost(s_crash, complete)
            self._extend_ghost(None if complete else s_crash, best)
        except Exception:
            pass
        if best and s_crash is not None:
            rep["committed"], rep["commit_reason"] = self._commit(s_crash)
        else:
            grew = self._promote_seen()
            if grew or self.fired or len(self.origins) != _n_orig:
                self._save()
        if self.verbose and not self.best_events and not self.fired and not rep.get("committed"):
            _sc = None if s_crash is None else round(s_crash, 1)
            print(f"[landmarks] no replay yet (locked={self.tracker.locked} ever={self.tracker.ever_locked} "
                  f"s_crash={_sc} best={best} presses={len(self.events)})")
        errs = rep["press_height_err"]
        if self.verbose and (self.fired or rep.get("committed")):
            msg = f"[landmarks] fired {rep['fired']} press(es), skipped {rep['skipped']}"
            if errs:
                msg += (f"; height at press vs best run: mean {np.mean(errs):+.2f}, "
                        f"worst {max(errs, key=abs):+.2f} blocks")
            if rep.get("blamed") is not None:
                msg += f"; death blamed on press {rep['blamed']}"
            print(msg)
        self.last_report = rep
        return rep

    GHOST_SAFE_BACK = 10.0   # blocks before a death not trusted as part of a working path

    def _extend_ghost(self, s_crash, best: bool) -> None:
        """Grow the ghost past its end with this attempt's path, up to GHOST_SAFE_BACK blocks
        before where it died (all of it on a completed run). A route made before the ghost
        existed gets one this way from its next attempts; nothing has to be started over."""
        if best or not self.ghost_log:
            return                              # a new best replaces the ghost in _commit instead
        end = self.ghost[-1][0] if self.ghost else -1e9
        safe = 1e9 if s_crash is None else float(s_crash) - self.GHOST_SAFE_BACK
        new = [g for g in self.ghost_log if end + GHOST_STEP * 0.5 < g[0] <= safe]
        if not new:
            return
        if self.ghost and not any(abs(g[0] - end) <= 1.0 for g in self.ghost_log):
            return                              # this attempt's path doesn't join the ghost's end
        self.ghost = list(self.ghost or []) + new
        self._save()

    GHOST_ARMS = ("0", "1", "-1", "2", "-2", "3", "-3", "r")

    def _pick_ghost_arm(self, z: str) -> str:
        """Learn from deaths: per stretch of the level, which timing the ghost follows its path
        with (k steps of GHOST_SHIFT later / earlier) -- or "r", hand the stretch back to the press
        replay. Untried choices get an optimistic prior once the stretch dies >= 10% of >= 8
        tries, so it tries them one by one until each proves worse (like the replay's offsets)."""
        st = self.ghost_stats.get(z, {})
        tot_ok = sum(int(v[0]) for v in st.values())
        tot_bad = sum(int(v[1]) for v in st.values())
        explore = (tot_ok + tot_bad) >= 8 and tot_bad >= 0.10 * (tot_ok + tot_bad)
        a, b = (1.8, 0.2) if explore else (1.0, 1.0)
        best, best_score = "0", -1e9
        arms = list(self.GHOST_ARMS)
        # the best tried shift is the furthest one there is (live 055505: a stretch settled on
        # -3, 17 of 20) -- the right timing may lie beyond: open the next two steps that way
        tried = {k: v for k, v in st.items() if k != "r" and sum(v) >= 3}
        if tried:
            top = max(tried, key=lambda k: (tried[k][0] + 1) / (sum(tried[k]) + 2))
            k_top = int(top)
            if abs(k_top) >= 3:
                sgn = 1 if k_top > 0 else -1
                arms += [str(sgn * (abs(k_top) + 1)), str(sgn * (abs(k_top) + 2))]
        for arm in arms:
            ok, bad = (st.get(arm) or [0, 0])
            pen = 0.006 if arm == "r" else 0.003 * abs(int(arm))
            score = (ok + a) / (ok + bad + a + b) - pen
            if score > best_score:
                best, best_score = arm, score
        return best

    def _score_ghost(self, s_crash, complete: bool) -> None:
        """After an attempt: the stretch the death came from (the decisions ~1.5 blocks before
        the crash) scores a death for the choice it played; every stretch passed scores a success."""
        if not self._ghost_arm:
            return
        dz = None if (complete or s_crash is None) else str(int((float(s_crash) - 1.5) // GHOST_ZONE))
        for z, arm in self._ghost_arm.items():
            row = self.ghost_stats.setdefault(z, {}).setdefault(arm, [0, 0])
            if z == dz:
                row[1] += 1
            elif dz is None or int(z) < int(dz):
                row[0] += 1
        self._save()

    def ghost_covers(self, s0: float, s1: float) -> bool:
        g = self.ghost
        return bool(g) and g[0][0] <= s0 and g[-1][0] >= s1

    def ghost_y(self, s):
        g = self.ghost
        return np.interp(s, [q[0] for q in g], [q[1] for q in g])

    def ghost_action(self, arcs, mode):
        """--ghost: 1 / 0 = the action whose predicted arc keeps the icon closest to the ghost
        path over the next few blocks, or None (no ghost here, not locked, a mode the arcs aren't
        trusted for, or too far off the path to be guided). arcs = env.last_arcs: the planner's
        [(plan name, crashes?, [(blocks ahead, height rel. to the icon)])].
        A replay presses at remembered places; a jump that went out a decision late then lands
        somewhere the next remembered press doesn't fit. Following the path re-plans every
        decision from where the icon really is: a late jump's next press moves to match."""
        if not GHOST or not arcs or mode not in GHOST_MODES or not self.locked or self._snap is None:
            return None
        s, y = float(self._snap["s"]), float(self._snap["y"])
        if not self.ghost_covers(s, s + GHOST_XS[-1]):
            self.ghost_report["uncovered"] += 1
            return None
        dev = abs(y - float(self.ghost_y(s)))
        if dev > GHOST_OFF_PATH:
            self.ghost_report["off_path"] += 1
            return None
        if dev < GHOST_ON_PATH:
            return None                         # on the path: the replay's presses are right here
        z = str(int(s // GHOST_ZONE))
        arm = self._ghost_arm.get(z)
        if arm is None:
            arm = self._ghost_arm[z] = self._pick_ghost_arm(z)
        if arm == "r":
            self.ghost_report["replay_zones"] = self.ghost_report.get("replay_zones", 0) + 1
            return None                         # this stretch kept killing the ghost: the replay plays it
        shift = int(arm) * GHOST_SHIFT          # + = the ghost's path moved later: press later
        xs = np.asarray(GHOST_XS)
        want = self.ghost_y(s + xs - shift)
        best = {0: 1e9, 1: 1e9}
        for name, hit, pts in arcs:
            first = PLAN_FIRST.get(name)
            if first is None or len(pts) < 3:
                continue
            px = np.asarray([q[0] for q in pts], float)
            py = np.asarray([q[1] for q in pts], float)
            ok = xs <= px[-1]
            if ok.sum() < 3:
                continue
            pred = y + np.interp(xs[ok], px, py)
            # a crash further ahead than the compared stretch is a LATER decision's problem (the
            # arcs only press once; the next decisions press again) -- only a crash inside it counts
            crash_here = bool(hit) and px[-1] <= xs[-1]
            score = float(np.mean(np.abs(pred - want[ok]))) + (GHOST_HIT if crash_here else 0.0)
            best[first] = min(best[first], score)
        if best[0] >= 1e9 and best[1] >= 1e9:
            return None
        a = 1 if best[1] < best[0] - GHOST_TIE else 0
        self.ghost_report["decided"] += 1
        self.ghost_report["pressed"] += a
        return a

    def _promote_seen(self, lag_steps: int = 0) -> int:
        """Committed = seen long enough. Promote provisionally-tracked landmarks
        (built on ANY attempt, not just best runs) and report how many were new."""
        if lag_steps:
            try:
                self._last_lag_s = float(lag_steps) * float(getattr(self, "dt", 1 / 30))
            except Exception:
                pass
        n = 0
        for e in self.tracker.pool:
            _need_n = 6 if self._is_tiled(e) else MIN_OBS
            if not e.committed and e.n >= _need_n:
                e.committed = True
                n += 1
        if n:
            keep = [e for e in self.tracker.pool if e.committed]
            keep.sort(key=lambda e: e.x)
            self.entries = [dict(x=round(e.x, 3), w=round(e.w, 3), h=round(e.h, 3), kind=e.kind, n=e.n) for e in keep]
        return n

    def _is_tiled(self, e) -> bool:
        """Repetitive row? >=2 same-kind neighbours within 1.5 blocks."""
        try:
            c = 0
            for q in self.tracker.pool:
                if q is not e and q.kind == e.kind and abs(q.x - e.x) < 1.5:
                    c += 1
                    if c >= 2:
                        return True
            return False
        except Exception:
            return False

    SPLICE_BACK = 15.0    # blocks before the old best run's end where the new run takes over
    SPLICE_QUIET = 2.5    # no press of either run this close to the switch point

    def _splice(self, events: list, s_crash: float) -> list:
        """A new best run keeps the OLD run's presses where the old run already went (minus its
        last SPLICE_BACK blocks, where it was failing) and uses the new run's from there on.
        In --play the new run's early part is mostly the old run replayed, plus whatever the
        shield / brain changed on that one lucky attempt: replacing everything let one-off
        changes overwrite presses that had worked hundreds of times -- live (064500 on) the
        89% run's replay sat at a 22% median where the 78% run's gave 42-50%. Switches at a
        point where neither run pressed within SPLICE_QUIET blocks; no such point = no splice."""
        old = list(self.best_events or [])
        if not old or not self.best_s or self.best_s < self.SPLICE_BACK + 10.0 or s_crash <= self.best_s:
            return events
        target = float(self.best_s) - self.SPLICE_BACK
        presses = [float(e["s"]) for e in old] + [float(e["s"]) for e in events]
        x = target
        while x > target - 20.0:
            if all(abs(p - x) > self.SPLICE_QUIET for p in presses):
                kept = [e for e in old if float(e["s"]) < x]
                fresh = [e for e in events if float(e["s"]) >= x]
                self._splice_x = x
                if self.verbose:
                    print(f"[landmarks] new best: kept the old run's {len(kept)} presses up to {x:.0f} blocks, "
                          f"the new run's {len(fresh)} from there")
                return kept + fresh
            x -= 0.5
        return events

    def _commit(self, s_crash: float):
        tr = self.tracker
        # First commit is allowed from dead reckoning (never locked): positions come
        # from scroll, presses from what was sent. Precision comes later once the
        # map locks and offsets/blame refine them. Without this the memory can never
        # bootstrap: no lock -> no commit -> no presses -> no replay.
        if not tr.ever_locked and not self.events:
            return False, "no presses recorded and never locked"
        if not self.events:
            return False, "no presses recorded this run"
        self._promote_seen()                                # provisional landmarks seen long enough are kept
        keep = [e for e in tr.pool if e.committed]
        keep.sort(key=lambda e: e.x)
        ids = {id(e) for e in keep}
        # presses made before the lock were placed by dead reckoning: only trusted once the
        # respawn's place in the map is known (else they could be blocks off -- dropped)
        trust_dr = self.origin_known() and self._att_clock_ok
        drop_prefix = int(self.pre_lock_events) if (tr.ever_locked and not trust_dr) else 0
        events = []
        for ev in self.events[drop_prefix:]:
            if ev["s"] > s_crash + 0.5:
                continue
            e = dict(ax=None, d=None, s=round(ev["s"], 3), y=round(ev["y"], 3), v=round(ev["v"], 3),
                     delay=ev["delay"], steps=ev["steps"], t=round(ev["t"], 3))
            if ev.get("age") is not None:
                e["age"] = round(float(ev["age"]), 4)
            if ev["_e"] is not None and id(ev["_e"]) in ids and ev["_d"] is not None:
                e["ax"], e["d"] = round(ev["_e"].x, 3), round(float(ev["_d"]), 3)
            rel = ev.get("_rel")
            if rel is not None and rel[0] is not None and id(rel[0]) in ids and rel[1] is not None:
                e["rel"] = dict(ax=round(rel[0].x, 3), d=round(float(rel[1]), 3), steps=ev.get("rel_steps", 1))
                if ev.get("rel_age") is not None:
                    e["rel"]["age"] = round(float(ev["rel_age"]), 4)
            else:
                e["rel"] = dict(ax=None, d=None, steps=ev.get("rel_steps", ev["steps"]))
            events.append(e)
        self._splice_x = None
        events = self._splice(events, s_crash)
        # the ghost path follows the same switch: the old run's path up to the splice point,
        # this run's after it (up to just before where it crashed)
        _x = self._splice_x
        _new = [g for g in self.ghost_log if g[0] <= s_crash - 1.0 and (_x is None or g[0] >= _x)]
        self.ghost = ([g for g in (self.ghost or []) if g[0] < _x] if _x is not None else []) + _new
        # level position -> level time: keep the old table's start, extend it with this run's
        old = [p for p in self.times]
        end_old = old[-1][0] if old else -1e9
        new = [p for p in self.tlog if p[0] > end_old and p[0] <= s_crash]
        table = old + new if old else [p for p in self.tlog if p[0] <= s_crash]
        self.entries = [dict(x=round(e.x, 3), w=round(e.w, 3), h=round(e.h, 3), kind=e.kind, n=e.n) for e in keep]
        try:
            _old_stats = dict(self.stats or {})
            _new_stats = {}
            for _j, _e in enumerate(events):
                _bi, _bd = None, 1e9
                for _k, _v in _old_stats.items():
                    try:
                        _oi = int(_k)
                    except Exception:
                        continue
                    if _oi < 0 or _oi >= len(self.best_events):
                        continue
                    _d = abs(float(self.best_events[_oi].get("s", 0.0)) - float(_e.get("s", 0.0)))
                    if _d < _bd:
                        _bd, _bi = _d, _k
                if _bi is not None and _bd <= 2.0:
                    _new_stats[str(_j)] = _old_stats[_bi]
            self.stats = _new_stats
        except Exception:
            self.stats = {}
        self.best_events, self.times, self.best_s = events, [tuple(p) for p in table], s_crash
        self._pick_offsets()
        self._save()
        if self.verbose:
            print(f"[landmarks] remembered {len(events)} press(es) against {len(self.entries)} landmarks "
                  f"({s_crash:.0f} blocks)")
        return True, "ok"