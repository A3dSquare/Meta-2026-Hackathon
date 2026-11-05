"""Route memory: per-level practice that the brain never trains on.

A GD level is the same every attempt: an obstacle is always the same time after the
attempt starts. When the bot keeps dying at the same moment, this replays the best run
it has had on that level up to just before that spot and tries a different press
timing there -- one change per attempt -- like a player practising a hard part.

What it remembers, per level (routes/<level id>/route.json, kept forever; the same folder holds
frames.npz, the best run's pictures, and landmarks.json, the landmark memory):
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

try:
    import cv2 as _CV2
except Exception:
    _CV2 = None


def _get_cv2():
    global _CV2
    if _CV2 is None:
        import cv2 as _cv
        globals()['_CV2'] = _cv
        return _cv
    return _CV2

PRINT_TIMES = (0.3, 0.6, 0.9)   # seconds into an attempt where the level's look is sampled
PRINT_W, PRINT_H = 48, 27


def level_print(frame_bgr: np.ndarray) -> np.ndarray:
    """A small colour picture of the level, without the parts that change between
    attempts or players: the top (progress bar, "Attempt N"), the icon's column."""
    cv2 = _get_cv2()
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
    cv2 = _get_cv2()
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


PRINT_SHIFT = 14     # px (of 48): the most a print may be slid sideways when matching (~5 blocks)


def print_distance_shifted(a, b, max_shift: int = PRINT_SHIFT, step: int = 2) -> float:
    """print_distance, but the two pictures may be slid sideways against each other first.
    The prints are taken by the ATTEMPT CLOCK, which starts when the respawn is detected --
    that detection jitters by up to ~0.5 s = ~5 blocks of scroll, so the same level was
    sampled at different places and read as "different" (live: one level saved as up to 4
    separate levels inside an hour, same-level differences 6-8 against a cut-off of 9).
    Only pixels unmasked in BOTH pictures count."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    W = a.shape[1]
    va, vb = a.sum(axis=2) > 0, b.sum(axis=2) > 0
    best = print_distance(a, b)
    for k in range(-max_shift, max_shift + 1, step):
        if k == 0:
            continue
        if k > 0:
            A, B, m = a[:, k:], b[:, :W - k], va[:, k:] & vb[:, :W - k]
        else:
            A, B, m = a[:, :W + k], b[:, -k:], va[:, :W + k] & vb[:, -k:]
        n = int(m.sum())
        if n < 150:
            continue
        d = float(np.abs(A - B)[m].sum() / (3 * n))
        if d < best:
            best = d
    return best


def _print_blank(p) -> bool:
    """A level picture with no signal (black frame, capture not ready). Such prints
    match every other blank level at distance 0 and poison recognition -- they must
    never be stored or matched."""
    try:
        a = np.asarray(p, np.float32)
        return float(np.abs(a).mean()) < 1.0
    except Exception:
        return True


def _prints_valid(ps) -> bool:
    return bool(ps) and not any(_print_blank(p) for p in ps)


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
    MATCH = 13.0      # picture difference below this = the same level. With sideways sliding
                      # (print_distance_shifted) on the 18 saved Oct-29 routes: the same level
                      # 0.8..8.0, different levels >= 22.4 (was 9.0 unshifted: too close to the
                      # same level's 6-8 -- one level kept being saved as a new one)
    SHIFTS = (-1, 1, -2, 2, -3, 3, -4, 4, -6, 6)
    # Picture alignment is only allowed to fine-tune the attempt clock, never to replace it.
    # Live (015528, a level that repeats every few blocks): the start offset was steady
    # (-48..+33 ms) but pictures kept matching an EARLIER look-alike part, and since each match
    # moved the search window, the offset ran away to -1.3..-1.9 s within 4 s -- every replayed
    # press and every death time after that was on a broken clock.
    ALIGN_START = 0.30     # s: the most the respawn detection can be off (measured ~0.05)
    ALIGN_DEV = 0.10       # s: once anchored, the most the offset may move from its anchor
    ALIGN_ANCHOR = 5       # accepted pictures that set the anchor

    def __init__(self, folder: str = "routes", dt: float = 1 / 30, seed: int = 0, verbose: bool = True,
                 fixed_level: str | None = None, play: bool = False, frontier: bool = False) -> None:
        # play: --play (beat the level, no training): EVERY attempt follows the best run, and
        # the stuck spot is practised from the first death there
        self.play = play
        # frontier: --frontier (train where it's stuck): every OTHER attempt replays the best
        # run to LEAD seconds before its end, then the brain plays -- and learns -- from there
        self.frontier = frontier and not play
        self.frontier_attempt = False
        # --frontier hand-over cap for this session (level time): a replay that dies before the
        # hand-over moves the next hand-over to just before where it died -- the brain practises
        # the part the replay can't do, instead of frontier switching itself off. Live (020639):
        # two replays died at 5.0 s (hand-over 5.7 s) and frontier ran 2 of 49 attempts.
        self._fcut = None
        self._fcut_T = None
        self.frontier_plan = None             # the last attempt's plan, for the timing log
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
        # the REAL time between decisions: dt is the nominal frame_skip/60 (33 ms) but live
        # decisions take 33 or 50 ms (mean ~45). A tap shorter than a decision is replayed on
        # the decision whose window holds it -- windows of 33 ms with decisions 50 ms apart
        # left gaps, and a tap falling in a gap was never pressed.
        self.dec_dt = dt
        self._dts: list = []
        self._dts_att: list = []
        self._align_log: list = []
        self._align_rej: dict = {}
        self._anchor_offs: list = []
        self._t_last = None

    # -- storage -------------------------------------------------------------------
    # One folder per level: routes/<level id>/{route.json, frames.npz, landmarks.json}
    def level_dir(self, lid: str) -> str:
        return os.path.join(self.folder, lid)

    def _migrate_flat(self) -> None:
        """Old layout (everything flat in routes/): move each level's files into its folder."""
        try:
            names = os.listdir(self.folder)
        except OSError:
            return
        moves = []
        for fn in names:
            src = os.path.join(self.folder, fn)
            if not os.path.isfile(src):
                continue
            for suffix, new in (("_landmarks.json.bak", "landmarks.json.bak"), ("_landmarks.json.tmp", "landmarks.json.tmp"),
                                ("_landmarks.json", "landmarks.json"), ("_frames.npz", "frames.npz"),
                                (".json.tmp", "route.json.tmp"), (".json", "route.json")):
                if fn.endswith(suffix):
                    moves.append((src, fn[: -len(suffix)], new))
                    break
        for src, lid, new in moves:
            if not lid:
                continue
            dst_dir = self.level_dir(lid)
            dst = os.path.join(dst_dir, new)
            try:
                os.makedirs(dst_dir, exist_ok=True)
                if not os.path.exists(dst):
                    os.replace(src, dst)
            except OSError as ex:
                print(f"[route] couldn't move {src} into {dst_dir}: {ex!r}")
        if moves and self.verbose:
            print(f"[route] moved {len(moves)} file(s) into one folder per level ({self.folder}/<level>/)")

    def _recover_tmps(self) -> None:
        try:
            for d in os.listdir(self.folder):
                tmp = os.path.join(self.folder, d, "route.json.tmp")
                dst = os.path.join(self.folder, d, "route.json")
                if os.path.isfile(tmp) and not os.path.exists(dst):
                    try:
                        os.replace(tmp, dst)
                    except OSError:
                        pass
        except OSError:
            pass

    def _load_all(self) -> dict:
        self._migrate_flat()
        self._recover_tmps()
        out = {}
        for dn in sorted(os.listdir(self.folder)):
            fn = os.path.join(dn, "route.json")
            if os.path.isfile(os.path.join(self.folder, fn)):
                try:
                    with open(os.path.join(self.folder, fn)) as _f:
                        d = json.load(_f)
                    d["prints"] = [np.asarray(p, np.float32) for p in d.get("prints", [])]
                    if d.get("prints") and not _prints_valid(d["prints"]):
                        # blank capture saved by an old run: no signal, would match
                        # every other blank level at distance 0 -- ignore it so the
                        # real level is recognised fresh instead of merged into garbage
                        if self.verbose:
                            print(f"[route] ignoring {fn}: blank level prints (no signal)")
                        continue
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
        if lv.get("prints") and not _prints_valid(lv["prints"]):
            # never persist a blank capture: it would match any blank run at 0
            lv["prints"] = []
        d = dict(lv)
        d["prints"] = [np.round(p, 1).tolist() for p in lv["prints"]]
        os.makedirs(self.level_dir(lv["id"]), exist_ok=True)
        path = os.path.join(self.level_dir(lv["id"]), "route.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
            try:
                f.flush()
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)

    def _save_thumbs(self, lv: dict, t_end: float) -> None:
        """The new best run's pictures: what later replays line themselves up with."""
        th = [(t, p) for t, p in self._thumbs if t <= t_end - 0.1]   # (not the crash / explosion)
        if len(th) < 5:
            return
        th.sort(key=lambda q: q[0])
        os.makedirs(self.level_dir(lv["id"]), exist_ok=True)
        path = os.path.join(self.level_dir(lv["id"]), "frames.npz")
        try:
            np.savez_compressed(path, t=np.array([q[0] for q in th], np.float32),
                                th=np.stack([q[1] for q in th]).astype(np.uint8))
            self._ref = None                # reload on next use
        except Exception as ex:
            print(f"[route] couldn't save {path}: {ex!r}")

    def _new_level(self, prints) -> dict:
        lid = "level_" + time.strftime("%Y%m%d_%H%M%S")
        if lid in self.levels or os.path.exists(self.level_dir(lid)):  # never reuse an id / folder
            lid += f"_{self.rng.randrange(65536):04x}"
            while lid in self.levels or os.path.exists(self.level_dir(lid)):
                lid += f"_{self.rng.randrange(65536):04x}"
        lv = {"id": lid, "name": "", "created": time.time(), "last_seen": time.time(), "attempts": 0,
              "prints": list(prints), "best": [], "best_T": 0.0, "spots": {}}
        self.levels[lid] = lv
        if self.verbose:
            print(f"[route] a level it hasn't seen before -- remembering it as {lid} "
                  f"(rename it: \"name\" in {os.path.join(self.level_dir(lid), 'route.json')})")
        return lv

    def _recognise(self) -> dict:
        if self.fixed_level:
            return self.levels.get(self.fixed_level) or self._new_fixed()
        if not _prints_valid(self._prints):
            # capture wasn't ready (black frames): no signal to match on -- keep
            # the last level so the attempt still replays instead of forking a
            # blank level record that matches every other blank run at distance 0
            if self.level is not None:
                return self.level
            return self._new_level(self._prints)
        best, best_d = None, 1e9
        cands = [lv for lv in self.levels.values()
                 if len(lv.get("prints") or []) == len(self._prints) and _prints_valid(lv["prints"])]
        # (old blank records are never a match). Plain comparison first: cheap, and the usual
        # case (same level as last attempt, sampled at the same place) ends here
        for lv in cands:
            d = float(np.median([print_distance(a, b) for a, b in zip(lv["prints"], self._prints)]))
            if d < best_d:
                best, best_d = lv, d
        if best_d >= self.MATCH:
            # maybe a known level sampled a little earlier / later (respawn clock jitter):
            # slide the pictures -- ~1 ms per level, only on attempts the plain match missed
            for lv in cands:
                d = float(np.median([print_distance_shifted(a, b) for a, b in zip(lv["prints"], self._prints)]))
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
        self._t_last = None
        self.offset, self._offs, self._thumbs = 0.0, [], []
        self._align_log, self._align_rej = [], {}   # timing diagnostics (attempt_timing())
        self._anchor_offs = []
        self._dts_att = []
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
            if self._fcut_T != lv["best_T"]:
                self._fcut, self._fcut_T = None, lv["best_T"]     # a new best: try its full replay
            if self._parity and lv["best_T"] > self.LEAD + 1.0 and spf.get("drift", 0) < 8:
                f0 = lv["best_T"] - self.LEAD
                if self._fcut is not None:
                    f0 = min(f0, self._fcut)
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
            if sp.get("drift", 0) >= 6:
                # replaying the best run keeps dying BEFORE the spot: the saved run doesn't hold
                # (its timing is off) -- the brain plays the whole attempt instead of a replay
                # that dies at the same early spot every time. Threshold is high ON PURPOSE:
                # --play is pure playback, so two unlucky jitters must not retire a good run.
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
        if self.verbose and sp.get("deaths", 0) < self.STUCK:
            print(f"[route] the brain's last {lv.get('early', 0)} attempts died well before the best "
                  f"run's {lv['best_T']:.1f} s -- replaying it to there")
        base = [iv for iv in lv["best"] if iv[1] > w0 and iv[0] < lv["best_T"] + self.AFTER]
        change, iv = self._next_change(sp, base, w0, lv["best_T"])
        self.plan = (w0, lv["best_T"] + self.AFTER, iv, change)
        if self.verbose:
            print(f"[route] {lv.get('name') or lv['id']}: stuck at {lv['best_T']:.2f} s ({sp['deaths']} deaths) -- "
                  f"replaying the best run, then trying: {change}")

    def _stuck_spot(self, lv):
        key = f"{lv['best_T']:.1f}"
        sp = lv["spots"].get(key)
        if lv.get("early", 0) >= self.STUCK and lv["best_T"] > self.WINDOW + 1.0:
            # the brain keeps dying long BEFORE the best spot (live 004613: best 29%, then 120
            # attempts at ~4% -- the spot's death count never moved, so the memory never
            # replayed the run it had). Practise the spot anyway on the route's attempts.
            sp = sp or lv["spots"].setdefault(key, {"deaths": 0, "tried": [], "worked": None, "drift": 0})
            return key, sp
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
        path = os.path.join(self.level_dir(lv["id"]), "frames.npz")
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
        if ref is None:
            self._reject("no_reference")
            return
        if t_raw < 0.12:
            return
        T, P, _ = ref
        anchored = len(getattr(self, "_anchor_offs", [])) >= self.ALIGN_ANCHOR
        center = float(np.median(self._anchor_offs)) if anchored else 0.0
        span = self.ALIGN_DEV if anchored else self.ALIGN_START
        guess = t_raw + center               # search around the CLOCK (+ anchor), not the last match
        lo, hi = np.searchsorted(T, guess - span - 0.05), np.searchsorted(T, guess + span + 0.05)
        if hi - lo < 3:
            self._reject("past_reference" if guess > float(T[-1]) else "no_frames_near")
            return
        m = th > 0
        if m.sum() < 50:
            self._reject("blank")
            return
        d = np.abs(P[lo:hi][:, m] - th[m].astype(np.float32)).mean(axis=1)
        k = int(np.argmin(d))
        if d[k] > 25.0 or (np.median(d) - d[k]) < 2.0:
            self._reject("no_match" if d[k] > 25.0 else "ambiguous")
            return                          # no clear match (a flash, a menu...): keep the old offset
        tk = float(T[lo + k])
        if 0 < k < len(d) - 1:              # between two pictures: where the minimum really is
            a_, b_, c_ = d[k - 1], d[k], d[k + 1]
            den = a_ - 2 * b_ + c_
            if den > 1e-6:
                frac = 0.5 * (a_ - c_) / den
                tk += float(np.clip(frac, -0.5, 0.5)) * float(T[lo + k + 1] - T[lo + k - 1]) / 2.0
        if abs((tk - t_raw) - center) > span:
            self._reject("implausible_jump")  # a look-alike part of the level, not where we are
            return
        if not anchored:
            self._anchor_offs.append(tk - t_raw)
        self._offs.append(tk - t_raw)
        self._offs = self._offs[-7:]
        self.offset = float(np.median(self._offs))
        self.align_ok += 1
        if len(self._align_log) < 2000:
            # (attempt clock, this picture's offset, match difference, margin over the rest)
            self._align_log.append((round(t_raw, 4), round(tk - t_raw, 4), round(float(d[k]), 2),
                                    round(float(np.median(d) - d[k]), 2)))

    def _reject(self, why: str) -> None:
        r = getattr(self, "_align_rej", None)
        if r is not None:
            r[why] = r.get(why, 0) + 1

    def attempt_timing(self) -> dict:
        """How this attempt lined up with the best run, for the timing log: the offset at the
        start (respawn-detection jitter) and how it moved during the run (does the game itself
        drift?). Offsets in ms: + = this attempt was BEHIND the best run's clock."""
        log = list(getattr(self, "_align_log", []))
        out = dict(level=(self.level or {}).get("id"), best_T=round(float((self.level or {}).get("best_T", 0.0)), 3),
                   align_samples=len(log), align_rejected=dict(getattr(self, "_align_rej", {})),
                   plan=(self.plan[3] if self.plan else None))
        dts = list(getattr(self, "_dts_att", []))
        if dts:
            a = np.asarray(dts) * 1000.0
            out.update(dec_ms_median=round(float(np.median(a)), 1), dec_ms_p95=round(float(np.percentile(a, 95)), 1),
                       dec_ms_max=round(float(a.max()), 1), decisions=len(a))
        if len(log) >= 3:
            t = np.array([q[0] for q in log]); o = np.array([q[1] for q in log]) * 1000.0
            early = o[t < 1.0]
            out["offset_start_ms"] = round(float(np.median(early if len(early) else o[:3])), 1)
            out["offset_end_ms"] = round(float(np.median(o[-5:])), 1)
            out["span_s"] = round(float(t[-1] - t[0]), 2)
            if t[-1] - t[0] >= 1.0:
                slope, icpt = np.polyfit(t, o, 1)
                res = o - (slope * t + icpt)
                out["drift_ms_per_s"] = round(float(slope), 2)
                out["residual_ms_std"] = round(float(np.std(res)), 1)
                out["residual_ms_max"] = round(float(np.max(np.abs(res))), 1)
            # the raw series, thinned to <=120 points (t s, offset ms, match difference)
            step = max(1, len(log) // 120)
            out["series"] = [[q[0], round(q[1] * 1000.0, 1), q[2]] for q in log[::step]]
        return out

    def action(self, t: float, a: int, frame=None) -> tuple[int, bool]:
        """(action to send, came from here?) at attempt clock t, given the brain + shield's a."""
        t_raw = t
        if self._t_last is not None and 0.0 < t_raw - self._t_last < 4 * self.dt:
            if len(getattr(self, "_dts_att", [])) < 5000:
                self._dts_att.append(t_raw - self._t_last)
            self._dts.append(t_raw - self._t_last)
            del self._dts[:-31]
            if len(self._dts) >= 5:
                self.dec_dt = float(np.clip(np.median(self._dts), 0.8 * self.dt, 3.0 * self.dt))
        self._t_last = t_raw
        if frame is not None and self.level is not None:
            th = level_thumb(frame)
            if not _print_blank(th):
                self._align(t_raw, th)
                self._thumbs.append((t_raw + self.offset, th))
            # blank frame (capture not ready): no signal -- keep the old offset
            # instead of aligning to black and poisoning the replay timing
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
                out = pressed_at(self.level["best"], t, self.dec_dt)
            self.phase = "replay"
        elif t < w1 and iv is not None:
            out = pressed_at(iv, t, self.dec_dt)
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
        if self._prints and not _prints_valid(self._prints):
            # capture never delivered a real frame this attempt: the level id and
            # the recorded times are untrustworthy -- don't fork a blank record and
            # don't overwrite a good best with garbage timing
            self.plan = None
            return
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
        self.frontier_plan = plan[3] if plan else None
        if plan is not None and plan[3] == "frontier":
            f0_full = lv["best_T"] - self.LEAD
            if t_crash < plan[0] - 0.1:
                # the replay didn't reach the hand-over: hand over ~1 s before this spot next time
                self._fcut = max(0.5, t_crash - 1.0)
                if self.verbose:
                    print(f"[frontier] the replay died at {t_crash:.1f} s, before the hand-over at "
                          f"{plan[0]:.1f} s -- next time the brain takes over at {self._fcut:.1f} s")
            elif self._fcut is not None:
                self._fcut += 0.5                          # it held: give the replay a bit more
                if self._fcut >= f0_full:
                    self._fcut = None
        T = lv["best_T"]
        key = f"{T:.1f}"
        sp = lv["spots"].setdefault(key, {"deaths": 0, "tried": [], "worked": None, "drift": 0}) if T > 0 else None
        if plan is None and T > 0:
            # the brain's own attempt: did it get near the best spot, or die long before it?
            lv["early"] = lv.get("early", 0) + 1 if t_crash < T - self.WINDOW else 0
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
