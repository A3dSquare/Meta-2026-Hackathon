"""Approximate Geometry Dash physics for every game mode, used to PREVIEW actions.

Units: blocks and seconds, +y = up (screen up), x = distance ahead of the
icon. Values are approximations of GD 2.2 at 1x (vertical motion does not
depend on speed; horizontal speed is measured live). The cube's jump is
calibrated from your icon's real jumps; mode and gravity are recognised from
how the icon moves (ModeEstimator) -- no art involved.

Modes: cube, ship, ball, ufo, wave, robot, spider  (x mini, x upside-down)

The policy never sees these constants -- it sees the OUTCOMES of 8 fixed
action plans (hit? when? how much room?), which have the same meaning in every
mode. So improving or adding physics later doesn't change what the network
reads.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

MODES = ["cube", "ship", "ball", "ufo", "wave", "robot", "spider"]
MODE_ID = {m: i for i, m in enumerate(MODES)}


@dataclass
class Phys:
    # cube: measured from a real Stereo Madness recording (9 clean jumps, 1080p60):
    # apex 2.21 blocks, 0.43 s in the air. Still fine-tuned live from your own jumps.
    cube_v0: float = 20.5       # jump speed (blocks/s)
    cube_g: float = 95.0        # gravity (blocks/s^2)
    ship_a: float = 55.0        # (old single value; ship_up / ship_down are used)
    ship_up: float = 55.0       # ship acceleration while holding / falling while released
    ship_down: float = 55.0     # (measured live per mode: ModeCalibrator)
    ship_vmax: float = 12.0
    ball_g: float = 80.0
    ufo_v0: float = 14.0
    ufo_g: float = 80.0
    wave_slope: float = 1.0     # 45 degrees (mini wave: 2.0)
    robot_v0: float = 16.0
    robot_boost: float = 0.25   # seconds of extra rise while held
    spider_snap: float = 12.0   # how far it searches for the opposite surface
    mini: float = 0.8           # impulses / accelerations scale when mini
    # cube calibration from observed jumps
    _track: list | None = field(default=None, repr=False)
    n_cal: int = 0

    @property
    def apex(self) -> float:
        return self.cube_v0 ** 2 / (2 * self.cube_g)

    @property
    def airtime(self) -> float:
        return 2 * self.cube_v0 / self.cube_g

    def calibrate(self, y: float, vy_step: float, pressed: bool, grounded_before: bool, dt: float) -> None:
        """Learn the cube's apex / airtime from clean floor-to-floor jumps.

        pressed: a press took effect this step (after input delay);
        grounded_before: the icon was resting BEFORE this step's motion.
        Each jump's samples get a parabola fit (y = a + b t + c t^2), so the
        apex and airtime don't depend on where the frames happen to fall."""
        if self._track is None:
            if pressed and grounded_before and vy_step > 0.15:
                self._track = [y - vy_step, [(dt, y)]]
            return
        y0, pts = self._track
        t = pts[-1][0] + dt
        if y - y0 <= 0.05:                       # back on (about) the takeoff level
            self._track = None
            pts = [(tt, yy - y0) for tt, yy in pts if yy - y0 > 0.05]
            if len(pts) < 5:
                return
            T = np.array([q[0] for q in pts]); Y = np.array([q[1] for q in pts])
            c, b_, a_ = np.polyfit(T, Y, 2)
            disc = b_ * b_ - 4 * a_ * c
            if c >= 0 or disc <= 0:
                return
            rise = a_ - b_ * b_ / (4 * c)
            air = float(np.sqrt(disc) / -c)
            if 1.2 <= rise <= 3.5 and 0.25 <= air <= 0.8 and float(np.max(np.abs(np.polyval([c, b_, a_], T) - Y))) < 0.15:
                k = 0.3 if self.n_cal < 5 else 0.15     # first few jumps: adapt faster
                apex = self.apex + k * (rise - self.apex)
                air = self.airtime + k * (air - self.airtime)
                self.cube_v0 = 4 * apex / air
                self.cube_g = 8 * apex / air ** 2
                self.n_cal += 1
        elif t > 1.0:
            self._track = None
        else:
            pts.append((t, y))


@dataclass
class Body:
    y: float = 0.0
    vy: float = 0.0
    g: int = 1                  # +1 normal gravity, -1 upside down
    grounded: bool = True
    mode: str = "cube"
    mini: bool = False
    boost: float = 0.0          # robot: remaining boost time
    prev_hold: bool = False


def _scale(b: Body, p: Phys) -> float:
    return p.mini if b.mini else 1.0


def advance(b: Body, hold: bool, dt: float, speed: float, p: Phys, surfaces) -> Body:
    """One physics step (no collision with hazards). surfaces(x_unused) gives
    (floors, ceilings): heights the icon's centre may rest at from above / below."""
    s = _scale(b, p)
    press = hold and not b.prev_hold
    m = b.mode
    floors, ceils = surfaces
    was_grounded = b.grounded
    dv = 0.0   # velocity lost to gravity / thrust this step (for an exact position update)
    if m == "cube":
        if b.grounded and hold:
            b.vy, b.grounded = b.g * p.cube_v0 * s, False
        if not b.grounded:
            dv = b.g * p.cube_g * dt
            b.vy -= dv
    elif m == "robot":
        if b.grounded and hold:
            b.vy, b.grounded, b.boost = b.g * p.robot_v0 * s, False, p.robot_boost
        if not b.grounded:
            if hold and b.boost > 0:
                b.boost -= dt
            else:
                b.boost = 0.0
                dv = b.g * p.cube_g * dt
                b.vy -= dv
    elif m == "ball":
        if b.grounded and press:
            b.g, b.grounded, b.vy = -b.g, False, -b.g * 2.0   # flip and start moving away
        if not b.grounded:
            dv = b.g * p.ball_g * dt
            b.vy -= dv
    elif m == "ufo":
        if press:
            b.vy, b.grounded = b.g * p.ufo_v0 * s, False
        if not b.grounded:
            dv = b.g * p.ufo_g * dt
            b.vy -= dv
    elif m == "ship":
        dv = -b.g * (p.ship_up if hold else -p.ship_down) * s * dt
        b.vy -= dv
        b.vy = max(-p.ship_vmax * s, min(p.ship_vmax * s, b.vy))
        b.grounded = False
    elif m == "wave":
        b.vy = b.g * (1 if hold else -1) * p.wave_slope * (2.0 if b.mini else 1.0) * speed
        b.grounded = False
    elif m == "spider":
        if b.grounded and press:
            # teleport to the opposite surface and flip gravity
            if b.g == 1:
                tgt = [c for c in ceils if c > b.y + 0.5 and c - b.y <= p.spider_snap]
                b.y = min(tgt) if tgt else b.y + p.spider_snap
            else:
                tgt = [f for f in floors if f < b.y - 0.5 and b.y - f <= p.spider_snap]
                b.y = max(tgt) if tgt else b.y - p.spider_snap
            b.g, b.vy = -b.g, 0.0
            b.prev_hold = hold
            return b
        if not b.grounded:
            dv = b.g * p.cube_g * dt
            b.vy -= dv
    y_new = b.y + (b.vy + 0.5 * dv) * dt   # exact for constant acceleration
    # resting surfaces: falling onto a floor / rising into a ceiling
    if b.vy <= 0:
        land = [f for f in floors if y_new <= f <= b.y + 0.06]
        if land:
            y_new = max(land)
            if b.g == 1 or m in ("ship", "wave", "ufo"):
                b.grounded = b.g == 1 and m not in ("ship", "wave")
            b.vy = 0.0 if m != "wave" else b.vy
    if b.vy >= 0:
        hit_c = [c for c in ceils if b.y - 0.06 <= c <= y_new]
        if hit_c:
            y_new = min(hit_c)
            if b.g == -1 or m in ("ship", "wave", "ufo"):
                b.grounded = b.g == -1 and m not in ("ship", "wave")
            b.vy = 0.0 if m != "wave" else b.vy
    if b.grounded and hold and m in ("cube", "robot") and not was_grounded:
        # holding SPACE as it lands: GD jumps again straight away (no frame on the ground)
        b.vy, b.grounded = b.g * (p.robot_v0 if m == "robot" else p.cube_v0) * s, False
        if m == "robot":
            b.boost = p.robot_boost
    if b.grounded and m not in ("ship", "wave"):
        on = floors if b.g == 1 else ceils
        if not any(abs(y_new - f) < 0.07 for f in on):
            b.grounded = False                      # walked off an edge
    b.y = y_new
    b.prev_hold = hold
    return b


# --------------------------------------------------------------------------- #
# Action plans: the same 8 for every mode
# --------------------------------------------------------------------------- #
PLANS = [
    ("release", lambda k: False),
    ("hold", lambda k: True),
    ("tap now", lambda k: k == 0),
    ("tap in 2", lambda k: k == 2),
    ("tap in 4", lambda k: k == 4),
    ("tap in 6", lambda k: k == 6),
    ("hold 3 then release", lambda k: k < 3),
    ("release 3 then hold", lambda k: k >= 3),
]
N_PLANS = len(PLANS)
# Two more timings, so "press in k steps" exists for every k up to 4: with only 0/2/4/6
# the shield could not see "the only way is to press in 1 step" coming (live deaths).
# They live in the brain's spare inputs, so existing brains keep working.
EXTRA_PLANS = [
    ("tap in 1", lambda k: k == 1),
    ("tap in 3", lambda k: k == 3),
]
ALL_PLANS = PLANS + EXTRA_PLANS
N_ALL = len(ALL_PLANS)
N_EXTRA_FEATS = 3 * len(EXTRA_PLANS)


def surfaces_at(x: float, solid, floor, ceil, half: float = 0.4):
    """Heights the icon's centre can rest at (floors, from above) / hang at
    (ceilings, from below) when the icon is at distance x ahead."""
    floors, ceils = [], []
    for s0, s1, sb, st in solid:
        if s0 < x + half - 0.05 and s1 > x - half + 0.05:
            floors.append(st + 0.5)          # centre height standing on top
            ceils.append(sb - 0.5)           # centre height hanging under it
    if floor is not None:
        floors.append(floor + 0.5)
    if ceil is not None:
        ceils.append(ceil - 0.5)
    return floors, ceils


def simulate_plans(body: Body, haz, solid, floor, ceil, speed: float, p: Phys,
                   frame_skip: int = 2, horizon: float = 1.5, half: float = 0.4,
                   dec_dt: float | None = None, lag: int = 0, pending=(), motion=None,
                   extra_out: list | None = None):
    """Run every plan forward; return (features [N_PLANS*3 + 1], paths).

    haz / solid: boxes (x0, x1, bottom, top) in blocks relative to the icon.
    floor / ceil: surface heights (None = unknown). Per plan: hit?, time to
    hit / horizon (1 = never), clearance to the nearest hazard. Plus: time
    until the "release" plan leaves its current surface (ledge).
    dec_dt: real seconds per decision; lag: input delay in decisions --
    the next `lag` steps still play the already-sent actions in `pending`
    (oldest first), so "tap now" really takes effect `lag` steps later.
    motion: optional MotionLearner; its learned correction is added to every
    simulated step once it has proven itself (all plans step together so the
    network runs once per step).
    """
    dt = 1.0 / 30.0
    steps = int(horizon / dt)
    dec_dt = dec_dt or frame_skip / 60.0
    pending = list(pending)[-lag:] if lag > 0 else []
    use_m = motion is not None and motion.active_for(body.mode)   # off = zero cost
    feats = np.zeros(N_PLANS * 3 + 1, dtype=np.float32)
    B = [Body(body.y, body.vy, body.g, body.grounded, body.mode, body.mini, body.boost, body.prev_hold)
         for _ in ALL_PLANS]
    pts = [[(0.0, body.y)] for _ in ALL_PLANS]
    hit_t = [None] * N_ALL
    clear = [3.0] * N_ALL
    live = list(range(N_ALL))
    t_ledge = None
    x_prev = 0.0
    corr_by: dict = {}
    corr_key: dict = {}
    # 1/30 s steps for the first 0.5 s (where every plan does its pressing), then 1/15 s:
    # the physics is exact for any step, it only makes far-away timing a bit coarser
    fine = min(steps, int(round(0.5 / dt)))
    times = [k * dt for k in range(1, fine + 1)]
    while times[-1] + 1e-9 < horizon:
        times.append(min(horizon, times[-1] + 2 * dt))
    t_prev = 0.0
    for i, t in enumerate(times, start=1):
        if not live:
            break
        dt_i = t - t_prev
        x = speed * t
        jd = int(t_prev / dec_dt + 1e-9)            # decision index during this step (0 = now)
        if jd < len(pending):
            holds = [bool(pending[jd])] * len(live)  # already sent, still in the pipeline
        else:
            holds = [bool(ALL_PLANS[pi][1](jd - len(pending))) for pi in live]
        floors, ceils = surfaces_at(x, solid, floor, ceil, half)
        corr = None
        if use_m:
            # smooth motion: reuse the correction for 3 steps; but anything event-like
            # (press, release, landing / leaving a surface) gets a fresh one that step only
            keys = [(h, h and not B[pi].prev_hold, B[pi].grounded) for pi, h in zip(live, holds)]
            if (i - 1) % 3 == 0 or any(corr_key.get(pi) != k_ for pi, k_ in zip(live, keys)):
                c_ = motion.predict(MotionLearner.feats_batch([B[pi] for pi in live], holds,
                                                              floors, ceils, speed))
                corr_by = {pi: float(c_[k]) for k, pi in enumerate(live)}
                corr_key = dict(zip(live, keys))
            corr = [corr_by[pi] for pi in live]
        lo, hi = x_prev - half, x + half                 # swept horizontally (no tunnelling)
        still = []
        for k, pi in enumerate(live):
            b = B[pi]
            was = b.grounded
            advance(b, holds[k], dt_i, speed, p, (floors, ceils))
            if corr is not None and not b.grounded:
                s_ = dt_i / dt                            # learned leftover error (it was learned
                b.y += float(corr[k]) * dt_i * (1 + s_) / 2   # per 1/30 s step; a 2x step gets
                b.vy += float(corr[k]) * s_               # what two of them would add)
            if pi == 0 and was and not b.grounded and t_ledge is None and b.mode not in ("ship", "wave", "ufo"):
                t_ledge = t
            y = b.y
            pts[pi].append((x, y))
            for h0, h1, hb, ht in haz:
                if h0 < hi and h1 > lo:
                    gap = (y - half) - ht if y > (hb + ht) / 2 else hb - (y + half)
                    clear[pi] = min(clear[pi], gap)
                    if hb < y + half and ht > y - half:
                        hit_t[pi] = t
            if hit_t[pi] is None:
                for s0, s1, sb, st in solid:
                    if s0 < hi - 0.05 and s1 > lo + 0.05 and sb < y + half - 0.12 and st > y - half + 0.12:
                        hit_t[pi] = t                  # into the side of a block
                        break
            if hit_t[pi] is None and (y < -12 or y > 14):
                hit_t[pi] = t                          # out of the world
            if hit_t[pi] is None:
                still.append(pi)
        live = still
        x_prev = x
        t_prev = t
    paths = []
    extra = np.zeros(N_EXTRA_FEATS, dtype=np.float32)
    for pi in range(N_ALL):
        h = hit_t[pi]
        f3 = (1.0 if h is not None else 0.0, (h if h is not None else horizon) / horizon,
              float(np.clip(clear[pi], -2.0, 3.0)) / 2.0)
        if pi < N_PLANS:
            feats[3 * pi:3 * pi + 3] = f3
        else:
            extra[3 * (pi - N_PLANS):3 * (pi - N_PLANS) + 3] = f3
        paths.append((ALL_PLANS[pi][0], h is not None, pts[pi]))
    feats[-1] = (t_ledge if t_ledge is not None else horizon) / horizon
    if extra_out is not None:
        extra_out.append(extra)
    return feats, paths


# --------------------------------------------------------------------------- #
# Search: "is there ANY way to survive if I press / don't press now?"
# --------------------------------------------------------------------------- #
# The 8 fixed plans press at most once (or hold forever), so two obstacles in a row
# ("jump, land, jump again") made EVERY plan crash 0.5-1 s early and the safety shield
# had nothing to compare (live logs: most steps before a death looked "doomed either
# way"). This search plays every later decision well: at each decision every surviving
# state branches into press / release, identical states are merged, and a small beam
# per first action keeps it fast (~1-2 ms). It answers the question the shield needs.
SEARCH_KEYS = ("1", "00", "01")      # press now / release now + next / release now, press next


def search_best(body: Body, haz, solid, floor, ceil, speed: float, p: Phys,
                dec_dt: float, lag: int = 0, pending=(), horizon: float = 1.5,
                half: float = 0.5, beam: int = 10, want_path: bool = False,
                late_every: float = 0.0, portals=None) -> dict:
    """Longest survival (fraction of `horizon`, 1.0 = survives it all) for each first
    action ("1" = press now, "00" = release now and at the next decision, "01" =
    release now, press at the next decision), with every later decision chosen well.
    With want_path, also the path (x, y) of the best line found and whether it crashes.

    Each state carries the SET of first actions that can reach it (bit 0 = "1",
    bit 1 = "00", bit 2 = "01"), so lines that end up in the same state (e.g. back on
    the ground) are simulated once."""
    dt = 1.0 / 30.0
    pending = list(pending)[-lag:] if lag > 0 else []
    n_pend = len(pending)
    fine = int(round(0.5 / dt))
    times = [k * dt for k in range(1, fine + 1)]
    while times[-1] + 1e-9 < horizon:
        times.append(min(horizon, times[-1] + 2 * dt))
    root_node = (0.0, body.y, None, 0.0)          # path node: (x, y, parent, effort so far)
    ALL, B1, B00, B01 = 7, 1, 2, 4
    # a state: [Body, root mask, hold this step, path node]
    states = [(Body(body.y, body.vy, body.g, body.grounded, body.mode, body.mini, body.boost, body.prev_hold),
               ALL, False, root_node)]
    best = [0.0, 0.0, 0.0]
    best_node = [root_node, root_node, root_node]

    def credit(mask: int, t_frac: float, node) -> None:
        for i in range(3):
            if mask >> i & 1 and (t_frac > best[i] or (want_path and t_frac == best[i]
                                                         and node[3] < best_node[i][3])):
                best[i] = t_frac
                best_node[i] = node

    every = max(1, int(round(late_every / dec_dt))) if late_every > 0 else 1
    t_prev, x_prev, last_jd = 0.0, 0.0, -1
    for t in times:
        if not states:
            break
        x = speed * t
        jd = int(t_prev / dec_dt + 1e-9)            # decision index during this step (0 = now)
        if jd < n_pend:
            hp = bool(pending[jd])
            states = [(b, m, hp, nd) for b, m, _h, nd in states]
        elif jd != last_jd and (t_prev < 0.5 or jd % every == 0):
            k = jd - n_pend                         # which free decision this is (0 = now)
            new = []
            for b, m, _h, nd in states:
                for h in (True, False):
                    if k == 0:
                        m2 = m & (B1 if h else (B00 | B01))
                    elif k == 1:
                        m2 = m & ((B1 | B01) if h else (B1 | B00))
                    else:
                        m2 = m
                    if m2:
                        new.append((Body(b.y, b.vy, b.g, b.grounded, b.mode, b.mini, b.boost, b.prev_hold),
                                    m2, h, nd))
            states = new
        last_jd = jd
        dt_i = t - t_prev
        floors, ceils = surfaces_at(x, solid, floor, ceil, half)
        lo, hi = x_prev - half, x + half                 # swept horizontally (no tunnelling)
        hz = [(h_[2], h_[3]) for h_ in haz if h_[0] < hi and h_[1] > lo]
        so = [(s_[2], s_[3]) for s_ in solid if s_[0] < hi - 0.05 and s_[1] > lo + 0.05]
        # portals: a path that flies through one continues in that mode / gravity
        # (whole-level test: every jump into the ship portal "crashed" as a cube)
        pt_here = [q for q in (portals or ()) if q[0] < hi and q[1] > lo]
        if pt_here:
            for b, m, h, nd in states:
                for _x0, _x1, pb, ptop, kind, val in pt_here:
                    if pb < b.y + half and ptop > b.y - half:
                        if kind == "mode" and b.mode != val:
                            b.mode, b.boost = val, 0.0
                            if val in ("ship", "wave", "ufo"):
                                b.grounded = False
                            b.vy *= 0.5
                        elif kind == "g" and b.g != val:
                            b.g, b.grounded = val, False
        alive: list = []
        index: dict = {}
        for b, m, h, nd in states:
            pressed = h and not b.prev_hold
            advance(b, h, dt_i, speed, p, (floors, ceils))
            y = b.y
            # the drawn line = the surviving path with the LEAST effort (presses, holding):
            # with nothing dangerous ahead every path survives, and an arbitrary one was
            # drawn -- full of pointless jumps
            node = (x, y, nd, nd[3] + (1.0 if pressed else 0.0) + (0.05 if h else 0.0)) if want_path else nd
            dead = y < -12 or y > 14
            if not dead:
                yt, yb_ = y + half, y - half
                for hb, ht in hz:
                    if hb < yt and ht > yb_:
                        dead = True
                        break
                if not dead:
                    for sb, st in so:
                        if sb < yt - 0.12 and st > yb_ + 0.12:
                            dead = True
                            break
            if dead:
                credit(m, t / horizon, node)
                continue
            key = (h, b.prev_hold, round(y * 20.0), round(b.vy * 1.6667), b.grounded, b.g, round(b.boost, 2), b.mode)
            j = index.get(key)
            if j is not None:                       # same state reached by other first actions
                s0 = alive[j]
                keep_node = node if (want_path and node[3] < s0[3][3]) else s0[3]
                if m & ~s0[1] or keep_node is not s0[3]:
                    alive[j] = (s0[0], s0[1] | m, s0[2], keep_node)
                continue
            index[key] = len(alive)
            alive.append((b, m, h, node))
        if len(alive) > beam:                       # keep a spread of heights for each first action
            keep = set()
            for i in range(3):
                lst = [j for j, s_ in enumerate(alive) if s_[1] >> i & 1]
                if len(lst) > beam:
                    # evenly spread over height INCLUDING both extremes (the top one is
                    # e.g. a ship that kept climbing -- losing it lost every way over a wall)
                    lst.sort(key=lambda j: (alive[j][0].y, alive[j][0].vy))
                    n_ = len(lst) - 1
                    lst = [lst[int(round(q * n_ / (beam - 1)))] for q in range(beam)]
                keep.update(lst)
            alive = [alive[j] for j in sorted(keep)]
        states = alive
        t_prev, x_prev = t, x
    for b, m, h, nd in states:
        credit(m, 1.0, nd)
    out = {k: float(best[i]) for i, k in enumerate(SEARCH_KEYS)}
    if want_path:
        # ties -> prefer waiting: the line then shows the LATEST safe moment to press
        i_best = max((1, 2, 0), key=lambda i: best[i])
        node = best_node[i_best]
        pts = []
        while node is not None:
            pts.append((node[0], node[1]))
            node = node[2]
        out["path"] = pts[::-1]
        out["path_hit"] = best[i_best] < 1.0
    return out


def robust_search(body: Body, haz, solid, floor, ceil, speed: float, p: Phys, dec_dt: float,
                  lag: int, pending, pending_long=None, horizon: float = 1.5, half: float = 0.5,
                  want_path: bool = False, lag_alt: int | None = None, portals=None) -> dict:
    """search_best in the measured world AND a slightly different one: the press arrives
    one step EARLIER or LATER (lag_alt: whichever the measured delays say is likely --
    live they split between 1 and 2 steps, so with "2" a jump often started a step early
    and landed on the spike's far end) and the level scrolls 4% faster. Replaying the
    real last seconds before live deaths, timing half a step off explained many crashes.

    Per first action one SAFETY score (what the brain sees and the shield uses):
      1.0          survives in both worlds
      0.90..0.99   survives as measured, crashes in the worse world (later = higher)
      0..0.85      crashes even as measured (0.85 x the fraction of the horizon survived)
    The raw values are kept under "nominal" / "worse"."""
    if not haz and not solid:          # nothing ahead to hit: every choice is safe (fast path)
        a = {k: 1.0 for k in SEARCH_KEYS}
        a.update(nominal=dict(a), worse=dict(a), path=[], path_hit=False)
        return a
    a = search_best(body, haz, solid, floor, ceil, speed, p, dec_dt, lag, pending, horizon, half,
                    want_path=want_path, portals=portals)
    if all(a[k] < 1.0 for k in SEARCH_KEYS):
        b = {k: 0.0 for k in SEARCH_KEYS}   # crashes as measured -> the worse world can't help
    else:
        alt = lag + 1 if lag_alt is None else max(0, int(lag_alt))
        pl = list(pending_long) if pending_long is not None else [False] * max(0, alt - len(pending)) + list(pending)
        b = search_best(body, haz, solid, floor, ceil, speed * 1.04, p, dec_dt, alt, pl[len(pl) - alt:] if alt else [],
                        horizon, half, portals=portals)
    a["nominal"] = {k: a[k] for k in SEARCH_KEYS}
    a["worse"] = {k: b[k] for k in SEARCH_KEYS}
    for k in SEARCH_KEYS:
        a[k] = safety_score(a["nominal"][k], b[k])
    return a


def safety_score(nominal: float, worse: float) -> float:
    if nominal < 1.0:
        return 0.85 * float(nominal)
    return 1.0 if worse >= 1.0 else 0.90 + 0.09 * float(worse)


def safety_level(v: float) -> int:
    """2 = survives even in the worse world, 1 = only as measured, 0 = crashes."""
    return 2 if v >= 1.0 - 1e-6 else (1 if v >= 0.895 else 0)


def portal_effect(hue: float):
    """What a portal of this colour does: ("mode", name) / ("g", +1 or -1) / None (unsure)."""
    if hue is None or hue < 0:
        return None
    d = {m: min(abs(hue - h), 1 - abs(hue - h)) for m, h in {**PORTAL_HUES, **GRAVITY_HUES}.items()}
    m = min(d, key=d.get)
    if d[m] >= 0.04:
        return None
    return ("g", GRAVITY_DIR[m]) if m in GRAVITY_HUES else ("mode", m)


def resolve_start(y: float, solid, half: float = 0.5, max_move: float = 0.45) -> float:
    """The icon's start height, moved out of any block it overlaps (by at most max_move):
    a live icon is never inside a block, so an overlap is detection error, and the
    preview would call it a crash at t=0 (whole-level test on a recorded completion)."""
    for _ in range(3):
        hit = [(sb, st) for s0, s1, sb, st in solid
               if s0 < half - 0.05 and s1 > -half + 0.05 and sb < y + half - 0.12 and st > y - half + 0.12]
        if not hit:
            return y
        sb, st = hit[0]
        up, down = st + half - y, y - (sb - half)        # move needed to sit on / hang under it
        if min(up, down) > max_move:
            return y
        y = y + up if up <= down else y - down
    return y


def inst_vy(mode: str, g: int, mini: bool, v_avg: float, hold: bool, grounded: bool,
            dt: float, p: Phys) -> float:
    """Measured speed is the AVERAGE over the last step; the simulation wants the
    speed at the END of it (they differ by half a step of gravity / thrust)."""
    if grounded or mode == "wave":
        return v_avg
    s = p.mini if mini else 1.0
    if mode == "ship":
        a = g * (p.ship_up if hold else -p.ship_down) * s
        return max(-p.ship_vmax * s, min(p.ship_vmax * s, v_avg + a * dt / 2))
    G = {"ball": p.ball_g, "ufo": p.ufo_g}.get(mode, p.cube_g)
    return v_avg - g * G * dt / 2


# --------------------------------------------------------------------------- #
# Learned correction: what the formulas get wrong, learned from real motion
# --------------------------------------------------------------------------- #
class MotionLearner:
    """A small network that learns the LEFTOVER error of the physics formulas.

    Every step of real play is a free example: the formulas predicted the
    icon would move at speed v_pred, the screen showed v_obs; the network
    learns v_obs - v_pred from the icon's state (mode, gravity, speed, held,
    near a floor / ceiling...). Its output is added to every simulated step.

    Safety: it is only used once it beats the plain formulas on NEW examples
    (checked before training on them), and its weight ("trust") follows that
    margin -- so it can't make the preview worse than the formulas alone.
    Plain numpy, ~2k weights, trained online a little every step.
    """

    N_IN = 16
    H = 32
    CLIP = 15.0          # largest correction it may learn (blocks/s)

    def __init__(self, seed: int = 0, buf: int = 20000) -> None:
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 1 / np.sqrt(self.N_IN), (self.N_IN, self.H))
        self.b1 = np.zeros(self.H)
        self.W2 = r.normal(0, 1 / np.sqrt(self.H), (self.H, self.H))
        self.b2 = np.zeros(self.H)
        self.W3 = np.zeros((self.H, 1))          # starts as "no correction"
        self.b3 = np.zeros(1)
        self._params = ["W1", "b1", "W2", "b2", "W3", "b3"]
        self._m = {k: np.zeros_like(getattr(self, k)) for k in self._params}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in self._params}
        self._t = 0
        self.X = np.zeros((buf, self.N_IN), np.float32)
        self.Y = np.zeros(buf, np.float32)
        self.n = 0                                # examples seen (ever)
        # per game mode: running |error| of the formulas alone / with the correction,
        # examples seen, and how much the correction is used ("trust")
        self.m_err_phys = np.zeros(len(MODES))
        self.m_err_corr = np.zeros(len(MODES))
        self.m_n = np.zeros(len(MODES), np.int64)
        self.m_trust = np.zeros(len(MODES))
        self._rng = r

    # -- features of a simulated / observed state ------------------------------
    @staticmethod
    def feats(b: Body, hold: bool, floors, ceils, speed: float) -> np.ndarray:
        f = np.zeros(MotionLearner.N_IN, np.float32)
        f[MODE_ID.get(b.mode, 0)] = 1.0
        f[7] = float(b.g)
        f[8] = 1.0 if b.mini else 0.0
        f[9] = max(-2.0, min(2.0, b.vy / 20.0))
        f[10] = 1.0 if hold else 0.0
        f[11] = 1.0 if (hold and not b.prev_hold) else 0.0
        f[12] = 1.0 if b.grounded else 0.0
        below = [b.y - q for q in floors if q <= b.y + 0.06]
        above = [q - b.y for q in ceils if q >= b.y - 0.06]
        f[13] = min(4.0, max(0.0, min(below))) / 4.0 if below else 1.0
        f[14] = min(4.0, max(0.0, min(above))) / 4.0 if above else 1.0
        f[15] = speed / 20.0
        return f

    # -- network ----------------------------------------------------------------
    def _fwd(self, X):
        h1 = np.tanh(X @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        return h1, h2, (h2 @ self.W3 + self.b3)[:, 0]

    def raw(self, X) -> np.ndarray:
        return np.clip(self._fwd(np.atleast_2d(X))[2], -self.CLIP, self.CLIP)

    # overall numbers (for the console): example-weighted over modes
    @property
    def trust(self) -> float:
        n = self.m_n.sum()
        return float((self.m_trust * self.m_n).sum() / n) if n else 0.0

    @property
    def err_phys(self) -> float:
        n = self.m_n.sum()
        return float((self.m_err_phys * self.m_n).sum() / n) if n else 0.0

    @property
    def err_corr(self) -> float:
        n = self.m_n.sum()
        return float((self.m_err_corr * self.m_n).sum() / n) if n else 0.0

    def active(self) -> bool:
        return bool((self.m_trust > 0.05).any())

    def active_for(self, mode: str) -> bool:
        return bool(self.m_trust[MODE_ID.get(mode, 0)] > 0.05)

    @staticmethod
    def feats_batch(bodies, holds, floors, ceils, speed: float) -> np.ndarray:
        """feats() for many bodies at once (same surfaces), without per-row numpy calls."""
        n = len(bodies)
        F = np.zeros((n, MotionLearner.N_IN), np.float32)
        rows = []
        for b, h in zip(bodies, holds):
            lo = hi = None
            for q in floors:
                if q <= b.y + 0.06:
                    d = b.y - q
                    lo = d if lo is None or d < lo else lo
            for q in ceils:
                if q >= b.y - 0.06:
                    d = q - b.y
                    hi = d if hi is None or d < hi else hi
            v = b.vy / 20.0
            rows.append((MODE_ID.get(b.mode, 0), float(b.g), 1.0 if b.mini else 0.0,
                         -2.0 if v < -2.0 else (2.0 if v > 2.0 else v), 1.0 if h else 0.0,
                         1.0 if (h and not b.prev_hold) else 0.0, 1.0 if b.grounded else 0.0,
                         1.0 if lo is None else min(4.0, max(0.0, lo)) / 4.0,
                         1.0 if hi is None else min(4.0, max(0.0, hi)) / 4.0))
        R = np.array(rows, np.float32)
        F[np.arange(n), R[:, 0].astype(int)] = 1.0
        F[:, 7:15] = R[:, 1:]
        F[:, 15] = speed / 20.0
        return F

    def predict(self, X) -> np.ndarray:
        """Correction to add (blocks/s), scaled by how much it has earned in each row's mode."""
        X = np.atleast_2d(X)
        return self.raw(X) * self.m_trust[np.argmax(X[:, :len(MODES)], axis=1)]

    # -- learning ---------------------------------------------------------------
    def add(self, x: np.ndarray, resid: float, train: bool = True) -> None:
        resid = float(np.clip(resid, -self.CLIP, self.CLIP))
        # judge on the new example BEFORE learning from it (honest score), per mode
        mi = int(np.argmax(x[:len(MODES)]))
        e_p = abs(resid)
        e_c = abs(resid - float(self.raw(x)[0]))
        k = max(0.003, 1.0 / (self.m_n[mi] + 1))          # running mean, then a slow moving average
        self.m_err_phys[mi] += k * (e_p - self.m_err_phys[mi])
        self.m_err_corr[mi] += k * (e_c - self.m_err_corr[mi])
        self.m_n[mi] += 1
        if self.m_n[mi] >= 300 and self.m_err_phys[mi] > 1e-6:
            gain = (self.m_err_phys[mi] - self.m_err_corr[mi]) / self.m_err_phys[mi]   # error removed
            self.m_trust[mi] = float(np.clip((gain - 0.1) * 4.0, 0.0, 1.0))   # must remove >10% first
        i = self.n % len(self.Y)
        self.X[i], self.Y[i] = x, resid
        self.n += 1
        if train and self.n >= 64:
            self.train_step()

    WD = 0.0             # weight decay (keeps it from memorising detector jitter)

    def train_step(self, batch: int = 64, lr: float = 1e-3) -> None:
        m = min(self.n, len(self.Y))
        idx = self._rng.integers(0, m, batch)
        X, Y = self.X[idx].astype(np.float64), self.Y[idx].astype(np.float64)
        h1, h2, out = self._fwd(X)
        d = np.clip(out - Y, -3.0, 3.0)[:, None] / batch   # Huber-ish: big outliers can't dominate
        g = {"W3": h2.T @ d, "b3": d.sum(0)}
        dh2 = (d @ self.W3.T) * (1 - h2 ** 2)
        g["W2"], g["b2"] = h1.T @ dh2, dh2.sum(0)
        dh1 = (dh2 @ self.W2.T) * (1 - h1 ** 2)
        g["W1"], g["b1"] = X.T @ dh1, dh1.sum(0)
        self._t += 1
        for k_ in self._params:
            self._m[k_] = 0.9 * self._m[k_] + 0.1 * g[k_]
            self._v[k_] = 0.999 * self._v[k_] + 0.001 * g[k_] ** 2
            mh = self._m[k_] / (1 - 0.9 ** self._t)
            vh = self._v[k_] / (1 - 0.999 ** self._t)
            w = getattr(self, k_)
            setattr(self, k_, w - lr * (mh / (np.sqrt(vh) + 1e-8) + self.WD * w))

    # -- one observed step --------------------------------------------------------
    def observe(self, body: Body, hold: bool, floors, ceils, speed: float, dt: float,
                p: Phys, v_obs: float) -> None:
        """body: the icon's state at the START of the step just played (as the
        preview saw it); hold: what reached the game during it; v_obs: the
        icon's measured vertical speed over that step (blocks/s)."""
        b = Body(body.y, body.vy, body.g, body.grounded, body.mode, body.mini, body.boost, body.prev_hold)
        x = self.feats(b, hold, floors, ceils, speed)
        advance(b, hold, dt, speed, p, (floors, ceils))
        v_pred = (b.y - body.y) / dt
        if b.grounded:
            return      # resting, or the formulas land it this step: landings are timing
                        # artefacts (where in the step it touches down), not physics to learn
        y_end = body.y + v_obs * dt
        if any(abs(y_end - q) < 0.15 for q in floors) or any(abs(y_end - q) < 0.15 for q in ceils):
            return      # it really touched a surface this step: same reason
        self.add(x, v_obs - v_pred)

    # -- persistence ----------------------------------------------------------------
    def state(self) -> dict:
        d = {k: getattr(self, k) for k in self._params}
        m = min(self.n, len(self.Y))
        d.update(X=self.X[:m], Y=self.Y[:m], n=np.array([self.n]), t=np.array([self._t]),
                 m_err=np.stack([self.m_err_phys, self.m_err_corr, self.m_n, self.m_trust]))
        for k in self._params:
            d["m_" + k], d["v_" + k] = self._m[k], self._v[k]
        return d

    def load_state(self, d) -> None:
        for k in self._params:
            setattr(self, k, np.array(d[k], np.float64))
            self._m[k], self._v[k] = np.array(d["m_" + k]), np.array(d["v_" + k])
        m = len(d["Y"])
        self.X[:m], self.Y[:m] = d["X"], d["Y"]
        self.n, self._t = int(d["n"][0]), int(d["t"][0])
        e = d["m_err"]
        self.m_err_phys, self.m_err_corr = np.array(e[0]), np.array(e[1])
        self.m_n, self.m_trust = np.array(e[2]).astype(np.int64), np.array(e[3])


class ModeCalibrator:
    """Measures every mode's physics from the icon's REAL motion, the way the cube jump is
    measured (Phys.calibrate): ship climb / fall rate and top speed, ball and UFO gravity,
    UFO tap strength, wave angle. The defaults are guesses; your game's values replace them.

    It cuts the motion into pieces where nothing changes (same mode / gravity / size,
    same button state after the input delay, in the air, not touching anything) and fits a
    parabola to each piece's heights (y = a + b t + c t^2): 2c is the acceleration, b the
    speed. Fitting many frames makes the player box's wobble (~0.2 blocks) average out.
    A value is used once it has 3 clean pieces (the median of the last 25), only within a
    sane range, and only while the recogniser is sure of the mode (a wrong mode would teach
    the wrong numbers)."""

    PARAMS = {   # name: (low, high) sane range
        "ship_up": (20.0, 140.0), "ship_down": (20.0, 140.0), "ship_vmax": (5.0, 25.0),
        "ball_g": (30.0, 180.0), "ufo_g": (30.0, 180.0), "ufo_v0": (5.0, 30.0),
        "wave_slope": (0.5, 2.0),
    }
    MIN_N = 3

    def __init__(self) -> None:
        self.samples: dict = {k: deque(maxlen=25) for k in self.PARAMS}
        self._seg: list = []
        self._key = None
        self._t = 0.0
        self._stable = 0
        self._last_mode = None
        self._prev_hold = False
        self._seg_dt0 = 0.0

    def counts(self) -> dict:
        return {k: len(v) for k, v in self.samples.items()}

    def update(self, y: float, eff_hold: bool, dt: float, mode: str, g: int, mini: bool,
               resting: bool, speed: float, p: Phys, confident: bool = True) -> None:
        """One step: y = icon height (blocks, up), eff_hold = the button state that drove
        the motion INTO this sample (after the input delay), resting = on a surface."""
        self._t += dt
        self._stable = self._stable + 1 if mode == self._last_mode else 0
        self._last_mode = mode
        edge = bool(eff_hold) and not self._prev_hold
        self._prev_hold = bool(eff_hold)
        key = (mode, g, mini, bool(eff_hold) if mode in ("ship", "wave") else None)
        usable = (confident and not resting and self._stable >= 5
                  and mode in ("ship", "wave", "ball", "ufo"))
        jumped = bool(self._seg) and abs(y - self._seg[-1][1]) > 2.0
        split = (key != self._key or not usable or jumped
                 or (mode == "ufo" and edge) or len(self._seg) >= 10)
        if split:
            self._finish(p, speed)
            last = self._seg[-1] if self._seg else None
            self._seg = []
            self._key = key
            self._seg_edge = mode == "ufo" and edge     # this piece starts right after a tap
            self._seg_dt0 = dt
            # the sample before a button change belongs to the new piece too (the new
            # button state drove the motion from it); a UFO tap starts after the tap
            if usable and last is not None and not jumped and not (mode == "ufo" and edge):
                self._seg.append(last)
        if usable:
            self._seg.append((self._t, float(y)))

    def _finish(self, p: Phys, speed: float) -> None:
        seg, key = self._seg, self._key
        if key is None or len(seg) < 4:
            return
        mode, g, mini, hold = key
        T = np.array([q[0] for q in seg]) - seg[0][0]
        Y = np.array([q[1] for q in seg])
        try:
            c, b, a = np.polyfit(T, Y, 2)
        except Exception:
            return
        if float(np.max(np.abs(np.polyval([c, b, a], T) - Y))) > 0.22:
            return                                 # not one smooth piece (hit something?)
        s = p.mini if mini else 1.0
        acc = 2.0 * c * g / s                      # acceleration, + = away from the floor
        v0, v1 = b * g / s, (b + 2 * c * T[-1]) * g / s
        add = self._add
        if mode == "ship":
            # top speed: the configured one, or what the ship is actually seen reaching
            sg = 1.0 if hold else -1.0              # direction it's being pushed
            a_fit, v_fit = self._ship_fit(T, (Y - Y[0]) * g / s, sg, p.ship_vmax)
            if a_fit is not None:
                add("ship_up" if hold else "ship_down", a_fit)
            if v_fit is not None:
                add("ship_vmax", v_fit)
            elif a_fit is None and abs(acc) < 12.0 and (v1 > 0) == bool(hold) and abs(v1) > 4.0:
                add("ship_vmax", abs(0.5 * (v0 + v1)))   # flat at full speed the whole piece
        elif mode == "wave":
            if abs(acc) < 15.0 and speed > 1.0:
                add("wave_slope", abs(0.5 * (v0 + v1)) / (speed * (2.0 if mini else 1.0)))
        elif mode in ("ball", "ufo"):
            add("ball_g" if mode == "ball" else "ufo_g", -acc)
            if mode == "ufo" and getattr(self, "_seg_edge", False) and v0 > 0:
                add("ufo_v0", v0 + (-acc) * float(self._seg_dt0))   # the tap was a step earlier
        self._apply(p)

    @staticmethod
    def _ship_fit(T, Y, sg: float, vmax_guess: float):
        """(acceleration, top speed) of a ship piece, allowing for the speed being capped
        (a plain parabola fit reads a capped climb as a slow one). Exact closed form on a
        grid of (acceleration, start speed, top speed). Each is None unless the piece pins
        it down: a third lower / higher has to fit clearly worse."""
        A = np.arange(15.0, 152.5, 2.5)[:, None, None, None]
        VM = np.arange(5.0, 20.5, 1.0)[None, None, :, None]
        V0 = np.linspace(-1.0, 1.0, 33)[None, :, None, None] * VM
        t = np.asarray(T, float)[None, None, None, :]
        u = sg * V0                                   # start speed in the pushed direction
        tc = np.clip((VM - u) / A, 0.0, None)         # time until it hits top speed
        tm = np.minimum(t, tc)
        y = u * tm + 0.5 * A * tm ** 2 + VM * (t - tm)
        R = sg * y - np.asarray(Y, float)[None, None, None, :]
        R = R - R.mean(axis=3, keepdims=True)          # free start height
        err = np.sqrt((R ** 2).mean(axis=3))           # (a, v0, vmax)
        e = err.min(axis=1)                            # best start speed
        ia, iv = np.unravel_index(int(np.argmin(e)), e.shape)
        e0 = float(e[ia, iv])
        if e0 > 0.15:
            return None, None
        a_, v_ = float(A[ia, 0, 0, 0]), float(VM[0, 0, iv, 0])
        e_a = e.min(axis=1)                            # best over top speed, per acceleration
        e_v = e.min(axis=0)                            # best over acceleration, per top speed
        def pinned(curve, grid, x):
            lo = curve[int(np.argmin(np.abs(grid - x * 0.7)))]
            hi = curve[int(np.argmin(np.abs(grid - x * 1.4)))]
            return min(lo, hi) > 1.15 * e0 + 0.005
        a_ok = pinned(e_a, A[:, 0, 0, 0], a_)
        v_ok = pinned(e_v, VM[0, 0, :, 0], v_) and v_ < 19.5
        return (a_ if a_ok else None), (v_ if v_ok else None)

    def _add(self, name: str, v: float) -> None:
        lo, hi = self.PARAMS[name]
        if lo * 0.7 <= v <= hi * 1.3:              # far outside = a bad piece, not a value
            self.samples[name].append(float(v))

    def _apply(self, p: Phys) -> None:
        for name, (lo, hi) in self.PARAMS.items():
            q = self.samples[name]
            if len(q) >= self.MIN_N:
                setattr(p, name, float(np.clip(np.median(q), lo, hi)))

    def state(self) -> np.ndarray:
        """[value, count] per parameter (for saving)."""
        return np.array([[float(np.median(self.samples[k])) if self.samples[k] else 0.0,
                          len(self.samples[k])] for k in self.PARAMS], np.float64)

    def load_state(self, arr, p: Phys) -> None:
        for (name, _rng), (v, n) in zip(self.PARAMS.items(), np.asarray(arr)):
            if n >= self.MIN_N:                    # keep it as that many votes for the value
                self.samples[name].extend([float(v)] * int(min(n, 10)))
        self._apply(p)

    def summary(self, p: Phys) -> str:
        n = self.counts()
        parts = []
        for k, label in (("ship_up", "ship up"), ("ship_down", "ship down"), ("ship_vmax", "ship max"),
                         ("ball_g", "ball g"), ("ufo_g", "ufo g"), ("ufo_v0", "ufo tap"),
                         ("wave_slope", "wave")):
            if n[k]:
                parts.append(f"{label} {getattr(p, k):.1f}{'' if n[k] >= self.MIN_N else '?'}({n[k]})")
        return ", ".join(parts) if parts else "no other modes seen yet"


def save_motion(path: str, learner: MotionLearner, phys: Phys, est, calib=None) -> None:
    """Learned correction + tuned jump + measured input delay, in one small file
    shared by every brain and level (it's the game's physics, not a level's)."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    d = learner.state()
    d["phys"] = np.array([phys.cube_v0, phys.cube_g, phys.n_cal, getattr(est, "lag", 0)], np.float64)
    if calib is not None:
        d["modes"] = calib.state()
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **d)
    os.replace(tmp, path)


def load_motion(path: str, learner: MotionLearner, phys: Phys, est, calib=None) -> bool:
    import os
    if not os.path.exists(path):
        return False
    try:
        with np.load(path) as d:
            learner.load_state(d)
            v0, g, n_cal, lag = d["phys"]
            if calib is not None and "modes" in d.files:
                calib.load_state(d["modes"], phys)
        if n_cal > 0:
            phys.cube_v0, phys.cube_g, phys.n_cal = float(v0), float(g), int(n_cal)
        est.lag = int(lag)
        return True
    except Exception as ex:
        print(f"[motion] couldn't load {path}: {ex!r} -- starting it fresh")
        return False


# --------------------------------------------------------------------------- #
# Which mode / gravity / size is the icon in?  (from how it moves)
# --------------------------------------------------------------------------- #
PORTAL_HUES = {  # approximate GD portal colours (hue 0..1) -> mode hint
    "cube": 0.33, "ship": 0.86, "ball": 0.02, "ufo": 0.10, "wave": 0.52, "spider": 0.75,
}


GRAVITY_HUES = {"grav_flip": 0.15, "grav_norm": 0.58}   # yellow = upside down, blue = normal
GRAVITY_DIR = {"grav_flip": -1, "grav_norm": 1}


class ModeEstimator:
    """Picks the (mode, gravity, mini) whose physics best explains the icon's
    recent vertical motion given the actions actually taken."""

    def __init__(self, window: int = 20) -> None:
        self.hist: deque = deque(maxlen=window)   # (y_blocks, hold)
        self.mode, self.g, self.mini = "cube", 1, False
        self.conf = 0.0
        self._cand_streak = (None, 0)
        self._hint, self._hint_t = None, 0
        self.lag = 0                  # input delay (decisions) between a press and the icon reacting
        self.lag_alt = None           # the other likely delay (measured presses split between two)
        self._lag_votes = (None, 0)
        self._lag_ok = 0              # checks in a row that confirmed the current delay
        self._n_upd = 0
        self._rest = []

    def eff_hold(self, back: int = 0) -> bool:
        """The action that actually drove the motion `back` steps ago (after input delay)."""
        i = len(self.hist) - 1 - back - self.lag
        return bool(self.hist[i][1]) if 0 <= i < len(self.hist) else False

    def pending(self, n: int | None = None) -> list:
        """Actions already sent that haven't reached the icon yet (oldest first).
        n: how many (default: the measured input delay)."""
        n = self.lag if n is None else int(n)
        h = list(self.hist)
        out = [x[1] for x in h[max(0, len(h) - n):]] if n > 0 else []
        return [0.0] * (n - len(out)) + out

    def resting_before(self) -> bool:
        return bool(self._rest and self._rest[-1])

    def reset(self) -> None:
        self.hist.clear()
        self.mode, self.g, self.mini = "cube", 1, False   # GD attempts start as a normal cube
        self.conf = 0.0
        self._cand_streak = (None, 0)
        self._hint, self._hint_t = None, 0     # the last attempt's portal doesn't carry over
        self._ghint_t = 0
        self._rest = []                        # (the measured input delay does carry over)

    def gravity_hint(self) -> None:
        """An orb / pad is at the icon: gravity may flip now (allow a fast gravity change)."""
        self._ghint_t = 15

    def portal_hint(self, hue: float) -> None:
        self._hint_t = 30              # a portal was crossed: allow a fast switch
        self._hint = None
        while len(self.hist) > 2:      # motion from before the portal follows other physics
            self.hist.popleft()
        if hue < 0:
            return
        d = {m: min(abs(hue - h), 1 - abs(hue - h)) for m, h in {**PORTAL_HUES, **GRAVITY_HUES}.items()}
        m = min(d, key=d.get)
        if m in GRAVITY_HUES:
            if d[m] < 0.04:
                self.g = GRAVITY_DIR[m]
            return
        if d[m] < 0.04:                # colour clearly says which mode: take it now, the
            self._hint = m             # motion check can still overrule it within the window.
            if m != self.mode:         # (pillars that looked like portals are filtered out
                self.mode, self.mini = m, False   # by the env: they overlap a detected block)

    REST = 2.0   # |vy| below this (blocks/s) = sitting on a surface (detector jitter is ~1.3)

    @staticmethod
    def _pred(mode, g, mini, v0, h, hp, resting, dt, speed, p):
        """Possible average vertical speeds over the next step (None = teleport).
        Several options where GD allows them (e.g. it may have just landed)."""
        sc = p.mini if mini else 1.0
        press = h and not hp
        if mode == "wave":
            return [g * (1 if h else -1) * p.wave_slope * (2.0 if mini else 1.0) * speed, 0.0]
        if mode == "ship":
            v = max(-p.ship_vmax * sc, min(p.ship_vmax * sc, v0 + g * (p.ship_up if h else -p.ship_down) * sc * dt))
            return [v, 0.0] if resting else [v]
        G = {"cube": p.cube_g, "robot": p.cube_g, "spider": p.cube_g, "ball": p.ball_g, "ufo": p.ufo_g}[mode]
        jump = g * (p.robot_v0 if mode == "robot" else p.cube_v0) * sc - (0.0 if mode == "robot" else g * G * dt / 2)
        if resting:
            if mode == "spider":
                return None if press else [0.0]
            if mode == "ball":
                return [g * 2.0 + g * G * dt / 2] if press else [0.0]
            if mode == "ufo":
                return [g * p.ufo_v0 * sc - g * G * dt / 2] if press else [0.0]
            return [jump] if h else [0.0]
        if mode == "ufo" and press:
            return [g * p.ufo_v0 * sc - g * G * dt / 2]
        opts = [v0 - g * G * dt, 0.0]                  # keeps falling / just landed
        if mode == "robot" and h and hp and g * v0 > 0:
            opts.append(v0)                            # boosting
        if mode in ("cube", "robot") and h and g * v0 < 0:
            opts.append(jump)                          # landed and re-jumped (held)
        return opts

    def _score(self, mode: str, g: int, mini: bool, vys, holds, dt: float, speed: float, p: Phys, lag: int = 0):
        """Mean prediction error over the window, starting with gravity g.
        Ball and spider flip gravity themselves, so g is tracked through the
        window; returns (error, n, gravity at the end of the window)."""
        err, n = 0.0, 0
        for k in range(3 + lag, len(vys)):    # vys[0] is a placeholder: start where the
            v0, v1 = vys[k - 1], vys[k]       # previous two speeds are real measurements
            h, hp = holds[k - lag], holds[k - 1 - lag]   # what reached the icon (input delay)
            resting = self._rest[k]
            dt_k = dt[k] if isinstance(dt, list) else dt      # real duration of THIS step
            pred = self._pred(mode, g, mini, v0, h, hp, resting, dt_k, speed, p)
            if pred is None:                       # spider teleport expected, towards +g
                e = 0.0 if (abs(v1) > 20 and v1 * g > 0) else 25.0
            else:
                # options after the first cost a little (landing / boost / re-jump), so a
                # model that allows more things doesn't win just by being flexible
                e = min(min(abs(v1 - q) + (0.0 if i == 0 or (resting and q == 0.0) else 0.8)
                            for i, q in enumerate(pred)), 25.0)
                e = min(e, 25.0)
            if resting and h and not hp and mode in ("ball", "spider"):
                g = -g                             # this press flipped gravity
            err += e
            n += 1
        return (err / n if n else None), n, g

    MAX_LAG = 3

    def _fit_lag(self, vys, holds, dt, speed, p) -> None:
        """Which input delay (0..MAX_LAG decisions) explains the motion best?
        Real GD + screen capture react a frame or two after a press; the
        simulator reacts instantly. Switch only after 3 agreeing checks."""
        best = {}
        for L in range(self.MAX_LAG + 1):
            s = [self._score(m, g, mi, vys, holds, dt, speed, p, L)
                 for m in MODES for g in (1, -1) for mi in (False, True)]
            s = [e for e, n, _ in s if e is not None and n >= 3]
            if s:
                best[L] = min(s)
        if not best or self.lag not in best:
            return
        L = min(best, key=best.get)
        if L != self.lag and best[L] < 0.75 * best[self.lag]:
            c, k = self._lag_votes
            k = k + 1 if c == L else 1
            self._lag_votes = (L, k)
            if k >= 3:
                self.lag, self._lag_votes = L, (None, 0)
            self._lag_ok = 0
        else:
            self._lag_votes = (None, 0)
            self._lag_ok += 1

    def update(self, y: float, hold: bool, dt: float, speed: float, p: Phys) -> None:
        if self.hist and abs(y - self.hist[-1][0]) > 3.0 and self.mode != "spider" and self._hint != "spider":
            self.hist.clear()          # respawn / camera jump: not motion, start the window fresh
        self.hist.append((y, bool(hold), float(dt)))
        if len(self.hist) < 6:
            return
        ys = [h[0] for h in self.hist]
        holds = [h[1] for h in self.hist]
        dts = [h[2] for h in self.hist]       # each step's real duration (live: 33 or 50 ms)
        vys = [0.0] + [(ys[k] - ys[k - 1]) / dts[k] for k in range(1, len(ys))]
        if not any(holds[1:]) and sum(abs(v) > 2 * self.REST for v in vys[1:]) < 2:
            return                    # sitting still / sliding, not pressing: nothing to learn
        if self._hint_t > 0:          # (the portal window only counts informative steps)
            self._hint_t -= 1
        if getattr(self, "_ghint_t", 0) > 0:
            self._ghint_t -= 1
        # 'resting before step k' = barely moved over the two steps before it (two-step
        # displacement: much less sensitive to detector jitter than single-step speeds)
        self._rest = [k >= 3 and abs(ys[k - 1] - ys[k - 3]) < self.REST * (dts[k - 1] + dts[k - 2])
                      for k in range(len(ys))]
        self._n_upd += 1
        # input-delay check: often at first, rarely once it has been confirmed a few times
        if not getattr(self, "lag_locked", False) and self._n_upd % (8 if self._lag_ok < 5 else 40) == 0 and any(holds):
            self._fit_lag(vys, holds, dts, speed, p)
        L = self.lag
        if (self.mode in ("ball", "spider") and len(holds) > L + 1 and holds[-1 - L] and not holds[-2 - L]
                and self._rest[-1]):
            self.g = -self.g          # a press on a surface flips ball / spider gravity
        # full re-check (all 28 mode hypotheses) only when something happened -- a press /
        # release reaching the icon, a sudden speed change (landing, orb, pad), a portal,
        # a switch in progress -- or every 3rd step; steady motion can't change the answer
        h_eff = holds[: len(holds) - L] if L else holds
        event = (len(h_eff) >= 4 and len(set(h_eff[-4:])) > 1) or abs(vys[-1] - vys[-2]) > 8.0 \
            or self._hint_t > 0 or self._cand_streak[1] > 0
        if not event and self._n_upd % 3:
            return
        scores = []
        for mode in MODES:
            for g in (1, -1):
                for mini in (False, True):
                    sc, n, g_end = self._score(mode, g, mini, vys, holds, dts, speed, p, self.lag)
                    if sc is None or n < 3:
                        continue
                    if self._hint == mode and self._hint_t > 0:
                        sc *= 0.8
                    if (mode, g_end, mini) == (self.mode, self.g, self.mini):
                        sc *= 0.85    # hysteresis: stay unless clearly worse
                    scores.append((sc, mode, g_end, mini))
        if not scores:
            return
        scores.sort()
        best = scores[0]
        second = scores[1][0] if len(scores) > 1 else best[0] * 2
        cand = best[1:]
        cur = next((sc for sc, *c in scores if tuple(c) == (self.mode, self.g, self.mini)), None)
        if cand == (self.mode, self.g, self.mini):
            self._cand_streak = (None, 0)
        else:
            # GD only changes mode at a portal (gravity: portals, orbs, ball/spider
            # presses -- the models track those). Right after a portal, switch fast;
            # otherwise only when the current model keeps failing badly.
            # fast switch only right after a portal, or (gravity alone) at an orb / pad;
            # ball / spider press flips are modelled above. Anything else needs overwhelming,
            # long-lasting evidence: a normal cube jump once looked like "falling upward"
            gravity_only = cand[0::2] == (self.mode, self.mini)
            portal = self._hint_t > 0 or (gravity_only and getattr(self, "_ghint_t", 0) > 0)
            # without a portal, demand overwhelming, long-lasting evidence: real live runs
            # showed a cube whose presses weren't reaching the game turn into an
            # "upside-down ship" within a second -- and then every prediction was wrong
            # Live deaths still showed the cube taken for a "spider" / "wave" / "ship" for a
            # second or more without any portal (3 of 30 deaths) -- every prediction wrong.
            # GD changes MODE only at portals: without one it now takes ~1.6 s of a far
            # better fit (gravity alone: ~0.8 s), so a wobbly box can't do it.
            clearly = cur is None or best[0] < (0.85 if portal else (0.4 if gravity_only else 0.3)) * cur
            c, k = self._cand_streak
            k = k + 1 if (c == cand and clearly) else (1 if clearly else 0)
            self._cand_streak = (cand, k)
            if k >= (2 if portal else (20 if gravity_only else 40)):
                self.mode, self.g, self.mini = cand
                self._cand_streak = (None, 0)
        self.conf = float(np.clip(1.0 - best[0] / max(second, 1e-6), 0.0, 1.0))
