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

import math
import numpy as np

MODES = ["cube", "ship", "ball", "ufo", "wave", "robot", "spider"]
MODE_ID = {m: i for i, m in enumerate(MODES)}
# the swing copter (GD 2.2) has its own physics below but shares the ship's slot in the
# brain's inputs (both fly; a new slot would change the input size = need a new brain)
MODE_ID["swing"] = MODE_ID["ship"]


@dataclass
class Phys:
    # cube: measured from a real Stereo Madness recording (9 clean jumps, 1080p60):
    # apex 2.21 blocks, 0.43 s in the air. Still fine-tuned live from your own jumps.
    cube_v0: float = 20.5       # jump speed (blocks/s)
    cube_g: float = 95.0        # gravity (blocks/s^2)
    ship_a: float = 55.0        # (old single value; ship_up / ship_down are used)
    # (starting values measured on a recording of your game -- custom-for-claude.mp4; the
    #  ModeCalibrator keeps measuring them live)
    ship_up: float = 58.0       # ship acceleration while holding / falling while released
    ship_down: float = 45.0     # (was 55 / 55: the fall was ~20% too strong)
    ship_vmax: float = 12.0
    ball_g: float = 80.0
    ufo_v0: float = 12.5        # (was 14)
    ufo_g: float = 57.0         # (was 80: ~30% too strong)
    wave_slope: float = 0.86    # (was 1.0 = 45 degrees; measured ~40; mini wave: 2x)
    robot_v0: float = 16.0
    robot_boost: float = 0.25   # seconds of extra rise while held
    spider_snap: float = 12.0   # how far it searches for the opposite surface
    # GD only kills on a block when the icon's small INNER hitbox touches it; touching it
    # with the rest of the icon puts the icon on top. So a cube coming in a little BELOW a
    # block's top still lands on it (live: a jump onto a pillar landed with its bottom 0.1
    # below the top, the preview called it a crash and the shield gave up). How far below
    # the top (blocks, measured at the icon's bottom) that still lands:
    snap: float = 0.3
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


@dataclass(slots=True)
class Body:
    y: float = 0.0
    vy: float = 0.0
    g: int = 1                  # +1 normal gravity, -1 upside down
    grounded: bool = True
    mode: str = "cube"
    mini: bool = False
    boost: float = 0.0          # robot: remaining boost time
    prev_hold: bool = False
    used: int = 0               # search: pads / orbs this path already used (bitmask)
    dash: bool = False          # dashing (dash orb, while SPACE stays held)
    crashed: bool = False       # hit a block head-first (cube / robot): dead


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
    vy_before = b.vy
    dv = 0.0   # velocity lost to gravity / thrust this step (for an exact position update)
    if b.dash:
        if hold:                 # dash orb: straight ahead while held, no gravity
            b.vy, b.grounded, b.prev_hold = 0.0, False, hold
            return b
        b.dash = False
    if m == "swing":
        # swing copter: every press flips which way it accelerates (like gravity), speed
        # capped like the ship's
        if press:
            b.g = -b.g
        dv = b.g * p.ship_down * s * dt
        b.vy -= dv
        b.vy = max(-p.ship_vmax * s, min(p.ship_vmax * s, b.vy))
        b.grounded = False
    elif m == "cube":
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
    if m in ('ship','swing') and dt > 0:
        # Integrate only until reaching the speed cap, then travel at the
        # cap. Applying half the full acceleration after clamping velocity
        # underestimated every sustained climb/fall (even already at vmax).
        acceleration = -dv/dt
        if abs(acceleration) > 1e-12:
            cap = math.copysign(p.ship_vmax*s,acceleration)
            start_v = max(-p.ship_vmax*s,min(p.ship_vmax*s,vy_before))
            tc = max(0.,min(dt,(cap-start_v)/acceleration))
            y_new = b.y + start_v*tc + .5*acceleration*tc*tc + cap*(dt-tc)
    # resting surfaces: falling onto a floor / rising into a ceiling
    up = p.snap if b.g == 1 else 0.06           # landing side: GD's inner-hitbox leniency
    dn = p.snap if b.g == -1 else 0.06
    # A cube / robot can only stand on the side gravity pulls it to. Hitting a block with
    # its HEAD (jumping into the underside; upside down: falling onto a block's top) kills
    # it in GD -- this used to be a harmless bump, so the blue line drew jumps straight up
    # into low ceilings and called them safe.
    head_kills = m in ("cube", "robot")
    # (only a real hit: moving INTO it, and the block's edge was clear of the head before
    #  this step -- sliding along under a block in a 1-block gap is fine in GD; the
    #  first version flagged that and the whole-level test went from 18 to 28 false crashes)
    # Takeoff is included: a ground jump into the underside is fatal too.
    # Merely sliding underneath has no headward velocity and stays valid.
    if b.vy < -0.5 and head_kills and b.g == -1:
        if any(y_new <= f <= b.y + 1e-8 for f in floors):
            b.crashed = True
    if b.vy > 0.5 and head_kills and b.g == 1:
        if any(b.y - 1e-8 <= c <= y_new for c in ceils):
            b.crashed = True
    if b.vy <= 0:
        land = [f for f in floors if y_new <= f <= max(b.y, y_new) + up]
        if land:
            y_new = max(land)
            if b.g == 1 or m in ("ship", "wave", "ufo", "swing"):
                b.grounded = b.g == 1 and m not in ("ship", "wave", "swing")
            b.vy = 0.0 if m != "wave" else b.vy
    if b.vy >= 0:
        hit_c = [c for c in ceils if min(b.y, y_new) - dn <= c <= y_new]
        if hit_c:
            y_new = min(hit_c)
            if b.g == -1 or m in ("ship", "wave", "ufo", "swing"):
                b.grounded = b.g == -1 and m not in ("ship", "wave", "swing")
            b.vy = 0.0 if m != "wave" else b.vy
    if b.grounded and hold and m in ("cube", "robot") and not was_grounded:
        # holding SPACE as it lands: GD jumps again straight away (no frame on the ground)
        b.vy, b.grounded = b.g * (p.robot_v0 if m == "robot" else p.cube_v0) * s, False
        if m == "robot":
            b.boost = p.robot_boost
    if b.grounded and m not in ("ship", "wave", "swing"):
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


def block_side_contacts(x_prev, x, solid, half):
    """Times within this step when the inner hitbox enters a block's side."""
    if x <= x_prev:
        return []
    edge = half - .05
    return [((s0 - edge - x_prev) / (x - x_prev), sb, st)
            for s0, s1, sb, st in solid
            if x_prev + edge <= s0 < x + edge]


def block_side_hit(body, hold, dt, speed, p, contacts, half):
    """Check first horizontal contact BEFORE a landing snap can erase a crash.

    The normal landing tolerance still applies to the inner hitbox. A cube
    that is already too low at the leading edge cannot be lifted onto the
    platform merely because its top is under the icon at the step's end.
    Returns (fraction, centre_y) for the earliest hit, or None.
    """
    if body.grounded or body.mode not in ('cube', 'robot'):
        return None
    first = None
    for fraction, bottom, top in contacts:
        if first is not None and fraction >= first[0]:
            continue
        contact = Body(body.y, body.vy, body.g, False, body.mode, body.mini,
                       body.boost, body.prev_hold, body.used, body.dash, body.crashed)
        advance(contact, hold, dt * fraction, speed, p, ([], []))
        upper_margin, lower_margin = (.12, p.snap) if body.g == 1 else (p.snap, .12)
        if bottom < contact.y + half - upper_margin - 1e-9 and top > contact.y - half + lower_margin + 1e-9:
            first = (fraction, contact.y)
    return first


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
    times = _fine_times(len(pending) * dec_dt, dec_dt, len(pending), dt, horizon)
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
        side_contacts = block_side_contacts(x_prev, x, solid, half)
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
            side_hit = block_side_hit(b, holds[k], dt_i, speed, p, side_contacts, half)
            if side_hit is not None:
                fraction, contact_y = side_hit
                hit_t[pi] = t_prev + fraction * dt_i
                pts[pi].append((x_prev + fraction * (x - x_prev), contact_y))
                continue
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
                    tu, td = (0.12, p.snap) if b.g == 1 else (p.snap, 0.12)   # landing side: lenient
                    if s0 < hi - 0.05 and s1 > lo + 0.05 and sb < y + half - tu and st > y - half + td:
                        hit_t[pi] = t                  # into the side of a block
                        break
            if hit_t[pi] is None and (y < -12 or y > 14 or b.crashed):
                hit_t[pi] = t                          # out of the world / hit a block head-first
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


def _fine_times(D: float, dec_dt: float, n_pend: int, dt: float, horizon: float) -> list:
    """The search's time points: every moment an action reaches the icon (D + k * dec_dt,
    the measured delay is NOT a whole number of 1/30 s steps) is a step boundary, the gaps
    split into steps of at most ~1/30 s, for the first 0.5 s; then 1/15 s steps."""
    fine_end = min(horizon, 0.5)
    b = [D + k * dec_dt for k in range(-n_pend, int((fine_end - D) / dec_dt) + 2)]
    pts = [0.0]
    for q in sorted(q for q in b if 0.003 < q < fine_end - 0.003):
        if q - pts[-1] > 0.003:             # (a boundary a hair from the last: no 0-length step)
            pts.append(q)
    pts.append(fine_end)
    times: list = []
    for a_, b_ in zip(pts[:-1], pts[1:]):
        m = max(1, int(math.ceil((b_ - a_) / dt - 0.25)))
        times.extend(a_ + (b_ - a_) * i / m for i in range(1, m + 1))
    while times[-1] + 1e-9 < horizon:
        times.append(min(horizon, times[-1] + 2 * dt))
    return times


def search_best(body: Body, haz, solid, floor, ceil, speed: float, p: Phys,
                dec_dt: float, lag: int = 0, pending=(), horizon: float = 1.5,
                half: float = 0.5, beam: int = 10, want_path: bool = False,
                late_every: float = 0.0, portals=None, specials=None, prefer=None,
                delay: float | None = None, root_action: int | None = None,
                flight_target: float = 0., flight_view=None, ground_clearance=False,
                action_schedule=None,work_cache=None) -> dict:
    """Longest survival (fraction of `horizon`, 1.0 = survives it all) for each first
    action ("1" = press now, "00" = release now and at the next decision, "01" =
    release now, press at the next decision), with every later decision chosen well.
    With want_path, also the path (x, y) of the best line found and whether it crashes.

    Each state carries the SET of first actions that can reach it (bit 0 = "1",
    bit 1 = "00", bit 2 = "01"), so lines that end up in the same state (e.g. back on
    the ground) are simulated once.

    delay: the input delay in SECONDS (measured in ms, not rounded to whole steps): the
    action chosen now reaches the icon `delay` s from now, the already-sent ones in
    `pending` (oldest first, one per decision) before that. None = `lag` whole decisions."""
    dt = 1.0 / 30.0
    if delay is None:
        pending = list(pending)[-lag:] if lag > 0 else []
        D = len(pending) * dec_dt
    else:
        D = max(0.0, float(delay))
        need = int(math.ceil(D / dec_dt - 1e-6))
        pending = list(pending)[-need:] if need > 0 else []
        pending = [0.0] * (need - len(pending)) + pending
    n_pend = len(pending)
    # Step-based delay still means n_pend decisions, not a rounded 30 Hz
    # interval. Split on action boundaries in BOTH delay modes. Otherwise a
    # 41 ms decision can actuate at 100 ms instead of 82 ms, inventing a
    # different jump and changing the shield's safe first action.
    time_key=('times',D,dec_dt,n_pend,dt,horizon)
    times=work_cache.get(time_key) if work_cache is not None else None
    if times is None:
        times=_fine_times(D,dec_dt,n_pend,dt,horizon)
        if work_cache is not None:work_cache[time_key]=times
    root_node = (0.0, body.y, None, 0.0, 0., flight_target)
    trajectory_meta = {id(root_node):(0.,body.vy,None,-1,0.)} if want_path else None
    def node_key(node):
        return (node[4],node[3]) if flight_target > 0 else (node[3],)
    if flight_target > 0:
        from flight_control import flight_segment_hit, path_clearance
        margin_floor,margin_ceil = floor,ceil
        if flight_view is not None:
            margin_floor = max(floor,flight_view[0]) if floor is not None else flight_view[0]
            margin_ceil = min(ceil,flight_view[1]) if ceil is not None else flight_view[1]
    pref_x = pref_y = None
    if want_path and prefer and len(prefer) >= 3:
        pp = sorted(prefer)
        pref_x = np.array([q[0] for q in pp], float)
        pref_y = np.array([q[1] for q in pp], float)
    ALL, B1, B00, B01 = 7, 1, 2, 4
    fixed_last=action_schedule[max(action_schedule)] if action_schedule else False
    # a state: [Body, root mask, hold this step, path node]
    states = [(Body(body.y, body.vy, body.g, body.grounded, body.mode, body.mini, body.boost, body.prev_hold),
               ALL if root_action is None else (1 << root_action), False, root_node)]
    best = [0.0, 0.0, 0.0]
    best_node = [root_node, root_node, root_node]
    best_unknown = [False, False, False]

    def credit(mask: int, t_frac: float, node, unknown=False) -> None:
        for i in range(3):
            if mask >> i & 1 and (t_frac > best[i] or ((want_path or flight_target > 0) and t_frac == best[i]
                                                         and node_key(node) < node_key(best_node[i]))
                                  or (unknown and t_frac==0. and best_node[i] is root_node)):
                best[i] = t_frac
                best_node[i] = node
                best_unknown[i] = unknown

    every = max(1, int(round(late_every / dec_dt))) if late_every > 0 else 1
    t_prev, x_prev, last_jd = 0.0, 0.0, -1
    spd_now = speed
    for t in times:
        if not states:
            break
        x = x_prev + spd_now * (t - t_prev)        # (speed portals change spd_now)
        # which decision's action reaches the icon during this step: the free decision k
        # (0 = the one chosen now) once its delay has passed, an already-sent one before
        if delay is None:
            jd = int(t_prev / dec_dt + 1e-9)
            k_ = jd - n_pend
        else:
            k_ = int(math.floor((t_prev - D) / dec_dt + 1e-6))
            jd = k_ + n_pend
        if k_ < 0:
            hp = bool(pending[max(0, n_pend + k_)]) if n_pend else bool(body.prev_hold)
            states = [(b, m, hp, nd) for b, m, _h, nd in states]
        elif jd != last_jd and (t_prev < 0.5 or jd % every == 0):
            k = k_                                  # which free decision this is (0 = now)
            new = []
            for b, m, _h, nd in states:
                choices=(bool(action_schedule.get(k,fixed_last)),) if action_schedule else (True,False)
                for h in choices:
                    if k == 0:
                        m2 = m & (B1 if h else (B00 | B01))
                    elif k == 1:
                        m2 = m & ((B1 | B01) if h else (B1 | B00))
                    else:
                        m2 = m
                    if m2:
                        # A fixed input sequence has one successor. No sibling
                        # can mutate this body, so cloning at every decision is
                        # unnecessary. Branching searches still own each child.
                        child=b if action_schedule else Body(b.y,b.vy,b.g,b.grounded,b.mode,b.mini,b.boost,b.prev_hold,b.used,b.dash,b.crashed)
                        new.append((child,m2,h,nd))
            states = new
        last_jd = jd
        dt_i = t - t_prev
        lo, hi = x_prev - half, x + half                 # swept horizontally (no tunnelling)
        geometry_key=('geometry',x_prev,x,half)
        geometry=work_cache.get(geometry_key) if work_cache is not None else None
        if geometry is None:
            floors,ceils=surfaces_at(x,solid,floor,ceil,half)
            side_contacts=block_side_contacts(x_prev,x,solid,half)
            hz_boxes=[h_ for h_ in haz if h_[0]<hi and h_[1]>lo]
            so_boxes=[s_ for s_ in solid if s_[0]<hi-.05 and s_[1]>lo+.05]
            hz=[(h_[2],h_[3]) for h_ in hz_boxes];so=[(s_[2],s_[3]) for s_ in so_boxes]
            pt_here=[q for q in (portals or ()) if q[0]<hi and q[1]>lo]
            sp_here=[q for q in (specials or ()) if q[0]<hi and q[1]>lo]
            geometry=(floors,ceils,side_contacts,hz_boxes,so_boxes,hz,so,pt_here,sp_here)
            if work_cache is not None:work_cache[geometry_key]=geometry
        floors,ceils,side_contacts,hz_boxes,so_boxes,hz,so,pt_here,sp_here=geometry
        # portals: a path that flies through one continues in that mode / gravity
        # (whole-level test: every jump into the ship portal "crashed" as a cube)
        if pt_here:
            for _x0, _x1, pb, ptop, kind, val in pt_here:
                if kind == "speed":                  # speed portals: the whole level speeds up
                    spd_now = float(val)
            for b, m, h, nd in states:
                for _x0, _x1, pb, ptop, kind, val in pt_here:
                    if pb < b.y + half and ptop > b.y - half:
                        if kind == "mode" and b.mode != val:
                            b.mode, b.boost = val, 0.0
                            if val in ("ship", "wave", "ufo", "swing"):
                                b.grounded = False
                            b.vy *= 0.5
                        elif kind == "g" and val == 0 and not (b.used >> 30 & 1):
                            b.g, b.grounded = -b.g, False           # toggle: once per portal
                            b.used |= 1 << 30
                        elif kind == "g" and val != 0 and b.g != val:
                            b.g, b.grounded = val, False
                        elif kind == "size":
                            b.mini = bool(val)
        alive: list = []
        index: dict = {}
        if flight_target > 0:
            near_key=('near',t_prev,t,x,spd_now,half)
            near=work_cache.get(near_key) if work_cache is not None else None
            if near is None:
                weight=math.exp(-t_prev/.5)-math.exp(-t/.5)
                reach=max(1.,spd_now*.3)
                near_haz=[q for q in haz if q[1]>=x-half and q[0]<=x+half+reach]
                near_solid=[q for q in solid if q[1]>=x-half and q[0]<=x+half+reach]
                near=(weight,near_haz,near_solid)
                if work_cache is not None:work_cache[near_key]=near
            weight,near_haz,near_solid=near
        for b, m, h, nd in states:
            pressed = h and not b.prev_hold
            # An orb/pad in this interval may change motion before contact;
            # retain its existing collision handling rather than assuming a
            # free ballistic arc through that event.
            side_hit = block_side_hit(b, h, dt_i, spd_now, p, side_contacts, half) if not sp_here else None
            if side_hit is not None:
                fraction, contact_y = side_hit
                hit_time = t_prev + fraction * dt_i
                side_node = (x_prev + fraction * (x - x_prev), contact_y, nd,
                             nd[3] + (1.0 if pressed else 0.0) + (.05 if h else 0.0),
                             nd[4],nd[5]) if flight_target > 0 else (
                             (x_prev + fraction * (x-x_prev),contact_y,nd,nd[3]) if want_path else nd)
                if want_path:
                    contact_body=Body(b.y,b.vy,b.g,b.grounded,b.mode,b.mini,b.boost,b.prev_hold,b.used,b.dash,b.crashed)
                    advance(contact_body,h,fraction*dt_i,spd_now,p,(floors,ceils))
                    trajectory_meta[id(side_node)]=(hit_time,contact_body.vy,bool(h),k_,t_prev)
                credit(m, hit_time / horizon, side_node)
                continue
            before_y, before_mode = b.y, b.mode
            advance(b, h, dt_i, spd_now, p, (floors, ceils))
            if sp_here:
                special_effect(b, sp_here, pressed, half, p, (floors, ceils))
            y = b.y
            # the drawn line = the surviving path with the LEAST effort (presses, holding):
            # with nothing dangerous ahead every path survives, and an arbitrary one was
            # drawn -- full of pointless jumps
            if want_path or flight_target > 0:
                eff = nd[3] + (1.0 if pressed else 0.0) + (0.05 if h else 0.0)
                if pref_x is not None and x <= pref_x[-1]:
                    # stay on LAST step's line unless it stopped working: among equally safe
                    # routes the least-effort one flipped from step to step ("the line is on
                    # the right track, then blips to another one")
                    eff += PREFER_W * min(1.0, abs(y - float(np.interp(x, pref_x, pref_y))))
                if flight_target > 0:
                    cache_key=('clearance',spd_now,x,y,b.mode,b.g,b.mini,half,flight_target)
                    reusable=ground_clearance and b.mode not in ('ship','wave','swing') and work_cache is not None
                    clearance=work_cache.get(cache_key) if reusable else None
                    if clearance is None:
                        clearance=path_clearance(x,y,b.vy,b.mode,b.g,b.mini,near_haz,near_solid,margin_floor,margin_ceil,
                                                 spd_now,p,half,flight_target,dec_dt,ground_route=ground_clearance)
                        if reusable:work_cache[cache_key]=clearance
                    score,gap=clearance
                    # Near-term recovery matters more than distant empty space;
                    # otherwise one late rescue diluted across 1.5 s looks fine.
                    node = (x,y,nd if want_path else None,eff,nd[4]+(1.-score)*weight,min(nd[5],gap))
                else:
                    node = (x, y, nd, eff)
            else:
                node = nd
            if want_path:
                trajectory_meta[id(node)] = (t,b.vy,bool(h),k_,t_prev)
            # A route above/below the captured area has unobserved obstacles
            # and boundaries. Do not call it collision-safe merely because
            # the detector cannot supply geometry outside the image. End its
            # VERIFIED horizon here, without reporting a physical collision.
            # This matters near hanging blocks: "fly over it, off screen"
            # used to defeat every otherwise safer route underneath it.
            if ((flight_target > 0 or action_schedule is not None) and flight_view is not None
                    and (y-half < flight_view[0] or y+half > flight_view[1])):
                edge = flight_view[0]+half if y-half < flight_view[0] else flight_view[1]-half
                fraction = max(0.,min(1.,(edge-before_y)/(y-before_y))) if y != before_y else 0.
                edge_node = (x_prev+(x-x_prev)*fraction,edge,node[2],node[3],node[4],node[5]) if flight_target>0 else node
                if want_path:
                    trajectory_meta[id(edge_node)] = (t_prev+dt_i*fraction,b.vy,bool(h),k_,t_prev)
                credit(m,(t_prev+dt_i*fraction)/horizon,edge_node,unknown=True)
                continue
            dead = y < -12 or y > 14 or b.crashed
            if not dead and flight_target > 0 and before_mode in ('ship','wave','swing') and not sp_here:
                # Wave motion is linear. Ship acceleration bows away from the
                # chord by at most a*dt^2/8; include that small uncertainty.
                pad = max(p.ship_up,p.ship_down)*dt_i*dt_i/8 if before_mode in ('ship','swing') else 0.
                dead = flight_segment_hit(x_prev,before_y,x,y,hz_boxes,so_boxes,half,b.g,pad,p.snap)
            if not dead:
                yt, yb_ = y + half, y - half
                for hb, ht in hz:
                    if hb < yt and ht > yb_:
                        dead = True
                        break
                if not dead:
                    tu, td = (0.12, p.snap) if b.g == 1 else (p.snap, 0.12)   # landing side: lenient
                    for sb, st in so:
                        if sb < yt - tu and st > yb_ + td:
                            dead = True
                            break
            if dead:
                credit(m, t / horizon, node)
                continue
            if action_schedule:
                alive.append((b,m,h,node))
                continue
            key = (h, b.prev_hold, round(y * 20.0), round(b.vy * 1.6667), b.grounded, b.g, round(b.boost, 2), b.mode, b.used)
            j = index.get(key)
            if j is not None:                       # same state reached by other first actions
                s0 = alive[j]
                keep_node = node if ((want_path or flight_target > 0) and node_key(node) < node_key(s0[3])) else s0[3]
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
                    pick = [lst[int(round(q * n_ / (beam - 1)))] for q in range(beam)]
                    if want_path or flight_target > 0:
                        # among states at the same height and speed keep the one with the least
                        # effort (else a "pad launched it, no presses" line could be dropped for
                        # a copy that holds / presses, and the drawn line showed a needless jump)
                        grp: dict = {}
                        for j in lst:
                            k_ = (round(alive[j][0].y * 20.0), round(alive[j][0].vy * 1.6667))
                            if k_ not in grp or node_key(alive[j][3]) < node_key(alive[grp[k_]][3]):
                                grp[k_] = j
                        pick = [grp[(round(alive[j][0].y * 20.0), round(alive[j][0].vy * 1.6667))] for j in pick]
                    lst = pick
                keep.update(lst)
            alive = [alive[j] for j in sorted(keep)]
        states = alive
        t_prev, x_prev = t, x
    for b, m, h, nd in states:
        credit(m, 1.0, nd)
    out = {k: float(best[i]) for i, k in enumerate(SEARCH_KEYS)}
    if flight_target > 0:
        out['clearance'] = {k:max(0.,1.-best_node[i][4]/(1.-math.exp(-horizon/.5))) for i,k in enumerate(SEARCH_KEYS)}
        out['minimum_clearance'] = {k:best_node[i][5] for i,k in enumerate(SEARCH_KEYS)}
        out['visibility_limited'] = {k:best_unknown[i] for i,k in enumerate(SEARCH_KEYS)}
    if want_path:
        # ties -> prefer waiting: the line then shows the LATEST safe moment to press
        i_best = root_action if root_action is not None else max((1, 2, 0), key=lambda i: best[i])
        node = best_node[i_best]
        pts = []
        trajectory = []
        while node is not None:
            pts.append((node[0], node[1]))
            tm,velocity,held,decision,start = trajectory_meta.get(id(node),(0.,0.,None,-1,0.))
            trajectory.append(dict(t=float(tm),x=float(node[0]),y=float(node[1]),vy=float(velocity),
                                   hold=held,decision=int(decision),start=float(start)))
            node = node[2]
        out["path"] = pts[::-1]
        out["path_unknown"] = best_unknown[i_best]
        out["path_hit"] = best[i_best] < 1.0 and not best_unknown[i_best]
        controls = {}
        for point in reversed(trajectory):
            if point['decision'] >= 0:
                controls.setdefault(point['decision'],dict(decision=point['decision'],
                    applies_at=point['start'],hold=bool(point['hold'])))
        out['plan'] = dict(root=SEARCH_KEYS[i_best],first_action=int(i_best==0),
                           trajectory=trajectory[::-1],controls=list(controls.values()),
                           verified=best[i_best]>=1.-1e-6 and not best_unknown[i_best])
    return out


PREFER_W = 0.15   # path effort per step for being 1 block off last step's line (a press = 1.0)


def flight_search_best(body,haz,solid,floor,ceil,speed,p,dec_dt,lag,pending,
                       horizon,half,want_path,portals,specials,prefer,delay,target,view,work_cache=None):
    # Keep histories separate per first action. Merging first-action masks
    # would otherwise assign one root the other root's clearance history.
    results = [search_best(body,haz,solid,floor,ceil,speed,p,dec_dt,lag,pending,horizon,half,
                           beam=6,want_path=want_path,portals=portals,specials=specials,prefer=prefer,
                           delay=delay,root_action=i,flight_target=target,flight_view=view,work_cache=work_cache) for i in range(3)]
    result = {k:results[i][k] for i,k in enumerate(SEARCH_KEYS)}
    result['clearance'] = {k:results[i]['clearance'][k] for i,k in enumerate(SEARCH_KEYS)}
    result['minimum_clearance'] = {k:results[i]['minimum_clearance'][k] for i,k in enumerate(SEARCH_KEYS)}
    result['visibility_limited'] = {k:results[i]['visibility_limited'][k] for i,k in enumerate(SEARCH_KEYS)}
    winner = max(SEARCH_KEYS,key=lambda k:(result[k],result['clearance'][k]))
    if want_path:
        selected = results[SEARCH_KEYS.index(winner)]
        result.update(path=selected['path'],path_hit=selected['path_hit'],path_unknown=selected['path_unknown'],plan=selected['plan'])
        result['_root_paths'] = {k:(results[i]['path'],results[i]['path_hit'],results[i]['path_unknown']) for i,k in enumerate(SEARCH_KEYS)}
        result['_root_plans'] = {k:results[i]['plan'] for i,k in enumerate(SEARCH_KEYS)}
    return result


def robust_search(body: Body, haz, solid, floor, ceil, speed: float, p: Phys, dec_dt: float,
                  lag: int, pending, pending_long=None, horizon: float = 1.5, half: float = 0.5,
                  want_path: bool = False, lag_alt: int | None = None, portals=None, specials=None,
                  prefer=None, delay: float | None = None, delay_alt: float | None = None,
                  flight_buffer: float = 0., flight_view=None, explicit_plan=False) -> dict:
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
    flight = body.mode in ('ship','wave') and flight_buffer > 0
    # Valid only for this observed geometry; never carry it across frames.
    work_cache={}
    if not explicit_plan and not haz and not solid and (not flight or (floor is None and ceil is None and flight_view is None)):
        a = {k: 1.0 for k in SEARCH_KEYS}
        a.update(nominal=dict(a), worse=dict(a), path=[], path_hit=False)
        if flight:
            a['flight'] = dict(mode=body.mode,survival={k:1. for k in SEARCH_KEYS},
                               clearance={k:1. for k in SEARCH_KEYS},buffer=flight_buffer)
        return a
    def run(world_speed,world_lag,queue,world_delay,draw=False):
        if flight:
            return flight_search_best(body,haz,solid,floor,ceil,world_speed,p,dec_dt,world_lag,queue,
                                      horizon,half,draw,portals,specials,prefer,world_delay,flight_buffer,flight_view,work_cache)
        if explicit_plan:
            roots=[search_best(body,haz,solid,floor,ceil,world_speed,p,dec_dt,world_lag,queue,horizon,half,
                    beam=8,want_path=draw,portals=portals,specials=specials,prefer=prefer,
                    delay=world_delay,root_action=i,flight_target=.5,flight_view=flight_view,
                    ground_clearance=True,work_cache=work_cache) for i in range(3)]
            result={k:roots[i][k] for i,k in enumerate(SEARCH_KEYS)}
            result['clearance']={k:roots[i]['clearance'][k] for i,k in enumerate(SEARCH_KEYS)}
            if draw:
                result['_root_results']={k:roots[i] for i,k in enumerate(SEARCH_KEYS)}
                result['_root_plans']={k:roots[i]['plan'] for i,k in enumerate(SEARCH_KEYS)}
                result['_root_paths']={k:(roots[i]['path'],roots[i]['path_hit'],roots[i]['path_unknown']) for i,k in enumerate(SEARCH_KEYS)}
            return result
        return search_best(body,haz,solid,floor,ceil,world_speed,p,dec_dt,world_lag,queue,horizon,half,
                           want_path=draw,portals=portals,specials=specials,prefer=prefer,delay=world_delay)
    a = run(speed,lag,pending,delay,want_path)
    alt = lag + 1 if lag_alt is None else max(0,int(lag_alt))
    da = None if delay is None else (delay+dec_dt if delay_alt is None else max(0.,float(delay_alt)))
    pl = (list(pending_long) if pending_long is not None else
          ([False]*max(0,alt-len(pending))+list(pending) if delay is None else list(pending)))
    worst_lag=alt if delay is None else lag
    worst_queue=pl[-alt:] if delay is None and alt else ([] if delay is None else pl)
    draw_ground=bool(explicit_plan and want_path and not flight)
    if draw_ground and delay is None:
        # An earlier alternative must not remove the LATER delay bound.
        worst_lag=max(lag+1,alt)
        worst_queue=([body.prev_hold]*max(0,worst_lag-len(pl))+pl)[-worst_lag:]
    if all(a[k] < 1.0 for k in SEARCH_KEYS):
        b = {k: 0.0 for k in SEARCH_KEYS}   # crashes as measured -> the worse world can't help
    elif delay is not None:
        # delay measured in ms: the worse world's press arrives `delay_alt` s after (the far
        # end of the measured spread, at least a step off)
        b = run(speed*1.04,worst_lag,worst_queue,da,draw_ground)
    else:
        b = run(speed*1.04,worst_lag,worst_queue,None,draw_ground)
    # Two searches can survive using DIFFERENT future commands. Their maxima
    # do not certify the one sequence drawn on screen. Replay the actual
    # commands in both worlds, also trying the slower world's route when the
    # nominal route misses a landing/jump deadline there.
    sequence_checks={}
    if draw_ground and '_root_plans' in a:
        for i,k in enumerate(SEARCH_KEYS):
            candidates=[a['_root_plans'][k]]
            if '_root_plans' in b:candidates.append(b['_root_plans'][k])
            best_score=-1.;selected=None
            seen_schedules=set()
            for plan in candidates:
                schedule={c['decision']:bool(c['hold']) for c in plan['controls']}
                if not schedule:continue
                identity=tuple(sorted(schedule.items()))
                if identity in seen_schedules:continue
                seen_schedules.add(identity)
                def replay(world_speed,world_lag,queue,world_delay,draw):
                    return search_best(body,haz,solid,floor,ceil,world_speed,p,dec_dt,world_lag,queue,horizon,half,
                        beam=1,want_path=draw,portals=portals,specials=specials,delay=world_delay,root_action=i,
                        flight_target=.5,flight_view=flight_view,ground_clearance=True,action_schedule=schedule,work_cache=work_cache)
                # The search already simulated its own selected sequence.
                # Reuse that result; only cross-world sequences need replay.
                nominal=(dict(a['_root_results'][k]) if plan is a['_root_plans'][k]
                         else replay(speed,lag,pending,delay,True))
                worse=(dict(b['_root_results'][k]) if plan is b.get('_root_plans',{}).get(k)
                       else replay(speed*1.04,worst_lag,worst_queue,da,False))
                # Brief holds around a landing are not monotone in delay:
                # both endpoints can survive while a middle timing fails.
                upper=worst_lag*dec_dt if delay is None else max(delay+dec_dt,da)
                lower=min(lag,alt)*dec_dt if delay is None else max(0.,min(delay-dec_dt,da))
                checked=[]
                if nominal[k]>=1.-1e-6 and worse[k]>=1.-1e-6:
                    for phase in range(9):
                        seconds=lower+(upper-lower)*phase/8.
                        need=int(math.ceil(seconds/dec_dt-1e-6))
                        history=([body.prev_hold]*max(0,need-len(pl))+pl)[-need:] if need else []
                        result=search_best(body,haz,solid,floor,ceil,speed,p,dec_dt,lag,history,horizon,half,
                            beam=1,want_path=False,portals=portals,specials=specials,delay=seconds,root_action=i,
                            flight_view=flight_view,action_schedule=schedule,work_cache=work_cache)
                        checked.append(dict(delay=seconds,survival=result[k]))
                        worse[k]=min(worse[k],result[k])
                        if result[k]<1.-1e-6:break
                score=safety_score(nominal[k],worse[k])
                quality=min(nominal['clearance'][k],worse['clearance'][k])
                if selected is None or (score,quality)>(best_score,selected[2]):
                    best_score=score;selected=(nominal,worse,quality,checked)
            if selected is not None:
                nominal,worse,quality,checked=selected
                a[k]=nominal[k];b[k]=worse[k]
                a['clearance'][k]=quality
                b.setdefault('clearance',{})[k]=quality
                a['_root_plans'][k]=nominal['plan']
                a['_root_paths'][k]=(nominal['path'],nominal['path_hit'],nominal['path_unknown'])
                sequence_checks[k]=dict(nominal=nominal[k],worse=worse[k])
                a['_root_plans'][k]['delay_checks']=checked
        a['sequence_checks']=sequence_checks
    a.pop('_root_results',None)
    a["nominal"] = {k: a[k] for k in SEARCH_KEYS}
    a["worse"] = {k: b[k] for k in SEARCH_KEYS}
    for k in SEARCH_KEYS:
        a[k] = safety_score(a["nominal"][k], b[k])
    if flight:
        from flight_control import recovery_hint
        quality = {k:min(a['clearance'][k],b.get('clearance',a['clearance'])[k]) for k in SEARCH_KEYS}
        a['flight'] = dict(mode=body.mode,survival={k:a[k] for k in SEARCH_KEYS},clearance=quality,
                           minimum_clearance={k:min(a['minimum_clearance'][k],b.get('minimum_clearance',a['minimum_clearance'])[k]) for k in SEARCH_KEYS},
                           buffer=flight_buffer,
                           nominal={k:a['nominal'][k] for k in SEARCH_KEYS},worse={k:b[k] for k in SEARCH_KEYS})
        need = lag if delay is None else int(math.ceil(delay/dec_dt-1e-6))
        queue = list(pending)[-need:] if need > 0 else []
        queue = [False]*max(0,need-len(queue))+queue
        margin_floor,margin_ceil = floor,ceil
        if flight_view is not None:
            margin_floor = max(floor,flight_view[0]) if floor is not None else flight_view[0]
            margin_ceil = min(ceil,flight_view[1]) if ceil is not None else flight_view[1]
        a['flight']['visibility_bounds'] = flight_view
        a['flight']['visibility_limited'] = a.get('visibility_limited',{})
        a['flight']['recovery'] = recovery_hint(body,haz,solid,margin_floor,margin_ceil,speed,p,half,flight_buffer,dec_dt,
                                               queue,delay)
        if want_path:
            winner = max(SEARCH_KEYS,key=lambda k:(safety_level(a[k]),
                         a[k] if safety_level(a[k]) == 0 else 0.,quality[k],a[k]))
            a['path'],a['path_hit'],a['path_unknown'] = a.pop('_root_paths')[winner]
            a['plan'] = a.pop('_root_plans')[winner]
            a['plan']['verified'] = safety_level(a[winner])==2 and not a['path_unknown'] and not a['path_hit']
            a['flight']['preferred_root'] = winner
    elif want_path and explicit_plan and '_root_plans' in a:
        quality={k:min(a['clearance'][k],b.get('clearance',a['clearance'])[k]) for k in SEARCH_KEYS}
        winner=max(('00','01','1'),key=lambda k:(safety_level(a[k]),
            a[k] if safety_level(a[k])==0 else 0.,quality[k],a[k]))
        a['route_clearance']=quality
        a['path'],a['path_hit'],a['path_unknown']=a.pop('_root_paths')[winner]
        a['plan']=a.pop('_root_plans')[winner]
        a['plan']['sequence_safety']=a.get('sequence_checks',{}).get(winner)
        a['plan']['verified']=safety_level(a[winner])==2 and not a['path_hit'] and not a['path_unknown']
    elif want_path and a.get('plan'):
        root=a['plan']['root']
        a['plan']['verified'] = safety_level(a[root])==2 and not a.get('path_unknown',False) and not a.get('path_hit',False)
    return a


def safety_score(nominal: float, worse: float) -> float:
    if nominal < 1.0:
        return 0.85 * float(nominal)
    return 1.0 if worse >= 1.0 else 0.90 + 0.09 * float(worse)


def safety_level(v: float) -> int:
    """2 = survives even in the worse world, 1 = only as measured, 0 = crashes."""
    return 2 if v >= 1.0 - 1e-6 else (1 if v >= 0.895 else 0)


# Pads and orbs: what each type does is in gd_objects.OBJECTS (GD 2.2 strengths relative
# to the cube's jump). Pads fire on touch, orbs on a press while touching.
def pad_orb_kind(cls: int, hue: float, kind: str | None = None):
    """("pad" | "orb", effect, type name) for a detected object: from its exact type (YOLO)
    when known, else from its class (orbs) / colour (pads). None = not a pad / orb, unknown."""
    from gd_objects import orb_pad_effect, pad_kind_from_hue
    name = kind
    if not name and cls in (1, 2, 3):
        name = {1: "orb_y", 2: "orb_b", 3: "orb_p"}[cls]
    if not name and cls == 4:
        name = pad_kind_from_hue(hue)
    r = orb_pad_effect(name) if name else None
    return (r[0], r[1], name) if r else None


LAUNCH_LEARNED: dict = {}     # (type name, mode) -> measured strength (x cube jump), see LaunchCalibrator


class LaunchCalibrator:
    """Learns every pad / orb type's REAL launch strength, per game mode, from the icon's
    motion -- the way the cube jump is measured. When the icon touches a pad, or a press
    reaches the game while it touches an orb, the next few heights are fitted with a
    parabola; the launch speed / the measured cube jump = that type's strength in this mode.
    Used (in the blue line) once 3 launches agree (median of the last 15); saved with the
    rest of the measured physics (motion_model.npz)."""

    MIN_N = 3
    LO, HI = 0.15, 3.0
    LEARN = ("jump", "down", "flip", "flipjump")

    def __init__(self) -> None:
        self.samples: dict = {}
        self.pending = None
        self._cool: dict = {}
        self._prev_eff = False

    def strength(self, name: str, mode: str):
        q = self.samples.get((name, mode))
        return float(np.median(q)) if q is not None and len(q) >= self.MIN_N else None

    def _publish(self) -> None:
        LAUNCH_LEARNED.clear()
        for (name, mode) in self.samples:
            v = self.strength(name, mode)
            if v is not None:
                LAUNCH_LEARNED[(name, mode)] = v

    def update(self, y: float, dt: float, mode: str, g: int, mini: bool, eff_hold: bool,
               touching, p: Phys, confident: bool = True) -> None:
        """One step. y: icon centre height (blocks, up = +); touching: [(group, effect, name)]
        for the pads / orbs the icon overlaps right now."""
        pressed = bool(eff_hold) and not self._prev_eff
        self._prev_eff = bool(eff_hold)
        y_prev, self._last_y = getattr(self, "_last_y", None), y
        for k in list(self._cool):
            self._cool[k] -= 1
            if self._cool[k] <= 0:
                del self._cool[k]
        if self.pending is not None:
            pe = self.pending
            pe["t"] += dt
            pe["pts"].append((pe["t"], y))
            if mode != pe["mode"] or len(pe["pts"]) >= 6:
                self._finish(p)
            return
        if not confident or mode in ("wave", "ship", "swing"):
            return                       # (ship / swing: thrust hides the launch; wave: none)
        for group, fx, name in touching:
            if not name or fx is None or fx[0] not in self.LEARN or name in self._cool:
                continue
            if group == "orb" and not pressed:
                continue
            g_after = -g if fx[0] in ("flip", "flipjump") else g
            self.pending = {"name": name, "mode": mode, "fx": fx[0], "g": g_after, "mini": mini,
                            "t": 0.0, "pts": [(0.0, y)], "dt": dt, "y_prev": y_prev}
            self._cool[name] = 20        # one launch per object
            return

    def _finish(self, p: Phys) -> None:
        pe, self.pending = self.pending, None
        pts = pe["pts"][1:]              # the heights AFTER the launch step
        if len(pts) < 4:
            return
        T = np.array([q[0] for q in pts])
        Y = np.array([q[1] for q in pts])
        c2, c1, _c0 = np.polyfit(T, Y, 2)
        G = {"ball": p.ball_g, "ufo": p.ufo_g}.get(pe["mode"], p.cube_g)
        acc = 2.0 * c2
        if abs(acc + pe["g"] * G) > 0.5 * G:
            return                       # didn't fall freely (hit something, landed): don't learn
        # when did it launch? The frame that shows the contact may already show the launch:
        # where the new arc meets the height the step BEFORE it (else: half a step back)
        dt_ = pe["dt"]
        t_l = -0.5 * dt_
        if pe.get("y_prev") is not None:
            r = np.roots([c2, c1, _c0 - pe["y_prev"]])
            r = [float(np.real(q)) for q in r if abs(np.imag(q)) < 1e-9 and -1.5 * dt_ <= np.real(q) <= 0.5 * dt_]
            if r:
                t_l = min(r, key=lambda q: abs(q + 0.5 * dt_))
        vy0 = c1 + acc * t_l             # launch speed
        sc = p.mini if pe["mini"] else 1.0
        sign = -1.0 if pe["fx"] in ("down", "flip") else 1.0
        strength = sign * pe["g"] * vy0 / max(1e-6, p.cube_v0 * sc)
        if not (self.LO <= strength <= self.HI):
            return
        self.samples.setdefault((pe["name"], pe["mode"]), deque(maxlen=15)).append(float(strength))
        self._publish()

    def state(self) -> np.ndarray:
        import json
        return np.array([json.dumps({f"{n}|{m}": list(q) for (n, m), q in self.samples.items()})])

    def load_state(self, arr) -> None:
        import json
        try:
            d = json.loads(str(np.asarray(arr).reshape(-1)[0]))
        except Exception:
            return
        for k, v in d.items():
            n, m = k.split("|", 1)
            self.samples[(n, m)] = deque([float(x) for x in v], maxlen=15)
        self._publish()

    def summary(self) -> str:
        parts = []
        for (n, m), q in sorted(self.samples.items()):
            v = self.strength(n, m)
            parts.append(f"{n}/{m} {v:.2f}x" if v is not None else f"{n}/{m} ?({len(q)})")
        return ", ".join(parts) if parts else "no pad / orb launches measured yet"


def special_effect(b: Body, here, pressed: bool, half: float, p: Phys, surfaces=None) -> None:
    """Apply a pad / orb the icon touches this step.
    here: [(x0, x1, bot, top, "pad" | "orb", effect, idx, name)]; surfaces = (floors, ceilings).
    A strength measured on your game (LaunchCalibrator) replaces the built-in one."""
    from gd_objects import MODE_SCALE
    for x0, x1, sb, st, kind, fx, idx, name in here:
        learned = LAUNCH_LEARNED.get((name, b.mode))
        s = _scale(b, p) * (1.0 if learned is not None else MODE_SCALE.get(b.mode, 1.0))
        if learned is not None and fx[0] in ("jump", "down", "flip", "flipjump"):
            fx = (fx[0], learned)
        if b.used >> idx & 1 or not (sb < b.y + half and st > b.y - half):
            continue
        if kind == "orb" and not pressed:
            continue
        what = fx[0]
        if what == "none":
            continue
        b.used |= 1 << idx
        b.boost = 0.0
        if what == "jump":
            if b.mode != "wave":
                b.vy, b.grounded = b.g * fx[1] * p.cube_v0 * s, False
        elif what == "down":
            if b.mode != "wave":
                b.vy, b.grounded = -b.g * fx[1] * p.cube_v0 * s, False
        elif what == "flip":
            b.g, b.grounded = -b.g, False
            if b.mode != "wave":
                b.vy = -b.g * fx[1] * p.cube_v0 * s       # small push toward the new floor
        elif what == "flipjump":
            b.g, b.grounded = -b.g, False
            if b.mode != "wave":
                b.vy = b.g * fx[1] * p.cube_v0 * s
        elif what == "dash":
            if fx[1]:
                b.g = -b.g
            b.dash, b.vy, b.grounded = True, 0.0, False
        elif what == "teleport_flip":
            floors, ceils = surfaces if surfaces else ([], [])
            if b.g == 1:
                tgt = [c for c in ceils if c > b.y + 0.5 and c - b.y <= p.spider_snap]
                b.y = min(tgt) if tgt else b.y + p.spider_snap
            else:
                tgt = [f for f in floors if f < b.y - 0.5 and b.y - f <= p.spider_snap]
                b.y = max(tgt) if tgt else b.y - p.spider_snap
            b.g, b.vy = -b.g, 0.0
        return


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
    if mode == "swing":
        return max(-p.ship_vmax * s, min(p.ship_vmax * s, v_avg - g * p.ship_down * s * dt / 2))
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


def save_motion(path: str, learner: MotionLearner, phys: Phys, est, calib=None, launch=None) -> None:
    """Learned correction + tuned jump + measured input delay, in one small file
    shared by every brain and level (it's the game's physics, not a level's)."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    d = learner.state()
    d["phys"] = np.array([phys.cube_v0, phys.cube_g, phys.n_cal, getattr(est, "lag", 0)], np.float64)
    if getattr(est, "delay_s", None) is not None:
        d["delay"] = np.array([est.delay_s, est.delay_alt_s if est.delay_alt_s is not None else -1.0], np.float64)
    if calib is not None:
        d["modes"] = calib.state()
    if launch is not None:
        d["launch"] = launch.state()
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **d)
    os.replace(tmp, path)


def load_motion(path: str, learner: MotionLearner, phys: Phys, est, calib=None, launch=None) -> bool:
    import os
    if not os.path.exists(path):
        return False
    try:
        with np.load(path) as d:
            learner.load_state(d)
            v0, g, n_cal, lag = d["phys"]
            if "delay" in d.files:
                ds, da = (float(q) for q in d["delay"])
                est.delay_s, est.delay_alt_s = ds, (da if da >= 0 else None)
            if calib is not None and "modes" in d.files:
                calib.load_state(d["modes"], phys)
            if launch is not None and "launch" in d.files:
                launch.load_state(d["launch"])
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
        self._actions: deque = deque(maxlen=window)
        self.mode, self.g, self.mini = "cube", 1, False
        self.conf = 0.0
        self._cand_streak = (None, 0)
        self._hint, self._hint_t = None, 0
        self.lag = 0                  # input delay (decisions) between a press and the icon reacting
        self.lag_alt = None           # the other likely delay (measured presses split between two)
        self.delay_s = None           # measured input delay in SECONDS (the blue line uses this)
        self.delay_alt_s = None       # the worse world's delay (the far end of the measured spread)
        self._lag_votes = (None, 0)
        self._lag_ok = 0              # checks in a row that confirmed the current delay
        self._n_upd = 0
        self._rest = []

    def eff_hold(self, back: int = 0) -> bool:
        """The action that actually drove the motion `back` steps ago (after input delay)."""
        actions = list(self._actions) if self._actions else [h[1] for h in self.hist]
        i = len(actions) - 1 - back - self.lag
        return bool(actions[i]) if 0 <= i < len(actions) else False

    def pending(self, n: int | None = None) -> list:
        """Actions already sent that haven't reached the icon yet (oldest first).
        n: how many (default: the measured input delay)."""
        n = self.lag if n is None else int(n)
        h = list(self._actions) if self._actions else [q[1] for q in self.hist]
        out = h[max(0, len(h) - n):] if n > 0 else []
        return [0.0] * (n - len(out)) + out

    def resting_before(self) -> bool:
        return bool(self._rest and self._rest[-1])

    def reset(self) -> None:
        self.hist.clear()
        self._actions.clear()
        self.mode, self.g, self.mini = "cube", 1, False   # GD attempts start as a normal cube
        self.conf = 0.0
        self._cand_streak = (None, 0)
        self._hint, self._hint_t = None, 0     # the last attempt's portal doesn't carry over
        self._ghint_t = 0
        self._rest = []                        # (the measured input delay does carry over)

    def gravity_hint(self) -> None:
        """An orb / pad is at the icon: gravity may flip now (allow a fast gravity change)."""
        self._ghint_t = 15

    def icon_hint(self, mode: str | None, conf: float) -> None:
        """The detector RECOGNISED the icon (YOLO: cube / ship / ball ...). Three frames in a
        row of the same answer switch the mode at once -- no portal needed -- and while the
        detector keeps saying so, the motion check can't switch it away."""
        self._icon_lock = max(0, getattr(self, "_icon_lock", 0) - 1)
        if mode is None or mode not in MODE_ID or conf < 0.5:
            self._icon_run = (None, 0)
            return
        m, n = getattr(self, "_icon_run", (None, 0))
        n = n + 1 if m == mode else 1
        self._icon_run = (mode, n)
        if n >= 3:
            self._icon_mode = mode
            self._icon_lock = 15
            if mode != self.mode:
                self.mode, self.mini = mode, False
                self._cand_streak = (None, 0)
                while len(self.hist) > 2:      # motion from before follows other physics
                    self.hist.popleft()

    def portal_fx(self, fx) -> None:
        """A portal of a KNOWN type was crossed (YOLO named it): apply it directly."""
        self._hint_t = 30
        self._hint = None
        while len(self.hist) > 2:
            self.hist.popleft()
        what = fx[0]
        if what == "mode":
            self._hint = fx[1]
            if fx[1] != self.mode:
                self.mode = fx[1]
        elif what == "g":
            self.g = -self.g if fx[1] == 0 else int(fx[1])
        elif what == "size":
            self.mini = bool(fx[1])

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
        if mode == "swing":                    # a press flips its pull: both are possible
            vs = [max(-p.ship_vmax * sc, min(p.ship_vmax * sc, v0 - gg * p.ship_down * sc * dt)) for gg in (g, -g)]
            return vs + [0.0] if resting else vs
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

    def missing(self, hold: bool) -> None:
        """Commands still advance when the icon is not measured. Motion cannot."""
        self._actions.append(bool(hold))
        self.hist.clear()
        self._rest = []
        self._cand_streak = (None, 0)

    def update(self, y: float, hold: bool, dt: float, speed: float, p: Phys) -> None:
        self._actions.append(bool(hold))
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
            icon_says = getattr(self, "_icon_lock", 0) > 0 and getattr(self, "_icon_mode", None) == self.mode
            if k >= (2 if portal else (20 if gravity_only else 40)) and not (icon_says and cand[0] != self.mode):
                self.mode, self.g, self.mini = cand
                self._cand_streak = (None, 0)
        self.conf = float(np.clip(1.0 - best[0] / max(second, 1e-6), 0.0, 1.0))
