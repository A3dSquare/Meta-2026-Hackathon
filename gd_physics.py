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
    cube_v0: float = 22.36      # jump speed (blocks/s)  -> apex ~2.17 blocks
    cube_g: float = 115.0       # gravity (blocks/s^2)   -> ~0.39 s in the air
    ship_a: float = 55.0        # ship up/down acceleration
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
        """Learn the cube's apex / airtime from clean floor-to-floor jumps."""
        if self._track is None:
            if pressed and grounded_before and vy_step > 0.15:
                self._track = [dt, y - vy_step, max(0.0, vy_step)]
            return
        t, y0, rise = self._track
        t += dt
        rise = max(rise, y - y0)
        if y - y0 <= 0.05 and t > 0.15:
            if 1.2 <= rise <= 3.5 and 0.25 <= t <= 0.8:
                apex = self.apex + 0.15 * (rise - self.apex)
                air = self.airtime + 0.15 * (t - self.airtime)
                self.cube_v0 = 4 * apex / air
                self.cube_g = 8 * apex / air ** 2
                self.n_cal += 1
            self._track = None
        elif t > 1.0:
            self._track = None
        else:
            self._track = [t, y0, rise]


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
    if m == "cube":
        if b.grounded and hold:
            b.vy, b.grounded = b.g * p.cube_v0 * s, False
        if not b.grounded:
            b.vy -= b.g * p.cube_g * dt
    elif m == "robot":
        if b.grounded and hold:
            b.vy, b.grounded, b.boost = b.g * p.robot_v0 * s, False, p.robot_boost
        if not b.grounded:
            if hold and b.boost > 0:
                b.boost -= dt
            else:
                b.boost = 0.0
                b.vy -= b.g * p.cube_g * dt
    elif m == "ball":
        if b.grounded and press:
            b.g, b.grounded, b.vy = -b.g, False, -b.g * 2.0   # flip and start moving away
        if not b.grounded:
            b.vy -= b.g * p.ball_g * dt
    elif m == "ufo":
        if press:
            b.vy, b.grounded = b.g * p.ufo_v0 * s, False
        if not b.grounded:
            b.vy -= b.g * p.ufo_g * dt
    elif m == "ship":
        b.vy += b.g * (p.ship_a if hold else -p.ship_a) * s * dt
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
            b.vy -= b.g * p.cube_g * dt
    y_new = b.y + b.vy * dt
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


def simulate_plans(body: Body, haz, solid, floor, ceil, speed: float, p: Phys,
                   frame_skip: int = 2, horizon: float = 1.5, half: float = 0.4):
    """Run every plan forward; return (features [N_PLANS*3 + 1], paths).

    haz / solid: boxes (x0, x1, bottom, top) in blocks relative to the icon.
    floor / ceil: surface heights (None = unknown). Per plan: hit?, time to
    hit / horizon (1 = never), clearance to the nearest hazard. Plus: time
    until the "release" plan leaves its current surface (ledge).
    """
    dt = 1.0 / 30.0
    steps = int(horizon / dt)
    dec_dt = frame_skip / 60.0
    feats = np.zeros(N_PLANS * 3 + 1, dtype=np.float32)
    paths = []
    t_ledge = None
    for pi, (_name, plan) in enumerate(PLANS):
        b = Body(body.y, body.vy, body.g, body.grounded, body.mode, body.mini, body.boost, body.prev_hold)
        pts = [(0.0, b.y)]
        hit_t, clear = None, 3.0
        x_prev = 0.0
        for i in range(1, steps + 1):
            t = i * dt
            x = speed * t
            hold = plan(int((t - dt) / dec_dt + 1e-9))   # the action during this step (decision 0 = now)
            floors, ceils = [], []
            for s0, s1, sb, st in solid:
                if s0 < x + half - 0.05 and s1 > x - half + 0.05:
                    floors.append(st + 0.5)          # centre height standing on top
                    ceils.append(sb - 0.5)           # centre height hanging under it
            if floor is not None:
                floors.append(floor + 0.5)
            if ceil is not None:
                ceils.append(ceil - 0.5)
            was = b.grounded
            b = advance(b, hold, dt, speed, p, (floors, ceils))
            if pi == 0 and was and not b.grounded and t_ledge is None and b.mode not in ("ship", "wave", "ufo"):
                t_ledge = t
            y = b.y
            pts.append((x, y))
            lo, hi = x_prev - half, x + half               # swept horizontally (no tunnelling)
            for h0, h1, hb, ht in haz:
                if h0 < hi and h1 > lo:
                    gap = (y - half) - ht if y > (hb + ht) / 2 else hb - (y + half)
                    clear = min(clear, gap)
                    if hb < y + half and ht > y - half:
                        hit_t = t
            if hit_t is None:
                for s0, s1, sb, st in solid:
                    if s0 < hi - 0.05 and s1 > lo + 0.05 and sb < y + half - 0.12 and st > y - half + 0.12:
                        hit_t = t                  # into the side of a block
                        break
            if hit_t is None and (y < -12 or y > 14):
                hit_t = t                          # out of the world
            if hit_t is not None:
                break
            x_prev = x
        feats[3 * pi] = 1.0 if hit_t is not None else 0.0
        feats[3 * pi + 1] = (hit_t if hit_t is not None else horizon) / horizon
        feats[3 * pi + 2] = float(np.clip(clear, -2.0, 3.0)) / 2.0
        paths.append((PLANS[pi][0], hit_t is not None, pts))
    feats[-1] = (t_ledge if t_ledge is not None else horizon) / horizon
    return feats, paths


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

    def reset(self) -> None:
        self.hist.clear()
        self.mode, self.g, self.mini = "cube", 1, False   # GD attempts start as a normal cube
        self.conf = 0.0
        self._cand_streak = (None, 0)

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
        if d[m] < 0.04:                # colour clearly says which mode: take it now,
            self._hint = m             # the motion check can still overrule it
            if m != self.mode:
                self.mode, self.mini = m, False

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
            v = max(-p.ship_vmax * sc, min(p.ship_vmax * sc, v0 + g * (p.ship_a if h else -p.ship_a) * sc * dt))
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

    def _score(self, mode: str, g: int, mini: bool, vys, holds, dt: float, speed: float, p: Phys):
        """Mean prediction error over the window, starting with gravity g.
        Ball and spider flip gravity themselves, so g is tracked through the
        window; returns (error, n, gravity at the end of the window)."""
        err, n = 0.0, 0
        for k in range(3, len(vys)):          # vys[0] is a placeholder: start where the
            v0, v1 = vys[k - 1], vys[k]       # previous two speeds are real measurements
            h, hp = holds[k], holds[k - 1]
            resting = self._rest[k]
            pred = self._pred(mode, g, mini, v0, h, hp, resting, dt, speed, p)
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

    def update(self, y: float, hold: bool, dt: float, speed: float, p: Phys) -> None:
        self.hist.append((y, bool(hold)))
        if len(self.hist) < 6:
            return
        ys = [h[0] for h in self.hist]
        holds = [h[1] for h in self.hist]
        vys = [0.0] + [(ys[k] - ys[k - 1]) / dt for k in range(1, len(ys))]
        if not any(holds[1:]) and sum(abs(v) > 2 * self.REST for v in vys[1:]) < 2:
            return                    # sitting still / sliding, not pressing: nothing to learn
        if self._hint_t > 0:          # (the portal window only counts informative steps)
            self._hint_t -= 1
        # 'resting before step k' = barely moved over the two steps before it (two-step
        # displacement: much less sensitive to detector jitter than single-step speeds)
        self._rest = [k >= 3 and abs(ys[k - 1] - ys[k - 3]) < self.REST * 2 * dt for k in range(len(ys))]
        if self.mode in ("ball", "spider") and holds[-1] and not holds[-2] and self._rest[-1]:
            self.g = -self.g          # a press on a surface flips ball / spider gravity
        scores = []
        for mode in MODES:
            for g in (1, -1):
                for mini in (False, True):
                    sc, n, g_end = self._score(mode, g, mini, vys, holds, dt, speed, p)
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
            # gravity alone changes often (orbs, pads, ball/spider presses): treat like a portal
            portal = self._hint_t > 0 or cand[0::2] == (self.mode, self.mini)
            clearly = cur is None or best[0] < (0.85 if portal else 0.6) * cur
            c, k = self._cand_streak
            k = k + 1 if (c == cand and clearly) else (1 if clearly else 0)
            self._cand_streak = (cand, k)
            if k >= (2 if portal else 5):
                self.mode, self.g, self.mini = cand
                self._cand_streak = (None, 0)
        self.conf = float(np.clip(1.0 - best[0] / max(second, 1e-6), 0.0, 1.0))
