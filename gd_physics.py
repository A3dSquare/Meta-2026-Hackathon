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
        dv = -b.g * (p.ship_a if hold else -p.ship_a) * s * dt
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
                   dec_dt: float | None = None, lag: int = 0, pending=(), motion=None):
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
         for _ in PLANS]
    pts = [[(0.0, body.y)] for _ in PLANS]
    hit_t = [None] * N_PLANS
    clear = [3.0] * N_PLANS
    live = list(range(N_PLANS))
    t_ledge = None
    x_prev = 0.0
    corr_by: dict = {}
    corr_key: dict = {}
    for i in range(1, steps + 1):
        if not live:
            break
        t = i * dt
        x = speed * t
        jd = int((t - dt) / dec_dt + 1e-9)          # decision index during this step (0 = now)
        if jd < len(pending):
            holds = [bool(pending[jd])] * len(live)  # already sent, still in the pipeline
        else:
            holds = [bool(PLANS[pi][1](jd - len(pending))) for pi in live]
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
            advance(b, holds[k], dt, speed, p, (floors, ceils))
            if corr is not None and not b.grounded:
                b.y += float(corr[k]) * dt                # learned leftover error
                b.vy += float(corr[k])
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
    paths = []
    for pi in range(N_PLANS):
        h = hit_t[pi]
        feats[3 * pi] = 1.0 if h is not None else 0.0
        feats[3 * pi + 1] = (h if h is not None else horizon) / horizon
        feats[3 * pi + 2] = float(np.clip(clear[pi], -2.0, 3.0)) / 2.0
        paths.append((PLANS[pi][0], h is not None, pts[pi]))
    feats[-1] = (t_ledge if t_ledge is not None else horizon) / horizon
    return feats, paths


def inst_vy(mode: str, g: int, mini: bool, v_avg: float, hold: bool, grounded: bool,
            dt: float, p: Phys) -> float:
    """Measured speed is the AVERAGE over the last step; the simulation wants the
    speed at the END of it (they differ by half a step of gravity / thrust)."""
    if grounded or mode == "wave":
        return v_avg
    s = p.mini if mini else 1.0
    if mode == "ship":
        a = g * (p.ship_a if hold else -p.ship_a) * s
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


def save_motion(path: str, learner: MotionLearner, phys: Phys, est) -> None:
    """Learned correction + tuned jump + measured input delay, in one small file
    shared by every brain and level (it's the game's physics, not a level's)."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    d = learner.state()
    d["phys"] = np.array([phys.cube_v0, phys.cube_g, phys.n_cal, getattr(est, "lag", 0)], np.float64)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **d)
    os.replace(tmp, path)


def load_motion(path: str, learner: MotionLearner, phys: Phys, est) -> bool:
    import os
    if not os.path.exists(path):
        return False
    try:
        with np.load(path) as d:
            learner.load_state(d)
            v0, g, n_cal, lag = d["phys"]
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
        self._lag_votes = (None, 0)
        self._lag_ok = 0              # checks in a row that confirmed the current delay
        self._n_upd = 0
        self._rest = []

    def eff_hold(self, back: int = 0) -> bool:
        """The action that actually drove the motion `back` steps ago (after input delay)."""
        i = len(self.hist) - 1 - back - self.lag
        return bool(self.hist[i][1]) if 0 <= i < len(self.hist) else False

    def pending(self) -> list:
        """Actions already sent that haven't reached the icon yet (oldest first)."""
        return [h[1] for h in list(self.hist)[len(self.hist) - self.lag:]] if self.lag > 0 else []

    def resting_before(self) -> bool:
        return bool(self._rest and self._rest[-1])

    def reset(self) -> None:
        self.hist.clear()
        self.mode, self.g, self.mini = "cube", 1, False   # GD attempts start as a normal cube
        self.conf = 0.0
        self._cand_streak = (None, 0)
        self._hint, self._hint_t = None, 0     # the last attempt's portal doesn't carry over
        self._rest = []                        # (the measured input delay does carry over)

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

    def _score(self, mode: str, g: int, mini: bool, vys, holds, dt: float, speed: float, p: Phys, lag: int = 0):
        """Mean prediction error over the window, starting with gravity g.
        Ball and spider flip gravity themselves, so g is tracked through the
        window; returns (error, n, gravity at the end of the window)."""
        err, n = 0.0, 0
        for k in range(3 + lag, len(vys)):    # vys[0] is a placeholder: start where the
            v0, v1 = vys[k - 1], vys[k]       # previous two speeds are real measurements
            h, hp = holds[k - lag], holds[k - 1 - lag]   # what reached the icon (input delay)
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
        self._n_upd += 1
        # input-delay check: often at first, rarely once it has been confirmed a few times
        if self._n_upd % (8 if self._lag_ok < 5 else 40) == 0 and any(holds):
            self._fit_lag(vys, holds, dt, speed, p)
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
                    sc, n, g_end = self._score(mode, g, mini, vys, holds, dt, speed, p, self.lag)
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
