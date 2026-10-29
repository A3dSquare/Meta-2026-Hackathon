"""Tiny Geometry Dash simulator for testing the learning pipeline offline.

It is NOT the real game. It reproduces the parts that decide whether the
agent can learn:
  * cube physics at 60 Hz (gravity, ~2-block jump, auto-jump while held),
    spikes with GD-sized hitboxes (small rectangle at the base), solid blocks
    you can land on and die against,
  * GD's death timing: on a crash the camera FREEZES for ~1 s (explosion),
    the level restarts, and only then does the % reset -- which is when the
    real env reports the death,
  * detector imperfections: missed boxes, position noise,
  * the exact observation encoding (env_boxes.encode_obs), death-lag estimate
    (env_boxes.estimate_death_lag) and trainer (trainer.Trainer) used live.

    python sim_gd.py                    # train on the Stereo-like level
    python sim_gd.py --level random --seed 3
    python sim_gd.py --no-trim          # ablation: penalty at %-reset time
    python sim_gd.py --eval checkpoints/sim_stereo.zip --level random

Use it to smoke-test any change to env_boxes / trainer in a minute before
spending an hour of live training on it.
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError("gymnasium is required") from e

from detect import CLS_ID, Entities, Entity
import gd_physics as gp
from env_boxes import FAR, OBS_DIM, encode_obs, estimate_death_lag

SPEED = 10.38          # blocks / s (GD 1x)
G = 92.0               # blocks / s^2
V0 = 19.0              # jump velocity -> ~1.96 block apex, 0.41 s airtime
VIEW_W = 19.0          # blocks visible horizontally
VIEW_H = 10.67         # blocks visible vertically
PLAYER_SX = 0.28       # player screen x
GROUND_SY = 0.75       # ground line screen y
DT = 1.0 / 60.0


A_SHIP = 42.0          # ship vertical acceleration, blocks / s^2
VMAX_SHIP = 8.5        # ship max vertical speed
CEIL = 9.0             # top of the playfield (ship ceiling)
SPEEDS = {0.5: 8.36, 1: 10.38, 2: 12.91, 3: 15.6, 4: 19.2}   # GD speed portals, blocks/s
PORTAL_HUE = {"ship": 0.86, "cube": 0.33, "ball": 0.02, "ufo": 0.10, "wave": 0.52, "robot": -1.0,
              "spider": 0.75, "grav_flip": 0.15, "grav_norm": 0.58}   # same colours gd_physics expects


@dataclass
class Obj:
    kind: str          # spike | wall | ceil (downward spike) | portal | orb
    x: float           # left edge (blocks)
    y: float           # bottom (blocks above ground)
    w: float
    h: float
    what: str = ""     # portal: ship | cube | grav_flip | grav_norm | speed
    val: float = 0.0   # speed portal: blocks/s

    def hitbox(self):
        if self.kind == "wall":
            return (self.x, self.y, self.x + self.w, self.y + self.h)
        hw = 0.2 * min(1.0, self.w)
        cx = self.x + self.w / 2
        if self.kind == "ceil":
            top = self.y + self.h
            return (cx - hw / 2, top - 0.45 * self.h, cx + hw / 2, top)
        return (cx - hw / 2, self.y, cx + hw / 2, self.y + 0.45 * self.h)


def stereo_like() -> tuple[list, float]:
    """Hand-made opening in the spirit of Stereo Madness (not a copy)."""
    o: list = []
    S = lambda x, w=1.0, h=0.95, y=0.0: o.append(Obj("spike", x, y, w, h))
    B = lambda x, y, w, h=1.0: o.append(Obj("wall", x, y, w, h))
    S(16)                       # the first spike
    S(27)
    S(36); S(37)                # double
    B(46, 0, 1); B(50, 0, 1, 2)     # steps up
    B(54, 0, 4, 2); S(56, y=2.0)    # platform with a spike on it
    S(62); S(63)
    S(70, 1.0, 0.45); S(71, 1.0, 0.45); S(72, 1.0, 0.45)  # small spike row
    B(80, 0, 3); S(83); S(84)      # block then spikes after it
    B(92, 0, 2, 1); B(96, 0, 2, 2); B(100, 0, 2, 3)  # staircase
    S(104); S(105); S(106)         # triple
    S(114, 0.55, 0.45); S(118)     # mini then regular
    B(124, 3.2, 3); o.append(Obj("ceil", 125, 2.25, 1.0, 0.95))  # hanging spike
    S(130); S(131)
    S(140, 1.0, 0.45); S(144); S(145)
    return o, 155.0


def modes_level() -> tuple[list, float]:
    """Cube -> ship corridor -> cube at 2x speed with an orb gap -> upside-down
    cube on a ceiling -> normal. Exercises speed, mode, gravity and orbs."""
    o: list = []
    S = lambda x, w=1.0, h=0.95, y=0.0: o.append(Obj("spike", x, y, w, h))
    Wl = lambda x, y, w, h: o.append(Obj("wall", x, y, w, h))
    P = lambda x, what, val=0.0: o.append(Obj("portal", x, 0.0, 1.0, 3.0, what, val))
    S(16); S(26); S(27); S(36, 1.0, 0.45)
    P(46, "ship")
    Wl(46, CEIL - 1, 62, 1)                       # corridor ceiling 46..108
    Wl(58, 0, 2, 3)                               # fly over
    Wl(68, 4.5, 2, CEIL - 1 - 4.5)                # fly under
    S(76); S(77); S(78)
    Wl(86, 0, 1, 2.5); Wl(86, 5.5, 1, CEIL - 1 - 5.5)   # gap 2.5..5.5
    Wl(96, 0, 1, 4.0)
    P(108, "cube")
    P(112, "speed", SPEEDS[2])
    S(120); S(130); S(131)
    for k in range(7):                            # 7-wide spike pit: needs the orb
        S(140 + k)
    o.append(Obj("orb", 141.5, 1.6, 1.0, 1.0))
    P(156, "speed", SPEEDS[1])
    P(160, "grav_flip")
    Wl(160, 5.0, 36, 1)                           # ceiling to run on, 160..196
    o.append(Obj("ceil", 172, 4.05, 1.0, 0.95))   # spikes hanging from that ceiling
    o.append(Obj("ceil", 181, 4.05, 1.0, 0.95)); o.append(Obj("ceil", 182, 4.05, 1.0, 0.95))
    P(192, "grav_norm")
    S(206)
    return o, 215.0


def allmodes_level() -> tuple[list, float]:
    """One section per game mode, entered through its portal (ends as an upside-down cube)."""
    o: list = []
    S = lambda x, w=1.0, h=0.95, y=0.0: o.append(Obj("spike", x, y, w, h))
    Wl = lambda x, y, w, h: o.append(Obj("wall", x, y, w, h))
    P = lambda x, what, val=0.0: o.append(Obj("portal", x, 0.0, 1.0, 3.0, what, val))
    S(16); S(26); S(27)                                   # cube
    P(40, "ship"); Wl(52, 0, 2, 3); Wl(64, 5, 2, CEIL - 5); S(74); S(75); S(76)
    P(88, "ball"); Wl(88, 6, 40, 1); S(100); S(101); o.append(Obj("ceil", 114, 5.05, 1, 0.95))
    P(130, "ufo"); Wl(142, 0, 2, 3); S(154); S(155); S(156)
    P(168, "wave"); Wl(180, 0, 2, 3); Wl(194, 5, 2, CEIL - 5)
    P(208, "robot"); S(220); S(228)
    P(242, "spider"); Wl(242, 6, 34, 1); S(254); S(255); o.append(Obj("ceil", 266, 5.05, 1, 0.95))
    P(280, "cube"); P(284, "grav_flip"); Wl(284, 5.0, 30, 1); o.append(Obj("ceil", 296, 4.05, 1, 0.95))
    P(310, "grav_norm"); S(322)
    return o, 332.0


def random_level(seed: int, length: float = 160.0) -> tuple[list, float]:
    rng = random.Random(seed)
    o: list = []
    x = 14.0
    while x < length - 10:
        r = rng.random()
        if r < 0.45:
            kind = rng.choice([(1.0, 0.95), (1.0, 0.95), (1.0, 0.45), (0.55, 0.45)])
            n = rng.choice([1, 1, 2, 3]) if kind[0] == 1.0 else rng.choice([1, 2])
            for k in range(n):
                o.append(Obj("spike", x + k * kind[0], 0.0, *kind))
            x += n * kind[0]
        elif r < 0.75:
            h = 1  # a cube can't jump onto a 2-high wall from the floor
            w = rng.choice([1, 2, 3, 4])
            o.append(Obj("wall", x, 0.0, w, h))
            if w >= 3 and rng.random() < 0.5:
                o.append(Obj("spike", x + 1, float(h), 1.0, 0.95))
            x += w
        else:
            o.append(Obj("wall", x, 3.2, 2, 1))
            o.append(Obj("ceil", x + 0.5, 2.25, 1.0, 0.95))
            x += 2
        x += rng.uniform(5.0, 9.0)
    return o, length


class SimGD(gym.Env):
    """Same interface as env_boxes.EntityBoxEnv."""

    def __init__(self, level=None, frame_skip: int = 2, dead_ticks: int = 60, signal_delay: int = 2,
                 p_miss: float = 0.03, pos_noise: float = 0.002, seed: int = 0, use_jump: bool = True,
                 input_lag: int = 0, world_phys: "gp.Phys | None" = None, motion_fix: bool = False) -> None:
        self.use_jump = use_jump
        # the simulated game's true physics; the preview's formulas (self.phys) can differ,
        # which is what the learned correction (motion_fix) is there to make up for
        self.world = world_phys if world_phys is not None else gp.Phys()
        self.motion_fix = gp.MotionLearner(seed) if motion_fix else None
        self._mstate = None
        self.input_lag = int(input_lag)   # decisions between sending SPACE and the icon reacting
        self._sent: list = []
        self.phys = gp.Phys()
        self.est = gp.ModeEstimator()
        self._hinted: set = set()
        self.objs, self.length = level if level is not None else stereo_like()
        self.objs.sort(key=lambda o: o.x)
        self.frame_skip = frame_skip
        self.dead_ticks = dead_ticks
        self.signal_delay = signal_delay
        self.p_miss = p_miss
        self.pos_noise = pos_noise
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(-FAR, FAR, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.fallback_steps = int(round(0.6 * 60 / frame_skip))
        self.attempts = 1
        self.game_ticks = 0
        self.moving_hist: list = []
        self.prev_py = None
        self.hist: list = []
        self._respawn()

    # -- physics ---------------------------------------------------------------
    def _respawn(self) -> None:
        self.x, self.y, self.vy = 0.0, 0.0, 0.0
        self.grounded = True
        self.mode, self.gdir, self.speed = "cube", 1, SPEEDS[1]
        self.boost, self.prev_hold = 0.0, False
        self.used: set = set()  # portals / orbs already triggered this attempt
        self.dead = 0           # >0: ticks left in the death animation
        self.signal = -1        # >=0: ticks until the "% reset" death signal

    def _surfaces(self, x0, x1):
        tops, bots = [0.0], [CEIL] if self.mode == "ship" else []
        for o in self.objs:
            if o.kind != "wall" or o.x > x1 or o.x + o.w < x0:
                continue
            tops.append(o.y + o.h)
            bots.append(o.y)
        return tops, bots

    def _tick(self, hold: bool) -> bool:
        """Advance one frame. Returns True if the camera moved."""
        self.game_ticks += 1
        if self.dead > 0:
            self.dead -= 1
            if self.dead == 0:
                self._respawn()
                self.signal = self.signal_delay
                return True
            return False
        if self.signal >= 0:
            self.signal -= 1
        self.x += self.speed * DT
        px0, px1 = self.x, self.x + 1.0
        for i, o in enumerate(self.objs):
            if o.x > px1 + 1:
                break
            if i in self.used:
                continue
            if o.kind == "portal" and o.x <= self.x + 0.5:
                self.used.add(i)
                if o.what in gp.MODES:
                    self.mode = o.what
                    self.vy *= 0.5
                    self.grounded = False
                elif o.what == "grav_flip":
                    self.gdir, self.grounded = -1, False
                elif o.what == "grav_norm":
                    self.gdir, self.grounded = 1, False
                elif o.what == "speed":
                    self.speed = o.val
            elif (o.kind == "orb" and hold and self.mode in ("cube", "robot", "ball", "spider")
                  and abs(o.x + 0.5 - (self.x + 0.5)) < 0.9 and abs(o.y + 0.5 - (self.y + 0.5)) < 0.9):
                self.used.add(i)
                self.vy, self.grounded = self.world.cube_v0 * self.gdir, False
        floors, ceils = [0.5], [CEIL - 0.5]
        for o in self.objs:
            if o.kind == "wall" and o.x < px1 - 0.05 and o.x + o.w > px0 + 0.05:
                floors.append(o.y + o.h + 0.5)
                ceils.append(o.y - 0.5)
        b = gp.Body(self.y + 0.5, self.vy, self.gdir, self.grounded, self.mode, False, self.boost, self.prev_hold)
        b = gp.advance(b, hold, DT, self.speed, self.world, (floors, ceils))
        self.y, self.vy, self.gdir, self.grounded = b.y - 0.5, b.vy, b.g, b.grounded
        self.boost, self.prev_hold = b.boost, b.prev_hold
        if self.y > CEIL + 3 or self.y < -0.5:
            self.dead = self.dead_ticks
            return True
        for o in self.objs:
            if o.x > px1 + 1:
                break
            if o.x + o.w < px0 - 1 or o.kind in ("portal", "orb"):
                continue
            hx0, hy0, hx1, hy1 = o.hitbox()
            if hx0 < px1 - 0.02 and hx1 > px0 + 0.02 and hy0 < self.y + 0.98 and hy1 > self.y + 0.02:
                self.dead = self.dead_ticks
                return True
        return True

    # -- observation ---------------------------------------------------------
    def _entities(self) -> Entities:
        items = []
        py_s = GROUND_SY - (self.y + 0.5) / VIEW_H
        pw, ph = {"ship": (1.5, 0.9), "wave": (0.7, 0.7), "ufo": (1.3, 1.0), "spider": (1.4, 0.8),
                  "robot": (1.1, 1.2)}.get(self.mode, (1.0, 1.0))
        items.append(Entity(CLS_ID["player"], PLAYER_SX, py_s, pw / VIEW_W, ph / VIEW_H, 0.9))
        for o in self.objs:
            sx0 = PLAYER_SX + (o.x - self.x - 0.5) / VIEW_W
            if sx0 > 1.0:
                break
            sx1 = sx0 + o.w / VIEW_W
            if sx1 < 0:
                continue
            if o.kind == "portal" and o.what == "speed":
                continue          # speed is sensed directly from the scroll
            if self.rng.random() < self.p_miss:
                continue
            sy1 = GROUND_SY - o.y / VIEW_H
            sy0 = sy1 - o.h / VIEW_H
            n = self.rng.normal(0, self.pos_noise, 2)
            hue = -1.0
            if o.kind == "wall":
                cls = CLS_ID["wall"]
            elif o.kind == "portal":
                cls, hue = CLS_ID["portal"], PORTAL_HUE.get(o.what, 0.5)
            elif o.kind == "orb":
                cls, hue = CLS_ID["orb_y"], 0.15
            else:
                cls = CLS_ID["spike"]
            items.append(Entity(cls, (sx0 + sx1) / 2 + n[0], (sy0 + sy1) / 2 + n[1],
                                sx1 - sx0, sy1 - sy0, 0.9, hue))
        return Entities(items)

    def _obs(self, hold: bool) -> np.ndarray:
        from env_boxes import player_vy
        ents = self._entities()
        speed = self.speed * (1 + self.rng.normal(0, 0.02))
        from env_boxes import BLOCK_H, physics_features
        dt = self.frame_skip / 60.0
        prev_hold = bool(self.hist and self.hist[-1][1] > 0.5)
        vy_step = player_vy(ents, self.prev_py)
        recent = [h[0] for h in self.hist[-2:]] + [vy_step]
        grounded = len(recent) >= 2 and all(abs(v) / dt < gp.ModeEstimator.REST for v in recent)
        if self.use_jump:
            yb = self.y + 0.5 + self.rng.normal(0, 0.03)
            self.est.update(yb, hold, dt, speed, self.phys)
            for o in self.objs:
                if o.kind == "portal" and abs(o.x + 0.5 - (self.x + 0.5)) < 0.6 and id(o) not in self._hinted:
                    self._hinted.add(id(o))
                    self.est.portal_hint(PORTAL_HUE.get(o.what, -1.0))
            e = self.est
            key = (e.mode, e.g, e.mini)
            if self.motion_fix is not None and self._mstate is not None and self._mstate[3] == key \
                    and self.dead == 0 and e._hint_t < 25 and abs(vy_step / dt) < 40:
                b0, (f0, c0), s0, _ = self._mstate
                self.motion_fix.observe(b0, e.eff_hold(0), f0, c0, s0, dt, self.phys, vy_step / dt)
            out: dict = {}
            self._pf_args = (ents, speed, self.est, self.phys, self.frame_skip, vy_step / dt, grounded, hold,
                             -(self.y + 0.5), CEIL - (self.y + 0.5))   # (tests: redo the preview)
            mf, plan, self.last_arcs = physics_features(*self._pf_args, motion=self.motion_fix, out=out)
            self._mstate = (out["body"], out["surf"], speed, key) if self.dead == 0 else None
        else:
            mf = plan = None
            self.last_arcs = []
        v = encode_obs(ents, hold=hold, prev_py=self.prev_py, player_found=True,
                       speed=speed, hist=self.hist, mode=mf, plan=plan,
                       extra=out.get("extra_plans") if self.use_jump else None,
                       search=out.get("search") if self.use_jump else None)
        self.hist.append((player_vy(ents, self.prev_py), 1.0 if hold else 0.0))
        self.hist = self.hist[-8:]
        self.prev_py = ents.player_pos()[1]
        return v

    # -- gym API -------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        return self._obs(False), {"attempts": self.attempts}

    def step(self, action):
        sent = int(action) == 1
        self._sent = (self._sent + [sent])[-(self.input_lag + 1):]
        hold = self._sent[0] if len(self._sent) > self.input_lag else False   # what reaches the game
        moved = False
        died = complete = False
        for _ in range(self.frame_skip):
            moved = self._tick(hold) or moved
            if self.signal == 0:
                died = True
                break
            if self.dead == 0 and self.x >= self.length:
                complete = True
                break
        self.moving_hist.append(moved)
        info = {"moving": moved, "prog": min(1.0, self.x / self.length) if self.dead == 0 else None,
                "died": died, "complete": complete, "attempts": self.attempts,
                "death_lag": 0, "death_lag_src": ""}
        if died:
            hist = self.moving_hist
            lag, src = estimate_death_lag(hist, self.fallback_steps)
            info.update(death_lag=lag, death_lag_src=src)
            self.attempts += 1
            self.signal = -1
            self.moving_hist = []
            self.prev_py = None
            self.hist = []
            self._mstate = None
            self.est.reset()
            self._hinted = set()
        if complete:
            self.attempts += 1
            self.moving_hist = []
            self.hist = []
            self._respawn()
        return self._obs(sent), 0.0, complete, False, info

    def pause(self):
        pass

    def resume(self):
        pass

    def mark_trained(self):
        pass


def evaluate(trainer, env: SimGD, episodes: int = 5, deterministic: bool = True) -> list:
    """Peak progress of a few attempts with the current policy."""
    out = []
    for _ in range(episodes):
        obs, _ = env.reset()
        peak = 0.0
        for _ in range(20000):
            a, _ = trainer.act(obs, deterministic=deterministic)
            obs, _, _, _, info = env.step(a)
            if info.get("prog") is not None:
                peak = max(peak, info["prog"])
            if info["died"] or info["complete"]:
                break
        out.append(peak)
    return out


def main() -> int:
    from trainer import Trainer, TrainConfig
    p = argparse.ArgumentParser(description="train the box policy on a simulated GD level")
    p.add_argument("--level", default="stereo", choices=("stereo", "random", "one", "modes", "allmodes"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--algo", default="dqn", choices=("dqn", "ppo"))
    p.add_argument("--frame-skip", type=int, default=2)
    p.add_argument("--no-trim", action="store_true", help="ablation: no death-time correction")
    p.add_argument("--eval", default=None, help="evaluate a checkpoint instead of training")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-shield", action="store_true", help="no physics-preview safety shield")
    p.add_argument("--motion-fix", action="store_true", help="also learn a correction of the physics preview (experimental)")
    p.add_argument("--input-lag", type=int, default=0,
                   help="decisions between pressing and the icon reacting (real GD + capture: ~1)")
    a = p.parse_args()
    if a.level == "stereo":
        level = stereo_like()
    elif a.level == "modes":
        level = modes_level()
    elif a.level == "allmodes":
        level = allmodes_level()
    elif a.level == "one":
        level = ([Obj("spike", 16, 0, 1, 0.95)], 40.0)
    else:
        level = random_level(a.seed)
    env = SimGD(level, frame_skip=a.frame_skip, seed=a.seed, input_lag=a.input_lag,
                motion_fix=a.motion_fix)
    cfg = TrainConfig(algo=a.algo, total_steps=a.steps, shield=not a.no_shield, name=f"sim_{a.level}", log_dir="./logs_sim",
                      trim_death=not a.no_trim, verbose=0 if a.quiet else 1,
                      stop_on_complete=False)
    tr = Trainer(env, cfg, model_path=a.eval)
    if a.eval:
        print("peak % per attempt:", [round(100 * v) for v in evaluate(tr, env, 5)])
        return 0
    tr.run()
    r = evaluate(tr, env, 5)
    print(f"done: {tr.attempts} attempts, {tr.steps} steps ({tr.steps * a.frame_skip / 60 / 60:.1f} game-minutes); "
          f"greedy policy peak % = {[round(100 * v) for v in r]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
