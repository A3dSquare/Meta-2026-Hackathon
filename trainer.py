"""Box-policy trainer: one loop for the live game, headless runs and the simulator.

Default learner: Double DQN (dueling, n-step returns, replay buffer).
Optional learner: PPO (--algo ppo), kept for comparison.

Why this replaces the old per-death PPO loop in preview_boxes.py:

  OLD                                          NEW
  -------------------------------------------  -------------------------------------------
  reward = a dozen shaped terms (spike-        +0.1 per step alive, -10 at the crash, +50
  proximity rent, "clearance" bonus, hold      for finishing. In GD you can't stall, so
  bonus...) driven by detector noise; holding  staying alive IS making progress.
  SPACE near spikes was paid -> early jumps
  death penalty landed ~1 s after the crash    steps after the crash are cut off; the
  (when the % resets): almost no credit        penalty lands on the step that caused it
  reached the jump that mattered
  PPO update on ONE attempt (~90 samples)      every step goes into a replay buffer and is
  then thrown away -- each sample costs real   reused for many updates (an order of
  game time and is used once                   magnitude fewer attempts in the simulator)
  ent_coef 0.08 at 60 Hz = near-constant       epsilon-greedy with a decaying epsilon,
  random jumping                               30 decisions/s (frame_skip=2)

The env must look like env_boxes.EntityBoxEnv: reset() -> (obs, info),
step(a) -> (obs, r, term, trunc, info) with info keys died / death_lag /
complete / prog, plus pause() / resume() / mark_trained().
"""
from __future__ import annotations

import csv
import math
import os
import time
from training_diagnostics import SessionLog, file_sha256
from flight_reward import clearance_weight, shaping_reward
from flight_control import FlightShield, teacher_action


def flight_mode_mask(obs):
    import sys as _s
    _eb = _s.modules.get("env_boxes")
    _gp = _s.modules.get("gd_physics")
    OFF_MODE = _eb.OFF_MODE if _eb is not None else __import__("env_boxes").OFF_MODE
    MODE_ID = _gp.MODE_ID if _gp is not None else __import__("gd_physics").MODE_ID
    if obs.shape[-1] <= OFF_MODE + max(MODE_ID['ship'],MODE_ID['wave']):
        return obs[...,0] < -math.inf
    return (obs[..., OFF_MODE+MODE_ID['ship']] > .5) | (obs[..., OFF_MODE+MODE_ID['wave']] > .5)
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable

import numpy as np


@dataclass
class TrainConfig:
    algo: str = "dqn"              # dqn | ppo
    total_steps: int = 3_000_000
    # --- shared -----------------------------------------------------------
    gamma: float = 0.98
    lr: float = 5e-4
    alive_reward: float = 0.1
    hold_cost: float = 0.02        # tiny price per step SPACE is held: makes "release" win when holding
                                   # doesn't matter (mid-air), so it stops bunny-hopping by default
    clearance_weight: float = 0.0 # optional bounded ship/wave potential; zero preserves old rewards
    flight_margin: float = 0.0    # thrust has no cube-jump threshold, including Double-DQN targets
    flight_buffer: float = .75    # desired trajectory clearance in blocks; zero disables buffered shield
    flight_teacher_weight: float = .05 # hinge coaching from verified, unambiguous safe flight interventions
    death_penalty: float = -10.0
    complete_bonus: float = 50.0
    net_arch: tuple = (256, 256)
    device: str = "cpu"
    cpu_threads: int = 1          # small DQN inference benchmark: fewer scheduling outliers
    route_controller: bool = False
    controller_eval_min: int = 32
    strict_offline_gate: bool = False  # old promotion gate (human-label accuracy + TD); default: safety+motion, then the live A/B
    shield_log: bool = True        # log every shield-vs-brain disagreement (route controller)
    shield_auto_off: bool = True   # trusted modes: auto-disable shield when logged evidence passes
    shield_min_overrides: int = 30 # minimum trusted-mode overrides before auto-off may trigger
    max_grad_norm: float = 10.0
    update_steps: int = 128        # train once this many new steps are banked (checked at deaths)
    max_rollout: int = 2048        # ...or mid-attempt once this many are pending
    trim_death: bool = True        # cut post-crash steps (ablation switch)
    shield: bool = True            # never take an action the physics preview says crashes when
                                   # the other action has a path that survives (see safety_shield)
    stop_on_complete: bool = True  # live GD sits on the level-complete screen: save and stop
    route: bool = True             # per-level practice of stuck spots, kept out of the brain's learning (route.py)
    route_dir: str = "./routes"    # one file per level it has seen (kept forever)
    route_level: str | None = None # tests only: skip recognising the level
    landmarks: bool = False        # --landmarks: replay the best run's presses by WHERE the level is (landmark_route.py)
    play: bool = False             # --play: beat the level -- no training, best route + practice, brain past it
    frontier: bool = False         # --frontier: every other attempt replays to just before the furthest point, brain learns from there
    save_dir: str = "./checkpoints"
    checkpoint_file: str | None = None
    name: str = "preview_boxes"
    log_dir: str = "./logs_boxes"
    verbose: int = 1
    # --- DQN --------------------------------------------------------------
    n_step: int = 4
    buffer_size: int = 100_000
    batch_size: int = 128
    replay_ratio: float = 4.0      # gradient steps per new environment step
    max_grad_steps: int = 1000     # cap per update (bounds the pause length)
    pause_on_death: bool = True    # ESC-pause at every death until all after-death work is done
    warmup: int = 256
    target_every: int = 250        # hard target-net sync, in gradient steps
    eps_start: float = 0.15
    eps_end: float = 0.01
    eps_decay_steps: int = 20_000   # ~11 min of play
    explore_hold: float = 0.5      # P(hold) for a random action
    jump_margin: float = 0.3       # only hold SPACE when Q(hold) beats Q(release) by this much:
                                   # "don't jump until confident". Used when acting AND in the
                                   # learning target, so the two agree.
    # --- PPO --------------------------------------------------------------
    epochs: int = 10
    minibatch: int = 64
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    target_kl: float = 0.03
    init_hold_prob: float = 0.10


# =========================================================================== #
# Learners
# =========================================================================== #
class _QNet:
    """Dueling MLP Q-network (plain torch so checkpoints are tiny .pt files)."""

    @staticmethod
    def build(obs_dim: int, n_act: int, arch):
        import torch.nn as nn

        class Q(nn.Module):
            def __init__(self):
                super().__init__()
                layers, d = [], obs_dim
                for h in arch:
                    layers += [nn.Linear(d, h), nn.ReLU()]
                    d = h
                self.trunk = nn.Sequential(*layers)
                self.v = nn.Linear(d, 1)
                self.a = nn.Linear(d, n_act)

            def forward(self, x):
                z = self.trunk(x)
                a = self.a(z)
                return self.v(z) + a - a.mean(dim=1, keepdim=True)
        return Q()


class DQNLearner:
    def __init__(self, obs_dim: int, n_act: int, cfg: TrainConfig) -> None:
        import torch as th
        self.th, self.cfg = th, cfg
        self.dev = th.device(cfg.device)
        if self.dev.type == 'cpu' and cfg.cpu_threads > 0:
            th.set_num_threads(cfg.cpu_threads)
        self.obs_dim, self.n_act = obs_dim, n_act
        self.q = _QNet.build(obs_dim, n_act, cfg.net_arch).to(self.dev)
        with th.no_grad():  # fresh brain waits by default; jumps come from exploration
            try:   # spare inputs start disconnected, so filling them later changes nothing at first
                from env_boxes import OBS_DIM as _OD, RESERVED as _RS
                if obs_dim == _OD:
                    self.q.trunk[0].weight[:, _RS] = 0.0
            except Exception:
                pass
            self.q.a.weight.mul_(0.1)
            self.q.a.bias.copy_(th.tensor([0.25, -0.25][:n_act] + [0.0] * max(0, n_act - 2)))
        self.qt = _QNet.build(obs_dim, n_act, cfg.net_arch).to(self.dev)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = th.optim.Adam(self.q.parameters(), lr=cfg.lr)
        C = cfg.buffer_size
        self.b_obs = np.zeros((C, obs_dim), np.float32)
        self.b_nxt = np.zeros((C, obs_dim), np.float32)
        self.b_act = np.zeros(C, np.int64)
        self.b_ret = np.zeros(C, np.float32)
        self.b_disc = np.zeros(C, np.float32)   # gamma^m, 0 if terminal
        self.b_teacher = np.full(C,-1,np.int64)
        self.ptr = 0
        self.full = False
        self.grad_steps = 0
        self.rng = np.random.default_rng(0)
        self.steps_seen = 0
        self.last_stats: dict = {}

    # -- acting --------------------------------------------------------------
    def epsilon(self) -> float:
        c = self.cfg
        f = min(1.0, self.steps_seen / max(1, c.eps_decay_steps))
        return c.eps_start + (c.eps_end - c.eps_start) * f

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        th = self.th
        with th.no_grad():
            return self.q(th.as_tensor(obs[None], dtype=th.float32, device=self.dev))[0].cpu().numpy()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[int, dict]:
        qv = self.q_values(obs)
        if not deterministic:
            self.steps_seen += 1
            if self.rng.random() < self.epsilon():
                return int(self.rng.random() < self.cfg.explore_hold), {"q": qv}
        return self._greedy(qv, obs), {"q": qv}

    def _greedy(self, qv, obs=None) -> int:
        if len(qv) == 2:
            margin = self.cfg.flight_margin if obs is not None and bool(flight_mode_mask(obs)) else self.cfg.jump_margin
            return int(qv[1] - qv[0] > margin)
        return int(np.argmax(qv))

    def hold_prob(self, obs: np.ndarray) -> float:
        qv = self.q_values(obs)
        return float(1.0 / (1.0 + math.exp(-(qv[1] - qv[0]))))  # display only

    # -- data ----------------------------------------------------------------
    def size(self) -> int:
        return self.cfg.buffer_size if self.full else self.ptr

    def add_segment(self, obs: list, act: list, rew: list, next_obs_last: np.ndarray | None,
                    terminal: bool, upto: int | None = None, teacher=None) -> int:
        """Push n-step transitions for steps [0, upto) of a segment."""
        c = self.cfg
        T = len(act)
        upto = T if upto is None else min(upto, T)
        g = c.gamma
        for t in range(upto):
            m = min(c.n_step, T - t)
            R = 0.0
            for i in range(m):
                R += (g ** i) * rew[t + i]
            end = t + m
            if end == T and terminal:
                disc, nxt = 0.0, obs[t]          # next obs unused
            else:
                disc = g ** m
                nxt = obs[end] if end < T else next_obs_last
                if nxt is None:
                    continue
            i = self.ptr
            self.b_obs[i] = obs[t]; self.b_act[i] = act[t]; self.b_ret[i] = R
            self.b_disc[i] = disc; self.b_nxt[i] = nxt
            self.b_teacher[i] = int(teacher[t]) if teacher is not None else -1
            self.ptr = (self.ptr + 1) % c.buffer_size
            self.full = self.full or self.ptr == 0
        return upto

    # -- learning --------------------------------------------------------------
    def train(self, new_steps: int, tick=None) -> dict:
        th, c = self.th, self.cfg
        n = self.size()
        if n < c.warmup:
            return {}
        k = int(min(c.max_grad_steps, max(1, c.replay_ratio * new_steps)))
        losses, qs, guide_losses = [], [], []
        teacher_draws = 0
        self.q.train()
        for j in range(k):
            if tick is not None and j % 50 == 0:
                tick()   # e.g. pump the GUI window so Windows doesn't flag it "Not Responding"
            idx = self.rng.integers(0, n, size=c.batch_size)
            o = th.as_tensor(self.b_obs[idx], device=self.dev)
            a = th.as_tensor(self.b_act[idx], device=self.dev)
            R = th.as_tensor(self.b_ret[idx], device=self.dev)
            d = th.as_tensor(self.b_disc[idx], device=self.dev)
            nx = th.as_tensor(self.b_nxt[idx], device=self.dev)
            with th.no_grad():
                qn = self.q(nx)
                if qn.shape[1] == 2:   # same "confident jump" rule as when acting
                    margins = th.where(flight_mode_mask(nx), c.flight_margin, c.jump_margin)
                    a_star = ((qn[:, 1] - qn[:, 0]) > margins).long()[:, None]
                else:
                    a_star = qn.argmax(dim=1, keepdim=True)
                tq = self.qt(nx).gather(1, a_star).squeeze(1)
                y = R + d * tq
            values = self.q(o)
            q = values.gather(1, a[:, None]).squeeze(1)
            loss = th.nn.functional.smooth_l1_loss(q, y)
            if c.flight_teacher_weight > 0:
                labels = th.as_tensor(self.b_teacher[idx],device=self.dev)
                eligible = (labels >= 0) & flight_mode_mask(o)
                if bool(eligible.any()):
                    selected = labels[eligible]
                    logits = values[eligible]
                    advantage = (logits.gather(1,selected[:,None])-logits.gather(1,(1-selected)[:,None])).squeeze(1)
                    guide = th.relu(max(.2,c.flight_margin+.05)-advantage).mean()
                    loss = loss + c.flight_teacher_weight*guide
                    guide_losses.append(float(guide.detach().cpu()))
                    teacher_draws += int(eligible.sum())
            self.opt.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.q.parameters(), c.max_grad_norm)
            self.opt.step()
            self.grad_steps += 1
            if self.grad_steps % c.target_every == 0:
                self.qt.load_state_dict(self.q.state_dict())
            losses.append(float(loss.item()))
            qs.append(float(q.mean().item()))
        self.q.eval()
        self.last_stats = dict(grad_steps=k, loss=float(np.mean(losses)), q=float(np.mean(qs)),
                               eps=self.epsilon(), buffer=n,teacher_loss=float(np.mean(guide_losses)) if guide_losses else 0.,
                               teacher_draws=teacher_draws)
        return self.last_stats

    def save(self, path: str, extra: dict | None = None) -> None:
        th = self.th
        from env_boxes import OBS_VERSION
        th.save({"algo": "dqn", "obs_dim": self.obs_dim, "obs_version": OBS_VERSION,
                 "arch": list(self.cfg.net_arch), "q": self.q.state_dict(),
                 "qt": self.qt.state_dict(), "opt": self.opt.state_dict(),
                 "steps_seen": self.steps_seen, "grad_steps": self.grad_steps,
                 **(extra or {})}, path)

    def load(self, path: str) -> None:
        th = self.th
        ck = th.load(path, map_location=self.dev, weights_only=False)
        if 'route_controller_version' in ck:
            raise ValueError('This checkpoint requires --route-controller; it is not a legacy DQN checkpoint.')
        if ck.get("obs_dim") != self.obs_dim:
            raise SystemExit(
                f"\n{path} is a brain from an older version (it sees {ck.get('obs_dim')} numbers, "
                f"this version sees {self.obs_dim}: speed, icon shape, motion history, colours were added).\n"
                f"It can't be resumed -- start a fresh brain by running without --model.\n")
        self.q.load_state_dict(ck["q"])
        self.qt.load_state_dict(ck.get("qt", ck["q"]))
        try:
            self.opt.load_state_dict(ck["opt"])
        except Exception:
            pass
        self.steps_seen = int(ck.get("steps_seen", 0))
        self.grad_steps = int(ck.get("grad_steps", 0))


class PPOLearner:
    """On-policy PPO via SB3's ActorCriticPolicy (comparison / legacy)."""

    def __init__(self, obs_space, act_space, cfg: TrainConfig, model=None) -> None:
        import torch as th
        self.th, self.cfg = th, cfg
        if model is None:
            from stable_baselines3 import PPO
            import gymnasium as gym

            class _E(gym.Env):
                observation_space = obs_space
                action_space = act_space

                def reset(self, *, seed=None, options=None):
                    return obs_space.sample(), {}

                def step(self, a):
                    return obs_space.sample(), 0.0, True, False, {}
            model = PPO("MlpPolicy", _E(), learning_rate=cfg.lr, n_steps=cfg.update_steps,
                        batch_size=cfg.minibatch, gamma=cfg.gamma, device=cfg.device, verbose=0,
                        policy_kwargs=dict(net_arch=dict(pi=list(cfg.net_arch), vf=list(cfg.net_arch)),
                                           activation_fn=th.nn.ReLU))
            p = min(max(cfg.init_hold_prob, 1e-3), 0.5)
            with th.no_grad():
                model.policy.action_net.bias.copy_(th.tensor([0.0, math.log(p / (1 - p))]))
        self.model = model
        self.policy = model.policy
        self.opt = self.policy.optimizer
        for g in self.opt.param_groups:
            g["lr"] = cfg.lr
        self.segs: list = []
        self.last_stats: dict = {}

    def _t(self, obs):
        th = self.th
        return th.as_tensor(obs[None], dtype=th.float32, device=self.model.device)

    def act(self, obs, deterministic=False):
        with self.th.no_grad():
            o = self._t(obs)
            dist = self.policy.get_distribution(o)
            a = dist.mode() if deterministic else dist.sample()
            return int(a.item()), {"logp": float(dist.log_prob(a).item()),
                                   "v": float(self.policy.predict_values(o).item())}

    def value(self, obs) -> float:
        with self.th.no_grad():
            return float(self.policy.predict_values(self._t(obs)).item())

    def hold_prob(self, obs) -> float:
        with self.th.no_grad():
            return float(self.policy.get_distribution(self._t(obs)).distribution.probs[0, 1].item())

    def size(self) -> int:
        return sum(len(s[1]) for s in self.segs)

    def add_segment(self, obs, act, rew, next_obs_last, terminal, logp, val):
        boot = 0.0 if terminal else self.value(next_obs_last)
        self.segs.append((obs, act, rew, logp, val, terminal, boot))

    def train(self, new_steps: int, tick=None) -> dict:
        th, cfg = self.th, self.cfg
        if not self.segs:
            return {}
        O, A, LP, ADV, RET = [], [], [], [], []
        for obs, act, rew, logp, val, terminal, boot in self.segs:
            r = np.asarray(rew, np.float32); v = np.asarray(val, np.float32)
            T = len(r); adv = np.zeros(T, np.float32); gae = 0.0
            for t in reversed(range(T)):
                last = t == T - 1
                nv = (0.0 if terminal else boot) if last else v[t + 1]
                nt = 0.0 if (last and terminal) else 1.0
                delta = r[t] + cfg.gamma * nv * nt - v[t]
                gae = delta + cfg.gamma * cfg.lam * nt * gae
                adv[t] = gae
            O.append(np.asarray(obs, np.float32)); A.append(np.asarray(act, np.int64))
            LP.append(np.asarray(logp, np.float32)); ADV.append(adv); RET.append(adv + v)
        self.segs = []
        dev = self.model.device
        O, A, LP, ADV, RET = (th.as_tensor(np.concatenate(x), device=dev) for x in (O, A, LP, ADV, RET))
        n = len(A)
        self.policy.set_training_mode(True)
        kls, ents = [], []
        stop = False
        for _ in range(cfg.epochs):
            perm = th.randperm(n, device=dev)
            for i in range(0, n, cfg.minibatch):
                idx = perm[i:i + cfg.minibatch]
                if len(idx) < 8:
                    continue
                values, new_lp, ent = self.policy.evaluate_actions(O[idx], A[idx])
                adv = ADV[idx]; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = th.exp(new_lp - LP[idx])
                pg = -th.min(adv * ratio, adv * th.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)).mean()
                vl = th.nn.functional.smooth_l1_loss(values.flatten(), RET[idx])
                loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent.mean()
                with th.no_grad():
                    lr_ = new_lp - LP[idx]
                    kl = float(((th.exp(lr_) - 1) - lr_).mean().item())
                kls.append(kl)
                if cfg.target_kl and kl > 1.5 * cfg.target_kl:
                    stop = True
                    break
                self.opt.zero_grad(); loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.opt.step()
                ents.append(float(ent.mean().item()))
            if stop:
                break
        self.policy.set_training_mode(False)
        self.last_stats = dict(samples=n, kl=float(np.mean(kls or [0])), entropy=float(np.mean(ents or [0])))
        return self.last_stats

    def save(self, path: str, extra: dict | None = None) -> None:
        self.model.save(path)


# =========================================================================== #
# Trainer loop
# =========================================================================== #
@dataclass
class _Seg:
    obs: list = field(default_factory=list)
    act: list = field(default_factory=list)
    rew: list = field(default_factory=list)
    logp: list = field(default_factory=list)
    val: list = field(default_factory=list)
    potential: list = field(default_factory=list)
    next_potential: list = field(default_factory=list)
    shaping: list = field(default_factory=list)
    teacher: list = field(default_factory=list)
    route_records: list = field(default_factory=list)
    discounts: list = field(default_factory=list)
    completed: bool = False

    def __len__(self) -> int:
        return len(self.act)

    def trim(self, n: int) -> None:
        if n <= 0:
            return
        k = max(0, len(self) - n)
        for name in ("obs", "act", "rew", "logp", "val", "potential", "next_potential", "shaping", "teacher", "route_records", "discounts"):
            lst = getattr(self, name)
            if lst:
                setattr(self, name, lst[:k])

    def drop_front(self, k: int) -> None:
        for name in ("obs", "act", "rew", "logp", "val", "potential", "next_potential", "shaping", "teacher", "route_records", "discounts"):
            lst = getattr(self, name)
            if lst:
                setattr(self, name, lst[k:])


def checkpoint_path(cfg: TrainConfig, best: bool = False) -> str:
    if cfg.checkpoint_file:
        if not best:
            return cfg.checkpoint_file
        stem, extension = os.path.splitext(cfg.checkpoint_file)
        return stem + "_best" + extension
    base = os.path.join(cfg.save_dir, ("best_boxes" if cfg.name == "preview_boxes" else cfg.name + "_best") if best else cfg.name)
    return base + (".pt" if cfg.algo == "dqn" else "")


def make_learner(env, cfg: TrainConfig, model_path: str | None = None):
    if cfg.algo == "dqn":
        if cfg.route_controller:
            from route_learner import RouteLearner
            lr=RouteLearner(int(np.prod(env.observation_space.shape)),int(env.action_space.n),cfg)
        else:
            lr = DQNLearner(int(np.prod(env.observation_space.shape)), int(env.action_space.n), cfg)
        if model_path:
            lr.load(model_path)
            print(f"resumed {model_path}")
        return lr
    model = None
    if model_path:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device=cfg.device)
        if tuple(model.observation_space.shape) != tuple(env.observation_space.shape):
            raise ValueError(f"{model_path} has obs shape {model.observation_space.shape}; this "
                             f"version uses {env.observation_space.shape}. Start fresh (drop --model).")
    return PPOLearner(env.observation_space, env.action_space, cfg, model)


def replay_shield(obs: np.ndarray, a: int) -> int:
    """The safety net over a REPLAYED run: a lighter touch than safety_shield. The run being
    replayed got past here, so only step in when the replayed action crashes even as
    measured and the other one is safe even with worse timing (the full shield also swapped "fine, a bit tight"
    actions; in the sim that changed replays that would have worked and killed them)."""
    try:
        from env_boxes import OFF_SEARCH, OBS_DIM
        from gd_physics import safety_level
    except Exception:
        return a
    if obs is None or len(obs) < OBS_DIM or obs[OFF_SEARCH + 3] <= 0.5:
        return a
    p1, w00, w01 = (float(v) for v in obs[OFF_SEARCH:OFF_SEARCH + 3])
    mine, other = (safety_level(p1), max(safety_level(w00), safety_level(w01))) if int(a) == 1 else \
        (max(safety_level(w00), safety_level(w01)), safety_level(p1))
    return 1 - int(a) if (mine == 0 and other >= 2) else int(a)


def replay_guard(obs: np.ndarray, a: int) -> int:
    """The safety net over a replayed run: the full shield's protection (take the clearly safer
    first action; both doomed -> buy time) WITHOUT its "press now, the next step is too tight"
    rule. That rule hedges a late-pressing brain; on a replay of a run that already cleared the
    spot it moves the jump a decision EARLY -- live (050917) the replay of the 24% run kept landing
    short of the 9% pillar (press 0.95 / wait 0.94, shield pressed early). The light replay_shield
    let presses rated 0.27 vs wait 0.96 through (11 of 16 attempts died at 1%)."""
    try:
        from env_boxes import OFF_SEARCH, OBS_DIM
        from gd_physics import safety_level
    except Exception:
        return a
    if obs is None or len(obs) < OBS_DIM or obs[OFF_SEARCH + 3] <= 0.5:
        return safety_shield(obs, a)
    p1, w00, w01 = (float(v) for v in obs[OFF_SEARCH:OFF_SEARCH + 3])
    L1, Lw = safety_level(p1), max(safety_level(w00), safety_level(w01))
    if int(a) == 1 and L1 < Lw:
        return 0
    if int(a) == 0 and Lw < L1:
        return 1
    if L1 == 0 and Lw == 0:
        mine, other = (p1, max(w00, w01)) if int(a) == 1 else (max(w00, w01), p1)
        if other > mine + 0.05:
            return 1 - int(a)
    return int(a)


EARLY_PRESS = True   # --no-early-press turns off the "next step is too tight: press now" rule
# --trust-replay: while the landmarks are LOCKED, a replayed press/wait goes out as remembered --
# the safety shield doesn't edit it (it still guards the brain, and the replay while unlocked).
# Live (004233, a human run's presses as the route) the shield rewrote the replay ~60 times an
# attempt -- ~2 a second through the first 20 s, adding presses the human run never made -- and
# 14 of 16 deaths came within 3 s of it doing so.
TRUST_REPLAY = False
TRUST_REPLAY_AB = False   # --trust-replay-ab: every other attempt trusted, the rest shielded (a fair test)
# --guide [margin]: the BRAIN plays and the remembered route only guides it. Where the brain and the
# route disagree, the brain's choice goes out when it clearly prefers it (its two Q values differ by
# >= margin and the shield didn't have to correct it); otherwise the route's. Live (deaths of
# 230811/004233/013303) |Q1-Q0| sits at ~3.3 when the brain doesn't care either way (half of all
# decisions, and the median where replay and brain disagreed) and 12-50 when it does.
GUIDE_MARGIN = None
# --play explores at the frontier: from 1.5 s before the best run's end to 20 blocks past it, each
# decision the brain makes is swapped for the other action with this probability -- only when the
# planner rates the other action at least as safe. Live (050624): 6 of 12 attempts died at the
# exact same spot past the end of the route, the brain (deterministic in --play) repeating the
# same moves every time; the old nudge flipped ONE decision. 0 = off (--no-frontier-explore).
FRONTIER_EXPLORE = 0.15   # (no longer used: see FRONTIER_EVERY)
# Live (052046): 15% random swaps (7-15 per attempt) still died at the same 43% every time. Now
# every FRONTIER_EVERY-th attempt CHANGES the brain's decisions there on purpose, cycling through
# kinds of change so those attempts differ from each other too; the other attempts play as usual.
FRONTIER_EVERY = 3
FRONTIER_PATTERNS = ("swap every decision", "presses become waits", "waits become presses",
                     "swap every 2nd decision")


def safety_shield(obs: np.ndarray, a: int) -> int:
    """Use the physics preview as a safety net.

    The preview simulated 8 short plans; some start by pressing now, the rest by
    not pressing. If every plan that starts with action `a` crashes, but a plan
    starting with the other action survives (or survives clearly longer), take
    the other action. When both are fine -- or both doomed -- the brain decides.
    Works the same for random (exploring) and chosen actions, and the brain
    learns from the action actually taken.
    """
    try:
        from env_boxes import OFF_PLAN, OFF_RES, OFF_SEARCH, OBS_DIM
        from gd_physics import EXTRA_PLANS, N_PLANS, PLANS
    except Exception:
        return a
    if obs is None or len(obs) < OBS_DIM:
        return a
    if obs[OFF_SEARCH + 3] > 0.5:
        # The search (every later decision played well, checked in a slightly worse world
        # too). Replaying live deaths: the 8 fixed plans called most steps before a crash
        # "doomed either way" (they can't jump twice), so this rule had nothing to go on.
        from gd_physics import safety_level
        p1, w00, w01 = (float(v) for v in obs[OFF_SEARCH:OFF_SEARCH + 3])
        L1, L00, L01 = safety_level(p1), safety_level(w00), safety_level(w01)
        Lw = max(L00, L01)
        # levels: 2 = survives even if the timing is a bit worse than measured, 1 = only as
        # measured, 0 = crashes. Take the safer first action; same level -> the brain decides
        if int(a) == 1 and L1 < Lw:
            return 0                        # pressing now is less safe than waiting
        if int(a) == 0 and Lw < L1:
            return 1                        # only pressing now is that safe
        if EARLY_PRESS and int(a) == 0 and L00 < L01 == Lw and L1 >= Lw and Lw > 0:
            return 1                        # waiting only works by pressing at the very next step:
                                            # timing that tight flips with jitter -- press now
        if L1 == 0 and Lw == 0:             # doomed both ways (as far as we can see): buy time
            mine, other = (p1, max(w00, w01)) if int(a) == 1 else (max(w00, w01), p1)
            if other > mine + 0.05:
                return 1 - int(a)
        return int(a)
    f = obs[OFF_PLAN:OFF_PLAN + 3 * N_PLANS]
    if not np.any(f):                       # no preview this step (icon not found)
        return a
    fx = obs[OFF_RES:OFF_RES + 3 * len(EXTRA_PLANS)]   # "tap in 1" / "tap in 3" (spare inputs)
    best = {0: 0.0, 1: 0.0}                 # longest survival (1 = whole horizon) per first action
    plans = [(plan, f[3 * pi:3 * pi + 3]) for pi, (_n, plan) in enumerate(PLANS)]
    if np.any(fx):
        plans += [(plan, fx[3 * pi:3 * pi + 3]) for pi, (_n, plan) in enumerate(EXTRA_PLANS)]
    slack_ok = False                        # a release-first plan that survives AND doesn't
    for plan, f3 in plans:                  # need a press within the next step
        first = int(bool(plan(0)))
        surv = 1.0 if f3[0] < 0.5 else float(f3[1])
        best[first] = max(best[first], surv)
        if first == 0 and surv >= 1.0 and not plan(1):
            slack_ok = True
    if int(a) == 0 and best[0] >= 1.0 and not slack_ok and best[1] >= 1.0:
        # waiting is only safe if it presses in EXACTLY one step; pressing now is safe too.
        # Tight timings flip between steps (step-time jitter, input delay): press now.
        # (live deaths: "press in 1" safe -> next step nothing safe)
        return 1
    mine, other = best[int(a)], best[1 - int(a)]
    if mine < 1.0 and other >= 1.0:
        return 1 - int(a)                   # only the other action has a path that survives
    if mine < 0.5 and other > mine + 0.3:
        return 1 - int(a)                   # both crash eventually, the other buys much more time
    return int(a)


class Trainer:
    def __init__(self, env, cfg: TrainConfig | None = None, learner=None, model_path: str | None = None,
                 on_step: Callable[..., Any] | None = None,
                 on_attempt: Callable[..., Any] | None = None) -> None:
        self.env = env
        self.cfg = cfg or TrainConfig()
        clearance_weight(self.cfg.clearance_weight)
        if not math.isfinite(self.cfg.flight_buffer) or not 0 <= self.cfg.flight_buffer <= 2:
            raise ValueError('flight_buffer must be between 0 and 2 blocks')
        if not math.isfinite(self.cfg.flight_margin):
            raise ValueError('flight_margin must be finite')
        if not math.isfinite(self.cfg.flight_teacher_weight) or not 0 <= self.cfg.flight_teacher_weight <= .2:
            raise ValueError('flight_teacher_weight must be between 0 and .2')
        if not 0 <= self.cfg.gamma <= 1:
            raise ValueError('gamma must be between zero and one')
        self.env.clearance_enabled = self.cfg.clearance_weight > 0 and not self.cfg.play
        self.env.flight_buffer = self.cfg.flight_buffer
        self.flight_shield = FlightShield()
        self.replay_flight_shield = FlightShield()
        self.learner = learner if learner is not None else make_learner(env, self.cfg, model_path)
        if self.cfg.route_controller and model_path and not getattr(self.learner, 'imported_legacy', False):
            destination = os.path.abspath(os.fspath(model_path))
            self.cfg = replace(self.cfg, checkpoint_file=destination,
                               save_dir=os.path.dirname(destination))
        if self.cfg.route_controller:
            print(f'[controller] live updates save to {checkpoint_path(self.cfg)}')
        self.on_step = on_step
        self.on_attempt = on_attempt
        self.on_train_tick = None     # called every 50 gradient steps during an update
        self._tm = {"brain": 0.0, "env": 0.0, "gui": 0.0}   # seconds since the last [health] line
        self.shield_n = 0             # actions the safety shield changed since the last update
        self.shield_steps = 0
        # Shield-vs-brain disagreement evidence (route controller only).
        # Attribution is indirect: the shield's action is executed, so a
        # death <=0.5s after an override scores policy_right, survival
        # >=1.0s scores shield_right, anything between is inconclusive.
        self._shield_pending = []     # overrides awaiting outcome attribution
        self._shield_stats = {}       # mode -> [overrides, policy_right, shield_right]
        self._shield_off_modes = set()  # trusted modes where evidence disabled the shield
        self._step_t0 = time.perf_counter()
        self._step_n0 = 0
        self.best_prog = -1.0
        if model_path:
            # keep the record across runs, so the first attempt of a new run
            # doesn't overwrite best_boxes with a worse brain
            try:
                import torch as _th
                bp = checkpoint_path(self.cfg, best=True)
                if os.path.exists(bp) and self.cfg.algo == "dqn":
                    self.best_prog = float(_th.load(bp, map_location="cpu", weights_only=False).get("best_pct", -100)) / 100
            except Exception:
                pass
        self.steps = 0
        self.updates = 0
        self.attempts = 0
        self.best_steps = -1
        self.recent: list = []           # (survived_steps, peak_prog), last 50 attempts
        self.new_steps = 0               # env steps since the last update
        self.pause_s = 0.0               # wall time spent paused for training
        self._pause_depth = 0            # nested game pauses (death bookkeeping > training update)
        try:
            # --play saves no checkpoints after a death (the snapshot is written in the
            # background): no reason to stop the game there
            self.env.pause_on_death = bool(self.cfg.pause_on_death) and not self.cfg.play
        except Exception:
            pass
        self.stop = False
        self.last_q = None
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        self.route = None
        if self.cfg.route:
            from route import RouteMemory
            self.route = RouteMemory(self.cfg.route_dir, dt=getattr(env, "frame_skip", 2) / 60.0,
                                     verbose=bool(self.cfg.verbose), fixed_level=self.cfg.route_level,
                                     play=bool(self.cfg.play), frontier=bool(self.cfg.frontier))
        self.lm = None                   # landmark-triggered replay (landmark_route.py); rides on the route memory
        self._lm_overrides = 0
        if self.cfg.landmarks:
            if self.route is None:
                print("[landmarks] needs the route memory (don't use --no-route): landmarks are OFF")
            else:
                from landmark_route import LandmarkRoute
                self.lm = LandmarkRoute(self.cfg.route_dir, verbose=bool(self.cfg.verbose))
                print(f"[landmarks] ON: the best run's presses are replayed by level position ({self.cfg.route_dir}/<level>/landmarks.json)")
        self._att_k = 0                  # decisions since the attempt started (clock fallback)
        self._clock_ok = False           # the first attempt after starting isn't timed from a respawn
        self._att_ages: list = []        # this attempt: how old each observation was when its key went out
        self._att_times: list = []
        self.session = SessionLog(self.cfg.log_dir, self.cfg, model_path)
        self._csv_path = os.path.join(self.session.directory, 'attempts.csv')
        self._csv, self._w = self.session.file, self.session
        if hasattr(self.env, 'death_dir') and self.env.death_dir:
            self.env.death_dir = os.path.join(self.session.directory, 'deaths')
        self.session.event('environment', dict(backend=type(getattr(self.env, 'det', None)).__name__,
                           frame_skip=getattr(self.env, 'frame_skip', None),
                           fast_steps=getattr(self.env, 'fast_steps', None),
                           capture_prefetch=getattr(getattr(getattr(getattr(self.env,'base',None),'cap',None),'cfg',None),'prefetch',None),
                           motion_fix=getattr(self.env, 'use_motion_fix', None)))
        detector = getattr(getattr(self.env, 'det', None), 'yolo', None)
        path = getattr(detector, 'loaded_path', None)
        if path and os.path.isfile(path):
            try:
                self.session.event('detector', dict(path=path, sha256=file_sha256(path),
                                   modified=os.path.getmtime(path), engine=bool(detector.engine)))
            except Exception as exc:
                print(f'[session] detector fingerprint unavailable: {type(exc).__name__}: {exc}; continuing')
        print(f'[session] {self.session.directory}')

    # -- helpers ------------------------------------------------------------
    def _clearance_potential(self) -> float:
        if self.cfg.clearance_weight <= 0 or self.cfg.play:
            return 0.
        sample = getattr(self.env, 'last_clearance', None) or {}
        if not sample.get('valid') or sample.get('mode') not in ('ship', 'wave'):
            return 0.
        score = sample.get('score', 0.)
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            return 0.
        return self.cfg.clearance_weight * min(1., max(0., score))

    # -- shield disagreement evidence -------------------------------------
    def _shield_mode(self, obs):
        """Mode name for an observation, or None (logging must never crash play)."""
        try:
            from route_controller import MODES as _CM, mode_indices
            return _CM[int(mode_indices(obs))]
        except Exception:
            try:
                me = getattr(self.env, 'mode_est', None)
                return getattr(me, 'mode', None)
            except Exception:
                return None

    def _shield_trusted(self, mode):
        learner = self.learner
        modes = getattr(learner, 'trusted_modes', set()) or set()
        try:
            from route_controller import MODES as _CM
            idx = _CM.index(mode) if mode in _CM else None
        except Exception:
            idx = None
        return mode in modes or idx in modes

    def _shield_note_override(self, mode, brain_action, applied_action, role):
        now = time.time()
        self._shield_pending.append(dict(mode=mode, t=now, attempt=self.attempts + 1,
                                         brain=int(brain_action), applied=int(applied_action),
                                         role=role))
        st = self._shield_stats.setdefault(mode, [0, 0, 0])
        st[0] += 1
        if self.cfg.shield_log and self.cfg.route_controller:
            try:
                self.session.event('shield_disagreement', dict(kind='override', mode=mode,
                    brain=int(brain_action), applied=int(applied_action), role=role,
                    attempt=self.attempts + 1))
            except Exception:
                pass

    def _shield_attribute(self, died: bool, completed: bool = False):
        """Score pending overrides against the just-observed outcome."""
        if not self._shield_pending:
            return
        now = time.time()
        keep = []
        for rec in self._shield_pending:
            age = now - rec['t']
            # Replay-shield overrides protect a replay a frame off: a death right
            # after one says nothing about the brain/policy -- don't score it.
            if rec.get('role') in ('replay', 'route') and died and age <= 1.0:
                continue
            if died and age <= 0.5:
                verdict = 'policy_right'
            elif age >= 1.0 or completed:
                verdict = 'shield_right'
            elif died:
                verdict = 'inconclusive'
            else:
                keep.append(rec)
                continue
            st = self._shield_stats.setdefault(rec['mode'], [0, 0, 0])
            if verdict == 'policy_right':
                st[1] += 1
            elif verdict == 'shield_right':
                st[2] += 1
            if self.cfg.shield_log and self.cfg.route_controller:
                try:
                    self.session.event('shield_disagreement', dict(kind='outcome', mode=rec['mode'],
                        verdict=verdict, age_s=round(age, 3), brain=rec['brain'],
                        applied=rec['applied'], attempt=rec['attempt']))
                except Exception:
                    pass
            self._shield_maybe_disable(rec['mode'])
        self._shield_pending = keep if not died else []

    def _shield_maybe_disable(self, mode):
        if mode in self._shield_off_modes or not self.cfg.shield_auto_off:
            return
        n, pol, shi = self._shield_stats.get(mode, [0, 0, 0])
        if n >= max(1, self.cfg.shield_min_overrides) and pol > shi and self._shield_trusted(mode):
            self._shield_off_modes.add(mode)
            if self.cfg.verbose:
                print(f'[shield] AUTO-OFF for trusted mode {mode}: {n} overrides, '
                      f'policy-right {pol} > shield-right {shi}. '
                      f'Re-enable any time with --shield (shield stays on for other modes).')
            try:
                self.session.event('shield_disagreement', dict(kind='auto_off', mode=mode,
                    overrides=n, policy_right=pol, shield_right=shi))
            except Exception:
                pass

    def act(self, obs, deterministic: bool = False):
        if self.cfg.route_controller:self.learner.set_context(self.env)
        # Lossless hot-path trim: the motion forecast costs an extra network
        # forward per decision. Skip it only when no consumer can read it
        # (no route controller, no death snapshots, no step callback such as
        # the preview overlay). Otherwise keep the exact old behaviour.
        _env = getattr(self, "env", None)
        _need_pred = bool(self.cfg.route_controller) or bool(getattr(_env, "death_dir", None)) or (self.on_step is not None)
        try:
            a, extra = self.learner.act(obs, deterministic, with_prediction=_need_pred)
        except TypeError:
            a, extra = self.learner.act(obs, deterministic)
        self.last_q = extra.get("q")
        a_brain = a
        label = -1
        _mode = self._shield_mode(obs) if self.cfg.route_controller else None
        _shield_on = self.cfg.shield and not (_mode is not None and _mode in self._shield_off_modes)
        if _shield_on and self.cfg.algo == "dqn":
            a2 = safety_shield(obs, a)
            metadata = (getattr(self.env,'_last_search',None) or {}).get('flight')
            if self.cfg.flight_buffer > 0 and metadata and len(obs) > 3 and obs[3] > .5 and bool(flight_mode_mask(obs)):
                # Use the raw brain choice: the flight selector already applies
                # collision priority before considering clearance or hysteresis.
                flight_choice = self.flight_shield.choose(a, metadata)
                if self.flight_shield.last is not None:
                    a2 = flight_choice
                    counts = getattr(getattr(self.env,'mode_cal',None),'counts',lambda:{})()
                    params = ('ship_up','ship_down','ship_vmax') if metadata['mode'] == 'ship' else ('wave_slope',)
                    verified = (all(counts.get(k,0) >= 3 for k in params)
                                and bool(getattr(getattr(self.env,'mode_est',None),'lag_locked',False))
                                and bool(getattr(self.env,'_calibration_valid',False)))
                    if self.cfg.flight_teacher_weight > 0 and not self.cfg.play:
                        label = teacher_action(metadata,self.flight_shield.last,verified)
            else:
                self.flight_shield.reset()
            if a2 != a:
                self.shield_n += 1
                extra = dict(extra, shielded=True)
                if self.cfg.route_controller:
                    self._shield_note_override(_mode, a_brain, a2,
                                                extra.get('controller_role', '') or 'brain')
                a = a2
        self.shield_steps += 1
        if self.cfg.route_controller:
            label=self.learner.planner_label()
            extra=dict(extra,route_record=dict(self.learner.context))
            if self.learner.evaluating:self.learner.trial.note_action(int(a)!=int(a_brain))
        extra = dict(extra,flight_teacher=label)
        try:    # for the death snapshots: what the brain wanted vs what was sent
            q = self.last_q
            self.env._dbg = {"a_brain": int(a_brain), "shielded": int(a) != int(a_brain), "attempt": self.attempts + 1,
                             "qdiff": float(q[1] - q[0]) if q is not None and len(q) == 2 else None,
                             "flight_control": self.flight_shield.last}
            self.env._dbg['flight_teacher'] = label
            self.env._dbg['controller_role']=extra.get('controller_role','legacy')
            if self.cfg.route_controller:self.env._dbg['motion_prediction']=extra.get('motion_prediction')
        except Exception:
            pass
        return a, extra

    def hold_prob(self, obs) -> float:
        return self.learner.hold_prob(obs)

    def _attempt_t(self) -> float:
        """Seconds into the current attempt: the env's frame clock if it has one, else
        decisions x nominal step length."""
        f = getattr(self.env, "attempt_time", None)
        t = f() if callable(f) else None
        if t is None:
            t = self._att_k * getattr(self.env, "frame_skip", 2) / 60.0
        return float(t)

    def save(self, best: bool = False) -> None:
        if best and self.cfg.route_controller and not self.learner.promotion_pending:return
        if self.cfg.route_controller and not best:
            # A live trial is starting: the checkpoint on disk still holds the brain in use --
            # keep a copy, since a promotion overwrites it (and *_best) for good
            ev = bool(getattr(self.learner, "evaluating", False))
            if ev and not getattr(self, "_trial_backed_up", False):
                src = checkpoint_path(self.cfg)
                if os.path.exists(src):
                    try:
                        import shutil
                        stem, ext = os.path.splitext(src)
                        shutil.copy2(src, stem + "_before_trial" + ext)
                        print(f"[controller evaluation] live trial starting -- the current brain is kept in "
                              f"{os.path.basename(stem + '_before_trial' + ext)} (copy it over the checkpoint to undo a promotion)")
                    except OSError as ex:
                        print(f"[controller evaluation] couldn't back up {src}: {ex!r}")
                self._trial_backed_up = True
            elif not ev:
                self._trial_backed_up = False
        try:
            self.learner.save(checkpoint_path(self.cfg, best),
                              {"attempts": self.attempts, "best_pct": self.best_prog * 100,
                               "reward_config": {k: getattr(self.cfg,k) for k in
                                   ('clearance_weight','hold_cost','gamma','alive_reward','death_penalty','flight_margin','flight_buffer','flight_teacher_weight')}})
            if best and self.cfg.route_controller:self.learner.promotion_pending=False
        except Exception as e:
            print(f"save failed: {e}")

    def _pause_game(self) -> None:
        """Nested pause: only the outermost pause / resume talks to the game (a training update
        inside the after-death bookkeeping must not un-pause it halfway)."""
        if self._pause_depth == 0:
            try:
                self.env.pause()
            except Exception as e:
                print(f"PAUSE FAILED: {e}")
        self._pause_depth += 1

    def _resume_game(self) -> None:
        self._pause_depth = max(0, self._pause_depth - 1)
        if self._pause_depth == 0:
            try:
                self.env.resume()
            except Exception as e:
                print(f"RESUME FAILED: {e}")

    def _push(self, seg: _Seg, next_obs, terminal: bool, upto: int | None = None) -> int:
        if self.cfg.algo == "dqn":
            if self.cfg.route_controller:
                return self.learner.add_segment(seg.obs,seg.act,seg.rew,next_obs,terminal,upto,
                                               teacher=seg.teacher,route_records=seg.route_records,completed=seg.completed)
            if seg.teacher and any(t >= 0 for t in seg.teacher):
                return self.learner.add_segment(seg.obs,seg.act,seg.rew,next_obs,terminal,upto,teacher=seg.teacher)
            return self.learner.add_segment(seg.obs, seg.act, seg.rew, next_obs, terminal, upto)
        self.learner.add_segment(seg.obs, seg.act, seg.rew, next_obs, terminal, seg.logp, seg.val)
        return len(seg)

    def _do_update(self, why: str) -> None:
        t0 = time.perf_counter()
        self._pause_game()
        st = self.learner.train(self.new_steps, tick=self.on_train_tick)
        if self.cfg.route_controller and self.learner.evaluating and why=='long attempt':
            self.learner.trial_skip_until_reset=True
        self.new_steps = 0
        if st:
            self.updates += 1
        self.save()
        if self.cfg.route_controller:
            self.session.event('controller_evaluation',self.learner.last_evaluation)
            if self.learner.promotion_pending:self.save(best=True)
        try:
            self.env.mark_trained()
        except Exception:
            pass
        self._resume_game()
        dt = time.perf_counter() - t0
        self.pause_s += dt
        if self.cfg.verbose and st:
            self._health(dt)
        if self.cfg.verbose and st:
            mean_s = np.mean([r[0] for r in self.recent[-20:]]) if self.recent else 0
            extra = "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in st.items())
            sh = (f"  shield saved {self.shield_n}x ({100 * self.shield_n / max(1, self.shield_steps):.1f}% of steps)"
                  if self.cfg.shield and self.cfg.algo == "dqn" else "")
            print(f"[update {self.updates}] {why}: {extra}  avg survive (last 20)={mean_s / 30:.1f}s  "
                  f"(paused {dt:.2f}s){sh}")
            self.shield_n = self.shield_steps = 0

    def _health(self, pause_dt: float) -> None:
        """One status line per update: if the run degrades, this shows what grew."""
        now = time.perf_counter()
        n = self.steps - self._step_n0
        play = max(1e-6, now - self._step_t0 - pause_dt)
        ms = 1000.0 * play / max(1, n)
        self._step_t0, self._step_n0 = now, self.steps
        mem = ""
        try:
            import sys as _sys
            _ps = _sys.modules.get("psutil")
            if _ps is None:
                import psutil as _ps
            _proc = getattr(_ps, "_zoo_proc", None)
            if _proc is None:
                _proc = _ps.Process()
                _ps._zoo_proc = _proc
            mem = f"  RAM {_proc.memory_info().rss / 1e6:.0f} MB"
        except Exception:
            pass
        cap = ""
        try:
            h = self.env.base.cap.health
            cap = f"  capture restarts {h['restarts']} overruns {h['overruns']}"
        except Exception:
            pass
        slow = "  <-- SLOW (target ~33 ms)" if ms > 45 else ""
        # where the time went, per decision (game = capture + waiting for the next frame
        # + reading the %: the part that's NOT our code; ~33 ms total is the target)
        parts = ""
        components = {}
        try:
            k = 1000.0 / max(1, n)
            et = getattr(self.env, "timing", None) or {}
            det, phy = et.get("detect", 0.0) * k, et.get("physics", 0.0) * k
            game = max(0.0, self._tm["env"] * k - det - phy)
            wait = None
            try:
                cap_ = self.env.base.cap
                w_now = float(getattr(cap_, "wait_s", 0.0))
                wait = (w_now - getattr(self, "_wait0", w_now)) * k
                self._wait0 = w_now
            except Exception:
                pass
            # game/capture split: WAITING for the next game frame (idle -- shrinks by itself when
            # the rest gets faster) vs real work (grabbing / shrinking frames, % read, death check)
            gtxt = (f"game/capture {game:.1f} (waiting {min(wait, game):.1f}, work {max(0.0, game - wait):.1f})"
                    if wait is not None else f"game/capture {game:.1f}")
            parts = (f"  = detect {det:.1f} + physics {phy:.1f} + brain {self._tm['brain'] * k:.1f}"
                     f" + gui {self._tm['gui'] * k:.1f} + {gtxt}")
            components = dict(detect_ms=det, physics_ms=phy, brain_ms=self._tm['brain'] * k,
                              gui_ms=self._tm['gui'] * k, game_capture_ms=game,
                              capture_wait_ms=wait,
                              game_work_ms=max(0., game - wait) if wait is not None else None)
            for key in et:
                et[key] = 0.0
        except Exception:
            pass
        for key in self._tm:
            self._tm[key] = 0.0
        warn = ""
        if getattr(self.env, "presses_working", True) is False:
            warn = "  <-- PRESSES AREN'T REACHING GD (click into the game window)"
        fpd = getattr(self.env, "frames_per_dec", None)
        fpd = f"  ({fpd:.1f} game frames/decision, target {getattr(self.env, 'frame_skip', 2)})" if fpd else ""
        print(f"[health] {ms:.1f} ms/decision{fpd}{parts}{mem}{cap}{slow}{warn}")
        profile = getattr(self.env, 'profile', None)
        stages = profile.summary() if profile is not None else {}
        self.session.event('timing', dict(steps=self.steps, decision_mean_ms=ms, stages=stages, components=components))
        if stages:
            print('[timing] ' + ', '.join(f"{k} {v['mean_ms']:.1f}/{v['p95_ms']:.1f} ms mean/p95" for k, v in stages.items()))

    def _note_cause(self, info: dict) -> None:
        """Tally how it died (death_cause.py); a summary every 25 deaths."""
        d = info.get("death_cause_d")
        if not d:
            return
        if not hasattr(self, "_causes"):
            from death_cause import Tally
            self._causes = Tally(50)
            self._n_cause = 0
        self._causes.add(d)
        self._n_cause += 1
        if self._n_cause % 25 == 0:
            print(f"[deaths] how the last {len(self._causes.q)} died: {self._causes.summary()}")

    def _log_route_attempt(self, seg: _Seg, info: dict, peak: float, tag: str = "route") -> None:
        """A route-memory practice attempt (or any --play attempt): counted and logged, never
        learned from."""
        self.attempts += 1
        lag = int(info.get("death_lag", 0)) if self.cfg.trim_death else 0
        survived = max(0, int(info.get("attempt_total", len(seg))))
        self._note_cause(info)
        self._w.writerow([f"{time.time():.1f}", self.attempts, self.steps, survived,
                          f"{peak * 100:.1f}", lag, info.get("death_lag_src", ""), self.updates, tag,
                          info.get("death_cause", "")])
        self._csv.flush()
        self.best_prog = max(self.best_prog, peak) if tag == "play" else self.best_prog
        if self.cfg.verbose:
            what = {"play": "--play, not learned from",
                    'controller_eval_incumbent':'controller evaluation: incumbent, not learned from',
                    'controller_eval_candidate':'controller evaluation: candidate, not learned from',
                    "frontier": "frontier: died in the replay, before the brain's turn"}.get(
                tag, "route practice, not learned from")
            print(f"[attempt {self.attempts}] ({what}) "
                  f"{'COMPLETE' if info.get('complete') else 'died'} peak={peak * 100:.0f}%")

    def _finish_attempt(self, seg: _Seg, info: dict, peak: float, next_obs, tag: str = "") -> None:
        cfg = self.cfg
        lag = int(info.get("death_lag", 0)) if cfg.trim_death else 0
        seg.trim(lag)
        seg.completed=bool(info.get('complete'))
        if not info.get('complete') and seg.teacher:
            # A collision invalidates recommendations inside their own planning
            # horizon. The 64-step tail keeps these labels out of banked data.
            dt = getattr(self.env,'_step_dt',lambda **kw: 1/30)(peek=True)
            tail = min(len(seg.teacher),int(math.ceil(1.5/max(1e-3,dt))))
            if cfg.route_controller and seg.discounts and 0<cfg.gamma<1:
                elapsed=0.;tail=0
                for discount in reversed(seg.discounts):
                    elapsed+=math.log(max(1e-12,discount))/math.log(cfg.gamma)/30.
                    tail+=1
                    if elapsed>=1.5:break
            seg.teacher[-tail:] = [-1]*tail
        if info.get("complete") and seg.rew:
            seg.rew[-1] += cfg.complete_bonus
            if seg.next_potential:
                final_discount=seg.discounts[-1] if seg.discounts else cfg.gamma
                seg.rew[-1] -= final_discount * seg.next_potential[-1]
                seg.shaping[-1] = -seg.potential[-1]
        elif seg.rew:
            seg.rew[-1] = cfg.death_penalty
            if seg.potential:
                seg.rew[-1] -= seg.potential[-1]
                seg.shaping[-1] = -seg.potential[-1]
        if seg.next_potential:
            seg.next_potential[-1] = 0.
        if cfg.clearance_weight > 0:
            self.session.event('clearance', dict(attempt=self.attempts + 1, tag=tag,
                               tail_steps=len(seg), weight=cfg.clearance_weight,
                               tail_shaping_sum=sum(seg.shaping), terminal_phi=0.,
                               tail_potential_mean=sum(seg.potential)/max(1,len(seg.potential))))
        if len(seg):
            self._push(seg, next_obs, True)
        self.attempts += 1
        survived = max(0, int(info.get("attempt_total", len(seg))))
        self._note_cause(info)
        self.recent.append((survived, peak))
        self.recent = self.recent[-50:]
        self._w.writerow([f"{time.time():.1f}", self.attempts, self.steps, survived,
                          f"{peak * 100:.1f}", lag, info.get("death_lag_src", ""), self.updates, tag,
                          info.get("death_cause", "")])
        self._csv.flush()
        if tag == "frontier":
            # (the brain only played the end of it: doesn't count as the brain's own best)
            if self.cfg.verbose:
                print(f"[attempt {self.attempts}] (frontier: brain from the hand-over, learned from) "
                      f"{'COMPLETE' if info.get('complete') else 'died'} peak={peak * 100:.0f}%")
            return
        improved = (peak > self.best_prog + 1e-6) if peak > 0 else (survived > self.best_steps)
        self.best_prog = max(self.best_prog, peak)
        self.best_steps = max(self.best_steps, survived)
        if improved:
            self.save(best=True)
        if cfg.verbose:
            tag = "COMPLETE" if info.get("complete") else "died"
            print(f"[attempt {self.attempts}] {tag} after {survived / 30:.1f}s  peak={peak * 100:.0f}%  "
                  f"best={max(self.best_prog, 0) * 100:.0f}%  (cut {lag} post-crash steps, "
                  f"{info.get('death_lag_src', '')}"
                  + (f", this one read {info['death_lag_raw']}" if info.get("death_lag_src") == "median" else "")
                  + ")" + ("  NEW BEST" if improved else "")
                  + (f"\n    how: {info['death_cause']}" if info.get("death_cause") else ""))
        if self.on_attempt is not None:
            self.on_attempt(self, survived, peak, info)

    def _trust_mode(self) -> bool:
        """This attempt replays without shield edits? (--trust-replay, or --trust-replay-ab on odd
        attempts; self.attempts only counts up after the attempt is logged)"""
        return bool(TRUST_REPLAY or (TRUST_REPLAY_AB and self.attempts % 2 == 1))

    def _log_route_timing(self, tc: float, info: dict, peak: float) -> None:
        """One line per attempt in <session>/route_timing.jsonl: how well this attempt lined up
        with the best run (start offset, drift during the run), the decision intervals and the
        input latency. timing_report.py summarises it. Never lets logging take the bot down."""
        try:
            rec = self.route.attempt_timing()
            rec["plan"] = getattr(self.route, "frontier_plan", None)
            rec["trust_replay"] = self._trust_mode()
            rec["frontier_flips"] = int(getattr(self, "_frontier_flips", 0))
            rec["orb_presses"] = int(getattr(self, "_orb_presses", 0))
            self._orb_presses = 0
            rec["frontier_pattern"] = getattr(self, "_frontier_pattern", None)
            self._frontier_flips, self._frontier_pattern = 0, None
            import landmark_route as _LRg
            if _LRg.GHOST and self.lm is not None and getattr(self.lm, "ghost_report", None) is not None:
                rec["ghost"] = dict(self.lm.ghost_report, path_blocks=round(float(self.lm.ghost[-1][0]), 1)
                                    if self.lm.ghost else 0.0)
            if GUIDE_MARGIN is not None:
                rec["guide_brain_choices"] = int(getattr(self, "_guide_n", 0))
                self._guide_n = 0
            rec.update(attempt=int(info.get("attempts", 0) or 0), clock_ok=bool(self._clock_ok),
                       died_at_s=round(float(tc), 3), peak_pct=round(float(peak) * 100.0, 1),
                       complete=bool(info.get("complete")), death_lag=int(info.get("death_lag", 0) or 0),
                       replay_controlled=bool(getattr(self.route, "controlled", False)))
            ages = np.asarray(self._att_ages, float) * 1000.0
            if ages.size:
                rec.update(age_ms_median=round(float(np.median(ages)), 1),
                           age_ms_p05=round(float(np.percentile(ages, 5)), 1),
                           age_ms_p95=round(float(np.percentile(ages, 95)), 1))
            me = getattr(self.env, "mode_est", None)
            d = getattr(me, "delay_s", None) if me is not None else None
            if d is not None:
                rec["input_delay_ms"] = round(float(d) * 1000.0, 1)
            lm = self.lm
            if lm is not None:
                tr = lm.tracker
                rec.update(lm_ever_locked=bool(tr.ever_locked), lm_first_lock=(
                    None if tr.first_lock is None else [round(float(x), 3) for x in tr.first_lock]),
                    lm_origin=round(float(lm.origin()), 3), lm_origins_n=len(lm.origins))
                try:   # which presses fired, their timing shift k and distance from the point (blocks)
                    rec["lm_fired"] = [[int(f["i"]), int(f.get("k", 0)), f.get("r")] for f in lm.fired]
                    pw = (lm.report or {}).get("press_wait_ms") or []
                    if pw:
                        rec["precise_wait_ms_median"] = round(float(np.median(pw)), 1)
                    sw = getattr(self.env, "_slot_wait", None)
                    if sw is not None:
                        rec["slot_wait_ms"] = round(float(sw) * 1000.0, 1)
                except Exception:
                    pass
            self.session.event("route_timing", rec)
        except Exception as ex:
            print(f"[route] timing log failed: {ex!r}")

    # -- main loop -------------------------------------------------------------
    def _frontier_explore(self, lm, a: int) -> int:
        """Near and just past the end of the best run, every FRONTIER_EVERY-th attempt changes the
        brain's decisions (FRONTIER_PATTERNS, a different kind each time); never to an action the
        planner rates as less safe."""
        try:
            if FRONTIER_EVERY <= 0 or not self.cfg.play:
                return a
            n = self.attempts + 1                         # this attempt's number
            s_, b_ = float(lm.tracker.s), float(lm.best_s or 0.0)
            lead = float(getattr(lm, "speed", 10.0) or 10.0) * 1.5
            if b_ <= 0 or not (b_ - lead <= s_ <= b_ + 20.0):
                return a
            # count attempts that GET here, not attempt numbers: live (055502) a third of the
            # attempts died at an earlier wall, and only 1 of 42 got a change at the frontier
            if getattr(self, "_fe_seen", None) != n:
                self._fe_seen = n
                # kept in the route, so it carries across sessions (sessions are often only
                # 7-10 attempts: a per-session count rarely got to its 3rd visit)
                lm.frontier_visits = int(getattr(lm, "frontier_visits", 0) or 0) + 1
            v = int(lm.frontier_visits)
            if v % FRONTIER_EVERY:
                return a
            pattern = FRONTIER_PATTERNS[(v // FRONTIER_EVERY) % len(FRONTIER_PATTERNS)]
            if getattr(self, "_fe_attempt", None) != n:
                self._fe_attempt, self._fe_k = n, 0
                print(f"[frontier] attempt {n} (visit {v} to the end of the best run): {pattern}")
            self._fe_k += 1
            self._frontier_pattern = pattern
            if pattern == "presses become waits" and a != 1:
                return a
            if pattern == "waits become presses" and a != 0:
                return a
            if pattern == "swap every 2nd decision" and self._fe_k % 2:
                return a
            sr = getattr(self.env, "_last_search", None) or {}
            k_new, k_old = ("1", "00") if a == 0 else ("00", "1")
            if sr.get(k_new) is not None and sr.get(k_old) is not None \
                    and float(sr[k_new]) < float(sr[k_old]) - 0.05:
                return a                                  # the other action is clearly less safe
            self._frontier_flips = getattr(self, "_frontier_flips", 0) + 1
            return 1 - a
        except Exception:
            pass
        return a

    def _orb_contact(self, lm, a_lm: int, a_brain: int) -> int:
        """An orb works only if you press WHILE touching it, so its remembered press position is
        only right if the icon arrives on the same path as last time. Live (Polargeist 044852,
        26 deaths around 21 s): in the orb section over the thorn floor the brain pressed as
        the icon reached an orb, the replay said "wait" (its remembered press for that orb fell
        a little earlier or later), the orb was missed and the cube fell into the thorns. So:
        when the brain presses, an orb is in reach when the key lands (input delay ahead), and
        the remembered run used an orb there, the press goes out."""
        try:
            if a_lm != 0 or a_brain != 1:
                return a_lm
            ents = getattr(self.env, "last_ents", None)
            if ents is None or ents.player() is None:
                return a_lm
            from env_boxes import special_boxes as _sb
            me = getattr(self.env, "mode_est", None)
            delay = float(getattr(me, "delay_s", None) or 0.084)
            lead = float(getattr(lm, "speed", None) or 10.38) * delay      # blocks moved before it lands
            s_now = float(lm.tracker.s)
            for x0, x1, bot, top, kind, fx, _i, _n in _sb(ents, 4.0):
                if kind != "orb" or (fx and fx[0] == "none"):
                    continue
                if not (x0 - 0.8 < lead and x1 + 0.8 > lead - 0.3):       # in reach as the key lands
                    continue
                if not (bot < 1.2 and top > -1.2):                       # at the icon's height
                    continue
                s_orb = s_now + 0.5 * (x0 + x1)
                if any(abs(float(e["s"]) - s_orb) <= 3.0 for e in (lm.best_events or [])):
                    self._orb_presses = getattr(self, "_orb_presses", 0) + 1
                    return 1
        except Exception:
            pass
        return a_lm

    def _ghost_choice(self, lm, a_lm: int) -> int:
        """--ghost: where the remembered PATH covers what's ahead, the action whose predicted arc
        follows it replaces the remembered button press (landmark_route.ghost_action). The
        replay keeps running underneath (hand-overs, stats) and decides wherever there's no ghost."""
        try:
            import landmark_route as _LR
            if not _LR.GHOST:
                return a_lm
            _me = getattr(self.env, "mode_est", None)
            # the arcs the ghost compares are the planner's fixed plans, which don't include
            # pads / orbs: near one, "stay on the path" means jumping where the pad would have
            # launched it (live 041703: jumps over the yellow pad) -- the replay + shield decide
            try:
                from env_boxes import special_boxes as _sb
                _ents = getattr(self.env, "last_ents", None)
                if _ents is not None and _ents.player() is not None and any(
                        q[1] > -0.5 and q[0] < 7.5 for q in _sb(_ents, 8.0)):
                    return a_lm
            except Exception:
                pass
            a_g = lm.ghost_action(getattr(self.env, "last_arcs", None), getattr(_me, "mode", None))
            if a_g is not None:
                self._ghost_n = getattr(self, "_ghost_n", 0) + 1
                return int(a_g)
        except Exception as ex:
            print(f"[ghost] {ex!r}")
        return a_lm

    def _landmark_action(self, t_att: float, a_brain: int, a2: int, forced: bool):
        """Landmark replay for this decision. Always OBSERVES (builds / follows the level map).
        --play (landmarks-only): whenever the position is locked and the map has presses,
        the landmarks DRIVE -- the clock replay is ignored entirely (its timing is trash).
        Past the remembered presses (replay_action() None) the brain takes over, never
        the clock. Training / frontier keep the old gating: landmarks only refine the
        clock's replay window, so the brain still gets its own attempts."""
        lm = self.lm
        try:
            lv = self.route.level
            if lv is not None and lm.level_id != lv["id"]:
                lm.set_level(lv["id"])
            lm.clock_ok = bool(self._clock_ok)      # this attempt began at a respawn (origin learning)
            lm.cur_mode = getattr(getattr(self.env, "mode_est", None), "mode", None)
            lm.observe_env(self.env, t_att)
            if self.cfg.play:
                # Landmarks-only play: replay as soon as there is a position.
                # Locked is best, but early in the attempt (first ~40 blocks)
                # dead reckoning has barely drifted (scroll accumulates), so
                # replaying by position still beats the brain -- waiting for a
                # lock instead hands the whole opening to the brain every
                # attempt, which is why replay "wasn't following" (only 1/117
                # overrides in 000327: almost nothing ever came from replay).
                # Past the map (None) the brain takes over.
                try:
                    _early = float(getattr(getattr(lm, "tracker", None), "s", 1e9) or 1e9) < 40.0
                except Exception:
                    _early = False
                if bool(getattr(lm, "best_events", None)) and (lm.locked or _early):
                    # 1s hand-over: replay stops ~1s of scroll before the best
                    # run's end, the brain+shield plays the ending live so it
                    # can beat the best instead of replaying into its crash.
                    try:
                        a_lm = lm.replay_action(handover_s=1.0)
                    except TypeError:
                        a_lm = lm.replay_action()
                    if a_lm is not None:
                        a_lm = self._ghost_choice(lm, a_lm)
                        a_lm = self._orb_contact(lm, a_lm, int(a_brain))
                        # Unlocked (dead-reckoning) position drifts: enforcing a
                        # RELEASE (0) here suppresses the brain/shield's press
                        # and walks into the first spikes -- the 1-2% odd
                        # deaths. Presses (1) are still replayed (missing a
                        # jump is certain death); releases fall through to the
                        # brain+shield until the lock confirms the position.
                        if not bool(getattr(lm, "locked", False)) and int(a_lm) == 0:
                            self.route.phase = None
                            return int(a_brain), False
                        if int(a_lm) != int(a2):
                            self._lm_overrides += 1
                        self.route.phase = "replay"
                        self.route.controlled = True
                        return int(a_lm), True
                    # hand-over to the brain: --play is deterministic, so without
                    # a nudge it repeats the exact inputs that died at the best
                    # point every attempt. First consult the death-spot memory:
                    # if the brain wants the move that killed here before, do
                    # the opposite (the shield still guards it). Unknown spots
                    # fall back to alternating-attempt probing.
                    _entering = getattr(self.route, "phase", None) == "replay"
                    self.route.phase = None
                    if _entering:
                        try:
                            _lead = float(getattr(lm, "speed", 10.0) or 10.0) * 1.0
                            _near_best = float(lm.tracker.s) >= float(lm.best_s or 0.0) - _lead
                        except Exception:
                            _near_best = True
                        if _near_best:
                            try:
                                _bad = lm.taboo_action(float(lm.tracker.s))
                            except Exception:
                                _bad = None
                            if _bad is not None and int(a_brain) == int(_bad):
                                return int(1 - int(a_brain)), False
                            if _bad is None and (self.attempts % 2 == 1):
                                return int(1 - int(a_brain)), False
                    return self._frontier_explore(lm, int(a_brain)), False
                # empty map: the brain explores, NOT the clock
                # (clock timing is trash -- it must never drive in play)
                self.route.phase = None
                return int(a_brain), False
            if forced and getattr(self.route, "phase", None) == "replay" and lm.ready():
                a_lm = lm.replay_action()
                if a_lm is not None:
                    if int(a_lm) != int(a2):
                        self._lm_overrides += 1
                    return int(a_lm), True
        except Exception as ex:                  # never let the memory take the bot down
            print(f"[landmarks] disabled after an error: {ex!r}")
            self.lm = None
        return a2, forced

    def run(self) -> None:
        cfg = self.cfg
        obs, _ = self.env.reset()
        potential = self._clearance_potential()
        seg = _Seg()
        peak = 0.0
        flushed = 0          # steps of the current attempt already pushed (long attempts)
        keep = max(64,int(math.ceil(1.5*60/max(1,getattr(self.env,'frame_skip',2))))+4)
        try:
            while self.steps < cfg.total_steps and not self.stop:
                _ta = time.perf_counter()
                evaluation=bool(cfg.route_controller and self.learner.evaluating and not cfg.play)
                a, extra = self.act(obs, deterministic=cfg.play)     # (--play: no random moves)
                if self.route is not None and not evaluation:
                    if not self._att_times:
                        self.route.start_attempt()
                        if cfg.play and self.lm is not None and self.route.level is not None:
                            self.lm.clock_ok = bool(self._clock_ok)
                            self.lm.begin_attempt(self.route.level["id"])
                    t_att = self._attempt_t()
                    a2, forced = self.route.action(t_att, a, getattr(self.env, "last_hires", None))
                    if self.lm is not None:
                        # --play: landmarks-only, the clock replay is never trusted
                        # for an action. Training/frontier: landmarks refine the clock.
                        a2, forced = self._landmark_action(t_att, a, a2, forced)
                    if GUIDE_MARGIN is not None and forced and getattr(self.route, "phase", None) == "replay" \
                            and int(a2) != int(a):
                        _q = self.last_q
                        try:
                            _sure = _q is not None and len(_q) == 2 and abs(float(_q[1]) - float(_q[0])) >= GUIDE_MARGIN
                        except Exception:
                            _sure = False
                        if _sure and not extra.get("shielded"):
                            forced = False               # the brain's (already shielded) choice goes out
                            self._guide_n = getattr(self, "_guide_n", 0) + 1
                    _trusted = bool(self._trust_mode() and self.lm is not None and getattr(self.lm, "locked", False))
                    if forced and getattr(self.route, "phase", None) == "replay" and cfg.shield and cfg.algo == "dqn" \
                            and not _trusted:
                        # replaying the best run: the safety shield still watches -- a replay a
                        # frame off can walk into what the recorded run just missed.
                        # This stays ON in --play too: the 233446 alternating 17%/9%
                        # pattern is the landmark replay mistiming press 16 (stats
                        # hunt 0 vs +/-1) while every shield verdict says shield_right.
                        # Unshielded replay dies at 9%; the shield's correction is what
                        # reaches 17%. Pure playback without the guard just repeats the miss.
                        # The FULL shield, not the lighter replay_shield: live (050609, --play) the
                        # replayed best run pressed where the planner said press 0.27 / wait 0.96 --
                        # the light shield only steps in when waiting is safe even in the worse
                        # world (>= 1.0), so 11 of 16 attempts died at 1% on replayed presses.
                        # (replay_guard -- the full shield minus its "press now" rule -- was tried
                        #  live and was clearly worse: median 3% vs 9%; the rule helps the early part)
                        a_sh = safety_shield(obs, a2)
                        meta = (getattr(self.env,'_last_search',None) or {}).get('flight')
                        if self.cfg.flight_buffer > 0 and meta and obs[3] > .5 and bool(flight_mode_mask(obs)):
                            proposed = self.replay_flight_shield.choose(a2,meta)
                            if self.replay_flight_shield.last is not None:
                                a_sh = proposed
                            self.env._dbg = dict(getattr(self.env,'_dbg',{}) or {},
                                                 flight_control=self.replay_flight_shield.last)
                        else:
                            self.replay_flight_shield.reset()
                        if a_sh != a2:
                            self.shield_n += 1
                            print(f"[replay-shield] t={t_att:.2f}s replay said {a2}, shield sent {a_sh}")
                            if self.cfg.route_controller:
                                self._shield_note_override(self._shield_mode(obs), a2, a_sh, 'replay')
                            a2 = a_sh
                    if forced:
                        extra = dict(extra,flight_teacher=-1)
                        try:
                            self.env._dbg = dict(getattr(self.env, "_dbg", {}) or {}, shielded=a2 != a, route=True,
                                                 flight_teacher=-1)
                        except Exception:
                            pass
                        a = a2
                    self._att_times.append(t_att)
                self._att_k += 1
                if self.lm is not None:
                    # --precise-press: the replay fired this decision and asked for an exact send time
                    _pa = getattr(self.lm, "press_at", None)
                    self.lm.press_at = None
                    if _pa is not None and int(a) == 1 and self.route is not None and \
                            getattr(self.route, "phase", None) == "replay":
                        self.env._press_at = _pa
                _tb = time.perf_counter()
                nobs, _r, _term, _trunc, info = self.env.step(a)
                _age = getattr(self.env, "_action_observation_age_s", None)
                if isinstance(_age, (int, float)) and 0.0 <= _age < 1.0 and len(self._att_ages) < 5000:
                    self._att_ages.append(float(_age))
                next_potential = self._clearance_potential()
                time_scale=float(np.clip(self.env._motion_dt(),.005,.1)/(1/30)) if cfg.route_controller else 1.
                transition_discount=cfg.gamma**time_scale
                shape = shaping_reward(potential, next_potential, transition_discount)
                a = int(info.get('action_applied', a))
                if extra.get('flight_teacher',-1) != a:
                    extra['flight_teacher'] = -1
                if self.route is not None and not evaluation:
                    self.route.record(t_att, a)
                    if self.lm is not None:
                        self.lm.record(a, t_att + self.route.offset,     # (level time, like route.py)
                                       age=getattr(self.env, "_action_observation_age_s", None),
                                       wait_extra=getattr(self.env, "_press_wait_extra", 0.0) or 0.0)
                _tc = time.perf_counter()
                self._tm["brain"] += _tb - _ta
                self._tm["env"] += _tc - _tb
                profile = getattr(self.env, 'profile', None)
                if profile is not None:
                    profile.add('brain_route', _tb - _ta)
                    profile.add('env_total', _tc - _tb)
                self.steps += 1
                # menus / loading screens before an attempt: nothing to learn (inside an attempt
                # every step is kept, so the post-crash trimming still lines up)
                in_level = info.get("in_level", True) or len(seg) > 0 or info.get("died") or info.get("complete")
                # --frontier: the replayed part isn't the brain's -- it learns from the hand-over on
                replaying = (not evaluation and cfg.frontier and self.route is not None and self.route.frontier_attempt
                             and getattr(self.route, "phase", None) in ("replay", "wait_align"))
                if in_level and not (replaying and not info.get("died") and not info.get("complete")):
                    if not evaluation:self.new_steps += 1
                    tracking=float(info.get('route_tracking_reward',0.)) if cfg.route_controller else 0.
                    seg.obs.append(obs); seg.act.append(a); seg.rew.append((cfg.alive_reward - cfg.hold_cost * a)*time_scale+shape+tracking)
                    seg.teacher.append(int(extra.get('flight_teacher',-1)))
                    if cfg.route_controller:
                        seg.route_records.append(extra.get('route_record',{}));seg.discounts.append(transition_discount)
                    if cfg.clearance_weight > 0 and not cfg.play:
                        seg.potential.append(potential)
                        seg.next_potential.append(next_potential)
                        seg.shaping.append(shape)
                    if "logp" in extra:
                        seg.logp.append(extra["logp"]); seg.val.append(extra["v"])
                p = info.get("prog")
                if cfg.route_controller:
                    self._shield_attribute(bool(info.get('died')), bool(info.get('complete')))
                if evaluation and not info.get('died'):
                    self.learner.trial.observe(getattr(self.env,'last_hires',None),self._attempt_t())
                if isinstance(p, (int, float)) and not info.get("died"):
                    peak = max(peak, float(p))
                obs = nobs
                potential = next_potential
                if self.on_step is not None:
                    _td = time.perf_counter()
                    _keep = self.on_step(self, obs, a, info)
                    self._tm["gui"] += time.perf_counter() - _td
                    if profile is not None:
                        profile.add('gui', time.perf_counter() - _td)
                    if _keep is False:
                        break
                if info.get("died") or info.get("complete"):
                    _death_paused = bool(info.get("paused_for_death"))
                    if _death_paused:
                        self._pause_depth += 1     # the env already paused the game at the death
                    # Full attempt length includes replay; the learning segment doesn't.
                    # A death during replay must never produce negative survival time.
                    lag_count = int(info.get('death_lag', 0)) if cfg.trim_death else 0
                    info = dict(info, attempt_total=max(0, int(info.get('attempt_steps', self._att_k)) - lag_count))
                    if self.route is not None and self._att_times:
                        lag = int(info.get("death_lag", 0))
                        tc = self._att_times[max(0, len(self._att_times) - 1 - lag)]
                        _lv = self.route.level
                        _T0 = float(_lv["best_T"]) if _lv is not None else 0.0
                        self.route.end_attempt(tc, bool(info.get("complete")), clock_ok=self._clock_ok)
                        self._log_route_timing(tc, info, peak)
                        if self.lm is not None:
                            _lv = self.route.level
                            _T1 = float(_lv["best_T"]) if _lv is not None else 0.0
                            if cfg.play:
                                # landmarks-only: the clock decides NOTHING. Best = further
                                # in BLOCKS than the map's best (0.5 block margin). The
                                # clock's best_T is ignored -- its timing is trash.
                                # No _clock_ok gate either: that flag exists because the
                                # CLOCK's first-attempt timing includes menu time. The
                                # landmark crash position comes from scroll (0 in menus),
                                # so the first attempt's distance is valid and must save
                                # -- otherwise the 58% first run is always thrown away.
                                try:
                                    _s_crash = self.lm._crash_position(tc)
                                except Exception:
                                    _s_crash = None
                                try:
                                    _s_best = float(self.lm.best_s)
                                except Exception:
                                    _s_best = 0.0
                                _best = _s_crash is not None and (
                                    bool(info.get("complete")) or _s_crash > _s_best + 0.5)
                            else:
                                # remember this run's presses when it is the route's new best, OR when it got
                                # further into the level than anything the landmark memory holds yet (a short
                                # prefix of a good run is still worth replaying; the route's clock covers the rest)
                                try:
                                    _cover = float(self.lm.times[-1][1]) if self.lm.times else 0.0
                                except Exception:
                                    _cover = 0.0
                                _tl = tc + self.route.offset
                                _best = self._clock_ok and (_T1 > _T0 + 1e-9 or (_tl >= 3.0 and _tl > _cover + 0.5))
                            # Death-spot memory: what move died, and where. The next
                            # attempt's hand-over avoids that exact move there instead
                            # of coin-flipping forever. Recorded even when NOT best
                            # (a failed ending still teaches what kills there).
                            if cfg.play and bool(info.get("died")) and not bool(info.get("complete")):
                                try:
                                    _lm_s = self.lm._crash_position(tc)
                                    if _lm_s is not None:
                                        self.lm.note_death_move(_lm_s, int(a))
                                except Exception:
                                    pass
                            _rep = None
                            try:
                                _rep = self.lm.end_attempt(tc, bool(_best), bool(info.get("complete")),
                                                           lag_steps=int(info.get("death_lag", 0) or 0))
                            except Exception as ex:
                                print(f"[landmarks] end_attempt failed: {ex!r}")
                            if _best and _rep is not None:
                                if _rep.get("committed"):
                                    print(f"[landmarks] saved this run's presses ({len(self.lm.best_events)} presses, {len(self.lm.entries)} landmarks)")
                                else:
                                    print(f"[landmarks] NOT saved: {_rep.get('commit_reason')}")
                            if self._lm_overrides:
                                print(f"[landmarks] {self._lm_overrides} decision(s) where landmark replay differed from the clock replay")
                            self._lm_overrides = 0
                    self._att_times = []
                    self._att_ages = []
                    # after a death the env's step already ran into the new attempt: count it
                    self._att_k = 1 if info.get("died") else 0
                    self._clock_ok = bool(info.get("died"))   # the next attempt starts at a respawn
                    if evaluation:
                        arm=self.learner.trial.arm
                        self._log_route_attempt(seg,info,peak,tag='controller_eval_'+arm)
                        report=self.learner.finish_evaluation_attempt(peak,info['attempt_total'],bool(info.get('complete')))
                        self.session.event('controller_live_evaluation',report)
                        if cfg.verbose:print(f'[controller evaluation] {arm}: progress={peak*100:.1f}% -- {report["reason"]}')
                        self.save()
                        if self.learner.promotion_pending:self.save(best=True)
                    elif cfg.play:
                        self._log_route_attempt(seg, info, peak, tag="play")   # nothing is learned
                        if self.attempts % 5 == 0:
                            # --play never trains, so the per-stage timing (timing.jsonl + the
                            # [health] line) was never written: log it every 5 attempts
                            try:
                                self._health(0.0)
                            except Exception as ex:
                                print(f"[health] timing log failed: {ex!r}")
                    elif cfg.frontier and self.route is not None and self.route.frontier_attempt:
                        if len(seg) >= 2:       # learn from the brain's part (from the hand-over)
                            self._finish_attempt(seg, info, peak, nobs, tag="frontier")
                        else:                   # died during the replay: nothing of the brain's
                            self._log_route_attempt(seg, info, peak, tag="frontier")
                    elif self.route is not None and self.route.attempt_was_route:
                        # practice run of the route memory: NOT for the brain (no learning from
                        # it, no "best brain" save) -- just logged
                        self._log_route_attempt(seg, info, peak)
                    else:
                        self._finish_attempt(seg, info, peak, nobs)
                    seg, peak, flushed = _Seg(), 0.0, 0
                    if _death_paused:
                        self._resume_game()        # all after-death work done: play the new attempt
                    self.flight_shield.reset()
                    self.replay_flight_shield.reset()
                    potential = 0. # new attempt boundary; normal GD respawns as a cube
                    if info.get("complete") and cfg.play:
                        print(f"LEVEL COMPLETE on attempt {self.attempts} (--play: the route is saved in "
                              f"{cfg.route_dir}; the brain wasn't changed)")
                        break
                    if info.get('complete') and evaluation and cfg.stop_on_complete:
                        print('LEVEL COMPLETE during controller evaluation: progress saved; restart the same level and resume the controller to finish comparison.')
                        break
                    if info.get("complete") and cfg.stop_on_complete and not evaluation:
                        stats=self.learner.train(self.new_steps)
                        if cfg.route_controller:
                            if stats:self.updates+=1
                            self.new_steps=0;self.save()
                            self.session.event('controller_evaluation',self.learner.last_evaluation)
                            if self.learner.promotion_pending:self.save(best=True)
                            print(f'LEVEL COMPLETE: experience/candidate saved to {checkpoint_path(cfg)}; controller promotion still requires validation.')
                        else:
                            self.save(best=True)
                            print(f"LEVEL COMPLETE on attempt {self.attempts}! Brain saved to "
                                  f"{checkpoint_path(cfg, best=True)}. Restart the level (or pick a new one) "
                                  f"and resume with --model.")
                        break
                    if self.new_steps >= cfg.update_steps and not cfg.play:
                        self._do_update("after death")
                    if info.get("complete"):
                        obs, _ = self.env.reset()
                        potential = self._clearance_potential()
                elif len(seg) >= cfg.max_rollout and (evaluation or cfg.play or (self.route is not None and self.route.controlled
                                                                   and not self.route.frontier_attempt)):
                    n = len(seg) - keep                      # route practice: keep it out of the brain
                    seg.drop_front(n)
                    flushed += n
                elif len(seg) >= cfg.max_rollout:
                    # long attempt: bank everything but the tail, train mid-run
                    if cfg.algo == "dqn":
                        n = self._push(seg, None, False, upto=len(seg) - keep)
                        seg.drop_front(n)
                        flushed += n
                    else:
                        self._push(seg, nobs, False)
                        flushed += len(seg)
                        seg = _Seg()
                    self._do_update("long attempt")
        except KeyboardInterrupt:
            print("interrupted")
        finally:
            if (cfg.route_controller and not cfg.play and not self.learner.evaluating and len(seg)>keep):
                # Preserve completed safe prefixes on a clean stop. Keep the
                # uncertain death tail out, just as for a mid-run bank.
                count=self._push(seg,None,False,upto=len(seg)-keep)
                seg.drop_front(count)
            if cfg.play:
                print("--play: the brain wasn't changed (nothing saved); the route memory is in "
                      f"{cfg.route_dir}")
            else:
                self.save()
                print(f"saved {checkpoint_path(cfg)}")
            self.session.event('summary', dict(attempts=self.attempts, steps=self.steps,
                               updates=self.updates, best_pct=100*self.best_prog,
                               stages=getattr(self.env, 'profile', None).summary()
                               if getattr(self.env, 'profile', None) is not None else {}))
            try:
                self.session.close()
            except Exception:
                pass
            self._csv.close()