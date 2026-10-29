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
from dataclasses import asdict, dataclass, field
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
    death_penalty: float = -10.0
    complete_bonus: float = 50.0
    net_arch: tuple = (256, 256)
    device: str = "cpu"
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
    play: bool = False             # --play: beat the level -- no training, best route + practice, brain past it
    frontier: bool = False         # --frontier: every other attempt replays to just before the furthest point, brain learns from there
    save_dir: str = "./checkpoints"
    name: str = "preview_boxes"
    log_dir: str = "./logs_boxes"
    verbose: int = 1
    # --- DQN --------------------------------------------------------------
    n_step: int = 4
    buffer_size: int = 100_000
    batch_size: int = 128
    replay_ratio: float = 4.0      # gradient steps per new environment step
    max_grad_steps: int = 1000     # cap per update (bounds the pause length)
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
        return self._greedy(qv), {"q": qv}

    def _greedy(self, qv) -> int:
        if len(qv) == 2:
            return int(qv[1] - qv[0] > self.cfg.jump_margin)
        return int(np.argmax(qv))

    def hold_prob(self, obs: np.ndarray) -> float:
        qv = self.q_values(obs)
        return float(1.0 / (1.0 + math.exp(-(qv[1] - qv[0]))))  # display only

    # -- data ----------------------------------------------------------------
    def size(self) -> int:
        return self.cfg.buffer_size if self.full else self.ptr

    def add_segment(self, obs: list, act: list, rew: list, next_obs_last: np.ndarray | None,
                    terminal: bool, upto: int | None = None) -> int:
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
        losses, qs = [], []
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
                    a_star = ((qn[:, 1] - qn[:, 0]) > c.jump_margin).long()[:, None]
                else:
                    a_star = qn.argmax(dim=1, keepdim=True)
                tq = self.qt(nx).gather(1, a_star).squeeze(1)
                y = R + d * tq
            q = self.q(o).gather(1, a[:, None]).squeeze(1)
            loss = th.nn.functional.smooth_l1_loss(q, y)
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
                               eps=self.epsilon(), buffer=n)
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

    def __len__(self) -> int:
        return len(self.act)

    def trim(self, n: int) -> None:
        if n <= 0:
            return
        k = max(1, len(self) - n)
        for name in ("obs", "act", "rew", "logp", "val"):
            lst = getattr(self, name)
            if lst:
                setattr(self, name, lst[:k])

    def drop_front(self, k: int) -> None:
        for name in ("obs", "act", "rew", "logp", "val"):
            lst = getattr(self, name)
            if lst:
                setattr(self, name, lst[k:])


def checkpoint_path(cfg: TrainConfig, best: bool = False) -> str:
    base = os.path.join(cfg.save_dir, ("best_boxes" if cfg.name == "preview_boxes" else cfg.name + "_best") if best else cfg.name)
    return base + (".pt" if cfg.algo == "dqn" else "")


def make_learner(env, cfg: TrainConfig, model_path: str | None = None):
    if cfg.algo == "dqn":
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
    if obs is None or len(obs) != OBS_DIM or obs[OFF_SEARCH + 3] <= 0.5:
        return a
    p1, w00, w01 = (float(v) for v in obs[OFF_SEARCH:OFF_SEARCH + 3])
    mine, other = (safety_level(p1), max(safety_level(w00), safety_level(w01))) if int(a) == 1 else \
        (max(safety_level(w00), safety_level(w01)), safety_level(p1))
    return 1 - int(a) if (mine == 0 and other >= 2) else int(a)


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
    if obs is None or len(obs) != OBS_DIM:
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
        if int(a) == 0 and L00 < L01 == Lw and L1 >= Lw and Lw > 0:
            return 1                        # waiting only works by pressing at the very next step:
                                            # timing that tight flips with jitter -- press now
        if L1 == 0 and Lw == 0:             # doomed both ways (as far as we can see): buy time
            mine, other = (p1, max(w00, w01)) if int(a) == 1 else (max(w00, w01), p1)
            if other > mine + 0.12:
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
        self.learner = learner if learner is not None else make_learner(env, self.cfg, model_path)
        self.on_step = on_step
        self.on_attempt = on_attempt
        self.on_train_tick = None     # called every 50 gradient steps during an update
        self._tm = {"brain": 0.0, "env": 0.0, "gui": 0.0}   # seconds since the last [health] line
        self.shield_n = 0             # actions the safety shield changed since the last update
        self.shield_steps = 0
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
        self._att_k = 0                  # decisions since the attempt started (clock fallback)
        self._clock_ok = False           # the first attempt after starting isn't timed from a respawn
        self._att_times: list = []
        self._csv_path = os.path.join(self.cfg.log_dir, "attempts.csv")
        new = not os.path.exists(self._csv_path)
        self._csv = open(self._csv_path, "a", newline="")
        self._w = csv.writer(self._csv)
        if new:
            self._w.writerow(["time", "attempt", "total_steps", "survived_steps", "peak_pct",
                              "death_lag", "lag_src", "updates", "route", "cause"])

    # -- helpers ------------------------------------------------------------
    def act(self, obs, deterministic: bool = False):
        a, extra = self.learner.act(obs, deterministic)
        self.last_q = extra.get("q")
        a_brain = a
        if self.cfg.shield and self.cfg.algo == "dqn":
            a2 = safety_shield(obs, a)
            if a2 != a:
                self.shield_n += 1
                extra = dict(extra, shielded=True)
                a = a2
        self.shield_steps += 1
        try:    # for the death snapshots: what the brain wanted vs what was sent
            q = self.last_q
            self.env._dbg = {"a_brain": int(a_brain), "shielded": int(a) != int(a_brain), "attempt": self.attempts + 1,
                             "qdiff": float(q[1] - q[0]) if q is not None and len(q) == 2 else None}
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
        try:
            self.learner.save(checkpoint_path(self.cfg, best),
                              {"attempts": self.attempts, "best_pct": self.best_prog * 100})
        except Exception as e:
            print(f"save failed: {e}")

    def _push(self, seg: _Seg, next_obs, terminal: bool, upto: int | None = None) -> int:
        if self.cfg.algo == "dqn":
            return self.learner.add_segment(seg.obs, seg.act, seg.rew, next_obs, terminal, upto)
        self.learner.add_segment(seg.obs, seg.act, seg.rew, next_obs, terminal, seg.logp, seg.val)
        return len(seg)

    def _do_update(self, why: str) -> None:
        t0 = time.perf_counter()
        try:
            self.env.pause()
        except Exception as e:
            print(f"PAUSE FAILED: {e}")
        st = self.learner.train(self.new_steps, tick=self.on_train_tick)
        self.new_steps = 0
        if st:
            self.updates += 1
        self.save()
        try:
            self.env.mark_trained()
            self.env.resume()
        except Exception as e:
            print(f"RESUME FAILED: {e}")
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
            import psutil
            mem = f"  RAM {psutil.Process().memory_info().rss / 1e6:.0f} MB"
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
        survived = int(info.get("attempt_total", len(seg)))
        self._note_cause(info)
        self._w.writerow([f"{time.time():.1f}", self.attempts, self.steps, survived,
                          f"{peak * 100:.1f}", lag, info.get("death_lag_src", ""), self.updates, tag,
                          info.get("death_cause", "")])
        self._csv.flush()
        self.best_prog = max(self.best_prog, peak) if tag == "play" else self.best_prog
        if self.cfg.verbose:
            what = {"play": "--play, not learned from",
                    "frontier": "frontier: died in the replay, before the brain's turn"}.get(
                tag, "route practice, not learned from")
            print(f"[attempt {self.attempts}] ({what}) "
                  f"{'COMPLETE' if info.get('complete') else 'died'} peak={peak * 100:.0f}%")

    def _finish_attempt(self, seg: _Seg, info: dict, peak: float, next_obs, tag: str = "") -> None:
        cfg = self.cfg
        lag = int(info.get("death_lag", 0)) if cfg.trim_death else 0
        seg.trim(lag)
        if info.get("complete"):
            seg.rew[-1] += cfg.complete_bonus
        else:
            seg.rew[-1] = cfg.death_penalty
        self._push(seg, next_obs, True)
        self.attempts += 1
        survived = int(info.get("attempt_total", len(seg)))
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

    # -- main loop -------------------------------------------------------------
    def run(self) -> None:
        cfg = self.cfg
        obs, _ = self.env.reset()
        seg = _Seg()
        peak = 0.0
        flushed = 0          # steps of the current attempt already pushed (long attempts)
        keep = 64            # tail kept un-pushed so a later death can still be trimmed
        try:
            while self.steps < cfg.total_steps and not self.stop:
                _ta = time.perf_counter()
                a, extra = self.act(obs, deterministic=cfg.play)     # (--play: no random moves)
                if self.route is not None:
                    if not self._att_times:
                        self.route.start_attempt()
                    t_att = self._attempt_t()
                    a2, forced = self.route.action(t_att, a, getattr(self.env, "last_hires", None))
                    if forced and getattr(self.route, "phase", None) == "replay" and cfg.shield and cfg.algo == "dqn":
                        # replaying the best run: the safety shield still watches -- a replay a
                        # frame off can walk into what the recorded run just missed
                        a_sh = replay_shield(obs, a2)
                        if a_sh != a2:
                            self.shield_n += 1
                            a2 = a_sh
                    if forced:
                        try:
                            self.env._dbg = dict(getattr(self.env, "_dbg", {}) or {}, shielded=a2 != a, route=True)
                        except Exception:
                            pass
                        a = a2
                    self.route.record(t_att, a)
                    self._att_times.append(t_att)
                self._att_k += 1
                _tb = time.perf_counter()
                nobs, _r, _term, _trunc, info = self.env.step(a)
                _tc = time.perf_counter()
                self._tm["brain"] += _tb - _ta
                self._tm["env"] += _tc - _tb
                self.steps += 1
                # menus / loading screens before an attempt: nothing to learn (inside an attempt
                # every step is kept, so the post-crash trimming still lines up)
                in_level = info.get("in_level", True) or len(seg) > 0 or info.get("died") or info.get("complete")
                # --frontier: the replayed part isn't the brain's -- it learns from the hand-over on
                replaying = (cfg.frontier and self.route is not None and self.route.frontier_attempt
                             and getattr(self.route, "phase", None) == "replay")
                if in_level and not (replaying and not info.get("died") and not info.get("complete")):
                    self.new_steps += 1
                    seg.obs.append(obs); seg.act.append(a); seg.rew.append(cfg.alive_reward - cfg.hold_cost * a)
                    if "logp" in extra:
                        seg.logp.append(extra["logp"]); seg.val.append(extra["v"])
                p = info.get("prog")
                if isinstance(p, (int, float)) and not info.get("died"):
                    peak = max(peak, float(p))
                obs = nobs
                if self.on_step is not None:
                    _td = time.perf_counter()
                    _keep = self.on_step(self, obs, a, info)
                    self._tm["gui"] += time.perf_counter() - _td
                    if _keep is False:
                        break
                if info.get("died") or info.get("complete"):
                    info = dict(info, attempt_total=flushed + len(seg) - (int(info.get("death_lag", 0)) if cfg.trim_death else 0))
                    if self.route is not None and self._att_times:
                        lag = int(info.get("death_lag", 0))
                        tc = self._att_times[max(0, len(self._att_times) - 1 - lag)]
                        self.route.end_attempt(tc, bool(info.get("complete")), clock_ok=self._clock_ok)
                    self._att_times = []
                    # after a death the env's step already ran into the new attempt: count it
                    self._att_k = 1 if info.get("died") else 0
                    self._clock_ok = bool(info.get("died"))   # the next attempt starts at a respawn
                    if cfg.play:
                        self._log_route_attempt(seg, info, peak, tag="play")   # nothing is learned
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
                    if info.get("complete") and cfg.play:
                        print(f"LEVEL COMPLETE on attempt {self.attempts} (--play: the route is saved in "
                              f"{cfg.route_dir}; the brain wasn't changed)")
                        break
                    if info.get("complete") and cfg.stop_on_complete:
                        self.learner.train(self.new_steps)
                        self.save(best=True)
                        print(f"LEVEL COMPLETE on attempt {self.attempts}! Brain saved to "
                              f"{checkpoint_path(cfg, best=True)}. Restart the level (or pick a new one) "
                              f"and resume with --model.")
                        break
                    if self.new_steps >= cfg.update_steps and not cfg.play:
                        self._do_update("after death")
                    if info.get("complete"):
                        obs, _ = self.env.reset()
                elif len(seg) >= cfg.max_rollout and (cfg.play or (self.route is not None and self.route.controlled
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
            if cfg.play:
                print("--play: the brain wasn't changed (nothing saved); the route memory is in "
                      f"{cfg.route_dir}")
            else:
                self.save()
                print(f"saved {checkpoint_path(cfg)}")
            self._csv.close()
