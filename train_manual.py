"""Manual train-per-death loop (replaces model.learn for live GD).

Validates the user's design: collect_rollouts runs WITHOUT any callback
(so nothing pauses mid-rollout and no frozen filler accumulates);
when a death lands inside the collected chunk, compute GAE over exactly
that chunk and run model.train() immediately -- all while the game is
already ESC-paused by pause-on-death... except here nothing pauses
mid-rollout at all: the rollout STOPS at the death because we only ever
ask for (death_step - start) more steps. Then resume + repeat.

This file is the PROOF, not the product: preview_boxes.py keeps the
shared callback for the GUI path. Run:
  python train_manual.py --steps 2000
"""
from __future__ import annotations

import argparse
import time


def main() -> int:
    p = argparse.ArgumentParser(description="manual truncate-train proof")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    import numpy as np
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from env_boxes import EntityBoxEnv
    from detect import Entities, Entity

    class FakeCap:
        _running = True
        def read(self):
            import numpy as _np
            return _np.zeros((4, 180, 320, 3), _np.uint8)
        @property
        def last_fullres(self): return None
    class FakePct:
        def __init__(self): self.n = 0
        def tick(self, f):
            self.n += 1
            m = self.n % 40
            return (m / 100.0 if m < 38 else 0.0, True, 0)
        def load_templates(self): return 0
    from env import EnvConfig, GymGDEnv, SmallObsWrapper
    base = SmallObsWrapper(GymGDEnv(EnvConfig(), cap=FakeCap(), pct_reader=FakePct()))
    benv = EntityBoxEnv(base, backend="heuristic")
    import gymnasium as _g
    vec = DummyVecEnv([lambda: benv])
    m = PPO("MlpPolicy", vec, n_steps=256, batch_size=64, n_epochs=2,
            verbose=0, device="cpu", policy_kwargs=dict(net_arch=[64, 64]))
    import tempfile as _tf
    m.set_logger(__import__("stable_baselines3.common.logger", fromlist=["configure"]).configure(_tf.mkdtemp(), ["stdout"]))
    before = [p.detach().clone() for p in m.policy.parameters()]

    # manual loop: collect until death (max n_steps), then train partial
    import torch as _th
    from stable_baselines3.common.utils import obs_as_tensor
    steps, updates, pauses = 0, 0, 0
    buf = m.rollout_buffer
    vec.reset()
    m.policy.set_training_mode(False)
    obs = vec.reset()
    m._last_obs = obs
    m._last_episode_starts = __import__("numpy").ones((1,), dtype=bool)
    t0 = time.perf_counter()
    while steps < args.steps:
        buf.reset()
        n = 0
        # collect up to n_steps OR death, whichever first (no callback)
        while n < m.n_steps and steps < args.steps:
            with _th.no_grad():
                ot = obs_as_tensor(obs, m.device)
                acts, vals, lps = m.policy(ot)
            acts_np = acts.cpu().numpy()
            new_o, rews, dones, infos = vec.step(acts_np)
            m.num_timesteps += 1
            steps += 1
            buf.add(m._last_obs, acts_np.reshape(-1, 1), rews,
                    m._last_episode_starts, vals, lps)
            m._last_obs = new_o
            m._last_episode_starts = dones
            n += 1
            dead = str((infos[0] if infos else {}).get("event")) == "death"
            if dead:
                obs = vec.reset()
                m._last_obs = obs
                m._last_episode_starts = __import__("numpy").ones((1,), dtype=bool)
                break
            obs = new_o
        if n < 8:
            continue
        t1 = time.perf_counter()
        with _th.no_grad():
            last_v = m.policy.predict_values(obs_as_tensor(
                buf.observations[:n].reshape(-1, *buf.obs_shape), m.device)[-1:])
        buf.buffer_size = n
        buf.compute_returns_and_advantage(last_values=last_v,
                                          dones=m._last_episode_starts)
        buf.full = True
        old_e, old_b = m.n_epochs, m.batch_size
        m.n_epochs, m.batch_size = 2, min(64, n)
        try:
            m.train()
        finally:
            m.n_epochs, m.batch_size = old_e, old_b
            buf.buffer_size = m.n_steps
        updates += 1
        pauses += 1
        if updates <= 3 or updates % 10 == 0:
            print(f"update {updates}: partial={n} steps={steps} "
                  f"{(time.perf_counter()-t1)*1000:.0f}ms")
        buf.reset()
    after = [p.detach().clone() for p in m.policy.parameters()]
    moved = sum(1 for b, a in zip(before, after) if not __import__("numpy").allclose(b.numpy(), a.numpy()))
    dt = time.perf_counter() - t0
    print(f"MANUAL_OK updates={updates} pauses={pauses} moved={moved}/12 {dt:.1f}s")
    vec.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
