"""Synthetic PPO update check: forced hyperparams + partial-buffer train."""
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class _Box(gym.Env):
    observation_space = gym.spaces.Box(-2, 2, (6, 9), np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((6, 9), np.float32), {}

    def step(self, a):
        return np.zeros((6, 9), np.float32), 0.1, False, False, {}


vec = DummyVecEnv([_Box])
m = PPO("MlpPolicy", vec, n_steps=64, policy_kwargs=dict(net_arch=[128, 128]),
        device="cpu", verbose=0)
# force the exact settings preview_boxes.py now forces on resumed models
from stable_baselines3.common.utils import FloatSchedule
m.learning_rate, m.ent_coef = 1e-3, 0.08
m.clip_range = FloatSchedule(0.2)
m.n_epochs, m.batch_size, m.gamma, m.gae_lambda = 3, 1024, 0.99, 0.95
try:
    m.policy.optimizer.param_groups[0]["lr"] = 1e-3
except Exception:
    pass
print(f"forced lr={m.learning_rate} ent={m.ent_coef} clip={m.clip_range(1.0)} "
      f"epochs={m.n_epochs} opt_lr={m.policy.optimizer.param_groups[0]['lr']:.0e}")

from stable_baselines3.common import logger as _lg
m.set_logger(_lg.configure(None, []))
_r = vec.reset()
m._last_obs = _r[0] if isinstance(_r, tuple) else _r
m._last_episode_starts = np.ones((1,), dtype=bool)
buf = m.rollout_buffer
buf.reset()
from stable_baselines3.common.utils import obs_as_tensor
import torch as th
obs = m._last_obs
n = 40
for i in range(n):
    with th.no_grad():
        a, v, lp = m.policy(obs_as_tensor(obs, m.device))
    no, rw, dn, _ = vec.step(a.cpu().numpy())
    buf.add(obs, a.cpu().numpy(), rw, m._last_episode_starts, v, lp)
    obs = no
    m._last_episode_starts = dn
    m.num_timesteps += 1
with th.no_grad():
    lv = m.policy.predict_values(obs_as_tensor(obs, m.device))
buf.buffer_size = n
buf.compute_returns_and_advantage(last_values=lv, dones=m._last_episode_starts)
buf.full = True
oe, ob = m.n_epochs, m.batch_size
m.n_epochs, m.batch_size = 4, min(128, n)
m.train()
m.n_epochs, m.batch_size = oe, ob
buf.buffer_size = m.n_steps
vals = m.logger.name_to_value
print(f"train OK: pg={vals.get('train/policy_gradient_loss', 0):+.4f} "
      f"ent={vals.get('train/entropy_loss', 0):+.3f} "
      f"vl={vals.get('train/value_loss', 0):.2f} steps={m.num_timesteps}")
print("TRAIN_PROBE_DONE")
