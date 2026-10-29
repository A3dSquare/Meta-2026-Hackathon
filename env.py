"""Step 1.5 -- resets: Gymnasium env over capture + input + reward.

Contract:
  - Vanilla GD Auto-Retry ON = free episode resets: death already
    restarts the attempt in-game, so [`reset()`](env.py:92) only zeroes
    [`ProgressReward`](gd.py:209) state + releases SPACE. No menu nav.
  - Pause protocol: when the level auto-retries (pause menu / death
    screen holds), training pauses via ESC; when the model is done
    training, SPACE resumes. [`GymGDEnv`](env.py:56) exposes
    [`pause()`](env.py:151)/[`resume()`](env.py:159) driving
    [`GDInput`](gd.py:118) -- ESC to pause, SPACE to continue.
  - [`SubprocVecEnv`](env.py:1) CANNOT carry a dxcam handle on Windows
    (handle not picklable, DXGI duplicator is per-process). Use
    [`DummyVecEnv`](env.py:1) only -- see [`make_vec_env()`](env.py:195).
  - Obs stay uint8 `(N,H,W,3)` RGB; [`Monitor`](env.py:1) + best-%
    callback checkpoint the best-% model, never latest -- see
    [`BestPercentCallback`](env.py:218).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover - import-time hint
    raise ImportError("gymnasium is required: pip install gymnasium") from e

from capture import CaptureConfig, GDCapture
from gd import (
    ACTION_SPACE,
    DEFAULT_KEY,
    GDInput,
    ProgressReward,
    RewardConfig,
)

__all__ = [
    "GymGDEnv",
    "EnvConfig",
    "make_vec_env",
    "seal_and_train",
    "train_episode_on_death",
    "BestPercentCallback",
    "PAUSE_KEY",
    "RESUME_KEY",
    "_stacked_to_hwc",
    "SmallObsWrapper",
]

PAUSE_KEY = "esc"  # ESC freezes GD for training
RESUME_KEY = "space"  # SPACE restarts/plays after settle (user contract:
# death -> ESC pause -> train -> wait 3s -> SPACE play)


def _stacked_to_hwc(stacked):
    """(N,H,W,3) RGB stack -> (H,W,3*N) channel-last for SB3 CnnPolicy."""
    import numpy as _np
    s = _np.asarray(stacked)
    if s.ndim != 4:
        return _np.ascontiguousarray(s)
    n, h, w, c = s.shape
    return _np.ascontiguousarray(s.transpose(1, 2, 0, 3).reshape(h, w, n * c))



from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Wiring for GymGDEnv. Capture owns the 60Hz grid."""

    capture: CaptureConfig = field(default_factory=CaptureConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    input_key: str = DEFAULT_KEY  # SPACE -- the only gameplay input
    max_steps: int = 60 * 120  # 2min cap per episode (safety, GD kills first)
    pct_hook: Optional[Callable[[np.ndarray | None], Optional[float]]] = None
    complete_hook: Optional[Callable[[np.ndarray], bool]] = None


class GymGDEnv(gym.Env):
    """Pixels -> Discrete(2) -> shaped reward. Auto-retry = free resets."""

    metadata = {"render_mode": None}

    def __init__(self, config: EnvConfig | None = None,
                 cap: GDCapture | None = None,
                 inp: GDInput | None = None,
                 pct_reader: Any | None = None) -> None:
        super().__init__()
        self.cfg: EnvConfig = config or EnvConfig()
        self.cap: GDCapture = cap if cap is not None else GDCapture(self.cfg.capture)
        self.inp: GDInput = inp if inp is not None else GDInput(key=self.cfg.input_key)
        self.reward = ProgressReward(cfg=self.cfg.reward)
        self.pct = pct_reader  # PctReader or None; built lazily to avoid cv2 import cost
        self.action_space = ACTION_SPACE
        h, w = self.cfg.capture.height, self.cfg.capture.width
        n = self.cfg.capture.n_stack
        self.observation_space = spaces.Box(0, 255, shape=(h, w, 3 * n), dtype=np.uint8)
        self._owns_cap = cap is None
        self._owns_inp = inp is None
        self._steps = 0
        self._paused = False

    # ------------------------------------------------------------------ #
    def _ensure_started(self) -> None:
        if not self.cap._running:
            self.cap.start()
            for _ in range(10):  # warm the 4-frame stack
                self.cap.read()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Free reset: release SPACE, zero reward baseline. Game already retried."""
        super().reset(seed=seed)
        self._ensure_started()
        self.inp.release()
        self.reward.reset()
        self._steps = 0
        obs = self.cap.read()
        return _stacked_to_hwc(obs), {"attempts": self.reward.attempts}

    def step(self, action: Any):
        """1 tick = 1 frame + 1 actuation + 1 reward. Pacing by read()."""
        if self._paused:
            # Training paused: hold position, pay nothing, burn no decisions.
            # Reward is suspended so the hidden bar can't fake a death.
            obs = self.cap.read()
            self.reward.suspend()
            self.reward.step(obs)
            return (_stacked_to_hwc(obs), 0.0, False, False,
                    {"paused": True, "attempts": self.reward.attempts})
        obs = self.cap.read()  # hard 60Hz lock lives here
        self.inp.step(action)  # strict Discrete(2) -> SPACE, change-only
        if self.cfg.pct_hook is not None:
            full = getattr(self.cap, "last_fullres", None)
            pct = self.cfg.pct_hook(full)
        elif self.pct is not None:
            pct = self.pct.tick(getattr(self.cap, "last_fullres", None))
        else:
            pct = None
        done_flag = self.cfg.complete_hook(obs) if self.cfg.complete_hook else False
        from gd import action_to_hold
        try:
            _held = bool(action_to_hold(action))
        except Exception:
            _held = False
        _edge = getattr(self, "_pending_edge", None)
        try:
            self._pending_edge = None
        except Exception:
            pass
        r, done, info = self.reward.step(obs, pct=pct, completed=bool(done_flag), held=_held, spike_edge=_edge)
        if isinstance(pct, tuple) and len(pct) == 3:    # how long since the % was last read (ticks)
            info = dict(info, pct_age=int(pct[2]))
        self._steps += 1
        terminated = bool(done)  # level complete
        truncated = self._steps >= self.cfg.max_steps
        info.update({
            "steps": self._steps,
            "attempts": self.reward.attempts,
            "best_fill": self.reward.best_fill,
            "total": self.reward.total,
        })
        return _stacked_to_hwc(obs), float(r), terminated, truncated, info

    # --- pause protocol: ESC to pause, SPACE to resume ---------------- #
    # User contract: death -> ESC pause -> train -> wait 3s -> SPACE play.
    # pause() freezes (one ESC tap, already-paused is a no-op so the menu
    # is never toggled closed mid-training). resume() blocks the full
    # settle_s (1.0s) after mark_trained(), then SPACE to play.
    settle_s: float = 1.0  # class-level default; instance may override
    _train_done_ts: float = 0.0  # wall time the last TRAINED update finished

    def pause(self) -> None:
        """Level auto-retried / training time: ESC freezes the game."""
        if self._paused:
            # Already paused: do NOT re-tap ESC. ESC toggles the menu --
            # a second tap CLOSES it, which un-pauses mid-training (the
            # first-pause-only bug's second half).
            self.reward.suspend()
            return
        self.inp.release()  # never leave SPACE down under pause menu
        try:
            # Direct global keyboard (pydirectinput): GD is foreground,
            # one ESC tap freezes it. No window targeting anymore.
            self.inp._backend.keyDown(PAUSE_KEY)
            self.inp._backend.keyUp(PAUSE_KEY)
        except Exception:  # noqa: BLE001 - pause is best-effort
            pass
        self.reward.suspend()  # bar hidden under menu: track, never penalize
        self._paused = True
        self._paused_ts = __import__("time").perf_counter()

    def mark_trained(self) -> None:
        """Call the instant the PPO update finishes (starts settle clock)."""
        self._train_done_ts = __import__("time").perf_counter()

    def resume(self) -> None:
        """Model done training: SPACE plays (after 1s settle)."""
        if not self._paused:
            return
        import time as _t
        now = _t.perf_counter()
        train_ts = getattr(self, "_train_done_ts", 0.0) or 0.0
        wait = self.settle_s - (now - train_ts)
        if train_ts > 0.0 and wait > 0:
            _t.sleep(wait)  # game stays ESC-paused; no frames lost (frozen)
        self._paused = False
        self.reward.unsuspend()  # snap baseline to post-menu bar, pay 0
        try:
            self.inp._backend.keyDown(RESUME_KEY)
            self.inp._backend.keyUp(RESUME_KEY)
        except Exception:  # noqa: BLE001 - resume is best-effort
            pass

    @property
    def paused(self) -> bool:
        return self._paused


class SmallObsWrapper(gym.Wrapper):
    """84x84x12 RGB (COLOR) frame-stack for live --train ONLY.

    The 180x320x12 RGB stack is 5.6GB per 2048-step rollout as float32
    plus a ~23M-param NatureCNN head -> first PPO update OOMs/crashes on
    CPU. Color is REQUIRED: orbs/portals are color-coded (blue/orange/
    pink change the required input), so gray is NOT acceptable here.
    84x84x12 keeps all 4 RGB frames downscaled: 85kB/obs, ~43MB per
    512-step rollout, first NatureCNN layer grows 4->12 channels but
    stays ~1.7M params. GUI shows the live 320x180 color capture; this
    wrapper only shrinks what the CNN trains on.

    SPIKE-SALIENCE ASSIST (NOT object detection, NOT SAM): channel 12 is a
    84x84 motion/edge map (frame t minus frame t-1, abs, downscaled). Spikes
    are static white triangles on scrolling backgrounds -- the difference
    map lights up exactly the lethal geometry one frame before contact,
    while flat ground cancels out. The CNN still gets all 12 raw RGB
    channels (orbs/portals need color); channel 13 just hands it the
    "what moved" cue for free so it stops learning edges from scratch.
    SAM would segment every frame on GPU (~100ms, kills 60Hz) and still
    output no policy; this is one abs-diff at 84x84 (~0.05ms, CPU).
    """

    OBS_SHAPE = (84, 84, 13)

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(0, 255, shape=self.OBS_SHAPE, dtype=np.uint8)
        import cv2 as _cv2
        self._cv2 = _cv2
        self._stack: list = []
        self.last_display = None  # latest (4,180,320,3) RGB stack, for the GUI

    def _convert(self, stacked_hwc) -> "np.ndarray":
        import numpy as _np
        s = _np.asarray(stacked_hwc)
        if s.ndim == 4 and s.shape[1:] == (180, 320, 3):
            self.last_display = _np.ascontiguousarray(s)
        if s.ndim == 3 and s.shape == (180, 320, 12):
            self.last_display = _np.ascontiguousarray(
                s.reshape(180, 320, 4, 3).transpose(2, 0, 1, 3))
        if s.ndim == 3 and s.shape == (84, 84, 12):
            # legacy 12ch (pre-salience checkpoints): pad zero salience
            z = _np.zeros((84, 84, 1), dtype=_np.uint8)
            return _np.ascontiguousarray(_np.concatenate([s, z], axis=-1))
        if s.ndim == 3 and s.shape == (84, 84, 13):
            return _np.ascontiguousarray(s)
        if s.ndim == 4:
            # (N,H,W,3) RGB stack -> per-frame COLOR 84x84, channels stacked
            smalls = []
            for f in s:
                small = self._cv2.resize(_np.asarray(f), (84, 84),
                                         interpolation=self._cv2.INTER_AREA)
                smalls.append(_np.ascontiguousarray(small))
            out = _np.concatenate(smalls, axis=-1)
            # Channel 13: spike-salience = newest-vs-previous abs motion.
            try:
                g0 = (0.299 * smalls[-1][:, :, 0] + 0.587 * smalls[-1][:, :, 1]
                      + 0.114 * smalls[-1][:, :, 2])
                g1 = (0.299 * smalls[-2][:, :, 0] + 0.587 * smalls[-2][:, :, 1]
                      + 0.114 * smalls[-2][:, :, 2]) if len(smalls) > 1 else g0
                sal = _np.abs(g0 - g1)
                m = sal.max()
                if m > 1e-6:
                    sal = sal / m * 255.0
                out = _np.concatenate([out, sal[:, :, None].astype(_np.uint8)], axis=-1)
            except Exception:
                out = _np.concatenate([out, _np.zeros((84, 84, 1), dtype=_np.uint8)], axis=-1)
            return _np.ascontiguousarray(out)
        if s.ndim == 3 and s.shape == (180, 320, 12):
            frames = s.reshape(180, 320, 4, 3).transpose(2, 0, 1, 3)
            return self._convert(_np.ascontiguousarray(frames))
        # unknown: zero obs, never crash training
        return _np.zeros(self.OBS_SHAPE, dtype=_np.uint8)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # self.env obs is already HWC (180,320,12); _convert turns it into
        # the 4-frame color stack. Never double-read the capture.
        try:
            return self._convert(obs), info
        except Exception:
            import numpy as _np
            return _np.zeros(self.OBS_SHAPE, dtype=_np.uint8), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        small = self._convert(obs)
        return small, r, term, trunc, info

    def pause(self) -> None:
        try:
            return self.env.pause()
        except Exception:
            pass

    def resume(self) -> None:
        try:
            return self.env.resume()
        except Exception:
            pass

    @property
    def settle_s(self) -> float:
        try:
            return float(self.env.settle_s)
        except Exception:
            return 2.0

    @settle_s.setter
    def settle_s(self, v: float) -> None:
        try:
            self.env.settle_s = float(v)
        except Exception:
            pass

    def mark_trained(self) -> None:
        try:
            return self.env.mark_trained()
        except Exception:
            pass

    @property
    def _paused(self) -> bool:
        try:
            return bool(self.env._paused)
        except Exception:
            return False

    @_paused.setter
    def _paused(self, v: bool) -> None:
        try:
            self.env._paused = bool(v)
        except Exception:
            pass

    def close(self) -> None:
        try:
            return self.env.close()
        except Exception:
            pass
        # Fallback if wrapped env lacks close (never crash teardown).
        try:
            if getattr(self.env, "_owns_inp", False):
                self.env.inp.close()
        except Exception:
            pass
        if getattr(self.env, "_owns_cap", False):
            self.cap.stop()


# ---------------------------------------------------------------------- #
# Vec + checkpointing: DummyVecEnv (dxcam handle can't cross processes).
# ---------------------------------------------------------------------- #
def make_vec_env(cfg: EnvConfig | None = None, monitor_dir: str = "./logs",
                 env_kwargs: dict | None = None):
    """Single-env DummyVecEnv wrapped in Monitor for SB3 PPO."""
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    os.makedirs(monitor_dir, exist_ok=True)

    def _thunk():
        e = GymGDEnv(cfg, **(env_kwargs or {}))
        return Monitor(e, os.path.join(monitor_dir, "monitor.csv"))
    return DummyVecEnv([_thunk])


def seal_and_train(model, n_epochs: int = 2, batch_size: int = 64,
                   ent_floor: float = 0.3) -> dict:
    """Run the NATURAL PPO update on the sealed rollout, while PAUSED.

    Call ONLY when rollout_buffer.full (n_steps reached, mid-rollout here
    means the sealed episode boundary). compute_returns_and_advantage +
    PPO.train() are SB3's own code path -- no hand-rolled gradients, no
    GAE corruption. Runs OFF the 60Hz grid (game is ESC-paused), so no lag.
    Entropy floor: collapsed exploration -> bump ent_coef for next attempt.
    """
    import time as _t
    t0 = _t.perf_counter()
    buf = model.rollout_buffer
    assert buf.full, "seal_and_train needs a full rollout"
    with __import__("torch").no_grad():
        last_v = model.policy.predict_values(buf.observations[-1])
    buf.compute_returns_and_advantage(last_values=last_v,
                                      dones=model._last_episode_starts)
    old_epochs, old_batch = model.n_epochs, model.batch_size
    model.n_epochs, model.batch_size = n_epochs, batch_size
    try:
        model.train()
    finally:
        model.n_epochs, model.batch_size = old_epochs, old_batch
    ms = (_t.perf_counter() - t0) * 1000.0
    try:
        lg = getattr(getattr(model, "logger", None), "name_to_value", {})
        ent = float(lg.get("train/entropy_loss", 0.0) or 0.0)
        pg = float(lg.get("train/policy_gradient_loss", 0.0) or 0.0)
        vl = float(lg.get("train/value_loss", 0.0) or 0.0)
        if abs(ent) < ent_floor and hasattr(model, "ent_coef"):
            model.ent_coef = min(0.05, float(model.ent_coef) * 1.5 + 0.005)
    except Exception:
        ent = pg = vl = 0.0
    return {"loss_ms": ms, "pg": pg, "vl": vl, "ent": ent}


def train_episode_on_death(model, rollout_steps: int = 128) -> dict:
    """Back-compat alias: seals + trains naturally (death-gated callers)."""
    return seal_and_train(model)


class BestPercentCallback:
    """Checkpoint best-% model, not latest. Reads best_fill from infos.

    Works as an SB3 BaseCallback: pass to model.learn(). Saves
    best_model.zip whenever mean best_fill over recent episodes improves.
    """

    def __init__(self, save_path: str = "./checkpoints/best_model",
                 check_freq: int = 2048, verbose: int = 1):
        from stable_baselines3.common.callbacks import BaseCallback

        class _CB(BaseCallback):
            def __init__(inner_self):
                super().__init__(verbose)
                inner_self._best = -1.0
                inner_self._fills: list[float] = []

            def _on_step(inner_self) -> bool:
                infos = inner_self.locals.get("infos", [])
                dones = inner_self.locals.get("dones", [False] * len(infos))
                for done, info in zip(dones, infos):
                    if not done:
                        continue
                    bf = info.get("best_fill")
                    if isinstance(bf, (int, float)):
                        inner_self._fills.append(float(bf))
                if inner_self.n_calls % check_freq == 0 and inner_self._fills:
                    cur = sum(inner_self._fills) / len(inner_self._fills)
                    if cur > inner_self._best:
                        inner_self._best = cur
                        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                        inner_self.model.save(save_path)
                        if verbose:
                            print(f"best {cur*100:.1f}% -> saved {save_path}")
                    inner_self._fills.clear()
                return True

        self._cb = _CB()

    @property
    def callback(self):
        return self._cb

    def __call__(self, *a, **k):
        return self._cb(*a, **k)
