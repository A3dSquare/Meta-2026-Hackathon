"""Level-agnostic policy env: entity BOXES in, pixels out.

Wraps GymGDEnv: each tick runs the detector on the 320x180 frame, takes
the 6 nearest entities ahead of the player as a (6,8) float32 vector
(dx, dy, w, h, class/7, edge, closing, hold, player-y), and exposes THAT as the observation. PPO trains
MlpPolicy on boxes -- no CNN, no backgrounds, no per-level memorization.
A spike 0.2 ahead at dy 0.0 means the same jump on Stereo, Jumper, and
Finger: that is what transfers.

Reward/pacing/death/pause/settle are inherited from GymGDEnv unchanged
(progress-delta + jump + air + death penalty + pct0 counter).
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError("gymnasium is required") from e

from detect import Entities, make_detector

N_ENT, FEAT = 6, 9


class EntityBoxEnv(gym.Wrapper):
    """(6,9) box-vector obs for cross-level PPO (dx,dy,w,h,cls/7,edge,closing,hold,py)."""

    def __init__(self, env: gym.Env, backend: str = "heuristic",
                 detector: Any | None = None) -> None:
        try:
            super().__init__(env)
        except Exception:
            # gym.Wrapper type-checks; fall back to duck-typing so a broken
            # detector/env can never crash training -- zeros flow instead.
            self.env = env
            self._wrap_failed = True
        else:
            self._wrap_failed = False
        try:
            self.det = detector if detector is not None else make_detector(backend)
        except Exception:
            self.det = lambda f: Entities([])
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(N_ENT, FEAT),
                                            dtype=np.float32)
        self._last_ents: Entities = Entities([])
        self._last_edge: float | None = None
        self._jitter_rng = np.random.default_rng(1234)
        self._jitter: bool = True  # obs jitter for transfer (set False for eval)
        self._prev_attempts: int = 1  # attempts-delta death gate baseline
        self._dbg_ring: list = []

    def _jittered(self, v):
        """Level-invariant obs: dx/edge/closing jitter +-1/32, py +-1/64.
        Same jump on a different background = same FEATURES, not same tape."""
        if not self._jitter:
            return np.ascontiguousarray(v)
        import numpy as _np
        out = v.copy()
        out[:, 0] = _np.clip(out[:, 0] + self._jitter_rng.uniform(-1/32, 1/32, size=out.shape[0]), -2, 2)
        out[:, 5] = _np.clip(out[:, 5] + self._jitter_rng.uniform(-1/32, 1/32, size=out.shape[0]), -2, 2)
        out[:, 6] = _np.clip(out[:, 6] + self._jitter_rng.uniform(-1/64, 1/64, size=out.shape[0]), -2, 2)
        out[:, 8] = _np.clip(out[:, 8] + self._jitter_rng.uniform(-1/64, 1/64, size=out.shape[0]), -2, 2)
        return _np.ascontiguousarray(out)

    _err_shown: bool = False  # print-once: zeros must NEVER be silent

    def _loud(self, where: str, exc: BaseException) -> None:
        if not self._err_shown:
            self._err_shown = True
            print(f"[EntityBoxEnv:{where}] {type(exc).__name__}: {exc} -> zero obs (fix me)")

    def _frame(self):
        """Newest 180x320x3 RGB frame. SmallObsWrapper caches it in
        last_display; the policy obs it returns (84x84x13) is NOT a frame."""
        w = self.env
        for _ in range(4):
            d = getattr(w, "last_display", None)
            if d is not None:
                f = np.asarray(d)
                return f[-1] if f.ndim == 4 else f
            w = getattr(w, "env", None)
            if w is None:
                break
        return None

    def _vec(self, stacked, hold: bool = False) -> np.ndarray:
        try:
            ents = self.det(stacked)
        except Exception as ex:
            self._loud("detect", ex)
            ents = Entities([])
        self._last_ents = ents
        try:
            px, _py = ents.player_pos()
            # Only spikes still AHEAD count: a stale/false detection behind
            # the player (edge very negative) must never become the "nearest
            # spike" feeding proximity rent / clearance in ProgressReward.
            sp = [x for x in ents.items if int(x.cls) == 0
                  and ((x.x - px) - (x.w / 2 + 0.02)) > -0.06]
            self._last_edge = (min((((x.x - px) - (x.w / 2 + 0.02)) for x in sp), default=None))
        except Exception:
            self._last_edge = None
        try:
            return self._jittered(ents.nearest(N_ENT, hold=hold))
        except Exception as ex:
            self._loud("nearest", ex)
            return np.zeros((N_ENT, FEAT), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        try:
            self._prev_attempts = int(info.get("attempts", 1))
        except Exception:
            self._prev_attempts = 1
        try:
            from detect import Entities as _E
            self._last_ents = _E([])
        except Exception:
            pass
        frame = self._frame()
        if frame is None:
            self._loud("reset", RuntimeError("no last_display frame"))
            return np.zeros((N_ENT, FEAT), dtype=np.float32), info
        return self._vec(frame), info

    def step(self, action):
        try:
            from gd import action_to_hold as _ath
            _held = bool(_ath(action))
        except Exception:
            _held = False
        try:
            _px, _ = self._last_ents.player_pos()
            _sp = [x for x in self._last_ents.items if int(x.cls) == 0
                   and ((x.x - _px) - (x.w / 2 + 0.02)) > -0.06]
            _ed = (min((((x.x - _px) - (x.w / 2 + 0.02)) for x in _sp), default=None))
        except Exception:
            _ed = None
        try:
            # walk to the GymGDEnv and stash the edge for this tick
            _w = self.env
            for _ in range(4):
                if hasattr(_w, "reward") and hasattr(_w, "cap"):
                    try:
                        _w._pending_edge = _ed
                    except Exception:
                        pass
                    break
                _w = getattr(_w, "env", None)
                if _w is None:
                    break
        except Exception:
            pass
        obs, r, term, trunc, info = self.env.step(action)
        try:
            frame = self._frame()
            if frame is None or frame.shape != (180, 320, 3):
                raise RuntimeError(f"bad frame {None if frame is None else frame.shape}")
            v = self._vec(frame, hold=_held)
        except Exception as ex:
            self._loud("step", ex)
            v = np.zeros((N_ENT, FEAT), dtype=np.float32)
        try:
            info["n_entities"] = len(self._last_ents.items)
        except Exception:
            pass
        # EPISODE TERMINATION ON DEATH (the 500k-step stall fix):
        # the inner env NEVER terminates (done=False always; death is just
        # info['event']). SB3 therefore stitched 70k-tick timeouts into one
        # giant episode: every rollout held 3% of an episode, GAE bootstrapped
        # forever, explained_variance sat at exactly 0.0 for 240 updates.
        # Death now ENDS the episode so each update sees complete attempts.
        try:
            ev = str(info.get("event"))
        except Exception:
            ev = "?"
        try:
            att = int(info.get("attempts", self._prev_attempts))
        except Exception:
            att = self._prev_attempts
        died = (ev == "death") or (att > self._prev_attempts)
        try:
            if died:
                print(f"[death-signal] event={ev} attempts {self._prev_attempts}->{att}")
                self._dbg_ring.append((ev, self._prev_attempts, att))
                if len(self._dbg_ring) >= 5:
                    print(f"[death-signal] last5={self._dbg_ring[-5:]}")
        except Exception:
            pass
        self._prev_attempts = max(self._prev_attempts, att)
        if died:
            try:
                info["terminal_observation"] = v.copy()
            except Exception:
                pass
            return v, r, True, False, info
        return v, r, term, trunc, info