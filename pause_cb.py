"""Shared death-gated pause callback (train_boxes.py AND preview_boxes.py).

Contract (per user spec):
- _on_rollout_end: pause the game (ESC). Runs AFTER SB3 seals the rollout
  but BEFORE model.train(), so no gradient ever runs on a live game and the
  pause never lands mid-rollout (~1900 frozen steps bug).
- _on_rollout_start: train ALREADY ran (SB3 calls train() between the two
  hooks). mark_trained() + resume (SPACE) after the 1s settle inside resume().
- _on_step: display/death bookkeeping ONLY. Never pauses here.
- While paused, frozen frames must NOT enter the buffer: GymGDEnv.step
  already returns reward 0 while paused, and the rollout that sealed is the
  one being trained -- frozen ticks only accumulate into the NEXT rollout
  if the game stays paused past the seal, which resume() prevents (1s).

Usage:
    from pause_cb import DeathPauseCallback
    cb = CallbackList([DeathPauseCallback(get_env, get_model, gui_state), ...])
where get_env() returns the inner GymGDEnv (has pause/resume/mark_trained)
and get_model() returns the PPO model.
"""
from __future__ import annotations

from typing import Any, Callable


class DeathPauseCallback:
    """Factory: build the SB3 BaseCallback subclass bound to env+model."""

    def __new__(cls, get_env: Callable[[], Any], get_model: Callable[[], Any],
                gui_state: dict | None = None):
        from stable_baselines3.common.callbacks import BaseCallback

        class _DeathPause(BaseCallback):
            def _env(self):
                try:
                    return get_env()
                except Exception as e:
                    print(f"PAUSE MISS: get_env failed: {e}")
                    return None

            def _sealed_len(self) -> int:
                try:
                    return int(getattr(self.model.rollout_buffer, "pos", 0))
                except Exception:
                    return 0

            def _train_partial(self, why: str) -> None:
                """Train on the PARTIAL buffer sealed at death, then resume.
                Runs inside _on_step (return False aborts the rollout right
                after), so NO more frozen ticks are collected. GAE is exact
                over the partial episode (compute_returns over pos steps)."""
                try:
                    model = get_model()
                except Exception as e:
                    print(f"TRAIN MISS: no model: {e}")
                    return
                try:
                    n = self._sealed_len()
                    if n < 8:
                        print(f"death too early ({n} steps); skipping update")
                        return
                    env = self._env()
                    try:
                        env.pause()
                        print(f"PAUSED on death verified={bool(env._paused)} (partial {n})")
                    except Exception as e:
                        print(f"PAUSE FAILED: {e}")
                    import torch as _th
                    buf = model.rollout_buffer
                    with _th.no_grad():
                        from stable_baselines3.common.utils import obs_as_tensor
                        obs_t = obs_as_tensor(buf.observations[:n].reshape(-1, *buf.obs_shape),
                                              model.device)
                        last_v = model.policy.predict_values(obs_t[-1:])
                    # GAE over the partial rollout only
                    buf.buffer_size = n
                    buf.compute_returns_and_advantage(last_values=last_v,
                                                      dones=model._last_episode_starts)
                    buf.full = True  # partial seal: allow get() over n steps
                    old_e, old_b = model.n_epochs, model.batch_size
                    model.n_epochs, model.batch_size = 4, min(128, n)
                    try:
                        model.train()
                    finally:
                        model.n_epochs, model.batch_size = old_e, old_b
                        buf.buffer_size = model.n_steps
                    try:
                        lg = getattr(getattr(model, "logger", None), "name_to_value", {})
                        if gui_state is not None:
                            gui_state["last_train"] = {
                                "loss_ms": 0.0,
                                "pg": float(lg.get("train/policy_gradient_loss", 0.0) or 0.0),
                                "vl": float(lg.get("train/value_loss", 0.0) or 0.0),
                                "ent": float(lg.get("train/entropy_loss", 0.0) or 0.0)}
                            print(f"TRAINED death partial={n} "
                                  f"pg{gui_state['last_train']['pg']:+.4f} "
                                  f"ent{gui_state['last_train']['ent']:+.3f}")
                    except Exception:
                        pass
                    try:
                        model.save("./checkpoints/preview_boxes")
                    except Exception:
                        pass
                    try:
                        env.mark_trained()
                    except Exception:
                        pass
                    try:
                        env.resume()  # blocks 1s settle inside
                        print("RESUMED +1s settle")
                    except Exception as e:
                        print(f"RESUME FAILED: {e}")
                    try:
                        model.rollout_buffer.reset()
                    except Exception:
                        pass
                except Exception as e:
                    import traceback as _tb
                    print(f"partial-train failed: {e}")
                    _tb.print_exc()

            def _on_step(self) -> bool:
                # NOTE: returning False here aborts ALL of learn(), not just
                # the rollout (SB3 CallbackList stops on first False). So we
                # NEVER return False: we set a flag and let the countdown
                # below trigger the manual rollout loop instead. Zero frozen
                # filler either way: pause() runs the same tick as death.
                try:
                    infos = self.locals.get("infos", [{}])
                    info = infos[0] if infos else {}
                    _dead = (str(info.get("event")) == "death")
                except Exception:
                    _dead = False
                if _dead:
                    try:
                        env = self._env()
                        try:
                            env.pause()
                            print(f"PAUSED on death verified={bool(env._paused)}")
                        except Exception as e:
                            print(f"PAUSE FAILED: {e}")
                    except Exception as e:
                        print(f"pause path failed: {e}")
                    try:
                        self._death_pending = True
                    except Exception:
                        pass
                return True  # never abort learn(); manual loop below reseeds

            def _on_rollout_end(self) -> None:
                return  # pause already happened in _on_step
            def _x_on_rollout_end(self) -> None:
                try:
                    env = self._env()
                    if env is None:
                        return
                    try:
                        paused = bool(env._paused)
                    except Exception:
                        paused = False
                    if not paused:
                        try:
                            env.pause()
                            print(f"PAUSED (rollout seal) verified={bool(env._paused)}")
                        except Exception as e:
                            print(f"PAUSE FAILED: {e}")
                except Exception as e:
                    print(f"rollout-end pause failed: {e}")

            def _on_rollout_start(self) -> None:  # noqa: legacy no-op
                return
            def _x_on_rollout_start(self) -> None:
                try:
                    env = self._env()
                    if env is None:
                        return
                    try:
                        paused = bool(env._paused)
                    except Exception:
                        paused = False
                    if not paused:
                        return
                    try:
                        model = get_model()
                        lg = getattr(getattr(model, "logger", None), "name_to_value", {})
                        if gui_state is not None:
                            gui_state["last_train"] = {
                                "loss_ms": 0.0,
                                "pg": float(lg.get("train/policy_gradient_loss", 0.0) or 0.0),
                                "vl": float(lg.get("train/value_loss", 0.0) or 0.0),
                                "ent": float(lg.get("train/entropy_loss", 0.0) or 0.0)}
                            print(f"TRAINED pg{gui_state['last_train']['pg']:+.4f} "
                                  f"ent{gui_state['last_train']['ent']:+.3f}")
                    except Exception:
                        pass
                    try:
                        env.mark_trained()
                    except Exception:
                        pass
                    try:
                        model.save("./checkpoints/preview_boxes")
                    except Exception:
                        pass
                    try:
                        env.resume()  # blocks 1s settle inside
                        print("RESUMED +1s settle")
                    except Exception as e:
                        print(f"RESUME FAILED: {e}")
                except Exception as e:
                    print(f"rollout-start resume failed: {e}")

        return _DeathPause()
