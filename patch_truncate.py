import pathlib

# ============ 1) pause_cb.py: abort-on-death truncate-train ============
p = pathlib.Path("pause_cb.py")
t = p.read_text()
old_cls = '''        class _DeathPause(BaseCallback):
            def _env(self):
                try:
                    return get_env()
                except Exception as e:
                    print(f"PAUSE MISS: get_env failed: {e}")
                    return None

            def _on_step(self) -> bool:
                return True  # bookkeeping only; pause lives in rollout hooks'''
assert old_cls in t
t = t.replace(old_cls, '''        class _DeathPause(BaseCallback):
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
                        obs_t = buf.observations[:n].reshape(-1, *buf.obs_shape)
                        last_v = model.policy.predict_values(obs_t[-1:])
                    # GAE over the partial rollout only
                    buf.buffer_size = n
                    buf.compute_returns_and_advantage(last_values=last_v,
                                                      dones=model._last_episode_starts)
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
                try:
                    infos = self.locals.get("infos", [{}])
                    info = infos[0] if infos else {}
                    _dead = (str(info.get("event")) == "death")
                except Exception:
                    _dead = False
                if _dead:
                    self._train_partial("death")
                    return False  # ABORT rollout at death: zero frozen filler
                return True''')
# neutralize old rollout hooks (keep as no-ops for compat)
t = t.replace("            def _on_rollout_end(self) -> None:",
              "            def _on_rollout_end(self) -> None:  # noqa: legacy no-op (pause lives in _on_step)\n                return\n            def _x_on_rollout_end(self) -> None:")
t = t.replace("            def _on_rollout_start(self) -> None:",
              "            def _on_rollout_start(self) -> None:  # noqa: legacy no-op\n                return\n            def _x_on_rollout_start(self) -> None:")
p.write_text(t)
print("pause_cb truncate wired")
