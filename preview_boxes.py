"""Live GUI for the cross-level box policy (thin fork of preview.py --train).

Shows the 320x180 COLOR frame + entity boxes + the MlpPolicy action, while
SB3 learns on (16,5) box vectors. Same death-gated PAUSE(ESC)/TRAIN/
+1s/RESUME(SPACE) contract as preview.py.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description="box-policy live GUI + train")
    p.add_argument("--backend", default="heuristic", choices=("heuristic", "yolo"))
    p.add_argument("--model", default=None)
    p.add_argument("--timesteps", type=int, default=500000)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--levels", nargs="*", default=[],
                    help="rotate these GD levels (by name) for general data: stereo jumper finger")
    p.add_argument("--rotate-min", type=float, default=10.0,
                    help="minutes per level before auto-rotating")
    args = p.parse_args()

    from stable_baselines3 import PPO
    import torch as _th  # noqa: F401
    _dev = getattr(args, "device", "auto")
    _dev = _dev if _dev != "auto" else "cpu"
    if args.levels:
        print(f"GENERAL DATA: rotating {args.levels} every {args.rotate_min}min (boxes only, no backgrounds)")
    print(f"device={_dev} torch={_th.__version__}" + (f" gpu={_th.cuda.get_device_name(0)}" if _dev == 'cuda' else ''))

    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from capture import CaptureConfig, GDCapture
    from env import EnvConfig, GymGDEnv, SmallObsWrapper, BestPercentCallback
    from env_boxes import EntityBoxEnv
    from pct import PctReader
    import os as _os
    _os.makedirs("./logs_boxes", exist_ok=True)
    _os.makedirs("./checkpoints", exist_ok=True)

    cap = GDCapture(CaptureConfig())
    cap.start()
    for _ in range(10):
        cap.read()
    rdr = PctReader()
    rdr.load_templates()
    base = SmallObsWrapper(GymGDEnv(EnvConfig(), cap=cap, pct_reader=rdr))
    benv = EntityBoxEnv(base, backend=args.backend)
    vec = DummyVecEnv([lambda: Monitor(benv, "./logs_boxes/monitor.csv")])
    if args.model:
        model = PPO.load(args.model, env=vec)
        # A checkpoint keeps its BIRTH hyperparameters. NOTE: right now
        # checkpoints/ is EMPTY (no preview_boxes.zip on disk), so this
        # force only matters once a run has saved again. Force the intended
        # live settings onto every resumed model.
        try:
            model.learning_rate = 1e-3
            model.n_steps = 4096
            model.batch_size = 1024
            model.n_epochs = 3
            model.ent_coef = 0.08
            model.vf_coef = 1.0
            # PPO stores clip_range as a SCHEDULE callable (FloatSchedule);
            # a raw float crashes train() with 'float' object not callable.
            from stable_baselines3.common.utils import FloatSchedule as _FS
            model.clip_range = _FS(0.2)
            model.gamma = 0.99
            model.gae_lambda = 0.95
            try:
                model.policy.optimizer.param_groups[0]["lr"] = 1e-3
            except Exception:
                pass
            print("resumed hyperparams FORCED: lr=1e-3 ent=0.08 clip=0.2 epochs=3 batch=1024")
        except Exception as _he:
            print(f"hyperparam force failed (continuing): {_he}")
        print(f"resumed {args.model} (BOX policy)")
    else:
        model = PPO("MlpPolicy", vec, learning_rate=1e-3, n_steps=4096,
                    batch_size=1024, n_epochs=3, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.08, vf_coef=1.0,
                    max_grad_norm=0.5, verbose=1, tensorboard_log="./tb_boxes",
                    policy_kwargs=dict(net_arch=[128, 128]), device=_dev)
        print("fresh BOX MlpPolicy 6x7: learns entities, transfers levels.")
    # Manual truncate-train loop calls model.train() directly, bypassing
    # learn()'s _setup_learn (which is what normally creates _logger).
    # Initialize it here so _update_learning_rate doesn't crash on first death.
    try:
        from stable_baselines3.common import logger as _sb_logger
        import os as _os2
        _os2.makedirs("./tb_boxes", exist_ok=True)
        model.set_logger(_sb_logger.configure("./tb_boxes", ["stdout", "tensorboard"]))
    except Exception as _e:
        print(f"logger init failed (non-fatal): {_e}")
    zoom = int(args.scale)
    cv2.namedWindow("BOX policy view", cv2.WINDOW_NORMAL)
    state: dict = {"act": None}

    class Gui(BaseCallback):
        _prev_att: int = 1
        def _on_step(self) -> bool:
            from detect import CLASSES as _CL
            infos = self.locals.get("infos", [{}])
            info = infos[0] if infos else {}
            try:
                state["act"] = int(np.asarray(self.locals.get("actions")).flat[0])
            except Exception:
                pass
            # Display-only: pausing lives in DeathPauseCallback
            # (rollout hooks), never mid-rollout here.
            try:
                _att = int(info.get("attempts", 0) or 0)
                Gui._prev_att = max(Gui._prev_att, _att)
            except Exception:
                pass
            # --- SPIKE OVERLAY: live color + per-entity boxes + edges ---
            # Green = spike (cls 0), yellow/cyan/magenta = orbs, gray = wall.
            # The closing edge distance for the nearest spike is burned into
            # the header so you SEE what the (6,8) vector sees. Zero-entity
            # frames draw "NO ENTS" instead of silently showing nothing.
            try:
                disp = getattr(base, "last_display", None)
                frame = np.ascontiguousarray(disp[-1]).copy() if disp is not None else np.zeros((180, 320, 3), np.uint8)
            except Exception:
                frame = np.zeros((180, 320, 3), np.uint8)
            _spike_txt = ""
            try:
                H0, W0 = 180, 320
                big0 = cv2.resize(frame[:, :, ::-1], (W0 * zoom, H0 * zoom))
                _ents = list(getattr(benv, "_last_ents", __import__("detect").Entities([])).items)
                _px = getattr(benv, "_last_ents").player_pos()[0]
                _spikes = [e for e in _ents if int(e.cls) == 0]
                _shown = 0
                _edge_txt = ""
                # NEAREST SPIKE FIRST: sorted by edge gap, not x. The spike
                # that kills you is the smallest edge, even if another box
                # sits further right.
                _order = sorted(range(min(6, len(_ents))),
                                key=lambda i: ((_ents[i].x - _px) - (_ents[i].w / 2 + 0.02)))
                _COL = {0: (0, 255, 0), 1: (0, 255, 255), 2: (255, 255, 0), 3: (255, 0, 255),
                        4: (0, 165, 255), 5: (255, 0, 255), 6: (128, 128, 128)}
                for _rank, _i in enumerate(_order):
                    e = _ents[_i]
                    x0 = max(0, int((e.x - e.w / 2) * W0 * zoom))
                    y0 = max(0, int((e.y - e.h / 2) * H0 * zoom))
                    x1 = min(W0 * zoom - 1, int((e.x + e.w / 2) * W0 * zoom))
                    y1 = min(H0 * zoom - 1, int((e.y + e.h / 2) * H0 * zoom))
                    col = _COL.get(int(e.cls), (255, 255, 255))
                    edge = e.x - _px - (e.w / 2 + 0.02)
                    if _rank == 0 and int(e.cls) == 0:
                        # NEAREST SPIKE: thick double border + filled label
                        # so the lethal spike reads at a glance.
                        cv2.rectangle(big0, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), (0, 255, 0), 3)
                        cv2.rectangle(big0, (x0, y0), (x1, y1), (0, 0, 0), 1)
                        lbl = f"SPIKE {edge:+.2f}"
                        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(big0, (x0, max(0, y0 - th - 6)), (x0 + tw + 4, y0), (0, 255, 0), -1)
                        cv2.putText(big0, lbl, (x0 + 2, max(10, y0 - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        _edge_txt = f" edge={edge:+.2f}"
                    else:
                        cv2.rectangle(big0, (x0, y0), (x1, y1), col, max(1, zoom // 2))
                        cv2.putText(big0, _CL[int(e.cls)] if int(e.cls) < len(_CL) else str(int(e.cls)),
                                    (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
                    _shown += 1
                big = big0
                if _shown == 0:
                    cv2.putText(big, "NO ENTS", (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    _edge_txt = " (clear)"
                # SPIKE zoom crop: bottom-right inset of the nearest spike at
                # 3x, so you can confirm the detector sees the triangle.
                try:
                    if _spikes:
                        _s = min(_spikes, key=lambda e: (e.x - _px) - (e.w / 2 + 0.02))
                        _fx0 = max(0, int((_s.x - _s.w) * W0))
                        _fy0 = max(0, int((_s.y - _s.h) * H0))
                        _fx1 = min(W0, int((_s.x + _s.w) * W0))
                        _fy1 = min(H0, int((_s.y + _s.h) * H0))
                        _crop = frame[max(0, _fy0 - 10):_fy1 + 10, max(0, _fx0 - 10):_fx1 + 10]
                        if _crop.size:
                            _ch, _cw = 60 * zoom // 2, 60 * zoom // 2
                            _zin = cv2.resize(_crop[:, :, ::-1], (_cw, _ch), interpolation=cv2.INTER_NEAREST)
                            big[H0 * zoom - _ch - 4:H0 * zoom - 4, W0 * zoom - _cw - 4:W0 * zoom - 4] = _zin
                            cv2.rectangle(big, (W0 * zoom - _cw - 4, H0 * zoom - _ch - 4),
                                          (W0 * zoom - 4, H0 * zoom - 4), (0, 255, 0), 1)
                except Exception:
                    pass
            except Exception:
                big = cv2.resize(frame[:, :, ::-1], (320 * zoom, 180 * zoom))
                _edge_txt = ""
            hold = state["act"] == 1
            if hold:
                cv2.rectangle(big, (0, 0), (big.shape[1] - 1, big.shape[0] - 1), (0, 255, 0), 3)
            _n = 0
            _ns = 0
            try:
                _items = getattr(benv, "_last_ents", __import__("detect").Entities([])).items
                _n = len(_items)
                _ns = sum(1 for _e in _items if int(_e.cls) == 0)
            except Exception:
                pass
            try:
                _spike_txt = f" SPIKES={_ns}" if _ns else " NO-SPIKE"
            except Exception:
                _spike_txt = ""
            cv2.putText(big, f"BOX act={state['act']} n={_n}{_spike_txt}{_edge_txt} d={info.get('attempts', 1) - 1}",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if _gui_state.get("paused"):
                cv2.putText(big, "PAUSED (SPACE to resume)", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("BOX policy view", big)
            _key = cv2.waitKey(1) & 0xFF
            if _key == 32:  # SPACE on the GUI window: pause/resume training
                if _gui_state.get("paused"):
                    _gui_state["paused"] = False
                    print("RESUMED [space] (training un-paused)")
                else:
                    _gui_state["paused"] = True
                    print("PAUSED [space] (hit SPACE again to resume)")
                import time as _tp
                state["space_ts"] = _tp.perf_counter()
            if _key in (ord("q"), 27):
                return False
            return True

    from pause_cb import DeathPauseCallback as _DPC
    _gui_state: dict = {}
    # Shared callback kept for PAUSE/RESUME + header stats; GRADIENTS run in
    # the manual loop below (collect -> death? train partial immediately).
    _shared = _DPC(lambda: base.env, lambda: model, _gui_state)
    _best = BestPercentCallback(save_path="./checkpoints/best_boxes",
                                check_freq=2048, verbose=1).callback
    from stable_baselines3.common.callbacks import CallbackList as _CL2
    _gui = Gui()
    cb = _CL2([_shared, _gui, _best])
    print("BOX live: manual truncate-train loop (collect->death?train->resume). Gui renders every tick.")
    import torch as _thm
    from stable_baselines3.common.utils import obs_as_tensor as _oat
    _gui_state["manual_updates"] = 0
    try:
        import numpy as _np
        try:
            _r = vec.reset()
            _obs = _r[0] if isinstance(_r, tuple) else _r
        except Exception:
            _obs = None
        model._last_obs = _obs
        model._last_episode_starts = _np.ones((vec.num_envs,), dtype=bool)
        buf = model.rollout_buffer
        while model.num_timesteps < int(args.timesteps):
            buf.reset()
            _n = 0
            _dead = False
            _peak = 0.0  # best x% reached this attempt
            while _n < model.n_steps and model.num_timesteps < int(args.timesteps):
                # SPACE-pause (GUI window): freeze collection, keep rendering
                # so the window stays responsive. Game stays ESC-paused.
                try:
                    import time as _tpu
                    while _gui_state.get("paused"):
                        _disp = getattr(base, "last_display", None)
                        _fr = _np.ascontiguousarray(_disp[-1]).copy() if _disp is not None else _np.zeros((180, 320, 3), _np.uint8)
                        _bigp = cv2.resize(_fr[:, :, ::-1], (320 * zoom, 180 * zoom))
                        cv2.putText(_bigp, "PAUSED (SPACE to resume)", (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.imshow("BOX policy view", _bigp)
                        if (cv2.waitKey(50) & 0xFF) == 32:
                            _gui_state["paused"] = False
                            print("RESUMED [space]")
                            break
                except Exception:
                    pass
                with _thm.no_grad():
                    _ot = _oat(_obs, model.device)
                    _acts, _vals, _lps = model.policy(_ot)
                _an = _acts.cpu().numpy()
                _no, _rw, _dn, _in = vec.step(_an)
                model.num_timesteps += vec.num_envs
                buf.add(model._last_obs, _an.reshape(vec.num_envs, -1), _rw,
                        model._last_episode_starts, _vals, _lps)
                model._last_obs = _no
                model._last_episode_starts = _dn
                _n += 1
                _info0 = (_in[0] if isinstance(_in, (list, tuple)) else {})
                try:
                    _peak = max(_peak, float(_info0.get("prog") or 0.0))
                except Exception:
                    pass
                try:  # the manual loop bypasses learn(), so drive the overlay ourselves
                    _gui.locals = {"infos": [_info0], "actions": _an}
                    if _gui._on_step() is False:
                        raise KeyboardInterrupt
                except KeyboardInterrupt:
                    raise
                except Exception as _ge:
                    if not _gui_state.get("ov_err"):
                        _gui_state["ov_err"] = True
                        print(f"overlay failed (shown once): {_ge!r}")
                try:
                    state["act"] = int(__import__("numpy").asarray(_an).flat[0])
                except Exception:
                    pass
                if str(_info0.get("event")) == "death":
                    _dead = True
                    try:
                        base.env.pause()
                        print(f"PAUSED on death verified={bool(base.env._paused)} (partial {_n})")
                    except Exception as _e:
                        print(f"PAUSE FAILED: {_e}")
                    break
                _obs = _no
            if _n < 8:
                if _dead:
                    try:
                        base.env.mark_trained()
                        base.env.resume()
                    except Exception as _e:
                        print(f"RESUME FAILED: {_e}")
                try:
                    _r = vec.reset()
                    _obs = _r[0] if isinstance(_r, tuple) else _r
                except Exception:
                    pass
                model._last_obs = _obs
                model._last_episode_starts = _np.ones((vec.num_envs,), dtype=bool)
                continue
            if not _dead:
                try:
                    base.env.pause()
                    print(f"PAUSED (no death in {_n} steps) verified={bool(base.env._paused)}")
                except Exception as _e:
                    print(f"PAUSE FAILED: {_e}")
            with _thm.no_grad():  # bootstrap from the NEWEST obs (masked to 0 on death)
                _lv = model.policy.predict_values(_oat(_no, model.device))
            buf.buffer_size = _n
            buf.compute_returns_and_advantage(last_values=_lv, dones=model._last_episode_starts)
            buf.full = True
            _oe, _ob = model.n_epochs, model.batch_size
            model.n_epochs, model.batch_size = 4, min(128, _n)
            try:
                model.train()
            finally:
                model.n_epochs, model.batch_size = _oe, _ob
                buf.buffer_size = model.n_steps
            try:
                _lg = getattr(getattr(model, "logger", None), "name_to_value", {})
                _gui_state["best"] = max(_gui_state.get("best", 0.0), _peak)
                _gui_state["attempt_n"] = _gui_state.get("attempt_n", 0) + 1
                print(f"[attempt {_gui_state['attempt_n']}] survived={_n} steps ({_n/60:.1f}s) "
                      f"peak={_peak*100:.0f}% best={_gui_state['best']*100:.0f}%")
                print(f"TRAINED death partial={_n} pg{float(_lg.get('train/policy_gradient_loss', 0.0) or 0.0):+.4f} "
                      f"ent{float(_lg.get('train/entropy_loss', 0.0) or 0.0):+.3f} @upd={int(model.num_timesteps)}")
            except Exception:
                pass
            try:
                model.save("./checkpoints/preview_boxes")
                # disk I/O every death (~2s train each): keep it, it's the
                # crash-recovery guarantee. Async would risk corruption.
            except Exception:
                pass
            try:
                base.env.mark_trained()
            except Exception:
                pass
            try:
                base.env.resume()
                print("RESUMED +1s settle")
            except Exception as _e:
                print(f"RESUME FAILED: {_e}")
            try:
                buf.reset()
            except Exception:
                pass
            _gui_state["manual_updates"] = _gui_state.get("manual_updates", 0) + 1
            if _dead:
                # Fresh baseline for the NEXT attempt: clear the box env's
                # entity memory so the first spike reads as new, not stale.
                try:
                    benv.reset()
                except Exception:
                    pass
                try:
                    _r = vec.reset()
                    _obs = _r[0] if isinstance(_r, tuple) else _r
                except Exception:
                    pass
                model._last_episode_starts = _np.ones((vec.num_envs,), dtype=bool)
            else:  # attempt still running: continue it
                _obs = _no
                model._last_episode_starts = _np.zeros((vec.num_envs,), dtype=bool)
            model._last_obs = _obs
    except KeyboardInterrupt:
        print("interrupted, saving...")
    model.save("./checkpoints/preview_boxes")
    print("saved ./checkpoints/preview_boxes.zip")
    vec.close()
    cap.stop()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())