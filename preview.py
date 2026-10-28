"""Live GUI preview for 1.1: shows exactly what the policy sees.
 
Layout (recomputed every frame from the current zoom):
  Big (full panel): latest 320x180 RGB frame, NEAREST-upscaled.
  Header          : gui_fps, tick, exact tick_fps, drift vs wall clock,
                    latency, frames/drops, restarts, shape. Frame count is
                    exact: every read() = 1 tick, ticks == frames + drops.

The capture loop runs in THIS thread at a hard 60Hz. Rendering runs every
tick (60Hz) against preallocated buffers so imshow never allocates in the
hot loop. Obs is (4,180,320,3) RGB; the 4-frame stack is still
captured/stacked in code (see capture.GDCapture) — only the newest frame
is drawn.

Scalable GUI:
  - Window is WINDOW_NORMAL | WINDOW_KEEPRATIO: drag any edge to resize.
  - Zoom keys +/- (or scroll): 1x..6x upscale of the 320x180 view.
  - F toggles fullscreen. --scale sets the starting zoom.

Controls: Q/ESC quit. R restart capture. +/- zoom. F fullscreen.
  Region: --region l,t,r,b (default: full screen).
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from capture import CaptureConfig, GDCapture
from gd import ProgressReward
from pct import PctReader


def _np_asarray(x):
    return np.asarray(x)


def _np_zeros_frame():
    return np.zeros((180, 320, 3), dtype=np.uint8)

WIN = "GD capture preview (policy view)"
HDR_H = 44  # two short header lines so nothing clips


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GD capture preview (policy view)")
    p.add_argument("--region", default=None,
                   help="crop region l,t,r,b e.g. 0,0,1920,1080 (default: full screen)")
    p.add_argument("--output-idx", type=int, default=0)
    p.add_argument("--scale", type=int, default=2, choices=(1, 2, 3, 4, 5, 6),
                   help="starting zoom factor (1-6, change live with +/-)")
    p.add_argument("--train", action="store_true",
                   help="run PPO training live in the GUI: model acts + learns, you watch what it sees")
    p.add_argument("--model", default=None,
                   help="path to .zip to resume (default: fresh CnnPolicy)")
    p.add_argument("--timesteps", type=int, default=1000000)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--train-steps", type=int, default=128,
                    help="gradient steps per death-pause (default 128, ~0.5s)")
    return p.parse_args()


# Preallocated per-zoom buffers so the 60Hz hot loop never allocates.
# NOTE: the 4-frame stack still exists in capture.GDCapture (policy input);
# we only stop *drawing* the right-side tiles. obs[-1] is the newest frame.
_PANEL_CACHE: dict[int, dict] = {}


def _get_buffers(zoom: int, big_w: int, big_h: int) -> dict:
    e = _PANEL_CACHE.get(zoom)
    strip_h = 22  # mask debug strip under the game view
    shape = (HDR_H + big_h + strip_h, big_w, 3)
    if e is None or e["panel"].shape != shape:
        e = {
            "panel": np.zeros(shape, dtype=np.uint8),
            "rgb": np.empty((big_h, big_w, 3), dtype=np.uint8),
        }
        _PANEL_CACHE.clear()  # one zoom live at a time; drop stale sizes
        _PANEL_CACHE[zoom] = e
    return e


def build_panel(obs: np.ndarray, zoom: int, header: str,
                died: bool = False, attempts: int = 1,
                deaths: int = 0,
                fill: float | None = None, pending: bool = False,
                bar_box: tuple | None = None, pct_box: tuple | None = None,
                mask: np.ndarray | None = None,
                flash: float = 0.0, holding: bool = False) -> np.ndarray:
    """Compose the BGR panel at the current zoom. Nearest = true pixels.

    Obs is (N,H,W,3) RGB; cv2 wants BGR so swap R<->B in place.
    Visual-only: draws newest frame fullscreen-width. Stacking (obs) is
    untouched in capture.py -- callers still pass the full (N,H,W,3) stack.
    Zero per-frame allocs: resize writes into a cached RGB buffer, then
    2 channel blits + middle passthrough fill BGR. Text uses LINE_8 (AA
    off) which is ~2x faster and indistinguishable at 0.55 scale.
    died draws the fullscreen red wash (flash 0..1 alpha) + persistent
    DEATHS counter top-right + DIED banner. fill shows the TEXT % readout
    (cyan), pending tints it yellow; bar_box dim debug, pct_box is the
    primary reward ROI (cyan, 2px).
    """
    H, W = obs.shape[1], obs.shape[2]  # 180, 320
    big_w, big_h = W * zoom, H * zoom
    bufs = _get_buffers(zoom, big_w, big_h)
    panel = bufs["panel"]
    rgb = bufs["rgb"]

    try:
        newest = obs[-1]
    except Exception:
        newest = _np_zeros_frame()
    try:
        _nd = _np_asarray(newest).ndim
    except Exception:
        newest = _np_zeros_frame()
        _nd = 3
    if _nd == 2:
        # Back-compat: old grayscale obs (N,H,W) still renders.
        cv2.resize(newest, (big_w, big_h), dst=rgb[:, :, 0],
                   interpolation=cv2.INTER_NEAREST)
        rgb[:, :, 1] = rgb[:, :, 0]
        rgb[:, :, 2] = rgb[:, :, 0]
    else:
        # NEAREST upscale in place -- no temp allocation. Wrong-channel
        # frames (e.g. transposed 3ch post-update) fall back to newest-frame
        # reinterpret instead of crashing the tick.
        try:
            cv2.resize(newest, (big_w, big_h), dst=rgb,
                       interpolation=cv2.INTER_NEAREST)
        except Exception:
            try:
                _nz = _np_asarray(newest)
                if _nz.ndim == 3 and _nz.shape[0] in (3, 4, 12):
                    _nz = _nz.transpose(1, 2, 0)
                if _nz.ndim == 3 and _nz.shape[2] not in (3, 4):
                    _nz = _nz.reshape(_nz.shape[0], _nz.shape[1], -1)[:, :, :3]
                cv2.resize(_nz[:, :, :3], (big_w, big_h), dst=rgb,
                           interpolation=cv2.INTER_NEAREST)
            except Exception:
                rgb[:] = 0
    # RGB->BGR via in-place R<->B channel blits (no cvtColor alloc).
    body = panel[HDR_H:HDR_H + big_h]
    body[:, :, 0] = rgb[:, :, 2]
    body[:, :, 1] = rgb[:, :, 1]
    body[:, :, 2] = rgb[:, :, 0]
    # Clear header strip (kills text ghosting) + fast non-AA text.
    # Header is "line1|line2": two compact rows, nothing clipped.
    panel[0:HDR_H] = 0
    parts = header.split("|", 1)
    cv2.putText(panel, parts[0], (8, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_8)
    if len(parts) > 1:
        cv2.putText(panel, parts[1], (8, 37), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_8)
    # --- tracked-ROI highlight: % box PRIMARY (cyan), bar box debug (dim) --- #
    if pct_box is not None:
        (px0, py0, px1, py1) = (int(v) for v in pct_box)
        pcol = (0, 255, 255) if pending else (255, 255, 0)
        cv2.rectangle(panel, (px0, HDR_H + py0), (px1, HDR_H + py1), pcol, 2)
        cv2.putText(panel, "PCT", (px0, HDR_H + py0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, pcol, 1, cv2.LINE_8)
    if bar_box is not None:
        (x0, y0, x1, y1) = (int(v) for v in bar_box)
        col = (0, 255, 255) if pending else (80, 80, 80)
        cv2.rectangle(panel, (x0, HDR_H + y0), (x1, HDR_H + y1), col, 1)
    if fill is not None:
        fcol = (0, 255, 255) if pending else (255, 255, 0)
        cv2.putText(panel, f"{fill*100:5.1f}%", (10, HDR_H + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, fcol, 2, cv2.LINE_8)
    # --- persistent death counter: always visible top-right --- #
    cv2.putText(panel, f"DEATHS {deaths}  ATT {attempts}", (big_w - 210, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_8)
    # --- mask debug strip: what the white-bar gate sees (white = counts) --- #
    strip_y = HDR_H + big_h
    panel[strip_y:strip_y + 22] = 0
    cv2.putText(panel, "mask", (4, strip_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_8)
    if mask is not None and mask.size:
        m = (mask > 0).astype(np.uint8) * 255
        mh, mw = m.shape[:2]
        dx = min(big_w - 60, mw * zoom)
        dy = min(18, mh * zoom)
        if dx > 0 and dy > 0:
            small = cv2.resize(m, (dx, dy), interpolation=cv2.INTER_NEAREST)
            panel[strip_y + 2:strip_y + 2 + dy, 56:56 + dx, 0] = small
            panel[strip_y + 2:strip_y + 2 + dy, 56:56 + dx, 1] = small
            panel[strip_y + 2:strip_y + 2 + dy, 56:56 + dx, 2] = small
    if holding and not (died or flash > 0.0):
        # AI input LIVE: green border + corner ticks so a hold reads
        # instantly even mid-flash-free gameplay. Death red takes priority.
        gh = max(2, zoom) * 2
        panel[HDR_H:HDR_H+gh, :] = (0, 255, 0)
        panel[HDR_H+big_h-gh:HDR_H+big_h, :] = (0, 255, 0)
        panel[HDR_H:, :gh] = (0, 255, 0)
        panel[HDR_H:, big_w-gh:] = (0, 255, 0)
    if died or flash > 0.0:
        # Fullscreen red wash, alpha = flash (1.0 fresh -> 0.0 expired).
        # In-place blend, no alloc: body = body*(1-a) + red*a.
        a = min(1.0, max(0.0, flash if flash > 0.0 else 1.0))
        body = panel[HDR_H:HDR_H + big_h]
        body[:, :, 0] = (body[:, :, 0].astype(np.float32) * (1.0 - a)).astype(np.uint8)
        body[:, :, 1] = (body[:, :, 1].astype(np.float32) * (1.0 - a)).astype(np.uint8)
        body[:, :, 2] = (body[:, :, 2].astype(np.float32) * (1.0 - a * 0.55)).astype(np.uint8)
        body[:, :, 2] = np.clip(body[:, :, 2].astype(np.int16)
                                + int(255 * a * 0.55), 0, 255).astype(np.uint8)
        t = max(2, zoom) * 2
        panel[HDR_H:HDR_H+t, :] = (0, 0, 255)
        panel[HDR_H+big_h-t:HDR_H+big_h, :] = (0, 0, 255)
        panel[HDR_H:, :t] = (0, 0, 255)
        panel[HDR_H:, big_w-t:] = (0, 0, 255)
        cv2.putText(panel, f"DIED x{attempts}", (10, HDR_H+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_8)
    return panel


def main() -> int:
    args = parse_args()
    region = None
    if args.region:
        try:
            l, t, r, b = (int(v) for v in args.region.split(","))
            region = (l, t, r, b)
        except ValueError:
            print("bad --region, want l,t,r,b")
            return 2
    zoom = int(args.scale)
    cfg = CaptureConfig(region=region, output_idx=args.output_idx)
    cap = GDCapture(cfg)
    try:
        cap.start()
    except Exception as e:  # noqa: BLE001
        print(f"capture failed to start: {e}")
        return 1

    # Resizable + aspect-kept window: user can drag to ANY size.
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(WIN, 320 * zoom, 180 * zoom + HDR_H)
    for _ in range(10):  # warm up 4-frame stack in capture.py (still stacked, just not drawn)
        cap.read()

    fullscreen = False
    latest_obs: np.ndarray = cap.read()
    gui_fps = 60.0  # EMA of imshow rate, shown in header
    _last_show = time.perf_counter()
    rew = ProgressReward()
    flash_until = 0.0
    tick_dbg = 0
    pct = PctReader()
    pct.load_templates()
    pct.load_cnn() if hasattr(pct, "load_cnn") else None
    pct_val: float | None = None

    # --- optional live training: SB3 OWNS the loop, GUI only DISPLAYS ---
    # Root cause of "holding instead of learning": the old GUI called
    # predict() itself AND stepped a second env per tick (double capture
    # reads = double input + double reward), while a hand-rolled learn_buf
    # list never ran a gradient step. Now SB3 drives: DummyVecEnv(GymGDEnv)
    # owns cap+input+reward, model.learn() runs the rollout + PPO update,
    # and a render callback imshows + handles keys every tick so you see
    # exactly what it sees WHILE weights update.
    model = None
    vec = None
    gui_state: dict = {"act": None, "rinfo": {}, "died": False,
                       "flash_until": 0.0, "paused": False, "zoom": int(args.scale),
                       "fullscreen": False, "training": False}
    if args.train:
        from stable_baselines3 import PPO
        import torch as _th
        _dev = args.device if getattr(args, 'device', 'auto') != 'auto' else ('cuda' if _th.cuda.is_available() else 'cpu')
        print(f"device={_dev} torch={_th.__version__}" + (f" gpu={_th.cuda.get_device_name(0)}" if _dev == 'cuda' else ''))
        from stable_baselines3.common.callbacks import BaseCallback
        from env import EnvConfig, GymGDEnv
        from pct import PctReader as _PR
        import gymnasium as _gym  # noqa: F401 (SB3 cold-start import)
        _rdr = _PR()
        _rdr.load_templates()
        from env import SmallObsWrapper
        _env = SmallObsWrapper(GymGDEnv(EnvConfig(), cap=cap, pct_reader=_rdr))
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        import os as _os
        _os.makedirs("./logs", exist_ok=True)
        _os.makedirs("./checkpoints", exist_ok=True)
        vec = DummyVecEnv([lambda: Monitor(_env, "./logs/monitor.csv")])
        if args.model:
            model = PPO.load(args.model, env=vec)
            print(f"resumed {args.model} on LIVE env")
        else:
            # Death-gated training: n_steps=256 seals an episode-sized
            # rollout; the _DeathGate below pauses (ESC) on every counted
            # death, runs the PPO update OFF-GRID (no gameplay lag), then
            # resumes (SPACE). Update cadence = death cadence, never a
            # fixed 2048-clock. Generous entropy (0.02) + small net keep
            # early attempts exploring instead of camping SPACE.
            model = PPO("CnnPolicy", vec, learning_rate=1e-3, n_steps=512,
                        batch_size=128, n_epochs=4, gamma=0.95, gae_lambda=0.9,
                        clip_range=0.15, ent_coef=0.05, vf_coef=0.5,
                        max_grad_norm=0.5, verbose=1, tensorboard_log="./tb", device=_dev)
            print("fresh CnnPolicy on LIVE env: SB3 learns, GUI displays. T=pause, M=save.")
        gui_state["training"] = True
        _learn_total = int(args.timesteps)

        class _GuiRender(BaseCallback):
            """Display-only: SB3 already stepped the env + chose the action.

            locals: actions (what the policy chose THIS tick), new_obs
            (resulting HWC stack), rewards, dones, infos (reward/death).
            NO cap.read() here (that would double-step the 60Hz grid), NO
            predict() (that would double-act). Pure render + keys.
            """

            def _on_step(self) -> bool:
                import time as _t
                import numpy as _np
                import traceback as _tb
                try:
                    return self._render_tick(_t, _np)
                except Exception as _e:
                    print(f"render tick failed (training continues): {_e}")
                    _tb.print_exc()
                    try:
                        _k = cv2.waitKey(1) & 0xFF
                        if _k in (ord("q"), ord("Q"), 27):
                            return False
                    except Exception:
                        pass
                    return True

            def _render_tick(self, _t, _np) -> bool:
                gs = gui_state
                infos = self.locals.get("infos", [{}])
                info = infos[0] if infos else {}
                new_obs = self.locals.get("new_obs")
                latest = None
                if new_obs is not None:
                    try:
                        _arr = _np.asarray(new_obs[0])
                        if _arr.ndim == 3 and _arr.shape in ((84, 84, 12), (84, 84, 13), (84, 84, 4),
                                                           (12, 84, 84), (13, 84, 84), (4, 84, 84)):
                            # TRAIN input is 84x84; GUI must show the LIVE
                            # 320x180 COLOR capture, never an upscale of the
                            # train tensor. Fall through to cap.read() below.
                            latest = None
                        elif _arr.ndim == 3 and _arr.shape == (180, 320, 12):
                            latest = _arr.reshape(180, 320, 4, 3).transpose(2, 0, 1, 3).copy()
                        elif _arr.ndim == 3 and _arr.shape == (12, 180, 320):
                            latest = _arr.transpose(1, 2, 0).reshape(180, 320, 4, 3).transpose(2, 0, 1, 3).copy()
                        elif _arr.ndim == 4 and _arr.shape[1:] == (180, 320, 3):
                            latest = _np.ascontiguousarray(_arr)
                        elif _arr.ndim == 4 and _arr.shape[1:] == (3, 180, 320):
                            latest = _np.ascontiguousarray(_arr.transpose(0, 2, 3, 1))
                        else:
                            latest = None
                    except Exception:
                        latest = None
                if latest is None:
                    # GUI = live 320x180 COLOR capture. Prefer the wrapper's
                    # cached display stack (same tick, zero extra reads);
                    # NEVER cap.read() here (that eats the env's 60Hz tick).
                    try:
                        _disp = None
                        try:
                            _inner = vec.envs[0]
                            for _ in range(4):
                                if hasattr(_inner, "last_display"):
                                    _disp = getattr(_inner, "last_display")
                                    break
                                _inner = getattr(_inner, "env", None)
                                if _inner is None:
                                    break
                        except Exception:
                            _disp = None
                        if _disp is not None and _np.asarray(_disp).ndim == 4 and _np.asarray(_disp).shape[1:] == (180, 320, 3):
                            latest = _np.ascontiguousarray(_disp)
                        else:
                            latest = _np.zeros((4, 180, 320, 3), dtype=_np.uint8)
                    except Exception:
                        latest = _np.zeros((4, 180, 320, 3), dtype=_np.uint8)
                gs["act"] = None
                try:
                    _a = self.locals.get("actions")
                    if _a is not None:
                        gs["act"] = int(_np.asarray(_a).flat[0])
                except Exception:
                    pass
                try:
                    _ev = info.get("event")
                    _is_death = (str(_ev) == "death") if not isinstance(_ev, bool) else bool(_ev)
                except Exception:
                    _is_death = False
                if _is_death:
                    gs["flash_until"] = _t.perf_counter() + 0.2
                    try:
                        _att = int(_np.asarray(info.get("attempts", 1)).flat[0])
                    except Exception:
                        _att = 1
                    try:
                        _bf = float(_np.asarray(info.get("best_fill", 0)).flat[0])
                    except Exception:
                        _bf = 0.0
                    try:
                        _tot = float(_np.asarray(info.get("total", 0)).flat[0])
                    except Exception:
                        _tot = 0.0
                    print(f"DIED attempt {_att} best {_bf:.2f} total {_tot:.2f} "
                          f"upd={int(getattr(model, 'num_timesteps', 0))}")
                gs["died"] = _t.perf_counter() < gs["flash_until"]
                flash = max(0.0, (gs["flash_until"] - _t.perf_counter()) / 0.2) if gs["died"] else 0.0
                zoom = gs["zoom"]
                pv = info.get("best_fill", info.get("prog", 0.0))
                try:
                    pvf = float(_np.asarray(pv).flat[0]) if pv is not None else 0.0
                except Exception:
                    pvf = 0.0
                try:
                    upd = int(getattr(model, "num_timesteps", 0))
                except Exception:
                    upd = 0
                # PPO entropy: read from the rollout buffer tensors when the
                # update just ran, else show last known.
                # Learning proof: SB3 logger holds the last PPO update's
                # train/entropy_loss + train/policy_gradient_loss once the
                # first 2048-step rollout trains. Zeros before that = still
                # collecting the first rollout (random acts expected).
                try:
                    _lg = getattr(getattr(model, "logger", None), "name_to_value", {})
                    _ent = float(_lg.get("train/entropy_loss", 0.0) or 0.0)
                    _pg = float(_lg.get("train/policy_gradient_loss", 0.0) or 0.0)
                    _vl = float(_lg.get("train/value_loss", 0.0) or 0.0)
                except Exception:
                    _ent = _pg = _vl = 0.0
                # _att2 MUST be computed before the header that uses it.
                try:
                    _att2 = int(_np.asarray(info.get("attempts", 1)).flat[0])
                except Exception:
                    _att2 = 1
                _lt = gs.get("last_train") or {}
                _tms = f"{_lt.get('loss_ms', 0):.0f}ms" if _lt else "--"
                try:
                    _paused_now = False
                    _ii = vec.envs[0]
                    for _ in range(5):
                        if hasattr(_ii, "_paused"):
                            _paused_now = bool(_ii._paused)
                            break
                        _ii = getattr(_ii, "env", None)
                        if _ii is None:
                            break
                except Exception:
                    _paused_now = False
                _pz = "PAUSED" if _paused_now else ("SETTLE" if (gs.get("last_train") and _paused_now) else "LIVE")
                hdr = (f"DEATH-TRAIN upd={upd}/{_learn_total} [{_pz}] last={_tms} ent{_ent:+.4f} pg{_pg:+.4f} vl{_vl:.4f} "
                       f"act={gs['act']} best={pvf * 100:4.1f}% "
                       f"d{_att2 - 1} a{_att2}"
                       f"|pause-train-+1s-resume")
                _hold = bool(gs.get("act") == 1)
                cv2.imshow(WIN, build_panel(
                    latest, zoom, hdr, gs["died"], _att2, _att2 - 1,
                    fill=pvf, pending=False,
                    bar_box=None, pct_box=None, mask=None, flash=flash,
                    holding=_hold))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    return False
                if key in (ord("m"), ord("M")):
                    try:
                        model.save("./checkpoints/preview_model")
                        print("saved ./checkpoints/preview_model.zip")
                    except Exception as _e:
                        print(f"save failed: {_e}")
                if key in (ord("+"), ord("=")):
                    gs["zoom"] = min(6, gs["zoom"] + 1)
                if key in (ord("-"), ord("_")):
                    gs["zoom"] = max(1, gs["zoom"] - 1)
                if key in (ord("f"), ord("F")):
                    gs["fullscreen"] = not gs["fullscreen"]
                    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if gs["fullscreen"] else cv2.WINDOW_NORMAL)
                return True

        from env import seal_and_train

        class _DeathGate(BaseCallback):
            _prev_att: int = 1
            """PAUSE(ESC) on death -> RESUME(SPACE) at next sealed update.

            Ordering constraint (verified against SB3 2.9.2 source):
            collect_rollouts calls on_rollout_end BEFORE train(), so the
            gate cannot run gradients in that hook -- there is nothing to
            train yet. Contract instead:
            - _on_step: counted death -> env.pause() IMMEDIATELY (game
              frozen mid-rollout, SPACE released, reward suspended). The
              60Hz grid is NOT blocked: SB3 keeps collecting frozen ticks
              (reward 0, suspended) until the buffer seals.
            - _on_rollout_start: fires right AFTER train() on the sealed
              buffer. If paused-for-death, env.resume() (SPACE) and print
              the fresh pg/ent/vl. Update cadence still lands on deaths
              (death seals the episode that triggers the update), all
              gradients run on SB3's natural sealed path -- no GAE hacks,
              no lag mid-gameplay, no weird stutter.
            """

            def _on_step(self) -> bool:
                try:
                    infos = self.locals.get("infos", [{}])
                    info = infos[0] if infos else {}
                    # Two independent death signals, either pauses:
                    #   (a) reward info event == 'death' (pct0 counter)
                    #   (b) attempts counter increased by exactly 1 since the
                    #       last tick (reads the SAME counter the GUI shows,
                    #       so a death the GUI counts ALWAYS pauses here too,
                    #       even if the event field was dropped by wrapping).
                    _dead = (str(info.get("event")) == "death")
                    try:
                        _att_n = int(_np_asarray(info.get("attempts", self._prev_att)).flat[0])
                    except Exception:
                        try:
                            _att_n = int(info.get("attempts", self._prev_att))
                        except Exception:
                            _att_n = self._prev_att
                    if _att_n > self._prev_att:
                        _dead = True
                    self._prev_att = max(self._prev_att, _att_n)
                    if _dead:
                        # EVERY counted death pauses: call the DIRECT
                        # GymGDEnv reference (_env.env) -- no unwrapping.
                        try:
                            _env.env.pause()
                            _now = bool(_env.env._paused)
                            print(f"PAUSED on death @upd={int(getattr(model, 'num_timesteps', 0))} "
                                  f"verified={_now} (training at rollout seal)")
                        except Exception as _e:
                            print(f"PAUSE FAILED: {_e}")
                except Exception as _e:
                    print(f"death-step failed: {_e}")
                return True

            def _on_rollout_start(self) -> None:
                try:
                    # DIRECT env ref, no unwrapping.
                    _inner = _env.env
                    if _inner._paused:
                        try:
                            lg = getattr(getattr(model, "logger", None), "name_to_value", {})
                            gui_state["last_train"] = {
                                "loss_ms": 0.0,
                                "pg": float(lg.get("train/policy_gradient_loss", 0.0) or 0.0),
                                "vl": float(lg.get("train/value_loss", 0.0) or 0.0),
                                "ent": float(lg.get("train/entropy_loss", 0.0) or 0.0)}
                            print(f"TRAINED death upd={int(getattr(model, 'num_timesteps', 0))} "
                                  f"pg{gui_state['last_train']['pg']:+.4f} "
                                  f"ent{gui_state['last_train']['ent']:+.3f}")
                        except Exception:
                            pass
                        try:
                            _inner.mark_trained()  # settle 1.0s from NOW
                            print(f"SETTLE 1.0s @upd={int(getattr(model, 'num_timesteps', 0))}")
                        except Exception:
                            pass
                        try:
                            model.save("./checkpoints/preview_model")
                        except Exception:
                            pass
                        try:
                            _inner.resume()  # blocks settle_s (1.0s: train->wait->SPACE) inside
                            print(f"RESUMED +1.0s settle @upd={int(getattr(model, 'num_timesteps', 0))}")
                        except Exception as _e:
                            print(f"RESUME FAILED: {_e}")
                except Exception as _e:
                    print(f"death-resume failed: {_e}")

        _gui_cb = _GuiRender()
        from env import BestPercentCallback
        _best_cb = BestPercentCallback(save_path="./checkpoints/best_model",
                                       check_freq=2048, verbose=1).callback
        from stable_baselines3.common.callbacks import CallbackList
        _cb = CallbackList([_DeathGate(), _gui_cb, _best_cb])
        print("preview --train: SB3 learn() driving, GUI rendering every tick")
        try:
            model.learn(total_timesteps=_learn_total, callback=_cb)
        except KeyboardInterrupt:
            print("interrupted, saving...")
        except Exception as _e:
            import traceback as _tb
            print(f"learn() crashed after {int(getattr(model, 'num_timesteps', -1))} steps: {_e}")
            _tb.print_exc()
        try:
            model.save("./checkpoints/preview_model")
            print("saved ./checkpoints/preview_model.zip (best stays at ./checkpoints/best_model.zip)")
        except Exception as _e:
            print(f"final save failed: {_e}")
        try:
            vec.close()
        except Exception:
            pass
        try:
            cap.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        import sys as _sys
        _sys.exit(0)
    training = False
    print("preview running: drag edges to resize | +/- zoom | F fullscreen | "
          "R restart | P pause(suspend death) | Q quit"
          + (" | T train ON/OFF | M save model" if args.train else ""))
    paused = False
    try:
        while True:
            latest_obs = cap.read()  # hard-locked to 60Hz inside

            # Render EVERY tick: buffer reuse keeps this ~1-2ms, so the
            # 60Hz capture grid never slips.
            h = cap.health
            now = time.perf_counter()
            dt = now - _last_show
            if dt > 0:
                inst = 1.0 / dt
                gui_fps += 0.05 * (inst - gui_fps)  # fast EMA, stable readout
            _last_show = now
            if paused:
                rew.suspend()
                pct_tup = (None, False, 999)
            else:
                # PHASE 2: 60Hz text-first read -> (accepted, fresh, age).
                pct_tup = pct.tick(cap.last_fullres)
            dbg: dict = {}
            _, _, rinfo = rew.step(latest_obs, pct=pct_tup, _mask_out=dbg)
            # Plain preview (no --train): local reward + act display only.
            # With --train we never reach here (SB3 owns the loop above).
            act = None
            if rinfo.get("event") == "death":
                # deaths/attempts are the SINGLE SOURCE OF TRUTH on rew
                # (see ProgressReward._commit_death): GUI only reads them.
                # Fullscreen red wash for exactly 0.2s from the counted tick.
                flash_until = time.perf_counter() + 0.2
                print(f"DIED attempt {rew.attempts} deaths {rew.deaths} "
                      f"pct {rinfo.get('prog', 0.0):.2f} via {rinfo.get('via')}")
            tick_dbg += 1
            if (rinfo.get("pending") or rinfo.get("snap_pending")
                    or rinfo.get("snap")) and tick_dbg % 30 == 0:
                print(f"arm? prog {rinfo.get('prog', 0.0):.2f} "
                      f"fill {rinfo.get('fill', 0.0):.2f} "
                      f"diff {rinfo.get('diff', 0.0):.3f} src {rinfo.get('src')}")
            now2 = time.perf_counter()
            died = now2 < flash_until
            flash = max(0.0, (flash_until - now2) / 0.2) if died else 0.0
            H0, W0 = latest_obs.shape[1], latest_obs.shape[2]
            bx0, bx1 = int(W0 * rew.cfg.bar_x0), int(W0 * rew.cfg.bar_x1)
            by0, by1 = int(H0 * rew.cfg.bar_y0), int(H0 * rew.cfg.bar_y1)
            # % box: prefer the AUTO-LOCATED bar interior (resolution-free,
            # zero config); fall back to config fractions when not found.
            bb = pct.health.get("bar_box")
            if bb is not None:
                qx0, qy0, qx1, qy1 = (int(W0 * bb[0]), int(H0 * bb[1]),
                                      int(W0 * bb[2]), int(H0 * bb[3]))
            else:
                qx0 = int(W0 * pct.cfg.x0)
                qx1 = int(W0 * pct.cfg.x1)
                qy0 = int(H0 * pct.cfg.y0)
                qy1 = int(H0 * pct.cfg.y1)
            pv = rinfo.get("prog")
            if act is not None:
                pv = pv  # keep prog display; action shown below
            raw = pct.health.get("raw")
            fr = "F" if rinfo.get("pct_fresh") else "S"
            age = rinfo.get("pct_age", pct.health.get("age", 0))
            raw_s = f"{raw*100:4.1f}" if raw is not None else "----"
            pend = ("SNAP" if rinfo.get("snap_pending") or rinfo.get("snap")
                    else ("ARM" if rinfo.get("pending") else ""))
            frz = "FRZ" if rinfo.get("frozen") else ""
            act_s = f" act={act}" if act is not None else ""
            train_s = " TRAIN" if training else ""
            hdr = (f"{gui_fps:4.1f}fps pct {pv*100:4.1f}%({fr}{age}) raw {raw_s} "
                   f"d{rew.deaths} a{rew.attempts} df{rinfo.get('diff', 0.0):.3f}{pend}{frz}{act_s}{train_s}"
                   f"|t{h['tick']} {h['tick_fps']:4.1f}Hz p95 {h['lat_p95_ms']:3.0f}ms "
                   f"f{h['frames']} x{h['drops']} z{zoom}x")
            _plain_hold = bool(act == 1)
            cv2.imshow(WIN, build_panel(
                latest_obs, zoom, hdr, died, rew.attempts, rew.deaths,
                fill=pv, pending=bool(rinfo.get("pending")),
                bar_box=(bx0 * zoom, by0 * zoom, bx1 * zoom, by1 * zoom),
                pct_box=(qx0 * zoom, qy0 * zoom, qx1 * zoom, qy1 * zoom),
                mask=dbg.get("mask"), flash=flash, holding=_plain_hold))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("r"), ord("R")):
                cap.restart("user keypress")
                print("capture restarted")
            elif key in (ord("p"), ord("P")):
                paused = not paused
                if paused:
                    rew.suspend()
                    print("paused: death detection suspended")
                else:
                    rew.unsuspend()
                    print("resumed: baseline re-synced")
            elif key in (ord("t"), ord("T")):
                if model is not None:
                    training = not training
                    print(f"training {'ON' if training else 'OFF'} (model acts live)")
            elif key in (ord("m"), ord("M")):
                if model is not None:
                    model.save("./checkpoints/preview_model")
                    print("saved ./checkpoints/preview_model.zip")
            elif key in (ord("+"), ord("=")):
                zoom = min(6, zoom + 1)
            elif key in (ord("-"), ord("_")):
                zoom = max(1, zoom - 1)
            elif key in (ord("f"), ord("F")):
                fullscreen = not fullscreen
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key == 82:  # Up arrow: zoom in (opencv special key, rarely fires with waitKey(1))
                zoom = min(6, zoom + 1)
            elif key == 84:  # Down arrow: zoom out
                zoom = max(1, zoom - 1)
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
        cv2.destroyAllWindows()
        h = cap.health
        n = h["tick"]
        ok = (h["frames"] + h["drops"] == n)
        print(f"soak: ticks={n} frames={h['frames']} drops={h['drops']} "
              f"exact={'OK' if ok else 'MISMATCH!'} restarts={h['restarts']} "
              f"tick_fps={h['tick_fps']:.2f} drift={h['tick_drift']:+d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
