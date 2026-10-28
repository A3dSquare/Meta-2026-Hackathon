"""Headless live training of the cross-level box policy (no GUI window).

    python train_boxes.py                          # fresh brain, Double DQN
    python train_boxes.py --model checkpoints/preview_boxes.pt   # resume

Same loop as preview_boxes.py (which adds the live overlay). Setup: GD
foreground, practice OFF, Auto-Retry ON, the level open and playing.
"""
from __future__ import annotations

import argparse
import os


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", default="heuristic", choices=("heuristic", "yolo"))
    p.add_argument("--model", default=None, help="checkpoint to resume (.pt for dqn, .zip for ppo)")
    p.add_argument("--algo", default="dqn", choices=("dqn", "ppo"))
    p.add_argument("--steps", "--timesteps", dest="steps", type=int, default=3_000_000)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    p.add_argument("--frame-skip", type=int, default=2,
                   help="game frames per decision (2 = 30 decisions/s)")
    p.add_argument("--no-hires", action="store_true",
                   help="detect on 320x180 only (faster, misses mini spikes)")
    p.add_argument("--death-lag", default="auto",
                   help="'auto' (measure camera freeze) or seconds between crash and %% reset")
    p.add_argument("--settle", type=float, default=1.0,
                   help="seconds to wait after training before un-pausing GD")
    p.add_argument("--update-steps", type=int, default=128,
                   help="train after a death once this many new steps were played")
    p.add_argument("--replay-ratio", type=float, default=4.0)
    p.add_argument("--eps", type=float, default=None,
                   help="override starting exploration (default 0.15 fresh / checkpoint's value)")
    p.add_argument("--name", default="preview_boxes", help="checkpoint name in ./checkpoints")
    p.add_argument("--minutes", type=float, default=0,
                   help="stop cleanly (save + release keys) after this many minutes (0 = run until q)")
    p.add_argument("--jump-margin", type=float, default=0.3,
                   help="only jump when it's this confident jumping beats not jumping (0 = off)")
    p.add_argument("--hold-cost", type=float, default=0.02,
                   help="tiny penalty per step SPACE is held (0 = off)")
    p.add_argument("--icon-hue", type=int, nargs=2, default=(55, 95), metavar=("LO", "HI"),
                   help="OpenCV hue range (0-179) of your icon's main colour; default green/cyan. "
                        "e.g. yellow 20 35, orange 8 20, blue 95 125, pink 140 170, red 0 8")


def build(args):
    """Capture + % reader + GymGDEnv + EntityBoxEnv + Trainer config."""
    from capture import CaptureConfig, GDCapture
    from env import EnvConfig, GymGDEnv
    from env_boxes import EntityBoxEnv
    from pct import PctReader
    from trainer import TrainConfig

    dev = args.device
    if dev == "auto":
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            dev = "cpu"
    cap = GDCapture(CaptureConfig())
    cap.start()
    for _ in range(10):
        cap.read()
    rdr = PctReader()
    n_t = rdr.load_templates()
    if not n_t:
        print("WARNING: no %-digit templates loaded -> deaths can't be detected. "
              "Run from the repo folder (needs ./templates).")
    base = GymGDEnv(EnvConfig(max_steps=10**9), cap=cap, pct_reader=rdr)
    base.settle_s = float(args.settle)
    lag = None if str(args.death_lag).lower() == "auto" else float(args.death_lag)
    from detect import make_detector
    det = make_detector(args.backend, icon_hue=tuple(args.icon_hue))
    env = EntityBoxEnv(base, backend=args.backend, detector=det, frame_skip=args.frame_skip,
                       hires=not args.no_hires, death_lag_s=lag)
    cfg = TrainConfig(algo=args.algo, total_steps=args.steps, device=dev, name=args.name,
                      update_steps=args.update_steps, replay_ratio=args.replay_ratio,
                      jump_margin=args.jump_margin, hold_cost=args.hold_cost)
    if args.eps is not None:
        cfg.eps_start = float(args.eps)
    print(f"algo={cfg.algo} device={dev} frame_skip={args.frame_skip} hires={not args.no_hires} "
          f"death_lag={args.death_lag} detector={args.backend}")
    return cap, base, env, cfg


def main() -> int:
    p = argparse.ArgumentParser(description="headless live box-policy training")
    add_common_args(p)
    args = p.parse_args()
    from trainer import Trainer
    cap, base, env, cfg = build(args)
    os.makedirs("./checkpoints", exist_ok=True)
    import time as _time
    _t_end = _time.time() + args.minutes * 60 if args.minutes > 0 else None
    tr = Trainer(env, cfg, model_path=args.model,
                 on_step=(lambda *_: _time.time() < _t_end) if _t_end else None)
    if args.eps is not None and cfg.algo == "dqn":
        tr.learner.steps_seen = 0
        tr.best_prog = -1.0   # --eps = new level: start a fresh best %
    try:
        tr.run()
    finally:
        try:
            base.inp.close()
        except Exception:
            pass
        cap.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
