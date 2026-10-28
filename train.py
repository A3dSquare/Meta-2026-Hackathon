"""2.2 PPO config (PLAN.md S8): SB3 CnnPolicy, single env, 60Hz GD loop.

python train.py --timesteps 1000000
Wires: make_vec_env (DummyVecEnv + Monitor) -> PPO CnnPolicy ->
model.learn() with BestPercentCallback. PctReader injected as pct_reader
so text flows every tick. Auto-Retry ON = free resets.
"""
from __future__ import annotations

import argparse
import os


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1000000)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--n-steps", type=int, default=2048, help="1024-2048 per spec")
    p.add_argument("--batch", type=int, default=256, help="256-512 per spec")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--logdir", default="./logs")
    p.add_argument("--ckpt", default="./checkpoints/best_model")
    args = p.parse_args()

    from stable_baselines3 import PPO
    import torch as _th
    _dev = args.device if getattr(args, 'device', 'auto') != 'auto' else ('cuda' if _th.cuda.is_available() else 'cpu')
    print(f"device={_dev} torch={_th.__version__}" + (f" gpu={_th.cuda.get_device_name(0)}" if _dev == 'cuda' else ''))

    from env import EnvConfig, make_vec_env, BestPercentCallback
    from pct import PctReader

    rdr = PctReader()
    n = rdr.load_templates()
    print(f"templates: {n} (text-first reader active)")

    vec = make_vec_env(EnvConfig(), monitor_dir=args.logdir,
                       env_kwargs={"pct_reader": rdr})
    cb = BestPercentCallback(save_path=args.ckpt, check_freq=2048, verbose=1)

    model = PPO(
        "CnnPolicy",
        vec,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tb",
        device=_dev,
    )
    print(f"PPO CnnPolicy lr={args.lr} clip=0.1 ent=0.01 vf=0.5 "
          f"n_steps={args.n_steps} batch={args.batch} -> learn {args.timesteps}")
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    try:
        model.learn(total_timesteps=args.timesteps, callback=cb.callback)
    except KeyboardInterrupt:
        print("interrupted, saving last...")
    model.save(args.ckpt + "_last")
    print(f"saved {args.ckpt}_last.zip (best stays at {args.ckpt}.zip)")
    vec.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
