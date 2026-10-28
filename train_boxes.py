"""Cross-level PPO: MlpPolicy on entity boxes (NOT pixels).

python train_boxes.py --backend heuristic --timesteps 500000

General-data training: pass --levels stereo jumper finger to rotate the
source recording/live level every --rotate-min minutes. The policy never
sees backgrounds -- only (16,5) entity vectors -- so rotation only
changes WHICH boxes appear, forcing reactive play over memorization.
"""
from __future__ import annotations

import argparse
import os


def main() -> int:
    p = argparse.ArgumentParser(description="cross-level box PPO")
    p.add_argument("--timesteps", type=int, default=500000)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--backend", default="heuristic", choices=("heuristic", "yolo"))
    p.add_argument("--levels", nargs="*", default=[],
                   help="level names to rotate for general data (live play)")
    p.add_argument("--rotate-min", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--debug", action="store_true",
                    help="every 200 steps: n_ents, obs[0], action, reward")
    p.add_argument("--logdir", default="./logs_boxes")
    p.add_argument("--ckpt", default="./checkpoints/best_boxes")
    args = p.parse_args()

    from stable_baselines3 import PPO
    import torch as _th  # noqa: F401
    _dev = getattr(args, "device", "auto")
    _dev = _dev if _dev != "auto" else "cpu"
    print(f"device={_dev} torch={_th.__version__}" + (f" gpu={_th.cuda.get_device_name(0)}" if _dev == 'cuda' else ''))

    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from env import EnvConfig, GymGDEnv, SmallObsWrapper
    from env_boxes import EntityBoxEnv
    from pct import PctReader

    # NOTE: SmallObsWrapper stays in the chain ONLY for its pause/resume/
    # settle plumbing and GUI display cache; EntityBoxEnv sits OUTSIDE it
    # and replaces the CNN input with boxes. Pixels never reach the policy.
    _rdr = PctReader()
    _rdr.load_templates()
    base = SmallObsWrapper(GymGDEnv(EnvConfig(), pct_reader=_rdr))
    benv = EntityBoxEnv(base, backend=args.backend)
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    vec = DummyVecEnv([lambda: Monitor(benv, os.path.join(args.logdir, "monitor.csv"))])

    from env import BestPercentCallback
    from pause_cb import DeathPauseCallback
    gui_state: dict = {}
    cb_pause = DeathPauseCallback(lambda: base.env, lambda: model, gui_state)
    from stable_baselines3.common.callbacks import CallbackList
    cb = CallbackList([cb_pause,
        BestPercentCallback(save_path=args.ckpt, check_freq=2048, verbose=1).callback])

    model = PPO("MlpPolicy", vec, learning_rate=3e-4, n_steps=2048,
                batch_size=512, n_epochs=10, gamma=0.95, gae_lambda=0.9,
                clip_range=0.15, ent_coef=0.05, vf_coef=1.0,
                max_grad_norm=0.5, verbose=1, tensorboard_log="./tb_boxes",
                policy_kwargs=dict(net_arch=[128, 128]), device=_dev)
    print(f"box PPO MlpPolicy 6x8 lr=3e-4 levels={args.levels or 'live-only'}")
    if args.debug:
        from stable_baselines3.common.callbacks import BaseCallback as _BC
        _dbg_n = {"n": 0}
        class _Dbg(_BC):
            def _on_step(self) -> bool:
                _dbg_n["n"] += 1
                if _dbg_n["n"] % 200 == 0:
                    try:
                        ino = self.locals.get("infos", [{}])[0]
                        import numpy as _np
                        _o = self.locals.get("new_obs")
                        _row = _np.asarray(_o[0][0]).round(3).tolist() if _o is not None else None
                        print(f"[debug] step={_dbg_n['n']} n_ents={ino.get('n_entities')} obs0={_row} "
                              f"act={self.locals.get('actions')} rew={self.locals.get('rewards')}")
                    except Exception as _e:
                        print(f"[debug] failed: {_e}")
                return True
        from stable_baselines3.common.callbacks import CallbackList as _CL
        cb = _CL([cb, _Dbg()])
    try:
        model.learn(total_timesteps=args.timesteps, callback=cb)
    except KeyboardInterrupt:
        print("interrupted, saving...")
    model.save(args.ckpt + "_last")
    print(f"saved {args.ckpt}_last.zip")
    vec.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
