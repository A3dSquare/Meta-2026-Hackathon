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
    p.add_argument("--backend", default="heuristic", choices=("heuristic", "yolo", "yolo-only"),
                   help="heuristic = built-in rules; yolo = rules + YOLO for what they miss; "
                        "yolo-only = everything from YOLO incl. the icon's mode (needs yolo_gd.pt)")
    p.add_argument("--model", default=None, help="checkpoint to resume (.pt for dqn, .zip for ppo)")
    p.add_argument("--algo", default="dqn", choices=("dqn", "ppo"))
    p.add_argument("--steps", "--timesteps", dest="steps", type=int, default=3_000_000)
    p.add_argument("--device", default="auto", choices=("cpu", "cuda", "auto"))
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
    p.add_argument('--cpu-threads',type=int,default=1,
                   help='PyTorch threads for the CPU brain (default 1; 0 keeps library default)')
    p.add_argument('--route-controller',action='store_true',
                   help='new route-conditioned mode controllers, learned motion, persistent replay and evaluation gate; separate checkpoint')
    capture_mode=p.add_mutually_exclusive_group()
    capture_mode.add_argument('--threaded-capture',action='store_true',
                   help='EXPERIMENTAL frame-reader thread; can worsen freshness under Python CPU load (default off)')
    capture_mode.add_argument('--sync-capture',action='store_true',
                   help='explicitly select the default synchronous capture path')
    p.add_argument("--eps", type=float, default=None,
                   help="override starting exploration (default 0.15 fresh / checkpoint's value)")
    p.add_argument("--name", default="preview_boxes", help="checkpoint name in ./checkpoints")
    p.add_argument("--minutes", type=float, default=0,
                   help="stop cleanly (save + release keys) after this many minutes (0 = run until q)")
    p.add_argument("--jump-margin", type=float, default=0.3,
                   help="only jump when it's this confident jumping beats not jumping (0 = off)")
    p.add_argument('--flight-margin', type=float, default=0.,
                   help='separate ship/wave thrust threshold (default 0; jump-margin applies to other modes)')
    p.add_argument('--flight-buffer', type=float, default=.75,
                   help='ship/wave trajectory clearance in blocks (0 disables buffered flight shield; default .75, max 2)')
    p.add_argument('--flight-teacher-weight', type=float, default=.05,
                   help='coaching from verified safe ship/wave interventions (default .05, 0 disables; max .2)')
    p.add_argument("--hold-cost", type=float, default=0.02,
                   help="tiny penalty per step SPACE is held (0 = off)")
    from flight_reward import clearance_weight
    p.add_argument('--clearance-reward', nargs='?', const=.05, default=0., type=clearance_weight,
                   help='optional ship/wave clearance potential (default off; flag alone = 0.05; range 0..0.1)')
    p.add_argument("--play", action="store_true",
                   help="beat the level with what it knows: NO training -- every attempt follows the "
                        "best run saved for this level, practises the spot it died at, and the brain "
                        "(no random moves) plays past it. Needs --model (and the route memory)")
    p.add_argument("--ms-delay", action="store_true",
                   help="blue line / shield use the input delay measured in ms (not whole steps); "
                        "experimental -- off by default")
    p.add_argument("--frontier", action="store_true",
                   help="train where it gets stuck: every other attempt replays the best run saved for "
                        "this level to ~3 s before its furthest point and the brain plays and LEARNS "
                        "from there (the replayed part isn't learned from); the other attempts are "
                        "normal training from the start")
    p.add_argument("--trust-replay", action="store_true",
                   help="--landmarks: while the landmarks are locked, replay the remembered presses exactly "
                        "(the safety shield doesn't edit them). Use with a route built from YOUR run "
                        "(import_human_route.py)")
    p.add_argument("--no-frontier-explore", action="store_true",
                   help="--play: don't vary the brain's moves near the end of the best run (it then "
                        "repeats the same moves there every attempt)")
    p.add_argument("--ghost", action="store_true",
                   help="--landmarks: follow the remembered PATH (the icon's height along the level) in cube "
                        "sections instead of replaying the remembered presses; learns per stretch from deaths "
                        "(timing shifts, or back to the press replay where it keeps dying). Other modes replay")
    p.add_argument("--guide", nargs="?", type=float, const=8.0, default=None, metavar="MARGIN",
                   help="--landmarks: the brain plays and the remembered route guides it -- where they "
                        "disagree the brain wins when it clearly prefers its choice (Q difference >= MARGIN, "
                        "default 8), else the route. The safety shield checks both as usual")
    p.add_argument("--trust-replay-ab", action="store_true",
                   help="test --trust-replay fairly: every other attempt trusts the replay, the others "
                        "keep the shield on it (logged per attempt in route_timing.jsonl)")
    p.add_argument("--no-early-press", action="store_true",
                   help="shield: don't press a step early when waiting only works by pressing at the "
                        "very next step. That rule covers a planner that thinks presses land sooner than "
                        "they do; with --ms-delay the timing is measured, and at tight gaps (a triple "
                        "spike under hanging spikes) it jumped a decision too early")
    p.add_argument("--no-death-pause", action="store_true",
                   help="don't ESC-pause the game after each death while the bot saves its logs, "
                        "route memory and checkpoints (that work then lags the next attempt's start)")
    p.add_argument("--strict-offline-gate", action="store_true",
                   help="route controller: the old promotion gate (a candidate must also match the "
                        "held-out human labels and TD error). Default: it only has to be no less safe "
                        "and predict motion no worse, then the live A/B comparison on the level decides")
    p.add_argument("--no-route", action="store_true",
                   help="turn off the route memory (per-level practice of spots it keeps dying at, routes/)")
    p.add_argument("--routes", default="routes", help="folder the route memory keeps its levels in")
    p.add_argument("--level", default=None,
                   help="route folder to use (e.g. level_20261103_172553) instead of recognising the level "
                        "from its first frames -- recognition can miss and start a new, empty route")
    p.add_argument("--landmarks", action="store_true",
                   help="replay the best run's presses by WHERE the level is (spikes / blocks ahead) instead of "
                        "the clock; needs the route memory (not --no-route). Planner shield still vetoes. "
                        "Saved to <routes>/<level>/landmarks.json")
    p.add_argument("--fast-steps", action="store_true",
                   help="decide as soon as 2 game frames have passed even if processing ran long "
                        "(more decisions/s, but their length varies; off = the tested behaviour)")
    p.add_argument("--death-snaps", type=int, default=30,
                   help="save what the AI saw before each death to logs_boxes/sessions/<session>/deaths/ "
                        "(keeps the newest N; 0 = off)")
    p.add_argument("--no-shield", action="store_true",
                   help="don't let the physics preview veto actions it predicts will crash")
    p.add_argument("--shield", action="store_true",
                   help="force the physics shield ON (overrides evidence-based auto-off for trusted modes)")
    p.add_argument("--motion-fix", action="store_true",
                   help="also use the learned physics correction (experimental; off by default)")
    p.add_argument("--hazards", default="hazards",
                   help="folder of object pictures: deadly ones, learned/ (from deaths), safe/ "
                        "(see hazards.py; 'none' = off). Learning from deaths needs --death-snaps")
    p.add_argument("--motion-model", default="checkpoints/motion_model.npz",
                   help="learned physics correction + tuned jump + input delay (shared by all "
                        "brains and levels); 'none' = don't load or save one")
    p.add_argument("--icon-hue", type=int, nargs=2, default=(55, 95), metavar=("LO", "HI"),
                   help="OpenCV hue range (0-179) of your icon's main colour; default green/cyan. "
                        "e.g. yellow 20 35, orange 8 20, blue 95 125, pink 140 170, red 0 8")


def build(args):
    """Capture + % reader + GymGDEnv + EntityBoxEnv + Trainer config."""
    if str(args.motion_model).lower() != 'none' and not os.path.isfile(args.motion_model):
        raise ValueError(f'Physics calibration not found: {args.motion_model}. '
                         'Pass --motion-model with your existing motion_model.npz. '
                         'Use --motion-model none only to deliberately run without saved calibration.')
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
    if args.cpu_threads < 0:
        raise ValueError('--cpu-threads must be nonnegative')
    cap = GDCapture(CaptureConfig(prefetch=args.threaded_capture))
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
    # Brain device only: YOLO keeps its own auto (GPU/TensorRT engine) path so an
    # explicit --device cpu never forces the detector onto the CPU.
    det = make_detector(args.backend, icon_hue=tuple(args.icon_hue),
                        device=(dev if str(dev) == "cuda" else None))
    env = EntityBoxEnv(base, backend=args.backend, detector=det, frame_skip=args.frame_skip,
                       hires=not args.no_hires, death_lag_s=lag,
                       motion_path=None if str(args.motion_model).lower() == "none" else args.motion_model,
                       motion_fix=bool(args.motion_fix))
    env.fast_steps = bool(getattr(args, "fast_steps", False))
    if args.death_snaps > 0:
        env.death_dir = os.path.join("logs_boxes", "deaths")
        env.death_keep = int(args.death_snaps)
    if str(args.hazards).lower() != "none":
        from hazards import HazardMemory   # pictures of deadly / harmless objects (hazards.py)
        env.hazmem = HazardMemory(args.hazards)
        if env.hazmem.active:
            print(f"[hazards] {len(env.hazmem.templates)} object picture(s) in {args.hazards}/ in use")
    if getattr(args, "trust_replay", False):
        import trainer as _trainer_mod
        _trainer_mod.TRUST_REPLAY = True
        print("[landmarks] --trust-replay: locked replay goes out as remembered (no shield edits)")
    if getattr(args, "no_frontier_explore", False):
        import trainer as _trainer_mod
        _trainer_mod.FRONTIER_EXPLORE = 0.0
        _trainer_mod.FRONTIER_EVERY = 0
    if getattr(args, "ghost", False):
        import landmark_route as _lm_mod
        _lm_mod.GHOST = True
        print("[landmarks] --ghost: cube sections follow the remembered path (learns per stretch from deaths)")
    if getattr(args, "guide", None) is not None:
        import trainer as _trainer_mod
        _trainer_mod.GUIDE_MARGIN = float(args.guide)
        print(f"[landmarks] --guide {args.guide:g}: the brain plays; the route decides only where the brain "
              f"is unsure (Q difference < {args.guide:g})")
    if getattr(args, "trust_replay_ab", False):
        import trainer as _trainer_mod
        _trainer_mod.TRUST_REPLAY_AB = True
        print("[landmarks] --trust-replay-ab: odd attempts replay as remembered, even ones shielded")
    if getattr(args, "no_early_press", False):
        import trainer as _trainer_mod
        _trainer_mod.EARLY_PRESS = False
        print("[shield] --no-early-press: no pressing a step early at tight timings")
    cfg = TrainConfig(algo=args.algo, total_steps=args.steps, device=dev, name=args.name,cpu_threads=args.cpu_threads,
                      update_steps=args.update_steps, replay_ratio=args.replay_ratio,
                      jump_margin=args.jump_margin, hold_cost=args.hold_cost,
                      clearance_weight=args.clearance_reward,
                      flight_margin=args.flight_margin,flight_buffer=args.flight_buffer,
                      flight_teacher_weight=args.flight_teacher_weight,
                      shield=not args.no_shield, route=not args.no_route or args.play or args.frontier,
                      route_dir=args.routes, route_level=getattr(args, 'level', None),
                      landmarks=bool(getattr(args, "landmarks", False)),
                      play=bool(args.play), frontier=bool(args.frontier) and not args.play,
                      strict_offline_gate=bool(getattr(args, "strict_offline_gate", False)),
                      pause_on_death=not bool(getattr(args, "no_death_pause", False)))
    if args.route_controller:
        if cfg.algo!='dqn':raise ValueError('--route-controller currently requires --algo dqn')
        if getattr(args,'shield',False):cfg.shield_auto_off=False  # explicit --shield wins over evidence
        from route_controller import RouteObservationEnv
        cfg.route_controller=True
        if cfg.name=='preview_boxes':cfg.name='route_controller'
        # Conservative candidate updates. Deployed weights are protected by
        # the held-out gate; optimizer/replay/exploration persist on restart.
        cfg.lr=1e-4;cfg.buffer_size=20_000;cfg.replay_ratio=min(cfg.replay_ratio,.5)
        cfg.max_grad_steps=128;cfg.jump_margin=0.;cfg.flight_margin=0.
        env=RouteObservationEnv(env)
        print('[controller] separate route_controller checkpoint; planner bootstrap until each mode passes validation')
    if getattr(args, "ms_delay", False):
        import env_boxes as _eb
        _eb.USE_MS_DELAY = True
        print("--ms-delay: the blue line uses the input delay in ms")
    if args.clearance_reward:
        print(f'--clearance-reward: ship/wave potential weight {args.clearance_reward:g}'
              + (' (ignored in --play)' if args.play else '; base survival/death rewards unchanged'))
    if args.frontier and args.play:
        print("--frontier and --play: --play wins (no training)")
    elif args.frontier:
        print("--frontier: every other attempt starts the brain just before its furthest point")
    if args.play:
        if args.no_route:
            print("--play uses the route memory: ignoring --no-route")
        if not args.model:
            print("--play without --model: a fresh brain only plays where the route doesn't reach yet")
        print("--play: beating the level, NOT training (no learning, no pauses after deaths)")
    if args.eps is not None:
        cfg.eps_start = float(args.eps)
    print(f"algo={cfg.algo} device={dev} frame_skip={args.frame_skip} hires={not args.no_hires} "
          f"death_lag={args.death_lag} detector={args.backend}")
    print(f'[flight] buffer={cfg.flight_buffer:g} blocks; thrust margin={cfg.flight_margin:g}; '
          f'coach weight={cfg.flight_teacher_weight:g} (requires verified physics)')
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
    import winfocus

    def on_step(*_):
        if _t_end and _time.time() >= _t_end:
            return False
        cmd = winfocus.user_command()          # F8 = pause / resume, F9 = save + quit
        if cmd == "quit":
            print("F9: saving and stopping")
            return False
        if cmd == "pause":
            env.pause()
            print("PAUSED -- press F8 to continue (F9 to quit)")
            while True:
                _time.sleep(0.05)
                c2 = winfocus.user_command()
                if c2 == "quit":
                    return False
                if c2 == "pause":
                    break
            print("resuming in 1 s")
            _time.sleep(1.0)
            env.mark_trained()
            env.resume()
        return True

    print("F8 = pause / resume, F9 = save + quit (from any window)")
    tr = Trainer(env, cfg, model_path=args.model, on_step=on_step)
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