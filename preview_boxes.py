"""Live training with an overlay window (the main way to train).

    python preview_boxes.py                                  # fresh brain
    python preview_boxes.py --model checkpoints/preview_boxes.pt   # resume
    python preview_boxes.py --model checkpoints/best_boxes.pt --eps 0.05   # new level, same brain

Window: the 640x360 frame the detector sees, with
  red = spike, gray = wall/block, yellow/blue/pink = orbs, orange = pad,
  magenta = portal (labelled), green = player,
  thick box = nearest hazard ahead, bar on the left = how much the policy
  wants to jump right now, green border = SPACE held.
  Predicted paths: thick green = if it jumps now, thin yellow = if it jumps
  in 4 steps, thin magenta = if it keeps holding (ship / wave / robot), white
  dots = if it doesn't press (follows ledges / falls); red X = that path
  crashes there.
Keys in the window: q / ESC quit (saves).  p pause/unpause training (GD
gets ESC; click back into GD within 2 s after unpausing).

Console per attempt:   [attempt N] died after 3.4s peak=12% best=18% (cut 31 post-crash steps, measured)
Console per update:    [update N] after death: grad_steps=.. loss=.. q=.. eps=..
Learning curve: logs_boxes/attempts.csv (one row per attempt).
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

_COL = {0: (0, 0, 255), 1: (0, 255, 255), 2: (255, 160, 0), 3: (255, 0, 255),
        4: (0, 165, 255), 5: (255, 0, 200), 6: (170, 170, 170), 7: (0, 255, 0)}


def render(env, tr, a: int, info: dict, scale: float) -> np.ndarray:
    from detect import CLASSES
    from gd_physics import MODE_ID
    hi = env.last_hires
    if hi is not None:
        img = hi.copy()
    else:
        img = cv2.resize(np.ascontiguousarray(env.last_frame[:, :, ::-1]), (640, 360),
                         interpolation=cv2.INTER_NEAREST)
    H, W = img.shape[:2]
    ents = env.last_ents
    px, _ = ents.player_pos()
    ahead = [e for e in ents.items if e.cls in (0, 6) and e.x + e.w / 2 > px]
    near = min(ahead, key=lambda e: e.x - e.w / 2) if ahead else None
    for e in ents.items:
        x0, y0 = int((e.x - e.w / 2) * W), int((e.y - e.h / 2) * H)
        x1, y1 = int((e.x + e.w / 2) * W), int((e.y + e.h / 2) * H)
        c = _COL.get(int(e.cls), (255, 255, 255))
        cv2.rectangle(img, (x0, y0), (x1, y1), c, 3 if e is near else 1)
        if e is near or int(e.cls) in (1, 2, 3, 4, 5):
            cv2.putText(img, CLASSES[int(e.cls)], (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
    # physics preview (in the current mode): thin magenta = keep holding (ship / wave /
    # robot), white dots = don't press, thick cyan = the best line the search found
    # (what the safety shield goes by),
    # red X = where that path would crash
    try:
        from env_boxes import BLOCK_W, BLOCK_H
        ppx, ppy = ents.player_pos()
        for name, hit, pts in getattr(env, "last_arcs", []):
            # (green "tap now" / yellow "tap in 4" are no longer drawn: the shield goes by the
            # cyan search line; the brain still gets those plans as inputs)
            if name not in ("release", "hold", "best") or len(pts) < 2:
                continue
            if name == "hold" and getattr(getattr(env, "mode_est", None), "mode", "cube") not in ("ship", "wave", "robot"):
                continue
            P = np.array([[(ppx + x * BLOCK_W) * W, (ppy - y * BLOCK_H) * H] for x, y in pts], np.int32)
            col = {"tap now": (0, 255, 0), "tap in 4": (0, 255, 255), "hold": (255, 0, 255),
                   "best": (255, 255, 0)}.get(name, (255, 255, 255))
            if name == "release":
                for q in P[::3]:
                    cv2.circle(img, tuple(int(v) for v in q), 2, col, -1)
            else:
                thick = name in ("tap now", "best")
                cv2.polylines(img, [P], False, (0, 0, 0), 4 if thick else 3, cv2.LINE_AA)   # outline
                cv2.polylines(img, [P], False, col, 2 if thick else 1, cv2.LINE_AA)
            if hit:                                   # red X where that path would crash
                ex, ey = (int(v) for v in P[-1])
                cv2.line(img, (ex - 5, ey - 5), (ex + 5, ey + 5), (0, 0, 255), 2)
                cv2.line(img, (ex - 5, ey + 5), (ex + 5, ey - 5), (0, 0, 255), 2)
    except Exception as ex:                     # say why once instead of silently drawing nothing
        if not getattr(render, "_warned", False):
            render._warned = True
            print(f"[overlay] couldn't draw the predicted paths: {ex!r}")
    for q in getattr(env, "_lines", None) or ():   # detected ground / ceiling line
        if q is not None:
            cv2.line(img, (0, int(q * H)), (W - 1, int(q * H)), (255, 200, 120), 1)
    if a == 1:
        cv2.rectangle(img, (0, 0), (W - 1, H - 1), (0, 255, 0), 3)
    # jump-urge bar
    try:
        ph = tr.hold_prob(tr._last_obs)
    except Exception:
        ph = 0.0
    bh = int(ph * (H - 40))
    cv2.rectangle(img, (4, H - 10 - bh), (14, H - 10), (0, 255, 0) if ph > 0.5 else (0, 200, 255), -1)
    cv2.rectangle(img, (4, 30), (14, H - 10), (255, 255, 255), 1)
    ns = sum(1 for e in ents.items if e.cls == 0)
    eps = getattr(tr.learner, "epsilon", lambda: 0.0)()
    spd = (env.speed or 10.38) / 10.38
    pl = ents.player()
    icon = f"{pl.w / 0.0527:.1f}x{pl.h / 0.0938:.1f}" if pl is not None else "--"
    me = getattr(env, "mode_est", None)
    mtxt = (me.mode + (" mini" if me.mini else "") + (" UPSIDE-DOWN" if me.g < 0 else "")) if me else "?"
    ph_ = getattr(env, "phys", None)
    if ph_ is not None and me is not None:   # tuned jump (blocks / seconds) + measured input delay (steps)
        mtxt += f"  jump {ph_.apex:.1f}b/{ph_.airtime:.2f}s{'*' if ph_.n_cal else ''}  delay {me.lag}"
        mfx = getattr(env, "motion_fix", None)
        if mfx is not None and mfx.n:   # how much the learned correction is used in this mode
            mtxt += f"  fix {mfx.m_trust[MODE_ID.get(me.mode, 0)]:.0%}"
    txt = (f"att {tr.attempts + 1}  best {max(tr.best_prog, 0) * 100:.0f}%  {mtxt}  spikes {ns}  "
           f"speed {spd:.1f}x  icon {icon}  eps {eps:.2f}  "
           f"{'MOVING' if info.get('moving') else 'FROZEN'}  upd {tr.updates}")
    cv2.rectangle(img, (0, 0), (W, 22), (0, 0, 0), -1)
    cv2.putText(img, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    if scale != 1.0:
        img = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_NEAREST)
    return img


def main() -> int:
    from train_boxes import add_common_args, build
    from trainer import Trainer

    p = argparse.ArgumentParser(description="box-policy live GUI + train")
    add_common_args(p)
    p.add_argument("--scale", type=float, default=1.5, help="overlay window zoom")
    p.add_argument("--render-every", type=int, default=1,
                   help="draw every N decisions (drawing costs several ms; >1 makes the decision "
                        "length uneven, so only together with --fast-steps)")
    args = p.parse_args()
    cap, base, env, cfg = build(args)
    cv2.namedWindow("BOX policy view", cv2.WINDOW_NORMAL)
    state = {"n": 0}
    t_end = time.time() + args.minutes * 60 if args.minutes > 0 else None

    import winfocus

    def pause_until_resumed() -> bool:
        """Pause GD + training until F8 (any window) or p (this window). False = quit."""
        env.pause()
        print("PAUSED -- press F8 (from any window) or p (in this window) to continue, F9 / q to quit")
        while True:
            k = cv2.waitKey(50) & 0xFF
            cmd = winfocus.user_command()
            if cmd == "quit" or k in (ord("q"), 27):
                return False
            if cmd == "pause" or k == ord("p"):
                break
        print("resuming in 1 s (Geometry Dash is brought back to the front automatically)")
        time.sleep(1.0)
        env.mark_trained()
        env.resume()
        return True

    def on_step(tr, obs, a, info):
        tr._last_obs = obs
        state["n"] += 1
        if t_end is not None and time.time() >= t_end:
            print(f"--minutes {args.minutes} reached: saving and stopping")
            return False
        cmd = winfocus.user_command()          # F8 / F9 work from any window
        if cmd == "quit":
            print("F9: saving and stopping")
            return False
        if cmd == "pause" and not pause_until_resumed():
            return False
        if state["n"] % max(1, args.render_every):
            return True
        cv2.imshow("BOX policy view", render(env, tr, a, info, args.scale))
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            return False
        if k == ord("p") and not pause_until_resumed():
            return False
        return True

    tr = Trainer(env, cfg, model_path=args.model, on_step=on_step)
    tr.on_train_tick = lambda: cv2.waitKey(1)   # keep the window responsive while training
    if args.eps is not None and cfg.algo == "dqn":
        tr.learner.steps_seen = 0
        tr.best_prog = -1.0   # --eps = new level: start a fresh best %
    print("BOX live: Double DQN on detector boxes. F8 = pause / resume, F9 = save + quit "
          "(both work from any window; q / p also work in the overlay window).")
    try:
        tr.run()
    finally:
        try:
            base.inp.close()
        except Exception:
            pass
        cap.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
