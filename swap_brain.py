"""Put one checkpoint's PLAYING brain in charge of another checkpoint's learning.

A route-controller checkpoint holds, separately:
  q          the brain that plays (and that --play uses)
  candidate  the brain still learning (trained from demonstrations + gameplay; it only takes
             over by winning a live trial)
  replay / validation / optimiser / schedule  -- everything learned so far

So when a bad brain gets promoted (v16/v17: one that presses when you didn't), the learning
isn't lost: keep it, and swap in a better playing brain.

  python swap_brain.py --brain checkpoints/manual_v15/route_controller.pt \
      --rest checkpoints/manual_v17/route_controller.pt --out checkpoints/manual_v18

--out gets route_controller.pt (and motion_model.npz copied from --rest's folder). The next
training session runs a live trial straight away: --rest's candidate vs the swapped-in brain.
"""
import argparse
import os
import shutil


def reset_candidate(ck: dict) -> dict:
    """The learning brain (candidate) and its target network start again from the brain in
    charge (q). Use when the candidate drifted (live v19: ship hold recall 25% vs q's 64% after
    a day of cube-only training) before retraining on new recordings."""
    out = dict(ck)
    out["candidate"] = {k: v.clone() if hasattr(v, "clone") else v for k, v in ck["q"].items()}
    out["target"] = {k: v.clone() if hasattr(v, "clone") else v for k, v in ck["q"].items()}
    # (the optimiser state is kept: the loader requires it, and Adam re-adapts within steps)
    out.pop("live_trial", None)
    return out


def swap(brain_ck: dict, rest_ck: dict) -> dict:
    """rest_ck with brain_ck's playing brain; refuses incompatible networks."""
    qa, qb = brain_ck["q"], rest_ck["q"]
    if set(qa) != set(qb):
        raise ValueError("the two checkpoints' brains have different layers: not interchangeable")
    for k in qa:
        if tuple(qa[k].shape) != tuple(qb[k].shape):
            raise ValueError(f"layer {k} differs in shape {tuple(qa[k].shape)} vs {tuple(qb[k].shape)}")
    if brain_ck.get("obs_dim") != rest_ck.get("obs_dim"):
        raise ValueError("the checkpoints use different observation sizes")
    out = dict(rest_ck)
    out["q"] = {k: v.clone() if hasattr(v, "clone") else v for k, v in qa.items()}
    out.pop("live_trial", None)                 # a half-run trial was against the old brain
    out["promotion_pending"] = False
    ep, gs = int(out.get("episode_id", 0)), int(out.get("grad_steps", 0))
    # trial as soon as possible: the candidate vs the swapped-in brain
    out["trial_schedule"] = dict(failures=0, next_episode=ep, next_grad=gs)
    out["brain_swapped_from"] = brain_ck.get("_path", "?")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--brain", required=True, help="checkpoint whose PLAYING brain to use")
    p.add_argument("--rest", required=True, help="checkpoint whose learning (candidate, data, schedule) to keep")
    p.add_argument("--out", required=True, help="new, empty folder")
    p.add_argument("--reset-candidate", action="store_true",
                   help="also restart the learning brain (candidate) from the brain in charge")
    a = p.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        p.error("--out must be new or empty (protects existing checkpoints)")
    import torch
    brain = torch.load(a.brain, map_location="cpu", weights_only=False)
    rest = torch.load(a.rest, map_location="cpu", weights_only=False)
    for name, ck in (("--brain", brain), ("--rest", rest)):
        if "route_controller_version" not in ck or "q" not in ck:
            p.error(f"{name} is not a route-controller checkpoint")
    brain["_path"] = os.path.abspath(a.brain)
    out = swap(brain, rest)
    if a.reset_candidate:
        out = reset_candidate(out)
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "route_controller.pt")
    torch.save(out, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    motion = os.path.join(os.path.dirname(os.path.abspath(a.rest)), "motion_model.npz")
    if os.path.exists(motion):
        shutil.copy2(motion, os.path.join(a.out, "motion_model.npz"))
    print(f"saved {dst}: playing brain from {a.brain}, learning from {a.rest}; "
          f"a live trial (candidate vs this brain) runs at the start of the next training session")


if __name__ == "__main__":
    main()
