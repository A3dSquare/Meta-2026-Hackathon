"""How repeatable is the timing? Reads <session>/route_timing.jsonl (written by trainer.py, one
line per attempt) and says which part of the timing varies:

  start   the attempt clock starts at a different moment each time (respawn detection)
  drift   during a run, the level gets ahead of / behind the best run's clock
  jitter  decision intervals and observation age (how old a picture is when its key goes out)

  python timing_report.py                 # newest session in logs_boxes/sessions
  python timing_report.py <session dir>   # a specific one
  python timing_report.py --all           # every session that has the log
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

SPEED = 10.38                     # blocks / s at 1x: turns ms into blocks for the verdicts
ROOT = os.path.join("logs_boxes", "sessions")


def load(d: str) -> list:
    f = os.path.join(d, "route_timing.jsonl")
    if not os.path.exists(f):
        return []
    out = []
    with open(f) as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def blocks(ms: float) -> str:
    return f"{ms * SPEED / 1000.0:.2f} blocks"


def report(d: str) -> None:
    rows = load(d)
    print(f"\n=== {os.path.basename(d)}: {len(rows)} attempt(s)")
    if not rows:
        print("  no route_timing.jsonl (run with the route memory on, i.e. without --no-route)")
        return
    print(f"  {'att':>4} {'peak%':>6} {'died s':>7} {'best s':>7} {'n':>4} {'start ms':>9} {'end ms':>8} "
          f"{'drift ms/s':>10} {'resid ms':>9} {'dec ms':>7} {'age ms':>7}  rejected")
    for r in rows:
        print(f"  {r.get('attempt', 0):>4} {r.get('peak_pct', 0):>6.1f} {r.get('died_at_s', 0):>7.2f} "
              f"{r.get('best_T', 0):>7.2f} {r.get('align_samples', 0):>4} "
              f"{_f(r.get('offset_start_ms')):>9} {_f(r.get('offset_end_ms')):>8} {_f(r.get('drift_ms_per_s')):>10} "
              f"{_f(r.get('residual_ms_std')):>9} {_f(r.get('dec_ms_median')):>7} {_f(r.get('age_ms_median')):>7}  "
              f"{_rej(r.get('align_rejected'))}" + ("" if r.get("clock_ok", True) else "  (not from a respawn)"))
    ok = [r for r in rows if r.get("clock_ok") and r.get("offset_start_ms") is not None]
    print()
    if len(ok) < 3:
        print("  VERDICT: not enough aligned attempts yet (need >= 3 that started at a respawn and had a\n"
              "  best run to compare with). Play more attempts -- the first one never counts.")
        return
    # offsets are measured against the best run AT THE TIME: compare attempts with the same reference
    by_ref: dict = {}
    for r in ok:
        by_ref.setdefault(r.get("best_T"), []).append(r)
    ref, grp = max(by_ref.items(), key=lambda kv: len(kv[1]))
    st = np.array([r["offset_start_ms"] for r in grp])
    dr = np.array([r["drift_ms_per_s"] for r in grp if r.get("drift_ms_per_s") is not None])
    rs = np.array([r["residual_ms_std"] for r in grp if r.get("residual_ms_std") is not None])
    dec = np.array([r["dec_ms_median"] for r in rows if r.get("dec_ms_median") is not None])
    dec95 = np.array([r["dec_ms_p95"] for r in rows if r.get("dec_ms_p95") is not None])
    a05 = np.array([r["age_ms_p05"] for r in rows if r.get("age_ms_p05") is not None])
    a95 = np.array([r["age_ms_p95"] for r in rows if r.get("age_ms_p95") is not None])
    print(f"  compared {len(grp)} attempts against the same best run ({ref} s)")
    start_sd = float(np.std(st))
    print(f"  start:  offset {np.median(st):+.0f} ms typical, spread (std) {start_sd:.0f} ms = {blocks(start_sd)}, "
          f"range {st.min():+.0f}..{st.max():+.0f} ms")
    if dr.size:
        print(f"  drift:  {np.median(np.abs(dr)):.1f} ms per second of play (median |slope|), worst {np.max(np.abs(dr)):.1f}")
    if rs.size:
        print(f"  wobble: {np.median(rs):.0f} ms around that line (median std) = {blocks(float(np.median(rs)))}")
    if dec.size:
        print(f"  decisions: every {np.median(dec):.0f} ms (p95 {np.median(dec95):.0f} ms) = up to "
              f"{blocks(float(np.median(dec95)))} between chances to press")
    if a05.size:
        print(f"  observation age when a key goes out: {np.median(a05):.0f}..{np.median(a95):.0f} ms (5-95%) "
              f"= {blocks(float(np.median(a95) - np.median(a05)))} of spread")
    print()
    v = []
    if start_sd > 25:
        v.append("START varies a lot: line each attempt up by its first frames, not by when the respawn "
                 "was detected (fixable).")
    else:
        v.append("START is steady: the attempt clock is a good base.")
    if dr.size and np.median(np.abs(dr)) > 10 or rs.size and np.median(rs) > 20:
        v.append("The level DRIFTS against the best run's clock during a run: a replay must keep "
                 "correcting itself from the frames the whole way (or the game's own timing varies).")
    elif dr.size:
        v.append("No real drift during runs: once lined up at the start, a timed replay should hold.")
    if dec95.size and np.median(dec95) > 45:
        v.append("Decisions are too far apart for precise presses: send replayed presses from their own "
                 "timer instead of waiting for the next decision.")
    for line in v:
        print("  VERDICT: " + line)


def _f(x) -> str:
    return "-" if x is None else f"{x:+.1f}" if isinstance(x, float) and x < 0 else "-" if x is None else f"{x:.1f}"


def _rej(r) -> str:
    if not r:
        return ""
    return ", ".join(f"{k} {v}" for k, v in sorted(r.items(), key=lambda kv: -kv[1])[:3])


def main(argv) -> None:
    if argv and argv[0] == "--all":
        for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
            if os.path.exists(os.path.join(d, "route_timing.jsonl")):
                report(d)
        return
    if argv:
        report(argv[0])
        return
    ds = [d for d in sorted(glob.glob(os.path.join(ROOT, "*"))) if os.path.isdir(d)]
    with_log = [d for d in ds if os.path.exists(os.path.join(d, "route_timing.jsonl"))]
    if not with_log:
        print("No session has route_timing.jsonl yet: run the bot (route memory on) for a few attempts.")
        return
    report(with_log[-1])


if __name__ == "__main__":
    main(sys.argv[1:])
