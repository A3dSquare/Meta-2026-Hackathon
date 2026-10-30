"""How did it die? One short answer per death, from the death snapshot's records.

Replaces the hazard-picture learning (hazards/learned/): instead of guessing that
whatever was in front of the icon must be deadly, it says WHAT it hit and WHY it
didn't avoid it:

  what:  spike          touching a detected hazard
         block side     ran into the side of a block
         head hit       hit the underside of a block
         unknown        nothing detected touches it (detection missed it, or a timing death)
  why:   blue line wrong   the prediction said a safe move existed ~0.3 s before, and the
                           move taken WAS that safe move -- the physics / detection was off
         chose wrong       a safe move existed, but the brain / shield took the other one
         trapped           no safe move left 0.3 s before: the mistake was earlier
         shield            (added when the shield had overridden the brain in the last 0.3 s)

Printed after each death, written into the death snapshot (title + JSON "cause"), the
env's info ("death_cause") and attempts.csv (last column); the trainer prints a tally.
"""
from __future__ import annotations

BLOCK_FRAC_H = 30.0 / 320.0      # one block as a fraction of the frame height
ASPECT = 16.0 / 9.0
LOOK = 10                        # steps before the crash to judge "could it still be saved?" (~0.3 s)
SAFE = 0.9                       # search value counted as safe (1.0 both worlds, 0.90-0.99 as measured)


def _chosen(rec) -> float | None:
    sr = rec.get("search") or {}
    if not sr:
        return None
    return float(sr.get("1", 0.0)) if int(rec.get("a", 0)) == 1 else max(float(sr.get("00", 0.0)), float(sr.get("01", 0.0)))


def _best(rec) -> float | None:
    sr = rec.get("search") or {}
    if not sr:
        return None
    return max(float(sr.get("1", 0.0)), float(sr.get("00", 0.0)), float(sr.get("01", 0.0)))


def trace_decisions(buf, crash):
    """Locate the final loss of verified planning, not a proven causal mistake.

    Missing observations/plans are unknown, never evidence of a safe route.
    Times use measured intervals rather than assuming a fixed frame rate.
    """
    rows=buf[:crash+1]
    def state(r):
        plan=r.get('route_plan') or {}
        if not r.get('player') or not plan:return 'unknown'
        return 'verified' if plan.get('verified') else 'unverified'
    states=[state(r) for r in rows]
    last=next((i for i in range(len(rows)-1,-1,-1) if states[i]=='verified'),None)
    start=last+1 if last is not None else 0
    def detail(i):
        r=rows[i];plan=r.get('route_plan') or {};p=r.get('preview_inputs') or {}
        return dict(index=i,seconds_before_crash=round(sum(
            max(0.,float(q.get('motion_dt',q.get('dt',0.)) or 0.)) for q in rows[i+1:]),3),
            action=r.get('a'),planned_action=plan.get('first_action'),
            role=r.get('controller_role'),state=states[i],
            sequence_safety=plan.get('sequence_safety'),
            measured_velocity=r.get('vy'),estimated_velocity=p.get('vy0'),
            grounded=p.get('grounded'),state_source=p.get('state_source'))
    alternatives=[i for i,r in enumerate(rows) if r.get('player')
                  and _best(r) is not None and _best(r)>=SAFE
                  and _chosen(r) is not None and _chosen(r)<SAFE]
    return dict(last_verified=detail(last) if last is not None else None,
                first_after_verified=detail(start) if start<len(rows) else None,
                last_action_with_safer_scored_alternative=detail(alternatives[-1]) if alternatives else None,
                history_limited=last is None,
                limitation='Retrospective planner evidence, not proof that another action would survive.')


def classify(buf: list, lag: int) -> dict:
    """buf: the env's death-snapshot records (oldest first); lag: steps from the crash to the
    death signal. Returns {"what", "why", "text", ...}."""
    out = {"what": "unknown", "why": "", "text": "unknown"}
    if not buf:
        return out
    crash = max(0, len(buf) - 1 - int(lag))
    # the last record before the crash that still has the icon
    i = crash
    while i > 0 and not buf[i].get("player"):
        i -= 1
    rec = buf[i]
    pp = rec.get("player_pos")
    bh = BLOCK_FRAC_H
    bw = bh / ASPECT
    if pp is not None:
        px, py = pp
        pw, ph = bw, bh
        for b in rec.get("boxes") or []:
            if int(b[0]) == 7:
                pw, ph = max(pw, b[3]), max(ph, b[4])
        for b in rec.get("boxes") or []:
            c, x, y, w, h = (float(q) for q in b[:5])
            c = int(c)
            near_x = abs(x - px) < (w + pw) / 2 + 0.35 * bw
            near_y = abs(y - py) < (h + ph) / 2 + 0.35 * bh
            if c == 0 and near_x and near_y:
                out["what"] = "spike"
                break
            if c == 6:
                if abs(x - px) < (w + pw) / 2 + 0.2 * bw and 0 <= (py - ph / 2) - (y + h / 2) < 0.5 * bh:
                    out["what"] = "head hit"           # block just above the icon
                elif 0 <= (x - w / 2) - (px + pw / 2) < 0.6 * bw and abs(y - py) < h / 2 + 0.2 * bh:
                    out["what"] = "block side"         # block face just ahead, at the icon's height
    # why: judged LOOK steps before the crash
    j = max(0, crash - LOOK)
    before = buf[j:crash + 1]
    bv = _best(buf[j])
    cv = _chosen(buf[j])
    shielded = any(r.get("shielded") for r in before)
    # was a safe move taken all the way in? (each step's chosen action safe)
    chosen_ok = [(_chosen(r) or 0.0) >= SAFE for r in before if r.get("search")]
    if bv is None:
        out["why"] = "no prediction"
    elif bv < SAFE:
        out["why"] = "trapped"
    elif chosen_ok and all(chosen_ok):
        out["why"] = "blue line wrong"
    elif any((_best(r) or 0.)>=SAFE and (_chosen(r) or 0.)<SAFE for r in before if r.get('player')):
        out["why"] = "chose wrong"
    else:
        out["why"] = "forecast became unsafe"
    out['decision_trace']=trace_decisions(buf,crash)
    out["shielded"] = bool(shielded)
    out["mode"] = rec.get("mode")
    out["best_before"] = bv
    out["chosen_before"] = cv
    out["text"] = out["what"] + (f", {out['why']}" if out["why"] else "") + (" (shield acted)" if shielded else "")
    return out


class Tally:
    """Running counts of causes (the last `n` deaths)."""

    def __init__(self, n: int = 50) -> None:
        from collections import deque
        self.q = deque(maxlen=n)

    def add(self, cause: dict) -> None:
        self.q.append(f"{cause.get('what', '?')}/{cause.get('why', '')}")

    def summary(self) -> str:
        from collections import Counter
        c = Counter(self.q)
        return ", ".join(f"{k} {v}" for k, v in c.most_common(5))
