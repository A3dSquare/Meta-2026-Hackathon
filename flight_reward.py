"""Bounded geometry potential for ship/wave reward shaping.

No physics rollout or screen-centre target. All distances are in blocks,
relative to the detected player's centre, with positive y upwards.
"""
import math


def clearance_weight(raw):
    import argparse
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('clearance weight must be a number between 0 and 0.1')
    if not math.isfinite(value) or not 0 <= value <= .1:
        raise argparse.ArgumentTypeError('clearance weight must be between 0 and 0.1')
    return value


def flight_clearance(mode, player_half, objects, speed, *, reliable=True,
                     floor=None, ceiling=None, target=.75, lookahead=.5):
    """objects: (class, left, right, bottom, top, confidence).

    Only hazards (0) and solids (6) ahead or horizontally overlapping count.
    An obstacle further ahead has a smoothly reduced influence; a close
    obstacle intersecting the current flight height has zero clearance.
    Capping at target avoids rewarding unlimited distance from obstacles.
    """
    def unavailable(reason):
        return dict(score=0., valid=False, mode=mode, reason=reason)
    if mode not in ('ship', 'wave'):
        return unavailable('inactive mode')
    if not reliable:
        return unavailable('unreliable tracking')
    hw, hh = player_half
    if not all(math.isfinite(x) for x in (hw, hh, speed, target, lookahead)) or not (
            0 < hw <= 1.5 and 0 < hh <= 1.5 and speed > 0 and target > 0 and lookahead > 0):
        return unavailable('invalid geometry')
    hw, hh = max(.15, hw), max(.15, hh)
    reach = max(1., speed * lookahead)
    score, count = 1., 0
    for cls, left, right, bottom, top, confidence in objects:
        if cls not in (0, 6) or not all(math.isfinite(x) for x in
                (left, right, bottom, top, confidence)) or confidence < .5:
            continue
        if left >= right or bottom >= top or right < -hw:
            continue
        dx = max(0., left - hw)
        if dx > reach:
            continue
        # Distance between vertical edges, not centres. Approaching a row of
        # spikes at the same height progressively reduces safety before contact.
        dy = max(0., bottom - hh, -hh - top)
        score = min(score, min(1., dy / target + dx / reach))
        count += 1
    for boundary, sign in ((floor, -1.), (ceiling, 1.)):
        if boundary is not None and math.isfinite(boundary) and sign * boundary > 0:
            score = min(score, min(1., max(0., sign * boundary - hh) / target))
            count += 1
    if not count:
        return unavailable('no relevant geometry')
    return dict(score=max(0., score), valid=True, mode=mode, reason='geometry', objects=count)


def shaping_reward(current, following, gamma):
    """Inputs are already weighted potentials. Never clip this difference:
    gamma*Phi(next)-Phi(now) must telescope, including mode/gap boundaries.
    Terminal states must use following=0, even after delayed-death trimming.
    """
    return gamma * following - current
