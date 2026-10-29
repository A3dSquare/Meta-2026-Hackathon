"""Level-agnostic policy env: entity BOXES in, SPACE out.  (v2)

Wraps GymGDEnv (capture + input + % reader). Every decision step it:
  1. holds the chosen action for `frame_skip` game frames (default 2 ->
     30 decisions/s; GD's jump windows are several frames wide, and halving
     the number of decisions halves how far back credit has to travel),
  2. measures how far the screen scrolled each frame (phase correlation),
     which is how we find the exact moment of death later,
  3. runs the detector on the 640x360 frame (small spikes need the pixels),
  4. encodes the boxes with encode_obs() -> flat float32 vector.

What changed vs v1 and why it matters for learning:
  * EMPTY SLOTS: v1 filled unused entity rows with zeros, and an all-zero
    row means "spike (class 0) at dx=0", i.e. a spike touching the player.
    Every frame with fewer than 6 entities told the policy it was about to
    die. v2 has a `present` flag and parks empty slots far away.
  * No observation jitter (+-10 px of fake noise on spike distance -- about
    3 frames of timing error on a jump that needs ~3-frame precision) and no
    1/64 rounding.
  * Edges (left/right/top/bottom) in BLOCK units instead of centers, one-hot
    class, player vertical speed.
  * DEATH TIMING: GD shows the explosion for ~1 s before the level restarts,
    and the % reader only sees the death when the % resets. v1 therefore
    punished whatever the agent did ~60 frames AFTER it had already died.
    v2 reports `death_lag` = how many decision steps ago the screen froze
    (= the moment of the crash), and the trainer cuts those steps off and
    puts the penalty on the step that actually caused the death.

The reward the trainer uses is computed in trainer.py from info flags
(alive / died / complete); the shaped reward from gd.ProgressReward is
ignored (its spike-proximity terms were driven by detector noise and
rewarded holding SPACE near spikes, i.e. jumping too early).
"""
from __future__ import annotations

from typing import Any

import time
from collections import deque

import math
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError("gymnasium is required") from e

from detect import BLOCK_FRAC_H, CLS_ID, Entities, make_detector

# ----------------------------------------------------------------------------
# Observation encoding (shared with sim_gd.py)
# ----------------------------------------------------------------------------
N_ENT = 8                         # nearest entities ahead of the player
ENT_CLASSES = 7                   # spike, orb_y, orb_b, orb_p, pad, portal, wall
ENT_FEAT = 7 + ENT_CLASSES        # present, x0, x1, y0, y1, hue cos, hue sin, one-hot
N_HIST = 8                        # past decision steps of (vertical speed, action)
from gd_physics import (MODES, MODE_ID, N_PLANS, SEARCH_KEYS, Body, ModeCalibrator, ModeEstimator,
                        MotionLearner, Phys, inst_vy, load_motion, portal_effect, resolve_start, robust_search,
                        LaunchCalibrator,
                        save_motion,
                        simulate_plans, surfaces_at)
N_MODE = len(MODES) + 3            # mode one-hot, gravity (+1/-1), mini, confidence
N_PLAN = 3 * N_PLANS + 1           # per action plan: hits?, time to hit, clearance; + time to ledge
N_RESERVED = 12                    # spare inputs (zero today) so future features don't need a new brain
OFF_HIST = 7
OFF_MODE = OFF_HIST + 2 * N_HIST
OFF_PLAN = OFF_MODE + N_MODE
OFF_RES = OFF_PLAN + N_PLAN
N_GLOBAL = OFF_RES + N_RESERVED
OBS_DIM = N_GLOBAL + N_ENT * ENT_FEAT
OBS_VERSION = 5
RESERVED = slice(OFF_RES, OFF_RES + N_RESERVED)
# spare inputs in use: +0..+5 "tap in 1" / "tap in 3" plans; +6..+8 search: survival if
# press now / release now then release / release now then press (1 = survives); +9 = 1
OFF_SEARCH = OFF_RES + 6

ASPECT = 16.0 / 9.0
BLOCK_W = BLOCK_FRAC_H / ASPECT   # one block as a fraction of screen width
BLOCK_H = BLOCK_FRAC_H
UNIT = 8.0                        # features are in units of 8 blocks (|x| <~ 2)
FAR = 2.0                         # empty slot: 16 blocks away (off screen)
SPEED_1X = 10.38                  # GD 1x speed, blocks per second


def _fast_downscale(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """= cv2.INTER_AREA, ~3-4x faster on big frames (same as capture.fast_downscale)."""
    import cv2
    while img.shape[1] // 2 >= 0.95 * w and img.shape[0] // 2 >= 0.95 * h:
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
    if img.shape[1] != w or img.shape[0] != h:
        near = abs(img.shape[1] - w) <= 0.06 * w and abs(img.shape[0] - h) <= 0.06 * h
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR if near else cv2.INTER_AREA)
    return img


def player_vy(ents: Entities, prev_py: float | None) -> float:
    """Icon's vertical speed in blocks per decision step (+ = up)."""
    if prev_py is None:
        return 0.0
    return float(np.clip((prev_py - ents.player_pos()[1]) / BLOCK_H, -3.0, 3.0))


def preview_boxes(ents: Entities, horizon_blocks: float):
    """Hazard / solid boxes in blocks relative to the icon centre (+y = up)."""
    px, py = ents.player_pos()
    haz, solid, portals = [], [], []
    spikes = []                      # the spikes' full boxes (the hitboxes are smaller)
    for e in ents.items:
        c = int(e.cls)
        x0 = (e.x - e.w / 2 - px) / BLOCK_W
        x1 = (e.x + e.w / 2 - px) / BLOCK_W
        if x1 < -0.6 or x0 > horizon_blocks:
            continue
        top = (py - (e.y - e.h / 2)) / BLOCK_H
        bot = (py - (e.y + e.h / 2)) / BLOCK_H
        if c == 0 and x0 < 0 < x1 and bot < 0 < top:
            continue     # on the icon's centre: the icon is alive, so it's part of the icon
        if c == 0:   # spike hitboxes are smaller than the sprite
            # (Size checked on a full recorded completion of a real level: with 0.25 / 0.3
            # the preview called 23 frames 'doomed' that the player survived; at 0.3 / 0.35
            # it's 18, and the simulator shield scores the same. Still a bit bigger than
            # GD's real hitbox on purpose -- that's the safety margin.)
            # The hitbox sits toward the spike's BASE: which end that is depends on where
            # the tip points (the detector knows; hanging spikes used to get an upright
            # spike's hitbox -- shifted toward the tip, missing the part by the ceiling)
            w = x1 - x0
            hh = top - bot
            pt = int(getattr(e, "pt", -1))
            if pt == 1:                                  # hanging: tip down
                haz.append((x0 + 0.3 * w, x1 - 0.3 * w, bot + 0.35 * hh, top - 0.1 * hh))
            elif pt == 2:                                # tip left (side of a pillar)
                haz.append((x0 + 0.3 * w, x1 - 0.1 * w, bot + 0.25 * hh, top - 0.25 * hh))
            elif pt == 3:                                # tip right
                haz.append((x0 + 0.1 * w, x1 - 0.3 * w, bot + 0.25 * hh, top - 0.25 * hh))
            elif pt == 0:                                # standing: tip up
                haz.append((x0 + 0.3 * w, x1 - 0.3 * w, bot + 0.1 * hh, top - 0.35 * hh))
            else:                                        # unknown (e.g. YOLO): centred
                haz.append((x0 + 0.25 * w, x1 - 0.25 * w, bot + 0.2 * hh, top - 0.2 * hh))
        elif c == 6:
            solid.append((x0, x1, bot, top))
        elif c == CLS_ID["portal"]:
            portals.append((x0, x1, bot, top, float(getattr(e, "hue", -1.0)), getattr(e, "kind", None)))
        if c == 0:
            spikes.append((x0, x1, bot, top))
    # a block box drawn over its own spikes (a hanging block's box reaching down over the
    # spikes under it): the block ends where the spikes start. With head hits now deadly,
    # the too-low bottom made every jump under it a "crash" (whole-level test, 77% part).
    if spikes and solid:
        fixed = []
        for s0, s1, sb, st in solid:
            for h0, h1, hb, ht in spikes:
                cx = (h0 + h1) / 2
                if not (s0 <= cx <= s1) or (h1 - h0) > 0.9 * (s1 - s0) + 0.3:
                    continue
                if sb - 0.1 <= hb and ht < st - 0.2 and hb < sb + 0.6 * (ht - hb) + 0.1:
                    sb = max(sb, ht - 0.05)          # spike hangs from inside the box's bottom
                elif st + 0.1 >= ht and hb > sb + 0.2 and ht > st - 0.6 * (ht - hb) - 0.1:
                    st = min(st, hb + 0.05)          # spike stands inside the box's top
            if st - sb > 0.1:
                fixed.append((s0, s1, sb, st))
        solid = fixed
    return haz, solid, portals


def special_boxes(ents: Entities, reach: float = 17.0) -> list:
    """Pads and orbs ahead, in blocks relative to the icon centre:
    [(x0, x1, bottom, top, "pad" | "orb", effect, index, type name)] (see gd_physics.special_effect)."""
    from gd_physics import pad_orb_kind
    px, py = ents.player_pos()
    out = []
    for e in ents.items:
        c = int(e.cls)
        if c not in (1, 2, 3, 4):
            continue
        k = pad_orb_kind(c, float(getattr(e, "hue", -1.0)), getattr(e, "kind", None))
        if k is None:
            continue
        x0 = (e.x - e.w / 2 - px) / BLOCK_W
        x1 = (e.x + e.w / 2 - px) / BLOCK_W
        if x1 < -1.0 or x0 > reach:
            continue
        bot = (py - (e.y + e.h / 2)) / BLOCK_H
        top = (py - (e.y - e.h / 2)) / BLOCK_H
        if k[0] == "pad":
            top = max(top, bot + 0.3)          # the touch zone is the strip and a bit above it
        out.append((x0, x1, bot, top, k[0], k[1], len(out), k[2]))
        if len(out) >= 30:
            break
    return out


USE_MS_DELAY = False     # train_boxes --ms-delay turns it on


def physics_features(ents: Entities, speed: float, est: ModeEstimator, phys: Phys,
                     frame_skip: int, vy_bps: float, grounded: bool, prev_hold: bool,
                     floor_rel: float | None, ceil_rel: float | None, horizon: float = 1.5,
                     dec_dt: float | None = None, motion: MotionLearner | None = None,
                     out: dict | None = None, state=None, search: bool = True, half: float = 0.5,
                     prefer=None):
    """Mode block + action-plan outcomes for the current frame.

    floor_rel / ceil_rel: surface heights relative to the icon centre (None =
    unknown). Returns (mode_feats[N_MODE], plan_feats[N_PLAN], paths).
    """
    haz, solid, portals_raw = preview_boxes(ents, speed * horizon + 1)
    if floor_rel is not None and est.g == 1:
        # hazards sitting at / just under the floor line (a floor covered in thorns): landing
        # on the floor there kills -- reach their hitboxes up to the floor (live: thorns
        # 0.3-0.5 blocks under the detected line, so falling onto them looked like a safe
        # landing)
        low = [h for h in haz if floor_rel - 1.0 < h[3] <= floor_rel + 0.05]
        if len(low) >= 2:
            haz = [(h[0], h[1], h[2], floor_rel + 0.15) if h in low else h for h in haz]
    portals = []
    for x0_, x1_, b_, t_, hue_, kind_ in portals_raw:   # real portals only (same test as the env's hint)
        on_block = any(s0 < x1_ and s1 > x0_ and sb < t_ and st > b_ for s0, s1, sb, st in solid)
        if kind_:                                # YOLO named the exact portal: trust it
            from gd_objects import portal_fx
            eff = portal_fx(kind_)
            if eff is not None and eff[0] != "none" and not on_block:
                portals.append((x0_, x1_, b_, t_) + tuple(eff))
            continue
        eff = portal_effect(hue_)
        if eff is not None and (t_ - b_) >= 2.4 and (x1_ - x0_) >= 0.6 and not on_block:
            portals.append((x0_, x1_, b_, t_) + eff)
    specials = special_boxes(ents)
    # just landed? Standing level with a known surface and not moving counts as on the
    # ground right away (the 3-still-frames rule alone misses taps right after a landing)
    if not grounded and est.mode not in ("ship", "wave") and est.g * vy_bps < 1.5 * ModeEstimator.REST:
        # (the landing step's measured speed is still the fall speed; "not rising" is enough)
        fl, ce = surfaces_at(0.0, solid, floor_rel, ceil_rel)
        # (0.25, not 0.12: the box centre sits up to ~0.3 off the icon's real centre)
        grounded = any(abs(q) < 0.25 for q in (fl if est.g == 1 else ce))
    # input delay: the icon is still reacting to actions sent `est.lag` steps ago
    eff_prev = est.eff_hold(0) if est.lag > 0 else prev_hold
    dtd = dec_dt or frame_skip / 60.0
    vy0 = inst_vy(est.mode, est.g, est.mini, vy_bps, eff_prev, grounded, dtd, phys)
    y0 = 0.0
    if state is not None and not grounded:      # arc fit (env): steadier than 2 frames
        y0, vy0 = float(state[0]), float(state[1])
    if grounded and est.mode not in ("ship", "wave"):
        # The player box is only the coloured part of the icon, so its centre sits up to
        # ~0.3 blocks off the icon's real centre (which part depends on how it landed).
        # Standing on a known surface, the real centre is exactly half a block above it.
        fl_, ce_ = surfaces_at(0.0, solid, floor_rel, ceil_rel, half)
        rest = [q for q in (fl_ if est.g == 1 else ce_) if abs(q) < 0.35]
        if rest:
            y0, vy0 = min(rest, key=abs), 0.0
    # detection is off by a few pixels: a live icon is never INSIDE a block -- move it out
    y0 = resolve_start(y0, solid, half)
    body = Body(y0, vy0, est.g, grounded, est.mode, est.mini, 0.0, eff_prev)
    xo: list = []
    pend = est.pending()
    plan, paths = simulate_plans(body, haz, solid, floor_rel, ceil_rel, speed, phys,
                                 frame_skip=frame_skip, horizon=horizon, dec_dt=dec_dt, half=half,
                                 lag=est.lag, pending=pend, motion=motion, extra_out=xo)
    if search:
        try:
            alt = est.lag_alt if getattr(est, "lag_alt", None) is not None else est.lag + 1
            # (the ms delay is opt-in: --ms-delay. Live, the runs after it went in did worse
            #  -- back to whole steps by default until it's proven)
            D = getattr(est, "delay_s", None) if USE_MS_DELAY else None
            if D is not None:              # the input delay measured in ms (not whole steps)
                Da = getattr(est, "delay_alt_s", None)
                Da = D + dtd if Da is None else Da
                sr = robust_search(body, haz, solid, floor_rel, ceil_rel, speed, phys, dtd, est.lag,
                                   est.pending(int(np.ceil(D / dtd - 1e-6))),
                                   pending_long=est.pending(int(np.ceil(Da / dtd - 1e-6))), horizon=horizon,
                                   half=half, want_path=True, portals=portals, specials=specials,
                                   prefer=prefer, delay=D, delay_alt=Da)
            else:
                sr = robust_search(body, haz, solid, floor_rel, ceil_rel, speed, phys, dtd, est.lag, pend,
                                   pending_long=est.pending(alt), horizon=horizon, half=half,
                                   want_path=True, lag_alt=alt, portals=portals, specials=specials,
                                   prefer=prefer)
            paths = list(paths) + [("best", bool(sr.get("path_hit")), sr.get("path") or [])]
            if out is not None:
                out["search"] = sr
        except Exception as ex:   # never let the search take the preview down
            if out is not None:
                out["search_error"] = repr(ex)
    if out is not None and xo:
        out["extra_plans"] = xo[0]   # "tap in 1" / "tap in 3" -> the brain's spare inputs
    if out is not None:   # the state the preview started from (to learn from what happens next)
        out["inputs"] = {"grounded": bool(grounded), "vy0": float(vy0), "y0": float(y0),
                         "floor_rel": floor_rel, "ceil_rel": ceil_rel,
                         "lag": int(est.lag), "pending": [bool(x) for x in pend],
                         "motion": bool(motion is not None and motion.active_for(est.mode)),
                         "cube_v0": float(phys.cube_v0), "cube_g": float(phys.cube_g)}
        out["body"] = body
        out["surf"] = surfaces_at(0.0, solid, floor_rel, ceil_rel)
    mf = np.zeros(N_MODE, dtype=np.float32)
    mf[MODE_ID[est.mode]] = 1.0
    mf[len(MODES)] = float(est.g)
    mf[len(MODES) + 1] = 1.0 if est.mini else 0.0
    mf[len(MODES) + 2] = est.conf
    return mf, plan, paths


def encode_obs(ents: Entities, *, hold: bool, prev_py: float | None,
               player_found: bool | None = None, speed: float = SPEED_1X,
               hist=None, mode=None, plan=None, extra=None, search=None) -> np.ndarray:
    """Entities -> flat (OBS_DIM,) float32. Player-relative, in block units / 8.

    Globals tell the policy what the boxes can't:
      speed      scroll speed / 1x speed (speed portals change jump timing)
      icon w, h  size of the icon box (cube ~1x1, ship wider, wave/mini smaller)
      history    the last N_HIST steps of (vertical speed, action). How the icon
                 responds to SPACE is what tells cube from ship from ball, and
                 which way gravity points -- the policy reads mode from it.
    """
    px, py = ents.player_pos()
    pl = ents.player()
    found = (pl is not None and pl.conf > 0.6) if player_found is None else bool(player_found)
    out = np.zeros(OBS_DIM, dtype=np.float32)
    out[0] = (0.75 - py) / BLOCK_H / UNIT          # height above the usual floor line
    out[1] = player_vy(ents, prev_py)
    out[2] = 1.0 if hold else 0.0
    out[3] = 1.0 if found else 0.0
    out[4] = float(np.clip(speed / SPEED_1X, 0.0, 2.0))
    if pl is not None:
        out[5] = float(np.clip(pl.w / BLOCK_W / 2, 0, 2))
        out[6] = float(np.clip(pl.h / BLOCK_H / 2, 0, 2))
    if hist:
        hh = list(hist)[-N_HIST:]
        for k, (vy_k, a_k) in enumerate(reversed(hh)):   # newest first
            out[7 + 2 * k] = vy_k
            out[8 + 2 * k] = float(a_k)
    if mode is not None:            # which game mode / gravity / size (from how it moves)
        out[OFF_MODE:OFF_MODE + N_MODE] = mode
    if plan is not None:            # physics preview: outcome of 8 fixed action plans
        out[OFF_PLAN:OFF_PLAN + N_PLAN] = plan
    if extra is not None:           # extra plan timings, in the first spare inputs
        out[OFF_RES:OFF_RES + len(extra)] = extra
    if search is not None:          # best survival per first action (later decisions played well)
        out[OFF_SEARCH:OFF_SEARCH + 3] = [float(search[k]) for k in SEARCH_KEYS]
        out[OFF_SEARCH + 3] = 1.0
    # the rest of out[RESERVED] stays 0: room for future features without a new brain
    half_pw = 0.5 * BLOCK_W
    ahead = []
    for e in ents.items:
        c = int(e.cls)
        if c == CLS_ID["player"] or c >= ENT_CLASSES:
            continue
        x0, x1 = e.x - e.w / 2, e.x + e.w / 2
        if x1 < px - half_pw:      # fully behind the icon: irrelevant now
            continue
        ahead.append((x0, x1, e.y - e.h / 2, e.y + e.h / 2, c, float(getattr(e, "hue", -1.0))))
    ahead.sort(key=lambda t: t[0])
    base = N_GLOBAL
    for i in range(N_ENT):
        o = base + i * ENT_FEAT
        if i < len(ahead):
            x0, x1, y0, y1, c, hue = ahead[i]
            out[o] = 1.0
            out[o + 1] = (x0 - px) / BLOCK_W / UNIT
            out[o + 2] = (x1 - px) / BLOCK_W / UNIT
            out[o + 3] = (py - y1) / BLOCK_H / UNIT    # bottom edge, + = above player center
            out[o + 4] = (py - y0) / BLOCK_H / UNIT    # top edge
            if hue >= 0:                               # colour = orb/pad/portal type
                out[o + 5] = float(np.cos(2 * np.pi * hue))
                out[o + 6] = float(np.sin(2 * np.pi * hue))
            out[o + 7 + c] = 1.0
        else:
            out[o + 1] = FAR
            out[o + 2] = FAR
    return np.clip(out, -FAR, FAR)


def estimate_death_lag(moving: list, fallback: int, *, min_run: int = 3,
                       max_post: int = 8, max_gap: int = 2) -> tuple[int, str]:
    """How many of the most recent decision steps happened after the crash.

    `moving[i]` says whether the camera scrolled during decision step i of
    this attempt. On death GD freezes the camera while the explosion plays,
    then restarts the level (camera moves again) and only then does the %
    reader report the death. So the history ends:
        ...moving, [frozen x N], [moving x few]
    and the crash happened right before the frozen run. Explosion particles
    can make a frozen step look like motion for a step or two, so blips of
    <= max_gap moving steps inside the frozen run are bridged (a live camera
    never stops for just one step).
    Returns (lag_steps, "measured"|"fallback").
    """
    n = len(moving)
    if n <= 1:
        return 0, "fallback"
    i, post = n - 1, 0
    while i >= 0 and moving[i] and post < max_post:
        i -= 1
        post += 1
    if i < 0 or moving[i]:
        return min(fallback, n - 1), "fallback"
    j = i
    while j >= 0:
        if not moving[j]:
            j -= 1
            continue
        k = j
        while k >= 0 and moving[k]:
            k -= 1
        if j - k <= max_gap and k >= 0 and not moving[k]:
            j = k          # short blip inside the freeze: keep walking back
            continue
        break
    run = i - j
    if run < min_run:
        return min(fallback, n - 1), "fallback"
    return min(post + run, n - 1), "measured"


class MotionMeter:
    """Pixel fallback for "did the camera scroll?" when no hazard is tracked.

    Compares the new frame with the previous one shifted left by 1..8 px (GD
    scrolls ~2.4 px/frame at 0.5x up to ~5.4 px at 4x, at 320 wide). Returns
    the best shift if it matches clearly better than no shift, 0.0 if the
    frame is unchanged (frozen / dropped), None if it can't tell (e.g. a big
    slow-parallax background dominates the frame).
    The primary signal is the hazard tracker's displacement (Entities.scroll):
    at a crash the thing you hit is on screen and stops moving.
    """

    MAX_SHIFT = 8

    def __init__(self) -> None:
        import cv2
        self._cv2 = cv2
        self._prev = None

    def reset(self) -> None:
        self._prev = None

    def update(self, rgb: np.ndarray):
        cv2 = self._cv2
        f = np.asarray(rgb)
        h = f.shape[0]
        band = np.ascontiguousarray(f[int(0.40 * h):int(0.97 * h)])
        g = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY).astype(np.int16)
        prev, self._prev = self._prev, g
        if prev is None or prev.shape != g.shape:
            return None
        S = self.MAX_SHIFT
        W = g.shape[1]
        cur = g[:, S:W - S]
        d0 = float(np.abs(cur - prev[:, S:W - S]).mean())
        if d0 <= 0.02:
            return 0.0
        best, best_s = d0, 0
        for s in range(1, S + 1):
            d = float(np.abs(cur - prev[:, S + s:W - S + s]).mean())
            if d < best:
                best, best_s = d, s
        if best_s and best < 0.75 * d0:
            return float(best_s)
        return None


def _find_base(env):
    """Walk wrappers down to the GymGDEnv (has .cap, .reward, .pause)."""
    w = env
    for _ in range(6):
        if hasattr(w, "cap") and hasattr(w, "reward"):
            return w
        w = getattr(w, "env", None)
        if w is None:
            break
    return None


class EntityBoxEnv(gym.Wrapper):
    """GymGDEnv -> (OBS_DIM,) box vector, frame-skipped, with death timing.

    info keys added every step:
      moving (bool), prog (float|None), attempts (int)
      died (bool), death_lag (int steps), death_lag_src ("measured"|"fallback")
      complete (bool), n_entities (int)
    """

    def __init__(self, env: gym.Env, backend: str = "heuristic", detector: Any | None = None,
                 frame_skip: int = 2, hires: bool = True, death_lag_s: float | None = None,
                 fallback_lag_s: float = 0.6, move_px: float = 0.6,
                 motion_path: str | None = None, motion_fix: bool = False) -> None:
        super().__init__(env)
        self.base = _find_base(env)
        self.det = detector if detector is not None else make_detector(backend)
        self.frame_skip = max(1, int(frame_skip))
        self.hires = bool(hires)
        self.death_lag_s = death_lag_s
        self.fallback_steps = max(0, int(round(fallback_lag_s * 60 / self.frame_skip)))
        self.move_px = float(move_px)
        self.observation_space = spaces.Box(-FAR, FAR, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.motion = MotionMeter()
        from collections import deque
        self._scroll_hist = deque(maxlen=120)   # |tracker scroll| on moving steps
        self._last_moved = True
        self._lag_hist = deque(maxlen=25)        # measured crash lags (steps)
        self._hist = deque(maxlen=N_HIST)        # (vy, action) per step this attempt
        self.phys = Phys()                       # GD physics (cube jump self-calibrates)
        self.mode_est = ModeEstimator()          # which mode / gravity, from how the icon moves
        # learned correction of the physics preview (+ tuned jump + input delay), shared
        # by all brains and levels; trains itself on every step of play
        self._game_t = 0                 # captured frames while the game was running (see step)
        self._in_level = False           # see _update_in_level: no SPACE on menus
        self._no_icon = 0
        self._pct_ever = False
        self.fast_steps = False          # see step(): observe as soon as frame_skip capture slots passed
        self.timing = {"detect": 0.0, "physics": 0.0}   # seconds since the last [health] line
        # death snapshots (death_snaps.py): the last ~2.5 s of steps, saved on each death
        self.death_dir: str | None = None
        self.death_keep = 30
        self._snap: deque = deque(maxlen=75)
        self._dbg: dict = {}          # set by the trainer each step: brain's choice, shield, Q
        # input sanity: is GD the active window, and do presses do anything?
        self._focus_n = 0
        self._focus_warn_t = 0.0
        self._focus_unknown_shown = False
        self._press_pending: list = []
        self._press_ok: deque = deque(maxlen=10)
        self._press_delay: deque = deque(maxlen=15)
        self._press_delay_s: deque = deque(maxlen=21)   # the same presses' delays in seconds
        self._lag_vote = (None, 0)
        self._last_yb = None
        self._air: list = []             # (time, height) since leaving the ground (arc fit)
        self._debris: list = []          # per step: debris boxes around the icon (crash timing)
        self._last_pp = None
        self._press_warn_t = 0.0
        self._peak = 0.0
        # off by default: on live data its corrections made "don't press" float over
        # the first spike in the preview (so the shield trusted it). --motion-fix = on.
        self.use_motion_fix = bool(motion_fix)
        self.motion_fix = MotionLearner()
        self.motion_path = motion_path
        self._mstate = None
        self._motion_t = time.time()
        self._n_deaths = 0
        self.mode_cal = ModeCalibrator()         # ship / ball / UFO / wave physics, measured live
        self.launch_cal = LaunchCalibrator()     # every pad / orb type's launch strength, measured live
        if motion_path and load_motion(motion_path, self.motion_fix, self.phys, self.mode_est, self.mode_cal,
                                       self.launch_cal):
            m = self.motion_fix
            print(f"[motion] loaded {motion_path}: {m.n} examples, correction trust {m.trust:.0%}, "
                  f"jump {self.phys.apex:.2f} blocks / {self.phys.airtime:.2f}s, input delay {self._delay_txt()}; "
                  f"{self.mode_cal.summary(self.phys)}")
        if motion_path:
            import atexit
            atexit.register(self.save_motion)
        self.last_arcs: list = []                # predicted paths (for the overlay)
        self._floor_y = None                     # screen y of the floor (lowest place stood on)
        self._floor_hist: deque = deque(maxlen=30)   # recent standing positions on the ground
        self._ceil_hist: deque = deque(maxlen=30)
        self._ceil_y = None                      # ... of the ceiling when upside down
        self._portal_cool = 0
        self.speed = None                        # blocks/s, from the tracker
        self._speed_hist: deque = deque(maxlen=15)
        self._line_raw = [None, None]            # last raw ground / ceiling line (screen y)
        self._line_hold = [(None, 0), (None, 0)] # last confirmed line + frames since seen
        self._lines = (None, None)               # confirmed lines this step
        self.use_lines = True
        self.use_search = True                   # the best-continuation search (shield + brain)
        self.hazmem = None                       # hazards.HazardMemory (train_boxes sets it)
        self._last_search = None
        self._rest_true = None                   # real resting centre height (y_b) on the ground
        self._dt_frame = None
        self._moving: list = []
        self._prev_py: float | None = None
        self._prev_attempts = 1
        self._started = False
        self._err_shown = False
        self.last_frame = np.zeros((180, 320, 3), np.uint8)   # RGB, for the GUI
        self.last_hires = None                                # BGR 640x360
        self.last_ents = Entities([])
        self.last_shift = 0.0
        self._dbg_ring: list = []   # legacy attribute (probe_diag.py)

    # -- frames ---------------------------------------------------------------
    def _newest(self, obs) -> np.ndarray | None:
        a = np.asarray(obs)
        if a.ndim == 3 and a.shape[0] == 180 and a.shape[1] == 320 and a.shape[2] >= 3:
            return a[:, :, -3:]
        d = getattr(self.env, "last_display", None)
        if d is not None:
            d = np.asarray(d)
            return d[-1] if d.ndim == 4 else d
        return None

    def _hires(self):
        if not self.hires or self.base is None:
            return None
        try:
            import cv2
            hi = getattr(self.base.cap, "last_hires", None)   # already shrunk by the capture
            if hi is not None and hi.shape[:2] == (360, 640):
                return hi
            full = getattr(self.base.cap, "last_fullres", None)
            if full is None:
                return None
            full = np.asarray(full)
            if full.ndim != 3 or full.shape[0] < 360:
                return None
            return _fast_downscale(np.ascontiguousarray(full[:, :, :3]), 640, 360)
        except Exception as ex:
            self._loud("hires", ex)
            return None

    def _loud(self, where: str, exc: BaseException) -> None:
        if not self._err_shown:
            self._err_shown = True
            print(f"[EntityBoxEnv:{where}] {type(exc).__name__}: {exc}")

    def _step_dt(self, peek: bool = False) -> float:
        """Real seconds per decision (smoothed). Nominal is frame_skip/60, but if
        detection can't keep up the game moves further per step than assumed."""
        nom = self.frame_skip / 60.0
        if peek:
            return getattr(self, "_dt_s", nom)
        now = time.perf_counter()
        last = getattr(self, "_t_obs", None)
        self._t_obs = now
        cur = getattr(self, "_dt_s", nom)
        raw = None
        if last is not None and 0.6 * nom <= now - last <= 3.0 * nom:   # ignore pauses / hiccups
            raw = now - last
            cur = 0.9 * cur + 0.1 * raw
        self._dt_s = cur
        # THIS step's real duration (live decisions take 2 or 3 frames: 33 or 50 ms; using
        # the average for every step made the measured speeds wobble by ~8 blocks/s)
        self._dt_raw = raw if raw is not None else cur
        return cur

    def _update_speed(self, sc, moved: bool) -> None:
        """Scroll speed from the hazard tracker (blocks/s): the scroll of THIS step over
        THIS step's real length, then the median of the last ~15 (was the average step
        length + a fast average: live speeds wobbled 7.6..13 blocks/s around a true 10.4,
        so where a spike would be in 1 s moved by +-2 blocks step to step)."""
        if sc is None or not moved:
            return
        bps = abs(sc) / BLOCK_W / max(1e-3, self._motion_dt())
        if 3.0 <= bps <= 30.0:
            self._speed_hist.append(bps)
            self.speed = float(np.median(self._speed_hist))

    def _motion_dt(self) -> float:
        """Real time between the last two frames: the screen capture's own frame times
        when it has them (dxcam 0.3+), else the time between our observations."""
        d = getattr(self, "_dt_frame", None)
        return d if d else (getattr(self, "_dt_raw", None) or self._step_dt(peek=True))

    def _frame_time_step(self) -> None:
        """Update _dt_frame from the capture's frame timestamps (None if unavailable)."""
        cap = getattr(self.base, "cap", None) if self.base is not None else None
        ts = getattr(cap, "last_frame_ts", None) if cap is not None else None
        last = getattr(self, "_ts_last", None)
        self._dt_frame = None
        nom = self.frame_skip / 60.0
        if ts is not None and last is not None and 0.4 * nom <= ts - last <= 3.0 * nom:
            self._dt_frame = float(ts - last)
        if ts is not None:
            self._ts_last = ts
        # when THIS observation's picture was on screen (perf_counter seconds): the capture's
        # own frame time if it is on that clock, else now -- for the input delay in ms
        now = time.perf_counter()
        t_obs = float(ts) if ts is not None and abs(float(ts) - now) < 0.5 else now
        self._obs_ts_prev, self._obs_ts = getattr(self, "_obs_ts", None), t_obs

    def _surface_lines(self, frame) -> tuple:
        """(floor, ceiling) screen heights (fractions, None = not seen). GD draws a bright,
        paler line along the top of the ground and the bottom of a ceiling, all the way
        across the screen. That line is exact (live: the floor estimated from standing
        positions was off by up to 1.2 blocks after deaths froze the icon in mid-air), and
        it follows the camera. A line must be seen at the same height twice in a row."""
        try:
            import cv2
            img = np.asarray(frame)
            h, w = img.shape[:2]
            cols = np.r_[int(0.05 * w):int(0.28 * w), int(0.48 * w):int(0.95 * w)]   # not the icon
            hsv = cv2.cvtColor(np.ascontiguousarray(img[:, cols, :3]), cv2.COLOR_RGB2HSV)
            sc = hsv[:, :, 2].astype(np.int16) - (hsv[:, :, 1].astype(np.int16) >> 1)
            up, dn = np.roll(sc, 2, axis=0), np.roll(sc, -2, axis=0)
            ridge_m = (sc - np.maximum(up, dn)) > 12
            ridge = ridge_m.mean(axis=1)
            ridge[:2] = 0
            ridge[-2:] = 0
            # The ground line is the LOWEST long bright row with plain ground below it.
            # A long platform's top edge is a bright row too, but under it come the
            # platform's own grid lines (the whole-level test on a recorded completion:
            # platform edges taken for the ground / a ceiling "doomed" 9 frames the player
            # survived). (GD's ground line fades toward the screen edges: don't demand them.)
            res = []
            room = max(3, int(0.04 * h))
            # (ceiling: not used -- a platform top above the icon looks just like one and
            # blocked every jump in the test; ceilings come from detected blocks instead)
            for floor_side in (True,):
                lo, hi_ = (0.45, 0.99) if floor_side else (0.07, 0.55)
                rows = range(int(hi_ * h) - 1, int(lo * h) - 1, -1) if floor_side else range(int(lo * h), int(hi_ * h))
                found_r = None
                for r in rows:
                    if ridge[r] < 0.5:
                        continue
                    beyond = ridge[r + 3:] if floor_side else ridge[int(0.07 * h):max(int(0.07 * h), r - 2)]
                    if len(beyond) >= room and float(beyond.max(initial=0.0)) < 0.4:
                        found_r = r
                    break                          # only the outermost long row can be it
                res.append(None if found_r is None else (found_r + 0.5) / h)
            res.append(None)
        except Exception as ex:
            self._loud("surface-lines", ex)
            res = [None, None]
        out = []
        for k, y in enumerate(res):
            prev = self._line_raw[k]
            self._line_raw[k] = y
            ok = y is not None and prev is not None and abs(y - prev) < 0.012
            held, age = self._line_hold[k]
            if ok:
                self._line_hold[k] = (y, 0)
                out.append(y)
            elif held is not None and age < 15 and (y is None or abs(y - held) < 0.012):
                # not seen this frame (a spike / the icon / an effect covering it): keep the
                # last line for up to ~0.6 s instead of falling back to the guessed floor,
                # which made the predicted paths jump for a frame ("blips")
                self._line_hold[k] = (held, age + 1)
                out.append(held)
            else:
                if age >= 15:
                    self._line_hold[k] = (None, 0)
                out.append(None)
        return tuple(out)

    def _observe(self, frame, hold: bool) -> np.ndarray:
        self.last_frame = frame
        _t0 = time.perf_counter()
        hi = self._hires()
        self.last_hires = hi
        try:
            ents = self.det(frame, hi) if hi is not None else self.det(frame)
        except TypeError:
            ents = self.det(frame)  # a detector that takes one frame
        except Exception as ex:
            self._loud("detect", ex)
            ents = Entities([])
        try:                             # the detector recognised the icon's mode (YOLO)
            _pl = ents.player()
            self.mode_est.icon_hint(getattr(_pl, "mode", None) if _pl is not None else None,
                                    float(getattr(_pl, "conf", 0.0)) if _pl is not None else 0.0)
        except Exception as ex:
            self._loud("icon-mode", ex)
        if self.hazmem is not None:      # hazards taught by picture / learned from deaths
            try:
                ents = self.hazmem.apply(ents, hi if hi is not None else self.last_hires, getattr(ents, "scroll", None))
            except Exception as ex:
                self._loud("hazards", ex)
        self.last_ents = ents
        _t1 = time.perf_counter()
        self.timing["detect"] += _t1 - _t0
        pl = ents.player()
        spd = self.speed or SPEED_1X
        dt_plan = self._step_dt()                       # typical step (plan timing)
        self._frame_time_step()
        dt = self._motion_dt()                          # this step's real length (motion)
        line_fl, line_ce = self._surface_lines(frame) if self.use_lines else (None, None)
        self._lines = (line_fl, line_ce)
        prev_hold = bool(self._hist and self._hist[-1][1] > 0.5)
        vy_step = player_vy(ents, self._prev_py) if pl is not None else 0.0
        vy_bps = vy_step / dt
        recent = [h[0] for h in list(self._hist)[-2:]] + [vy_step]
        grounded = len(recent) >= 2 and all(abs(v) / dt < ModeEstimator.REST for v in recent)
        floor_rel = ceil_rel = None
        if pl is not None:
            py = ents.player_pos()[1]
            y_b = (0.75 - py) / BLOCK_H
            self.mode_est.update(y_b, hold, dt, spd, self.phys)
            me = self.mode_est
            sc0 = getattr(ents, "scroll", None)
            self._watch_press(y_b, hold, me.resting_before(), sc0 is not None and abs(sc0) > 0.0025 * self.frame_skip)
            if me.mode == "cube" and me.g == 1 and not me.mini and self.presses_working:
                # the press that reached the icon this step (input delay), and whether it
                # was resting BEFORE this step (not counting the jump's own first motion)
                self.phys.calibrate(y_b, vy_step, me.eff_hold(0), me.resting_before(), dt)
            px_ = ents.player_pos()[0]
            walls_ = [w for w in ents.items if int(w.cls) == CLS_ID["wall"]]
            for e in ents.items:            # portal crossing the icon: mode may change now
                c_ = int(e.cls)
                kind_ = getattr(e, "kind", None)
                if c_ == CLS_ID["portal"] and kind_ and abs(e.x - px_) < 0.6 * BLOCK_W and self._portal_cool <= 0:
                    # YOLO named the exact portal: apply it (no size / colour guessing)
                    from gd_objects import portal_fx
                    fx = portal_fx(kind_)
                    if fx is not None and abs(e.y - py) < 0.6 * max(e.h, 2.0 * BLOCK_H):
                        if fx[0] == "speed":
                            self.speed = float(fx[1])
                            self._speed_hist.clear()
                            self._speed_hist.extend([float(fx[1])] * 5)
                        else:
                            self.mode_est.portal_fx(fx)
                        self._portal_cool = 10
                elif c_ == CLS_ID["portal"] and abs(e.x - px_) < 0.6 * BLOCK_W and self._portal_cool <= 0:
                    # mode / gravity portals are ~3 blocks tall (speed portals are shorter and
                    # don't change the mode); a "portal" that sits on a detected block is a
                    # pillar, not a portal (live runs: pillars turned the cube into a "UFO")
                    # Live: thin glowing pillars in Stereo Madness (~0.4 blocks wide, below the
                    # icon) were taken for portals and made the cube a "spider" / "robot".
                    # A real portal is ~1 block wide and the icon flies THROUGH it.
                    if (e.h / BLOCK_H >= 2.4 and e.w / BLOCK_W >= 0.6 and abs(e.y - py) < 0.45 * e.h
                            and not any(abs(w.x - e.x) < (w.w + e.w) / 2
                                        and abs(w.y - e.y) < (w.h + e.h) / 2 for w in walls_)):
                        self.mode_est.portal_hint(float(getattr(e, "hue", -1.0)))
                        self._portal_cool = 10
                elif c_ in (CLS_ID["orb_y"], CLS_ID["orb_b"], CLS_ID["orb_p"], CLS_ID["pad"]) \
                        and abs(e.x - px_) < 1.0 * BLOCK_W and abs(e.y - py) < 1.5 * BLOCK_H:
                    self.mode_est.gravity_hint()   # orbs / pads can flip gravity
            self._portal_cool -= 1
            # Floor / ceiling. The player box is often only part of the icon (and which part
            # changes as it spins), so its centre wobbles by ~0.3 blocks: track the ground as
            # the MEDIAN of recent standing positions, and while standing on the ground (not
            # on a detected block) the floor is exactly half a block down by definition.
            # Live snapshots showed the floor placed 0.3 blocks too high, so the "don't press"
            # path sank through it and slid UNDER the first spike -- "safe" until too late.
            g_ = self.mode_est.g
            on_block = self._on_block(ents, py, g_)
            # (only while the level scrolls: a death freezes the icon wherever it died,
            # and those "standing" heights dragged the floor up to 1.2 blocks off)
            scrolling = sc0 is not None and abs(sc0) > 0.0025 * self.frame_skip
            # standing at the height of detected platform tops nearby: on a platform whose
            # piece under the icon wasn't detected -- not the level's ground
            at_plat = any(int(e.cls) == CLS_ID["wall"] and abs((e.y - e.h / 2) - (py + 0.5 * BLOCK_H)) < 0.3 * BLOCK_H
                          and abs(e.x - ents.player_pos()[0]) < 8 * BLOCK_W for e in ents.items)
            if grounded and not on_block and scrolling and not at_plat:
                if g_ == 1 and (self._ceil_y is None or py - self._ceil_y > 1.5 * BLOCK_H):
                    if self._floor_hist and abs(py - float(np.median(self._floor_hist))) > 3 * BLOCK_H:
                        self._floor_hist.clear()          # the level's ground moved (new section)
                    self._floor_hist.append(py)
                    self._floor_y = float(np.median(self._floor_hist))
                elif g_ == -1 and (self._floor_y is None or self._floor_y - py > 1.5 * BLOCK_H):
                    # only a real ceiling: standing still ON the floor while the recogniser
                    # briefly thinks "upside down" must not turn the floor into a ceiling
                    if self._ceil_hist and abs(py - float(np.median(self._ceil_hist))) > 3 * BLOCK_H:
                        self._ceil_hist.clear()
                    self._ceil_hist.append(py)
                    self._ceil_y = float(np.median(self._ceil_hist))
            if line_fl is not None and self._line_is_platform(ents, line_fl):
                # the "ground line" is the top edge of a detected block (a floating platform):
                # not the ground -- using it as an endless floor made walking off a platform
                # look safe (live: 90% of deaths were falling off platforms instead of jumping)
                line_fl = None
            if line_fl is not None and line_fl > py + 0.2 * BLOCK_H:
                # the ground line: exact. Relative to the box centre, which may sit off the
                # icon's real centre (the preview corrects for that while standing)
                floor_rel = min(-0.15, (py - line_fl) / BLOCK_H)
            elif self._floor_y is not None:
                floor_rel = (py - self._floor_y) / BLOCK_H - 0.5
                if (grounded and g_ == 1 and not on_block and floor_rel > -1.0) or floor_rel > -0.5:
                    floor_rel = -0.5                      # standing on it / never inside the icon
            if line_ce is not None and line_ce < py - 0.2 * BLOCK_H:
                ceil_rel = max(0.15, (py - line_ce) / BLOCK_H)   # a real ceiling line
            elif self._ceil_y is not None and g_ == -1:
                ceil_rel = (py - self._ceil_y) / BLOCK_H + 0.5   # only matters upside down
                if (grounded and not on_block) or ceil_rel < 0.5:
                    ceil_rel = 0.5
        self._last_floor_rel, self._last_ceil_rel = floor_rel, ceil_rel
        # learn from the step that just happened: what the preview expected vs what the icon did
        me = self.mode_est
        key = (me.mode, me.g, me.mini)
        sc_ = getattr(ents, "scroll", None)
        cam_moving = sc_ is not None and abs(sc_) > 0.0025 * self.frame_skip
        if (self._mstate is not None and pl is not None and cam_moving and self._mstate[3] == key
                and me._hint_t < 25 and abs(vy_bps) < 40 and self.presses_working and self.use_motion_fix):
            body0, (fl0, ce0), spd0 = self._mstate[:3]
            try:
                self.motion_fix.observe(body0, me.eff_hold(0), fl0, ce0, spd0, dt, self.phys, vy_bps)
            except Exception as ex:
                self._loud("motion-learn", ex)
        out: dict = {}
        if pl is not None and grounded and floor_rel is not None and self.mode_est.g == 1:
            # where the icon's REAL centre rests (for the takeoff model of the next jump)
            self._rest_true = (0.75 - ents.player_pos()[1]) / BLOCK_H + floor_rel + 0.5 \
                if abs(floor_rel + 0.5) < 0.35 else None
        elif grounded:
            self._rest_true = None
        fit = self._air_fit(pl, grounded)
        mf, plan, self.last_arcs = physics_features(ents, spd, self.mode_est, self.phys, self.frame_skip,
                                                    vy_bps, grounded, hold, floor_rel, ceil_rel, dec_dt=dt_plan,
                                                    state=fit,
                                                    motion=self.motion_fix if self.use_motion_fix else None,
                                                    out=out, search=self.use_search,
                                                    prefer=self._prefer_path(spd, dt_plan, pl))
        sr_ = out.get("search") or {}
        self._last_path = (list(sr_.get("path") or []), float(pl.y)) if pl is not None else None
        self._mstate = (out["body"], out["surf"], spd, key) if (pl is not None and "body" in out) else None
        if pl is not None and "surf" in out:
            # every mode's physics from its real motion (like the cube jump). Touching or
            # about to touch a floor / ceiling / block doesn't count: a landing inside a
            # piece made the fall look slower than it is
            fl_s, ce_s = out["surf"]
            near = grounded or any(abs(q) < 0.45 for q in list(fl_s) + list(ce_s))
            self.mode_cal.update((0.75 - ents.player_pos()[1]) / BLOCK_H, me.eff_hold(0), dt, me.mode, me.g,
                                 me.mini, near, spd, self.phys,
                                 confident=(self.presses_working and cam_moving
                                            and getattr(me, "lag_locked", False)))   # delay measured
            # pads / orbs the icon is touching now -> their real launch strength
            touching = [(k_, fx_, n_) for x0_, x1_, b_, t_, k_, fx_, _i, n_ in special_boxes(ents, 2.0)
                        if x0_ < 0.5 and x1_ > -0.5 and b_ < 0.5 and t_ > -0.5]
            self.launch_cal.update((0.75 - ents.player_pos()[1]) / BLOCK_H, dt, me.mode, me.g, me.mini,
                                   me.eff_hold(0), touching, self.phys,
                                   confident=self.presses_working and cam_moving)
        self._last_inputs = out.get("inputs")
        self._last_search = out.get("search")
        if "search_error" in out:
            self._loud("search", RuntimeError(out["search_error"]))
        self.timing["physics"] += time.perf_counter() - _t1
        v = encode_obs(ents, hold=hold, prev_py=self._prev_py,
                       speed=spd, hist=self._hist, mode=mf, plan=plan, extra=out.get("extra_plans"),
                       search=out.get("search"))
        self._hist.append((player_vy(ents, self._prev_py) if pl is not None else 0.0, 1.0 if hold else 0.0))
        self._prev_py = ents.player_pos()[1] if pl is not None else self._prev_py
        return v

    def save_motion(self) -> None:
        # (was: only when the learned correction had data -- it's off by default, so the
        # measured jump and input delay were never saved either)
        if not self.motion_path or not (self.motion_fix.n or self.phys.n_cal or self.mode_est.lag
                                        or any(self.mode_cal.counts().values()) or self.launch_cal.samples):
            return
        try:
            save_motion(self.motion_path, self.motion_fix, self.phys, self.mode_est, self.mode_cal,
                        self.launch_cal)
        except Exception as ex:
            self._loud("motion-save", ex)

    def attempt_time(self) -> float | None:
        """Seconds since this attempt started, counted in captured game frames from the
        respawn (exact to one frame, whatever the decision steps did). None = can't tell."""
        s0 = getattr(self, "_att_slot", None)
        if s0 is None or getattr(self, "_gt_last", None) is None:
            return None
        return (self._game_t - s0) / 60.0

    def _prefer_path(self, spd, dt, pl):
        """Last step's drawn line, moved into this step's frame (the icon went spd*dt blocks
        forward), for the search to stick to when other routes are just as safe."""
        lp = getattr(self, "_last_path", None)
        if pl is None or not lp or not lp[0]:
            return None
        dx = float(spd or SPEED_1X) * float(dt or self.frame_skip / 60.0)
        dy = (lp[1] - float(pl.y)) / BLOCK_H        # how far the icon went UP on screen since
        pts = [(x - dx, y - dy) for x, y in lp[0] if x - dx >= 0.0]
        return pts if len(pts) >= 3 else None

    def _snap_record(self, a: int, v) -> dict:
        """One step for the death snapshots (frame + what the AI knew and did)."""
        import death_snaps
        hi = self.last_hires
        if hi is None:
            import cv2
            hi = cv2.resize(np.ascontiguousarray(self.last_frame[:, :, ::-1]), (640, 360))
        ents = self.last_ents
        pl = ents.player()
        me = self.mode_est
        prog = None
        try:
            prog = float(self.base.reward.last_pct) / 100.0     # best effort, may not exist
        except Exception:
            pass
        d = self._dbg
        return {
            "frame": np.array(hi, copy=True),
            "boxes": [(int(e.cls), float(e.x), float(e.y), float(e.w), float(e.h), int(getattr(e, "pt", -1)))
                      for e in ents.items],
            "player": pl is not None,
            "player_pos": tuple(float(q) for q in ents.player_pos()) if pl is not None else None,
            "arcs": list(self.last_arcs),
            "a": int(a), "a_brain": int(d.get("a_brain", a)), "shielded": bool(d.get("shielded", False)),
            "qdiff": d.get("qdiff"),
            "mode": me.mode, "g": int(me.g), "mini": bool(me.mini), "lag": int(me.lag),
            "plan": death_snaps.plan_summary(v[OFF_PLAN:OFF_PLAN + N_PLAN], v[OFF_RES:OFF_RES + 6]),
            "n_spikes": sum(1 for e in ents.items if int(e.cls) == 0),
            "y": (0.75 - ents.player_pos()[1]) / BLOCK_H if pl is not None else None,
            "vy": float(self._hist[-1][0]) / self._step_dt(peek=True) if self._hist else 0.0,
            "dt": self._step_dt(peek=True), "speed": float(self.speed or SPEED_1X), "prog": prog,
            "floor_rel": getattr(self, "_last_floor_rel", None), "ceil_rel": getattr(self, "_last_ceil_rel", None),
            "preview_inputs": getattr(self, "_last_inputs", None),
            "search": ({k: round(float(self._last_search[k]), 3) for k in SEARCH_KEYS}
                       if self._last_search else None),
            "lines": [None if q is None else round(float(q), 4) for q in self._lines],
        }

    def _new_attempt(self) -> None:
        self.launch_cal.pending = None       # (a launch measurement cut by the death)
        self._press_pending = []
        self._last_yb = None
        self._air = []
        self._debris = []
        self._ceil_y = None        # ceilings belong to a section; the floor is the level's
        self._ceil_hist.clear()
        self._snap.clear()
        self._peak = 0.0
        self._mstate = None
        self._n_deaths += 1
        m = self.motion_fix
        if self._n_deaths % 10 == 0:
            print(f"[physics] jump {self.phys.apex:.2f}b/{self.phys.airtime:.2f}s ({self.phys.n_cal} jumps), "
                  f"delay {self._delay_txt()}; {self.mode_cal.summary(self.phys)}  (? = not enough yet)")
            print(f"[physics] pads / orbs (x cube jump): {self.launch_cal.summary()}")
        if self._n_deaths % 10 == 0 and m.n:
            print(f"[motion] {m.n} examples  formulas off by {m.err_phys:.2f} blocks/s, "
                  f"with learned fix {m.err_corr:.2f}  -> using it at {m.trust:.0%}  "
                  f"(jump {self.phys.apex:.2f}b/{self.phys.airtime:.2f}s, delay {self._delay_txt()})")
        if time.time() - self._motion_t > 30:
            self._motion_t = time.time()
            self.save_motion()
        self._moving = []
        self.mode_est.reset()      # every attempt starts as a normal cube
        self._hist.clear()
        self._last_moved = True
        self._prev_py = None
        try:
            self.det.reset()
        except Exception:
            pass
        if self.base is not None:
            try:
                self.base._steps = 0  # never let GymGDEnv's 2-min cap truncate
            except Exception:
                pass

    # -- gym API --------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        info: dict = {}
        self._att_slot = self._game_t      # attempt clock (the first attempt is a little off: it
                                           # isn't started by a respawn; the route memory copes)
        if not self._started:
            self._ensure_focus(force=True)
            obs, info = self.env.reset(seed=seed, options=options)
            self._started = True
            frame = self._newest(obs)
        else:
            frame = self.last_frame
        try:
            self._prev_attempts = int(info.get("attempts", self._prev_attempts))
        except Exception:
            pass
        self._new_attempt()
        self.motion.reset()
        if frame is None:
            frame = np.zeros((180, 320, 3), np.uint8)
        return self._observe(frame, False), info

    def step(self, action):
        a = int(np.asarray(action).reshape(-1)[0])
        if not self._in_level:
            a = 0          # menus / loading / level-complete screen: never press SPACE there
        died = complete = paused = False
        px_moved = None          # pixel-shift verdict over this step's frames
        info: dict = {}
        frame = None
        # (OFF by default -- --fast-steps turns it on. Live, it made decision lengths flip
        # between 2 and 3 frames and the agent did worse than at a steady 3.)
        # Frame pacing: the capture runs on a fixed 60 Hz grid and each env.step waits for
        # the NEXT slot. Counting reads, a decision whose processing ran past one slot (> 17
        # ms) took 3 slots (50 ms) instead of 2: live runs went at ~20 decisions/s, not 30.
        # Count grid slots instead: once frame_skip slots have passed since the last
        # observation, observe now.
        cap = getattr(self.base, "cap", None) if self.base is not None else None
        slots = (lambda: int(getattr(cap, "_ticks", 0)) + int(getattr(cap, "_overruns", 0))) \
            if cap is not None and hasattr(cap, "_ticks") and self.fast_steps else None
        s_mark = getattr(self, "_slot_mark", None)
        for _k in range(self.frame_skip):
            obs, _r, term, _trunc, info = self.env.step(a)
            f = self._newest(obs)
            if f is not None:
                frame = f
                shift = self.motion.update(f)
                if cap is not None and hasattr(cap, "_ticks"):
                    # GAME time: capture frames while the picture moves (a paused game --
                    # training pauses, menus -- doesn't advance the level, so doesn't count)
                    now_s = int(getattr(cap, "_ticks", 0)) + int(getattr(cap, "_overruns", 0))
                    last_s = getattr(self, "_gt_last", None)
                    self._gt_last = now_s
                    if last_s is not None and shift != 0.0:
                        self._game_t += min(6, now_s - last_s)
                if shift is not None:
                    px_moved = bool(px_moved) or shift > self.move_px
            if info.get("paused"):
                paused = True
                break
            try:
                att = int(info.get("attempts", self._prev_attempts))
            except Exception:
                att = self._prev_attempts
            if str(info.get("event")) == "death" or att > self._prev_attempts:
                died = True
                self._att_slot = self._game_t                    # the attempt clock restarts here
                self._prev_attempts = max(self._prev_attempts, att)
                break
            if str(info.get("event")) == "complete" or term:
                complete = True
                try:
                    self.base.reward.reset()   # don't stay latched on "complete"
                except Exception:
                    pass
                break
            if slots is not None and s_mark is not None and slots() - s_mark >= self.frame_skip:
                break                          # the slots already passed (processing ran long)
        if slots is not None:
            self._slot_mark = slots()
            self.frames_per_dec = 0.95 * getattr(self, "frames_per_dec", float(self.frame_skip)) + \
                0.05 * max(1, self._slot_mark - (s_mark if s_mark is not None else self._slot_mark - self.frame_skip))
        if frame is None:
            frame = self.last_frame
        v = self._observe(frame, a == 1)
        self._ensure_focus()
        self._debris.append(self._debris_count())
        if self.death_dir:
            try:
                self._snap.append(self._snap_record(a, v))
            except Exception as ex:
                self._loud("death-snap", ex)
        sc = getattr(self.last_ents, "scroll", None)
        if sc is not None:   # tracked hazards: frozen camera == they stopped moving
            ref = float(np.median(self._scroll_hist)) if len(self._scroll_hist) >= 5 else 0.009 * self.frame_skip
            moved = abs(sc) > max(0.0025 * self.frame_skip, 0.4 * ref)
            if moved:
                self._scroll_hist.append(abs(sc))
        elif px_moved is not None:
            moved = px_moved
        else:
            moved = self._last_moved   # can't tell this step: keep the last verdict
        self._last_moved = moved
        self._update_speed(sc, moved)
        self.last_shift = float(sc) if sc is not None else -1.0
        if isinstance(info.get("prog"), (int, float)) and not died:
            self._peak = max(self._peak, float(info["prog"]))
        out_info = {
            "moving": moved, "prog": info.get("prog"), "attempts": self._prev_attempts,
            "died": died, "complete": complete, "paused": paused,
            "death_lag": 0, "death_lag_src": "",
        }
        if not paused:
            self._moving.append(moved)
        if died:
            if self.death_lag_s is not None:
                lag = min(int(round(self.death_lag_s * 60 / self.frame_skip)), max(0, len(self._moving) - 1))
                src = "fixed"
            else:
                raw, src = estimate_death_lag(self._moving, self.fallback_steps)
                out_info["death_lag_raw"] = raw
                if src == "measured":
                    self._lag_hist.append(raw)
                # GD's crash -> respawn delay is the same every death, but a single
                # measurement is noisy (explosion over the hazard, % read jitter):
                # live logs showed 15..42 steps around a true ~24. Over-cutting
                # removes the very jump that caused the death, so once a few deaths
                # are measured use their running median for every death.
                if len(self._lag_hist) >= 5:
                    lag, src = int(np.median(self._lag_hist)), "median"
                else:
                    lag = raw
                lag = min(lag, max(0, len(self._moving) - 1))
            lag2 = self._explosion_lag(lag)
            if lag2 is not None and lag2 != lag:
                lag, src = lag2, src + "+explosion"
            out_info["death_lag"], out_info["death_lag_src"] = lag, src
            out_info["attempt_steps"] = len(self._moving)
            cause = None
            if self._snap:
                try:                             # how it died (death_cause.py)
                    import death_cause
                    cause = death_cause.classify(list(self._snap), lag)
                    out_info["death_cause"] = cause["text"]
                    out_info["death_cause_d"] = cause
                except Exception as ex:
                    self._loud("death-cause", ex)
            if self.death_dir and self._snap:
                try:
                    import death_snaps
                    death_snaps.save(list(self._snap), lag, int(self._dbg.get("attempt", self._n_deaths + 1)),
                                     self.death_dir, self.death_keep, peak_pct=100.0 * self._peak, cause=cause)
                except Exception as ex:
                    self._loud("death-snap-save", ex)
            self._new_attempt()
        out_info["n_entities"] = len(self.last_ents.items)
        self._update_in_level(info)
        out_info["in_level"] = self._in_level
        return v, 0.0, complete, False, out_info

    def _update_in_level(self, info: dict) -> None:
        """In a level = the icon is on screen AND the level's % is showing (it isn't on the
        menus). If the % has never been read this session (turned off in GD's settings),
        the icon + a moving level have to do."""
        pl = self.last_ents.player() if self.last_ents is not None else None
        self._no_icon = 0 if pl is not None else self._no_icon + 1
        age = info.get("pct_age")
        if age is not None and age < 900:
            self._pct_ever = True
        if age is not None and self._pct_ever:
            ok = age < 90                       # the level's % read within the last 1.5 s
        else:                                   # (% turned off: the icon has to do; deaths hide
            ok = self._no_icon <= 60            #  it for ~1 s, so allow 2 s without it)
        was = self._in_level
        # (a frozen picture = paused / a menu over the level: no presses either)
        self._in_level = bool(ok and not info.get("frozen"))
        if was != self._in_level and getattr(self, "_started", False):
            print("[input] in a level: inputs on" if self._in_level else
                  "[input] not in a level (menu / loading?): inputs off until a level starts")

    # -- pause protocol passthrough ------------------------------------------
    def pause(self) -> None:
        if self.base is not None:
            self._ensure_focus(force=True)     # ESC must reach GD, not another window
            self.base.pause()

    def resume(self) -> None:
        if self.base is not None:
            self._ensure_focus(force=True)     # the unpause key too
            self.base.resume()
            self.motion.reset()

    # -- input sanity -----------------------------------------------------------
    def _debris_count(self) -> int:
        """Coloured bits (orb / pad / portal boxes) right around the icon: the crash
        explosion makes lots of them appear at once."""
        ents = self.last_ents
        if ents.player() is not None:
            self._last_pp = ents.player_pos()
        if self._last_pp is None:
            return 0
        px, py = self._last_pp
        return sum(1 for e in ents.items if 1 <= int(e.cls) <= 5 and abs(e.x - px) < 0.12 and abs(e.y - py) < 0.2)

    def _explosion_lag(self, lag: int):
        """Refine one death's crash time: the explosion starts within a second of where
        the usual crash->% reset delay puts it. Live snapshots: the fixed delay was right
        within ~4 steps for most deaths, but 12-20 steps late for some."""
        deb = self._debris
        n = len(deb)
        crash = n - 1 - int(lag)
        lo = max(1, crash - 22)
        for i in range(lo, min(n - 1, crash + 7)):
            if deb[i] >= 2 and deb[i + 1] >= 2:          # a burst (normal play: ~never 2+)
                j = i
                while j > lo and deb[j - 1] >= 1:        # back to where it started building up
                    j -= 1
                return max(0, n - 1 - (j - 1))           # visible one step after the impact
        return None

    def _air_fit(self, pl, grounded: bool):
        """Height and speed in the air from an ARC FIT through the recent heights (known
        gravity, real step times) instead of the last two frames. The player box wobbles
        ~0.24 blocks, which made the 2-frame speed off by ~8 blocks/s (a jump starts at
        ~20) and the predicted landing jumped around step to step.
        Returns (height offset vs the measured centre, speed) or None (use the raw values)."""
        me = self.mode_est
        y = self._last_yb if pl is not None else None
        if y is None or grounded or me.mode not in ("cube", "robot", "ball", "ufo", "spider"):
            self._air = [] if y is None else [(0.0, y)]
            self._air_from_rest = y is not None and grounded     # an arc starting from the ground
            self._air_rest = self._rest_true if (grounded and self._rest_true is not None) else y
            return None
        t = (self._air[-1][0] + self._motion_dt()) if self._air else 0.0
        if self._air and abs(y - self._air[-1][1]) > 3.0:
            self._air = []                       # teleport / respawn: start over
            self._air_from_rest = False
        G = {"ball": self.phys.ball_g, "ufo": self.phys.ufo_g}.get(me.mode, self.phys.cube_g) * me.g
        v_jump = self.phys.cube_v0 * (self.phys.mini if me.mini else 1.0) * me.g
        if me.mode == "cube" and len(self._air) >= 2 and me.g * (
                (y - self._air[-1][1]) / max(1e-3, t - self._air[-1][0])
                - (self._air[-1][1] - self._air[-2][1]) / max(1e-3, self._air[-1][0] - self._air[-2][0])) > 0.5 * abs(v_jump):
            # bounced: held SPACE through a landing and jumped again without a resting frame.
            # A new arc from the ground (the old one made the fit fail -> raw 2-frame speed)
            self._air = [(t - self._motion_dt(), self._air[-1][1])]
            self._air_from_rest = True
            self._air_rest = self._rest_true if self._rest_true is not None else self._air[0][1]
        if me.mode in ("ufo", "ball") and me.eff_hold(0) and not me.eff_hold(1):
            self._air = self._air[-8:]           # a UFO tap / ball flip starts a new arc
        self._air.append((t, y))
        self._air = self._air[-8:]
        from_rest = getattr(self, "_air_from_rest", False) and me.mode == "cube"
        if len(self._air) < (2 if from_rest else 3):
            return None
        if from_rest and len(self._air) <= 12:
            # From the ground the takeoff speed is KNOWN (a jump: the measured jump speed;
            # walking off a ledge: 0): only the takeoff moment inside the first step is
            # unknown. Much steadier than fitting the speed from a few wobbly heights --
            # and usable from the FIRST frame in the air (live: the raw speed there was
            # ~4 blocks/s for a jump really going up at ~18, so it "fell" onto the spike).
            T = np.array([q[0] for q in self._air]); Y = np.array([q[1] for q in self._air])
            rest = getattr(self, "_air_rest", None)
            y_rest, t1 = (Y[0] if rest is None else float(rest)), (T[1] if len(T) > 1 else 0.0)
            best = None
            for v0 in (v_jump, 0.0):
                for tau in np.linspace(0.0, t1, 6):
                    dtm = np.clip(T - tau, 0.0, None)
                    m = y_rest + v0 * dtm - 0.5 * G * dtm ** 2
                    err = float(np.sqrt(np.mean((Y[1:] - m[1:]) ** 2)))
                    if best is None or err < best[0]:
                        best = (err, v0, tau)
            if best is not None and best[0] < 0.5:
                err, v0, tau = best
                tc = T[-1] - tau
                return (float(y_rest + v0 * tc - 0.5 * G * tc * tc - y), float(v0 - G * tc))
        for pts in (self._air, self._air[-3:]):
            T = np.array([q[0] for q in pts]); Y = np.array([q[1] for q in pts])
            A = np.stack([np.ones_like(T), T], 1)
            (a, b), *_ = np.linalg.lstsq(A, Y + 0.5 * G * T ** 2, rcond=None)
            r = Y - (a + b * T - 0.5 * G * T ** 2)
            if float(np.max(np.abs(r))) < 0.45:
                tc = T[-1]
                return (float(a + b * tc - 0.5 * G * tc * tc - y), float(b - G * tc))
        return None

    @staticmethod
    def _line_is_platform(ents, line_y: float) -> bool:
        """Is this bright horizontal line the top of a detected block (in the icon's half of
        the screen)? Then it's a platform's edge, not the level's ground."""
        px = ents.player_pos()[0]
        for e in ents.items:
            if int(e.cls) != CLS_ID["wall"] or e.x + e.w / 2 < px - 6 * BLOCK_W or e.x - e.w / 2 > px + 8 * BLOCK_W:
                continue
            if abs((e.y - e.h / 2) - line_y) < 0.35 * BLOCK_H:
                return True
        return False

    @staticmethod
    def _on_block(ents, py: float, g: int) -> bool:
        """Is a detected block right under (or, upside down, over) the icon?"""
        px = ents.player_pos()[0]
        for e in ents.items:
            if int(e.cls) != CLS_ID["wall"] or abs(e.x - px) > e.w / 2 + 0.3 * BLOCK_W:
                continue
            top, bot = e.y - e.h / 2, e.y + e.h / 2
            if g == 1 and 0.15 * BLOCK_H < top - py < 0.9 * BLOCK_H:
                return True
            if g == -1 and 0.15 * BLOCK_H < py - bot < 0.9 * BLOCK_H:
                return True
        return False

    def _ensure_focus(self, force: bool = False) -> None:
        """SPACE only reaches GD when GD is the active window: check every ~2 s
        (and before pausing / resuming); bring it back to the front if not."""
        if self.base is None:
            return
        self._focus_n += 1
        if not force and self._focus_n % 60:
            return
        try:
            import winfocus
            fg = winfocus.gd_is_foreground()
            if fg is False:
                ok = winfocus.focus_gd()
                now = time.time()
                if now - self._focus_warn_t > 20:
                    self._focus_warn_t = now
                    print("[input] Geometry Dash was NOT the active window -- SPACE was going to another "
                          "window. " + ("Brought GD back to the front." if ok else
                                        "Couldn't bring it back: click into the GD window!"))
            elif fg is None and not self._focus_unknown_shown and self._focus_n > 1:
                self._focus_unknown_shown = True
                print("[input] (can't find a window titled 'Geometry Dash' to keep it in front -- "
                      "make sure GD stays the active window yourself)")
        except Exception as ex:
            self._loud("focus", ex)

    def _delay_txt(self) -> str:
        d = getattr(self.mode_est, "delay_s", None)
        return f"{d * 1000:.0f} ms" if d is not None else f"{self.mode_est.lag} step(s)"

    def _measure_delay_ms(self, dy: float, t_dec, t_still, t_now) -> None:
        """One press's input delay in SECONDS: from when the picture it was decided on was
        shown to when the icon started to move. The first moving picture shows the icon
        already `dy` up; for a cube, the jump's own arc says how long ago it started
        (else: halfway between the last still picture and this one)."""
        if t_dec is None or t_still is None or t_now is None or not (t_dec <= t_still < t_now):
            return
        me = self.mode_est
        start = 0.5 * (t_still + t_now)
        if me.mode in ("cube", "robot") and not me.mini:
            v0, g = self.phys.cube_v0, self.phys.cube_g
            disc = v0 * v0 - 2.0 * g * dy
            tau = (v0 - math.sqrt(disc)) / g if disc > 0 else v0 / g
            start = min(t_now, max(t_still, t_now - tau))
        d = start - t_dec
        if 0.0 <= d <= 0.4:
            self._press_delay_s.append(d)

    def _publish_delay_ms(self) -> None:
        """The blue line's input delay: the median of the measured ones, in ms -- not
        rounded to whole steps (live: rounding 1.5 steps up to 2 made it expect every jump
        later than it really started, so its presses came early). The worse world uses the
        far end of the spread, at least a whole step off (as before)."""
        q = list(self._press_delay_s)
        if len(q) < 5:
            return
        q = np.asarray(q, np.float64)
        d = float(np.median(q))
        lo, hi = float(np.percentile(q, 15)), float(np.percentile(q, 85))
        nom = self._step_dt(peek=True)
        off = max(nom, 1.5 * max(d - lo, hi - d))     # (at least a step: as safe as before)
        alt = d - off if (d - lo) > (hi - d) and d - off >= 0 else d + off
        me = self.mode_est
        old = getattr(me, "delay_s", None)
        me.delay_s, me.delay_alt_s = d, alt
        if old is None or abs(old - d) > 0.008:
            print(f"[input] input delay {d * 1000:.0f} ms ({d / nom:.2f} steps; spread "
                  f"{lo * 1000:.0f}-{hi * 1000:.0f} ms) -- the blue line uses it as is")

    def _watch_press(self, y_b, hold: bool, resting: bool, alive: bool) -> None:
        """Did SPACE do anything? After a press from rest, the icon should move
        within ~0.4 s in every mode. If most presses do nothing, the key isn't
        reaching the game (focus, wrong key, game paused...) -- say so loudly."""
        if y_b is None or not alive:
            self._press_pending = []
            return
        prev = bool(self._hist and self._hist[-1][1] > 0.5)
        y_before = self._last_yb
        self._last_yb = y_b
        self._since_press = getattr(self, "_since_press", 99) + 1
        if hold and not prev:
            # only isolated presses from rest: a press soon after another one can catch the
            # first one's (delayed) jump starting and look like "no delay"
            if resting and y_before is not None and not self._press_pending and self._since_press >= 5:
                # (steps since the press, height before it, when the picture the press was
                #  decided on was shown, when the last still picture was shown)
                t_dec = getattr(self, "_obs_ts_prev", None)
                self._press_pending.append((0, y_before, t_dec, t_dec))
            self._since_press = 0
        still = []
        t_now = getattr(self, "_obs_ts", None)
        for age, y0, t_dec, t_still in self._press_pending:
            if abs(y_b - y0) > 0.3:
                self._press_ok.append(True)
                self._press_delay.append(age)             # 0 = it reacted during the press step
                self._measure_delay_ms(abs(y_b - y0), t_dec, t_still, t_now)
            elif age >= 11:
                self._press_ok.append(False)
            else:
                still.append((age + 1, y0, t_dec, t_now))
        self._press_pending = still
        if len(self._press_delay) >= 5:
            # the measured input delay beats inferring it from motion. Live it sits between
            # 1 and 2 steps (decisions take 33 or 50 ms): round UP (pressing a hair early is
            # much safer than late) and only change after it has said so 5 presses running
            med = float(np.median(self._press_delay))
            want = int(np.clip(np.ceil(med - 0.25), 0, ModeEstimator.MAX_LAG))   # 1.5 -> 2
            c, k = self._lag_vote
            k = k + 1 if c == want else 1
            self._lag_vote = (want, k)
            if want != self.mode_est.lag and (k >= 5 or not self.mode_est.__dict__.get("lag_locked")):
                print(f"[input] measured input delay: {want} step(s) between pressing and the icon moving")
                self.mode_est.lag = want
            self.mode_est.lag_locked = True
            # the other likely delay, for the search's second world: live the delays split
            # between 1 and 2 steps, and "2" made jumps start a step early half the time
            L = self.mode_est.lag
            lo = float(np.mean([d < L for d in self._press_delay]))
            self.mode_est.lag_alt = L - 1 if (L >= 1 and lo >= 0.25) else L + 1
        self._publish_delay_ms()
        r = list(self._press_ok)
        if len(r) >= 8 and sum(r) <= 1:
            now = time.time()
            if now - self._press_warn_t > 30:
                self._press_warn_t = now
                print(f"[input] WARNING: {len(r) - sum(r)} of the last {len(r)} presses did NOTHING (the icon "
                      f"didn't move). SPACE isn't reaching Geometry Dash: click into the GD window so it's "
                      f"the active one. Training on this teaches the AI that jumping is useless!")
                self._ensure_focus(force=True)

    @property
    def presses_working(self) -> bool:
        """False when recent presses mostly did nothing (then don't learn physics from it)."""
        r = list(self._press_ok)
        return not (len(r) >= 5 and sum(r) < 0.5 * len(r))

    def mark_trained(self) -> None:
        if self.base is not None:
            self.base.mark_trained()

    @property
    def is_paused(self) -> bool:
        return bool(getattr(self.base, "_paused", False))
