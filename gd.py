"""Step 1.2 + 1.3 -- input + action space: SPACE hold/release at 60Hz.

 
Contract:
  - [`pydirectinput.PAUSE`](gd.py:22) = 0 and [`pydirectinput.FAILSAFE`](gd.py:23) = False
    at import. The default 0.1s pause silently caps throughput at
    ~10 actions/sec, which breaks a 60-decision/sec loop.
  - [`GDInput.set()`](gd.py:54) / [`GDInput.step()`](gd.py:76) send
    [`keyDown`](gd.py:66) / [`keyUp`](gd.py:70) ONLY on action-state
    change. Holding across ticks costs zero syscalls.
  - Non-blocking: no sleep, no per-step alloc. Caller paces via
    [`GDCapture.read()`](capture.py:336); input never throttles capture.
  - [`close()`](gd.py:88) / context-manager exit always releases the key
    so training crashes never leave SPACE stuck down.
  - Action space [`Discrete(2)`](gd.py:34): 0 = release, 1 = hold (SPACE
    only). One decision per capture tick: 240 TPS / 4 ticks = 60 dec/sec.

Default key is SPACE (Geometry Dash jump/hold) -- the ONLY input used.

Hybrid reward (pixels only, no game-state access):
  - Dense: top-center bar fill delta every tick (progress made = reward).
  - Anchor: x% text re-read at ~1Hz via digit templates to kill bar drift.
  - Death: fill reset -> -1 + attempt++ ; completion: template match -> +10.
  - Stalling pays nothing: no fill delta = no reward.
"""

from __future__ import annotations

from typing import Any

try:
    import pydirectinput
except ImportError as e:  # pragma: no cover - import-time hint
    raise ImportError(
        "pydirectinput is required on Windows: pip install pydirectinput"
    ) from e

try:
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover - import-time hint
    raise ImportError(
        "gymnasium is required: pip install gymnasium"
    ) from e

# Kill the hidden 0.1s inter-call pause and the corner-failsafe. Both add
# latency/jitter a 60Hz policy loop cannot afford.
pydirectinput.PAUSE = 0
pydirectinput.FAILSAFE = False

__all__ = [
    "GDInput",
    "DEFAULT_KEY",
    "RELEASE",
    "HOLD",
    "ACTION_SPACE",
    "PHYSICS_TPS",
    "TICKS_PER_DECISION",
    "DECISIONS_PER_SEC",
    "validate_action",
    "action_to_hold",
    "RewardConfig",
    "ProgressReward",
]

DEFAULT_KEY = "space"

# --- 1.3 Action space: Discrete(2), SPACE only ------------------------ #
RELEASE: int = 0  # action 0 -> SPACE up
HOLD: int = 1  # action 1 -> SPACE down
PHYSICS_TPS: int = 240  # GD physics rate
TICKS_PER_DECISION: int = 4  # physics ticks per policy decision
DECISIONS_PER_SEC: int = PHYSICS_TPS // TICKS_PER_DECISION  # 60
assert DECISIONS_PER_SEC == 60
ACTION_SPACE: spaces.Discrete = spaces.Discrete(2)


def validate_action(action: Any) -> int:
    """Strict Discrete(2) check: unwrap size-1 arrays/scalars, require 0/1.

    Returns the int action (0 or 1). Raises ValueError on anything outside
    the space (e.g. 2, -1, NaN, None, size!=1 arrays) so bad policy output
    fails loudly instead of silently pinning SPACE down.
    """
    import math

    a = action
    if hasattr(a, "size") and hasattr(a, "flat"):
        try:
            if a.size != 1:
                raise ValueError(f"action {action!r} not in Discrete(2): size != 1")
            a = a.flat[0]
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"action {action!r} not in Discrete(2): {e}") from e
    if isinstance(a, bool):
        return int(a)
    if isinstance(a, float):
        if math.isnan(a) or math.isinf(a) or not a.is_integer():
            raise ValueError(f"action {action!r} not in Discrete(2)")
        a = int(a)
    else:
        try:
            a = int(a)
        except (TypeError, ValueError) as e:
            raise ValueError(f"action {action!r} not in Discrete(2): {e}") from e
    if a not in (RELEASE, HOLD):
        raise ValueError(f"action {action!r} not in Discrete(2): want 0 or 1")
    if a not in ACTION_SPACE:
        raise ValueError(f"action {a} rejected by ACTION_SPACE")
    return a


def action_to_hold(action: Any) -> bool:
    """Map a validated Discrete(2) action to hold state: 1 -> True."""
    return bool(validate_action(action))


class GDInput:
    """Edge-triggered holder: one syscall per transition, none per hold.

    Input goes STRAIGHT to the global keyboard (pydirectinput, PAUSE=0).
    GD must be the FOREGROUND window for the game to receive keys -- the
    earlier GD-window PostMessage targeting never actuated GD's input
    thread, so pause-on-death silently did nothing. No hwnd, no
    PostMessage, no foreground logic: exactly the pre-window-backend
    behavior (plus SPACE hold/release change-only).
    """

    def __init__(self, key: str = DEFAULT_KEY, backend: Any | None = None) -> None:
        self.key: str = key
        self._backend: Any = backend if backend is not None else pydirectinput
        self._held: bool = False
        self._downs: int = 0  # keyDown syscalls issued
        self._ups: int = 0  # keyUp syscalls issued
        self._changes: int = 0  # transitions (downs + ups)
        self._steps: int = 0  # set()/step() calls
        self._closed: bool = False

    # ------------------------------------------------------------------ #
    @property
    def held(self) -> bool:
        """Currently-commanded hold state (no I/O)."""
        return self._held

    @property
    def health(self) -> dict:
        """Counters for logging: steps vs actual syscalls."""
        return {
            "key": self.key,
            "held": self._held,
            "steps": self._steps,
            "downs": self._downs,
            "ups": self._ups,
            "changes": self._changes,
            "closed": self._closed,
        }

    def set(self, hold: bool) -> bool:
        """Command hold/release. Returns True iff a syscall was issued.

        Idempotent: repeated identical commands are free no-ops so a
        60Hz loop calling every tick pays only for real transitions.
        """
        want = bool(hold)
        self._steps += 1
        if want == self._held or self._closed:
            return False
        if want:
            self._backend.keyDown(self.key)
            self._downs += 1
        else:
            self._backend.keyUp(self.key)
            self._ups += 1
        self._held = want
        self._changes += 1
        return True

    def step(self, action: Any) -> bool:
        """Policy-facing entry: strict Discrete(2) -> SPACE hold/release.

        0 -> release, 1 -> hold. Anything outside the space raises
        ValueError (see validate_action) so a corrupt head can never
        silently pin SPACE down. Only transitions hit the OS.
        """
        return self.set(action_to_hold(action))

    def release(self) -> None:
        """Force key up now (no-op if already up). Counts as a transition."""
        self.set(False)

    def close(self) -> None:
        """Release the key once and latch closed; further sets are no-ops."""
        if not self._closed:
            if self._held:
                try:
                    self._backend.keyUp(self.key)
                finally:
                    self._ups += 1
                    self._changes += 1
                    self._held = False
            self._closed = True

    # Context-manager + destructor safety: never leave SPACE stuck down.
    def __enter__(self) -> "GDInput":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:  # best-effort only, never raise
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------- #
# 1.4 Reward: x% text PRIMARY, bar fill FALLBACK/debug only, SCENE-SNAP
# death detector independent of both. Pixels only: never touches game
# memory. The x% text (full-res crop, digit templates at ~1Hz, no OCR
# dep) is the dense progress signal; bar fill on 320x180 proved
# unreliable and only fills gaps between % reads. Death does NOT trust
# bar/% ROIs alone (both are guessed screen spots until calibrated):
# a real death is a full-frame SCENE SNAP -- mean abs frame diff spikes
# far above scroll rate, then settles on the retry start. Pause/menus are
# time-FROZEN frames (diff ~0 for 30+ ticks), auto-suppressed without any
# manual P key. No SAM, no segmentation: two stats, both O(pixels).
# ---------------------------------------------------------------------- #
from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    """Tunable knobs for ProgressReward. Fractions, not percents (0..1)."""

    # Measured from user 2560x1440 screenshot: hollow bar outline spans
    # x ~900-1870 (0.352-0.730), y ~10-60 (0.007-0.042); the white FILL
    # lives strictly inside (inset ~10px). Fractions are resolution-free,
    # so the same values map onto the 320x180 obs.
    bar_y0: float = 0.008  # top-center band start (bar sits near top)
    bar_y1: float = 0.038  # band end (fill interior only, not outline)
    bar_x0: float = 0.356  # left edge of bar fill interior
    bar_x1: float = 0.726  # right edge of bar fill interior
    # GD progress bar: SOLID fill in the CHARACTER's color (blue, orange,
    # green -- whatever cube/ship color is equipped) inside a thin WHITE
    # hollow outline. Fixed hue bands always break on color change, so the
    # gate is hue-free: ANY saturated+bright pixel counts (S >= sat_min AND
    # V >= val_min). White outline fails saturation, dark bg fails value,
    # gray level geometry fails saturation. Hue is intentionally NOT tested.
    sat_min: int = 90  # below = gray/white outline/dark bg, not fill
    val_min: int = 90  # below = dark background, ignore
    hue_lo: int = 0  # unused (kept for back-compat); hue is NOT gated
    hue_hi: int = 179  # unused (kept for back-compat); hue is NOT gated
    row_frac_min: float = 0.15  # kept for back-compat, unused by _left_run
    # Reward shaping: PROGRESS-DELTA + survival floor + death penalty.
    # tb diagnosis: pure survive 0.01/tick * 7200-step episodes = +26.75 flat
    # while a death pays -1 (1/27th of the episode). Model learned camping
    # the spawn = +reward. Now forward steps pay scaled by prog_delta_scale,
    # backward steps cost down_scale, survival floor keeps time alive only
    # when flat (0.001/tick), deaths cost -5 (a full attempt's progress).
    # Learned-behavior target: survive AND advance, not camp.
    survive_pay: float = -0.02  # every alive tick costs (death = 500 ticks of life)
    dense_scale: float = 1.0  # legacy multiplier (kept for compat)
    prog_delta_scale: float = 20.0  # climb must dominate survival math
    down_scale: float = 10.0  # -per unit backward progress per tick
    deadband: float = 1e-4  # progress jitter below this pays floor, not delta
    jump_pay: float = 2.0  # lump sum on NEW best % (salient, survives GAE)
    air_pay: float = 0.05  # +per held tick while airborne-climbing (jump timing)
    hold_pay: float = 0.005  # +per held tick (any posture): holding is the action, teach it first
    prox_radius: float = 0.25  # edge distance inside which spikes cost rent
    prox_scale: float = 2.0  # -per tick at edge 0, scaled by (1 - edge/radius)
    clear_pay: float = 3.0  # +lump sum when edge GROWS while held (jump cleared it)
    clear_cool: int = 10  # ticks between clearance payouts (no farming one spike)
    first_spike_pay: float = 5.0  # +lump when the FIRST spike of an attempt is cleared (the jump that matters most)
    peak_pay: float = 3.0  # +lump per NEW best-% point this attempt (curiosity: forces searching past 2%)
    first_spike_edge: float = 0.30  # only when nearest spike started inside this range (the opener, not far spam)
    max_delta_clip: float = 0.05  # clip single-tick jump (spam reads)
    death_penalty: float = -5.0  # death must exceed any camp+climb sum (peak-scaled below)
    completion_bonus: float = 10.0  # level complete
    completion_frac: float = 0.995  # kept for back-compat; unused
    # PHASE 3 primary: pct-drop death. Value was above drop_hi, then at or
    # below drop_lo for drop_hold ticks (2+). Fresh text only; stale never
    # arms. Commits through _commit_death as via="pctdrop".
    drop_hi: float = 0.03
    drop_lo: float = 0.02
    drop_hold: int = 2
    # Ablation flags: each detector independently on/off for eval ranking.
    # Phase 3 spec: pctdrop primary; snap only cross-checks, never commits
    # alone; losers get deleted after the numbers come in.
    use_pctdrop: bool = True
    use_snap: bool = True  # cross-check only (arm), never sole commit
    use_reset: bool = True
    use_sink: bool = True
    use_banner: bool = True
    use_freeze: bool = False  # OFF: identical-frame stall killed learning
    # ATTEMPT banner: big white text mid-screen on retry ("ATTEMPT N").
    # White fraction over this band confirms a death (geometry, not OCR:
    # any large white blob here = banner, not gameplay).
    banner_y0: float = 0.18
    banner_y1: float = 0.32
    banner_x0: float = 0.05
    banner_x1: float = 0.55
    banner_white_frac: float = 0.04  # white fraction = banner present
    banner_confirm_ticks: int = 240  # snap then banner within ~4s = death
    # Death = fill drops by more than this between ticks (not a stall:
    # stall = same fill = zero delta = zero reward, no penalty).
    # Gated: a drop only ARMS a pending death; it CONFIRMS only when fill
    # climbs back above death_rearm_fill within death_confirm_ticks
    # (auto-retry restarted the attempt). A pause/menu hides the bar and
    # stays low, so it expires as "occluded" with no penalty. This is what
    # kills the pause-false-positive without any SAM/segmentation model.
    death_drop: float = 0.08  # arm on a modest single-tick collapse ...
    death_floor: float = 0.05  # ... or when progress sinks to ~empty while moving
    death_confirm_ticks: int = 240  # ~4s at 60Hz: retry can be slow, menu longer
    death_rearm_fill: float = 0.02  # climb above this after a drop = new attempt
    # Scene-snap death: full-frame mean abs diff (grayscale, 0..1).
    snap_arm: float = 0.18  # single-tick scene change this big = snap (death)
    snap_settle: float = 0.06  # ...confirmed when diff falls back under this
    snap_confirm_ticks: int = 240  # ~4s to settle on retry start
    # Freeze = pause/menu: diff under this for freeze_ticks = auto-suspend.
    # (reward gate disabled via use_freeze=False above; env.pause() still
    # suspends explicitly. Identical black probe frames locked suspended.)
    freeze_diff: float = 0.004  # below = frame is time-frozen
    freeze_ticks: int = 30
    # 1Hz anchor: caller passes pct or None; resyncs bar baseline.
    anchor_alpha: float = 1.0  # 1.0 = snap to anchor, <1 = EMA blend
    max_delta_clip: float = 0.05  # clip single-tick delta (particle flash)


@dataclass
class ProgressReward:
    """Tracks bar fill -> dense delta reward + death/completion events."""

    cfg: RewardConfig = field(default_factory=RewardConfig)
    prev_fill: float = 0.0  # last tick's x% progress (text only, never bar)
    best_fill: float = 0.0  # max progress this attempt (anti-farm)
    prev_src: str = "none"  # "pct" | "pct_hold" | "pct_init" | "none"
    _pct_seen: bool = False  # latched on first text read; bar never drives after
    attempts: int = 1  # attempt-counter, incremented on death
    done: bool = False  # latched on completion template match
    total: float = 0.0  # cumulative shaped reward (logging)
    deaths: int = 0  # SINGLE SOURCE OF TRUTH for GUI death count (== attempts - 1)
    alive_ticks: int = 0  # survival ticks this attempt (telemetry)
    _pending: bool = False  # a drop was seen, awaiting retry-climb to confirm
    _pending_age: int = 0  # ticks since the drop armed pending
    _sink: float = 0.0  # cumulative monotonic text decrease since last climb
    _drop_hi_seen: bool = False  # pctdrop: saw value above drop_hi
    _drop_lo_run: int = 0  # pctdrop: consecutive ticks at/below drop_lo
    _zero_fired: bool = False  # pctzero: fired this low episode (anti-double)
    _snap_armed: bool = False  # scene snap seen, awaiting settle to confirm
    _snap_age: int = 0  # ticks since the snap
    _prev_gray: Any | None = None  # last newest-frame gray, for frame diff
    _freeze_run: int = 0  # consecutive frozen ticks
    _auto_frozen: bool = False  # freeze detector currently suppressing death
    _last_diff: float = 0.0  # last full-frame diff (surfaced in GUI)
    _suspended: bool = False  # True while env/preview is paused: no death ever
    _resync: bool = False  # first step after unsuspend: snap baseline, pay 0

    def reset(self) -> None:
        """Start a new attempt: zero baseline, keep attempt count + pct latch."""
        self.prev_fill = 0.0
        self.best_fill = 0.0
        self.prev_src = "none"
        self.done = False
        self._pending = False
        self._pending_age = 0
        self._sink = 0.0
        self._drop_hi_seen = False
        self._drop_lo_run = 0
        self._zero_fired = False
        self._snap_armed = False
        self._snap_age = 0
        self.alive_ticks = 0
        self._last_edge = None
        self._last_closing = 0.0
        self._last_clear_tick = -10**9
        # NOTE: _pct_seen is NOT cleared: text readability persists across
        # attempts. Bar must never drive reward even on attempt 1 frame 1.

    def _commit_death(self, prog: float, via: str) -> tuple[float, dict]:
        """Single choke point: EVERY death increments attempts AND deaths."""
        r = self.cfg.death_penalty
        self.attempts += 1
        self.deaths += 1
        self.total += r
        info: dict = {"event": "death", "done": False, "via": via}
        self.reset()
        self.prev_fill = prog
        self.prev_src = "pct"
        return r, info

    def suspend(self) -> None:
        """Pause gate: while suspended step() tracks text but never fires death."""
        self._suspended = True
        self._pending = False
        self._pending_age = 0
        self._sink = 0.0
        self._snap_armed = False
        self._snap_age = 0

    def unsuspend(self) -> None:
        """Resume gate: re-sync baseline to current bar so unpause jump pays 0."""
        self._suspended = False
        self._pending = False
        self._pending_age = 0
        # prev_fill re-syncs on next step via _resync flag below.
        self._resync = True

    def bar_fill(self, obs_rgb: Any, _mask_out: Any | None = None) -> float:
        """Fill fraction 0..1 from the top-center bar crop of 320x180 RGB.

        Accepts (H,W,3) RGB or (N,H,W,3) stacked obs (uses newest).
        The fill is the CHARACTER's color (blue/orange/green, changes with
        equipped cube), so hue is NEVER gated: any saturated+bright pixel
        counts (S >= sat_min AND V >= val_min). White outline fails
        saturation, dark bg fails value, gray level geometry fails
        saturation. Fill = left-anchored saturated run / interior width
        via median-row consensus. Pass _mask_out to get {"mask",
        "crop_box"} for the preview debug strip.
        """
        import cv2
        import numpy as np

        img = np.asarray(obs_rgb)
        if img.ndim == 4:
            img = img[-1]  # newest frame of stack
        if img.ndim == 2:
            h, w = img.shape
            y0, y1 = int(h * self.cfg.bar_y0), int(h * self.cfg.bar_y1)
            x0, x1 = int(w * self.cfg.bar_x0), int(w * self.cfg.bar_x1)
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                return 0.0
            mask = (crop > self.cfg.val_min)
            if _mask_out is not None:
                _mask_out["mask"] = mask
                _mask_out["crop_box"] = (x0, y0, x1, y1)
            return self._left_run(mask)
        h, w = img.shape[:2]
        y0, y1 = int(h * self.cfg.bar_y0), int(h * self.cfg.bar_y1)
        x0, x1 = int(w * self.cfg.bar_x0), int(w * self.cfg.bar_x1)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        mask = ((hsv[:, :, 1] >= self.cfg.sat_min)
                & (hsv[:, :, 2] >= self.cfg.val_min))
        if _mask_out is not None:
            _mask_out["mask"] = mask
            _mask_out["crop_box"] = (x0, y0, x1, y1)
        return self._left_run(mask)

    @staticmethod
    def _left_run(mask: Any) -> float:
        """Longest left-anchored run / width, best row wins (thin lines)."""
        import numpy as np

        m = np.asarray(mask)
        if m.size == 0:
            return 0.0
        best = 0.0
        for rr in m:
            if not rr.any():
                continue
            bw2 = rr.shape[0]
            start = int(np.argmax(rr))
            if start > max(1, bw2 // 40):
                continue
            end = start
            while end < bw2 and rr[end]:
                end += 1
            run = (end - start) / max(1, bw2)
            if run > best:
                best = run
        return float(best)

    def _frame_diff(self, obs_rgb: Any) -> float:
        """Full-frame scene change 0..1 vs last tick (grayscale mean abs)."""
        import numpy as np

        img = np.asarray(obs_rgb)
        if img.ndim == 4:
            img = img[-1]
        if img.ndim == 3:
            g = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1]
                 + 0.114 * img[:, :, 2])
        else:
            g = img.astype(float)
        if self._prev_gray is None or self._prev_gray.shape != g.shape:
            self._prev_gray = g.copy()
            return 0.0
        d = float(abs(g - self._prev_gray).mean() / 255.0)
        self._prev_gray = g.copy()
        self._last_diff = d
        return d

    def _banner_present(self, obs_rgb: Any) -> tuple[bool, float]:
        """ATTEMPT-banner check: big white text mid-screen (geometry, no OCR).

        White fraction over the banner band; gameplay rarely puts >4% solid
        white there (spikes are thin outlines, bg is saturated color). Returns
        (present, frac) for GUI telemetry.
        """
        import numpy as np

        img = np.asarray(obs_rgb)
        if img.ndim == 4:
            img = img[-1]
        if img.ndim == 3:
            mx = img.max(axis=2).astype(np.float32)
            mn = img.min(axis=2).astype(np.float32)
        else:
            mx = img.astype(np.float32)
            mn = mx
        h, w = mx.shape[:2]
        y0, y1 = int(h * self.cfg.banner_y0), int(h * self.cfg.banner_y1)
        x0, x1 = int(w * self.cfg.banner_x0), int(w * self.cfg.banner_x1)
        crop_max = mx[y0:y1, x0:x1]
        crop_min = mn[y0:y1, x0:x1]
        if crop_max.size == 0:
            return False, 0.0
        white = (crop_max >= 200) & ((crop_max - crop_min) <= 40)
        frac = float(white.mean())
        return bool(frac >= self.cfg.banner_white_frac), frac

    def step(self, obs_rgb: Any,
             pct: float | tuple | None = None,
             held: bool = False,
             completed: bool = False,
             spike_edge: float | None = None,
             _mask_out: Any | None = None) -> tuple[float, bool, dict]:
        """One 60Hz tick: returns (reward, done, info).

        PHASE 2: pct is (value|None, fresh, age_ticks) from PctReader (or a
        bare float for back-compat, treated as fresh). fresh=True moves the
        baseline and pays/arms; stale holds WITHOUT dragging anything (the
        old pct_hold workaround is deleted: staleness is explicit now).
        SURVIVAL-FIRST still pays every alive tick; death via snap/banner/
        pct-reset through _commit_death only.
        """
        if isinstance(pct, tuple):
            pct_val, pct_fresh, pct_age = pct
        elif pct is None:
            pct_val, pct_fresh, pct_age = None, False, 999
        else:
            pct_val, pct_fresh, pct_age = float(pct), True, 0
        try:
            import math as _m
            if pct_val is not None and not _m.isfinite(float(pct_val)):
                pct_val, pct_fresh, pct_age = None, False, 999
        except Exception:
            pass
        self._last_hold = bool(held)
        diff = self._frame_diff(obs_rgb)
        # Freeze tracker: pause/menu frames are time-frozen (diff ~0).
        if diff <= self.cfg.freeze_diff:
            self._freeze_run += 1
        else:
            self._freeze_run = 0
            self._auto_frozen = False
        auto_frozen = self.cfg.use_freeze and self._freeze_run >= self.cfg.freeze_ticks
        if auto_frozen and not self._auto_frozen:
            self._auto_frozen = True
            self._pending = False
            self._pending_age = 0
            self._snap_armed = False
            self._snap_age = 0
        # Bar/text are TELEMETRY ONLY now: computed for GUI/debug, but the
        # survival reward below does not depend on them at all.
        fill = self.bar_fill(obs_rgb, _mask_out=_mask_out)
        banner, banner_frac = self._banner_present(obs_rgb)
        if pct_val is not None and (pct_fresh or float(pct_val) <= self.cfg.drop_lo):
            # Fresh text: prog vs stored prev_fill IS the signal.
            # Low text commits even when stale (retry screens hold 0% with
            # no new decodes; diag_death2 shows lows arriving before labels).
            p = min(1.0, max(0.0, float(pct_val)))
            prog, src = (p, "pct") if pct_fresh else (p, "pct_stale_lo")
            if not self._pct_seen:
                self._pct_seen = True
                # First-ever read seeds the baseline: pay 0, arm nothing.
                self.prev_fill = p
                self.best_fill = max(self.best_fill, p)
                self.prev_src = "pct"
                info: dict = {"fill": fill, "prog": prog, "src": "pct_init",
                              "prev": self.prev_fill, "attempts": self.attempts,
                              "diff": diff, "frozen": auto_frozen,
                              "pct_seen": True, "pct_fresh": True, "pct_age": 0}
                info.update({"event": None, "done": False, "delta": 0.0,
                             "init": True})
                return 0.0, False, info
        else:
            if self._pct_seen:
                # Stale: HOLD last accepted baseline, touch nothing.
                prog, src = self.prev_fill, "stale"
            elif diff >= self.cfg.snap_arm or self._snap_armed:
                prog, src = 0.0, "snap"
            else:
                prog, src = 0.0, "none"
        info = {"fill": fill, "prog": prog, "src": src,
                "prev": self.prev_fill, "attempts": self.attempts,
                "diff": diff, "frozen": auto_frozen,
                "pct_seen": True}
        # 100% only counts when it came from the 90s: a misread "100" (live: twice in a row,
        # in Stereo Madness's ship part at ~30%) used to end the run as "level complete".
        self._rt = getattr(self, "_rt", 0) + 1
        if src.startswith("pct") and 0.90 <= prog < self.cfg.completion_frac:
            self._last90 = self._rt
        if not completed and prog >= self.cfg.completion_frac and \
                self._rt - getattr(self, "_last90", -10**9) > 180:
            if not getattr(self, "_warned100", False) or self._rt - self._warned100 > 600:
                self._warned100 = self._rt
                print(f"[progress] ignored a 100% read (it came from {self.prev_fill * 100:.0f}%, not the 90s)")
            prog, src = self.prev_fill, "stale"     # treat it as no read
            info["prog"], info["src"] = prog, src
        if completed or prog >= self.cfg.completion_frac:
            r = self.cfg.completion_bonus
            self.done = True
            self.total += r
            info.update({"event": "complete", "done": True})
            return r, True, info
        if self._resync:
            # First tick after resume: baseline jumps (unpause/menuswap).
            # Snap to it, pay nothing, arm nothing.
            self._resync = False
            self.prev_fill = prog
            self.prev_src = src if src != "pct_hold" else self.prev_src
            if prog > self.best_fill:
                self.best_fill = prog
            info.update({"event": None, "done": False, "delta": 0.0,
                         "resync": True})
            return 0.0, False, info
        if self._suspended or auto_frozen:
            # Obscured while paused (manual P OR auto freeze-detect): follow
            # the signal so resume pays ~0, never arm, never confirm.
            self.prev_fill = prog
            if prog > self.best_fill:
                self.best_fill = prog
            self._pending = False
            self._pending_age = 0
            self._snap_armed = False
            self._snap_age = 0
            info.update({"event": None, "done": False, "delta": 0.0,
                         "suspended": True,
                         "auto_frozen": auto_frozen})
            return 0.0, False, info
        # Re-arm: ANY fresh positive progress is a new attempt underway, so
        # the next fall to 0 must count. Previously the latch only cleared
        # above drop_lo (2%): an agent dying at 1% every attempt (first
        # spike, see Stereo screenshot) never re-armed and every later
        # 1%->0% death was silently swallowed.
        if src == "pct" and prog > 1e-9:
            self._zero_fired = False
        # --- PHASE 3 + user rule: ANY LOW commits, no climb needed ---
        # Rule A (high->low edge): fresh prog at/below drop_lo with the
        # immediate pre-low baseline ABOVE drop_lo. Stale never fires.
        # Attempt-start guard: baseline 0.0 + prev_src init/none/snap is a
        # spawn, not a fall, so it never fires (covers replay-at-1% starts).
        # probe_best shows labelled deaths arriving already-low (collapse
        # happens ticks before the human-marked flash), so prev_fill gating
        # alone misses them; _pct_seen covers low-from-start-of-replay.
        # 1%-to-0% edge: best_fill stays 0.01 so the gate above misses it;
        # strict-decrease fallback below catches ANY fresh drop to ~0.
        # Anti-double-fire per low episode. via="pctzero".
        if (src == "pct" and prog <= self.cfg.drop_lo
                and self.prev_fill > self.cfg.drop_lo
                and not (self.prev_fill <= 1e-9
                     and self.prev_src in ("none", "pct_init", "snap"))):
            if not self._zero_fired:
                r, dinfo = self._commit_death(prog, via="pctzero")
                self._zero_fired = True
                self._drop_hi_seen = False
                self._drop_lo_run = 0
                info.update(dinfo)
                return r, False, info
        elif src == "pct" and prog > self.cfg.drop_lo:
            self._zero_fired = False
        # Rule B (low->zero edge): ZERO-OVERRIDE fresh 0.0 after ANY
        # positive baseline commits. Covers 1%->0%, 5%->0%, 8%->0% where
        # prev never exceeded drop_lo. Attempt-start guard: baseline 0.0 is
        # a spawn, not a fall, so it never fires.
        if (src == "pct" and prog <= 1e-9 and self.prev_fill > 1e-9
                and not self._zero_fired):
            r, dinfo = self._commit_death(prog, via="pctzero")
            self._zero_fired = True
            self._drop_hi_seen = False
            self._drop_lo_run = 0
            info.update(dinfo)
            return r, False, info
        # --- PHASE 3 (ablation winner): pct-drop dwell death ---
        # Fresh text above drop_hi then at/below drop_lo for drop_hold ticks.
        # Commits via _commit_death as via="pctdrop". Kept alongside pctzero
        # for noisy reads that hover at 1-2% instead of clean 0.
        if src == "pct":
            if prog >= self.cfg.drop_hi:
                self._drop_hi_seen = True
            if self._drop_hi_seen and prog <= self.cfg.drop_lo:
                self._drop_lo_run += 1
            elif prog > self.cfg.drop_lo:
                self._drop_lo_run = 0
                if prog >= self.cfg.drop_hi:
                    self._drop_hi_seen = True
            else:
                self._drop_lo_run = 0
            if self._drop_hi_seen and self._drop_lo_run >= self.cfg.drop_hold:
                r, dinfo = self._commit_death(prog, via="pctzero" if prog <= 0.005 else "pctdrop")
                self._drop_hi_seen = False
                self._drop_lo_run = 0
                info.update(dinfo)
                return r, False, info
        # Deleted per ablation: reset/sink/banner arming (all P/R-neutral or
        # worse; pctdrop_only == ALL). Only pctdrop commits now.
        # PROGRESS-DELTA shaping: forward pays, backward costs, flat pays a
        # small floor (keeps time alive on plateaus without rewarding camp).
        delta = prog - self.prev_fill
        if self.prev_src == "pct" and src == "pct":
            clipped = max(-self.cfg.max_delta_clip,
                          min(self.cfg.max_delta_clip, delta))
        else:
            clipped = delta
        if clipped > self.cfg.deadband:
            r = clipped * self.cfg.prog_delta_scale
        elif clipped < -self.cfg.deadband:
            r = clipped * self.cfg.down_scale  # negative = penalty
        else:
            r = self.cfg.survive_pay * self.cfg.dense_scale
        held_now = bool(getattr(self, "_last_hold", False))
        if not hasattr(self, "_last_closing"):
            self._last_closing = 0.0
        if not hasattr(self, "_last_clear_tick"):
            self._last_clear_tick = -10**9
        # SPIKE AVOIDANCE shaping (dense, per-tick):
        # (1) proximity rent: grounded near a spike bleeds, scaled by closeness;
        #     holding (airborne) waives the rent -- the jump itself is the escape.
        # (2) clearance bonus: nearest-spike edge grew while held = the jump
        #     carried the cube PAST the spike. Cooled down anti-farm.
        try:
            _edge = None if spike_edge is None else float(spike_edge)
            if _edge is not None and not (_edge == _edge):  # NaN guard
                _edge = None
        except Exception:
            _edge = None
        if _edge is not None and _edge < self.cfg.prox_radius and not held_now:
            r += -self.cfg.prox_scale * max(0.0, 1.0 - _edge / max(1e-6, self.cfg.prox_radius))
        _prev_edge = getattr(self, "_last_edge", None)
        _closing_neg = False
        try:
            _closing_neg = bool(getattr(self, "_last_closing", 0.0) < -1e-9)
        except Exception:
            pass
        # Clearance requires the edge to actually GROW (jump carried the
        # spike behind us). _closing_neg was inverted: it fired while the
        # spike was still APPROACHING (closing < 0), farming +3.0 for
        # holding toward an unjumped spike and inflating attempt rewards.
        if (_edge is not None and held_now
                and _prev_edge is not None and _edge > _prev_edge + 1e-9
                and self.alive_ticks - getattr(self, "_last_clear_tick", -10**9) >= self.cfg.clear_cool):
            r += self.cfg.clear_pay
            self._last_clear_tick = self.alive_ticks
            try:
                _first = (_prev_edge is not None and _prev_edge <= self.cfg.first_spike_edge
                          and self.best_fill <= 0.03)
            except Exception:
                _first = False
            if _first:
                r += self.cfg.first_spike_pay
        try:
            _pe = getattr(self, "_last_edge", None)
            if _pe is not None and _edge is not None:
                self._last_closing = float(_edge - _pe)
            elif _edge is None:
                self._last_closing = 0.0
            self._last_edge = float(_edge) if _edge is not None else None
        except Exception:
            self._last_edge = None
            self._last_closing = 0.0
        if held_now:
            r += self.cfg.hold_pay  # action itself pays: breaks 0-hold trap
        # PEAK curiosity: each new best-% tick pays, so 2% -> 3% is worth
        # the same risk as 0% -> 1%. Without it the 2% camp is optimal.
        if src == "pct" and prog > self.best_fill + 1e-9:
            r += self.cfg.peak_pay * (prog - self.best_fill)
        if src == "pct" and prog > self.best_fill + 1e-9:
            r += self.cfg.jump_pay
            if held_now:
                r += self.cfg.air_pay  # held THROUGH the climb = timed jump
            self.best_fill = prog
        elif prog > self.best_fill:
            self.best_fill = prog
        self.prev_fill = prog
        self.prev_src = "pct" if src == "pct" else self.prev_src
        self.alive_ticks += 1
        self.total += r
        info.update({"event": None, "done": False, "delta": delta,
                     "alive": self.alive_ticks,
                     "banner_frac": round(banner_frac, 4)})
        return r, False, info
