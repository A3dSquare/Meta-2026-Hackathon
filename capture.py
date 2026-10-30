"""Second Attempt 1.1 — fast screen capture pipeline.

Pipeline: dxcam -> crop (via DXGI region) -> BGR -> RGB -> downscale to
320x180 -> 4-frame stack. Full RGB so orb/portal colors survive (yellow
bounce vs blue flip vs pink vs portals are hue-coded, near-identical
in luma at 320x180).

Optimizations for stable 60 FPS:
  - Capture only `region` so fewer pixels cross the bus.
  - Keep native BGR from DXGI, swap to RGB + downscale in place.
  - `video_mode=True` + `target_fps=60` so DXGI paces acquisition.
  - Small `max_buffer_len` so we never build latency backlog.
  - Preallocated numpy buffers; cv2.resize writes into `dst` in place.
  - Ring-buffer frame stack: no np.roll / no per-step allocation for
    the history itself (one small copy only when emitting obs).
  - Precise FPS cap (sleep + spin) so decisions lock to <= 60 Hz.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

log = logging.getLogger("gd.capture")

import cv2
import numpy as np

try:
    import dxcam
except ImportError as e:  # pragma: no cover - import-time hint
    raise ImportError(
        "dxcam is required on Windows: pip install dxcam comtypes"
    ) from e

Region = Tuple[int, int, int, int]  # (left, top, right, bottom)


def fast_downscale(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Same result as cv2.INTER_AREA, ~3-4x faster on big frames.

    An exact 2x LINEAR shrink averages 2x2 pixel blocks, which is what AREA
    does -- so halve with LINEAR while the frame is >= 2x the target, then
    finish with a small AREA resize (e.g. 2560x1440 -> 1280x720 -> 640x360).
    """
    while img.shape[1] // 2 >= 0.95 * w and img.shape[0] // 2 >= 0.95 * h:
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
    if img.shape[1] != w or img.shape[0] != h:
        near = abs(img.shape[1] - w) <= 0.06 * w and abs(img.shape[0] - h) <= 0.06 * h
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR if near else cv2.INTER_AREA)
    return img


@dataclass
class CaptureConfig:
    width: int = 320
    height: int = 180
    n_stack: int = 4
    target_fps: int = 60  # hard lock: read() emits exactly this many ticks/sec
    region: Optional[Region] = None
    output_idx: int = 0  # primary monitor
    max_buffer_len: int = 8
    prefetch: bool = False  # experimental: Python-heavy search can starve the extra reader
    prefetch_wait_s: float = 0.  # polling never invokes coarse Windows condition timeouts
    max_frame_age_s: float = .1  # an old buffered frame is a drop, not a fresh observation
    # INTER_AREA is best for downscale quality; INTER_LINEAR is ~2x faster.
    # At 320x180 both are <<1ms, so keep quality by default.
    interpolation: int = cv2.INTER_AREA
    # --- consistency knobs ---
    stale_timeout_s: float = 1.0  # no fresh frame for this long -> restart
    max_consec_none: int = 120  # ~2s of None polls at 60Hz -> restart
    restart_cooldown_s: float = 2.0  # min gap between auto-restarts
    restart_backoff_s: float = 0.5  # sleep before recreating duplicator
    latency_window: int = 240  # rolling window for p95/max stats


class GDCapture:
    """Optimized dxcam capture with RGB + resize + frame stack.

    Also caches the last native full-res BGR frame (`last_fullres`) for
    the 1Hz x% digit-template reader: 320x180 is too coarse for OCR, so
    percent text must be matched on full-res crops, never on the policy obs.
    """

    def __init__(self, config: CaptureConfig | None = None, **kwargs) -> None:
        self.cfg: CaptureConfig = config or CaptureConfig(**kwargs)
        self._period_s: float = 1.0 / float(self.cfg.target_fps)
        self._period_ns: int = int(round(1e9 / float(self.cfg.target_fps)))
        self._camera = None
        self._mailbox = None
        self._clock = None
        self._prefetch_aligned = False
        self.profile = None
        self.last_frame_ts: float | None = None   # when the newest frame was shown (seconds)
        self._ts_mode: str | None = None          # "dxcam" (exact) / "none" (arrival times)
        self._bgr_direct: bool = True  # True once we confirm native BGR path
        self._running: bool = False
        self._lock = threading.Lock()

        H, W, N = self.cfg.height, self.cfg.width, self.cfg.n_stack
        # Ring buffer history: (N, H, W, 3) RGB uint8, written in place.
        self._stack: np.ndarray = np.zeros((N, H, W, 3), dtype=np.uint8)
        self._ptr: int = 0  # next slot to overwrite
        self._filled: int = 0
        # Scratch destination for cv2.resize (avoids one alloc per frame).
        self._small: np.ndarray = np.zeros((H, W, 3), dtype=np.uint8)
        self._fullres: np.ndarray | None = None  # last native BGR frame (no copy beyond this)
        self._hires: np.ndarray | None = None    # last frame at 640x360 BGR (for the detector)
        self._last_tick_ns: int = 0
        # --- exact tick/frame accounting ---
        self._ticks: int = 0  # every read() call = exactly one 60Hz tick
        self._frames: int = 0  # ticks that carried a fresh screen frame
        self._drops: int = 0  # ticks that reused the last obs (None/corrupt)
        self._overruns: int = 0  # full-period overruns (grid slots skipped)
        self._consec_none: int = 0
        self._restarts: int = 0
        self._last_restart_ts: float = 0.0
        self._last_good_ts: float = 0.0
        self._lat_ms: deque = deque(maxlen=self.cfg.latency_window)
        self._t_start: float = 0.0
        self._tick_ts: deque = deque(maxlen=self.cfg.target_fps * 5)  # wall times for exact fps

    # ------------------------------------------------------------------ #
    @property
    def obs_shape(self) -> Tuple[int, int, int, int]:
        """Stacked observation shape: (n_stack, height, width, 3) RGB."""
        return (self.cfg.n_stack, self.cfg.height, self.cfg.width, 3)

    @property
    def obs_dtype(self):
        return np.uint8

    def _create_and_start(self) -> None:
        self._prefetch_aligned = False
        create_kwargs: dict = dict(
            output_idx=self.cfg.output_idx,
            output_color="BGR",  # native DXGI order; swap to RGB in place
            max_buffer_len=self.cfg.max_buffer_len,
        )
        # dxcam 0.3.x accepts region in create(); older builds only in start().
        # Try create-with-region first, fall back gracefully.
        if self.cfg.region is not None:
            try:
                self._camera = dxcam.create(region=self.cfg.region, **create_kwargs)
            except TypeError:
                self._camera = dxcam.create(**create_kwargs)
        else:
            self._camera = dxcam.create(**create_kwargs)

        start_kwargs: dict = dict(target_fps=self.cfg.target_fps, video_mode=True)
        if self.cfg.region is not None:
            try:
                self._camera.start(region=self.cfg.region, **start_kwargs)
            except TypeError:
                self._camera.start(**start_kwargs)
        else:
            self._camera.start(**start_kwargs)
        if self.cfg.prefetch:
            from frame_mailbox import FrameMailbox
            camera = self._camera
            # get_latest_frame returns an owned copy by default. Preserve
            # that ownership and its paired timestamp across this handoff.
            self._mailbox = FrameMailbox(lambda: self._latest_with_time(camera)).start()

    def _stop_camera(self):
        """Stop the producer before replacing its camera; never reuse its queue."""
        mailbox, self._mailbox = self._mailbox, None
        if mailbox is not None:
            mailbox.request_stop()
        try:
            if self._camera is not None:
                self._camera.stop()  # unblocks get_latest_frame in the reader
        finally:
            self._camera = None
            if mailbox is not None:
                mailbox.join()

    def start(self) -> "GDCapture":
        """Create the DXGI duplicator and begin background capture."""
        with self._lock:
            if self._running:
                return self
            self._create_and_start()
            self._running = True
            now = time.perf_counter()
            now_ns = time.perf_counter_ns()
            # Anchor the 60Hz grid to start: tick k lands at t_start + k*period.
            self._t_start = now
            self._t_start_ns = now_ns
            self._next_tick_ns = now_ns
            self._last_tick_ns = now_ns
            self._last_good_ts = now
            self._consec_none = 0
            self._ticks = 0
            self._frames = 0
            self._drops = 0
            self._overruns = 0
            self._tick_ts.clear()
            return self

    def _restart_locked(self, reason: str) -> bool:
        """Recreate the DXGI duplicator. Caller must hold lock. Returns ok."""
        now = time.perf_counter()
        if now - self._last_restart_ts < self.cfg.restart_cooldown_s:
            return False
        self._last_restart_ts = now
        log.warning("capture restart: %s", reason)
        try:
            if self._camera is not None:
                try:
                    self._stop_camera()
                except Exception:  # noqa: BLE001 - best effort teardown
                    pass
            time.sleep(self.cfg.restart_backoff_s)
            self._create_and_start()
            self._consec_none = 0
            self._last_good_ts = time.perf_counter()
            self._last_tick_ns = time.perf_counter_ns()  # avoid throttle debt
            self._restarts += 1
            return True
        except Exception:  # noqa: BLE001 - keep serving last-known obs
            log.exception("capture restart failed")
            return False

    def restart(self, reason: str = "manual") -> bool:
        with self._lock:
            return self._restart_locked(reason)

    def stop(self) -> None:
        with self._lock:
            try:
                self._stop_camera()
            finally:
                self._running = False
                if self._clock is not None:
                    self._clock.close()
                    self._clock = None

    @property
    def tick(self) -> int:
        """Monotonic 60Hz tick counter: increments exactly once per read()."""
        return self._ticks

    @property
    def last_hires(self) -> np.ndarray | None:
        """The last frame already shrunk to 640x360 BGR (None if not made)."""
        with self._lock:
            return self._hires

    @property
    def last_fullres(self) -> np.ndarray | None:
        """Last native full-res BGR frame (no downscale). Same-tick read-only.

        Used by the 1Hz x% digit-template reader: percent glyphs are
        ~6-10px at 320x180 (unreadable) but crisp at native res. Returns
        None before the first fresh frame. Do NOT hold across ticks: dxcam
        reuses the buffer, so copy the crop if you need it later.
        """
        with self._lock:
            return self._fullres

    @property
    def expected_ticks(self) -> int:
        """Ticks that SHOULD have elapsed since start at exactly target_fps."""
        if not self._t_start:
            return 0
        return int((time.perf_counter() - self._t_start) * self.cfg.target_fps)

    @property
    def health(self) -> dict:
        """Snapshot of consistency stats (lock-free read of counters).

        ticks == frames + drops ALWAYS (exact accounting, asserted in tests).
        tick_fps is computed from real wall-clock span of the tick window,
        so it cannot lie the way a frame-count/elapsed estimate can.
        """
        lat = np.asarray(self._lat_ms, dtype=np.float64) if self._lat_ms else np.zeros(1)
        elapsed = max(1e-6, time.perf_counter() - self._t_start) if self._t_start else 0.0
        ticks = self._ticks
        if len(self._tick_ts) >= 2:
            span = self._tick_ts[-1] - self._tick_ts[0]
            tick_fps = (len(self._tick_ts) - 1) / span if span > 0 else 0.0
        else:
            tick_fps = 0.0
        return {
            "running": self._running,
            "tick": ticks,
            "expected_ticks": self.expected_ticks,
            "tick_drift": ticks - self.expected_ticks,
            "frames": self._frames,
            "drops": self._drops,
            "overruns": self._overruns,
            "grid_slots": ticks + self._overruns,
            "drop_rate": self._drops / max(1, ticks),
            "restarts": self._restarts,
            "fps_avg": self._frames / elapsed if elapsed else 0.0,
            "tick_fps": tick_fps,  # exact measured tick rate from timestamps
            "target_fps": self.cfg.target_fps,
            "lat_mean_ms": float(lat.mean()),
            "lat_p95_ms": float(np.percentile(lat, 95)),
            "lat_max_ms": float(lat.max()),
            "stale_s": (time.perf_counter() - self._last_good_ts) if self._last_good_ts else 0.0,
            "bgr_direct": self._bgr_direct,
            "prefetch": self.cfg.prefetch,
            "reader_alive": self._mailbox.alive if self._mailbox is not None else False,
            "high_resolution_pacing": self._clock.high_resolution if self._clock is not None else False,
            "gray_direct": self._bgr_direct,  # back-compat alias for old tests
            "obs_shape": self.obs_shape,
        }

    def __enter__(self) -> "GDCapture":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def set_region(self, region: Optional[Region]) -> None:
        """Change the crop region (restarts the duplicator)."""
        was_running = self._running
        if was_running:
            self.stop()
        self.cfg.region = region
        if was_running:
            self.start()

    # ------------------------------------------------------------------ #
    def _throttle_to_fps_cap(self) -> int:
        """Wait until the next tick on the fixed 60Hz grid. Returns tick index.

        The grid is anchored at start: tick k lands at t_start + k*period.
        Oversleep (OS jitter) never shifts the grid — the next tick is still
        on schedule — and overrun ticks (>=1 period late) are reported as
        missed so read() can count them honestly in drops.
        """
        period_ns = self._period_ns
        target_ns = self._next_tick_ns
        now = time.perf_counter_ns()
        remaining_ns = target_ns - now
        missed = 0
        if remaining_ns > 0:
            if self._clock is None:
                from capture_clock import CaptureClock
                self._clock = CaptureClock()
            # Python 3.10's coarse Windows sleep can overshoot an entire
            # frame. Use a private high-resolution timer where available.
            margin = 250_000 if self._clock.high_resolution else 1_000_000
            if remaining_ns > margin+250_000:
                self._clock.sleep((remaining_ns-margin)/1e9)
            while time.perf_counter_ns() < target_ns:
                pass
        else:
            # Late past this tick's deadline: count whole periods skipped.
            missed = int((now - target_ns) // period_ns)
        self._next_tick_ns = target_ns + (missed + 1) * period_ns
        self._last_tick_ns = time.perf_counter_ns()
        return missed

    def _to_rgb_small(self, frame: np.ndarray) -> np.ndarray:
        """Crop is already done by DXGI region; swap to RGB + downscale."""
        if frame is None:
            raise RuntimeError("dxcam returned None (minimized/protected content?)")
        # Native path: dxcam with output_color="BGR" gives (H, W, 3) BGR.
        if frame.ndim == 3 and frame.shape[2] == 3:
            bgr = frame
        elif frame.ndim == 3 and frame.shape[2] == 4:
            # BGRA (alpha unused): drop channel in place via scratch.
            if not hasattr(self, "_scratch") or self._scratch is None:
                self._scratch = np.empty_like(frame)
            np.copyto(self._scratch, frame)
            bgr = self._scratch[:, :, :3]
        elif frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            # Backend gave GRAY despite BGR request: replicate to RGB.
            self._bgr_direct = False
            g = frame[:, :, 0] if frame.ndim == 3 else frame
            if g.shape[0] == self.cfg.height and g.shape[1] == self.cfg.width:
                self._small[:, :, 0] = g
                self._small[:, :, 1] = g
                self._small[:, :, 2] = g
            else:
                cv2.resize(
                    g,
                    (self.cfg.width, self.cfg.height),
                    dst=self._small[:, :, 0],
                    interpolation=self.cfg.interpolation,
                )
                self._small[:, :, 1] = self._small[:, :, 0]
                self._small[:, :, 2] = self._small[:, :, 0]
            return self._small
        else:  # pragma: no cover - defensive
            raise ValueError(f"unexpected frame ndim={frame.ndim} shape={frame.shape}")

        if bgr.shape[0] == self.cfg.height and bgr.shape[1] == self.cfg.width:
            # Exact size: BGR->RGB channel swap in place, no resize needed.
            self._small[:, :, 0] = bgr[:, :, 2]
            self._small[:, :, 1] = bgr[:, :, 1]
            self._small[:, :, 2] = bgr[:, :, 0]
        elif (self.cfg.interpolation == cv2.INTER_AREA and bgr.shape[0] >= 360 and bgr.shape[1] >= 640
              and self.cfg.width <= 640 and self.cfg.height <= 360):
            # Shrink ONCE: a 640x360 copy for the detector, and the small frame from it
            # (was: two full-size AREA resizes per frame, the slowest step on 1440p screens)
            hi = fast_downscale(bgr, 640, 360)
            self._hires_next = hi
            cv2.resize(hi, (self.cfg.width, self.cfg.height), dst=self._small, interpolation=cv2.INTER_AREA)
            tmp = self._small[:, :, 0].copy()
            self._small[:, :, 0] = self._small[:, :, 2]
            self._small[:, :, 2] = tmp
        else:
            # Resize all 3 channels at once, then swap R<->B in place.
            cv2.resize(
                bgr,
                (self.cfg.width, self.cfg.height),
                dst=self._small,
                interpolation=self.cfg.interpolation,
            )
            # In-place R<->B swap via temp view (no alloc).
            tmp = self._small[:, :, 0].copy()
            self._small[:, :, 0] = self._small[:, :, 2]
            self._small[:, :, 2] = tmp
        return self._small

    # Back-compat alias (old name); health key renamed below.
    def _to_grayscale_small(self, frame: np.ndarray) -> np.ndarray:
        return self._to_rgb_small(frame)

    def _push(self, small: np.ndarray) -> None:
        self._stack[self._ptr] = small
        self._ptr = (self._ptr + 1) % self.cfg.n_stack
        self._filled = min(self._filled + 1, self.cfg.n_stack)

    def _stacked_obs(self) -> np.ndarray:
        """Return contiguous (N,H,W) obs, oldest -> newest. Copies once."""
        N = self.cfg.n_stack
        if self._filled < N:
            # Not warmed up yet: replicate the newest frame into empty slots
            # so the policy never sees stale zeros mid-episode start.
            newest = self._stack[(self._ptr - 1) % N]
            out = np.empty_like(self._stack)
            out[:] = newest
            filled = [self._stack[(self._ptr - self._filled + i) % N] for i in range(self._filled)]
            for i, f in enumerate(filled):
                out[N - self._filled + i] = f
            return out
        idx = [(self._ptr + i) % N for i in range(N)]  # oldest first
        return self._stack[idx]  # advanced indexing already returns an owned contiguous copy

    # ------------------------------------------------------------------ #
    def read(self) -> np.ndarray:
        """Grab latest frame, preprocess, push to stack, return stacked obs.

        HARD 60Hz LOCK: blocks until the next tick on the fixed grid, so a
        tight loop yields exactly `target_fps` decisions/second.
        Invariants (asserted in tests):
          - every read() call increments `tick` by exactly 1.
          - `tick == frames + drops` at all times.
          - fresh screen frame -> `frames` + 1, else `drops` + 1.
          - a full-period overrun additionally counts missed slots in
            `overruns` (and in `drops`), never as phantom ticks.

        Never raises on transient DXGI hiccups: repeated None polls or a
        stale duplicator trigger an auto-restart, and the last-known obs
        is served meanwhile so training loops never crash.
        """
        if not self._running or self._camera is None:
            raise RuntimeError("GDCapture.read() called before start()")
        t_in = time.perf_counter()
        missed = self._throttle_to_fps_cap()
        t_fetch = time.perf_counter()
        self.wait_s = getattr(self, "wait_s", 0.0) + (t_fetch - t_in)
        if self.profile is not None:
            self.profile.add('capture_pacing',t_fetch-t_in)
        try:
            frame, ts = (self._mailbox.take(self.cfg.prefetch_wait_s) if self._mailbox is not None
                         else self._latest_with_time())
            if self.profile is not None:
                self.profile.add('capture_fetch',time.perf_counter()-t_fetch)
            # Timestamp domains differ on old backends; only use age when
            # the timestamp demonstrably belongs to perf_counter's clock.
            age = time.perf_counter()-ts if ts is not None else None
            if age is not None and 0 <= age < 10.:
                if self.profile is not None:
                    self.profile.add('capture_frame_age',age)
                if age > self.cfg.max_frame_age_s:
                    frame = None
            if frame is not None and self._mailbox is not None and not self._prefetch_aligned:
                # Phase the consumer just after frame publication, instead
                # of sampling just BEFORE each fresh frame for the whole run.
                arrival = self._mailbox.last_arrival
                if arrival is not None:
                    self._next_tick_ns = int((arrival+self._period_s+.001)*1e9)
                    self._prefetch_aligned = True
        except Exception:  # noqa: BLE001 - DXGI lost device etc.
            with self._lock:
                self._restart_locked("get_latest_frame raised")
            self._note_tick(missed, fresh=False)
            self._lat_ms.append((time.perf_counter() - t_in) * 1000.0)
            return self._stacked_obs()
        if frame is None:
            # No new frame since last poll (common in video_mode); reuse last
            # stack state but still return a valid obs.
            with self._lock:
                self._consec_none += 1
                stale_for = time.perf_counter() - self._last_good_ts
                need_restart = (
                    self._consec_none >= self.cfg.max_consec_none
                    or stale_for >= self.cfg.stale_timeout_s
                )
                if need_restart:
                    self._restart_locked(
                        f"stale feed none={self._consec_none} stale={stale_for:.1f}s"
                    )
            self._note_tick(missed, fresh=False)
            self._lat_ms.append((time.perf_counter() - t_in) * 1000.0)
            return self._stacked_obs()
        try:
            t_resize = time.perf_counter()
            self._hires_next = None
            small = self._to_rgb_small(frame)
            if self.profile is not None:
                self.profile.add('capture_resize',time.perf_counter()-t_resize)
        except Exception:  # noqa: BLE001 - corrupt frame; keep last obs
            log.warning("dropping corrupt frame shape=%s", getattr(frame, "shape", "?"))
            self._note_tick(missed, fresh=False)
            return self._stacked_obs()
        with self._lock:
            self._push(small)
            self.last_frame_ts = ts
            self._fullres = frame  # dxcam owns buffer; read-only use, same-tick only
            self._hires = getattr(self, "_hires_next", None)
            self._consec_none = 0
            self._last_good_ts = time.perf_counter()
            obs = self._stacked_obs()
        self._note_tick(missed, fresh=True)
        self._lat_ms.append((time.perf_counter() - t_in) * 1000.0)
        return obs

    def _latest_with_time(self, camera=None):
        """(frame, seconds) -- when the game put that frame on screen if dxcam can tell
        (0.3+: get_latest_frame(with_timestamp=True)), else when we received it. Motion is
        measured over the real time between two frames: the newest frame is up to a frame
        older or newer than our decision tick, which made measured speeds wobble ~20%."""
        camera = self._camera if camera is None else camera
        if getattr(self, "_ts_mode", None) != "none":
            try:
                got = camera.get_latest_frame(with_timestamp=True)
                if isinstance(got, tuple) and len(got) == 2:
                    frame, ts = got
                    if ts is not None and float(ts) > 0:
                        if getattr(self, "_ts_mode", None) is None:
                            self._ts_mode = "dxcam"
                            log.info("frame times: from dxcam (exact)")
                        return frame, float(ts)
                    return frame, time.perf_counter()
                return got, time.perf_counter()
            except TypeError:
                self._ts_mode = "none"       # older dxcam: no timestamps
                log.info("frame times: dxcam has none, using arrival times")
        return camera.get_latest_frame(), time.perf_counter()

    def _note_tick(self, missed: int, *, fresh: bool) -> None:
        """Exact accounting, one read() = exactly one tick.

        - `ticks` counts read() calls only, so reads == ticks always holds.
        - `frames` counts ticks with a fresh screen frame, `drops` counts
          ticks that reused the last obs. Hence ticks == frames + drops.
        - `overruns` counts grid slots skipped by a full-period overrun.
          They carried no decision and are NOT ticks; total grid slots
          elapsed = ticks + overruns.
        """
        with self._lock:
            self._ticks += 1
            self._overruns += missed
            if fresh:
                self._frames += 1
            else:
                self._drops += 1
            self._tick_ts.append(time.perf_counter())

    def reset(self, fill_value: int = 0) -> np.ndarray:
        """Clear history; returns a zero obs with the canonical shape."""
        self._stack.fill(fill_value)
        self._ptr = 0
        self._filled = 0
        return self._stacked_obs()

    # ------------------------------------------------------------------ #
    def benchmark(self, seconds: float = 5.0, warmup: int = 30) -> dict:
        """Run the live pipeline and report achieved FPS + timing breakdown."""
        if not self._running:
            raise RuntimeError("call start() before benchmark()")
        for _ in range(warmup):
            self.read()
        n = max(1, int(seconds * self.cfg.target_fps))
        lat: list[float] = []
        t0 = time.perf_counter()
        for _ in range(n):
            s = time.perf_counter()
            self.read()
            lat.append((time.perf_counter() - s) * 1000.0)
        dt = time.perf_counter() - t0
        arr = np.asarray(lat, dtype=np.float64)
        return {
            "frames": n,
            "seconds": dt,
            "fps": n / dt,
            "target_fps": self.cfg.target_fps,
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(arr.max()),
            "obs_shape": self.obs_shape,
            "bgr_direct": self._bgr_direct,
        }
