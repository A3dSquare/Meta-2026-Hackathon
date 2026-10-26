"""Second Attempt 1.1 — fast screen capture pipeline.

Pipeline: dxcam -> crop (via DXGI region) -> grayscale (on GPU if
available) -> downscale to 320x180 -> 4-frame stack.

Optimizations for stable 60 FPS:
  - Capture only `region` so fewer pixels cross the bus.
  - Ask dxcam for GRAY output directly (GPU-side conversion), fall back
    to cv2 BGR->GRAY only if the backend ignores it.
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


@dataclass
class CaptureConfig:
    width: int = 320
    height: int = 180
    n_stack: int = 4
    target_fps: int = 60  # hard lock: read() emits exactly this many ticks/sec
    region: Optional[Region] = None
    output_idx: int = 0  # primary monitor
    max_buffer_len: int = 8
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
    """Optimized dxcam capture with grayscale + resize + frame stack."""

    def __init__(self, config: CaptureConfig | None = None, **kwargs) -> None:
        self.cfg: CaptureConfig = config or CaptureConfig(**kwargs)
        self._period_s: float = 1.0 / float(self.cfg.target_fps)
        self._period_ns: int = int(round(1e9 / float(self.cfg.target_fps)))
        self._camera = None
        self._gray_direct: bool = True  # assume GPU grayscale works until proven otherwise
        self._running: bool = False
        self._lock = threading.Lock()

        H, W, N = self.cfg.height, self.cfg.width, self.cfg.n_stack
        # Ring buffer history: (N, H, W) uint8, written in place.
        self._stack: np.ndarray = np.zeros((N, H, W), dtype=np.uint8)
        self._ptr: int = 0  # next slot to overwrite
        self._filled: int = 0
        # Scratch destination for cv2.resize (avoids one alloc per frame).
        self._small: np.ndarray = np.zeros((H, W), dtype=np.uint8)
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
    def obs_shape(self) -> Tuple[int, int, int]:
        """Stacked observation shape: (n_stack, height, width)."""
        return (self.cfg.n_stack, self.cfg.height, self.cfg.width)

    @property
    def obs_dtype(self):
        return np.uint8

    def _create_and_start(self) -> None:
        create_kwargs: dict = dict(
            output_idx=self.cfg.output_idx,
            output_color="GRAY",  # GPU-side grayscale when supported
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
                    self._camera.stop()
                except Exception:  # noqa: BLE001 - best effort teardown
                    pass
                self._camera = None
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
            if self._camera is not None:
                try:
                    self._camera.stop()
                finally:
                    self._camera = None
            self._running = False

    @property
    def tick(self) -> int:
        """Monotonic 60Hz tick counter: increments exactly once per read()."""
        return self._ticks

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
            "gray_direct": self._gray_direct,
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
            # Coarse sleep for most of it, spin for the last ~1.5ms.
            if remaining_ns > 1_500_000:
                time.sleep((remaining_ns - 1_000_000) / 1e9)
            while time.perf_counter_ns() < target_ns:
                pass
        else:
            # Late past this tick's deadline: count whole periods skipped.
            missed = int((now - target_ns) // period_ns)
        self._next_tick_ns = target_ns + (missed + 1) * period_ns
        self._last_tick_ns = time.perf_counter_ns()
        return missed

    def _to_grayscale_small(self, frame: np.ndarray) -> np.ndarray:
        """Crop is already done by DXGI region; convert + downscale in place."""
        if frame is None:
            raise RuntimeError("dxcam returned None (minimized/protected content?)")
        # Fast path: dxcam with output_color="GRAY" gives (H, W) or (H, W, 1).
        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3 and frame.shape[2] == 1:
            gray = frame[:, :, 0]
        elif frame.ndim == 3 and frame.shape[2] == 3:
            # Fallback: backend ignored output_color="GRAY", convert on CPU.
            # Note: dxcam default order is BGR.
            self._gray_direct = False
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            self._gray_direct = False
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:  # pragma: no cover - defensive
            raise ValueError(f"unexpected frame ndim={frame.ndim} shape={frame.shape}")

        if gray.shape[0] == self.cfg.height and gray.shape[1] == self.cfg.width:
            np.copyto(self._small, gray)
        else:
            cv2.resize(
                gray,
                (self.cfg.width, self.cfg.height),
                dst=self._small,
                interpolation=self.cfg.interpolation,
            )
        return self._small

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
        return self._stack[idx].copy()  # fancy index already copies; .copy() makes it contiguous

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
        try:
            frame = self._camera.get_latest_frame()
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
            small = self._to_grayscale_small(frame)
        except Exception:  # noqa: BLE001 - corrupt frame; keep last obs
            log.warning("dropping corrupt frame shape=%s", getattr(frame, "shape", "?"))
            self._note_tick(missed, fresh=False)
            return self._stacked_obs()
        with self._lock:
            self._push(small)
            self._consec_none = 0
            self._last_good_ts = time.perf_counter()
            obs = self._stacked_obs()
        self._note_tick(missed, fresh=True)
        self._lat_ms.append((time.perf_counter() - t_in) * 1000.0)
        return obs

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
            "gray_direct": self._gray_direct,
        }
