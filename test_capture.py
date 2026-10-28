"""Smoke test for 1.1 capture pipeline: shape + hard 60Hz lock + exact counts."""
import sys
import time

import numpy as np

from capture import CaptureConfig, GDCapture


def main() -> int:
    cfg = CaptureConfig(width=320, height=180, n_stack=4, target_fps=60)
    cap = GDCapture(cfg)
    print(f"expected obs_shape={cap.obs_shape} dtype=uint8")

    # Offline shape test (no screen needed): push synthetic RGB frames.
    fake = (np.random.rand(180, 320, 3) * 255).astype(np.uint8)
    for _ in range(4):
        cap._push(fake)
    obs = cap._stacked_obs()
    assert obs.shape == (4, 180, 320, 3), f"bad shape {obs.shape}"
    assert obs.dtype == np.uint8, f"bad dtype {obs.dtype}"
    print(f"offline stack OK: shape={obs.shape} dtype={obs.dtype}")

    # Offline conversion test: BGR in -> RGB out (red <-> blue swapped).
    bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255  # pure blue in BGR
    small = cap._to_rgb_small(bgr)
    assert small.shape == (180, 320, 3), f"bad small {small.shape}"
    assert small[:, :, 0].mean() < 8 and small[:, :, 2].mean() > 247, "BGR->RGB swap wrong"
    print("offline BGR->RGB OK")
    cap.reset()

    # Live test: real dxcam capture with EXACT tick accounting.
    try:
        cap.start()
    except Exception as e:  # noqa: BLE001
        print(f"LIVE CAPTURE SKIPPED: {e}")
        return 0
    try:
        for _ in range(30):  # warmup duplicator (not counted)
            cap.read()
        t0_tick = cap.tick
        n = 600  # 10s at 60Hz
        t0 = time.perf_counter()
        for _ in range(n):
            obs = cap.read()
        dt = time.perf_counter() - t0
        ticks_done = cap.tick - t0_tick
        h = cap.health

        print(f"live obs shape={obs.shape} dtype={obs.dtype}")
        assert obs.shape == (4, 180, 320, 3), f"bad live shape {obs.shape}"
        # EXACTNESS: one read() == one tick, ticks == frames + drops.
        assert ticks_done == n, f"tick miscount: {ticks_done} != {n} reads"
        assert h["frames"] + h["drops"] == h["tick"], (
            f"count mismatch: frames {h['frames']} + drops {h['drops']} != tick {h['tick']}"
        )
        fps = n / dt
        print(f"loop {n} reads in {dt:.2f}s -> {fps:.2f} Hz "
              f"(tick_fps={h['tick_fps']:.2f} drift={h['tick_drift']:+d})")
        print(f"frames={h['frames']} drops={h['drops']} "
              f"drop_rate={h['drop_rate'] * 100:.2f}% restarts={h['restarts']} "
              f"p95={h['lat_p95_ms']:.1f}ms max={h['lat_max_ms']:.1f}ms "
              f"bgr_direct={h['bgr_direct']}")
        ok = True
        if abs(fps - 60.0) > 1.0:
            print(f"FAIL: loop rate {fps:.2f}Hz not locked to 60Hz (+/-1)")
            ok = False
        if abs(h["tick_fps"] - 60.0) > 1.0:
            print(f"FAIL: tick_fps {h['tick_fps']:.2f}Hz not 60Hz (+/-1)")
            ok = False
        if abs(h["tick_drift"]) > 5:
            print(f"FAIL: tick drift {h['tick_drift']:+d} too large (>5 ticks)")
            ok = False
        if h["drop_rate"] > 0.05:
            print(f"FAIL: drop rate {h['drop_rate'] * 100:.2f}% > 5%")
            ok = False
        if not ok:
            return 1
        print("PASS: shape (4,180,320,3) RGB uint8, hard-locked 60Hz, counts exact")
        return 0
    finally:
        cap.stop()


if __name__ == "__main__":
    sys.exit(main())
