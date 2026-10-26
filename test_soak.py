"""30s consistency soak: exact tick accounting + hard 60Hz lock."""
import time

from capture import GDCapture

cap = GDCapture().start()
try:
    for _ in range(60):
        cap.read()
    t0 = time.perf_counter()
    n = 1800
    for _ in range(n):
        cap.read()
    dt = time.perf_counter() - t0
    h = cap.health
    print(f"30s soak: {n} reads in {dt:.2f}s -> {n / dt:.3f} Hz")
    print(f"tick={h['tick']} expected~{h['expected_ticks']} drift={h['tick_drift']:+d} "
          f"tick_fps={h['tick_fps']:.3f} overruns={h['overruns']} drops={h['drops']} "
          f"restarts={h['restarts']} p95={h['lat_p95_ms']:.1f}ms max={h['lat_max_ms']:.1f}ms")
    assert h["frames"] + h["drops"] == h["tick"], "count invariant broken"
    assert abs(n / dt - 60.0) < 1.0, "loop rate off"
    assert abs(h["tick_drift"]) <= 5, "drift too large"
    print("SOAK PASS")
finally:
    cap.stop()
