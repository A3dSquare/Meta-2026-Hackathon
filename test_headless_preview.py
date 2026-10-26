"""Headless check: replicate preview.py build_panel at all zooms, save PNG."""
import cv2

from capture import GDCapture
from preview import build_panel

cap = GDCapture().start()
try:
    for _ in range(10):
        cap.read()
    obs = cap.read()
    assert obs.shape == (4, 180, 320), obs.shape
    for zoom in (1, 2, 3, 4):
        panel = build_panel(obs, zoom, f"tick {cap.tick} zoom {zoom}x test header")
        eh, ew = 30 + 180 * zoom, 320 * zoom + 10 + (320 * zoom) // 2
        assert panel.shape == (eh, ew, 3), f"zoom {zoom}: {panel.shape} != {(eh, ew, 3)}"
        print(f"zoom {zoom}x OK: panel {panel.shape}")
    cv2.imwrite("preview_check.png", build_panel(obs, 2, "tick test header"))
    h = cap.health
    print(f"counts: ticks={h['tick']} frames={h['frames']} drops={h['drops']} "
          f"exact={'OK' if h['frames'] + h['drops'] == h['tick'] else 'MISMATCH'}")
finally:
    cap.stop()
print("DONE")
