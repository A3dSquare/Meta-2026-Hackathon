"""Live GUI preview for 1.1: shows exactly what the policy sees.

Layout (recomputed every frame from the current zoom):
  Left  (big)    : latest 320x180 grayscale frame, NEAREST-upscaled.
  Right (4 small): the 4-frame stack, oldest -> newest.
  Header         : tick, exact tick_fps, drift vs wall clock, latency,
                   frames/drops, restarts, shape. Frame count is exact:
                   every read() = 1 tick, ticks == frames + drops.

The capture loop runs in THIS thread at a hard 60Hz. Rendering is
decoupled (<= ~30Hz) so imshow can never throttle the decision loop.

Scalable GUI:
  - Window is WINDOW_NORMAL | WINDOW_KEEPRATIO: drag any edge to resize.
  - Zoom keys +/- (or scroll): 1x..6x upscale of the 320x180 view.
  - F toggles fullscreen. --scale sets the starting zoom.

Controls: Q/ESC quit. R restart capture. +/- zoom. F fullscreen.
  Region: --region l,t,r,b (default: full screen).
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from capture import CaptureConfig, GDCapture

WIN = "GD 1.1 capture preview (policy view)  [+/- zoom, F fullscreen, R restart, Q quit]"
HDR_H = 30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GD capture preview (policy view)")
    p.add_argument("--region", default=None,
                   help="crop region l,t,r,b e.g. 0,0,1920,1080 (default: full screen)")
    p.add_argument("--output-idx", type=int, default=0)
    p.add_argument("--scale", type=int, default=2, choices=(1, 2, 3, 4, 5, 6),
                   help="starting zoom factor (1-6, change live with +/-)")
    return p.parse_args()


def build_panel(obs: np.ndarray, zoom: int, header: str) -> np.ndarray:
    """Compose the BGR panel at the current zoom. Nearest = true pixels."""
    H, W = obs.shape[1], obs.shape[2]  # 180, 320
    big_w, big_h = W * zoom, H * zoom
    # Right column: 4 tiles, each half the big width, stacked to big height.
    tile_w = big_w // 2
    tile_h = big_h // 4

    newest = obs[-1]
    big = cv2.resize(newest, (big_w, big_h), interpolation=cv2.INTER_NEAREST)
    big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)

    rows = []
    for i, f in enumerate(obs):
        tag = "new" if i == len(obs) - 1 else f"t-{len(obs) - 1 - i}"
        t = cv2.resize(f, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
        t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        fs = 0.5 if tile_h < 80 else 0.7
        cv2.putText(t, tag, (6, int(12 + 10 * fs)), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (0, 255, 0), 1, cv2.LINE_AA)
        rows.append(t)
    col = np.vstack(rows)

    pw = big_w + 10 + tile_w
    ph = HDR_H + big_h
    panel = np.zeros((ph, pw, 3), dtype=np.uint8)
    panel[HDR_H:HDR_H + big_h, 0:big_w] = big
    panel[HDR_H:HDR_H + big_h, big_w + 10:big_w + 10 + tile_w] = col
    cv2.putText(panel, header, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def main() -> int:
    args = parse_args()
    region = None
    if args.region:
        try:
            l, t, r, b = (int(v) for v in args.region.split(","))
            region = (l, t, r, b)
        except ValueError:
            print("bad --region, want l,t,r,b")
            return 2
    zoom = int(args.scale)
    cfg = CaptureConfig(region=region, output_idx=args.output_idx)
    cap = GDCapture(cfg)
    try:
        cap.start()
    except Exception as e:  # noqa: BLE001
        print(f"capture failed to start: {e}")
        return 1

    # Resizable + aspect-kept window: user can drag to ANY size.
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(WIN, 320 * zoom + 160 * zoom + 10, 180 * zoom + HDR_H)
    for _ in range(10):  # warm up stack so all 4 tiles are live
        cap.read()

    last_gui = 0.0
    fullscreen = False
    latest_obs: np.ndarray = cap.read()

    print("preview running: drag edges to resize | +/- zoom | F fullscreen | R restart | Q quit")
    try:
        while True:
            latest_obs = cap.read()  # hard-locked to 60Hz inside

            # Render at most every 33ms so GUI never slows capture.
            now = time.perf_counter()
            if now - last_gui >= 1 / 30:
                last_gui = now
                h = cap.health
                hdr = (f"tick {h['tick']}  {h['tick_fps']:5.1f}Hz "
                       f"drift {h['tick_drift']:+d}  p95 {h['lat_p95_ms']:4.1f}ms  "
                       f"fr {h['frames']} dr {h['drops']} rst {h['restarts']}  "
                       f"zoom {zoom}x  {tuple(h['obs_shape'])}")
                cv2.imshow(WIN, build_panel(latest_obs, zoom, hdr))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("r"), ord("R")):
                cap.restart("user keypress")
                print("capture restarted")
            elif key in (ord("+"), ord("=")):
                zoom = min(6, zoom + 1)
            elif key in (ord("-"), ord("_")):
                zoom = max(1, zoom - 1)
            elif key in (ord("f"), ord("F")):
                fullscreen = not fullscreen
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key == 82:  # Up arrow: zoom in (opencv special key, rarely fires with waitKey(1))
                zoom = min(6, zoom + 1)
            elif key == 84:  # Down arrow: zoom out
                zoom = max(1, zoom - 1)
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
        cv2.destroyAllWindows()
        h = cap.health
        n = h["tick"]
        ok = (h["frames"] + h["drops"] == n)
        print(f"soak: ticks={n} frames={h['frames']} drops={h['drops']} "
              f"exact={'OK' if ok else 'MISMATCH!'} restarts={h['restarts']} "
              f"tick_fps={h['tick_fps']:.2f} drift={h['tick_drift']:+d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
