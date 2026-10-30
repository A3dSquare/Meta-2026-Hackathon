"""Raw YOLO viewer. No policy, physics, hazard memory, or keyboard automation.

python preview_yolo.py
python preview_yolo.py --video recs/level.mp4 --conf .25
"""
import argparse
import time


def annotate(frame, detections, names):
    import cv2
    out = frame.copy()
    h, w = out.shape[:2]
    for c, x0, y0, x1, y1, confidence in detections:
        color = ((73 * c + 50) % 200 + 55, (131 * c + 20) % 200 + 55, (47 * c) % 200 + 55)
        a, b = (int(x0*w), int(y0*h)), (int(x1*w), int(y1*h))
        cv2.rectangle(out, a, b, color, 2)
        cv2.putText(out, f'{names[c]} {confidence:.2f}', (a[0], max(15, a[1]-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='yolo_gd.pt')
    parser.add_argument('--conf', type=float, default=.35)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default=None)
    parser.add_argument('--video', help='Optional recording instead of live capture')
    parser.add_argument('--monitor', type=int, default=0)
    parser.add_argument('--region', type=int, nargs=4, metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'))
    args = parser.parse_args()
    import cv2
    from detect_yolo import YoloDetector
    from gd_objects import NAMES
    detector = YoloDetector(args.model, conf=args.conf, imgsz=args.imgsz, device=args.device)
    source = None
    title = 'YOLO only - Q/Esc quit, Space pause'
    try:
        if args.video:
            source = cv2.VideoCapture(args.video)
            if not source.isOpened():
                raise RuntimeError(f'Cannot open {args.video}')
        else:
            from capture import GDCapture, CaptureConfig
            source = GDCapture(CaptureConfig(output_idx=args.monitor,
                               region=tuple(args.region) if args.region else None)).start()
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        paused = False
        while True:
            if not paused:
                start = time.perf_counter()
                if args.video:
                    ok, frame = source.read()
                    if not ok:
                        break
                else:
                    source.read()
                    frame = source.last_hires
                    if frame is None:
                        continue
                t = time.perf_counter()
                detections = detector.detect(frame)
                infer_ms = (time.perf_counter()-t)*1000
                display = annotate(frame, detections, NAMES)
                cv2.putText(display, f'YOLO {infer_ms:.1f} ms | loop {(time.perf_counter()-start)*1000:.1f} ms | {len(detections)} boxes',
                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
                cv2.imshow(title, display)
            key = cv2.waitKey(15 if paused else 1) & 255
            if key in (27, ord('q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 32:
                paused = not paused
    finally:
        if source is not None:
            source.release() if args.video else source.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
