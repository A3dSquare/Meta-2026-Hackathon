"""YOLO backend (opt-in): detector upgrade when heuristic misses art.

Train: collect recs/*.boxes.json via label_boxes.py (~500 boxes, 3+ levels),
then:  python detect_yolo.py --train
which writes ./yolo_gd.pt (YOLOv8n, 7 classes). Use with
--backend yolo in train_boxes.py / preview_boxes.py.

CPU fallback: if ultralytics/torch is missing, YoloDetector raises a clear
hint and callers should use --backend heuristic.
"""
from __future__ import annotations

import glob
import json
import os

from detect import CLASSES, Entities, Entity

MODEL_PATH = "./yolo_gd.pt"


class YoloDetector:
    def __init__(self, path: str = MODEL_PATH) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("pip install ultralytics torch  (or use --backend heuristic)") from e
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing: run python detect_yolo.py --train first")
        from ultralytics import YOLO as _Y
        self._m = _Y(path)

    def __call__(self, rgb320):
        import numpy as np
        img = np.asarray(rgb320)
        if img.ndim == 4:
            img = img[-1]
        h, w = img.shape[:2]
        res = self._m.predict(img, verbose=False, conf=0.35)[0]
        items = []
        try:
            for b in res.boxes:
                x0, y0, x1, y1 = (float(v) for v in b.xyxy[0].tolist())
                items.append(Entity(int(b.cls[0]), ((x0 + x1) / 2) / w,
                                    ((y0 + y1) / 2) / h, (x1 - x0) / w,
                                    (y1 - y0) / h, float(b.conf[0])))
        except Exception:
            pass
        return Entities(items)


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="train YOLO entity detector")
    p.add_argument("--train", action="store_true")
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args()
    if not args.train:
        p.print_help()
        return 2
    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics torch")
        return 1
    # Export boxes/*.json to YOLO dataset dirs
    import cv2
    ds = "./yolo_ds"
    for split in ("train", "val"):
        os.makedirs(os.path.join(ds, "images", split), exist_ok=True)
        os.makedirs(os.path.join(ds, "labels", split), exist_ok=True)
    files = sorted(glob.glob("recs/*.boxes.json"))
    print(f"box files: {files}")
    n = 0
    for i, bf in enumerate(files):
        rec = bf.replace(".boxes.json", ".mp4")
        if not os.path.exists(rec):
            continue
        boxes = json.load(open(bf))
        by_fi: dict[int, list] = {}
        for b in boxes:
            by_fi.setdefault(int(b["fi"]), []).append(b)
        cap = cv2.VideoCapture(rec)
        for fi, bl in by_fi.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok:
                continue
            split = "val" if i % 5 == 4 else "train"
            H, W = fr.shape[:2]
            name = f"{os.path.basename(rec).replace('.mp4','')}_{fi}"
            cv2.imwrite(os.path.join(ds, "images", split, name + ".jpg"), fr)
            with open(os.path.join(ds, "labels", split, name + ".txt"), "w") as f:
                for b in bl:
                    cx = (b["x0"] + b["x1"]) / 2
                    cy = (b["y0"] + b["y1"]) / 2
                    f.write(f"{b['cls']} {cx:.4f} {cy:.4f} {b['x1']-b['x0']:.4f} {b['y1']-b['y0']:.4f}\n")
            n += 1
        cap.release()
    with open(os.path.join(ds, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(ds)}\ntrain: images/train\nval: images/val\n"
                f"names: {CLASSES}\nnc: {len(CLASSES)}\n")
    print(f"dataset: {n} frames")
    m = YOLO("yolov8n.pt")
    m.train(data=os.path.join(ds, "data.yaml"), epochs=args.epochs, imgsz=320)
    print("train a YOLO model, then copy runs/detect/train/weights/best.pt -> ./yolo_gd.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
