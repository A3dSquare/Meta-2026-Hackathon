"""YOLO detector: learns object art from your labels (record.py -> label_boxes.py).

    pip install ultralytics                 # once (uses your torch; a GPU helps a lot)
    python detect_yolo.py --train           # all recs/*.boxes.json -> ./yolo_gd.pt
    python detect_yolo.py --train --epochs 80
    python detect_yolo.py --check recs/deadlocked.mp4    # writes yolo_check_*.png previews

Used by `--backend yolo` in preview_boxes.py / train_boxes.py as a HELPER to
the built-in detector (detect.HeuristicDetector(extra=...)): the built-in one
keeps doing spikes/blocks/icon/speed exactly as before, and YOLO only ADDS
objects it doesn't already have (saws, monsters, slime, custom art). So on
levels the built-in detector already handles, the boxes -- and the brain's
behaviour -- stay the same.

Classes (same numbers as the policy uses):
  0 hazard  1 orb_y  2 orb_b  3 orb_p  4 pad  5 portal  6 solid
"""
from __future__ import annotations

import glob
import json
import os
import shutil

import numpy as np

MODEL_PATH = "./yolo_gd.pt"
NAMES = ["hazard", "orb_y", "orb_b", "orb_p", "pad", "portal", "solid"]


class YoloDetector:
    def __init__(self, path: str = MODEL_PATH, conf: float = 0.35, imgsz: int = 640, device=None) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("pip install ultralytics   (or use --backend heuristic)") from e
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing: label some frames, then run python detect_yolo.py --train")
        self._m = YOLO(path)
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def detect(self, bgr: np.ndarray) -> list:
        """BGR frame (any size) -> [(cls, x0, y0, x1, y1, conf)] normalized 0..1."""
        h, w = bgr.shape[:2]
        res = self._m.predict(np.ascontiguousarray(bgr[:, :, :3]), imgsz=self.imgsz, conf=self.conf,
                              verbose=False, device=self.device)[0]
        out = []
        try:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            cf = res.boxes.conf.cpu().numpy()
            for (x0, y0, x1, y1), c, p in zip(xyxy, cls, cf):
                if 0 <= c <= 6:
                    out.append((int(c), x0 / w, y0 / h, x1 / w, y1 / h, float(p)))
        except Exception:
            pass
        return out

    def __call__(self, rgb320, hires_bgr=None):
        """YOLO-only Entities (no icon / speed) -- prefer the hybrid (make_detector('yolo'))."""
        from detect import Entities, Entity
        img = hires_bgr if hires_bgr is not None else np.ascontiguousarray(np.asarray(rgb320)[:, :, ::-1])
        return Entities([Entity(c, (a + b) / 2, (y0 + y1) / 2, b - a, y1 - y0, p)
                         for c, a, y0, b, y1, p in self.detect(img)])


def _load_labels(path: str):
    d = json.load(open(path))
    if isinstance(d, list):
        return d, sorted({int(b["fi"]) for b in d})
    return d.get("boxes", []), sorted(int(f) for f in d.get("done", []))


def build_dataset(ds: str = "./yolo_ds") -> tuple[str, int, int]:
    import cv2
    if os.path.exists(ds):
        shutil.rmtree(ds)
    for split in ("train", "val"):
        os.makedirs(os.path.join(ds, "images", split), exist_ok=True)
        os.makedirs(os.path.join(ds, "labels", split), exist_ok=True)
    files = sorted(glob.glob("recs/*.boxes.json"))
    if not files:
        raise SystemExit("no recs/*.boxes.json yet: python record.py --level NAME, then python label_boxes.py --rec recs/NAME.mp4")
    n_img = n_box = 0
    for bf in files:
        rec = bf.replace(".boxes.json", ".mp4")
        if not os.path.exists(rec):
            print(f"skip {bf}: {rec} missing")
            continue
        boxes, done = _load_labels(bf)
        by_fi: dict = {}
        for b in boxes:
            by_fi.setdefault(int(b["fi"]), []).append(b)
        cap = cv2.VideoCapture(rec)
        stem = os.path.basename(rec)[:-4]
        for k, fi in enumerate(done):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok:
                continue
            split = "val" if k % 5 == 4 else "train"
            name = f"{stem}_{fi:05d}"
            cv2.imwrite(os.path.join(ds, "images", split, name + ".jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(os.path.join(ds, "labels", split, name + ".txt"), "w") as f:
                for b in by_fi.get(fi, []):      # done frame with no boxes = "nothing here" example
                    cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
                    f.write(f"{int(b['cls'])} {cx:.5f} {cy:.5f} {b['x1'] - b['x0']:.5f} {b['y1'] - b['y0']:.5f}\n")
                    n_box += 1
            n_img += 1
        cap.release()
        print(f"  {bf}: {len(done)} done frames")
    yaml = os.path.join(ds, "data.yaml")
    with open(yaml, "w") as f:
        f.write(f"path: {os.path.abspath(ds)}\ntrain: images/train\nval: images/val\n"
                f"nc: {len(NAMES)}\nnames: {NAMES}\n")
    return yaml, n_img, n_box


def train(epochs: int = 60, imgsz: int = 640, base: str = "yolov8n.pt", device=None) -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics")
        return 1
    yaml, n_img, n_box = build_dataset()
    print(f"dataset: {n_img} frames, {n_box} boxes")
    if n_img < 10:
        print("fewer than 10 labelled frames -- label more first (aim for 40-60 per level)")
        return 1
    m = YOLO(base)
    r = m.train(data=yaml, epochs=epochs, imgsz=imgsz, batch=-1 if device != "cpu" else 8,
                device=device, project="runs_yolo", name="gd", exist_ok=True, verbose=False,
                mosaic=0.5, fliplr=0.0, hsv_s=0.5, hsv_v=0.4)  # GD never mirrors; hue kept (orb colours matter)
    best = os.path.join(str(getattr(r, "save_dir", "runs_yolo/gd")), "weights", "best.pt")
    if not os.path.exists(best):
        best = os.path.join("runs_yolo", "gd", "weights", "best.pt")
    if os.path.exists(MODEL_PATH):
        shutil.copy(MODEL_PATH, MODEL_PATH.replace(".pt", "_prev.pt"))
    shutil.copy(best, MODEL_PATH)
    print(f"saved {MODEL_PATH}  -> use it with:  python preview_boxes.py --backend yolo ...")
    return 0


def check(rec: str, n: int = 6) -> int:
    import cv2
    from detect import HeuristicDetector, draw
    det = HeuristicDetector(temporal=False, extra=YoloDetector())
    cap = cv2.VideoCapture(rec)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    for k, fi in enumerate(np.linspace(0, max(0, total - 1), n).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(cv2.resize(fr, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        hi = cv2.resize(fr, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"yolo_check_{k}.png", draw(hi, det(rgb, hi)))
    print(f"wrote yolo_check_0..{n - 1}.png")
    return 0


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="train / check the YOLO object detector")
    p.add_argument("--train", action="store_true")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="cpu, 0 (first GPU), ... default: auto")
    p.add_argument("--check", default=None, help="mp4 to preview detections on")
    a = p.parse_args()
    if a.train:
        return train(a.epochs, a.imgsz, device=a.device)
    if a.check:
        return check(a.check)
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
