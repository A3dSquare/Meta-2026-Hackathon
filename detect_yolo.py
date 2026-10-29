"""YOLO detector: learns object art from your labels (record.py -> label_boxes.py).

    pip install ultralytics                 # once (uses your torch; a GPU helps a lot)
    python detect_yolo.py --train           # all recs/*.boxes.json -> ./yolo_gd.pt
    python detect_yolo.py --train --epochs 80
    python detect_yolo.py --check recs/deadlocked.mp4    # writes yolo_check_*.png previews
    python preview_boxes.py --backend yolo-only ...      # YOLO does ALL the detecting

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
# All the classes (gd_objects.py): 0-6 the objects the policy has always used, 7-14 the
# ICON by game mode, then every orb / pad / portal / speed portal type. Each maps to a
# base class for the brain; the exact type drives the blue line's physics.
from gd_objects import BASE, GROUP, ICON_CLASSES, NAMES


class YoloDetector:
    def __init__(self, path: str = MODEL_PATH, conf: float = 0.35, imgsz: int = 640, device=None) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("pip install ultralytics   (or use --backend heuristic)") from e
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing: label some frames, then run python detect_yolo.py --train")
        # a TensorRT engine made from this model (python detect_yolo.py --export) is ~2-3x faster
        eng = os.path.splitext(path)[0] + ".engine"
        self.engine = os.path.exists(eng) and os.path.getmtime(eng) >= os.path.getmtime(path)
        if os.path.exists(eng) and not self.engine:
            print(f"[yolo] {eng} is older than {path} (retrained since): using the .pt -- re-run --export")
        self._m = YOLO(eng, task="detect") if self.engine else YOLO(path)
        try:
            import torch
            dev = "GPU (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else "CPU -- slow!"
        except Exception:
            dev = "CPU -- slow!"
        print(f"[yolo] {'TensorRT engine' if self.engine else os.path.basename(path)} on the {dev}")
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def detect(self, bgr: np.ndarray) -> list:
        """BGR frame (any size) -> [(cls, x0, y0, x1, y1, conf)] normalized 0..1."""
        h, w = bgr.shape[:2]
        if not hasattr(self, "_half"):
            try:   # half precision on an NVIDIA GPU: ~2x faster, same boxes for a model this size
                import torch
                self._half = (bool(torch.cuda.is_available()) and str(self.device) != "cpu"
                              and not self.engine)       # (a TensorRT engine has its precision built in)
            except Exception:
                self._half = False
            try:   # newer ultralytics warns about 'half' on EVERY call: keep the console readable
                import logging
                from ultralytics.utils import LOGGER
                LOGGER.setLevel(logging.ERROR)
            except Exception:
                pass
        res = self._m.predict(np.ascontiguousarray(bgr[:, :, :3]), imgsz=self.imgsz, conf=self.conf,
                              verbose=False, device=self.device, half=self._half)[0]
        out = []
        try:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            cf = res.boxes.conf.cpu().numpy()
            for (x0, y0, x1, y1), c, p in zip(xyxy, cls, cf):
                if 0 <= c < len(NAMES):
                    out.append((int(c), x0 / w, y0 / h, x1 / w, y1 / h, float(p)))
        except Exception:
            pass
        return out

    def __call__(self, rgb320, hires_bgr=None):
        """YOLO-only Entities (no icon / speed) -- prefer the hybrid (make_detector('yolo'))."""
        from detect import Entities, Entity
        img = hires_bgr if hires_bgr is not None else np.ascontiguousarray(np.asarray(rgb320)[:, :, ::-1])
        return Entities([Entity(BASE[c], (a + b) / 2, (y0 + y1) / 2, b - a, y1 - y0, p)
                         for c, a, y0, b, y1, p in self.detect(img) if c not in ICON_CLASSES])


def _tip(gray, x0, y0, x1, y1) -> int:
    """Which way a hazard points (0 up, 1 down, 2 left, 3 right, -1 can't tell): a spike is
    wide at its base and narrow at its tip. The blue line puts the hitbox at the base."""
    import cv2
    H, W = gray.shape
    a, b = max(0, int(x0 * W)), min(W, int(x1 * W) + 1)
    c, d = max(0, int(y0 * H)), min(H, int(y1 * H) + 1)
    if b - a < 6 or d - c < 6:
        return -1
    e = cv2.Canny(gray[c:d, a:b], 50, 150) > 0
    h, w = e.shape
    def span(m):
        idx = np.nonzero(m.any(axis=0))[0]
        return (idx[-1] - idx[0] + 1) if len(idx) else 0
    q = max(1, h // 4)
    top, bot = span(e[:q]), span(e[-q:])
    qx = max(1, w // 4)
    lef, rig = span(e[:, :qx].T), span(e[:, -qx:].T)
    if bot > 1.6 * max(1, top) and bot >= 0.5 * w:
        return 0
    if top > 1.6 * max(1, bot) and top >= 0.5 * w:
        return 1
    if rig > 1.6 * max(1, lef) and rig >= 0.5 * h:
        return 2
    if lef > 1.6 * max(1, rig) and lef >= 0.5 * h:
        return 3
    return -1


class YoloOnlyDetector:
    """Everything from YOLO: hazards, blocks, orbs, pads, portals AND the icon with its mode.
    (--backend yolo-only.) Same Entities as the built-in detector, so the brain, the blue
    line and the shield work unchanged. Kept from the built-in detector: hazard tracking
    (smooths boxes, measures the scroll = speed) and, only for frames where YOLO doesn't
    see the icon, the built-in icon finder."""

    def __init__(self, path: str = MODEL_PATH, icon_hue: tuple = (55, 95), conf: float = 0.35, device=None) -> None:
        from detect import HeuristicDetector, _HazardTracker
        self.yolo = YoloDetector(path, conf=conf, device=device)
        names = list(getattr(self.yolo._m, "names", {}).values()) if isinstance(getattr(self.yolo._m, "names", None), dict) \
            else list(getattr(self.yolo._m, "names", []) or [])
        if names and names != NAMES[:len(names)]:
            raise SystemExit(f"{path} was trained with other classes ({names}): retrain with python detect_yolo.py --train")
        self.has_icon = len(names) > 7
        if not self.has_icon:
            print("[yolo-only] this model has no icon classes (old labels): using the built-in icon finder. "
                  "Label the icon (keys 7 8 9 U V R P in label_boxes.py) and retrain to get the mode from YOLO.")
        self._fallback = HeuristicDetector(icon_hue=icon_hue)
        self._tracker = _HazardTracker()
        self.n_icon_yolo = self.n_icon_fallback = 0

    def reset(self) -> None:
        self._tracker.reset()

    def __call__(self, rgb320, hires_bgr=None):
        import cv2
        from detect import CLS_ID, Entities, Entity, _PL_LAST
        img = np.asarray(hires_bgr if hires_bgr is not None else np.asarray(rgb320)[-1] if np.asarray(rgb320).ndim == 4
                         else rgb320)
        if hires_bgr is None:
            img = np.ascontiguousarray(img[:, :, ::-1])
        img = np.ascontiguousarray(img[:, :, :3])
        H, W = img.shape[:2]
        dets = self.yolo.detect(img)
        ents = []
        icons = [d for d in dets if d[0] in ICON_CLASSES]
        pl = None
        if icons:
            c, x0, y0, x1, y1, p = max(icons, key=lambda d: d[5])
            pl = Entity(CLS_ID["player"], (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, p)
            pl.mode = ICON_CLASSES[c]
            _PL_LAST[0] = (pl.x, pl.y)          # keep the built-in finder in step
            self.n_icon_yolo += 1
        else:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            pl = self._fallback._player(hsv, W, H)
            if pl is not None:
                pl.mode = None
                self.n_icon_fallback += 1
        if pl is not None:
            ents.append(pl)
        # every other class -> its base class (what the brain sees) + its exact type (kind)
        objs = [(BASE[d[0]],) + tuple(d[1:]) + (NAMES[d[0]] if d[0] > 6 else None,)
                for d in dets if d[0] not in ICON_CLASSES]
        if pl is not None:                       # nothing on the icon itself is a hazard / block
            px0, py0, px1, py1 = pl.x - pl.w / 2, pl.y - pl.h / 2, pl.x + pl.w / 2, pl.y + pl.h / 2
            objs = [d for d in objs if not (d[0] in (0, 6) and d[1] < px1 and d[3] > px0 and d[2] < py1 and d[4] > py0
                                            and (d[1] + d[3]) / 2 > px0 and (d[1] + d[3]) / 2 < px1)]
        gray = None
        track = []
        for c, x0, y0, x1, y1, p, _k in objs:
            if c == 6:
                track.append((c, x0, y0, x1, y1, p, -1))
            elif c == 0:
                if gray is None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                track.append((c, x0, y0, x1, y1, p, _tip(gray, x0, y0, x1, y1)))
        ents.extend(self._tracker.update(track))
        hsv = None
        for c, x0, y0, x1, y1, p, kind in objs:
            if c in (0, 6):
                continue
            if hsv is None:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            reg = hsv[int(y0 * H):int(y1 * H) + 1, int(x0 * W):int(x1 * W) + 1].reshape(-1, 3)
            reg = reg[reg[:, 1] > 80]
            hue = float(np.median(reg[:, 0])) / 180.0 if len(reg) else -1.0
            e = Entity(c, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, p, hue)
            e.kind = kind                        # exact type (red orb, 2x speed portal ...)
            ents.append(e)
        return Entities(ents, self._tracker.last_scroll)


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
    any_icon = any(int(b["cls"]) in ICON_CLASSES for bf in files for b in _load_labels(bf)[0])
    for bf in files:
        rec = bf.replace(".boxes.json", ".mp4")
        if not os.path.exists(rec):
            print(f"skip {bf}: {rec} missing")
            continue
        boxes, done = _load_labels(bf)
        by_fi: dict = {}
        for b in boxes:
            by_fi.setdefault(int(b["fi"]), []).append(b)
        # frames labelled before the icon classes existed have no icon box: training on them
        # would teach YOLO that the icon is background -- leave them out (add the icon with
        # I in label_boxes.py to use them again)
        no_icon = [fi for fi in done if not any(int(b["cls"]) in ICON_CLASSES for b in by_fi.get(fi, []))]
        if any_icon and no_icon:
            print(f"  {bf}: {len(no_icon)} of {len(done)} done frames have no icon box -- left out "
                  f"(open them in label_boxes.py and press I on each to add it)")
            done = [fi for fi in done if fi not in set(no_icon)]
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
        names = NAMES
        if not any_icon:
            print("  (no icon labels anywhere: the icon / mode will come from the built-in finder)")
        cnt = {}
        for bf in files:
            for b in _load_labels(bf)[0]:
                cnt[int(b["cls"])] = cnt.get(int(b["cls"]), 0) + 1
        few = [f"{NAMES[c]} ({n})" for c, n in sorted(cnt.items()) if n < 15]
        if few:
            print(f"  only a few examples of: {', '.join(few)} -- aim for 15+ boxes of each type you use")
        f.write(f"path: {os.path.abspath(ds)}\ntrain: images/train\nval: images/val\n"
                f"nc: {len(names)}\nnames: {names}\n")
    return yaml, n_img, n_box


def train(epochs: int = 60, imgsz: int = 640, base: str = "yolov8n.pt", device=None) -> int:
    """base: yolov8n.pt (fastest) or yolov8s.pt (more accurate, still ~3 ms on a 5070 Ti)."""
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
    print(f"saved {MODEL_PATH}  -> check it:  python detect_yolo.py --check recs/<level>.mp4\n"
          f"            then use it:  python preview_boxes.py --backend yolo-only --model checkpoints/preview_boxes.pt")
    return 0


def export(imgsz: int = 640) -> int:
    """yolo_gd.pt -> yolo_gd.engine (TensorRT, half precision): the fastest way to run it on
    an NVIDIA GPU. Needs: pip install tensorrt. Used automatically while newer than the .pt."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics")
        return 1
    if not os.path.exists(MODEL_PATH):
        print(f"{MODEL_PATH} missing: train first (python detect_yolo.py --train)")
        return 1
    out = YOLO(MODEL_PATH).export(format="engine", half=True, imgsz=imgsz, device=0)
    print(f"exported {out} -- it's used automatically from now on (until you retrain)")
    return 0


def check(rec: str, n: int = 6, hybrid: bool = False) -> int:
    """Preview images of what the detector sees (YOLO-only by default, the icon labelled
    with the mode YOLO thinks it's in)."""
    import cv2
    from detect import HeuristicDetector, draw
    det = HeuristicDetector(temporal=False, extra=YoloDetector()) if hybrid else YoloOnlyDetector()
    cap = cv2.VideoCapture(rec)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    for k, fi in enumerate(np.linspace(0, max(0, total - 1), n).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(cv2.resize(fr, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        hi = cv2.resize(fr, (640, 360), interpolation=cv2.INTER_AREA)
        ents = det(rgb, hi)
        img = draw(hi, ents)
        pl = ents.player()
        if pl is not None:
            lab = getattr(pl, "mode", None) or "icon (built-in finder)"
            cv2.putText(img, lab, (int((pl.x - pl.w / 2) * img.shape[1]), max(12, int((pl.y - pl.h / 2) * img.shape[0]) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.imwrite(f"yolo_check_{k}.png", img)
    print(f"wrote yolo_check_0..{n - 1}.png")
    return 0


def speed(n: int = 60) -> int:
    """How long YOLO takes per frame on this PC (and the code around it)."""
    import cv2
    import time
    vids = sorted(glob.glob("recs/*.mp4"))
    if not vids:
        print("no recs/*.mp4 to time on")
        return 1
    cap = cv2.VideoCapture(vids[0])
    frames = []
    while len(frames) < n:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(fr, (640, 360), interpolation=cv2.INTER_AREA))
    d = YoloOnlyDetector()
    for f in frames[:5]:
        d(cv2.resize(f, (320, 180)), f)
    t = time.perf_counter()
    for f in frames:
        d.yolo.detect(f)
    ty = (time.perf_counter() - t) / len(frames)
    t = time.perf_counter()
    for f in frames:
        d(cv2.resize(f, (320, 180)), f)
    tt = (time.perf_counter() - t) / len(frames)
    print(f"YOLO itself {1000 * ty:.1f} ms/frame, whole detector {1000 * tt:.1f} ms/frame "
          f"(on a 5070 Ti: ~3-6 with the .pt, ~2-3 with --export)")
    return 0


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="train / check the YOLO object detector")
    p.add_argument("--train", action="store_true")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="cpu, 0 (first GPU), ... default: auto")
    p.add_argument("--base", default="yolov8n.pt", help="starting model: yolov8n.pt (fast) or yolov8s.pt (better)")
    p.add_argument("--check", default=None, help="mp4 to preview detections on (writes yolo_check_*.png)")
    p.add_argument("--hybrid", action="store_true", help="--check the built-in + YOLO mix instead of YOLO only")
    p.add_argument("--n", type=int, default=6, help="--check: how many frames")
    p.add_argument("--export", action="store_true",
                   help="make a TensorRT engine of yolo_gd.pt (fastest on an NVIDIA GPU; pip install tensorrt)")
    p.add_argument("--speed", action="store_true", help="time the detector on recs/*.mp4 frames")
    a = p.parse_args()
    if a.export:
        return export(a.imgsz)
    if a.speed:
        return speed()
    if a.train:
        return train(a.epochs, a.imgsz, base=a.base, device=a.device)
    if a.check:
        return check(a.check, a.n, a.hybrid)
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
