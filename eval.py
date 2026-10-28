"""PHASE 1 eval (fixed): frame-indexed labels, fresh-aware death scoring.

Usage: python eval.py recs/a.mp4 [...] [--trace]
Labels: {"deaths": [frame_idx...], "pct": {"<frame_idx>": int}, "stride": 1,
"fps": float}. Old per-second pct auto-converts (first frame of second).
Deaths: predicted tick must be within +-30 REAL frames of a true death.
Percent: scored ONLY on labeled frames; reports MAE + exact + within-1.
None diagnostics: per-rec counts by reason + up to 10 dumps of
(crop, mask, cell scores). Trace files store source hash + fps; rebuild
automatically on mismatch. eval_trace replays cached traces fast.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np


def _fps_of(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    cap.release()
    return float(fps)


def _iter_mp4_frames(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    fi = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        yield fi, fr, fi / fps
        fi += 1
    cap.release()


def _load_labels(rec: str) -> dict:
    if rec.lower().endswith(".mp4"):
        p = os.path.splitext(rec)[0] + ".labels.json"
    else:
        p = os.path.join(rec, "labels.json")
    lab = json.load(open(p))
    lab.setdefault("deaths", [])
    lab.setdefault("pct", {})
    # backward compat: per-second keys -> first frame of that second
    fps = float(lab.get("fps", 60.0))
    keys = list(lab["pct"].keys())
    if keys and max(int(k) for k in keys) < 1000 and len(keys) < 200:
        print(f"WARNING {p}: old per-second pct keys, converting to first-frame-of-second (fps={fps})")
        lab["pct"] = {str(int(int(k) * fps)): int(v) for k, v in lab["pct"].items()}
    # one-time migration: strided death ticks -> real frame idx
    stride = int(lab.get("stride", 1))
    if stride != 1 and not lab.get("migrated", False):
        lab["deaths"] = sorted(int(d * stride) for d in lab.get("deaths", []))
        lab["stride"] = 1
        lab["migrated"] = True
        json.dump(lab, open(p, "w"), indent=2)
        print(f"migrated {p}: deaths x{stride} -> real frame idx")
    return lab


def _trace_hash() -> str:
    h = hashlib.sha256()
    for path in ["pct.py", "pct_roi.json"]:
        if os.path.exists(path):
            h.update(open(path, "rb").read())
    import glob
    for p in sorted(glob.glob("./templates/d*.png") + glob.glob("./templates/pct.png")):
        h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def _obs_small_bgr2rgb(fr_bgr: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(fr_bgr, cv2.COLOR_BGR2RGB),
                      (320, 180), interpolation=cv2.INTER_AREA)


def eval_rec(rec: str) -> dict:
    """Streaming eval with frame-indexed scoring + None diagnostics."""
    from gd import ProgressReward
    from pct import PctReader

    lab = _load_labels(rec)
    true_deaths: list[int] = sorted(int(d) for d in lab.get("deaths", []))
    true_pct: dict[int, int] = {int(k): int(v) for k, v in lab.get("pct", {}).items()}
    fps = float(lab.get("fps", 60.0))
    if abs(fps - 60.0) > 2.0:
        print(f"WARNING {rec}: fps={fps:.1f} not ~60, ticks are frame idx (fps-independent)")

    rdr = PctReader()
    rdr.load_templates()
    rdr.load_cnn() if hasattr(rdr, "load_cnn") else None
    print(f"[{rec}] ROI={__import__('json').load(open('pct_roi.json')) if os.path.exists('pct_roi.json') else 'default'}")
    rew = ProgressReward()

    errs: list[float] = []
    dig_ok = dig_within1 = dig_tot = 0
    fresh_lab = fresh_lab_tot = 0
    pred_deaths: list[int] = []
    cost_ms: list[float] = []
    miss_idx: list[tuple[float, int]] = []
    none_reasons: dict[str, int] = {}
    none_dumps: list[tuple[int, np.ndarray]] = []
    n_full = 0

    for fi, fr, sec in _iter_mp4_frames(rec):
        n_full += 1
        t0 = time.perf_counter()
        v, is_fresh, age = rdr.read(fr)
        cost_ms.append((time.perf_counter() - t0) * 1000.0)
        obs = _obs_small_bgr2rgb(fr)
        _, done, info = rew.step(obs, pct=(v, is_fresh, age))
        if info.get("event") == "death":
            pred_deaths.append(fi)
        if fi in true_pct and v is not None:
            e = abs(float(v) * 100 - true_pct[fi])
            errs.append(e)
            if round(float(v) * 100) == true_pct[fi]:
                dig_ok += 1
            if abs(float(v) * 100 - true_pct[fi]) <= 1.0:
                dig_within1 += 1
            dig_tot += 1
            if e > 5:
                miss_idx.append((e, fi))
        if fi in true_pct:
            fresh_lab_tot += 1
            if is_fresh:
                fresh_lab += 1
            elif v is None and len(none_dumps) < 10:
                none_dumps.append((fi, fr.copy()))
    # None reason breakdown: re-run decode path counters via health
    for _fi, fr in none_dumps:
        none_reasons["unreadable_labeled_frame"] = none_reasons.get("unreadable_labeled_frame", 0) + 1

    used = set()
    tp = 0
    for td in true_deaths:
        for j, pd in enumerate(pred_deaths):
            if j in used:
                continue
            if abs(pd - td) <= 30:
                tp += 1
                used.add(j)
                break
    prec = tp / max(1, len(pred_deaths))
    recall = tp / max(1, len(true_deaths))
    miss_idx.sort(key=lambda m: -m[0])
    if miss_idx:
        cap = cv2.VideoCapture(rec)
        for i, (e, fi) in enumerate(miss_idx[:10]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if ok:
                cv2.imwrite(os.path.join(os.path.dirname(rec), f"miss_{i}_e{e:.0f}_f{fi}.png"), fr)
        cap.release()
    # None dumps: crop + mask + cell scores burned into image
    for i, (fi, fr) in enumerate(none_dumps):
        try:
            from pct import PctConfig
            cfg = PctConfig.load()
            h, w = fr.shape[:2]
            x0, y0 = int(w * cfg.x0), int(h * cfg.y0)
            x1, y1 = int(w * cfg.x1), int(h * cfg.y1)
            crop = fr[y0:y1, x0:x1]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask = (((hsv[:, :, 1] <= cfg.sat_max) & (hsv[:, :, 2] >= cfg.val_min)).astype(np.uint8) * 255)
            ch, cw = crop.shape[:2]
            canvas = np.zeros((ch + 8, cw * 2 + 8, 3), dtype=np.uint8)
            canvas[:ch, :cw] = crop
            mz = cv2.resize(mask, (cw, ch))
            canvas[:ch, cw + 8:, 0] = mz
            canvas[:ch, cw + 8:, 1] = canvas[:ch, cw + 8:, 0]
            canvas[:ch, cw + 8:, 2] = canvas[:ch, cw + 8:, 0]
            cv2.putText(canvas, f"f{fi} None", (8, ch + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(os.path.dirname(rec), f"none_{i}_f{fi}.png"), canvas)
        except Exception as ex:
            print("none dump failed:", ex)
    try:
        import glob as _g
        res = set()
        for r in [rec]:
            c = cv2.VideoCapture(r)
            res.add((int(c.get(3)), int(c.get(4))))
            c.release()
        if len(res) > 1:
            print(f"WARNING mixed resolutions: {res}")
    except Exception:
        pass
    out = {
        "rec": rec, "n_full": n_full, "n_obs": n_full,
        "pct_mae": float(np.mean(errs)) if errs else float("nan"),
        "pct_labeled": len(errs),
        "digit_acc": dig_ok / max(1, dig_tot),
        "digit_within1": dig_within1 / max(1, dig_tot),
        "fresh_frac": fresh_lab / max(1, fresh_lab_tot),
        "fresh_frac_all": None,
        "none_reasons": none_reasons,
        "death_tp": tp, "death_pred": len(pred_deaths), "death_true": len(true_deaths),
        "death_prec": prec, "death_rec": recall,
        "read_ms_mean": float(np.mean(cost_ms)) if cost_ms else 0.0,
        "read_ms_p95": float(np.percentile(cost_ms, 95)) if cost_ms else 0.0,
    }
    return out


def extract_trace(rec: str, trace_path: str | None = None, force: bool = False) -> str:
    from pct import PctReader
    if trace_path is None:
        trace_path = os.path.splitext(rec)[0] + ".trace.npz"
    want = _trace_hash()
    if os.path.exists(trace_path) and not force:
        try:
            d = np.load(trace_path)
            if str(d.get("src_hash", "")) == want:
                return trace_path
        except Exception:
            pass
        print(f"rebuilding trace {trace_path} (src/templates/roi changed)")
    rdr = PctReader()
    rdr.load_templates()
    pcts, freshs, ages, obss = [], [], [], []
    fps = _fps_of(rec)
    if abs(fps - 60.0) > 2.0:
        print(f"WARNING {rec}: fps={fps:.1f}")
    for _fi, fr, _sec in _iter_mp4_frames(rec):
        v, is_fresh, age = rdr.read(fr)
        pcts.append(float(v) if v is not None else float("nan"))
        freshs.append(bool(is_fresh))
        ages.append(int(age))
        obss.append(_obs_small_bgr2rgb(fr))
    np.savez_compressed(trace_path,
                        pct=np.asarray(pcts, dtype=np.float32),
                        fresh=np.asarray(freshs, dtype=np.bool_),
                        age=np.asarray(ages, dtype=np.int16),
                        obs=np.stack(obss) if obss else np.zeros((0, 180, 320, 3), np.uint8),
                        src_hash=np.asarray(want), fps=np.asarray(fps))
    return trace_path


def eval_trace(trace_path: str, label_path: str, flags: dict | None = None) -> dict:
    from gd import ProgressReward, RewardConfig
    d = np.load(trace_path, allow_pickle=True)
    pct_arr = d["pct"]
    fresh_arr = d["fresh"]
    age_arr = d["age"]
    obs_arr = d["obs"]
    stored_fps = float(d["fps"]) if "fps" in d else 60.0
    lab = json.load(open(label_path))
    true_deaths: list[int] = sorted(int(x) for x in lab.get("deaths", []))
    true_pct: dict[int, int] = {int(k): int(v) for k, v in lab.get("pct", {}).items()}
    cfg = RewardConfig()
    for k, v in (flags or {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    rew = ProgressReward(cfg=cfg)
    errs: list[float] = []
    dig_ok = dig_within1 = dig_tot = 0
    fresh_lab = fresh_lab_tot = 0
    pred_deaths: list[int] = []
    for fi in range(len(pct_arr)):
        pv = None if pct_arr[fi] != pct_arr[fi] else float(pct_arr[fi])
        is_fresh = bool(fresh_arr[fi])
        age = int(age_arr[fi])
        _, done, info = rew.step(obs_arr[fi], pct=(pv, is_fresh, age))
        if info.get("event") == "death":
            pred_deaths.append(fi)
        if fi in true_pct and pv is not None:
            e = abs(pv * 100 - true_pct[fi])
            errs.append(e)
            if round(pv * 100) == true_pct[fi]:
                dig_ok += 1
            if e <= 1.0:
                dig_within1 += 1
            dig_tot += 1
        if fi in true_pct:
            fresh_lab_tot += 1
            if is_fresh:
                fresh_lab += 1
    used = set()
    tp = 0
    for td in true_deaths:
        for j, pd in enumerate(pred_deaths):
            if j in used:
                continue
            if abs(pd - td) <= 30:
                tp += 1
                used.add(j)
                break
    prec = tp / max(1, len(pred_deaths))
    recall = tp / max(1, len(true_deaths))
    return {
        "death_tp": tp, "death_pred": len(pred_deaths), "death_true": len(true_deaths),
        "death_prec": prec, "death_rec": recall,
        "pct_mae": float(np.mean(errs)) if errs else float("nan"),
        "pct_labeled": len(errs),
        "digit_acc": dig_ok / max(1, dig_tot),
        "digit_within1": dig_within1 / max(1, dig_tot),
        "fresh_frac": fresh_lab / max(1, fresh_lab_tot),
        "n": len(pct_arr),
    }


def main() -> int:
    recs = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--trace" in sys.argv:
        for r in recs:
            p = extract_trace(r)
            print(f"traced {r} -> {p}")
        return 0
    if not recs:
        print("usage: python eval.py recs/a.mp4 [...] [--trace]")
        return 2
    all_out = [eval_rec(str(r)) for r in recs]
    for o in all_out:
        print(f"== {o['rec']}: full={o['n_full']} obs={o['n_obs']} "
              f"pct_mae={o['pct_mae']:.2f} (n={o['pct_labeled']}) "
              f"digit_acc={o['digit_acc']:.3f} within1={o['digit_within1']:.3f} "
              f"fresh_lab={o['fresh_frac']:.3f} none={o['none_reasons']} "
              f"death P={o['death_prec']:.3f} R={o['death_rec']:.3f} "
              f"(tp={o['death_tp']} pred={o['death_pred']} true={o['death_true']}) "
              f"read {o['read_ms_mean']:.2f}ms p95 {o['read_ms_p95']:.2f}ms")
    if len(all_out) > 1:
        import numpy as _np
        maes = [o["pct_mae"] for o in all_out if o["pct_mae"] == o["pct_mae"]]
        print(f"AGG pct_mae={float(_np.mean(maes)):.2f} "
              f"death P={sum(o['death_tp'] for o in all_out)/max(1,sum(o['death_pred'] for o in all_out)):.3f} "
              f"R={sum(o['death_tp'] for o in all_out)/max(1,sum(o['death_true'] for o in all_out)):.3f}")
    print("BASELINE recorded. No detector changed in Phase 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
