"""Death snapshots: what the AI saw in the last second before each crash.

On every death, EntityBoxEnv hands this the last ~2 s of steps it kept in
memory. One PNG per death goes to logs_boxes/deaths/: 12 tiles, 0.1 s apart,
ending just after the moment of the crash (red border), each with

  - the boxes the detector found (red = spike, gray = block, green = player, ...)
  - the predicted paths (thick green = tap now, yellow = tap in 4,
    magenta = hold, white dots = don't press, thick cyan = the best line the search
    found, red X = that path crashes), light blue line = detected ground / ceiling
  - a header: time before the crash, what was sent (PRESS / ----), whether the
    safety shield changed the brain's choice, the mode the recogniser thinks
    it's in, the input delay, how much the brain wanted to jump (Q hold-release)
  - a second line: the search's safety score if it presses now / waits, with every later
    decision played well (1 = safe, 0.9-0.99 = safe only if the timing is exactly as
    measured, below 0.85 = crashes) -- what the shield uses.

Plus a .json with the same numbers for every kept step (for analysis).
Only the newest `keep` deaths are kept, so the folder never grows past ~60 MB.
"""
from __future__ import annotations

import json
import os

import numpy as np

_COL = {0: (0, 0, 255), 1: (0, 255, 255), 2: (255, 160, 0), 3: (255, 0, 255),
        4: (0, 165, 255), 5: (255, 0, 200), 6: (170, 170, 170), 7: (0, 255, 0)}
TILE_W, TILE_H = 426, 240
COLS = 3
N_TILES = 12
EVERY = 3            # steps between tiles (3 steps = 0.1 s at 30 decisions/s)


def plan_summary(plan_feats, extra_feats=None) -> dict:
    """Per plan: survival 0..1 (1 = survives the whole 1.5 s) + best per first action."""
    from gd_physics import EXTRA_PLANS, PLANS
    out, best = {}, {0: 0.0, 1: 0.0}
    if plan_feats is None:
        return {"plans": out, "best_press": None, "best_release": None}
    f = np.asarray(plan_feats, np.float32)
    items = [(name, plan, f[3 * pi:3 * pi + 3]) for pi, (name, plan) in enumerate(PLANS)]
    if extra_feats is not None and np.any(extra_feats):
        fx = np.asarray(extra_feats, np.float32)
        items += [(name, plan, fx[3 * pi:3 * pi + 3]) for pi, (name, plan) in enumerate(EXTRA_PLANS)]
    for name, plan, f3 in items:
        s = 1.0 if f3[0] < 0.5 else float(f3[1])
        out[name] = round(s, 3)
        first = int(bool(plan(0)))
        best[first] = max(best[first], s)
    any_ = bool(np.any(f[:3 * len(PLANS)]))
    return {"plans": out, "best_press": round(best[1], 3) if any_ else None,
            "best_release": round(best[0], 3) if any_ else None}


def _draw(frame, rec) -> np.ndarray:
    import cv2
    from env_boxes import BLOCK_H, BLOCK_W
    img = frame.copy()
    H, W = img.shape[:2]
    for bx in rec["boxes"]:
        cls, x, y, w, h = bx[:5]
        x0, y0 = int((x - w / 2) * W), int((y - h / 2) * H)
        x1, y1 = int((x + w / 2) * W), int((y + h / 2) * H)
        cv2.rectangle(img, (x0, y0), (x1, y1), _COL.get(cls, (255, 255, 255)), 2)
    pp = rec.get("player_pos")
    if pp is not None:
        ppx, ppy = pp
        for name, hit, pts in rec["arcs"]:
            if name not in ("tap now", "tap in 4", "release", "hold", "best") or len(pts) < 2:
                continue
            if name == "hold" and rec["mode"] not in ("ship", "wave", "robot"):
                continue
            P = np.array([[(ppx + px * BLOCK_W) * W, (ppy - py * BLOCK_H) * H] for px, py in pts], np.int32)
            col = {"tap now": (0, 255, 0), "tap in 4": (0, 255, 255), "hold": (255, 0, 255),
                   "best": (255, 255, 0)}.get(name, (255, 255, 255))
            if name == "release":
                for q in P[::3]:
                    cv2.circle(img, (int(q[0]), int(q[1])), 2, col, -1)
            else:
                thick = name in ("tap now", "best")
                cv2.polylines(img, [P], False, (0, 0, 0), 4 if thick else 3, cv2.LINE_AA)
                cv2.polylines(img, [P], False, col, 2 if thick else 1, cv2.LINE_AA)
            if hit:
                ex, ey = (int(v) for v in P[-1])
                cv2.line(img, (ex - 6, ey - 6), (ex + 6, ey + 6), (0, 0, 255), 3)
                cv2.line(img, (ex - 6, ey + 6), (ex + 6, ey - 6), (0, 0, 255), 3)
    for q in rec.get("lines") or []:          # detected ground / ceiling line (light blue)
        if q is not None:
            cv2.line(img, (0, int(q * H)), (W - 1, int(q * H)), (255, 200, 120), 1)
    tile = cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    return tile


def _header(tile, rec, t_rel: float, is_crash: bool) -> None:
    import cv2
    sent = "PRESS" if rec["a"] else "-----"
    sh = f" SHIELD(brain {'PRESS' if rec['a_brain'] else '-----'})" if rec.get("shielded") else ""
    q = rec.get("qdiff")
    m = rec["mode"] + (" mini" if rec["mini"] else "") + (" UPSIDE" if rec["g"] < 0 else "")
    l1 = f"{t_rel:+.2f}s {sent}{sh}  {m} d{rec['lag']}" + (f" Q{q:+.2f}" if q is not None else "")
    s = rec["plan"]
    bp, br = s.get("best_press"), s.get("best_release")
    fl, ce = rec.get("floor_rel"), rec.get("ceil_rel")
    surf = (f" floor{fl + 0.5:+.1f}" if fl is not None else " floor?") + (f" ceil{ce - 0.5:+.1f}" if ce is not None else "")
    sr = rec.get("search")
    if sr:   # the search (later decisions played well): what the shield goes by
        l2 = (f"safe? press {sr['1']:.2f} wait {max(sr['00'], sr['01']):.2f}"
              f"{' (only if press next)' if sr['00'] < sr['01'] else ''} spikes {rec['n_spikes']}{surf}"
              + ("" if rec["player"] else "  NO PLAYER"))
    else:
        l2 = ("no preview (player not found)" if bp is None else
              f"press-now {bp:.2f} release-now {br:.2f} spikes {rec['n_spikes']}{surf}"
              + ("" if rec["player"] else "  NO PLAYER"))
    cv2.rectangle(tile, (0, 0), (TILE_W, 30), (0, 0, 0), -1)
    cv2.putText(tile, l1, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, l2, (3, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 255, 200), 1, cv2.LINE_AA)
    if is_crash:
        cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H - 1), (0, 0, 255), 3)


def save(buf: list, lag: int, attempt: int, out_dir: str, keep: int = 30, peak_pct=None) -> str | None:
    """buf: per-step records (oldest first); lag: steps from the crash to the death
    signal (the env's measured death lag). Returns the PNG path."""
    import cv2
    if not buf:
        return None
    os.makedirs(out_dir, exist_ok=True)
    crash = max(0, len(buf) - 1 - int(lag))            # the last step before the crash
    first = crash + EVERY - EVERY * N_TILES            # 11 tiles before the crash tile, 1 after
    idx = [i for i in range(first, crash + 2 * EVERY, EVERY) if 0 <= i < len(buf)][-N_TILES:]
    dt = buf[-1].get("dt", 1 / 30)
    tiles = []
    for i in idx:
        rec = buf[i]
        tile = _draw(rec["frame"], rec)
        _header(tile, rec, (i - crash) * dt, i == crash or (i < crash < i + EVERY))
        tiles.append(tile)
    while len(tiles) % COLS:
        tiles.append(np.zeros((TILE_H, TILE_W, 3), np.uint8))
    rows = [np.hstack(tiles[k:k + COLS]) for k in range(0, len(tiles), COLS)]
    sheet = np.vstack(rows)
    title = np.zeros((26, sheet.shape[1], 3), np.uint8)
    pk = f"  died at {peak_pct:.0f}%" if peak_pct is not None else ""
    cv2.putText(title, f"attempt {attempt}{pk}  (red border = crash moment, cut {lag} steps after it)",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    sheet = np.vstack([title, sheet])
    base = os.path.join(out_dir, f"death_{attempt:05d}")
    cv2.imwrite(base + ".png", sheet)
    lo = max(0, crash - 45)
    data = {"attempt": attempt, "peak_pct": peak_pct, "cut_steps": int(lag), "crash_index": crash - lo,
            "steps": [{k: v for k, v in r.items() if k not in ("frame", "arcs")} for r in buf[lo:]]}
    with open(base + ".json", "w") as f:
        json.dump(data, f, default=float)
    # keep only the newest `keep` deaths
    old = sorted(p for p in os.listdir(out_dir) if p.startswith("death_") and p.endswith(".png"))
    for p in old[:-keep] if keep > 0 else []:
        for ext in (".png", ".json"):
            try:
                os.remove(os.path.join(out_dir, p[:-4] + ext))
            except OSError:
                pass
    return base + ".png"
