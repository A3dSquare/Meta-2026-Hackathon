"""Box labeler for the YOLO upgrade path (heuristic needs NO labels).

Usage: python label_boxes.py --rec recs/stereo.mp4
Keys: drag = new box, 0-6 = set class of current box, TAB = next class,
D = delete box under cursor, A/D = step, W/S = jump 10, ENTER = save,
Q = save+quit. Writes recs/<name>.boxes.json: [{fi,x0,y0,x1,y1,cls}] in
0..1 fractions of the FULL frame (resolution-free).

You do NOT need this for the heuristic backend. Only label if/when the
heuristic misses entity art on a new level and you want --backend yolo.
Target ~500 boxes across 3+ levels, then train detect_yolo.py.
"""
from __future__ import annotations

import argparse
import json
import os

CLASSES = ["spike", "orb_y", "orb_b", "orb_p", "pad", "portal", "wall"]


def main() -> int:
    p = argparse.ArgumentParser(description="entity box labeler")
    p.add_argument("--rec", required=True)
    args = p.parse_args()
    import cv2
    from label import FastMP4
    mp4 = FastMP4(args.rec)
    out = os.path.splitext(args.rec)[0] + ".boxes.json"
    boxes: list[dict] = []
    if os.path.exists(out):
        try:
            boxes = json.load(open(out))
        except Exception:
            boxes = []
    fi, cur_cls = 0, 0
    print("drag=box, 0-6=class, A/D=step, W/S=x10, ENTER=save, Q=quit")
    while True:
        fr = mp4.get(fi)
        if fr is None:
            break
        disp = fr.copy()
        H, W = disp.shape[:2]
        for b in boxes:
            if b["fi"] != fi:
                continue
            x0, y0, x1, y1 = int(b["x0"] * W), int(b["y0"] * H), int(b["x1"] * W), int(b["y1"] * H)
            cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(disp, CLASSES[b["cls"]], (x0, y0 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(disp, f"f{fi} cls={CLASSES[cur_cls]} n={len([b for b in boxes if b['fi']==fi])}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("boxes", disp)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("a"), ord("A")):
            fi = max(0, fi - 1)
        elif key in (ord("d"), ord("D")):
            fi = min(mp4.n - 1, fi + 1)
        elif key in (ord("w"), ord("W")):
            fi = max(0, fi - 10)
        elif key in (ord("s"), ord("S")):
            fi = min(mp4.n - 1, fi + 10)
        elif ord("0") <= key <= ord("6"):
            cur_cls = key - ord("0")
        elif key == 13:
            json.dump(boxes, open(out, "w"), indent=1)
            print(f"saved {len(boxes)} boxes -> {out}")
    json.dump(boxes, open(out, "w"), indent=1)
    print(f"saved {len(boxes)} boxes -> {out}")
    mp4.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
