"""Live detector check. Run with GD in the foreground:  python detect_debug.py
Saves detect_debug.png: [frame+boxes | white mask (spike outline)] / [dark mask (spike body) | cube hue mask]
so you can SEE why spikes/player are or aren't found, and prints what it found."""
import cv2, numpy as np
from capture import GDCapture
from detect import HeuristicDetector, CLASSES

cap = GDCapture(); cap.start()
for _ in range(15): cap.read()
obs = cap.read()                       # (4,180,320,3) RGB
frame = np.ascontiguousarray(obs[-1])
ents = HeuristicDetector()(frame)
print("shape", frame.shape, "entities:", len(ents.items))
for e in ents.items: print(f"  {CLASSES[e.cls]:7s} x={e.x:.2f} y={e.y:.2f} w={e.w:.3f} h={e.h:.3f}")

bgr = frame[:, :, ::-1].copy()
vis = bgr.copy()
for e in ents.items:
    x0, y0 = int((e.x-e.w/2)*320), int((e.y-e.h/2)*180)
    x1, y1 = int((e.x+e.w/2)*320), int((e.y+e.h/2)*180)
    cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,0), 1)
    cv2.putText(vis, CLASSES[e.cls], (x0, max(8,y0-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
white = cv2.cvtColor(cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1], cv2.COLOR_GRAY2BGR)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
cube = cv2.cvtColor(cv2.inRange(hsv, (55,100,100), (95,255,255)), cv2.COLOR_GRAY2BGR)
dark = cv2.cvtColor(cv2.inRange(hsv, (0,0,0), (179,255,70)), cv2.COLOR_GRAY2BGR)
out = np.vstack([np.hstack([vis, white]), np.hstack([dark, cube])])
cv2.imwrite("detect_debug.png", cv2.resize(out, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
cv2.imwrite("detect_debug_full.png", cap.last_fullres)
cap.stop(); print("wrote detect_debug.png (and detect_debug_full.png)")