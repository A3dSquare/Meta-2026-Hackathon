"""Headless regression probe for the learning-failure fixes.

- Detector: counts corner-flicker spikes and spike coverage on recs/stereo.mp4
- 1-frame flash test: a lone triangle in ONE frame must NOT be emitted
- Reward: holding toward an approaching spike must NOT pay clearance
- Env: _prev_attempts/_dbg_ring initialised (old dead-code position crashed)
"""
import cv2
import numpy as np

from detect import HeuristicDetector

# --- 1+2: video scan + corner flicker count -------------------------------
det = HeuristicDetector()
cap = cv2.VideoCapture("recs/stereo.mp4")
tot = sp = frames = corner = frames_with_spike = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    small = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (320, 180))
    e = det(small)
    frames += 1
    tot += len(e.items)
    s = [x for x in e.items if int(x.cls) == 0]
    sp += len(s)
    if s:
        frames_with_spike += 1
    corner += sum(1 for x in s if (x.x < 0.12 or x.x > 0.88) and x.y > 0.75)
cap.release()
print(f"[detector] frames={frames} avg_ents={tot/max(1,frames):.2f} "
      f"avg_spikes={sp/max(1,frames):.2f} spike_frames={frames_with_spike} "
      f"({frames_with_spike/max(1,frames)*100:.0f}%) corner_spike_hits={corner}")

# --- 1b: the wall attempt: frames 0..170 of the recording ------------------
det1b = HeuristicDetector()
cap = cv2.VideoCapture("recs/stereo.mp4")
f170 = s170 = c170 = pl170 = 0
for fi in range(171):
    ok, fr = cap.read()
    if not ok:
        break
    small = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (320, 180))
    e = det1b(small)
    s = [x for x in e.items if int(x.cls) == 0]
    f170 += 1
    s170 += 1 if s else 0
    c170 += sum(1 for x in s if (x.x < 0.12 or x.x > 0.88) and x.y > 0.75)
    pl170 += 1 if any(int(x.cls) == 7 for x in e.items) else 0
cap.release()
print(f"[wall-attempt f0-170] spike_frames={s170}/{f170} player_frames={pl170} "
      f"corner_hits={c170}")

# --- 3: single-frame flash must be suppressed ------------------------------
det2 = HeuristicDetector()
blank = np.zeros((180, 320, 3), np.uint8)
flash = blank.copy()
cv2.drawContours(flash, [np.array([[10, 170], [30, 170], [20, 140]])], -1,
                 (255, 255, 255), -1)  # bottom-LEFT white triangle, 1 frame only
n0 = len([e for e in det2(blank).items if int(e.cls) == 0])
n1 = len([e for e in det2(flash).items if int(e.cls) == 0])
n2 = len([e for e in det2(flash).items if int(e.cls) == 0])  # 2nd tick: confirmed
print(f"[flash] blank_spikes={n0} flash_1tick_spikes={n1} (want 0) "
      f"flash_2tick_spikes={n2} (want >=0, gate test is the 1-tick one)")

# --- 4: clearance must not pay while spike approaches ----------------------
from gd import ProgressReward
r = ProgressReward()
r.reset()
tot_r = 0.0
edge = 0.24
for _ in range(30):  # approach: edge shrinks 0.24 -> ~0.06 while HOLDING
    rw, _, _ = r.step(np.zeros((4, 180, 320, 3), np.uint8), pct=(0.0, False, 999),
                      held=True, spike_edge=edge)
    tot_r += rw
    edge -= 0.006
print(f"[reward] 30 held approaching ticks total={tot_r:.2f} "
      f"(clear_pay+first_spike would have added >=+6 if inverted bug lived)")

# --- 5: box-env attribute init (old code never ran these lines) ------------
from env_boxes import EntityBoxEnv
import gymnasium as gym


class _Dummy(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (4, 180, 320, 3), np.uint8)
    action_space = gym.spaces.Discrete(2)


e = EntityBoxEnv(_Dummy(), detector=lambda f: __import__("detect").Entities([]))
print(f"[env] _prev_attempts={e._prev_attempts} _dbg_ring={e._dbg_ring} (init OK)")

print("PROBE_DONE")
