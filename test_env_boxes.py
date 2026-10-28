"""End-to-end check of env_boxes.EntityBoxEnv on RENDERED frames (no game).

A fake GymGDEnv draws GD-looking frames from the simulator's physics
(scrolling ground, black spikes/blocks, cube, camera freeze while the
explosion plays, restart, then the % reset that reports the death). The real
EntityBoxEnv + HeuristicDetector + MotionMeter run on those pixels.

    python test_env_boxes.py

Checks: detector sees the spikes ahead, the camera-freeze death lag is
measured (not the fallback) and matches the true number of post-crash steps,
and observations are finite with the right shape.
"""
from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import sim_gd as sg
from env_boxes import OBS_DIM, EntityBoxEnv

W, H = 640, 360
BLK = H * 30.0 / 320.0


class _Cap:
    def __init__(self):
        self.last_fullres = None
        self._running = True


class _Reward:
    def __init__(self):
        self.attempts = 1


class FakeGD(gym.Env):
    """Quacks like env.GymGDEnv: step() -> (180,320,12) RGB stack."""

    observation_space = spaces.Box(0, 255, shape=(180, 320, 12), dtype=np.uint8)
    action_space = spaces.Discrete(2)

    def __init__(self, level):
        self.sim = sg.SimGD(level, frame_skip=1, p_miss=0.0, pos_noise=0.0)
        self.cap = _Cap()
        self.reward = _Reward()
        self._paused = False
        self._steps = 0
        self.cam_x = 0.0
        self.stack = []
        self.true_crash_tick = None
        self.tick = 0
        self.bg_off = 0

    def _render(self) -> np.ndarray:
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = (200, 95, 35)
        gy = int(0.75 * H)
        tile = int(BLK * 4)
        off = int(self.cam_x * BLK * 0.3) % tile  # parallax background
        for tx in range(-tile, W + tile, tile):
            if ((tx - off) // tile) % 2 == 0:
                img[:gy, max(0, tx - off):max(0, tx - off + tile)] = (215, 105, 40)
        img[gy:] = (150, 60, 20)
        cv2.line(img, (0, gy), (W, gy), (250, 240, 230), 2)
        goff = int(self.cam_x * BLK) % int(2 * BLK)
        for gx in range(-goff, W, int(2 * BLK)):
            cv2.line(img, (gx, gy + 3), (gx, H), (110, 40, 10), 2)
        px_screen = sg.PLAYER_SX * W
        for o in self.sim.objs:
            x0 = px_screen + (o.x - self.cam_x - 0.5) * BLK
            if x0 > W + BLK:
                break
            x1 = x0 + o.w * BLK
            if x1 < -BLK:
                continue
            yb = gy - o.y * BLK
            yt = yb - o.h * BLK
            if o.kind == "wall":
                cv2.rectangle(img, (int(x0), int(yt)), (int(x1), int(yb)), (10, 10, 10), -1)
                cv2.rectangle(img, (int(x0), int(yt)), (int(x1), int(yb)), (235, 235, 235), 1)
            else:
                if o.kind == "ceil":
                    pts = np.array([[x0, yt], [x1, yt], [(x0 + x1) / 2, yb]])
                else:
                    pts = np.array([[x0, yb], [x1, yb], [(x0 + x1) / 2, yt]])
                pts = pts.round().astype(np.int32)
                cv2.fillPoly(img, [pts], (8, 8, 8), lineType=cv2.LINE_AA)
                cv2.polylines(img, [pts], True, (245, 245, 245), 1, lineType=cv2.LINE_AA)
        if self.sim.dead == 0:
            cy = gy - (self.sim.y + 0.5) * BLK
            s = BLK * 0.95
            cv2.rectangle(img, (int(px_screen - s / 2), int(cy - s / 2)), (int(px_screen + s / 2), int(cy + s / 2)), (0, 0, 0), -1)
            cv2.rectangle(img, (int(px_screen - s / 3), int(cy - s / 3)), (int(px_screen + s / 3), int(cy + s / 3)), (200, 255, 60), -1)
        else:  # explosion particles
            rng = np.random.default_rng(self.tick)
            for _ in range(12):
                cv2.circle(img, (int(px_screen + rng.normal(0, 20)), int(gy - BLK / 2 + rng.normal(0, 20))), 3, (255, 255, 255), -1)
        return img

    def _obs(self):
        bgr = self._render()
        self.cap.last_fullres = bgr
        rgb = cv2.cvtColor(cv2.resize(bgr, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self.stack = (self.stack + [rgb])[-4:]
        while len(self.stack) < 4:
            self.stack.insert(0, rgb)
        return np.concatenate(self.stack, axis=2)

    def reset(self, *, seed=None, options=None):
        return self._obs(), {"attempts": self.reward.attempts}

    def step(self, a):
        self.tick += 1
        self._steps += 1
        was_alive = self.sim.dead == 0
        self.sim._tick(int(a) == 1)
        if was_alive and self.sim.dead > 0:
            self.true_crash_tick = self.tick
        if self.sim.dead == 0:
            self.cam_x = self.sim.x
        info = {"attempts": self.reward.attempts, "prog": self.sim.x / self.sim.length, "event": None}
        if self.sim.signal == 0:
            self.reward.attempts += 1
            self.sim.signal = -1
            info["event"] = "death"
            info["attempts"] = self.reward.attempts
        return self._obs(), 0.0, False, False, info

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def mark_trained(self):
        pass


def main() -> int:
    level = ([sg.Obj("spike", 16, 0, 1, 0.95), sg.Obj("spike", 24, 0, 1, 0.45),
              sg.Obj("spike", 25, 0, 1, 0.45), sg.Obj("spike", 33, 0, 0.55, 0.45),
              sg.Obj("wall", 40, 0, 2, 1)], 60.0)
    fake = FakeGD(level)
    env = EntityBoxEnv(fake, frame_skip=2)
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,), obs.shape
    seen_spikes = []
    deaths = []
    step_of_crash = None
    for t in range(600):
        obs, r, term, trunc, info = env.step(0)        # never jump: die at the first spike
        assert np.all(np.isfinite(obs))
        seen_spikes.append(sum(1 for e in env.last_ents.items if e.cls == 0))
        if fake.true_crash_tick is not None and step_of_crash is None:
            step_of_crash = t
        if info["died"]:
            true_lag = t - step_of_crash
            deaths.append((info["death_lag"], info["death_lag_src"], true_lag))
            fake.true_crash_tick = None
            step_of_crash = None
        if len(deaths) >= 3:
            break
    print("spikes visible per step (first 40):", seen_spikes[:40])
    print("deaths (measured_lag, source, true_lag):", deaths)
    ok = deaths and all(src == "measured" and abs(m - tl) <= 1 for m, src, tl in deaths)
    ok = ok and max(seen_spikes) >= 1
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
