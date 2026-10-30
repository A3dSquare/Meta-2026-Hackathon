"""Run the REAL live stack against a fake game (no GD, no Windows needed).

    python test_live_stack.py [steps]

Real code under test: env.GymGDEnv (60 Hz step, pause/resume protocol),
gd.ProgressReward (death detection from the % value), gd.GDInput,
env_boxes.EntityBoxEnv (detector on rendered pixels, crash timing) and
trainer.Trainer (DQN). Faked: the screen (rendered GD-like frames from the
simulator's physics), the % text (exact value instead of template OCR), and
the keyboard (pydirectinput is replaced by a stub that the fake game reads:
SPACE held = jump, ESC = pause menu, SPACE while paused = resume).
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import types

import numpy as np


class _Keys:
    def __init__(self):
        self.down = set()
        self.events = []

    def keyDown(self, k):
        self.down.add(k)
        self.events.append(("down", k))

    def keyUp(self, k):
        self.down.discard(k)
        self.events.append(("up", k))


KEYS = _Keys()
_stub = types.ModuleType("pydirectinput")
_stub.keyDown, _stub.keyUp = KEYS.keyDown, KEYS.keyUp
_stub.PAUSE, _stub.FAILSAFE = 0, False
sys.modules.setdefault("pydirectinput", _stub)
sys.modules.setdefault("dxcam", types.ModuleType("dxcam"))

import cv2  # noqa: E402

import sim_gd as sg  # noqa: E402
import test_env_boxes as T  # noqa: E402


class FakeGame:
    """Rendered game + pause menu, driven by the stub keyboard."""

    def __init__(self, level):
        self.r = T.FakeGD(level)          # reuse its renderer + sim
        self.paused = False
        self.pct = 0.0
        self.ev_i = 0
        self.crashes = 0

    def _keys(self):
        ev = KEYS.events[self.ev_i:]
        self.ev_i = len(KEYS.events)
        for kind, k in ev:
            if kind == "down" and k == "esc":
                self.paused = True
            elif kind == "down" and k == "space" and self.paused:
                self.paused = False
                KEYS.down.discard("space")  # the resume tap doesn't count as a jump

    def tick(self):
        self._keys()
        if self.paused:
            return
        sim = self.r.sim
        was_alive = sim.dead == 0
        sim._tick("space" in KEYS.down)
        if was_alive and sim.dead > 0:
            self.crashes += 1
        if sim.dead == 0:
            self.r.cam_x = sim.x
            self.pct = math.floor(100 * sim.x / sim.length) / 100.0   # GD shows whole %
        if sim.signal >= 0:
            sim.signal = -1


class FakeCap:
    """Quacks like capture.GDCapture."""

    def __init__(self, game: FakeGame):
        self.g = game
        self._running = True
        self.last_fullres = None
        self._stack = []

    def start(self):
        return self

    def stop(self):
        pass

    def read(self):
        self.g.tick()
        bgr = self.g.r._render()
        self.last_fullres = bgr
        rgb = cv2.cvtColor(cv2.resize(bgr, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self._stack = (self._stack + [rgb])[-4:]
        while len(self._stack) < 4:
            self._stack.insert(0, rgb)
        return np.stack(self._stack)


def main() -> int:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    from env import EnvConfig, GymGDEnv
    from env_boxes import EntityBoxEnv
    from gd import GDInput
    from trainer import Trainer, TrainConfig

    level = ([sg.Obj("spike", 16, 0, 1, 0.95), sg.Obj("spike", 28, 0, 1, 0.45),
              sg.Obj("spike", 29, 0, 1, 0.45), sg.Obj("wall", 40, 0, 2, 1),
              sg.Obj("spike", 52, 0, 0.55, 0.45)], 300.0)
    game = FakeGame(level)
    cap = FakeCap(game)
    cfg_env = EnvConfig(max_steps=10**9, pct_hook=lambda full: game.pct)
    base = GymGDEnv(cfg_env, cap=cap, inp=GDInput(backend=KEYS))
    base.settle_s = 0.0
    env = EntityBoxEnv(base, frame_skip=2)
    d = tempfile.mkdtemp()
    # shield off: this test checks the death / crash-time / pause pipeline, so it needs deaths
    tr = Trainer(env, TrainConfig(total_steps=steps, save_dir=d, log_dir=d, name="live", verbose=0, shield=False,
                                route=False))   # (route memory is tested on its own)
    tr.run()
    import csv
    rows = list(csv.DictReader(open(tr._csv_path)))
    meas = sum(1 for r in rows if r["lag_src"] == "measured")
    peaks = [float(r["peak_pct"]) for r in rows]
    n_esc = sum(1 for k, v in KEYS.events if k == "down" and v == "esc")
    print(f"crashes in game: {game.crashes}   deaths seen by env: {len(rows)}   "
          f"crash time measured: {meas}/{len(rows)}   ESC pauses: {n_esc}   updates: {tr.updates}")
    print("peak % by attempt (first 40):", [round(p) for p in peaks[:40]])
    print("peak % by attempt (last 20): ", [round(p) for p in peaks[-20:]])
    deaths = [r for r in rows if float(r["peak_pct"]) < 99]
    meas = sum(1 for r in deaths if r["lag_src"] in ("measured", "median"))
    completed = len(rows) > len(deaths)
    print(f"deaths: {len(deaths)}  crash time measured on {meas}/{len(deaths)}  completed level: {completed}")
    ok = (abs(len(deaths) - game.crashes) <= 1 and meas >= 0.8 * max(1, len(deaths))
          and n_esc >= tr.updates > 0 and not game.paused)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
