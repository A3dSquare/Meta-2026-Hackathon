"""Route memory accuracy + replay timing (route.py, landmark_route.py).

  python -m unittest test_route_memory -v
"""
import tempfile
import time
import unittest

import numpy as np

import landmark_route as LR
import route as R

SPEED = LR.SPEED_1X
DT = 1 / 30
HAZ = [9.0, 12.5, 14.0, 19.0, 23.5, 25.0, 31.0, 36.5, 40.0, 47.0, 52.5, 55.0]   # true positions (blocks)


def view(p):
    """What the detector would show with the icon at TRUE level position p."""
    return [LR.Box(x - p, x - p + 0.9, 0.0, 0.9, "hazard") for x in HAZ if LR.BEHIND <= x - p <= 8.0]


def run_attempt(lm, p_start, n_steps, press_at=(), t0=0.0, empty_first=0):
    """Feed one attempt. p_start = true position when the attempt clock started."""
    p = p_start
    for k in range(n_steps):
        t = t0 + k * DT
        boxes = [] if k < empty_first else view(p)
        lm.observe(boxes, SPEED * DT if k else 0.0, SPEED, 0.0, t)
        a = int(any(abs(p - q) < 0.2 for q in press_at))
        lm.record(a, t)
        p += SPEED * DT
    return p


class LandmarkOrigin(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.lm = LR.LandmarkRoute(self.dir, verbose=False)
        self.lm.set_level("lv")

    def build(self):
        self.lm.clock_ok = True
        run_attempt(self.lm, 0.0, 120, press_at=(8.0, 18.0))
        rep = self.lm.end_attempt(119 * DT, True)
        self.assertTrue(rep["committed"], rep)

    def test_first_attempt_not_at_respawn_builds_no_map(self):
        # live 010907: the bot's first attempt began 5.6 blocks in; the map built from it put
        # every later attempt 5.6 blocks off and replays pressed ~0.5 s early
        self.lm.clock_ok = False
        run_attempt(self.lm, 5.6, 120, press_at=(8.0, 18.0))
        rep = self.lm.end_attempt(119 * DT, True)
        self.assertFalse(rep["committed"])
        self.assertEqual((self.lm.entries, self.lm.best_events), ([], []))
        self.build()                                   # the next one, from a respawn, does
        self.assertTrue(self.lm.origin_known())
        self.assertAlmostEqual(self.lm.origin(), 0.0, delta=0.05)

    def test_origin_learned_from_respawn_attempts(self):
        self.build()
        for _ in range(4):
            self.lm.clock_ok = True
            run_attempt(self.lm, 1.5, 60)            # first locks 1.5 blocks off (live 020035)
            self.lm.end_attempt(59 * DT, False)
        self.assertEqual(self.lm.origin(), 0.0)      # must not creep: the map's 0 IS the respawn
        self.lm._begin_tracker()
        self.assertEqual(self.lm.tracker.s, 0.0)
        self.assertEqual(self.lm.tracker.max_jump, LR.LOCK_MAX_JUMP)

    def test_origin_ignores_attempts_not_started_at_a_respawn(self):
        self.build()
        n = len(self.lm.origins)
        self.lm.clock_ok = False
        run_attempt(self.lm, 3.0, 60)
        self.lm.end_attempt(59 * DT, False)
        self.assertEqual(len(self.lm.origins), n)

    def test_origin_persists(self):
        self.build()
        lm2 = LR.LandmarkRoute(self.dir, verbose=False)
        lm2.set_level("lv")
        self.assertTrue(lm2.origin_known())

    def test_replay_presses_at_the_right_place(self):
        self.build()
        self.lm.clock_ok = True
        fired, p = [], 0.0
        for k in range(150):
            t = k * DT
            self.lm.observe([] if k < 3 else view(p), SPEED * DT if k else 0.0, SPEED, 0.0, t)
            a = self.lm.replay_action()
            if a == 1 and (not fired or abs(fired[-1] - p) > 1.0):
                fired.append(p)
            self.lm.record(int(a or 0), t)
            p += SPEED * DT
        self.assertEqual(len(fired), 2, fired)
        for q, want in zip(fired, (8.0, 18.0)):
            self.assertAlmostEqual(q, want, delta=0.5)


class LandmarkLatency(unittest.TestCase):
    def test_lat_moves_press_by_the_age_difference(self):
        lm = LR.LandmarkRoute(tempfile.mkdtemp(), verbose=False)
        self.assertAlmostEqual(lm._lat(SPEED, SPEED, 0.040, 0.090), SPEED * 0.05, places=6)
        self.assertEqual(lm._lat(SPEED, SPEED, 0.040, None), 0.0)
        self.assertEqual(lm._lat(SPEED, SPEED, 0.0, 1.0), LR.LAT_CLIP)          # clipped
        for _ in range(6):
            lm.record(0, 0.0, age=0.05)
        self.assertAlmostEqual(lm._lat(SPEED, SPEED, None, 0.05), 0.0, places=6)  # typical age as reference

    def test_age_now_reads_the_observation_timestamp(self):
        lm = LR.LandmarkRoute(tempfile.mkdtemp(), verbose=False)
        lm._obs_ts = time.perf_counter() - 0.06
        self.assertAlmostEqual(lm._age_now(), 0.06, delta=0.02)
        lm._obs_ts = time.perf_counter() - 5.0                                     # a pause: not latency
        self.assertIsNone(lm._age_now())


def fake_print(shift_px=0, seed=1):
    rng = np.random.default_rng(seed)
    wide = rng.uniform(20, 230, (R.PRINT_H, R.PRINT_W + 40, 3)).astype(np.float32)
    p = wide[:, 20 + shift_px:20 + shift_px + R.PRINT_W].copy()
    p[: int(0.33 * R.PRINT_H)] = 0
    p[:, int(0.26 * R.PRINT_W):int(0.50 * R.PRINT_W)] = 0
    return p


class RouteRecognition(unittest.TestCase):
    def test_shifted_print_is_the_same_level(self):
        a, b = fake_print(0), fake_print(6)
        self.assertGreater(R.print_distance(a, b), R.RouteMemory.MATCH)
        self.assertLess(R.print_distance_shifted(a, b), 1.0)
        self.assertGreater(R.print_distance_shifted(a, fake_print(0, seed=2)), 30.0)   # another level

    def test_recognises_level_sampled_late(self):
        rm = R.RouteMemory(tempfile.mkdtemp(), verbose=False)
        lv = rm._new_level([fake_print(0, 3), fake_print(0, 4), fake_print(0, 5)])
        rm._prints = [fake_print(6, 3), fake_print(6, 4), fake_print(6, 5)]
        self.assertIs(rm._recognise(), lv)
        self.assertEqual(len(rm.levels), 1)


class RouteTiming(unittest.TestCase):
    def make(self):
        rm = R.RouteMemory(tempfile.mkdtemp(), dt=DT, verbose=False, fixed_level="t")
        rm.start_attempt()
        return rm

    def test_decision_interval_is_measured(self):
        rm = self.make()
        for k in range(20):
            rm.action(k * 0.05, 0)
        self.assertAlmostEqual(rm.dec_dt, 0.05, places=3)

    def test_short_tap_survives_50ms_decisions(self):
        rm = self.make()
        rm.level["best"] = [[0.62, 0.64]]            # a 20 ms tap, between two 50 ms decisions
        rm.level["best_T"] = 5.0
        rm.plan = (5.0, 5.5, None, "x")
        hits = sum(rm.action(k * 0.05, 0)[0] for k in range(40))
        self.assertGreaterEqual(hits, 1)


class RouteRegression(unittest.TestCase):
    def test_practises_best_run_when_brain_keeps_dying_early(self):
        rm = R.RouteMemory(tempfile.mkdtemp(), dt=DT, verbose=False, fixed_level="t")
        rm.start_attempt()
        lv = rm.level
        lv["best"], lv["best_T"] = [[1.0, 1.1], [5.0, 5.1]], 17.0
        lv["best_dec"] = [[0.0, 0]]
        lv["spots"]["17.0"] = {"deaths": 1, "tried": [], "worked": None, "drift": 0}
        planned = 0
        for _ in range(12):
            rm.start_attempt()
            planned += rm.plan is not None
            rm.end_attempt(2.7)                    # the brain dies at ~4% every time
        self.assertGreater(planned, 0)            # was: never (the spot's deaths stay at 1)


class LandmarkNoisyScroll(unittest.TestCase):
    """Live 013804: per-step scroll readings dropped to 0 and jumped; the level repeats
    (spike + pillar pattern every ~11 blocks). Positions must still agree run to run."""
    REP = [6.6, 9.0, 12.4, 17.6, 20.0, 23.4, 28.6, 31.0, 34.4, 39.6, 42.0, 45.4, 50.6, 53.0, 56.4]

    def view(self, p):
        return [LR.Box(x - p, x - p + 0.9, 0.0, 0.9 if i % 3 else 2.0, "hazard" if i % 3 else "solid")
                for i, x in enumerate(self.REP) if LR.BEHIND <= x - p <= 8.0]

    def attempt(self, lm, p0, n, rng, press_at=(), replay=False):
        p, fired = p0, []
        for k in range(n):
            t = k * DT
            sc = SPEED * DT
            r = rng.random()
            if r < 0.15:
                sc = 0.0                       # "not moving" glitch
            elif r < 0.2:
                sc *= rng.uniform(1.5, 3.0)    # a bad shift reading
            lm.observe(self.view(p), sc if k else 0.0, SPEED, 0.0, t)
            if replay:
                a = lm.replay_action()
                if a == 1 and (not fired or abs(fired[-1] - p) > 1.0):
                    fired.append(p)
                a = int(a or 0)
            else:
                a = int(any(abs(p - q) < 0.2 for q in press_at))
            lm.record(a, t)
            p += SPEED * DT
        return fired

    def test_origin_tight_and_presses_in_place(self):
        rng = np.random.default_rng(0)
        lm = LR.LandmarkRoute(tempfile.mkdtemp(), verbose=False)
        lm.set_level("lv")
        lm.clock_ok = False                          # bot started mid-level: no map from it
        self.attempt(lm, 5.6, 150, rng, press_at=(5.9, 16.5, 27.5, 38.5))
        self.assertFalse(lm.end_attempt(149 * DT, True)["committed"])
        lm.clock_ok = True
        self.attempt(lm, 0.0, 150, rng, press_at=(5.9, 16.5, 27.5, 38.5))
        self.assertTrue(lm.end_attempt(149 * DT, True)["committed"])
        for _ in range(6):
            lm.clock_ok = True
            self.attempt(lm, 0.0, 60, rng)
            lm.end_attempt(59 * DT, False)
        o = np.asarray(lm.origins)
        self.assertLess(float(o.max() - o.min()), 1.0, lm.origins)
        self.assertAlmostEqual(lm.origin(), 0.0, delta=0.5)
        lm.clock_ok = True
        fired = self.attempt(lm, 0.0, 150, rng, replay=True)
        self.assertEqual(len(fired), 4, fired)
        for q, want in zip(fired, (5.9, 16.5, 27.5, 38.5)):
            self.assertAlmostEqual(q, want, delta=0.5)

    def test_old_maps_are_rebuilt(self):
        import json, os
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "lv_landmarks.json"), "w") as f:
            json.dump(dict(version=1, entries=[dict(x=1.0, w=1.0, h=1.0, kind="hazard", n=9)], events=[{"s": 1}]), f)
        lm = LR.LandmarkRoute(d, verbose=False)
        lm.set_level("lv")
        self.assertEqual((lm.entries, lm.best_events), ([], []))


class RouteTimingLog(unittest.TestCase):
    """The timing log must measure a known start offset and a known drift."""
    def frame(self, x):
        rng = np.random.default_rng(7)
        level = (rng.random((180, 4000, 3)) * 255).astype(np.uint8)
        level = np.repeat(level[:, ::8], 8, axis=1)                  # blocky, like a level
        px = int(round(x * 60))                                       # 60 px per level second
        return level[:, px:px + 320]

    def test_offset_and_drift_measured(self):
        rm = R.RouteMemory(tempfile.mkdtemp(), dt=DT, verbose=False, fixed_level="t")
        rm.start_attempt()
        for k in range(150):                                          # the best run: level time = clock
            t = k * DT
            rm.action(t, 0, self.frame(t))
            rm.record(t, 0)
        rm.end_attempt(150 * DT)
        self.assertGreater(rm.level["best_T"], 4.0)
        rm.start_attempt()
        for k in range(120):                                          # 60 ms late start, 15 ms/s drift
            t = k * DT
            rm.action(t, 0, self.frame(t + 0.060 + 0.015 * t))
        rep = rm.attempt_timing()
        self.assertGreater(rep["align_samples"], 50)
        self.assertAlmostEqual(rep["offset_start_ms"], 60 + 7, delta=12)
        self.assertAlmostEqual(rep["drift_ms_per_s"], 15, delta=5)
        self.assertLess(rep["residual_ms_std"], 10)


class AlignmentRunaway(unittest.TestCase):
    """Live 015528: on a repeating level picture alignment ran away to -1.9 s in 4 s."""
    def frame(self, x, period=24):
        rng = np.random.default_rng(3)
        tile = (rng.random((180, period, 3)) * 255).astype(np.uint8)
        level = np.tile(tile, (1, 4000 // period + 2, 1))
        level = level + (rng.random(level.shape) * 12).astype(np.uint8)     # small differences
        px = int(round(x * 60))
        return level[:, px:px + 320]

    def test_offset_stays_on_the_clock(self):
        rm = R.RouteMemory(tempfile.mkdtemp(), dt=DT, verbose=False, fixed_level="t")
        rm.start_attempt()
        for k in range(150):
            t = k * DT
            rm.action(t, 0, self.frame(t))
            rm.record(t, 0)
        rm.end_attempt(150 * DT)
        rm.start_attempt()
        for k in range(130):
            t = k * DT
            rm.action(t, 0, self.frame(t + 0.03))
            self.assertLess(abs(rm.offset), R.RouteMemory.ALIGN_START + 0.01, (t, rm.offset))
        self.assertLess(abs(rm.offset - 0.03), R.RouteMemory.ALIGN_DEV + 0.035)


class LandmarkLockGuard(unittest.TestCase):
    """Live 015528: at a respawn the lock jumped 2.2-3.1 blocks onto a look-alike spike row.
    Once the respawn is known, a lock the clock rules out must be refused."""
    MAP = ((5.0, 0.9), (8.0, 2.0))

    def setUp(self):
        self.tr = LR.Tracker()
        self.tr.begin([dict(x=x, w=0.9, h=h, kind="hazard") for x, h in self.MAP])

    def view(self, p):
        return [LR.Box(x - p, x - p + 0.9, 0.0, h, "hazard") for x, h in self.MAP]

    def test_unbounded_lock_jumps(self):
        self.tr.update(self.view(3.5), 0.0)            # the old behaviour: believes the picture
        self.assertTrue(self.tr.locked)
        self.assertAlmostEqual(self.tr.s, 3.5, delta=0.05)

    def test_bounded_lock_stays_with_the_clock(self):
        self.tr.max_jump = LR.LOCK_MAX_JUMP
        self.tr.update(self.view(3.5), 0.0)            # 3.5 blocks from where the clock says
        self.assertFalse(self.tr.locked)
        self.assertAlmostEqual(self.tr.s, 0.0, delta=0.01)
        self.tr.update(self.view(0.3), 0.0)            # a picture that fits the clock: locks
        self.assertTrue(self.tr.locked)
        self.assertAlmostEqual(self.tr.s, 0.3, delta=0.05)

class FrontierHandover(unittest.TestCase):
    """Live 020639: replays died at 5.0 s before a 5.7 s hand-over; frontier gave up after 2."""
    def make(self):
        rm = R.RouteMemory(tempfile.mkdtemp(), dt=DT, verbose=False, fixed_level="t", frontier=True)
        rm.start_attempt()
        lv = rm.level
        lv["best"], lv["best_T"], lv["best_dec"] = [[1.0, 1.1]], 8.7, [[0.0, 0]]
        return rm

    def test_keeps_going_and_hands_over_before_the_failure(self):
        rm = self.make()
        handovers = []
        for _ in range(20):
            rm.start_attempt()
            if rm.frontier_attempt:
                handovers.append(rm.plan[0])
                rm.end_attempt(5.0)            # the replay dies at 5.0 s
            else:
                rm.end_attempt(2.7)            # the brain from the start
        self.assertGreaterEqual(len(handovers), 8)            # was: 2, then off
        self.assertAlmostEqual(handovers[0], 5.7, places=3)
        self.assertTrue(all(h <= 5.5 + 1e-6 for h in handovers[1:]), handovers)   # never the failing 5.7 again
        self.assertIn(4.0, [round(h, 3) for h in handovers])                       # 1 s before the failure

    def test_hand_over_moves_back_when_replay_holds(self):
        rm = self.make()
        rm.start_attempt()
        if not rm.frontier_attempt:
            rm.end_attempt(2.7); rm.start_attempt()
        rm.end_attempt(5.0)                                    # died: cut to 4.0
        seen = []
        for _ in range(12):
            rm.start_attempt()
            if rm.frontier_attempt:
                seen.append(rm.plan[0])
                rm.end_attempt(rm.plan[0] + 0.5)              # reached the hand-over
            else:
                rm.end_attempt(2.7)
        self.assertAlmostEqual(seen[0], 4.0, places=3)
        self.assertAlmostEqual(seen[-1], 5.7, places=3)        # back to the normal hand-over


class RouteFolders(unittest.TestCase):
    def test_one_folder_per_level_and_old_files_moved(self):
        import os, json
        d = tempfile.mkdtemp()
        # the old flat layout
        old = R.RouteMemory(d, verbose=False)
        lv = old._new_level([fake_print(0, 3), fake_print(0, 4), fake_print(0, 5)])
        lid = lv["id"]
        old._save(lv)
        os.replace(os.path.join(d, lid, "route.json"), os.path.join(d, lid + ".json"))
        os.rmdir(os.path.join(d, lid))
        with open(os.path.join(d, lid + "_landmarks.json"), "w") as f:
            json.dump(dict(version=2, entries=[], events=[]), f)
        np.savez_compressed(os.path.join(d, lid + "_frames.npz"), t=np.zeros(5, np.float32), th=np.zeros((5, 2, 2), np.uint8))
        rm = R.RouteMemory(d, verbose=False)                       # migrates on load
        self.assertIn(lid, rm.levels)
        self.assertEqual(sorted(os.listdir(os.path.join(d, lid))), ["frames.npz", "landmarks.json", "route.json"])
        self.assertEqual([n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n))], [])
        lm = LR.LandmarkRoute(d, verbose=False)
        lm.set_level(lid)
        lm._save()
        self.assertTrue(os.path.exists(os.path.join(d, lid, "landmarks.json")))
        lv2 = rm._new_level([fake_print(0, 9)] * 3)                 # new levels go straight into folders
        rm._save(lv2)
        self.assertTrue(os.path.isfile(os.path.join(d, lv2["id"], "route.json")))


class BestRunSplice(unittest.TestCase):
    """A new best keeps the old run's presses where the old run already went."""
    def lm(self):
        lm = LR.LandmarkRoute(tempfile.mkdtemp(), verbose=False)
        lm.best_events = [dict(s=float(x), tag="old") for x in range(10, 101, 10)]   # old run to 100
        lm.best_s = 100.0
        return lm

    def test_old_presses_kept_new_ones_after(self):
        lm = self.lm()
        new = [dict(s=float(x) + 1, tag="new") for x in range(10, 150, 10)]          # new run to 150
        out = lm._splice(new, 150.0)
        cut = max(e["s"] for e in out if e["tag"] == "old")
        self.assertTrue(all(e["tag"] == "old" for e in out if e["s"] < 80))
        self.assertTrue(all(e["tag"] == "new" for e in out if e["s"] > 90))
        self.assertLess(cut, 86)
        self.assertEqual([e["s"] for e in out], sorted(e["s"] for e in out))

    def test_no_quiet_point_no_splice(self):
        lm = self.lm()
        lm.best_events = [dict(s=x * 0.5, tag="old") for x in range(0, 201)]          # a press every 0.5
        new = [dict(s=x * 0.5 + 0.25, tag="new") for x in range(0, 300)]
        self.assertIs(lm._splice(new, 150.0), new)

    def test_first_best_or_short_old_run_unchanged(self):
        lm = LR.LandmarkRoute(tempfile.mkdtemp(), verbose=False)
        new = [dict(s=5.0)]
        self.assertIs(lm._splice(new, 50.0), new)



class PrecisePress(unittest.TestCase):
    """--precise-press: a replayed press goes out at its point, not at the nearest decision."""
    def replay_errors(self, precise, phase):
        lm = LR.LandmarkRoute(tempfile.mkdtemp(), verbose=False)
        lm.set_level("lv")
        lm.clock_ok = True
        run_attempt(lm, 0.0, 120, press_at=(8.0, 18.0))
        self.assertTrue(lm.end_attempt(119 * DT, True)["committed"])
        want = [e["s"] for e in lm.best_events]
        old, LR.PRECISE_PRESS = LR.PRECISE_PRESS, precise
        try:
            lm.begin_attempt("lv")
            dt = 1 / 20                               # live: 50 ms decisions
            p, errs, prev = phase, [], 0
            for k in range(100):
                t = k * dt
                lm.observe([] if k < 3 else view(p), SPEED * dt if k else 0.0, SPEED, 0.0, t)
                a = int(lm.replay_action() or 0)
                if a and not prev:
                    wait = 0.0
                    if lm.press_at is not None:
                        wait = lm.press_at - time.perf_counter()
                        self.assertGreaterEqual(wait, -0.005)
                        self.assertLessEqual(wait, LR.PRECISE_REACH * dt + 0.01)
                    errs.append(p + SPEED * max(0.0, wait) - want[len(errs)])
                lm.record(a, t)
                prev = a
                p += SPEED * dt
        finally:
            LR.PRECISE_PRESS = old
        self.assertEqual(len(errs), 2, errs)
        return errs

    def test_precise_press_lands_on_the_point(self):
        worst_old = worst_new = 0.0
        for phase in np.linspace(0.0, SPEED / 20, 7, endpoint=False):
            worst_old = max(worst_old, max(abs(e) for e in self.replay_errors(False, phase)))
            worst_new = max(worst_new, max(abs(e) for e in self.replay_errors(True, phase)))
        self.assertGreater(worst_old, 0.15)           # +-half a 50 ms decision (0.26 block)
        self.assertLess(worst_new, 0.08)

    def test_precise_press_off_by_default(self):
        self.assertFalse(LR.PRECISE_PRESS)


class GhostPath(unittest.TestCase):
    """--ghost: follow the remembered path; learn per stretch from deaths."""
    def lm_on(self, ghost):
        lm = LR.LandmarkRoute(tempfile.mkdtemp(), verbose=False)
        lm.set_level("lv")
        lm.ghost = ghost
        lm.tracker.locked = lm.tracker.has_map = True
        return lm

    @staticmethod
    def arc(rise_at, h=2.0):
        """a jump arc that leaves the ground rise_at blocks ahead (None: stays on the ground)"""
        pts = []
        for x in np.arange(0.0, 15.0, 0.35):
            y = 0.0 if rise_at is None or x < rise_at else max(0.0, h * np.sin(min(np.pi, (x - rise_at) / 4.0 * np.pi)))
            pts.append((float(x), float(y)))
        return pts

    def arcs(self):
        return [("release", False, self.arc(None)), ("tap now", False, self.arc(0.0)),
                ("tap in 1", False, self.arc(0.5)), ("tap in 2", False, self.arc(1.0)),
                ("tap in 4", False, self.arc(2.0))]

    def ghost_jump_at(self, sj):
        return [(float(x), float(0.0 if x < sj else max(0.0, 2.0 * np.sin(min(np.pi, (x - sj) / 4.0 * np.pi)))))
                for x in np.arange(0.0, 60.0, 0.25)]

    def test_presses_when_the_path_rises_now_and_waits_otherwise(self):
        old, LR.GHOST = LR.GHOST, True
        try:
            lm = self.lm_on(self.ghost_jump_at(20.0))
            lm._snap = dict(s=20.0, y=0.0, locked=True)
            self.assertEqual(lm.ghost_action(self.arcs(), "cube"), 1)
            lm._snap = dict(s=18.0, y=0.0, locked=True)       # the path rises 2 blocks ahead: wait
            self.assertEqual(lm.ghost_action(self.arcs(), "cube"), 0)
            lm._snap = dict(s=10.0, y=0.0, locked=True)       # flat ahead: never an extra jump
            self.assertEqual(lm.ghost_action(self.arcs(), "cube"), 0)
            self.assertIsNone(lm.ghost_action(self.arcs(), "ship"))     # ship keeps the replay
        finally:
            LR.GHOST = old

    def test_learns_a_timing_shift_then_the_replay_where_it_keeps_dying(self):
        lm = self.lm_on(self.ghost_jump_at(20.0))
        z = str(int(20.0 // LR.GHOST_ZONE))
        self.assertEqual(lm._pick_ghost_arm(z), "0")
        tried = []
        for _ in range(80):                       # this stretch dies whatever the ghost does
            lm._ghost_arm = {z: lm._pick_ghost_arm(z)}
            tried.append(lm._ghost_arm[z])
            lm._score_ghost(20.0 + 2.0, False)
        self.assertTrue({"1", "-1", "r"} <= set(tried), set(tried))
        # stretches before the death count as passed
        lm._ghost_arm = {"2": "0", z: "1"}
        lm._score_ghost(20.0 + 2.0, False)
        self.assertEqual(lm.ghost_stats["2"]["0"], [1, 0])

    def test_shift_moves_the_press(self):
        old, LR.GHOST = LR.GHOST, True
        try:
            lm = self.lm_on(self.ghost_jump_at(20.0))
            z = str(int(20.0 // LR.GHOST_ZONE))
            lm._snap = dict(s=20.0, y=0.0, locked=True)
            lm._ghost_arm = {z: "3"}                   # ghost path moved 0.5 block later
            self.assertEqual(lm.ghost_action(self.arcs(), "cube"), 0)
        finally:
            LR.GHOST = old

    def test_ghost_grows_from_attempts_and_is_saved(self):
        lm = self.lm_on([])
        lm.ghost_log = [(float(x), 0.0) for x in np.arange(0.0, 50.0, 0.25)]
        lm._extend_ghost(40.0, best=False)
        self.assertAlmostEqual(lm.ghost[-1][0], 30.0, delta=0.3)       # 10 blocks before the death
        lm.ghost_log = [(float(x), 0.0) for x in np.arange(0.0, 80.0, 0.25)]
        lm._extend_ghost(None, best=False)                             # completed: all of it
        self.assertAlmostEqual(lm.ghost[-1][0], 79.75, delta=0.01)
        lm.ghost_stats = {"5": {"0": [3, 1]}}
        lm._save()
        lm2 = LR.LandmarkRoute(lm.folder, verbose=False)
        lm2.set_level("lv")
        self.assertEqual(len(lm2.ghost), len(lm.ghost))
        self.assertEqual(lm2.ghost_stats, {"5": {"0": [3, 1]}})


class HeightGuard(unittest.TestCase):
    def test_press_made_high_in_the_air_is_not_replayed_on_the_ground(self):
        lm = LR.LandmarkRoute(tempfile.mkdtemp(), verbose=False)
        lm.set_level("lv")
        lm.best_events = [dict(ax=None, d=None, s=10.0, y=2.9, v=SPEED, delay=0.066, steps=2, t=1.0,
                               rel=dict(ax=None, d=None, steps=2))]
        lm.best_s = 50.0
        lm._pick_offsets()
        lm._snap = dict(s=9.9, y=0.9, locked=True)
        lm.tracker.s, lm.y, lm.speed = 9.9, 0.9, SPEED
        lm.cur_mode = "cube"
        self.assertNotEqual(lm.replay_action(), 1)
        self.assertEqual(lm.report.get("height_skips"), 1)
        # the same press at the same height IS replayed, and the guard is cube-only
        for y, mode in ((2.9, "cube"), (0.9, "ship")):
            lm._reset_attempt(); lm._pick_offsets()
            lm._snap = dict(s=9.9, y=y, locked=True)
            lm.tracker.s, lm.y = 9.9, y
            lm.cur_mode = mode
            self.assertEqual(lm.replay_action(), 1, (y, mode))

if __name__ == "__main__":
    unittest.main()
