import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from training_diagnostics import SessionLog, Timings, flight_errors, supported_ground, terrain_scroll, select_death_lag


class DiagnosticsTests(unittest.TestCase):
    def test_empty_ship_room_clears_stale_freeze_without_hiding_player(self):
        import numpy as np
        from test_env_boxes import FakeGD
        from env_boxes import EntityBoxEnv
        from detect import Entities,Entity
        objects = [(6,.5+i*.1,.9,.04,.08) for i in range(4)]
        class Detector:
            shift=0.
            empty=False
            def __call__(self,*args):
                terrain = [] if self.empty else [Entity(c,x+self.shift,y,w,h) for c,x,y,w,h in objects]
                return Entities(terrain+[Entity(7,.3,.7,.05,.09,1.)],scroll=None)
        detector=Detector(); env=EntityBoxEnv(FakeGD(([],100.)),detector=detector)
        env.use_search=False
        frame=np.zeros((180,320,3),np.uint8)
        for i in range(5):
            detector.shift=-.018*i;env._observe(frame,False)
        for _ in range(3):env._observe(frame,False)
        self.assertTrue(env._death_frozen)
        detector.empty=True
        env.mode_est.mode='ship'
        env._observe(frame,True)
        self.assertFalse(env._death_frozen)
        self.assertIsNotNone(env.last_ents.player())
        self.assertTrue(env._raw_player_seen)
        detector.empty=False
        env._pixel_moving=True
        for _ in range(4):env._observe(frame,True)
        self.assertFalse(env._death_frozen)

    def test_recorded_platform_edge_wait_is_no_longer_safe(self):
        from detect import Entities, Entity
        from env_boxes import preview_boxes
        from gd_physics import Body, Phys, robust_search
        r = json.loads((Path(__file__).parent/'test_fixtures/platform_side_22pct.json').read_text())
        q = r['preview_inputs']
        e = Entities([Entity(*b[:5], pt=b[5]) for b in r['boxes']])
        hazards, solids, _ = preview_boxes(e, 17.)
        body = Body(q['y0'], q['vy0'], r['g'], q['grounded'], r['mode'], r['mini'])
        phys = Phys(cube_v0=q['cube_v0'], cube_g=q['cube_g'])
        result = robust_search(body, hazards, solids, q['floor_rel'], q['ceil_rel'],
                               r['speed'], phys, r['dt'], q['lag'], q['pending'])
        self.assertEqual(r['search']['00'], 1.)
        self.assertLess(result['00'], .9)
        self.assertGreaterEqual(result['1'], .9)

    def test_falling_cube_hits_side_before_landing_snap(self):
        from gd_physics import Body, Phys, block_side_contacts, block_side_hit, search_best, simulate_plans
        p = Phys()
        solid = [(.8, 2., -2., 0.)]
        contacts = block_side_contacts(0., .5, solid, .5)
        hit = block_side_hit(Body(.55,-12,1,False), False, .05, 10., p, contacts, .5)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit[0], .7)
        self.assertIsNone(block_side_hit(Body(1.3,-10,1,False), False, .05, 10., p, contacts, .5))
        self.assertIsNone(block_side_hit(Body(-3.,-1,1,False), False, .05, 10., p, contacts, .5))
        values = search_best(Body(.55,-12,1,False), [], solid, None, None, 10., p, .05, horizon=.2)
        self.assertTrue(all(values[k] < .5 for k in ('1','00','01')))
        _, paths = simulate_plans(Body(.55,-12,1,False), [], solid, None, None, 10., p,
                                  dec_dt=.05, horizon=.2, half=.5)
        self.assertTrue(all(hit for _, hit, _ in paths))
        safe = search_best(Body(1.3,-10,1,False), [], solid, None, None, 10., p, .05, horizon=.2)
        self.assertEqual(safe['1'], 1.)

    def test_side_contact_is_symmetric_and_respects_landing_margin(self):
        from gd_physics import Body, Phys, block_side_contacts, block_side_hit
        p = Phys()
        for mode in ('cube','robot'):
            up = block_side_contacts(0., .5, [(.8,2.,-2.,0.)], .5)
            down = block_side_contacts(0., .5, [(.8,2.,0.,2.)], .5)
            a = block_side_hit(Body(.55,-12,1,False,mode), False, .05, 10., p, up, .5)
            b = block_side_hit(Body(-.55,12,-1,False,mode), False, .05, 10., p, down, .5)
            self.assertIsNotNone(a); self.assertIsNotNone(b)
            self.assertAlmostEqual(a[0], b[0]); self.assertAlmostEqual(a[1], -b[1])
        # A lenient corner landing stays legal when the inner hitbox clears the top.
        contacts = block_side_contacts(0., .5, [(.8,2.,-2.,0.)], .5)
        self.assertIsNone(block_side_hit(Body(.9,-12,1,False), False, .05, 10., p, contacts, .5))

    def test_platform_observation_rejects_cached_infinite_floor(self):
        import numpy as np
        from test_env_boxes import FakeGD
        from detect import Entities, Entity
        from env_boxes import EntityBoxEnv
        detector = lambda *args: Entities([Entity(7,.28,.7,.05,.09),
                                           Entity(6,.28,.8,.15,.1)], scroll=-.018)
        env = EntityBoxEnv(FakeGD(([],100.)), detector=detector)
        env.use_lines = env.use_search = False
        env._floor_y = .7
        env._floor_hist.extend([.7]*5)
        env._hist.extend([(0.,0.)]*5)
        env._tracking_age = 1
        env._observe(np.zeros((180,320,3),np.uint8), False)
        self.assertIsNone(env._last_floor_rel)
        self.assertIsNone(env._floor_y)
        self.assertEqual(env._floor_source, 'rejected inferred floor')
        # The detected platform still supplies support without an infinite plane.
        self.assertTrue(env._last_inputs['grounded'])

    def test_cached_surface_does_not_follow_ship_or_bridge_platform_gap(self):
        from training_diagnostics import inferred_surface
        self.assertIsNone(inferred_surface(.9, .7, .1, 1, False, False, 'ship'))
        self.assertIsNone(inferred_surface(.7, .7, .1, 1, True, True, 'cube'))
        self.assertEqual(inferred_surface(.71, .7, .1, 1, True, False, 'cube'), -.5)
        self.assertAlmostEqual(inferred_surface(.5, .7, .1, 1, False, False, 'ship'), -2.5)
        self.assertIsNone(inferred_surface(.1, .3, .1, -1, False, False, 'ship'))
        self.assertEqual(inferred_surface(.3, .3, .1, -1, True, False, 'cube'), .5)

    def test_fingerprint_without_file_digest_and_nonfatal_failure(self):
        import hashlib
        from unittest.mock import patch
        from trainer import Trainer, TrainConfig
        from training_diagnostics import file_sha256
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            model = Path(root, 'model.engine')
            model.write_bytes(b'abc')
            with patch.object(hashlib, 'file_digest', None, create=True):
                self.assertEqual(file_sha256(model), hashlib.sha256(b'abc').hexdigest())
                env = SimpleNamespace(det=SimpleNamespace(yolo=SimpleNamespace(loaded_path=str(model), engine=True)))
                cfg = TrainConfig(log_dir=root, route=False, verbose=0)
                tr = Trainer(env, cfg, learner=SimpleNamespace())
                tr._csv.close()
                event = json.loads(Path(tr.session.directory, 'detector.jsonl').read_text())
                self.assertEqual(event['sha256'], hashlib.sha256(b'abc').hexdigest())
            with patch('trainer.file_sha256', side_effect=OSError('test read failure')):
                tr = Trainer(env, cfg, learner=SimpleNamespace())
                tr._csv.close()

    def test_yolo_selects_current_engine_but_not_stale_engine(self):
        import os
        import sys
        from unittest.mock import patch
        from detect_yolo import YoloDetector
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            weights, engine = Path(root,'model.pt'), Path(root,'model.engine')
            weights.write_bytes(b'weights'); engine.write_bytes(b'engine')
            os.utime(weights,(100,100)); os.utime(engine,(110,110))
            with patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=lambda *args, **kw: object())}):
                self.assertEqual(YoloDetector(str(weights)).loaded_path, str(engine.resolve()))
                self.assertEqual(YoloDetector(str(weights), device='cpu').loaded_path, str(weights.resolve()))
                os.utime(weights,(120,120))
                self.assertEqual(YoloDetector(str(weights)).loaded_path, str(weights.resolve()))

    def test_focus_recovery_releases_cached_hold(self):
        from unittest.mock import patch
        from test_live_stack import _Keys
        from test_env_boxes import FakeGD
        from gd import GDInput
        from env_boxes import EntityBoxEnv
        keys = _Keys()
        env = EntityBoxEnv(FakeGD(([],100.)))
        env.base.inp = GDInput(backend=keys)
        env.base.inp.set(True)
        with patch('winfocus.gd_is_foreground', return_value=False), patch('winfocus.focus_gd', return_value=True):
            env._ensure_focus(force=True)
        self.assertFalse(env.base.inp.held)
        env.base.inp.step(1)
        self.assertEqual(keys.events[-2:], [('up','space'),('down','space')])
        env.base.inp.close()

    def test_pause_resume_resets_motion_and_rearms_input(self):
        from test_live_stack import KEYS, FakeGame, FakeCap
        from gd import GDInput
        from env import GymGDEnv, EnvConfig
        from env_boxes import EntityBoxEnv
        KEYS.down.clear(); KEYS.events.clear()
        game = FakeGame(([],100.))
        inp = GDInput(backend=KEYS)
        base = GymGDEnv(EnvConfig(pct_hook=lambda _: game.pct), cap=FakeCap(game), inp=inp)
        base.settle_s = 0.
        env = EntityBoxEnv(base)
        env._ensure_focus = lambda **kwargs: None
        env.reset()
        inp.set(True)
        env.mode_est._actions.extend([True,True])
        env.phys._track = [0.,[(.04,1.)]]
        env.pause()
        self.assertFalse(inp.held)
        env.resume()
        self.assertFalse(env.mode_est._actions)
        self.assertIsNone(env.phys._track)
        self.assertEqual(env._tracking_age, 0)
        inp.step(1)
        _,_,_,_,info = base.step(1)
        self.assertEqual(info['action_applied'], 1)
        self.assertTrue(info['input_health']['held'])
        inp.close()

    def test_env_reports_suppressed_input(self):
        from test_env_boxes import FakeGD
        from env_boxes import EntityBoxEnv
        env = EntityBoxEnv(FakeGD(([],100.)))
        env._ensure_focus = lambda **kwargs: None
        env.reset()
        env._in_level = False
        _, _, _, _, info = env.step(1)
        self.assertEqual(info['action_applied'], 0)
        self.assertEqual(env._input_trace['requested'], 1)
        self.assertEqual(env._input_trace['applied'], 0)

    def test_missing_measurement_preserves_command_delay(self):
        from gd_physics import ModeEstimator, Phys
        mode = ModeEstimator()
        mode.lag = 2
        mode.update(0., False, .04, 10., Phys())
        mode.missing(True)
        mode.missing(False)
        self.assertEqual(mode.pending(), [True, False])
        self.assertFalse(mode.hist)
        mode.update(2., False, .04, 10., Phys())
        self.assertEqual(len(mode.hist), 1)
        self.assertEqual(mode.pending(), [False, False])
        self.assertTrue(mode.eff_hold())

    def test_gap_discards_fits_not_learned_values(self):
        import numpy as np
        from test_env_boxes import FakeGD
        from env_boxes import EntityBoxEnv
        from detect import Entities, Entity
        class Detector:
            y = .7
            def __call__(self, *args):
                return Entities([] if self.y is None else [Entity(7,.28,self.y,.05,.09)], scroll=-.018)
        detector = Detector()
        env = EntityBoxEnv(FakeGD(([],100.)), detector=detector)
        env.use_search = False
        frame = np.zeros((180,320,3),np.uint8)
        env._observe(frame, False)
        env.phys._track = [0.,[(.04, 1.)]]
        env.mode_cal._seg = [(0.,0.),(.04,1.)]
        before = (env.phys.cube_g, env.phys.cube_v0, env.phys.n_cal)
        detector.y = None
        env._observe(frame, True)
        self.assertIsNone(env.phys._track)
        self.assertFalse(env.mode_cal._seg)
        self.assertIsNone(env._prev_py)
        detector.y = .4
        env._observe(frame, False)
        self.assertEqual(env._raw_vy, 0.)
        self.assertFalse(env._calibration_valid)
        self.assertEqual(before, (env.phys.cube_g, env.phys.cube_v0, env.phys.n_cal))

    def test_keyboard_recovers_a_lost_hold(self):
        from test_live_stack import _Keys
        from gd import GDInput
        keys = _Keys()
        inp = GDInput(backend=keys, state_reader=lambda: 'space' in keys.down)
        inp.set(True)
        for _ in range(6):
            inp.set(True)
        self.assertEqual(inp.health['downs'], 1)
        keys.down.clear()  # Windows/another input source released it behind our cache.
        for _ in range(6):
            inp.set(True)
        self.assertIn('space', keys.down)
        self.assertEqual(inp.health['resends'], 1)
        inp.close()
        self.assertNotIn('space', keys.down)

    def test_unknown_keyboard_state_does_not_resend(self):
        from test_live_stack import _Keys
        from gd import GDInput
        keys = _Keys()
        inp = GDInput(backend=keys, state_reader=lambda: None)
        for _ in range(12):
            inp.set(True)
        self.assertEqual(inp.health['downs'], 1)
        self.assertIsNone(inp.health['os_key_down'])
        inp.close()

    def test_recorded_5pct_jump_was_falsely_approved(self):
        from detect import Entities, Entity
        from env_boxes import preview_boxes
        from gd_physics import Body, Phys, robust_search
        record = json.loads((Path(__file__).parent/'test_fixtures/search_timing_5pct.json').read_text())
        p = record['preview_inputs']
        entities = Entities([Entity(*b[:5], pt=b[5]) for b in record['boxes']])
        hazards, solids, _ = preview_boxes(entities, 17.)
        body = Body(p['y0'], p['vy0'], record['g'], p['grounded'], record['mode'], record['mini'])
        phys = Phys(cube_v0=p['cube_v0'], cube_g=p['cube_g'])
        result = robust_search(body, hazards, solids, p['floor_rel'], p['ceil_rel'],
                               record['speed'], phys, record['dt'], p['lag'], p['pending'])
        self.assertEqual(record['search']['1'], 1.)
        self.assertLess(result['1'], .5)
        self.assertEqual(result['00'], 1.)

    def test_measured_crash_is_not_replaced_by_history(self):
        self.assertEqual(select_death_lag(45, 'measured', [29]*10), (45, 'measured'))
        self.assertEqual(select_death_lag(18, 'fallback', [29]*10), (29, 'median-fallback'))

    def test_terrain_motion_ignores_false_player(self):
        old = [(6, .3+i*.1, .8, .04, .08) for i in range(4)]
        now = [(c, x-.018, y, w, h) for c,x,y,w,h in old]
        self.assertAlmostEqual(terrain_scroll(old, now), -.018)
        self.assertEqual(terrain_scroll(now, now+[(7,.7,.2,.04,.08)]), 0.)
        self.assertIsNone(terrain_scroll([], now))

    def test_repeated_floor_tiles_use_leftward_motion_prior(self):
        # A 67 ms decision moves almost one tile. The nearest tile appears
        # to move RIGHT, falsely halving the old speed estimate.
        old=[(6,.15+i*.052,.8,.05,.08) for i in range(12)]
        now=[(c,x-.036,y,w,h) for c,x,y,w,h in old]
        self.assertAlmostEqual(terrain_scroll(old,now,expected_shift=-.036),-.036)
        self.assertEqual(terrain_scroll(old,old,expected_shift=-.036),0.)
        # A real slower scene is still measurable; the prior is not a speed lock.
        slow=[(c,x-.01,y,w,h) for c,x,y,w,h in old]
        self.assertAlmostEqual(terrain_scroll(old,slow,expected_shift=-.027),-.01)

    def test_recorded_tight_arc_scroll_does_not_reverse_or_halve_speed(self):
        from env_boxes import BLOCK_W
        d=json.loads((Path(__file__).parent/'test_fixtures/scroll_alias.json').read_text())
        self.assertGreater(d['old_scroll'],0.)
        shift=terrain_scroll(d['previous'],d['current'],expected_shift=-10.38*BLOCK_W*d['dt'])
        self.assertLess(shift,0.)
        self.assertAlmostEqual(-shift/BLOCK_W/d['dt'],10.38,delta=.5)

    def test_positive_scroll_cannot_train_forward_speed(self):
        from collections import deque
        from env_boxes import EntityBoxEnv
        env=SimpleNamespace(speed=10.38,_speed_hist=deque(maxlen=15),_motion_dt=lambda:.05)
        EntityBoxEnv._update_speed(env,.012,True)
        self.assertEqual(env.speed,10.38);self.assertEqual(len(env._speed_hist),0)

    def test_doubled_tile_speed_is_rejected_but_portal_reference_is_respected(self):
        from collections import deque
        from env_boxes import EntityBoxEnv,BLOCK_W
        env=SimpleNamespace(speed=10.38,_speed_hist=deque(maxlen=15),_motion_dt=lambda:.033)
        EntityBoxEnv._update_speed(env,-20.76*BLOCK_W*.033,True)
        self.assertEqual(env.speed,10.38)
        EntityBoxEnv._update_speed(env,-8.*BLOCK_W*.033,True)
        self.assertAlmostEqual(env.speed,8.)
        # Named portal handling resets speed/history before the next sample.
        env.speed=20.76;env._speed_hist.clear()
        EntityBoxEnv._update_speed(env,-20.76*BLOCK_W*.033,True)
        self.assertAlmostEqual(env.speed,20.76)

    def test_false_icon_is_suppressed_only_after_freeze(self):
        import numpy as np
        from test_env_boxes import FakeGD
        from env_boxes import EntityBoxEnv
        from detect import Entities, Entity
        objects = [(6, .5+i*.1, .9, .04, .08) for i in range(4)]
        class Detector:
            shift = 0.
            def __call__(self, *args):
                return Entities([Entity(*q[:1], q[1]+self.shift, *q[2:]) for q in objects]
                                + [Entity(7,.28,.7,.05,.09)], scroll=-.018)
        detector = Detector()
        env = EntityBoxEnv(FakeGD(([],100.)), detector=detector)
        env.use_search = False
        frame = np.zeros((180,320,3),np.uint8)
        for i in range(5):
            detector.shift = -.018*i
            env._observe(frame, False)
        self.assertIsNotNone(env.last_ents.player())
        for _ in range(3):
            env._observe(frame, False)
        self.assertTrue(env._death_frozen)
        self.assertIsNone(env.last_ents.player())
        detector.shift -= .018
        env._observe(frame, False)
        self.assertFalse(env._death_frozen)
        self.assertIsNotNone(env.last_ents.player())

    def test_step_delay_and_seconds_schedule_agree(self):
        from gd_physics import search_best, Body, Phys, simulate_plans
        for dt in (.03333333333333333, .04, .043, .05):
            args = (Body(), [(2.8,3.2,-.5,.3)], [], -.5, None, 10., Phys(), dt, 2, [False,False])
            a = search_best(*args)
            b = search_best(*args, delay=2*dt)
            for k in ('1','00','01'):
                self.assertAlmostEqual(a[k], b[k], places=10)
        _, paths = simulate_plans(Body(), [], [], -.5, None, 10., Phys(), dec_dt=.041,
                                  lag=2, pending=[False,False])
        tap = next(points for name, hit, points in paths if name == 'tap now')
        # The pending queue ends exactly at 82 ms, not at 100 ms on a 30 Hz grid.
        self.assertTrue(any(abs(x/10-.082)<1e-9 for x,y in tap))

    def test_apex_is_not_ground(self):
        self.assertFalse(supported_ground(True, 0., 1, [-2.5]))
        self.assertTrue(supported_ground(True, 0., 1, [.1]))
        self.assertFalse(supported_ground(False, 12., 1, [.1]))
        self.assertTrue(supported_ground(False, -8., 1, [.1]))
        self.assertTrue(supported_ground(False, 8., -1, [-.1]))

    def test_ballistic_error_uses_actual_time(self):
        steps = []
        for i in range(7):
            t = i * .04
            steps.append(dict(mode='cube', player=True, g=1, y=2+10*t-47.5*t*t,
                              dt=.033, motion_dt=.04,
                              preview_inputs=dict(grounded=False, vy0=10-95*t, y0=0, cube_g=95)))
        errors = list(flight_errors(steps))
        self.assertTrue(errors)
        self.assertLess(max(abs(e['error']) for e in errors), 1e-12)
        steps[1]['preview_inputs']['grounded'] = True
        self.assertFalse(any(e['index'] == 0 for e in flight_errors(steps)))
        self.assertFalse(list(flight_errors(steps, crash_index=0)))

    def test_sessions_preserve_legacy_and_have_cause(self):
        from dataclasses import dataclass
        @dataclass
        class Config:
            frontier: bool = False
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            legacy = Path(root, 'attempts.csv')
            legacy.write_text('old header\n')
            a, b = SessionLog(root, Config(), None), SessionLog(root, Config(), None)
            self.assertNotEqual(a.id, b.id)
            a.writerow([0, 1, 20, 0, 5, 30, 'median', 0, 'frontier', 'spike'])
            a.file.close(); b.file.close()
            with open(Path(a.directory, 'attempts.csv')) as f:
                row = list(csv.DictReader(f))[0]
            self.assertEqual(row['cause'], 'spike')
            self.assertEqual(row['session_id'], a.id)
            self.assertEqual(legacy.read_text(), 'old header\n')

    def test_timing_summary(self):
        t = Timings()
        for i in range(1, 101):
            t.add('search', i/1000)
        self.assertEqual(t.summary()['search']['p95_ms'], 95.)
        self.assertEqual(t.summary(), {})

    def test_frontier_empty_after_trim(self):
        from trainer import Trainer, TrainConfig, _Seg
        import numpy as np
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            cfg = TrainConfig(log_dir=root, save_dir=root, route=False, verbose=0)
            learner = SimpleNamespace(add_segment=lambda *args: 0)
            env = SimpleNamespace()
            tr = Trainer(env, cfg, learner=learner)
            seg = _Seg()
            seg.obs = [np.zeros(1), np.zeros(1)]
            seg.act = [0, 0]; seg.rew = [.1, .1]
            tr._finish_attempt(seg, dict(death_lag=30, attempt_total=-20), .05, np.zeros(1), tag='frontier')
            tr._csv.close()
            with open(tr._csv_path) as f:
                row = list(csv.DictReader(f))[0]
            self.assertEqual(row['survived_steps'], '0')

    def test_supported_ground_in_physics(self):
        from env_boxes import physics_features
        from detect import Entities, Entity
        from gd_physics import ModeEstimator, Phys
        out = {}
        physics_features(Entities([Entity(7, .3, .4, .05, .09, 1)]), 10,
                         ModeEstimator(), Phys(), 2, 0., True, False, -3., None,
                         out=out, search=False)
        self.assertFalse(out['inputs']['grounded'])

    def test_yolo_annotation_keeps_source(self):
        import numpy as np
        from preview_yolo import annotate
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        output = annotate(frame, [(0, .1, .2, .3, .5, .9)], ['spike'])
        self.assertEqual(int(frame.sum()), 0)
        self.assertGreater(int(output.sum()), 0)


if __name__ == '__main__':
    unittest.main()
