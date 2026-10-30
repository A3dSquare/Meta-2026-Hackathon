"""Reward geometry, telescoping, terminal trimming and trainer integration."""
import argparse
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

from flight_reward import clearance_weight, flight_clearance, shaping_reward


class FlightGeometryTests(unittest.TestCase):
    def score(self, objects=(), **kwargs):
        return flight_clearance('ship', (.5,.5), objects, 10., **kwargs)

    def test_clearance_improves_away_from_floor_and_ceiling(self):
        low = self.score(floor=-.55, ceiling=3.)['score']
        middle = self.score(floor=-1.5, ceiling=1.5)['score']
        high = self.score(floor=-3., ceiling=.55)['score']
        self.assertGreater(middle, low)
        self.assertAlmostEqual(low, high)
        self.assertEqual(middle, 1.)
        self.assertEqual(self.score(floor=-5., ceiling=5.)['score'], middle)

    def test_ahead_hazards_matter_progressively_and_behind_dont(self):
        def hazard(x):
            return [(0,x,x+.4,-.5,.5,.9)]
        self.assertLess(self.score(hazard(.7))['score'], self.score(hazard(3.))['score'])
        self.assertFalse(self.score(hazard(-2.))['valid'])
        self.assertFalse(self.score(hazard(10.))['valid'])
        self.assertEqual(self.score(hazard(.3))['score'], 0.)
        self.assertFalse(self.score([(1,.3,.7,-.5,.5,1.), (5,.3,.7,-.5,.5,1.)])['valid'])

    def test_mode_and_tracking_gates(self):
        for mode in ('cube','robot','ball','ufo','spider','swing'):
            self.assertEqual(flight_clearance(mode,(.5,.5),[],10.,floor=-1.)['score'], 0.)
        for mode in ('ship','wave'):
            self.assertTrue(flight_clearance(mode,(.5,.5),[],10.,floor=-1.)['valid'])
        self.assertFalse(self.score(floor=-1., reliable=False)['valid'])
        self.assertFalse(self.score()['valid'])
        self.assertFalse(self.score([(0,1.,2.,0.,1.,.1)])['valid'])
        self.assertFalse(self.score([(0,1.,float('nan'),0.,1.,1.)])['valid'])
        self.assertFalse(flight_clearance('ship', (float('nan'),.5), [], 10.)['valid'])

    def test_mirrored_scene_and_bounds(self):
        rng = random.Random(42)
        for _ in range(100):
            y = rng.uniform(-3.,3.)
            boxes = [(rng.choice([0,6]),1.,2.,y,y+.5,1.)]
            flipped = [(c,a,b,-t,-s,p) for c,a,b,s,t,p in boxes]
            a = self.score(boxes, floor=-2., ceiling=3.)['score']
            b = self.score(flipped, floor=-3., ceiling=2.)['score']
            self.assertAlmostEqual(a,b)
            self.assertTrue(0. <= a <= 1.)

    def test_no_inferred_boundaries_enter_env_sample(self):
        from test_env_boxes import FakeGD
        from detect import Entities, Entity
        from env_boxes import EntityBoxEnv
        env = EntityBoxEnv(FakeGD(([],100.)))
        env.mode_est.mode = 'ship'; env.mode_est._icon_lock = 15
        env._tracking_age = 3; env._in_level = True
        env._last_floor_rel = -.55; env._last_ceil_rel = None
        env._floor_source = 'inferred'; env._lines = (None,None)
        entities = Entities([Entity(7,.3,.7,.05,.09,1.)])
        self.assertFalse(env._clearance_sample(entities)['valid'])
        env._floor_source = 'line'
        self.assertTrue(env._clearance_sample(entities)['valid'])
        env._death_frozen = True
        self.assertFalse(env._clearance_sample(entities)['valid'])

    def test_cli_validation_and_default(self):
        from train_boxes import add_common_args
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        self.assertEqual(parser.parse_args([]).clearance_reward, 0.)
        self.assertEqual(parser.parse_args(['--clearance-reward']).clearance_reward, .05)
        self.assertEqual(parser.parse_args(['--clearance-reward','.02']).clearance_reward, .02)
        for value in ('nan','inf','-.01','.2','garbage'):
            with self.assertRaises(argparse.ArgumentTypeError):
                clearance_weight(value)


class RewardAccountingTests(unittest.TestCase):
    def test_n_step_shaping_has_correct_discounted_endpoint(self):
        rng = random.Random(82)
        for n in range(1,30):
            phi = [rng.uniform(0.,.05) for _ in range(n+1)]
            total = sum(.98**j*shaping_reward(phi[j],phi[j+1],.98) for j in range(n))
            self.assertAlmostEqual(total, .98**n*phi[-1]-phi[0])

    def test_discounted_telescoping_through_modes_and_detection_gaps(self):
        # Zero potentials represent missing tracking and non-flight modes.
        potentials = [.02,.04,0.,.03,0.,.01,0.]
        gamma = .98
        total = sum(gamma**i * shaping_reward(a,b,gamma)
                    for i,(a,b) in enumerate(zip(potentials,potentials[1:])))
        self.assertAlmostEqual(total, -potentials[0])
        for a,b in zip(potentials,potentials[1:]):
            self.assertLessEqual(abs(shaping_reward(a,b,gamma)), .05)

    def make_trainer(self, root, weight=.05, play=False, steps=4, frontier=False):
        import numpy as np
        from trainer import Trainer, TrainConfig
        class Environment:
            last_clearance = None
            def reset(self):
                self.i = 0
                self.last_clearance = dict(valid=True, mode='ship', score=.2)
                return np.zeros(1), {}
            def step(self,a):
                self.i += 1
                self.last_clearance = dict(valid=True, mode='ship', score=.2+self.i*.1)
                return np.zeros(1),0.,False,False,dict(in_level=True, died=self.i==steps,
                                                      death_lag=0, prog=.01*self.i)
        class Learner:
            def __init__(self):
                self.segments = []
                self.saved = 0
            def act(self, obs, deterministic=False):
                return 0, {}
            def add_segment(self,obs,act,rew,next_obs,terminal,upto=None):
                self.segments.append((list(act),list(rew),terminal))
                return len(act)
            def save(self,*args):
                self.saved += 1
        cfg = TrainConfig(log_dir=root, save_dir=root, total_steps=steps, route=False, frontier=frontier,
                          shield=False, verbose=0, clearance_weight=weight, play=play)
        learner = Learner()
        return Trainer(Environment(),cfg,learner=learner),learner

    def test_delayed_death_uses_terminal_zero_at_trimmed_boundary(self):
        from trainer import _Seg
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            tr, learner = self.make_trainer(root)
            phi = [.02,.03,.04,.01,.02,.03]
            shape = [shaping_reward(a,b,tr.cfg.gamma) for a,b in zip(phi,phi[1:])]
            seg = _Seg(obs=[0]*5,act=[0]*5,rew=[.1+s for s in shape],
                       potential=phi[:-1],next_potential=phi[1:],shaping=shape)
            tr._finish_attempt(seg,dict(death_lag=2),.1,0,tag='frontier')
            tr._csv.close()
            self.assertEqual(len(learner.segments[0][0]), 3)
            self.assertAlmostEqual(seg.rew[-1], -10.04)
            self.assertAlmostEqual(sum(tr.cfg.gamma**i*s for i,s in enumerate(seg.shaping)), -.02)
            self.assertEqual(seg.next_potential[-1], 0.)

    def test_completion_refunds_next_potential_and_long_segment_keeps_alignment(self):
        from trainer import _Seg
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            tr, _ = self.make_trainer(root)
            seg = _Seg(obs=[0]*3,act=[0]*3,rew=[.1]*2+[.1+shaping_reward(.03,.04,.98)],
                       potential=[.01,.02,.03],next_potential=[.02,.03,.04],shaping=[0.,0.,.0092])
            seg.drop_front(2)
            self.assertEqual(seg.potential,[.03])
            tr._finish_attempt(seg,dict(complete=True),1.,0,tag='frontier')
            tr._csv.close()
            self.assertAlmostEqual(seg.rew[-1], 50.1-.03)
            self.assertEqual(seg.next_potential,[0.])

    def test_fully_trimmed_frontier_has_no_fake_training_transition(self):
        from trainer import _Seg
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            tr, learner = self.make_trainer(root)
            seg = _Seg(obs=[0]*2,act=[0]*2,rew=[.1]*2,potential=[.02]*2,
                       next_potential=[.02]*2,shaping=[-.0004]*2)
            tr._finish_attempt(seg,dict(death_lag=2),.1,0,tag='frontier')
            tr._csv.close()
            self.assertFalse(learner.segments)

    def test_training_loop_enabled_disabled_and_play(self):
        for weight,play in ((0.,False),(.05,False),(.05,True)):
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
                tr, learner = self.make_trainer(root,weight,play)
                tr.run()
                if play:
                    self.assertFalse(learner.segments)
                    self.assertEqual(learner.saved,0)
                    self.assertFalse(tr.env.clearance_enabled)
                else:
                    actions, rewards, terminal = learner.segments[0]
                    self.assertTrue(terminal)
                    base = [.1,.1,.1,-10.]
                    if weight == 0:
                        self.assertEqual(rewards,base)
                    else:
                        delta = [r-b for r,b in zip(rewards,base)]
                        self.assertAlmostEqual(sum(.98**i*d for i,d in enumerate(delta)), -.01)

    def test_frontier_replay_prefix_is_not_learned(self):
        class Route:
            phase = 'replay'
            frontier_attempt = True
            attempt_was_route = True
            controlled = True
            i = 0
            def start_attempt(self): pass
            def action(self,t,a,frame):
                self.i += 1
                self.phase = 'replay' if self.i <= 2 else 'brain'
                return a, self.i <= 2
            def record(self,*args): pass
            def end_attempt(self,*args,**kwargs): pass
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            tr,learner = self.make_trainer(root,frontier=True)
            tr.route = Route()
            tr.run()
            actions,rewards,_ = learner.segments[0]
            self.assertEqual(len(actions),2)
            # Handover starts at Phi=.05*.4=.02, not the attempt's initial Phi.
            self.assertAlmostEqual((rewards[0]-.1)+.98*(rewards[1]+10.), -.02)

    def test_route_practice_does_not_train_even_with_shaping(self):
        class Route:
            phase = 'replay'
            frontier_attempt = False
            attempt_was_route = True
            controlled = True
            def start_attempt(self): pass
            def action(self,t,a,frame): return a, True
            def record(self,*args): pass
            def end_attempt(self,*args,**kwargs): pass
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            tr,learner = self.make_trainer(root)
            tr.route = Route()
            tr.run()
            self.assertFalse(learner.segments)


if __name__ == '__main__':
    unittest.main()
