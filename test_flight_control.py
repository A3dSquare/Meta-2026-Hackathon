"""Flight safety priorities, recovery physics, controller stability and DQN policy."""
import importlib.util
import math
import unittest
import numpy as np

from flight_control import FlightShield, flight_segment_hit, path_clearance, recovery_hint, teacher_action
from gd_physics import Body, Phys, robust_search


def metadata(press=1.,wait=1.,next_press=1.,q1=.8,q0=.8,q01=None,mode='ship'):
    return dict(mode=mode,survival={'1':press,'00':wait,'01':next_press},
                clearance={'1':q1,'00':q0,'01':q0 if q01 is None else q01})


class FlightSelectorTests(unittest.TestCase):
    def test_teacher_requires_verified_unambiguous_robust_intervention(self):
        s = FlightShield(); m = metadata(q1=.95,q0=.5)
        s.choose(0,m)
        self.assertEqual(teacher_action(m,s.last,False),-1)
        self.assertEqual(teacher_action(m,s.last,True),1)
        s.choose(1,m)
        self.assertEqual(teacher_action(m,s.last,True),-1)
        m = metadata(q1=.86,q0=.8);s.choose(0,m)
        self.assertEqual(teacher_action(m,s.last,True),-1)
        m = metadata(press=.95,wait=.2,next_press=.2);s.choose(0,m)
        self.assertEqual(teacher_action(m,s.last,True),-1)

    def test_collision_safety_wins_over_clearance_and_hysteresis(self):
        shield = FlightShield()
        self.assertEqual(shield.choose(0,metadata(q1=1.,q0=.2)),1)
        self.assertEqual(shield.choose(1,metadata(press=.2,q1=1.,q0=0.)),0)
        self.assertEqual(shield.choose(0,metadata(wait=.2,next_press=.2,q1=0.,q0=1.)),1)

    def test_doomed_paths_choose_longest_survival(self):
        self.assertEqual(FlightShield().choose(0,metadata(.6,.2,.2,q1=0.,q0=1.)),1)
        self.assertEqual(FlightShield().choose(1,metadata(.1,.6,.2,q1=1.,q0=0.)),0)
        self.assertEqual(FlightShield().choose(1,metadata(.2,.2,.2,q1=.1,q0=.8)),0)

    def test_deadband_and_hysteresis(self):
        s = FlightShield()
        self.assertEqual(s.choose(0,metadata(q1=.82,q0=.8)),0)
        self.assertEqual(s.choose(1,metadata(q1=.82,q0=.8)),1)
        self.assertEqual(s.choose(0,metadata(q1=.95,q0=.8)),1)
        self.assertEqual(s.choose(0,metadata(q1=.84,q0=.8)),1)
        self.assertEqual(s.choose(0,metadata(q1=.82,q0=.8)),0)

    def test_mode_change_and_invalid_metadata_reset_hysteresis(self):
        s = FlightShield()
        s.choose(0,metadata(q1=.95,q0=.8))
        self.assertEqual(s.choose(0,metadata(q1=.84,q0=.8,mode='wave')),0)
        self.assertEqual(s.choose(1,None),1)
        self.assertEqual(s.choose(0,metadata(q1=math.nan)),0)
        self.assertEqual(s.choose(1,{'mode':'ship'}),1)

    def test_does_not_postpone_a_rescue_because_next_step_press_is_good(self):
        s = FlightShield()
        self.assertEqual(s.choose(0,metadata(q1=.9,q0=.6,q01=.9)),1)
        self.assertEqual(s.choose(0,metadata(wait=.8,next_press=1.,q1=.6,q0=.9)),1)

    def test_braking_hint_cannot_override_collision_priority(self):
        m = metadata(press=.1)
        m['recovery'] = {'action':1}
        self.assertEqual(FlightShield().choose(1,m),0)
        m = metadata(q1=.1,q0=.9)
        m['recovery'] = {'action':1}
        self.assertEqual(FlightShield().choose(0,m),0)


class FlightPhysicsTests(unittest.TestCase):
    def test_visible_frame_limit_stops_blind_climb_without_inventing_ceiling_collision(self):
        r=robust_search(Body(0.,2.,1,False,'ship'),[],[],None,None,10.,Phys(),.04,1,[True],
                        flight_buffer=.75,flight_view=(-9.,.8),want_path=True)
        self.assertTrue(all(r[k]<1. for k in ('1','00','01')))
        self.assertTrue(all(r['flight']['visibility_limited'].values()))
        self.assertTrue(r['path_unknown'])
        self.assertFalse(r['path_hit'])
        self.assertLessEqual(max(y for x,y in r['path']),.3+1e-8)
        self.assertEqual(FlightShield().choose(1,r['flight']),0)
        self.assertNotEqual(r['flight']['preferred_root'],'1')
        self.assertEqual(r['flight']['visibility_bounds'],(-9.,.8))

    def test_recorded_hanging_block_does_not_offer_offscreen_escape(self):
        import json
        from pathlib import Path
        from audit_ship_planner import replay_step
        step=json.loads((Path(__file__).parent/'test_fixtures/ship_ceiling_39pct.json').read_text())
        result=replay_step(step)
        self.assertEqual(step['search']['1'],1.)
        self.assertLess(result['1'],.85)
        self.assertEqual(result['00'],1.)
        self.assertEqual(FlightShield().choose(1,result['flight']),0)
        self.assertEqual(result['flight']['preferred_root'],'00')

    def test_ship_velocity_uses_capture_interval_not_smoothed_plan_interval(self):
        from env_boxes import physics_features
        from detect import Entities,Entity
        from gd_physics import ModeEstimator
        est=ModeEstimator();est.mode='ship';est.lag=1;est._actions.extend([True,False])
        ents=Entities([Entity(7,.36,.5,.08,.1,1.)]);p=Phys(ship_up=40.)
        out={}
        physics_features(ents,10.,est,p,2,2.,False,False,None,None,
                         dec_dt=.08,motion_dt=.04,search=False,out=out)
        self.assertAlmostEqual(out['inputs']['vy0'],2.8)
        self.assertAlmostEqual(out['inputs']['decision_dt'],.08)
        self.assertAlmostEqual(out['inputs']['motion_dt'],.04)

    def test_ship_speed_cap_displacement_matches_piecewise_motion(self):
        from gd_physics import advance
        p=Phys(ship_up=40.,ship_down=30.,ship_vmax=11.)
        for gravity in (-1,1):
            b=Body(0.,11.*gravity,gravity,False,'ship')
            advance(b,True,.1,10.,p,([],[]))
            self.assertAlmostEqual(b.y,1.1*gravity)
            b=Body(0.,10.*gravity,gravity,False,'ship')
            advance(b,True,.1,10.,p,([],[]))
            self.assertAlmostEqual(b.y,(10*.025+.5*40*.025**2+11*.075)*gravity)
            b=Body(0.,-11.*gravity,gravity,False,'ship')
            advance(b,False,.1,10.,p,([],[]))
            self.assertAlmostEqual(b.y,-1.1*gravity)

    def test_constant_hold_brain_is_kept_visible_in_empty_room(self):
        from gd_physics import advance
        physical=Body(0.,0.,1,False,'ship');p=Phys();control=FlightShield();pending=False
        heights=[]
        for _ in range(40):
            result=robust_search(Body(0.,physical.vy,1,False,'ship'),[],[],None,None,10.,p,.04,1,[pending],
                                 flight_buffer=.75,flight_view=(-8.-physical.y,.8-physical.y))
            selected=control.choose(1,result['flight'])
            advance(physical,pending,.04,10.,p,([],[]))
            pending=bool(selected);heights.append(physical.y)
        self.assertLess(max(heights),.3)
        self.assertGreater(min(heights),-7.5)
    def test_fast_fall_requires_more_clearance_than_rest(self):
        p = Phys()
        def score(v):
            return path_clearance(0,1.5,v,'ship',1,False,[],[],0.,None,10.,p,.5,.75,.04)[0]
        self.assertGreater(score(0.),score(-8.))
        a = path_clearance(0,1.5,-8,'ship',1,False,[],[],0.,None,10.,p,.5,.75,.04)
        b = path_clearance(0,-1.5,8,'ship',-1,False,[],[],None,0.,10.,p,.5,.75,.04)
        self.assertEqual(a,b)

    def test_wave_recovery_is_not_ship_acceleration(self):
        def score(up):
            p = Phys(ship_up=up)
            return path_clearance(0,1.,-8.,'wave',1,False,[],[],0.,None,10.,p,.5,.75,.04)
        self.assertEqual(score(20.),score(120.))

    def test_thin_obstacle_cannot_be_tunnelled_by_wave(self):
        hazard = [(.45,.55,-.1,.1)]
        self.assertTrue(flight_segment_hit(0.,1.,1.,-1.,hazard,[],.1,1))
        self.assertFalse(flight_segment_hit(0.,1.,1.,1.,hazard,[],.1,1))
        # Exact tangency is not a collision.
        self.assertFalse(flight_segment_hit(0.,.2,1.,.2,hazard,[],.1,1))

    def test_pending_commands_are_projected_with_fractional_delay(self):
        b = Body(0.,-5.,1,False,'ship')
        p = Phys()
        hint = recovery_hint(b,[],[],-.8,5.,10.,p,.5,.75,.04,[False,True],delay=.06)
        self.assertIsNotNone(hint)
        self.assertAlmostEqual(hint['queued_seconds'],.06)
        self.assertEqual((b.y,b.vy),(0.,-5.))

    def test_planner_and_controller_recover_from_bottom_and_top(self):
        for mode,vy,floor,ceiling,queued,brain,expected in (
            ('ship',-8.,-1.5,4.,False,0,1),('ship',8.,-4.,1.5,True,1,0),
            ('wave',-10.,-1.5,4.,False,0,1),('wave',10.,-4.,1.5,True,1,0)):
            r = robust_search(Body(0.,vy,1,False,mode),[],[],floor,ceiling,10.,Phys(),.04,1,[queued],
                              flight_buffer=.75,want_path=True)
            self.assertEqual(FlightShield().choose(brain,r['flight']),expected,(mode,r))
            self.assertTrue(r['path'])
            self.assertFalse(r['path_hit'])

    def test_empty_space_leaves_brain_in_control_and_zero_buffer_is_legacy(self):
        r = robust_search(Body(0.,0.,1,False,'ship'),[],[],None,None,10.,Phys(),.04,0,[],flight_buffer=.75)
        self.assertEqual(FlightShield().choose(0,r['flight']),0)
        self.assertEqual(FlightShield().choose(1,r['flight']),1)
        legacy = robust_search(Body(0.,0.,1,False,'ship'),[],[],-2.,2.,10.,Phys(),.04,0,[],flight_buffer=0.)
        self.assertNotIn('flight',legacy)

    def test_narrow_corridor_does_not_turn_soft_buffer_into_collision(self):
        r = robust_search(Body(0.,0.,1,False,'ship'),[],[],-.65,.65,10.,Phys(),.04,0,[],flight_buffer=.75)
        self.assertTrue(any(r[k] == 1. for k in ('1','00','01')))
        self.assertIn(FlightShield().choose(0,r['flight']),(0,1))

    def test_cube_predictions_unchanged_by_flight_buffer(self):
        args = (Body(),[(3.,3.5,-.5,.3)],[],-.5,None,10.,Phys(),.04,1,[False])
        a,b = robust_search(*args,flight_buffer=0.),robust_search(*args,flight_buffer=.75)
        self.assertEqual(a,b)

    def test_buffered_planner_mirrors_normal_and_upside_down_flight(self):
        for mode in ('ship','wave'):
            for mini in (False,True):
                a = robust_search(Body(0.,-5.,1,False,mode,mini),[(2.,4.,-2.,-1.)],[],
                                  -2.5,3.,10.,Phys(),.04,1,[False],flight_buffer=.75)
                b = robust_search(Body(0.,5.,-1,False,mode,mini),[(2.,4.,1.,2.)],[],
                                  -3.,2.5,10.,Phys(),.04,1,[False],flight_buffer=.75)
                for k in ('1','00','01'):
                    self.assertAlmostEqual(a[k],b[k],places=8)
                    self.assertAlmostEqual(a['flight']['clearance'][k],b['flight']['clearance'][k],places=8)


class FlightPolicyTests(unittest.TestCase):
    def test_teacher_labels_near_failed_prediction_are_revoked_after_death_trim(self):
        from pathlib import Path
        import tempfile
        from types import SimpleNamespace
        from trainer import Trainer,TrainConfig,_Seg
        captured = {}
        def push(*args,**kw):
            captured['labels'] = kw['teacher'][:]
            return len(args[1])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            env = SimpleNamespace(_step_dt=lambda **kw: .04)
            tr = Trainer(env,TrainConfig(log_dir=root,save_dir=root,route=False,verbose=0),
                         learner=SimpleNamespace(add_segment=push))
            seg = _Seg(obs=[np.zeros(1)]*70,act=[1]*70,rew=[.1]*70,teacher=[1]*70)
            tr._finish_attempt(seg,{'death_lag':5},.2,np.zeros(1),tag='frontier')
            tr._csv.close()
            self.assertEqual(len(captured['labels']),65)
            self.assertEqual(captured['labels'][-38:],[-1]*38)
            self.assertTrue(all(v == 1 for v in captured['labels'][:-38]))

    def test_trainer_wires_controller_and_coaching_with_confidence_gates(self):
        from pathlib import Path
        from types import SimpleNamespace
        import tempfile
        from trainer import Trainer,TrainConfig
        from env_boxes import OBS_DIM,OFF_MODE,OFF_SEARCH
        from gd_physics import MODE_ID
        obs = np.zeros(OBS_DIM,np.float32); obs[3] = 1.
        obs[OFF_MODE+MODE_ID['ship']] = 1.
        obs[OFF_SEARCH:OFF_SEARCH+4] = [1.,.2,.2,1.]
        env = SimpleNamespace(_last_search={'flight':metadata(wait=.2,next_press=.2)},
                              mode_cal=SimpleNamespace(counts=lambda:{k:3 for k in ('ship_up','ship_down','ship_vmax')}),
                              mode_est=SimpleNamespace(lag_locked=True),_calibration_valid=True)
        learner = SimpleNamespace(act=lambda *a,**kw:(0,{'q':np.array([1.,0.])}))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            tr = Trainer(env,TrainConfig(log_dir=root,save_dir=root,route=False,verbose=0),learner=learner)
            action,extra = tr.act(obs)
            self.assertEqual(action,1);self.assertEqual(extra['flight_teacher'],1)
            env._calibration_valid = False
            self.assertEqual(tr.act(obs)[1]['flight_teacher'],-1)
            env._last_search = {'flight':{'mode':'ship'}}
            # Malformed flight metadata must not cancel the ordinary collision veto.
            self.assertEqual(tr.act(obs)[0],1)
            tr._csv.close()

    def test_mode_specific_action_margin(self):
        from trainer import DQNLearner, TrainConfig
        from env_boxes import OBS_DIM,OFF_MODE
        from gd_physics import MODE_ID
        learner = object.__new__(DQNLearner)
        learner.cfg = TrainConfig(jump_margin=1.,flight_margin=0.)
        for mode,expected in (('cube',0),('ship',1),('wave',1),('ufo',0)):
            obs = np.zeros(OBS_DIM,np.float32); obs[OFF_MODE+MODE_ID[mode]] = 1.
            self.assertEqual(learner._greedy([0.,.4],obs),expected)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires training PyTorch environment')
    def test_actual_dqn_bootstrap_uses_same_flight_margin(self):
        import torch
        from trainer import DQNLearner,TrainConfig
        from env_boxes import OBS_DIM,OFF_MODE
        from gd_physics import MODE_ID
        for mode,expected_loss in (('cube',0.),('ship',.5),('wave',.5)):
            cfg = TrainConfig(jump_margin=1.,flight_margin=0.,net_arch=(4,),buffer_size=4,
                              batch_size=1,warmup=1,replay_ratio=1.,max_grad_steps=1)
            learner = DQNLearner(OBS_DIM,2,cfg)
            with torch.no_grad():
                for network in (learner.q,learner.qt):
                    for param in network.parameters(): param.zero_()
                learner.q.v.bias.fill_(.2)
                learner.q.a.bias.copy_(torch.tensor([0.,.4]))
                learner.qt.v.bias.fill_(1.)
                learner.qt.a.bias.copy_(torch.tensor([0.,2.]))
            learner.b_nxt[0,OFF_MODE+MODE_ID[mode]] = 1.
            learner.b_disc[0] = .5
            learner.ptr = 1
            stats = learner.train(1)
            self.assertAlmostEqual(stats['loss'],expected_loss,places=5)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires training PyTorch environment')
    def test_coaching_moves_real_network_toward_safe_action_and_ignores_cube(self):
        import torch
        from trainer import DQNLearner,TrainConfig
        from env_boxes import OBS_DIM,OFF_MODE
        from gd_physics import MODE_ID
        for mode,improves in (('ship',True),('wave',True),('cube',False)):
            cfg = TrainConfig(net_arch=(4,),buffer_size=4,batch_size=1,warmup=1,replay_ratio=1.,max_grad_steps=1)
            lr = DQNLearner(OBS_DIM,2,cfg)
            with torch.no_grad():
                for net in (lr.q,lr.qt):
                    for param in net.parameters(): param.zero_()
                lr.q.v.bias.fill_(.2);lr.q.a.bias.copy_(torch.tensor([.4,0.]))
            obs = np.zeros(OBS_DIM,np.float32);obs[OFF_MODE+MODE_ID[mode]] = 1.
            # Match the float32 TD target exactly so only coaching can move Q.
            lr.add_segment([obs],[0],[float(lr.q_values(obs)[0])],obs,True,teacher=[1])
            before = lr.q_values(obs)[1]-lr.q_values(obs)[0]
            stats = lr.train(1)
            after = lr.q_values(obs)[1]-lr.q_values(obs)[0]
            self.assertEqual(stats['teacher_draws']>0,improves)
            if improves: self.assertGreater(after,before)
            else: self.assertAlmostEqual(after,before)


if __name__ == '__main__': unittest.main()
