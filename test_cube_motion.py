import unittest
from types import SimpleNamespace
import numpy as np
from cube_motion import fit_arc,CubePhysicsLearner
from gd_physics import Phys,Body,advance,robust_search
from env_boxes import EntityBoxEnv


class CubeTrackingTests(unittest.TestCase):
    def test_takeoff_into_low_ceiling_is_fatal_in_both_gravity_directions(self):
        for mode in ('cube','robot'):
            for gravity in (1,-1):
                for gap in (0.,.1,.25):
                    body=Body(0.,0.,gravity,True,mode)
                    surfaces=([0.],[gap]) if gravity==1 else ([-gap],[0.])
                    advance(body,True,.033,10.,Phys(),surfaces)
                    self.assertTrue(body.crashed,(mode,gravity,gap))

    def test_sliding_under_low_ceiling_does_not_kill_without_jumping(self):
        for gravity in (1,-1):
            body=Body(0.,0.,gravity,True,'cube')
            surfaces=([0.],[0.])
            advance(body,False,.033,10.,Phys(),surfaces)
            self.assertFalse(body.crashed)

    def test_planner_releases_under_low_platform_instead_of_certifying_a_jump(self):
        for gravity in (1,-1):
            solids=[(-1.,3.,.6,1.)] if gravity==1 else [(-1.,3.,-1.,-.6)]
            floor,ceil=(-.5,None) if gravity==1 else (None,.5)
            result=robust_search(Body(0.,0.,gravity,True,'cube'),[],solids,floor,ceil,10.,Phys(),.04,
                lag=0,pending=[],horizon=.6,want_path=True,explicit_plan=True,flight_view=(-5.,7.))
            self.assertLess(result['1'],1.)
            self.assertEqual(result['plan']['first_action'],0)

    def test_recorded_cube_jump_does_not_keep_previous_falling_velocity(self):
        import json
        from pathlib import Path
        from audit_cube_motion import replay
        data=json.loads((Path(__file__).parent/'test_fixtures/cube_stale_fall.json').read_text())
        rows=replay(data['steps']);row=next(r for r in rows if r['index']==data['index'])
        self.assertGreater(row['raw_velocity'],8.)
        self.assertLess(row['old_velocity'],-3.)
        self.assertGreater(row['new_velocity'],3.)
        self.assertLessEqual(abs(row['new_offset']),.2)
    def test_old_fall_cannot_override_observed_launch(self):
        points=[(0.,2.8),(.05,2.5),(.1,2.3),(.15,3.1)]
        fit=fit_arc(points,95.,20.)
        self.assertTrue(fit is None or fit[1]>0.)

    def test_takeoff_window_does_not_move_as_the_history_rolls(self):
        physics=Phys(cube_v0=20.,cube_g=95.)
        for start in (0.,3.5):
            points=[(t,20*(t-start)-.5*95*(t-start)**2) for t in np.linspace(start+.1,start+.4,8)]
            # Shift to the real latest contact, not the first retained frame.
            points=[(t-start,y) for t,y in points]
            fit=fit_arc(points,95.,20.,rest=0.,launch_window=.04)
            self.assertIsNotNone(fit);self.assertAlmostEqual(fit[1],-18.,places=6)

    def test_late_platform_bounce_rebases_time_and_uses_real_support(self):
        env=EntityBoxEnv.__new__(EntityBoxEnv)
        env.mode_est=SimpleNamespace(mode='cube',g=1,mini=False)
        env.phys=Phys(cube_v0=20.,cube_g=95.);env.route_controller=False
        env._air=[(1.1,2.0),(1.15,1.5)];env._cube_key=('cube',1,False)
        env._air_from_rest=False;env._rest_true=None;env._cube_supports=[1.5]
        env._calibration_valid=True;env._last_yb=1.5+20*.05-.5*95*.05**2
        env._motion_dt=lambda:.05
        fit=env._air_fit(SimpleNamespace(conf=1.),False)
        self.assertIsNotNone(fit);self.assertGreater(fit[1],10.)
        self.assertLessEqual(abs(fit[0]),.2)

    def test_fits_cannot_move_the_player_more_than_point_two_blocks(self):
        points=[(i*.04,20*i*.04-.5*95*(i*.04)**2) for i in range(8)]
        points[-1]=(points[-1][0],points[-1][1]+.7)
        fit=fit_arc(points,95.,20.,rest=0.,launch_window=.04)
        self.assertTrue(fit is None or abs(fit[0]-points[-1][1])<=.2)


class CubePhysicsTests(unittest.TestCase):
    def arc(self,learner,physics,gravity=1,mini=False,noise=0.):
        learner.observe(0.,.04,True,gravity,mini,True,physics,rest=0.)
        scale=physics.mini if mini else 1.
        for i in range(1,10):
            t=i*.04;y=gravity*(19.5*scale*t-.5*90*t*t)+noise*((i%2)*2-1)
            learner.observe(y,.04,False,gravity,mini,True,physics)

    def test_partial_arcs_learn_gravity_without_waiting_for_original_floor(self):
        physics=Phys(cube_g=108.,cube_v0=22.);learner=CubePhysicsLearner()
        for _ in range(20):self.arc(learner,physics)
        self.assertGreaterEqual(learner.accepted,3)
        self.assertLess(abs(physics.cube_g-90.),abs(108.-90.))
        self.assertLess(abs(physics.cube_v0-19.5),abs(22.-19.5))
        self.assertGreater(learner.updates,0)

    def test_one_arc_does_not_change_parameters_and_invalid_frames_do_not_vote(self):
        physics=Phys(cube_g=108.,cube_v0=22.);learner=CubePhysicsLearner()
        self.arc(learner,physics)
        self.assertEqual((physics.cube_g,physics.cube_v0),(108.,22.))
        count=learner.accepted
        for i in range(10):learner.observe(float(i),.04,False,1,False,False,physics)
        self.assertEqual(learner.accepted,count)

    def test_reacquisition_cannot_count_the_same_jump_as_a_new_calibration_arc(self):
        physics=Phys(cube_g=108.,cube_v0=22.);learner=CubePhysicsLearner()
        self.arc(learner,physics);count=learner.accepted
        learner.observe(None,.04,False,1,False,False,physics)
        for i in range(1,10):
            t=i*.04;learner.observe(4.+3*t-.5*90*t*t,.04,False,1,False,True,physics)
        self.assertEqual(learner.accepted,count)
        self.arc(learner,physics);self.assertGreater(learner.accepted,count)

    def test_upside_down_mini_arcs_use_same_gravity_and_normalized_impulse(self):
        physics=Phys(cube_g=108.,cube_v0=22.);learner=CubePhysicsLearner()
        for _ in range(20):self.arc(learner,physics,-1,True)
        self.assertLess(abs(physics.cube_g-90.),18.)
        self.assertTrue(all(14.<=a['speed']<=26. for a in learner.arcs if a['speed'] is not None))


if __name__=='__main__':unittest.main()
