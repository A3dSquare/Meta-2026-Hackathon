import unittest
from types import SimpleNamespace
from cube_motion import fit_arc
from death_cause import trace_decisions, classify
from env_boxes import EntityBoxEnv
from gd_physics import Phys


class TakeoffEvidenceTests(unittest.TestCase):
    def test_stationary_or_small_rise_cannot_invent_full_jump(self):
        for g in (1,-1):
            for rise in (-.002,.0,.067,.116):
                result=fit_arc([(0.,0.),(.033,rise*g)],90*g,19.55*g,0.,.033)
                self.assertTrue(result is None or result[1]*g<8.,result)

    def test_observed_jump_and_walkoff_remain_supported(self):
        for g in (1,-1):
            for speed in (0.,20.):
                points=[(t,g*(speed*t-45*t*t)) for t in (0.,.033,.066)]
                result=fit_arc(points,90*g,speed*g,0.,.033)
                self.assertIsNotNone(result)
                self.assertAlmostEqual(result[1],g*(speed-90*.066),places=5)

    def test_invalid_supported_frame_does_not_seed_launch(self):
        env=EntityBoxEnv.__new__(EntityBoxEnv)
        env.mode_est=SimpleNamespace(mode='cube',g=1,mini=False)
        env.phys=Phys();env._last_yb=1.;env._motion_dt=lambda:.033
        env._calibration_valid=False;env._air=[(0.,0.),(.033,1.)]
        self.assertIsNone(env._cube_air_fit(SimpleNamespace(conf=1.),True))
        self.assertEqual(env._air,[])
        self.assertFalse(env._air_from_rest)


class DecisionTraceTests(unittest.TestCase):
    def row(self,verified,a=0):
        return dict(player=True,dt=.04,a=a,search={'00':1.,'01':1.,'1':.1},
                    route_plan=dict(verified=verified,first_action=0))

    def test_final_loss_of_verification_uses_elapsed_time(self):
        rows=[self.row(True),self.row(False),self.row(True),self.row(False),self.row(False)]
        rows[-1]['motion_dt']=.08
        trace=trace_decisions(rows,4)
        self.assertEqual(trace['last_verified']['index'],2)
        self.assertEqual(trace['first_after_verified']['seconds_before_crash'],.08)
        self.assertIsNone(trace['last_action_with_safer_scored_alternative'])

    def test_unknown_history_and_alternative_are_explicit(self):
        rows=[self.row(False,1),dict(player=False,dt=.04)]
        trace=trace_decisions(rows,1)
        self.assertTrue(trace['history_limited'])
        self.assertEqual(trace['last_action_with_safer_scored_alternative']['index'],0)
        self.assertIsNone(trace['last_verified'])

    def test_later_trap_is_not_mislabeled_as_wrong_choice(self):
        rows=[self.row(True) for _ in range(12)]
        rows[-1]['search']={'00':.1,'01':.1,'1':.1}
        self.assertEqual(classify(rows,0)['why'],'forecast became unsafe')


if __name__=='__main__':unittest.main()
