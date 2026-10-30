"""Route semantics, measured action effects, replay isolation and deployment gates."""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch

from route_controller import (BalancedReplay,CONTROLLER_DIM,HEADER,HORIZON,TARGET_START,
                              ANALYTIC_START,RouteObservationEnv)
from route_controller import CONTROLLER_MODE_START,MODES as CONTROL_MODES
from route_learner import RouteLearner
from trainer import TrainConfig,DQNLearner
from env_boxes import OBS_DIM,OFF_MODE,OFF_SEARCH
from gd_physics import Body,Phys,MODES,advance,robust_search


def observation(mode=0):
    o=np.zeros(CONTROLLER_DIM,np.float32)
    o[3]=1.;o[OFF_MODE+(1 if mode==7 else mode)]=1.;o[OFF_SEARCH:OFF_SEARCH+4]=[1.,.2,.2,1.]
    o[CONTROLLER_MODE_START+mode]=1.
    o[OBS_DIM:OBS_DIM+14]=[0.,0.,.04,.04,0.,0.,1.,0.,1.,1.,1.,0.,.1,.1]
    o[TARGET_START:ANALYTIC_START]=np.tile([0.,0.,0.,1.],5)
    return o


def row(action=1,mode=0,teacher=1):
    o=observation(mode)
    return dict(obs=o,nxt=o.copy(),action=action,ret=.1,discount=.98,mode=mode,success=bool(action),
        teacher=teacher,actions=np.full(HORIZON,action,np.float32),durations=np.full(HORIZON,.04,np.float32),
        outcomes=np.zeros((HORIZON,2),np.float32),mask=np.ones(HORIZON,np.float32),episode=5)


def config(**kw):
    return TrainConfig(route_controller=True,buffer_size=512,net_arch=(32,32),warmup=8,batch_size=16,
                       replay_ratio=1.,max_grad_steps=32,controller_eval_min=8,**kw)


class RoutePlanTests(unittest.TestCase):
    def test_scalar_route_lookup_matches_numpy_at_boundaries_and_duplicate_times(self):
        import json
        from route_controller import route_point
        cases=json.loads((Path(__file__).parent/'test_fixtures/planner_timing_cases.json').read_text())
        plans=[c['expected']['plan'] for c in cases]
        plans.append(dict(trajectory=[dict(t=t,x=x,y=x*x,vy=-x) for t,x in [(0.,0.),(.1,1.),(.1,2.),(.2,3.)]]))
        for plan in plans:
            points=plan['trajectory'];times=[p['t'] for p in points]
            for t in [times[0]-.1,*times,*np.linspace(times[0],times[-1],17),times[-1]+.5e-6]:
                expected=[np.interp(t,times,[p[k] for p in points]) for k in ('x','y','vy')]
                np.testing.assert_allclose(route_point(plan,t),expected,rtol=0.,atol=1e-12)
            self.assertIsNone(route_point(plan,times[-1]+2e-6))

    def test_geometry_cache_matches_uncached_rollouts_through_portals_for_all_modes(self):
        from benchmark_planner import equivalent
        from gd_physics import search_best
        for mode in CONTROL_MODES:
            for gravity in (1,-1):
                body=Body(1.,3.*gravity,gravity,False,mode)
                args=(body,[(7.,8.,0.,1.)],[(4.,5.,0.,1.)],-.5,6.,10.,Phys(),.04)
                kw=dict(lag=1,pending=[False],want_path=True,flight_target=.5,flight_view=(-5.,7.),
                        portals=[(2.,2.5,-5.,7.,'speed',12.9),(6.,6.5,-5.,7.,'mode','ship')],
                        ground_clearance=mode not in ('ship','wave','swing'))
                with self.subTest(mode=mode,gravity=gravity):
                    self.assertTrue(equivalent(search_best(*args,**kw),search_best(*args,**kw,work_cache={})))
    def test_optimized_planner_preserves_recorded_outputs_and_scopes_cache_to_frame(self):
        import json
        from benchmark_planner import run_case,equivalent
        cases=json.loads((Path(__file__).parent/'test_fixtures/planner_timing_cases.json').read_text())
        # Different frames/geometry back to back also guard against retaining
        # cached clearance or input sequences from a previous observation.
        for i,case in enumerate(cases):
            with self.subTest(case=i):
                self.assertTrue(equivalent(json.loads(json.dumps(run_case(case))),case['expected']))
    def test_recorded_landing_can_fail_between_safe_delay_endpoints(self):
        import json
        from detect import Entities,Entity
        from env_boxes import preview_boxes,special_boxes
        from gd_physics import search_best,SEARCH_KEYS
        d=json.loads((Path(__file__).parent/'test_fixtures/landing_delay_gap.json').read_text())
        q=d['preview_inputs'];e=Entities([Entity(*b[:5],pt=b[5]) for b in d['boxes']]);h,s,_=preview_boxes(e,20.)
        body=Body(q['y0'],q['vy0'],d['g'],q['grounded'],d['mode'],d['mini'],prev_hold=q['eff_prev'])
        p=Phys(cube_v0=q['cube_v0'],cube_g=q['cube_g']);old=d['route_plan'];key=old['root']
        schedule={c['decision']:c['hold'] for c in old['controls']}
        def at_delay(seconds):
            return search_best(body,h,s,q['floor_rel'],q['ceil_rel'],d['speed'],p,q['decision_dt'],q['lag'],
                q['pending_long'],delay=seconds,root_action=SEARCH_KEYS.index(key),action_schedule=schedule,
                specials=special_boxes(e))[key]
        self.assertEqual(at_delay(q['decision_dt']),1.)
        self.assertEqual(at_delay(2*q['decision_dt']),1.)
        self.assertLess(at_delay(1.25*q['decision_dt']),1.)
        result=robust_search(body,h,s,q['floor_rel'],q['ceil_rel'],d['speed'],p,q['decision_dt'],q['lag'],q['pending'],
            pending_long=q['pending_long'],lag_alt=q['lag_alt'],want_path=True,explicit_plan=True,
            specials=special_boxes(e),flight_view=q['flight_view'])
        if result['plan']['verified']:
            self.assertEqual(len(result['plan']['delay_checks']),9)
            self.assertTrue(all(c['survival']==1. for c in result['plan']['delay_checks']))
        else:self.assertLess(result['plan']['sequence_safety']['worse'],1.)

    def test_earlier_alternative_does_not_drop_the_later_delay_bound(self):
        result=robust_search(Body(0.,0.,1,True,'cube'),[],[],-.5,None,10.,Phys(),.04,1,[False],
            pending_long=[False,False],lag_alt=0,want_path=True,explicit_plan=True,flight_view=(-5.,7.))
        self.assertTrue(result['plan']['verified'])
        delays=[c['delay'] for c in result['plan']['delay_checks']]
        self.assertAlmostEqual(min(delays),0.)
        self.assertAlmostEqual(max(delays),.08)

    def test_recorded_23pct_route_requires_the_same_future_commands_in_both_worlds(self):
        import json
        from detect import Entities,Entity
        from env_boxes import preview_boxes,special_boxes
        from gd_physics import search_best,SEARCH_KEYS
        d=json.loads((Path(__file__).parent/'test_fixtures/route_sequence_23pct.json').read_text())
        q=d['preview_inputs'];e=Entities([Entity(*box[:5],pt=box[5]) for box in d['boxes']])
        h,s,_=preview_boxes(e,20.);p=Phys(cube_v0=q['cube_v0'],cube_g=q['cube_g'])
        body=Body(q['y0'],q['vy0'],d['g'],q['grounded'],d['mode'],d['mini'],prev_hold=q['eff_prev'])
        old=d['route_plan'];schedule={c['decision']:c['hold'] for c in old['controls']}
        # The old route claimed full safety even though its own command
        # sequence hits a hazard with the very delay used by the robust check.
        self.assertTrue(old['verified'])
        failed=search_best(body,h,s,q['floor_rel'],q['ceil_rel'],d['speed']*1.04,p,q['decision_dt'],
            q['lag_alt'],q['pending_long'],root_action=SEARCH_KEYS.index(old['root']),action_schedule=schedule,
            specials=special_boxes(e))
        self.assertLess(failed[old['root']],1.)
        result=robust_search(body,h,s,q['floor_rel'],q['ceil_rel'],d['speed'],p,q['decision_dt'],q['lag'],q['pending'],
            pending_long=q['pending_long'],lag_alt=q['lag_alt'],want_path=True,explicit_plan=True,
            specials=special_boxes(e),flight_view=q['flight_view'])
        self.assertFalse(result['plan']['verified'])
        self.assertLess(result['plan']['sequence_safety']['worse'],1.)
        self.assertEqual(result['plan']['first_action'],int(result['plan']['controls'][0]['hold']))

    def test_fixed_command_sequence_can_still_certify_a_safe_route(self):
        result=robust_search(Body(0.,0.,1,True,'cube'),[],[],-.5,None,10.,Phys(),.04,1,[False],
            pending_long=[False,False],lag_alt=2,want_path=True,explicit_plan=True,flight_view=(-5.,7.))
        self.assertTrue(result['plan']['verified'])
        self.assertEqual(result['plan']['sequence_safety'],dict(nominal=1.,worse=1.))

    def test_explicit_input_sequence_reproduces_predicted_positions(self):
        for mode in ('cube','ship','wave','robot','ball','ufo','spider','swing'):
            b=Body(0.,0.,1,mode not in ('ship','wave','swing'),mode)
            r=robust_search(b,[],[],-.5,6.,10.,Phys(),.04,1,[False],horizon=.4,
                            want_path=True,explicit_plan=True,flight_buffer=.75 if mode in ('ship','wave') else 0.,
                            flight_view=(-5.,7.))
            plan=r['plan'];self.assertTrue(plan['trajectory']);self.assertTrue(plan['controls'])
            self.assertEqual(plan['controls'][0]['hold'],bool(plan['first_action']))
            physical=copy.copy(b);last=0.
            from gd_physics import surfaces_at
            for p in plan['trajectory'][1:]:
                advance(physical,p['hold'],p['t']-last,10.,Phys(),surfaces_at(p['x'],[],-.5,6.,.5))
                self.assertAlmostEqual(physical.y,p['y'],places=6)
                self.assertAlmostEqual(physical.vy,p['vy'],places=6)
                last=p['t']

    def test_recorded_platform_plans_cannot_mislabel_the_first_command(self):
        import json
        from detect import Entities,Entity
        from env_boxes import preview_boxes
        d=json.loads((Path(__file__).parent/'test_fixtures/search_timing_5pct.json').read_text())
        e=Entities([Entity(*box[:5],pt=box[5]) for box in d['boxes']]);h,s,_=preview_boxes(e,18.)
        p=d['preview_inputs'];r=robust_search(Body(p['y0'],p['vy0'],d['g'],p['grounded'],d['mode']),h,s,
            p['floor_rel'],p['ceil_rel'],d['speed'],Phys(cube_v0=p['cube_v0'],cube_g=p['cube_g']),
            d['dt'],p['lag'],p['pending'],want_path=True,explicit_plan=True)
        self.assertTrue(r['plan']['verified'])
        self.assertEqual(r['plan']['first_action'],0)
        self.assertFalse(r['plan']['controls'][0]['hold'])


class ReplayTests(unittest.TestCase):
    def test_returns_use_elapsed_time_and_completions_are_successes(self):
        learner=RouteLearner(CONTROLLER_DIM,2,config())
        states=[observation() for _ in range(4)]
        for o in states:o[OBS_DIM+3]=.05
        learner.add_segment(states[:-1],[0]*3,[.1]*3,states[-1],False)
        r=learner.replay.rows()[0]
        discount=learner.cfg.gamma**1.5
        self.assertAlmostEqual(r['ret'],.1*(1+discount+discount**2),places=6)
        self.assertAlmostEqual(r['discount'],discount**3,places=6)
        winner=RouteLearner(CONTROLLER_DIM,2,config())
        winner.add_segment(states[:-1],[0]*3,[.1,.1,50.],states[-1],True,completed=True)
        self.assertTrue(all(r['success'] for r in winner.replay.rows()))

    def test_teacher_requires_matching_observed_motion_after_delay(self):
        learner=RouteLearner(CONTROLLER_DIM,2,config())
        states=[observation() for _ in range(10)]
        plan=dict(trajectory=[dict(t=0.,x=0.,y=0.,vy=0.),dict(t=1.,x=10.,y=0.,vy=0.)],
                  controls=[dict(decision=j,hold=True,applies_at=j*.04) for j in range(25)])
        learner.add_segment(states[:-1],[1]*9,[.1]*9,states[-1],False,
                            teacher=[1]*9,route_records=[dict(plan=plan)]*9)
        self.assertGreater(sum(r['teacher']>=0 for r in learner.replay.rows()),0)
        bad=RouteLearner(CONTROLLER_DIM,2,config())
        states[3][0]=.2  # recorded motion disagrees by more than one block
        bad.add_segment(states[:-1],[1]*9,[.1]*9,states[-1],False,
                        teacher=[1]*9,route_records=[dict(plan=plan)]*9)
        first=bad.replay.rows()[0]
        self.assertEqual(first['teacher'],-1)

    def test_rare_modes_and_failures_survive_many_cube_successes(self):
        replay=BalancedReplay(320)
        replay.add(row(mode=1,action=0));replay.add(row(mode=1))
        for _ in range(5000):replay.add(row())
        self.assertEqual(len(replay.buckets[(1,False)]),1)
        self.assertEqual(len(replay.buckets[(1,True)]),1)
        sample=replay.sample(3000)
        ship=sum(r['mode']==1 for r in sample)
        self.assertGreater(ship,1700);self.assertLess(ship,2300)

    def test_validation_is_whole_episode_and_multi_frame_actions_are_actual(self):
        learner=RouteLearner(CONTROLLER_DIM,2,config())
        states=[observation() for _ in range(12)];actions=[i%2 for i in range(11)]
        for i,o in enumerate(states):o[0]=i*.01
        learner.episode_id=5
        learner.add_segment(states[:-1],actions,[.1]*11,states[-1],True)
        self.assertEqual(len(learner.replay),0);self.assertGreater(len(learner.validation),0)
        first=learner.validation.rows()[0]
        np.testing.assert_array_equal(first['actions'],actions[:5])
        np.testing.assert_allclose(first['outcomes'][:,0],[.01,.02,.03,.04,.05],atol=1e-7)
        learner.add_segment(states[:-1],actions,[.1]*11,states[-1],True)
        self.assertGreater(len(learner.replay),0)
        self.assertTrue(all(r['episode']==5 for r in learner.validation.rows()))

    def test_bad_observations_do_not_create_motion_targets_or_teachers(self):
        learner=RouteLearner(CONTROLLER_DIM,2,config());states=[observation() for _ in range(8)]
        states[1][OBS_DIM+10]=0.
        learner.add_segment(states[:-1],[1]*7,[.1]*7,states[-1],True,teacher=[1]*7)
        r=learner.replay.rows()[0]
        self.assertEqual(r['mask'].sum(),0.)
        self.assertEqual(r['teacher'],-1)


class LearningTests(unittest.TestCase):
    def learner(self):return RouteLearner(CONTROLLER_DIM,2,config())

    def test_failed_motion_validation_revokes_candidate_trust_not_deployed_trust(self):
        learner=self.learner()
        with torch.no_grad():
            for parameter in learner.candidate.parameters():parameter.zero_()
            learner.candidate.motion_heads[0].bias.fill_(1.)
            learner.candidate.motion_trust[0]=1.
            learner.target.motion_trust[0]=1.
            learner.q.motion_trust[0]=1.
        for i in range(32):learner.validation.add(row(i%2,teacher=i%2))
        report=learner.evaluate_candidate()
        self.assertEqual(float(learner.candidate.motion_trust[0]),0.)
        self.assertEqual(float(learner.target.motion_trust[0]),0.)
        self.assertEqual(float(learner.q.motion_trust[0]),1.)
        self.assertIn('motion validation',report['modes']['cube']['blockers'])

    def test_motion_batches_preserve_rare_commands_without_invalid_targets(self):
        learner=self.learner()
        for i in range(25):learner.replay.add(row(0,teacher=-1))
        learner.replay.add(row(1,teacher=-1))
        invalid=row(1,mode=1);invalid['mask'][:]=0;learner.replay.add(invalid)
        rows=learner._coaching_rows(200,learner._motion_buckets())
        self.assertGreater(sum(r['actions'][0]==1 for r in rows),60)
        self.assertTrue(all(np.any(r['mask']) and r['mode']==0 for r in rows))

    def test_old_analytic_replay_migrates_without_changing_weights(self):
        learner=self.learner();sample=row()
        sample['obs'][ANALYTIC_START:ANALYTIC_START+10:2]=np.arange(1,6)*.025
        learner.replay.add(sample)
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'brain.pt');learner.save(path)
            state=torch.load(path,weights_only=False);state.pop('analytic_velocity_version')
            torch.save(state,path)
            restored=self.learner();restored.load(path)
            migrated=restored.replay.rows()[0]['obs']
            np.testing.assert_allclose(migrated[ANALYTIC_START+1:ANALYTIC_START+10:2],.25,atol=1e-6)
            for name,value in learner.q.state_dict().items():
                self.assertTrue(torch.equal(value,restored.q.state_dict()[name]))
            self.assertEqual(len(restored.replay),1)

    def test_velocity_targets_come_from_positions_not_the_tracker_model(self):
        learner=self.learner();states=[observation() for _ in range(8)]
        for i,o in enumerate(states):
            o[0]=i*.02;o[OBS_DIM]=-1.  # deliberately wrong old fitted velocity
        learner.add_segment(states[:-1],[1]*7,[.1]*7,states[-1],False)
        first=learner.replay.rows()[0]
        np.testing.assert_allclose(first['outcomes'][:,1],.2,atol=1e-6)  # .16 blocks / .04s / 20

    def test_old_inconsistent_samples_are_filtered_with_backup_and_deployed_weights_preserved(self):
        from training_diagnostics import file_sha256
        learner=self.learner();good=row();bad=row()
        bad['outcomes'][:,0]=np.arange(1,6)*.1;bad['outcomes'][:,1]=-1.
        learner.replay.add(good);learner.replay.add(bad);learner.validation.add(copy.deepcopy(bad))
        before={k:v.clone() for k,v in learner.q.state_dict().items()}
        with torch.no_grad():learner.candidate.advantages[0].bias.add_(2.)
        learner.steps_seen=1234
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'controller.pt');learner.save(path)
            state=torch.load(path,weights_only=False);state.pop('motion_targets_version')
            torch.save(state,path);digest=file_sha256(path)
            restored=self.learner();restored.load(path)
            self.assertEqual(restored.steps_seen,1234)
            self.assertEqual(len(restored.replay),1);self.assertEqual(len(restored.validation),0)
            self.assertEqual(restored.data_migration['discarded_inconsistent'],2)
            self.assertTrue(all(torch.equal(v,restored.q.state_dict()[k]) for k,v in before.items()))
            self.assertTrue(all(torch.equal(v,restored.candidate.state_dict()[k]) for k,v in before.items()))
            self.assertEqual(file_sha256(path),digest)
            restored.save(path)
            self.assertEqual(file_sha256(path+'.before_cube_targets'),digest)
            self.assertEqual(torch.load(path,weights_only=False)['motion_targets_version'],2)

    def test_velocity_validation_still_rejects_wrong_velocity_with_correct_positions(self):
        learner=self.learner()
        with torch.no_grad():
            for parameter in learner.candidate.parameters():parameter.zero_()
            learner.candidate.advantages[0].bias[1]=1.
            learner.candidate.motion_heads[0].bias[1]=1.
        for i in range(32):learner.validation.add(row(i%2,teacher=1 if i%2 else -1))
        report=learner.evaluate_candidate()
        self.assertEqual(report['modes']['cube']['candidate']['position_rmse'],0.)
        self.assertGreater(report['modes']['cube']['candidate']['velocity_rmse'],4.)
        self.assertFalse(report.get('offline_passed',False))

    def test_shadow_training_changes_candidate_not_deployed_network(self):
        learner=self.learner()
        for i in range(32):learner.replay.add(row(i%2,teacher=1 if i%2 else -1))
        before={k:v.clone() for k,v in learner.q.state_dict().items()}
        candidate={k:v.clone() for k,v in learner.candidate.state_dict().items()}
        result=learner.train(16)
        self.assertFalse(result['promoted'])
        self.assertTrue(all(torch.equal(v,learner.q.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(any(not torch.equal(v,learner.candidate.state_dict()[k]) for k,v in candidate.items()))

    def test_held_out_safety_gate_promotes_good_candidate_and_rejects_regression(self):
        learner=self.learner()
        with torch.no_grad():
            for net in (learner.q,learner.candidate):
                for parameter in net.parameters():parameter.zero_()
            learner.q.advantages[0].bias.copy_(torch.tensor([1.,0.]))
            learner.candidate.trunk[0].weight[0,0]=1.
            learner.candidate.trunk[2].weight[0,0]=1.
            learner.candidate.advantages[0].bias[0]=.5
            learner.candidate.advantages[0].weight[1,0]=2.
        for i in range(32):
            sample=row(i%2,teacher=i%2);sample['obs'][0]=i%2
            sample['obs'][OFF_SEARCH:OFF_SEARCH+4]=[1.,1.,1.,1.]
            learner.validation.add(sample)
        report=learner.evaluate_candidate()
        self.assertTrue(report['offline_passed']);self.assertFalse(report['promoted'])
        self.assertEqual(learner.trusted_modes,set())
        learner.q.load_state_dict(learner.candidate.state_dict());learner.trusted_modes={0}
        before={k:v.clone() for k,v in learner.q.state_dict().items()}
        with torch.no_grad():
            learner.candidate.advantages[0].weight.zero_()
            learner.candidate.advantages[0].bias.copy_(torch.tensor([1.,0.]))
        self.assertFalse(learner.evaluate_candidate().get('offline_passed',False))
        self.assertTrue(all(torch.equal(v,learner.q.state_dict()[k]) for k,v in before.items()))

    def test_wrong_motion_blocks_promotion_even_with_perfect_demonstration_agreement(self):
        learner=self.learner()
        with torch.no_grad():
            for parameter in learner.candidate.parameters():parameter.zero_()
            learner.candidate.advantages[0].bias[1]=1.
            learner.candidate.motion_heads[0].bias.fill_(1.)
        for i in range(32):learner.validation.add(row(i%2,teacher=1 if i%2 else -1))
        report=learner.evaluate_candidate()
        self.assertFalse(report['promoted'])
        self.assertFalse(report.get('offline_passed',False))
        self.assertGreater(report['modes']['cube']['candidate']['position_rmse'],.3)

    def test_controller_heads_are_distinct_and_motion_depends_on_future_inputs(self):
        learner=self.learner();net=learner.candidate
        o=torch.from_numpy(observation()[None]);d=torch.full((1,HORIZON),.04)
        a=net.dynamics(o,torch.zeros_like(d),d);b=net.dynamics(o,torch.ones_like(d),d)
        self.assertFalse(torch.equal(a,b))
        ship_before=net.advantages[1].weight.detach().clone()
        loss=net(o).sum();loss.backward();learner.opt.step()
        self.assertTrue(torch.equal(ship_before,net.advantages[1].weight))
        self.assertEqual(len(net.advantages),8)
        self.assertEqual(int(net.modes(torch.from_numpy(observation(7)[None]))[0]),7)

    def test_checkpoint_restores_replay_candidate_optimizer_and_exploration(self):
        learner=self.learner()
        for i in range(32):learner.replay.add(row(i%2));learner.validation.add(row(i%2))
        learner.train(4);learner.steps_seen=777;learner.episode_id=16
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'controller.pt');learner.save(path)
            restored=self.learner();restored.load(path)
            self.assertEqual((restored.steps_seen,restored.episode_id),(777,16))
            self.assertEqual(len(restored.replay),len(learner.replay))
            self.assertEqual(len(restored.validation),len(learner.validation))
            self.assertEqual(restored.epsilon(),learner.epsilon())
            self.assertTrue(all(torch.equal(v,restored.candidate.state_dict()[k]) for k,v in learner.candidate.state_dict().items()))
            self.assertEqual([r['action'] for r in restored.replay.sample(10)],[r['action'] for r in learner.replay.sample(10)])

    def test_importing_old_brain_preserves_file_and_resets_new_controller_exploration(self):
        from training_diagnostics import file_sha256
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'old.pt');old=DQNLearner(OBS_DIM,2,config())
            old.steps_seen=99999;old.save(path);digest=file_sha256(path)
            learner=self.learner();learner.load(path)
            self.assertEqual(file_sha256(path),digest);self.assertEqual(learner.steps_seen,0)
            np.testing.assert_allclose(learner.q.trunk[0].weight[:,:OBS_DIM].detach().numpy(),old.q.trunk[0].weight.detach().numpy())
            self.assertEqual(float(learner.q.trunk[0].weight[:,OBS_DIM:].detach().abs().sum()),0.)


class LiveEvaluationTests(unittest.TestCase):
    def test_single_cpu_mode_inference_matches_batched_mode_heads(self):
        learner=RouteLearner(CONTROLLER_DIM,2,config());net=learner.candidate
        states=torch.from_numpy(np.array([observation(i) for i in range(len(CONTROL_MODES))]))
        actions=torch.zeros((len(states),HORIZON));durations=torch.full_like(actions,.04)
        with torch.no_grad():
            expected_q=net(states);expected_motion=net.dynamics(states,actions,durations)
            for i in range(len(states)):
                torch.testing.assert_close(net(states[i:i+1])[0],expected_q[i],rtol=1e-5,atol=1e-7)
                torch.testing.assert_close(net.dynamics(states[i:i+1],actions[i:i+1],durations[i:i+1])[0],expected_motion[i],rtol=1e-5,atol=1e-7)
        net.zero_grad(set_to_none=True)
        net(states[:1]).sum().backward()
        # A size-one TRAINING batch retains the old zero-gradient behavior
        # for other heads, including optimizer momentum semantics.
        self.assertIsNotNone(net.advantages[1].weight.grad)
        self.assertEqual(float(net.advantages[1].weight.grad.abs().sum()),0.)

    def test_one_action_perfect_agreement_cannot_queue_a_trial(self):
        learner=RouteLearner(CONTROLLER_DIM,2,config())
        with torch.no_grad():
            for parameter in learner.candidate.parameters():parameter.zero_()
        for i in range(32):learner.validation.add(row(i%2,teacher=0))
        report=learner.evaluate_candidate()
        self.assertEqual(report['modes']['cube']['candidate']['accuracy'],1.)
        self.assertEqual(report['modes']['cube']['candidate']['action_label_counts'],[32,0])
        self.assertFalse(report.get('offline_passed',False))

    def test_coaching_samples_rare_holds_and_excludes_unlabelled_rows(self):
        learner=RouteLearner(CONTROLLER_DIM,2,config())
        for i in range(25):learner.replay.add(row(0,teacher=0))
        learner.replay.add(row(1,teacher=1))
        learner.replay.add(row(1,teacher=-1))
        draws=learner._coaching_rows(200)
        self.assertTrue(all(r['teacher']>=0 for r in draws))
        self.assertGreater(sum(r['teacher']==1 for r in draws),60)

    def test_cached_coaching_partition_preserves_sampling_sequence(self):
        learner=RouteLearner(CONTROLLER_DIM,2,config())
        for i in range(32):learner.replay.add(row(i%2,mode=i%3,teacher=i%2))
        state=copy.deepcopy(learner.rng.bit_generator.state)
        expected=[learner._coaching_rows(16) for _ in range(4)]
        learner.rng.bit_generator.state=state;buckets=learner._coaching_buckets()
        actual=[learner._coaching_rows(16,buckets) for _ in range(4)]
        self.assertEqual([[id(r) for r in batch] for batch in actual],[[id(r) for r in batch] for batch in expected])

    def test_planner_coaching_is_separate_from_observed_success_and_excludes_death_tail(self):
        states=[observation() for _ in range(9)]
        records=[dict(first_action=1,verified=True,tracking_valid=True) for _ in range(8)]
        learner=RouteLearner(CONTROLLER_DIM,2,config())
        learner.add_segment(states[:-1],[0]*8,[.1]*8,states[-1],False,teacher=[-1]*8,route_records=records)
        self.assertTrue(all(r['planner']==1 and r['teacher']==-1 for r in learner.replay.rows()))
        dying=RouteLearner(CONTROLLER_DIM,2,config())
        dying.add_segment(states[:-1],[0]*8,[.1]*8,states[-1],True,teacher=[-1]*8,route_records=records)
        self.assertTrue(all(r['planner']==-1 for r in dying.replay.rows()))

    def test_failed_trial_requires_learning_before_retry_and_persists_schedule(self):
        from controller_evaluation import LiveComparison
        learner=RouteLearner(CONTROLLER_DIM,2,config())
        learner.trial=LiveComparison();learner.trial_model=copy.deepcopy(learner.candidate);learner.trial_modes={0}
        for i in range(6):
            learner.trial.prints=[np.zeros((27,48,3),np.float32)]
            result=learner.finish_evaluation_attempt(.3 if i%2==0 else .1,100)
        self.assertFalse(result['passed'])
        self.assertEqual(learner.next_trial_episode,learner.episode_id+24)
        self.assertEqual(learner.next_trial_grad,learner.grad_steps+512)
        for i in range(32):learner.replay.add(row(i%2))
        learner.evaluate_candidate=lambda:dict(promoted=False,offline_passed=True,eligible_modes=[0])
        learner.train(8)
        self.assertFalse(learner.evaluating)
        self.assertGreater(learner.last_evaluation['training_episodes_remaining'],0)
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'schedule.pt');learner.save(path)
            restored=RouteLearner(CONTROLLER_DIM,2,config());restored.load(path)
            self.assertEqual(restored.next_trial_episode,learner.next_trial_episode)
            self.assertEqual(restored.next_trial_grad,learner.next_trial_grad)
            self.assertEqual(restored.trial_failures,1)
        learner.episode_id=learner.next_trial_episode;learner.grad_steps=learner.next_trial_grad
        learner.train(8)
        self.assertTrue(learner.evaluating)
        for i in range(6):
            learner.trial.prints=[np.zeros((27,48,3),np.float32)]
            learner.finish_evaluation_attempt(.3 if i%2==0 else .1,100)
        self.assertEqual(learner.next_trial_episode,learner.episode_id+48)

    def test_old_evaluation_policy_cancels_pending_trial_without_changing_weights(self):
        from controller_evaluation import LiveComparison
        learner=RouteLearner(CONTROLLER_DIM,2,config())
        learner.trial=LiveComparison();learner.trial_model=copy.deepcopy(learner.candidate);learner.trial_modes={0}
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'old_gate.pt');learner.save(path)
            ck=torch.load(path,weights_only=False);ck.pop('evaluation_policy_version');ck.pop('trial_schedule')
            torch.save(ck,path)
            restored=RouteLearner(CONTROLLER_DIM,2,config());restored.load(path)
            self.assertFalse(restored.evaluating)
            self.assertTrue(all(torch.equal(v,restored.q.state_dict()[k]) for k,v in learner.q.state_dict().items()))
            self.assertTrue(all(torch.equal(v,restored.candidate.state_dict()[k]) for k,v in learner.candidate.state_dict().items()))
            self.assertEqual(len(restored.replay),len(learner.replay))
            self.assertEqual(restored.next_trial_episode,restored.episode_id+24)

    def test_live_pass_is_required_before_deployed_weights_change(self):
        from controller_evaluation import LiveComparison
        learner=RouteLearner(CONTROLLER_DIM,2,config())
        learner.trial=LiveComparison();learner.trial_model=copy.deepcopy(learner.candidate)
        learner.trial_modes={0}
        with torch.no_grad():learner.trial_model.advantages[0].bias[1].add_(3.)
        before=learner.q.advantages[0].bias.clone()
        for i in range(6):
            learner.trial.prints=[np.ones((27,48,3),np.float32)]
            learner.trial.steps=100;learner.trial.interventions=10
            result=learner.finish_evaluation_attempt(.2 if i%2==0 else .4,100)
            if i<5:self.assertTrue(torch.equal(before,learner.q.advantages[0].bias))
        self.assertTrue(result['passed']);self.assertFalse(learner.evaluating)
        self.assertTrue(learner.promotion_pending)
        self.assertFalse(torch.equal(before,learner.q.advantages[0].bias))

    def test_trial_restart_discards_partial_attempt_without_losing_completed_comparisons(self):
        from controller_evaluation import LiveComparison
        learner=RouteLearner(CONTROLLER_DIM,2,config());learner.trial=LiveComparison()
        learner.trial_model=copy.deepcopy(learner.candidate);learner.trial_modes={0}
        learner.trial.prints=[np.ones((27,48,3),np.float32)]
        learner.finish_evaluation_attempt(.2,100)
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'trial.pt');learner.save(path)
            restored=RouteLearner(CONTROLLER_DIM,2,config());restored.load(path)
            self.assertEqual(len(restored.trial.records),1)
            restored.finish_evaluation_attempt(.9,50)
            self.assertEqual(len(restored.trial.records),1)
            self.assertFalse(restored.trial_skip_until_reset)

    def comparison(self,old,new,old_shield=.2,new_shield=.1):
        from controller_evaluation import LiveComparison
        comparison=LiveComparison()
        for i in range(6):
            comparison.prints=[np.zeros((27,48,3),np.float32)]
            comparison.steps=100;comparison.interventions=int(100*(old_shield if i%2==0 else new_shield))
            result=comparison.record(old[i//2] if i%2==0 else new[i//2],100)
        return result

    def test_lucky_best_does_not_override_worse_consistency(self):
        result=self.comparison([.4,.4,.4],[.01,.01,.9])
        self.assertFalse(result['passed'])

    def test_consistent_progress_without_more_interventions_passes(self):
        self.assertTrue(self.comparison([.2,.3,.4],[.3,.4,.5])['passed'])

    def test_more_shield_dependence_blocks_promotion(self):
        self.assertFalse(self.comparison([.2,.3,.4],[.3,.4,.5],.1,.6)['passed'])

    def test_level_change_makes_comparison_inconclusive(self):
        from controller_evaluation import LiveComparison
        comparison=LiveComparison()
        for i in range(6):
            comparison.prints=[np.full((27,48,3),255 if i>=3 else 0,np.float32)]
            result=comparison.record(.4 if i%2 else .2,100)
        self.assertFalse(result['passed']);self.assertIn('identity',result['reason'])

    def test_frozen_trial_is_deterministic_and_training_pauses(self):
        from controller_evaluation import LiveComparison
        learner=RouteLearner(CONTROLLER_DIM,2,config())
        learner.trial=LiveComparison();learner.trial_model=copy.deepcopy(learner.candidate);learner.trial_modes={0}
        learner.context=dict(verified=True,first_action=1)
        before=learner.steps_seen
        actions=[learner.act(observation())[0] for _ in range(8)]
        self.assertEqual(actions,[1]*8);self.assertEqual(learner.steps_seen,before)
        self.assertEqual(learner.train(100),{})

    def test_trainer_excludes_live_evaluation_from_experience_and_logs_arms(self):
        import gymnasium as gym
        from trainer import Trainer
        from controller_evaluation import LiveComparison
        class Env:
            observation_space=gym.spaces.Box(-2.,2.,(CONTROLLER_DIM,),np.float32)
            action_space=gym.spaces.Discrete(2)
            route_context=dict(verified=True,calibrated=True,tracking_valid=True,first_action=1,mode='cube')
            _last_search={};last_hires=np.zeros((180,320,3),np.uint8)
            frame_skip=2
            def __init__(self):self.t=0
            def reset(self,**kw):return observation(),{}
            def step(self,action):
                self.t+=1;died=self.t%6==0
                return observation(),0.,False,False,dict(in_level=True,died=died,death_lag=1,
                    death_lag_src='measured',attempt_steps=6,action_applied=action,prog=.2)
            def attempt_time(self):return ((self.t-1)%6+1)*.1
            def _motion_dt(self):return .04
            def pause(self):pass
            def resume(self):pass
            def mark_trained(self):pass
        with tempfile.TemporaryDirectory() as directory:
            cfg=config(total_steps=36,save_dir=directory,log_dir=directory,name='route_controller',route=False,verbose=0)
            learner=RouteLearner(CONTROLLER_DIM,2,cfg);learner.trial=LiveComparison()
            learner.trial_model=copy.deepcopy(learner.candidate);learner.trial_modes={0}
            trainer=Trainer(Env(),cfg,learner=learner);trainer.run()
            self.assertEqual(len(learner.replay),0);self.assertEqual(len(learner.validation),0)
            self.assertEqual(trainer.updates,0)
            import csv
            with open(trainer._csv_path) as source:rows=list(csv.DictReader(source))
            self.assertEqual([r['route'] for r in rows],['controller_eval_incumbent','controller_eval_candidate']*3)


class WorldModelLearningTests(unittest.TestCase):
    def test_future_commands_cannot_change_earlier_predictions(self):
        learner=RouteLearner(CONTROLLER_DIM,2,config());o=torch.from_numpy(observation()[None])
        durations=torch.full((1,HORIZON),.04);a=torch.zeros((1,HORIZON));b=a.clone();b[:,2:]=1.
        first=learner.candidate.dynamics(o,a,durations);second=learner.candidate.dynamics(o,b,durations)
        self.assertTrue(torch.equal(first[:,:2],second[:,:2]))
        self.assertFalse(torch.equal(first[:,2:],second[:,2:]))
        o[:,OBS_DIM+5]=2/8.
        first=learner.candidate.dynamics(o,a,durations);second=learner.candidate.dynamics(o,torch.ones_like(a),durations)
        self.assertTrue(torch.equal(first[:,:2],second[:,:2]))

    def test_measured_multi_step_motion_error_falls_after_supervised_training(self):
        learner=RouteLearner(CONTROLLER_DIM,2,config(lr=.003))
        # Independent phase values in validation; actions and durations are
        # supplied explicitly, so this tests learning effects, not copying
        # the requested target route.
        for i in range(80):
            r=row(i%2,teacher=-1);r['obs'][0]=i/1000.
            direction=1. if i%2 else -1.
            r['outcomes'][:,0]=direction*np.arange(1,6)*.01
            r['outcomes'][:,1]=direction*.1
            learner.replay.add(r)
        samples=[row(i%2,teacher=-1) for i in range(16)]
        for i,r in enumerate(samples):
            direction=1. if i%2 else -1.;r['obs'][0]=(i+.5)/1000.
            r['outcomes'][:,0]=direction*np.arange(1,6)*.01;r['outcomes'][:,1]=direction*.1
        before=learner._metrics(learner.candidate,samples)
        for _ in range(4):learner.train(32)
        after=learner._metrics(learner.candidate,samples)
        self.assertLess(after['position_rmse'],before['position_rmse']*.5)
        self.assertLess(after['velocity_rmse'],before['velocity_rmse']*.5)


class ObservationLoopTests(unittest.TestCase):
    def test_completion_saves_experience_and_pending_trial_without_fake_champion(self):
        import gymnasium as gym
        from trainer import Trainer
        from controller_evaluation import LiveComparison
        class Env:
            observation_space=gym.spaces.Box(-2.,2.,(CONTROLLER_DIM,),np.float32)
            action_space=gym.spaces.Discrete(2);frame_skip=2;_last_search={}
            last_hires=np.zeros((180,320,3),np.uint8)
            route_context=dict(verified=True,calibrated=True,tracking_valid=True,first_action=1,mode='cube')
            def __init__(self):self.t=0
            def reset(self,**kw):return observation(),{}
            def step(self,a):
                self.t+=1;done=self.t==5
                return observation(),0.,done,False,dict(in_level=True,prog=1. if done else .1,
                    complete=done,action_applied=a,attempt_steps=self.t,death_lag=0)
            def _motion_dt(self):return .04
            def attempt_time(self):return self.t*.1
        for evaluation in (False,True):
            with tempfile.TemporaryDirectory() as directory:
                cfg=config(total_steps=100,save_dir=directory,log_dir=directory,name='route_controller',route=False,verbose=0)
                learner=RouteLearner(CONTROLLER_DIM,2,cfg)
                if evaluation:
                    learner.trial=LiveComparison();learner.trial_model=copy.deepcopy(learner.candidate);learner.trial_modes={0}
                trainer=Trainer(Env(),cfg,learner=learner);trainer.run()
                self.assertEqual(trainer.steps,5)
                self.assertTrue((Path(directory)/'route_controller.pt').exists())
                self.assertFalse((Path(directory)/'route_controller_best.pt').exists())
                if evaluation:self.assertEqual(len(learner.trial.records),1)
                else:self.assertTrue(all(r['success'] for r in learner.replay.rows()))

    def test_gui_snapshots_render_new_and_legacy_controllers_without_live_window(self):
        from detect import Entities,Entity
        from preview_boxes import _Env,_Tr,render
        env=SimpleNamespace(last_hires=np.zeros((360,640,3),np.uint8),last_frame=np.zeros((180,320,3),np.uint8),
            last_ents=Entities([Entity(7,.35,.5,.08,.1,1.)]),last_arcs=[('best',False,[(0.,0.),(2.,1.)])],
            _last_search=dict(path_unknown=False,plan=dict(verified=True)),_dbg=dict(controller_role='planner bootstrap'),
            mode_est=SimpleNamespace(mode='cube',mini=False,g=1,lag=0),phys=Phys(),motion_fix=None,speed=10.)
        tr=SimpleNamespace(cfg=TrainConfig(route_controller=True),_last_obs=observation(),hold_prob=lambda _: .5,
            attempts=0,best_prog=.2,updates=1,learner=SimpleNamespace(epsilon=lambda:.1))
        for new in (True,False):
            tr.cfg.route_controller=new
            snapshot=_Env(env,tr);view=render(snapshot,_Tr(tr),0,dict(moving=True),1.)
            self.assertEqual(view.shape,(360,640,3));self.assertEqual(view.dtype,np.uint8)
            self.assertEqual(snapshot._dbg['controller_role'],'planner bootstrap')
            np.testing.assert_array_equal(view[330,10],[0,200,255])

    def test_preview_explains_short_collision_horizons_and_missing_player(self):
        from preview_boxes import prediction_status
        plan=dict(trajectory=[dict(t=0.),dict(t=.024)],verified=False)
        self.assertEqual(prediction_status(dict(plan=plan,path_hit=True),True),'collision predicted in 0.02s')
        self.assertEqual(prediction_status(dict(plan=plan,path_hit=True,path_unknown=True),True),'view limit at 0.02s')
        self.assertIn('preview hidden',prediction_status(dict(plan=plan,path_hit=True),False))
        plan['sequence_safety']=dict(nominal=1.,worse=.5)
        self.assertIn('timing-sensitive',prediction_status(dict(plan=plan),True))

    def test_gui_uses_action_q_values_without_another_network_forward(self):
        from preview_boxes import _Tr
        def unexpected(_):raise AssertionError('GUI must not run the policy again')
        trainer=SimpleNamespace(last_q=np.array([0.,1.]),_last_obs=observation(),hold_prob=unexpected,
            cfg=config(),attempts=1,best_prog=.2,updates=1,learner=SimpleNamespace(epsilon=lambda:.1))
        view=_Tr(trainer)
        self.assertAlmostEqual(view.hold_prob(None),1./(1.+np.exp(-1.)))

    def test_missing_player_preview_does_not_draw_paths_at_default_origin(self):
        from detect import Entities
        from preview_boxes import render
        env=SimpleNamespace(last_hires=np.zeros((360,640,3),np.uint8),last_ents=Entities([]),
            last_arcs=[('best',False,[(0.,0.),(4.,1.)])],mode_est=None,phys=None,motion_fix=None,speed=10.,
            _last_search={},_dbg={},_lines=None)
        tr=SimpleNamespace(route_controller=True,attempts=0,best_prog=0.,updates=0,_last_obs=None,
            hold_prob=lambda _:0.,learner=SimpleNamespace(epsilon=lambda:0.))
        with_path=render(env,tr,0,{},1.)
        env.last_arcs=[]
        np.testing.assert_array_equal(with_path,render(env,tr,0,{},1.))

    def test_replanning_cannot_manufacture_tracking_reward(self):
        from detect import Entities,Entity
        def plan(y):return dict(verified=True,first_action=0,controls=[],trajectory=[
            dict(t=0.,x=0.,y=0.,vy=0.),dict(t=1.,x=10.,y=y,vy=0.)])
        base=observation()[:OBS_DIM]
        env=SimpleNamespace(mode_est=SimpleNamespace(mode='cube',g=1,mini=False,lag=0,lag_locked=True,conf=1.,hist=[]),
            phys=SimpleNamespace(n_cal=3),_calibration_valid=True,_tracking_age=3,_death_frozen=False,_in_level=True,
            _input_trace={},_tracked_vy=0.,_last_inputs=dict(vy0=0.,grounded=True,pending=[],y0=0.),
            _last_search=dict(plan=plan(0.)),speed=10.,last_arcs=[('release',False,[(0.,0.),(10.,0.)])],
            last_ents=Entities([Entity(7,.35,.5,.08,.1,1.)]),_step_dt=lambda **kw:.04,_motion_dt=lambda:.04,
            reset=lambda **kw:(base.copy(),{}))
        def step(action):
            env._last_search=dict(plan=plan(4.))
            return base.copy(),0.,False,False,dict(died=False)
        env.step=step
        from route_controller import augment
        env.last_arcs=[('release',False,[(0.,0.),(.4,.6),(.8,.8),(2.,.2)])]
        encoded,_=augment(base,env)
        # At t=.04 the interval speed is .6/.04=15, not the
        # derivative of the next segment or a centered gradient.
        self.assertAlmostEqual(float(encoded[ANALYTIC_START+1])*20.,15.,places=5)
        self.assertAlmostEqual(float(encoded[ANALYTIC_START+3])*20.,5.,places=5)
        env.last_arcs=[('release',False,[(0.,0.),(10.,0.)])]
        wrapper=RouteObservationEnv(env);wrapper.reset()
        _,_,_,_,info=wrapper.step(0)
        self.assertEqual(info['route_position_error'],0.)
        self.assertEqual(info['route_tracking_reward'],0.)
        # The next OLD reference now asks for an ascent, but the player stays
        # still. The calculated penalty must actually reach the learner.
        _,reward,_,_,info=wrapper.step(0)
        self.assertLess(info['route_tracking_reward'],0.)
        self.assertAlmostEqual(reward,info['route_tracking_reward'])
        env._pixel_moving=False;env._calibration_valid=False
        obs,_=wrapper.reset()
        self.assertEqual(obs[OBS_DIM+10],0.)
        self.assertFalse(wrapper.route_context['verified'])
        env._pixel_moving=True
        obs,_=wrapper.reset()
        self.assertEqual(obs[OBS_DIM+10],1.)

    def test_safe_unfinished_episode_is_banked_on_stop(self):
        import gymnasium as gym
        from trainer import Trainer
        class Env:
            observation_space=gym.spaces.Box(-2.,2.,(CONTROLLER_DIM,),np.float32)
            action_space=gym.spaces.Discrete(2);frame_skip=2;_last_search={}
            route_context=dict(verified=True,calibrated=True,tracking_valid=True,first_action=1,mode='cube')
            def reset(self,**kw):return observation(),{}
            def step(self,a):return observation(),0.,False,False,dict(in_level=True,prog=.1,action_applied=a)
            def _motion_dt(self):return .04
        with tempfile.TemporaryDirectory() as directory:
            cfg=config(total_steps=100,save_dir=directory,log_dir=directory,name='route_controller',route=False,verbose=0)
            tr=Trainer(Env(),cfg);tr.run()
            self.assertGreater(tr.learner.size(),0);self.assertEqual(tr.updates,0)
            self.assertTrue(tr.learner.episode_open)
            restored=RouteLearner(CONTROLLER_DIM,2,cfg)
            restored.load(str(Path(directory)/'route_controller.pt'))
            self.assertEqual(restored.episode_id,tr.learner.episode_id+1)


if __name__=='__main__':unittest.main()
