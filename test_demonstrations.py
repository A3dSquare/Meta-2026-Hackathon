import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from demonstrations import InputTimeline,fit_planner,make_observer,replay_clip
from gd_physics import Phys,Body,advance,MotionLearner,ModeEstimator,ModeCalibrator,save_motion,load_motion


def cube_clip(gravity=100.):
    rows=[]
    for arc in range(3):
        start=arc*1.
        rows.append(dict(t=start,y=0.,mode='cube',g=1,mini=False,action=1,valid=True,
                         clean=True,special=False,grounded=True,speed=10.))
        for i in range(1,13):
            t=i/30
            rows.append(dict(t=start+t,y=20*t-.5*gravity*t*t,mode='cube',g=1,mini=False,action=1,
                             valid=True,clean=True,special=False,grounded=False,speed=10.))
    return rows


class TimelineTests(unittest.TestCase):
    def test_recorder_writes_input_edges_and_markers_without_injecting_input(self):
        from record_demonstrations import InputRecorder
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'inputs.jsonl';calls=[]
            def read():
                calls.append(1)
                if len(calls)==3:recorder.stop.set()
                return len(calls)>1,True,[0x75] if len(calls)==2 else []
            recorder=InputRecorder(path,reader=read);recorder.run()
            events=[json.loads(x) for x in path.read_text().splitlines()]
            self.assertIsNone(recorder.error)
            self.assertEqual([x['down'] for x in events],[False,True])
            self.assertEqual(recorder.markers.get_nowait()[1],0x75)

    def test_capture_distinguishes_exact_and_fallback_timestamps(self):
        from capture import GDCapture
        class Camera:
            def __init__(self):self.calls=0
            def get_latest_frame(self,**kwargs):
                self.calls+=1
                return np.zeros((2,2,3),np.uint8),123. if self.calls==1 else None
        cap=GDCapture.__new__(GDCapture);camera=Camera()
        cap._latest_with_time(camera);self.assertTrue(cap.last_timestamp_exact)
        cap._latest_with_time(camera);self.assertFalse(cap.last_timestamp_exact)

    def test_mixed_input_focus_loss_and_long_poll_gaps_are_rejected(self):
        events=[dict(t=0.,down=False,focus=True,uncertainty=.002),
                dict(t=.05,down=True,focus=True,uncertainty=.002),
                dict(t=.15,down=True,focus=True,uncertainty=.09),
                dict(t=.2,down=True,focus=False,uncertainty=.002)]
        timeline=InputTimeline(events)
        self.assertFalse(timeline.clean(-.04,-.01))
        self.assertTrue(timeline.clean(.01,.04))
        self.assertFalse(timeline.clean(.04,.07))
        self.assertFalse(timeline.clean(.07,.1))  # gap finishes AFTER this frame
        self.assertFalse(timeline.clean(.19,.22))
        self.assertTrue(timeline.state(.06)['down'])

    def test_human_press_is_kept_while_partial_interval_motion_is_rejected(self):
        t=InputTimeline([dict(t=0.,down=False,focus=True,uncertainty=.002),
                         dict(t=.02,down=True,focus=True,uncertainty=.002)])
        self.assertEqual(t.command(.01,.04),1)
        self.assertFalse(t.clean(.01,.04))
        self.assertTrue(t.clean(.01,.04,allow_changes=True))
        tap=InputTimeline(t.events+[dict(t=.03,down=False,focus=True,uncertainty=.002)])
        self.assertEqual(tap.command(.01,.04),-1)

    def test_observer_has_no_input_backend_and_uses_recorded_time(self):
        from detect import Entities
        env=make_observer(lambda *args:Entities([]),Phys())
        self.assertFalse(hasattr(env.base,'inp'))
        env.replay_time=123.;env.replay_dt=.05
        env._frame_time_step()
        self.assertEqual(env._obs_ts,123.);self.assertEqual(env._motion_dt(),.05)
        with self.assertRaises(RuntimeError):env._ensure_focus()

    def test_frame_replay_runs_production_observer_without_game_input(self):
        import cv2
        from detect import Entities,Entity
        class Detector:
            def reset(self):self.n=0
            def __call__(self,*args):
                self.n+=1
                return Entities([Entity(7,.3,.5,.0527,.09375,1.)])
        with tempfile.TemporaryDirectory() as folder:
            records=[]
            for i in range(8):
                name=f'{i}.jpg';cv2.imwrite(str(Path(folder)/name),np.zeros((360,640,3),np.uint8))
                records.append(dict(t=1+i/30,file=name,timestamp_exact=True))
            timeline=InputTimeline([dict(t=0.,down=False,focus=True,uncertainty=.002)])
            rows=replay_clip(folder,records,timeline,Detector(),Phys())
            self.assertEqual(len(rows),8)
            self.assertTrue(all(np.isfinite(r['obs']).all() for r in rows))

    def test_label_lead_teaches_the_press_earlier(self):
        # the human pressed at t=1.20; live the bot acts on a frame ~45 ms after it was shown,
        # so the brain must be taught to press on the frames before 1.20 by that much
        import cv2
        from detect import Entities,Entity
        class Detector:
            def reset(self):pass
            def __call__(self,*args):return Entities([Entity(7,.3,.5,.0527,.09375,1.)])
        with tempfile.TemporaryDirectory() as folder:
            records=[]
            for i in range(12):
                name=f'{i}.jpg';cv2.imwrite(str(Path(folder)/name),np.zeros((360,640,3),np.uint8))
                records.append(dict(t=1+i/30,file=name,timestamp_exact=True))
            timeline=InputTimeline([dict(t=0.,down=False,focus=True,uncertainty=.002),
                                    dict(t=1.2,down=True,focus=True,uncertainty=.002)])
            def first_press(lead):
                rows=replay_clip(folder,records,timeline,Detector(),Phys(),lead=lead)
                return min(r['t'] for r in rows if r['action']==1)
            plain,led=first_press(0.),first_press(.045)
            self.assertAlmostEqual(plain,1+5/30,places=6)      # frame interval holding the press
            self.assertAlmostEqual(led,1+4/30,places=6)        # one frame (>= 45 ms) earlier
            self.assertLess(led,plain)


class PlannerTeachingTests(unittest.TestCase):
    def test_ship_joint_fit_and_wave_slope_improve_real_planner_rollouts(self):
        def clip(mode):
            rows=[];truth=Phys(ship_up=70.,ship_down=60.,ship_vmax=9.,wave_slope=1.2)
            for k in range(6):
                hold=k%2;body=Body(0.,-4. if hold else 4.,1,False,mode)
                for i in range(12):
                    if i:advance(body,bool(hold),1/30,10.,truth,([],[]))
                    rows.append(dict(t=k+i/30,y=body.y,mode=mode,g=1,mini=False,action=hold,valid=True,
                                     clean=True,special=False,grounded=False,speed=10.))
                rows.append(dict(rows[-1],t=k+.5,valid=False))
            return rows
        for mode,params in [('ship',('ship_up','ship_down','ship_vmax')),('wave',('wave_slope',))]:
            new,report=fit_planner([clip(mode) for _ in range(4)],[clip(mode)],Phys())
            for name in params:self.assertTrue(report[name]['accepted'],report)

    def test_held_out_improvement_changes_saved_physics_and_actual_rollout(self):
        old=Phys(cube_g=90.,cube_v0=20.)
        new,report=fit_planner([cube_clip() for _ in range(4)],[cube_clip()],old)
        self.assertTrue(report['cube_g']['accepted'],report)
        self.assertAlmostEqual(new.cube_g,100.,places=5)
        with tempfile.TemporaryDirectory() as folder:
            path=str(Path(folder)/'motion.npz');new.n_cal=3
            save_motion(path,MotionLearner(),new,ModeEstimator(),ModeCalibrator())
            loaded=Phys();self.assertTrue(load_motion(path,MotionLearner(),loaded,ModeEstimator()))
            a=Body(1.,5.,1,False,'cube');b=copy.deepcopy(a)
            advance(a,False,.2,10.,old,([],[]));advance(b,False,.2,10.,loaded,([],[]))
            self.assertAlmostEqual(a.y-b.y,.2,places=5)

    def test_validation_cannot_fit_parameters_or_accept_a_regression(self):
        old=Phys(cube_g=110.,cube_v0=20.)
        new,report=fit_planner([cube_clip(90.) for _ in range(4)],[cube_clip(110.)],old)
        self.assertFalse(report['cube_g']['accepted'])
        self.assertAlmostEqual(report['cube_g']['proposed'],90.,places=5)
        self.assertEqual(new.cube_g,old.cube_g)

    def test_bad_tracking_cannot_teach_physics(self):
        bad=cube_clip()
        for r in bad:r['valid']=False
        new,report=fit_planner([bad]*4,[cube_clip()],Phys())
        self.assertTrue(all(not r['accepted'] for r in report.values()))


class HumanLearningTests(unittest.TestCase):
    def test_human_press_teaches_action_without_false_constant_hold_motion(self):
        from test_route_controller import observation,config
        from route_controller import CONTROLLER_DIM
        from route_learner import RouteLearner
        learner=RouteLearner(CONTROLLER_DIM,2,config())
        states=[observation() for _ in range(20)]
        learner.add_segment(states[:-1],[1]*19,[.1]*19,states[-1],False,completed=True,human_demo=True,
                            motion_valid=[False]*19,human_labels=[1]*19)
        rows=learner.replay.rows()
        self.assertTrue(any(r['teacher']==1 for r in rows))
        self.assertTrue(all(not np.any(r['mask']) for r in rows))

    def test_human_labels_do_not_require_agreement_with_planner_and_stay_held_out(self):
        from test_route_controller import observation,config
        from route_controller import CONTROLLER_DIM
        from route_learner import RouteLearner
        learner=RouteLearner(CONTROLLER_DIM,2,config())
        states=[observation() for _ in range(20)]
        # No route records and no teacher recommendation: human holds still teach.
        learner.episode_id=5
        learner.add_segment(states[:-1],[1]*19,[.1]*19,states[-1],True,completed=True,human_demo=True)
        self.assertEqual(len(learner.replay),0)
        labels=[r for r in learner.validation.rows() if r['teacher']==1]
        self.assertTrue(labels)
        self.assertTrue(all(r['source']=='human' for r in labels))
        self.assertFalse(learner._coaching_buckets())

    def test_invalid_observations_do_not_become_human_labels(self):
        from test_route_controller import observation,config
        from route_controller import CONTROLLER_DIM
        from route_learner import RouteLearner
        from env_boxes import OBS_DIM
        learner=RouteLearner(CONTROLLER_DIM,2,config())
        states=[observation() for _ in range(10)]
        for s in states:s[OBS_DIM+10]=0.
        learner.add_segment(states[:-1],[1]*9,[.1]*9,states[-1],True,completed=True,human_demo=True)
        self.assertTrue(all(r['teacher']==-1 and not np.any(r['mask']) for r in learner.replay.rows()))

    def test_training_cli_writes_new_candidate_preserves_source_and_persists_imports(self):
        import sys
        from unittest.mock import patch
        import torch
        from test_route_controller import observation,config
        from route_controller import CONTROLLER_DIM
        from route_learner import RouteLearner
        from training_diagnostics import file_sha256
        import train_demonstrations
        learner=RouteLearner(CONTROLLER_DIM,2,config())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);brain=root/'source.pt';learner.save(str(brain));digest=file_sha256(brain)
            recording=root/'recording';recording.mkdir();clips=[];frames=[]
            for i in range(8):
                clips.append(dict(id=str(i),start=0.,end=2.,result='success'))
                frames.extend(dict(t=j/30,clip=str(i),timestamp_exact=True,file='unused.jpg') for j in range(2 if i==7 else 40))
            (recording/'meta.json').write_text(json.dumps(dict(version=1,id='fixture',level='test-level',clips=clips[:5])))
            tail=root/'tail';tail.mkdir()
            (tail/'meta.json').write_text(json.dumps(dict(version=1,id='fixture_tail',level='test-level',clips=clips[5:])))
            (recording/'frames.jsonl').write_text('\n'.join(json.dumps(f) for f in frames))
            (recording/'inputs.jsonl').write_text(json.dumps(dict(t=0.,down=True,focus=True,uncertainty=.002)))
            (tail/'frames.jsonl').write_bytes((recording/'frames.jsonl').read_bytes())
            (tail/'inputs.jsonl').write_bytes((recording/'inputs.jsonl').read_bytes())
            def replay(*args):
                result=[]
                for i in range(40):
                    r=dict(t=i/30,obs=observation(),action=i%2,human_label=i%2,motion_valid=True,
                           valid=True,clean=True,special=False,mode='cube',g=1,mini=False,grounded=True,y=0.,speed=10.)
                    r['obs'][0]=float(i%2);result.append(r)
                clip_id=args[1][0]['clip']
                if clip_id in ('5','6'):
                    for i,r in enumerate(result):
                        r['valid']=clip_id=='6' and i%4!=0
                        r['rejected']=[] if r['valid'] else ['mode uncertain']
                return result
            out=root/'trained'
            out.mkdir()  # an empty directory from a failed older import is reusable
            physics_path=root/'source_motion.npz'
            source_physics=Phys();source_physics.n_cal=10
            source_timing=ModeEstimator();source_timing.lag=2;source_timing.delay_s=.069
            source_timing.delay_alt_s=.083
            save_motion(str(physics_path),MotionLearner(),source_physics,source_timing)
            argv=['train_demonstrations.py',str(recording),str(tail),'--model',str(brain),'--motion-model',str(physics_path),
                  '--out',str(out),'--cycles','1']
            with patch.object(sys,'argv',argv),patch('detect_yolo.YoloOnlyDetector'),patch('demonstrations.replay_clip',side_effect=replay):
                train_demonstrations.main()
            self.assertEqual(file_sha256(brain),digest)
            ck=torch.load(out/'route_controller.pt',weights_only=False)
            self.assertEqual(len(ck['demo_imports']),5)
            self.assertEqual(ck['trusted_modes'],[])
            for key,value in learner.q.state_dict().items():self.assertTrue(torch.equal(value,ck['q'][key]))
            self.assertTrue(any(not torch.equal(value,ck['candidate'][key]) for key,value in learner.candidate.state_dict().items()))
            self.assertTrue((out/'motion_model.npz').exists())
            restored_timing=ModeEstimator()
            self.assertTrue(load_motion(str(out/'motion_model.npz'),MotionLearner(),Phys(),restored_timing))
            self.assertEqual(restored_timing.lag,2)
            self.assertAlmostEqual(restored_timing.delay_s,.069)
            self.assertAlmostEqual(restored_timing.delay_alt_s,.083)
            self.assertFalse(json.loads((out/'report.json').read_text())['live_promotion'])
            entries=json.loads((out/'import_progress.json').read_text())['recordings']
            skipped={r['id']:r for r in entries if r['status']=='skipped'}
            self.assertEqual(set(skipped),{'fixture_tail/5','fixture_tail/6','fixture_tail/7'})
            self.assertEqual(skipped['fixture_tail/5']['usable'],0)
            self.assertEqual(skipped['fixture_tail/6']['usable'],30)
            self.assertIn('continuous',skipped['fixture_tail/6']['reason'])
            self.assertTrue(next(r for r in entries if r['id']=='fixture/4')['held_out'])
            # Rebuild old demonstrations into protected storage, never double
            # the import manifest or move a held-out episode into training.
            refresh_argv=list(argv)+['--refresh-demonstrations']
            refresh_argv[refresh_argv.index('--model')+1]=str(out/'route_controller.pt')
            refreshed=root/'refreshed'
            refresh_argv[refresh_argv.index('--out')+1]=str(refreshed)
            with patch.object(sys,'argv',refresh_argv),patch('detect_yolo.YoloOnlyDetector'),patch('demonstrations.replay_clip',side_effect=replay):
                train_demonstrations.main()
            refreshed_ck=torch.load(refreshed/'route_controller.pt',weights_only=False)
            self.assertEqual(set(refreshed_ck['demo_imports']),set(ck['demo_imports']))
            self.assertEqual(len(refreshed_ck['demo_imports']),5)
            self.assertTrue(json.loads((refreshed/'report.json').read_text())['data_audit'])
            self.assertTrue(all(r['episode']%5==0 for b in refreshed_ck['validation']['buckets'].values() for r in b))
            # Losing the held-out run cannot be repaired by leaking a training
            # run into validation or by accepting an unvalidated candidate.
            def missing_validation(*args):
                rows=replay(*args)
                if args[1][0]['clip']=='4':
                    for r in rows:r['valid']=False
                return rows
            no_valid=root/'no_validation';argv[argv.index('--out')+1]=str(no_valid)
            with patch.object(sys,'argv',argv),patch('detect_yolo.YoloOnlyDetector'),patch('demonstrations.replay_clip',side_effect=missing_validation):
                with self.assertRaisesRegex(ValueError,'training AND held-out'):train_demonstrations.main()
            self.assertFalse((no_valid/'route_controller.pt').exists())
            self.assertTrue((no_valid/'import_progress.json').exists())


if __name__=='__main__':unittest.main()
