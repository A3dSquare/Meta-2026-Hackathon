"""Pure replay, input alignment, and held-out physics fitting for manual demos."""
from bisect import bisect_right
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np


class InputTimeline:
    def __init__(self,events):
        self.events=sorted(events,key=lambda e:e['t']);self.times=[e['t'] for e in self.events]

    def state(self,t):
        i=bisect_right(self.times,t)-1
        return self.events[i] if i>=0 else dict(down=False,focus=False,uncertainty=1.)

    def clean(self,start,end,allow_changes=False):
        """Reject unknown focus, polling gaps, or mixed-input frame intervals."""
        if not .005<=end-start<=.1:return False
        initial=self.state(start)
        if not initial['focus']:return False
        # An event's uncertainty covers the interval BEFORE its timestamp.
        for event in self.events[max(0,bisect_right(self.times,start)-1):bisect_right(self.times,end)+1]:
            if event['t']-event.get('uncertainty',0.)<end and event['t']>start:
                if event.get('uncertainty',0.)>.01:return False
            if start<event['t']<=end and (not event['focus'] or (not allow_changes and event['down']!=initial['down'])):return False
        return True

    def command(self,start,end):
        """Imitate a press/release within the upcoming frame, never invent a tap.

        The command timing is quantized to at most one frame early. Such a
        partial interval is excluded from constant-command motion training.
        """
        if not self.clean(start,end,allow_changes=True):return -1
        changes=[];down=self.state(start)['down']
        for e in self.events[bisect_right(self.times,start):bisect_right(self.times,end)]:
            if e['down']!=down:changes.append(e);down=e['down']
        return int(down) if len(changes)<=1 else -1


def read_recording(folder):
    folder=Path(folder);meta=json.loads((folder/'meta.json').read_text())
    if meta.get('version')!=1:raise ValueError('Unsupported demonstration version')
    frames=[json.loads(x) for x in (folder/'frames.jsonl').read_text().splitlines() if x]
    events=[json.loads(x) for x in (folder/'inputs.jsonl').read_text().splitlines() if x]
    return meta,frames,InputTimeline(events)


def valid_runs(rows):
    """Never bootstrap TD targets across tracking gaps or mixed-input intervals."""
    run=[]
    for row in rows:
        if not row['valid']:
            if len(run)>=8:yield run
            run=[]
        else:run.append(row)
    if len(run)>=8:yield run


def clip_quality(rows):
    """Shared preflight for original and recalibrated observations."""
    from collections import Counter
    usable=sum(bool(r['valid']) for r in rows)
    runs=[len(run) for run in valid_runs(rows)]
    reasons=Counter(reason for r in rows if not r['valid'] for reason in r.get('rejected',[]))
    # Count the exact action-label eligibility used by RouteLearner.add_segment,
    # after continuity filtering, including its end-of-section exclusion.
    from env_boxes import OBS_DIM
    labels=Counter()
    for run in valid_runs(rows):
        remaining=np.cumsum([float(r['obs'][OBS_DIM+3]) for r in run[:-1]][::-1])[::-1]
        for i,r in enumerate(run[:-1]):
            label=r.get('human_label',r.get('action',-1))
            if label>=0 and remaining[i]>.15 and min(r['obs'][OBS_DIM+10],run[i+1]['obs'][OBS_DIM+10])>.5:
                labels[(r['mode'],int(label))]+=1
    reason=('too few reliable observations (minimum 20)' if usable<20 else
            'no continuous valid segment (minimum 8 frames)' if not runs else None)
    return dict(frames=len(rows),usable=usable,continuous_runs=runs,
                eligible_human_labels=[dict(mode=mode,action=action,count=count)
                                       for (mode,action),count in sorted(labels.items())],
                rejected_observations=dict(reasons),status='skipped' if reason else 'accepted',reason=reason)


def make_observer(detector,physics):
    """Reuse production observation/planning code, with no game/input object."""
    import gymnasium as gym
    from env_boxes import EntityBoxEnv
    class Base(gym.Env):
        observation_space=gym.spaces.Box(0,255,(180,320,3),np.uint8)
        action_space=gym.spaces.Discrete(2)
        def __init__(self):
            self.cap=SimpleNamespace(last_hires=None,last_frame_ts=None)
            self.reward=SimpleNamespace(last_pct=0.)
    class Observer(EntityBoxEnv):
        def _step_dt(self,peek=False):return self.replay_dt
        def _frame_time_step(self):
            self._dt_frame=self.replay_dt
            self._obs_ts_prev,self._obs_ts=getattr(self,'_obs_ts',None),self.replay_time
        def _watch_press(self,y_b,*args):self._last_yb=y_b
        def _ensure_focus(self,*args,**kwargs):raise RuntimeError('Offline replay cannot focus the game')
    env=Observer(Base(),detector=detector,motion_path=None)
    env.phys=copy.deepcopy(physics);env.route_controller=True;env._in_level=True
    env.mode_est.lag=0;env.mode_est.lag_locked=True
    # Fits are proposed separately from training clips; held-out observations
    # must not silently recalibrate the model being evaluated.
    env.cube_physics_learner=SimpleNamespace(observe=lambda *a,**k:None,reset=lambda *a,**k:None,last={})
    env.mode_cal.update=lambda *a,**k:None
    env.launch_cal.update=lambda *a,**k:None
    env.replay_dt=1/30;env.replay_time=0.
    return env


def replay_clip(folder,frames,timeline,detector,physics,lead=0.):
    """lead (s): teach the brain the key state this much AFTER each frame. A human's key is
    paired with the frame on screen at that moment, but live the bot only acts on a frame
    after capturing and processing it (live: 30-60 ms, median ~45). Labels at lead=0 taught
    "press when the screen looks like this", so every press came that much late -- the shield
    overrode a waiting brain 198 times in one session (284 shield-right verdicts vs 1) and
    late jumps off the floating slabs at 4% missed the next one. The motion/physics data keep
    the real key state at each frame (that's what the game did)."""
    import cv2
    from route_controller import augment
    from env_boxes import OBS_DIM
    detector.reset();env=make_observer(detector,physics);result=[];previous=None
    for record in frames:
        ts=record['t'];dt=ts-previous if previous is not None else 1/30
        if previous is not None and ts<=previous:continue
        image_path=(Path(folder)/record['file']).resolve()
        if Path(folder).resolve() not in image_path.parents:raise ValueError('Frame path escapes recording')
        image=cv2.imread(str(image_path))
        if image is None:raise ValueError(f'Unreadable frame: {image_path}')
        rgb=cv2.cvtColor(cv2.resize(image,(320,180)),cv2.COLOR_BGR2RGB)
        clean=previous is not None and record.get('timestamp_exact',False) and timeline.clean(previous,ts,allow_changes=True)
        incoming=timeline.state(previous if previous is not None else ts)
        if not clean:env._invalidate_tracking(bool(incoming['down']))
        env.replay_dt=dt;env.replay_time=ts;env.base.cap.last_hires=image
        env._input_trace={'paused':not clean};env._pixel_moving=True
        base=env._observe(rgb,bool(incoming['down']))
        obs,context=augment(base,env)
        if not clean:obs[OBS_DIM+10]=0.
        player=env.last_ents.player()
        rejected=[]
        if not clean:rejected.append('input timing or focus')
        if player is None:rejected.append('player not detected')
        elif player.conf<=.6:rejected.append('player confidence too low')
        if env._tracking_age<2:rejected.append('tracking not established')
        if env.mode_est.conf<.6 and getattr(env.mode_est,'_icon_lock',0)<=0:rejected.append('mode uncertain')
        if env._death_frozen:rejected.append('scene frozen')
        # Human commands are timestamped at the observation, with no bot
        # decision/capture queue. Do not import this zero queue into live physics.
        action=int(timeline.state(ts)['down'])
        obs[OBS_DIM+5]=0.
        # the previous COMMAND the bot would have sent, on the same shifted timeline as the labels
        obs[OBS_DIM+30]=float(timeline.state((previous if previous is not None else ts)+lead)['down'] if lead else incoming['down'])
        if lead:action=int(timeline.state(ts+lead)['down'])
        result.append(dict(t=ts,obs=obs,action=action,human_label=action,motion_valid=False,valid=bool(obs[OBS_DIM+10]),rejected=rejected,
            mode=env.mode_est.mode,g=env.mode_est.g,mini=env.mode_est.mini,
            y=float((.75-player.y)/(30/320)) if player else None,
            grounded=bool((env._last_inputs or {}).get('grounded')),
            speed=float(env.speed or 10.38),clean=clean,
            calibratable=bool(env._calibration_valid and
                              (env.mode_est.mode=='cube' or timeline.clean(ts-.09,ts))),
            special=any(int(e.cls) in (1,2,3,4,5) and abs(e.x-env.last_ents.player_pos()[0])<.08 for e in env.last_ents.items)))
        previous=ts
    # Keep human press/release teaching at action edges. Only the motion loss
    # excludes partial intervals that the fixed-command schema cannot express.
    for i in range(len(result)-1):
        a,b=result[i]['t'],result[i+1]['t']
        label=timeline.command(a+lead,b+lead)
        result[i]['human_label']=label
        result[i]['motion_valid']=timeline.clean(a,b)
        if label>=0:result[i]['action']=label
        else:
            result[i]['obs'][OBS_DIM+10]=0.;result[i]['valid']=False
            result[i]['rejected'].append('ambiguous upcoming input interval')
    return result


def motion_windows(rows):
    """Nonoverlapping clean free-flight pieces, including pre-launch support."""
    windows=[];segment=[];rest=None;key=None
    def finish():
        if len(segment)>=7 and segment[-1]['t']-segment[0]['t']>=.18:
            windows.append(dict(rows=list(segment),rest=rest))
    for r in rows:
        newkey=(r['mode'],r['g'],r['mini'],r['action'] if r['mode'] in ('ship','wave') else None)
        usable=r['valid'] and r['clean'] and r.get('calibratable',True) and not r['special'] and r['mode'] in ('cube','ship','wave')
        if not usable or r['grounded'] or newkey!=key or len(segment)>=12:
            finish();segment=[]
            rest=r if usable and r['grounded'] else None
        if usable and not r['grounded']:segment.append(r)
        key=newkey
    finish()
    return windows


def window_fit(window):
    rows=window['rows'];t=np.array([r['t']-rows[0]['t'] for r in rows]);y=np.array([r['y'] for r in rows])
    c,b,a=np.polyfit(t,y,2)
    if np.max(np.abs(np.polyval([c,b,a],t)-y))>.12:return None
    return t,y,float(c),float(b),float(a)


def parameter_votes(windows,physics):
    from gd_physics import ModeCalibrator
    votes={k:[] for k in ('cube_g','cube_v0','ship_up','ship_down','ship_vmax','wave_slope')}
    for w in windows:
        fit=window_fit(w)
        if fit is None:continue
        t,y,c,b,a=fit;r=w['rows'][0];g=r['g'];scale=physics.mini if r['mini'] else 1.
        if r['mode']=='cube':
            gravity=-2*c*g
            if 60<=gravity<=130:
                votes['cube_g'].append(gravity)
                rest=w['rest']
                if rest is not None:
                    discriminant=b*b+2*gravity*g*(a-rest['y'])
                    if discriminant>0:
                        speed=np.sqrt(discriminant);tau=(b*g-speed)/gravity
                        if rest['t']-r['t']-.01<=tau<=.01 and 14<=speed/scale<=26:
                            votes['cube_v0'].append(float(speed/scale))
        elif r['mode']=='ship':
            acc,cap=ModeCalibrator._ship_fit(t,(y-y[0])*g/scale,1 if r['action'] else -1,physics.ship_vmax)
            if acc is not None:votes['ship_up' if r['action'] else 'ship_down'].append(acc)
            if cap is not None:votes['ship_vmax'].append(cap)
        elif r['mode']=='wave' and abs(2*c)<15:
            slope=abs(b+c*t[-1])/(r['speed']*(2 if r['mini'] else 1))
            if .5<=slope<=2:votes['wave_slope'].append(float(slope))
    return votes


def forecast_errors(windows,physics,mode,hold=None):
    from gd_physics import Body,advance
    errors=[]
    for w in windows:
        rows=w['rows'];r=rows[0]
        if r['mode']!=mode or (hold is not None and r['action']!=hold) or window_fit(w) is None:continue
        # Fit ONLY preceding observations for initial velocity. Evaluate the
        # actual planner integrator on later frames, not on fitted endpoints.
        t=np.array([x['t']-r['t'] for x in rows]);y=np.array([x['y'] for x in rows])
        c,b,a=np.polyfit(t[:4],y[:4],2)
        body=Body(float(y[3]),float(b+2*c*t[3]),r['g'],False,mode,r['mini'])
        per=[]
        for j in range(4,len(rows)):
            advance(body,bool(rows[j-1]['action']),float(t[j]-t[j-1]),rows[j-1]['speed'],physics,([],[]))
            per.append(abs(body.y-y[j]))
        if per:errors.append(float(np.mean(per)))
    return errors


def fit_planner(training,validation,physics):
    """Fit training clips only, accept each parameter on held-out motion."""
    train_windows=[w for clip in training for w in motion_windows(clip)]
    valid_windows=[w for clip in validation for w in motion_windows(clip)]
    votes=parameter_votes(train_windows,physics);held=parameter_votes(valid_windows,physics)
    result=copy.deepcopy(physics);report={}
    # Ship acceleration and its speed cap interact. A correct acceleration
    # paired with the OLD cap can look worse, so validate a jointly fitted
    # ship first, separately on climbing and descending held-out pieces.
    ship_names=('ship_up','ship_down','ship_vmax')
    if all(len(votes[k])>=3 for k in ship_names):
        joint=copy.deepcopy(result)
        for k in ship_names:setattr(joint,k,float(np.median(votes[k])))
        checks=[]
        for action in (0,1):
            old_errors=forecast_errors(valid_windows,result,'ship',action)
            new_errors=forecast_errors(valid_windows,joint,'ship',action)
            checks.append((old_errors,new_errors))
        if all(len(a)>=3 and np.mean(b)<np.mean(a)*.95-.002 and
               np.quantile(b,.95)<=np.quantile(a,.95)+.005 for a,b in checks):
            before=[v for a,b in checks for v in a];after=[v for a,b in checks for v in b]
            for k in ship_names:
                report[k]=dict(training_pieces=len(votes[k]),validation_pieces=len(before),accepted=True,
                               old=float(getattr(result,k)),proposed=float(getattr(joint,k)),
                               before=float(np.mean(before)),after=float(np.mean(after)),
                               reason='joint ship fit improves both hold and release on held-out runs',
                               metric='held-out multi-frame planner position error (blocks)',joint_group='ship')
            result=joint
    for name,values in votes.items():
        if name in report:continue
        entry=dict(training_pieces=len(values),accepted=False,old=float(getattr(result,name)))
        report[name]=entry
        if len(values)<3:entry['reason']='need at least 3 clean training pieces';continue
        proposed=float(np.median(values));trial=copy.deepcopy(result);setattr(trial,name,proposed)
        entry['proposed']=proposed
        if name=='cube_v0':
            before=[abs(x-result.cube_v0) for x in held[name]];after=[abs(x-proposed) for x in held[name]]
            entry['metric']='held-out launch-speed error (blocks/s)'
        else:
            mode='cube' if name.startswith('cube') else 'wave' if name.startswith('wave') else 'ship'
            hold=1 if name=='ship_up' else 0 if name=='ship_down' else None
            before=forecast_errors(valid_windows,result,mode,hold);after=forecast_errors(valid_windows,trial,mode,hold)
            entry['metric']='held-out multi-frame planner position error (blocks)'
        entry['validation_pieces']=len(before)
        if len(before)<3:entry['reason']='need at least 3 clean held-out pieces';continue
        entry.update(before=float(np.mean(before)),after=float(np.mean(after)),
                     before_p95=float(np.quantile(before,.95)),after_p95=float(np.quantile(after,.95)))
        accepted=entry['after']<entry['before']*.95-.002 and entry['after_p95']<=entry['before_p95']+.005
        entry['accepted']=bool(accepted);entry['reason']='held-out improvement' if accepted else 'no reliable held-out improvement'
        if accepted:result=trial
    return result,report
