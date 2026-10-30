"""Route-conditioned observations and episode-separated, balanced experience.

Coordinates are player-relative blocks; routes contain timestamps and the
actual delayed input schedule. This module does not send keyboard inputs.
"""
from collections import defaultdict
import numpy as np
from env_boxes import OBS_DIM, OFF_MODE, BLOCK_H, UNIT
from gd_physics import MODES as LEGACY_MODES

# The legacy observation aliases swing to ship. Give it its own controller
# slot without changing the legacy observation/checkpoint format.
MODES=tuple(LEGACY_MODES)+('swing',)
MODE_ID={name:i for i,name in enumerate(MODES)}

VERSION=1
HORIZON=5
TARGET_TIMES=np.array([.05,.1,.2,.4,.8],np.float32)
HEADER=31+len(MODES)
CONTROLLER_MODE_START=OBS_DIM+31
TARGET_START=OBS_DIM+HEADER
ANALYTIC_START=TARGET_START+len(TARGET_TIMES)*4
VALID_START=ANALYTIC_START+2*HORIZON*2
ROUTE_DIM=HEADER+len(TARGET_TIMES)*4+2*HORIZON*2+2*HORIZON
CONTROLLER_DIM=OBS_DIM+ROUTE_DIM
# Header: measured vy/20, instantaneous vy/20, decision dt, observed dt,
# grounded, lag/8, gravity, mini, calibrated, route verified, tracking valid, y0/8.


def mode_indices(observations):
    o=np.asarray(observations)
    dedicated=o[...,CONTROLLER_MODE_START:CONTROLLER_MODE_START+len(MODES)]
    legacy=np.argmax(o[...,OFF_MODE:OFF_MODE+len(LEGACY_MODES)],axis=-1)
    return np.where(dedicated.sum(axis=-1)>.5,dedicated.argmax(axis=-1),legacy)


def route_point(plan,t):
    points=(plan or {}).get('trajectory') or []
    if len(points)<2 or t>points[-1]['t']+1e-6:return None
    # Binary search directly in the trajectory instead of building four lists
    # and entering NumPy three times for each scalar lookup. Equal timestamps
    # choose the rightmost sample, matching np.interp.
    if t<points[0]['t']:return tuple(float(points[0][k]) for k in ('x','y','vy'))
    lo,hi=0,len(points)
    while lo<hi:
        mid=(lo+hi)//2
        if points[mid]['t']<=t:lo=mid+1
        else:hi=mid
    if lo>=len(points):return tuple(float(points[-1][k]) for k in ('x','y','vy'))
    left,right=points[lo-1],points[lo]
    elapsed=t-left['t'];duration=right['t']-left['t']
    return tuple(float((right[k]-left[k])/duration*elapsed+left[k]) for k in ('x','y','vy'))


def calibrated(env,mode):
    if not getattr(env,'_calibration_valid',False):return False
    est=env.mode_est
    if not getattr(est,'lag_locked',False):return False
    if mode in getattr(env,'model_verified_modes',()):return True
    if mode in ('ship','wave','ball','ufo'):
        keys={'ship':('ship_up','ship_down','ship_vmax'),'wave':('wave_slope',),
              'ball':('ball_g',),'ufo':('ufo_g','ufo_v0')}[mode]
        counts=env.mode_cal.counts()
        return all(counts.get(k,0)>=3 for k in keys)
    # Robot/spider/swing have no accepted live parameter fitter yet. Do not
    # turn guessed physics for those modes into demonstration labels.
    return mode=='cube' and env.phys.n_cal>=3


def augment(observation,env):
    out=np.zeros(CONTROLLER_DIM,np.float32);out[:OBS_DIM]=observation
    inputs=getattr(env,'_last_inputs',None) or {}
    search=getattr(env,'_last_search',None) or {};plan=search.get('plan') or {}
    mode=env.mode_est.mode;dt=float(env._step_dt(peek=True))
    valid=bool(observation[3]>.5 and getattr(env,'_tracking_age',0)>=2
               and not getattr(env,'_death_frozen',False)
               and getattr(env,'_in_level',True)
               and not getattr(env,'_input_trace',{}).get('paused',False)
               and (env.mode_est.conf>=.6 or getattr(env.mode_est,'_icon_lock',0)>0))
    observed_dt=getattr(env,'_dt_frame',None)
    if observed_dt is None and getattr(env,'_obs_ts_prev',None) is not None:
        observed_dt=env._obs_ts-env._obs_ts_prev
    if observed_dt is not None and not .005<=observed_dt<=.1:valid=False
    # Explosion fragments can retain a confident player box while sparse
    # terrain provides no freeze vote. No pixel motion AND no valid scene
    # motion is insufficient evidence for teaching "held, but did not jump".
    if getattr(env,'_pixel_moving',None) is False and not getattr(env,'_calibration_valid',False):valid=False
    verified=bool(valid and plan.get('verified') and not search.get('path_unknown',False))
    trusted=bool(valid and calibrated(env,mode))
    uncertainty=getattr(env,'controller_uncertainty',{}).get(mode,(2.,10.))
    pending=list(inputs.get('pending',[]))[-8:]
    durations=[float(item[2]) for item in list(env.mode_est.hist)[-len(pending):]] if pending else []
    out[OBS_DIM:OBS_DIM+14]=[getattr(env,'_tracked_vy',0.)/20.,inputs.get('vy0',0.)/20.,
        dt,env._motion_dt(),float(inputs.get('grounded',False)),env.mode_est.lag/8.,
        env.mode_est.g,float(env.mode_est.mini),float(trusted),float(verified),float(valid),inputs.get('y0',0.)/8.,
        uncertainty[0]/8.,uncertainty[1]/20.]
    out[OBS_DIM+14:OBS_DIM+22]=pending+[0.]*(8-len(pending))
    out[OBS_DIM+22:OBS_DIM+30]=durations+[0.]*(8-len(durations))
    out[OBS_DIM+30]=float(inputs.get('eff_prev',False))
    out[CONTROLLER_MODE_START+MODE_ID[mode]]=1.
    for i,t in enumerate(TARGET_TIMES):
        point=route_point(plan,float(t))
        if point is not None:
            x,y,vy=point
            out[TARGET_START+i*4:TARGET_START+(i+1)*4]=[x/8.,y/8.,vy/20.,1.]
    # Existing analytical fixed-input rollouts provide a warm-start model
    # before the learned action-conditioned predictor earns trust.
    arcs={name:(hit,points) for name,hit,points in getattr(env,'last_arcs',[]) if name in ('hold','release')}
    for action,name in enumerate(('release','hold')):
        hit,points=arcs.get(name,(False,[]))
        if len(points)<2:continue
        times=np.array([p[0]/max(1.,env.speed or 10.38) for p in points])
        heights=np.array([p[1] for p in points])
        for j in range(HORIZON):
            t=(j+1)*dt;k=ANALYTIC_START+(action*HORIZON+j)*2
            y=float(np.interp(t,times,heights))
            # Match learned targets: mean velocity over this observation
            # interval, not instantaneous velocity at its endpoint.
            previous_y=float(np.interp(j*dt,times,heights))
            out[k:k+2]=[y/8.,(y-previous_y)/dt/20.]
            py=env.last_ents.player_pos()[1]
            visible=(py-1)/BLOCK_H+.5<=y<=py/BLOCK_H-.5
            out[VALID_START+action*HORIZON+j]=float(t<=times[-1]+1e-6 and
                (not hit or t<times[-1]-1e-6) and visible and valid)
    context=dict(plan=plan,verified=verified,calibrated=trusted,mode=mode,
                 first_action=plan.get('first_action'),tracking_valid=valid,uncertainty=uncertainty)
    return np.nan_to_num(np.clip(out,-2.,2.)),context


class RouteObservationEnv:
    """Keep legacy feature offsets; append route targets and measured state."""
    def __init__(self,env,tracking_weight=.05):
        from gymnasium import spaces
        object.__setattr__(self,'inner',env)
        object.__setattr__(self,'observation_space',spaces.Box(-2.,2.,(CONTROLLER_DIM,),np.float32))
        object.__setattr__(self,'tracking_weight',tracking_weight)
        object.__setattr__(self,'route_context',{})
        env.route_controller=True

    def __getattr__(self,name):return getattr(self.inner,name)

    def __setattr__(self,name,value):
        if name in ('inner','observation_space','tracking_weight','route_context'):
            object.__setattr__(self,name,value)
        else:setattr(self.inner,name,value)

    def reset(self,**kw):
        obs,info=self.inner.reset(**kw)
        obs,self.route_context=augment(obs,self.inner)
        return obs,info

    def step(self,action):
        previous=self.route_context
        start_y=self.inner.last_ents.player_pos()[1]
        obs,reward,terminated,truncated,info=self.inner.step(action)
        obs,self.route_context=augment(obs,self.inner)
        # Compare the actual next state to the OLD route in the OLD reference
        # frame. Never reward moving the target itself during replanning.
        target=route_point(previous.get('plan'),self.inner._motion_dt())
        cost=0.
        if (target is not None and previous.get('verified') and previous.get('calibrated')
                and self.route_context.get('tracking_valid') and not info.get('died')
                and previous.get('mode')==self.route_context.get('mode')):
            actual_y=(start_y-self.inner.last_ents.player_pos()[1])/BLOCK_H
            position_error=actual_y-target[1]
            start=route_point(previous.get('plan'),0.)
            expected_velocity=(target[1]-(start[1] if start is not None else 0.))/max(.005,self.inner._motion_dt())
            velocity_error=actual_y/max(.005,self.inner._motion_dt())-expected_velocity
            cost=-self.tracking_weight*min(1.,position_error**2+.02*velocity_error**2)*min(3.,self.inner._motion_dt()/(1/30))
            info=dict(info,route_position_error=position_error,route_velocity_error=velocity_error)
        info=dict(info,route_tracking_reward=cost)
        return obs,reward+cost,terminated,truncated,info


class BalancedReplay:
    """Reservoir quotas per mode and safe/failure continuation, not global FIFO."""
    def __init__(self,capacity,seed=0):
        self.quota=max(1,capacity//(len(MODES)*2))
        self.buckets=defaultdict(list);self.seen=defaultdict(int)
        self.rng=np.random.default_rng(seed)

    def __len__(self):return sum(map(len,self.buckets.values()))

    def add(self,row):
        key=(int(row['mode']),bool(row['success']))
        self.seen[key]+=1;bucket=self.buckets[key]
        if len(bucket)<self.quota:bucket.append(row)
        else:
            i=int(self.rng.integers(self.seen[key]))
            if i<self.quota:bucket[i]=row

    def sample(self,n):
        keys=[k for k,b in self.buckets.items() if b]
        if not keys:return []
        return self._sample(keys,n)

    def _sample(self,keys,n):
        result=[]
        for _ in range(n):
            key=keys[int(self.rng.integers(len(keys)))];bucket=self.buckets[key]
            result.append(bucket[int(self.rng.integers(len(bucket)))])
        return result

    def rows(self):return [r for b in self.buckets.values() for r in b]

    def state(self):return dict(quota=self.quota,buckets=dict(self.buckets),seen=dict(self.seen),rng=self.rng.bit_generator.state)

    def load(self,state):
        self.quota=state['quota'];self.buckets=defaultdict(list,state['buckets']);self.seen=defaultdict(int,state['seen'])
        self.rng.bit_generator.state=state['rng']
