"""Measured cube arc fitting and independently checked online physics calibration."""
from collections import deque
import numpy as np


def fit_arc(points,gravity,jump_speed,rest=None,launch_window=None):
    """Current (height, velocity) from causal positions; never invent a jump.

    The newest position and interval must agree, not merely the average of
    an old arc. launch_window is fixed at the FIRST airborne interval.
    """
    if len(points)<2:return None
    times=np.asarray([p[0] for p in points],float)
    heights=np.asarray([p[1] for p in points],float)
    last_dt=times[-1]-times[-2]
    if last_dt<=0:return None
    # A launch at the very end of an interval can fit almost no movement
    # while inventing full jump velocity. Require observed motion evidence.
    measured_velocity=(heights[-1]-heights[-2])/last_dt-.5*gravity*last_dt

    def acceptable(predicted,velocity):
        return (abs(velocity-measured_velocity)<=5.
                and abs(predicted[-1]-heights[-1])<=.2
                and abs((predicted[-1]-predicted[-2])-(heights[-1]-heights[-2]))<=.15
                and np.max(np.abs(predicted-heights))<=.3)

    if rest is not None and launch_window is not None:
        # Times are relative to the actual last supporting observation,
        # even after the rolling position window has discarded that frame.
        best=None
        for speed in (jump_speed,0.):
            if speed and (heights[-1]-heights[-2])*np.sign(speed)<=.12:
                continue
            for tau in np.linspace(0.,launch_window,12):
                elapsed=np.maximum(0.,times-tau)
                predicted=rest+speed*elapsed-.5*gravity*elapsed**2
                velocity=speed-gravity*elapsed[-1]
                if acceptable(predicted,velocity):
                    error=float(np.mean((predicted-heights)**2))
                    if best is None or error<best[0]:best=(error,predicted[-1],velocity)
        if best is not None:return float(best[1]),float(best[2])
    # More history reduces box wobble, but it must still explain the most
    # recent interval. Shorten after a real contact/impulse, not before it.
    for n in sorted(set((min(8,len(points)),min(5,len(points)),3)),reverse=True):
        if n<3 or n>len(points):continue
        t=times[-n:]-times[-1];y=heights[-n:]
        design=np.stack((np.ones(n),t),axis=1)
        a,b=np.linalg.lstsq(design,y+.5*gravity*t*t,rcond=None)[0]
        predicted=a+b*t-.5*gravity*t*t
        if (abs(b-measured_velocity)<=5.
                and np.max(np.abs(predicted-y))<=.2 and abs(predicted[-1]-y[-1])<=.2
                and abs((predicted[-1]-predicted[-2])-(y[-1]-y[-2]))<=.15):
            return float(a),float(b)
    return None


class CubePhysicsLearner:
    """One vote per clean arc, including interrupted/platform-to-platform jumps.

    Fit on earlier observations and check a later observation. Parameter
    proposals must improve validation across at least three separate arcs.
    """
    def __init__(self):
        self.arcs=deque(maxlen=25);self.points=[];self.rest=None;self.key=None
        self.time=0.;self.first_dt=None;self.voted=False;self.accepted=0;self.updates=0
        self.suspended=False
        self.last={}

    def reset(self,rest=None,key=None):
        self.points=[];self.rest=rest;self.key=key;self.time=0.;self.first_dt=None;self.voted=False

    @staticmethod
    def gravity_error(arc,gravity):
        t,y,test_t,test_y=arc['t'],arc['y'],arc['test_t'],arc['test_y']
        a,b=np.linalg.lstsq(np.stack((np.ones_like(t),t),1),y+.5*gravity*t*t,rcond=None)[0]
        return abs(test_y-(a+b*test_t-.5*gravity*test_t**2))

    @staticmethod
    def launch_error(arc,gravity,speed):
        if arc['rest'] is None:return None
        taus=np.linspace(0.,arc['first_dt'],24)
        elapsed=np.maximum(0.,arc['t'][None,:]-taus[:,None])
        predictions=arc['rest']+speed*arc['scale']*elapsed-.5*gravity*elapsed**2
        index=int(np.argmin(np.mean((predictions-arc['y'][None,:])**2,axis=1)))
        dt=max(0.,arc['test_t']-taus[index])
        return abs(arc['test_y']-(arc['rest']+speed*arc['scale']*dt-.5*gravity*dt*dt))

    def observe(self,y,dt,grounded,gravity,mini,valid,physics,reset=False,rest=None):
        key=(gravity,mini)
        if not valid or y is None or not .005<=dt<=.1:
            self.reset();self.suspended=True;return
        if grounded:
            self.reset(float(y*gravity if rest is None else rest*gravity),key);self.suspended=False;return
        if reset or key!=self.key:
            self.reset(None,key)
        if self.suspended:return
        self.time+=dt
        if self.first_dt is None:self.first_dt=dt
        self.points.append((self.time,float(y*gravity)))
        if self.voted or len(self.points)<6:return
        points=np.asarray(self.points[-12:],float)
        if points[-1,0]-points[0,0]<.18:return
        train=points[:-1];t,y_train=train[:,0],train[:,1]
        c,b,a=np.polyfit(t,y_train,2)
        g=float(-2*c)
        test_t,test_y=points[-1]
        validation=abs(float(np.polyval([c,b,a],test_t)-test_y))
        if not 60.<=g<=130. or validation>.12 or np.max(np.abs(np.polyval([c,b,a],t)-y_train))>.12:return
        scale=physics.mini if mini else 1.
        speed=None
        if self.rest is not None:
            disc=b*b+2*g*(a-self.rest)
            if disc>0:
                impulse=float(np.sqrt(disc));launch_time=float((b-impulse)/g)
                if -.025<=launch_time<=self.first_dt+.025 and 14.<=impulse/scale<=26.:
                    speed=impulse/scale
        arc=dict(t=t.copy(),y=y_train.copy(),test_t=float(test_t),test_y=float(test_y),
                 rest=self.rest,first_dt=self.first_dt,scale=scale,g=g,speed=speed)
        self.arcs.append(arc);self.voted=True;self.accepted+=1
        physics.n_cal+=1  # validation counts even when current constants already fit
        self.last=dict(heldout_error=float(validation),accepted_arcs=self.accepted,updates=self.updates)
        if len(self.arcs)<3:return
        values=np.array([s['g'] for s in self.arcs]);proposal=float(np.median(values))
        old=float(np.mean([self.gravity_error(s,physics.cube_g) for s in self.arcs]))
        new=float(np.mean([self.gravity_error(s,proposal) for s in self.arcs]))
        changed=False
        if np.quantile(values,.75)-np.quantile(values,.25)<=12. and new<old*.9-.003:
            physics.cube_g=.85*physics.cube_g+.15*proposal;changed=True
        launches=[s for s in self.arcs if s['speed'] is not None]
        if len(launches)>=3:
            speeds=np.array([s['speed'] for s in launches]);proposal_v=float(np.median(speeds))
            old_v=float(np.mean([self.launch_error(s,physics.cube_g,physics.cube_v0) for s in launches]))
            new_v=float(np.mean([self.launch_error(s,physics.cube_g,proposal_v) for s in launches]))
            if np.quantile(speeds,.75)-np.quantile(speeds,.25)<=2. and new_v<old_v*.9-.003:
                physics.cube_v0=.85*physics.cube_v0+.15*proposal_v;changed=True
        if changed:self.updates+=1
        self.last.update(updates=self.updates,gravity_validation_before=old,gravity_validation_proposed=new,
                         gravity=float(physics.cube_g),jump_speed=float(physics.cube_v0))
