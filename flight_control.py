"""Ship/wave trajectory margins and a stateful, collision-first selector."""
import math


def segment_hits_box(x0, y0, x1, y1, box, hx, bottom_half, top_half):
    """Swept inner rectangle vs a stationary rectangle, including thin hazards."""
    left, right, bottom, top = box
    enter, leave = 0., 1.
    for start, end, lo, hi in ((x0,x1,left-hx,right+hx),
                                (y0,y1,bottom-top_half,top+bottom_half)):
        lo += 1e-8; hi -= 1e-8
        delta = end-start
        if abs(delta) < 1e-12:
            if not lo < start < hi:
                return False
        else:
            a,b = (lo-start)/delta,(hi-start)/delta
            enter = max(enter,min(a,b)); leave = min(leave,max(a,b))
            if enter > leave:
                return False
    return enter <= leave


def flight_segment_hit(x0,y0,x1,y1,hazards,solids,half,gravity,curve_padding=0.,snap=.3):
    if any(segment_hits_box(x0,y0,x1,y1,h,half,half+curve_padding,half+curve_padding) for h in hazards):
        return True
    upper, lower = (.12,snap) if gravity == 1 else (snap,.12)
    return any(segment_hits_box(x0,y0,x1,y1,s,half-.05,
                               half-lower,half-upper) for s in solids)


def path_clearance(x,y,vy,mode,gravity,mini,hazards,solids,floor,ceil,speed,phys,half,target,decision_dt,ground_route=False):
    """Clearance includes room to recover; wave does not use ship inertia.

    Physical input delay is simulated by the planner's pending-action queue.
    A ship's braking distance uses 85% of fitted acceleration as a modest
    uncertainty allowance. A wave needs one further decision's travel room.
    """
    if mode not in ('ship','wave','swing'):
        if not ground_route:return 1.,target
        # Ground routes have no braking envelope and may touch supports.
        # Avoid allocating a filtered solids list and clamping scores that
        # cannot lower the current minimum in the beam's hottest loop.
        lower,upper=y-half,y+half
        score,physical=1.,target;reach=max(1.,speed*.3)
        for boxes,is_solid in ((hazards,False),(solids,True)):
            for left,right,bottom,top in boxes:
                if is_solid and left-half<=x<=right+half and (abs(lower-top)<=phys.snap+.05 or abs(upper-bottom)<=.12):continue
                if right<x-half:continue
                dx=max(0.,left-x-half)
                if dx>reach:continue
                gap=lower-top if lower>=top else bottom-upper if upper<=bottom else 0.
                candidate=(gap-0.)/target+dx/reach
                if candidate<score:score=max(0.,candidate)
                if dx<=.05:physical=min(physical,gap)
                if score==0. and physical==0.:return score,physical
        return score,physical
    scale = phys.mini if mini else 1.
    if mode in ('ship','swing'):
        up = (phys.ship_down if mode=='swing' else phys.ship_up if gravity == 1 else phys.ship_down)*scale*.85
        down = (phys.ship_down if mode=='swing' else phys.ship_down if gravity == 1 else phys.ship_up)*scale*.85
        stop_down = min(vy,0.)**2/(2*max(1.,up))
        stop_up = max(vy,0.)**2/(2*max(1.,down))
        if mode=='swing':
            stop_down+=max(0.,-vy)*decision_dt*2
            stop_up+=max(0.,vy)*decision_dt*2
    elif mode=='wave':
        stop_down = max(0.,-vy)*decision_dt
        stop_up = max(0.,vy)*decision_dt
    else:
        stop_down=stop_up=0.
        floor=ceil=None  # touching a supporting surface is valid for ground modes
        solids=[s for s in solids if not (s[0]-half<=x<=s[1]+half and
                (abs(y-half-s[3])<=phys.snap+.05 or abs(y+half-s[2])<=.12))]
    score, physical = 1., target
    reach = max(1.,speed*.3)
    for left,right,bottom,top in (*hazards,*solids):
        if right < x-half:
            continue
        dx = max(0.,left-x-half)
        if dx > reach:
            continue
        if y-half >= top:
            gap, stop = y-half-top, stop_down
        elif y+half <= bottom:
            gap, stop = bottom-y-half, stop_up
        else:
            gap, stop = 0.,0.
        score = min(score,max(0.,min(1.,(gap-stop)/target+dx/reach)))
        if dx <= .05:
            physical = min(physical,gap)
    for gap, stop in ((y-half-floor if floor is not None else None,stop_down),
                      (ceil-y-half if ceil is not None else None,stop_up)):
        if gap is not None:
            physical = min(physical,max(0.,gap))
            score = min(score,max(0.,min(1.,(gap-stop)/target)))
    return score, physical


def recovery_hint(body,hazards,solids,floor,ceil,speed,phys,half,target,decision_dt,pending,delay=None):
    """Project queued inputs before asking how much braking room remains."""
    if body.mode not in ('ship','wave'):
        return None
    from dataclasses import replace
    from gd_physics import advance
    b = replace(body)
    elapsed = 0.
    remaining = len(pending)*decision_dt if delay is None else max(0.,delay)
    for i,hold in enumerate(pending):
        # For a fractional delay the OLDEST pending command has the short
        # residual interval; the newest still occupies a full decision.
        step = min(remaining,max(0.,(remaining-(len(pending)-i-1)*decision_dt)))
        if step <= 0: break
        advance(b,bool(hold),step,speed,phys,([],[]))
        elapsed += step; remaining -= step
    if remaining > 1e-9:
        advance(b,b.prev_hold,remaining,speed,phys,([],[])); elapsed += remaining
    if abs(b.vy) < .1:
        return None
    falling = b.vy < 0
    gaps = []
    if falling and floor is not None: gaps.append(b.y-half-floor)
    if not falling and ceil is not None: gaps.append(ceil-b.y-half)
    x = speed*elapsed
    for left,right,bottom,top in (*hazards,*solids):
        if right < x-half or left > x+half+speed*.2:
            continue
        if falling and b.y >= top: gaps.append(b.y-half-top)
        if not falling and b.y <= bottom: gaps.append(bottom-b.y-half)
    if not gaps: return None
    scale = phys.mini if b.mini else 1.
    if b.mode == 'ship':
        acceleration = (phys.ship_up if (falling == (b.g == 1)) else phys.ship_down)*scale*.85
        required = b.vy*b.vy/(2*max(1.,acceleration))
    else:
        required = abs(b.vy)*decision_dt
    gap = min(gaps)
    if gap >= required+target:
        return None
    return dict(action=int(falling == (b.g == 1)),gap=gap,braking_distance=required,
                queued_seconds=elapsed,mode=b.mode)


class FlightShield:
    def __init__(self, switch_margin=.05):
        self.switch_margin = switch_margin
        self.override = None
        self.mode = None
        self.last = None

    def reset(self):
        self.override = self.mode = self.last = None

    def choose(self, brain, metadata):
        """Survival level first; clearance hysteresis never overrides collision safety."""
        self.last = None
        if not metadata or metadata.get('mode') not in ('ship','wave'):
            self.reset()
            return int(brain)
        from gd_physics import safety_level
        if self.mode != metadata['mode']:
            self.reset(); self.mode = metadata['mode']
        try:
            survival, quality = metadata['survival'], metadata['clearance']
            valid = all(math.isfinite(float(m[k])) for m in (survival,quality) for k in ('1','00','01'))
        except (KeyError,TypeError,ValueError):
            valid = False
        if not valid:
            self.reset(); return int(brain)
        levels = {k:safety_level(survival[k]) for k in survival}
        l1,l0 = levels['1'],max(levels['00'],levels['01'])
        q1 = quality['1']
        # When continuing to release is collision-safe, judge its clearance.
        # Taking max(00,01) here repeatedly postpones an urgent rescue by one
        # decision, because "wait, then thrust" always looks almost as good.
        q0 = quality['00'] if levels['00'] == l0 else quality['01']
        choice, reason = int(brain), 'brain'
        if l1 != l0:
            choice,reason = int(l1>l0),'collision safety'
        elif l1 == 0:
            p,w = survival['1'],max(survival['00'],survival['01'])
            if abs(p-w) > 1e-6:
                choice,reason = int(p>w),'longest survival'
            elif abs(q1-q0) > self.switch_margin:
                # A queued input can make every first action leave the
                # verified region at the same instant. Do not keep thrusting
                # into that boundary simply because survival times tie.
                choice,reason = int(q1>q0),'clearance at tied horizon'
        elif levels['00'] < l0 and l1 >= l0:
            choice,reason = 1,'tight response window'
        else:
            q = (q0,q1)
            hint = metadata.get('recovery')
            if hint and q[int(hint['action'])] >= max(q)-self.switch_margin:
                choice,reason = int(hint['action']),'braking envelope'
            else:
                preferred = int(q1>q0)
                gain = q[preferred]-q[int(brain)]
                threshold = self.switch_margin*.5 if self.override == preferred else self.switch_margin
                if preferred != int(brain) and gain > threshold:
                    choice,reason = preferred,'recovery clearance'
        self.override = choice if choice != int(brain) and reason == 'recovery clearance' else None
        self.last = dict(action=choice,brain=int(brain),reason=reason,press_clearance=q1,
                         release_clearance=q0,press_level=l1,release_level=l0)
        return choice


def teacher_action(metadata, decision, calibrated):
    """Only unambiguous, robustly safe interventions from verified physics.

    Failures within the predicted 1.5 s horizon are removed later at death.
    """
    if not calibrated or not metadata or not decision or decision['action'] == decision['brain']:
        return -1
    chosen = decision['action']
    level = decision['press_level'] if chosen else decision['release_level']
    other_level = decision['release_level'] if chosen else decision['press_level']
    gain = (decision['press_clearance']-decision['release_clearance'])*(1 if chosen else -1)
    return chosen if level == 2 and (other_level < 2 or gain >= .1) else -1
