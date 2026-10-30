"""Replay cube tracking from snapshots; does not train or control the game."""
import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace
from env_boxes import EntityBoxEnv,preview_boxes
from detect import Entities,Entity
from gd_physics import Phys,inst_vy,surfaces_at


def replay(steps):
    env=EntityBoxEnv.__new__(EntityBoxEnv)
    env._air=[];env._rest_true=None;env.route_controller=False;env._calibration_valid=False
    env.mode_est=SimpleNamespace(mode='cube',g=1,mini=False)
    rows=[]
    for i,s in enumerate(steps):
        inputs=s.get('preview_inputs') or {}
        if s.get('mode')!='cube':env._air=[];env._cube_key=None;continue
        env.mode_est.g=s['g'];env.mode_est.mini=s['mini']
        env.phys=Phys(cube_v0=inputs.get('cube_v0',20.),cube_g=inputs.get('cube_g',95.))
        env._last_yb=s.get('y');env._motion_dt=lambda s=s:s.get('motion_dt',s['dt'])
        env._calibration_valid=bool(s.get('calibration_valid'))
        env._cube_supports=[]
        grounded=bool(inputs.get('grounded'))
        if s.get('player'):
            entities=Entities([Entity(*b[:5],pt=b[5]) for b in s['boxes']])
            _,solids,_=preview_boxes(entities,1.)
            fl,ce=surfaces_at(0.,solids,inputs.get('floor_rel'),inputs.get('ceil_rel'),.5)
            env._cube_supports=[s['y']+q for q in (fl if s['g']==1 else ce)]
            if grounded:env._rest_true=(s['y']+min(fl if s['g']==1 else ce,key=abs)) if (fl if s['g']==1 else ce) else s['y']
        fit=env._air_fit(SimpleNamespace(conf=1.) if s.get('player') else None,grounded)
        current=fit[1] if fit else inst_vy('cube',s['g'],s['mini'],s.get('vy',0.),
            inputs.get('eff_prev',False),grounded,s.get('motion_dt',s['dt']),env.phys)
        rows.append(dict(index=i,raw_velocity=s.get('vy',0.),old_velocity=inputs.get('vy0',0.),
                         new_velocity=current,new_offset=fit[0] if fit else 0.,grounded=grounded))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots',default='logs_boxes/sessions/20261030_044225_3d6e5163/deaths')
    parser.add_argument('--output',default='logs_boxes/cube_state_audit.json')
    args=parser.parse_args();rows=[];errors=[]
    for path in sorted(Path(args.snapshots).glob('death_*.json')):
        try:data=json.loads(path.read_text())
        except FileNotFoundError:continue
        steps=data['steps'][:data['crash_index']]
        for row in replay(steps):
            row['file']=str(path);rows.append(row)
            i=row['index']
            if i+1>=len(steps):continue
            a,b=steps[i:i+2];p=a.get('preview_inputs') or {};q=b.get('preview_inputs') or {}
            dt=b.get('motion_dt',b['dt']);old_dt=a.get('motion_dt',a['dt'])
            if (not a.get('player') or not b.get('player') or b.get('mode')!='cube' or row['grounded'] or q.get('grounded')
                    or a['g']!=b['g'] or a['mini']!=b['mini'] or not .005<=dt<=.1):continue
            gravity=p.get('cube_g',95.)*a['g']
            # Exclude observed contact/impulse intervals from a ballistic
            # audit. This check does not evaluate obstacle collision search.
            if abs(b['vy']-(a['vy']-gravity*(dt+old_dt)/2.))>8.:continue
            observed=b['y']-a['y']
            before=p.get('y0',0.)+row['old_velocity']*dt-.5*gravity*dt*dt
            after=row['new_offset']+row['new_velocity']*dt-.5*gravity*dt*dt
            errors.append((abs(before-observed),abs(after-observed)))
    wrong=[r for r in rows if r['raw_velocity']>8 and r['old_velocity']<-3 and not r['grounded']]
    report=dict(samples=len(rows),rising_predicted_falling_before=len(wrong),
        rising_predicted_falling_after=sum(r['new_velocity']<-3 for r in wrong),examples=wrong,
        limitation='Recorded state-estimation replay. Not a live collision or level-completion benchmark.')
    if errors:
        def summarize(values):
            values=sorted(values)
            return dict(median_blocks=statistics.median(values),p95_blocks=values[min(len(values)-1,int(len(values)*.95))])
        report['one_observation_free_flight']=dict(samples=len(errors),
            before=summarize([x[0] for x in errors]),after=summarize([x[1] for x in errors]))
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='examples'}))


if __name__=='__main__':main()
