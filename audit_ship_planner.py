"""Replay saved ship observations through the planner without sending game inputs."""
import argparse
import json
from pathlib import Path
import time

from detect import Entities, Entity
from env_boxes import BLOCK_H, preview_boxes
import gd_physics as gp
from flight_control import FlightShield


def replay_step(s, beam=6):
    items=[]
    for cls,x,y,w,h,pt in s['boxes']:
        e=Entity(cls,x,y,w,h,1.); e.pt=pt;items.append(e)
    hz,solid,_=preview_boxes(Entities(items),s['speed']*1.5+1)
    p=gp.Phys();inp=s['preview_inputs']
    for k in ('ship_up','ship_down','ship_vmax','wave_slope'):
        if k in inp:setattr(p,k,inp[k])
    body=gp.Body(inp['y0'],inp['vy0'],s['g'],False,s['mode'],s['mini'])
    original=gp.search_best
    def search(*args,**kw):
        kw['beam']=beam
        return original(*args,**kw)
    gp.search_best=search
    try:
        py=s['player_pos'][1]
        return gp.robust_search(body,hz,solid,inp['floor_rel'],inp['ceil_rel'],s['speed'],p,inp.get('decision_dt',s['dt']),
                                inp['lag'],inp['pending'],want_path=True,flight_buffer=.75,
                                lag_alt=inp.get('lag_alt'),pending_long=inp.get('pending_long'),
                                flight_view=((py-1)/BLOCK_H,py/BLOCK_H))
    finally:gp.search_best=original


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot')
    parser.add_argument('--output',default='logs_boxes/ship_planner_audit.json')
    args=parser.parse_args();data=json.loads(Path(args.snapshot).read_text());rows=[]
    for i,s in enumerate(data['steps'][:data['crash_index']]):
        if s.get('mode')!='ship' or not s.get('player'):continue
        row=dict(index=i,y=s['player_pos'][1],results={})
        for beam in (6,12,24):
            start=time.perf_counter();r=replay_step(s,beam)
            f=r['flight'];action=FlightShield().choose(1,f)
            row['results'][beam]=dict(survival={k:r[k] for k in gp.SEARCH_KEYS},
                clearance=f['clearance'],root=f['preferred_root'],action_if_brain_press=action,
                visibility_limited=f.get('visibility_limited'),path_unknown=r.get('path_unknown'),
                ms=(time.perf_counter()-start)*1000)
        rows.append(row)
        if i>=34:print(json.dumps(row),flush=True)
    Path(args.output).write_text(json.dumps(dict(snapshot=args.snapshot,rows=rows,
        limitation='Observation replay, not a game rollout. Old snapshots omit alternate pending inputs and portal types; those use planner defaults.'),indent=2))


if __name__=='__main__':main()
