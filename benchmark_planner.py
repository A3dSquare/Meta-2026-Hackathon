"""Read-only recorded-state planner benchmark; never sends gameplay inputs."""
import argparse
import json
import statistics
import time
import math
from pathlib import Path


def equivalent(a,b):
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):return a==b or math.isclose(a,b,rel_tol=0.,abs_tol=1e-12)
    return a==b


def run_case(s,planner=None):
    from detect import Entities,Entity
    from env_boxes import preview_boxes,special_boxes
    from gd_physics import Body,Phys,robust_search
    q=s['preview_inputs'];e=Entities([Entity(*b[:5],pt=b[5]) for b in s['boxes']])
    h,solid,_=preview_boxes(e,20.)
    return (planner or robust_search)(Body(q['y0'],q['vy0'],s['g'],q['grounded'],s['mode'],s['mini'],prev_hold=q['eff_prev']),
        h,solid,q['floor_rel'],q['ceil_rel'],s['speed'],Phys(cube_v0=q['cube_v0'],cube_g=q['cube_g']),
        q['decision_dt'],q['lag'],q['pending'],pending_long=q['pending_long'],lag_alt=q['lag_alt'],
        want_path=True,explicit_plan=True,specials=special_boxes(e),flight_view=q['flight_view'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',default='logs_boxes/planner_timing_head_fixed_baseline.json')
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--output',default='logs_boxes/planner_timing_optimized.json')
    parser.add_argument('--reference-module',help='Saved pre-change physics module for paired same-process timings')
    args=parser.parse_args();baseline=json.loads(Path(args.baseline).read_text())
    if args.repeats<1:parser.error('--repeats must be positive')
    changed=[];samples=[];reference_samples=[];reference=None
    if args.reference_module:
        import importlib.util,sys
        spec=importlib.util.spec_from_file_location('_planner_reference',args.reference_module)
        reference=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reference;spec.loader.exec_module(reference)
    for i,(case,expected) in enumerate(zip(baseline['cases'],baseline['outputs'])):
        result=json.loads(json.dumps(run_case(case)))
        if not equivalent(result,expected):changed.append(i)
        for _ in range(args.repeats):
            # Alternate order to reduce systematic warm-cache/load bias.
            order=((reference.robust_search,reference_samples),(None,samples)) if reference else ((None,samples),)
            if _%2:order=order[::-1]
            for fn,dest in order:
                start=time.perf_counter();run_case(case,fn);dest.append((time.perf_counter()-start)*1000.)
    def summary(v):
        return dict(mean_ms=statistics.mean(v),median_ms=statistics.median(v),
                    p95_ms=sorted(v)[min(len(v)-1,int(.95*len(v)))])
    report=dict(cases=len(baseline['cases']),changed_cases=changed,baseline=summary(baseline['ms']),
                optimized=summary(samples),limitation='Recorded CPU search only; live end-to-end timing must be measured after restart.')
    if reference_samples:
        report['paired_reference']=summary(reference_samples)
        report['paired_speedup_percent']=100.*(1.-statistics.mean(samples)/statistics.mean(reference_samples))
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    return bool(changed)


if __name__=='__main__':raise SystemExit(main())
