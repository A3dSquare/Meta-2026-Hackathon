import json, glob, statistics as st
from collections import defaultdict
from training_diagnostics import flight_errors

def q(v,p): return sorted(v)[min(len(v)-1,int(len(v)*p))] if v else None
def summ(v):
    v=sorted(abs(x) for x in v)
    return dict(n=len(v), bias=round(st.mean(v),3) if v else None,
                med=round(st.median(v),3) if v else None,
                p95=round(q(v,.95),3) if v else None,
                gt1=sum(x>1 for x in v), gt2=sum(x>2 for x in v))

base='logs_boxes/sessions/20261102_003110_852e5ff9/deaths/death_*.json'
allrows=[]
for path in sorted(glob.glob(base)):
    d=json.load(open(path))
    steps=d['steps'][:d['crash_index']]
    # enrich each indexed step with flight errors starting there
    errs={}
    for e in flight_errors(steps):
        errs.setdefault(e['index'],[]).append(e)
    for i,s in enumerate(steps):
        p=s.get('preview_inputs') or {}
        for e in errs.get(i,[]):
            e=dict(e)
            e['attempt']=d['attempt']; e['peak']=d['peak_pct']
            e['source']=p.get('state_source','missing')
            e['grounded_start']=bool(p.get('grounded'))
            e['lag']=p.get('lag')
            e['vy0']=p.get('vy0',0); e['rawvy']=s.get('vy',0)
            e['vydiff']=abs((p.get('vy0',0) or 0)-s.get('vy',0))
            allrows.append(e)

print('TOTAL',len(allrows))
for h in (1,3,5):
    print('H',h,summ([e['error'] for e in allrows if e['horizon']==h]))
print('\nBY SOURCE horizon5')
for src, rows in sorted(defaultdict(list,{k:[e for e in allrows if e['source']==k] for k in set(e['source'] for e in allrows)}).items()):
    print(src, summ([e['error'] for e in rows if e['horizon']==5]))
print('\nBY VYDIFF horizon5')
for name, fn in [('<=2',lambda e:e['vydiff']<=2),('2-5',lambda e:2<e['vydiff']<=5),('>5',lambda e:e['vydiff']>5)]:
    print(name,summ([e['error'] for e in allrows if e['horizon']==5 and fn(e)]))
print('\nBY INDEX BUCKET horizon5')
for lo,hi in [(0,10),(10,25),(25,50),(50,100),(100,999)]:
    print(lo,hi,summ([e['error'] for e in allrows if e['horizon']==5 and lo<=e['index']<hi]))
print('\nTOP BIG H5 SAMPLES')
for e in sorted([e for e in allrows if e['horizon']==5],key=lambda e:abs(e['error']),reverse=True)[:20]:
    print({k:e[k] for k in ('attempt','peak','index','seconds','error','source','lag','vy0','rawvy','vydiff')})

# Is the huge error explained by an in-window ballistic impulse/contact?
# flight_errors deliberately excludes a tail that ENDS grounded, but not an
# intermediate grounded/impulse sample. Proxy: any tail step grounded OR an
# observed vy discontinuity incompatible with gravity (same test cube audit uses).
import glob as _g, json as _j
imp_rows=[]; clean_rows=[]
for path in sorted(_g.glob(base)):
    d=_j.load(open(path)); steps=d['steps'][:d['crash_index']]
    byidx={}
    for e in flight_errors(steps): byidx.setdefault(e['index'],[]).append(e)
    for i,es in byidx.items():
        first=steps[i]; tail=steps[i+1:]
        impulse=any((s.get('preview_inputs') or {}).get('grounded') for s in tail)
        # observed velocity jumps from stored vy sequence
        seq=[first]+tail
        for a,b in zip(seq,seq[1:]):
            dt=b.get('motion_dt',b['dt']); g=(first.get('preview_inputs') or {}).get('cube_g',95.)*first.get('g',1)
            if abs(b['vy']-(a['vy']-g*dt))>8.: impulse=True
        for e in es:
            (imp_rows if impulse else clean_rows).append(e)
print('\nIN-WINDOW CONTACT/IMPULSE SPLIT')
for h in (1,3,5):
    print('h',h,'impulse',summ([e['error'] for e in imp_rows if e['horizon']==h]),'clean',summ([e['error'] for e in clean_rows if e['horizon']==h]))
