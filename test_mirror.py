"""Upside-down check: every mode, orb, pad and portal must give the same safety scores
with gravity flipped (the scene mirrored). Run: python test_mirror.py"""
import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import gd_physics as G
from gd_objects import EFFECT
p=G.Phys()
def flip(boxes): return [(a,b,-t,-bt) for a,b,bt,t in boxes]
def flip_sp(sp): return [(a,b,-t,-bt,k,fx,i,n) for a,b,bt,t,k,fx,i,n in sp]
scenes={
 'spike ahead':      dict(haz=[(3.3,3.7,-0.45,0.1)], solid=[]),
 'block step':       dict(haz=[], solid=[(3.0,6.0,-0.5,0.5)]),
 'low ceiling':      dict(haz=[], solid=[(-1.0,6.0,1.2,3.0)]),
 'pillar + spikes':  dict(haz=[(2.0,6.0,-0.5,-0.2)], solid=[(1.0,2.0,-0.5,1.0),(6.0,7.0,-0.5,1.0)]),
 'yellow pad':       dict(haz=[(3.5,7.0,-0.45,0.1)], solid=[], sp=[(2.0,3.0,-0.5,-0.2,'pad',EFFECT['pad_yellow'],0,'pad_yellow')]),
 'yellow orb':       dict(haz=[(3.5,7.0,-0.45,0.1)], solid=[], sp=[(2.5,3.5,1.0,2.0,'orb',EFFECT['orb_y'],0,'orb_y')]),
 'blue orb':         dict(haz=[(3.5,7.0,-0.45,0.1)], solid=[], sp=[(2.5,3.5,1.0,2.0,'orb',EFFECT['orb_b'],0,'orb_b')]),
 'green orb':        dict(haz=[(3.5,7.0,-0.45,0.1)], solid=[], sp=[(2.5,3.5,1.0,2.0,'orb',EFFECT['orb_green'],0,'orb_green')]),
 'black orb':        dict(haz=[(5.0,7.0,-0.45,0.1)], solid=[], sp=[(2.5,3.5,1.0,2.0,'orb',EFFECT['orb_black'],0,'orb_black')]),
 'spider orb':       dict(haz=[(3.5,7.0,-0.45,0.1)], solid=[], sp=[(2.5,3.5,1.0,2.0,'orb',EFFECT['orb_spider'],0,'orb_spider')]),
 'dash orb':         dict(haz=[(3.5,7.0,-0.45,0.1)], solid=[], sp=[(2.5,3.5,1.0,2.0,'orb',EFFECT['orb_dash_green'],0,'orb_dash_green')]),
 'blue pad':         dict(haz=[(3.5,7.0,-0.45,0.1)], solid=[], sp=[(2.0,3.0,-0.5,-0.2,'pad',EFFECT['pad_blue'],0,'pad_blue')]),
 'red pad':          dict(haz=[(3.5,9.0,-0.45,0.1)], solid=[], sp=[(2.0,3.0,-0.5,-0.2,'pad',EFFECT['pad_red'],0,'pad_red')]),
}
portal_scenes={
 'gravity portal':  [(3.0,4.0,-3.5,3.5,'g',0)],
 'mini portal':     [(3.0,4.0,-3.5,3.5,'size',True)],
 'speed portal':    [(3.0,4.0,-3.5,3.5,'speed',15.6)],
 'ship portal':     [(3.0,4.0,-3.5,3.5,'mode','ship')],
}
bad=0
for mode in ('cube','ship','ball','ufo','wave','robot','spider','swing'):
    row=[]
    for name,sc in scenes.items():
        sp=sc.get('sp')
        up=G.search_best(G.Body(0.0,0.0,1,mode not in ('ship','wave','swing','ufo'),mode,False,0,False),
                         sc['haz'],sc['solid'],-0.5,4.0,10.39,p,1/30,0,[],want_path=True,specials=sp)
        dn=G.search_best(G.Body(0.0,0.0,-1,mode not in ('ship','wave','swing','ufo'),mode,False,0,False),
                         flip(sc['haz']),flip(sc['solid']),-4.0,0.5,10.39,p,1/30,0,[],want_path=True,specials=flip_sp(sp) if sp else None)
        su={k:round(up[k],2) for k in ('1','00','01')}; sd={k:round(dn[k],2) for k in ('1','00','01')}
        ok = su==sd
        # the drawn line should be the mirror image too
        pu=[(round(x,1),round(y,1)) for x,y in up['path']]; pd=[(round(x,1),round(-y,1)) for x,y in dn['path']]
        okp = pu==pd
        if not (ok and okp): bad+=1; row.append(f"{name}: up {su} / upside-down {sd}{'' if okp else ' (line differs)'}")
    print(mode.ljust(7), 'all mirrored OK' if not row else ' | '.join(row))
for mode in ('cube','ship','ball','ufo','wave','robot','spider','swing'):
    row=[]
    for name,pt in portal_scenes.items():
        up=G.search_best(G.Body(0.0,0.0,1,mode not in ('ship','wave','swing','ufo'),mode,False,0,False),[(5.0,7.0,-0.45,0.1)],[],-0.5,4.0,10.39,p,1/30,0,[],want_path=True,portals=pt)
        dn=G.search_best(G.Body(0.0,0.0,-1,mode not in ('ship','wave','swing','ufo'),mode,False,0,False),flip([(5.0,7.0,-0.45,0.1)]),[],-4.0,0.5,10.39,p,1/30,0,[],want_path=True,portals=pt)
        su={k:round(up[k],2) for k in ('1','00','01')}; sd={k:round(dn[k],2) for k in ('1','00','01')}
        if su!=sd: bad+=1; row.append(f"{name}: {su} / {sd}")
    print(mode.ljust(7),'portals mirrored OK' if not row else ' | '.join(row))
print('mismatches:',bad)
