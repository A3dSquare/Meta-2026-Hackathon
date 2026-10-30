"""Offline ship detection audit against existing labelled validation images.

Writes JSON and a contact sheet. Never trains, exports, or sends game inputs.
Validation frames may share source videos with training; this is not an
independent cross-level/generalization benchmark.
"""
import argparse
import json
from pathlib import Path
import statistics
import time


def iou(a,b):
    x=max(0.,min(a[2],b[2])-max(a[0],b[0]))
    y=max(0.,min(a[3],b[3])-max(a[1],b[1]))
    area=x*y
    return area/max(1e-12,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


def replay_ship_video(detector, out):
    """Check temporal continuity separately from the validation split.

    These labels were used for training too, so this is a replay sanity check.
    Unlabelled frames are not counted as ground-truth detections or misses.
    """
    import cv2
    video=Path('recs/ship.mp4');labels=video.with_suffix('.boxes.json')
    if not video.exists() or not labels.exists():return None
    data=json.loads(labels.read_text())
    expected={}
    for box in data['boxes']:
        if box['cls']==8:
            expected.setdefault(box['fi'],[]).append(tuple(box[k] for k in ('x0','y0','x1','y1')))
    capture=cv2.VideoCapture(str(video));detector.reset();rows=[]
    try:
        fi=0
        while True:
            ok,frame=capture.read()
            if not ok:break
            hi=cv2.resize(frame,(640,360))
            rgb=cv2.cvtColor(cv2.resize(frame,(320,180)),cv2.COLOR_BGR2RGB)
            pl=detector(rgb,hi).player()
            ship=pl is not None and getattr(pl,'mode',None)=='ship'
            box=(pl.x-pl.w/2,pl.y-pl.h/2,pl.x+pl.w/2,pl.y+pl.h/2) if pl is not None else None
            gt=expected.get(fi,[])
            good=bool(ship and any(iou(box,b)>=.5 for b in gt))
            rows.append(dict(frame=fi,labelled_ship=bool(gt),ship_found=ship,
                             matched=good if gt else None,confidence=float(pl.conf) if pl else None))
            if gt and not good:cv2.imwrite(str(out/f'video_miss_{fi:05d}.jpg'),frame)
            fi+=1
    finally:capture.release()
    return dict(video=str(video),frames=len(rows),labelled_ship_frames=len(expected),
                labelled_ship_matches=sum(r['matched'] is True for r in rows),
                labelled_ship_misses=[r['frame'] for r in rows if r['matched'] is False],
                rows=rows,limitation='Training-source video at 5 fps; not independent validation or a live 60 fps tracking test.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='logs_boxes/ship_detection_audit')
    args=parser.parse_args()
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from detect_yolo import YoloOnlyDetector
    from gd_objects import ICON_CLASSES
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cases=[]
    for label in sorted(Path('yolo_ds/labels/val').glob('*.txt')):
        image=Path('yolo_ds/images/val')/(label.stem+'.jpg')
        if not image.exists():continue
        gt=[]
        for line in label.read_text().splitlines():
            cls,x,y,w,h=map(float,line.split())
            if int(cls) in ICON_CLASSES:
                gt.append((int(cls),(x-w/2,y-h/2,x+w/2,y+h/2)))
        cases.append((image,gt))
    device=0 if torch.cuda.is_available() else 'cpu'
    rows=[];summaries=[];tiles=[]
    for model_path in ('yolo_gd.engine','yolo_gd.pt'):
        model=YOLO(model_path,task='detect')
        for conf in (.35,.15):
            counts=dict(images=len(cases),ship_labels=0,ship_matches=0,ship_misses=0,
                        wrong_mode_matches=0,false_ship_boxes=0,matched_ship_boxes=0)
            times=[]
            for idx,(path,gt) in enumerate(cases):
                frame=cv2.imread(str(path));h,w=frame.shape[:2]
                start=time.perf_counter()
                result=model.predict(frame,imgsz=640,conf=conf,device=device,
                                     half=device!='cpu' and model_path.endswith('.pt'),verbose=False)[0]
                elapsed=(time.perf_counter()-start)*1000
                if idx>1:times.append(elapsed)
                boxes=result.boxes
                predictions=[(int(c),tuple((box/[w,h,w,h]).tolist()),float(p))
                             for c,box,p in zip(boxes.cls.cpu().numpy(),boxes.xyxy.cpu().numpy(),boxes.conf.cpu().numpy())
                             if int(c) in ICON_CLASSES]
                ships=[p for p in predictions if p[0]==8]
                expected=[b for c,b in gt if c==8]
                matched=set();details=[]
                for target in expected:
                    candidates=[(iou(target,p[1]),j,p) for j,p in enumerate(ships) if j not in matched]
                    score,j,p=max(candidates,default=(0.,-1,None),key=lambda z:z[0])
                    good=score>=.5
                    if good:matched.add(j)
                    wrong=any(c!=8 and iou(target,b)>=.5 for c,b,p in predictions)
                    counts['ship_labels']+=1
                    counts['ship_matches']+=int(good);counts['ship_misses']+=int(not good)
                    counts['wrong_mode_matches']+=int(not good and wrong)
                    details.append(dict(matched=good,iou=score,confidence=p[2] if p else None))
                counts['matched_ship_boxes']+=len(matched)
                counts['false_ship_boxes']+=len(ships)-len(matched)
                row=dict(image=str(path),model=model_path,threshold=conf,expected_ships=len(expected),
                         predicted_icons=predictions,matches=details,inference_ms=elapsed)
                rows.append(row)
                if model_path.endswith('.engine') and conf==.35 and expected:
                    tile=cv2.resize(frame,(480,270))
                    for target in expected:
                        x0,y0,x1,y1=target
                        cv2.rectangle(tile,(int(x0*480),int(y0*270)),(int(x1*480),int(y1*270)),(0,255,0),2)
                    for c,(x0,y0,x1,y1),p in predictions:
                        color=(255,255,0) if c==8 else (0,0,255)
                        cv2.rectangle(tile,(int(x0*480),int(y0*270)),(int(x1*480),int(y1*270)),color,1)
                        cv2.putText(tile,f'{ICON_CLASSES[c]} {p:.2f}',(max(0,int(x0*480)),max(12,int(y0*270)-3)),
                                    cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
                    title=np.zeros((28,480,3),np.uint8)
                    cv2.putText(title,path.stem+(' OK' if all(d['matched'] for d in details) else ' MISS'),
                                (4,19),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
                    tiles.append(np.vstack((title,tile)))
            counts['recall']=counts['ship_matches']/max(1,counts['ship_labels'])
            counts['precision']=counts['matched_ship_boxes']/max(1,counts['matched_ship_boxes']+counts['false_ship_boxes'])
            counts['mean_inference_ms']=statistics.mean(times) if times else None
            counts.update(model=model_path,threshold=conf)
            summaries.append(counts);print(json.dumps(counts),flush=True)
        del model
    detector=YoloOnlyDetector()
    pipeline=dict(ship_labels=0,player_found=0,correct_ship_mode=0,usable_by_shield=0)
    for path,gt in cases:
        expected=[b for c,b in gt if c==8]
        if not expected:continue
        frame=cv2.imread(str(path));hi=cv2.resize(frame,(640,360))
        rgb=cv2.cvtColor(cv2.resize(frame,(320,180)),cv2.COLOR_BGR2RGB)
        pl=detector(rgb,hi).player()
        pipeline['ship_labels']+=len(expected)
        if pl is not None:
            box=(pl.x-pl.w/2,pl.y-pl.h/2,pl.x+pl.w/2,pl.y+pl.h/2)
            good=any(iou(box,b)>=.5 for b in expected)
            pipeline['player_found']+=int(good)
            pipeline['correct_ship_mode']+=int(good and getattr(pl,'mode',None)=='ship')
            pipeline['usable_by_shield']+=int(good and getattr(pl,'mode',None)=='ship' and pl.conf>.6)
    print('PIPELINE',json.dumps(pipeline),flush=True)
    replay=replay_ship_video(detector,out)
    if replay:print('REPLAY',json.dumps({k:v for k,v in replay.items() if k!='rows'}),flush=True)
    if tiles:
        while len(tiles)%3:tiles.append(np.zeros_like(tiles[0]))
        sheet=np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)])
        cv2.imwrite(str(out/'ship_validation.jpg'),sheet)
    report=dict(summaries=summaries,pipeline=pipeline,replay=replay,rows=rows,
                limitation='Existing validation split; may share source videos with training. Not a generalization estimate.')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print('REPORT',str(out.resolve()),flush=True)


if __name__=='__main__':main()
