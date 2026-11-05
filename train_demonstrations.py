"""Train controller/motion from manual play and validate blue-line physics offline.

Writes a NEW output directory; never controls the game or overwrites the source brain.
Recordings are grouped by level, with whole clips reserved for validation.
"""
import argparse
import copy
import json
import hashlib
import io
import tempfile
from pathlib import Path
import numpy as np


class DetectionCache:
    """YOLO boxes per recorded frame, saved to disk. YOLO is ~all of the import time and its
    boxes depend only on the frame and the YOLO model, never on planner physics or label lead,
    so the "Rebuilding observations with validated planner physics" pass and every later
    --refresh reuse them instead of re-running YOLO on ~85k frames. Keyed by a hash of the
    exact frame pixels; the file is named after the YOLO model's contents + settings, so a
    retrained YOLO starts a new cache automatically."""
    def __init__(self,detector,model_path):
        yolo=detector.yolo
        h=hashlib.sha256()
        with open(getattr(yolo,'loaded_path',model_path),'rb') as f:
            for block in iter(lambda:f.read(1<<20),b''):h.update(block)
        h.update(f'{yolo.conf}|{yolo.imgsz}'.encode())
        self.path=Path('detection_cache')/f'{h.hexdigest()[:16]}.npz'
        self.boxes={};self.new=0;self.hits=0;self.misses=0
        if self.path.is_file():
            try:
                with np.load(self.path) as data:
                    keys,counts,rows=data['keys'],data['counts'],data['rows']
                ends=np.cumsum(counts);starts=ends-counts
                self.boxes={k.tobytes():rows[a:b] for k,a,b in zip(keys.reshape(-1,16),starts,ends)}
                print(f'[detection cache] {len(self.boxes)} frames of saved YOLO boxes ({self.path})',flush=True)
            except Exception as exc:
                print(f'[detection cache] ignoring unreadable {self.path}: {exc}',flush=True);self.boxes={}
        detect=yolo.detect
        def cached(bgr):
            img=np.ascontiguousarray(bgr[:,:,:3])
            key=hashlib.blake2b(img.tobytes(),digest_size=16,person=str(img.shape).encode()[:16]).digest()
            rows=self.boxes.get(key)
            if rows is None:
                self.misses+=1
                out=detect(img)
                self.boxes[key]=np.asarray([(c,x0,y0,x1,y1,p) for c,x0,y0,x1,y1,p in out],np.float32).reshape(-1,6)
                self.new+=1
                return out
            self.hits+=1
            # same types as YoloDetector.detect: int class, float32 coordinates, float conf
            return [(int(r[0]),r[1],r[2],r[3],r[4],float(r[5])) for r in rows]
        yolo.detect=cached
    def save(self):
        print(f'[detection cache] {self.hits} frames reused, {self.misses} detected',flush=True)
        if not self.new:return
        self.path.parent.mkdir(exist_ok=True)
        keys=list(self.boxes)
        rows=[self.boxes[k] for k in keys]
        temporary=self.path.with_name(self.path.stem+'.tmp.npz')
        np.savez(temporary,keys=np.frombuffer(b''.join(keys),np.uint8).reshape(-1,16),
                 counts=np.asarray([len(r) for r in rows],np.int64),
                 rows=np.concatenate(rows) if rows else np.zeros((0,6),np.float32))
        temporary.replace(self.path);self.new=0
        print(f'[detection cache] saved {len(keys)} frames -> {self.path}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('recordings',nargs='+',type=Path)
    p.add_argument('--model',default='checkpoints/route_controller.pt')
    p.add_argument('--motion-model',default='checkpoints/motion_model.npz')
    p.add_argument('--yolo-model',default='yolo_gd.pt')
    p.add_argument('--out',type=Path,required=True,help='new directory for trained brain, planner calibration and report')
    p.add_argument('--cycles',type=int,default=20,help='offline training cycles, at most 128 updates each')
    p.add_argument('--label-lead-ms',type=float,default=45.,
                   help='teach the key state this many ms after each frame: the bot acts on a frame ~30-60 ms '
                        'after it was shown (timing_report.py: "observation age when a key goes out"). '
                        '0 = the old behaviour (presses learned this way come late live)')
    p.add_argument('--refresh-demonstrations',action='store_true',
                   help='rebuild human replay from all previously imported recordings; preserves weights and gameplay replay')
    p.add_argument('--no-detection-cache',action='store_true',
                   help='always re-run YOLO on every frame (default: reuse boxes saved in detection_cache/ for this exact YOLO model)')
    args=p.parse_args()
    if args.cycles<1:p.error('--cycles must be positive')
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        p.error('--out must be new or empty (protects existing checkpoints)')
    if not Path(args.motion_model).is_file():
        p.error(f'Physics calibration not found: {args.motion_model}. '
                'Pass --motion-model with your existing motion_model.npz; '
                'training will not silently reset calibrated physics or input timing.')
    from demonstrations import read_recording,replay_clip,fit_planner,valid_runs,clip_quality
    recordings=[];training=[];validation=[]
    entries=[]
    from demonstration_storage import discover_recordings,grouped_clips
    try:
        sessions=[(folder,*read_recording(folder)) for folder in discover_recordings(args.recordings)]
        clips=grouped_clips(sessions)
    except ValueError as exc:
        p.error(str(exc))
    for folder,meta,clip,frames,timeline,heldout in clips:
        selected=[f for f in frames if f['clip']==clip['id'] and clip['start']<=f['t']<=clip['end']]
        reason=('fewer than 30 recorded frames' if len(selected)<30 else
                'capture lacks exact frame timestamps' if any(not f.get('timestamp_exact') for f in selected) else None)
        if reason:
            identity=meta['id']+'/'+clip['id']
            entries.append(dict(id=identity,held_out=heldout,frames=len(selected),status='skipped',reason=reason))
            print(f'Skipping {identity}: {reason}',flush=True)
            continue
        recordings.append((folder,meta,clip,selected,timeline,heldout))
    import torch
    from trainer import TrainConfig
    from route_controller import CONTROLLER_DIM
    from route_learner import RouteLearner
    from gd_physics import Phys,MotionLearner,ModeEstimator,ModeCalibrator,LaunchCalibrator,load_motion,save_motion
    from detect_yolo import YoloOnlyDetector
    # Read once: the live bot may atomically save the source while this job
    # trains. Architecture, weights and fingerprint must refer to one snapshot.
    source_bytes=Path(args.model).read_bytes();source_digest=hashlib.sha256(source_bytes).hexdigest()
    source=torch.load(io.BytesIO(source_bytes),map_location='cpu',weights_only=False)
    cfg=TrainConfig(route_controller=True,device='cpu',cpu_threads=1,net_arch=tuple(source.get('arch',(256,256))),
                    buffer_size=20000,lr=1e-4,warmup=32,batch_size=128,max_grad_steps=128,replay_ratio=.5)
    learner=RouteLearner(CONTROLLER_DIM,2,cfg)
    with tempfile.TemporaryDirectory() as temporary:
        snapshot=Path(temporary)/'source.pt';snapshot.write_bytes(source_bytes)
        learner.load(str(snapshot))
    del source_bytes,source
    if args.refresh_demonstrations:
        available={meta['id']+'/'+clip['id'] for _,meta,clip,_,_,_ in recordings}
        missing=set(learner.demo_imports)-available
        if missing:
            p.error(f'Cannot refresh: {len(missing)} previously imported clips were not supplied or failed preflight. '
                    'Supply the complete recording library; no brain was saved.')
        for replay in (learner.replay,learner.validation):replay.clear_human()
        learner.demo_imports=[]
    # A frozen trial describes the old candidate. Updating it from human data
    # requires another offline/live comparison; never manufacture promotion.
    learner.trial=learner.trial_model=None;learner.trial_modes=set();learner.trial_skip_until_reset=False
    physics=Phys();motion=MotionLearner();est=ModeEstimator();cal=ModeCalibrator();launch=LaunchCalibrator()
    if not load_motion(args.motion_model,motion,physics,est,cal,launch):
        raise ValueError('Could not load source planner calibration')
    detector=YoloOnlyDetector(args.yolo_model)
    cache=None if args.no_detection_cache else DetectionCache(detector,args.yolo_model)
    args.out.mkdir(parents=True,exist_ok=True)
    fresh=[]
    def progress():
        temporary=args.out/'import_progress.tmp'
        temporary.write_text(json.dumps(dict(recordings=entries),indent=2))
        temporary.replace(args.out/'import_progress.json')
    progress()
    for n,(folder,meta,clip,frames,timeline,heldout) in enumerate(recordings):
        identity=meta['id']+'/'+clip['id']
        if identity in learner.demo_imports:
            print('Already imported:',identity);continue
        print(f'Observing demonstration {n+1}/{len(recordings)} ({"validation" if heldout else "training"}, {len(frames)} frames)',flush=True)
        rows=replay_clip(folder,frames,timeline,detector,physics,lead=args.label_lead_ms/1000.)
        quality=clip_quality(rows)
        entries.append(dict(id=identity,held_out=heldout,**quality));progress()
        if quality['status']=='skipped':
            print(f'Skipping {identity}: {quality["reason"]}; {quality["usable"]}/{quality["frames"]} reliable frames; {quality["rejected_observations"]}',flush=True)
            continue
        (validation if heldout else training).append(rows)
        fresh.append((identity,heldout,rows))
    if not training or not validation:
        raise ValueError(f'After filtering, need fresh training AND held-out clips. See {args.out / "import_progress.json"}; no controller was saved.')
    if cache:cache.save()
    updated,planner_report=fit_planner(training,validation,physics)
    # Persist accepted physics through the same loader used by physics_features
    # and robust_search. Manual capture timing must NOT replace the bot delay.
    if any(planner_report[k]['accepted'] for k in ('cube_g','cube_v0')):updated.n_cal=max(3,updated.n_cal)
    for name in cal.PARAMS:
        if planner_report.get(name,{}).get('accepted'):
            cal.samples[name].clear();cal.samples[name].extend([float(getattr(updated,name))]*3)
    save_motion(str(args.out/'motion_model.npz'),motion,updated,est,cal,launch)
    # Rebuild planner features using the accepted physics for both sets. The
    # fitted parameters came only from training clips; validation chose whether
    # to accept them and remains excluded from gradient training.
    if any(v['accepted'] for v in planner_report.values()):
        rebuilt=[]
        for folder,meta,clip,frames,timeline,heldout in recordings:
            identity=meta['id']+'/'+clip['id']
            if identity in {x[0] for x in fresh}:
                print('Rebuilding observations with validated planner physics:',identity,flush=True)
                rows=replay_clip(folder,frames,timeline,detector,updated,lead=args.label_lead_ms/1000.)
                quality=clip_quality(rows)
                entry=next(e for e in entries if e['id']==identity)
                entry['recalibrated_quality']=quality
                if quality['status']=='skipped':
                    entry.update(status='skipped',reason='recalibrated observations: '+quality['reason'])
                    print(f'Skipping {identity} after recalibration: {quality["reason"]}',flush=True)
                else:rebuilt.append((identity,heldout,rows))
                progress()
        fresh=rebuilt
        if cache:cache.save()
    if not any(heldout for _,heldout,_ in fresh) or not any(not heldout for _,heldout,_ in fresh):
        raise ValueError('No usable training/held-out pair remains after recalibration; see import_progress.json. No controller was saved.')
    for identity,heldout,rows in fresh:
        while (learner.episode_id%5==0)!=heldout:learner.episode_id+=1
        count=0
        for run in valid_runs(rows):
            observations=[r['obs'] for r in run]
            # Import human commands, never substitute planner labels. A marked
            # successful section is a truncation, not a terminal level victory.
            actions=[r['action'] for r in run[:-1]]
            rewards=[.1*min(3.,max(.005,run[i+1]['t']-run[i]['t'])/(1/30)) for i in range(len(actions))]
            count+=learner.add_segment(observations[:-1],actions,rewards,observations[-1],False,
                                      completed=True,human_demo=True,
                                      motion_valid=[r['motion_valid'] for r in run[:-1]],
                                      human_labels=[r['human_label'] for r in run[:-1]])
        if not count:raise ValueError(f'{identity}: no continuous valid demonstration segments')
        learner.episode_id+=1;learner.episode_open=False
        learner.demo_imports.append(identity)
    before=learner.evaluate_candidate();stats=[]
    data_audit=learner.data_audit()
    for cycle in range(args.cycles):
        stats.append(learner.train(256,queue_live_trial=False))
        print(f'Training {cycle+1}/{args.cycles}: {stats[-1]}',flush=True)
    learner.save(str(args.out/'route_controller.pt'))
    report=dict(recordings=entries,source_model=str(Path(args.model).resolve()),source_sha256=source_digest,
                planner=planner_report,controller_before=before,controller_after=learner.last_evaluation,
                training=stats,live_promotion=False,data_audit=data_audit,
                refreshed_demonstrations=args.refresh_demonstrations,
                limitations=['Successful section labels are user marks, not verified level completions.',
                    'Mixed-input intervals, tracking gaps and uncertain timing are excluded.',
                    'Planner fitting covers cube jump/gravity, ship acceleration/speed and wave slope; collision geometry and YOLO are unchanged.',
                    'Offline improvement is not proof of better live completion; controller still requires live comparison.'])
    from demonstration_report import annotate_controller_thresholds
    annotate_controller_thresholds(report, cfg.controller_eval_min)
    (args.out/'report.json').write_text(json.dumps(report,indent=2))
    print('Saved candidate and planner calibration:',args.out)
    print('Accepted planner parameters:',', '.join(k for k,v in planner_report.items() if v['accepted']) or 'none (existing physics retained)')
    print(f'Run: python preview_boxes.py --backend yolo-only --route-controller --no-route --clearance-reward --model "{args.out / "route_controller.pt"}" --motion-model "{args.out / "motion_model.npz"}"')


if __name__=='__main__':main()
