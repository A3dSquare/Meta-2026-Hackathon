"""Record manual Geometry Dash play and timestamped left-button input; never sends inputs.

F6 begins a run, F7 ends a successful section, F8 discards a run, F9 saves/exits.
Record at least five separate successful runs for episode-held-out validation.
"""
import argparse
import json
from pathlib import Path
import queue
import threading
import time
import uuid


class InputRecorder:
    """Timestamp state changes with polling uncertainty and focus changes."""
    def __init__(self, path, reader=None):
        self.path=Path(path);self.reader=reader;self.stop=threading.Event()
        self.markers=queue.Queue();self.error=None

    def run(self):
        from capture_clock import CaptureClock
        clock=CaptureClock()
        try:
            if self.reader is None:
                import ctypes
                import winfocus
                query=ctypes.windll.user32.GetAsyncKeyState
                self.reader=lambda: (bool(query(1)&0x8000),winfocus.gd_is_foreground() is True,
                                     [key for key in (0x75,0x76,0x77,0x78) if query(key)&0x8000])
            previous=None;previous_keys=set();last=time.perf_counter()
            with self.path.open('x') as f:
                while not self.stop.is_set():
                    down,focus,keys=self.reader();now=time.perf_counter();gap=now-last
                    state=(bool(down),bool(focus))
                    if state!=previous or gap>.01:
                        f.write(json.dumps(dict(t=now,down=state[0],focus=state[1],uncertainty=gap))+'\n')
                        f.flush()
                    for key in set(keys)-previous_keys:self.markers.put((now,key))
                    previous=state;previous_keys=set(keys);last=now
                    clock.sleep(.002)
        except BaseException as exc:self.error=exc
        finally:clock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='demonstrations')
    p.add_argument('--level',required=True)
    p.add_argument('--fps',type=int,default=30,choices=(30,60))
    args=p.parse_args()
    import cv2
    import numpy as np
    from capture import GDCapture,CaptureConfig
    from demonstration_storage import create_session
    folder=create_session(args.out,args.level)
    meta=dict(version=1,id=folder.name,level=args.level,fps=args.fps,input='left mouse button',
              clock='perf_counter seconds',clips=[],dropped_frames=0)
    def save_meta():
        temporary=folder/'meta.tmp';temporary.write_text(json.dumps(meta,indent=2))
        temporary.replace(folder/'meta.json')
    save_meta()
    inputs=InputRecorder(folder/'inputs.jsonl');thread=threading.Thread(target=inputs.run,daemon=True)
    pending=queue.Queue(maxsize=120);errors=[]
    def writer():
        try:
            with (folder/'frames.jsonl').open('x') as f:
                while True:
                    item=pending.get()
                    if item is None:break
                    record,frame=item
                    if not cv2.imwrite(str(folder/record['file']),frame,[cv2.IMWRITE_JPEG_QUALITY,95]):
                        raise OSError('Could not write demonstration frame')
                    f.write(json.dumps(record)+'\n');f.flush()
        except BaseException as exc:errors.append(exc)
    worker=threading.Thread(target=writer,daemon=True)
    cap=GDCapture(CaptureConfig(target_fps=args.fps))
    active=None;index=0;last_ts=None;timestamp_warning=False
    print('Manual input only. Stop the AI bot before recording. Use LEFT CLICK to play.')
    print('F6: begin run BEFORE starting/resuming play; F7: successful section; F8: discard/death; F9: save/exit.')
    print('Mark before each restart. Record at least five successful runs across sessions of this level. Output:',folder)
    thread.start();worker.start()
    try:
        cap.start()
        while True:
            if inputs.error:raise RuntimeError('Input recorder failed') from inputs.error
            if errors:raise RuntimeError('Frame writer failed') from errors[0]
            quit_requested=False
            while not inputs.markers.empty():
                now,key=inputs.markers.get()
                if key==0x75:
                    if active:active.update(end=now,result='discarded')
                    active=dict(id=uuid.uuid4().hex,start=now,result='unfinished')
                    meta['clips'].append(active);print('Recording run',len(meta['clips']))
                elif key in (0x76,0x77):
                    if active:
                        active.update(end=now,result='success' if key==0x76 else 'discarded')
                        print('Run marked',active['result']);active=None
                elif key==0x78:quit_requested=True
                save_meta()
            if quit_requested:break
            cap.read();arrival=time.perf_counter();ts=cap.last_frame_ts
            exact=cap.last_timestamp_exact and ts is not None and abs(ts-arrival)<.5
            if not exact and not timestamp_warning:
                print('WARNING: exact capture timestamps unavailable. These frames cannot be used for controller training.')
                timestamp_warning=True
            ts=float(ts) if exact else arrival
            if active is None or ts<active['start'] or ts==last_ts:continue
            last_ts=ts
            frame=cap.last_hires
            if frame is None:
                # never drop every frame silently (a capture change once left last_hires empty:
                # 7 clips recorded inputs and 0 frames, found only when training skipped them all)
                full=cap.last_fullres
                if full is None:continue
                frame=cv2.resize(np.ascontiguousarray(np.asarray(full)[:,:,:3]),(640,360),interpolation=cv2.INTER_AREA)
            record=dict(index=index,t=ts,arrival=arrival,timestamp_exact=exact,
                        clip=active['id'],file=f'frames/{index:08d}.jpg')
            try:pending.put_nowait((record,frame.copy()));index+=1
            except queue.Full:meta['dropped_frames']+=1
    except KeyboardInterrupt:pass
    finally:
        if active:active.update(end=time.perf_counter(),result='unfinished')
        inputs.stop.set();thread.join(timeout=3);cap.stop()
        while worker.is_alive():
            try:pending.put(None,timeout=.1);break
            except queue.Full:
                if errors:break
        worker.join(timeout=30);save_meta()
    if errors or inputs.error or worker.is_alive():raise RuntimeError('Recording incomplete; inspect files before training')
    print('Saved',folder,'— successful runs:',sum(c['result']=='success' for c in meta['clips']))


if __name__=='__main__':main()
