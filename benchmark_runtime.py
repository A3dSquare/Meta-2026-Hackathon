"""Offline CPU thread/capture-preprocessing benchmark; never sends game inputs."""
import argparse
import json
from pathlib import Path
import statistics
import time


def measure(fn, samples):
    for _ in range(10):fn()
    values=[]
    for _ in range(samples):
        start=time.perf_counter();fn();values.append((time.perf_counter()-start)*1000)
    values.sort()
    return dict(mean_ms=statistics.mean(values),p50_ms=statistics.median(values),
                p95_ms=values[min(len(values)-1,int(len(values)*.95))],max_ms=max(values))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=100)
    parser.add_argument('--output',default='logs_boxes/runtime_benchmark.json')
    args=parser.parse_args()
    import cv2
    import numpy as np
    import torch
    from capture import GDCapture
    from gd import ProgressReward
    cv_original=cv2.getNumThreads();torch_original=torch.get_num_threads()
    cap=cv2.VideoCapture('recs/stereo.mp4');ok,frame=cap.read();cap.release()
    if not ok:raise RuntimeError('recs/stereo.mp4 could not be read')
    source=cv2.resize(frame,(2560,1440));capture=GDCapture();reward=ProgressReward()
    def pixels():
        small=capture._to_rgb_small(source)
        capture._push(small);capture._stacked_obs()
        reward._frame_diff(small);reward._banner_present(small);reward.bar_fill(small)
    result=dict(original_threads=dict(opencv=cv_original,torch=torch_original),opencv={},torch={})
    try:
        for threads in sorted(set((1,2,4,cv_original))):
            cv2.setNumThreads(threads)
            result['opencv'][threads]=measure(pixels,args.samples)
        cv2.setNumThreads(1)
        from trainer import _QNet
        from env_boxes import OBS_DIM
        network=_QNet.build(OBS_DIM,2,[256,256]).eval();observation=torch.zeros((1,OBS_DIM))
        for threads in sorted(set((1,2,4,torch_original))):
            torch.set_num_threads(threads)
            with torch.no_grad():result['torch'][threads]=measure(lambda:network(observation),args.samples)
    finally:
        cv2.setNumThreads(cv_original);torch.set_num_threads(torch_original)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
