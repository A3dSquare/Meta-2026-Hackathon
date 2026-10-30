"""Read-only live capture timing comparison. Does not send keyboard inputs."""
import argparse
import json
from pathlib import Path
import time

from capture import CaptureConfig,GDCapture
from training_diagnostics import Timings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=90)
    parser.add_argument('--output',default='logs_boxes/capture_benchmark.json')
    args=parser.parse_args();results=[]
    for prefetch in (False,True):
        for work_ms in (0.,18.):
            cap=GDCapture(CaptureConfig(prefetch=prefetch))
            try:
                cap.start()
                for _ in range(15):cap.read()
                profile=Timings();cap.profile=profile
                before=cap.health
                start=time.perf_counter()
                for _ in range(args.samples):
                    tick=time.perf_counter();cap.read()
                    profile.add('read_total',time.perf_counter()-tick)
                    # Exercise Python/GIL contention like the trajectory
                    # search, rather than yielding the CPU with sleep().
                    until=time.perf_counter()+work_ms/1000.
                    while time.perf_counter()<until:pass
                elapsed=time.perf_counter()-start
                after=cap.health
                result=dict(prefetch=prefetch,simulated_work_ms=work_ms,samples=args.samples,
                            simulated_work_kind='Python busy loop',
                            elapsed_s=elapsed,drops=after['drops']-before['drops'],
                            overruns=after['overruns']-before['overruns'],
                            stages=profile.summary())
                results.append(result);print(json.dumps(result),flush=True)
            finally:cap.stop()
    Path(args.output).write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
