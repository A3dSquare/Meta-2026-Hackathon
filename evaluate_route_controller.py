"""Re-run a route checkpoint's episode-held-out checks without playing or saving weights."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--output',default='logs_boxes/controller_offline_evaluation.json')
    args=parser.parse_args()
    import torch
    from trainer import TrainConfig
    from route_learner import RouteLearner
    from route_controller import CONTROLLER_DIM
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if 'route_controller_version' not in ck:raise SystemExit('This evaluator requires a route-controller checkpoint.')
    learner=RouteLearner(CONTROLLER_DIM,2,TrainConfig(route_controller=True,net_arch=tuple(ck['arch']),buffer_size=20_000))
    learner.load(args.checkpoint)
    report=learner.evaluate_candidate()
    report['limitation']='Offline held-out metrics only; deployment also requires the frozen live comparison. No weights were saved.'
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
