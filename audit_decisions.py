"""Inspect earlier decisions in saved deaths without touching training state."""
import argparse
import json
from pathlib import Path
from death_cause import trace_decisions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('deaths',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=[]
    for path in sorted(args.deaths.glob('death_*.json')):
        data=json.loads(path.read_text())
        rows.append(dict(file=path.name,attempt=data.get('attempt'),peak_pct=data.get('peak_pct'),
                         trace=trace_decisions(data['steps'],data['crash_index'])))
    args.output.write_text(json.dumps(rows,indent=2))
    print(f'Examined {len(rows)} deaths; wrote {args.output}')


if __name__=='__main__':main()
