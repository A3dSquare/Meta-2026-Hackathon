"""Dependency-free session logging and recorded free-flight physics checks."""
import csv
import json
import math
import os
import statistics
import sys
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict


def file_sha256(path):
    """Streaming fingerprint compatible with Python versions before file_digest."""
    import hashlib
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def terrain_scroll(previous, current, expected_shift=None):
    """Independent scroll evidence from repeated hazard/block boxes, not icons.

    Returns None if there are too few comparable static objects. Small box
    jitter is tolerated; unrelated rows/sizes cannot vote for a camera freeze.
    """
    shifts = []
    # Repeated floor tiles can make the nearest box belong to the NEXT tile,
    # especially when a slow decision moves the scene more than .04 screen
    # widths. Match near the predicted LEFTWARD translation instead of zero.
    old=[tuple(q[:5]) for q in previous if q[0] in (0,6)]
    now=[tuple(q[:5]) for q in current if q[0] in (0,6)]
    if len(old)>=3 and old==now:return 0.
    used = set()
    for c, x, y, w, h, *_ in current:
        if c not in (0, 6):
            continue
        matches = [(abs(x-q[1]-(expected_shift or 0.)), i, q) for i, q in enumerate(previous)
                   if i not in used and q[0] == c and abs(q[2] - y) < .008
                   and abs(q[3] - w) < .015 and abs(q[4] - h) < .015
                   and ((abs(q[1]-x)<.04) if expected_shift is None else
                        (-.12<=x-q[1]<=.002 and abs(x-q[1]-expected_shift)<.03))]
        if matches:
            _, i, q = min(matches, key=lambda v: v[0])
            used.add(i)
            shifts.append(x - q[1])
    return statistics.median(shifts) if len(shifts) >= 3 else None


def select_death_lag(raw, source, history):
    # A median of decision COUNTS is not a reliable clock: processing rate and
    # the delay until the percent reader sees the respawn both vary per death.
    if source == 'measured' or not history:
        return raw, source
    return int(statistics.median(history)), 'median-fallback'


class SessionLog:
    fields = ['time', 'attempt', 'total_steps', 'survived_steps', 'peak_pct',
              'death_lag', 'lag_src', 'updates', 'route', 'cause', 'session_id']

    def __init__(self, root, cfg, model_path):
        self.id = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        self.directory = os.path.join(root, 'sessions', self.id)
        os.makedirs(self.directory, exist_ok=True)
        with open(os.path.join(self.directory, 'settings.json'), 'w') as f:
            json.dump(dict(session_id=self.id, started=time.time(), argv=sys.argv,
                           config=asdict(cfg), model_path=model_path), f, indent=2)
        self.file = open(os.path.join(self.directory, 'attempts.csv'), 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.fields)
        self.file.flush()

    def writerow(self, row):
        self.writer.writerow(list(row) + [self.id])

    def event(self, name, data):
        with open(os.path.join(self.directory, name + '.jsonl'), 'a') as f:
            f.write(json.dumps(dict(time=time.time(), session_id=self.id, **data)) + '\n')


class Timings:
    def __init__(self):
        self.samples = defaultdict(lambda: deque(maxlen=600))

    def add(self, name, seconds):
        self.samples[name].append(seconds * 1000)

    def summary(self):
        result = {}
        for name, values in self.samples.items():
            if values:
                ordered = sorted(values)
                result[name] = dict(n=len(ordered), mean_ms=statistics.mean(ordered),
                                    p95_ms=ordered[math.ceil(.95 * len(ordered)) - 1])
        self.samples.clear()
        return result


def supported_ground(resting, vy, gravity, surfaces, tolerance=.25):
    """Stationary at an apex is not grounded. Require a nearby resting surface.

    Surface coordinates already include the player's half-height. Negative
    gravity-relative velocity means rising, which rules out landing inference.
    """
    near = any(abs(q) < tolerance for q in surfaces)
    return near and (resting or gravity * vy <= 1.5)


def inferred_surface(player_y, rest_y, block_h, gravity, grounded, platform, mode):
    """A cached resting height is not a movable floor under a flying icon.

    In platform scenes only detected blocks establish support; never extend a
    guessed resting height through a gap. Heights on the wrong side of the
    icon indicate a stale screen-space estimate (e.g. after camera movement).
    """
    if rest_y is None or platform:
        return None
    distance = gravity * (player_y - rest_y) / block_h - .5
    resting_here = (grounded and mode not in ('ship', 'wave', 'swing')
                    and abs(player_y - rest_y) < .3 * block_h)
    if resting_here:
        distance = -.5
    elif distance > -.5:
        return None
    return gravity * distance


def flight_errors(steps, crash_index=None, horizons=(1, 3, 5)):
    """Compare cube ballistic predictions to later detections, in blocks.

    Only uninterrupted airborne segments are eligible; contacts, mode changes,
    large gaps and post-crash pictures are excluded. This measures combined
    model/tracking error, not ground-truth accuracy or collision correctness.
    """
    end = len(steps) if crash_index is None else min(len(steps), crash_index + 1)
    for i, first in enumerate(steps[:end]):
        p = first.get('preview_inputs') or {}
        if first.get('mode') != 'cube' or not first.get('player') or p.get('grounded', True):
            continue
        if first.get('y') is None or p.get('vy0') is None:
            continue
        for h in horizons:
            if i + h >= end:
                continue
            tail = steps[i + 1:i + h + 1]
            if any(not s.get('player') or s.get('mode') != 'cube'
                   or s.get('g', 1) != first.get('g', 1)
                   or s.get('mini', False) != first.get('mini', False)
                   or (s.get('preview_inputs') or {}).get('grounded', True)
                   or not 0 < s.get('motion_dt', s.get('dt', 0)) <= .1 for s in tail):
                continue
            t = sum(s.get('motion_dt', s['dt']) for s in tail)
            predicted = first['y'] + p.get('y0', 0) + p['vy0'] * t - .5 * first.get('g', 1) * p.get('cube_g', 95) * t*t
            yield dict(index=i, horizon=h, seconds=t, predicted=predicted,
                       observed=tail[-1]['y'], error=tail[-1]['y'] - predicted)


def audit(paths, output):
    results = []
    for path in paths:
        with open(path) as f:
            d = json.load(f)
        errors = list(flight_errors(d['steps'], d.get('crash_index')))
        results.append(dict(file=path, attempt=d.get('attempt'), peak_pct=d.get('peak_pct'), errors=errors))
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    for h in (1, 3, 5):
        es = [abs(e['error']) for r in results for e in r['errors'] if e['horizon'] == h]
        if es:
            print(f'{h}-step free flight: n={len(es)}, median={statistics.median(es):.3f} blocks, max={max(es):.3f}')
    print('Old logs use smoothed dt; new logs record motion_dt. Errors include detector noise.')


if __name__ == '__main__':
    import argparse
    import glob
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', help='Snapshot glob; default: legacy and session death snapshots')
    parser.add_argument('--output', default='physics_audit.json')
    args = parser.parse_args()
    paths = glob.glob(args.snapshots) if args.snapshots else (
        glob.glob('logs_boxes/deaths/death_*.json') + glob.glob('logs_boxes/sessions/*/deaths/death_*.json'))
    if not paths:
        parser.error('No death snapshots found')
    audit(sorted(paths), args.output)
