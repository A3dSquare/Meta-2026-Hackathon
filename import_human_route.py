"""Make YOUR recorded run the bot's remembered route (landmark replay in --play).

Live (20261103) the bot's own best Stereo Madness run pressed 59 times in the first 223 blocks
where a human run pressed 28: every extra jump is one more chance to die, and the 19 s wall
(press 54, 19% deaths at its best timing) is a jump the human never made. This puts a human
run's presses, placed by the same landmarks the bot replays from, in place of the bot's.

  1. python record_demonstrations.py --level stereo_madness
     F6 as the attempt starts (normal mode, not practice), play to the end, F7 on the
     level-complete screen, F9 to save.
  2. python import_human_route.py demonstrations/stereo_madness/<that folder>
  3. python preview_boxes.py ... --landmarks --play --ms-delay   (as usual)

Uses the longest successful clip unless --clip is given. Backs up the old landmarks.json to
landmarks.before_human_<time>.json first; copy it back to undo.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('recording', type=Path)
    p.add_argument('--clip', type=int, default=None, help='clip number (default: the longest successful one)')
    p.add_argument('--level', default=None, help='route folder name under routes/ (default: the only one there)')
    p.add_argument('--routes', default='routes')
    p.add_argument('--yolo-model', default='yolo_gd.pt')
    p.add_argument('--motion-model', default=None, help='motion_model.npz for the planner physics (optional)')
    p.add_argument('--device', default=None, help='YOLO device (default: GPU if there is one)')
    p.add_argument('--dry-run', action='store_true', help='show what it would remember; change nothing')
    p.add_argument('--partial', action='store_true',
                   help='the clip is NOT a full run to the end: only replace the presses it covers')
    args = p.parse_args()

    import cv2
    import landmark_route as LR
    from demonstrations import read_recording, make_observer
    from detect_yolo import YoloOnlyDetector
    from gd_physics import Phys, MotionLearner, ModeEstimator, ModeCalibrator, LaunchCalibrator, load_motion

    routes = Path(args.routes)
    level = args.level
    if level is None:
        dirs = [d.name for d in routes.iterdir() if d.is_dir() and (d / 'landmarks.json').is_file()]
        if len(dirs) != 1:
            p.error(f'pass --level: routes/ has {len(dirs)} level folders with landmarks ({dirs})')
        level = dirs[0]
    lm_path = routes / level / 'landmarks.json'
    if not lm_path.is_file():
        p.error(f'{lm_path} not found: run the bot on this level first (--landmarks) so it has a map')

    meta, frames, timeline = read_recording(args.recording)
    clips = [c for c in meta.get('clips', []) if c.get('result') == 'success']
    if not clips:
        p.error('no successful clips in this recording (F7 marks one)')
    clip = meta['clips'][args.clip] if args.clip is not None else max(clips, key=lambda c: c['end'] - c['start'])
    sel = [f for f in frames if f['clip'] == clip['id'] and clip['start'] <= f['t'] <= clip['end']]
    if len(sel) < 30:
        p.error('clip too short')
    inputs = [json.loads(l) for l in open(args.recording / 'inputs.jsonl')]
    edges, prev_down = [], False
    for x in inputs:
        d = bool(x.get('down'))
        if d != prev_down:
            edges.append((float(x['t']), d))
            prev_down = d
    print(f'clip {clip["id"][:8]}: {clip["end"] - clip["start"]:.1f} s, {len(sel)} frames, '
          f'{sum(1 for t, d in edges if d and clip["start"] <= t <= clip["end"])} presses')

    det = YoloOnlyDetector(args.yolo_model, device=args.device)
    try:
        from train_demonstrations import DetectionCache
        cache = DetectionCache(det, args.yolo_model)
    except Exception:
        cache = None
    phys = Phys()
    if args.motion_model:
        load_motion(args.motion_model, MotionLearner(), phys, ModeEstimator(), ModeCalibrator(), LaunchCalibrator())
    env = make_observer(det, phys)

    lm = LR.LandmarkRoute(args.routes, verbose=False)
    lm.clock_ok = True
    lm.begin_attempt(level)
    old_events, old_best = list(lm.best_events), float(lm.best_s or 0.0)

    def down_in(a, b):
        """Key state for the decision made on frame a: down at b, or tapped in between."""
        st = timeline.state(b)['down']
        return bool(st or any(a < t <= b and d for t, d in edges))

    def edge_age(a, b, want):
        for t, d in edges:
            if a < t <= b and d == want:
                return t - a
        return None

    # Start at the first frame with the icon in view (a recording begun in a menu would
    # otherwise dead-reckon through the menu), end at the last one (the level-complete screen
    # has no icon, and its mouse clicks are not jumps).
    seen = []
    t0 = None
    prev = None
    s_hist, lock_hist, n_used = [], [], 0
    prev_a = 0
    for k, f in enumerate(sel):
        ts = f['t']
        dt = ts - prev if prev is not None else 1 / 30
        if prev is not None and ts <= prev:
            continue
        img = cv2.imread(str(args.recording / f['file']))
        if img is None:
            continue
        rgb = cv2.cvtColor(cv2.resize(img, (320, 180)), cv2.COLOR_BGR2RGB)
        env.replay_dt = dt
        env.replay_time = ts
        env.base.cap.last_hires = img
        env._input_trace = {'paused': False}
        env._pixel_moving = True
        env._observe(rgb, bool(timeline.state(prev if prev is not None else ts)['down']))
        player = env.last_ents.player()
        has_player = player is not None and player.conf > .6
        if t0 is None:
            if not has_player:
                prev = ts
                continue
            t0 = ts
        seen.append((k, has_player))
        env._obs_ts = ts
        lm.observe_env(env, ts - t0)
        nxt = sel[k + 1]['t'] if k + 1 < len(sel) else ts + 1 / 30
        a = int(down_in(ts, nxt))
        age = edge_age(ts, nxt, bool(a)) if a != prev_a else None
        lm.record(a, ts - t0, age=age)
        prev_a = a
        s_hist.append((ts - t0, float(lm.tracker.s), bool(lm.locked), has_player))
        n_used += 1
        prev = ts
    if cache:
        cache.save()
    if not s_hist:
        sys.exit('the icon was never detected in this clip')
    last_seen = max(i for i, h in enumerate(s_hist) if h[3])
    t_end, s_end = s_hist[last_seen][0], s_hist[last_seen][1]
    first_lock = next((i for i, h in enumerate(s_hist) if h[2]), None)
    if first_lock is None:
        sys.exit('the landmarks never locked onto this run: is it the same level as routes/' + level + '?')
    corr = 0.0
    if first_lock > 0:
        # the lock's correction to dead reckoning, applied to the presses made before it
        t_a, s_a = s_hist[first_lock - 1][:2]
        t_b, s_b = s_hist[first_lock][:2]
        corr = s_b - (s_a + (lm.speed or LR.SPEED_1X) * (t_b - t_a))
    n_pre = int(lm.pre_lock_events)
    for ev in lm.events[:n_pre]:
        ev['s'] += corr
        ev['_e'] = ev['_d'] = None            # replayed by position (they were placed by dead reckoning)
        if ev.get('_rel') is not None:
            ev['_rel'] = (None, None)
    lm.pre_lock_events = 0
    print(f'icon from {s_hist[0][1]:.1f} to {s_end:.1f} blocks ({t_end:.1f} s); locked from {s_hist[first_lock][0]:.1f} s; '
          f'{n_pre} press(es) before the lock moved {corr:+.2f} blocks')

    # hold lengths: recorded in 30 fps frames, replayed in ~50 ms decisions (only used when a
    # release has no landmark to go by)
    fdt = float(np.median(np.diff([h[0] for h in s_hist]))) if len(s_hist) > 2 else 1 / 30
    for ev in lm.events:
        ev['steps'] = max(1, int(round(ev['steps'] * fdt / 0.05)))
        if ev.get('rel_steps'):
            ev['rel_steps'] = max(1, int(round(ev['rel_steps'] * fdt / 0.05)))

    human = [ev for ev in lm.events if ev['s'] <= s_end + 0.5]
    lo, hi = (human[0]['s'], human[-1]['s']) if human else (0, 0)
    bot_same = [e for e in old_events if lo - 0.5 <= e['s'] <= hi + 0.5]
    print(f'your run: {len(human)} presses between {lo:.0f} and {hi:.0f} blocks; '
          f"the bot's remembered run: {len(bot_same)} there ({len(old_events)} in all, to {old_best:.0f} blocks)")
    if args.dry_run:
        print('--dry-run: nothing changed')
        return

    stamp = time.strftime('%Y%m%d_%H%M%S')
    backup = lm_path.with_name(f'landmarks.before_human_{stamp}.json')
    shutil.copy2(lm_path, backup)
    old_stats, old_times = dict(lm.stats or {}), list(lm.times or [])
    old_ghost, old_ghost_stats = list(lm.ghost or []), dict(lm.ghost_stats or {})
    if args.partial:
        keep_tail = [(i, e) for i, e in enumerate(old_events) if e['s'] > s_end + 1.0]
    lm.best_events, lm.best_s, lm.times, lm.stats, lm.taboos = [], 0.0, [], {}, []
    ok, why = lm._commit(s_end)
    if not ok:
        shutil.copy2(backup, lm_path)
        sys.exit(f'not saved: {why}')
    if args.partial:
        # the bot's own presses after the clip, with what it learned about each (timing shifts,
        # presses handed to the brain) and its position -> time table
        n_h = len(lm.best_events)
        lm.best_events = list(lm.best_events) + [e for _, e in keep_tail]
        lm.stats = {str(n_h + j): old_stats[str(i)] for j, (i, _) in enumerate(keep_tail) if str(i) in old_stats}
        end_new = lm.times[-1][0] if lm.times else -1e9
        lm.times = list(lm.times) + [tuple(p) for p in old_times if p[0] > end_new]
        lm.best_s = max(old_best, s_end)
        # --ghost path: yours up to the clip's end, the bot's (if it has one) after it
        lm.ghost = list(lm.ghost) + [g for g in old_ghost if g[0] > s_end + 1.0]
        lm.ghost_stats = {z: v for z, v in old_ghost_stats.items() if int(z) * LR.GHOST_ZONE > s_end + 1.0}
        print(f'kept the bot\'s {len(keep_tail)} presses after {s_end:.0f} blocks')
    else:
        # a full run: never hand over to the brain before the end (it does ~1 s before best_s)
        lm.best_s = s_end + 20.0
    lm._pick_offsets()
    lm._save()
    print(f'path for --ghost: {len(lm.ghost)} points' + (f' ({lm.ghost[0][0]:.0f}-{lm.ghost[-1][0]:.0f} blocks)' if lm.ghost else ''))
    print(f'saved: {len(lm.best_events)} presses, {len(lm.entries)} landmarks -> {lm_path}\n'
          f'(old route backed up to {backup.name}; copy it back over landmarks.json to undo)')


if __name__ == '__main__':
    main()
