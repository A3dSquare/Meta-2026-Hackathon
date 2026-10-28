# Training Guide — cross-level GD box policy (v2)

The box policy sees **detector boxes, not pixels**: spikes, blocks, orbs, pads,
portals, player. It also gets the scroll **speed**, the icon's **size**, and
the last 8 steps of how the icon moved and what it pressed (that's how it
tells cube from ship and normal from upside-down gravity). The same brain can
be pointed at any level. Train it with `preview_boxes.py` (live overlay) or
`train_boxes.py` (headless).

**Brains from before the speed/mode update can't be resumed** (the AI sees
more numbers now). Start fresh once; it tells you if you pass an old one.

## 0. One-time setup

- GD windowed-fullscreen on the primary monitor, **foreground**, Auto-Retry ON,
  practice mode OFF, the level open.
- `pip install stable-baselines3 gymnasium opencv-python numpy torch dxcam pydirectinput`
- Run from the repo folder (it needs `./templates` for the % reader).

## 1. Sanity checks (no game needed, ~1 min each)

```
python test_detect.py --hard          # detector recall on synthetic frames
python test_env_boxes.py              # env wrapper + crash-time measurement on rendered frames
python sim_gd.py --steps 20000        # the learning loop on a simulated level
```

`test_env_boxes.py` must print PASS. `sim_gd.py` should get past the first spike
in most attempts within ~50 attempts.

## 2. Train

```
python preview_boxes.py                                    # fresh brain
python preview_boxes.py --model checkpoints/preview_boxes.pt     # resume
```

What happens: the agent plays at 30 decisions/s. When it dies, the death is
seen when the % resets (~1 s after the crash). The env measures when the
camera froze, **cuts the post-crash steps off**, and puts the death penalty
on the step that actually caused it. After a death, once enough new play has
been banked (128 steps by default), GD is paused (ESC), the network trains
on its replay buffer (up to 1000 gradient steps), it waits `--settle` seconds,
and resumes (SPACE).

Console:

```
[attempt 37] died after 3.4s  peak=12%  best=18%  (cut 31 post-crash steps, measured)
[update 21] after death: grad_steps=880 loss=0.21 q=4.1 eps=0.13 buffer=5210 ...
```

- `measured` = crash time found from the camera freeze. After 5 measured
  deaths it switches to `median` (the running median of those measurements,
  since GD's respawn delay is constant and single readings are noisy). If you mostly see
  `fallback`, the freeze isn't being detected on your setup: pass
  `--death-lag 1.0` (seconds from crash to % reset; time it once by hand).
- `eps` = exploration (random actions), decays 0.15 -> 0.01 over ~20k steps (~11 min of play).
- Learning curve: `logs_boxes/attempts.csv`, one row per attempt.

Window keys: `q`/ESC quit (saves), `p` pause/unpause (click back into GD
within 2 s after unpausing). Ctrl+C in the console also saves.

## 3. Other levels, same brain

```
python preview_boxes.py --model checkpoints/best_boxes.pt --eps 0.05
python train_general.py --levels stereo backontrack polargeist --rotate-min 10
```

`--eps 0.05` restarts exploration low, so the brain uses what it knows and
explores only a little. `train_general.py` resumes the same checkpoint for
every level (the old version started each level from scratch).

## 4. What good looks like

- First ~20 attempts: mostly dies at the first spike, `eps` ~0.15.
- By ~50-100 attempts: the first spike is cleared almost every attempt and
  `best` keeps climbing. (In the simulator, the Stereo-like level is
  completed within ~100 attempts.)
- Boxes in the overlay: red = spike (small and mini spikes too), gray =
  block/wall, green = player. The thick box is the nearest hazard ahead.

## 5. If it stalls

- **No `[attempt ...]` line when the cube dies**: the % reader missed the
  reset. Re-calibrate: `python pct.py --calibrate`.
- **Spikes not boxed in the overlay**: `python detect.py` (one live frame ->
  `detect_probe.png`) or `python detect.py --probe recs/<level>.mp4 --show`.
  Level art the heuristic can't read -> YOLO (below).
- **Lots of `fallback` lags**: see section 2.
- **Jumps constantly**: it only jumps when it's confident (`--jump-margin`,
  default 0.3) and pays a tiny cost per held step (`--hold-cost`, default 0.02).
  Raise `--jump-margin` to 0.5 for even fewer jumps; 0 turns it off.
- **Training pauses feel long**: `--replay-ratio 2` halves them; learning is
  a bit slower per attempt.
- **Player not found (no green box)**: the icon detector looks for a
  green/cyan icon by default. For another colour pass its hue range, e.g.
  `--icon-hue 20 35` (yellow), `8 20` (orange), `95 125` (blue),
  `140 170` (pink), `0 8` (red). The overlay header shows `icon WxH` (in
  blocks) when it's found.
- **Speed looks wrong** (header `speed 1.0x` at a 2x section): it's measured
  from how fast spikes/blocks scroll, so it needs some on screen.

## 5b. New mechanics (ship, speed, gravity, orbs)

`python sim_gd.py --level modes` trains on a simulated level with a cube
section, a ship corridor, a 2x speed section with an orb you must hit, and an
upside-down section. In the simulator it learns the ship corridor within ~100
attempts; expect the real game to be slower. Portal/orb/pad detection is
tuned on synthetic frames -- check the overlay labels on a real level.

## 6. YOLO upgrade (only if the heuristic misses a level's art)

1. `python label_boxes.py --rec recs/<level>.mp4` (~150 boxes/level, 3 levels)
2. `python detect_yolo.py --train --epochs 50` -> copy best.pt to `yolo_gd.pt`
3. `python preview_boxes.py --backend yolo`

## 7. Files

- `checkpoints/preview_boxes.pt` — latest brain (resume this)
- `checkpoints/best_boxes.pt` — brain at the best % so far (keep this safe)
- `logs_boxes/attempts.csv` — per-attempt survival / peak % / crash-lag
- v1 `.zip` box checkpoints are not compatible (the observation changed)
- `--algo ppo` still exists for comparison; DQN learns several times faster
  per attempt here, because every attempt is replayed many times instead of
  being used once.
- Pixel legacy: `preview.py --train`, `preview_model.zip` — per-level only.
