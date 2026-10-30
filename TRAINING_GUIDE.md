# Training Guide — cross-level GD box policy (v2)

### Tracking availability and the visible flight area

A frozen-terrain vote is no longer latched through frames with no comparable
terrain, or through ambiguous shifts. Coherent pixel motion vetoes that vote.
This fixes a regression where a live ship in an empty room was hidden from
the controller while its last hold continued. Snapshots now distinguish
`player_raw` (before the freeze filter) from `player`, and record
`pixel_moving` alongside the terrain/freezing evidence.

The flight planner also uses the captured viewport as a **soft visibility
limit** for clearance and recovery. It does not feed these limits into the
physical landing/collision surfaces: no imaginary floor or ceiling is added.
Holding into unobserved space therefore loses clearance preference even when
no ceiling objects have been detected. This is not a screen-centre reward;
positions with adequate visible headroom remain equivalent. The limits are
logged as `flight_search.visibility_bounds`. A constant-hold controller is
covered by an empty-room closed-loop regression test.

## Buffered flight shield and learning to use thrust

Ship/wave now use a trajectory-based safety selector by default. It checks
survival in the measured and alternate-timing/faster-scroll worlds first;
clearance cannot overrule a safer survival level. Among surviving options,
it prefers recovery-adjusted clearance from lower AND upper hazards/solids
and known boundaries. The desired soft buffer is **0.75 blocks**; narrow
corridors can still be traversed when that margin is unattainable. Use
`--flight-buffer 0` to restore the ordinary collision-only flight shield, or
choose a value up to 2 blocks.

The planner keeps first-action histories separate so one action cannot
inherit another action's clearance. Near-term clearance receives greater
weight than distant empty space. A two-decision wait is checked separately
from "wait, then thrust", preventing a rescue from being postponed every
frame. Thin hazards are checked against swept flight segments rather than
only the endpoint. Ship chord checks include a small acceleration-curvature
allowance; wave segments use their linear motion.

Queued inputs are projected before computing a braking-envelope hint. Ship
braking room uses velocity squared divided by twice the opposing acceleration,
with 85% of the fitted acceleration as a conservative allowance. Wave uses
one decision's travel distance, not ship inertia. A hint is only considered
among options at the same survival level and cannot force a materially worse
clearance path. With similar safety/clearance the brain remains in control.
Clearance overrides require a score advantage above 0.05; an existing override
can persist down to 0.025 to limit switching. Collision vetoes bypass this
hysteresis immediately. Flight replay receives the same buffered protection;
cube/other-mode replay keeps its lighter existing shield.

**Separate thrust threshold:** `--flight-margin` defaults to **0** for ship
and wave. `--jump-margin` now applies to the other modes. The same distinction
is used in Double-DQN's next-action targets, not just inference. Thus keeping
`--jump-margin 1.0` no longer imposes that extra threshold on thrust.
Checkpoint dimensions are unchanged; existing brains can be resumed.

**Conservative coaching:** `--flight-teacher-weight` defaults to **0.05**
(range 0..0.2; 0 disables). A small hinge loss teaches the brain an unambiguous
safe shield action, rather than relying only on sparse death feedback. Labels
require a robustly surviving intervention and either a survival-level advantage
or a clearance advantage of at least 0.1. Ship ascent, descent and maximum
speed must EACH have three accepted calibration samples; wave needs three
slope samples. Input delay must be verified and current tracking/calibration
usable. Mere braking hints or near ties are not automatically taught.

Replay-forced/menu-suppressed actions are not coaching examples, and examples
inside the final 1.5 s before a death are discarded after crash-time trimming.
The unbanked rollout tail covers that horizon even at frame-skip 1. Coaching
is DQN-only and is disabled in `--play` or with the shield off. Unlike the
optional potential reward, this is an additional learning objective; its
quality depends on the model's recommendations, hence these gates and opt-out.
Cube/other-mode samples never receive the coaching loss.

Death snapshots record `flight_control` (brain choice, selected action and
reason), `flight_search` (per-first-action survival, clearance and recovery
hint), `flight_teacher` (candidate label, potentially discarded later), and
the fitted ship/wave parameters. Updates report `teacher_loss` and
`teacher_draws` (sample draws, not unique examples). Search time remains in
the timing log. A local replay benchmark on 27 saved ship states averaged
about 6.4 ms with the buffer versus 3.9 ms without; live decision timing and
completion improvements require a new run.

## Optional ship/wave clearance shaping

`--clearance-reward` enables a small geometry potential for ship and wave;
the flag alone uses weight **0.05**, or pass a value from 0 to 0.1, e.g.
`--clearance-reward 0.02`. Default is **off**. Both live entry points accept
it, including `--backend yolo-only`. No checkpoint dimensions or YOLO weights
change; existing brains can be resumed. `--play` ignores shaping and never
trains.

The score uses edge-to-edge clearance from reliable detected hazards and
solids ahead/overlapping the player. Objects up to about half a second ahead
have increasing influence as they approach. Portals, orbs, pads, objects
behind the player, low-confidence boxes and guessed floor/ceiling planes
do not contribute. Only confirmed surface lines contribute as boundaries.
Clearance saturates at 0.75 blocks; there is no screen-centre target, infinite
distance bonus, or preferred thrust direction. Missing/uncertain player or
mode tracking and absent relevant geometry have zero potential.

The added reward is exactly `gamma * Phi(next) - Phi(current)`, with Phi the
weighted score. Potentials are bounded and the difference is not separately
clipped. Inactive modes have zero Phi; mode/tracking boundaries retain the
necessary correction, rather than skipping one side and creating free reward.
Deaths and completions have zero terminal Phi. Death corrections apply at
the retained crash transition **after** delayed-death trimming. Replay
prefixes remain excluded from learning, and long-rollout cuts retain Phi
alignment. For a complete episode, discounted shaping sums to minus the
initial Phi: it redistributes feedback without changing the base objective.
This is a learning aid, not a guarantee of faster learning or safer flight.

Session settings and checkpoint metadata record the weight. Death snapshots
include the normalized `clearance` score, validity and mode; `clearance.jsonl`
contains per-attempt **retained-tail** accounting (long-run steps may already
have been banked). Timing reports include the clearance calculation.

To preserve the original brain while trying it:

```
python preview_boxes.py --backend yolo-only --model checkpoints/preview_boxes.pt --name clearance_test --clearance-reward --minutes 10
```

Compare with the flag absent using the same starting checkpoint and all
other settings held fixed. Keep jump-margin, hold-cost, route mode and level
the same in both runs. Compare ship survival, independent thrust choices,
clearance and death locations over multiple attempts. This shaping does not
train YOLO or directly calibrate the blue-line physics.

## Session diagnostics and raw YOLO viewer

Run `python preview_yolo.py` for a standalone live detection window. It shows
raw YOLO boxes, exact class names, confidence, and inference time, with no
policy, physics, heuristic icon fallback, route memory, or game inputs.
Use `--video recs/level.mp4` for a recording, `--conf .25` to change the
threshold, `--model yolo_gd.pt`, `--device cpu` or `--device 0`, and
`--region LEFT TOP RIGHT BOTTOM` to crop live capture. Q/Escape or closing the
window exits; Space pauses the viewer. Live capture defaults to monitor 0.

New training runs print their directory under `logs_boxes/sessions/`.
Each session has `settings.json` (arguments and training configuration),
`environment.jsonl` (effective backend/settings), `attempts.csv` (including
cause and session ID), `timing.jsonl` (mean and p95 stage times plus capture
wait/work estimates), and `summary.jsonl`. Attempt numbers are local to a
unique session; old logs are preserved. Death images and JSON live in that
session's `deaths/` folder. Survival counts include the replay prefix and
cannot go negative; replay steps remain excluded from brain learning.

The grounded check now requires nearby surface support, preventing a still
icon at a jump apex from becoming grounded or teaching a false floor height.
Accepted cube arc fits also stabilize the policy's velocity/history inputs;
raw positions and velocities remain available for calibration and diagnosis.
The observation dimensions and checkpoint format are unchanged.

`python training_diagnostics.py --output logs_boxes/physics_audit.json`
compares airborne cube predictions against detections 1, 3, and 5 decisions
later. It scans both legacy and session snapshots; use `--snapshots` with a
quoted glob to select one session. New snapshots include actual motion dt,
raw velocity, and tracked velocity. Per-death errors are also written to
`deaths/physics_errors.jsonl` when snapshots are enabled. These are combined
tracking/model residuals, not ground-truth collision tests; old snapshots use
smoothed dt. Contacts and mode changes are excluded, but unseen special-object
impulses or camera motion can still create outliers.

The twelve takeoff-fit candidates are evaluated as one vectorized operation,
preserving candidate selection and reducing small per-decision overhead.
Timing reports separate fixed plans from continuation search, plus detector,
brain/route, GUI, and environment time. `physics_total` includes plans and
search; these overlapping values must not be added together. Live timing and
completion improvements require a new game run to measure.

### Crash feedback and decision-boundary correction

Current camera-freeze measurements now take precedence over historical median
death delays. The median is only a fallback when the current attempt has no
measurable freeze. Explosion-object guesses cannot override a measured freeze
or an explicitly fixed delay. Multiple static terrain boxes provide independent
scroll evidence; after three frozen observations following active scrolling,
player detections are suppressed until terrain movement resumes. This keeps
green explosion effects/UI out of player-motion calibration. New snapshots
record `terrain_scroll` and `death_frozen` for inspection.

Both fixed plans and the continuation search now split their near-term time
grid at actual decision boundaries. This applies even with `--ms-delay` off:
the default still uses the measured number of delayed decisions. Previously
two 41 ms decisions could become 100 ms on the search's 33 ms grid instead of
82 ms, changing the predicted jump. The millisecond-delay estimator remains
opt-in; no brain or YOLO checkpoint format changes are required.

### Tracking gaps and input diagnostics

Missing player detections now interrupt cube, ship/mode, launch and motion
fits. The first reacquired position is a new baseline, not a one-frame jump
from the last sighting. Delayed action history continues advancing through
these gaps independently of position history. Training pause/resume also
breaks unfinished fits and clears the pre-pause action queue. Accepted physics
values and the brain remain intact.

Snapshots include `input` (requested/applied action, paused/focus state,
cached keyboard hold, OS key state when available, and resend count),
`tracking_age`, `calibration_valid`, `held_rest_steps`, and ship calibration
counts/delay verification. OS key state is not confirmation that GD accepted
an input. A sustained held-but-grounded cube produces an `[input]` diagnostic.
Focus is checked before a new key press; a lost focus invalidates the cached
hold. If Windows reports that a supposedly held SPACE is up, the real input
backend periodically re-sends keyDown. Normal uninterrupted holds remain
change-only. The trainer/route memory record the action actually passed through
the environment's input gate, including menu/paused suppression.

`detector.jsonl` records the selected model/engine path and SHA-256 fingerprint
per session. Restart the trainer after retraining/exporting YOLO: model weights
are loaded once, not hot-reloaded. CPU explicitly selects the weights rather
than a TensorRT engine.

For an isolated diagnostic run preserving the original brain and physics file:

```
python preview_boxes.py --backend yolo-only --model checkpoints/preview_boxes.pt --name diagnostic_boxes --motion-model none --minutes 10
```

This resumes the brain but writes `diagnostic_boxes.pt` and
`diagnostic_boxes_best.pt`. It starts from default physics, calibrates in
memory, and does not load or overwrite `motion_model.npz`. Route memory still
operates as usual. Inspect the new session before deciding to reset a brain.

### Inferred floors near gaps and in flight

An inferred resting height is no longer clamped to half a block under an
airborne player when the cached height is on the wrong side of the icon.
When there is no real ground line in a platform scene, detected blocks supply
support individually; the cached height cannot become an infinite floor
through a gap. Rejected estimates are cleared so they cannot reappear after
the icon leaves a block. The corresponding inferred-ceiling clamp is also
guarded. Snapshots record `floor_source` (`line`, `inferred`, `unknown`, or
`rejected inferred floor`) for checking the next live run. These conservative
checks can remove support predictions when terrain detection is incomplete;
they do not guarantee a level completion or prevent all repeated deaths.

### Platform-side contact while falling

Fixed plans and the continuation search now test the cube/robot's inner
hitbox at the instant its leading edge enters a detected block. This happens
before vertical landing snap: a cube already too low at that edge is marked
as a side crash instead of being lifted onto the top. Existing corner-landing
tolerance is retained, including upside-down robot/cube cases. Intervals
containing orb/pad events retain their existing event handling. This uses the
shared physics model, so it applies to YOLO-only mode without retraining YOLO.
The recorded 22% regression fixture changes a previously safe two-step wait
to an unsafe wait while retaining a surviving press-first prediction.

The box policy sees **detector boxes, not pixels**: spikes, blocks, orbs, pads,
portals, player. It also gets the scroll **speed**, the icon's **size**, and
the last 8 steps of how the icon moved and what it pressed (that's how it
tells cube from ship and normal from upside-down gravity). The same brain can
be pointed at any level. Train it with `preview_boxes.py` (live overlay) or
`train_boxes.py` (headless).

It also gets a **physics preview** every step, in **every game mode**: cube,
ship, ball, UFO, wave, robot, spider, mini versions, and **upside-down**
gravity. A mode recogniser works out which one it is from how the icon
actually moves when it presses or doesn't (plus the colour of a portal it just
crossed). The preview then simulates 8 short plans in that mode: don't press,
hold, tap now / in 2 / 4 / 6 steps, hold-then-release, release-then-hold.
For each it checks the boxes: does it hit something, how soon, with how much
room to spare, and when it would walk off the ledge. The cube's jump comes
from a real Stereo Madness recording (2.2 blocks high, 0.43 s in the air) and
keeps fine-tuning itself from your icon's jumps. It also measures the **input
delay** (how many steps between pressing SPACE and the icon reacting) and
times the paths with it. In the simulator the
recogniser gets the mode right ~99% of the time; the robot is only recognised
at its first jump (its portal is grey, so there's no colour hint). The overlay
draws the paths and shows the mode in the header (e.g. `ship UPSIDE-DOWN`),
plus `jump 2.2b/0.43s* delay 1` (the * appears once it has tuned the jump from
your own jumps; `delay` is the measured input delay in steps, 0-3).

**Learned correction (experimental, OFF by default):** a small network can learn
what the physics formulas get wrong from real play (`--motion-fix` turns it on).
On live data it made the "don't press" path float over the first spike, so the
shield trusted it -- keep it off unless you're testing it. The tuned jump and the
measured input delay are still saved in `checkpoints/motion_model.npz` (shared
by every brain and level; delete it to re-measure).

**Input delay:** measured from your own presses (steps from pressing to the icon
moving; the console prints `[input] measured input delay: N step(s)`). Live it
was 1 step. Since fix49 it is also measured in **milliseconds** (from when the
picture the press was decided on was shown to when the jump started, using the
jump's own arc), and with `--ms-delay` (opt-in since fix53: the runs right after it went in did worse)
the blue line / search use that exact value:
`[input] input delay 52 ms (1.56 steps; spread 45-60 ms)`. Before, 1.5 steps was
rounded up to 2, so the line expected every jump ~17 ms (about 0.2 blocks) later
than it really started. The "worse world" check still uses a delay at least a
whole step off, so the shield is as careful as before.

**Floating platforms (fix52):** the bright "ground line" is only used as the floor when
it isn't the top edge of a detected block, and standing on a platform no longer counts
as standing on the ground. Before, a platform's top became an endless floor, so
walking off the platform looked safe and it fell instead of jumping. Hazards lying
just under the floor (a floor covered in thorns) now make landing there deadly, and
YOLO no longer drops the block the icon is standing on.

**Safety shield:** before every action (chosen or random), a SEARCH asks
"is there any way to survive the next 1.5 s if I press now / if I wait?",
with every later decision played well (jump, land, jump again...). It checks
twice: as measured, and with the press arriving a step late and the level 4%
faster. Each choice gets a safety score: 1.00 = safe both times, 0.90-0.99 =
safe only if the timing is exactly as measured, below 0.85 = crashes. The
shield takes the safer choice; when both are equally safe the brain decides.
It learns from what was actually done, so it picks up the habit itself. The
`[update]` line shows `shield saved Nx (P% of steps)`, which should shrink as
it learns. (The old shield only compared 8 fixed one-press plans, which all
crash before any "jump, land, jump" pattern, so it had nothing to go on.) In
the simulator, an agent that never pressed by itself got 87% of the way
through the Stereo-like level on the shield alone (old shield: 34%).
`--no-shield` turns it off.

**Every mode's physics is measured from your game:** like the cube jump, the
ship's climb / fall rate and top speed, ball and UFO gravity, the UFO tap and
the wave angle are fitted from the icon's real motion (once the input delay is
measured), saved in `checkpoints/motion_model.npz` with the jump and the delay,
and used by the predictions from then on. Every 10 deaths the console prints
`[physics] jump ..., delay N; ship up .., ship down .., ...` (`?` = not enough
seen yet; a value is used after 3 clean pieces of motion). The mode itself only
changes at a portal now (without one it takes ~1.6 s of overwhelming evidence):
live, glowing pillars were taken for portals and turned the cube into a
"spider" / "robot".

**Floor:** read from the bright line GD draws on top of the ground (light blue
line in the overlay), not guessed from where the cube stood. The guess drifted
by up to 1.2 blocks after deaths froze the cube in mid-air. If a level has no
visible ground line it falls back to the guess.

**Future-proofing:** the brain has 12 spare inputs that are wired in but
switched off. New features can be added there later without starting a new
brain. **Brains from before this update (v4 and older) can't be resumed** --
start fresh this one time; it tells you if you pass an old one.

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

Keys: **F8 = pause / resume, F9 = save + quit** -- these work from ANY window
(GD is kept in front automatically, so the overlay rarely has focus). In the
overlay window `q`/ESC (quit, saves) and `p` (pause) also work. Space is GD's
jump key, so it can't be used. Ctrl+C in the console also saves.

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
- Paths in the overlay: thin magenta = keep holding (ship / wave / robot),
  dots = don't press,
  **thick cyan = the best line the search found** (waits as long as it safely
  can, so where it leaves the ground is the latest safe moment to jump). A red
  X marks where a path would hit something. Light blue = detected ground line.
  (The death snapshots still also draw green = tap now and yellow = tap in 4.)

## 4b. Death snapshots (what it saw before each crash)

Every death saves `logs_boxes/deaths/death_NNNNN.png`: 12 frames, 0.1 s apart,
ending just after the crash (red border). Each frame shows the boxes, the
predicted paths, what was pressed (`PRESS` / `-----`), `SHIELD(brain ...)`
when the shield overrode the brain, the mode, the input delay `dN`, the
brain's jump urge `Q`, and `safe? press X wait Y` = the search's safety
scores the shield used (1.00 = safe, 0.90-0.99 = safe only with exact timing,
below 0.85 = crashes; "only if press next" = waiting works only by pressing on
the very next step, so the shield presses now). The
`.json` next to it has the same numbers per step. The newest 30 deaths are
kept (`--death-snaps N` to change, 0 = off). Send a few of these when asking
why it dies somewhere.

**GD must be the active window.** Key presses go to whatever window is in
front. If the overlay, the console or anything else is active, SPACE never
reaches GD: the cube can't jump and the AI learns that jumping is useless.
The env now brings GD back to the front every ~2 s (and before pausing /
resuming), and prints `[input] WARNING: ... presses did NOTHING` if the icon
stops reacting to presses. If you see that, click into GD.

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
- **Header shows the wrong mode** for more than a second after a portal: the
  recogniser needs the icon to move (it can't tell modes apart while the icon
  sits on the floor). If it stays wrong while the icon flies around, send me a
  recording (`python record.py`).
- **Speed looks wrong** (header `speed 1.0x` at a 2x section): it's measured
  from how fast spikes/blocks scroll, so it needs some on screen.

## 5b. New mechanics (ship, speed, gravity, orbs)

`python sim_gd.py --level modes` trains on a simulated level with a cube
section, a ship corridor, a 2x speed section with an orb you must hit, and an
upside-down section. `--level allmodes` has one section per game mode (cube,
ship, ball, UFO, wave, robot, spider, then upside-down). In the simulator it learns the ship corridor within ~100
attempts; expect the real game to be slower. Portal/orb/pad detection is
tuned on synthetic frames -- check the overlay labels on a real level.

## 5c. Teach it objects by picture (and let deaths teach it)

`hazards/` holds pictures of objects (see `hazards.py`). Every other frame they
are searched for ahead of the icon; a match becomes a hazard box that the cyan
line, the shield and the brain treat like a spike.

- `python hazards.py --grab hazard` -- with GD paused and the object on screen,
  drag a tight box around something deadly the detector misses (a saw, a
  coloured spike...) and press ENTER. Training picks it up within ~10 s.
- `python hazards.py --grab safe` -- the same for something harmless that the
  detector calls a hazard (red box on decoration): its hazard boxes are dropped.
- **Learning hazards from deaths is OFF** (since fix51): it once learned a yellow
  jump pad as "deadly", and the shield then jumped over the pad onto the spikes
  1000+ times. `hazards/learned/` is no longer read (delete it). Pictures you
  grab yourself (above) still work.
- **How it died** (`death_cause.py`, every death): what it hit -- `spike`,
  `block side`, `head hit`, `unknown` (nothing detected there) -- and why:
  `blue line wrong` (the prediction said its move was safe 0.3 s before: physics
  or detection off), `chosen wrong` (a safe move existed, the brain / shield took
  the other), `trapped` (no safe move left by then: the mistake came earlier);
  `(shield acted)` if the shield overrode the brain. It is on each death
  snapshot's title, in its JSON (`"cause"`), after each `[attempt N]` line
  (`how: ...`), in attempts.csv's last column, and every 25 deaths as a tally:
  `[deaths] how the last 50 died: spike/blue line wrong 30, ...`. Lots of
  `blue line wrong` at one spot = send me that snapshot (physics); lots of
  `chosen wrong` = the brain is still learning; `unknown` = detection misses something.
- `python hazards.py --list` shows what's in use. `--hazards none` turns it off.

Spikes also carry which way they point now: hanging spikes (tip down) and
spikes on pillar sides get their hitbox at the BASE, like GD (before, every
spike got a standing spike's hitbox, shifted toward a hanging spike's tip).

## 6. YOLO: teach the detector a level's art (e.g. Deadlocked)

The built-in detector uses rules (black triangle + outline = spike). Heavily
decorated levels break those rules. YOLO learns from boxes you draw instead.
With `--backend yolo` it runs TOGETHER with the built-in detector and only
adds objects the built-in one missed -- so on levels that already work the
boxes (and the brain's behaviour) stay the same.

1. `pip install ultralytics`  (a CUDA GPU is strongly recommended; on CPU it
   only runs every 3rd frame and may still be too slow -- watch `[health]`)
2. Record: `python record.py --level deadlocked` (120 s, 5 fps -> recs/deadlocked.mp4)
3. Label: `python label_boxes.py --rec recs/deadlocked.mp4`
   - frames open pre-filled with the built-in detector's boxes; fix them
   - drag = box, right-click = delete, number = type, D = done + next
   - 0 hazard (anything deadly), 1/2/3 orbs, 4 pad, 5 portal, 6 solid
   - 40-60 done frames per level, spread over the whole level
4. Train: `python detect_yolo.py --train` -> `yolo_gd.pt` (all recs/*.boxes.json)
5. Check: `python detect_yolo.py --check recs/deadlocked.mp4` -> yolo_check_*.png
6. Use: add `--backend yolo` to preview_boxes.py / train_boxes.py
   (`--extra "--backend yolo"` for train_general.py)

### YOLO only (`--backend yolo-only`)

YOLO does ALL the detecting -- spikes, blocks, orbs, pads, portals and the ICON,
including which mode it's in (so the blue line switches to ship physics the
moment the icon turns into a ship, no portal needed). Kept from the built-in
detector: box tracking (smooth boxes, scroll speed) and, only on frames where
YOLO doesn't see the icon, the built-in icon finder.

- Label the icon too: ONE box around the whole icon, class = its mode:
  `7` cube, `8` ship, `9` ball, `U` ufo, `V` wave, `R` robot, `P` spider, `N` swing.
- Every orb / pad / portal type has its own class (gd_objects.py), so the blue
  line knows exactly what each does (a red orb's big jump, a 2x speed portal,
  a mini portal...). Keys step through a group -- press again for the next
  type; with the mouse over a box, the key changes that box:
  `O` orbs (yellow, pink, red, blue, green, black, dash green, dash pink,
  spider, teleport, toggle), `K` pads (yellow, pink, red, blue, spider),
  `T` mode portals, `G` gravity / size / mirror / dual / teleport portals,
  `E` speed portals (0.5x .. 4x). The top bar shows the type you're drawing.
  Aim for 15+ boxes of every type you use (training warns about rare ones).
  Mirror / dual / teleport portals and teleport orbs are recognised but their
  effect isn't simulated yet.
  New frames come pre-filled with a guessed icon box; old labelled frames
  show `NO ICON BOX (I)` -- press `I` on each to add it (training leaves out
  frames without an icon box once any frame has one).
- Train on the GPU: `python detect_yolo.py --train --device 0 --epochs 80`
  (`--base yolov8s.pt` = more accurate, still fast on a 5070 Ti)
- Check: `python detect_yolo.py --check recs/<level>.mp4 --n 12`: the icon is
  labelled with the mode YOLO thinks it's in.
- Use: `python preview_boxes.py --backend yolo-only --model checkpoints/preview_boxes.pt`
  (same brain: the inputs are the same kinds of objects).

Label more levels any time and re-run step 4; every level you label makes it
better on levels it has never seen. `recs/` is git-ignored -- back up the
`.boxes.json` files, they are your labelling work.
Limitation: the AI sees rectangles. Diagonal corridor walls (Deadlocked's
wave parts) don't fit boxes well; spikes, saws, monsters, blocks do.

## 7. Files

- `checkpoints/preview_boxes.pt` — latest brain (resume this)
- `checkpoints/best_boxes.pt` — brain at the best % so far (keep this safe)
- `checkpoints/motion_model.npz` — learned physics correction + tuned jump +
  input delay (shared by all brains; safe to delete, it relearns)
- `logs_boxes/attempts.csv` — per-attempt survival / peak % / crash-lag
- v1 `.zip` box checkpoints are not compatible (the observation changed)
- `--algo ppo` still exists for comparison; DQN learns several times faster
  per attempt here, because every attempt is replayed many times instead of
  being used once.
- Pixel legacy: `preview.py --train`, `preview_model.zip` — per-level only.

## Route memory (routes/): practising the spots it keeps dying at

On by default (`--no-route` turns it off). It recognises each level by how its start
looks -- no names needed -- and keeps one file per level in `routes/` forever:
the best run's presses, every spot it got stuck at, every change it tried there
and the one that got it through. When it has died 4 times at the same spot, every
OTHER attempt replays the best run up to just before the spot and tries the next
untried change there (a press a bit earlier / later, left out, held longer, an
extra tap). Coming back to a level weeks later continues where it left off.
Those practice attempts are never given to the brain to learn from and don't
count for best_boxes.pt; the console marks them `(route practice, not learned
from)`. Rename a level by editing `"name"` in its file. Back up `routes/` if you
care about it (it isn't the brain, it's per-level practice).

### --play: beat a level with what it knows (no training)

    python preview_boxes.py --backend yolo-only --model checkpoints/preview_boxes.pt --play

Nothing is learned and there are no training pauses. Every attempt follows the best
run saved for this level up to the spot it died at last; there it first lets the brain
try, then the route memory's next untried change, one per attempt; past that spot the
brain plays (no random moves). New progress is saved in routes/ as usual; the brain
file is not touched. The first attempt of a session is the brain's own while the level
is recognised.

### --frontier: train AND play (learn where it's stuck)

    python preview_boxes.py --backend yolo-only --model checkpoints/preview_boxes.pt --frontier

Training, but every other attempt skips the part it can already do: it replays the
best run saved for this level (shield watching) to 3 s before that run's end, then the
brain takes over and plays -- and learns -- from there. The replayed part is never
learned from. The other attempts are normal training from the start, so the brain
still learns the whole level. Only those full attempts count for best_boxes.pt; the
console marks the others `(frontier: ...)`, attempts.csv column 9 says `frontier`.
If a replay keeps dying before the hand-over (timing doesn't hold), that spot goes
back to normal attempts. (Replays in --play / --frontier / route practice have a
lighter safety net than the brain: it only changes a replayed press when that press
crashes even as measured and the other choice is safe. The full shield also
"corrected" presses that were fine and killed good replays.) Getting past the best run's end on a frontier attempt saves
the new, longer best run, so the hand-over point moves forward with it.

Inputs are also only sent while a level is on screen (its % showing, nothing
paused / frozen): on the menus SPACE is never pressed.

### Ship paths near the screen edge

With the flight buffer enabled (the default), ship/wave search only certifies
routes while the full simulated player remains inside the captured view. A
route that leaves the view ends at that edge as **unverified**, rather than
being labelled safe because no offscreen geometry was detected. This prevents
the search from recommending an unseen detour above a hanging block. This is
a limit of the observation, not a claim that the screen edge is a solid wall;
the search rebuilds the visible region on every capture as the camera moves.
It can be conservative during vertical camera movement.

The cyan line is the best planned route, including future input changes; it
does not predict holding the current button forever. The preview identifies
when its continuation is unverified. Green/other fixed-action paths retain
their existing meaning. Snapshot JSON includes `path_unknown` and per-root
`visibility_limited` to distinguish an unseen continuation from a collision.

Ship/swing displacement now integrates the time before and after reaching the
speed cap exactly. Velocity initialization uses the actual capture interval,
while future decisions use the smoothed decision interval. This avoids
confusing a slow planning step with the duration of the observed motion.

`python audit_ship_planner.py <death_snapshot.json>` replays saved observations
offline at several search widths. It never controls the game. Old snapshots
omit some alternate input history and portal types, so this is a diagnostic
comparison, not a recreation of the entire run or proof of live success.

### Decision timing and capture experiments

For the new predictor/planner/mode-controller architecture, separate checkpoints,
persistent balanced experience and frozen evaluation, see [CONTROLLER_GUIDE.md](CONTROLLER_GUIDE.md).

Normal startup now uses one PyTorch CPU thread for the small brain
(`--cpu-threads 1`). `--cpu-threads 0` keeps the library's thread setting;
positive values select a thread count. This does not move YOLO off its GPU.
OpenCV's thread setting is unchanged because limiting it did not improve
the recorded-frame benchmark on this machine.

The attempt-banner check processes only the band it uses, with exactly the
same pixel criteria as before. Frame-stack output also avoids a redundant
copy. Capture pacing uses a private Windows high-resolution waitable timer
when available, with a short final spin and a portable fallback. It does not
alter the system timer resolution or the capture FPS.

Timing logs now separate `capture_pacing`, `capture_fetch`, `capture_resize`,
`percent_read`, and `reward_check`. `capture_frame_age` and `observation_age`
measure freshness at acquisition and after perception/planning respectively
when the backend timestamp shares the process clock. Capture/reward samples
are per captured tick; search/observation samples are per decision, so do not
sum all rolling stage means as if their sample counts were identical.

`--threaded-capture` is an **experimental, opt-in** additional reader. It
keeps only the newest owned frame and paired timestamp; no unbounded queue.
Repeated/old frames count as drops, and old reader queues are discarded on
camera restart. The default remains synchronous (`--sync-capture` explicitly
selects it). A live CPU-stress comparison found shorter reads but older frames
and many drops with the extra reader, so it is not enabled for normal play.
DXCam already acquires frames in its own background thread. Moving the Python
trajectory search to another Python thread would still compete for the GIL.

`python benchmark_runtime.py` compares CPU settings on recorded frames and a
temporary network, without changing trained models. `python benchmark_capture.py`
compares live read-only capture paths, with and without synthetic Python work;
it never sends gameplay inputs or saves screen images. These are component
benchmarks, not claims of a particular live gameplay decision rate.
