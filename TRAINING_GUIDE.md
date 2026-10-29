# Training Guide — cross-level GD box policy (v2)

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
