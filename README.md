# Geometry Dash Vision AI

An AI that plays Geometry Dash **from the screen alone**. It sees what a player sees and controls
one input (jump, or hold for the ship). No game memory, no mods that read game state.

Built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026).

---

## TL;DR: the best way to run it

NOTE: In practice this needs an NVIDIA GPU. YOLO falls back to the CPU without one, but that is far too slow to play.
Run `python detect_yolo.py --export` once to build the faster TensorRT engine (optional).

Open Geometry Dash, start the level in **normal mode**, keep the GD window in front, then run:

```powershell
python preview_boxes.py --backend yolo-only --route-controller --clearance-reward ^
  --model checkpoints/manual_v21/route_controller.pt ^
  --motion-model checkpoints/manual_v21/motion_model.npz ^
  --hold-cost 0.0 --landmarks --play --ms-delay --ghost --guide
```

* The first attempt on a new level is played by the brain plus the safety shield. That attempt builds
  a landmark map of the level and a remembered route. Every later attempt replays and improves it.
* Nothing needs deleting between sessions. Routes live in `routes/<level>/` and keep improving.
* When the level is beaten, the console prints `LEVEL COMPLETE` and the program exits on its own.
  It never presses anything on the level-complete screen.
* Quit any time with `q` or ESC in the preview window.

### What each flag does

| Flag | What it does |
|---|---|
| `--backend yolo-only` | YOLO detects everything: spikes, blocks, orbs, pads, portals and the icon (with its mode). |
| `--route-controller` | Uses the route-conditioned brain (DQN) with the physics planner's features. |
| `--model … --motion-model …` | The brain checkpoint, plus the calibrated physics (gravity, jump speed, ship rates, input delay). |
| `--hold-cost 0.0 --clearance-reward` | Reward settings the v21 brain was trained with. |
| `--landmarks` | Route memory is positioned by landmarks in the level (spikes and blocks), not by a clock. |
| `--play` | Deterministic play. The brain doesn't train, and route memory is used and improved. |
| `--ms-delay` | The planner uses the input delay measured in milliseconds (median about 83 ms across all sessions), not whole steps. |
| `--ghost` | In cube sections, follows the remembered **path** rather than replaying button presses. Learns per stretch from deaths. |
| `--guide` | The brain overrides the route when it clearly prefers its own choice. |

Optional:
* `--no-frontier-explore` turns off the deliberate variation at the end of the best run.
* `--level <route folder>` forces a route instead of recognising the level.
* `--no-death-pause` and `--no-early-press` are timing and shield tweaks; see `python train_boxes.py -h`.

---

## How it works

```
screen (dxcam) ─► YOLO detector ─► hazard tracking + landmark map ─► physics planner (robust search)
                                                                         │
                     route memory (landmarks: replay / ghost / guide) ───┤
                                                                         ▼
                                  brain (route-conditioned DQN) ─► safety shield ─► SPACE key
```

1. **Vision.** `detect_yolo.py` runs a YOLO model (`yolo_gd.pt`, trained on frames labelled with
   `label_boxes.py`) with 51 classes: hazards, solids, every orb, pad and portal type, and the icon
   per game mode. A hazard tracker smooths boxes and measures scroll speed.
2. **Physics planner** (`gd_physics.py`). It simulates cube, ship, ball, UFO, wave, robot, spider and
   swing physics, including pads, orbs, portals, gravity flips and the input delay. A beam search answers
   "can I survive if I press now or wait?" in two worlds: as measured, and slightly worse. Gravity,
   jump speed, ship rates and pad strengths are **calibrated live** from the icon's real motion.
   Live changes to the cube's gravity and jump speed are limited to ±2% per session (see below).
3. **Brain** (`route_controller.py`, `route_learner.py`). A DQN that sees the planner's outcomes,
   the detections and its own motion forecast. It was trained from live play plus 164 recordings of
   human demonstrations (`train_demonstrations.py`).
4. **Safety shield** (`trainer.py: safety_shield`). Overrides any action the planner says is fatal when
   a safer action exists. Ship and wave use a buffered flight shield.
5. **Route memory** (`landmark_route.py`). It builds a map of each level from landmarks and locks onto
   it every attempt. The best run is remembered and reused, with **per-press learning**: a press that
   keeps failing is tried earlier or later, then with a different hold length, then handed to the
   brain. A new best run is spliced in from where the old one started failing.
   * `--ghost`: remembers the icon's **height along the level** and, in cube sections, picks
     the action whose predicted arc stays on that path. It learns a timing shift for each 4-block
     stretch, or hands the stretch back to the press replay where it keeps dying. It steps aside near
     pads and orbs.
   * `--guide`: where the brain and the route disagree, a confident brain wins.
   * **Frontier exploration**: every 3rd time it reaches the end of its best run, the decisions there
     are deliberately changed. It cycles through: swap every decision, presses become waits, waits become
     presses, swap every 2nd decision. A change is never allowed if the planner rates it as less safe.
     Without this, deterministic play repeats the same death at the frontier forever.

---

## Results (Nov 1–4, 2026)

* **6,726 attempts** over **172 sessions** in 4 days.
* **A full clear (100%)** of the level saved as `routes/level_20261104_005044`, on attempt 15 of a
  `--ghost --guide` session. Its previous best was 43%.
* **Stereo Madness:**
  * Best runs reach about **94%** in-game. The log shows 89% because the % reader stops reading near
    the end (see the bugs section).
  * The saved route covers about 850 blocks of the level. A bug that switched inputs off near the
    end was found and fixed on the last day.
  * Over 1,942 attempts the median run reached 22% (a few hard walls), and 158 runs reached 75% or more.

---

## Other workflows

### Teach it your own run (strongly recommended)
The bot's own best runs contain unnecessary jumps. In Stereo Madness's first quarter, its route had
**59 presses where a human uses 28**, and every extra jump is a chance to die. Record yourself:

```powershell
python record_demonstrations.py --level stereo_madness    # F6 at the start, play, F7 at the end, F9 to save
python import_human_route.py demonstrations/stereo_madness/<new folder>   # stop the bot first
```
The level must already have a route (play it with the bot once). The import backs up the old route.
Use `--partial` for a clip that doesn't reach the end of the level.

### Retrain the brain from demonstrations
```powershell
python swap_brain.py --brain checkpoints/manual_v21/route_controller.pt --rest checkpoints/manual_v21/route_controller.pt --out checkpoints/manual_v22 --reset-candidate
python train_demonstrations.py demonstrations --model checkpoints/manual_v22/route_controller.pt ^
  --motion-model checkpoints/manual_v22/motion_model.npz --out checkpoints/manual_v23 --refresh-demonstrations
```
* `demonstrations` (the whole folder) imports every recording.
* YOLO boxes are cached in `detection_cache/`, so a refresh only runs detection on new frames.
* The report is written to `report.json`. A learned brain only takes over in `--play` after it passes
  the offline gate **and** a live A/B comparison. So far only the cube is trusted.

### Detector
```powershell
python label_boxes.py --rec recs/<video>.mp4     # fix pre-filled boxes, D = next frame
python detect_yolo.py --train --epochs 80        # -> yolo_gd.pt (then --export for TensorRT)
```

### Diagnostics
* `logs_boxes/sessions/<id>/` contains:
  * `attempts.csv` and `route_timing.jsonl`: one row per attempt, including route, ghost and frontier
    statistics.
  * `shield_disagreement.jsonl`.
  * `deaths/`: a PNG and JSON per death, showing the last second, what the planner predicted, what the
    brain wanted and what was sent.
* `python timing_report.py` summarises decision timing and input delay.
* Tests: `python -m pytest test_route_memory.py test_cube_motion.py` (44 tests). The full bot loop on
  a simulated game is in `test_route_live.py`.

---

## What we tried

| Idea | Result | Kept? |
|---|---|---|
| **Clock-based replay** (`route.py`) | Timing drifted run to run; the replay missed. | Replaced by landmarks |
| **Landmark replay** (`landmark_route.py`) | Positions repeat to about 0.1 block; got to 89% on Stereo Madness. | ✅ core |
| Per-press offset learning, hold variation, hand to brain | Fixes individual bad presses slowly. Untried timings needed an optimistic prior to get tested at all. | ✅ |
| Splicing a new best into the old route | Stops one lucky run from overwriting presses that worked hundreds of times. | ✅ |
| **Human route import** | 28 presses instead of 59 in the first quarter. Removed the bot's extra jump at the 19 s wall. | ✅ |
| **`--guide`** (brain overrides when confident) | Polargeist median **21% vs 14%** (18 attempts). | ✅ |
| **`--ghost`** (follow the path) | Without learning, worse in the simulated game (median 5% vs 11%). With per-stretch learning it matched replay in the sim. Live with `--guide` it pushed the Stereo Madness best from 79 s to 82 s and got the first full clear. | ✅ opt-in |
| `--precise-press` (send presses between decisions) | Presses landed on time, but the delayed key confused the live physics learner. Gravity drifted from 93 to 86 and the bot died at 9% instead of 22%. | ❌ removed |
| `--trust-replay` (replay with no shield) | **Much worse:** median 1% vs 14% on Polargeist. The shield is what keeps replay alive. | ❌ off |
| Ghost used only to recover when off the path | Median 11% vs 27% in the sim. | ❌ |
| `--fast-steps` (variable 2/3-frame decisions) | Worse: deaths at the 19–21 s wall rose to 57%. | ❌ |
| `replay_guard` (shield without the "press now" rule) | Median 3% vs 9%. | ❌ |
| Grouping solids from YOLO | Changed 8% of verdicts with no speed gain. | ❌ |
| Promoting a learned brain after a short live trial | It over-pressed, so it was reverted (`swap_brain.py`). The gate is now stricter. | ❌ reverted |
| Learned ship controller | Agreement 65–74%, hold recall 0.40–0.60. It doesn't pass the gate. The planner and replay fly the ship. | ⏳ |

### Bugs found along the way
* **Inputs switched off near the end of Stereo Madness.** "In a level" was decided by reading the %
  text, and in the last section it stops reading (the last read is 89% while the game shows about 94%).
  After 1.5 s without a read, inputs went off and the ship fell, every attempt. Fixed: once in a level
  with the icon on screen, inputs stay on. A backup end-of-level detector stops the run on a win,
  since 100% is never read there.
* **Pads.**
  * A single missed detection of a yellow pad made waiting look fatal, so the bot jumped over the pad.
    Fixed: pads and orbs persist for 3 decisions.
  * A remembered press made mid-air (after a pad launch) was replayed on the ground. Fixed: a height
    guard skips such presses.
  * The ghost's arcs don't model pads. Fixed: the ghost steps aside near pads and orbs.
* **Physics corruption.** One bad session's live calibration was saved and broke every later session.
  Fixed: live updates are limited to ±2% of the session's starting physics.
* **Missed gravity portals.** Upcoming portals were only remembered from 2.5 blocks away, and the
  detector often loses a portal right before the icon reaches it (the flash). On Jumper, the bot
  missed the yellow gravity portal in 4 of 14 attempts and kept playing an upside-down cube as an
  upright one. Fixed: portals are remembered from 5 blocks away and applied when the icon touches
  them, as GD does, rather than after it has passed the centre. That gives back 2–3 decisions.
* **Detection ran twice on every frame during demonstration training** (about 70 minutes). Fixed with
  `detection_cache/`.
* Recordings with 0 frames (capture bug), the resume key causing a jump (SPACE → ESC), skip probes
  never scored, a learned respawn origin that ratcheted, and others: see `logs_boxes` notes and the
  test names.

### Known limits / next steps
* Most remaining deaths are logged as **"trapped"**: the planner saw the danger after the last safe
  moment. The limit is prediction accuracy (jump arcs, and floor detection after big drops), not
  route memory.
* The % reader fails on bright backgrounds.
* While the icon crosses a portal, the flash can hide it for 1–3 decisions, and the bot plans from a
  stale position until it reappears.
* Learned ship control.
* Teach routes from more full human runs.

### Meta tools used
* Muse Glimmer
* Muse Spark 1.3 Contributor
* Muse Wonder

### Honesty note
Route memory replays **the bot's own remembered presses**, and optionally a run recorded from a
human (`import_human_route.py`). The planner, shield and brain still decide live on every frame.
Nothing reads game memory or alters the game.

---

## Files

| File | Role |
|---|---|
| `preview_boxes.py` / `train_boxes.py` | Entry point with the overlay window / all flags |
| `trainer.py` | Main loop: brain, shield, route memory, ghost/guide/frontier, logging |
| `env_boxes.py`, `env.py`, `gd.py` | Capture → observation, input, death/progress detection |
| `detect_yolo.py`, `detect.py`, `label_boxes.py` | Detector, tracking, labelling |
| `gd_physics.py`, `cube_motion.py`, `flight_control.py` | Physics, calibration, search, flight shield |
| `landmark_route.py`, `route.py` | Route memory (landmarks / clock, level recognition) |
| `route_controller.py`, `route_learner.py`, `controller_evaluation.py` | Brain, learning, promotion gates |
| `record_demonstrations.py`, `train_demonstrations.py`, `import_human_route.py` | Human demonstrations → brain / route |
| `swap_brain.py`, `timing_report.py` | Checkpoint surgery, timing diagnostics |
| `routes/<level>/` | `landmarks.json` (map, presses, ghost, stats), `route.json`, `frames.npz` |
| `checkpoints/manual_v21/` | Current brain + physics |
