# Teach the controller and blue-line physics from manual play

Stop the AI bot before recording so only your inputs drive the game. Use left
click for jumping/thrust; this recorder does not record Space or controller input.
It never sends inputs or focuses a window. Keep Geometry Dash in the foreground.

```powershell
python record_demonstrations.py --level stereo_madness
```

The console prints a session folder such as
`demonstrations/base-after-base/20261031_201500_ab12cd34/`.
Repeated launches with `--level "Base After Base"` or `--level base-after-base`
reuse the same level folder and create separate sessions inside it. Frames,
input timelines and metadata are never overwritten or mixed between sessions.
Existing timestamp-only recording folders remain supported.

| Key | Action |
| --- | --- |
| F6 | Begin a new run, before starting/resuming the level or practice section |
| F7 | End and mark a successful section |
| F8 | Discard the run after a death or an accidental input |
| F9 | Save and quit; an unfinished run is not used for training |

Repeat F6/play/F7 for **at least five separate successful runs across recordings of the same
level**. Ten or more gives a better validation set. Mark each practice
restart separately; do not include deaths/restarts inside a “successful” run.
You can mark a successfully cleared section without completing the whole level.
For ship and wave, include both climbing and descending and several recovery
maneuvers. Do not use Space during a left-click demonstration.

Capture defaults to 30 FPS at 640×360; `--fps 60` reduces action quantization
when your PC can sustain it. Frames are JPEGs and inputs use a separate timed
poller with a private high-resolution timer. A bounded disk queue drops frames
instead of accumulating lag; dropped frames and polling gaps are not silently
treated as normal timing. Recording performs no YOLO inference or training.
Exact capture timestamps are required by the importer. If the recorder warns
they are unavailable, fix the capture backend before making a long recording.

## Train offline

Replace `RECORDING_FOLDER` with the folder printed by the recorder. Choose a
new output directory each time; the source checkpoint is never overwritten by
the offline importer. Always pass your working physics calibration explicitly.
A missing calibration file now stops the import instead of resetting physics
and input timing. For the current working calibration, use
`--motion-model checkpoints/manual_v1/motion_model.npz`.

```powershell
python train_demonstrations.py "demonstrations/RECORDING_FOLDER" --model checkpoints/route_controller.pt --motion-model checkpoints/manual_v1/motion_model.npz --out checkpoints/manual_v3
```

Supply individual sessions, a level folder, or `demonstrations` to collect all
sessions recursively. Overlapping paths are deduplicated. Sessions are grouped
by the normalized level label in `meta.json`, not their folder names. Five
successful clips are required per level across all supplied sessions.

```powershell
python train_demonstrations.py demonstrations --model checkpoints/route_controller.pt --motion-model checkpoints/manual_v1/motion_model.npz --out checkpoints/manual_v3
```

Multiple recording folders can also be supplied before the options. YOLO defaults
to `yolo_gd.pt` and its normal compatible engine-selection logic; select another
detector with `--yolo-model PATH`. `--cycles 20` is the default offline training
budget, up to 128 gradient updates per cycle. More cycles do not guarantee
better validation. The importer can take time: it replays the production
detector, tracker, observation encoder and planner offline.

It writes:

- `import_progress.json`: updated after each observed clip, including reliable
  frame counts, continuous valid segments and rejection reasons. Short or
  unreliable clips are skipped instead of aborting the entire import. Training
  still requires usable training and held-out clips; filtering never reshuffles
  the split or lowers confidence thresholds. A new or empty output folder is
  accepted, so an empty folder left by an older failed import can be reused.

- `route_controller.pt`: existing deployed weights plus the trained candidate,
  optimizer and replay. Human labels train the candidate independently of the
  planner's recommendations. No automatic approval is fabricated.
- `motion_model.npz`: existing physics with only validated parameter changes.
- `report.json`: input quality counts, before/after controller evaluation,
  planner parameter proposals and acceptance/rejection reasons.

Whole clips are reserved for validation across recordings of each level.
Legacy sessions with at least five clips retain their original every-fifth
assignments; shorter sessions use their position in the combined level.
Assignments are saved in each session's `training_split.json`, separately from
recorder metadata, so adding recordings cannot change an assigned clip's role.
Keep these split files. Unreliable clips are still filtered, and usable training
and validation data are both required.
Imports are recorded in the checkpoint so the same run cannot be accidentally
added twice. Keep using the resulting checkpoint when adding later recordings.
An old frozen live trial is cancelled because the candidate has changed.

### Rebuild demonstrations from an older checkpoint

Older replay buffers mixed manual recordings with all historical gameplay,
which could discard most rare human presses. Human examples now have bounded
per-mode hold/release reserves, separate from gameplay (up to an additional
half of the original replay capacity). Coaching balances mode/action first,
then available human, witnessed-gameplay and planner sources. Validation
hold/release scores use witnessed labels only; planner agreement is reported
separately and cannot satisfy the approval gate. Approval thresholds are unchanged.

To recover discarded examples from existing recordings, stop the bot and run:

```powershell
python train_demonstrations.py demonstrations --model checkpoints/manual_v4/route_controller.pt --motion-model checkpoints/manual_v4/motion_model.npz --refresh-demonstrations --out checkpoints/manual_v5
```

Use a new output folder. Refresh keeps the source weights and gameplay replay,
replaces human replay in the new copy, and rereads your original recordings.
It requires all previously imported clips to be supplied and preserves the
saved whole-clip training/validation assignments. Existing checkpoints remain
unchanged. No new recordings are needed simply to perform this repair.

Reports now include `eligible_human_labels` per clip after reliability,
continuity and end-of-section filtering, `data_audit` for retained examples by
mode/source/action in both splits, and actual `coaching_draws` during training.
`teacher_draws` counts witnessed examples in the coaching batch, rather than
unrelated examples drawn for the TD loss. Human-only validation scores are
reported separately. Counts are frame examples, not independent jumps.
Because the validation storage and label definitions changed, compare the
new run's before/after scores on its rebuilt validation set, not directly
against percentages in older reports.

### Independent action policy

The controller now uses a separate policy encoder and per-mode action heads
for imitation. Reward-based TD learning updates the original value encoder
and value heads; it cannot backpropagate into the policy parameters. Motion
learning remains separate as well. Action selection, action validation and live
comparison all use the same policy logits. The overlay's action-score difference
now reflects policy logits rather than reward-value differences.

Old checkpoints automatically initialize the policy from their existing
encoder and advantage heads, preserving their initial action choices. Existing
weights, replay, imports and old optimizer moments are retained; new policy
parameters start with fresh optimizer moments. Old pending trials are cancelled
for revalidation. Approval thresholds and the safety shield are unchanged.

Normal live training continues: the policy learns from retained human inputs
and validated coaching, while reward experience trains the separate value
network. Scalar rewards do not directly update the action policy in this version.
Training reports include separate `policy_loss`, `value_loss` and `motion_loss`.

To retrain the independent policy using existing recordings without replacing v5:

```powershell
python train_demonstrations.py demonstrations --model checkpoints/manual_v5/route_controller.pt --motion-model checkpoints/manual_v5/motion_model.npz --refresh-demonstrations --out checkpoints/manual_v6
```

## Use the taught controller and blue line

```powershell
python preview_boxes.py --backend yolo-only --route-controller --no-route --clearance-reward --model checkpoints/manual_v1/route_controller.pt --motion-model checkpoints/manual_v1/motion_model.npz
```

**Both paths matter.** The first loads the controller and neural motion model;
the second loads the physical parameters actually used to draw/search the blue
line. Controller control still requires the offline gate and live comparison.
Accepted planner calibration takes effect on startup, even while the candidate
controller remains in shadow.

When live training resumes a route controller with `--model`, updates now save
back to that exact checkpoint file. A promoted best checkpoint is saved beside
it with `_best` appended to the filename. The destination is printed at startup
and recorded in session settings. Thus loading `manual_v3/route_controller.pt`
updates that file, rather than the unrelated root `checkpoints/route_controller.pt`.
Fresh runs without a loaded model retain their default save location.
Live startup also rejects missing physics files before starting capture;
`--motion-model none` explicitly opts out of saved calibration for live play.

## What is learned, and how accuracy is checked

Human button edges are quantized to the preceding captured observation: at
30 FPS a press can be taught at most about one frame early. The recorded event
timestamps are retained. Frames containing multiple button edges are excluded.
Mixed-input intervals can teach the human decision but cannot teach a false
“held for the entire interval” motion target. Prediction targets come from
measured displacement and elapsed time; unreliable tracking, lost focus, and
input polling gaps are filtered. TD targets do not cross filtered gaps.

Planner fitting uses clean, unobstructed motion pieces from training runs. It
supports cube gravity/jump speed, ship thrust/fall acceleration and maximum
speed, and wave slope. Gravity, ship and wave changes must improve multi-frame
predictions made by the actual planner integrator on held-out pieces. Cube
jump-speed changes additionally use independently fitted held-out launch speeds.
When all three ship parameters have sufficient evidence, they are tested jointly
on both held and released motion: acceleration and maximum speed interact.
At least three usable pieces are required in both sets; insufficient evidence
or no improvement leaves that parameter unchanged. Observation/planner features
are rebuilt after accepted calibration. The existing bot input-delay settings
are preserved: manual clicks do not have the bot's observation/decision queue.

This does **not** retrain YOLO, invent missing geometry, or change collision
rules. Other modes still receive controller/motion supervision, but their
planner parameters are not fitted by this importer. F7 is your assertion of a
successful section, not automatic proof of level completion. Offline tests
cannot guarantee better live progress; inspect the report and subsequent runs.
