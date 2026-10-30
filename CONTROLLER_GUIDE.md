# Route-conditioned controller

This is a separate controller architecture. The legacy DQN and its checkpoints
remain available. YOLO and the existing calibrated physics are reused.

## Start and resume

First run, importing the useful encoder from the preserved best legacy brain:

```powershell
python preview_boxes.py --backend yolo-only --route-controller --model checkpoints/best_boxes.pt --no-route --hold-cost 0.0 --clearance-reward
```

Subsequent runs must resume the NEW checkpoint, which includes experience,
optimizer, exploration, deployed/candidate weights and any pending evaluation:

```powershell
python preview_boxes.py --backend yolo-only --route-controller --model checkpoints/route_controller.pt --no-route --hold-cost 0.0 --clearance-reward
```

The default output is `checkpoints/route_controller.pt`. The validated champion
is `checkpoints/route_controller_best.pt`. The original `preview_boxes.pt`,
`best_boxes.pt`, and YOLO weights are not overwritten by this controller.
Starting repeatedly from a LEGACY checkpoint reimports its encoder and starts
the new experience again; resume `route_controller.pt` after the first run.
Omit `--model` to initialize the new controller without importing an encoder.

The default CPU thread/capture settings still apply. No GPU training process,
real-time OS priority, or background gameplay process is introduced. The new
planner traces root actions separately, which can cost more search time; use
the timing logs to assess the complete live loop on your machine.

## What the line and brain mean

The cyan line predicts a particular **planned input sequence**, with explicit
timestamps, velocity and delayed inputs. It is conditional on that sequence,
not a promise about arbitrary future button presses. Search now keeps root
histories separate so the displayed route cannot inherit another first
action's trajectory. All modes use known-view limits and collision-first
planning; ground modes also prefer clearance without penalizing valid
standing contacts. The planner rebuilds the route from each new observation.

The controller sees upcoming target positions and velocities, current measured
and estimated velocity, grounded state, mode/gravity/size, decision timing,
pending commands and their observed durations, motion uncertainty, and fixed
hold/release forecasts. A shared encoder has separate control heads for cube,
ship, ball, UFO, wave, robot, spider and swing.

A separate action-sequence-conditioned model learns measured displacement and
velocity at the next **one through five decisions**, including changing inputs
and actual observation intervals. Its recurrent rollout only sees inputs that
have reached the current interval, along with the previous effective input:
queued commands precede new ones, and future feedback commands cannot leak
backward into predictions. It is trained on observed outcomes, with a
displacement/velocity consistency term. Until independently validated, control
uses the analytical short forecasts; the long cyan route still comes from
calibrated physics and collision search. Validity flags prevent extrapolating
a fixed-action forecast through its collision or beyond the observed area.

During bootstrap the planner supplies actions while the new controller learns.
The overlay and snapshots explicitly identify bootstrap, learned control, and
evaluation. Unvalidated heads do not quietly take over from the planner.
The safety shield continues to check both exploration and learned actions.

## Learning and demonstrations

Planner first-action labels are provisional. They only enter learning when
the executed inputs match the traced plan and several subsequent recorded
positions/velocities agree, including time after the pending-input delay.
Disagreement invalidates the label. The final 1.5 seconds before a death are
also stripped of demonstration labels. Missing observations, mode/gravity/
size changes and invalid time intervals cannot create motion targets.

A small tracking cost compares observed motion to the **previous** trusted
route in its original coordinate frame. Replanning cannot move a target to
manufacture a tracking reward. Survival/death rewards remain in use.
For this controller, alive/hold/tracking costs and discounting use elapsed
observation time with a 30 Hz reference. A slower decision loop cannot change
the objective simply by producing fewer reward/discount steps. Optional
clearance shaping uses the same transition discount and terminal correction.

Replay has separate reservoir quotas per mode and safe/failure continuation.
Sampling balances populated buckets, retaining rare modes and earlier useful
experience. Every fifth learning episode is held out entirely, including all
its rollout fragments, from both control and motion training. Complete replay
and RNG state persist in atomic checkpoints. Safe prefixes are banked at a
clean stop too; the uncertain death tail is intentionally excluded. A
partially banked episode gets a new episode identity after restart.

## Deployment checks

Candidate weights train in shadow with a lower learning rate and bounded
updates. Held-out checks cover demonstrated action agreement, avoidable unsafe
choices, TD consistency, and multi-frame position/velocity error per mode.
Previously trusted modes must retain validation coverage and pass regression
checks. Offline checks alone **cannot promote a candidate**.

After passing, the candidate is frozen for six alternating gameplay attempts:
three current-controller and three candidate attempts. Exploration, learning,
and route-memory forcing are disabled for this comparison. Evaluation samples
are never put in replay. The shield remains on; its intervention rate is
measured. The controller must improve typical progress/completion or reduce
interventions while preserving median/lower-quartile progress and completion.
A single lucky maximum is insufficient.

Keep the SAME level and game settings throughout an evaluation block.
Opening-frame fingerprints check comparability. A level change or missing
fingerprints makes the comparison inconclusive and prevents promotion. These
are small comparisons, not statistical guarantees or evaluation of every
unseen level/mode. Model/error checks depend on detector/tracker quality.

Evaluation progress persists on restart; the first partial attempt after a
restart is discarded and comparison resumes after respawn. `--play` uses the
deployed controller and never completes/promotes a pending candidate trial.
Trials queued during a long run wait for a respawn before collecting results.
Completing a level saves and stops normally; resume the same level/controller
to finish a pending comparison. A training completion saves the candidate and
experience without falsely labelling an unvalidated model as the champion.
`--frontier`/route practice can still be used for training, but are bypassed
during controller evaluation.

## Inspect results

Session files include:

- `controller_evaluation.jsonl`: held-out model/action checks and queueing.
- `controller_live_evaluation.jsonl`: frozen comparison outcomes.
- `attempts.csv`: evaluation arms are marked `controller_eval_incumbent` and
  `controller_eval_candidate`; those rows are not training attempts.
- Death snapshots: explicit `route_plan`, `controller_role`, and pre-action
  `motion_prediction` for hold/release with source, origin height, times and
  validity. The motion forecast is made before the snapshot's executed input;
  its origin height allows comparison to the next observed position.

Offline report, without playing or saving weights:

```powershell
python evaluate_route_controller.py checkpoints/route_controller.pt
```

Tests use recorded or rendered scenes, synthetic transitions, fake input and
temporary checkpoints. Passing them verifies the implementation; it does not
establish that the new controller already outperforms your old brain live.

## Cube prediction and calibration update

Cube support detection now runs before arc estimation, so a platform landing
ends the previous falling arc before the next jump is predicted. The takeoff
window remains anchored to the actual contact when the observation history
rolls forward. Fits must explain the newest position and interval, and can
correct the measured center by at most 0.2 blocks. Rejected fits fall back to
measured motion instead of preserving a contradictory old velocity.

In route-controller mode, clean partial airborne arcs can also calibrate cube
gravity and jump speed; returning to the original floor is not required. One
vote is accepted per arc. Earlier observations fit the proposal and a later
observation checks it. Parameters only change gradually after at least three
consistent arcs show better held-out prediction than the current constants.
Tracking gaps suspend calibration until a confirmed contact or new impulse.
Learned constants use the existing motion-model persistence. Snapshots include
`cube_physics` and the preview input's `state_source` for diagnosis.

Motion-model velocity targets now come from recorded displacement divided by
elapsed time. Fitted tracker velocities are state estimates, not ground-truth
labels. Old checkpoints are migrated in memory: strongly contradictory samples
are excluded and retained velocity targets are made consistent with recorded
positions. Deployed weights and exploration progress remain. If there is no
approved learned controller, an affected unapproved shadow candidate is restarted
from the deployed baseline. Pending old-metric trials are cancelled.

Before overwriting a migrated source checkpoint for the first time, the program
copies it to `<checkpoint>.before_cube_targets`. Resume the same controller path;
a fresh brain or YOLO retraining is not required.

Validation reports both interval-velocity error and velocity error averaged over
each prediction window. The latter compares integrated predicted velocity with
measured displacement, reducing amplification of frame-to-frame box jitter.
Position checks and live comparison remain mandatory; a model with correct
positions but incorrect velocity is still rejected.

`python audit_cube_motion.py --snapshots <session/deaths>` replays saved cube
states and checks short free-flight predictions. It does not play the game or
claim to validate collision geometry or whole-level performance.

## Avoiding repeated failed controller trials

Offline action checks now require at least four held-out recommendations for
each input, hold and release, with at least 85% agreement on each. A perfect
score on release-only labels cannot queue a trial. Existing motion, witnessed
demonstration, safety and live promotion checks still apply. Reports include
`action_label_counts`, `action_recall` and `balanced_action_supervision`.

Training uses a separate mode/action-balanced coaching batch so rare jump
examples are sampled. Observed, validated demonstrations have full weight;
verified planner recommendations on safe continuations have quarter weight.
The latter are stored separately as `planner`, never relabelled as observed
successful actions. Death-tail recommendations are excluded. Held-out episodes
remain excluded from training. The bounded old-route tracking penalty is now
actually added to reward; previously it was calculated and logged only.

After a failed live comparison, at least 24 training episodes and 512 gradient
updates must finish before another can start. Repeated failures increase the
episode wait to 48, then 96. The schedule persists across restarts. Evaluation
remains frozen and excluded from replay, and promotion still requires the live
comparison to pass. This improves the training/evaluation allocation; it does
not establish better live performance by itself.

Resume the existing checkpoint. Old one-action-gate trials are cancelled in
memory on load and a new training interval begins. Deployed and candidate
weights, replay, optimizer and exploration progress are preserved. New planner
coaching fields fill as fresh training episodes are recorded.

## Blue-line command sequence safety

Cube route checks now replay one explicit future input sequence in both the
nominal and slower-input/faster-scroll simulations. Previously, each simulation
could find a different successful continuation and the single displayed route
was still marked verified. The search tries the sequence from each simulation,
checks both against both worlds, and selects using the resulting safety scores.
An uncertain sequence cannot supply a verified planner coaching label.

Snapshots include `route_plan.sequence_safety` with nominal and worse-world
survival fractions for the displayed commands. A verified route must survive
both. Replanning can still change the sequence at the next observation; this
check does not guarantee safe execution under arbitrary timing, tracking errors
or unseen geometry. It does not change YOLO weights or reset the controller.

## Tight jump arcs and scroll measurements

Repeated floor tiles can alias between observations: matching the nearest
tile to its old position can look like rightward motion when the level actually
scrolled left by almost one tile. This previously drove the measured speed from
about 10.4 to 4.6 blocks/s and compressed the blue arc horizontally.

Terrain matching now uses expected leftward displacement over the actual frame
interval, allows longer shifts on slower decisions, and still recognizes an
unchanged scene. Rightward matches cannot update forward speed or validate
physics calibration. Actual slower motion remains measurable. No jump constants
or learned weights are reset. Restart to clear the old runtime speed samples.

The preview hides paths with no detected player rather than anchoring them to
a default position. Its status distinguishes a predicted collision, view limit,
and timing-sensitive sequence. A short collision horizon is not a different
jump strength or a change in the line's drawing thickness.

## Landing timing checks

A short hold around a landing can succeed at the two tested input delays but
fail between them. Cube plans now check nine fractional delay phases as well
as the faster-scroll world. Choosing an earlier alternative delay cannot drop
the later bound of one additional decision. Duplicate command sequences are
checked once and failed candidates stop their phase checks early. Snapshots
record the selected plan's `delay_checks`. Passing this finite set is still
conditional on the observed geometry and motion, not a guarantee for all timing.

Large isolated speed samples outside 65–135% of the current reference are
excluded. Named speed portals reset that reference before sampling continues;
an undetected large speed change may therefore need portal detection corrected.
Player boxes without either pixel-motion or valid scene-motion evidence are
excluded from controller motion targets and coaching. This prevents stationary
explosion fragments from teaching that a held cube should fail to jump.

## Planner runtime optimization

The planner reuses each beam search's result when checking that search's own
selected command sequence; cross-world sequences still get replayed. Fixed
sequence checks avoid branch-state copies and state-merging work. Ground
clearance scores are cached only within a single observed frame, and the ground
clearance calculation avoids unnecessary allocation/clamping. Flight braking
calculations remain separate. All nine delay phases are retained.

`python benchmark_planner.py` compares the optimized planner with the saved
pre-optimization outputs on 45 recorded states. The regression fixture checks
the full result to 1e-12 numerical tolerance, including paths, controls, safety
scores and timing checks. Recorded CPU search timings are not live FPS results.

Timing logs now include `action_observation_age`, measured just before entering
the input loop after focus checks. Snapshots expose the same measurement as
`input.observation_age_ms`. This includes processing and intervening GUI/policy
work after the captured picture, providing a closer measure of input freshness
than the existing end-of-observation-processing metric. Empirically calibrated
input delay remains in use; age is not blindly added again to that delay.

Further CPU work is shared within each observation: timestep grids, collision
geometry, portal/pad lists and nearby clearance geometry. Each new observation
gets a fresh cache, so a previous frame or level cannot supply stale geometry.
Physics bodies use compact fixed-field objects. Search breadth, collision
handling and all delay-phase checks remain unchanged.

Scalar trajectory lookups use binary search rather than repeatedly constructing
NumPy arrays; regression checks cover recorded routes and duplicate timestamps.
The preview's urge bar uses the Q values for the displayed action rather than
running the neural network again. Coaching buckets are built once per paused
training cycle with the same sampling sequence and weighting as before.

For paired timing comparisons under the same CPU load, the benchmark accepts
`--reference-module <saved-physics-file.py>`. It alternates the old and new
planners in one process. Use only a trusted local reference file, since it is
imported as Python code. These development benchmarks use recorded states and
do not access the live game, YOLO engine or trained checkpoint.

Single-frame CPU inference evaluates only the active mode head, including
learned motion rollouts when trusted. This shortcut is restricted to gradients
disabled; gradient-enabled training retains its existing computation/optimizer
semantics even for batch size one. Network layout, checkpoint format, mode
coverage and candidate promotion checks are unchanged.

## Takeoff head collisions

Cube/robot takeoff into a platform underside is now fatal in the simulator,
including an underside already touching the player's head. Previously the
head-collision test skipped a step that started grounded, then the ceiling
handler stopped vertical movement as if the collision were a harmless bump.
The correction also covers inverted gravity. Horizontal movement under a low
ceiling without jumping remains valid. The recorded performance fixture was
updated against the unoptimized reference with the same collision correction;
old expected results are retained on changed cases for audit.
# Landing/takeoff evidence and earlier-decision diagnostics

Controller prediction inputs now use interval-average velocity for both the
analytical prior and learned motion, matching the measured training targets.
Old replay observations are migrated on load while preserving weights and
experience; an old frozen live trial must revalidate under the corrected inputs.
The existing pre-migration backup mechanism applies before overwriting the source.
Motion-only training batches balance modes and initial commands, excluding rows
without valid motion targets. TD replay and episode-held-out validation remain
separate. Candidate motion trust is revoked when its current validation fails;
deployed weights are unchanged. Evaluation logs list per-mode blockers so a
failure of action recall is not confused with motion accuracy or safety checks.

Cube arc fits now require their current velocity to agree with the observed
interval (within 5 blocks/s). An anchored jump also requires more than 0.12
blocks of observed upward movement. This prevents a fitted launch at the end
of an almost stationary interval from inventing full jump velocity. Ambiguous
frames fall back to measured motion until further evidence arrives. Invalid
tracking or timing clears the arc history, including on apparent landings.
The tradeoff is that a very late real takeoff can require another observation
before receiving a full fitted jump velocity.

Death JSON `cause.decision_trace` records the last verified plan, the first
subsequent observation, and the last executed action with a safer *scored*
alternative. It includes elapsed time, action, planned action, and measured
versus estimated velocity. This is retrospective evidence, not a counterfactual
proof or a new training reward. Missing history is explicitly marked. The
snapshot buffer retains 150 decisions rather than 75 to preserve the approach
before the death-animation tail; this increases snapshot memory/disk use.

Existing snapshots can be inspected without changing the brain:

```powershell
python audit_decisions.py logs_boxes/sessions/SESSION/deaths --output logs_boxes/decision_audit.json
```

Restart training with the same checkpoint to load these code changes.

