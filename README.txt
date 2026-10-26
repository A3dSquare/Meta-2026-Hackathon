Second Attempt — an AI that learns Geometry Dash from pixels
================================================================

WHAT IT IS
-----------
Second Attempt is a reinforcement-learning agent that learns to play
Geometry Dash from raw screen capture alone. It sees what a player sees
and it has one button: hold or release. It learns the same way the game
forces humans to learn — by dying thousands of times and getting
slightly further each time.

The hook: GRIEF, the hardest verified level in Geometry Dash, took a
human player 436,879 attempts over two years to beat once. This project
asks what happens when the learner is a machine that sees only pixels.

HOW IT WORKS (IN BRIEF)
------------------------
- Screen capture (dxcam) feeds frames to a PPO policy: crop, grayscale,
  downscale, 4-frame stack.
- The policy outputs hold or release at 60 decisions per second, sent
  through simulated keyboard input.
- The reward comes from pixels only: the progress bar's fill fraction,
  read with OpenCV. Death is a progress reset, completion is a bonus.
- Training uses a speed curriculum (starting slower, annealing up) and
  section training from checkpoints, so the agent can learn hard
  sections instead of only ever starting from 0%.

THE HONESTY LINE
----------------
No macros. No replayed inputs. No reading the game's memory. No
game-state access of any kind. The only assists are Geode/GDH tools that
move the start line and the clock (speedhack, start-position training) —
they never touch inputs during play. Everything the agent knows, it
learned from pixels.

THE EVENT
---------
Built for the Meta Global AI Developer Hackathon, Indie track,
"Push the Frontier" problem statement, October 26 – November 4, 2026.

Meta tooling used in this project:
- SAM 3.1 — offline analysis of recorded runs (what killed the agent,
  and where), plus labeled overlays for the demo.
- Muse Spark — experiment copilot and analyst in the loop: it reads run
  logs and learning curves and recommends curriculum changes the
  training pipeline actually consumes.
- Muse Code — the build tool used to scaffold and iterate on the
  harness during the event.
- Modal — rented GPU compute and model endpoints for offline jobs
  (video processing, training experiments on recorded data, serving
  open-weight models). Modal cannot play the game itself; live play
  happens on the training rig.

CURRENT STATUS
--------------
This project is at the very start of the build, on day one of the
hackathon. The only file in this repository so far is the Modal setup
for API credits — the rented-compute configuration that later steps
will use for offline jobs and model endpoints.

Nothing is runnable yet. The agent harness, training code, and demo are
being built during the hackathon window and will land here as they are
completed. The full step-by-step build plan lives alongside this file.

ROADMAP
-------
1. Capture smoke test — prove screen capture really runs at ~60 FPS.
2. Environment wrapper — capture, input, and pixel-only reward, wired
   together end to end.
3. Reward validation and first PPO training on Stereo Madness.
4. SAM 3.1 validation gate, speed curriculum, and the Spark
   analyst-in-the-loop.
5. Long training blocks, then section training on a real demon level.
6. Demo video (3 minutes) and submission before the November 4
   deadline.

HOW TO RUN
----------
Nothing runnable yet — setup and run instructions will appear here as
each component lands.

LICENSE AND NOTES
-----------------
Built during the Meta Global AI Developer Hackathon, October 26 –
November 4, 2026. Repository is public per the hackathon's rules. The
method is the public demo: pixels in, one button out, learning by
dying.