Geometry Dash Vision AI
=======================

An AI that plays Geometry Dash from the screen alone. YOLO detects the level
from screen capture, a physics planner predicts what pressing or waiting will
do, a DQN brain picks the action, a safety shield vetoes moves the planner
says are fatal, and a landmark-based route memory remembers and improves the
best run. It sends one input, the SPACE key. Nothing reads game memory.

Built for the Meta Global AI Developer Hackathon (Oct 26 - Nov 4, 2026).

Meta tools used:
- Muse Glimmer
- Muse Spark 1.3 Contributor
- Muse Wonder

Full documentation (how to run it, how it works, results, what we tried):
see README.md.
