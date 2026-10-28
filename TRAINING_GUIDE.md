# Training Guide — cross-level GD box policy

Two policies exist. Train the BOX one (general). The pixel one is legacy.

## 0. One-time setup

GD windowed/foreground, Stereo Madness practice, Auto-Retry ON.
`pip install stable-baselines3 gymnasium opencv-python numpy torch`

## 1. Start training (general data)

```
python preview_boxes.py --backend heuristic
```

What happens per death: `PAUSED verified=True` (ESC) → PPO update off-grid
→ `SETTLE 1.0s` → `RESUMED` (SPACE). Saves `preview_boxes.zip` every death,
`best_boxes.zip` on new best-%. Console `TRAINED pg/ent` proves learning.

Resume: `python preview_boxes.py --backend heuristic --model ./checkpoints/preview_boxes.zip`
Fresh level, same brain: keep `--model ./checkpoints/best_boxes.zip`, change
GD level, run again. Boxes transfer; backgrounds never existed to the policy.

## 2. What good looks like (first 30 min)

- `PAUSED` every death you see in-game. Missing line = detector miss, not training.
- `pg` magnitude grows `0.001→0.01`, `ent` stops sliding to 0.
- Best-% climbs 1→5→10 on Stereo, then holds when you switch levels.

## 3. If it stalls

- No `PAUSED` on visible death: `python detect.py --probe recs/stereo.mp4`
  mean should be <2ms; if entities=0 on your level, label 50 boxes and train YOLO.
- `ent` pinned at min: raise `--backend yolo` candidate or restart fresh
  (entropy floor auto-bumps, but a collapsed head needs a fresh net).
- OOM/crash: rollout is 0.19GB by design; if RAM spikes, lower n_steps in
  `train_boxes.py` (1024→512).

## 4. YOLO upgrade (only if heuristic misses art)

1. `python label_boxes.py --rec recs/<level>.mp4` (~150 boxes/level, 3 levels)
2. `python detect_yolo.py --train --epochs 50` → copy best.pt to `yolo_gd.pt`
3. `python preview_boxes.py --backend yolo`

## 5. Files

- `checkpoints/preview_boxes.zip` — latest brain (resume this)
- `checkpoints/best_boxes.zip` — best-% brain (keep this safe)
- `tb_boxes/` — tensorboard (`tensorboard --logdir tb_boxes`)
- Pixel legacy: `preview.py --train`, `preview_model.zip` — per-level only.
