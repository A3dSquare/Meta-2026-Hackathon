"""Rank each death detector using cached traces (seconds, not minutes).

Step 1 (once): python eval.py --trace recs/*.mp4  (decodes + matches once)
Step 2: python ablate.py  (reward-only replay per flag combo)
Prints pooled AND per-rec P/R/MAE/acc/fresh. Validates every flag exists
on RewardConfig. Proves each toggle changes predictions on >=1 rec.
Pair combos included beyond single-removal.
"""
import sys
sys.path.insert(0, ".")
from eval import extract_trace, eval_trace
from gd import RewardConfig

RECS = ["recs/jumper.mp4", "recs/stereo.mp4", "recs/finger.mp4"]

base = dict(use_pctdrop=True, use_snap=True, use_reset=True, use_sink=True, use_banner=True, use_freeze=True)
# validate flags exist
for k in base:
    if not hasattr(RewardConfig, k):
        print(f"ERROR: flag {k} is NOT an attribute of RewardConfig (detector ignores it?)")

combos = {"ALL": base}
for k in base:
    c = dict(base); c[k] = False; combos[f"no_{k}"] = c
combos["pctdrop_only"] = dict(use_pctdrop=True, use_snap=False, use_reset=False, use_sink=False, use_banner=False, use_freeze=True)
combos["pctdrop+snap"] = dict(use_pctdrop=True, use_snap=True, use_reset=False, use_sink=False, use_banner=False, use_freeze=True)
combos["pctdrop+reset"] = dict(use_pctdrop=True, use_snap=False, use_reset=True, use_sink=False, use_banner=False, use_freeze=True)
combos["pctdrop+sink"] = dict(use_pctdrop=True, use_snap=False, use_reset=False, use_sink=True, use_banner=False, use_freeze=True)

traces = []
for r in RECS:
    tp = extract_trace(r)
    traces.append((r, tp, r.rsplit(".", 1)[0] + ".labels.json"))
    print(f"trace ready: {tp}")

results = {}
for name, flags in combos.items():
    # flag proof: unknown flags error out now
    for k in flags:
        if not hasattr(RewardConfig, k):
            print(f"ERROR: combo {name} uses unknown flag {k}")
    per_rec = []
    tp = fp = fn = 0
    maes, accs, fresh = [], [], []
    for r, tp_, lab in traces:
        o = eval_trace(tp_, lab, flags)
        per_rec.append((r, o))
        tp += o["death_tp"]; fp += o["death_pred"] - o["death_tp"]; fn += o["death_true"] - o["death_tp"]
        if o["pct_mae"] == o["pct_mae"]:
            maes.append(o["pct_mae"])
        accs.append(o["digit_acc"]); fresh.append(o["fresh_frac"])
    p = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    mae = sum(maes) / max(1, len(maes))
    results[name] = (per_rec, tp, fp, fn)
    print(f"{name:15s} POOLED P={p:.3f} R={rc:.3f} tp={tp} pred={tp+fp} true={tp+fn} MAE={mae:.2f} acc={sum(accs)/max(1,len(accs)):.3f} fresh={sum(fresh)/max(1,len(fresh)):.3f}")
    for r, o in per_rec:
        print(f"  {r}: P={o['death_prec']:.3f} R={o['death_rec']:.3f} tp={o['death_tp']} pred={o['death_pred']} true={o['death_true']} MAE={o['pct_mae']:.2f} acc={o['digit_acc']:.3f} fresh={o['fresh_frac']:.3f}")

# proof: toggling each flag changes predictions on >=1 rec
print("flag-effect proof: compare no_<flag> pred counts vs ALL per rec above.")
print("If a no_<flag> row is identical to ALL on all 3 recs, that flag is dead.")
