#!/usr/bin/env python3
"""drift_check.py — do false positives grow with the absolute host counters (tx_MB / rx_MB)?
   python3 revision/analysis/drift_check.py results_rev           (or results_rev_oldmodel)
Prints one line per run in time order: training runs show the counter range the model saw."""
import sys, glob, os, json
import pandas as pd
rows = []
for d in sorted(glob.glob(f"{sys.argv[1]}/*/rep*")):
    f = f"{d}/ctrl_logs/live_dataset.csv"; t = f"{d}/ctrl_logs/timing.csv"
    if not (os.path.exists(f) and os.path.getsize(f) > 0): continue
    x = pd.read_csv(f, usecols=["timestamp", "cpu", "mem", "tx_MB", "rx_MB", "Label"], low_memory=False)
    tm = pd.read_csv(t, usecols=["true_label", "pred"], low_memory=False)
    b = tm[tm.true_label == 0]
    rows.append(dict(run=d.replace(sys.argv[1] + "/", ""), start=x.timestamp.min(),
                     tx_MB=x.tx_MB.mean(), rx_MB=x.rx_MB.mean(), cpu=x.cpu.mean(), mem=x.mem.mean(),
                     benign_FP_pct=100 * b.pred.mean() if len(b) else None))
df = pd.DataFrame(rows).sort_values("start")
df["hours"] = (df.start - df.start.min()) / 3600
pd.set_option("display.width", 200)
print(df[["run", "hours", "tx_MB", "rx_MB", "cpu", "mem", "benign_FP_pct"]].round(3).to_string(index=False))
s = df.dropna(subset=["benign_FP_pct"])
s = s[~s.run.str.startswith("T")]
if len(s) > 2:
    import warnings; warnings.filterwarnings("ignore")
    print("\ncorrelation of benign FP% with:", {c: round(float(s[c].corr(s.benign_FP_pct)), 3) for c in ["hours", "tx_MB", "rx_MB", "cpu", "mem"]})
