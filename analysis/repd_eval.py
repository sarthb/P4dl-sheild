#!/usr/bin/env python3
"""
repd_eval.py — REPD (Rényi-entropy) baseline evaluated on EXACTLY the same runs and unit as the DNN
(Reviewer 3 #23: "same traffic and tuning methods").

Differences from comparison_scripts/renyi_baseline.py (state these in the paper):
  * the reference (normal) entropy comes from benign windows of the TRAINING runs, not from the test file
  * alpha, window size, the entropy field (dst IP, as in REPD, or src IP) and the distance threshold are
    chosen on the VALIDATION runs (max F1); test runs are scored once
  * results are reported per window (REPD's own unit) AND per digest (each digest inherits its window's
    decision), next to the DNN's per-digest result on the same digests
  * the information distance is used as a continuous score, so AUC/EER come from real ROC curves

  python3 revision/analysis/repd_eval.py --train 'results_rev/T*/rep[123]' --val 'results_rev/T*/rep4' \
      --test 'results_rev/S*_detect/rep*' --out results_rev/repd_eval
"""
import argparse, glob, os, json
from collections import Counter
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True); ap.add_argument("--val", required=True)
ap.add_argument("--test", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--alphas", default="2,3"); ap.add_argument("--windows", default="50,80,200")
ap.add_argument("--fields", default="dst_ip,src_ip")
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)


def runs(p):   # comma-separated globs allowed
    ds = {d for q in p.split(",") for d in glob.glob(q.strip())}
    return sorted(d for d in ds if os.path.exists(f"{d}/ctrl_logs/timing.csv")
                  and os.path.getsize(f"{d}/ctrl_logs/timing.csv") > 0)


def load(d):
    t = pd.read_csv(f"{d}/ctrl_logs/timing.csv", low_memory=False,
                    usecols=["src_ip", "dst_ip", "true_label", "prob", "pred", "t_ctrl_rx"])
    return t.sort_values("t_ctrl_rx").reset_index(drop=True)


def renyi(vals, a):
    c = np.array(list(Counter(vals).values()), dtype=float); p = c / c.sum()
    return float(-np.sum(p * np.log2(p))) if a == 1 else float(np.log2(np.sum(p ** a)) / (1 - a))


def windows(t, w, field, a):
    """One row per window: entropy, majority label; plus the window index of every digest."""
    idx = np.arange(len(t)) // w
    rows = [dict(win=i, H=renyi(g[field].tolist(), a), y=int(g.true_label.mean() >= 0.5), n=len(g))
            for i, g in t.groupby(idx)]
    return pd.DataFrame(rows), idx


def eer(y, s):
    f, tp, _ = roc_curve(y, s); n = 1 - tp; i = np.nanargmin(np.abs(n - f)); return float((f[i] + n[i]) / 2)


def mets(y, s, thr):
    p = (s >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    m = dict(acc=accuracy_score(y, p), prec=precision_score(y, p, zero_division=0), rec=recall_score(y, p, zero_division=0),
             f1=f1_score(y, p, zero_division=0), fpr=fp / max(fp + tn, 1), TN=tn, FP=fp, FN=fn, TP=tp)
    if 0 < y.sum() < len(y): m.update(auc=roc_auc_score(y, s), eer=eer(y, s))
    return m


TR, VA, TE = runs(A.train), runs(A.val), runs(A.test)
assert not (set(TR) & set(VA) or set(TR) & set(TE) or set(VA) & set(TE)), "a run is in two splits"
print(f"runs: train={len(TR)} val={len(VA)} test={len(TE)}")
tr, va = [load(d) for d in TR], [load(d) for d in VA]

# ---- choose (alpha, window, field, threshold) on validation
best = None
for a in [float(x) for x in A.alphas.split(",")]:
    for w in [int(x) for x in A.windows.split(",")]:
        for fld in A.fields.split(","):
            ref = [x for t in tr for x in windows(t[t.true_label == 0], w, fld, a)[0].H]
            href = float(np.mean(ref)) if ref else 0.0
            vw = pd.concat([windows(t, w, fld, a)[0] for t in va]); s = (vw.H - href).abs().values
            for thr in np.unique(np.quantile(s, np.linspace(0.01, 0.99, 99))):
                f = f1_score(vw.y, (s >= thr).astype(int), zero_division=0)
                if best is None or f > best["val_f1"]:
                    best = dict(alpha=a, window=w, field=fld, H_ref=href, thr=float(thr), val_f1=float(f))
json.dump(best, open(f"{A.out}/repd_chosen_on_val.json", "w"), indent=2); print("chosen on validation:", best)

# ---- test once, per scenario
rows = []
for d in TE:
    t = load(d); scen = os.path.basename(os.path.dirname(d))
    w, idx = windows(t, best["window"], best["field"], best["alpha"])
    s_win = (w.H - best["H_ref"]).abs().values
    s_dig = s_win[idx]                                    # each digest inherits its window's score
    rows.append(dict(scenario=scen, run=d, method="REPD", unit="window", **mets(w.y.values, s_win, best["thr"])))
    rows.append(dict(scenario=scen, run=d, method="REPD", unit="digest", **mets(t.true_label.values, s_dig, best["thr"])))
    rows.append(dict(scenario=scen, run=d, method="DNN (live, same digests)", unit="digest",
                     **mets(t.true_label.values, t.prob.values, 0.5 if t.pred.sum() == 0 else
                            float(t.prob[t.pred == 1].min()))))
df = pd.DataFrame(rows); df.to_csv(f"{A.out}/repd_vs_dnn_per_run.csv", index=False)
c = ["acc", "prec", "rec", "f1", "fpr", "auc", "eer"]
agg = df.groupby(["scenario", "method", "unit"])[c].agg(["mean", "std"]).round(4)
agg.columns = [f"{x}_{y}" for x, y in agg.columns]; agg["n_runs"] = df.groupby(["scenario", "method", "unit"]).run.nunique()
agg.to_csv(f"{A.out}/repd_vs_dnn_summary.csv")
allg = df.groupby(["method", "unit"])[c].mean().round(4); allg.to_csv(f"{A.out}/repd_vs_dnn_overall.csv")
print(allg.to_string())
