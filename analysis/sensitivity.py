#!/usr/bin/env python3
"""
sensitivity.py — offline, from EXISTING detection-only test runs (no new testbed runs).

Answers Reviewer 4 (full report):
  Methods 1: the tag's own precision / recall / F1 as a detector (digest level), per scenario.
  Results 4: sensitivity to the decision threshold
             (a) digest level: precision, recall, F1, FPR of the deployed DNN at several thresholds
             (b) flow level: deployed blocking policy replayed at several thresholds and K values
                 -> benign flows blocked, attack flows blocked, onset-to-block time.

Uses the probabilities the controller logged live (ctrl_logs/timing.csv, column prob), so the scores are
exactly the deployed model's. Flow-level replay uses the same rule as the controller (see policy_replay.py).

  python3 revision/analysis/sensitivity.py \
     --runs 'results_rev/S1_single_A2_detect/rep*,results_rev/S2_distributed_detect/rep*,results_rev/S3_*/rep*,results_rev/S4_*/rep*' \
     --policy models/block_policy.json --out ~/sensitivity
"""
import argparse, glob, json, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True, help="comma-separated globs of detection-only run folders")
ap.add_argument("--policy", default="models/block_policy.json")
ap.add_argument("--out", required=True)
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)

THRS = [0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999]
KS = [1, 5, 20, 50, 100]
pol = json.load(open(A.policy))
GAP, SUS = pol.get("gap_s", 0.3), pol.get("sustain_s", 0.5)


def first_block(pos, K, W, B, R, G=GAP, L=SUS):
    lo = 0; eps = []; last = None
    for i, now in enumerate(pos):
        if last is None or now - last > G: eps.append(now)
        last = now
        while now - pos[lo] > W: lo += 1
        while eps and now - eps[0] > R and len(eps) > 1: eps.pop(0)
        n_eps = sum(1 for e in eps if now - e <= R)
        if i - lo + 1 >= K and (n_eps >= B or now - eps[-1] >= L):
            return float(now)
    return None


def prf(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=pr, recall=rc, f1=f1, fpr=fp / (fp + tn) if fp + tn else 0.0)


runs = []
for g in A.runs.split(","):
    for d in sorted(glob.glob(g.strip())):
        f = f"{d}/ctrl_logs/timing.csv"
        if not os.path.exists(f) or os.path.getsize(f) == 0: continue
        x = pd.read_csv(f, low_memory=False, usecols=["src_ip", "dst_ip", "proto", "dport", "true_label",
                                                      "tag", "prob", "t_ctrl_rx"]).dropna()
        if len(x) == 0: continue
        x["scenario"] = os.path.basename(os.path.dirname(d.rstrip("/"))); x["run"] = d
        runs.append(x)
assert runs, "no runs found"
print(f"{len(runs)} runs, {sum(len(x) for x in runs):,} digests")

# ---- 1) tag as a detector, and 2a) digest-level threshold sweep
tag_rows, thr_rows = [], []
for x in runs:
    y = x.true_label.astype(int).values
    tag_rows.append(dict(scenario=x.scenario.iloc[0], run=x.run.iloc[0], **prf(y, (x.tag.astype(int).values > 0).astype(int))))
    for t in THRS:
        thr_rows.append(dict(scenario=x.scenario.iloc[0], run=x.run.iloc[0], thr=t, **prf(y, (x.prob.values >= t).astype(int))))
tag = pd.DataFrame(tag_rows); thr = pd.DataFrame(thr_rows)
tag.to_csv(f"{A.out}/tag_detector_per_run.csv", index=False); thr.to_csv(f"{A.out}/threshold_digest_per_run.csv", index=False)
m = ["precision", "recall", "f1", "fpr"]
tag_s = tag.groupby("scenario")[m].mean()
tag_s.loc["MEAN over scenarios"] = tag_s.mean()
pooled = prf(np.concatenate([x.true_label.astype(int).values for x in runs]),
             np.concatenate([(x.tag.astype(int).values > 0).astype(int) for x in runs]))
tag_s.loc["POOLED (all digests)"] = [pooled[k] for k in m]
thr_s = thr.groupby(["thr", "scenario"])[m].mean().groupby("thr").agg(["mean"])
thr_s.columns = [a for a, _ in thr_s.columns]

# ---- 2b) flow-level policy sensitivity (deployed policy, vary threshold, then vary K)
flows = []
for x in runs:
    for key, g in x.sort_values("t_ctrl_rx").groupby(["src_ip", "dst_ip", "proto", "dport"]):
        flows.append((x.scenario.iloc[0], x.run.iloc[0], int(g.true_label.max()), g.t_ctrl_rx.values, g.prob.values))


def replay(t, K):
    rows = []
    for sc, r, lab, ts, p in flows:
        b = first_block(ts[p >= t], K, pol["window_s"], pol["min_bins"], pol["recur_window_s"])
        rows.append(dict(scenario=sc, run=r, label=lab, delay=(b - ts[0]) if b is not None else np.nan, blocked=b is not None))
    fl = pd.DataFrame(rows); ben, att = fl[fl.label == 0], fl[fl.label == 1]
    return dict(thr=t, K=K, benign_flows=len(ben), benign_blocked=int(ben.blocked.sum()),
                attack_flows=len(att), attack_blocked=int(att.blocked.sum()),
                block_delay_s_mean=att.delay.mean(), block_delay_s_max=att.delay.max())


pol_rows = [replay(t, pol["K"]) for t in THRS] + [replay(pol["threshold"], k) for k in KS if k != pol["K"]]
pol_t = pd.DataFrame(pol_rows); pol_t.to_csv(f"{A.out}/threshold_policy_flow_level.csv", index=False)

pd.set_option("display.width", 200)
with open(f"{A.out}/summary.txt", "w") as fo:
    for title, df in (("TAG AS A DETECTOR (digest level, mean over runs per scenario)", tag_s),
                      ("DEPLOYED DNN, DIGEST LEVEL vs THRESHOLD (mean over scenarios)", thr_s),
                      (f"FLOW LEVEL: deployed policy (K={pol['K']}, W={pol['window_s']}s, B={pol['min_bins']}, "
                       f"R={pol['recur_window_s']}s) at other thresholds, then other K at thr={pol['threshold']}", pol_t)):
        s = f"\n=== {title}\n{df.round(4).to_string()}\n"; print(s); fo.write(s)
print(f"Send back: {A.out}/summary.txt")
