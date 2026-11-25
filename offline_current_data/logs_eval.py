#!/usr/bin/env python3
"""Recompute Table III / IV inputs from the committed logs (results/pluto_*), with the
latency breakdown. Labels use the same rule as evaluate_pluto_comparison.py.
Usage: python3 logs_eval.py <repo_root>"""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
R = sys.argv[1] + "/results/"
def eer(y, s):
    f, t, _ = roc_curve(y, s); n = 1 - t; i = np.nanargmin(np.abs(n - f)); return (f[i] + n[i]) / 2
rows = []
for m in ["detect", "mitigate"]:
    for c in ["A1", "A2", "A3"]:
        d = R + f"pluto_{m}_{c}/"; df = pd.read_csv(d + "ml_predictions.csv")
        st = float(open(d + "attack_start.txt").read()); en = float(open(d + "attack_end.txt").read())
        y = ((df.src_ip == 167772674) & (df.dst_ip == 167773185) & (df.proto == 17) & (df.dport == 5001)).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, df.label, labels=[0, 1]).ravel()
        a = df[y == 1]; first = a.timestamp.min(); det = a[a.label == 1].timestamp.min()
        r = dict(run=f"{m}_{c}", rows=len(df), attack_digests=int(y.sum()), TN=tn, FP=fp, FN=fn, TP=tp,
                 attack_cmd_to_first_attack_digest_s=round(first - st, 3),
                 first_attack_digest_to_detection_s=round(det - first, 3),
                 attack_digests_before_detection=int((a.timestamp < det).sum()),
                 attack_digests_per_switch_per_s=round(len(a[a.device_id == 4]) / (en - st), 2),
                 benign_prob_constant=round(float(df[y == 0].prob.iloc[0]), 6))
        if y.sum() > 10:
            r.update(auc=round(roc_auc_score(y, df.prob), 4), eer=round(eer(y, df.prob), 4),
                     one_minus_auc=round(1 - roc_auc_score(y, df.prob), 4))
        rows.append(r)
pd.set_option("display.width", 250); print(pd.DataFrame(rows).to_string(index=False))
