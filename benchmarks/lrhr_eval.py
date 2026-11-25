#!/usr/bin/env python3
"""
lrhr_eval.py — LR-HR DDoS 2024 (github.com/nqmn/lr-hr_ddos_2024_ddos_dataset) evaluation for Table V.

Dataset facts this script measures and reports:
  * 61.3 % of rows are exact duplicates  -> a random row split leaks copies of test rows into training
  * protocol == 0 (and ports == 0) for ALL attack rows, only 280 benign rows -> protocol gives the label away
  * the file is ordered (all benign, then all attack) -> no time-ordered split is possible

Protocols compared (5 repeated stratified 60/20/20 splits, seed = split seed = model seed):
  P1_random_rows      : random split of raw rows (what most papers do, incl. duplicates)
  P2_dedup            : duplicates removed first, then random split (no shared rows across splits)
Feature sets:
  all_incl_proto_ports  | cleaned_no_proto_ports (the repo's cleaned CSV) |
  live_mapped_7         (proto, Pkt Size Avg, packet_count, byte_count, flow_duration, Flow Pkts/s, IAT=duration/pkts)
  live_mapped_no_proto
Validation (20 %) chooses the DNN early stop and the threshold for every model; test (20 %) is used once.

  python3 lrhr_eval.py "<path>/LR-HR DDoS 2024 Dataset for SDN-Based Networks.csv" <out_dir> [seeds]
"""
import sys, os, copy, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
warnings.filterwarnings("ignore")
CSV, OUT = sys.argv[1], sys.argv[2]; SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
os.makedirs(OUT, exist_ok=True)

d = pd.read_csv(CSV)
d["iat"] = d.flow_duration / d.packet_count.clip(lower=1)
FS = {
    "all_incl_proto_ports": [c for c in d.columns if c not in ("Label", "iat")],
    "cleaned_no_proto_ports": [c for c in d.columns if c not in ("Label", "iat", "protocol", "srcport", "dstport")],
    "live_mapped_7": ["protocol", "Pkt Size Avg", "packet_count", "byte_count", "flow_duration", "Flow Pkts/s", "iat"],
    "live_mapped_no_proto": ["Pkt Size Avg", "packet_count", "byte_count", "flow_duration", "Flow Pkts/s", "iat"],
}
facts = dict(rows=len(d), attack=int(d.Label.sum()), benign=int((d.Label == 0).sum()),
             exact_duplicate_rows=int(d.duplicated().sum()), unique_rows=int((~d.duplicated()).sum()),
             attack_rows_with_protocol0=int(((d.protocol == 0) & (d.Label == 1)).sum()),
             benign_rows_with_protocol0=int(((d.protocol == 0) & (d.Label == 0)).sum()),
             label_runs_in_file_order=int((d.Label != d.Label.shift()).sum()))
pd.Series(facts).to_csv(f"{OUT}/dataset_facts.csv"); print(facts)


def eer(y, s):
    f, t, _ = roc_curve(y, s); n = 1 - t; i = np.nanargmin(np.abs(n - f)); return float((f[i] + n[i]) / 2)


def mets(y, s, thr):
    p = (s >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(acc=accuracy_score(y, p), bal_acc=balanced_accuracy_score(y, p), prec=precision_score(y, p, zero_division=0),
                rec=recall_score(y, p, zero_division=0), f1=f1_score(y, p, zero_division=0), fpr=fp / max(fp + tn, 1),
                auc=roc_auc_score(y, s), eer=eer(y, s), TN=tn, FP=fp, FN=fn, TP=tp)


def thr_on_val(y, s):
    g = np.linspace(0.02, 0.98, 49); return float(g[int(np.argmax([f1_score(y, (s >= t).astype(int)) for t in g]))])


def dnn(X, y, Xv, yv, seed):
    m = MLPClassifier((128, 64, 32), learning_rate_init=1e-3, batch_size=256, alpha=1e-5, random_state=seed)
    best, bs, bad, rng = None, -1, 0, np.random.RandomState(seed)
    for ep in range(80):
        idx = rng.permutation(len(X))
        for b in range(0, len(idx), 4096): m.partial_fit(X[idx[b:b + 4096]], y[idx[b:b + 4096]], classes=[0, 1])
        s = f1_score(yv, (m.predict_proba(Xv)[:, 1] >= .5).astype(int))
        if s > bs + 1e-6: bs, best, bad = s, copy.deepcopy(m), 0
        else:
            bad += 1
            if bad >= 10: break
    return best


rows = []
for proto_name, data in (("P1_random_rows", d), ("P2_dedup", d.drop_duplicates().reset_index(drop=True))):
    for seed in range(SEEDS):
        tr, tmp = train_test_split(data, test_size=0.4, stratify=data.Label, random_state=seed)
        va, te = train_test_split(tmp, test_size=0.5, stratify=tmp.Label, random_state=seed)
        key = lambda x: pd.util.hash_pandas_object(x.drop(columns=["iat"]), index=False)
        leak = float(key(te).isin(set(key(tr))).mean())
        # single-field rule baseline
        s = (te.protocol == 0).astype(float).values
        rows.append(dict(protocol=proto_name, featset="rule: protocol==0", model="rule", seed=seed,
                         test_rows_also_in_train=leak, **mets(te.Label.values, s, 0.5)))
        for fs, cols in FS.items():
            sc = StandardScaler().fit(tr[cols]); Xtr, Xva, Xte = (sc.transform(x[cols]) for x in (tr, va, te))
            ytr, yva, yte = tr.Label.values, va.Label.values, te.Label.values
            for name in ("DNN", "LogReg", "RandomForest", "GradBoost"):
                if name == "DNN": m = dnn(Xtr, ytr, Xva, yva, seed)
                elif name == "LogReg": m = LogisticRegression(max_iter=3000).fit(Xtr, ytr)
                elif name == "RandomForest": m = RandomForestClassifier(200, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
                else: m = HistGradientBoostingClassifier(random_state=seed).fit(Xtr, ytr)
                t = thr_on_val(yva, m.predict_proba(Xva)[:, 1])
                rows.append(dict(protocol=proto_name, featset=fs, model=name, seed=seed, thr=t,
                                 test_rows_also_in_train=leak, **mets(yte, m.predict_proba(Xte)[:, 1], t)))
            print(proto_name, seed, fs, flush=True)
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/raw_results.csv", index=False)
c = ["acc", "bal_acc", "prec", "rec", "f1", "fpr", "auc", "eer", "test_rows_also_in_train"]
agg = df.groupby(["protocol", "featset", "model"])[c].agg(["mean", "std"])
agg.columns = [f"{a}_{b}" for a, b in agg.columns]; agg["n_splits"] = df.groupby(["protocol", "featset", "model"]).seed.nunique()
agg.round(4).to_csv(f"{OUT}/summary.csv")
print(agg[["acc_mean", "acc_std", "f1_mean", "fpr_mean", "auc_mean", "eer_mean", "test_rows_also_in_train_mean"]].round(4).to_string())
