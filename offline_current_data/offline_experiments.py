#!/usr/bin/env python3
"""
Offline experiments for the P4-DL Shield revision (Reviewer 3 #12,#15-#17,#21-#22; Reviewer 4; Reviewer 1 #3,#7).

Protocol (run-level, no leakage):
  TRAIN : dataset_runs/{attack_ldos_L02, attack_ldos_L025, attack_ldos_port6001, normal, normal_udp_5001_1M}
  VAL   : dataset_runs/{attack_ldos_L03, normal_udp_5001}  -> early stopping + threshold selection ONLY
  TEST  : results/pluto_detect_{A1,A2,A3}                   -> evaluated once, never used for any choice

Usage:  python3 offline_experiments.py <repo_root> <out_dir> [n_seeds]
"""
import sys, os, json, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)

warnings.filterwarnings("ignore")
ROOT, OUT = sys.argv[1], sys.argv[2]
NSEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
os.makedirs(OUT, exist_ok=True)

FULL = ["proto", "tag", "pkt_len", "flow_packet_count", "flow_byte_count", "flow_duration",
        "flow_packets_per_second", "inter_arrival_time", "cpu", "mem", "tx_MB", "rx_MB"]
SYS = ["cpu", "mem", "tx_MB", "rx_MB"]
FLOW = ["flow_packet_count", "flow_byte_count", "flow_duration", "flow_packets_per_second", "inter_arrival_time"]
FEATSETS = {
    "full_12": FULL,
    "no_tag": [f for f in FULL if f != "tag"],
    "no_system_telemetry": [f for f in FULL if f not in SYS],
    "no_protocol": [f for f in FULL if f != "proto"],
    "no_pkt_len": [f for f in FULL if f != "pkt_len"],
    "no_pkt_len_no_proto": [f for f in FULL if f not in ("pkt_len", "proto")],
    "flow_temporal_only": [f for f in FLOW if f != "flow_byte_count"],
    "pkt_len_only": ["pkt_len"],
    "tag_only": ["tag"],
}

TRAIN = ["attack_ldos_L02", "attack_ldos_L025", "attack_ldos_port6001", "normal", "normal_udp_5001_1M"]
VAL = ["attack_ldos_L03", "normal_udp_5001"]
TEST = ["A1", "A2", "A3"]


def load():
    def rd(p, run):
        d = pd.read_csv(p); d["run"] = run; return d
    tr = pd.concat([rd(f"{ROOT}/dataset_runs/{r}/live_dataset.csv", r) for r in TRAIN])
    va = pd.concat([rd(f"{ROOT}/dataset_runs/{r}/live_dataset.csv", r) for r in VAL])
    te = pd.concat([rd(f"{ROOT}/results/pluto_detect_{c}/live_dataset.csv", c) for c in TEST])
    return [d.dropna(subset=FULL + ["Label"]).reset_index(drop=True) for d in (tr, va, te)]


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr)); return float((fpr[i] + fnr[i]) / 2)


def metrics(y, s, thr):
    p = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    m = dict(acc=accuracy_score(y, p), prec=precision_score(y, p, zero_division=0),
             rec=recall_score(y, p, zero_division=0), f1=f1_score(y, p, zero_division=0),
             fpr=fp / max(fp + tn, 1), TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))
    if 0 < y.sum() < len(y):
        m["auc"] = roc_auc_score(y, s); m["eer"] = eer(y, s)
    return m


def best_thr(y, s):
    grid = np.unique(np.concatenate([np.linspace(0.05, 0.95, 19), [0.5, 0.7]]))
    f = [f1_score(y, (s >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f))])


def fit_mlp(Xtr, ytr, Xva, yva, seed):
    """DNN 128-64-32 (same widths as the paper), Adam lr=1e-3, batch 256,
    early stopping on VALIDATION F1 with patience 10 (max 100 epochs)."""
    m = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
                      learning_rate_init=1e-3, batch_size=256, alpha=1e-5, random_state=seed)
    rng = np.random.RandomState(seed)
    best, best_f1, bad = None, -1, 0
    import copy
    for ep in range(100):
        idx = rng.permutation(len(Xtr))
        for b in range(0, len(idx), 4096):  # partial_fit internally batches at 256
            m.partial_fit(Xtr[idx[b:b + 4096]], ytr[idx[b:b + 4096]], classes=[0, 1])
        f = f1_score(yva, (m.predict_proba(Xva)[:, 1] >= 0.5).astype(int), zero_division=0)
        if f > best_f1 + 1e-6:
            best_f1, best, bad = f, copy.deepcopy(m), 0
        else:
            bad += 1
            if bad >= 10: break
    return best


class Rule:
    """Single-feature threshold detector; direction and cut chosen on TRAIN+VAL by F1."""
    def __init__(self, feat): self.feat = feat
    def fit(self, X, y, cols):
        x = X[:, cols.index(self.feat)]; best = (-1, None, None)
        for c in np.unique(np.quantile(x, np.linspace(0, 1, 201))):
            for sgn in (1, -1):
                f = f1_score(y, (sgn * x >= sgn * c).astype(int), zero_division=0)
                if f > best[0]: best = (f, c, sgn)
        _, self.c, self.sgn = best; self.cols = cols; return self
    def score(self, X):
        x = X[:, self.cols.index(self.feat)]
        return (self.sgn * (x - self.c) >= 0).astype(float)


def run_model(name, feats, tr, va, te, seed):
    Xtr, Xva, Xte = (d[feats].astype(float).values for d in (tr, va, te))
    ytr, yva, yte = (d.Label.astype(int).values for d in (tr, va, te))
    if name.startswith("rule:"):
        r = Rule(name[5:]).fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]), feats)
        sva, ste, thr = r.score(Xva), r.score(Xte), 0.5
    else:
        sc = StandardScaler().fit(Xtr); Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
        if name == "DNN": mdl = fit_mlp(Xtr, ytr, Xva, yva, seed)
        elif name == "LogReg": mdl = LogisticRegression(max_iter=2000, random_state=seed).fit(Xtr, ytr)
        elif name == "RandomForest": mdl = RandomForestClassifier(200, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
        elif name == "GradBoost": mdl = HistGradientBoostingClassifier(random_state=seed).fit(Xtr, ytr)
        sva, ste = mdl.predict_proba(Xva)[:, 1], mdl.predict_proba(Xte)[:, 1]
        thr = best_thr(yva, sva)  # threshold from VALIDATION only
    rows = []
    for case in ["ALL"] + TEST:
        mask = np.ones(len(te), bool) if case == "ALL" else (te.run == case).values
        m = metrics(yte[mask], ste[mask], thr)
        rows.append(dict(model=name, featset=None, seed=seed, case=case, thr=thr, **m))
    # robustness probe (R1 #3): attacker pads packets to the benign UDP size (1498 B)
    if "pkt_len" in feats and not name.startswith("rule:"):
        te2 = te.copy(); att = te2.Label == 1
        te2.loc[att, "pkt_len"] = 1498
        te2.loc[att, "flow_byte_count"] = te2.loc[att, "flow_packet_count"] * 1498
        X2 = sc.transform(te2[feats].astype(float).values)
        m = metrics(yte, mdl.predict_proba(X2)[:, 1], thr)
        rows.append(dict(model=name, featset=None, seed=seed, case="ALL_padded_attack", thr=thr, **m))
    return rows


def main():
    tr, va, te = load()
    # ---------- A1: audit table ----------
    aud = []
    for split, d in (("train", tr), ("val", va), ("test", te)):
        for r, g in d.groupby("run"):
            aud.append(dict(split=split, run=r, rows=len(g), attack_rows=int(g.Label.sum()),
                            benign_rows=int((g.Label == 0).sum()), duration_s=round(g.timestamp.max() - g.timestamp.min(), 1),
                            tag_eq_1=int((g.tag == 1).sum()),
                            attack_pkt_len=sorted(g[g.Label == 1].pkt_len.unique().tolist()),
                            benign_udp_pkt_len=sorted(g[(g.Label == 0) & (g.proto == 17)].pkt_len.unique().tolist()),
                            switches_reporting=g.device_id.nunique()))
    pd.DataFrame(aud).to_csv(f"{OUT}/A1_dataset_audit.csv", index=False)
    var = pd.concat([tr, va, te])[FULL].nunique().rename("distinct_values")
    var.to_csv(f"{OUT}/A1_feature_distinct_values.csv")

    # ---------- A2/A3/A4 ----------
    res = []
    plan = [("DNN", k) for k in FEATSETS] + [(m, "full_12") for m in ("LogReg", "RandomForest", "GradBoost")] \
        + [("RandomForest", "no_pkt_len_no_proto"), ("GradBoost", "no_pkt_len_no_proto")]
    for name, fs in plan:
        feats = FEATSETS[fs]
        seeds = range(NSEEDS) if name in ("DNN",) else range(min(NSEEDS, 5))
        for s in seeds:
            for r in run_model(name, feats, tr, va, te, s):
                r["featset"] = fs; res.append(r)
        print("done", name, fs, flush=True)
    for rule, fs in (("rule:pkt_len", "full_12"), ("rule:flow_packets_per_second", "full_12"),
                     ("rule:inter_arrival_time", "full_12"), ("rule:tag", "full_12")):
        for r in run_model(rule, FEATSETS[fs], tr, va, te, 0):
            r["featset"] = rule[5:]; res.append(r)
    df = pd.DataFrame(res)
    df.to_csv(f"{OUT}/raw_results_all_seeds.csv", index=False)
    cols = ["acc", "prec", "rec", "f1", "auc", "eer", "fpr", "thr"]
    agg = df.groupby(["model", "featset", "case"])[cols].agg(["mean", "std"]).round(4)
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = df.groupby(["model", "featset", "case"]).seed.nunique()
    agg.reset_index().to_csv(f"{OUT}/summary_mean_std.csv", index=False)
    cm = df[df.case.isin(TEST + ["ALL"])].groupby(["model", "featset", "case"])[["TN", "FP", "FN", "TP"]].mean().round(1)
    cm.reset_index().to_csv(f"{OUT}/confusion_matrices_mean.csv", index=False)
    print(agg.reset_index().query("case=='ALL' or case=='ALL_padded_attack'")[
        ["model", "featset", "case", "f1_mean", "f1_std", "auc_mean", "eer_mean", "fpr_mean", "thr_mean", "n_seeds"]].to_string())


if __name__ == "__main__":
    main()
