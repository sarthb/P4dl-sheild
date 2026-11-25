#!/usr/bin/env python3
"""
Offline experiments for the P4-DL Shield revision (Reviewer 3 #12,#15-#17,#21-#22; Reviewer 4; Reviewer 1 #3,#7).

Protocol (run-level, no leakage):
  TRAIN : dataset_runs/{attack_ldos_L02, attack_ldos_L025, attack_ldos_port6001, normal, normal_udp_5001_1M}
  VAL   : dataset_runs/{attack_ldos_L03, normal_udp_5001}  -> early stopping + threshold selection ONLY
  TEST  : results/pluto_detect_{A1,A2,A3}                   -> evaluated once, never used for any choice

Usage:  python3 offline_experiments.py <repo_root> <out_dir> [n_seeds]

REVISION v2 (training/train_eval_v2.py): same protocol, but
  * runs are given on the command line (glob of run dirs, each with ctrl_logs/live_dataset.csv)
  * the DNN is the paper's PyTorch model (BatchNorm + Dropout) when torch is installed
  * adds the 'delta' system-telemetry variant (Reviewer 1 #7)
  * --export saves the deployable model/scaler/threshold trained on TRAIN with VAL early stopping

  sudo -E python3 training/train_eval_v2.py \
      --train 'results_rev/T*/rep[123]' --val 'results_rev/T*/rep4' --test 'results_rev/S*/rep*' \
      --out results_rev/offline_eval --seeds 5 --export models/ddos_dnn_rev.pth
"""
import sys, os, json, warnings, time
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)

warnings.filterwarnings("ignore")
import argparse, glob
_ap = argparse.ArgumentParser()
_ap.add_argument("--train", required=True); _ap.add_argument("--val", required=True)
_ap.add_argument("--test", default=None, help="omit to only --export"); _ap.add_argument("--out", required=True)
_ap.add_argument("--seeds", type=int, default=5); _ap.add_argument("--export", default=None)
_ap.add_argument("--max-rows-per-run", type=int, default=0,
                 help="stratified cap per (run, class) after loading; 0 = use all rows. Keeps the ablations tractable")
_ap.add_argument("--featsets", default="", help="comma-separated subset of feature sets for the DNN ablation (default: all)")
_ap.add_argument("--export-featset", default="no_system_telemetry",
                 help="feature set of the deployed model (see FEATSETS); absolute host counters drift over time")
ARGS = _ap.parse_args()
OUT, NSEEDS = ARGS.out, ARGS.seeds
os.makedirs(OUT, exist_ok=True)
try:
    import torch, torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

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
    "delta_system_telemetry": [f for f in FULL if f not in SYS] + ["d_cpu", "d_mem", "d_tx_MB", "d_rx_MB"],
    # ablations of the DEPLOYED 8-feature model (no system telemetry)
    "deployed_no_tag": [f for f in FULL if f not in SYS and f != "tag"],
    "deployed_no_pkt_len": [f for f in FULL if f not in SYS and f != "pkt_len"],
    "deployed_no_protocol": [f for f in FULL if f not in SYS and f != "proto"],
}

def _runs(pattern):   # comma-separated globs allowed
    ds = {d for p in pattern.split(",") for d in glob.glob(p.strip())}
    return sorted(d for d in ds if os.path.exists(f"{d}/ctrl_logs/live_dataset.csv")
                  and os.path.getsize(f"{d}/ctrl_logs/live_dataset.csv") > 0)


TRAIN, VAL, TEST_DIRS = _runs(ARGS.train), _runs(ARGS.val), (_runs(ARGS.test) if ARGS.test else [])
assert not (set(TRAIN) & set(VAL)) and not (set(TRAIN) & set(TEST_DIRS)) and not (set(VAL) & set(TEST_DIRS)), \
    "a run appears in more than one split"
TEST = sorted({os.path.basename(os.path.dirname(d)) for d in TEST_DIRS})   # report per scenario


def load():
    def rd(d, scen_level):
        x = pd.read_csv(f"{d}/ctrl_logs/live_dataset.csv", low_memory=False)
        if ARGS.max_rows_per_run > 0:      # stratified cap per (run, class), fixed seed -> reproducible
            x = pd.concat([g.sample(min(len(g), ARGS.max_rows_per_run), random_state=0)
                           for _, g in x.groupby("Label")]).sort_values("timestamp")
        x["run"] = os.path.basename(os.path.dirname(d)) if scen_level else d
        x["run_dir"] = d
        print(f"  loaded {d}: {len(x)} rows", flush=True)
        return x
    tr = pd.concat([rd(d, False) for d in TRAIN]); va = pd.concat([rd(d, False) for d in VAL])
    te = pd.concat([rd(d, True) for d in TEST_DIRS]) if TEST_DIRS else tr.iloc[:0].copy()
    print(f"runs: train={len(TRAIN)} val={len(VAL)} test={len(TEST_DIRS)}; rows: {len(tr)}/{len(va)}/{len(te)}")
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


class _TorchDNN:
    """Exactly the paper's DDoSDNN: 128-BN-ReLU-Drop0.3-64-BN-ReLU-Drop0.3-32-ReLU-1."""
    def __init__(self, d, seed):
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def predict_proba(self, X):
        self.net.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net(torch.tensor(X, dtype=torch.float32))).numpy().ravel()
        return np.c_[1 - p, p]


def fit_torch(Xtr, ytr, Xva, yva, seed):
    import copy
    m = _TorchDNN(Xtr.shape[1], seed); opt = torch.optim.Adam(m.net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss(); Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1); g = torch.Generator().manual_seed(seed)
    best, best_f1, bad = None, -1, 0
    for ep in range(100):
        m.net.train()
        for idx in torch.randperm(len(Xt), generator=g).split(256):
            if len(idx) < 2: continue
            opt.zero_grad(); lossf(m.net(Xt[idx]), yt[idx]).backward(); opt.step()
        f = f1_score(yva, (m.predict_proba(Xva)[:, 1] >= 0.5).astype(int), zero_division=0)
        if f > best_f1 + 1e-6: best_f1, best, bad = f, copy.deepcopy(m), 0
        else:
            bad += 1
            if bad >= 10: break
    return best


def fit_mlp(Xtr, ytr, Xva, yva, seed):
    if HAVE_TORCH:
        return fit_torch(Xtr, ytr, Xva, yva, seed)
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
        elif name == "XGBoost":
            from xgboost import XGBClassifier
            mdl = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=seed,
                                n_jobs=-1, eval_metric="logloss").fit(Xtr, ytr)
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
    if not TEST_DIRS:
        assert ARGS.export, "give --test and/or --export"
        export(tr, va); return
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
    try:
        import xgboost  # noqa: F401
        extra = ["XGBoost"]
    except ImportError:
        extra = []
    fsets = [f for f in ARGS.featsets.split(",") if f] or list(FEATSETS)
    plan = [("DNN", k) for k in fsets] + [(m, "full_12") for m in ["LogReg", "RandomForest", "GradBoost"] + extra] \
        + [("RandomForest", "no_pkt_len_no_proto"), ("GradBoost", "no_pkt_len_no_proto")]
    for name, fs in plan:
        feats = FEATSETS[fs]
        seeds = range(NSEEDS) if name in ("DNN",) else range(min(NSEEDS, 5))
        for s in seeds:
            for r in run_model(name, feats, tr, va, te, s):
                r["featset"] = fs; res.append(r)
        print("done", name, fs, time.strftime("%H:%M:%S"), flush=True)
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
    if ARGS.export:
        export(tr, va)
    print(agg.reset_index().query("case=='ALL' or case=='ALL_padded_attack'")[
        ["model", "featset", "case", "f1_mean", "f1_std", "auc_mean", "eer_mean", "fpr_mean", "thr_mean", "n_seeds"]].to_string())


def export(tr, va):
    """Deployable model in the format controller/dnn_model.py loads (torch required)."""
    import joblib
    assert HAVE_TORCH, "export needs torch"
    FE = FEATSETS[ARGS.export_featset]
    sc = StandardScaler().fit(tr[FE].astype(float).values)
    Xtr, Xva = sc.transform(tr[FE].astype(float).values), sc.transform(va[FE].astype(float).values)
    m = fit_torch(Xtr, tr.Label.values, Xva, va.Label.values, 0)
    thr = best_thr(va.Label.values, m.predict_proba(Xva)[:, 1])
    # keys must be "net.<i>.<param>" to match controller/dnn_model.py:DDoSDNN (self.net = Sequential)
    sd = {"net." + k: v for k, v in m.net.state_dict().items()}
    torch.save({"model_state_dict": sd, "input_dim": len(FE), "features": FE, "featset": ARGS.export_featset,
                "threshold": thr, "train_runs": TRAIN, "val_runs": VAL}, ARGS.export)
    sp = ARGS.export.replace(".pth", "_scaler.pkl"); joblib.dump(sc, sp)
    json.dump({"model": ARGS.export, "scaler": sp, "threshold_from_val": thr}, open(f"{OUT}/exported_model.json", "w"), indent=2)
    json.dump({"model": ARGS.export, "scaler": sp, "threshold_from_val": thr, "featset": ARGS.export_featset,
               "features": FE}, open(f"{OUT}/exported_model.json", "w"), indent=2)
    print(f"exported {ARGS.export} (+{sp}); features={ARGS.export_featset} ({len(FE)}); threshold chosen on VAL = {thr}")


if __name__ == "__main__":
    main()
