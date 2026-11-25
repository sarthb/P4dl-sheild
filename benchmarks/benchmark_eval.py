#!/usr/bin/env python3
"""
benchmark_eval.py — clean benchmark evaluation for Table V (Reviewer 3 #18, Reviewer 4).

Protocol (state this in the paper):
  * One model is RETRAINED PER DATASET. The live model cannot be transferred, because the
    benchmarks have no suspicion tag (and CIC-DDoS2019 has no host telemetry).
  * Feature set "live_mapped" = the 7 live features with a direct equivalent (mapping below).
    Feature set "all_flow" = all flow statistics, with NO identifiers (IPs, MACs, ports, flow IDs, timestamps).
  * CIC-DDoS2019: train = 01-12 day, test = 03-11 day (the dataset's own split, different capture day).
    Validation = last 20 % (by timestamp) of every training-day file. Early stopping and the
    threshold use validation only; test is evaluated once.
  * Other datasets (generic CSV, --label-col / --time-col): a time-ordered 60/20/20 split.
  * Because benign traffic is only a tiny fraction of CIC-DDoS2019, accuracy is misleading on its own:
    balanced accuracy, benign recall (= 1 - FPR), PR-AUC and the majority-class baseline are reported too.

Mapping to live features (CICFlowMeter column -> live feature):
  Protocol -> proto | Packet Length Mean -> pkt_len | Total Fwd+Bwd Packets -> flow_packet_count
  Total Length Fwd+Bwd -> flow_byte_count | Flow Duration (us) -> flow_duration (s)
  Flow Packets/s -> flow_packets_per_second | Flow IAT Mean (us) -> inter_arrival_time (s)
  tag, cpu, mem, tx_MB, rx_MB -> not available

  python3 benchmark_eval.py --dataset cic2019 --csv cic2019_sample.csv.gz --out bench_cic --seeds 5
  python3 benchmark_eval.py --dataset generic --csv lrhr2024.csv --label-col Label --time-col Timestamp \
          --drop "Src IP,Dst IP,Flow ID,Src Port,Dst Port" --out bench_lrhr
"""
import argparse, os, json, copy, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, roc_curve, confusion_matrix)
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["cic2019", "generic"], required=True)
ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--label-col", default="Label")
ap.add_argument("--benign-value", default="BENIGN"); ap.add_argument("--time-col", default=None)
ap.add_argument("--drop", default="", help="comma-separated identifier columns to drop (generic)")
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
try:
    import torch, torch.nn as nn; HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
IDS = {"Flow ID", "Source IP", "Destination IP", "Source Port", "Destination Port", "Timestamp", "SimillarHTTP",
       "Unnamed: 0", "Inbound", "IP_Src", "IP_Dst", "Mac_Src", "Mac_Dst", "Port_Src", "Port_Dst", "FlowID",
       "DeviceID", "day", "file", "Label", "y", "_t"}


def load():
    d = pd.read_csv(A.csv, low_memory=False); d.columns = [c.strip() for c in d.columns]
    lab = d[A.label_col].astype(str).str.strip().str.upper()
    d["y"] = (lab != A.benign_value.upper()).astype(int) if not set(lab.unique()) <= {"0", "1"} else lab.astype(int)
    tcol = "Timestamp" if A.dataset == "cic2019" else A.time_col
    d["_t"] = pd.to_datetime(d[tcol], errors="coerce") if tcol else pd.RangeIndex(len(d))
    if A.dataset == "cic2019":
        g = lambda c: pd.to_numeric(d[c], errors="coerce")
        d["proto"] = g("Protocol"); d["pkt_len"] = g("Packet Length Mean")
        d["flow_packet_count"] = g("Total Fwd Packets") + g("Total Backward Packets")
        d["flow_byte_count"] = g("Total Length of Fwd Packets") + g("Total Length of Bwd Packets")
        d["flow_duration"] = g("Flow Duration") / 1e6; d["flow_packets_per_second"] = g("Flow Packets/s")
        d["inter_arrival_time"] = g("Flow IAT Mean") / 1e6
        live = ["proto", "pkt_len", "flow_packet_count", "flow_byte_count", "flow_duration",
                "flow_packets_per_second", "inter_arrival_time"]
        tr_all = d[d.day == "01-12"]; te = d[d.day == "03-11"]
        q = tr_all.groupby("file")["_t"].transform(lambda t: t.rank(pct=True, method="first"))
        tr, va = tr_all[q <= 0.8], tr_all[q > 0.8]
    else:
        drop = {c.strip() for c in A.drop.split(",") if c.strip()}
        d = d.drop(columns=[c for c in drop if c in d.columns]).sort_values("_t").reset_index(drop=True)
        n = len(d); tr, va, te = d.iloc[:int(.6 * n)], d.iloc[int(.6 * n):int(.8 * n)], d.iloc[int(.8 * n):]
        live = None
    num = [c for c in d.columns if c not in IDS and c not in (A.drop.split(",") if A.drop else [])
           and pd.api.types.is_numeric_dtype(d[c])]
    allflow = [c for c in num if c not in {"proto", "pkt_len", "flow_packet_count", "flow_byte_count",
                                            "flow_duration", "flow_packets_per_second", "inter_arrival_time"}]
    sets = {"all_flow_no_identifiers": allflow}
    if live: sets["live_mapped_7"] = live
    clean = lambda x: x.replace([np.inf, -np.inf], np.nan)
    return [clean(x) for x in (tr, va, te)], sets


def eer(y, s):
    f, t, _ = roc_curve(y, s); n = 1 - t; i = np.nanargmin(np.abs(n - f)); return float((f[i] + n[i]) / 2)


def mets(y, s, thr):
    p = (s >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    m = dict(acc=accuracy_score(y, p), bal_acc=balanced_accuracy_score(y, p), prec=precision_score(y, p, zero_division=0),
             rec=recall_score(y, p, zero_division=0), f1=f1_score(y, p, zero_division=0),
             benign_recall=tn / max(tn + fp, 1), TN=tn, FP=fp, FN=fn, TP=tp, thr=thr)
    if 0 < y.sum() < len(y):
        m.update(auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s), eer=eer(y, s))
    return m


def best_thr(y, s):  # maximise balanced accuracy on validation (robust to the class imbalance)
    g = np.linspace(0.02, 0.98, 49)
    return float(g[int(np.argmax([balanced_accuracy_score(y, (s >= t).astype(int)) for t in g]))])


def fit_dnn(X, y, Xv, yv, seed):
    """Paper architecture 128-64-32 (torch with BN+Dropout if available), early stop on val balanced acc."""
    if HAVE_TORCH:
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        pw = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], dtype=torch.float32)
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)
        lf = nn.BCEWithLogitsLoss(pos_weight=pw); Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32).view(-1, 1); gen = torch.Generator().manual_seed(seed)
        pred = lambda Z: (net.eval(), torch.sigmoid(net(torch.tensor(Z, dtype=torch.float32))).detach().numpy().ravel())[1]
        best, bs, bad = None, -1, 0
        for ep in range(60):
            net.train()
            for idx in torch.randperm(len(Xt), generator=gen).split(512):
                if len(idx) < 2: continue
                opt.zero_grad(); lf(net(Xt[idx]), yt[idx]).backward(); opt.step()
            sc = balanced_accuracy_score(yv, (pred(Xv) >= 0.5).astype(int))
            if sc > bs: bs, best, bad = sc, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= 8: break
        net.load_state_dict(best)
        return type("M", (), {"predict_proba": lambda self, Z: np.c_[1 - pred(Z), pred(Z)]})()
    m = MLPClassifier((128, 64, 32), learning_rate_init=1e-3, batch_size=512, alpha=1e-5, random_state=seed)
    best, bs, bad, rng = None, -1, 0, np.random.RandomState(seed)
    w0 = np.where(y == 1)[0]; w1 = np.where(y == 0)[0]
    for ep in range(60):  # class-balanced epochs (oversample the minority class)
        k = max(len(w0), len(w1)); idx = np.r_[rng.choice(w0, k), rng.choice(w1, k)]; rng.shuffle(idx)
        for b in range(0, len(idx), 8192): m.partial_fit(X[idx[b:b + 8192]], y[idx[b:b + 8192]], classes=[0, 1])
        sc = balanced_accuracy_score(yv, (m.predict_proba(Xv)[:, 1] >= 0.5).astype(int))
        if sc > bs: bs, best, bad = sc, copy.deepcopy(m), 0
        else:
            bad += 1
            if bad >= 8: break
    return best


def main():
    (tr, va, te), sets = load()
    info = {k: dict(rows=len(x), attack=int(x.y.sum()), benign=int((x.y == 0).sum())) for k, x in
            (("train", tr), ("val", va), ("test", te))}
    if "Label" in te: info["test_attack_types"] = te.Label.value_counts().to_dict()
    json.dump(info, open(f"{A.out}/split_counts.json", "w"), indent=2, default=int); print(info)
    rows = []
    for fs, feats in sets.items():
        Xs = [x[feats].astype(float) for x in (tr, va, te)]
        med = Xs[0].median(); Xs = [x.fillna(med).values for x in Xs]
        sc = StandardScaler().fit(Xs[0]); Xtr, Xva, Xte = (np.clip(sc.transform(x), -50, 50) for x in Xs)
        ytr, yva, yte = tr.y.values, va.y.values, te.y.values
        maj = int(ytr.mean() >= 0.5)
        rows.append(dict(model="majority_class", featset=fs, seed=0, **mets(yte, np.full(len(yte), float(maj)), 0.5)))
        for name in ("DNN", "LogReg", "RandomForest", "GradBoost"):
            for s in range(A.seeds if name == "DNN" else min(A.seeds, 3)):
                if name == "DNN": m = fit_dnn(Xtr, ytr, Xva, yva, s)
                elif name == "LogReg": m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
                elif name == "RandomForest": m = RandomForestClassifier(200, n_jobs=-1, class_weight="balanced_subsample",
                                                                        random_state=s).fit(Xtr, ytr)
                else: m = HistGradientBoostingClassifier(class_weight="balanced", random_state=s).fit(Xtr, ytr)
                thr = best_thr(yva, m.predict_proba(Xva)[:, 1])
                rows.append(dict(model=name, featset=fs, seed=s, **mets(yte, m.predict_proba(Xte)[:, 1], thr)))
                print(fs, name, s, round(rows[-1]["bal_acc"], 4), flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{A.out}/raw_results.csv", index=False)
    c = ["acc", "bal_acc", "prec", "rec", "f1", "benign_recall", "auc", "pr_auc", "eer"]
    agg = df.groupby(["featset", "model"])[c].agg(["mean", "std"]).round(4)
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = df.groupby(["featset", "model"]).seed.nunique()
    agg.to_csv(f"{A.out}/summary.csv"); print(agg[[x for x in agg.columns if x.endswith("_mean")] + ["n_seeds"]].to_string())


if __name__ == "__main__":
    main()
