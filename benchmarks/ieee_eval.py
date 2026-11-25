#!/usr/bin/env python3
"""
ieee_eval.py — clean evaluation on IEEE DataPort "SDN-SlowRate-DDoS" (DOI 10.21227/amrt-8y98), run locally (WSL).

The dataset ships NO labels and NO system statistics. Each of the 23 scenario folders has one
headerless flowStatistics.csv (ONOS flow-table snapshots) with 13 columns:
  DeviceID, FlowID, IP_Src, IP_Dst, Mac_Src, Mac_Dst, Port_Src, Port_Dst, Protocol, Packets, Bytes, Life, Timestamp
(the DataPort page lists "Bytes, Packets", but the data have ~78 bytes/packet only in this order).

LABEL RULE (documented, checked here): the attack is slowhttptest (slow HTTP read) against an HTTP
server on port 8000 -> a row is ATTACK if Port_Dst == 8000 or Port_Src == 8000, else benign.
The script reports how many rows the rule marks as attack in the two normal-only scenarios (should be 0).

Features (identity-free, per flow snapshot): Protocol, Packets, Bytes, Life, bytes/packet, packets/s,
bytes/s, and deltas since the previous snapshot of the same flow on the same switch
(d_packets, d_bytes, d_time, d_pkt_rate, d_byte_rate). IPs, MACs, ports, FlowID, DeviceID are NOT features
(except in P1, which reproduces the original ieee_dnn.py style on purpose).

Protocols:
  P1_rows_with_identity    random row split, identifiers included        (what ieee_dnn.py did)
  P2_rows_no_identity      random row split, identity-free features
  P3_flow_grouped          split by FlowID (no flow in two splits), identity-free
  P4_cross_session         train+val = T4000 scenarios (01-10; val = last 20% of time in each),
                           test = T1200 scenarios (11-23, a different capture session)   <- report this one
  P5_folder_labels_rows    FOLDER-NAME labels (every row of an attack scenario = attack, Normal* = benign),
                           random row split. Scored against folder labels AND against the port-8000 labels.
  P6_folder_labels_cross   folder labels, T4000 -> T1200 (same split as P4)
  P7_folder_labels_ident   folder labels + identifiers, random row split (closest to the original 99.48% setup)
label_counts.csv also reports, per scenario, how many rows the two labelling methods disagree on.
Validation chooses early stopping and the threshold; test is used once.

  python3 ieee_eval.py --root /mnt/c/Users/Sarath/Downloads/sdn-slowrate-ddos --out ~/bench_ieee
Send back: <out>/summary.csv, <out>/label_counts.csv, <out>/label_rule_check.json, <out>/per_test_scenario.csv
"""
import argparse, os, glob, re, copy, json, warnings, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, roc_curve, confusion_matrix)
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--per-class-per-scenario", type=int, default=20000,
                help="rows of each class sampled per scenario (after features are computed on the full file)")
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--attack-port", type=int, default=8000)
ap.add_argument("--protocols", default="P1,P2,P3,P4,P5,P6")
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
try:
    import torch, torch.nn as nn; HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

COLS = ["DeviceID", "FlowID", "IP_Src", "IP_Dst", "Mac_Src", "Mac_Dst", "Port_Src", "Port_Dst", "Protocol",
        "Packets", "Bytes", "Life", "Timestamp"]
FEAT = ["Protocol", "Packets", "Bytes", "Life", "bpp", "pps", "bps", "d_packets", "d_bytes", "d_time",
        "d_pkt_rate", "d_byte_rate"]
IDENT = ["DeviceID", "IP_Src", "IP_Dst", "Mac_Src", "Mac_Dst", "Port_Src", "Port_Dst"]


def load_scenario(path, scen, rng):
    d = pd.read_csv(path, header=None, names=COLS, low_memory=False)
    d["y"] = ((d.Port_Dst == A.attack_port) | (d.Port_Src == A.attack_port)).astype(np.int8)
    d["y_folder"] = np.int8(0 if "normal" in scen.lower() else 1)
    d = d.sort_values(["DeviceID", "FlowID", "Timestamp"])
    life = d.Life.clip(lower=1)
    d["bpp"] = d.Bytes / d.Packets.clip(lower=1); d["pps"] = d.Packets / life; d["bps"] = d.Bytes / life
    g = d.groupby(["DeviceID", "FlowID"], sort=False)
    d["d_packets"] = g.Packets.diff().fillna(d.Packets); d["d_bytes"] = g.Bytes.diff().fillna(d.Bytes)
    d["d_time"] = (g.Timestamp.diff() / 1e9).fillna(0.0)
    dt = d.d_time.where(d.d_time > 0, np.nan)
    d["d_pkt_rate"] = (d.d_packets / dt).fillna(0.0); d["d_byte_rate"] = (d.d_bytes / dt).fillna(0.0)
    full = dict(scenario=scen, rows=len(d), attack_rows=int(d.y.sum()), benign_rows=int((d.y == 0).sum()),
                flows=int(d.FlowID.nunique()), attack_flows=int(d[d.y == 1].FlowID.nunique()),
                exact_duplicate_rows=int(d[COLS].duplicated().sum()),
                snapshots_per_flow_median=float(d.groupby("FlowID").size().median()),
                duration_s=float((d.Timestamp.max() - d.Timestamp.min()) / 1e9),
                folder_label=int(d.y_folder.iloc[0]),
                rows_where_folder_and_port_labels_disagree=int((d.y != d.y_folder).sum()),
                disagree_pct=round(100 * float((d.y != d.y_folder).mean()), 2))
    # time position inside the scenario (for the P4 validation split)
    d["tpos"] = (d.Timestamp - d.Timestamp.min()) / max(d.Timestamp.max() - d.Timestamp.min(), 1)
    parts = [grp.sample(min(len(grp), A.per_class_per_scenario), random_state=int(rng.integers(1e9)))
             for _, grp in d.groupby(["y", "y_folder"])]
    s = pd.concat(parts); s["scenario"] = scen
    return s, full


def eer(y, s):
    f, t, _ = roc_curve(y, s); n = 1 - t; i = np.nanargmin(np.abs(n - f)); return float((f[i] + n[i]) / 2)


def mets(y, s, thr):
    p = (s >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    m = dict(acc=accuracy_score(y, p), bal_acc=balanced_accuracy_score(y, p), prec=precision_score(y, p, zero_division=0),
             rec=recall_score(y, p, zero_division=0), f1=f1_score(y, p, zero_division=0), fpr=fp / max(fp + tn, 1),
             TN=tn, FP=fp, FN=fn, TP=tp, thr=thr)
    if 0 < y.sum() < len(y):
        m.update(auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s), eer=eer(y, s))
    return m


def thr_on_val(y, s):
    g = np.linspace(0.02, 0.98, 49); return float(g[int(np.argmax([f1_score(y, (s >= t).astype(int)) for t in g]))])


def fit_dnn(X, y, Xv, yv, seed):
    """128-64-32 like the paper; early stopping on validation F1."""
    if HAVE_TORCH:
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5); lf = nn.BCEWithLogitsLoss()
        Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        gen = torch.Generator().manual_seed(seed)

        def pr(Z):
            net.eval()
            with torch.no_grad(): return torch.sigmoid(net(torch.tensor(Z, dtype=torch.float32))).numpy().ravel()
        best, bs, bad = None, -1, 0
        for ep in range(50):
            net.train()
            for idx in torch.randperm(len(Xt), generator=gen).split(512):
                if len(idx) < 2: continue
                opt.zero_grad(); lf(net(Xt[idx]), yt[idx]).backward(); opt.step()
            sc = f1_score(yv, (pr(Xv) >= .5).astype(int))
            if sc > bs + 1e-6: bs, best, bad = sc, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= 6: break
        net.load_state_dict(best)
        return type("M", (), {"predict_proba": lambda self, Z: np.c_[1 - pr(Z), pr(Z)]})()
    m = MLPClassifier((128, 64, 32), learning_rate_init=1e-3, batch_size=512, alpha=1e-5, random_state=seed)
    best, bs, bad, rng = None, -1, 0, np.random.RandomState(seed)
    for ep in range(50):
        idx = rng.permutation(len(X))
        for b in range(0, len(idx), 8192): m.partial_fit(X[idx[b:b + 8192]], y[idx[b:b + 8192]], classes=[0, 1])
        sc = f1_score(yv, (m.predict_proba(Xv)[:, 1] >= .5).astype(int))
        if sc > bs + 1e-6: bs, best, bad = sc, copy.deepcopy(m), 0
        else:
            bad += 1
            if bad >= 6: break
    return best


def run(tr, va, te, cols, proto, seed, rows, per_scen, ycol="y"):
    if any(c in IDENT for c in cols):
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(tr[IDENT].astype(str))
        def X(d):
            z = d[cols].copy(); z[IDENT] = enc.transform(d[IDENT].astype(str)); return z.astype(float).values
    else:
        X = lambda d: d[cols].astype(float).values
    Xtr, Xva, Xte = X(tr), X(va), X(te)
    Xtr, Xva, Xte = (np.nan_to_num(np.sign(v) * np.log1p(np.abs(v))) for v in (Xtr, Xva, Xte))  # heavy-tailed counters
    sc = StandardScaler().fit(Xtr); Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    ytr, yva, yte = tr[ycol].values, va[ycol].values, te[ycol].values
    for name in ("DNN", "LogReg", "RandomForest", "GradBoost"):
        t0 = time.time()
        if name == "DNN": m = fit_dnn(Xtr, ytr, Xva, yva, seed)
        elif name == "LogReg": m = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        elif name == "RandomForest": m = RandomForestClassifier(100, max_samples=min(200000, len(Xtr)), n_jobs=-1,
                                                                random_state=seed).fit(Xtr, ytr)
        else: m = HistGradientBoostingClassifier(random_state=seed).fit(Xtr, ytr)
        thr = thr_on_val(yva, m.predict_proba(Xva)[:, 1]); s = m.predict_proba(Xte)[:, 1]
        rows.append(dict(protocol=proto, model=name, seed=seed, train_rows=len(tr), test_rows=len(te),
                         fit_s=round(time.time() - t0, 1), **mets(yte, s, thr)))
        if ycol != "y":   # same predictions scored against the port-8000 (traffic-level) labels
            rows.append(dict(protocol=proto + "__vs_port_labels", model=name, seed=seed,
                             train_rows=len(tr), test_rows=len(te), **mets(te.y.values, s, thr)))
        if proto.startswith(("P4", "P6")):
            for scn, idx in te.groupby("scenario").indices.items():
                per_scen.append(dict(protocol=proto, model=name, seed=seed, scenario=scn, **mets(yte[idx], s[idx], thr)))
        print(f"{proto:26s} {name:12s} seed={seed} F1={rows[-1]['f1']:.4f} bal_acc={rows[-1]['bal_acc']:.4f}", flush=True)


def main():
    files = sorted(glob.glob(os.path.join(A.root, "**", "flowStatistics.csv"), recursive=True))
    print(f"{len(files)} flowStatistics.csv files"); rng = np.random.default_rng(0)
    data, facts = [], []
    for f in files:
        scen = os.path.basename(os.path.dirname(f))
        s, full = load_scenario(f, scen, rng); data.append(s); facts.append(full)
        print(f"  {scen:42s} rows={full['rows']:>8d} attack={full['attack_rows']:>8d} flows={full['flows']}", flush=True)
    pd.DataFrame(facts).to_csv(f"{A.out}/label_counts.csv", index=False)
    D = pd.concat(data, ignore_index=True)
    D["session"] = D.scenario.str.extract(r"_T(\d+)$")[0]
    normal = [x for x in facts if "Normal" in x["scenario"]]
    json.dump(dict(rule=f"attack iff Port_Src or Port_Dst == {A.attack_port}",
                   attack_rows_in_normal_only_scenarios={x["scenario"]: x["attack_rows"] for x in normal},
                   has_torch=HAVE_TORCH), open(f"{A.out}/label_rule_check.json", "w"), indent=2)
    rows, per_scen = [], []
    P = A.protocols.split(",")
    for seed in range(A.seeds):
        if "P1" in P or "P2" in P:
            tr, tmp = train_test_split(D, test_size=0.4, stratify=D.y, random_state=seed)
            va, te = train_test_split(tmp, test_size=0.5, stratify=tmp.y, random_state=seed)
            if "P1" in P: run(tr, va, te, FEAT + IDENT, "P1_rows_with_identity", seed, rows, per_scen)
            if "P2" in P: run(tr, va, te, FEAT, "P2_rows_no_identity", seed, rows, per_scen)
        if "P3" in P:
            g = D.scenario + "|" + D.FlowID.astype(str)
            i_tr, i_tmp = next(GroupShuffleSplit(1, test_size=0.4, random_state=seed).split(D, groups=g))
            tmp = D.iloc[i_tmp]
            i_va, i_te = next(GroupShuffleSplit(1, test_size=0.5, random_state=seed).split(tmp, groups=g.iloc[i_tmp]))
            run(D.iloc[i_tr], tmp.iloc[i_va], tmp.iloc[i_te], FEAT, "P3_flow_grouped", seed, rows, per_scen)
        if "P4" in P:
            a = D[D.session == "4000"]; te = D[D.session == "1200"]
            run(a[a.tpos <= 0.8], a[a.tpos > 0.8], te, FEAT, "P4_cross_session", seed, rows, per_scen)
        if "P5" in P:
            tr, tmp = train_test_split(D, test_size=0.4, stratify=D.y_folder, random_state=seed)
            va, te = train_test_split(tmp, test_size=0.5, stratify=tmp.y_folder, random_state=seed)
            run(tr, va, te, FEAT, "P5_folder_labels_rows", seed, rows, per_scen, ycol="y_folder")
        if "P6" in P:
            a = D[D.session == "4000"]; te = D[D.session == "1200"]
            run(a[a.tpos <= 0.8], a[a.tpos > 0.8], te, FEAT, "P6_folder_labels_cross", seed, rows, per_scen, ycol="y_folder")
        if "P7" in P:
            tr, tmp = train_test_split(D, test_size=0.4, stratify=D.y_folder, random_state=seed)
            va, te = train_test_split(tmp, test_size=0.5, stratify=tmp.y_folder, random_state=seed)
            run(tr, va, te, FEAT + IDENT, "P7_folder_labels_ident", seed, rows, per_scen, ycol="y_folder")
    df = pd.DataFrame(rows); df.to_csv(f"{A.out}/raw_results.csv", index=False)
    c = ["acc", "bal_acc", "prec", "rec", "f1", "fpr", "auc", "pr_auc", "eer"]
    agg = df.groupby(["protocol", "model"])[c].agg(["mean", "std"]).round(4)
    agg.columns = [f"{x}_{y}" for x, y in agg.columns]; agg["n_seeds"] = df.groupby(["protocol", "model"]).seed.nunique()
    agg.to_csv(f"{A.out}/summary.csv")
    if per_scen:
        ps = pd.DataFrame(per_scen).groupby(["protocol", "scenario", "model"])[["f1", "rec", "fpr", "bal_acc"]].mean().round(4)
        ps.to_csv(f"{A.out}/per_test_scenario.csv")
    print(agg[["acc_mean", "bal_acc_mean", "f1_mean", "f1_std", "fpr_mean", "auc_mean", "n_seeds"]].to_string())
    print(f"\nSend back: {A.out}/summary.csv, label_counts.csv, label_rule_check.json, per_test_scenario.csv")


if __name__ == "__main__":
    main()
