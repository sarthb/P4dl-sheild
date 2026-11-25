#!/usr/bin/env python3
"""
train_flow.py -- train and export the PER-FLOW model from live per-flow runs (controller/flow_msgs.csv).

  * training data: window summaries (msg_type 1) from results_flow/T*/rep1-3
  * validation:    results_flow/T*/rep4 (early stopping, threshold)
  * policy choice: validation + pilot runs (results_flow_pilot/*/rep1), replaying the controller's blocking rule
                   on the model's scores; order of preference: fewest benign flows blocked, all attacks blocked,
                   shortest mean time to block; ties -> larger K (more evidence)
  * exports: models/ddos_flow.pth (+ _scaler.pkl), models/block_policy_flow.json

  python3 revision/training/train_flow.py --root results_flow --pilot 'results_flow_pilot/*/rep1'
"""
import argparse, glob, json, os, copy
import numpy as np, pandas as pd
import torch, torch.nn as nn, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="results_flow")
ap.add_argument("--pilot", default="results_flow_pilot/*/rep1")
ap.add_argument("--export", default="models/ddos_flow.pth")
ap.add_argument("--policy-out", default="models/block_policy_flow.json")
ap.add_argument("--window-s", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
FEATS = ["protocol", "tag", "pkt_len", "flow_pkts", "flow_bytes", "flow_dur", "flow_pps", "iat", "win_pkts"]


def load(globs):
    out = []
    for g in globs.split(","):
        for d in sorted(glob.glob(g.strip())):
            f = f"{d}/ctrl_logs/flow_msgs.csv"
            if not os.path.exists(f) or os.path.getsize(f) == 0: continue
            x = pd.read_csv(f, low_memory=False)
            x["run"] = d
            out.append(x)
    if not out: raise SystemExit(f"no flow_msgs.csv under {globs}")
    x = pd.concat(out, ignore_index=True)
    w = x[x.msg_type == 1].copy()
    dur = (w.win_last_us - w.flow_first_us).clip(lower=0) / 1e6
    w["protocol"] = w.proto; w["tag"] = w.tag_max; w["pkt_len"] = w.win_bytes / w.win_pkts.clip(lower=1)
    w["flow_pkts"] = w.cum_pkts; w["flow_bytes"] = w.cum_bytes; w["flow_dur"] = dur
    w["flow_pps"] = w.cum_pkts / np.maximum(dur, 1e-3)
    w["iat"] = np.where(w.win_pkts > 1, (w.win_last_us - w.win_first_us) / 1e6 / (w.win_pkts - 1).clip(lower=1), A.window_s)
    w["flow_key"] = w.src_ip.astype(str) + "|" + w.dst_ip.astype(str) + "|" + w.proto.astype(str) + "|" + w.dport.astype(str)
    return w


class Net(nn.Module):           # identical to controller/dnn_model.py:DDoSDNN
    def __init__(s, d):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                              nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                              nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(s, x): return s.net(x)


def proba(m, X):
    m.eval()
    with torch.no_grad(): return torch.sigmoid(m(torch.tensor(X, dtype=torch.float32))).numpy().ravel()


def fit(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed); m = Net(Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5); lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    g = torch.Generator().manual_seed(seed); best, bf, bad = None, -1, 0
    for ep in range(100):
        m.train()
        for idx in torch.randperm(len(Xt), generator=g).split(256):
            if len(idx) < 2: continue
            opt.zero_grad(); lossf(m(Xt[idx]), yt[idx]).backward(); opt.step()
        f = f1_score(yva, proba(m, Xva) >= 0.5, zero_division=0)
        if f > bf + 1e-6: bf, best, bad = f, copy.deepcopy(m), 0
        else:
            bad += 1
            if bad >= 10: break
    return best


def first_block(pos, K, W, B, R, G=0.3, L=0.5):
    lo = 0; eps = []; last = None
    for i, now in enumerate(pos):
        if last is None or now - last > G: eps.append(now)
        last = now
        while now - pos[lo] > W: lo += 1
        while eps and now - eps[0] > R and len(eps) > 1: eps.pop(0)
        if i - lo + 1 >= K and (sum(1 for e in eps if now - e <= R) >= B or now - eps[-1] >= L):
            return float(now)
    return None


tr = load(f"{A.root}/T*/rep1,{A.root}/T*/rep2,{A.root}/T*/rep3")
va = load(f"{A.root}/T*/rep4")
print(f"train {len(tr):,} summaries ({tr.label.mean():.1%} attack), val {len(va):,}")
sc = StandardScaler().fit(tr[FEATS].astype(float).values)
Xtr, Xva = sc.transform(tr[FEATS].astype(float).values), sc.transform(va[FEATS].astype(float).values)
m = fit(Xtr, tr.label.values, Xva, va.label.values, A.seed)
pv = proba(m, Xva)
thr = max([0.5, 0.7, 0.9, 0.95, 0.99], key=lambda c: (round(f1_score(va.label.values, pv >= c), 4), c))
print(f"validation: F1 {f1_score(va.label.values, pv >= thr):.4f}  AUC {roc_auc_score(va.label.values, pv):.4f}  threshold {thr}")

# policy on validation + pilot (pilot runs are never reported)
pol_data = va.copy(); pol_data["p"] = pv
try:
    pi = load(A.pilot); pi["p"] = proba(m, sc.transform(pi[FEATS].astype(float).values))
    pol_data = pd.concat([pol_data, pi], ignore_index=True); print(f"pilot: {pi.run.nunique()} runs added for the policy choice")
except SystemExit:
    print("no pilot runs: policy chosen on validation only")
rows = []
for K in (1, 2, 3, 5, 8):
    for B in (1, 2):
        ben = att = bb = ab = 0; delays = []
        for (run, k), g in pol_data.groupby(["run", "flow_key"]):
            g = g.sort_values("t_rx"); pos = g.t_rx.values[g.p.values >= thr]
            b = first_block(pos, K, 1.0, B, 3.0)
            if g.label.max() == 1:
                att += 1; ab += b is not None
                if b is not None: delays.append(b - g.t_rx.min())
            else:
                ben += 1; bb += b is not None
        rows.append(dict(K=K, B=B, benign_flows=ben, benign_blocked=bb, attack_flows=att, attack_blocked=ab,
                         block_s_mean=np.mean(delays) if delays else np.nan))
grid = pd.DataFrame(rows)
print(grid.to_string(index=False))
best = grid.sort_values(["benign_blocked", "attack_blocked", "block_s_mean", "K"],
                        ascending=[True, False, True, False]).iloc[0]

os.makedirs(os.path.dirname(A.export) or ".", exist_ok=True)
torch.save({"model_state_dict": {k: v for k, v in m.state_dict().items()}, "input_dim": len(FEATS),
            "features": FEATS, "featset": "flow_window", "threshold": thr}, A.export)
joblib.dump(sc, A.export.replace(".pth", "_scaler.pkl"))
pol = dict(model=A.export, threshold=thr, K=int(best.K), window_s=1.0, min_bins=int(best.B), recur_window_s=3.0,
           gap_s=0.3, sustain_s=0.5, chosen_on="validation + pilot", grid=rows)
json.dump(pol, open(A.policy_out, "w"), indent=2, default=float)
print(f"exported {A.export} (+ scaler) and {A.policy_out}: K={pol['K']} B={pol['min_bins']} threshold={thr}")
