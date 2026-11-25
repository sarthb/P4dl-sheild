#!/usr/bin/env python3
"""
perflow_sim.py -- offline estimate of PER-FLOW (windowed) digests, from EXISTING per-packet logs. No testbed runs.

What it simulates (what a P4 switch could do with per-flow registers):
  * per switch and flow (src, dst, proto, dport) the switch keeps counters for the current window of W ms
    (windows on a global grid: window id = ingress_ts // W, as a P4 program would do with a shift);
  * when the first packet of a LATER window arrives, the switch sends ONE summary digest for the finished window;
  * optional (--new-flow-msg): also one digest for the first packet of every new flow (needed for the churn guard).

Each summary carries the same 8 features as the deployed model, computed for the window instead of the packet:
  protocol, tag (max in window), packet length (mean in window), flow packet count, flow byte count, flow duration,
  flow packets/s (cumulative), inter-arrival time (mean in window)   + window packet count (9th feature).

It then trains the same MLP (128-64-32, BN, dropout 0.3) on summaries from the training runs, picks threshold and
blocking policy on the validation runs, and compares on the test runs against the deployed per-packet model
(its live-logged probabilities), with times in SWITCH time from each flow's first packet:
  messages to the controller, peak messages/s, summary-level F1, flows blocked (benign / attack), time to block.

Run in WSL from ~/MTP1-main (needs results_rev with ctrl_logs/timing.csv + live_dataset.csv per run):
  python3 ~/revision/training/perflow_sim.py --out ~/perflow_sim
Defaults: train T*/rep1-3, validation T*/rep4, test = the 31 detection-only runs,
          rate runs = S5 (load) and S9 (control-plane) runs for message-rate comparison.
"""
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="results_rev")
ap.add_argument("--train", default="T*/rep1,T*/rep2,T*/rep3")
ap.add_argument("--val", default="T*/rep4")
ap.add_argument("--test", default="S1_single_A2_detect/rep*,S2_distributed_detect/rep*,S3_*/rep*,S4_*/rep*")
ap.add_argument("--rate", default="S5_*/rep*,S9_*/rep*,S1_single_A2/rep*,S0_*/rep*")
ap.add_argument("--policy", default="models/block_policy.json", help="deployed per-packet policy (baseline)")
ap.add_argument("--window-ms", type=float, default=100.0)
ap.add_argument("--new-flow-msg", action="store_true", help="also send one digest per new flow")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--out", required=True)
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
W = A.window_ms / 1000.0
FEATS = ["protocol", "tag", "pkt_len", "flow_pkts", "flow_bytes", "flow_dur", "flow_pps", "iat", "win_pkts"]
LOG = open(f"{A.out}/summary.txt", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n"); LOG.flush()

def run_dirs(globs):
    out = []
    for g in globs.split(","):
        out += [d for d in sorted(glob.glob(os.path.join(A.root, g.strip())))
                if os.path.exists(f"{d}/ctrl_logs/timing.csv")]
    return sorted(set(out))

# ------------------------------------------------------------------ load one run (timing + live, aligned by row)
def load_run(d):
    tcols = ["device_id", "src_ip", "dst_ip", "proto", "dport", "true_label", "tag", "prob", "sw_ingress_ts_us"]
    t = pd.read_csv(f"{d}/ctrl_logs/timing.csv", usecols=tcols, low_memory=False)
    l = pd.read_csv(f"{d}/ctrl_logs/live_dataset.csv", usecols=["src_ip", "dst_ip", "pkt_len"], low_memory=False)
    n = min(len(t), len(l))
    t, l = t.iloc[:n].reset_index(drop=True), l.iloc[:n].reset_index(drop=True)
    ok = (t.src_ip.astype(str).values == l.src_ip.astype(str).values).mean() if n else 0
    if n == 0 or ok < 0.999:
        say(f"  SKIP {d}: timing/live rows do not align (match {ok:.3f})"); return None
    t["pkt_len"] = pd.to_numeric(l.pkt_len, errors="coerce")
    t = t.dropna(subset=["sw_ingress_ts_us", "pkt_len", "prob"])
    t["ts"] = t.sw_ingress_ts_us.astype(float) / 1e6
    t["run"] = d; t["scenario"] = os.path.basename(os.path.dirname(d.rstrip("/")))
    t["label"] = t.true_label.astype(int)
    return t[["run", "scenario", "device_id", "src_ip", "dst_ip", "proto", "dport", "label", "tag", "prob", "pkt_len", "ts"]]

# ------------------------------------------------------------------ per-packet stream -> per-flow window summaries
KEY = ["device_id", "src_ip", "dst_ip", "proto", "dport"]
def summarize(t):
    t = t.sort_values(KEY + ["ts"]).copy()
    t["win"] = np.floor(t.ts / W).astype(np.int64)
    g = t.groupby(KEY + ["win"], sort=False)
    w = g.agg(n=("ts", "size"), bytes=("pkt_len", "sum"), first=("ts", "min"), last=("ts", "max"),
              tag=("tag", "max"), label=("label", "max"), proto_v=("proto", "first")).reset_index()
    w = w.sort_values(KEY + ["win"])
    fk = w.groupby(KEY, sort=False)
    w["flow_first"] = fk["first"].transform("min")
    w["cum_pkts"] = fk["n"].cumsum(); w["cum_bytes"] = fk["bytes"].cumsum()
    w["emit_ts"] = fk["first"].shift(-1)          # sent when the first packet of the next window arrives
    w["flow_dur"] = w["last"] - w["flow_first"]
    out = pd.DataFrame({
        "protocol": w.proto_v.astype(float), "tag": w.tag.astype(float),
        "pkt_len": w.bytes / w.n, "flow_pkts": w.cum_pkts.astype(float), "flow_bytes": w.cum_bytes.astype(float),
        "flow_dur": w.flow_dur, "flow_pps": w.cum_pkts / np.maximum(w.flow_dur, 1e-3),
        "iat": np.where(w.n > 1, (w["last"] - w["first"]) / np.maximum(w.n - 1, 1), W), "win_pkts": w.n.astype(float),
        "label": w.label.astype(int), "emit_ts": w.emit_ts, "flow_first": w.flow_first,
        "flow": w[KEY].astype(str).agg("|".join, axis=1)})
    n_new = t.groupby(KEY, sort=False).ngroups
    return out.dropna(subset=["emit_ts"]).reset_index(drop=True), n_new

# ------------------------------------------------------------------ model (same architecture as deployed)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
try:
    import torch, torch.nn as nn
    HAVE_TORCH = True
except ImportError:                      # fallback for testing without PyTorch (no BN/dropout)
    HAVE_TORCH = False
    from sklearn.neural_network import MLPClassifier

if HAVE_TORCH:
    class MLP(nn.Module):
        def __init__(s, k):
            super().__init__()
            s.net = nn.Sequential(nn.Linear(k, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        def forward(s, x): return s.net(x).squeeze(-1)

def train(Xtr, ytr, Xva, yva, seed):
    if not HAVE_TORCH:
        m = MLPClassifier((128, 64, 32), alpha=1e-5, batch_size=256, learning_rate_init=1e-3, max_iter=100,
                          early_stopping=True, n_iter_no_change=10, random_state=seed)
        return m.fit(Xtr, ytr)
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP(Xtr.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    pw = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    Xv = torch.tensor(Xva, dtype=torch.float32)
    best, best_state, bad = -1, None, 0
    for ep in range(100):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            if len(b) < 2: continue
            opt.zero_grad(); lossf(m(Xt[b]), yt[b]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): pv = torch.sigmoid(m(Xv)).numpy()
        f = f1_score(yva, pv >= 0.5)
        if f > best + 1e-4: best, best_state, bad = f, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 10: break
    m.load_state_dict(best_state); m.eval(); return m

def predict(m, X):
    if not HAVE_TORCH: return m.predict_proba(X)[:, 1]
    with torch.no_grad(): return torch.sigmoid(m(torch.tensor(X, dtype=torch.float32))).numpy()

# ------------------------------------------------------------------ blocking policy replay (same rule as controller)
def first_block(pos, K, Wp, B, R, G=0.3, L=0.5):
    lo = 0; eps = []; last = None
    for i, now in enumerate(pos):
        if last is None or now - last > G: eps.append(now)
        last = now
        while now - pos[lo] > Wp: lo += 1
        while eps and now - eps[0] > R and len(eps) > 1: eps.pop(0)
        if i - lo + 1 >= K and (sum(1 for e in eps if now - e <= R) >= B or now - eps[-1] >= L):
            return float(now)
    return None

def flow_level(df, tcol, pcol, thr, K, B, Wp=1.0, R=3.0, fcol="flow", first_col="flow_first", agg_switches=True):
    """Replay per flow (src,dst,proto,dport) aggregated over switches, like the controller's policy key."""
    d = df[[fcol, tcol, pcol, "label", first_col, "run", "scenario"]].copy()
    if agg_switches: d["pkey"] = d[fcol].str.split("|", n=1).str[1]     # drop device id
    else: d["pkey"] = d[fcol]
    rows = []
    for (run, k), g in d.groupby(["run", "pkey"], sort=False):
        g = g.sort_values(tcol); ts = g[tcol].values; pos = ts[g[pcol].values >= thr]
        b = first_block(pos, K, Wp, B, R)
        rows.append(dict(run=run, scenario=g.scenario.iloc[0], label=int(g.label.max()),
                         blocked=b is not None, delay=(b - g[first_col].min()) if b is not None else np.nan))
    return pd.DataFrame(rows)

def flow_summary(fl):
    ben, att = fl[fl.label == 0], fl[fl.label == 1]
    return dict(benign_flows=len(ben), benign_blocked=int(ben.blocked.sum()), attack_flows=len(att),
                attack_blocked=int(att.blocked.sum()), block_s_mean=round(att.delay.mean(), 3),
                block_s_max=round(att.delay.max(), 3))

def peak_rate(ts):
    if len(ts) == 0: return 0
    b = np.floor(np.asarray(ts)); _, c = np.unique(b, return_counts=True); return int(c.max())

# ================================================================== main
t0 = time.time()
say(f"window = {A.window_ms:.0f} ms, new-flow messages = {A.new_flow_msg}, model = {'PyTorch MLP' if HAVE_TORCH else 'sklearn MLP (fallback)'}")
sets = {}
for name, globs in (("train", A.train), ("val", A.val), ("test", A.test)):
    dirs = run_dirs(globs); S = []; P = []; msg = []
    for d in dirs:
        t = load_run(d)
        if t is None: continue
        s, n_new = summarize(t); s["run"] = d; s["scenario"] = t.scenario.iloc[0]
        S.append(s)
        if name == "test":
            t["flow"] = t[KEY].astype(str).agg("|".join, axis=1)
            t["flow_first"] = t.groupby("flow").ts.transform("min"); P.append(t)
        msg.append(dict(run=d, scenario=t.scenario.iloc[0], per_packet=len(t),
                        per_flow=len(s) + (n_new if A.new_flow_msg else 0)))
    sets[name] = (pd.concat(S, ignore_index=True) if S else None, pd.concat(P, ignore_index=True) if P else None,
                  pd.DataFrame(msg))
    say(f"{name}: {len(dirs)} runs, {sum(m['per_packet'] for m in msg):,} per-packet digests -> "
        f"{sum(m['per_flow'] for m in msg):,} per-flow summaries  ({time.time()-t0:.0f}s)")
tr, va, te = sets["train"][0], sets["val"][0], sets["test"][0]
assert tr is not None and va is not None and te is not None, "missing runs -- check --root and globs"

# ---- message reduction, incl. load and control-plane runs
rate_rows = []
for d in run_dirs(A.rate):
    t = load_run(d)
    if t is None: continue
    s, n_new = summarize(t)
    newf = t.sort_values("ts").drop_duplicates(KEY).ts.values if A.new_flow_msg else np.array([])
    rate_rows.append(dict(scenario=t.scenario.iloc[0], per_packet=len(t), per_flow=len(s) + len(newf),
                          peak_pp_per_s=peak_rate(t.ts.values),
                          peak_pf_per_s=peak_rate(np.concatenate([s.emit_ts.values, newf]))))
msgs = pd.concat([sets[k][2] for k in sets], ignore_index=True)
if rate_rows:
    rr = pd.DataFrame(rate_rows).groupby("scenario").mean(numeric_only=True)
    rr["reduction_x"] = rr.per_packet / rr.per_flow
    say("\n=== MESSAGES TO THE CONTROLLER per run (mean), load and control-plane runs\n" + rr.round(0).to_string())
mt = sets["test"][2].groupby("scenario")[["per_packet", "per_flow"]].mean()
mt["reduction_x"] = mt.per_packet / mt.per_flow
say("\n=== MESSAGES per run (mean), test runs\n" + mt.round(1).to_string())

# ---- train per-flow model (3 seeds), threshold on validation
sc = StandardScaler().fit(tr[FEATS].values)
Xtr, Xva, Xte = (sc.transform(x[FEATS].values) for x in (tr, va, te))
res, probs = [], []
for seed in [int(s) for s in A.seeds.split(",")]:
    m = train(Xtr, tr.label.values, Xva, va.label.values, seed)
    pv = predict(m, Xva)
    cand = [0.5, 0.7, 0.9, 0.95, 0.99]
    thr = max(cand, key=lambda c: (round(f1_score(va.label.values, pv >= c), 4), c))
    pt = predict(m, Xte); probs.append(pt)
    for scn, g in te.assign(p=pt).groupby("scenario"):
        y, p = g.label.values, g.p.values
        res.append(dict(seed=seed, scenario=scn, thr=thr, f1=f1_score(y, p >= thr),
                        auc=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
                        fpr=((p >= thr) & (y == 0)).sum() / max((y == 0).sum(), 1)))
res = pd.DataFrame(res)
say("\n=== PER-FLOW MODEL, summary level (mean over seeds)\n" +
    res.groupby("scenario")[["f1", "auc", "fpr"]].mean().round(4).to_string() +
    f"\nMEAN over scenarios: F1 {res.groupby('seed').f1.mean().mean():.4f} +- {res.groupby('seed').f1.mean().std():.4f}"
    f"   thresholds chosen on validation: {sorted(set(res.thr))}")
te["p"] = np.mean(probs, axis=0); thr_pf = float(np.median(res.thr))

# ---- deployed per-packet model at its threshold, per digest, same test runs (baseline)
pp = sets["test"][1]
pol = json.load(open(A.policy)) if os.path.exists(A.policy) else dict(threshold=0.99, K=50, window_s=1.0, min_bins=2, recur_window_s=3.0)
base = []
for scn, g in pp.groupby("scenario"):
    base.append(dict(scenario=scn, f1=f1_score(g.label.values, g.prob.values >= pol["threshold"])))
say("\n=== DEPLOYED PER-PACKET MODEL, digest level (same test runs)\n" +
    pd.DataFrame(base).set_index("scenario").round(4).to_string())

# ---- choose per-flow blocking policy on validation (fewest benign blocked, all attacks, fastest)
va["p"] = np.mean([predict(train(Xtr, tr.label.values, Xva, va.label.values, 0), Xva)], axis=0)
grid = []
for K in (1, 2, 3, 5, 8):
    for B in (1, 2):
        fl = flow_level(va, "emit_ts", "p", thr_pf, K, B)
        s = flow_summary(fl); s.update(K=K, B=B); grid.append(s)
grid = pd.DataFrame(grid)
best = grid.sort_values(["benign_blocked", "attack_blocked", "block_s_mean"], ascending=[True, False, True]).iloc[0]
say("\n=== PER-FLOW POLICY GRID on validation\n" + grid.to_string(index=False) +
    f"\nchosen: K={int(best.K)}, B={int(best.B)} (W=1 s, R=3 s, gap 0.3 s, sustain 0.5 s)")

# ---- flow-level comparison on test runs (switch time from each flow's first packet)
fl_pf = flow_level(te, "emit_ts", "p", thr_pf, int(best.K), int(best.B))
fl_pp = flow_level(pp, "ts", "prob", pol["threshold"], pol["K"], pol["min_bins"], pol["window_s"], pol["recur_window_s"])
fl_pf2 = flow_level(te, "emit_ts", "p", thr_pf, 1, 2)          # same recurrence rule as deployed (2 episodes)
cmp = pd.DataFrame([dict(design="per-packet (deployed)", **flow_summary(fl_pp)),
                    dict(design=f"per-flow {A.window_ms:.0f} ms, policy chosen K={int(best.K)} B={int(best.B)}", **flow_summary(fl_pf)),
                    dict(design=f"per-flow {A.window_ms:.0f} ms, recurrence K=1 B=2", **flow_summary(fl_pf2))])
say("\n=== FLOW LEVEL on test runs (times in switch time; BMv2 digest delay not included)\n" + cmp.to_string(index=False))
per_scn = []
for name, fl in (("per-packet", fl_pp), ("per-flow", fl_pf), ("per-flow-B2", fl_pf2)):
    for scn, g in fl.groupby("scenario"):
        a = g[g.label == 1]
        per_scn.append(dict(design=name, scenario=scn, benign_blocked=int(g[g.label == 0].blocked.sum()),
                            attacks=f"{int(a.blocked.sum())}/{len(a)}", block_s=round(a.delay.mean(), 2)))
tg = []
for K in (1, 2, 3, 5, 8):
    for B in (1, 2):
        s_ = flow_summary(flow_level(te, "emit_ts", "p", thr_pf, K, B)); s_.update(K=K, B=B); tg.append(s_)
say("\n=== ALL PER-FLOW POLICIES on test runs (for the trade-off only; the choice above is made on validation)\n"
    + pd.DataFrame(tg).to_string(index=False))
say("\n" + pd.DataFrame(per_scn).pivot(index="scenario", columns="design", values=["benign_blocked", "attacks", "block_s"]).to_string())
say(f"\ndone in {time.time()-t0:.0f}s.  Send back: {A.out}/summary.txt")
