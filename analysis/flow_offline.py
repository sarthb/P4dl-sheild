#!/usr/bin/env python3
"""
flow_offline.py -- every offline table of the paper, on PER-FLOW (50 ms window) data from the live runs in results_flow/.
No testbed runs. Uses ctrl_logs/flow_msgs.csv (window summaries = msg_type 1; new-flow messages = msg_type 0).

  A  ablation + baselines      retrain MLP per feature set (3 seeds) + RF / GB / LR + single-feature rules + tag alone
  B  tag as a detector         per summary, natural class mix
  C  threshold / policy        deployed model's LIVE scores: per-summary metrics vs threshold; flow-level replay of the
                               deployed policy vs threshold, K and B (times from the attacker's first packet)
  D  unseen variants           deployed policy per scenario: benign blocked, attackers blocked, onset -> block
  E  REPD                      Renyi-entropy baseline on the SAME messages (summaries weighted by packets), tuned on validation
  F  drift                     share of benign summaries flagged, per repetition
  S1 confusion matrices        TP / FP / FN / TN per test scenario at the deployed threshold

  cd ~/MTP1-main && python3 revision/analysis/flow_offline.py --out ~/flow_offline
"""
import argparse, glob, json, os, copy, time
from collections import Counter
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="results_flow")
ap.add_argument("--test", default="S1_single_A2_detect/rep*,S2_distributed_detect/rep*,S3_*/rep*,S4_*/rep*")
ap.add_argument("--repd-extra", default="S9_ctrl_flood_detect/rep*")
ap.add_argument("--drift", default="S0_benign_only/rep*,S1_single_A2/rep*,S2_distributed/rep*")
ap.add_argument("--policy", default="models/block_policy_flow.json")
ap.add_argument("--window-s", type=float, default=0.05)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--out", required=True)
ap.add_argument("--only", default="", help="e.g. A to run only the ablation")
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
FEATS = ["protocol", "tag", "pkt_len", "flow_pkts", "flow_bytes", "flow_dur", "flow_pps", "iat", "win_pkts"]
SEEDS = [int(s) for s in A.seeds.split(",")]
POL = json.load(open(A.policy)); THR = float(POL["threshold"])
LOG = open(f"{A.out}/summary.txt", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
t0 = time.time()


def dirs(globs):
    out = []
    for g in globs.split(","):
        out += [d for d in sorted(glob.glob(os.path.join(A.root, g.strip()))) if os.path.exists(f"{d}/ctrl_logs/flow_msgs.csv")]
    return sorted(set(out))


def onset_by_src(d):
    """attacker src IP (int) -> first packet time, from the sender logs (same mapping as analyze_run.py)."""
    try:
        man = json.load(open(f"{d}/manifest.json")); sc = json.load(open(f"{d}/scenario.json"))
    except Exception:
        return {}
    hosts = [t["host"] for t in sc["traffic"] if t.get("attack")]
    out = {}
    for f in glob.glob(f"{d}/send_attack_*.csv"):
        h = f.split("send_attack_")[1][:-4]
        try:
            ev = pd.read_csv(f); fp = ev[ev.event == "first_packet"].t
            src = man["attack_flows"][hosts.index(h)]["src"]
            a, b, c, e = (int(x) for x in src.split("."))
            out[(a << 24) | (b << 16) | (c << 8) | e] = float(fp.iloc[0])
        except Exception:
            pass
    return out


def load(globs, all_msgs=False):
    parts = []
    for d in dirs(globs):
        x = pd.read_csv(f"{d}/ctrl_logs/flow_msgs.csv", low_memory=False)
        x["run"] = d; x["scenario"] = os.path.basename(os.path.dirname(d)); x["rep"] = os.path.basename(d)
        on = onset_by_src(d); x["onset"] = x.src_ip.map(on)
        parts.append(x)
    if not parts: raise SystemExit(f"no flow_msgs.csv for {globs}")
    x = pd.concat(parts, ignore_index=True)
    if not all_msgs: x = x[x.msg_type == 1].copy()
    wp = x.win_pkts.clip(lower=1); dur = (x.win_last_us - x.flow_first_us).clip(lower=0) / 1e6
    x["protocol"] = x.proto; x["tag"] = x.tag_max; x["pkt_len"] = x.win_bytes / wp
    x["flow_pkts"] = x.cum_pkts; x["flow_bytes"] = x.cum_bytes; x["flow_dur"] = dur
    x["flow_pps"] = x.cum_pkts / np.maximum(dur, 1e-3)
    x["iat"] = np.where(x.win_pkts > 1, (x.win_last_us - x.win_first_us) / 1e6 / (x.win_pkts - 1).clip(lower=1), A.window_s)
    x["flow_key"] = x.src_ip.astype(str) + "|" + x.dst_ip.astype(str) + "|" + x.proto.astype(str) + "|" + x.dport.astype(str)
    return x


def eer(y, s):
    f, tp, _ = roc_curve(y, s); n = 1 - tp; i = np.nanargmin(np.abs(n - f)); return float((f[i] + n[i]) / 2)


def mets(y, s, thr):
    p = (np.asarray(s) >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    m = dict(prec=precision_score(y, p, zero_division=0), rec=recall_score(y, p, zero_division=0),
             f1=f1_score(y, p, zero_division=0), fpr=fp / max(fp + tn, 1), TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))
    if 0 < np.sum(y) < len(y): m.update(auc=roc_auc_score(y, s), eer=eer(y, s))
    return m


tr = load(f"T*/rep1,T*/rep2,T*/rep3"); va = load("T*/rep4"); te = load(A.test)
say(f"train {len(tr):,} summaries, val {len(va):,}, test {len(te):,} in {te.run.nunique()} runs "
    f"({te.label.mean():.2%} attack)   deployed threshold {THR}, policy K={POL['K']} B={POL['min_bins']}")

# =============================================================== A. ablation + baselines
try:
    import torch, torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    from sklearn.neural_network import MLPClassifier
    say("  (PyTorch not found: using sklearn MLP as a stand-in -- for testing only)")
    class nn:                       # minimal shim so the class definition below still parses
        Module = object
class Net(nn.Module if HAVE_TORCH else object):
    def __init__(s, d):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                              nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                              nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(s, x): return s.net(x)

def mlp_scores(cols, seed):
    sc = StandardScaler().fit(tr[cols].astype(float).values)
    Xtr, Xva, Xte = (sc.transform(d[cols].astype(float).values) for d in (tr, va, te))
    if not HAVE_TORCH:
        m = MLPClassifier((128, 64, 32), max_iter=50, early_stopping=True, random_state=seed).fit(Xtr, tr.label.values)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
    torch.manual_seed(seed); m = Net(len(cols)); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss(); Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(tr.label.values, dtype=torch.float32).view(-1, 1); g = torch.Generator().manual_seed(seed)
    def pr(X):
        m.eval()
        with torch.no_grad(): return torch.sigmoid(m(torch.tensor(X, dtype=torch.float32))).numpy().ravel()
    best, bf, bad = None, -1, 0
    for ep in range(100):
        m.train()
        for idx in torch.randperm(len(Xt), generator=g).split(256):
            if len(idx) < 2: continue
            opt.zero_grad(); lossf(m(Xt[idx]), yt[idx]).backward(); opt.step()
        f = f1_score(va.label.values, pr(Xva) >= 0.5, zero_division=0)
        if f > bf + 1e-6: bf, best, bad = f, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= 10: break
    m.load_state_dict(best); return pr(Xva), pr(Xte)

def sk_scores(model, cols):
    sc = StandardScaler().fit(tr[cols].astype(float).values)
    model.fit(sc.transform(tr[cols].astype(float).values), tr.label.values)
    return (model.predict_proba(sc.transform(va[cols].astype(float).values))[:, 1],
            model.predict_proba(sc.transform(te[cols].astype(float).values))[:, 1])

def val_thr(yv, sv):
    cands = np.unique(np.quantile(sv, np.linspace(0.5, 0.999, 60)))
    return max(cands, key=lambda c: (round(f1_score(yv, sv >= c, zero_division=0), 4), c))

DEPLOY_CANDS = [0.5, 0.7, 0.9, 0.95, 0.99]       # same candidates as training/train_flow.py used for the deployed model

def evaluate(name, sv, st, seed=0, prob=True):
    yv = va.label.values
    thr = (max(DEPLOY_CANDS, key=lambda c: (round(f1_score(yv, sv >= c, zero_division=0), 4), c)) if prob
           else val_thr(yv, sv))
    rows = []
    for scn, g in te.assign(s=st).groupby("scenario"):
        rows.append(dict(model=name, seed=seed, scenario=scn, thr=thr, **mets(g.label.values, g.s.values, thr)))
    return rows

ab = []
sets = {"MLP, 9 features (deployed)": FEATS, "MLP without tag": [f for f in FEATS if f != "tag"],
        "MLP without packet length": [f for f in FEATS if f != "pkt_len"],
        "MLP without protocol": [f for f in FEATS if f != "protocol"],
        "MLP without window packet count": [f for f in FEATS if f != "win_pkts"]}
for name, cols in sets.items():
    for sd in SEEDS:
        sv, st = mlp_scores(cols, sd); ab += evaluate(name, sv, st, sd)
    say(f"  A: {name} done ({time.time()-t0:.0f}s)")
for name, mk in (("Random forest", lambda sd: RandomForestClassifier(200, n_jobs=-1, random_state=sd, class_weight="balanced")),
                 ("Gradient boosting", lambda sd: GradientBoostingClassifier(random_state=sd)),
                 ("Logistic regression", lambda sd: LogisticRegression(max_iter=2000, class_weight="balanced"))):
    for sd in SEEDS:
        sv, st = sk_scores(mk(sd), FEATS); ab += evaluate(name, sv, st, sd)
    say(f"  A: {name} done ({time.time()-t0:.0f}s)")
for name, col in (("Rule: packet length", "pkt_len"), ("Rule: window packet count", "win_pkts"),
                  ("Rule: flow packet rate", "flow_pps"), ("Tag alone", "tag")):
    sv, st = va[col].astype(float).values, te[col].astype(float).values
    if roc_auc_score(va.label, sv) < 0.5: sv, st = -sv, -st
    ab += evaluate(name, sv, st, prob=False)
ab = pd.DataFrame(ab); ab.to_csv(f"{A.out}/A_ablation_per_scenario.csv", index=False)
per_seed = ab.groupby(["model", "seed"])[["f1", "auc", "eer", "fpr"]].mean()
tab = per_seed.groupby("model").mean().round(4); tab["f1_std_seeds"] = per_seed.groupby("model").f1.std().round(4)
tab["thr"] = ab.groupby("model").thr.median().round(3)
order = list(sets) + ["Random forest", "Gradient boosting", "Logistic regression", "Rule: packet length",
                      "Rule: window packet count", "Rule: flow packet rate", "Tag alone"]
say("\n=== A. ABLATION AND BASELINES (per summary; mean over test scenarios, then over seeds)\n" + tab.loc[order].to_string())
weak = ab[ab.scenario.isin(["S3_short_burst_50ms", "S3_tiny_pkts_28B", "S4_mimicry"])].groupby(["model", "scenario"]).f1.mean().unstack().round(3)
say("\nF1 on the hardest variants\n" + weak.loc[[m for m in order if m in weak.index]].to_string())

if A.only == "A":
    say(f"\ndone in {time.time()-t0:.0f}s (ablation only)"); raise SystemExit
# =============================================================== B. tag as a detector
tg = [dict(scenario=scn, **mets(g.label.values, (g.tag > 0).astype(int).values, 1)) for scn, g in te.groupby("scenario")]
tg = pd.DataFrame(tg).set_index("scenario")[["prec", "rec", "f1", "fpr"]]
pool = mets(te.label.values, (te.tag > 0).astype(int).values, 1)
tg.loc["MEAN over scenarios"] = tg.mean(); tg.loc["POOLED"] = [pool[k] for k in ("prec", "rec", "f1", "fpr")]
say("\n=== B. TAG AS A DETECTOR (per summary)\n" + tg.round(4).to_string())

# =============================================================== flow-level replay helper
def first_block(pos, K, W, B, R, G, L):
    lo = 0; eps = []; last = None
    for i, now in enumerate(pos):
        if last is None or now - last > G: eps.append(now)
        last = now
        while now - pos[lo] > W: lo += 1
        while eps and now - eps[0] > R and len(eps) > 1: eps.pop(0)
        if i - lo + 1 >= K and (sum(1 for e in eps if now - e <= R) >= B or now - eps[-1] >= L):
            return float(now)
    return None

def replay(d, thr, K, B):
    rows = []
    for (run, k), g in d.groupby(["run", "flow_key"], sort=False):
        g = g.sort_values("t_rx"); pos = g.t_rx.values[g.prob.values >= thr]
        b = first_block(pos, K, POL["window_s"], B, POL["recur_window_s"], POL["gap_s"], POL["sustain_s"])
        lab = int(g.label.max()); on = g.onset.iloc[0] if lab else np.nan
        rows.append(dict(run=run, scenario=g.scenario.iloc[0], label=lab, blocked=b is not None,
                         delay=(b - on) if (b is not None and lab and pd.notna(on)) else np.nan))
    return pd.DataFrame(rows)

def summ(fl):
    b, a = fl[fl.label == 0], fl[fl.label == 1]
    return dict(benign_flows=len(b), benign_blocked=int(b.blocked.sum()), attack_flows=len(a),
                attack_blocked=int(a.blocked.sum()), onset_to_block_mean=round(a.delay.mean(), 3),
                onset_to_block_max=round(a.delay.max(), 3))

# =============================================================== C. threshold and policy sensitivity (live scores)
rows = []
for t in (0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999):
    m = pd.DataFrame([mets(g.label.values, g.prob.values, t) for _, g in te.groupby("scenario")])[["prec", "rec", "f1", "fpr"]].mean()
    rows.append(dict(thr=t, **m.round(4).to_dict(), **summ(replay(te, t, POL["K"], POL["min_bins"]))))
say("\n=== C1. THRESHOLD SENSITIVITY (live scores of the deployed model; deployed policy)\n" + pd.DataFrame(rows).to_string(index=False))
rows = [dict(K=K, B=B, **summ(replay(te, THR, K, B))) for K in (1, 2, 3, 5, 8) for B in (1, 2)]
say("\n=== C2. POLICY SENSITIVITY at the deployed threshold\n" + pd.DataFrame(rows).to_string(index=False))

# =============================================================== D. unseen variants, deployed policy
fl = replay(te, THR, POL["K"], POL["min_bins"])
v = []
for scn, g in fl.groupby("scenario"):
    a = g[g.label == 1]
    v.append(dict(scenario=scn, runs=g.run.nunique(), benign_blocked=int(g[g.label == 0].blocked.sum()),
                  benign_flows=int((g.label == 0).sum()), attackers=f"{int(a.blocked.sum())}/{len(a)}",
                  onset_to_block_s=round(a.delay.mean(), 3), max_s=round(a.delay.max(), 3)))
say("\n=== D. UNSEEN VARIANTS (deployed policy replayed on detection-only runs; onset = attacker's first packet)\n"
    + pd.DataFrame(v).to_string(index=False))

# =============================================================== S1. confusion matrices
cm = te.assign(pred=(te.prob >= THR).astype(int)).groupby("scenario").apply(
    lambda g: pd.Series(dict(runs=g.run.nunique(), TP=int(((g.pred == 1) & (g.label == 1)).sum()),
                             FP=int(((g.pred == 1) & (g.label == 0)).sum()), FN=int(((g.pred == 0) & (g.label == 1)).sum()),
                             TN=int(((g.pred == 0) & (g.label == 0)).sum()))))
cm.to_csv(f"{A.out}/S1_confusion_matrices.csv")
say(f"\n=== S1. CONFUSION MATRICES per test scenario (window summaries, threshold {THR}, all runs pooled)\n" + cm.to_string())

# =============================================================== E. REPD on the same messages
def renyi_w(vals, w, a):
    c = pd.Series(w).groupby(np.asarray(vals)).sum().values.astype(float); p = c / c.sum()
    return float(-np.sum(p * np.log2(p))) if a == 1 else float(np.log2(np.sum(p ** a)) / (1 - a))

def repd_windows(d, n, fld, a):
    d = d.sort_values("t_rx").reset_index(drop=True); idx = np.arange(len(d)) // n
    H = d.groupby(idx).apply(lambda g: renyi_w(g[fld].values, g.win_pkts.clip(lower=1).values, a))
    y = d.groupby(idx).label.mean().ge(0.5).astype(int)
    return d, idx, H.values, y.values

tra = load("T*/rep1,T*/rep2,T*/rep3", all_msgs=True); vaa = load("T*/rep4", all_msgs=True)
best = None
for a in (2.0, 3.0):
    for n in (20, 50, 100):
        for fld in ("dst_ip", "src_ip"):
            ref = np.concatenate([repd_windows(g[g.label == 0], n, fld, a)[2] for _, g in tra.groupby("run")])
            href = float(np.mean(ref))
            ys, ss = [], []
            for _, g in vaa.groupby("run"):
                _, _, H, y = repd_windows(g, n, fld, a); ys.append(y); ss.append(np.abs(H - href))
            ys, ss = np.concatenate(ys), np.concatenate(ss)
            for thr in np.unique(np.quantile(ss, np.linspace(0.01, 0.99, 99))):
                f = f1_score(ys, ss >= thr, zero_division=0)
                if best is None or f > best["val_f1"]:
                    best = dict(alpha=a, window=n, field=fld, H_ref=href, thr=float(thr), val_f1=float(f))
say(f"\nREPD chosen on validation: {best}")
rp = []
tea = load(A.test + "," + A.repd_extra, all_msgs=True)
for run, g in tea.groupby("run"):
    d, idx, H, yw = repd_windows(g, best["window"], best["field"], best["alpha"]); s = np.abs(H - best["H_ref"])
    scn = g.scenario.iloc[0]; summ_mask = (d.msg_type == 1).values
    rp.append(dict(scenario=scn, method="REPD (per message)", **mets(d.label.values[summ_mask], s[idx][summ_mask], best["thr"])))
    rp.append(dict(scenario=scn, method="REPD (per window)", **mets(yw, s, best["thr"])))
    ds = d[summ_mask]
    rp.append(dict(scenario=scn, method="MLP (deployed, live)", **mets(ds.label.values, ds.prob.values, THR)))
rp = pd.DataFrame(rp); rp.to_csv(f"{A.out}/E_repd_per_run.csv", index=False)
say("\n=== E. REPD vs DEPLOYED MLP on the same window summaries (mean over runs)\n"
    + rp.groupby(["scenario", "method"])[["f1", "eer", "auc"]].mean().unstack().round(3).to_string())

# =============================================================== F. drift over repetitions
dr = load(A.drift)
b = dr[dr.label == 0]
f = b.groupby(["rep"]).apply(lambda g: pd.Series(dict(benign_summaries=len(g), flagged_pct=100 * (g.prob >= THR).mean())))
say("\n=== F. DRIFT: benign summaries flagged per repetition (S0, S1 A2, S2)\n" + f.round(4).to_string())
say(f"\ndone in {time.time()-t0:.0f}s. Send back {A.out}/summary.txt and S1_confusion_matrices.csv")
