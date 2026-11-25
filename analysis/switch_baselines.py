#!/usr/bin/env python3
"""
switch_baselines.py -- in-switch and hybrid P4 detection baselines (Reviewer 1, Comment 1.4), on the same test scenarios as the
offline tables of the paper. No testbed runs.

Two published families of detectors that run entirely in the switch data plane are re-implemented in software
and applied to the per-packet digest logs (results_rev/, one record per packet with switch, 5-tuple, length, time):

  NB   NetBeacon-style multi-phase in-switch trees (Zhou et al., USENIX Security 2023), reproduced as in
       Brain-on-Switch (NSDI 2024, App. A.5): one random forest of 3 trees, depth 7, per phase.
       Phase points are the 2^k-th packet of a flow (k = 1..11); the first packet uses packet-level features only.
       A flow's decision is made at each phase point from the features accumulated so far and holds until the
       next phase point. Flow key = 5-tuple per switch (NB5, as in NetBeacon) and, for reference, the
       4-tuple without source port that P4-DL Shield uses (NB4).
  ENT  in-switch entropy detection (P4DDoS, Ding et al., IEEE TDSC 2021; Euclid, Ilha et al.): per switch and time
       window, normalised Shannon entropy of source and destination addresses, compared with an EWMA of past
       windows; alarm when either deviates by more than k mean absolute deviations. Alarms are per window, not per
       flow: for flow-level numbers every flow towards the window's most frequent destination is counted as
       blocked (the destination-based mitigation such systems apply).

  ACMS P4RTHENON's in-switch stage (Al Sadi et al., IEEE TNSM 2024; asymmetric flow detection from P-SCOR): per switch
       and window dt, |packets src->dst - packets dst->src| for each address pair; the pair is flagged when it exceeds a
       threshold, and the heavier sender is blocked. (P4RTHENON then mirrors flagged traffic to a CNN; only the
       in-switch stage is evaluated here.)
  DOE  DOE-DTL (Tang et al., IEEE Trans. Networking 2026), hybrid LDoS defence on BMv2: the controller polls per-
       destination TCP/UDP packet and byte counters every 0.3 s, computes TB-MN, TP-Q1, UB-MD and UP-SD over windows
       of 1.5/3/4.5/6 s, gates on TB-MN < tau, classifies with an extreme learning machine (ELM), and declares an
       attack when three adjacent windows of any form are positive. In the switch, a double threshold reports a
       source whose UDP packets in a 0.2 s period reach Smt (R = 1); a reported source is blocked at its next
       report after the alarm. Differences: random search over the ELM's hidden size, C and input weights replaces
       the whale optimisation; tau, Smt and the ELM are chosen on validation runs.

Hyperparameters and thresholds are chosen on the validation runs (T*/rep4) only; models are trained on T*/rep1-3.
The test runs are scored once. P4-DL Shield is scored the same way from its per-flow detection-only runs
(results_flow/), first positive summary at the deployed threshold (the deployed policy, K = 1).

Common unit = flows: a flow is blocked at its first positive decision; times are from the attacker's first packet
(sender log), so they include BMv2's digest delivery delay for every method.

  cd ~/MTP1-main && python3 revision/analysis/switch_baselines.py --out ~/switch_baselines
"""
import argparse, glob, json, os, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_score, recall_score, confusion_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--pp-root", default="results_rev", help="per-packet runs (live_dataset.csv)")
ap.add_argument("--pf-root", default="results_flow", help="per-flow runs (flow_msgs.csv) for P4-DL Shield")
ap.add_argument("--train", default="T*/rep1,T*/rep2,T*/rep3")
ap.add_argument("--val", default="T*/rep4")
ap.add_argument("--test", default="S1_single_A2_detect/rep*,S2_distributed_detect/rep*,S3_*/rep*,S4_*/rep*")
ap.add_argument("--policy", default="models/block_policy_flow.json")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--ent-windows", default="0.05,0.1,0.25,0.5,1.0")
ap.add_argument("--ent-alphas", default="0.05,0.2")
ap.add_argument("--min-pkts", type=int, default=4, help="entropy: windows with fewer packets are not scored")
ap.add_argument("--out", required=True)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
SEEDS = [int(s) for s in A.seeds.split(",")]
THR_CANDS = [0.5, 0.7, 0.9, 0.95, 0.99]           # same candidate list as P4-DL Shield
PHASES = [2 ** k for k in range(1, 12)]            # 2 .. 2048
LOG = open(f"{A.out}/summary.txt", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
t0 = time.time()


# ------------------------------------------------------------------ loading
def dirs(root, globs, need):
    out = []
    for g in globs.split(","):
        out += [d for d in glob.glob(os.path.join(root, g.strip())) if os.path.exists(f"{d}/ctrl_logs/{need}")]
    return sorted(set(out))


def ip2int(s):
    if pd.api.types.is_numeric_dtype(s): return s.astype("int64")
    p = s.astype(str).str.split(".", expand=True).astype("int64")
    return (p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]


def onset_by_src(d):
    """attacker src IP (int) -> time of its first packet, from the sender logs (as in analyze_run.py)."""
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
            a, b, c, e = (int(x) for x in man["attack_flows"][hosts.index(h)]["src"].split("."))
            out[(a << 24) | (b << 16) | (c << 8) | e] = float(fp.iloc[0])
        except Exception:
            pass
    return out


PP_COLS = ["timestamp", "device_id", "src_ip", "dst_ip", "proto", "sport", "dport", "pkt_len", "Label"]
def load_pp(globs):
    parts = []
    for d in dirs(A.pp_root, globs, "live_dataset.csv"):
        x = pd.read_csv(f"{d}/ctrl_logs/live_dataset.csv", usecols=PP_COLS, low_memory=False).dropna()
        x = x.rename(columns={"timestamp": "t", "Label": "y"})
        x["src_ip"] = ip2int(x.src_ip); x["dst_ip"] = ip2int(x.dst_ip)
        for c in ("device_id", "proto", "sport", "dport", "y"): x[c] = x[c].astype("int64")
        x["pkt_len"] = x.pkt_len.astype("float32"); x["t"] = x.t.astype("float64")
        x["run"] = d; x["scenario"] = os.path.basename(os.path.dirname(d))
        x["onset"] = x.src_ip.map(onset_by_src(d))
        parts.append(x.sort_values("t", kind="stable"))
    if not parts: raise SystemExit(f"no live_dataset.csv under {A.pp_root} for {globs}")
    x = pd.concat(parts, ignore_index=True)
    x["fkey"] = x.src_ip.astype(str) + "|" + x.dst_ip.astype(str) + "|" + x.proto.astype(str) + "|" + x.dport.astype(str)
    return x


# ------------------------------------------------------------------ metrics
def eer(y, s):
    f, tp, _ = roc_curve(y, s); n = 1 - tp; i = np.nanargmin(np.abs(n - f)); return float((f[i] + n[i]) / 2)

def mets(y, s, thr):
    y = np.asarray(y); s = np.asarray(s); p = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    m = dict(prec=precision_score(y, p, zero_division=0), rec=recall_score(y, p, zero_division=0),
             f1=f1_score(y, p, zero_division=0), fpr=fp / max(fp + tn, 1))
    if 0 < y.sum() < len(y): m.update(auc=roc_auc_score(y, s), eer=eer(y, s))
    return m

def flow_level(rows):
    """rows: DataFrame(run, scenario, fkey, y, onset, t_block or NaN). One row per flow."""
    b, a = rows[rows.y == 0], rows[rows.y == 1]
    d = (a.t_block - a.onset)
    return dict(benign_flows=len(b), benign_blocked=int(b.t_block.notna().sum()),
                attack_flows=len(a), attack_blocked=int(a.t_block.notna().sum()),
                onset_to_block_mean=round(float(d.mean()), 3) if d.notna().any() else np.nan,
                onset_to_block_max=round(float(d.max()), 3) if d.notna().any() else np.nan)

def flows_from_scores(x, score, thr):
    """first packet of each flow (4-tuple, any switch) whose score >= thr."""
    pos = x[score >= thr].groupby(["run", "fkey"]).t.min()
    f = x.groupby(["run", "fkey"]).agg(scenario=("scenario", "first"), y=("y", "max"), onset=("onset", "first"))
    f["t_block"] = pos.reindex(f.index); return f.reset_index()


# ------------------------------------------------------------------ NetBeacon-style multi-phase trees
NB_FEATS = ["k", "sum_len", "mean_len", "min_len", "max_len", "var_len",
            "mean_iat", "min_iat", "max_iat", "var_iat", "dur", "pkt_len", "proto"]
PL_FEATS = ["pkt_len", "proto"]

def nb_features(x, five):
    key = ["run", "device_id", "src_ip", "dst_ip", "proto", "dport"] + (["sport"] if five else [])
    x = x.sort_values(key + ["t"], kind="stable").copy()
    g = x.groupby(key, sort=False)
    x["k"] = g.cumcount() + 1
    x["sum_len"] = g.pkt_len.cumsum(); sq = (x.pkt_len.astype("float64") ** 2).groupby([x[c] for c in key], sort=False).cumsum()
    x["mean_len"] = x.sum_len / x.k; x["var_len"] = (sq / x.k - x.mean_len ** 2).clip(lower=0)
    x["min_len"] = g.pkt_len.cummin(); x["max_len"] = g.pkt_len.cummax()
    iat = (x.t - g.t.shift(1)); ni = (x.k - 1).clip(lower=1)
    gi = iat.fillna(0).groupby([x[c] for c in key], sort=False)
    x["mean_iat"] = gi.cumsum() / ni
    x["var_iat"] = ((iat.fillna(0) ** 2).groupby([x[c] for c in key], sort=False).cumsum() / ni - x.mean_iat ** 2).clip(lower=0)
    x["min_iat"] = iat.groupby([x[c] for c in key], sort=False).cummin().fillna(0)
    x["max_iat"] = iat.groupby([x[c] for c in key], sort=False).cummax().fillna(0)
    x["dur"] = x.t - g.t.transform("first")
    x["phase"] = np.where(x.k == 1, 0, np.floor(np.log2(x.k.clip(lower=1))).astype(int))
    x["is_point"] = (x.k == 1) | ((x.k & (x.k - 1)) == 0) & (x.k <= PHASES[-1])
    x["_g"] = g.ngroup()
    return x

def nb_train(tr, seed):
    models = {}
    pts = tr[tr.is_point]
    for p in range(0, len(PHASES) + 1):
        d = pts[pts.phase == p]; cols = PL_FEATS if p == 0 else NB_FEATS
        if len(d) < 20 or d.y.nunique() < 2:
            models[p] = models.get(p - 1); continue          # too few flows reach this phase: reuse previous model
        m = RandomForestClassifier(n_estimators=3, max_depth=7, random_state=seed, n_jobs=-1)
        models[p] = (m.fit(d[cols].values, d.y.values), cols)
    return models

def nb_score(x, models):
    s = pd.Series(np.nan, index=x.index)
    pts = x[x.is_point]
    for p, mc in models.items():
        if mc is None: continue
        d = pts[pts.phase == p]
        if len(d): s.loc[d.index] = mc[0].predict_proba(d[mc[1]].values)[:, 1]
    return s.groupby(x._g).ffill().fillna(0).values          # decision holds until the next phase point


# ------------------------------------------------------------------ in-switch entropy (P4DDoS / Euclid style)
def norm_entropy(x, field):
    c = x.groupby(["run", "device_id", "win", field]).size()
    n = c.groupby(level=[0, 1, 2]).transform("sum"); p = c / n
    h = (-(p * np.log2(p))).groupby(level=[0, 1, 2]).sum()
    m = c.groupby(level=[0, 1, 2]).size()
    return (h / np.log2(m.clip(lower=2))).where(m > 1, 0.0)

def ent_windows(x, w):
    x = x.assign(win=np.floor((x.t - x.groupby("run").t.transform("min")) / w).astype("int64"))
    W = pd.DataFrame(dict(Hs=norm_entropy(x, "src_ip"), Hd=norm_entropy(x, "dst_ip")))
    g = x.groupby(["run", "device_id", "win"])
    W["n"] = g.size(); W["y"] = (g.y.mean() >= 0.5).astype(int); W["t_end"] = g.t.max()
    c = x.groupby(["run", "device_id", "win", "dst_ip"]).size().reset_index(name="c")
    c = c.sort_values("c", kind="stable").drop_duplicates(["run", "device_id", "win"], keep="last")
    W["top_dst"] = c.set_index(["run", "device_id", "win"]).dst_ip.reindex(W.index)
    return x, W.reset_index()

def ent_scores(W, alpha, init):
    """EWMA mean and mean absolute deviation per (run, switch); score = largest deviation in units of the MAD."""
    out = np.zeros(len(W))
    for _, idx in W.groupby(["run", "device_id"]).indices.items():
        mu = dict(init["mu"]); md = dict(init["md"])
        for i in idx:
            if W.n.iat[i] < A.min_pkts: continue
            z = 0.0
            for f in ("Hs", "Hd"):
                h = W[f].iat[i]; z = max(z, abs(h - mu[f]) / max(md[f], 1e-3))
                d = abs(h - mu[f]); mu[f] = (1 - alpha) * mu[f] + alpha * h; md[f] = (1 - alpha) * md[f] + alpha * d
            out[i] = z
    return out

def ent_init(trW):
    b = trW[(trW.y == 0) & (trW.n >= A.min_pkts)]
    return dict(mu=dict(Hs=b.Hs.mean(), Hd=b.Hd.mean()),
                md=dict(Hs=(b.Hs - b.Hs.mean()).abs().mean(), Hd=(b.Hd - b.Hd.mean()).abs().mean()))

def ent_flows(x, W, score, thr):
    """block every flow towards the most frequent destination of an alarmed window, from the window's end."""
    al = W[score >= thr][["run", "device_id", "win", "top_dst", "t_end"]]
    seen = x[["run", "device_id", "win", "dst_ip", "fkey"]].drop_duplicates()
    hit = seen.merge(al, on=["run", "device_id", "win"]); hit = hit[hit.dst_ip == hit.top_dst]
    first = hit.groupby(["run", "fkey"]).t_end.min()
    f = x.groupby(["run", "fkey"]).agg(scenario=("scenario", "first"), y=("y", "max"), onset=("onset", "first"))
    f["t_block"] = first.reindex(f.index); return f.reset_index()


# ------------------------------------------------------------------ ACMS (P4RTHENON's in-switch detector, from P-SCOR)
def acms_prep(x, dt):
    """per switch and window of dt seconds: running signed packet count of each address pair (+ for lo->hi)."""
    lo = np.minimum(x.src_ip.values, x.dst_ip.values); hi = np.maximum(x.src_ip.values, x.dst_ip.values)
    win = np.floor((x.t - x.groupby("run").t.transform("min")) / dt).astype("int64")
    d = pd.DataFrame(dict(run=x.run.values, dev=x.device_id.values, win=win.values, lo=lo, hi=hi,
                          s=np.where(x.src_ip.values == lo, 1, -1), t=x.t.values))
    d = d.sort_values(["run", "dev", "win", "lo", "hi", "t"], kind="stable")
    d["c"] = d.groupby(["run", "dev", "win", "lo", "hi"], sort=False).s.cumsum()
    return d

def acms_flows(x, d, th):
    """asym = |c| > th flags the pair; the heavier sender is blocked (all its flows towards the other address)."""
    f = d[d.c.abs() > th].groupby(["run", "lo", "hi"]).agg(t=("t", "min"), c=("c", "first")).reset_index()
    f["heavy"] = np.where(f.c > 0, f.lo, f.hi); f["other"] = np.where(f.c > 0, f.hi, f.lo)
    ft = f.groupby(["run", "heavy", "other"]).t.min().to_dict()
    fl = x.groupby(["run", "fkey"]).agg(scenario=("scenario", "first"), y=("y", "max"), onset=("onset", "first"),
                                         src=("src_ip", "first"), dst=("dst_ip", "first")).reset_index()
    fl["t_block"] = pd.Series(list(zip(fl.run, fl.src, fl.dst))).map(ft).values
    return fl.drop(columns=["src", "dst"])

def flow_f1(fl):
    tp = int(((fl.y == 1) & fl.t_block.notna()).sum()); fp = int(((fl.y == 0) & fl.t_block.notna()).sum())
    fn = int(((fl.y == 1) & fl.t_block.isna()).sum()); return 2 * tp / max(2 * tp + fp + fn, 1)


# ------------------------------------------------------------------ DOE-DTL (Tang et al., IEEE ToN 2026)
DOE_FORMS = (5, 10, 15, 20)          # windows of 1.5, 3, 4.5, 6 s of 0.3 s samples, step 0.3 s
def doe_series(x):
    """0.3 s samples per (run, switch, destination): TCP/UDP packets and bytes; sample label = inside the attack
    period towards that destination (first..last attack packet)."""
    x = x.assign(b=np.floor((x.t - x.groupby("run").t.transform("min")) / 0.3).astype("int64"),
                 tcp=(x.proto == 6), udp=(x.proto == 17))
    g = x.groupby(["run", "device_id", "dst_ip", "b"])
    S = pd.DataFrame(dict(tp=g.tcp.sum(), tb=(x.pkt_len * x.tcp).groupby([x.run, x.device_id, x.dst_ip, x.b]).sum(),
                          up=g.udp.sum(), ub=(x.pkt_len * x.udp).groupby([x.run, x.device_id, x.dst_ip, x.b]).sum(),
                          t=g.t.max())).reset_index()
    ap = x[x.y == 1].groupby(["run", "dst_ip"]).b.agg(["min", "max"])
    S = S.join(ap, on=["run", "dst_ip"]); S["lab"] = ((S.b >= S["min"]) & (S.b <= S["max"])).fillna(False).astype(int)
    keep = S.groupby(["run", "device_id", "dst_ip"]).tp.transform("sum") > 0       # destinations that receive TCP
    return S[keep].drop(columns=["min", "max"])

def doe_windows(S):
    """features of every window form: TB-MN (mean TCP bytes), TP-Q1 (first quartile of TCP packets),
    UB-MD (mean absolute deviation of UDP bytes), UP-SD (std of UDP packets); label 1 if > 80 % attack samples."""
    out = []
    for (run, dev, dst), s in S.groupby(["run", "device_id", "dst_ip"], sort=False):
        full = s.set_index("b").reindex(pd.RangeIndex(s.b.min(), s.b.max() + 1, name="b"))
        full[["tp", "tb", "up", "ub", "lab"]] = full[["tp", "tb", "up", "ub", "lab"]].fillna(0)
        full["t"] = full.t.ffill()
        for L in DOE_FORMS:
            r = full.rolling(L, min_periods=L)
            w = pd.DataFrame(dict(TB_MN=r.tb.mean(), TP_Q1=full.tp.rolling(L, min_periods=L).quantile(0.25),
                                  UB_MD=full.ub.rolling(L, min_periods=L).apply(lambda v: np.mean(np.abs(v - v.mean())), raw=True),
                                  UP_SD=r.up.std(ddof=0), att=r.lab.mean(), t=full.t)).dropna()
            w["y"] = (w.att > 0.8).astype(int); w["run"] = run; w["dev"] = dev; w["dst"] = dst; w["form"] = L
            out.append(w.reset_index())
    return pd.concat(out, ignore_index=True)

DOE_F = ["TB_MN", "TP_Q1", "UB_MD", "UP_SD"]
class ELM:
    """single-hidden-layer network, sigmoid, random input weights, ridge output weights (beta = (H'H + I/C)^-1 H'T)."""
    def __init__(s, L, C, seed): s.L, s.C, s.seed = L, C, seed
    def fit(s, X, y):
        r = np.random.default_rng(s.seed); s.mu, s.sd = X.mean(0), X.std(0) + 1e-9
        s.W = r.uniform(-1, 1, (X.shape[1], s.L)); s.bias = r.uniform(-1, 1, s.L)
        H = s._h(X); s.beta = np.linalg.solve(H.T @ H + np.eye(s.L) / s.C, H.T @ y); return s
    def _h(s, X): return 1 / (1 + np.exp(-(((X - s.mu) / s.sd) @ s.W + s.bias)))
    def score(s, X): return s._h(X) @ s.beta

def jaccard(y, p):   # DOE-DTL's fitness: JI = DR / (DR + FPR + FNR)
    tp = ((p == 1) & (y == 1)).sum(); fn = ((p == 0) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); tn = ((p == 0) & (y == 0)).sum()
    dr = tp / max(tp + fn, 1); fpr = fp / max(fp + tn, 1); return dr / max(dr + fpr + (1 - dr), 1e-9)

def doe_alarm(Wd, s, tau):
    """network-level alarm time per run: after the pre-detection gate (TB-MN < tau) opens, the first time any window
    form has three adjacent windows judged as attack."""
    Wd = Wd.assign(p=(s >= 0.5).astype(int)).sort_values(["run", "dev", "dst", "form", "b"])
    gate = Wd[Wd.TB_MN < tau].groupby("run").t.min()
    Wd = Wd.join(gate.rename("t_gate"), on="run"); Wd = Wd[Wd.t >= Wd.t_gate]
    k = Wd.groupby(["run", "dev", "dst", "form"]).p.transform(lambda v: v.rolling(3, min_periods=3).sum())
    return Wd[k >= 3].groupby("run").t.min()

def doe_suspicious(x, smt):
    """in-switch double threshold (R = 1): per switch and source, UDP packets per 0.2 s period; a period with
    >= smt packets reports the source. Returns (run, src) -> sorted report times."""
    u = x[x.proto == 17]
    b = np.floor((u.t - u.groupby("run").t.transform("min")) / 0.2).astype("int64")
    c = u.groupby([u.run, u.device_id, u.src_ip, b]).t.agg(["size", "max"])
    r = c[c["size"] >= smt].reset_index()
    return r.groupby(["run", "src_ip"])["max"].apply(lambda v: np.sort(v.values))

def doe_flows(x, alarm, rep):
    """a source is blocked at the first report after the alarm, if it was already reported before (DOE-DTL's list)."""
    fl = x.groupby(["run", "fkey"]).agg(scenario=("scenario", "first"), y=("y", "max"), onset=("onset", "first"),
                                         src=("src_ip", "first")).reset_index()
    tb = {}
    for (run, src), ts in rep.items():
        a = alarm.get(run)
        if a is None or len(ts) < 2: continue
        after = ts[ts >= a]; after = after[after > ts[0]]
        if len(after): tb[(run, src)] = float(after[0])
    fl["t_block"] = pd.Series(list(zip(fl.run, fl.src))).map(tb).values
    return fl.drop(columns=["src"])


# ================================================================== run
tr = load_pp(A.train); va = load_pp(A.val); te = load_pp(A.test)
say(f"per-packet logs: train {len(tr):,} records ({tr.run.nunique()} runs), val {len(va):,} ({va.run.nunique()}), "
    f"test {len(te):,} ({te.run.nunique()} runs, {te.y.mean():.2%} attack)")
assert not (set(tr.run) & set(te.run) or set(va.run) & set(te.run) or set(tr.run) & set(va.run)), "a run is in two splits"
RES = []          # one row per method (flow level + per-record metrics)
SCN = []          # per scenario flow level

import re
def record(name, unit, m, fl, extra=""):
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]
    fl.to_csv(f"{A.out}/flows_{slug}.csv", index=False)
    b = fl[(fl.y == 0) & fl.t_block.notna()]
    if len(b):
        kind = b.fkey.str.split("|").map(lambda k: f"proto {k[2]} dport {k[3]}")
        say(f"  [{name}] benign flows blocked, by protocol/port: " + ", ".join(f"{k}: {v}" for k, v in kind.value_counts().items()))
    RES.append(dict(method=name, unit=unit, **{k: m.get(k, np.nan) for k in ("prec", "rec", "f1", "fpr", "auc", "eer")},
                    **flow_level(fl), note=extra))
    for scn, g in fl.groupby("scenario"):
        SCN.append(dict(method=name, scenario=scn, **flow_level(g)))

# ---- NetBeacon-style
for five in (True, False):
    name = "NetBeacon-style trees, " + ("5-tuple" if five else "4-tuple")
    trf, vaf, tef = nb_features(tr, five), nb_features(va, five), nb_features(te, five)
    say(f"\n[{name}] phase points: train {int(trf.is_point.sum()):,}, flows per phase (train):",
        trf[trf.is_point].groupby("phase").size().to_dict())
    per_seed, flows = [], []
    for seed in SEEDS:
        models = nb_train(trf, seed)
        sv = nb_score(vaf, models)
        thr = max(THR_CANDS, key=lambda t: f1_score(vaf.y, (sv >= t).astype(int), zero_division=0))
        st = nb_score(tef, models)
        per_seed.append(dict(thr=thr, **mets(tef.y.values, st, thr)))
        flows.append(flows_from_scores(tef, st, thr))
        say(f"  seed {seed}: validation threshold {thr}; test F1 {per_seed[-1]['f1']:.3f}")
    m = pd.DataFrame(per_seed).mean().to_dict()
    # flow level: seed 0 for counts (seeds reported separately in nb_seeds.csv)
    pd.DataFrame(per_seed).to_csv(f"{A.out}/nb_{'5' if five else '4'}_seeds.csv", index=False)
    seed_fl = pd.DataFrame([flow_level(f) for f in flows]); seed_fl.to_csv(f"{A.out}/nb_{'5' if five else '4'}_flowlevel_seeds.csv", index=False)
    record(name, "packet", m, flows[0], f"F1 {pd.DataFrame(per_seed).f1.mean():.3f} +- {pd.DataFrame(per_seed).f1.std():.3f} over seeds")

# ---- entropy
best = None
for w in [float(v) for v in A.ent_windows.split(",")]:
    _, trW = ent_windows(tr, w); _, vaW = ent_windows(va, w); init = ent_init(trW)
    for a in [float(v) for v in A.ent_alphas.split(",")]:
        sv = ent_scores(vaW, a, init); ok = vaW.n >= A.min_pkts
        if vaW.y[ok].nunique() < 2: continue
        grid = np.unique(np.quantile(sv[ok], np.linspace(0.5, 0.995, 100)))
        f, thr = max((f1_score(vaW.y[ok], (sv[ok] >= t).astype(int), zero_division=0), t) for t in grid)
        say(f"  entropy w={w}s alpha={a}: validation window F1 {f:.3f} at k={thr:.2f}")
        if best is None or f > best[0]: best = (f, w, a, thr, init)
_, w, a, thr, init = best
tex, teW = ent_windows(te, w); st = ent_scores(teW, a, init); ok = teW.n >= A.min_pkts
mw = mets(teW.y[ok].values, st[ok], thr)
# per record: each packet inherits its window's score
pk = tex.merge(teW[["run", "device_id", "win"]].assign(s=st), on=["run", "device_id", "win"], how="left")
mp = mets(pk.y.values, pk.s.fillna(0).values, thr)
fl = ent_flows(tex, teW, st, thr)
record("In-switch entropy (P4DDoS/Euclid style)", "window", mw, fl,
       f"w={w}s alpha={a} k={thr:.2f}; per packet F1 {mp['f1']:.3f}")

# ---- ACMS (P4RTHENON in-switch stage): window and threshold chosen on validation (flow-level F1)
best = None
for dt in (1.0, 5.0, 30.0):
    d = acms_prep(va, dt)
    mx = d.groupby(["run", "dev", "win", "lo", "hi"]).c.agg(lambda v: np.abs(v).max()).values
    for th in np.unique(np.quantile(mx, np.linspace(0.5, 0.999, 60)).round()):
        f = flow_f1(acms_flows(va, d, th))
        if best is None or f > best[0]: best = (f, dt, th)
    say(f"  ACMS dt={dt}s: best validation flow F1 so far {best[0]:.3f} (dt={best[1]}, threshold={best[2]:.0f})")
_, dt, th = best
record("ACMS asymmetry (P4RTHENON in-switch stage)", "flow", {}, acms_flows(te, acms_prep(te, dt), th), f"dt={dt}s threshold={th:.0f}")

# ---- DOE-DTL: ELM size and C, pre-detection threshold and Smt chosen on validation
try:
    Wtr, Wva, Wte = doe_windows(doe_series(tr)), doe_windows(doe_series(va)), doe_windows(doe_series(te))
    say(f"\n[DOE-DTL] windows: train {len(Wtr):,} ({Wtr.y.mean():.1%} attack), val {len(Wva):,}, test {len(Wte):,}")
    best = None
    for L in (20, 50, 100):
        for C in (1.0, 10.0, 100.0):
            for seed in range(5):          # random search replaces DOE-DTL's whale-optimisation of the input weights
                m = ELM(L, C, seed).fit(Wtr[DOE_F].values, Wtr.y.values)
                ji = jaccard(Wva.y.values, (m.score(Wva[DOE_F].values) >= 0.5).astype(int))
                if best is None or ji > best[0]: best = (ji, L, C, seed, m)
    ji, L, C, seed, elm = best
    tau = float(Wva[Wva.y == 1].TB_MN.max())          # pre-detection gate passes every validation attack window
    sv = elm.score(Wva[DOE_F].values); st = elm.score(Wte[DOE_F].values)
    a_va = doe_alarm(Wva, sv, tau); a_te = doe_alarm(Wte, st, tau)
    best_s = None
    u = va[va.proto == 17]; bb = np.floor((u.t - u.groupby("run").t.transform("min")) / 0.2)
    cnt = u.groupby([u.run, u.device_id, u.src_ip, bb]).size()
    for smt in np.unique(np.quantile(cnt.values, np.linspace(0.5, 0.999, 40)).round()):
        f = flow_f1(doe_flows(va, a_va.to_dict(), doe_suspicious(va, smt)))
        if best_s is None or f > best_s[0] or (f == best_s[0] and smt > best_s[1]): best_s = (f, smt)
    smt = best_s[1]
    say(f"  ELM L={L} C={C} (validation JI {ji:.3f}); pre-detection TB-MN < {tau:.0f} B; Smt={smt:.0f} UDP packets / 0.2 s")
    mw = mets(Wte.y.values, st, 0.5)
    record("DOE-DTL (ELM + in-switch double threshold)", "window", mw,
           doe_flows(te, a_te.to_dict(), doe_suspicious(te, smt)), f"L={L} C={C} tau={tau:.0f} Smt={smt:.0f}")
    for scn, g in te.groupby("scenario"):
        on = g.groupby("run").onset.min(); al = a_te.reindex(on.index)
        d = (al - on)
        say(f"  DOE-DTL network alarm, {scn}: {int(al.notna().sum())}/{len(on)} runs alarmed; "
            f"{int((d < 0).sum())} before the attack started (false alarm); onset -> alarm when after onset: "
            f"{float(d[d >= 0].mean()) if (d >= 0).any() else float('nan'):.2f} s")
except Exception as e:
    say(f"  (DOE-DTL rows skipped: {type(e).__name__}: {e})")

# ---- P4-DL Shield (per-flow, deployed model's live scores, first positive summary)
try:
    THR = float(json.load(open(A.policy))["threshold"])
    parts = []
    for d in dirs(A.pf_root, A.test, "flow_msgs.csv"):
        x = pd.read_csv(f"{d}/ctrl_logs/flow_msgs.csv", low_memory=False)
        x.loc[x.msg_type != 1, "prob"] = np.nan          # new-flow messages carry no score but count as flows
        x["run"] = d; x["scenario"] = os.path.basename(os.path.dirname(d)); x["onset"] = x.src_ip.map(onset_by_src(d))
        parts.append(x)
    x = pd.concat(parts, ignore_index=True).rename(columns={"label": "y", "t_rx": "t"})
    x["fkey"] = x.src_ip.astype(str) + "|" + x.dst_ip.astype(str) + "|" + x.proto.astype(str) + "|" + x.dport.astype(str)
    sm = x.prob.notna()
    record("P4-DL Shield (per-flow MLP, deployed)", "summary", mets(x.y[sm].values, x.prob[sm].values, THR),
           flows_from_scores(x, x.prob.fillna(0).values, THR), f"threshold {THR}; {x.run.nunique()} per-flow runs")
except Exception as e:
    say(f"  (P4-DL Shield rows skipped: {e})")

R = pd.DataFrame(RES); R.to_csv(f"{A.out}/switch_baselines.csv", index=False)
S = pd.DataFrame(SCN); S.to_csv(f"{A.out}/switch_baselines_by_scenario.csv", index=False)
say("\n=== IN-SWITCH BASELINES vs P4-DL Shield (test scenarios; metrics in each method's own unit; "
    "flow level = first positive decision blocks the flow)\n" + R.round(4).to_string(index=False))
say("\n=== PER SCENARIO (flow level)\n" + S.to_string(index=False))
say(f"\ndone in {time.time() - t0:.0f} s")
