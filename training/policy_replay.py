#!/usr/bin/env python3
"""
policy_replay.py — choose the BLOCKING POLICY on validation runs by replaying the deployed model offline.

The DNN scores every digest; the policy decides when a flow gets a drop rule. Blocking on the first
positive digest blocks benign flows that look attack-like for one burst (e.g. the bursty benign h42) and
lets a handful of stray positives kill a legitimate flow. A persistence policy blocks a flow only when
  * at least K positive digests fall inside the last W seconds, AND
  * positives occurred in at least B different 1-second bins within the last R seconds (recurrence).

The policy (and score threshold) are chosen on VALIDATION runs (T*/rep4) plus PILOT runs: detection-only
runs recorded only for this choice (results_pilot/), never part of the reported results. Detection-only
matters: in runs with blocking on, a flow's digests stop once it is blocked, which biases a replay.
Selection order:
  1) fewest benign flows blocked, 2) all attack flows blocked, 3) shortest time to block.
Writes models/block_policy.json, which the controller loads automatically.

  python3 revision/training/policy_replay.py --model models/ddos_dnn_rev_nosys.pth \
      --val 'results_rev/T*/rep4' [--preview 'results_smoke/*/rep1']
"""
import argparse, glob, json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--val", default=None)
ap.add_argument("--preview", default=None, help="runs to show (NOT used for the choice)")
ap.add_argument("--out", default="models/block_policy.json")
ap.add_argument("--use-policy", default=None, help="only preview this existing policy JSON (no new choice)")
ap.add_argument("--min-bins", type=int, default=1,
                help="design rule fixed in advance: positives must recur in >= this many 1-s bins (LDoS is periodic)")
A = ap.parse_args()
sys.path.insert(0, "controller")
import torch, joblib                                  # noqa: E402
from dnn_model import load_model                       # noqa: E402

ck = torch.load(A.model, map_location="cpu")
FEATS = ck["features"]
M = load_model(A.model, device="cpu"); M["scaler"] = joblib.load(A.model.replace(".pth", "_scaler.pkl"))
net = M["model"].eval()


def score(df):
    X = M["scaler"].transform(df[FEATS].astype(float).values)
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(X, dtype=torch.float32))).numpy().ravel()


def load_runs(pattern):
    out = []
    for p in pattern.split(","):
        for d in sorted(glob.glob(p.strip())):
            f = f"{d}/ctrl_logs/live_dataset.csv"
            if os.path.exists(f) and os.path.getsize(f) > 0:
                x = pd.read_csv(f, low_memory=False)
                x = x.dropna(subset=FEATS)
                if len(x) == 0:                  # e.g. S14 (digests off)
                    continue
                x["prob"] = score(x); x["run"] = d
                out.append(x)
    return out


def flows_of(x):
    """Pre-group once: per flow (src,dst,proto,dport) across switches, time-ordered arrays."""
    out = []
    for key, g in x.sort_values("timestamp").groupby(["src_ip", "dst_ip", "proto", "dport"]):
        out.append((g.run.iloc[0], key, int(g.Label.max()), g.timestamp.values, g.prob.values))
    return out


GAP_S, SUSTAIN_S = 0.3, 0.5          # fixed design constants (same as the controller)


def first_block(pos, K, W, B, R, G=GAP_S, L=SUSTAIN_S):
    """Same rule as the controller (digest_handler_v2._policy_says_block), O(n).
    Block a flow when K positives fall in the last W seconds AND
      (positives formed >= B separate episodes within the last R seconds  OR  the current episode has lasted >= L s).
    An episode is a run of positives with no gap longer than G seconds. Fixed design constants: G = 0.3 s, L = 0.5 s
    (an LDoS attack recurs; a flood persists; one benign burst does neither)."""
    lo = 0; eps = []; last = None
    for i, now in enumerate(pos):
        if last is None or now - last > G: eps.append(now)
        last = now
        while now - pos[lo] > W: lo += 1
        while eps and now - eps[0] > R and len(eps) > 1: eps.pop(0)
        n_eps = sum(1 for e in eps if now - e <= R)
        if i - lo + 1 >= K and (n_eps >= B or now - eps[-1] >= L):
            return float(now)
    return None


def simulate(flows, thr, K, W, B, R):
    return pd.DataFrame([dict(run=r, flow=k, label=lab, first=t[0], blocked=first_block(t[p >= thr], K, W, B, R))
                         for r, k, lab, t, p in flows])


def evaluate(runs, pol):
    fl = pd.concat([simulate(x, *pol) for x in runs], ignore_index=True)
    ben, att = fl[fl.label == 0], fl[fl.label == 1]
    delay = (att.blocked - att["first"]).dropna()
    return dict(benign_flows=len(ben), benign_blocked=int(ben.blocked.notna().sum()),
                attack_flows=len(att), attack_blocked=int(att.blocked.notna().sum()),
                delay_s_mean=float(delay.mean()) if len(delay) else None,
                delay_s_max=float(delay.max()) if len(delay) else None), fl


if A.use_policy:
    pp = json.load(open(A.use_policy))
    pol_t = (pp["threshold"], pp["K"], pp["window_s"], pp["min_bins"], pp["recur_window_s"])
    rows = []
    for d in sorted({q for p in A.preview.split(",") for q in glob.glob(p.strip())}):
        runs = load_runs(d)
        if not runs: continue
        m, fl = evaluate([flows_of(x) for x in runs], pol_t)
        att = fl[fl.label == 1]
        rows.append(dict(scenario=d.split("/")[-2], run=d.split("/")[-1], **m,
                         attack_flows_with_10plus_digests=int(sum(len(t) >= 10 for _, _, lab, t, _ in flows_of(runs[0]) if lab == 1))))
    out = pd.DataFrame(rows); out.to_csv(A.out, index=False)
    pd.set_option("display.width", 200)
    print(out.groupby("scenario")[["benign_blocked", "benign_flows", "attack_blocked", "attack_flows", "delay_s_mean", "delay_s_max"]]
          .mean().round(3).to_string())
    sys.exit(0)

assert A.val, "--val is required unless --use-policy is given"
GRID = [(thr, K, W, B, R) for thr in (float(ck.get("threshold", 0.5)), 0.5, 0.9, 0.99)
        for (K, W) in ((1, 1.0), (5, 1.0), (20, 1.0), (50, 1.0))
        for (B, R) in ((1, 1.0), (2, 3.0), (3, 4.0))]
val_df = load_runs(A.val)
print(f"validation runs: {len(val_df)}  digests: {sum(len(x) for x in val_df)}  features: {FEATS}")
val = [flows_of(x) for x in val_df]
rows = []
for pol in GRID:
    m, _ = evaluate(val, pol)
    rows.append(dict(thr=pol[0], K=pol[1], W=pol[2], B=pol[3], R=pol[4], **m))
tab = pd.DataFrame(rows)
tab["all_attacks"] = tab.attack_blocked == tab.attack_flows
cand = tab[tab.B >= A.min_bins]
best = cand[cand.all_attacks].sort_values(["benign_blocked", "delay_s_mean", "thr", "K"],
                                          ascending=[True, True, False, False]).head(1)   # ties -> conservative
if best.empty:
    best = cand.sort_values(["benign_blocked", "attack_blocked", "delay_s_mean"], ascending=[True, False, True]).head(1)
b = best.iloc[0]
pol = dict(threshold=float(b.thr), K=int(b.K), window_s=float(b.W), min_bins=int(b.B), recur_window_s=float(b.R),
           gap_s=GAP_S, sustain_s=SUSTAIN_S,
           chosen_on=A.val, model=A.model, design_min_bins=A.min_bins, val_benign_blocked=int(b.benign_blocked),
           val_attack_blocked=f"{int(b.attack_blocked)}/{int(b.attack_flows)}", val_delay_s_mean=b.delay_s_mean)
json.dump(pol, open(A.out, "w"), indent=2)
pd.set_option("display.width", 200)
print("\nvalidation (top 12 by benign blocked, then delay):")
print(tab.sort_values(["benign_blocked", "delay_s_mean"]).head(12).round(3).to_string(index=False))
print(f"\nCHOSEN -> {A.out}: {pol}")
first = [r for r in rows if (r["K"], r["B"]) == (1, 1) and r["thr"] == float(ck.get("threshold", 0.5))][0]
print(f"current rule (block on first positive) on validation: benign blocked {first['benign_blocked']}/{first['benign_flows']}, "
      f"attacks blocked {first['attack_blocked']}/{first['attack_flows']}")
if A.preview:
    pv = [flows_of(x) for x in load_runs(A.preview)]
    for name, p in (("current rule", (float(ck.get("threshold", 0.5)), 1, 1.0, 1, 1.0)),
                    ("chosen policy", (pol["threshold"], pol["K"], pol["window_s"], pol["min_bins"], pol["recur_window_s"]))):
        m, fl = evaluate(pv, p)
        print(f"\npreview with {name}: {m}")
        bb = fl[(fl.label == 0) & fl.blocked.notna()]
        for _, r in bb.iterrows(): print("   benign blocked:", r.run.split("/")[-2], r.flow)
