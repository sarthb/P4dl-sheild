#!/usr/bin/env python3
"""
compare_flow.py -- per-packet (results_rev) vs per-flow (results_flow) runs of the same scenarios.
Needs analysis/analyze_run.py to have been run on both folders (all_runs.csv in each).

  python3 revision/analysis/compare_flow.py --pp results_rev --pf results_flow --out results_flow/compare
"""
import argparse, glob, os
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--pp", default="results_rev"); ap.add_argument("--pf", default="results_flow")
ap.add_argument("--out", default="results_flow/compare")
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)

COLS = [("digests", "messages per run"), ("digest_rate_per_switch_peak", "peak msgs/s per switch"),
        ("q_ms_p99", "queue delay p99 (ms)"), ("ctrl_cpu_pct_mean", "controller CPU %"),
        ("ctrl_rss_MB_max", "controller RSS MB"), ("bmv2_cpu_pct_mean", "BMv2 CPU %"),
        ("flows_benign_blocked", "benign flows blocked"), ("flows_attack_blocked", "attack flows blocked"),
        ("flows_attack", "attack flows"), ("att_detection_delay_s_mean", "detection delay (s)"),
        ("att_onset_to_block_s_mean", "onset -> block (s)"), ("att_leaked_pkts_before_block_mean", "attack pkts before block"),
        ("tcp_mbps_attack", "TCP during attack (Mb/s)"), ("ping_loss_attack_pct", "ping loss (%)"),
        ("ctrl_flow_table_max", "controller flow state max"), ("src_rules_installed", "source rules")]


def load(root):
    f = f"{root}/all_runs.csv"
    if not os.path.exists(f): raise SystemExit(f"{f} missing: run analysis/analyze_run.py {root} first")
    return pd.read_csv(f, low_memory=False)


pp, pf = load(A.pp), load(A.pf)
common = sorted(set(pp.scenario) & set(pf.scenario))
rows = []
for scn in common:
    for name, d in (("per-packet", pp), ("per-flow", pf)):
        g = d[d.scenario == scn]
        r = {"scenario": scn, "design": name, "runs": len(g)}
        for c, lab in COLS:
            if c in g: r[lab] = g[c].mean()
        rows.append(r)
t = pd.DataFrame(rows)
t.to_csv(f"{A.out}/compare_by_scenario.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
with open(f"{A.out}/compare.txt", "w") as fo:
    for scn in common:
        s = t[t.scenario == scn].set_index("design").drop(columns="scenario").T.round(3)
        txt = f"\n=== {scn}\n{s.to_string()}\n"; print(txt); fo.write(txt)

    # summary-level classification of the per-flow model on its own logged scores (window summaries only)
    res = []
    for d in sorted(glob.glob(f"{A.pf}/*/rep*")):
        f = f"{d}/ctrl_logs/flow_msgs.csv"
        if not os.path.exists(f): continue
        x = pd.read_csv(f, low_memory=False); x = x[(x.msg_type == 1) & (x.prob >= 0)]
        if len(x) == 0 or x.label.nunique() < 2: continue
        thr = 0.5
        res.append(dict(scenario=os.path.basename(os.path.dirname(d)), f1=f1_score(x.label, x.pred),
                        auc=roc_auc_score(x.label, x.prob), summaries=len(x),
                        newflow_msgs=None))
    if res:
        r = pd.DataFrame(res).groupby("scenario").mean(numeric_only=True).round(4)
        txt = "\n=== PER-FLOW MODEL, window summaries (live, deployed threshold)\n" + r.to_string() + "\n"
        print(txt); fo.write(txt)
print(f"written {A.out}/compare.txt and compare_by_scenario.csv")
