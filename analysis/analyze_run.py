#!/usr/bin/env python3
"""
analyze_run.py — metrics for one or more results_rev/<scenario>/rep<k> directories, then
aggregation (mean ± std over repetitions) per scenario.

    python3 revision/analysis/analyze_run.py results_rev            # all runs
    python3 revision/analysis/analyze_run.py results_rev/S2_distributed

Writes <run>/metrics.json and results_rev/summary_by_scenario.csv (+ .md).

Metric definitions (put these in the paper):
  onset            = first packet sent by an attacker (sender-side log, udp_pulse.py)
  first_digest     = first digest of that attack flow received by the controller (any switch)
  first_detection  = first digest of that flow with DNN prob >= threshold (end of inference)
  rule_ack         = P4Runtime Write for that flow's drop rule acknowledged (first switch)
  block            = rule_ack; verified by counting attack packets reaching the victim afterwards
  detection_delay  = first_detection - onset          (replaces the old 6.60 s definition)
  onset_to_block   = rule_ack - onset                 (total exposure; replaces "MRT")
  leaked_pkts      = attack packets that reached the victim in [onset, block]
"""
import sys, os, re, json, glob
import numpy as np, pandas as pd

LINK_MBPS = 10.0


def ip2int(s):
    a = [int(x) for x in s.split(".")]; return (a[0] << 24) | (a[1] << 16) | (a[2] << 8) | a[3]


def pct(x, q):
    x = [v for v in x if v is not None and not np.isnan(v)]
    return float(np.percentile(x, q)) if x else None


def read_victim(path):
    rx = re.compile(r"^(\d+\.\d+) IP (\d+\.\d+\.\d+\.\d+)\.(\d+) > (\d+\.\d+\.\d+\.\d+)\.(\d+): UDP, length (\d+)")
    rows = []
    if os.path.exists(path):
        for ln in open(path, errors="ignore"):
            m = rx.match(ln)
            if m: rows.append((float(m[1]), m[2], int(m[5]), int(m[6])))
    return pd.DataFrame(rows, columns=["t", "src", "dport", "len"])


def read_send(path):
    d = pd.read_csv(path)
    if len(d) == 0:
        raise RuntimeError(f"sender log has no rows: {path}")
    ev = {e: g for e, g in d.groupby("event")}
    first = float(ev["first_packet"].t.iloc[0]) if "first_packet" in ev else None
    end = ev.get("end", d.tail(1))
    bursts = []
    if "burst_start" in ev and "burst_end" in ev:
        bs = ev["burst_start"].set_index("burst_idx"); be = ev["burst_end"].set_index("burst_idx")
        j = bs.join(be, lsuffix="_s", rsuffix="_e", how="inner")
        j["n"] = j.pkts_sent_e - j.pkts_sent_s; j["dur"] = j.t_e - j.t_s
        bursts = j
    return dict(first=first, total_pkts=int(end.pkts_sent.iloc[-1]), total_bytes=int(end.bytes_ip.iloc[-1]),
                t_end=float(end.t.iloc[-1]), bursts=bursts, df=d)


def phase_mean(df, tcol, vcol, t0, t1):
    x = df[(df[tcol] >= t0) & (df[tcol] < t1)][vcol]
    return float(x.mean()) if len(x) else None


def _csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"empty or missing: {path}  (controller wrote nothing; see controller.log)")
    return pd.read_csv(path, low_memory=False)


def analyze(run):
    man = json.load(open(f"{run}/manifest.json")); sc = json.load(open(f"{run}/scenario.json"))
    tl = json.load(open(f"{run}/timeline.json"))
    L = f"{run}/ctrl_logs"
    notel = man.get("telemetry", True) is False      # S14: digests intentionally off
    tm = _csv(f"{L}/timing.csv") if not notel else None
    if not notel and len(tm) == 0:
        raise RuntimeError(f"{L}/timing.csv has a header but no digests: the controller received none")
    ev = pd.read_csv(f"{L}/rule_events.csv") if os.path.exists(f"{L}/rule_events.csv") else pd.DataFrame()
    occ = pd.read_csv(f"{L}/rule_occupancy.csv") if os.path.exists(f"{L}/rule_occupancy.csv") else pd.DataFrame()
    vic = read_victim(f"{run}/victim_udp.txt")
    t_att0, t_att1 = tl["t_attack_cmd"], tl["t_attack_end"]
    M = dict(run=run, scenario=os.path.basename(os.path.dirname(run)), mode=man["mode"],
             mitigation=man["mitigation"])

    att = []; M["attackers"] = []
    if notel:
        M["note"] = "telemetry/digests disabled: forwarding-only measurements"
    if not notel:
        # ---------------- detection quality (digest level and flow level)
        from sklearn.metrics import roc_auc_score, roc_curve
        y, p, s = tm.true_label.values, tm.pred.values, tm.prob.values
        tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
        M.update(digests=len(tm), TP=tp, FP=fp, FN=fn, TN=tn,
                 precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1) if y.sum() else None,
                 fpr=fp / max(fp + tn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1) if y.sum() else None)
        if 0 < y.sum() < len(y) and man["mode"] != "tag_only":
            f_, t_, _ = roc_curve(y, s); fn_ = 1 - t_; i = np.nanargmin(np.abs(fn_ - f_))
            M.update(auc=float(roc_auc_score(y, s)), eer=float((f_[i] + fn_[i]) / 2))
        fl = tm.groupby(["src_ip", "dst_ip", "proto", "dport"]).agg(lab=("true_label", "max"), hit=("pred", "max"))
        blk = tm[tm.rule_action.isin(["installed", "src_installed"])].groupby(["src_ip", "dst_ip", "proto", "dport"]).size()
        fl["blocked"] = fl.index.isin(blk.index)
        M["flows_benign_blocked"] = int(((fl.lab == 0) & fl.blocked).sum())
        M["flows_attack_blocked"] = int(((fl.lab == 1) & fl.blocked).sum())
        M.update(flows_benign=int((fl.lab == 0).sum()), flows_benign_flagged=int(((fl.lab == 0) & (fl.hit == 1)).sum()),
                 flows_attack=int((fl.lab == 1).sum()), flows_attack_detected=int(((fl.lab == 1) & (fl.hit == 1)).sum()))

        # ---------------- controller pipeline timing (all digests)
        tm["q_ms"] = (tm.t_dequeue - tm.t_ctrl_rx) * 1e3
        tm["feat_ms"] = (tm.t_features - tm.t_dequeue) * 1e3
        tm["inf_ms"] = (tm.t_inference - tm.t_features) * 1e3
        tm["rpc_ms"] = (tm.t_rule_ack - tm.t_rule_sent) * 1e3
        for c in ("q_ms", "feat_ms", "inf_ms", "rpc_ms"):
            M[f"{c}_p50"], M[f"{c}_p99"] = pct(tm[c], 50), pct(tm[c], 99)
        if (tm.sw_ingress_ts_us >= 0).any():  # switch->controller lag beyond the per-switch minimum
            lag = []
            for dev, g in tm[tm.sw_ingress_ts_us >= 0].groupby("device_id"):
                off = (g.t_ctrl_rx - g.sw_ingress_ts_us / 1e6)
                lag += list((off - off.min()) * 1e3)
            M["digest_lag_ms_p50"], M["digest_lag_ms_p99"], M["digest_lag_ms_max"] = pct(lag, 50), pct(lag, 99), max(lag)
        per_s = tm.groupby([tm.t_ctrl_rx.astype(int), "device_id"]).size()
        M["digest_rate_per_switch_mean"], M["digest_rate_per_switch_peak"] = float(per_s.mean()), int(per_s.max())
        M["queue_depth_max"] = int(tm.queue_depth_at_rx.max())
        if "flow_table_size" in tm:
            M["ctrl_flow_table_max"] = int(tm.flow_table_size.max())
            M["src_rules_installed"] = int((tm.rule_action == "src_installed").sum())
        # controller pipeline under attack vs. before it (control-plane impact)
        pre, dur = tm[tm.t_ctrl_rx < t_att0], tm[tm.t_ctrl_rx.between(t_att0, t_att1)]
        for nm, part in (("pre", pre), ("attack", dur)):
            M[f"q_ms_p99_{nm}"] = pct(part.q_ms, 99)
            M[f"digest_rate_{nm}"] = float(len(part) / max(part.t_ctrl_rx.max() - part.t_ctrl_rx.min(), 1)) if len(part) else None

        # ---------------- per-attacker characterisation + latency breakdown
        att = []
        for f in sorted(glob.glob(f"{run}/send_attack_*.csv")):
            host = f.split("send_attack_")[1][:-4]
            sd = read_send(f); src_ip = sc and [t for t in sc["traffic"] if t.get("attack") and t["host"] == host][0]
            src = [a["src"] for a in man["attack_flows"]][[t["host"] for t in sc["traffic"] if t.get("attack")].index(host)]
            si = ip2int(src); g = tm[(tm.src_ip == si) & (tm.true_label == 1)]
            r = dict(host=host, src=src, pkts_sent=sd["total_pkts"],
                     pkt_size_B=int(src_ip["size"]) + 28, burst_ms=src_ip["burst_ms"], period_ms=src_ip["period_ms"])
            b = sd["bursts"]
            if len(b):
                r["peak_rate_mbps"] = float((b.n * r["pkt_size_B"] * 8 / b.dur.clip(lower=1e-3)).median() / 1e6)
                r["peak_rate_pps"] = float((b.n / b.dur.clip(lower=1e-3)).median())
            dur = sd["t_end"] - sd["first"] if sd["first"] else None
            r["avg_rate_mbps"] = sd["total_bytes"] * 8 / dur / 1e6 if dur else None
            on = sd["first"]; r["onset"] = on
            r["first_digest"] = float(g.t_ctrl_rx.min()) if len(g) else None
            det = g[g.pred == 1]
            r["first_detection"] = float(det.t_inference.min()) if len(det) else None
            inst = g[g.rule_action.isin(["installed", "src_installed"])]
            r["rule_sent"] = float(inst.t_rule_sent.min()) if len(inst) else None
            r["rule_ack"] = float(inst.t_rule_ack.min()) if len(inst) else None
            v = vic[vic.src == src]
            if on:
                r["onset_to_first_digest_s"] = r["first_digest"] - on if r["first_digest"] else None
                r["digest_to_detection_s"] = (r["first_detection"] - r["first_digest"]) if r["first_detection"] else None
                r["detection_delay_s"] = r["first_detection"] - on if r["first_detection"] else None
                r["detection_to_rule_sent_s"] = (r["rule_sent"] - r["first_detection"]) if r["rule_sent"] and r["first_detection"] else None
                r["rule_rpc_s"] = (r["rule_ack"] - r["rule_sent"]) if r["rule_ack"] else None
                r["onset_to_block_s"] = r["rule_ack"] - on if r["rule_ack"] else None
                if r["rule_ack"]:
                    r["leaked_pkts_before_block"] = int(((v.t >= on) & (v.t <= r["rule_ack"])).sum())
                    # split what reaches the victim after the block:
                    #  - in-flight: packets already queued at the bottleneck when the rule went in (<= 1 s)
                    #  - after first rule expiry (TTL): leakage until the attack is re-detected and re-blocked
                    exp = ev[(ev.event == "expired_ttl") & (ev.src == si)].t.min() if len(ev) else np.nan
                    end = exp if not np.isnan(exp) else np.inf
                    r["attack_pkts_inflight_after_block"] = int(((v.t > r["rule_ack"]) & (v.t <= min(r["rule_ack"] + 1.0, end))).sum())
                    r["attack_pkts_after_block_before_expiry"] = int(((v.t > r["rule_ack"] + 1.0) & (v.t <= end)).sum())
                    if not np.isnan(exp):
                        re_ = inst[inst.t_rule_ack > exp].t_rule_ack.min() if len(inst) else np.nan
                        r["first_rule_expiry_s_after_block"] = float(exp - r["rule_ack"])
                        r["redetect_after_expiry_s"] = float(re_ - exp) if not np.isnan(re_) else None
                        r["attack_pkts_leaked_after_expiry"] = int((v.t > exp).sum())
                r["attack_pkts_at_victim"] = len(v)
                # digests of this flow missed before the first detection (window of vulnerability)
                if r["first_detection"]:
                    r["digests_before_detection"] = int((g.t_inference < r["first_detection"]).sum())
            pr = f"{run}/probe_{host}.csv"
            if os.path.exists(pr):
                ps = read_send(pr); p0 = ps["df"].t.min()
                dport_att = [t for t in sc["traffic"] if t.get("attack") and t["host"] == host][0]["dport"]
                pv = v[(v.t >= p0 - 0.01) & (v.t <= p0 + 1.0) & ((v.dport == dport_att) if dport_att > 0 else True)]
                r["probe_sent"], r["probe_received"] = ps["total_pkts"], len(pv)
                r["attack_loss_after_mitigation_pct"] = 100 * (1 - len(pv) / max(ps["total_pkts"], 1))
            att.append(r)
        M["attackers"] = att
        for k in ("detection_delay_s", "onset_to_block_s", "onset_to_first_digest_s", "digest_to_detection_s",
                  "rule_rpc_s", "leaked_pkts_before_block", "attack_pkts_inflight_after_block",
                  "attack_pkts_after_block_before_expiry", "attack_pkts_leaked_after_expiry", "redetect_after_expiry_s", "attack_loss_after_mitigation_pct", "peak_rate_mbps",
                  "avg_rate_mbps"):
            vals = [a.get(k) for a in att if a.get(k) is not None]
            M[f"att_{k}_mean"] = float(np.mean(vals)) if vals else None
        M["att_aggregate_avg_mbps"] = float(np.nansum([a.get("avg_rate_mbps") or 0 for a in att])) if att else 0.0

    # ---------------- benign impact
    blk = min([a["rule_ack"] for a in att if a.get("rule_ack")], default=None)
    for f in sorted(glob.glob(f"{run}/send_benign_*.csv")) + sorted(glob.glob(f"{run}/send_bg_*.csv")):
        nm = os.path.basename(f)[5:-4]; host = nm.split("_")[-1]
        sd = read_send(f); src = [t for t in sc["traffic"] if t["host"] == host and t["kind"] == "pulse"][0]
        from_ip = {"h11": "10.0.1.1", "h12": "10.0.1.2", "h21": "10.0.2.1", "h22": "10.0.2.2", "h31": "10.0.3.1",
                   "h32": "10.0.3.2", "h42": "10.0.4.2"}[host]
        v = vic[(vic.src == from_ip) & (vic.dport == src["dport"])]
        M[f"benign_{nm}_loss_pct"] = 100 * (1 - len(v) / max(sd["total_pkts"], 1))
        # loss during attack window vs. outside, using sender burst logs
        sent_att = sd["df"][(sd["df"].event == "burst_end") & (sd["df"].t.between(t_att0, t_att1))]
        if len(sent_att) > 1:
            n_sent = int(sent_att.pkts_sent.iloc[-1] - sent_att.pkts_sent.iloc[0])
            n_rx = int(v.t.between(sent_att.t.iloc[0], sent_att.t.iloc[-1]).sum())
            M[f"benign_{nm}_loss_during_attack_pct"] = 100 * (1 - n_rx / max(n_sent, 1))
    tcpf = sorted(f for f in glob.glob(f"{run}/tcp_*.csv") if os.path.getsize(f) > 0)
    tcpf.sort(key=lambda f: 0 if f.endswith("tcp_h12.csv") else 1)   # h12 = the reference benign TCP flow
    if not tcpf:
        M["warning_tcp"] = "tcp_*.csv empty: TCP throughput not measured"
    for f in tcpf[1:]:                 # extra TCP flows (e.g. legit TCP from the attacker host, S13)
        h = os.path.basename(f)[4:-4]
        t = pd.read_csv(f, header=None, on_bad_lines="skip")
        if t.shape[1] >= 9:
            t = t[t[6].astype(str).str.match(r"^\d+\.\d+-\d+\.\d+$")]
            st = tl["t_traffic_start"]; t["t"] = st + t[6].str.split("-").str[1].astype(float)
            t["mbps"] = t[8].astype(float) / 1e6
            M[f"tcp_{h}_mbps_pre"] = phase_mean(t, "t", "mbps", st + 2, t_att0)
            M[f"tcp_{h}_mbps_attack"] = phase_mean(t, "t", "mbps", t_att0, t_att1)
    if tcpf:
        t = pd.read_csv(tcpf[0], header=None, on_bad_lines="skip")
        if t.shape[1] >= 9:
            t = t[t[6].astype(str).str.match(r"^\d+\.\d+-\d+\.\d+$")]
            st = tl["t_traffic_start"]; t["t"] = st + t[6].str.split("-").str[1].astype(float)
            t["mbps"] = t[8].astype(float) / 1e6
            M["tcp_mbps_pre"] = phase_mean(t, "t", "mbps", st + 2, t_att0)
            M["tcp_mbps_attack"] = phase_mean(t, "t", "mbps", t_att0, t_att1)
            if blk: M["tcp_mbps_attack_after_block"] = phase_mean(t, "t", "mbps", blk + 1, t_att1)
            M["tcp_mbps_post"] = phase_mean(t, "t", "mbps", t_att1 + 2, tl["t_end"])
    pf = glob.glob(f"{run}/ping_*.txt")
    if pf:
        rows = [(float(m[1]), float(m[2])) for m in
                (re.search(r"\[(\d+\.\d+)\].*time=([\d.]+) ms", ln) for ln in open(pf[0])) if m]
        pg = pd.DataFrame(rows, columns=["t", "rtt"])
        M["rtt_ms_pre"] = phase_mean(pg, "t", "rtt", 0, t_att0); M["rtt_ms_attack"] = phase_mean(pg, "t", "rtt", t_att0, t_att1)
        M["rtt_ms_p50_all"], M["rtt_ms_p99_all"] = pct(pg.rtt, 50), pct(pg.rtt, 99)
        exp = (t_att1 - t_att0) / 0.2; got = pg.t.between(t_att0, t_att1).sum()
        M["ping_loss_attack_pct"] = 100 * (1 - got / exp) if exp > 0 else None

    # ---------------- bottleneck utilisation
    if os.path.exists(f"{run}/bottleneck.csv"):
        b = pd.read_csv(f"{run}/bottleneck.csv"); b["mbps"] = b.tx_bytes.diff() * 8 / b.t.diff() / 1e6
        ina = b[b.t.between(t_att0, t_att1)]
        M["bottleneck_util_pct_attack_mean"] = 100 * float(ina.mbps.mean()) / LINK_MBPS if len(ina) else None
        M["bottleneck_util_pct_attack_peak"] = 100 * float(ina.mbps.max()) / LINK_MBPS if len(ina) else None
        M["bottleneck_util_pct_pre_mean"] = 100 * float(b[b.t < t_att0].mbps.mean()) / LINK_MBPS
    try:
        d0 = int(re.search(r"dropped (\d+)", open(f"{run}/tc_before.txt").read())[1])
        d1 = int(re.search(r"dropped (\d+)", open(f"{run}/tc_after.txt").read())[1])
        M["bottleneck_queue_drops"] = d1 - d0
    except Exception:
        pass

    # ---------------- rule management
    if len(ev):
        for k in ("installed", "src_installed", "expired_ttl", "evicted_lru", "revoked", "insert_failed"):
            M[f"rules_{k}"] = int((ev.event == k).sum())
        gaps = []
        for key, g in ev.groupby(["switch", "src", "dst", "proto", "dport"]):
            g = g.sort_values("t"); exp_t = None
            for _, r in g.iterrows():
                if r.event == "expired_ttl": exp_t = r.t
                elif r.event == "installed" and exp_t is not None: gaps.append(r.t - exp_t); exp_t = None
        M["redetection_gap_s_mean"] = float(np.mean(gaps)) if gaps else None
    if len(occ): M["flow_guard_occupancy_max"] = int(occ.rules.max())
    rp = f"{L}/ctrl_resources.csv"
    if os.path.exists(rp) and os.path.getsize(rp) > 0:
        r = pd.read_csv(rp)
        if len(r):
            M["ctrl_cpu_pct_mean"], M["ctrl_cpu_pct_max"] = float(r.ctrl_cpu_pct.mean()), float(r.ctrl_cpu_pct.max())
            M["ctrl_rss_MB_max"] = float(r.ctrl_rss_MB.max())
            M["bmv2_cpu_pct_mean"] = float(r.bmv2_cpu_pct[r.bmv2_cpu_pct >= 0].mean())
            M["host_cpu_pct_attack_mean"] = float(r[r.t.between(t_att0, t_att1)].host_cpu_pct.mean())
    for f in man.get("force_fp_flows", []):   # FP recovery
        si, di = ip2int(f["src"]), ip2int(f["dst"])
        g = ev[(ev.src == si) & (ev.dst == di)].sort_values("t") if len(ev) else ev
        ins = g[g.event == "installed"].t.min() if len(g) else None
        exp_t = g[g.event.isin(["expired_ttl", "revoked"])].t
        exp_t = exp_t[exp_t > ins].min() if ins and len(exp_t) else None
        v = vic[vic.src == f["src"]]
        if ins and exp_t:
            M["fp_rule_lifetime_s"] = float(exp_t - ins)
            M["fp_benign_pkts_received_while_blocked"] = int(v.t.between(ins + 0.05, exp_t).sum())
            after = v[v.t > exp_t]
            M["fp_recovery_s"] = float(after.t.min() - exp_t) if len(after) else None
    json.dump(M, open(f"{run}/metrics.json", "w"), indent=2, default=float)
    return M


def main():
    base = sys.argv[1]
    runs = sorted(glob.glob(f"{base}/*/rep*")) or sorted(glob.glob(f"{base}/rep*"))
    rows = []
    for r in runs:
        try:
            m = analyze(r); m.pop("attackers"); rows.append(m); print("ok", r)
        except Exception as e:
            print("FAILED", r, e)
    if not rows: return
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number").columns
    g = df.groupby("scenario")[num]
    out = g.mean().round(4).add_suffix("_mean").join(g.std().round(4).add_suffix("_std")).join(g.size().rename("n_reps"))
    root = base if glob.glob(f"{base}/*/rep*") else os.path.dirname(base)
    out.to_csv(f"{root}/summary_by_scenario.csv")
    df.to_csv(f"{root}/all_runs.csv", index=False)
    key = ["n_reps", "f1_mean", "f1_std", "fpr_mean", "auc_mean", "eer_mean", "att_detection_delay_s_mean_mean",
           "att_onset_to_block_s_mean_mean", "att_leaked_pkts_before_block_mean_mean", "tcp_mbps_pre_mean",
           "tcp_mbps_attack_mean", "bottleneck_util_pct_attack_mean_mean", "inf_ms_p99_mean", "digest_lag_ms_p99_mean"]
    key = [k for k in key if k in out.columns]
    with open(f"{root}/summary_by_scenario.md", "w") as f:
        try:
            f.write(out[key].to_markdown())
        except ImportError:            # 'tabulate' not installed: plain text instead
            f.write(out[key].to_string())
    print(out[key].to_string())


if __name__ == "__main__":
    main()
