#!/usr/bin/env python3
"""
digest_handler_v2.py — REVISION drop-in replacement for digest_handler.py

What changes vs. digest_handler.py (and which reviewer comment it answers):
  * Digests are queued and processed in a worker thread, so DNN inference and
    P4Runtime writes no longer block the gRPC stream thread. Queue depth and
    queueing delay are logged (R1 #4, R3 #26).
  * Every digest gets a full timing record in logs/timing.csv: switch ingress
    timestamp -> controller receive -> dequeue -> features -> inference ->
    rule write sent -> rule write acknowledged (R1 #2, R3 #24, R4).
  * Ground-truth labels come ONLY from an experiment manifest (JSON) and are never
    used as features or for mitigation decisions (R3 #15). The old hard-coded
    `dport == 5001` mitigation condition is removed (R3 #25).
  * System telemetry is sampled live by a controller-side thread (the old code read a
    stale logs/system_profile.csv, so cpu/mem/tx/rx were constant). Absolute values
    and deltas relative to a pre-attack baseline are logged (R1 #7, R3 #20).
  * Supports the revised P4 digest (ingress_ts, burst_cnt) and the old 8-field digest.

Configuration: environment variable REV_MANIFEST=/path/to/manifest.json, e.g.
{
  "run_name": "S2_distributed_4att",
  "mode": "hybrid",                 # hybrid | dnn_only | tag_only | detect_only
  "mitigation": true,
  "attack_flows": [{"src":"10.0.2.2","dst":"10.0.4.1","proto":17,"dport":5001}],
  "force_fp_flows": [],             # benign flows to force-classify as malicious (FP-recovery test)
  "rule_ttl_s": 30, "flow_guard_capacity": 1024,
  "model_path": "models/ddos_dnn_live.pth", "threshold": 0.70,
  "baseline_seconds": 10
}
"""
import os, csv, json, time, queue, threading, ipaddress, signal, atexit, sys

import psutil

from dnn_model import load_model, predict_flow
import policy_manager_v2 as pm

# --------------------------------------------------------------------- config
CFG = {"run_name": "unnamed", "mode": "hybrid", "mitigation": False, "attack_flows": [],
       "force_fp_flows": [], "rule_ttl_s": 30, "flow_guard_capacity": 1024,
       "model_path": None, "threshold": None, "baseline_seconds": 10, "log_dir": "logs",
       "churn_guard": True, "churn_flows_per_s": 100, "flow_idle_s": 10, "telemetry": True,
       "block_policy": None,   # None -> models/block_policy.json if present, else block on first positive
       "digest_mode": "packet", "flow_window_s": 0.05}   # REV-FLOW: "flow" = per-flow windowed digests
if os.environ.get("REV_MANIFEST"):
    CFG.update(json.load(open(os.environ["REV_MANIFEST"])))
LOG = CFG["log_dir"]; os.makedirs(LOG, exist_ok=True)


def _ip(x):
    return int(ipaddress.IPv4Address(x)) if isinstance(x, str) else int(x)


def _flowset(lst):
    return {(_ip(f["src"]), _ip(f["dst"]), int(f["proto"]), int(f.get("dport", -1))) for f in lst}


ATTACK = _flowset(CFG["attack_flows"])
FORCE_FP = _flowset(CFG["force_fp_flows"])


def _match(fs, s, d, p, dp):
    return (s, d, p, dp) in fs or (s, d, p, -1) in fs


# ----------------------------------------------------------- live telemetry
class Telemetry:
    """Controller-host CPU/mem + machine-wide net counters, sampled every 0.5 s.
    NOTE for the paper: in Mininet all hosts share one kernel, so CPU/mem are machine-wide.
    They are therefore *controller-host* features, not per-flow or per-host features."""
    def __init__(self, interval=0.5):
        self.v = [0.0, 0.0, 0.0, 0.0]; self.base = None; self.t0 = time.time()
        self._samples = []
        psutil.cpu_percent(None)
        threading.Thread(target=self._loop, args=(interval,), daemon=True).start()

    def _loop(self, iv):
        me = psutil.Process(); me.cpu_percent(None)
        res = _CSV(f"{LOG}/ctrl_resources.csv", ["t", "ctrl_cpu_pct", "ctrl_rss_MB", "bmv2_cpu_pct", "host_cpu_pct"])
        bm, last = [], 0.0
        while True:
            if time.time() - last >= 1.0:          # resource usage once per second (R3 #26)
                last = time.time()
                if not bm:
                    bm = [p for p in psutil.process_iter(["name"]) if p.info["name"] == "simple_switch_grpc"]
                    for p in bm: p.cpu_percent(None)
                try:
                    bcpu = sum(p.cpu_percent(None) for p in bm)
                except Exception:
                    bm, bcpu = [], -1
                res.row([last, me.cpu_percent(None), me.memory_info().rss / 1e6, bcpu, psutil.cpu_percent(None)])
                res.f.flush()
            n = psutil.net_io_counters()
            v = [psutil.cpu_percent(None) / 100.0, psutil.virtual_memory().percent / 100.0,
                 n.bytes_sent / 1e6, n.bytes_recv / 1e6]
            self.v = v
            if self.base is None:
                self._samples.append(v)
                if time.time() - self.t0 >= CFG["baseline_seconds"]:
                    k = len(self._samples)
                    self.base = [sum(s[i] for s in self._samples) / k for i in range(4)]
            time.sleep(iv)

    def get(self):
        v = list(self.v)
        b = self.base or v
        return v, [v[i] - b[i] for i in range(4)]


# ---------------------------------------------------------------- state
_q = queue.Queue()
_flow = {}
_churn = {}                 # (device, src) -> list of new-flow timestamps in the last second
_last_purge = [0.0]
_stats = {"flows_expired": 0, "churn_triggers": 0}
_forced_done = set()        # FP-recovery test: flows whose forced false positive already produced a rule
_pos = {}                   # (src,dst,proto,dport) -> state of recent positive digests (all switches)
POLICY = {"K": 1, "window_s": 1.0, "min_bins": 1, "recur_window_s": 1.0, "gap_s": 0.3, "sustain_s": 0.5}


def _policy_says_block(key, now):
    """Persistence policy (chosen on validation by training/policy_replay.py).
    Block a flow when K positives fall in the last W seconds AND
      (positives formed >= B separate episodes within the last R seconds  OR  the current episode has lasted >= L s).
    An episode is a run of positives with no gap longer than G seconds. Fixed design constants: G = 0.3 s, L = 0.5 s
    (an LDoS attack recurs; a flood persists; one benign burst does neither)."""
    st = _pos.get(key)
    if st is None:
        st = _pos[key] = {"q": [], "eps": [], "last": None}
    q, eps = st["q"], st["eps"]
    if st["last"] is None or now - st["last"] > POLICY["gap_s"]:
        eps.append(now)                                   # a new episode starts
    st["last"] = now
    q.append(now)
    while q and now - q[0] > POLICY["window_s"]: q.pop(0)
    while eps and now - eps[0] > POLICY["recur_window_s"] and len(eps) > 1: eps.pop(0)
    n_eps = sum(1 for e in eps if now - e <= POLICY["recur_window_s"])
    sustained = now - eps[-1] >= POLICY["sustain_s"]
    return len(q) >= POLICY["K"] and (n_eps >= POLICY["min_bins"] or sustained)
_lock = threading.Lock()
_model = None
_tele = None
_dev = None
_started = False
TIMING_HDR = ["run", "device_id", "src_ip", "dst_ip", "proto", "dport", "true_label", "tag", "burst_cnt",
              "prob", "pred", "sw_ingress_ts_us", "t_ctrl_rx", "t_dequeue", "t_features", "t_inference",
              "t_rule_sent", "t_rule_ack", "rule_action", "queue_depth_at_rx", "flow_table_size", "new_flows_per_s"]
LIVE_HDR = ["timestamp", "device_id", "src_ip", "dst_ip", "proto", "sport", "dport", "tag", "pkt_len",
            "flow_packet_count", "flow_byte_count", "flow_duration", "flow_packets_per_second",
            "inter_arrival_time", "cpu", "mem", "tx_MB", "rx_MB", "Label",
            # extra REV columns (appended, so the old training script still works)
            "burst_cnt", "d_cpu", "d_mem", "d_tx_MB", "d_rx_MB", "run"]


class _CSV:
    def __init__(self, path, hdr):
        new = not os.path.exists(path)
        self.f = open(path, "a", newline=""); self.w = csv.writer(self.f)
        if new: self.w.writerow(hdr)
        self.n = 0

    def row(self, r):
        self.w.writerow(r); self.n += 1
        if self.n % 200 == 0: self.f.flush()


_timing = _CSV(f"{LOG}/timing.csv", TIMING_HDR)
_live = _CSV(f"{LOG}/live_dataset.csv", LIVE_HDR)
_timing.f.flush(); _live.f.flush()          # headers on disk immediately


def _flush_all():
    if _flowlog is not None:
        try: _flowlog.f.flush()
        except Exception: pass
    for c in (_timing, _live):
        try: c.f.flush()
        except Exception: pass


def _flusher():                             # nothing older than 1 s is ever lost
    while True:
        time.sleep(1.0); _flush_all()


def _on_term(signum, frame):                # runner stops the controller with SIGTERM
    _flush_all()
    print(f"[REV] stopping (signal {signum}); digests logged: timing={_timing.n} live={_live.n}; "
          f"flows_expired={_stats['flows_expired']} churn_triggers={_stats['churn_triggers']}", flush=True)
    os._exit(0)


atexit.register(_flush_all)
try:
    signal.signal(signal.SIGTERM, _on_term)
except ValueError:                          # not in main thread: rely on the flusher
    pass
threading.Thread(target=_flusher, daemon=True).start()


def _p4int(x):
    if x.HasField("bitstring"): return int.from_bytes(x.bitstring, "big")
    raise ValueError(x)


def _purge(now):
    """Idle-flow expiry (R3 #19): forget flow state not seen for flow_idle_s seconds."""
    if now - _last_purge[0] < 2.0: return
    _last_purge[0] = now; idle = CFG["flow_idle_s"]
    dead = [k for k, st in _flow.items() if now - st["last"] > idle]
    for k in dead: del _flow[k]
    _stats["flows_expired"] += len(dead)
    for k in [k for k, v in _churn.items() if not v or now - v[-1] > 1.0]: del _churn[k]


def _new_flow_rate(dev, s, now):
    q = _churn.setdefault((dev, s), [])
    q.append(now)
    while q and now - q[0] > 1.0: q.pop(0)
    return len(q)


def _flow_features(dev, s, d, p, dp, plen, now):
    k = (dev, s, d, p, dp)
    st = _flow.get(k)
    if st is None:
        st = _flow[k] = {"first": now, "last": now, "n": 0, "b": 0, "new": True}
    else:
        st["new"] = False
    iat = now - st["last"]; st["last"] = now; st["n"] += 1; st["b"] += plen
    dur = now - st["first"]
    return st["n"], st["b"], dur, (st["n"] / dur if dur > 0 else 0.0), iat


FLOW_HDR = ["t_rx", "device_id", "src_ip", "dst_ip", "proto", "dport", "msg_type", "label", "tag_max", "win_pkts",
            "win_bytes", "win_first_us", "win_last_us", "flow_first_us", "cum_pkts", "cum_bytes", "ingress_ts_us",
            "prob", "pred", "t_dequeue", "t_inference", "t_rule_sent", "t_rule_ack", "rule_action", "queue_depth_at_rx",
            "new_flows_per_s"]
_flowlog = None
FLOW_FEATS = ["protocol", "tag", "pkt_len", "flow_pkts", "flow_bytes", "flow_dur", "flow_pps", "iat", "win_pkts"]


def flow_summary_features(p, tagm, wp, wb, wf, wl, ff, cp, cb, W):
    """Features of one per-flow window summary (same definition as training/perflow_sim.py and train_flow.py)."""
    dur = max(wl - ff, 0) / 1e6
    return {"protocol": p, "tag": tagm, "pkt_len": wb / max(wp, 1), "flow_pkts": cp, "flow_bytes": cb,
            "flow_dur": dur, "flow_pps": cp / max(dur, 1e-3),
            "iat": ((wl - wf) / 1e6 / (wp - 1)) if wp > 1 else W, "win_pkts": wp}


def _handle_flow_msg(t_rx, qd, sw_conn, values, t_deq):
    global _flowlog
    if _flowlog is None:
        _flowlog = _CSV(f"{LOG}/flow_msgs.csv", FLOW_HDR)
    dev, s, d, p, sp, dp, tagm, mtype, ts_us, wp, wb, wf, wl, ff, cp, cb = values[:16]
    y = 1 if _match(ATTACK, s, d, p, dp) else 0
    churn = _new_flow_rate(dev, s, t_rx) if mtype == 0 else len(_churn.get((dev, s), []))
    _purge(t_rx)
    prob, pred = -1.0, 0
    if mtype == 1 and CFG["mode"] != "tag_only" and _model is not None:
        fv = flow_summary_features(p, tagm, wp, wb, wf, wl, ff, cp, cb, CFG["flow_window_s"])
        pred, prob = predict_flow(_model, _dev, [fv[f] for f in (_model.get("features") or FLOW_FEATS)])
    if mtype == 1 and _match(FORCE_FP, s, d, p, dp) and (s, d, p, dp) not in _forced_done:   # FP-recovery test
        pred = 1
    t_inf = time.time()
    t_sent = t_ack = None; action = "newflow" if mtype == 0 else ""
    active = CFG["mitigation"] and CFG["mode"] != "detect_only"
    if active and CFG["churn_guard"] and churn >= CFG["churn_flows_per_s"]:
        t_sent, t_ack, action = pm.install_src(sw_conn, s)
        if action == "src_installed": _stats["churn_triggers"] += 1
    elif pred == 1 and active:
        if _policy_says_block((s, d, p, dp), t_rx):
            t_sent, t_ack, action = pm.install(sw_conn, s, d, p, dp)
            if action == "installed" and _match(FORCE_FP, s, d, p, dp):
                _forced_done.add((s, d, p, dp))
        else:
            action = "pending"
    _flowlog.row([t_rx, dev, s, d, p, dp, mtype, y, tagm, wp, wb, wf, wl, ff, cp, cb, ts_us, round(prob, 6), pred,
                  t_deq, t_inf, t_sent, t_ack, action, qd, churn])
    # the generic timing log keeps one row per controller message, so analyze_run.py still works
    _timing.row([CFG["run_name"], dev, s, d, p, dp, y, tagm, wp, round(prob, 6), pred, ts_us,
                 t_rx, t_deq, t_deq, t_inf, t_sent, t_ack, action, qd, 0, churn])


def _worker():
    while True:
        t_rx, qd, sw_conn, values = _q.get()
        t_deq = time.time()
        if len(values) >= 16:                    # REV-FLOW: per-flow windowed digest
            try:
                _handle_flow_msg(t_rx, qd, sw_conn, values, t_deq)
            except Exception as e:
                print("[REV] flow msg error:", e, flush=True)
            continue
        dev, s, d, p, sp, dp, tag, plen = values[:8]
        ts_us = values[8] if len(values) > 8 else -1
        bcnt = values[9] if len(values) > 9 else -1
        n, b, dur, pps, iat = _flow_features(dev, s, d, p, dp, plen, t_rx)
        churn = _new_flow_rate(dev, s, t_rx) if n == 1 else len(_churn.get((dev, s), []))
        _purge(t_rx)
        (cpu, mem, tx, rx), (dcpu, dmem, dtx, drx) = _tele.get()
        y = 1 if _match(ATTACK, s, d, p, dp) else 0
        t_feat = time.time()

        mode = CFG["mode"]
        if mode == "tag_only":                   # in-switch-only baseline: decision = tag
            prob = float(tag); pred = int(tag)
        else:
            fv = {"proto": p, "tag": tag, "pkt_len": plen, "flow_packet_count": n, "flow_byte_count": b,
                  "flow_duration": dur, "flow_packets_per_second": pps, "inter_arrival_time": iat,
                  "cpu": cpu, "mem": mem, "tx_MB": tx, "rx_MB": rx,
                  "d_cpu": dcpu, "d_mem": dmem, "d_tx_MB": dtx, "d_rx_MB": drx}
            feats = _model.get("features") or ["proto", "tag", "pkt_len", "flow_packet_count", "flow_byte_count",
                                               "flow_duration", "flow_packets_per_second", "inter_arrival_time",
                                               "cpu", "mem", "tx_MB", "rx_MB"]
            pred, prob = predict_flow(_model, _dev, [fv[f] for f in feats])
        if _match(FORCE_FP, s, d, p, dp) and (s, d, p, dp) not in _forced_done:   # FP-recovery test: one-shot
            pred = 1
        t_inf = time.time()

        t_sent = t_ack = None; action = ""
        active = CFG["mitigation"] and mode != "detect_only"
        if active and CFG["churn_guard"] and mode != "tag_only" and churn >= CFG["churn_flows_per_s"]:
            # control-plane defence: one source creating >= N new flows/s at this switch
            t_sent, t_ack, action = pm.install_src(sw_conn, s)
            if action == "src_installed": _stats["churn_triggers"] += 1
        elif pred == 1 and active:
            if _policy_says_block((s, d, p, dp), t_rx):
                t_sent, t_ack, action = pm.install(sw_conn, s, d, p, dp)
                if action == "installed" and _match(FORCE_FP, s, d, p, dp):
                    _forced_done.add((s, d, p, dp))
            else:
                action = "pending"          # positive, but not persistent enough yet
        _timing.row([CFG["run_name"], dev, s, d, p, dp, y, tag, bcnt, round(prob, 6), pred, ts_us,
                     t_rx, t_deq, t_feat, t_inf, t_sent, t_ack, action, qd, len(_flow), churn])
        _live.row([t_rx, dev, s, d, p, sp, dp, tag, plen, n, b, dur, pps, iat, cpu, mem, tx, rx, y,
                   bcnt, dcpu, dmem, dtx, drx, CFG["run_name"]])


def start(device):
    """Call once from the controller after all switches are connected."""
    global _model, _tele, _dev, _started
    if _started: return
    _dev = device
    mp = CFG["model_path"] or "models/ddos_dnn_live.pth"
    if CFG["mode"] == "detect_only" and CFG["digest_mode"] == "flow" and not os.path.exists(mp):
        print(f"[REV] no model at {mp}: collecting per-flow training data only (no classification)")
        _model = None
    elif CFG["mode"] != "tag_only":
        _model = load_model(mp, device=device)
        sp = mp.replace(".pth", "_scaler.pkl")          # scaler exported by train_eval_v2.py
        if os.path.exists(sp):
            import joblib; _model["scaler"] = joblib.load(sp)
        if CFG["threshold"] is not None:
            _model["threshold"] = float(CFG["threshold"])
        else:                                           # threshold chosen on validation at export time
            import torch
            ck = torch.load(mp, map_location="cpu")
            if isinstance(ck, dict) and "threshold" in ck: _model["threshold"] = float(ck["threshold"])
        try:
            import torch
            ck = torch.load(mp, map_location="cpu")
            if isinstance(ck, dict) and ck.get("features"):
                _model["features"] = list(ck["features"])
        except ImportError:
            pass
        print(f"[REV] model features: {_model.get('features', 'legacy 12')}")
        bp = CFG["block_policy"]
        pfile = "models/block_policy_flow.json" if CFG["digest_mode"] == "flow" else "models/block_policy.json"
        if bp is None and os.path.exists(pfile):
            bp = json.load(open(pfile))
            if os.path.abspath(bp.get("model", mp)) != os.path.abspath(mp):
                print(f"[REV] models/block_policy.json is for {bp.get('model')}, not {mp}: ignored"); bp = None
        if bp:
            for k in POLICY: POLICY[k] = type(POLICY[k])(bp[k])
            if CFG["threshold"] is None and "threshold" in bp: _model["threshold"] = float(bp["threshold"])
        print(f"[REV] block policy: {POLICY} threshold={_model['threshold']}")
        print(f"[REV] model={mp} threshold={_model['threshold']}")
    _tele = Telemetry()
    pm.configure(ttl_s=CFG["rule_ttl_s"], capacity=CFG["flow_guard_capacity"], log_dir=LOG)
    threading.Thread(target=_worker, daemon=True).start()
    _started = True
    print(f"[REV] run={CFG['run_name']} mode={CFG['mode']} mitigation={CFG['mitigation']} "
          f"attack_flows={len(ATTACK)} ttl={CFG['rule_ttl_s']}s capacity={CFG['flow_guard_capacity']}")


def handle_digest(digest_msg, sw_conn, model=None, device=None):
    """Same signature as the old handler. Runs in the gRPC stream thread: only parse + enqueue."""
    t_rx = time.time()
    if not _started: start(device)
    qd = _q.qsize()
    for data in digest_msg.data:
        try:
            _q.put((t_rx, qd, sw_conn, [_p4int(v) for v in data.struct.members]))
        except Exception as e:
            print("[REV] bad digest:", e)
