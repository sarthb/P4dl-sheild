#!/usr/bin/env python3
"""
policy_manager_v2.py — REVISION rule management for MyIngress.flow_guard
(Reviewer 1 #6, Reviewer 3 #25, #26)

  * Rule lifetime: every drop rule has a hard TTL (rule_ttl_s). Once a rule is installed,
    the dropped flow sends no more digests, so an idle timeout based on digests is impossible.
    After the TTL the rule is deleted; if the attack is still running it is re-detected and
    re-installed. Each re-detection gap is logged, so leakage per TTL cycle can be reported.
  * Revocation: rules can be removed early (false-positive recovery) by writing
    "src dst proto dport" lines to <log_dir>/revoke_requests.txt (polled every 0.5 s),
    or by calling revoke(...).
  * Capacity: the controller keeps at most `capacity` rules per switch; when full, the
    least-recently-(re)confirmed rule is evicted (LRU). Occupancy is logged every second.
  * Timing: install() returns (t_sent, t_ack) around the P4Runtime Write RPC (the RPC is
    synchronous, so t_ack = time the switch acknowledged the write).
All events go to <log_dir>/rule_events.csv and <log_dir>/rule_occupancy.csv.
"""
import os, csv, time, threading, ipaddress
from collections import OrderedDict

_cfg = {"ttl_s": 30, "capacity": 1024, "log_dir": "logs"}
_rules = {}          # sw_name -> OrderedDict(key -> {"entry", "installed", "conn"})
_lock = threading.Lock()
_ev = None
_occ = None


def _w(path, hdr):
    new = not os.path.exists(path)
    f = open(path, "a", newline=""); w = csv.writer(f)
    if new: w.writerow(hdr); f.flush()
    return f, w


def _event(kind, sw, key, extra=""):
    f, w = _ev
    w.writerow([time.time(), kind, sw, *key, extra]); f.flush()


def configure(ttl_s=30, capacity=1024, log_dir="logs"):
    global _ev, _occ
    _cfg.update(ttl_s=ttl_s, capacity=capacity, log_dir=log_dir)
    os.makedirs(log_dir, exist_ok=True)
    _ev = _w(f"{log_dir}/rule_events.csv", ["t", "event", "switch", "src", "dst", "proto", "dport", "info"])
    _occ = _w(f"{log_dir}/rule_occupancy.csv", ["t", "switch", "rules"])
    threading.Thread(target=_housekeeping, daemon=True).start()


def _build(conn, s, d, p, dp):
    return conn.p4info_helper.buildTableEntry(
        table_name="MyIngress.flow_guard",
        match_fields={"hdr.ipv4.srcAddr": s, "hdr.ipv4.dstAddr": d,
                      "hdr.ipv4.protocol": p, "meta.feat.dport": dp},
        action_name="MyIngress.drop", action_params={})


def _delete(sw, key, why):
    r = _rules[sw].pop(key)
    try:
        r["conn"].DeleteTableEntry(r["entry"])
        _event(why, sw, key, f"age={time.time() - r['installed']:.2f}")
    except Exception as e:
        _event("delete_failed", sw, key, str(e))


def install(conn, s, d, p, dp):
    """Returns (t_sent, t_ack, action)."""
    sw = getattr(conn, "name", "unknown"); key = (s, d, p, dp)
    with _lock:
        tbl = _rules.setdefault(sw, OrderedDict())
        if key in tbl:
            tbl.move_to_end(key)
            return None, None, "exists"
        while len(tbl) >= _cfg["capacity"]:
            _delete(sw, next(iter(tbl)), "evicted_lru")
        entry = _build(conn, s, d, p, dp)
        t_sent = time.time()
        try:
            conn.WriteTableEntry(entry)
        except Exception as e:
            _event("insert_failed", sw, key, str(e)[:120])
            return t_sent, None, "failed"
        t_ack = time.time()
        tbl[key] = {"entry": entry, "installed": t_ack, "conn": conn}
        _event("installed", sw, key, f"rpc_ms={(t_ack - t_sent) * 1e3:.2f}")
        return t_sent, t_ack, "installed"


def install_src(conn, s):
    """Source-level drop in MyIngress.src_guard (key shown in logs as (src, 0, 0, -1))."""
    sw = getattr(conn, "name", "unknown"); key = (s, 0, 0, -1)
    with _lock:
        tbl = _rules.setdefault(sw, OrderedDict())
        if key in tbl:
            tbl.move_to_end(key); return None, None, "src_exists"
        entry = conn.p4info_helper.buildTableEntry(
            table_name="MyIngress.src_guard", match_fields={"hdr.ipv4.srcAddr": s},
            action_name="MyIngress.drop", action_params={})
        t_sent = time.time()
        try:
            conn.WriteTableEntry(entry)
        except Exception as e:
            _event("src_insert_failed", sw, key, str(e)[:120]); return t_sent, None, "failed"
        t_ack = time.time()
        tbl[key] = {"entry": entry, "installed": t_ack, "conn": conn}
        _event("src_installed", sw, key, f"rpc_ms={(t_ack - t_sent) * 1e3:.2f}")
        return t_sent, t_ack, "src_installed"


def revoke(s, d, p, dp, why="revoked"):
    key = (s, d, p, dp)
    with _lock:
        for sw, tbl in _rules.items():
            if key in tbl: _delete(sw, key, why)


def _housekeeping():
    path = f"{_cfg['log_dir']}/revoke_requests.txt"; last_occ = 0
    while True:
        now = time.time()
        with _lock:
            for sw, tbl in _rules.items():
                for key in [k for k, r in tbl.items() if now - r["installed"] > _cfg["ttl_s"]]:
                    _delete(sw, key, "expired_ttl")
        if os.path.exists(path):
            lines = open(path).read().split("\n"); os.remove(path)
            for ln in lines:
                parts = ln.split()
                if len(parts) == 4:
                    s, d = (int(ipaddress.IPv4Address(x)) if "." in x else int(x) for x in parts[:2])
                    revoke(s, d, int(parts[2]), int(parts[3]))
        if now - last_occ >= 1.0:
            f, w = _occ
            with _lock:
                for sw, tbl in _rules.items(): w.writerow([now, sw, len(tbl)])
            f.flush(); last_occ = now
        time.sleep(0.5)
