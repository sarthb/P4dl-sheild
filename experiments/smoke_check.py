#!/usr/bin/env python3
"""smoke_check.py — after `run_remaining.sh smoke`: does every new scenario do what it should?"""
import sys, os, glob, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import analyze_run as A

CHECKS = {   # scenario: [(metric, test, what it means)]
    "S9_ctrl_flood_mitigate": [("src_rules_installed", lambda v: v and v >= 1, "flow-churn guard installed a source rule")],
    "S10_hybrid_tagdrop": [("tcp_mbps_pre", lambda v: v is not None, "TCP measured (low value = tag drops benign TCP: a finding)")],
    "S11_table_full": [("flow_guard_occupancy_max", lambda v: v is not None and v <= 50, "rule table capped at 50"),
                       ("rules_evicted_lru", lambda v: v and v > 0, "LRU eviction happened")],
    "S12_cpu_stress": [("host_cpu_pct_attack_mean", lambda v: v and v > 60, "CPU stress active during attack")],
    "S13_attacker_host_legit": [("tcp_h22_mbps_attack", lambda v: v is not None, "legit TCP from attacker host measured"),
                                ("benign_benign_legit_h22_loss_pct", lambda v: v is not None, "legit UDP from attacker host measured")],
    "S14_telemetry_off": [("note", lambda v: v is not None, "digests off, forwarding-only run"),
                          ("tcp_mbps_pre", lambda v: v is not None, "TCP measured")],
    "S5_load_8kpps": [("digest_rate_attack", lambda v: v is not None, "digest rate measured")],
}
ok = True
found = {os.path.basename(os.path.dirname(os.path.dirname(r))) for r in glob.glob(f"{sys.argv[1]}/*/rep*/timeline.json")}
for sc in CHECKS:
    if sc not in found:
        print(f"FAIL  {sc:26s} MISSING: run did not finish (see the runner output above)"); ok = False
for run in sorted(glob.glob(f"{sys.argv[1]}/*/rep*")):
    sc = os.path.basename(os.path.dirname(run))
    try:
        m = A.analyze(run)
    except Exception as e:
        print(f"FAIL  {sc:26s} analysis error: {e}"); ok = False; continue
    base = f"digests={m.get('digests')} benign_flagged={m.get('flows_benign_flagged')} " \
           f"benign_BLOCKED={m.get('flows_benign_blocked')} " \
           f"attack_detected={m.get('flows_attack_detected')} onset_to_block={m.get('att_onset_to_block_s_mean')}"
    print(f"      {sc:26s} {base}")
    for k, test, what in CHECKS.get(sc, []):
        v = m.get(k); good = bool(test(v)); ok &= good
        print(f"{'PASS' if good else 'WARN'}    {k} = {v}   ({what})")
    if m.get("warning_tcp"): print(f"WARN    {m['warning_tcp']}")
print("\nALL CHECKS PASSED - start the full run" if ok else "\nSome checks did not pass - paste this output")
