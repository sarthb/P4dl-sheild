#!/usr/bin/env python3
"""
run_scenarios.py — automated revision experiments on the Mininet/BMv2 testbed.

Run from the repo root as root:
    sudo -E python3 revision/experiments/run_scenarios.py --scenarios S1_single_A2 S2_distributed --reps 5
    sudo -E python3 revision/experiments/run_scenarios.py --list

For every (scenario, repetition) it: builds a fresh topology with the REV P4 program, starts
the REV controller with a manifest (labels + mode), starts benign traffic, starts the
attack(s), then records everything into results_rev/<scenario>/rep<k>/ for analysis/analyze_run.py.

Hosts: h11 ping | h12 TCP iperf (victim of LDoS) | h21 small-packet benign UDP to the victim port
       h42 bursty benign UDP ("hard negative" that looks like pulses) | attackers: h22, h31, h32, h21*
Victim h41: iperf TCP server :5002, UDP sink :5001 (benign and attack share port 5001).
"""
import argparse, json, os, sys, time, subprocess, random, threading

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from mininet.log import setLogLevel, info          # noqa: E402
import topo                                         # noqa: E402  (repo's topo.py)

PULSE = os.path.join(ROOT, "revision", "experiments", "udp_pulse.py")
VICTIM = "10.0.4.1"
IP = {"h11": "10.0.1.1", "h12": "10.0.1.2", "h21": "10.0.2.1", "h22": "10.0.2.2",
      "h31": "10.0.3.1", "h32": "10.0.3.2", "h41": "10.0.4.1", "h42": "10.0.4.2"}

# ------------------------------------------------------------------ traffic building blocks
BENIGN_MIX = [  # always on, from t=0 to end (all share the victim; UDP ones use the attack port 5001)
    dict(host="h11", kind="ping"),
    dict(host="h12", kind="tcp"),
    dict(host="h21", kind="pulse", name="benign_voip", dport=5001, burst_ms=1000, period_ms=1000,
         rate_pps=50, size=144),                                   # small-packet constant UDP
]
HARD_BENIGN = [  # benign traffic whose shape overlaps the attack (Reviewer 3 #14)
    dict(host="h42", kind="pulse", name="benign_bursty", dport=5001, burst_ms=150, period_ms=1000,
         rate_pps=400, size=1000, jitter_ms=150),                  # bursty video-like chunks
]


def atk(host, burst, period=1000, pps=2000, size=1000, phase=0, sport=0, jitter=0, dport=5001):
    return dict(host=host, kind="pulse", name=f"attack_{host}", attack=True, dport=dport,
                burst_ms=burst, period_ms=period, rate_pps=pps, size=size, phase_ms=phase,
                sport=sport, jitter_ms=jitter)


# ------------------------------------------------------------------ scenarios
# rate 2000 pps x 1028 B = 16.4 Mb/s during a burst  > 10 Mb/s bottleneck (Shrew-style pulse)
SCEN = {
    # ---- TRAINING-DATA collection (detection only). Use ONLY these for training/validation;
    # ---- every S* scenario below is test-only. The S1 burst lengths (200/250/300) are kept out
    # ---- of training on purpose (train uses 150 and 350 ms) so the test cases are unseen.
    "T0_benign_train":   dict(traffic=BENIGN_MIX + HARD_BENIGN, attack_s=60, mode="detect_only"),
    "T1_attack_train_b150": dict(traffic=BENIGN_MIX + HARD_BENIGN + [atk("h22", 150)], mode="detect_only"),
    "T1_attack_train_b350": dict(traffic=BENIGN_MIX + HARD_BENIGN + [atk("h31", 350, 1000, 1500)],
                                 mode="detect_only"),
    # ---- TEST scenarios
    "S0_benign_only":    dict(traffic=BENIGN_MIX + HARD_BENIGN, attack_s=0, mitigation=True),
    # single attacker, the paper's three cases but with a real pulse rate
    "S1_single_A1":       dict(traffic=BENIGN_MIX + HARD_BENIGN + [atk("h22", 200)], mitigation=True),
    "S1_single_A2":       dict(traffic=BENIGN_MIX + HARD_BENIGN + [atk("h22", 250)], mitigation=True),
    "S1_single_A3":       dict(traffic=BENIGN_MIX + HARD_BENIGN + [atk("h22", 300)], mitigation=True),
    "S1_single_A2_detect": dict(traffic=BENIGN_MIX + HARD_BENIGN + [atk("h22", 250)], mitigation=False,
                                mode="detect_only"),
    # distributed: 3 attackers, different periods and phases, each below the single-attacker rate
    "S2_distributed":     dict(traffic=BENIGN_MIX + HARD_BENIGN + [
                                atk("h22", 200, 1000, 800, phase=0),
                                atk("h31", 250, 1200, 800, phase=330),
                                atk("h32", 300, 1500, 800, phase=660)], mitigation=True),
    "S2_distributed_detect": dict(traffic=BENIGN_MIX + HARD_BENIGN + [
                                atk("h22", 200, 1000, 800, phase=0),
                                atk("h31", 250, 1200, 800, phase=330),
                                atk("h32", 300, 1500, 800, phase=660)], mitigation=False, mode="detect_only"),
    # unseen variants (Reviewer 1 #3) — detection only, so every packet is scored
    "S3_short_burst_50ms":  dict(traffic=BENIGN_MIX + HARD_BENIGN + [atk("h22", 50)], mode="detect_only"),
    "S3_long_burst_500ms":  dict(traffic=BENIGN_MIX + HARD_BENIGN + [atk("h22", 500)], mode="detect_only"),
    "S3_period_500ms":      dict(traffic=BENIGN_MIX + HARD_BENIGN + [atk("h22", 100, 500)], mode="detect_only"),
    "S3_period_2000ms":     dict(traffic=BENIGN_MIX + HARD_BENIGN + [atk("h22", 300, 2000)], mode="detect_only"),
    "S3_tiny_pkts_28B":     dict(traffic=BENIGN_MIX + HARD_BENIGN + [atk("h22", 250, size=0)], mode="detect_only"),
    "S3_random_sport":      dict(traffic=BENIGN_MIX + HARD_BENIGN + [atk("h22", 250, sport=-1)], mode="detect_only"),
    # mimicry: same size and similar shape as the bursty benign flow, just stronger
    "S4_mimicry":           dict(traffic=BENIGN_MIX + HARD_BENIGN + [atk("h22", 150, 1000, 1200, 1000, jitter=150)],
                                 mode="detect_only"),
    # controller load (Reviewer 1 #4, Reviewer 3 #26): heavy benign background, detection only
    "S5_load_2kpps":  dict(traffic=BENIGN_MIX + [dict(host="h12", kind="pulse", name="bg", dport=5004,
                           burst_ms=1000, period_ms=1000, rate_pps=2000, size=200)] + [atk("h22", 250)],
                           mode="detect_only"),
    "S5_load_5kpps":  dict(traffic=BENIGN_MIX + [dict(host="h12", kind="pulse", name="bg", dport=5004,
                           burst_ms=1000, period_ms=1000, rate_pps=5000, size=200)] + [atk("h22", 250)],
                           mode="detect_only"),
    # false-positive recovery (Reviewer 1 #6, Reviewer 3 #25): force-block the benign VoIP flow
    "S6_fp_recovery":  dict(traffic=BENIGN_MIX, attack_s=0, mitigation=True, rule_ttl_s=10,
                            force_fp_flows=[dict(src=IP["h21"], dst=VICTIM, proto=17, dport=5001)]),
    # persistent attack with short TTL: leakage per re-detection cycle
    "S7_ttl_redetect": dict(traffic=BENIGN_MIX + [atk("h22", 250)], mitigation=True, rule_ttl_s=10),
    # control-plane LDoS (Reviewer 4): h31 sends 200 ms pulses at 5000 pps, every packet to a random port,
    # so every packet is a new flow -> digest storm + flow-state growth at the controller.
    # h22 runs the normal data-plane LDoS at the same time, to measure the impact on its detection.
    "S9_ctrl_flood_detect": dict(traffic=BENIGN_MIX + HARD_BENIGN + [
                                atk("h22", 250), atk("h31", 200, 1000, 5000, size=64, phase=500, dport=-1)],
                                mode="detect_only"),
    "S9_ctrl_flood_mitigate": dict(traffic=BENIGN_MIX + HARD_BENIGN + [
                                atk("h22", 250), atk("h31", 200, 1000, 5000, size=64, phase=500, dport=-1)],
                                mitigation=True),
    "S9_ctrl_flood_noguard": dict(traffic=BENIGN_MIX + HARD_BENIGN + [
                                atk("h22", 250), atk("h31", 200, 1000, 5000, size=64, phase=500, dport=-1)],
                                mitigation=True, churn_guard=False),
    # ---- batch 4 ----
    # R1 #2: drop tagged packets in the switch BEFORE the DNN confirms (hybrid + early drop)
    "S10_hybrid_tagdrop": dict(traffic=BENIGN_MIX + HARD_BENIGN + [atk("h22", 250)], mitigation=True,
                               mode="hybrid_tagdrop"),
    # R1 #4 / R3 #26: digest-rate sweep (with S5_load_2kpps and S5_load_5kpps)
    "S5_load_1kpps":  dict(traffic=BENIGN_MIX + [dict(host="h12", kind="pulse", name="bg", dport=5004,
                           burst_ms=1000, period_ms=1000, rate_pps=1000, size=200)] + [atk("h22", 250)],
                           mode="detect_only"),
    "S5_load_8kpps":  dict(traffic=BENIGN_MIX + [dict(host="h12", kind="pulse", name="bg", dport=5004,
                           burst_ms=1000, period_ms=1000, rate_pps=8000, size=200)] + [atk("h22", 250)],
                           mode="detect_only"),
    # R1 #6: fill flow_guard (capacity 50) with one rule per destination port -> LRU eviction under churn
    "S11_table_full":  dict(traffic=BENIGN_MIX + [atk("h31", 1000, 1000, 300, size=64, dport=-1)], mitigation=True,
                            capacity=50, churn_guard=False,
                            # table-management test: one rule per flagged flow, so block on first positive
                            block_policy=dict(K=1, window_s=1.0, min_bins=1, recur_window_s=1.0,
                                              gap_s=0.3, sustain_s=0.0),
                            force_fp_flows=[dict(src=IP["h31"], dst=VICTIM, proto=17, dport=-1)]),
    # R3 #20: unrelated CPU load on the controller host during the attack
    "S12_cpu_stress":  dict(traffic=BENIGN_MIX + HARD_BENIGN + [atk("h22", 250)], mitigation=True, cpu_stress=True),
    # R3 #25: legitimate TCP + UDP from the ATTACKER's own host must survive the drop rule
    "S13_attacker_host_legit": dict(traffic=BENIGN_MIX + HARD_BENIGN + [
                               dict(host="h22", kind="tcp"),
                               dict(host="h22", kind="pulse", name="benign_legit", dport=5004, burst_ms=1000,
                                    period_ms=1000, rate_pps=50, size=144),
                               atk("h22", 250)], mitigation=True),
    # blocking-policy comparison (R1 #6, R3 #25): same attack, three policies
    "S15_policy_first_positive": dict(traffic=BENIGN_MIX + HARD_BENIGN + [atk("h22", 250)], mitigation=True,
                                      block_policy=dict(K=1, window_s=1.0, min_bins=1, recur_window_s=1.0,
                                                        gap_s=0.3, sustain_s=0.0)),
    "S15_policy_fast":           dict(traffic=BENIGN_MIX + HARD_BENIGN + [atk("h22", 250)], mitigation=True,
                                      block_policy=dict(K=20, window_s=1.0, min_bins=1, recur_window_s=1.0,
                                                        gap_s=0.3, sustain_s=0.0),
                                      threshold=0.9),
    # (the deployed recurrent policy is what S1_single_A2 uses)
    # pilot-only (policy choice, never reported): S13 traffic with blocking off
    "S13_attacker_host_legit_detect": dict(traffic=BENIGN_MIX + HARD_BENIGN + [
                               dict(host="h22", kind="tcp"),
                               dict(host="h22", kind="pulse", name="benign_legit", dport=5004, burst_ms=1000,
                                    period_ms=1000, rate_pps=50, size=144),
                               atk("h22", 250)], mode="detect_only"),
    # R3 #26: forwarding throughput / latency with digests OFF (compare with S0_benign_only)
    "S14_telemetry_off": dict(traffic=BENIGN_MIX + HARD_BENIGN, attack_s=0, mitigation=False, telemetry=False),
    # in-switch-only baseline (Reviewer 1 #5): drop on tag, no DNN
    "S8_tag_only_single": dict(traffic=BENIGN_MIX + HARD_BENIGN + [atk("h22", 250)], mitigation=True,
                               mode="tag_only"),
    "S8_tag_only_distributed": dict(traffic=BENIGN_MIX + HARD_BENIGN + [
                                atk("h22", 200, 1000, 800, phase=0),
                                atk("h31", 250, 1200, 800, phase=330),
                                atk("h32", 300, 1500, 800, phase=660)], mitigation=True, mode="tag_only"),
}

CTRL_WAIT, PRE, ATTACK, POST = 15, 20, 60, 25


def sh(h, cmd):
    return h.cmd(cmd)


OFFLOAD = "ethtool -K {i} rx off tx off sg off tso off gso off gro off >/dev/null 2>&1"


def disable_offloads(net, out):
    """BMv2 is a software switch: with checksum/segmentation offload left on, hosts send TCP
    segments with unfinished checksums and the receiver drops them (ping/UDP still work).
    Same fix as the p4lang tutorials' P4Host. Needs `ethtool` (sudo apt install ethtool)."""
    if subprocess.call("which ethtool >/dev/null 2>&1", shell=True) != 0:
        raise SystemExit("ethtool not found: run  sudo apt install -y ethtool")
    done = []
    for h in net.hosts:
        for i in h.intfList():
            if i.name != "lo":
                h.cmd(OFFLOAD.format(i=i.name)); done.append(i.name)
    for s in net.switches:
        for i in s.intfList():
            if i.name != "lo":
                subprocess.call(OFFLOAD.format(i=i.name), shell=True); done.append(i.name)
    open(f"{out}/offloads_disabled.txt", "w").write("\n".join(done) + "\n")


P4NAME = "ddos_detect_rev"          # REV-FLOW: "ddos_detect_flow" with --p4 flow


def run_one(name, sc, rep, outroot, model, threshold):
    out = os.path.abspath(os.path.join(outroot, name, f"rep{rep}"))
    if os.path.exists(f"{out}/timeline.json"):          # finished earlier: resume-safe
        info(f"*** skip {name} rep{rep} (already done)\n"); return
    if os.path.exists(out):                             # half-finished (crash/sleep): redo cleanly
        import shutil; shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    attack_s = sc.get("attack_s", ATTACK)
    total = PRE + attack_s + POST
    attackers = [t for t in sc["traffic"] if t.get("attack")]
    if "threshold" in sc: threshold = sc["threshold"]
    manifest = dict(run_name=f"{name}_rep{rep}", mode=sc.get("mode", "hybrid"),
                    mitigation=sc.get("mitigation", False),
                    attack_flows=[dict(src=IP[t["host"]], dst=VICTIM, proto=17, dport=t["dport"]) for t in attackers],
                    force_fp_flows=sc.get("force_fp_flows", []), rule_ttl_s=sc.get("rule_ttl_s", 30),
                    flow_guard_capacity=sc.get("capacity", 1024), model_path=model, threshold=threshold,
                    churn_guard=sc.get("churn_guard", True), telemetry=sc.get("telemetry", True),
                    block_policy=sc.get("block_policy"),
                    digest_mode="flow" if P4NAME == "ddos_detect_flow" else "packet",
                    baseline_seconds=10, log_dir=os.path.join(out, "ctrl_logs"))
    json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=2)
    json.dump(sc, open(f"{out}/scenario.json", "w"), indent=2)

    net = topo.build_net(p4_json=os.path.join(ROOT, "build", f"{P4NAME}.json"))
    H = {h.name: h for h in net.hosts}
    disable_offloads(net, out)
    s2, s3 = net.get("s2"), net.get("s3")
    bott = [l.intf1 if l.intf1.node == s2 else l.intf2 for l in net.linksBetween(s2, s3)][0].name

    env = dict(os.environ, REV_MANIFEST=f"{out}/manifest.json", PYTHONUNBUFFERED="1",
               P4INFO=os.path.join(ROOT, "build", f"{P4NAME}.p4info.txt"), P4JSON=os.path.join(ROOT, "build", f"{P4NAME}.json"))
    ctrl = subprocess.Popen([sys.executable, os.path.join(ROOT, "controller", "ddos_ml_ctrl_rev.py")],
                            cwd=ROOT, env=env, stdout=open(f"{out}/controller.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(CTRL_WAIT)
    tl = {"t_ctrl_started": time.time()}

    v = H["h41"]
    sh(v, "pkill -f 'iperf -s' ; iperf -s -p 5002 >/dev/null 2>&1 &")
    sh(v, "iperf -s -u -p 5001 >/dev/null 2>&1 & iperf -s -u -p 5004 >/dev/null 2>&1 &")
    sh(v, f"tcpdump -i h41-eth0 -n -tt -l -q -U 'udp' > {out}/victim_udp.txt 2>/dev/null &")
    subprocess.call(f"tc -s qdisc show dev {bott} > {out}/tc_before.txt", shell=True)

    stop = threading.Event()

    def sample_bottleneck():
        p = f"/sys/class/net/{bott}/statistics/tx_bytes"
        with open(f"{out}/bottleneck.csv", "w") as f:
            f.write("t,tx_bytes\n")
            while not stop.is_set():
                f.write(f"{time.time()},{open(p).read().strip()}\n"); f.flush(); time.sleep(0.25)
    threading.Thread(target=sample_bottleneck, daemon=True).start()

    tl["t_traffic_start"] = time.time()
    for t in sc["traffic"]:
        h = H[t["host"]]
        if t["kind"] == "ping":
            sh(h, f"ping -D -i 0.2 -w {total} {VICTIM} > {out}/ping_{t['host']}.txt 2>&1 &")
        elif t["kind"] == "tcp":
            # line-buffered and ends 3 s before teardown, so its per-second CSV is never lost
            sh(h, f"stdbuf -oL iperf -c {VICTIM} -p 5002 -t {total - 3} -i 1 -y C > {out}/tcp_{t['host']}.csv 2>&1 &")
        elif t["kind"] == "pulse" and not t.get("attack"):
            sh(h, f"python3 {PULSE} --dst {VICTIM} --dport {t['dport']} --burst-ms {t['burst_ms']} "
                  f"--period-ms {t['period_ms']} --rate-pps {t['rate_pps']} --size {t['size']} "
                  f"--jitter-ms {t.get('jitter_ms', 0)} --duration {total} --log {out}/send_{t['name']}_{t['host']}.csv &")
    time.sleep(PRE)
    tl["t_attack_cmd"] = time.time()
    for t in attackers:
        sh(H[t["host"]], f"python3 {PULSE} --dst {VICTIM} --dport {t['dport']} --burst-ms {t['burst_ms']} "
                         f"--period-ms {t['period_ms']} --rate-pps {t['rate_pps']} --size {t['size']} "
                         f"--phase-ms {t.get('phase_ms', 0)} --sport {t.get('sport', 0)} "
                         f"--jitter-ms {t.get('jitter_ms', 0)} --duration {attack_s} "
                         f"--log {out}/send_{t['name']}.csv &")
    stress = None
    if sc.get("cpu_stress"):
        if subprocess.call("which stress-ng >/dev/null 2>&1", shell=True) != 0:
            raise SystemExit("stress-ng not found: run  sudo apt install -y stress-ng")
        stress = subprocess.Popen(f"stress-ng --cpu 0 --cpu-load 90 --timeout {int(attack_s)}s",
                                  shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tl["cpu_stress"] = "stress-ng --cpu 0 --cpu-load 90 during the attack window"
    time.sleep(attack_s)
    if stress: stress.wait(timeout=10)
    tl["t_attack_end"] = time.time()
    # post-attack probe: every attacker sends 20 more packets (are they still blocked?)
    for t in attackers:
        sh(H[t["host"]], f"python3 {PULSE} --dst {VICTIM} --dport {t['dport']} --burst-ms 100 --period-ms 100 "
                         f"--rate-pps 200 --size {t['size']} --duration 0.1 --log {out}/probe_{t['host']}.csv &")
    time.sleep(POST)
    tl["t_end"] = time.time()

    time.sleep(3)                      # let iperf/ping/udp_pulse finish and write their last lines
    stop.set()
    subprocess.call(f"tc -s qdisc show dev {bott} > {out}/tc_after.txt", shell=True)
    for h in net.hosts:
        sh(h, "pkill -f udp_pulse.py; pkill ping; pkill iperf; pkill tcpdump")
    time.sleep(2)
    ctrl.terminate()
    try: ctrl.wait(10)
    except subprocess.TimeoutExpired: ctrl.kill()
    tl["bottleneck_intf"] = bott
    json.dump(tl, open(f"{out}/timeline.json", "w"), indent=2)
    net.stop()
    subprocess.call("mn -c >/dev/null 2>&1", shell=True)
    info(f"*** done {name} rep{rep} -> {out}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=list(SCEN))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, "results_rev"))
    ap.add_argument("--model", default="models/ddos_dnn_live.pth")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--reps-start", type=int, default=1, help="first repetition number (e.g. to add reps later)")
    ap.add_argument("--p4", choices=["rev", "flow"], default="rev", help="flow = per-flow windowed digests (build/ddos_detect_flow.*)")
    a = ap.parse_args()
    global P4NAME
    P4NAME = "ddos_detect_flow" if a.p4 == "flow" else "ddos_detect_rev"
    if a.list:
        for k, v in SCEN.items():
            print(f"{k:28s} mode={v.get('mode', 'hybrid'):12s} mitigation={v.get('mitigation', False)}")
        return
    setLogLevel("info")
    for rep in range(a.reps_start, a.reps_start + a.reps):
        for s in a.scenarios:
            random.seed(rep)
            run_one(s, SCEN[s], rep, a.out, a.model, a.threshold)


if __name__ == "__main__":
    main()
