#!/usr/bin/env bash
# run_flow.sh -- PER-FLOW windowed digests (50 ms), live. Everything goes to results_flow*/; results_rev is untouched.
#   sudo -E bash revision/experiments/run_flow.sh build   # ~1 min : generate + compile P4, copy controller files
#   sudo -E bash revision/experiments/run_flow.sh smoke   # ~5 min : 1 short check run (S1_single_A2_detect)
#   sudo -E bash revision/experiments/run_flow.sh train   # ~35 min: 12 training runs + 3 pilot runs + train/export
#   sudo -E bash revision/experiments/run_flow.sh test    # ~80 min: 35 test runs
#   sudo -E bash revision/experiments/run_flow.sh analyze # ~5 min : analyze + compare with results_rev
#   sudo -E bash revision/experiments/run_flow.sh all     # train + test + analyze (~2 h, unattended)
#   sudo -E bash revision/experiments/run_flow.sh main    # missing reps to make per-flow the main design (~2.5 h) + analyze
# Safe to re-run: finished runs are skipped.
set -u
cd "$(dirname "$0")/../.."
U="${SUDO_USER:-$USER}"
PY="${PY:-/home/$U/p4env/bin/python3}"
MODEL=models/ddos_flow.pth
WIN_US="${WIN_US:-50000}"

run() { local out=$1 reps=$2; shift 2
  mn -c >/dev/null 2>&1; pkill -f simple_switch_grpc
  "$PY" revision/experiments/run_scenarios.py --p4 flow --model "$MODEL" --out "$out" --reps "$reps" --scenarios "$@"
  chown -R "$U:$U" "$out" 2>/dev/null; }

build() {
  "$PY" revision/p4/make_flow_p4.py --window-us "$WIN_US" || exit 1
  mkdir -p build
  p4c-bm2-ss --p4v 16 --p4runtime-files build/ddos_detect_flow.p4info.txt \
             -o build/ddos_detect_flow.json revision/p4/ddos_detect_flow.p4 || { echo "P4 COMPILE FAILED"; exit 1; }
  cp revision/controller/{ddos_ml_ctrl_rev.py,digest_handler_v2.py,policy_manager_v2.py} controller/
  grep -q _handle_flow_msg controller/digest_handler_v2.py && echo "P4 COMPILED, controller updated (flow window ${WIN_US} us)"
}

train() {
  # training data must be collected BEFORE a flow model exists: the controller runs detection-only here
  run results_flow 4 T0_benign_train T1_attack_train_b150 T1_attack_train_b350
  run results_flow_pilot 1 S1_single_A2_detect S2_distributed_detect S13_attacker_host_legit_detect
  sudo -u "$U" "$PY" revision/training/train_flow.py --root results_flow --pilot 'results_flow_pilot/*/rep1' \
       --window-s "$("$PY" -c "print($WIN_US/1e6)")" | tee results_flow/train_flow.log
}

testruns() {
  [ -f "$MODEL" ] || { echo "no $MODEL: run 'train' first"; exit 1; }
  run results_flow 3 S0_benign_only S1_single_A2 S2_distributed S13_attacker_host_legit
  run results_flow 3 S9_ctrl_flood_detect S9_ctrl_flood_noguard S9_ctrl_flood_mitigate
  run results_flow 3 S5_load_2kpps S5_load_8kpps
  run results_flow 2 S2_distributed_detect S3_short_burst_50ms S3_period_2000ms S4_mimicry
}

analyze() {
  sudo -u "$U" "$PY" revision/analysis/analyze_run.py results_flow > results_flow/analyze.log 2>&1
  sudo -u "$U" "$PY" revision/analysis/compare_flow.py --pp results_rev --pf results_flow --out results_flow/compare
  tar czf "/home/$U/flow_results.tgz" results_flow/compare results_flow/train_flow.log results_flow/all_runs.csv \
      results_flow/summary_by_scenario.csv models/block_policy_flow.json 2>/dev/null
  cp "/home/$U/flow_results.tgz" /mnt/c/Users/Sarath/Downloads/ 2>/dev/null
  echo "send back: flow_results.tgz (also copied to Downloads)"
}

main() {
  # bring per-flow to the same evidence as the per-packet main results (5 runs / 3 runs), so it can be the main design.
  # Existing reps are skipped automatically; only the missing ones run.  ~65 runs, ~2.5 h.
  [ -f "$MODEL" ] || { echo "no $MODEL: run 'train' first"; exit 1; }
  run results_flow 5 S0_benign_only S1_single_A1 S1_single_A2 S1_single_A3 S1_single_A2_detect S2_distributed
  run results_flow 5 S2_distributed_detect S9_ctrl_flood_detect S9_ctrl_flood_noguard S9_ctrl_flood_mitigate
  run results_flow 3 S13_attacker_host_legit S12_cpu_stress S6_fp_recovery S7_ttl_redetect
  run results_flow 3 S3_short_burst_50ms S3_long_burst_500ms S3_period_500ms S3_period_2000ms S3_tiny_pkts_28B \
                     S3_random_sport S4_mimicry
  run results_flow 3 S5_load_1kpps S5_load_2kpps S5_load_5kpps S5_load_8kpps
}

case "${1:-}" in
  build) build ;;
  smoke)
    # detection-only check that per-flow messages arrive and are logged (no model needed)
    run results_flow_smoke 1 S1_single_A2_detect
    f=results_flow_smoke/S1_single_A2_detect/rep1/ctrl_logs/flow_msgs.csv
    [ -s "$f" ] && echo "flow_msgs.csv rows: $(($(wc -l < "$f") - 1))" && head -3 "$f" || echo "SMOKE FAILED: no $f" ;;
  train) train ;;
  test) testruns ;;
  analyze) analyze ;;
  all) train; testruns; analyze ;;
  main) main; analyze ;;
  *) sed -n 2,9p "$0" ;;
esac
