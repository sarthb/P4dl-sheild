#!/usr/bin/env bash
# run_remaining.sh — everything left after batch 1, in one go.
#   sudo -E bash revision/experiments/run_remaining.sh smoke    # ~25 min: 1 run of each NEW scenario + checks
#   sudo -E bash revision/experiments/run_remaining.sh retrain  # ~30 min: drift check + drift-free model
#   sudo -E bash revision/experiments/run_remaining.sh policy   # ~15 min: 3 pilot runs + choose blocking policy
#   sudo -E bash revision/experiments/run_remaining.sh full     # ~6.5 h: batch 1 again + policy comparison + batches 2-4
# Safe to re-run: finished runs are skipped, half-finished ones are redone.
set -u
cd "$(dirname "$0")/../.."
U="${SUDO_USER:-$USER}"
PY="${PY:-/home/$U/p4env/bin/python3}"
MODEL="${MODEL:-models/ddos_dnn_rev_nosys.pth}"
WINDL="${WINDL:-/mnt/c/Users/Sarath/Downloads}"
for t in ethtool stress-ng p4c-bm2-ss simple_switch_grpc; do
  command -v $t >/dev/null || { echo "missing: $t   ->  sudo apt install -y ${t/p4c-bm2-ss/p4lang-p4c}"; exit 1; }
done
[ -f build/ddos_detect_rev.json ] && grep -q src_guard build/ddos_detect_rev.json || {
  echo "build/ddos_detect_rev.json is old (no src_guard table). Recompile first (README section 6)."; exit 1; }
grep -q install_src controller/policy_manager_v2.py || {
  echo "controller/ has old files. Run: cp revision/controller/*.py controller/"; exit 1; }

run() { local out=$1 reps=$2; shift 2
  mn -c >/dev/null 2>&1; pkill -f simple_switch_grpc
  "$PY" revision/experiments/run_scenarios.py --model "$MODEL" --out "$out" --reps "$reps" --scenarios "$@"
  chown -R "$U:$U" "$out" 2>/dev/null; }

case "${1:-}" in
retrain)
  # deployed model WITHOUT absolute host counters (they grow all night -> rising false positives)
  [ -d results_rev_oldmodel ] || { mkdir results_rev_oldmodel; mv results_rev/S* results_rev_oldmodel/ 2>/dev/null; }
  chown -R "$U:$U" results_rev results_rev_oldmodel models
  sudo -u "$U" "$PY" revision/analysis/drift_check.py results_rev_oldmodel > results_rev_oldmodel/drift_check.txt 2>&1
  sudo -u "$U" "$PY" revision/training/train_eval_v2.py --train 'results_rev/T*/rep[123]' --val 'results_rev/T*/rep4' \
       --out results_rev/train_only_nosys --export "$MODEL" --export-featset no_system_telemetry
  tail -4 results_rev_oldmodel/drift_check.txt ;;
policy)
  # pilot runs (detection only, NOT reported) + validation runs -> choose the blocking policy by replay
  run results_pilot 1 S1_single_A2_detect S2_distributed_detect S13_attacker_host_legit_detect
  sudo -u "$U" "$PY" revision/training/policy_replay.py --model "$MODEL" \
       --val 'results_rev/T*/rep4,results_pilot/*/rep1' --preview 'results_smoke/*/rep1' --min-bins 2 ;;
finish)
  # S6 rerun (one-shot forced FP), re-analysis, policy preview on detection-only runs, fresh export
  rm -rf results_rev/S6_fp_recovery
  run results_rev 3 S6_fp_recovery
  chown -R "$U:$U" results_rev
  sudo -u "$U" "$PY" revision/analysis/analyze_run.py results_rev > results_rev/analyze.log 2>&1
  sudo -u "$U" "$PY" revision/training/policy_replay.py --model "$MODEL" --use-policy models/block_policy.json \
       --preview 'results_rev/S1_single_A2_detect/rep*,results_rev/S2_distributed_detect/rep*,results_rev/S3_*/rep*,results_rev/S4_mimicry/rep*' \
       --out results_rev/policy_on_detect_runs.csv > results_rev/policy_preview.log 2>&1
  mkdir -p results_rev/export
  cp results_rev/summary_by_scenario.csv results_rev/all_runs.csv results_rev/analyze.log \
     results_rev/policy_on_detect_runs.csv results_rev/policy_preview.log models/block_policy.json results_rev/export/ 2>/dev/null
  cp results_rev/offline_eval/summary_mean_std.csv results_rev/export/offline_summary.csv 2>/dev/null
  (cd results_rev && tar czf export.tgz export) && cp results_rev/export.tgz "$WINDL/revision_results_v2.tgz"
  cat results_rev/policy_preview.log | tail -12
  date; echo "DONE. Upload $WINDL/revision_results_v2.tgz" ;;
finish2)
  # policy preview on detection-only runs + ablations of the deployed 8-feature model, then export v3
  chown -R "$U:$U" results_rev
  sudo -u "$U" "$PY" revision/training/policy_replay.py --model "$MODEL" --use-policy models/block_policy.json \
       --preview 'results_rev/S1_single_A2_detect/rep*,results_rev/S2_distributed_detect/rep*,results_rev/S3_*/rep*,results_rev/S4_mimicry/rep*' \
       --out results_rev/policy_on_detect_runs.csv > results_rev/policy_preview.log 2>&1
  tail -12 results_rev/policy_preview.log
  DET='results_rev/S1_single_A2_detect/rep*,results_rev/S2_distributed_detect/rep*,results_rev/S3_*/rep*,results_rev/S4_mimicry/rep*'
  sudo -u "$U" "$PY" -u revision/training/train_eval_v2.py --train 'results_rev/T*/rep[123]' --val 'results_rev/T*/rep4' \
       --test "$DET" --out results_rev/offline_eval_deployed --seeds 3 --max-rows-per-run 20000 \
       --featsets no_system_telemetry,deployed_no_tag,deployed_no_pkt_len,deployed_no_protocol 2>&1 | grep -E "runs:|done"
  mkdir -p results_rev/export3
  cp results_rev/policy_on_detect_runs.csv results_rev/policy_preview.log results_rev/export3/
  cp results_rev/offline_eval_deployed/summary_mean_std.csv results_rev/export3/offline_deployed_summary.csv
  (cd results_rev && tar czf export3.tgz export3) && cp results_rev/export3.tgz "$WINDL/revision_results_v3.tgz"
  date; echo "DONE. Upload $WINDL/revision_results_v3.tgz" ;;
smoke)
  run results_smoke 1 S9_ctrl_flood_mitigate S10_hybrid_tagdrop S11_table_full S12_cpu_stress \
                      S13_attacker_host_legit S14_telemetry_off S5_load_8kpps
  "$PY" revision/experiments/smoke_check.py results_smoke ;;
full)
  date; echo "== batch 1 again, with the drift-free model"
  run results_rev 5 S0_benign_only S1_single_A1 S1_single_A2 S1_single_A3 S1_single_A2_detect \
                    S2_distributed S2_distributed_detect S8_tag_only_single S8_tag_only_distributed
  date; echo "== blocking-policy comparison"
  run results_rev 5 S15_policy_first_positive S15_policy_fast
  date; echo "== batch 2 (control plane)"
  run results_rev 5 S9_ctrl_flood_detect S9_ctrl_flood_noguard S9_ctrl_flood_mitigate
  date; echo "== batch 3 (robustness, load, FP recovery, TTL)"
  run results_rev 3 S3_short_burst_50ms S3_long_burst_500ms S3_period_500ms S3_period_2000ms S3_tiny_pkts_28B \
                    S3_random_sport S4_mimicry S5_load_2kpps S5_load_5kpps S6_fp_recovery S7_ttl_redetect
  date; echo "== batch 4 (early drop, load sweep, table full, CPU stress, attacker-host legit, telemetry off)"
  run results_rev 3 S5_load_1kpps S5_load_8kpps S10_hybrid_tagdrop S11_table_full S12_cpu_stress \
                    S13_attacker_host_legit S14_telemetry_off
  date; echo "== analysis"
  chown -R "$U:$U" results_rev models
  sudo -u "$U" "$PY" revision/analysis/analyze_run.py results_rev > results_rev/analyze.log 2>&1
  DET='results_rev/S1_single_A2_detect/rep*,results_rev/S2_distributed_detect/rep*,results_rev/S3_*/rep*,results_rev/S4_mimicry/rep*,results_rev/S5_*/rep*,results_rev/S9_ctrl_flood_detect/rep*'
  sudo -u "$U" "$PY" revision/training/train_eval_v2.py --train 'results_rev/T*/rep[123]' --val 'results_rev/T*/rep4' \
       --test "$DET" --out results_rev/offline_eval --seeds 3 > results_rev/offline_eval.log 2>&1
  sudo -u "$U" "$PY" revision/analysis/repd_eval.py --train 'results_rev/T*/rep[123]' --val 'results_rev/T*/rep4' \
       --test "$DET" --out results_rev/repd_eval > results_rev/repd_eval.log 2>&1
  mkdir -p results_rev/export
  cp results_rev/summary_by_scenario.csv results_rev/all_runs.csv results_rev/analyze.log results_rev/export/ 2>/dev/null
  cp results_rev/offline_eval/summary_mean_std.csv results_rev/export/offline_summary.csv 2>/dev/null
  cp results_rev/offline_eval/A1_dataset_audit.csv results_rev/export/new_dataset_audit.csv 2>/dev/null
  cp results_rev/repd_eval/*.csv results_rev/repd_eval/*.json results_rev/export/ 2>/dev/null
  sudo -u "$U" "$PY" revision/analysis/drift_check.py results_rev > results_rev/export/drift_check_new.txt 2>&1
  cp results_rev_oldmodel/drift_check.txt results_rev/export/drift_check_oldmodel.txt 2>/dev/null
  cp results_rev_oldmodel/summary_by_scenario.csv results_rev/export/oldmodel_summary.csv 2>/dev/null
  (cd results_rev && tar czf export.tgz export) && cp results_rev/export.tgz "$WINDL/revision_results.tgz" 2>/dev/null
  date; echo "DONE. Upload $WINDL/revision_results.tgz" ;;
*) echo "usage: sudo -E bash $0 smoke|full"; exit 1 ;;
esac
