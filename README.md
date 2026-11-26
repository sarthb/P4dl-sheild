# P4-DL Shield — revision experiment package

Unzip into the root of `MTP1-main/` so that `revision/` sits next to `controller/` and `topo.py`.

```
revision/
  p4/ddos_detect_rev.p4         revised data plane (diff: ddos_detect_rev.diff)
  controller/ddos_ml_ctrl_rev.py   revised controller (generated from ddos_ml_ctrl.py)
  controller/digest_handler_v2.py  queued digest processing, timing, live telemetry, manifest labels
  controller/policy_manager_v2.py  drop rules with TTL, revocation, LRU eviction, timing
  experiments/udp_pulse.py      paced pulse sender (attack + bursty benign), logs true onset
  experiments/run_scenarios.py  automated Mininet scenarios (T* = training, S* = test)
  training/train_eval_v2.py     run-level train/val/test, PyTorch DNN, ablations, baselines, export
  analysis/analyze_run.py       per-run metrics + mean ± std per scenario
  offline_current_data/         analyses of the EXISTING data in the repo (already run; see RESULTS.md)
```

Nothing under `revision/` overwrites your files. Every revised file is new.

## 0. One-time setup

```bash
cd MTP1-main
cp revision/controller/{ddos_ml_ctrl_rev.py,digest_handler_v2.py,policy_manager_v2.py} controller/
pip install psutil xgboost          # xgboost optional (adds the XGBoost baseline)

# compile the revised P4 program (not compiled in my environment; fix any p4c complaints first)
p4c-bm2-ss --p4v 16 --p4runtime-files build/ddos_detect_rev.p4info.txt \
           -o build/ddos_detect_rev.json revision/p4/ddos_detect_rev.p4
```

For the load experiments (S5), run BMv2 without per-packet logging, i.e. `simple_switch_grpc`
built with `--disable-logging-macros`, or at least without `--log-console`. Otherwise BMv2 itself is the bottleneck.

## 1. Collect training data (new pipeline, tag + telemetry now real)

```bash
sudo -E python3 revision/experiments/run_scenarios.py --list
sudo -E python3 revision/experiments/run_scenarios.py \
     --scenarios T0_benign_train T1_attack_train_b150 T1_attack_train_b350 --reps 4
```
About 2.5 min per run, so roughly 30 min.

## 2. Train once, with no test data involved

```bash
sudo -E python3 revision/training/train_eval_v2.py \
     --train 'results_rev/T*/rep[123]' --val 'results_rev/T*/rep4' \
     --out results_rev/train_only --export models/ddos_dnn_rev.pth
```
The validation runs (rep4) choose early stopping **and** the threshold. No test run is read.

## 3. Run the test scenarios with the frozen model (5 reps each)

```bash
sudo -E python3 revision/experiments/run_scenarios.py --model models/ddos_dnn_rev.pth --reps 5 \
  --scenarios S0_benign_only S1_single_A1 S1_single_A2 S1_single_A3 S1_single_A2_detect \
              S2_distributed S2_distributed_detect S3_short_burst_50ms S3_long_burst_500ms \
              S3_period_500ms S3_period_2000ms S3_tiny_pkts_28B S3_random_sport S4_mimicry \
              S5_load_2kpps S5_load_5kpps S6_fp_recovery S7_ttl_redetect \
              S8_tag_only_single S8_tag_only_distributed
```
20 scenarios × 5 reps × about 2.5 min comes to roughly 4 h. Start with `S1_single_A2 S2_distributed S0_benign_only --reps 2`
to check everything works.

## 4. Analyse

```bash
python3 revision/analysis/analyze_run.py results_rev       # -> results_rev/summary_by_scenario.{csv,md}
# offline ablations/baselines on the new data, test = all S* detect-only runs, evaluated once:
sudo -E python3 revision/training/train_eval_v2.py \
     --train 'results_rev/T*/rep[123]' --val 'results_rev/T*/rep4' \
     --test 'results_rev/S*_detect/rep*' --out results_rev/offline_eval --seeds 5
```
