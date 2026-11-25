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

## Which experiment answers which comment

| Scenario / script | Reviewer comments |
|---|---|
| `analyze_run.py` latency fields (onset → digest → detection → rule RPC → block, leaked packets) | R1 #2, R3 #24, R4 (6.60 s) |
| S1 with mixed benign traffic on the **same port 5001** + bursty benign (h42) | R3 #14, #25, R4 (100%/0% "by construction") |
| S2 three attackers, different periods/phases; per-attacker peak/avg rate, packet size, bottleneck utilisation | R3 #13, R4 (not distributed) |
| S3 unseen burst/period/size/random sport, S4 mimicry | R1 #3 |
| S5 heavy background traffic: digest rate, queue depth, digest lag p99, inference p99 | R1 #4, R3 #26 |
| S6 forced false positive with 10 s TTL; S7 TTL re-detection leakage; occupancy, LRU eviction | R1 #6, R3 #25 |
| S8 tag-only in-switch baseline vs. hybrid; `tag_policy` table for pre-confirmation dropping | R1 #5, R1 #2 |
| S9 control-plane LDoS: random-port pulses flood digests/flow state; flow-churn guard -> `src_guard` (with / without guard / no mitigation); idle-flow expiry | R4 (control-plane attacks), R1 #4, R3 #19 |
| `train_eval_v2.py` ablations (no tag, tag only, no telemetry, delta telemetry, no pkt_len/proto) + LR/RF/GB/XGBoost + seeds | R3 #12, #15, #17, #21, R4, R1 #7 |
| Tag rule = BURST_PKTS/BURST_WIN_US in P4; labels only from manifest; identity-based `src_rate_limit` removed | R3 #15 |
| One digest per packet per switch (`max_list_size=1`, 1 ms); `digests` vs. unique packets in metrics | R3 #19 |

## 5. Benchmark datasets (Table V; Reviewer 3 #18, Reviewer 4)

```bash
# on your machine, where the 30 GB CIC-DDoS2019 CSVs are:
python3 revision/benchmarks/cic_sample.py --root /path/to/cic-ddos2019 --out cic2019_sample.csv.gz
# -> attach cic2019_sample.csv.gz (+ cic2019_sample_counts.csv), or run directly:
python3 revision/benchmarks/benchmark_eval.py --dataset cic2019 --csv cic2019_sample.csv.gz --out bench_cic --seeds 5
```
Train on the 01-12 day, test on the 03-11 day (a different capture day, with attack types it has never seen).
No IP, port, flow ID or timestamp features. The script reports both the 7 live-mapped features and all flow features,
plus a majority-class baseline, balanced accuracy and PR-AUC.

IEEE SDN-SlowRate-DDoS (DataPort needs a login, so run locally):
```bash
python3 revision/benchmarks/ieee_sample.py --root /path/to/sdn-slowrate-ddos --out ieee_sample.csv.gz
# attach ieee_sample.csv.gz + ieee_sample_stats.json (stats are computed on the FULL data)
```

IEEE SDN-SlowRate-DDoS (no labels or system stats in the download; labels derived: port 8000 = slowhttptest attack):
```bash
python3 revision/benchmarks/ieee_eval.py --root /mnt/c/Users/Sarath/Downloads/sdn-slowrate-ddos --out ~/bench_ieee
# send: ~/bench_ieee/{summary.csv,label_counts.csv,label_rule_check.json,per_test_scenario.csv}
```

## 6. Control-plane LDoS experiment (S9) — needs a recompile

The P4 program gained a `src_guard` table, so rebuild it once (after batch 1 has finished):
```bash
cp revision/controller/{ddos_ml_ctrl_rev.py,digest_handler_v2.py,policy_manager_v2.py} controller/
p4c-bm2-ss --p4v 16 --p4runtime-files build/ddos_detect_rev.p4info.txt \
           -o build/ddos_detect_rev.json revision/p4/ddos_detect_rev.p4
sudo nohup ~/p4env/bin/python3 revision/experiments/run_scenarios.py --model models/ddos_dnn_rev.pth --reps 5 \
     --scenarios S9_ctrl_flood_detect S9_ctrl_flood_noguard S9_ctrl_flood_mitigate > batch2.log 2>&1 &
```
No retraining needed: the new table is empty by default and the DNN features are unchanged.

## 7. TOMORROW — everything left, in order (after batch 1 shows 45 done)

```bash
cd ~/MTP1-main
# 0) batch 1 results (upload summary_by_scenario.csv + all_runs.csv)
sudo chown -R $USER:$USER results_rev
~/p4env/bin/python3 revision/analysis/analyze_run.py results_rev | tail -12
cp results_rev/summary_by_scenario.csv results_rev/all_runs.csv /mnt/c/Users/Sarath/Downloads/

# 1) new package, controller files, recompile P4 (src_guard table), one extra tool
unzip -o -q /mnt/c/Users/Sarath/Downloads/revision.zip
cp revision/controller/{ddos_ml_ctrl_rev.py,digest_handler_v2.py,policy_manager_v2.py} controller/
p4c-bm2-ss --p4v 16 --p4runtime-files build/ddos_detect_rev.p4info.txt \
           -o build/ddos_detect_rev.json revision/p4/ddos_detect_rev.p4 && echo "P4 COMPILED"
sudo apt install -y stress-ng

# 2) quick check of the 7 new scenarios (~25 min) -> paste the PASS/WARN lines
sudo -E bash revision/experiments/run_remaining.sh smoke

# 3) overnight (~3.5-4 h): 69 runs + all analyses; keep Windows awake
sudo -E nohup bash revision/experiments/run_remaining.sh full > full.log 2>&1 &
grep -c "\*\*\* done" full.log          # progress: 69 = all runs finished
tail -2 full.log                        # "DONE. Upload .../revision_results.tgz"
```
Then upload `revision_results.tgz` from Downloads. If anything stops overnight, just start step 3 again:
finished runs are skipped.

## 8. TODAY (after batch 1): drift fix, then everything in one overnight run

Batch 1 showed false positives growing with every repetition. The cause is the absolute host counters
(tx_MB / rx_MB are cumulative since boot and keep growing). The deployed model now uses the 8 features
without system telemetry (R1 #7). No new data is needed: the same training runs are reused.

```bash
cd ~/MTP1-main
unzip -o -q /mnt/c/Users/Sarath/Downloads/revision.zip
cp revision/controller/{ddos_ml_ctrl_rev.py,digest_handler_v2.py,policy_manager_v2.py} controller/
p4c-bm2-ss --p4v 16 --p4runtime-files build/ddos_detect_rev.p4info.txt \
           -o build/ddos_detect_rev.json revision/p4/ddos_detect_rev.p4 && echo "P4 COMPILED"
sudo apt install -y stress-ng
sudo -E bash revision/experiments/run_remaining.sh retrain     # ~30 min; paste the last lines
sudo -E bash revision/experiments/run_remaining.sh smoke       # ~25 min; paste PASS/WARN lines
sudo -E nohup bash revision/experiments/run_remaining.sh full > full.log 2>&1 &    # ~6 h overnight
```
Old-model batch 1 is kept in `results_rev_oldmodel/` as evidence of the drift.

## 9. Blocking policy (chosen on validation) — before the overnight run

The smoke runs showed benign flows blocked by a single burst of positives (h42's bursty flow) or by a few
stray positives (legit UDP from the attacker host, S13). The controller now blocks a flow only when its
positives persist (K positives within W s AND positives in B different 1-s bins within R s). The policy and
threshold are chosen on the validation runs (T*/rep4) by replaying the model; smoke runs are a preview only.

```bash
cd ~/MTP1-main
unzip -o -q /mnt/c/Users/Sarath/Downloads/revision.zip
cp revision/controller/{ddos_ml_ctrl_rev.py,digest_handler_v2.py,policy_manager_v2.py} controller/
sudo -E bash revision/experiments/run_remaining.sh policy          # ~5-15 min -> paste output
sudo rm -rf results_smoke
sudo -E bash revision/experiments/run_remaining.sh smoke           # ~25 min -> paste PASS/WARN/FAIL
sudo -E nohup bash revision/experiments/run_remaining.sh full > full.log 2>&1 &     # overnight
```

## 10. Per-flow windowed digests (optional extension, live)

Offline replay (RESULTS.md 11.11) showed that 50 ms per-flow summaries cut controller messages ~20x with equal or better
detection. This section runs the same idea live. Everything goes to `results_flow*/`; `results_rev/` is not touched.

Files: `p4/make_flow_p4.py` (generates `p4/ddos_detect_flow.p4`), controller per-flow path in
`controller/digest_handler_v2.py` (auto-detected from the 16-field digest; logs `ctrl_logs/flow_msgs.csv`),
`training/train_flow.py`, `analysis/compare_flow.py`, `experiments/run_flow.sh`, `--p4 flow` in `run_scenarios.py`.

```bash
cd ~/MTP1-main
cp "$(ls -t /mnt/c/Users/Sarath/Downloads/revision*.zip | head -1)" ~/revision.zip && (cd ~ && unzip -o -q revision.zip)
cp -r ~/revision .            # revision/ inside MTP1-main, as before
sudo -E bash revision/experiments/run_flow.sh build     # compile; must print "P4 COMPILED"
sudo -E bash revision/experiments/run_flow.sh smoke     # must print "flow_msgs.csv rows: ..." (a few thousand)
sudo -E nohup bash revision/experiments/run_flow.sh all > flow.log 2>&1 &    # ~2 h unattended
tail -f flow.log
```
Send back `~/flow_results.tgz` (also copied to Downloads).
