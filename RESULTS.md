# Revision experiments: results on the existing data, and what they mean

All numbers below were computed from the files committed in the repo (`dataset_runs/`, `results/pluto_*`, `comparison_results/`).
Scripts and raw outputs are in `offline_current_data/`.

## 1. Bottom line

The existing results cannot support the paper's claims. They aren't wrong arithmetically (Table III matches the logs),
but the experiment design lets a trivial shortcut separate the classes, and two of the paper's key components do nothing:

| # | Finding | Evidence | Reviewer comments it confirms |
|---|---|---|---|
| F1 | **The suspicion tag is always 0.** The `tag_suspicious()` action is defined but no table ever calls it. The tag reads `diffserv == 0xA8`, which no host ever sets. | `tag == 1` in 0 of 115,798 rows | R3 #12, #15; R4 (tag ablation) |
| F2 | **CPU/memory/TX/RX are constants** (0.022, 0.142, 14.55, 12.36) in every row. The controller reads `logs/system_profile.csv`, which nothing updates. Hosts write to `/tmp/<host>_sysmon.csv`. | 1 distinct value per feature | R1 #7, R3 #20; contradicts Fig. 9 and the "system-impact" argument |
| F3 | **Packet length alone separates the classes.** Attack packets are always 28 B (hping3 UDP, no payload); benign UDP is always 1498 B; benign ICMP is 84/56 B. A single threshold on `pkt_len` gives F1 = 1.000 on the held-out runs. | Table in §2 | R3 #14, R4 |
| F4 | **The held-out test runs (A1–A3) contain no benign UDP**: only ping plus the attack. So the test set doesn't check UDP-vs-UDP discrimination at all. | audit | R3 #14 |
| F5 | **The "attack" doesn't stress the network.** `timeout 0.2 hping3 --fast` sends about 2–4 packets per 1 s cycle: about 2.5–3.6 attack packets/s × 28 B ≈ 1 kb/s on a 10 Mb/s bottleneck. RTT doesn't change during the attack (29.5 → 29.3 ms). There is no measurable DoS effect to mitigate. | logs | R3 #13, R4 |
| F6 | **The 6.60 s detection delay is mostly a measurement artefact.** About 5.5 s passes between `attack_start.txt` and the first attack packet reaching the controller (hping3 start-up and the shell loop). Then about 1.0 s (9–13 attack digests) passes before the first positive. | §3 | R1 #2, R3 #24, R4 |
| F7 | **The detection and mitigation runs used different models.** The benign probability is 0.000182 in the detection runs and 0.000110 in the mitigation runs. The mitigation model flags the first attack packet (p = 0.999); the detection model scores the first ~9 at ≈0. That's why MRT (2.43 s) < detection delay (6.60 s). The two can't be combined into one pipeline claim. | §3 | R3 #24 |
| F8 | **Mitigation is hard-wired to the attack port.** `should_mitigate = pred==1 and proto==17 and dport==5001`. The "0% normal loss" partly follows from this rule, not from the model. "100% attack loss" is measured with 20 hping3 packets sent *after* the rule exists. | `digest_handler.py` | R3 #25, R4 |
| F9 | **Identity-based marking.** With `ENABLE_MITIGATION`, the controller installs `src_rate_limit` → `rl_mark` for the hard-coded attacker IP 10.0.2.2. This is the "source-based marking" R3 asked about. (It only affects TCP:80, so it didn't touch the UDP results.) | `ddos_ml_ctrl.py` | R3 #15 |
| F10 | **Threshold leakage.** `train_live_dnn.py` early-stops on its *test* runs (L03 + normal_udp_5001) and trains on `normal_udp_5001_1M`. `tune_threshold_live.py` then picks 0.70 on L03 + both normal_udp runs, i.e. partly on *training* data. The A1–A3 runs were never used, so Table III itself is still held out. | scripts | R3 #17 |
| F11 | **One digest per packet per switch** (`max_list_size=1`, 1 ms timeout, telemetry wildcard on). There is no flow summarising, sampling or batching. Each attack packet appears 3× (s2, s3, s4). The text's "compact flow-level digests" is inaccurate. | P4 + controller | R3 #19 |
| F12 | **REPD's EER = 1 − AUC is legitimate**, because the reproduced REPD outputs effectively binary scores and then the two coincide. But REPD was scored on 80-row windows of a different trace, while the DNN was scored per digest. PLUTO/Whisper/P&F/NetBeacon EERs are literature values that look derived as 1 − AUC; cite each one, or mark it "not reported". | §4 | R3 #22, #23 |

## 2. Offline experiments on the existing data (A1–A4)

Protocol: **train** = L02, L025, port6001, normal, normal_udp_5001_1M; **validation** = L03 + normal_udp_5001
(used for early stopping and threshold only); **test** = A1–A3 detection runs (evaluated once).
DNN = same widths as the paper (128-64-32, ReLU, Adam 1e-3, batch 256, early stopping on validation F1, patience 10),
implemented with scikit-learn. PyTorch wasn't available in my environment, so there's no BatchNorm/Dropout.
`training/train_eval_v2.py` reruns this with your exact PyTorch model. Mean ± std over seeds.
"Padded" = test attack packets set to 1498 B, simulating an attacker who pads to the benign UDP size.

| Model | Features | Seeds | F1 | AUC | EER | FPR | F1 (attack padded to 1498 B) |
|---|---|---|---|---|---|---|---|
| DNN | full_12 | 5 | 0.9993 ± 0.0000 | 0.9987 ± 0.0002 | 0.0011 ± 0.0004 | 0.0000 | 0.4030 ± 0.4008 |
| DNN | no_tag | 5 | 0.9992 ± 0.0001 | 0.9986 ± 0.0000 | 0.0012 ± 0.0004 | 0.0000 | 0.6453 ± 0.4157 |
| DNN | no_system_telemetry | 5 | 0.9993 ± 0.0000 | 0.9986 ± 0.0001 | 0.0013 ± 0.0004 | 0.0000 | 0.5892 ± 0.3886 |
| DNN | tag_only | 5 | 0.0000 | 0.5000 | 0.5000 | 0.0000 | n/a |
| DNN | no_protocol | 5 | 0.9755 ± 0.0046 | 0.9866 ± 0.0016 | 0.0128 ± 0.0026 | 0.0018 | 0.0000 |
| DNN | pkt_len_only | 5 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| DNN | no_pkt_len_no_proto | 5 | 0.5075 ± 0.4101 | 0.9864 ± 0.0016 | 0.0132 ± 0.0039 | 0.4928 ± 0.4857 | n/a |
| DNN | flow_temporal_only | 5 | 0.5446 ± 0.3915 | 0.9872 ± 0.0059 | 0.0190 ± 0.0169 | 0.3811 ± 0.4049 | n/a |
| LogReg | full_12 | 5 | 0.9993 | 0.9990 | 0.0015 | 0.0000 | 0.0000 |
| RandomForest | full_12 | 5 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0060 ± 0.0124 |
| Gradient boosting (XGBoost stand-in) | full_12 | 5 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| RandomForest | no_pkt_len_no_proto | 5 | 0.2494 ± 0.0346 | 0.9998 | 0.0022 | 0.6241 ± 0.1093 | n/a |
| Gradient boosting | no_pkt_len_no_proto | 5 | 0.6501 ± 0.0992 | 0.9952 | 0.0150 | 0.0018 | n/a |
| Rule `pkt_len < c` | 1 feature | – | **1.0000** | 1.0000 | 0.0000 | 0.0000 | n/a |
| Rule on packet rate | 1 feature | – | 0.2912 | 0.5972 | 0.4028 | 0.0355 | n/a |
| Rule on inter-arrival time | 1 feature | – | 0.3809 | 0.6479 | 0.3521 | 0.0461 | n/a |

What this shows:
* Removing the tag or the system telemetry changes nothing. That's expected, since both are constant (F1, F2).
* A one-line packet-size rule matches or beats the DNN, and random forest, gradient boosting and logistic regression do just as well.
  So the "DNN beats baselines" claim can't be made with this data.
* With only behavioural features (no size, no protocol), the classes are still *rankable* (AUC ≈ 0.99), but the
  threshold chosen on validation doesn't carry over to the test runs: F1 swings 0.25–0.65 with high variance across seeds.
  This is the honest version of the detector, and it isn't robust yet.
* When the attacker pads packets to the benign size, F1 collapses for every model (0.00–0.65).
  This is direct evidence for R1 #3's concern.

Per-case confusion matrices (A1–A3) and all seeds: `results/confusion_matrices_mean.csv`, `results/raw_results_all_seeds.csv`.

## 3. Verifying Table III / IV against the logs

`results/deployed_model_from_logs.txt`:

| run | TN | FP | FN | TP | AUC | EER (from ROC) | 1−AUC | attack cmd → 1st attack digest | 1st digest → detection |
|---|---|---|---|---|---|---|---|---|---|
| detect A1 | 6818 | 21 | 9 | 546 | 0.9831 | 0.0147 | 0.0169 | 5.73 s | 1.03 s (9 digests) |
| detect A2 | 6867 | 19 | 11 | 733 | 0.9882 | 0.0077 | 0.0118 | 5.58 s | 1.00 s (9) |
| detect A3 | 6892 | 11 | 19 | 779 | 0.9843 | 0.0100 | 0.0157 | 5.28 s | 1.18 s (13) |

* Table III (0.9960 acc, 0.9852 AUC, 0.0108 EER) is the **mean of these three runs**. It's reproducible, and the EER here *is*
  computed from the ROC curve.
* The contribution list's **96.46% / 0.9926** can't be reproduced from the committed logs. It's most likely the console
  output of `train_live_dnn.py` on its own test split (L03 + normal_udp_5001). Remove it, or label it as that split.
* The 6.60 s delay breaks down as ≈5.5 s waiting for the first attack packet + ≈1.0 s to the first positive. Inference and
  rule installation aren't measured in the current logs; the new controller logs them per digest.
* The mitigation runs saw only 1–6 attack digests in total (the rule went in on the first one). With a different model
  (F7), the MRT and detection delay belong to different systems.

## 4. Table VI notes

* Reproduced REPD (α=2, w=80): 641 windows, 408 attack. Its AUC computed from the entropy/distance score is 0.9399 and its EER is 0.0601,
  identical to the hard-label values, because the score is effectively two-valued. So "EER = 1−AUC" is correct *for REPD*,
  and the paper should say why.
* REPD was evaluated on a trace that isn't among the committed runs (timestamps around 1781155125), with 80-row windows.
  The DNN was evaluated per digest. Rerun both on the same runs and the same unit, or state the difference explicitly.
* PLUTO, Whisper, P&F, NetBeacon: mark these as "reported in [ref]". Put "–" where the original paper doesn't report an EER,
  and qualify the superiority claim.

## 5. What has to change in the paper regardless of new results

1. Title and abstract: no control-plane attack is evaluated. Drop "Control-Plane Attacks", or add an experiment
   (e.g. digest flooding, which S5 approaches).
2. Describe the tag exactly as implemented (the new burst rule: > 50 packets per (src,dst) pair in 100 ms), or remove it.
3. Describe CPU/mem/TX/RX as controller-host measurements. In Mininet all hosts share one kernel, so they can't be per-flow.
   Report the ablation with and without them.
4. Digests are per packet, per switch: fix the "flow-level digest" wording.
5. Define detection delay and onset-to-block from sender-side onset (see `analysis/analyze_run.py` docstring).
6. Report attack peak/average rate, packet size and bottleneck utilisation, and show actual harm (TCP throughput drop)
   before claiming mitigation.

## 6. Testbed experiments (code written, not run — needs your Mininet/BMv2 machine)

See `README.md` for the run order. Key design points:
* The attack is now a real Shrew-style pulse: 2000 pps × 1028 B = 16.4 Mb/s during a burst, above the 10 Mb/s bottleneck,
  sent by `udp_pulse.py`. The first packet's timestamp is logged at the sender.
* Benign traffic on the **same victim port 5001**: small-packet constant UDP (144 B, 50 pps), plus a bursty 1000-B
  "video chunk" flow that deliberately resembles the attack, plus TCP iperf (whose throughput shows the DoS impact) and ping.
* Training scenarios (T*) use burst lengths of 150/350 ms. The test cases at 200/250/300 ms, plus the unseen variants, are never trained on.
* The P4 changes (`ddos_detect_rev.p4`) were **not compiled** here. Expect to fix small p4c complaints.
  The analysis script and pulse sender were smoke-tested on synthetic and loopback data. The controller modules
  were only syntax-checked.

## 7. Benchmark: LR–HR DDoS 2024 (Table V; R3 #18, R4)

Source: github.com/nqmn/lr-hr_ddos_2024_ddos_dataset (mirror of the Kaggle release). Script: `benchmarks/lrhr_eval.py`,
output: `benchmarks/results_lrhr/`. Five repeated stratified 60/20/20 splits; validation chooses early stopping and the threshold;
DNN = 128-64-32 (scikit-learn stand-in for the PyTorch model).

**What the dataset contains**
* 113,407 rows, but **69,530 (61.3%) are exact duplicates**. Only 43,877 rows are distinct.
* **98.6% of attack rows are duplicates**: only 987 distinct attack rows (of 70,508). Benign rows are almost all distinct (42,890).
* **All attack rows have protocol = 0 and ports = 0**; only 280 benign rows do. A one-line rule `protocol == 0`
  gets 99.74% accuracy.
* The file is ordered (all benign, then all attack), so no time-ordered split is possible.

**Results (mean over 5 splits)**

| Split | Features | Model | Accuracy | F1 | FPR | AUC | Test rows also in train |
|---|---|---|---|---|---|---|---|
| Random rows (usual) | live-mapped 7 (incl. protocol) | DNN | 0.9997 | 0.9998 | 0.0003 | 0.9999 | 61.7% |
| Random rows | live-mapped 7 | Random forest | 0.9999 | 0.9999 | 0.0000 | 1.0000 | 61.7% |
| Random rows | — | Rule `protocol==0` | 0.9974 | 0.9979 | 0.0070 | 0.9965 | 61.7% |
| De-duplicated | live-mapped 7 | DNN | 0.9991 | 0.9797 ± 0.0066 | 0.0004 | 0.9980 | 0% |
| De-duplicated | live-mapped 7 | Random forest | 0.9996 | 0.9909 | 0.0002 | 1.0000 | 0% |
| De-duplicated | — | Rule `protocol==0` | 0.9935 | 0.8746 | 0.0066 | 0.9967 | 0% |
| De-duplicated | live-mapped, **no protocol** | DNN | 0.7616 ± 0.28 | 0.2331 ± 0.22 | 0.2393 | 0.9402 | 0% |
| De-duplicated | live-mapped, no protocol | Gradient boosting | 0.9948 | 0.8804 | 0.0018 | 0.9983 | 0% |
| De-duplicated | live-mapped, no protocol | Random forest | 0.9940 | 0.8550 | 0.0011 | 0.9851 | 0% |

(Full table including all 21 "cleaned" features and logistic regression: `results_lrhr/summary.csv`.)

**What this means for Table V**
* The reported 99.97% matches the **random-row split** exactly, where about 62% of test rows also appear in training.
* With duplicates removed and without the protocol shortcut, the DNN reaches F1 = 0.23 (very unstable across splits).
  Tree models do better (F1 0.86–0.88), so the DNN isn't the best model here either.
* Suggested reporting: describe the duplicates and the protocol artefact, report the de-duplicated split, give the result with
  and without protocol, and include the baselines. Or drop this dataset from the generalisation claim.

## 8. Benchmark: IEEE SDN-SlowRate-DDoS (Table V; R3 #18, R4)

Source: IEEE DataPort, DOI 10.21227/amrt-8y98 (CC BY 4.0), 23 scenarios, 16.0 M flow-table snapshots.
Run locally in WSL with `benchmarks/ieee_eval.py` (3 seeds; up to 20,000 rows per class per scenario, sampled after
features are computed on the full files; scikit-learn DNN stand-in, 128-64-32).

**What the dataset contains**
* **No label column and no CPU/memory/TX/RX** in any of the 23 folders. `ieee_dnn.py` reads `Label`, `cpu`, `mem`,
  `tx` and `rx`, so the CSV behind the reported 99.48% was not this download as-is. Its origin must be stated in the paper.
* Labels were derived with a documented rule: the attack is slowhttptest (slow HTTP read) against port 8000, so a row is
  attack iff its source or destination port is 8000. **Check: 0 attack rows in both normal-only scenarios (01, 11).**
* Attack scenarios still carry 12–36% benign background rows (iperf 5001/5201, SSH 22, web 8090), e.g. 28.5% in scenario 02.
  **Labelling by folder name therefore mislabels all of those rows as attack.**
* Each flow appears as many periodic snapshots (e.g. median 21–26 per flow in scenario 02) with cumulative counters.

**Results (mean ± std over 3 seeds)**

| Protocol | Labels | Features | DNN F1 | DNN FPR | Best baseline F1 (model) |
|---|---|---|---|---|---|
| P1 random rows | port 8000 | counters **+ IPs/MACs/ports/switch** (ieee_dnn.py style) | 0.9966 ± 0.0003 | 0.000 | 1.0000 (RF, GB) |
| P2 random rows | port 8000 | identity-free | 0.9351 ± 0.0003 | 0.079 | 0.9431 (GB) |
| P3 split by flow | port 8000 | identity-free | 0.9355 ± 0.0023 | 0.079 | 0.9425 (GB) |
| **P4 train T4000 → test T1200** | port 8000 | identity-free | **0.8665 ± 0.0014** | **0.284** | 0.8952 (RF), FPR 0.211 |
| P5 random rows | folder name | identity-free | 0.9788 (acc 0.959) | 0.872 | balanced acc. only 0.56–0.61 |
| P5 scored vs. port labels | folder → port | identity-free | 0.6495 | 0.986 | balanced acc. 0.51 |
| P6 T4000 → T1200 | folder name | identity-free | 0.9796 (acc 0.960) | 1.000 | balanced acc. 0.50 (= "always attack") |
| P7 random rows | folder name | counters **+ identifiers** (original labelling + ieee_dnn.py inputs) | 0.9820 ± 0.0002 (acc 0.965, bal. acc 0.63) | 0.731 | 0.9882 (RF), bal. acc 0.78, FPR 0.44 |
| P7 scored vs. port labels | folder → port | counters + identifiers | 0.6524 (bal. acc 0.51) | 0.973 | 0.6586 (RF) |

**P7 vs. the paper:** the original setup on the public download gives 96.5% accuracy, not 99.48%. "Always attack" alone scores 95.45%
on this test set (LogReg collapses to exactly that). The remaining ~3-point gap is most likely the CPU/mem/TX/RX columns of the original CSV,
which are absent from the download and are constant-ish per capture run, so they identify the folder; a different sample/split could
also contribute. The original CSV is not available, so this cannot be confirmed.

**Decision (authors, Option A):** Table V reports traffic-level labels (attack = port 8000) with P4 cross-session as the main IEEE
number (DNN F1 0.867, FPR 0.284; RF 0.895). The original 99.48% (folder-name labels + system counters, original CSV not available)
is kept only as a footnote, together with the P7 reproduction on the public data (acc 0.965, bal. acc 0.63, FPR 0.73).

**What this means for Table V**
* The reported ~99.5% is reproduced only when IPs, MACs, ports and switch IDs are inputs and rows are split at random (P1).
  With those identifiers, random forest and gradient boosting are perfect, so the score reflects host identity, not attack behaviour.
* Without identifiers, the honest cross-session result (P4) is **F1 ≈ 0.87 with a false-positive rate of ≈ 28%** for the DNN.
  Random forest and gradient boosting beat it (F1 0.88–0.90). Logistic regression matches it (0.867).
* Folder-name labels look excellent on accuracy (96%) and F1 (0.98), but that's the class imbalance:
  balanced accuracy is 0.50–0.61 and the FPR is 0.78–1.00, i.e. the model calls almost everything an attack. Scored against
  the traffic-level labels, it doesn't detect anything (balanced accuracy 0.50). If the original labels came from folder
  names, the 99.48% doesn't measure detection.
* Suggested reporting: the port-8000 labelling rule with its check, P4 as the main number, P1 as an explicit
  "with identifiers" comparison, baselines alongside, and the sample design (balanced per scenario). Or drop the dataset
  from the generalisation claim.

## 9. Benchmark: CICDDoS2019 (Table V; R3 #18, R4)

Run locally in WSL: `benchmarks/cic_sample.py` then `benchmarks/benchmark_eval.py --dataset cic2019 --seeds 5`.
**Train = 01-12 day (11 files), test = 03-11 day (7 files)**: a different capture day, including Portmap, which never appears in
training. Validation = last 20% (by timestamp) of each training-day file; it alone sets early stopping and the threshold.
No IPs, ports, flow IDs or timestamps as features. The sample keeps **all** 113,828 benign flows plus about 20,000 attack flows per file,
so the test set is 29% benign (in the full data benign is under 0.2%). Accuracy is therefore not comparable with numbers on the
natural class mix; F1, balanced accuracy and benign recall are.

Sample: train 221,489 rows (44,997 benign) | val 55,377 (11,866 benign) | test 196,965 (56,965 benign).

| Features | Model | Accuracy | Bal. acc. | F1 | Benign recall | AUC | EER |
|---|---|---|---|---|---|---|---|
| All flow statistics (37 CICFlowMeter columns, no identifiers) | DNN (5 seeds) | 0.9987 | 0.9984 | **0.9991** | 0.9977 | 0.9999 | 0.0010 |
| | Logistic regression | 0.9981 | 0.9972 | 0.9987 | 0.9951 | 0.9996 | 0.0017 |
| | Gradient boosting | 0.9978 | 0.9965 | 0.9985 | 0.9933 | 1.0000 | 0.0019 |
| | Random forest | 0.9824 | 0.9856 | 0.9875 | 0.9930 | 0.9987 | 0.0091 |
| **7 features mappable to the live P4 features** | DNN (5 seeds) | 0.8807 | 0.8981 | **0.9101** | 0.9394 | 0.9755 | 0.1029 |
| | Logistic regression | 0.8179 | 0.8374 | 0.8606 | 0.8837 | 0.8767 | 0.1556 |
| | Gradient boosting | 0.8071 | 0.8583 | 0.8445 | 0.9798 | 0.9614 | 0.1159 |
| | Random forest | 0.8092 | 0.8590 | 0.8466 | 0.9771 | 0.9842 | 0.0653 |
| — | Majority class ("all attack") | 0.7108 | 0.5000 | 0.8309 | 0.0000 | 0.5000 | 0.5000 |

**What this means for Table V**
* Unlike the other two datasets, the high CICDDoS2019 result **holds up under a clean cross-day split**: F1 0.999, AUC 0.9999,
  consistent with (slightly above) the paper's 99.58% / F1 0.99 / AUC 0.995.
* But it needs CICFlowMeter's full flow statistics, which the P4 switch does not compute. With only the 7 features the live system
  actually has, F1 drops to **0.910** (FPR ≈ 6%). This is the number that speaks to the deployed system.
* With full features, plain logistic regression matches the DNN (0.9987 vs 0.9991), so the dataset doesn't show the DNN is needed.
  With the 7 live features, the DNN is the best model (0.910 vs 0.84–0.86): the one setting where it clearly wins.
* Note the 03-11 `UDPLag.csv` actually contains mostly Syn-labelled flows (a known quirk of the release); labels were used as given.

## 10. Table V — proposed replacement rows (all reproducible from public data)

| Dataset | Split | Features | DNN F1 | DNN FPR | Best baseline F1 | Paper (old) |
|---|---|---|---|---|---|---|
| IEEE SDN-SlowRate | T4000 → T1200 sessions | flow counters, no identifiers | 0.867 | 0.284 | 0.895 (RF) | 99.48% acc |
| LR-HR DDoS 2024 | de-duplicated, random | 7 live-mapped incl. protocol | 0.980 | 0.0004 | 0.991 (RF) | 99.97% acc |
| LR-HR DDoS 2024 | de-duplicated, random | 6 live-mapped, no protocol | 0.233 | 0.239 | 0.880 (GB) | — |
| CICDDoS2019 | 01-12 → 03-11 days | 7 live-mapped | 0.910 | 0.061 | 0.861 (LR) | 99.58% acc |
| CICDDoS2019 | 01-12 → 03-11 days | 37 flow statistics | 0.999 | 0.002 | 0.999 (LR) | — |

Write-up guidance: say that each dataset was retrained separately, list the feature mapping, give the split and label rule,
report the baselines, and drop "generalises across datasets" as an unqualified claim. The fair claim is that with live-equivalent
features the DNN is competitive on CICDDoS2019, and weaker than tree models on IEEE and LR-HR without shortcut features.


## 11. Testbed results (revised system: 124 test runs + 12 training-data runs + 3 pilot runs; Mininet + BMv2, WSL2 Ubuntu 22.04)

All runs use the revised P4 program (burst tag, `src_guard`), the drift-free 8-feature DNN and the recurrent
blocking policy chosen on validation and pilot runs (threshold 0.99, ≥ 50 positives in 1 s, and either ≥ 2 episodes
within 3 s or one episode lasting ≥ 0.5 s; episode gap 0.3 s). Mean ± std over repetitions. Attack: Shrew-style UDP pulses at
16.4 Mb/s during a burst (2000 pps × 1028 B) on a 10 Mb/s bottleneck, plus benign TCP, ping, small-packet UDP and a bursty
"hard-negative" UDP flow on the **same victim port 5001**.

### 11.1 Detection, blocking and impact (R1 #2, R3 #13, #14, #24, #25, R4)
| Scenario | Runs | Benign flows blocked | Attackers blocked | Onset → block (s) | Attack pkts before block | TCP during attack (Mb/s) | Ping loss (%) |
|---|---|---|---|---|---|---|---|
| S0 benign only | 5 | 0 of 50 | – | – | – | 9.38 ± 0.04 (whole run) | – |
| S1 single, 200 ms bursts | 5 | 0 of 50 | 5 of 5 | 1.16 ± 0.01 | 246 ± 26 | 9.30 ± 0.08 | 1.3 ± 0.2 |
| S1 single, 250 ms bursts | 5 | 0 of 50 | 5 of 5 | 1.17 ± 0.02 | 308 ± 22 | 9.36 ± 0.03 | 1.4 ± 0.5 |
| S1 single, 300 ms bursts | 5 | 0 of 52 | 5 of 5 | 1.15 ± 0.04 | 379 ± 36 | 9.00 ± 0.42 | 1.9 ± 0.2 |
| S1 single, **no mitigation** | 5 | 0 of 61 | n/a (off) | – | – | 6.69 ± 0.04 | 14.0 ± 0.8 |
| S2 distributed (3 attackers) | 5 | 0 of 58 | 15 of 15 | 1.52 ± 0.02 | 364 ± 7 | 9.18 ± 0.25 | 0.9 ± 0.4 |
| S2 distributed, **no mitigation** | 5 | 0 of 91 | n/a (off) | – | – | 8.25 ± 0.01 | 1.1 ± 0.9 |
| S8 tag-only switch rule, single | 5 | 0 of 64 | 0 of 5 | – | – | 4.26 ± 0.14 | 0.8 ± 0.9 |
| S8 tag-only switch rule, distributed | 5 | 0 of 90 | 0 of 15 | – | – | 4.70 ± 0.05 | 0.2 ± 0.2 |
| S10 hybrid + early tag drop | 3 | 0 of 30 | 3 of 3 | 1.13 ± 0.05 | 221 ± 26 | 4.64 ± 0.32 | 0.3 ± 0.2 |
| S12 controller CPU at 93% | 3 | 0 of 18 | 3 of 3 | 1.96 ± 0.12 | 610 ± 63 | 6.63 ± 0.20 | 5.0 ± 0.7 |
| S13 legit traffic from attacker host | 3 | 1 of 46 | 3 of 3 | 1.30 ± 0.25 | 563 ± 400 | 4.10 ± 0.46 | 1.5 ± 0.2 |

* **Attack harm without mitigation:** TCP falls from 9.4 to 6.7 Mb/s (−29%) and ping loss rises to 14%. With mitigation TCP stays
  at 9.0–9.4 Mb/s and ping loss at 1–2%.
* **No benign flow was blocked** in any S0–S2 run (0 of 362 benign flows), including the bursty hard negative (loss 0.07%).
* **Where the time goes (S1, 250 ms):** first digest at the controller 0.14 s → DNN decision 0.05 s → rule write 0.6 ms
  → the remaining ≈ 1 s is the policy waiting for the attack's second episode. Detection itself takes ≈ 0.2 s; inference
  p99 is 0.32 ms. (Paper's old values: 6.60 s detection, 2.43 s mitigation, measured on different models.)
* **Tag-only in-switch rule (R1 #5):** blocks no attacker flow and halves benign TCP (4.3–4.7 Mb/s), because a fast TCP
  transfer also exceeds 500 packets/s. Early tag drop in the hybrid (S10) has the same TCP cost.
* **Legitimate traffic from the attacker's own host (R3 #25):** its UDP loses 1.6% and its TCP keeps flowing; only the attack flow
  is blocked. One benign flow was blocked in 1 of 3 runs (the bursty hard negative).

### 11.2 Blocking-policy comparison (R1 #6, R3 #25), same attack, 5 runs each
| Policy | Benign flows blocked / run | Bursty benign loss | Onset → block | Attack pkts before block |
|---|---|---|---|---|
| Block on first positive digest (original design) | 1.2 | 96% | 0.19 s | 85 |
| Fast: ≥ 20 positives in 1 s | 0 | 0.1% | 0.22 s | 98 |
| **Recurrent (deployed)** | **0** | **0.07%** | 1.17 s | 308 |

On the validation + pilot runs (80 benign flows) the original rule blocked 7 benign flows; both persistence policies blocked 0.

### 11.3 Unseen attack variants (R1 #3): deployed policy replayed on detection-only runs
| Variant | Benign blocked | Attacks blocked | Onset → block |
|---|---|---|---|
| Single (A2) / distributed | 0 / 0 | 1/1, 3/3 | 1.00 s / 1.27 s |
| 50 ms bursts / 500 ms bursts | 0 / 0 | 1/1, 1/1 | 1.00 s / 0.72 s |
| period 0.5 s / 2 s | 0 / 0 | 1/1, 1/1 | 0.60 s / 2.00 s |
| random source port / 28-byte packets | 0 / 0 | 1/1, 1/1 | 1.00 s / 1.00 s |
| mimicry (benign-like size and shape) | 0 | 1/1 | 2.30 s |

### 11.4 Feature ablation on the testbed data (R3 #12, #15, R4, R1 #7)
Train T*/rep1-3, validation T*/rep4, test = 31 detection-only runs of unseen variants (S1/S2 detect, S3, S4); at most
20,000 digests per run and class (fixed seed); 3 seeds. Mean over test scenarios:

| Model / features | F1 | AUC | EER | FPR | F1 spread over seeds |
|---|---|---|---|---|---|
| **DNN, 8 features (deployed)** | **0.951** | 1.000 | 0.001 | 0.0002 | 0.002 |
| DNN, 8 without tag | 0.900 | 1.000 | 0.001 | 0.0002 | 0.003 |
| DNN, 8 without packet length | 0.952 | 1.000 | 0.002 | 0.0003 | 0.001 |
| DNN, 8 without protocol | 0.840 | 0.936 | 0.060 | 0.0002 | 0.003 |
| DNN, 12 (+ CPU/mem/TX/RX) | 0.528 | 0.934 | 0.090 | 0.366 | **0.459** |
| Random forest, 12 | 0.909 | 1.000 | 0.002 | 0.0003 | 0.060 |
| Gradient boosting, 12 | 0.891 | 0.935 | 0.073 | 0.0006 | 0.005 |
| Logistic regression, 12 | 0.686 | 0.808 | 0.198 | 0.898 | 0 |
| Rule: packet length / packet rate | 0.836 / 0.745 | 0.809 / 0.681 | 0.191 / 0.319 | 0.38 / 0.58 | – |
| Tag alone | 0.576 | 0.640 | 0.360 | 0.230 | – |

* The tag matters for very short bursts (50 ms: F1 0.57 with tag, 0.12 without). Packet length is not a shortcut any more.
* Protocol matters for the 28-byte attack (F1 0.003 without it). Weakest case overall: 50 ms bursts (F1 0.57, AUC 0.999) —
  the policy still blocks them in 1.0 s.

### 11.5 Telemetry drift (R1 #7, R3 #20)
Benign digests wrongly flagged, by repetition through one night (S0–S2): old 12-feature model 0.001% → 0.22% → 0.66% → 1.11%
→ 1.27% (benign flows blocked: 0 in rep 1, up to 2 per run later, bursty-benign loss up to 94%); 8-feature model
0.002 / 0.002 / 0.001 / 0.001 / 0.002%. CPU/mem/TX/RX are cumulative or slowly drifting host values and are dropped.

### 11.6 Control-plane LDoS (R4, title claim)
Random-port pulses (5000 pps × 200 ms per second, 64 B) every packet a new flow, with the data-plane attack running at the same time:

| Defence | Controller flow state | Queue delay p99 | TCP during attack | Source blocked |
|---|---|---|---|---|
| None | 22,300 flows | 2.8 s | 4.3 Mb/s | – |
| Per-flow rules only | 22,100 flows | 12.6 s | 7.8 Mb/s | no |
| **Flow-churn guard** | 1,400 flows | 0.25 s | 9.2 Mb/s | 4 source rules, 0.74 s |

### 11.7 Controller load, resources, rule management (R1 #4, #6, R3 #19, #20, #26)
* **Digest capacity:** ~5,000 digests/s (S1) are handled with 0.07 s detection; at ≥ 7,500/s the queue grows without bound
  (p99 35–70 s) and detection takes 10–33 s. The controller is single-threaded Python; this is the scaling limit to report.
* **Digest delivery** from BMv2 to the controller is spread over 0–200 ms (median 100 ms, p99 199 ms) in every run, with the
  controller push-based and `max_timeout_ns` = 1 ms; this BMv2 behaviour is most of the 0.14 s "first digest" time.
* **Telemetry overhead (S14 vs S0):** same TCP (9.42 vs 9.38 Mb/s) and median RTT (115 ms); BMv2 CPU 74% → 98%.
* **CPU stress on the controller host (93%):** blocking 1.17 → 1.96 s, inference p99 0.32 → 0.67 ms.
* **Rule table full (capacity 50):** occupancy stays at 50, ≈ 32,000 LRU evictions per run, rule write 0.3 ms (p50).
* **False-positive recovery (S6):** a forced false positive blocks a benign flow for 10.1–10.2 s (TTL 10 s); traffic
  resumes 0.09–0.14 s after expiry.
* **Rule timeout (S7, TTL 10 s):** the attack is re-detected 1.7 s after each expiry.

### 11.8 REPD on exactly the same digests (R3 #23)
REPD's settings were chosen on the validation runs (α = 3, window 50, destination-IP entropy). Per digest, the DNN beats REPD
on every data-plane scenario (typical F1 0.99 vs 0.53–0.89; EER 0.00 vs 0.05–0.15). REPD does better on the control-plane flood
(F1 0.88 vs 0.53), which the DNN is not designed for and the churn guard handles.

### 11.10 Tag as a detector and threshold sensitivity (R4 full report: Methods 1, Results 4)
`analysis/sensitivity.py` on the 31 detection-only test runs (16,372,691 digests; live-logged probabilities of the deployed model).

**Tag alone (all digests, natural class mix):** precision 0.238, recall 0.530, F1 0.329, FPR 0.230 (pooled); weakest on 50 ms
bursts (F1 0.086) and mimicry (0.142), best on 500 ms bursts (0.560). (Ablation table 11.4 gives 0.576 on the class-balanced sample.)

| Threshold | Digest precision | Digest recall | Digest F1 | Digest FPR | Benign flows blocked (of 487) | Attack flows blocked (of 41) | Mean onset→block |
|---|---|---|---|---|---|---|---|
| 0.3 | 0.9976 | 0.9300 | 0.9493 | 0.0002 | 5 | 41 | 1.217 s |
| 0.5 | 0.9978 | 0.9296 | 0.9490 | 0.0002 | 1 | 41 | 1.217 s |
| 0.7 | 0.9998 | 0.9254 | 0.9469 | 0.0000 | 0 | 41 | 1.217 s |
| 0.9 | 0.9999 | 0.9239 | 0.9460 | 0.0000 | 0 | 41 | 1.217 s |
| 0.95 | 0.9999 | 0.9232 | 0.9456 | 0.0000 | 0 | 41 | 1.217 s |
| **0.99** | 0.9999 | 0.9206 | 0.9441 | 0.0000 | 0 | 41 | 1.217 s |
| 0.999 | 0.9999 | 0.9066 | 0.9364 | 0.0000 | 0 | 41 | 1.217 s |

K sweep at 0.99 (B = 2 kept): K = 1 / 5 / 20 / 50 / 100 -> 0 benign blocked, all 41 attacks blocked, mean 1.201 / 1.205 / 1.214 / 1.217 / 1.295 s;
max 4.91 s in every setting. The recurrence requirement (B = 2), not K, prevents false blocks (cf. 11.2: K = 1, B = 1 blocks benign flows).
Onset here = the flow's first digest at the controller (replay), not the sender timestamp used in 11.1.

### 11.11 Per-flow (windowed) digest export: offline replay (training/perflow_sim.py)
Replays the logged per-packet digests as if each switch sent ONE summary per flow (src, dst, proto, dport) per window
(global grid, sent when the next window's first packet arrives) plus one message per new flow. Same 8 features computed
per window + window packet count; same 128-64-32 MLP (PyTorch, 3 seeds); threshold and policy chosen on validation (T*/rep4).
Times are switch time from each flow's first packet (BMv2 digest delivery, 0-200 ms, not included). 31 test runs.

Messages to the controller (mean per run):
| Scenario | Per packet | 50 ms summaries | 100 ms summaries | Peak msgs/s: packet / 50 ms / 100 ms |
|---|---|---|---|---|
| S0 benign | 222,587 | 11,439 (19x) | 6,648 (33x) | 5,281 / 266 / 154 |
| S1 A2 | 536,298 | 27,377 (20x) | 15,851 (34x) | 5,571 / 271 / 157 |
| S5 load 8 kpps | 514,789 | 7,729 (67x) | 4,680 (110x) | 13,599 / 196 / 119 |
| S9 flood, no guard | 596,439 | 145,662 (4x) | 134,566 (4x) | 7,046 / 2,707 / 2,600 |
Test runs: 18-22x (50 ms), 31-37x (100 ms).

Detection (per message): per-packet deployed F1 mean 0.944; 50 ms summaries 0.958 +- 0.007; 100 ms 0.965 +- 0.008.
50 ms bursts: 0.54 (packet) vs 0.69 (50 ms) vs 0.76 (100 ms); 28 B packets 0.996 / 0.993 / 0.963.

Flow level (test runs; per-packet 487 benign flows, per-flow 224 = flows with >= 1 summary):
| Design | Policy | Benign blocked | Attacks blocked | Mean / max time to block |
|---|---|---|---|---|
| Per packet (deployed) | K=50, B=2 | 0 | 41/41 | 1.35 / 3.94 s |
| 50 ms summaries | K=2, B=1 (chosen on validation) | 0 | 41/41 | 0.12 / 0.98 s |
| 50 ms summaries | K=1, B=2 (recurrence) | 0 | 41/41 | 1.14 / 2.07 s |
| 100 ms summaries | K=5, B=1 (chosen on validation) | 0 | 41/41 | 2.90 / 35.97 s |
| 100 ms summaries | K=1, B=2 | 34 | 41/41 | 1.20 / 2.94 s |
50 ms, K=2 B=1 per scenario: 0.06-0.10 s, except 50 ms bursts 0.67 s (per packet: 3.32 s).
100 ms windows blur the evidence: zero false blocks only with K>=5, which is slow. 50 ms is the workable setting.
Caveats: offline replay, not a live run; validation has only 19 benign flows; single-window flows are never classified;
the S9 flood still costs one message per new flow (the churn guard would have to move into the switch).

### 11.12 Per-flow windowed digests, LIVE (experiments/run_flow.sh; results_flow/)
P4: p4/ddos_detect_flow.p4 (generated by make_flow_p4.py): per-flow registers (16,384 slots, crc32 of src,dst,proto,dport),
50 ms windows restarting at the first packet after the previous window; one digest per finished window (msg_type 1) and one
per new flow (msg_type 0). Controller: 9 window features, same MLP; trained on 12 per-flow training runs (211,347 summaries,
2.1% attack), threshold 0.7 on validation (F1 1.000); policy chosen on validation + 3 pilot runs: all 10 options blocked
0 of 42 benign flows, fastest = K=1, B=1 (first positive summary). Test: 3 runs per mitigation scenario, 2 per detection-only.

| Metric (mean) | S1 single pp / pf | S2 distributed pp / pf | S5 8,000 pps (detect-only) pp / pf | S9 + churn guard pp / pf |
|---|---|---|---|---|
| Messages per run | 536,298 / 23,851 | 535,392 / 24,986 | 514,789 / 15,868 | 535,281 / 24,679 |
| Peak msgs/s per switch | 1,667 / 87 | 1,789 / 93 | 3,394 / 102 | 1,930 / 408 |
| Controller queue delay p99 | 129 / 3.1 ms | 221 / 3.4 ms | 65.3 s / 0.36 s | 233 / 5.9 ms |
| Controller CPU | 76 / 4.5 % | 79 / 5.8 % | 87 / 3.8 % | 77 / 4.6 % |
| Detection delay | 0.20 / 0.18 s | 0.25 / 0.13 s | 32.9 / 0.17 s | -- |
| Onset -> block | 1.17 / 0.18 s | 1.52 / 0.13 s | -- | 0.74 / 0.13 s |
| Attack pkts before block | 308 / 58 | 364 / 65 | -- | 526 / 123 |
| Benign flows blocked | 0 / 0 | 0 / 0 | -- | 0 / 0 |
| TCP during attack (Mb/s) | 9.36 / 8.68 | 9.18 / 9.17 | -- | 9.21 / 8.88 |
| BMv2 CPU | 126 / 129 % | 126 / 133 % | 386 / 251 % | 134 / 139 % |

- Benign flows blocked, per-flow, S0 + S1 + S2: 0 of 106 (9 runs). S13: 1 of 3 runs blocked the bursty flow (same as per-packet);
  onset->block 1.30 -> 0.16 s. S5 2,000 pps: queue p99 56.4 s -> 4 ms, detection 22.4 -> 0.17 s.
- First-positive blocking is safe with summaries (0 benign blocks) while per-packet first-positive blocked 1.2 benign flows/run (11.2).
- Window summaries, live F1 (threshold 0.7): S1 0.967, S2 0.947, 50 ms bursts 0.989, period 2 s 1.000, mimicry 0.999, S13 0.986.
- Control-plane flood without the guard (S9 noguard): still 140k msgs/run (one per new flow; 4x fewer), queue p99 12.7 s -> 0.34 s;
  per-flow blocking of flood flows (8,114/run) and 1 benign flow blocked in 1 of 3 runs -> the churn guard is still needed.
- Cost: more switch work per packet. TCP before the attack unchanged (9.40-9.49 vs 9.43-9.47 Mb/s); during the attack unchanged in S2,
  -0.7 Mb/s in S1; benign-only T0 8.66 +- 0.98 vs 9.46 Mb/s; ping loss 1.1-2.4 % vs 0.9-1.4 %, more variable; median RTT +1-3 ms;
  BMv2 CPU +3-8 % (except under S5 load, where per-packet digests cost more). Only 3 runs per scenario.
- Artefacts to ignore: ping_loss in S0 (100 %, no attack window) and S9 noguard per-flow (negative); S9 mitigate per-flow detection
  delay (15.6 s in 2 runs) averages flood flows that the source rule already blocked; ctrl_flow_table_max = 0 because flow state
  lives in the switch.

### 11.13 Per-flow, FINAL (5 runs per mitigation scenario; this is the system in the revised paper)
Decision (author): per-flow summaries fully replace per-packet digests as the main design; per-packet runs are the baseline.
Per-packet-only scenarios kept: S8, S10 (tag drop), S14 (digests off), S15 (policies); drift history. Per-flow test runs: 98.

Live (results_flow/, 5 runs; S12/S13 3 runs):
| Scenario | Benign blocked | Onset -> block | Attack pkts before block | TCP during attack (Mb/s) | Ping loss |
|---|---|---|---|---|---|
| S0 benign | 0 of 46 | -- | -- | 8.63 +- 1.80 (whole run) | -- |
| S1 A1 200 ms | 0 of 60 | 0.18 +- 0.01 s | 61 +- 7 | 9.26 +- 0.35 | 2.3 % |
| S1 A2 250 ms | 0 of 50 | 0.18 +- 0.01 s | 54 +- 14 | 8.85 +- 0.37 | 1.1 % |
| S1 A3 300 ms | 0 of 60 | 0.18 +- 0.02 s | 55 +- 16 | 8.72 +- 1.20 | 1.6 % |
| S1 A2 no mitigation | -- | -- | -- | 6.46 +- 0.61 | 15.7 % |
| S2 distributed | 0 of 72 | 0.13 +- 0.01 s | 63 +- 5 | 9.23 +- 0.14 | 1.9 % |
| S2 no mitigation | -- | -- | -- | 7.58 +- 0.77 | 4.3 % |
| S13 attacker-host legit | 1 of 39 | 0.16 +- 0.05 s | 54 +- 35 | 4.22 +- 0.24 | 1.1 % |
| S12 CPU stress 93 % | 0 of 34 | 0.18 +- 0.03 s | 59 +- 19 | 8.79 +- 0.33 | 3.5 % |
- 0 of 400 benign flows blocked in the 36 mitigation runs of S0, S1, S2, S7, S9 (guard), S12.
- Timing (S1 A2): onset -> first summary 0.175 s; controller 0.002 s; rule write < 1 ms; total 0.177 +- 0.014 s.
  BMv2 digest lag median 99 ms, p99 268 ms.
- pp vs pf (5 runs): S1 messages 536,298 / 24,427; peak/s 1,667 / 85; queue p99 129 / 3.2 ms; ctrl CPU 76 / 4.7 %;
  S2 535,392 / 25,087, 221 / 3.2 ms; S9 guard 535,281 / 25,511, queue 233 / 5.9 ms, onset->block 0.74 / 0.13 s.
  S5 detection 10-33 s -> 0.14-0.17 s. BMv2 CPU +5-11 points (36 under CPU stress). TCP during attack up to -0.5 Mb/s in S1, same in S2.
- S9 (pf): none 141,825 msgs, queue p99 0.16 s, TCP 3.5 Mb/s; rules only 140,811, 0.27 s, 7.1 Mb/s, 2 of 57 benign blocked;
  guard 25,511, 6 ms, 9.1 Mb/s, 0 of 63.
- S6: benign flow blocked 10.4 s, resumed 0.09 s after expiry. S7: re-blocked 0.56 s after expiry, 218 attack pkts (pp: 1.68 s, 1,399).
- S11 full table (pf, 3 runs; added 2026-11-22): occupancy 50 throughout; rules installed ~1,678/run, LRU evictions ~1,578,
  TTL expiries 64-94, insert failures 0; rule write p50 0.32 ms, p99 0.5 ms; benign blocked 0 of 22; queue p99 21 ms (pp 33 s);
  ctrl CPU 6.6 % (pp 82 %); TCP 9.2 Mb/s. Random-port flows get a summary only when a port recurs within 50 ms, so ~1,670 of
  ~15,700 attack flows/run are blocked (pp: ~14,200, ~32,000 evictions). Corrects the earlier claim that per-flow cannot fill the table.
  Note: run manually as root -> chown results_flow before analyze (Permission denied on metrics.json otherwise).

Offline (analysis/flow_offline.py; 31 detection-only runs, 771,161 summaries, 34,868 attack; 3 seeds):
- MLP 9 features F1 0.988 +- 0.004 (AUC 1.000, EER 0.0001); 50 ms bursts 0.917. LR 0.955, GB 0.894, RF 0.808.
- Ablation: no tag 0.979 (50 ms: 0.847), no pkt_len 0.949, no protocol 0.863, no win_pkts 0.956.
- Rules: win_pkts 0.577, pkt_len 0.192, flow pps 0.082; tag alone F1 0.203 (P 0.12, R 0.68, FPR 0.24).
- Threshold replay (215 benign / 41 attack flows): 0 benign blocked at >= 0.95; 1 at 0.5-0.9; 18 at 0.3; all attacks blocked.
  K=5 -> 0 benign, 0.35 s; B=2 -> 1.24 s.
- Variants (deployed policy): 0.13-0.17 s; 50 ms bursts 0.84 s; mimicry 0.17 s; 1 of 215 benign blocked.
- Confusion (thr 0.7, totals): TP 34,651, FP 3, FN 217, TN 736,290 (per scenario: paper Table XVI).
- Drift: 0 % of benign summaries flagged in each of 5 repetitions.
- REPD on summaries: validation tuning collapsed (F1 0) -> compared on per-packet digests only (paper Table XIV).

### 11.14 Published P4 detectors on the same traffic (analysis/switch_baselines.py; Reviewer 1, Comment 1.4)
Per-packet logs of the 31 detection-only test runs (16.4 M records); training T*/rep1-3, validation T*/rep4. Flow level: first
decision that would block the flow; times from the attacker's first packet (include BMv2 digest delay for all methods).
| Method | Runs in | Attackers blocked | Benign flows blocked | Onset -> block mean (max) |
|---|---|---|---|---|
| P4-DL Shield (per-flow MLP, thr 0.7) | switch+controller | 41/41 | 1/424 (proto 17 dport 5001) | 0.20 (1.17) s |
| NetBeacon-style trees (3x7 RF per phase, 2^k phases) | switch | 41/41 | 31/487, all bursty UDP 5001 (1 per run) | 0.12 (0.18) s |
| Entropy P4DDoS/Euclid style (w=0.05 s, a=0.05, k=1.32) | switch | 41/41 | 243/487 (ICMP 86, UDP 5001 62, TCP 5002 31, replies) | 0.12 (0.19) s |
| ACMS asymmetry, P4RTHENON (dt=1 s, thr 177) | switch | 36/41 | 89/487 (ICMP 45, TCP 5002 31, UDP 5001 13) | 3.9 (56.7) s |
| DOE-DTL (ELM L=100 C=100, tau 401400 B, Smt 300/0.2 s) | switch+controller | 18/41 | 15/487 (ICMP from attacker hosts) | 10.9 (39.2) s |
- NetBeacon-style per-packet F1 0.872 +- 0.074 over 3 seeds (0.79-0.92); 4-tuple key gives the same flow-level result.
- DOE-DTL misses S2 (0/15, 800 pps each below Smt), 50 ms bursts (0/3), mimicry (0/3), period 500 ms (2 of 3 missed);
  network alarm before onset in 4 of 31 runs. Benign totals differ (424 vs 487): separate runs of the same scenarios.
  Benign flow sets: identical for UDP 5001 (62), TCP 5002 (31), TCP ephemeral (31); gap only in UDP to ephemeral ports
  (per-packet 201 / per-flow 159) and ICMP (162 / 141). Times in Table XV: mean over 41 attackers, all variants, to first
  positive summary (50 ms bursts 0.84 s; others 0.13-0.17 s); Table V 0.177 s = live S1 A2, to rule ack.
- Full output: results_summaries/switch_baselines_output.txt. Paper: Sec. VI-E12, Table XV.

### 11.15 Deployed classifier under fixed-point inference (analysis/quantize_eval.py)
BN folded into the linear layers (11,649 parameters); ranges calibrated on validation (99.99 pct); 31 test runs.
Check: recomputed float prob vs live-logged prob: median |diff| 3.5e-8, 0 decisions differ.
- 8-bit weights+activations (11.6 kB): 12 of 771,161 decisions change; F1 0.9969 (float 0.9968); flow level unchanged
  (41/41 attackers, 1/424 benign, 0.20 s). 6 bit: 92 change, flow level unchanged. 4 bit: 425 change, 34/424 benign blocked.
- Deployed single model F1 0.997 vs Table IX 0.988 +- 0.004 (mean of 3 retrained seeds).
- Full output: results_summaries/quantize_eval_output.txt. Paper: Sec. VI-E12 (paragraph), Sec. VII.

### 11.16 Benchmark reproducibility details (checked against the scripts)
- CICDDoS2019: 196,965 = 03-11 part of the cic_sample.py sample (all benign + ~20,000 attack per file, sampling seed 1);
  no rows dropped afterwards (inf -> NaN -> training-set median). 56,965 benign = 28.9 % of the same set.
- LR-HR: after drop_duplicates 43,877 rows (42,890 benign, 987 attack). Seeds 0-4 = split seed = model seed.
  Every split: train 26,326 (592 attack), val 8,775 (197), test 8,776 (198). MLP = scikit-learn MLPClassifier.
- SDN-SlowRate: sample drawn once (np.random.default_rng(0)) per scenario and (port label x folder label) group, <= 20,000 each;
  P4 split fixed (T4000 first 80 % by time / last 20 % / T1200). Seeds 0-2 change only model init and training order.
- CIC 0.999 (37 stats) vs 0.910 (7 live-mapped): only difference is the statistics the switch does not compute; LR reaches
  0.999 with them. Benchmarks retrain a model per dataset; the deployed model is never applied to them.

### 11.17 Benign-flow count check (400 flows, 36 runs)
- Runs: S0 5 + S1 A1-A3 15 + S2 5 + S7 3 + S9 guard 5 + S12 3 = 36. Flow = distinct (src, dst, proto, dport) in timing.csv,
  same code for all scenarios; blocked = a flow or source rule installed on that flow's own message.
- S9 mitigate: source rules (src_installed) only on 10.0.3.1 (h31, flood host) in all 5 runs; benign hosts are h11, h12, h21,
  h42 and the victim's replies -> no benign flow dropped by a source rule.

### 11.9 Limitations to state
Software switch (BMv2) on one laptop: no line-rate or hardware-portability claims; ~5k digests/s controller ceiling;
the 50 ms burst variant is the weakest; the S5 load background is flood-like and is used for load numbers only.
