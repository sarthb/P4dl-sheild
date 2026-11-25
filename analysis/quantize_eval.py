#!/usr/bin/env python3
"""
quantize_eval.py -- could the deployed classifier run inside the switch? (Reviewer 1, Comment 1.4; offline only)

In-switch neural-network systems run small models in fixed-point integer arithmetic: Pegasus (SIGCOMM 2025) keeps
full-precision weights with fixed-point activations; Quark (2025) and similar designs quantize weights and activations
to 8 bits or fewer. This script takes the DEPLOYED per-flow model (models/ddos_flow.pth + scaler), folds its batch-norm
layers into the linear layers (as a switch would store it), and simulates fixed-point inference:

  float     the deployed model (reference; checked against the probabilities logged live by the controller)
  A<b>      full-precision weights, b-bit activations and inputs              (Pegasus-style)
  W<b>A<b>  b-bit weights (per output neuron) and b-bit activations and inputs (Quark-style integer inference)

Quantization ranges are calibrated on the validation runs only (99.99th percentile of each layer's values); the test
runs are the same 31 detection-only runs of unseen variants as Table IX/XV. Decisions use the deployed threshold.
Reported: per-summary F1/AUC/EER, summaries whose decision changes, flow-level blocking (first positive summary,
the deployed policy), and model size at b bits.

  cd ~/MTP1-main && ~/p4env/bin/python3 revision/analysis/quantize_eval.py --out ~/quantize_eval
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_score, recall_score, confusion_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="results_flow")
ap.add_argument("--val", default="T*/rep4")
ap.add_argument("--test", default="S1_single_A2_detect/rep*,S2_distributed_detect/rep*,S3_*/rep*,S4_*/rep*")
ap.add_argument("--model", default="models/ddos_flow.pth")
ap.add_argument("--policy", default="models/block_policy_flow.json")
ap.add_argument("--window-s", type=float, default=0.05)
ap.add_argument("--bits", default="16,8,6,4")
ap.add_argument("--pct", type=float, default=99.99, help="calibration percentile for ranges")
ap.add_argument("--out", required=True)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
LOG = open(f"{A.out}/summary.txt", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
FEATS = ["protocol", "tag", "pkt_len", "flow_pkts", "flow_bytes", "flow_dur", "flow_pps", "iat", "win_pkts"]


# ------------------------------------------------------------------ data (same feature code as flow_offline.py)
def onset_by_src(d):
    try:
        man = json.load(open(f"{d}/manifest.json")); sc = json.load(open(f"{d}/scenario.json"))
    except Exception:
        return {}
    hosts = [t["host"] for t in sc["traffic"] if t.get("attack")]; out = {}
    for f in glob.glob(f"{d}/send_attack_*.csv"):
        h = f.split("send_attack_")[1][:-4]
        try:
            ev = pd.read_csv(f); fp = ev[ev.event == "first_packet"].t
            a, b, c, e = (int(x) for x in man["attack_flows"][hosts.index(h)]["src"].split("."))
            out[(a << 24) | (b << 16) | (c << 8) | e] = float(fp.iloc[0])
        except Exception:
            pass
    return out

def load(globs):
    parts = []
    for g in globs.split(","):
        for d in sorted(glob.glob(os.path.join(A.root, g.strip()))):
            f = f"{d}/ctrl_logs/flow_msgs.csv"
            if not os.path.exists(f) or os.path.getsize(f) == 0: continue
            x = pd.read_csv(f, low_memory=False)
            x["run"] = d; x["scenario"] = os.path.basename(os.path.dirname(d)); x["onset"] = x.src_ip.map(onset_by_src(d))
            parts.append(x)
    if not parts: raise SystemExit(f"no flow_msgs.csv under {A.root} for {globs}")
    x = pd.concat(parts, ignore_index=True)
    x["flow_key"] = x.src_ip.astype(str) + "|" + x.dst_ip.astype(str) + "|" + x.proto.astype(str) + "|" + x.dport.astype(str)
    w = x[x.msg_type == 1].copy()
    wp = w.win_pkts.clip(lower=1); dur = (w.win_last_us - w.flow_first_us).clip(lower=0) / 1e6
    w["protocol"] = w.proto; w["tag"] = w.tag_max; w["pkt_len"] = w.win_bytes / wp
    w["flow_pkts"] = w.cum_pkts; w["flow_bytes"] = w.cum_bytes; w["flow_dur"] = dur
    w["flow_pps"] = w.cum_pkts / np.maximum(dur, 1e-3)
    w["iat"] = np.where(w.win_pkts > 1, (w.win_last_us - w.win_first_us) / 1e6 / (w.win_pkts - 1).clip(lower=1), A.window_s)
    return x, w


# ------------------------------------------------------------------ model: load, fold batch norm
import torch, joblib
ck = torch.load(A.model, map_location="cpu", weights_only=False)
sd = {k.replace("net.", ""): v.numpy().astype(np.float64) for k, v in ck["model_state_dict"].items() if hasattr(v, "numpy")}
sc = joblib.load(A.model.replace(".pth", "_scaler.pkl"))
THR = float(json.load(open(A.policy))["threshold"])
assert list(ck.get("features", FEATS)) == FEATS, ck.get("features")

def fold(lin, bn=None, eps=1e-5):
    W, b = sd[f"{lin}.weight"], sd[f"{lin}.bias"]
    if bn is None: return W, b
    g, be, mu, var = sd[f"{bn}.weight"], sd[f"{bn}.bias"], sd[f"{bn}.running_mean"], sd[f"{bn}.running_var"]
    s = g / np.sqrt(var + eps)
    return W * s[:, None], (b - mu) * s + be
LAYERS = [fold("0", "1"), fold("4", "5"), fold("8"), fold("10")]      # Linear-BN-ReLU-Drop x2, Linear-ReLU, Linear
RELU = [True, True, True, False]
n_params = sum(W.size + b.size for W, b in LAYERS)

def forward(X, wq=None, aq=None, ranges=None, collect=False):
    """X: standardised inputs. wq/aq: bits for weights/activations (None = float). ranges: per-layer input ranges."""
    h = X; acts = [h]
    for i, ((W, b), r) in enumerate(zip(LAYERS, RELU)):
        if aq is not None:                      # quantize this layer's input (inputs signed, post-ReLU unsigned)
            lo = 0.0 if i > 0 else -ranges[i]; hi = ranges[i]
            levels = (2 ** aq - 1) if i > 0 else (2 ** (aq - 1) - 1)
            step = hi / levels
            h = np.clip(np.round(h / step), lo / step, levels) * step
        if wq is not None:                      # per-output-neuron symmetric weight quantization
            m = np.abs(W).max(axis=1, keepdims=True) + 1e-12; q = 2 ** (wq - 1) - 1
            W = np.round(W / m * q) * m / q
        h = h @ W.T + b
        if r: h = np.maximum(h, 0)
        acts.append(h)
    z = h.ravel()
    return (z, acts) if collect else z

def sig(z): return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------ metrics
def eer(y, s):
    f, tp, _ = roc_curve(y, s); n = 1 - tp; i = np.nanargmin(np.abs(n - f)); return float((f[i] + n[i]) / 2)

def mets(y, p_, thr):
    p = (p_ >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(f1=f1_score(y, p, zero_division=0), prec=precision_score(y, p, zero_division=0),
                rec=recall_score(y, p, zero_division=0), fpr=fp / max(fp + tn, 1), auc=roc_auc_score(y, p_), eer=eer(y, p_),
                FP=int(fp), FN=int(fn))

def flow_level(allmsgs, w, prob, thr):
    pos = w.assign(p=prob)[prob >= thr].groupby(["run", "flow_key"]).t_rx.min()
    f = allmsgs.groupby(["run", "flow_key"]).agg(y=("label", "max"), onset=("onset", "first")).reset_index()
    f["t_block"] = pd.Series(list(zip(f.run, f.flow_key))).map(pos.to_dict()).values
    b, a = f[f.y == 0], f[f.y == 1]; d = a.t_block - a.onset
    return dict(benign_blocked=f"{int(b.t_block.notna().sum())}/{len(b)}",
                attackers_blocked=f"{int(a.t_block.notna().sum())}/{len(a)}",
                onset_to_block_mean=round(float(d.mean()), 3), onset_to_block_max=round(float(d.max()), 3))


# ================================================================== run
va_all, va = load(A.val); te_all, te = load(A.test)
Xva = sc.transform(va[FEATS].values.astype(np.float64)); Xte = sc.transform(te[FEATS].values.astype(np.float64))
yte = te.label.values.astype(int)
say(f"test: {len(te):,} window summaries from {te.run.nunique()} runs ({yte.mean():.2%} attack); validation (calibration) {len(va):,}")
say(f"model: 9-128-64-32-1 MLP, batch norm folded -> {n_params:,} parameters; deployed threshold {THR}")

zf = forward(Xte); pf = sig(zf)
if "prob" in te:
    d = np.abs(pf - te.prob.values.astype(float))
    say(f"check: recomputed float probability vs probability logged live by the controller: "
        f"median |diff| {np.median(d):.2e}, max {d.max():.2e}, decisions that differ {int(((pf >= THR) != (te.prob.values >= THR)).sum())}")

# calibration ranges from validation runs only
_, acts = forward(Xva, collect=True)
ranges = [float(np.percentile(np.abs(a), A.pct)) for a in acts[:-1]]
say("calibrated input ranges per layer (validation, %.2f pct): %s" % (A.pct, ", ".join(f"{r:.2f}" for r in ranges)))

rows = []
def run(name, wq, aq, bits):
    p = pf if (wq is None and aq is None) else sig(forward(Xte, wq, aq, ranges))
    m = mets(yte, p, THR)
    m.update(flow_level(te_all, te, p, THR))
    size = n_params * (32 if wq is None else bits) / 8
    rows.append(dict(model=name, weight_bits=("float" if wq is None else wq), act_bits=("float" if aq is None else aq),
                     model_bytes=int(size), decisions_changed=int(((p >= THR) != (pf >= THR)).sum()), **m))
run("float (deployed)", None, None, 32)
for b in [int(v) for v in A.bits.split(",")]:
    run(f"A{b} (Pegasus-style)", None, b, b)
    run(f"W{b}A{b} (integer)", b, b, b)
R = pd.DataFrame(rows); R.to_csv(f"{A.out}/quantize_eval.csv", index=False)
say("\n=== DEPLOYED CLASSIFIER UNDER FIXED-POINT INFERENCE (31 test runs; flow level = first positive summary)\n"
    + R.round(4).to_string(index=False))
say("\nmodel_bytes: weights and biases at the given width (float = 32-bit, as stored by PyTorch). Activations need one register per neuron.")
