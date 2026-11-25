#!/usr/bin/env python3
"""
ieee_sample.py — shrink the IEEE DataPort "SDN-SlowRate-DDoS" dataset to a small sample, and compute
statistics on the FULL data that a sample cannot show (duplicates, identity leakage).
Run on YOUR machine (needs only pandas + numpy):

    python3 ieee_sample.py --root /path/to/sdn-slowrate-ddos --out ieee_sample.csv.gz

  * reads every .csv under --root (recursively) in chunks, so any size works
  * sample: up to --per-label-per-file rows of each label from each file (default 20000), all columns kept
  * full-data stats -> ieee_sample_stats.json:
      rows / label counts per file, exact duplicate rows, duplicate feature vectors,
      how well IP_Src / Mac_Src / IP_Dst alone predict the label (identity leakage), Timestamp range
Attach ieee_sample.csv.gz and ieee_sample_stats.json.
If the .gz is bigger than the upload limit, rerun with --per-label-per-file 5000.
"""
import argparse, os, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--out", default="ieee_sample.csv.gz")
ap.add_argument("--per-label-per-file", type=int, default=20000)
ap.add_argument("--label-col", default="Label"); ap.add_argument("--chunksize", type=int, default=500000)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
IDS = ["FlowID", "Timestamp"]                      # excluded when checking duplicate feature vectors
IDENT = ["IP_Src", "IP_Dst", "Mac_Src", "Mac_Dst", "Port_Src", "Port_Dst", "DeviceID"]

files = sorted(os.path.join(d, f) for d, _, fs in os.walk(a.root) for f in fs if f.lower().endswith(".csv"))
print(f"{len(files)} csv files")
stats = {"files": {}}; row_hash, feat_hash = [], []
ident_counts = {c: {} for c in IDENT}              # value -> [n_label0, n_label1]
parts = []; tmin = tmax = None

# pass 1: counts per label per file (to sample at a fixed rate) + full-data statistics
rates = {}
for p in files:
    cnt = {}
    for ch in pd.read_csv(p, chunksize=a.chunksize, low_memory=False, encoding_errors="ignore"):
        ch.columns = [c.strip() for c in ch.columns]
        lab = ch[a.label_col].astype(str)
        for k, v in lab.value_counts().items(): cnt[k] = cnt.get(k, 0) + int(v)
        row_hash.append(pd.util.hash_pandas_object(ch, index=False).values)
        fc = [c for c in ch.columns if c not in IDS]
        feat_hash.append(pd.util.hash_pandas_object(ch[fc], index=False).values)
        y = (pd.to_numeric(lab, errors="coerce").fillna(0) > 0).astype(int) if lab.str.isnumeric().all() \
            else (~lab.str.upper().isin(["0", "BENIGN", "NORMAL"])).astype(int)
        for c in IDENT:
            if c in ch:
                g = pd.crosstab(ch[c].astype(str), y)
                for v, r in g.iterrows():
                    e = ident_counts[c].setdefault(v, [0, 0]); e[0] += int(r.get(0, 0)); e[1] += int(r.get(1, 0))
        if "Timestamp" in ch:
            t = pd.to_numeric(ch.Timestamp, errors="coerce")
            if t.notna().any():
                tmin = t.min() if tmin is None else min(tmin, t.min()); tmax = t.max() if tmax is None else max(tmax, t.max())
    rates[p] = {k: min(1.0, a.per_label_per_file / v) for k, v in cnt.items()}
    stats["files"][os.path.basename(p)] = cnt
    print(os.path.basename(p), cnt)

rh = np.concatenate(row_hash); fh = np.concatenate(feat_hash)
stats["rows_total"] = int(len(rh))
stats["exact_duplicate_rows"] = int(len(rh) - len(np.unique(rh)))
stats["duplicate_rows_ignoring_FlowID_Timestamp"] = int(len(fh) - len(np.unique(fh)))
for c, dct in ident_counts.items():
    if not dct: continue
    tot = sum(x[0] + x[1] for x in dct.values()); pure = sum(max(x) for x in dct.values())
    stats[f"label_accuracy_from_{c}_alone"] = round(pure / max(tot, 1), 6)   # majority label per value
    stats[f"distinct_{c}"] = len(dct)
stats["timestamp_min"], stats["timestamp_max"] = (float(tmin), float(tmax)) if tmin is not None else (None, None)

# pass 2: sample
for p in files:
    for ch in pd.read_csv(p, chunksize=a.chunksize, low_memory=False, encoding_errors="ignore"):
        ch.columns = [c.strip() for c in ch.columns]
        lab = ch[a.label_col].astype(str)
        keep = [g.sample(frac=rates[p][k], random_state=a.seed) for k, g in ch.groupby(lab)]
        if keep:
            s = pd.concat(keep); s["file"] = os.path.basename(p); parts.append(s)
out = pd.concat(parts, ignore_index=True)
out.to_csv(a.out, index=False, compression="gzip")
stats["sample_rows"] = len(out)
json.dump(stats, open(a.out.replace(".csv.gz", "_stats.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in stats.items() if k != "files"}, indent=2))
print(f"wrote {a.out} ({len(out)} rows)")
