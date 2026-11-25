#!/usr/bin/env python3
"""
cic_sample.py — shrink CIC-DDoS2019 (30 GB of CSVs) to a small, representative sample.
Run on YOUR machine (needs only pandas):

    python3 cic_sample.py --root /path/to/cic-ddos2019 --out cic2019_sample.csv.gz
    python3 cic_sample.py --root /path/to/archive.zip  --out cic2019_sample.csv.gz   (no need to unzip)

  * walks --root recursively and reads every .csv in chunks (low memory)
  * keeps ALL benign rows (benign is very rare in CIC-DDoS2019)
  * keeps a random sample of attack rows from every file (--attack-per-file, default 20000)
  * keeps the columns needed for evaluation plus Timestamp, Source IP and Label
  * adds `day` (01-12 = training day, 03-11 = testing day, from the folder/file name) and `file`
Output is typically 30–100 MB gzipped. Attach that file in the chat.
"""
import argparse, os, re, random, zipfile
import pandas as pd

KEEP = ["Timestamp", "Source IP", "Destination IP", "Source Port", "Destination Port", "Protocol",
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Fwd Packet Length Mean", "Bwd Packet Length Mean", "Packet Length Mean", "Packet Length Std",
        "Min Packet Length", "Max Packet Length", "Average Packet Size",
        "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Mean", "Bwd IAT Mean", "Fwd Packets/s", "Bwd Packets/s",
        "SYN Flag Count", "ACK Flag Count", "URG Flag Count", "Down/Up Ratio",
        "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
        "Active Mean", "Idle Mean", "Label"]

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--out", default="cic2019_sample.csv.gz")
ap.add_argument("--attack-per-file", type=int, default=20000)
ap.add_argument("--chunksize", type=int, default=500000); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
random.seed(a.seed)

# --root may be a folder (with .csv files and/or .zip archives anywhere inside) or a single .zip:
# CSVs inside zip archives are read directly, without extracting.
def find_sources(root):
    srcs = []
    paths = [root] if os.path.isfile(root) else [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]
    for p in sorted(paths):
        if p.lower().endswith(".csv"):
            srcs.append((p, None))
        elif p.lower().endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                srcs += [(p, m) for m in sorted(z.namelist()) if m.lower().endswith(".csv")]
    return srcs


def reader(src, **kw):
    path, member = src
    if member is None:
        return pd.read_csv(path, **kw)
    return pd.read_csv(zipfile.ZipFile(path).open(member), **kw)


files = find_sources(a.root)
print(f"{len(files)} csv files found (inside zips: {sum(m is not None for _, m in files)})")
parts, report = [], []
for src in files:
    path = src[1] or src[0]
    day = "01-12" if re.search(r"01[-_]?12", path) else ("03-11" if re.search(r"03[-_]?11", path) else "unknown")
    n_rows = n_ben = 0; ben, att = [], []
    # first pass: count attack rows so we can sample at a fixed rate
    for ch in reader(src, usecols=lambda c: c.strip() == "Label", chunksize=a.chunksize,
                          low_memory=False, encoding_errors="ignore"):
        ch.columns = [c.strip() for c in ch.columns]
        n_rows += len(ch); n_ben += int((ch.Label.astype(str).str.upper() == "BENIGN").sum())
    rate = min(1.0, a.attack_per_file / max(n_rows - n_ben, 1))
    for ch in reader(src, usecols=lambda c: c.strip() in KEEP, chunksize=a.chunksize,
                          low_memory=False, encoding_errors="ignore"):
        ch.columns = [c.strip() for c in ch.columns]
        isb = ch.Label.astype(str).str.upper() == "BENIGN"
        ben.append(ch[isb]); att.append(ch[~isb].sample(frac=rate, random_state=a.seed))
    d = pd.concat(ben + att); d["day"] = day; d["file"] = os.path.basename(path)
    parts.append(d)
    report.append((os.path.basename(path), day, n_rows, n_ben, len(d)))
    print(f"{os.path.basename(path):40s} day={day} rows={n_rows:>10d} benign={n_ben:>8d} kept={len(d)}")
out = pd.concat(parts, ignore_index=True)
out.to_csv(a.out, index=False, compression="gzip")
pd.DataFrame(report, columns=["file", "day", "rows_total", "benign_total", "rows_kept"]).to_csv(
    a.out.replace(".csv.gz", "_counts.csv"), index=False)
print(f"\nwrote {a.out}: {len(out)} rows; label counts:\n{out.Label.value_counts()}")
