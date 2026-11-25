#!/usr/bin/env python3
"""
dataset_inventory.py — describe a dataset folder (all subfolders) in one small text file.

    python3 dataset_inventory.py /path/to/sdn-slowrate-ddos            (Linux / macOS)
    python dataset_inventory.py "C:\\path\\to\\sdn-slowrate-ddos"       (Windows)

Writes dataset_inventory.txt next to this script. Upload that file in the chat.
For every file: its folder path, size, and for CSV/TXT files the first 3 lines and the line count.
Small README / .txt / .json / .md files are copied in full (up to 5,000 characters).
"""
import os, sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_inventory.txt")
with open(out, "w", encoding="utf-8") as w:
    w.write(f"ROOT: {os.path.abspath(root)}\n\n== FOLDER TREE ==\n")
    for d, subs, files in sorted(os.walk(root)):
        subs.sort()
        depth = os.path.relpath(d, root).count(os.sep) + (0 if d == root else 1)
        w.write("  " * depth + os.path.basename(d.rstrip("/\\")) + "/\n")
        for f in sorted(files):
            p = os.path.join(d, f)
            w.write("  " * (depth + 1) + f"{f}  ({os.path.getsize(p) / 1e6:.1f} MB)\n")
    w.write("\n== FILE CONTENTS ==\n")
    for d, _, files in sorted(os.walk(root)):
        for f in sorted(files):
            p = os.path.join(d, f); ext = f.lower().rsplit(".", 1)[-1]
            rel = os.path.relpath(p, root)
            try:
                if ext in ("csv", "tsv", "log", "dat"):
                    with open(p, errors="ignore") as r:
                        head = [next(r, "") for _ in range(3)]
                        n = 3 + sum(1 for _ in r)
                    w.write(f"\n--- {rel}  [{n} lines]\n" + "".join(head))
                elif ext in ("txt", "md", "json", "readme", "yaml", "yml", "ini", "cfg") \
                        and os.path.getsize(p) < 2_000_000:
                    w.write(f"\n--- {rel}\n" + open(p, errors="ignore").read(5000) + "\n")
                else:
                    w.write(f"\n--- {rel}  (binary/other, not read)\n")
            except Exception as e:
                w.write(f"\n--- {rel}  (could not read: {e})\n")
print("wrote", out)
