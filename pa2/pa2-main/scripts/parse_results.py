#!/usr/bin/env python3
"""Collect the ROI statistics of every run into one CSV.

    python3 scripts/parse_results.py <results_dir> <out.csv>

<results_dir>/<tag>/<trace>.txt is one ChampSim run (scripts/run_one.sh).
"""
import csv
import os
import re
import sys

FIELDS = [
    "tag", "trace", "instructions", "cycles", "ipc",
    "l1d_load_access", "l1d_load_miss", "l1d_load_mpki",
    "l1d_pf_issued", "l1d_pf_useful", "l1d_pf_lower", "l1d_pf_accuracy",
    "l2c_load_miss", "l2c_load_mpki", "l2c_data_load_mpki", "l2c_instr_load_mpki", "l2c_total_mpki",
    "l2c_pf_useful", "l2c_pf_useless",
    "llc_total_mpki",
    "dyn_irrip_fraction", "dyn_mean_ratio", "dyn_pf_fills", "dyn_pf_first_use",
    "spp_epochs",
]

NUM = r"([-+0-9.eE]+|-?nan)"


def parse(path):
    text = open(path, errors="replace").read()
    roi = text.split("Region of Interest Statistics")[-1]
    r = {}

    m = re.search(r"CPU 0 cumulative IPC: " + NUM + r" instructions: (\d+) cycles: (\d+)", roi)
    r["ipc"], r["instructions"], r["cycles"] = float(m.group(1)), int(m.group(2)), int(m.group(3))

    def access_line(cache, kind):
        m = re.search(r"^%s %s\s+ACCESS:\s+(\d+)\s+HIT:\s+(\d+)\s+MISS:\s+(\d+).*MPKI: %s" % (cache, kind, NUM), roi, re.M)
        return (int(m.group(1)), int(m.group(3)), float(m.group(4))) if m else (0, 0, 0.0)

    r["l1d_load_access"], r["l1d_load_miss"], r["l1d_load_mpki"] = access_line("L1D", "LOAD     ")
    _, r["l2c_load_miss"], r["l2c_load_mpki"] = access_line("L2C", "LOAD     ")
    _, _, r["l2c_total_mpki"] = access_line("L2C", "TOTAL    ")
    _, _, r["llc_total_mpki"] = access_line("LLC", "TOTAL    ")

    m = re.search(r"^L2C DATA LOAD MPKI: " + NUM, roi, re.M)
    r["l2c_data_load_mpki"] = float(m.group(1))
    m = re.search(r"^L2C INSTRUCTION LOAD MPKI: " + NUM, roi, re.M)
    r["l2c_instr_load_mpki"] = float(m.group(1))

    m = re.search(r"^L1D PREFETCH\s+REQUESTED:\s+(\d+)\s+ISSUED:\s+(\d+)", roi, re.M)
    r["l1d_pf_issued"] = int(m.group(2))
    m = re.search(r"^L1D USEFUL LOAD PREFETCHES:\s+(\d+) PREFETCH ISSUED TO LOWER LEVEL:\s+(\d+)", roi, re.M)
    r["l1d_pf_useful"], r["l1d_pf_lower"] = int(m.group(1)), int(m.group(2))
    r["l1d_pf_accuracy"] = 100.0 * r["l1d_pf_useful"] / r["l1d_pf_lower"] if r["l1d_pf_lower"] else ""

    m = re.search(r"^L2C PREFETCH\s+REQUESTED:.*USEFUL:\s+(\d+)\s+USELESS:\s+(\d+)", roi, re.M)
    r["l2c_pf_useful"], r["l2c_pf_useless"] = (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    m = re.search(r"DYN_IRRIP decisions: \d+ irrip_decisions: \d+ irrip_fraction: " + NUM + " mean_ratio: " + NUM, text)
    r["dyn_irrip_fraction"], r["dyn_mean_ratio"] = (float(m.group(1)), float(m.group(2))) if m else ("", "")

    m = re.search(r"DYN_IRRIP pf_fills: (\d+) pf_first_use: (\d+)", text)
    r["dyn_pf_fills"], r["dyn_pf_first_use"] = (int(m.group(1)), int(m.group(2))) if m else ("", "")

    m = re.search(r"SPP epochs per degree:(.*)", text)
    r["spp_epochs"] = m.group(1).strip() if m else ""
    return r


def main():
    rdir, out = sys.argv[1], sys.argv[2]
    rows = []
    for tag in sorted(os.listdir(rdir)):
        d = os.path.join(rdir, tag)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".txt"):
                continue
            try:
                r = parse(os.path.join(d, f))
            except Exception as e:  # incomplete run
                print("skip %s/%s: %s" % (tag, f, e), file=sys.stderr)
                continue
            r["tag"], r["trace"] = tag, f[:-4]
            rows.append(r)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("%d runs -> %s" % (len(rows), out))


if __name__ == "__main__":
    main()
