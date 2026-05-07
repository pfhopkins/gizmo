#!/usr/bin/env python3
"""Legacy DIAG_NGL log summarizer — same shape as phase0_summarize but
parses the existing rank-0-only DIAG_NGL lines for retrospective Phase 0
preview from runs predating PHASE0_DIAG instrumentation.

Limitations vs PHASE0_NGL:
  - rank-0 only (existing gate)
  - no call_id, no dt_ghost_import
  - NO multi-rank scaling info (row 5 still requires PHASE0 sweep)
"""

import argparse
import re
import sys
from collections import defaultdict
from statistics import median

DIAG_RE = re.compile(
    r"\[DIAG_NGL caller=(\S+) tbm=0x([0-9a-fA-F]+) N=(\d+) Ntot=(\d+) "
    r"pairs=(\d+) ovflw=(\d+) sidx_cached=(\d+)(?:\s+earlyout=\d+)?\] "
    r"sidx_dec=(\S+) refresh_lnch=(\S+) refresh_fnc=(\S+) "
    r"drain=(\S+) noop_lnch=(\S+) noop_fnc=(\S+) "
    r"fused_lnch=(\S+) fused_fnc=(\S+) scan=(\S+) "
    r"compact_lnch=(\S+) compact_fnc=(\S+) free=(\S+) total=(\S+)")

N_BINS = [(0, 0), (1, 1), (2, 4), (5, 16), (17, 64), (65, 256),
          (257, 1024), (1025, 4096), (4097, 16384), (16385, 65536),
          (65537, 262144), (262145, 1048576), (1048577, 10**12)]


def n_bin_label(n):
    for lo, hi in N_BINS:
        if lo <= n <= hi:
            return f"{lo:>7}-{hi:<7}" if lo != hi else f"{lo:>7}       "
    return "       ?"


def percentile(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def fmt_us(s):
    if s < 0:
        return "    n/a"
    if s < 1e-3:
        return f"{s * 1e6:6.1f}us"
    if s < 1.0:
        return f"{s * 1e3:6.2f}ms"
    return f"{s:6.3f}s "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--caller", default=None)
    args = ap.parse_args()

    rows = []
    with open(args.log, "r", errors="replace") as f:
        for line in f:
            m = DIAG_RE.search(line)
            if not m:
                continue
            (caller, mode, N, Ntot, pairs, ovflw, cached,
             sdec, rl, rf, drain, nl, nf, fl, ff, scan, cl, cf, free_, total) = m.groups()
            row = dict(caller=caller, mode=int(mode, 16), N=int(N), Ntot=int(Ntot),
                       pairs=int(pairs), cached=int(cached),
                       dt_sidx_dec=float(sdec),
                       dt_refresh=float(rl) + float(rf),
                       dt_gpu=float(fl) + float(ff),
                       dt_total=float(total))
            if args.caller and row["caller"] != args.caller:
                continue
            rows.append(row)

    print(f"# Parsed {len(rows)} DIAG_NGL rows (rank 0 only — legacy format)")
    if not rows:
        sys.exit(0)

    print("\n## Nactive histogram + per-bin cost (rank 0)")
    print(f"{'N range':<16} {'count':>6}  {'tot_med':>9} {'tot_p90':>9}  "
          f"{'sidx_dec_med':>13}  {'refresh_med':>11} {'gpu_med':>9}")
    by_bin = defaultdict(list)
    for r in rows:
        by_bin[n_bin_label(r["N"])].append(r)
    for label in sorted(by_bin.keys()):
        rs = by_bin[label]
        tot = [r["dt_total"] for r in rs]
        sdec = [r["dt_sidx_dec"] for r in rs]
        ref = [r["dt_refresh"] for r in rs]
        gpu = [r["dt_gpu"] for r in rs]
        print(f"{label:<16} {len(rs):>6}  "
              f"{fmt_us(median(tot))} {fmt_us(percentile(tot, 90))}  "
              f"{fmt_us(median(sdec)):>13}  "
              f"{fmt_us(median(ref)):>11} {fmt_us(median(gpu)):>9}")

    print("\n## Per-caller (rank 0)")
    print(f"{'caller':<28} {'calls':>6}  {'N_min':>6} {'N_med':>6} {'N_max':>7}  "
          f"{'tot_med':>9} {'tot_sum':>9}  {'sidx_dec_sum':>13}")
    by_caller = defaultdict(list)
    for r in rows:
        by_caller[r["caller"]].append(r)
    for caller in sorted(by_caller.keys(), key=lambda c: -sum(r["dt_total"] for r in by_caller[c])):
        rs = by_caller[caller]
        Ns = sorted(r["N"] for r in rs)
        tot = [r["dt_total"] for r in rs]
        sdec = [r["dt_sidx_dec"] for r in rs]
        print(f"{caller:<28} {len(rs):>6}  "
              f"{Ns[0]:>6} {Ns[len(Ns)//2]:>6} {Ns[-1]:>7}  "
              f"{fmt_us(median(tot))} {fmt_us(sum(tot))}  {fmt_us(sum(sdec)):>13}")


if __name__ == "__main__":
    main()
