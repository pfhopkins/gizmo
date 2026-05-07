#!/usr/bin/env python3
"""Phase 0 PHASE0_NGL / PHASE0_GHOST log summarizer.

Drives Phase 0 decision rows 1+2 (Nactive histogram + tiny-N phase-cost
breakdown) and feeds row 6 (Mode B caller-target selection by histogram).

Usage: phase0_summarize.py <log file> [--rank R] [--caller NAME]
"""

import argparse
import re
import sys
from collections import defaultdict
from statistics import median

# log-spaced N bins (Nactive)
N_BINS = [(0, 0), (1, 1), (2, 4), (5, 16), (17, 64), (65, 256),
          (257, 1024), (1025, 4096), (4097, 16384), (16385, 65536),
          (65537, 262144), (262145, 1048576), (1048577, 10**12)]

NGL_RE = re.compile(
    r"^PHASE0_NGL rank=(\d+) call=(\d+) caller=(\S+) mode=0x([0-9a-fA-F]+) "
    r"cache=(\d+)(?: sidx_id=(\S+))? N=(\d+) Ntot=(\d+) "
    r"dt_ghost_import=(\S+) dt_sidx_dec=(\S+) dt_refresh=(\S+) "
    r"dt_gpu=(\S+) total_pairs=(\d+)")

GHOST_RE = re.compile(
    r"^PHASE0_GHOST rank=(\d+) call=(\d+) caller=(\S+) impl=(\S+) "
    r"nlocal_pre=(\d+) ghost_added=(-?\d+) ntotal_post=(\d+) "
    r"dt_ghost_import=(\S+)")


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
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--caller", default=None)
    args = ap.parse_args()

    ngl_rows = []
    ghost_rows = []
    with open(args.log, "r", errors="replace") as f:
        for line in f:
            m = NGL_RE.match(line)
            if m:
                rank, call, caller, mode, cache, sidx_id, N, Ntot, dghi, dsdec, dref, dgpu, pairs = m.groups()
                row = dict(rank=int(rank), call=int(call), caller=caller,
                           mode=int(mode, 16), cache=int(cache),
                           sidx_id=sidx_id or "?", N=int(N),
                           Ntot=int(Ntot), dt_ghost_import=float(dghi),
                           dt_sidx_dec=float(dsdec), dt_refresh=float(dref),
                           dt_gpu=float(dgpu), pairs=int(pairs))
                if args.rank is not None and row["rank"] != args.rank:
                    continue
                if args.caller is not None and row["caller"] != args.caller:
                    continue
                ngl_rows.append(row)
                continue
            m = GHOST_RE.match(line)
            if m:
                rank, call, caller, impl, nlpre, gadd, ntpost, dghi = m.groups()
                row = dict(rank=int(rank), call=int(call), caller=caller,
                           impl=impl, nlocal_pre=int(nlpre),
                           ghost_added=int(gadd), ntotal_post=int(ntpost),
                           dt_ghost_import=float(dghi))
                if args.rank is not None and row["rank"] != args.rank:
                    continue
                if args.caller is not None and row["caller"] != args.caller:
                    continue
                ghost_rows.append(row)

    print(f"# Parsed {len(ngl_rows)} PHASE0_NGL rows, {len(ghost_rows)} PHASE0_GHOST rows")
    if not ngl_rows and not ghost_rows:
        print("# No PHASE0 lines in log. Did you set GIZMO_PHASE0_DIAG=1?")
        sys.exit(0)

    # ---- 1. Nactive histogram + per-bin cost breakdown ----
    print("\n## Row 1+2: Nactive histogram + per-bin cost breakdown (all callers, all ranks)")
    print(f"{'N range':<16} {'count':>6}  "
          f"{'tot_med':>9} {'tot_p90':>9}  "
          f"{'sidx_dec_med':>13} {'sidx_dec_p90':>13}  "
          f"{'refresh_med':>11} {'gpu_med':>9}")
    by_bin = defaultdict(list)
    for r in ngl_rows:
        by_bin[n_bin_label(r["N"])].append(r)
    for label in sorted(by_bin.keys()):
        rs = by_bin[label]
        tot = [r["dt_sidx_dec"] + r["dt_refresh"] + r["dt_gpu"] for r in rs]
        sdec = [r["dt_sidx_dec"] for r in rs]
        ref = [r["dt_refresh"] for r in rs]
        gpu = [r["dt_gpu"] for r in rs]
        print(f"{label:<16} {len(rs):>6}  "
              f"{fmt_us(median(tot))} {fmt_us(percentile(tot, 90))}  "
              f"{fmt_us(median(sdec)):>13} {fmt_us(percentile(sdec, 90)):>13}  "
              f"{fmt_us(median(ref)):>11} {fmt_us(median(gpu)):>9}")

    # ---- 2. Per-caller breakdown ----
    print("\n## Row 6 input: per-caller call counts + Nactive distribution")
    print(f"{'caller':<28} {'calls':>6}  {'N_min':>6} {'N_med':>6} {'N_max':>7}  {'tot_med':>9} {'tot_sum':>9}")
    by_caller = defaultdict(list)
    for r in ngl_rows:
        by_caller[r["caller"]].append(r)
    for caller in sorted(by_caller.keys(), key=lambda c: -sum(r["dt_sidx_dec"] + r["dt_refresh"] + r["dt_gpu"] for r in by_caller[c])):
        rs = by_caller[caller]
        Ns = sorted(r["N"] for r in rs)
        tot = [r["dt_sidx_dec"] + r["dt_refresh"] + r["dt_gpu"] for r in rs]
        print(f"{caller:<28} {len(rs):>6}  "
              f"{Ns[0]:>6} {Ns[len(Ns)//2]:>6} {Ns[-1]:>7}  "
              f"{fmt_us(median(tot))} {fmt_us(sum(tot))}")

    # ---- 3. Ghost-import per caller ----
    if ghost_rows:
        print("\n## PHASE0_GHOST: per-caller dt_ghost_import")
        print(f"{'caller':<28} {'calls':>6}  {'med':>9} {'p90':>9} {'sum':>9}  {'avg_added':>10}")
        gby = defaultdict(list)
        for r in ghost_rows:
            gby[r["caller"]].append(r)
        for caller in sorted(gby.keys(), key=lambda c: -sum(x["dt_ghost_import"] for x in gby[c])):
            rs = gby[caller]
            ds = [r["dt_ghost_import"] for r in rs]
            adds = [r["ghost_added"] for r in rs]
            print(f"{caller:<28} {len(rs):>6}  "
                  f"{fmt_us(median(ds))} {fmt_us(percentile(ds, 90))} {fmt_us(sum(ds))}  "
                  f"{sum(adds) / len(adds):>10.0f}")

    # ---- 4. "No hidden global work" assertion (Mode B target) ----
    print("\n## Row 3 assertions (under Mode B these MUST be 0):")
    sus_sidx = sum(1 for r in ngl_rows if r["dt_sidx_dec"] > 1e-4 and r["N"] < 100)
    sus_gpu = sum(1 for r in ngl_rows if r["dt_gpu"] > 0 and r["N"] == 0)
    sus_ghost_tinyN_callers = set()
    print(f"  PHASE0_NGL calls with N<100 AND sidx_dec > 100us : {sus_sidx}  "
          f"(Mode B target: 0 for tiny-N callers)")
    print(f"  PHASE0_NGL calls with N=0 AND dt_gpu>0           : {sus_gpu}  "
          f"(should be 0 already from early-out)")
    print(f"  PHASE0_GHOST total calls                         : {len(ghost_rows)}  "
          f"(Mode B target: tiny-N steps don't trigger ghost)")


if __name__ == "__main__":
    main()
