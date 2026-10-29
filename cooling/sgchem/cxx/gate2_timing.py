#!/usr/bin/env python3
"""Dedicated CVODE-vs-DVODE wall-time benchmark.

The gate-(c) certification zones run in ~1-6 ms each, which is clock-granularity
and process-noise dominated -- not a usable timing number.  Here each
representative zone is driven with 2000 hydro steps and each measurement is
repeated NREP times; we report the MINIMUM (least-contaminated run), which is
the standard practice for short CPU benchmarks on a shared login node.

The measured interval is the integrator only (onezone3 --wall accumulates the
time inside EVOLVE_ABUNDANCES / cxx_evolve_abundances), so process startup,
COOLINMO table construction and the per-step calc_photo call are excluded.
"""
import subprocess, os, sys, csv

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "/orion/ptmp/uli/orion/fb_tests/_sgchem_cxx_build/gate2_data"
ONEZONE = "/orion/ptmp/uli/orion/stella/tests/onezone"
BIN = "/orion/ptmp/uli/orion/fb_tests/_sgchem_cxx_build/bin"
CWD = {5: os.path.join(ONEZONE, "build_net5"), 17: os.path.join(ONEZONE, "build_net17")}
DL_CM = 3.086e19
ZSTD, G0STD, ZETASTD = 0.1, 1.7, 1.8e-17
NREP = 3
NSTEPS = 2000

# representative regimes across the certification domain
ZONES = [
    ("diffuse_warm",  1e4, 1e-2,  30.0),
    ("wnm",           8e3, 1e0,   30.0),
    ("cnm",           1e2, 1e2,   30.0),
    ("molecular",     3e1, 1e4,   10.0),
    ("dense_cold",    3e1, 1e6,    3.0),
    ("shocked_hot",   1e6, 1e2,   10.0),
    ("tnochem_hot",   1e7, 1e1,   10.0),
    ("sn_bubble",     1e8, 1e0,   10.0),
]


def timed(net, backend, T0, nH, te):
    exe = os.path.join(BIN, "onezone3_net%d" % net)
    cmd = [exe, "--backend", backend, "--stats", "--wall",
           repr(T0), repr(nH), repr(ZSTD), repr(G0STD), repr(ZETASTD),
           repr(te), str(NSTEPS), repr(DL_CM), "0", str(NSTEPS)]
    best, stats = float('inf'), None
    for _ in range(NREP):
        p = subprocess.run(cmd, cwd=CWD[net], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True,
                           timeout=1800)
        if p.returncode != 0:
            return float('nan'), p.returncode, None
        w = float('nan')
        for l in p.stdout.splitlines():
            if l.startswith('WALL,'):
                w = float(l.split(',')[1])
        for l in p.stderr.splitlines():
            if l.startswith('STATS,'):
                stats = l.strip().split(',')
        if w == w and w < best:
            best = w
    return best, 0, stats


fn = os.path.join(OUT, "gate2_timing.csv")
with open(fn, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['net', 'zone', 'T0', 'nH', 'nsteps',
                'wall_dvode_s', 'wall_cvode_s', 'speedup',
                'us_per_step_dvode', 'us_per_step_cvode',
                'cvode_substeps', 'cvode_retries', 'cvode_rhs_evals'])
    print("%-4s %-14s %10s %10s %11s %11s %8s %10s %10s" %
          ("net", "zone", "T0", "nH", "DVODE s", "CVODE s", "x", "us/step_D", "us/step_C"))
    tot = {5: [0.0, 0.0], 17: [0.0, 0.0]}
    for net in (5, 17):
        for (tag, T0, nH, te) in ZONES:
            wd, rcd, _ = timed(net, 'f77', T0, nH, te)
            wc, rcc, st = timed(net, 'cxx', T0, nH, te)
            sp = wd / wc if wc and wc == wc else float('nan')
            tot[net][0] += wd; tot[net][1] += wc
            sub = ret = rhs = ''
            if st and len(st) >= 7:
                sub, ret, rhs = st[3], st[4], st[5]
            w.writerow([net, tag, '%.4e' % T0, '%.4e' % nH, NSTEPS,
                        '%.4f' % wd, '%.4f' % wc, '%.3f' % sp,
                        '%.2f' % (1e6 * wd / NSTEPS), '%.2f' % (1e6 * wc / NSTEPS),
                        sub, ret, rhs])
            f.flush()
            print("%-4d %-14s %10.2e %10.2e %11.4f %11.4f %8.2f %10.2f %10.2f" %
                  (net, tag, T0, nH, wd, wc, sp,
                   1e6 * wd / NSTEPS, 1e6 * wc / NSTEPS), flush=True)
    print()
    for net in (5, 17):
        print("net %-3d TOTAL  DVODE %.4f s   CVODE %.4f s   speedup x%.2f"
              % (net, tot[net][0], tot[net][1], tot[net][0] / tot[net][1]))
print("wrote", fn)
