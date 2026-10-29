#!/usr/bin/env python3
"""gate2_bench.py -- Gate-2 acceptance benchmark: C++/CVODE backend vs the
canonical Fortran/DVODE golden master, run through the SAME driver binary
(onezone3) so that everything except the integrator+RHS layer is identical.

Three gates (see GATE2_ACCEPTANCE.md):
  (a) equilibrium  : 20 points from the Tier-1 ladder grid; GATE |dT_eq| <= 1%
  (b) relaxation   : the fig6 trajectory cases; report max deviation
  (c) certification: the 80-zone (n,T0) DVODE domain grid per net;
                     GATE all zones complete, zero failures; report wall times

Usage: gate2_bench.py [a|b|c|all]   (default all)
Writes data/gate2_*.csv next to this script.
"""
import subprocess, os, sys, math, csv, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "/orion/ptmp/uli/orion/fb_tests/_sgchem_cxx_build/gate2_data"
ONEZONE = "/orion/ptmp/uli/orion/stella/tests/onezone"
BIN = "/orion/ptmp/uli/orion/fb_tests/_sgchem_cxx_build/bin"
CWD = {5: os.path.join(ONEZONE, "build_net5"), 17: os.path.join(ONEZONE, "build_net17")}

NETS = [5, 17]
NTRAC = {5: 3, 17: 7}
ZSTD, G0STD, ZETASTD = 0.1, 1.7, 1.8e-17
DL_CM = 3.086e19            # 10 pc shielding length
os.makedirs(OUT, exist_ok=True)


def logspace(a, b, n):
    return [10.0 ** (a + (b - a) * i / (n - 1)) for i in range(n)]


def run_zone(net, backend, T0, nH, Z, G0, zeta, te_Myr, nsteps, mode=0,
             stride=1, timeout=600):
    """Returns dict(status, wall, rows, lam, stats)."""
    exe = os.path.join(BIN, "onezone3_net%d" % net)
    cmd = [exe, "--backend", backend, "--stats", "--wall",
           repr(T0), repr(nH), repr(Z), repr(G0), repr(zeta),
           repr(te_Myr), str(nsteps), repr(DL_CM), str(mode), str(stride)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=CWD[net], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True,
                           timeout=timeout)
        rc = p.returncode
        out, err = p.stdout, p.stderr
        status = "OK" if rc == 0 else ("FATAL%d" % rc)
    except subprocess.TimeoutExpired:
        return dict(status="TIMEOUT", wall=timeout, rows=[], lam=None, solver_wall=float('nan'))
    wall = time.time() - t0

    rows, lam, solver_wall = [], None, float('nan')
    for l in out.splitlines():
        if l.startswith('#') or l.startswith('step'):
            continue
        if l.startswith('LAMBDA'):
            lam = [float(x) for x in l.split(',')[1:]]
            continue
        if l.startswith('WALL,'):
            solver_wall = float(l.split(',')[1]); continue
        try:
            rows.append([float(x) for x in l.split(',')])
        except ValueError:
            pass
    for l in err.splitlines():
        if l.startswith('STATS,'):
            solver_wall = float(l.split(',')[-1])
    return dict(status=status, wall=wall, rows=rows, lam=lam,
                solver_wall=solver_wall)


# ---------------------------------------------------------------------------
# (a) equilibrium: 20 Tier-1 ladder points, T_eq within 1%
# ---------------------------------------------------------------------------
def gate_a():
    # The Tier-1 ladders (audit_sweeps.py): NH = logspace(-3,6,19) at the
    # standard point, plus the Z / G0 / zeta ladder legs.  20 representative
    # points spanning all four ladders and the full density range.
    NH19 = logspace(-3, 6, 19)
    pts = []
    for n in [NH19[0], NH19[3], NH19[6], NH19[9], NH19[12], NH19[15], NH19[18]]:
        pts.append((n, ZSTD, G0STD, ZETASTD, 'nH-ladder'))
    for Z in [1.0, 0.01]:
        for n in [NH19[4], NH19[10], NH19[16]]:
            pts.append((n, Z, G0STD, ZETASTD, 'Z-ladder'))
    for G0 in [0.17, 17.0]:
        for n in [NH19[6], NH19[13]]:
            pts.append((n, ZSTD, G0, ZETASTD, 'G0-ladder'))
    for zeta in [3e-18, 1e-16]:
        for n in [NH19[6], NH19[13]]:
            pts.append((n, ZSTD, G0STD, zeta, 'CR-ladder'))
    assert len(pts) == 21, len(pts)
    pts = pts[:20]

    fn = os.path.join(OUT, "gate2_a_equilibrium.csv")
    with open(fn, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['net', 'ladder', 'nH', 'Z', 'G0', 'zeta',
                    'Teq_f77', 'Teq_cxx', 'relerr_T',
                    'status_f77', 'status_cxx',
                    'wall_f77', 'wall_cxx'] +
                   sum([['y%d_f77' % i, 'y%d_cxx' % i] for i in range(7)], []))
        for net in NETS:
            for (n, Z, G0, zeta, tag) in pts:
                res = {}
                for b in ('f77', 'cxx'):
                    # same protocol as audit_sweeps.final_T: 300 Myr from 1e4 K
                    res[b] = run_zone(net, b, 1e4, n, Z, G0, zeta, 300.0, 60,
                                      mode=0, stride=59, timeout=900)
                T = {}
                Y = {}
                for b in ('f77', 'cxx'):
                    r = res[b]['rows']
                    T[b] = r[-1][2] if r else float('nan')
                    Y[b] = (r[-1][4:4 + NTRAC[net]] if r else [])
                rel = abs(T['cxx'] - T['f77']) / T['f77'] if T['f77'] else float('nan')
                ycols = []
                for i in range(7):
                    for b in ('f77', 'cxx'):
                        ycols.append('%.6e' % Y[b][i] if i < len(Y[b]) else '')
                w.writerow([net, tag, '%.4e' % n, Z, G0, zeta,
                            '%.6e' % T['f77'], '%.6e' % T['cxx'], '%.3e' % rel,
                            res['f77']['status'], res['cxx']['status'],
                            '%.3f' % res['f77']['solver_wall'],
                            '%.3f' % res['cxx']['solver_wall']] + ycols)
                f.flush()
                print("A net%-2d %-10s n=%9.3e Z=%.3g G0=%.3g z=%.2g  "
                      "T %10.4g / %10.4g  rel=%.3e" %
                      (net, tag, n, Z, G0, zeta, T['f77'], T['cxx'], rel), flush=True)
    print("wrote", fn)


# ---------------------------------------------------------------------------
# (b) relaxation: fig6 trajectory cases, both backends, max deviation
# ---------------------------------------------------------------------------
RELAX_CASES = [   # (tag, nH, T0, te_long_Myr)   -- make_figures.py RELAX_CASES
    ('hot_dense',    1e4, 1e6,   10.0),
    ('wnm_cool',     1e2, 8e3,   30.0),
    ('h2_form',      1e4, 1e2,   30.0),
    ('diffuse_heat', 0.1, 1e2, 3000.0),
]


def gate_b():
    fn = os.path.join(OUT, "gate2_b_relaxation.csv")
    fnp = os.path.join(OUT, "gate2_b_points.csv")
    with open(fn, 'w', newline='') as f, open(fnp, 'w', newline='') as fp:
        w = csv.writer(f); wp = csv.writer(fp)
        w.writerow(['net', 'tag', 'scale', 'nH', 'T0', 'npts',
                    'maxrel_T', 'at_t_Myr', 'maxrel_xH2', 'maxrel_xHp',
                    'Tend_f77', 'Tend_cxx', 'relerr_Tend',
                    'wall_f77', 'wall_cxx', 'status_f77', 'status_cxx'])
        wp.writerow(['net', 'tag', 'scale', 't_Myr', 'T_f77', 'T_cxx',
                     'relT', 'xH2_f77', 'xH2_cxx', 'xHp_f77', 'xHp_cxx'])
        for net in NETS:
            for (tag, n, T0, teL) in RELAX_CASES:
                for scale, te in (('short', teL / 300.0), ('long', teL)):
                    res = {b: run_zone(net, b, T0, n, ZSTD, G0STD, ZETASTD,
                                       te, 150, mode=0, stride=1, timeout=900)
                           for b in ('f77', 'cxx')}
                    A, B = res['f77']['rows'], res['cxx']['rows']
                    npt = min(len(A), len(B))
                    mT = mH2 = mHp = 0.0; at = float('nan')
                    for k in range(npt):
                        ta, Ta = A[k][1], A[k][2]
                        Tb = B[k][2]
                        rT = abs(Tb - Ta) / Ta if Ta else 0.0
                        if rT > mT: mT, at = rT, ta
                        for col, acc in ((4, 'h2'), (5, 'hp')):
                            a, b2 = A[k][col], B[k][col]
                            r = abs(b2 - a) / a if a > 1e-30 else 0.0
                            if acc == 'h2': mH2 = max(mH2, r)
                            else:           mHp = max(mHp, r)
                        wp.writerow([net, tag, scale, '%.6e' % ta,
                                     '%.6e' % Ta, '%.6e' % Tb, '%.3e' % rT,
                                     '%.6e' % A[k][4], '%.6e' % B[k][4],
                                     '%.6e' % A[k][5], '%.6e' % B[k][5]])
                    Te = {b: (res[b]['rows'][-1][2] if res[b]['rows'] else float('nan'))
                          for b in ('f77', 'cxx')}
                    relend = abs(Te['cxx'] - Te['f77']) / Te['f77'] if Te['f77'] else float('nan')
                    w.writerow([net, tag, scale, n, T0, npt,
                                '%.3e' % mT, '%.4e' % at,
                                '%.3e' % mH2, '%.3e' % mHp,
                                '%.6e' % Te['f77'], '%.6e' % Te['cxx'],
                                '%.3e' % relend,
                                '%.3f' % res['f77']['solver_wall'],
                                '%.3f' % res['cxx']['solver_wall'],
                                res['f77']['status'], res['cxx']['status']])
                    f.flush(); fp.flush()
                    print("B net%-2d %-13s %-5s maxrelT=%.3e (t=%.3g Myr) "
                          "Tend %.5g/%.5g rel=%.3e" %
                          (net, tag, scale, mT, at, Te['f77'], Te['cxx'], relend),
                          flush=True)
    print("wrote", fn, fnp)


# ---------------------------------------------------------------------------
# (c) 80-zone certification grid per net
# ---------------------------------------------------------------------------
def gate_c():
    ngrid = logspace(-3, 6, 10)
    tgrid = logspace(1, 8, 8)
    fn = os.path.join(OUT, "gate2_c_certification.csv")
    with open(fn, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['net', 'T0', 'nH', 'status_f77', 'status_cxx',
                    'wall_f77', 'wall_cxx', 'speedup',
                    'Teq_f77', 'Teq_cxx', 'relerr_T'])
        for net in NETS:
            for T0 in tgrid:
                for n in ngrid:
                    te = min(max(100.0 / math.sqrt(n), 1.0), 30.0)
                    res = {b: run_zone(net, b, T0, n, ZSTD, G0STD, ZETASTD,
                                       te, 40, mode=0, stride=40, timeout=300)
                           for b in ('f77', 'cxx')}
                    Te = {b: (res[b]['rows'][-1][2] if res[b]['rows'] else float('nan'))
                          for b in ('f77', 'cxx')}
                    rel = (abs(Te['cxx'] - Te['f77']) / Te['f77']
                           if Te['f77'] == Te['f77'] and Te['f77'] else float('nan'))
                    wf, wc = res['f77']['solver_wall'], res['cxx']['solver_wall']
                    sp = wf / wc if wc and wc == wc and wc > 0 else float('nan')
                    w.writerow([net, '%.4e' % T0, '%.4e' % n,
                                res['f77']['status'], res['cxx']['status'],
                                '%.4f' % wf, '%.4f' % wc, '%.3f' % sp,
                                '%.5e' % Te['f77'], '%.5e' % Te['cxx'],
                                '%.3e' % rel])
                    f.flush()
                    print("C net%-2d T0=%9.3e n=%9.3e  %-8s/%-8s  "
                          "wall %7.3f/%7.3f (x%.2f)  rel=%.3e" %
                          (net, T0, n, res['f77']['status'], res['cxx']['status'],
                           wf, wc, sp, rel), flush=True)
    print("wrote", fn)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("a", "all"): gate_a()
    if what in ("b", "all"): gate_b()
    if what in ("c", "all"): gate_c()
