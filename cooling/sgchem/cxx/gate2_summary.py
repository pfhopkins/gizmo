#!/usr/bin/env python3
"""Summarise the Gate-2 benchmark CSVs into the tables used by
GATE2_ACCEPTANCE.md.  Pure reporting -- no tolerance tuning."""
import csv, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
D = "/orion/ptmp/uli/orion/fb_tests/_sgchem_cxx_build/gate2_data"


def f(x, d=float('nan')):
    try:
        return float(x)
    except Exception:
        return d


print("=" * 78)
print("GATE (a) EQUILIBRIUM -- 20 Tier-1 ladder points x 2 nets; GATE |dTeq| <= 1%")
print("=" * 78)
rows = list(csv.DictReader(open(os.path.join(D, "gate2_a_equilibrium.csv"))))
allrel = []
for net in ('5', '17'):
    R = [r for r in rows if r['net'] == net]
    rel = [f(r['relerr_T']) for r in R]
    allrel += rel
    over = [r for r in R if f(r['relerr_T']) > 0.01]
    bad = [r for r in R if r['status_f77'] != 'OK' or r['status_cxx'] != 'OK']
    print("net %-3s  points %2d  max|dT|/T %.3e  median %.3e  >1%%: %d  non-OK: %d"
          % (net, len(R), max(rel), st.median(rel), len(over), len(bad)))
    for r in sorted(R, key=lambda r: -f(r['relerr_T']))[:3]:
        print("    worst: %-10s nH=%-10s Z=%-5s G0=%-5s z=%-8s  Teq %s / %s  rel %s"
              % (r['ladder'], r['nH'], r['Z'], r['G0'], r['zeta'],
                 r['Teq_f77'], r['Teq_cxx'], r['relerr_T']))
print("OVERALL: %d points, max %.3e, >1%%: %d  => %s"
      % (len(allrel), max(allrel), sum(1 for x in allrel if x > 0.01),
         "PASS" if max(allrel) <= 0.01 else "FAIL"))

print()
print("=" * 78)
print("GATE (b) RELAXATION -- fig6 trajectory cases, both backends")
print("=" * 78)
rows = list(csv.DictReader(open(os.path.join(D, "gate2_b_relaxation.csv"))))
mx = 0.0
print("%-4s %-13s %-6s %6s %11s %11s %11s %11s" %
      ("net", "case", "scale", "npts", "max|dT|/T", "max|dxH2|", "max|dxH+|", "relTend"))
for r in rows:
    mx = max(mx, f(r['maxrel_T']))
    print("%-4s %-13s %-6s %6s %11.3e %11.3e %11.3e %11.3e" %
          (r['net'], r['tag'], r['scale'], r['npts'], f(r['maxrel_T']),
           f(r['maxrel_xH2']), f(r['maxrel_xHp']), f(r['relerr_Tend'])))
npts = sum(int(r['npts']) for r in rows)
print("OVERALL: %d cases, %d compared trajectory points, "
      "max pointwise |dT|/T = %.3e (%.4f%%)" % (len(rows), npts, mx, 100 * mx))

print()
print("=" * 78)
print("GATE (c) CERTIFICATION -- 80-zone (n,T0) grid per net; GATE zero failures")
print("=" * 78)
rows = list(csv.DictReader(open(os.path.join(D, "gate2_c_certification.csv"))))
for net in ('5', '17'):
    R = [r for r in rows if r['net'] == net]
    ff = [r for r in R if r['status_f77'] != 'OK']
    fc = [r for r in R if r['status_cxx'] != 'OK']
    wf = [f(r['wall_f77']) for r in R]
    wc = [f(r['wall_cxx']) for r in R]
    rel = [f(r['relerr_T']) for r in R if r['relerr_T'] not in ('nan', '')]
    print("net %-3s zones %d | DVODE complete %d/%d, CVODE complete %d/%d"
          % (net, len(R), len(R) - len(ff), len(R), len(R) - len(fc), len(R)))
    print("        wall/zone  DVODE mean %.4f s  median %.4f s  total %.3f s"
          % (st.mean(wf), st.median(wf), sum(wf)))
    print("        wall/zone  CVODE mean %.4f s  median %.4f s  total %.3f s"
          % (st.mean(wc), st.median(wc), sum(wc)))
    print("        speedup (DVODE/CVODE): total x%.2f  median-of-ratios x%.2f"
          % (sum(wf) / sum(wc),
             st.median([a / b for a, b in zip(wf, wc) if b > 0])))
    print("        Teq agreement: max %.3e  median %.3e  >1%%: %d"
          % (max(rel), st.median(rel), sum(1 for x in rel if x > 0.01)))
    for r in ff + fc:
        print("        NON-OK zone T0=%s n=%s  f77=%s cxx=%s"
              % (r['T0'], r['nH'], r['status_f77'], r['status_cxx']))

ff = [r for r in rows if r['status_f77'] != 'OK']
fc = [r for r in rows if r['status_cxx'] != 'OK']
print("OVERALL: %d zones, DVODE failures %d, CVODE failures %d  => %s"
      % (len(rows), len(ff), len(fc), "PASS" if not fc and not ff else "FAIL"))

print()
print("=" * 78)
print("WALL-TIME on the gate-(b) relaxation workload (150 hydro steps/case)")
print("=" * 78)
rows = list(csv.DictReader(open(os.path.join(D, "gate2_b_relaxation.csv"))))
for net in ('5', '17'):
    R = [r for r in rows if r['net'] == net]
    wf = [f(r['wall_f77']) for r in R]
    wc = [f(r['wall_cxx']) for r in R]
    print("net %-3s  DVODE total %7.3f s  CVODE total %7.3f s  speedup x%.2f"
          % (net, sum(wf), sum(wc), sum(wf) / sum(wc)))
    for r in R:
        print("        %-13s %-6s  DVODE %7.3f s  CVODE %7.3f s  x%.2f"
              % (r['tag'], r['scale'], f(r['wall_f77']), f(r['wall_cxx']),
                 f(r['wall_f77']) / max(f(r['wall_cxx']), 1e-9)))

print()
print("=" * 78)
print("WALL-TIME on the gate-(a) workload (300 Myr, 60 hydro steps/point)")
print("=" * 78)
rows = list(csv.DictReader(open(os.path.join(D, "gate2_a_equilibrium.csv"))))
for net in ('5', '17'):
    R = [r for r in rows if r['net'] == net]
    wf = [f(r['wall_f77']) for r in R]
    wc = [f(r['wall_cxx']) for r in R]
    print("net %-3s  DVODE total %7.3f s  CVODE total %7.3f s  speedup x%.2f  "
          "(median-of-ratios x%.2f)"
          % (net, sum(wf), sum(wc), sum(wf) / sum(wc),
             st.median([a / b for a, b in zip(wf, wc) if b > 0])))
