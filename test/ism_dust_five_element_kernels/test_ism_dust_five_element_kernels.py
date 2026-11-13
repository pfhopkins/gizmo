"""Unit test for src/solids/ism_dust_five_element_kernels.h (the five-element ISM dust model).

The reference is the stella_dev code (galaxy_sf/resolvedism_dust.cc: resolvedism_dust_evolve, resolvedism_dust_clamp_cell,
resolvedism_dust_seed_cell, resolvedism_dust_sn_destruction_frac and the rate rows of resolvedism_dust_selftest) transcribed
line by line below, with the rates in 1/yr and dt in yr. The header is compiled with plain -O2 and the production flags.
"""

import math
import subprocess
from os.path import abspath, dirname, join

import numpy as np
import pytest

FLAGSETS = {"O2": ["-O2"], "production": ["-O3", "-ffast-math", "-funroll-loops"]}
O, MG, SI, FE = 64.0 / 172.0, 24.0 / 172.0, 28.0 / 172.0, 56.0 / 172.0
YC = [-2.34333937e+02, 1.38485732e+02, -3.39021615e+01, 4.17705353e+00, -2.58281473e-01, 6.38827523e-03]
YS = [-2.34790500e+02, 1.33208637e+02, -3.13027448e+01, 3.71345730e+00, -2.21823668e-01, 5.31746427e-03]
DMIN, DMAX = (lambda a, b: a if a < b else b), (lambda a, b: a if a > b else b)
EXP = (lambda x: math.exp(x) if x < 709 else math.inf)  # C overflows to inf


def yth(T, c):
    if T < 3e4:
        return 0.0
    x = math.log10(min(T, 1e10))
    return 10 ** sum(c[k] * x ** k for k in range(6))


def ref_destr(n0, M, E51=1.0):
    return DMIN(0.4 * 2460.0 * E51 / (n0 ** 0.1 * 1.0 ** (9 / 7)) / M, 1.0)


def ref_step(nH, T, Td, dt, D, Z, literal=False, frac=(O, MG, SI, FE)):
    """resolvedism_dust_evolve for one cell, rates in 1/yr, dt in yr; literal=True keeps the Mg and Fe caps of stella_dev,
    which scale by an earlier O cap a second time (resolvedism_dust.cc:203-204), the port takes the plain minimum"""
    O, MG, SI, FE = frac
    D = list(D)
    if sum(D) <= 0:
        return D
    a = 0.1
    rC, rS = 3 * nH / a * yth(T, YC), 3 * nH / a * yth(T, YS)
    if rC > 0:
        D[0] *= math.exp(-dt * rC)
    if rS > 0:
        f = math.exp(-dt * rS)
        D[1:] = [x * f for x in D[1:]]
    stick = 1 / (1 + 0.04 * math.sqrt(T + Td))
    if stick > 1e-4:
        sq = math.sqrt(T)
        zC, zO, zMg, zSi, zFe = (DMAX(Z[k] - D[k], 0) for k in range(5))
        fC = D[0] / Z[0] if Z[0] > 0 else 1.0
        fSi = D[3] / Z[3] if Z[3] > 0 else 1.0
        fC, fSi = DMIN(fC, 1.0), DMIN(fSi, 1.0)
        tC = 2.15e-12 * nH / a * sq * (zC / 2.07e-3) * stick * (1 - fC)
        tSi = 2.01e-12 * nH / a * sq * (zSi / 6.83e-4) * stick * (1 - fSi)
        if tC > 0 and zC > 0:
            D[0] += DMIN(D[0] * (EXP(dt * tC) - 1), zC)
        if tSi > 0 and zSi > 0:
            dSi = DMIN(D[3] * (EXP(dt * tSi) - 1), zSi)
            tot = dSi / SI
            dO, dMg, dFe = tot * O, tot * MG, tot * FE
            if dO > zO:
                tot *= zO / dO
            if dMg > zMg:
                tot = DMIN(tot, zMg / MG * SI / dSi * tot) if literal else DMIN(tot, zMg / MG)
            if dFe > zFe:
                tot = DMIN(tot, zFe / FE * SI / dSi * tot) if literal else DMIN(tot, zFe / FE)
            if tot > 0:
                for k, fr in ((1, O), (2, MG), (3, SI), (4, FE)):
                    D[k] += tot * fr
    D = [DMAX(x, 0) for x in D]
    D[0] = DMIN(D[0], DMAX(Z[0], 0))
    f = 1
    for k in range(1, 5):
        if D[k] > 0:
            f = DMIN(f, DMAX(Z[k], 0) / D[k])
    if f < 1:
        D[1:] = [x * f for x in D[1:]]
    return D


def ref_seed(Zt, Z):
    dgr = 0.01 * (DMAX(Zt, 0) / 0.014)
    c = DMIN(0.3 * dgr, 0.5 * DMAX(Z[0], 0))
    sil = 0.7 * dgr
    for k, fr in ((1, O), (2, MG), (3, SI), (4, FE)):
        sil = DMIN(sil, 0.9 * DMAX(Z[k], 0) / fr)
    return [c, sil * O, sil * MG, sil * SI, sil * FE]


SOLAR = [2.47653512e-3, 6.00285929e-3, 7.41243589e-4, 6.96209192e-4, 1.35284992e-3]  # C, O, Mg, Si, Fe at Z = 0.014


def cases():
    """the self-test grids (DUSTTEST_RATE, DUSTTEST_DESTR) and cells across the regimes"""
    out = [("rate", [T], None) for T in (1e4, 3e4, 1e5, 3e5, 1e6, 1e7, 1e8)]
    out += [("destr", [n0, M], None) for n0 in (0.1, 1.0, 100.0, 1e4) for M in (1.0, 10.0, 100.0)]
    rng = np.random.default_rng(1)
    for nH, T, Td, dt in ((1.0, 1e7, 20.0, 1e4), (1e-2, 3e6, 20.0, 1e5), (1e3, 30.0, 15.0, 1e5), (1e4, 10.0, 10.0, 1e7),
                          (1e5, 20.0, 15.0, 1e9), (10.0, 8e3, 30.0, 1e6), (1.0, 1e5, 50.0, 1e3)):
        for scale in (0.1, 1.0, 3.0):
            Z = [s * scale for s in SOLAR]
            for fd in (0.01, 0.3, 0.9):
                D = [fd * Z[0]] + [fd * Z[3] / SI * fr for fr in (O, MG, SI, FE)]
                out.append(("step", [nH, T, Td, dt] + D + Z, None))
        Z = list(np.array(SOLAR) * rng.uniform(0.2, 2.0, 5))  # off-solar mixtures: other elements limit the silicate
        out.append(("step", [nH, T, Td, dt, 0.4 * Z[0], 0.2 * Z[3] / SI * O, 0.2 * Z[3] / SI * MG, 0.2 * Z[3], 0.2 * Z[3] / SI * FE] + Z, None))
    for Zt in (0.0, 1.4e-4, 0.014, 0.05):
        out.append(("seed", [Zt] + [s * Zt / 0.014 for s in SOLAR], None))
    out.append(("seed", [0.014, SOLAR[0], 1e-5, SOLAR[2], SOLAR[3], SOLAR[4]], None))  # O-poor: the silicate is capped by O
    return out


def reference(cmd, a):
    if cmd == "rate":
        return [yth(a[0], YC), yth(a[0], YS), 1 / (1 + 0.04 * math.sqrt(a[0] + 20.0))]
    if cmd == "destr":
        return [ref_destr(*a)]
    if cmd == "step":
        return ref_step(*a[:4], a[4:9], a[9:14])
    return ref_seed(a[0], a[1:6])


@pytest.mark.parametrize("flagset", sorted(FLAGSETS))
def test_ism_dust_five_element_kernels(flagset):
    test_dir = dirname(abspath(__file__))
    src_dir = abspath(join(test_dir, "..", "..", "src"))
    exe = join(test_dir, "test_ism_dust_five_element_kernels_" + flagset)
    cmd = ["mpicxx", "-std=c++17"] + FLAGSETS[flagset] + ["-I", src_dir, join(test_dir, "test_ism_dust_five_element_kernels.cc"), "-o", exe, "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, "Compilation failed:\n%s" % r.stderr
    cs = cases()
    stdin = "".join(f"{c} " + " ".join(repr(float(x)) for x in a) + "\n" for c, a, _ in cs)
    r = subprocess.run([exe], input=stdin, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = r.stdout.splitlines()
    assert len(rows) == len(cs)
    worst, nfail = 0.0, 0
    for (c, a, _), row in zip(cs, rows):
        got, want = np.array(row.split(), float), np.array(reference(c, a))
        err = np.abs(got - want) / np.maximum(np.abs(want), 1e-30)
        err[(np.abs(want) < 1e-25) & (np.abs(got) < 1e-25)] = 0
        worst = max(worst, err.max())
        if err.max() > 1e-9:
            nfail += 1
            print(f"FAIL {c} {a}: got {got}, stella_dev {want}")
    print(f"{len(cs)} cases, worst relative difference to the stella_dev formulas {worst:.2e}")
    assert nfail == 0


def test_double_cap_is_the_only_difference():
    """the literal stella_dev growth differs from the port only where an O cap is followed by a Mg or Fe cap"""
    differ = 0
    for c, a, _ in cases():
        if c == "step":
            p, l = ref_step(*a[:4], a[4:9], a[9:14]), ref_step(*a[:4], a[4:9], a[9:14], literal=True)
            if max(abs(x - y) / max(abs(x), 1e-30) for x, y in zip(p, l)) > 1e-12:
                differ += 1
                assert sum(l) < sum(p)  # stella_dev grows less silicate there
    print(f"cases where the literal stella_dev Mg/Fe cap differs: {differ}")
    assert differ <= 2


def test_cases_cover_the_regimes():
    """the step cases sputter (hot), grow (cold), saturate a pool, and hit both silicate limits"""
    kinds = {"sputter": 0, "grow": 0, "cap": 0}
    for c, a, _ in cases():
        if c != "step":
            continue
        d0, d1 = sum(a[4:9]), sum(ref_step(*a[:4], a[4:9], a[9:14]))
        kinds["sputter"] += d1 < 0.999 * d0
        kinds["grow"] += d1 > 1.001 * d0
        kinds["cap"] += abs(ref_step(*a[:4], a[4:9], a[9:14])[0] - a[9]) < 1e-12 * a[9]
    print(kinds)
    assert all(v > 0 for v in kinds.values())
