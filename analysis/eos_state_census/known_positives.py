#!/usr/bin/env python3
"""Self-test of census.py: a synthetic snapshot whose cells each sit in one constructed state (one CompositionType and one table slot per
case) is run through census.py; every case must raise its own flag and every control must not. Uses three tables from the table directory:
cd21_y0275.sesame (H/He with a validity-floor block), water_baseline.sesame (a few density intervals where u(T) falls: several roots) and
forsterite_S19_T4.sesame (a melt plateau).
    known_positives.py <workdir> --tables <dir>"""
import argparse, os, subprocess, sys
import numpy as np
import h5py
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from census import Lib, BIT, REPORTED

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("workdir"); ap.add_argument("--tables", required=True)
a = ap.parse_args()
T_FLOORED, T_WATER, T_S19 = (os.path.join(a.tables, f) for f in ("cd21_y0275.sesame", "water_baseline.sesame", "forsterite_S19_T4.sesame"))
work = os.path.abspath(a.workdir); out = os.path.join(work, "output"); os.makedirs(out, exist_ok=True)
lib = Lib()
cases = []   # (name, table, rho, u, must, must_not)

def add(name, table, rho, u, must=(), must_not=()):
    cases.append((name, table, float(rho), float(u), set(must), set(must_not)))

# densities, energies and the floor: the H/He table with its floor block
info = lib.load(T_FLOORED, 0)
lo, hi = info["logrho"]
r_mid = 10**(0.5 * (lo + hi))
u_ord, _ = lib.rhoT(0, [1e-2], [3000.0])
add("ordinary", T_FLOORED, 1e-2, u_ord[0], must_not=REPORTED)
add("rho_below", T_FLOORED, 10**(lo - 0.1), u_ord[0], must=["RHO_BELOW"])
add("rho_above", T_FLOORED, 10**(hi + 0.1), 1e13, must=["RHO_ABOVE"])
add("near_rho_low_in_band", T_FLOORED, 10**(lo + 0.045), u_ord[0], must=["NEAR_RHO_LOW"])
add("near_rho_low_outside", T_FLOORED, 10**(lo + 0.047), u_ord[0], must_not=["NEAR_RHO_LOW"])
add("near_rho_high_in_band", T_FLOORED, 10**(hi - 0.040), 1e13, must=["NEAR_RHO_HIGH"])
add("near_rho_high_outside", T_FLOORED, 10**(hi - 0.043), 1e13, must_not=["NEAR_RHO_HIGH"])
rgrid = 10**np.linspace(lo + 0.2, hi - 0.2, 400)
uf, ut = lib.energy_bounds(0, rgrid)
probe = lib.classify(0, rgrid, uf * 0.5, np.full(rgrid.size, 100.0), np.zeros(rgrid.size, np.uint8))
blk = np.flatnonzero((probe["flags"] & BIT["FLOOR_IS_BLOCK"]) != 0)
if not blk.size:
    sys.exit(f"{T_FLOORED}: no density with a floor-block row (wrong table?)")
rf = rgrid[blk[blk.size // 2]]; ufl = lib.energy_bounds(0, [rf])[0][0]
add("u_below_floor_block", T_FLOORED, rf, 0.5 * ufl, must=["U_AT_FLOOR"])
add("u_at_floor_exact", T_FLOORED, rf, ufl, must=["U_AT_FLOOR"])
add("near_floor_1.2", T_FLOORED, rf, 1.2 * ufl, must=["NEAR_FLOOR"], must_not=["U_AT_FLOOR"])
r13 = lib.classify(0, [rf], [1.3 * ufl], [100.0], [0])
if r13["floor_rows"][0] >= 1:   # beyond the floor row's cell: only the energy band decides
    add("near_floor_1.3_control", T_FLOORED, rf, 1.3 * ufl, must_not=["NEAR_FLOOR", "U_AT_FLOOR"])
else:
    add("near_floor_1.3_stencil", T_FLOORED, rf, 1.3 * ufl, must=["NEAR_FLOOR"])
u_top = lib.energy_bounds(0, [r_mid])[1][0]
add("u_above_top", T_FLOORED, r_mid, 2.0 * u_top, must=["U_AT_TOP"])
add("near_top", T_FLOORED, r_mid, 0.85 * u_top, must=["NEAR_TOP"], must_not=["U_AT_TOP"])
add("near_top_control", T_FLOORED, r_mid, 0.75 * u_top, must_not=["NEAR_TOP", "U_AT_TOP"])

# several roots and fold proximity: water, where u(T) falls in a few dense cold intervals
lib.load(T_WATER, 1)
found = None
for lr in np.linspace(np.log10(4.5), np.log10(6.0), 300):
    ub, ut2 = lib.energy_bounds(1, [10**lr])
    us = 10**np.linspace(np.log10(ub[0]) + 1e-4, np.log10(ut2[0]) - 1e-3, 3000)
    rr = lib.classify(1, np.full(us.size, 10**lr), us, np.full(us.size, 250.0), np.ones(us.size, np.uint8))
    k = np.flatnonzero(rr["n_roots"] >= 2); inside = np.flatnonzero(rr["fold_margin"] < 0)
    if k.size and inside.size:
        top = inside[-1]
        found = (10**lr, us[k[0]], us[top] * 10**(-rr["fold_margin"][top]))   # inside, margin = -(upper edge - log u)
        break
if not found:
    sys.exit(f"{T_WATER}: no several-root state found at 4.5-6 g/cc")
add("several_roots", T_WATER, found[0], found[1], must=["ROOTS_MULTI", "FOLD_IN"], must_not=["FOLD_GATE_MISS"])
add("fold_near", T_WATER, found[0], found[2] * 10**0.05, must=["FOLD_NEAR"], must_not=["FOLD_IN"])
add("fold_near_control", T_WATER, found[0], found[2] * 10**0.12, must_not=["FOLD_NEAR", "FOLD_IN"])

# plateau: forsterite's melt band near 3 g/cc
lib.load(T_S19, 2)
Ts = np.linspace(2100, 2250, 301); us19, _ = lib.rhoT(2, np.full(Ts.size, 3.0), Ts)
rp = lib.classify(2, np.full(Ts.size, 3.0), us19, Ts, np.ones(Ts.size, np.uint8))
kp = np.flatnonzero((rp["flags"] & BIT["PLATEAU"]) != 0)
if not kp.size:
    sys.exit(f"{T_S19}: no plateau at 3 g/cc, 2100-2250 K")
add("plateau_forsterite", T_S19, 3.0, us19[kp[kp.size // 2]], must=["PLATEAU"])

# the synthetic run: one CompositionType (= table slot) per case, three identical cells each
ncell = 3
with open(os.path.join(out, "parameters-usedvalues"), "w") as f:
    f.write(f"AneosNumMaterials {len(cases)}\n" + "".join(f"AneosTable{k} {c[1]}\n" for k, c in enumerate(cases)))
with h5py.File(os.path.join(out, "snapshot_000.hdf5"), "w") as f:
    H = f.create_group("Header")
    for k, v in dict(Time=0.0, ComovingIntegrationOn=np.int32(0), UnitMass_In_CGS=1.0, UnitLength_In_CGS=1.0, UnitVelocity_In_CGS=1.0,
                     Flag_DoublePrecision=np.int32(1), Compile_Effective_Options=np.bytes_(b"EOS_ANEOS\nEOS_GENERAL\n"),
                     Compile_Effective_Options_FNV1a64=np.uint64(0)).items():
        H.attrs[k] = v
    g = f.create_group("PartType0")
    n = ncell * len(cases)
    g["Density"] = np.repeat([c[2] for c in cases], ncell)
    g["InternalEnergy"] = np.repeat([c[3] for c in cases], ncell)
    g["CompositionType"] = np.repeat(np.arange(len(cases), dtype=np.int32), ncell)
    g["Masses"] = np.ones(n)
    g["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint64)
res = subprocess.run([sys.executable, os.path.join(HERE, "census.py"), out, "--out", os.path.join(work, "kp"), "--no-figures"],
                     capture_output=True, text=True)
if res.returncode:
    print(res.stdout[-3000:], res.stderr[-3000:]); sys.exit("census.py failed")
got = {k: set() for k in range(len(cases))}
for line in open(os.path.join(work, "kp.csv")).readlines()[1:]:
    v = line.split(",")
    if v[4] in REPORTED:
        got[int(v[3][2:])].add(v[4])
fail = 0
for k, (name, table, rho, u, must, must_not) in enumerate(cases):
    ok = must <= got[k] and not (must_not & got[k])
    fail += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:24s} rho {rho:.4g} u {u:.4g}  raised {sorted(got[k])}")
# a CompositionType that names a table the run did not load must stop the census, never pass as "not tabulated"
bad = os.path.join(work, "missing_slot", "output"); os.makedirs(bad, exist_ok=True)
with open(os.path.join(bad, "parameters-usedvalues"), "w") as f:
    f.write(f"AneosNumMaterials 1\nAneosTable0 {T_FLOORED}\n")
with h5py.File(os.path.join(out, "snapshot_000.hdf5"), "r") as src, h5py.File(os.path.join(bad, "snapshot_000.hdf5"), "w") as f:
    src.copy("Header", f); g = f.create_group("PartType0")
    for name in ("Density", "InternalEnergy", "Masses", "ParticleIDs"): g[name] = src["PartType0/" + name][:2]
    g["CompositionType"] = np.array([0, 5], dtype=np.int32)
res = subprocess.run([sys.executable, os.path.join(HERE, "census.py"), bad, "--out", os.path.join(work, "missing_slot", "kp"), "--no-figures"],
                     capture_output=True, text=True)
ok = res.returncode != 0 and "refused" in (res.stdout + res.stderr)
fail += not ok
print(f"{'PASS' if ok else 'FAIL'}  missing_table_slot       CompositionType 5 with 1 table loaded -> census refuses (rc {res.returncode})")
print(f"{len(cases) + 1 - fail} / {len(cases) + 1} pass")
sys.exit(1 if fail else 0)
