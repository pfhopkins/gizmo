"""G0 consistency: one 40 Msun Stella star, three radiation paths, chemistry-side photorates against the analytic field.

Paths (flags appended to Config.sh): TREE_RAD (GALSF_RESOLVEDISM_G0_VARIABLE, the gizmo2017 reference), TREE_RAY
(+ TREE_RAY) and M1 (RT_M1 with the PHOTOELECTRIC/LYMAN_WERNER bands; Stella adds the 6-8 eV NUV band). Media:
  thin    n_H = 0.3 cm^-3 uniform (A_V < 0.01): pure 1/r^2, checks each path's unit conversion at the chemistry hand-off
  uniform n_H = 100 cm^-3 uniform: attenuation by the gas between star and cell, plus gas behind the star
  slab    thin medium + an n_H = 200, 1.875 pc slab 2.5 pc from the star (N = 1.2e21 cm^-2) on one side
For every cell the chemistry's own rates (OUTPUT_CHEM_PHOTORATES) are compared with the TREE_RAD conventions applied to
the exact field L/(4 pi r^2 c) and the exact star-to-cell column N:
  k_H2 = 3.3e-11 G0_LW exp(-2e-21 N) fshield,  k_CO = 1.235e-10 G0 exp(-2.5 A_V),  G_dust = G0 exp(-2.5 A_V),
  dust-heating field = G0 chi_isrf / 1.7 (reference here G0 exp(-2.5 A_V) / 1.7),
with G0 = u(6-13.6 eV)/5.29e-14 x 1.7, G0_LW = u(11.2-13.6 eV)/2.11e-14 x 1.7, A_V = 5.348e-22 N.
"""

import json
import os
import shutil
import subprocess
from os.path import abspath, dirname, join

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, default_mpi_ranks

TEST_NAME = "g0_consistency"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
OUT = join(TEST_DIR, "output")

PATHS = {
    "tree_rad": ("GALSF_RESOLVEDISM_G0_VARIABLE",),
    "tree_ray": ("GALSF_RESOLVEDISM_G0_VARIABLE", "TREE_RAY"),
    "m1": ("RT_M1", "RT_CHEM_PHOTOION=1", "RT_PHOTOELECTRIC", "RT_LYMAN_WERNER", "RT_SPEEDOFLIGHT_REDUCTION=0.05"),
}
CASES = ("thin", "uniform", "slab")
QUANT = ("k_H2", "k_CO", "G_dust", "dust_heat")

PC = 3.085678e18
KPC = 1e3 * PC
MSUN = 1.989e33
M_P = 1.6726e-24
C_LIGHT = 2.99792458e10
X_H = 0.7065  # net 5 (ABHE = 0.1)
U_HAB_FUV, U_HAB_LW, G0_PARAM, AV_PER_NH = 5.29e-14, 2.11e-14, 1.7, 5.348e-22
UNIT_RHO = 1.989e43 / KPC**3
BOX_PC, NG = 20.0, 32
N_THIN, N_UNIFORM, N_SLAB = 0.3, 100.0, 200.0
SLAB_I0, SLAB_NI = 20, 3  # slab = lattice planes 20-22 in x: x = 12.5-14.375 pc, star at 10 pc
T_IC_TIME, STAR_AGE_YR, STAR_MASS = 0.001, 5.0e5, 40.0
CODE_TIME_YR = KPC / 1e5 / (365.25 * 86400)

# solar Metallicity[28] (total + 27 elements) and ElementAbundance[27], as Stella writes them at InitMetallicity = 1
METALLICITY = np.array([1.4000000e-02, 7.3750001e-01, 2.4850000e-01, 2.4765350e-03, 7.2545494e-04, 6.0028592e-03,
                        5.2841267e-07, 1.3157898e-03, 3.0605839e-05, 7.4124360e-04, 5.8255839e-05, 6.9620920e-04,
                        6.0990560e-06, 3.2376422e-04, 8.5877928e-06, 7.6870616e-05, 3.2093706e-06, 6.7169385e-05,
                        4.8646655e-08, 3.2681467e-06, 3.3215539e-07, 1.7387425e-05, 1.1327607e-05, 1.3528499e-03,
                        4.4118819e-06, 7.4619376e-05, 7.5396912e-07, 1.8184871e-06])


def nh_to_rho_code(n_h):
    return n_h * M_P / X_H / UNIT_RHO


def make_ics(fname, case):
    """Lattice of NG^3 cells in a 20 pc box (atomic, 100 K) with the star at the box centre, in snapshot format."""
    dx = BOX_PC / NG
    g = (np.arange(NG) + 0.5) * dx
    x = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    n_h = np.full(len(x), N_UNIFORM if case == "uniform" else N_THIN)
    if case == "slab":
        ix = np.floor(x[:, 0] / dx + 1e-6).astype(int)
        n_h[(ix >= SLAB_I0) & (ix < SLAB_I0 + SLAB_NI)] = N_SLAB
    vol = (dx * 1e-3) ** 3
    ng = len(x)
    with h5py.File(fname, "w") as f:
        p0 = f.create_group("PartType0")
        p0["Coordinates"] = x * 1e-3
        p0["Velocities"] = np.zeros((ng, 3))
        p0["ParticleIDs"] = np.arange(1, ng + 1, dtype=np.uint64)
        p0["Masses"] = nh_to_rho_code(n_h) * vol
        p0["InternalEnergy"] = np.full(ng, 1.0)  # ~100 K
        p0["KernelMaxRadius"] = np.full(ng, 2.0 * dx * 1e-3)  # ~32 neighbours
        p0["Metallicity"] = np.tile(METALLICITY, (ng, 1))
        p0["ElementAbundance"] = np.tile(METALLICITY[1:], (ng, 1))
        p0["ChemcoolTracAbund"] = np.tile([0.0, 1.0e-4, 0.0], (ng, 1))  # atomic, x_e = 1e-4, no CO
        p4 = f.create_group("PartType4")
        m_code = STAR_MASS * MSUN / 1.989e43
        p4["Coordinates"] = np.full((1, 3), 0.5 * BOX_PC * 1e-3)
        p4["Velocities"] = np.zeros((1, 3))
        p4["ParticleIDs"] = np.array([ng + 1], dtype=np.uint64)
        p4["Masses"] = np.array([m_code])
        p4["KernelMaxRadius"] = np.array([2.0 * dx * 1e-3])
        p4["StellarFormationTime"] = np.array([T_IC_TIME - STAR_AGE_YR / CODE_TIME_YR])
        p4["SampleIMFStellarMasses"] = np.array([[STAR_MASS, 0.0, 0.0, 0.0, m_code]])
        p4["IMFSampledFlag"] = np.array([1], dtype=np.int32)
        p4["BirthMetallicity"] = np.array([METALLICITY[0]])
        p4["FormationDensity"] = np.array([0.0])  # no PMS delay
        p4["Metallicity"] = METALLICITY[None, :]
        p4["ElementAbundance"] = METALLICITY[None, 1:]
        h = f.create_group("Header")
        npart = np.array([ng, 0, 0, 0, 1, 0])
        h.attrs["NumPart_ThisFile"] = npart.astype(np.int32)
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = T_IC_TIME
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = BOX_PC * 1e-3
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_DoublePrecision"] = 0  # HDF5 converts the float64 data on read
        h.attrs["Flag_Metals"] = len(METALLICITY)
        for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Feedback"):
            h.attrs[k] = 1


def run_case(exe, case, rundir, nranks):
    """Run one medium with one binary (snapshot start, RestartFlag 2); return the final snapshot."""
    shutil.rmtree(rundir, ignore_errors=True)
    os.makedirs(join(rundir, "output"))
    os.symlink(REPO, join(rundir, "gizmo"))  # stellar tables and chemistry data are opened under gizmo/
    make_ics(join(rundir, f"{TEST_NAME}_ics.hdf5"), case)
    shutil.copy(join(TEST_DIR, f"{TEST_NAME}.params"), rundir)
    if os.environ.get("SLURM_JOB_ID"):
        cmd = ["srun", "-n", str(nranks), "--cpu-bind=none"]
    else:
        cmd = ["mpirun", "-np", str(nranks), "--oversubscribe"]
    with open(join(rundir, "run.log"), "w") as log:
        r = subprocess.run(cmd + [exe, f"{TEST_NAME}.params", "2"], stdout=log, stderr=subprocess.STDOUT, cwd=rundir)
    assert r.returncode == 0, f"GIZMO failed for {rundir} (exit {r.returncode}), see run.log"
    snaps = sorted(f for f in os.listdir(join(rundir, "output")) if f.startswith("snapshot_"))
    return join(rundir, "output", snaps[-1])


def column_star_to_cell(case, d):
    """Exact N_H [cm^-2] along the straight line from the star to each cell (d = cell - star in pc)."""
    r = np.linalg.norm(d, axis=1)
    n_bg = N_UNIFORM if case == "uniform" else N_THIN
    col = n_bg * r * PC
    if case == "slab":
        dx = BOX_PC / NG
        x0, x1 = SLAB_I0 * dx - 0.5 * BOX_PC, (SLAB_I0 + SLAB_NI) * dx - 0.5 * BOX_PC  # slab in star-centred x
        cos = np.abs(d[:, 0]) / np.maximum(r, 1e-30)
        inside = np.clip(d[:, 0], x0, x1) - x0  # x-extent of the slab crossed by the ray
        inside[d[:, 0] <= x0] = 0.0
        col += (N_SLAB - n_bg) * inside * PC / np.maximum(cos, 1e-30)
    return col


def sample_mask(case, d):
    r = np.linalg.norm(d, axis=1)
    if case == "slab":  # shadowed cells beyond the slab, near the axis
        return (d[:, 0] > 5.0) & (r < 9.0) & (np.abs(d[:, 1]) < 3.0) & (np.abs(d[:, 2]) < 3.0)
    return (r > 3.0) & (r < 7.0)


def ratios(snap, case, l_fuv, l_lw):
    """Median chemistry-side rate / analytic rate over the sampled cells, per quantity."""
    with h5py.File(snap) as f:
        xs = f["PartType4/Coordinates"][0] * 1e3
        x = f["PartType0/Coordinates"][:] * 1e3
        pr = f["PartType0/ChemPhotoRates"][:].astype(float)
    d = x - xs
    m = sample_mask(case, d)
    r_cm = np.linalg.norm(d[m], axis=1) * PC
    av = AV_PER_NH * column_star_to_cell(case, d[m])
    g0 = l_fuv / (4 * np.pi * r_cm**2 * C_LIGHT) / U_HAB_FUV * G0_PARAM
    g0lw = l_lw / (4 * np.pi * r_cm**2 * C_LIGHT) / U_HAB_LW * G0_PARAM
    att = np.exp(-2.5 * av)
    ref = {"k_H2": 3.3e-11 * g0lw * np.exp(-2e-21 * av / AV_PER_NH) * pr[m, 4],
           "k_CO": 1.235e-10 * g0 * att, "G_dust": g0 * att, "dust_heat": g0 * att / 1.7}
    got = {"k_H2": pr[m, 0], "k_CO": pr[m, 1], "G_dust": pr[m, 2], "dust_heat": pr[m, 3]}
    return {q: float(np.median(got[q] / ref[q])) for q in QUANT}, int(m.sum()), float(np.median(av))


def star_luminosities(snap):
    with h5py.File(snap) as f:
        return 10 ** float(f["PartType4/UV_Luminosity"][0]), 10 ** float(f["PartType4/LW_Luminosity"][0])


# acceptance (median rate / analytic), measured at 32^3 (see README):
# thin: TREE_RAD/TREE_RAY 0.98-1.00, M1 1.21 (point-source M1 closure, E > F/c)
# uniform: TREE_RAY 0.93-1.00; TREE_RAD 0.001-0.01 (TreeCol column continues past the star); M1 1.3-1.9 (HLL ~1/(1+tau_cell) per cell)
# slab: TREE_RAD 0.84-0.89, TREE_RAY 0.93-0.99, M1 1.9-2.7 (3-cell slab, tau_cell 0.5-0.8, shadow filling)
# TREE_RAD dust heating uses the ISRF chi(A_V), not exp(-2.5 A_V): not checked against this reference when attenuated
TOL = {("thin", "tree_rad"): (0.95, 1.05), ("thin", "tree_ray"): (0.95, 1.05), ("thin", "m1"): (0.9, 1.4),
       ("uniform", "tree_rad"): (0.0, 0.1), ("uniform", "tree_ray"): (0.85, 1.15), ("uniform", "m1"): (0.9, 2.5),
       ("slab", "tree_rad"): (0.75, 1.1), ("slab", "tree_ray"): (0.85, 1.15), ("slab", "m1"): (0.9, 3.5)}
UNCHECKED = {("uniform", "tree_rad", "dust_heat"), ("slab", "tree_rad", "dust_heat")}


def test_g0_consistency():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO three times; nothing to validate from external snapshots")
    nranks = default_mpi_ranks(4)
    os.makedirs(OUT, exist_ok=True)
    results, lum = {}, {}
    for path, flags in PATHS.items():
        cwd = os.getcwd()
        os.chdir(REPO)
        try:
            build_gizmo_for_test(TEST_NAME, extra_config_flags=flags)
        finally:
            os.chdir(cwd)
        exe = join(OUT, f"GIZMO_{path}")
        shutil.copy(join(TEST_DIR, "GIZMO"), exe)
        for case in CASES:
            snap = run_case(exe, case, join(OUT, f"{path}_{case}"), nranks)
            if path == "tree_rad":
                lum[case] = star_luminosities(snap)  # same star and age in every path
            results[(path, case)] = ratios(snap, case, *lum[case])
    lines = ["path      case     ncell  <A_V>   " + "  ".join(f"{q:>9s}" for q in QUANT)]
    for (path, case), (rat, n, av) in results.items():
        lines.append(f"{path:9s} {case:8s} {n:5d}  {av:5.2f}  " + "  ".join(f"{rat[q]:9.3f}" for q in QUANT))
    table = "\n".join(lines)
    print(table)
    with open(join(OUT, "g0_consistency_table.txt"), "w") as f:
        f.write(table + "\n")
    with open(join(OUT, "g0_consistency.json"), "w") as f:
        json.dump({f"{p}/{c}": v[0] for (p, c), v in results.items()}, f, indent=1)
    bad = [f"{p}/{c}/{q}={rat[q]:.3f} not in {TOL[(c, p)]}" for (p, c), (rat, _, _) in results.items()
           for q in QUANT if (c, p, q) not in UNCHECKED and not (TOL[(c, p)][0] <= rat[q] <= TOL[(c, p)][1])]
    assert not bad, "chemistry-side rates off the analytic field:\n" + "\n".join(bad) + "\n" + table
