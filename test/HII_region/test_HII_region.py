"""HII region test: 10 Msun ZAMS star in a uniform gas box with mean density 100 H/cc (20 pc).

Baseline of the HII_region test family and the source of truth for the test *procedure*
(see hii_region_procedure.py). Checks that an ionization front forms and that the
photoionized gas settles at a physical HII-region temperature (~1e4 K). The subcycled
variants live in ../HII_region_subcycle and the ionizing-band-only variant in
../HII_region_simple; both import the procedure from here."""

import os
import h5py
import numpy as np
import pytest
from pathlib import Path
from gizmo.test import default_omp_threads, default_mpi_ranks, build_gizmo_for_test, run_test
import hii_region_procedure as hii

TEST_NAME = "HII_region"
TEST_DIR = Path(__file__).parent


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_HII_region(num_mpi_ranks, num_omp_threads):
    profiles, _, _ = hii.run_variant(TEST_NAME, TEST_DIR, num_mpi_ranks, num_omp_threads)
    hii.assert_hii_temperature(profiles)


def make_warm_neutral_ics(path, n_H=1.0, n_side=16):
    """uniform box of the same size and metallicity as the HII_region IC, without the star, at mean density n_H"""
    L = 20.0
    g = (np.arange(n_side) + 0.5) / n_side * L
    pos = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    pos = (pos + np.random.default_rng(1).normal(0, 0.02 * L / n_side, pos.shape)) % L
    n = len(pos)
    z = np.array([0.02, 0.28, 2.53e-3, 7.41e-4, 6.13e-3, 1.20e-3, 5.91e-4, 6.83e-4, 4.09e-4, 6.44e-5, 1.17e-3], dtype=np.float32)
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        npart = np.array([n, 0, 0, 0, 0, 0], dtype=np.int32)
        for k, v in dict(NumPart_ThisFile=npart, NumPart_Total=npart.astype(np.uint32), NumPart_Total_HighWord=np.zeros(6, dtype=np.uint32),
                         MassTable=np.zeros(6), Time=0.0, Redshift=0.0, BoxSize=L, NumFilesPerSnapshot=1, Omega0=0.0, OmegaLambda=0.0,
                         HubbleParam=1.0, Flag_Sfr=0, Flag_Cooling=0, Flag_StellarAge=1, Flag_Metals=11, Flag_Feedback=0,
                         Flag_DoublePrecision=0).items():
            h.attrs[k] = v
        p = f.create_group("PartType0")
        p["Coordinates"] = pos
        p["Velocities"] = np.zeros((n, 3), dtype=np.float32)
        p["Masses"] = np.full(n, 28244.6 * n_H / 100.0 / n, dtype=np.float32)  # 28244.6 Msun is n_H = 100 in this box
        p["ParticleIDs"] = np.arange(1, n + 1, dtype=np.int32)
        p["Metallicity"] = np.tile(z, (n, 1))


@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_warm_neutral_gas_keeps_metal_line_cooling(num_omp_threads):
    """The Kim+23 nebular term replaces the tabulated metal-line cooling only in the photoionized fraction of the gas. In
    warm neutral gas (x_HI > 0.9, 2e3-2e4 K) the tables are the main coolant: electron-excited lines of the singly
    ionized metals (Fe, Si, S, O, ...). Weighting them by the nebular taper alone removed them there (their share of
    the cooling fell to ~2e-6) and left warm neutral gas thermally stuck. Here: 8000 K, n_H = 1, no star, 0.5 Myr;
    with the fix the tables are ~45% of the cooling."""
    hii.prepare_inputs(TEST_DIR, TEST_NAME)
    build_gizmo_for_test(TEST_NAME, num_omp_threads)
    make_warm_neutral_ics(TEST_DIR / "warm_neutral_ics.hdf5")
    cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        run_test(TEST_NAME, 2, num_omp_threads, restart_flag=0, param_overrides={
            "InitCondFile": "warm_neutral_ics", "OutputDir": "output_warm_neutral", "InitGasTemp": "8000",
            "TimeMax": "0.5", "TimeBetSnapshot": "0.5"})
    finally:
        os.chdir(cwd)
    snap = sorted((TEST_DIR / "output_warm_neutral").glob("snapshot_*.hdf5"))[-1]
    with h5py.File(snap, "r") as f:
        g = f["PartType0"]
        T, x_HI = g["Temperature"][:], g["NeutralHydrogenAbundance"][:]
        share = g["MetalCoolingRate"][:].astype(np.float64) / g["CoolingRate"][:].astype(np.float64)
    warm = (T > 2e3) & (T < 2e4) & (x_HI > 0.9)
    print(f"{snap.name}: {warm.sum()} warm neutral cells, median T {np.median(T[warm]):.0f} K, "
          f"metal-line share of the cooling: median {np.median(share[warm]):.3g}, min {share[warm].min():.3g}")
    assert warm.sum() > 0.9 * len(T), "the box did not stay warm and neutral: the check is not exercised"
    assert np.median(share[warm]) > 0.1, f"metal lines are only {np.median(share[warm]):.2g} of the warm neutral gas's cooling"


def make_starless_ics(src, dst):
    """copy of the HII_region IC without the star"""
    with h5py.File(src, "r") as fi, h5py.File(dst, "w") as fo:
        fi.copy("PartType0", fo)
        h = fo.create_group("Header")
        for k, v in fi["Header"].attrs.items():
            h.attrs[k] = v
        npart = np.array(fi["Header"].attrs["NumPart_ThisFile"]).copy()
        npart[5] = 0
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)


@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_stellar_luminosity_factor(num_omp_threads):
    """StellarLuminosityFactor multiplies what single-star sources emit into RT, in every band. At 0 the radiation field
    within 2 pc of the star (which holds all of the stellar photons after 0.02 code units, ~20 kyr) must match the
    same box without the star, while the ISRF and IR field still evolve; at 1 the star must dominate it. Totals are
    compared locally because the star's own timestep (46 vs 8 steps here) perturbs the far-field ISRF/IR evolution
    at ~1e-5 of the band totals."""
    hii.prepare_inputs(TEST_DIR, TEST_NAME)
    build_gizmo_for_test(TEST_NAME, num_omp_threads)
    make_starless_ics(TEST_DIR / f"{TEST_NAME}_ics.hdf5", TEST_DIR / "starless_ics.hdf5")
    runs = {"factor1": (f"{TEST_NAME}_ics", "1"), "factor0": (f"{TEST_NAME}_ics", "0"), "nostar": ("starless_ics", "1")}
    E, M_ion = {}, {}
    cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        for name, (ic, factor) in runs.items():
            run_test(TEST_NAME, default_mpi_ranks(), num_omp_threads, restart_flag=0, param_overrides={
                "InitCondFile": ic, "OutputDir": f"output_lumfactor_{name}", "StellarLuminosityFactor": factor,
                "TimeMax": "0.02", "TimeBetSnapshot": "0.02"})
            used = (TEST_DIR / f"{TEST_NAME}_override.params-usedvalues").read_text().split()
            assert float(used[used.index("StellarLuminosityFactor") + 1]) == float(factor), f"{name}: factor not applied"
            snap = sorted((TEST_DIR / f"output_lumfactor_{name}").glob("snapshot_*.hdf5"))[-1]
            with h5py.File(snap, "r") as f:
                assert abs(f["Header"].attrs["Time"] - 0.02) < 1e-6, f"{name}: {snap.name} is not at TimeMax"
                near = np.linalg.norm(f["PartType0/Coordinates"][:] - 10.0, axis=1) < 2.0  # star at the box centre
                E[name] = f["PartType0/PhotonEnergy"][:][near].astype(np.float64).sum(axis=0)
                M_ion[name] = np.sum(f["PartType0/Masses"][:].astype(np.float64) * f["PartType0/HII"][:])
    finally:
        os.chdir(cwd)
    d1, d0 = E["factor1"] - E["nostar"], E["factor0"] - E["nostar"]
    for k in range(len(d1)):
        print(f"band {k} (r < 2 pc): E_nostar {E['nostar'][k]:.6g}  E_factor1 - E_nostar {d1[k]:.6g}  E_factor0 - E_nostar {d0[k]:.6g}")
    print(f"ionized mass [Msun]: nostar {M_ion['nostar']:.6g}  factor1 {M_ion['factor1']:.6g}  factor0 {M_ion['factor0']:.6g}")
    assert np.all(d1 > E["nostar"]), "the star does not dominate every band near it at factor 1: the check is not exercised"
    assert np.all(np.abs(d0) < 1e-3 * d1), f"factor 0 leaves a stellar contribution: {d0 / d1} of the factor-1 one per band"
    assert abs(M_ion["factor0"] - M_ion["nostar"]) < 1e-3 * abs(M_ion["factor1"] - M_ion["nostar"]), \
        "factor 0 still ionizes gas around the star"
