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
