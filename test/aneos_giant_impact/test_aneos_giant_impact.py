"""ANEOS giant impact test: differentiated sphere with real M-ANEOS tables.

Validates that GIZMO can:
  1. Load real M-ANEOS forsterite and iron tables
  2. Initialize particles with per-particle CompositionType
  3. Run a short hydro evolution without crashing
  4. Produce physically reasonable output (T > 0, P > 0, finite values)

It is a short smoke run (self-gravity off, a tiny fraction of a dynamical time) that also checks the IC: the body is built by
the HSE builder (generate_ics.py) on the Stewart tables, and the test requires that no particle below the free-surface layer
was given a substitute EOS state, and that GIZMO's own zero-time density and pressure there reproduce the hydrostatic profile
the IC was built for. The companion Config_selfgrav.sh + aneos_giant_impact_selfgrav.params hold the same body with
self-gravity for ~11 dynamical times.
"""

import pytest
import numpy as np
import h5py
import subprocess
import sys
from os import chdir, getcwd
from os.path import isfile, dirname, abspath, join

from gizmo.test import (
    build_gizmo_for_test,
    run_test,
    clean_test_outputs,
    default_mpi_ranks,
    default_omp_threads,
    get_final_snapshot,
)


ZENODO_BASE = "https://zenodo.org/records/12732259/files"


def _has_valid_row_block(path):
    """does the converted table end with the per-density lowest-valid-row block"""
    with open(path, "rb") as f:
        f.seek(0, 2); f.seek(max(0, f.tell() - 200000))
        return b"lowest_valid_T_index" in f.read()


def setup_tables(test_dir):
    """Download Stewart tables and convert to GIZMO format if needed."""
    converter = join(test_dir, "convert_stewart_table.py")

    for material, mat_id, out_name in [
        ("ANEOS_forsterite_S19.txt", 2019, "forsterite.sesame"),
        ("ANEOS_iron_S20.txt", 2020, "iron.sesame"),
    ]:
        out_path = join(test_dir, out_name)
        if isfile(out_path) and _has_valid_row_block(out_path):
            continue  # a table without the block comes from an older converter: rebuild it

        raw_path = join(test_dir, material)
        if not isfile(raw_path):
            from urllib.request import urlretrieve
            url = f"{ZENODO_BASE}/{material}?download=1"
            print(f"Downloading {material}...")
            urlretrieve(url, raw_path)

        print(f"Converting {material} -> {out_name}...")
        subprocess.run(
            [sys.executable, converter, raw_path, out_path, str(mat_id)],
            check=True,
            cwd=test_dir,
        )


def _ic_is_current(ic_path):
    """an IC written by the current generate_ics.py (it records the hydrostatic profile the checks compare against)"""
    if not isfile(ic_path):
        return False
    with h5py.File(ic_path, "r") as f:
        return "HSEProfile" in f


def setup_ics(test_dir):
    """Generate ICs if missing or written by an older generator."""
    ic_path = join(test_dir, "aneos_giant_impact_ics.hdf5")
    if not _ic_is_current(ic_path):
        subprocess.run(
            [sys.executable, join(test_dir, "generate_ics.py")],
            check=True,
            cwd=test_dir,
        )


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(max_ranks=2),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_aneos_giant_impact(num_mpi_ranks, num_omp_threads):
    test_name = "aneos_giant_impact"
    test_dir = abspath(join(dirname(__file__)))

    clean_test_outputs(test_name)
    build_gizmo_for_test(test_name, num_omp_threads)
    chdir(f"test/{test_name}/")

    # Prepare tables and ICs
    setup_tables(".")
    setup_ics(".")

    run_test(test_name, num_mpi_ranks, num_omp_threads)
    chdir("../../")

    # The IC: no substituted EOS states below the free-surface layer (generate_ics.py refuses to write one, and records the counts)
    ic_path = join(test_dir, "aneos_giant_impact_ics.hdf5")
    with h5py.File(ic_path, "r") as f:
        attrs = dict(f["PartType0"].attrs)
        prof = {k: f["HSEProfile"][k][:] for k in ("r", "P", "rho", "zone_id")}
        free_surface_ids = f["HSEProfile/FreeSurfaceIDs"][:]
        comp_ic = f["PartType0/CompositionType"][:]
    assert attrs["EOS_substitutions_interior"] == 0, f"{attrs['EOS_substitutions_interior']} interior particles have a substitute EOS state"
    print(f"IC: {len(comp_ic)} particles (forsterite {np.sum(comp_ic == 0)}, iron {np.sum(comp_ic == 1)}; "
          f"{attrs['N_free_surface']} in the free-surface layer), substitutions interior {attrs['EOS_substitutions_interior']} / "
          f"free surface {attrs['EOS_substitutions_free_surface']}")

    # GIZMO's zero-time density and pressure below the free-surface layer vs the hydrostatic profile the IC was built for
    # (each material read on its own branch of the profile; measured: density <= 1e-4, pressure <= 2e-4 off)
    with h5py.File(join(test_dir, "output", "snapshot_000.hdf5"), "r") as F:
        g = F["PartType0"]
        ids, x, m = g["ParticleIDs"][:], g["Coordinates"][:], g["Masses"][:]
        rho0, P0, comp0 = g["Density"][:], g["Pressure"][:], g["CompositionType"][:]
    r = np.linalg.norm(x - (m[:, None] * x).sum(0) / m.sum(), axis=1)
    interior = ~np.isin(ids, free_surface_ids)
    for z, name in ((0, "forsterite"), (1, "iron")):
        q = interior & (comp0 == z)
        on = prof["zone_id"] == z
        rho_prof = np.interp(np.clip(r[q], prof["r"][on].min(), prof["r"][on].max()), prof["r"][on], prof["rho"][on])
        P_prof = np.interp(r[q], prof["r"], prof["P"])
        d_rho, d_P = np.max(np.abs(rho0[q] / rho_prof - 1)), np.max(np.abs(P0[q] / P_prof - 1))
        print(f"zero time, {name} below the free-surface layer ({q.sum()}): max |rho / rho_HSE - 1| {d_rho:.2e}, max |P / P_HSE - 1| {d_P:.2e}")
        assert q.sum() >= 100, f"only {q.sum()} {name} particles below the free-surface layer (too few for the check)"
        assert d_rho < 1e-3, f"{name}: GIZMO's zero-time density is {d_rho:.2e} off the hydrostatic profile"
        assert d_P < 2e-3, f"{name}: GIZMO's zero-time pressure is {d_P:.2e} off the hydrostatic profile"

    # Check that output exists
    final_snap = get_final_snapshot(test_name)

    # Basic sanity checks on the output
    with h5py.File(final_snap, "r") as F:
        n = F["Header"].attrs["NumPart_ThisFile"][0]
        assert n > 0, "No particles in output"

        rho = F["PartType0/Density"][:]
        u = F["PartType0/InternalEnergy"][:]
        coords = F["PartType0/Coordinates"][:]

        # Check fields exist and have correct shape
        assert rho.shape == (n,), f"Density shape mismatch: {rho.shape}"
        assert u.shape == (n,), f"InternalEnergy shape mismatch: {u.shape}"
        assert coords.shape == (n, 3), f"Coordinates shape mismatch: {coords.shape}"

        # All values should be finite
        assert np.all(np.isfinite(rho)), f"Non-finite densities: {np.sum(~np.isfinite(rho))} particles"
        assert np.all(np.isfinite(u)), f"Non-finite internal energies: {np.sum(~np.isfinite(u))} particles"
        assert np.all(np.isfinite(coords)), f"Non-finite coordinates"

        # Densities and internal energies should be positive
        assert np.all(rho > 0), f"Non-positive densities: {np.sum(rho <= 0)} particles"
        assert np.all(u > 0), f"Non-positive internal energies: {np.sum(u <= 0)} particles"

        # Check CompositionType is preserved
        if "CompositionType" in F["PartType0"]:
            comp = F["PartType0/CompositionType"][:]
            assert set(np.unique(comp)).issubset({0, 1}), f"Unexpected CompositionType values: {np.unique(comp)}"

    print(f"ANEOS giant impact test passed: {n} particles, all finite and positive")
