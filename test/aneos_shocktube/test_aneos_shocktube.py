"""Sod shock tube test using ANEOS tabulated EOS.

Uses an ideal-gas SESAME table (gamma=1.4) to verify that the full
ANEOS pipeline (table loading, T-inversion, EOS dispatch, hydro coupling)
reproduces the standard Sod shock tube solution. Compares against the
same reference PPM solution as the standard shocktube test.

This test validates the ANEOS implementation end-to-end in a case where
the analytic answer is known exactly.
"""

import pytest
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import h5py
from os import chdir
from os.path import isfile
from glob import glob
from urllib.request import urlretrieve
import subprocess
import sys

from gizmo.test import (
    build_gizmo_for_test,
    run_test,
    clean_test_outputs,
    assert_final_time,
    default_mpi_ranks,
    default_omp_threads,
    get_final_snapshot,
)


WEBSITE = "http://www.tapir.caltech.edu/~phopkins/sims/"


def generate_aneos_table():
    """Generate the ideal-gas SESAME table for this test."""
    subprocess.run(
        [sys.executable, "generate_table.py"],
        check=True,
    )
    assert isfile("ideal_gas_gamma14.sesame"), "Failed to generate ANEOS table"


def add_composition_type_to_ics(ic_file):
    """Add CompositionType=0 to all particles in the IC file (required for EOS_ANEOS)."""
    with h5py.File(ic_file, "r+") as F:
        if "CompositionType" not in F["PartType0"]:
            npart = F["PartType0/Coordinates"].shape[0]
            F["PartType0"].create_dataset(
                "CompositionType", data=np.zeros(npart, dtype=np.int32)
            )


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(max_ranks=4),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_aneos_shocktube(num_mpi_ranks, num_omp_threads):
    test_name = "aneos_shocktube"
    clean_test_outputs(test_name)
    build_gizmo_for_test(test_name, num_omp_threads)
    chdir(f"test/{test_name}/")

    # Generate the ANEOS table
    generate_aneos_table()

    # Download ICs (same as standard shocktube) and exact solution
    for f in ("shocktube_ics_emass.hdf5", "shocktube_exact.txt"):
        if not isfile(f):
            urlretrieve(WEBSITE + f, f)

    # Add CompositionType field to ICs
    add_composition_type_to_ics("shocktube_ics_emass.hdf5")

    run_test(test_name, num_mpi_ranks, num_omp_threads)
    chdir("../../")

    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)

    # Load simulation data
    with h5py.File(final_snap, "r") as F:
        x_sim = F["PartType0/Coordinates"][:, 0]
        rho_sim = F["PartType0/Density"][:]
        vel_sim = F["PartType0/Velocities"][:, 0]
        u_sim = F["PartType0/InternalEnergy"][:]

    # Load exact solution: x, density, pressure, entropy, x_velocity
    exact = np.loadtxt(f"test/{test_name}/shocktube_exact.txt")
    x_exact = exact[:, 0]
    rho_exact = exact[:, 1]
    vel_exact = exact[:, 4]
    P_exact = exact[:, 2]
    gamma = 1.4
    u_exact = P_exact / (rho_exact * (gamma - 1))

    # Interpolate exact solution to particle positions
    rho_interp = interp1d(x_exact, rho_exact, bounds_error=False, fill_value="extrapolate")(x_sim)
    vel_interp = interp1d(x_exact, vel_exact, bounds_error=False, fill_value="extrapolate")(x_sim)
    u_interp = interp1d(x_exact, u_exact, bounds_error=False, fill_value="extrapolate")(x_sim)

    # Compute L1 errors
    L1_rho = np.mean(np.abs(rho_sim - rho_interp)) / np.mean(np.abs(rho_interp))
    L1_vel = np.mean(np.abs(vel_sim - vel_interp)) / (np.mean(np.abs(vel_interp)) + 1e-10)
    L1_u = np.mean(np.abs(u_sim - u_interp)) / np.mean(np.abs(u_interp))

    # Plot comparison
    order = x_sim.argsort()
    for label, sim, exact_vals in [
        ("Density", rho_sim, rho_interp),
        ("Velocity", vel_sim, vel_interp),
        ("InternalEnergy", u_sim, u_interp),
    ]:
        plt.figure()
        plt.plot(x_sim[order], sim[order], ".", markersize=1, label="GIZMO (ANEOS)")
        plt.plot(x_sim[order], exact_vals[order], "-", color="red", linewidth=0.5, label="Exact")
        plt.xlabel("x")
        plt.ylabel(label)
        plt.legend()
        plt.savefig(f"test/{test_name}/{label}.png")
        plt.close()

    # Allow slightly looser tolerances than the standard shocktube since
    # ANEOS adds interpolation error from the table lookup + T-inversion
    assert L1_rho < 0.08, f"Density L1 error {L1_rho:.4f} exceeds tolerance"
    assert L1_vel < 0.15, f"Velocity L1 error {L1_vel:.4f} exceeds tolerance"
    assert L1_u < 0.15, f"Internal energy L1 error {L1_u:.4f} exceeds tolerance"


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(max_ranks=4),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_aneos_shocktube_uniform_state_entropy(num_mpi_ranks, num_omp_threads):
    """The uniform low-density state ahead of the shock must keep its entropy, as the ideal-gas build does (to ~1e-10 at
    t=5). With the MFM face entropic switch compiled out for ANEOS, the Riemann energy flux grew its spread to ~6e-6."""
    test_name = "aneos_shocktube"
    clean_test_outputs(test_name)
    build_gizmo_for_test(test_name, num_omp_threads)
    chdir(f"test/{test_name}/")
    generate_aneos_table()
    if not isfile("shocktube_ics_emass.hdf5"):
        urlretrieve(WEBSITE + "shocktube_ics_emass.hdf5", "shocktube_ics_emass.hdf5")
    add_composition_type_to_ics("shocktube_ics_emass.hdf5")
    run_test(test_name, num_mpi_ranks, num_omp_threads)
    chdir("../../")

    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)
    with h5py.File(final_snap, "r") as F:
        x = F["PartType0/Coordinates"][:, 0]
        rho = F["PartType0/Density"][:]
        u = F["PartType0/InternalEnergy"][:]
    gamma = 1.4
    entropy = (gamma - 1) * u * rho ** (1 - gamma)
    untouched = (x > 12) & (x < 28)  # the initial rho=0.25 state, which the waves have not reached by t=5
    assert untouched.sum() > 10
    spread = np.ptp(entropy[untouched]) / np.median(entropy[untouched])
    assert spread < 1e-8, f"entropy spread {spread:.3e} in the uniform state ahead of the shock"


def write_cold_row_table(fname, T_valid):
    """The gamma=1.4 ideal-gas table of generate_table.py, smaller, ending with a lowest_valid_T_index block that
    marks every row below T_valid as invalid at every density (as the Stewart M-ANEOS tables mark their cold,
    tensile states). Returns the specific energy of the lowest valid row."""
    gamma, cv = 1.4, 1.0 / 0.4
    rho_arr, T_arr = np.logspace(-6, 4, 100), np.logspace(-4, 6, 100)
    k_valid = int(np.searchsorted(T_arr, T_valid))
    with open(fname, "w") as f:
        f.write(f"1400 {len(rho_arr)} {len(T_arr)} 6\n")
        f.writelines(f"{rho:.15e}\n" for rho in rho_arr)
        f.writelines(f"{T:.15e}\n" for T in T_arr)
        for rho in rho_arr:
            for T in T_arr:
                u = cv * T
                P = (gamma - 1.0) * rho * u
                S = cv * np.log(T) - (gamma - 1.0) * cv * np.log(rho) + 100.0
                f.write(f"{rho:.15e} {T:.15e} {P:.15e} {u:.15e} {S:.15e} {np.sqrt(gamma * P / rho):.15e}\n")
        f.write("lowest_valid_T_index\n" + " ".join([str(k_valid)] * len(rho_arr)) + "\n")
    return cv * T_arr[k_valid]


def write_double_rarefaction_ics(fname, n=400, boxsize=4.0):
    """Einfeldt's 1-2-3 problem (rho = 1, u = 1, v = -2 | +2) in a periodic box: the two rarefactions expand and cool
    the centre (the colliding edges shock instead)."""
    x = (np.arange(n) + 0.5) * boxsize / n
    vel = np.zeros((n, 3))
    vel[:, 0] = np.where(x < 0.5 * boxsize, -2.0, 2.0)
    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        h.attrs["NumPart_ThisFile"] = [n, 0, 0, 0, 0, 0]
        h.attrs["NumPart_Total"] = [n, 0, 0, 0, 0, 0]
        h.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
        h.attrs["MassTable"] = [0.0] * 6
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = boxsize
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_DoublePrecision"] = 1
        g = F.create_group("PartType0")
        g.create_dataset("Coordinates", data=np.column_stack([x, np.zeros(n), np.zeros(n)]))
        g.create_dataset("Velocities", data=vel)
        g.create_dataset("ParticleIDs", data=np.arange(1, n + 1, dtype=np.uint32))
        g.create_dataset("Masses", data=np.full(n, boxsize / n))
        g.create_dataset("InternalEnergy", data=np.ones(n))
        g.create_dataset("CompositionType", data=np.zeros(n, dtype=np.int32))


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(max_ranks=4),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_aneos_cold_row_floor(num_mpi_ranks, num_omp_threads):
    """An expansion that drives the specific energy below the table's lowest valid row (u = 0.43 here, from u = 1)
    must leave every element at or above that row, at every snapshot. With only the halving rule elements fell below
    it and kept falling, while the EOS returned the state at the row."""
    test_name = "aneos_shocktube"
    clean_test_outputs(test_name)
    build_gizmo_for_test(test_name, num_omp_threads)
    chdir(f"test/{test_name}/")
    try:
        u_floor = write_cold_row_table("ideal_gas_gamma14_coldrow.sesame", T_valid=0.16)
        write_double_rarefaction_ics("double_rarefaction_ics.hdf5")
        run_test(test_name, num_mpi_ranks, num_omp_threads, param_overrides={
            "InitCondFile": "double_rarefaction_ics", "AneosTable0": "ideal_gas_gamma14_coldrow.sesame",
            "BoxSize": 4, "TimeMax": 1.0, "TimeBetSnapshot": 0.1})
    finally:
        chdir("../../")

    snaps = sorted(glob(f"test/{test_name}/output/snapshot_*.hdf5"))
    assert_final_time(snaps[-1], test_name, time_max=1.0)
    worst = np.inf
    for snap in snaps:
        with h5py.File(snap, "r") as F:
            worst = min(worst, F["PartType0/InternalEnergy"][:].min() / u_floor)
    print(f"min u / u_floor over all snapshots: {worst:.6f}")
    assert worst >= 1 - 1e-9, f"an element fell to u = {worst:.4f} of the lowest valid row's energy"
