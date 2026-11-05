"""Sod shock tube test (Hopkins 2015)

Compares the final snapshot against the reference high-resolution PPM solution.
The exact solution file has columns: x, density, pressure, entropy, x_velocity.
"""

import pytest
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import h5py
from os import chdir, getcwd
from os.path import isfile
from urllib.request import urlretrieve
from gizmo.test import build_gizmo_for_test, run_test, stash_baseline_output, finalize_variant_output, clean_test_outputs, assert_final_time, assert_energy_conserved, default_mpi_ranks, default_omp_threads, get_final_snapshot, variant_suffix


WEBSITE = "http://www.tapir.caltech.edu/~phopkins/sims/"
# Both solvers conserve thermal+kinetic energy to 1.3e-6 here (measured 2026-11-05); 1e-4 leaves margin. A NaN
# CD10 viscosity limiter for gas at rest once wiped out the SPH thermal energy at the first kick (E: 118 -> 0.5).
ENERGY_TOL = 1e-4
# SPH L1 errors measured 2026-11-05: rho 0.0071, vel 0.060, u 0.0051 (MFM: 0.0057, 0.042, 0.0038); ~2x margin
SPH_L1_TOL = (0.015, 0.12, 0.012)
# artificial viscosity/conductivity parameters, read only by SPH builds
SPH_PARAMS = {"ArtBulkViscConst": 1.0, "ArtCondConstant": 0.25, "ViscosityAMin": 0.05, "ViscosityAMax": 2.0}

# per variant: extra Config flags, params overrides, L1 tolerances (rho, vel, u), energy-conservation tolerance
VARIANTS = [
    pytest.param((), None, (0.05, 0.1, 0.1), ENERGY_TOL, id="mfm"),
    pytest.param(("HYDRO_DENSITY_SPH",), SPH_PARAMS, SPH_L1_TOL, ENERGY_TOL, id="sph"),
]


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(max_ranks=4),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize("extra_config_flags,param_overrides,l1_tol,energy_tol", VARIANTS)
def test_shocktube(num_mpi_ranks, num_omp_threads, extra_config_flags, param_overrides, l1_tol, energy_tol):
    test_name = "shocktube"
    clean_test_outputs(test_name, extra_config_flags)

    build_gizmo_for_test(test_name, num_omp_threads, extra_config_flags)
    stash_baseline_output(test_name, extra_config_flags)
    cwd = getcwd()
    try:
        chdir(f"test/{test_name}/")
        try:
            # Download ICs (non-standard name, so not via download_test_files) and exact solution
            for f in ("shocktube_ics_emass.hdf5", "shocktube_exact.txt"):
                if not isfile(f):
                    urlretrieve(WEBSITE + f, f)
            run_test(test_name, num_mpi_ranks, num_omp_threads, param_overrides=param_overrides)
        finally:
            chdir(cwd)
    finally:
        finalize_variant_output(test_name, extra_config_flags)

    final_snap = get_final_snapshot(test_name, extra_config_flags)
    assert_final_time(final_snap, test_name)
    # closed periodic box without gravity: thermal + kinetic is the whole budget
    e_ratio = assert_energy_conserved(test_name, extra_config_flags, tol=energy_tol)
    print(f"shocktube[{extra_config_flags}]: E_final/E_initial - 1 = {e_ratio - 1:.3e}")

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
    # Internal energy from pressure: u = P / (rho * (gamma - 1))
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
        plt.plot(x_sim[order], sim[order], ".", markersize=1, label="GIZMO")
        plt.plot(x_sim[order], exact_vals[order], "-", color="red", linewidth=0.5, label="Exact")
        plt.xlabel("x")
        plt.ylabel(label)
        plt.legend()
        plt.savefig(f"test/{test_name}/{label}{variant_suffix(extra_config_flags)}.png")
        plt.close()

    print(f"shocktube[{extra_config_flags}]: L1 rho={L1_rho:.4f} vel={L1_vel:.4f} u={L1_u:.4f}")
    tol_rho, tol_vel, tol_u = l1_tol
    assert L1_rho < tol_rho, f"Density L1 error {L1_rho:.4f} exceeds tolerance {tol_rho}"
    assert L1_vel < tol_vel, f"Velocity L1 error {L1_vel:.4f} exceeds tolerance {tol_vel}"
    assert L1_u < tol_u, f"Internal energy L1 error {L1_u:.4f} exceeds tolerance {tol_u}"
