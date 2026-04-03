"""1D Hall MHD wave propagation test (Berlok & Pfrommer 2023, arXiv:2309.15907)

Tests propagation of whistler/ion-cyclotron waves with fixed non-ideal MHD
coefficients (eta_hall=0.01, eta_ohmic=0.0002, eta_ad=0). Compares the final
snapshot against a reference numerical solution.

The non-ideal MHD coefficients are injected by temporarily patching eos/eos.cc
before building GIZMO. The original file is always restored after the build,
so no permanent modifications are made to the source tree.
"""

import pytest
import shutil
import numpy as np
from glob import glob
from os import path
from os.path import isfile
from urllib.request import urlretrieve, HTTPError
from matplotlib import pyplot as plt
import matplotlib
import h5py
from gizmo.test import (
    build_gizmo_for_test,
    run_test,
    clean_test_outputs,
    assert_final_time,
    assert_snapshots_are_close,
    plot_1D_snapshot_comparison,
    default_mpi_ranks,
    default_omp_threads,
    get_final_snapshot,
)

# Fixed non-ideal MHD coefficients for this test (code units)
ETA_HALL = 0.01
ETA_OHMIC = 0.02 * ETA_HALL  # 0.0002
ETA_AD = 0.0

# Code block injected into calculate_and_assign_nonideal_mhd_coefficients
_PATCH_MARKER = "/* HALL_WAVE_TEST: fixed coefficients */"
_PATCH_CODE = f"""\
    {_PATCH_MARKER}
    cell[i].Eta_MHD_OhmicResistivity_Coeff = {ETA_OHMIC};
    cell[i].Eta_MHD_HallEffect_Coeff = {ETA_HALL};
    cell[i].Eta_MHD_AmbiPolarDiffusion_Coeff = {ETA_AD};
    return;
"""

# Anchor line in eos.cc right after which the patch is inserted
_ANCHOR = "#ifdef MHD_NON_IDEAL\n"

EOS_FILE = "eos/eos.cc"
EOS_BACKUP = "eos/eos.cc.hall_wave_backup"

# Reference snapshot for comparison (current known-good output at TimeMax=79)
WEBSITE = "http://www.tapir.caltech.edu/~phopkins/sims/"
REFERENCE_SNAP = "test/hall_wave/hall_wave_exact.hdf5"


def _download_if_missing(filename):
    """Download a file from the tapir server if not already present."""
    if not isfile(filename):
        try:
            urlretrieve(WEBSITE + path.basename(filename), filename)
        except HTTPError:
            print(f"Could not download {filename} from {WEBSITE}")


def _patch_eos():
    """Patch eos.cc to use fixed non-ideal MHD coefficients. Returns True if
    the patch was applied (so we know to restore later)."""
    if not path.isfile(EOS_FILE):
        raise FileNotFoundError(f"Cannot find {EOS_FILE} to patch")

    with open(EOS_FILE, "r") as f:
        src = f.read()

    if _PATCH_MARKER in src:
        return False  # already patched (shouldn't happen, but be safe)

    # Find the function and its first #ifdef MHD_NON_IDEAL
    func_sig = "void calculate_and_assign_nonideal_mhd_coefficients("
    func_pos = src.find(func_sig)
    if func_pos == -1:
        raise RuntimeError(f"Cannot find {func_sig} in {EOS_FILE}")

    # Find the first #ifdef MHD_NON_IDEAL after the function signature
    anchor_pos = src.find(_ANCHOR, func_pos)
    if anchor_pos == -1:
        raise RuntimeError(f"Cannot find '{_ANCHOR.strip()}' after function signature in {EOS_FILE}")

    insert_pos = anchor_pos + len(_ANCHOR)

    # Back up and write patched version
    shutil.copy2(EOS_FILE, EOS_BACKUP)
    patched = src[:insert_pos] + _PATCH_CODE + src[insert_pos:]
    with open(EOS_FILE, "w") as f:
        f.write(patched)

    return True


def _restore_eos():
    """Restore eos.cc from backup."""
    if path.isfile(EOS_BACKUP):
        shutil.move(EOS_BACKUP, EOS_FILE)


def plot_time_evolution(output_dir, plot_dir, x0=0.5):
    """Plot Bx, By, Bz, vx, vy, vz versus time for a specific cell,
    tracked by particle ID across all snapshots."""
    snaps = sorted(glob(path.join(output_dir, "snapshot_*.hdf5")))
    n_snaps = len(snaps)
    if n_snaps == 0:
        return

    t = np.zeros(n_snaps)
    vx = np.zeros(n_snaps); vy = np.zeros(n_snaps); vz = np.zeros(n_snaps)
    bx = np.zeros(n_snaps); by = np.zeros(n_snaps); bz = np.zeros(n_snaps)

    id0 = None
    for i, snap in enumerate(snaps):
        with h5py.File(snap, "r") as F:
            t[i] = float(F["Header"].attrs["Time"])
            ids = F["PartType0/ParticleIDs"][:]
            coords = F["PartType0/Coordinates"][:]
            vel = F["PartType0/Velocities"][:]
            mag = F["PartType0/MagneticField"][:]

        x = coords[:, 0]
        order = np.argsort(x)
        x = x[order]; ids = ids[order]
        vel = vel[order]; mag = mag[order]

        if id0 is None:
            id0 = ids[np.argmin(np.abs(x - x0))]

        q = np.where(ids == id0)[0][0]
        vx[i] = vel[q, 0]; vy[i] = vel[q, 1]; vz[i] = vel[q, 2]
        bx[i] = mag[q, 0]; by[i] = mag[q, 1]; bz[i] = mag[q, 2]

    # Normalization constants (same as hall_IC_analysis.py)
    dB_B0 = 1.e-3
    B0 = 0.1 / np.sqrt(4. * np.pi)
    dB = dB_B0 * B0
    k = 2. * np.pi * 5.
    rho0 = 1.
    vA = B0 / np.sqrt(rho0)
    sigma = ETA_HALL * k * k / 2.
    omega = np.sqrt(vA * vA * k * k + sigma * sigma)
    dv = -dB_B0 * omega / k
    omega_H = vA * vA / ETA_HALL
    t0plot = omega_H / (2. * np.pi)

    bx_plot = bx - B0  # subtract background field

    matplotlib.rcParams.update({'font.size': 14.})
    fig, ax = plt.subplots(1, 1, figsize=(6., 4.5))
    ax.plot(t * t0plot, vx / dv, color='blue', linestyle='--', label=r'$v_{x}/\delta v$', linewidth=1.)
    ax.plot(t * t0plot, vy / dv, color='blue', linestyle='-', label=r'$v_{y}/\delta v$', linewidth=1.)
    ax.plot(t * t0plot, vz / dv, color='blue', linestyle=':', label=r'$v_{z}/\delta v$', linewidth=1.)
    ax.plot(t * t0plot, bx_plot / dB, color='red', linestyle='--', label=r'$B_{x}/\delta B$', linewidth=2.)
    ax.plot(t * t0plot, by / dB, color='red', linestyle='-', label=r'$B_{y}/\delta B$', linewidth=2.)
    ax.plot(t * t0plot, bz / dB, color='red', linestyle=':', label=r'$B_{z}/\delta B$', linewidth=2.)

    ax.set_xlabel(r'Time: $\omega_{H} t/2\pi$')
    ax.set_ylabel(r'Value')
    ax.set_ylim(-1., 1.)
    ax.legend(loc='best', fontsize=14. * 0.8, handletextpad=0.5, borderpad=0.5,
              labelcolor='linecolor', handlelength=3., frameon=False,
              columnspacing=0.2, labelspacing=0.2, numpoints=1)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=1 - 0.01, top=1 - 0.01)
    fig.savefig(path.join(plot_dir, "hall_wave_time_evolution.png"),
                bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close(fig)


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(4),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_hall_wave(num_mpi_ranks, num_omp_threads):
    test_name = "hall_wave"
    clean_test_outputs(test_name)

    # Patch eos.cc with fixed non-ideal MHD coefficients, build, then restore
    patched = False
    try:
        patched = _patch_eos()
        build_gizmo_for_test(test_name, num_omp_threads)
    finally:
        if patched:
            _restore_eos()

    # Download ICs and reference if not present, then run
    from os import chdir
    chdir(f"test/{test_name}/")
    _download_if_missing("hall_wave_ics.hdf5")
    _download_if_missing("hall_wave_exact.hdf5")
    run_test(test_name, num_mpi_ranks, num_omp_threads)
    chdir("../../")

    # Verify the simulation completed
    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)

    # Compare final snapshot against reference numerical solution
    fields = ("Velocities", "MagneticField")
    assert_snapshots_are_close(
        REFERENCE_SNAP, final_snap,
        fields_to_compare=fields,
        rtol=0.05, atol=1e-5,
    )

    # Also plot 1D comparison of initial vs final and reference vs final
    plot_1D_snapshot_comparison(
        REFERENCE_SNAP, final_snap,
        fields_to_plot=fields,
        output_dir=f"test/{test_name}",
    )

    # Plot time evolution of B and v components for a tracked cell
    plot_time_evolution(
        output_dir=f"test/{test_name}/output",
        plot_dir=f"test/{test_name}",
    )
