"""1D C-type shock test with ambipolar diffusion

Tests propagation of a C-type shock driven by non-ideal MHD (ambipolar
diffusion dominated, with small Ohmic resistivity). The non-ideal MHD
coefficients are set following the original test problem:
  eta_ad = 1e5 * vA^2  (where vA^2 = B^2/rho)
  eta_ohmic = 1e-5 * eta_ad
  eta_hall = 0

The non-ideal MHD coefficients are injected by temporarily patching eos/eos.cc
before building GIZMO. The original file is always restored after the build,
so no permanent modifications are made to the source tree.
"""

import pytest
import shutil
import numpy as np
from os import path
from os.path import isfile
from urllib.request import urlretrieve, HTTPError
from matplotlib import pyplot as plt
import h5py
from gizmo.test import (
    build_gizmo_for_test,
    run_test,
    clean_test_outputs,
    assert_final_time,
    assert_snapshots_are_close,
    default_mpi_ranks,
    default_omp_threads,
    get_final_snapshot,
)

WEBSITE = "http://www.tapir.caltech.edu/~phopkins/sims/"

# Code block injected into calculate_and_assign_nonideal_mhd_coefficients
_PATCH_MARKER = "/* C_SHOCK_TEST: fixed coefficients */"
_PATCH_CODE = f"""\
    {_PATCH_MARKER}
    double eta_ad_base = 1.e5;
    double eps_om = 1.e-5;
    double B2 = cell[i].Bfield().norm_sq();
    double vA2 = B2 / cell[i].Density;
    double eta_ad_val = eta_ad_base * vA2;
    cell[i].Eta_MHD_OhmicResistivity_Coeff = eps_om * eta_ad_val;
    cell[i].Eta_MHD_HallEffect_Coeff = 0;
    cell[i].Eta_MHD_AmbiPolarDiffusion_Coeff = eta_ad_val;
    return;
"""

# Anchor line in eos.cc right after which the patch is inserted
_ANCHOR = "#ifdef MHD_NON_IDEAL\n"

EOS_FILE = "eos/eos.cc"
EOS_BACKUP = "eos/eos.cc.c_shock_backup"

# Reference snapshot for comparison
REFERENCE_SNAP = "test/c_shock/c_shock_exact.hdf5"

# Plot x-axis range (zoomed to shock region)
PLOT_XLIM = (3.05e7, 3.4e7)


def _download_if_missing(filename):
    """Download a file from the tapir server if not already present."""
    if not isfile(filename):
        try:
            urlretrieve(WEBSITE + path.basename(filename), filename)
        except HTTPError:
            print(f"Could not download {filename} from {WEBSITE}")


def _patch_eos():
    """Patch eos.cc to use C-shock non-ideal MHD coefficients. Returns True if
    the patch was applied (so we know to restore later)."""
    if not path.isfile(EOS_FILE):
        raise FileNotFoundError(f"Cannot find {EOS_FILE} to patch")

    with open(EOS_FILE, "r") as f:
        src = f.read()

    if _PATCH_MARKER in src:
        return False

    func_sig = "void calculate_and_assign_nonideal_mhd_coefficients("
    func_pos = src.find(func_sig)
    if func_pos == -1:
        raise RuntimeError(f"Cannot find {func_sig} in {EOS_FILE}")

    anchor_pos = src.find(_ANCHOR, func_pos)
    if anchor_pos == -1:
        raise RuntimeError(f"Cannot find '{_ANCHOR.strip()}' after function signature in {EOS_FILE}")

    insert_pos = anchor_pos + len(_ANCHOR)

    shutil.copy2(EOS_FILE, EOS_BACKUP)
    patched = src[:insert_pos] + _PATCH_CODE + src[insert_pos:]
    with open(EOS_FILE, "w") as f:
        f.write(patched)

    return True


def _restore_eos():
    """Restore eos.cc from backup."""
    if path.isfile(EOS_BACKUP):
        shutil.move(EOS_BACKUP, EOS_FILE)


def plot_c_shock_overview(snap1, snap2, output_dir, label1="Reference", label2="Test"):
    """Plot 6-panel overview of Density, InternalEnergy, Vx, Vy, Bx, By
    zoomed to the shock region."""
    data = {}
    for label, snap_path in [(label1, snap1), (label2, snap2)]:
        with h5py.File(snap_path, "r") as F:
            x = F["PartType0/Coordinates"][:, 0]
            order = x.argsort()
            data[label] = {
                "x": x[order],
                "Density": F["PartType0/Density"][:][order],
                "InternalEnergy": F["PartType0/InternalEnergy"][:][order],
                "Velocity_x": F["PartType0/Velocities"][:, 0][order],
                "Velocity_y": F["PartType0/Velocities"][:, 1][order],
                "MagneticField_x": F["PartType0/MagneticField"][:, 0][order],
                "MagneticField_y": F["PartType0/MagneticField"][:, 1][order],
            }

    fields = ["Density", "InternalEnergy", "Velocity_x", "Velocity_y",
              "MagneticField_x", "MagneticField_y"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for ax, name in zip(axes.flat, fields):
        d1 = data[label1]
        d2 = data[label2]
        ax.plot(d1["x"], d1[name], ".", markersize=1, label=label1, alpha=0.5)
        ax.plot(d2["x"], d2[name], ".", markersize=1, label=label2, color="red")
        ax.set_ylabel(name)
        ax.set_xlabel("x")
        ax.set_xlim(*PLOT_XLIM)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path.join(output_dir, "c_shock_overview.png"), dpi=150)
    plt.close(fig)


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(4),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_c_shock(num_mpi_ranks, num_omp_threads):
    test_name = "c_shock"
    clean_test_outputs(test_name)

    # Patch eos.cc with C-shock non-ideal MHD coefficients, build, then restore
    patched = False
    try:
        patched = _patch_eos()
        build_gizmo_for_test(test_name, num_omp_threads)
    finally:
        if patched:
            _restore_eos()

    # Download ICs and reference if not present, then run
    from os import chdir, getcwd
    cwd = getcwd()
    try:
        chdir(f"test/{test_name}/")
        _download_if_missing("c_shock_ics.hdf5")
        _download_if_missing("c_shock_exact.hdf5")
        run_test(test_name, num_mpi_ranks, num_omp_threads)
    finally:
        chdir(cwd)

    # Verify the simulation completed
    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)

    # Plot 6-panel overview zoomed to the shock region
    plot_c_shock_overview(
        REFERENCE_SNAP, final_snap,
        output_dir=f"test/{test_name}",
    )
    
    # Compare final snapshot against reference numerical solution
    fields = ("Density", "InternalEnergy", "Velocities", "MagneticField")
    assert_snapshots_are_close(
        REFERENCE_SNAP, final_snap,
        fields_to_compare=fields,
        rtol=0.05, atol=1e-5,
    )

