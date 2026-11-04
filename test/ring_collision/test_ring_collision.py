"""Ring collision elastic/solid body test (Hopkins 2015)

Tests elastic solid body physics by colliding two rings. The rings should
deform on impact and bounce apart, preserving their structure due to
elastic restoring forces. With the Tillotson EOS parameters set here,
P = cs^2 * (rho - rho_0) with cs = rho_0 = 1.
"""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from meshoid import Meshoid
from gizmo.test import build_and_run_test, default_mpi_ranks, clean_test_outputs, flush_colorbar, assert_final_time, default_omp_threads


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_ring_collision(num_mpi_ranks, num_omp_threads):
    test_name = "ring_collision"
    clean_test_outputs(test_name)
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    outputdir = f"test/{test_name}/output"
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError("GIZMO did not run successfully.")
    assert_final_time(snaps[-1], test_name)

    # Plot each snapshot using Meshoid slice interpolation
    for snap in snaps:
        with h5py.File(snap, "r") as F:
            pos = F["PartType0/Coordinates"][:]
            rho = F["PartType0/Density"][:]
            t = F["Header"].attrs["Time"]
        center = np.array([pos[:, 0].mean(), pos[:, 1].mean(), pos[:, 2].mean() if pos.shape[1] > 2 else 0.])
        size = max(pos[:, 0].max() - pos[:, 0].min(), pos[:, 1].max() - pos[:, 1].min()) * 1.1
        M = Meshoid(pos)
        rho_slice = M.Slice(rho, res=1024, plane="z", center=center, size=size, order=0)
        fig, ax = plt.subplots(figsize=(6, 6))
        extent = [center[0] - size/2, center[0] + size/2, center[1] - size/2, center[1] + size/2]
        im = ax.imshow(rho_slice.T, origin="lower", cmap="viridis", extent=extent)
        flush_colorbar(im, ax=ax, label="Density")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Ring Collision t={t:.0f}")
        fig.savefig(f"test/{test_name}/snapshot_t{t:.0f}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Load initial and final snapshots
    with h5py.File(snaps[0], "r") as F:
        mass0 = F["PartType0/Masses"][:]
        pos0 = F["PartType0/Coordinates"][:]
    with h5py.File(snaps[-1], "r") as F:
        mass_f = F["PartType0/Masses"][:]
        pos_f = F["PartType0/Coordinates"][:]

    # Mass conservation
    mass_err = abs(mass_f.sum() - mass0.sum()) / mass0.sum()
    assert mass_err < 1e-3, f"Mass not conserved: relative error {mass_err:.6f}"

    # After the collision, the rings should have bounced apart.
    # Check that the particles span a larger x-range than initially
    # (they started moving toward each other, collided, and bounced back)
    x_spread0 = pos0[:, 0].max() - pos0[:, 0].min()
    x_spread_f = pos_f[:, 0].max() - pos_f[:, 0].min()
    assert x_spread_f > 0.5 * x_spread0, (
        "Rings appear to have collapsed rather than bouncing"
    )


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_ring_collision_restitution(num_mpi_ranks, num_omp_threads):
    """By t=200 the rings have bounced apart. Their coefficient of restitution (relative centre-of-mass velocity after /
    before) measures the energy the elastic face dissipation takes from the collision: acting on the raw velocity
    difference, it damped every resolved gradient and the rings came apart at 0.53 of their approach speed (0.66 when
    it acts on the jump between the reconstructed face states)."""
    test_name = "ring_collision"
    time_max = 200
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads, param_overrides={"TimeMax": time_max})
    snaps = sorted(glob.glob(f"test/{test_name}/output/snapshot_*.hdf5"))
    assert_final_time(snaps[-1], test_name, time_max=time_max)

    def ring_velocities(snap, left_ids):
        with h5py.File(snap, "r") as F:
            ids = F["PartType0/ParticleIDs"][:]
            m = F["PartType0/Masses"][:]
            v = F["PartType0/Velocities"][:, 0]
        left = np.isin(ids, left_ids)
        return (m[left] * v[left]).sum() / m[left].sum(), (m[~left] * v[~left]).sum() / m[~left].sum()

    with h5py.File(snaps[0], "r") as F:
        ids0 = F["PartType0/ParticleIDs"][:]
        x0 = F["PartType0/Coordinates"][:, 0]
    left_ids = ids0[x0 < np.median(x0)]
    vl0, vr0 = ring_velocities(snaps[0], left_ids)
    vl, vr = ring_velocities(snaps[-1], left_ids)
    restitution = (vr - vl) / (vl0 - vr0)
    print(f"restitution {restitution:.4f} (relative velocity {vr0 - vl0:+.5f} -> {vr - vl:+.5f})")
    assert restitution > 0.6, f"restitution {restitution:.3f}"
