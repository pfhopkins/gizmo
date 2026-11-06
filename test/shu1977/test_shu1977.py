"""Shu (1977) singular isothermal sphere collapse test: checks that exactly one sink particle forms."""

import pytest
import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
from os import path
from meshoid import Meshoid
from gizmo.test import build_and_run_test, flush_colorbar, assert_final_time, get_final_snapshot, default_omp_threads


def plot_shu1977_density_slice(coords, rho, boxsize, output_dir="."):
    """Plot a density slice through the Shu 1977 collapse center."""
    center = np.average(coords, axis=0)
    size = boxsize * 0.1
    M = Meshoid(coords)
    rho_slice = M.Slice(np.log10(rho), res=1024, plane="z", center=center, size=size, order=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    half = size / 2
    im = ax.imshow(rho_slice.T, origin="lower", cmap="inferno", extent=[-half, half, -half, half])
    flush_colorbar(im, ax=ax, label="log10(Density)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Shu 1977 Collapse - Density Slice")
    fig.savefig(output_dir + "/Density_2D.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


@pytest.mark.xfail(strict=True, reason=(
    "fixed by pr/test-fixes: params in m/s against a km/s adiabat, c_s 0.2 m/s, m_sink/m_ff 0.983"))
@pytest.mark.parametrize("num_mpi_ranks", (8,))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_shu1977(num_mpi_ranks, num_omp_threads):
    test_name = "shu1977"
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)

    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)

    with h5py.File(final_snap, "r") as f:
        num_sinks = f["Header"].attrs["NumPart_ThisFile"][5]

    assert num_sinks == 1, f"Expected exactly 1 PartType5 particle, got {num_sinks}"

    # Pressure support. A shell of this overdense isothermal sphere (M(<r) = k r) would reach the
    # centre at t = (pi/2) r / sqrt(2 G k) without pressure, so the sink would hold
    # M_ff(t) = (2/pi) sqrt(2 G) k^1.5 t. Gas that is effectively pressureless (e.g. an
    # EOS_ENFORCE_ADIABAT given in different velocity units from the params file) still makes
    # exactly one sink, but accretes at that rate.
    with h5py.File(path.join(path.dirname(final_snap), "snapshot_000.hdf5"), "r") as F:
        G = F["Header"].attrs["Gravitational_Constant_In_Code_Inits"]
        x = F["PartType0/Coordinates"][:].astype(np.float64)
        m = F["PartType0/Masses"][:].astype(np.float64)
    r = np.linalg.norm(x - np.average(x, axis=0, weights=m), axis=1)
    order = np.argsort(r)
    k = np.median((np.cumsum(m[order]) / r[order])[len(m) // 10 : 9 * len(m) // 10])
    with h5py.File(final_snap, "r") as F:
        t_final = F["Header"].attrs["Time"]
        m_sink = F["PartType5/Masses"][:].sum()
    m_ff = (2 / np.pi) * np.sqrt(2 * G) * k**1.5 * t_final
    assert m_sink < 0.9 * m_ff, (
        f"sink mass {m_sink:.4g} is {m_sink / m_ff:.3f} of the pressureless free-fall value "
        f"{m_ff:.4g}: the gas has no effective pressure support"
    )

    with h5py.File(final_snap, "r") as F:
        coords = F["PartType0/Coordinates"][:]
        rho = F["PartType0/Density"][:]
        boxsize = F["Header"].attrs["BoxSize"]
    plot_shu1977_density_slice(coords, rho, boxsize, output_dir=f"test/{test_name}")
