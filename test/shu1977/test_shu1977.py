import os
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

    with h5py.File(final_snap, "r") as F:
        coords = F["PartType0/Coordinates"][:]
        rho = F["PartType0/Density"][:]
        boxsize = F["Header"].attrs["BoxSize"]
    plot_shu1977_density_slice(coords, rho, boxsize, output_dir=f"test/{test_name}")


@pytest.mark.parametrize("num_mpi_ranks", (8,))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_shu1977_newborn_sink_radius(num_mpi_ranks, num_omp_threads):
    """A sink formed from a cell takes the sink softening as the floor of its radius (sfr_eff.cc), not the cell's
    softening: the per-particle softening cache must follow the type change. SofteningBndry is raised so this floor
    (kernel radius = 2.8 x softening) exceeds the 0.79 G M / cs^2 term and sets the radius."""
    soft = 3.66e-5
    build_and_run_test("shu1977", num_mpi_ranks, num_omp_threads,
                       param_overrides={"SofteningBndry": soft, "OutputDir": "output_sinkradius"})
    snaps = sorted(glob.glob("test/shu1977/output_sinkradius/snapshot_*.hdf5"))
    with h5py.File(snaps[-1], "r") as f:
        r_sink = f["PartType5/Sink_Radius"][:]
    assert len(r_sink) >= 1, "no sink formed"
    assert np.all(r_sink >= 2.8 * soft * (1 - 1e-6)), f"sink radius {r_sink} below the sink softening {2.8 * soft:.4e}"


def _make_sinkveto_ics(src, dst, sink_pos, sink_mass):
    """shu1977 ICs plus one sink particle (PartType5) at sink_pos."""
    with h5py.File(src, "r") as fi, h5py.File(dst, "w") as fo:
        fi.copy("PartType0", fo)
        hdr = fo.create_group("Header")
        for k, v in fi["Header"].attrs.items():
            hdr.attrs[k] = v
        n_gas = fi["Header"].attrs["NumPart_ThisFile"][0]
        npf = np.array(fi["Header"].attrs["NumPart_ThisFile"]); npf[5] = 1; hdr.attrs["NumPart_ThisFile"] = npf
        npt = np.array(fi["Header"].attrs["NumPart_Total"]); npt[5] = 1; hdr.attrs["NumPart_Total"] = npt
        g = fo.create_group("PartType5")
        g.create_dataset("Coordinates", data=np.array([sink_pos], dtype=np.float64))
        g.create_dataset("Velocities", data=np.zeros((1, 3)))
        g.create_dataset("Masses", data=np.array([sink_mass]))
        g.create_dataset("ParticleIDs", data=np.array([fi["PartType0/ParticleIDs"][:].max() + 1], dtype=fi["PartType0/ParticleIDs"].dtype))


@pytest.mark.parametrize("num_mpi_ranks", (8,))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_shu1977_sink_veto_inactive_sink(num_mpi_ranks, num_omp_threads):
    """No sink may form within an existing sink's softening (sfr_eff.cc, SINGLE_STAR_SINK_FORMATION & 8), whether or
    not that sink is active on the step: the neighbour flag is set only by an active sink, so the veto also checks the
    gravity-walk distance against max(h, sink softening). A seed sink sits 1e-4 from the collapse centre: outside the
    core cell's kernel and the sink's capture radius (= SofteningBndry), inside its softening kernel (2.8 SofteningBndry).
    The core then cannot form a sink and keeps collapsing, so the run is capped by TimeLimitCPU; the check is on the
    formation log."""
    soft, d = 5e-5, 1e-4
    _make_sinkveto_ics("test/shu1977/shu1977_ics.hdf5", "test/shu1977/shu1977_sinkveto_ics.hdf5", [d, 0.0, 0.0], 7.8125e-07)
    try:
        build_and_run_test("shu1977", num_mpi_ranks, num_omp_threads,
                           param_overrides={"InitCondFile": "shu1977_sinkveto_ics", "SofteningBndry": soft,
                                            "OutputDir": "output_sinkveto", "TimeLimitCPU": 600,
                                            # with formation vetoed the isothermal core collapses unchecked: let it
                                            # crawl to the CPU cap instead of stopping at the timestep floor
                                            "MinSizeTimestep": 0,
                                            # the unit system and time axis this setup is defined in (some
                                            # parameter files carry km/s units, which make the gas far too hot to collapse)
                                            "UnitVelocity_in_cm_per_s": 1.0e2, "TimeMax": 2e-5, "MaxSizeTimestep": 5e-6,
                                            "TimeBetSnapshot": 1e-6, "TimeBetStatistics": 1e-6})
    except RuntimeError as e:  # the TimeLimitCPU cap is the intended end of this run
        if "never reported reaching TimeMax" not in str(e):
            raise
    ledger = "test/shu1977/output_sinkveto/sink_events/formation.npy"   # sink event ledger, where written
    if os.path.isfile(ledger):
        dist = [float(x) for x in np.load(ledger)["min_dist_to_sink"]]
    else:
        rows = [l.split() for f in glob.glob("test/shu1977/output_sinkveto/sink_details/sink_formation_*.txt")
                for l in open(f) if l.strip() and not l.startswith("#")]
        dist = [float(r[18]) for r in rows]
    assert all(x >= 2.8 * soft * (1 - 1e-6) for x in dist), f"sinks formed within the softening of a sink: {dist}"
