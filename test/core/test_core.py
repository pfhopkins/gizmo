"""Magnetized core collapse test (Hopkins 2015)

Tests the collapse of a magnetized molecular cloud core with a barotropic
EOS. Runs three variants (no constrained gradient, MHD_CONSTRAINED_GRADIENT=1,
MHD_CONSTRAINED_GRADIENT=2) to t=0.05 with snapshots every 0.005, then
produces a multi-panel plot comparing the divergence error |divB dx| / |B|
over time for each variant.
"""

import pytest
import numpy as np
import matplotlib
import h5py
import glob
from gizmo.test import (
    build_and_run_test, default_mpi_ranks, clean_test_outputs,
    assert_final_time, default_omp_threads, variant_output_dir,
)

# Variants: (label, extra Config.sh flags beyond OUTPUT_BFIELD_DIVCLEAN_INFO)
_VARIANTS = [
    (0, "Default",              ()),
    (1, "CONSTRAINED_GRADIENT=1", ("MHD_CONSTRAINED_GRADIENT=1",)),
    (2, "CONSTRAINED_GRADIENT=2", ("MHD_CONSTRAINED_GRADIENT=2",)),
]


def _extra_flags(variant_flags):
    return ("OUTPUT_BFIELD_DIVCLEAN_INFO",) + variant_flags


def _divb_error(snap_path):
    """Return per-particle |divB * dx| / |B| from a snapshot."""
    with h5py.File(snap_path, "r") as F:
        t    = float(F["Header"].attrs["Time"])
        divB = F["PartType0/DivergenceOfMagneticField"][:]
        B    = F["PartType0/MagneticField"][:]
        mass = F["PartType0/Masses"][:]
        rho  = F["PartType0/Density"][:]
    Bnorm = np.linalg.norm(B, axis=1)
    dx    = (mass / rho) ** (1.0 / 3.0)
    err   = np.abs(divB) * dx / np.maximum(Bnorm, 1e-30 * float(Bnorm.max()))
    return t, err


def _plot_divb_comparison(test_name, variant_snap_lists):
    """Multi-panel time-series plot of |divB dx| / |B| for each variant."""
    from matplotlib import pyplot as plt
    matplotlib.use("Agg")
    n = len(variant_snap_lists)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (cg, label, _flags, snaps) in zip(axes, variant_snap_lists):
        times, medians, p90s, maxs = [], [], [], []
        for snap in snaps:
            t, err = _divb_error(snap)
            times.append(t)
            medians.append(np.median(err))
            p90s.append(np.percentile(err, 90))
            maxs.append(err.max())

        times   = np.array(times)
        medians = np.array(medians)
        p90s    = np.array(p90s)
        maxs    = np.array(maxs)

        ax.semilogy(times, medians, label="median")
        ax.semilogy(times, p90s,    label="90th pct", linestyle="--")
        ax.semilogy(times, maxs,    label="max",       linestyle=":")
        ax.set_title(label)
        ax.set_xlabel("Time (code units)")
        ax.legend(fontsize=8)

    axes[0].set_ylabel(r"$|\nabla \cdot \mathbf{B}|\,\Delta x\;/\;|\mathbf{B}|$")
    fig.tight_layout()
    out = f"test/{test_name}/divb_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


@pytest.mark.parametrize("num_mpi_ranks",   (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_core(num_mpi_ranks, num_omp_threads):
    test_name = "core"
    variant_snap_lists = []

    for cg, label, vflags in _VARIANTS:
        flags = _extra_flags(vflags)
        clean_test_outputs(test_name, flags)
        build_and_run_test(test_name, num_mpi_ranks, num_omp_threads,
                           extra_config_flags=flags)

        outputdir = variant_output_dir(test_name, flags)
        snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
        if len(snaps) < 2:
            raise RuntimeError(
                f"GIZMO did not produce snapshots for variant '{label}'."
            )
        assert_final_time(snaps[-1], test_name)

        with h5py.File(snaps[0],  "r") as F:
            rho0   = F["PartType0/Density"][:]
            B0     = F["PartType0/MagneticField"][:]
            mass0  = F["PartType0/Masses"][:]
        with h5py.File(snaps[-1], "r") as F:
            rho_f  = F["PartType0/Density"][:]
            Bf     = F["PartType0/MagneticField"][:]
            mass_f = F["PartType0/Masses"][:]

        rho_max_ratio = rho_f.max() / rho0.max()
        assert rho_max_ratio > 10, (
            f"[{label}] Core did not collapse enough: "
            f"max density ratio = {rho_max_ratio:.1f}"
        )

        Bmag0_max = float(np.linalg.norm(B0, axis=1).max())
        Bmagf_max = float(np.linalg.norm(Bf, axis=1).max())
        assert Bmagf_max > Bmag0_max, (
            f"[{label}] Magnetic field should be amplified during collapse"
        )

        mass_err = abs(mass_f.sum() - mass0.sum()) / mass0.sum()
        assert mass_err < 1e-3, (
            f"[{label}] Mass not conserved: relative error {mass_err:.6f}"
        )

        variant_snap_lists.append((cg, label, vflags, snaps))

    _plot_divb_comparison(test_name, variant_snap_lists)
