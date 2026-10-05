"""Test of a 120msun protostellar core with a r^-2 density profile; should have approximately constant, high accretion rate and a smooth profile in gas, radiation, and dust temperature"""

import pytest
from gizmo.test import (
    build_and_run_test,
    get_cooling_tables,
    assert_final_time,
    default_omp_threads,
    default_mpi_ranks,
    get_final_snapshot,
)

from matplotlib import pyplot as plt
import h5py
from astropy import units as u, constants as c
from scipy.stats import binned_statistic
import numpy as np


def plot_quantiles_vs_radius(radius, quantity, radius_bins=np.logspace(-4, -1, 21), plotargs={}, label=None):
    quantiles = [
        binned_statistic(radius, quantity, lambda x: np.percentile(x, q), radius_bins)[0] for q in (16, 50, 84)
    ]

    centers = np.sqrt(radius_bins[1:] * radius_bins[:-1])
    plt.loglog(centers, quantiles[1], label=label, **plotargs)
    fill_kwargs = {k: v for k, v in plotargs.items()
                   if k not in ("marker", "markersize", "markerfacecolor", "linestyle", "alpha")}
    plt.fill_between(centers, quantiles[0], quantiles[2], **fill_kwargs, alpha=0.2)


_VARIANT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_VARIANT_COLORS = ["C0", "C2", "C1", "C3", "C4", "C5", "C6", "C7"]


def _short(label, maxlen=30):
    return label if len(label) <= maxlen else label[: maxlen - 3] + "..."


_variant_data = {}


def _variant_label(extra_config_flags):
    return "+".join(extra_config_flags) if extra_config_flags else "baseline"


def _load_shu_data(f):
    with h5py.File(f, "r") as F:
        # code energy density -> eV/cm^3. The velocity unit is read per file rather
        # than hardcoded, so the reference solution (written when code velocities were
        # m/s) stays comparable with runs in the current km/s units.
        v_unit = float(F["Header"].attrs["UnitVelocity_In_CGS"]) * u.cm / u.s
        code_to_evcm3 = (v_unit**2 * u.Msun / u.pc**3).to_value(u.eV / u.cm**3)
        rho = F["PartType0/Density"][:]
        d = {
            "xe": F["PartType0/ElectronAbundance"][:],
            "T": F["PartType0/Temperature"][:],
            "Trad": F["PartType0/IRBand_Radiation_Temperature"][:],
            "Tdust": F["PartType0/Dust_Temperature"][:],
            "urad_FIR": F["PartType0/PhotonEnergy"][:][:, 4] * (rho / F["PartType0/Masses"][:]) * code_to_evcm3,
        }
        rvec = F["PartType0/Coordinates"][:] - F["PartType5/Coordinates"][0]
        d["r"] = np.sum(rvec * rvec, axis=1) ** 0.5
    return d


def _render_combined_shu_plots(test_dir):
    """Re-render r-vs-X plots with the reference solution + all accumulated variants overlaid."""
    ref = _load_shu_data(f"{test_dir}/shu_M120_exact.hdf5")
    plot_specs = [
        ("T", r"$T (\rm K)$", "r_vs_T.png"),
        ("Tdust", r"$T_{\rm dust} (\rm K)$", "r_vs_Tdust.png"),
        ("Trad", r"$T_{\rm rad} (\rm K)$", "r_vs_Trad.png"),
        ("xe", r"$x_e$", "r_vs_xe.png"),
        ("urad_FIR", r"$u_{\rm rad} (\rm eV\,cm^{-3})$", "r_vs_urad.png"),
    ]
    for field, ylabel, fname in plot_specs:
        plot_quantiles_vs_radius(ref["r"], ref[field], label="Benchmark", plotargs={"color": "black"})
        for i, (vlabel, d) in enumerate(_variant_data.items()):
            plot_quantiles_vs_radius(
                d["r"], d[field], label=_short(vlabel),
                plotargs={"marker": _VARIANT_MARKERS[i % len(_VARIANT_MARKERS)],
                          "markersize": max(10 - 2 * i, 4), "markerfacecolor": "none",
                          "linestyle": "none", "alpha": 0.85,
                          "color": _VARIANT_COLORS[i % len(_VARIANT_COLORS)]},
            )
        plt.xlabel(r"$r\,\left(\rm pc\right)$")
        plt.ylabel(ylabel)
        plt.legend(loc="best", fontsize="x-small")
        plt.savefig(f"{test_dir}/{fname}", bbox_inches="tight")
        plt.close()


def compute_test_statistic(f):
    """Returns the test statistic to be compared with the reference solution."""
    d = _load_shu_data(f)
    r_bins = np.logspace(-4, 0, 21)
    stat_names = ["T", "Tdust", "Trad"]
    return {name: binned_statistic(d["r"], d[name], "median", r_bins)[0] for name in stat_names}


_SUBCYCLE_XFAIL = pytest.mark.xfail(
    reason="TRANSPORT_SUBCYCLE does not yet reproduce the un-subcycled result: T "
           "drifts ~30% from baseline. Tracked; the baseline variant is the live gate",
    strict=False,
)


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize(
    "extra_config_flags",
    [
        pytest.param((), id="baseline"),
        pytest.param(("TRANSPORT_SUBCYCLE=10",), id="subcycle_rt", marks=_SUBCYCLE_XFAIL),
        pytest.param(("TRANSPORT_SUBCYCLE=10", "TRANSPORT_SUBCYCLE_COOLING"),
                     id="subcycle_rt_cooling", marks=_SUBCYCLE_XFAIL),
        # Fatal tree audits through accretion with radiation on: every swallow rearranges the
        # particle list and the RT source injection from the sink walks the tree in the same step,
        # before any rebuild. Both integrators: Hermite's corrector is one such walk, and the
        # KDK variant proves the neighbour loops alone need the tree maintained through the swallow.
        pytest.param(("TREE_INTEGRITY_AUDITS",), id="tree_audits"),
        pytest.param(("DISABLE_HERMITE_INTEGRATION", "TREE_INTEGRITY_AUDITS"), id="kdk_tree_audits"),
    ],
)
def test_shu_M120(num_mpi_ranks, num_omp_threads, extra_config_flags):
    test_name = "shu_M120"
    test_dir = "test/shu_M120"
    get_cooling_tables(test_dir)
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads, extra_config_flags, timeout=600)
    final_snap = get_final_snapshot(test_name, extra_config_flags)
    assert_final_time(final_snap, test_name)

    from gizmo.test import variant_output_dir
    test_snap = variant_output_dir(test_name, extra_config_flags) + "/snapshot_005.hdf5"
    test_stats = compute_test_statistic(test_snap)
    benchmark_stats = compute_test_statistic(test_dir + "/shu_M120_exact.hdf5")

    # Accumulate this variant and re-render combined comparison plots
    _variant_data[_variant_label(extra_config_flags)] = _load_shu_data(test_snap)
    _render_combined_shu_plots(test_dir)
    for name in test_stats:
        assert test_stats[name] == pytest.approx(benchmark_stats[name], rel=0.1), \
            f"{name}: max rel diff = {np.max(np.abs(test_stats[name] - benchmark_stats[name]) / np.abs(benchmark_stats[name] + 1e-300)):.3f}"

    with h5py.File(test_snap, "r") as F:
        m_test = F["PartType5/Masses"][:]
    with h5py.File(test_dir + "/shu_M120_exact.hdf5", "r") as F:
        m_bench = F["PartType5/Masses"][:]

    assert len(m_test) == 1
    assert m_test[0] == pytest.approx(m_bench[0], rel=0.1)


def test_shu_M120_snapshot_restart():
    """Snapshot (flag-2) restart of the baseline run's final state with a real MinSizeTimestep.

    Also checks that the restart reads the gas state back (second run below): before any cooling step every cell must
    have the baseline's temperature (1e-2; measured 2e-3, from the H2 the baseline's last cooling step formed after
    its final EOS update). With COOL_MOLECFRAC_NONEQM the snapshot stores the H2 fraction per neutral H, and the
    total fraction, which sets the mean molecular weight in the u -> T conversion, has to be rebuilt from it; left at
    zero until the first cooling step, molecular cells come back up to 1.8x colder.

    Guards the restart ramp. On the first step after a snapshot restart every particle is put on the
    smallest allowed step. Sinks are coupled to their gas neighbours' timebin (dt_ngbs in get_timestep),
    so when that ramp started at 2 ticks every sink requested a sub-floor step for the first few steps
    and STOP_WHEN_BELOW_MINTIMESTEP killed the run; an interim exemption of the abort left the floor
    warnings in the log. The floor here is 40 ticks of the restart's own timebase: above the coupled
    requests, which reach ~33 ticks, and far below any physical step, so a clean log means the ramp
    started at the floor. Runs a short window from the last baseline snapshot; the baseline variant
    must have run first, which is the order in this module."""
    import re
    from glob import glob
    from os import chdir, getcwd, path
    from gizmo.test import run_test
    test_name = "shu_M120"
    snap = get_final_snapshot(test_name, ())
    if snap is None or not path.isfile(snap):
        pytest.skip("baseline snapshot missing; the restart needs it")
    with h5py.File(snap) as f:
        t0 = float(f["Header"].attrs["Time"])
    typedefs = path.join(path.dirname(__file__), "..", "..", "declarations", "typedefs.h")
    timebins = int(re.search(r"#define\s+TIMEBINS\s+(\d+)", open(typedefs).read()).group(1))
    dt_run = 3.0e-5                       # about a quarter of TimeBetSnapshot: dozens of steps up the ladder, a few seconds
    tick = dt_run / 2 ** timebins         # Timebase_interval of the restarted run (TimeBegin is rebased to the snapshot time)
    # With so short a window the tick is tiny, and a physical request above 2^63 ticks = 8 dt_run
    # (the parent run's MaxSizeTimestep allows ~50) overflows the integer conversion in get_timestep,
    # which then misreads it as a sub-floor step and aborts. Cap the step at a quarter of the window.
    cwd = getcwd()
    try:
        chdir("test/shu_M120/")
        run_test(test_name, default_mpi_ranks(), default_omp_threads(), timeout=300, restart_flag=2,
                 param_overrides={"InitCondFile": "output/" + path.basename(snap).replace(".hdf5", ""),
                                  "OutputDir": "output_restart", "TimeMax": t0 + dt_run,
                                  "TimeBetSnapshot": dt_run, "MaxSizeTimestep": dt_run / 4,
                                  "MinSizeTimestep": 40 * tick})
        log = open("test_shu_M120.out").read()
        # The state read back: a flag-2 run writes nothing at the restart time, so a second short restart takes its
        # first step at a floor of 1e6 ticks and writes a snapshot 1e5 ticks in, after the opening kick and the drift
        # (which re-evaluates the EOS) but before any cooling step.
        run_test(test_name, default_mpi_ranks(), default_omp_threads(), timeout=300, restart_flag=2,
                 param_overrides={"InitCondFile": "output/" + path.basename(snap).replace(".hdf5", ""),
                                  "OutputDir": "output_restart_state", "TimeMax": t0 + dt_run,
                                  "TimeOfFirstSnapshot": t0 + 1e5 * tick, "TimeBetSnapshot": dt_run,
                                  "MaxSizeTimestep": dt_run / 4, "MinSizeTimestep": 1e6 * tick})
    finally:
        chdir(cwd)
    n_floor = log.count("wants to be below the limit")
    assert n_floor == 0, f"{n_floor} sub-floor timestep requests after the snapshot restart: the ramp is not starting at MinSizeTimestep"
    assert "ENDRUN" not in log

    def gas(fname):
        with h5py.File(fname) as f:
            g = f["PartType0"]
            ids = [g[k][:].astype(np.int64) for k in ("ParticleIDs", "ParticleChildIDsNumber", "ParticleIDGenerationNumber")]
            order = np.lexsort(ids)
            return (f["Header"].attrs["Time"], np.array(ids)[:, order], g["Temperature"][:][order],
                    g["MolecularMassFraction"][:][order])
    t_r, ids_r, T_r, _ = gas(sorted(glob("test/shu_M120/output_restart_state/snapshot_*.hdf5"))[0])
    _, ids_p, T_p, fH2_p = gas(snap)
    assert t_r < t0 + 1e6 * tick and np.array_equal(ids_r, ids_p), "the restart's first snapshot is not the state read back"
    ratio = T_r / T_p
    worst = np.argmax(np.abs(ratio - 1))
    print(f"restart / baseline temperature: max |ratio - 1| = {abs(ratio[worst] - 1):.3e} (H2 per neutral H there "
          f"{fH2_p[worst]:.3f}); cells with H2 per neutral H > 0.5: median ratio {np.median(ratio[fH2_p > 0.5]):.6f}")
    assert abs(ratio[worst] - 1) < 1e-2, "the restart does not reproduce the gas temperature it read"
