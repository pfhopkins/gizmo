"""FIRE cosmological galaxy formation test (m11i-like dwarf galaxy)

Runs a short cosmological FIRE simulation of a downsampled dwarf galaxy
(m11i) from z~2.9 for a small time interval. Tests that the full FIRE
physics pipeline (cooling, star formation, stellar feedback, BH physics)
runs without crashing and produces physically reasonable results.

Checks: dark matter mass conservation, and compares gas/star/BH masses
and gas density-temperature distribution against a reference solution.
"""

import re
import pytest
import numpy as np
from os import path, system, chdir, environ, getcwd
from os.path import isfile
from shutil import copy2
from urllib.request import urlretrieve, HTTPError
from matplotlib import pyplot as plt
import h5py
import shlex
from gizmo.test import (
    run_test,
    mpi_launch_command,
    build_gizmo_for_test,
    clean_test_outputs,
    get_cooling_tables,
    default_mpi_ranks,
    default_omp_threads,
    get_final_snapshot,
    stash_baseline_output,
    finalize_variant_output,
)

WEBSITE = "http://www.tapir.caltech.edu/~phopkins/sims/"
REFERENCE_FILE = "test/fire/fire_exact.hdf5"


def get_particle_masses(snapshot):
    """Return dict of total mass per particle type."""
    masses = {}
    with h5py.File(snapshot, "r") as F:
        for grp in F.keys():
            if grp.startswith("PartType"):
                pt = int(grp.replace("PartType", ""))
                masses[pt] = float(F[grp]["Masses"][:].sum())
    return masses


def get_gas_density_temperature(snapshot):
    """Return gas density and temperature arrays."""
    with h5py.File(snapshot, "r") as F:
        rho = F["PartType0/Density"][:]
        u = F["PartType0/InternalEnergy"][:]
        xe = F["PartType0/ElectronAbundance"][:]
    X_H = 0.76
    gamma = 5.0 / 3.0
    mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * xe)
    m_p = 1.6726e-24  # g
    k_B = 1.3807e-16  # erg/K
    u_cgs = u * (1.0e5) ** 2  # convert from (km/s)^2 to (cm/s)^2
    T = (gamma - 1.0) * u_cgs * mu * m_p / k_B
    return rho, T


def plot_density_temperature(snapshot, output_dir, label=""):
    """Scatter plot of gas log(density) vs log(temperature)."""
    rho, T = get_gas_density_temperature(snapshot)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(np.log10(rho + 1e-30), np.log10(T + 1e-1),
               s=0.1, alpha=0.3, color="black", rasterized=True)
    ax.set_xlabel(r"$\log_{10}(\rho)$ [code units]")
    ax.set_ylabel(r"$\log_{10}(T)$ [K]")
    ax.set_title(f"Gas Density-Temperature {label}")
    fig.savefig(path.join(output_dir, "density_temperature.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_density_histogram(snapshot, output_dir, label=""):
    """Histogram of gas log(density)."""
    with h5py.File(snapshot, "r") as F:
        rho = F["PartType0/Density"][:]
        masses = F["PartType0/Masses"][:]
    log_rho = np.log10(rho + 1e-30)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(log_rho, bins=100, weights=masses, color="steelblue",
            edgecolor="none", alpha=0.8)
    ax.set_xlabel(r"$\log_{10}(\rho)$ [code units]")
    ax.set_ylabel("Mass-weighted count")
    ax.set_title(f"Gas Density Histogram {label}")
    ax.set_yscale("log")
    fig.savefig(path.join(output_dir, "density_histogram.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _download_if_missing(filename):
    """Download a file from the tapir server if not already present."""
    if not isfile(filename):
        try:
            urlretrieve(WEBSITE + filename, filename)
        except HTTPError:
            print(f"Could not download {filename} from {WEBSITE}")


def run_fire_test(test_name, num_mpi_ranks, num_omp_threads, extra_config_flags=()):
    """Build and run the FIRE test with restart flag 2."""
    build_gizmo_for_test(test_name, num_omp_threads, extra_config_flags)
    chdir(f"test/{test_name}/")
    # Download ICs and reference if not present
    _download_if_missing("fire_ics.hdf5")
    _download_if_missing("fire_exact.hdf5")
    # Download cooling tables; get_cooling_tables tries cp cooling/TREECOOL
    # from cwd which won't work here, so copy TREECOOL manually
    if not path.isdir("spcool_tables"):
        get_cooling_tables(".")
    if not isfile("TREECOOL"):
        copy2("../../cooling/TREECOOL", "TREECOOL")
    if num_omp_threads > 0:
        environ["OMP_NUM_THREADS"] = str(num_omp_threads)
    paramsfile = f"{test_name}.params"
    launch = shlex.join(mpi_launch_command(num_mpi_ranks, num_omp_threads) + ["./GIZMO", paramsfile, "2"])
    system(f"{launch} 1>test_{test_name}.out 2>test_{test_name}.err")
    chdir("../../")


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(2),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize(
    "extra_config_flags",
    [(), ("TIDAL_TIMESTEP_CRITERION", "ADAPTIVE_TREEFORCE_UPDATE=0.06")],
    ids=["baseline", "tidal_adaptive"],
)
def test_fire(num_mpi_ranks, num_omp_threads, extra_config_flags):
    test_name = "fire"
    clean_test_outputs(test_name, extra_config_flags)

    # Build and run, stashing any baseline output/ aside so non-default variants don't clobber it
    stash_baseline_output(test_name, extra_config_flags)
    try:
        run_fire_test(test_name, num_mpi_ranks, num_omp_threads, extra_config_flags)
    finally:
        finalize_variant_output(test_name, extra_config_flags)

    # Check simulation produced output
    final_snap = get_final_snapshot(test_name, extra_config_flags)

    # Get initial and final masses
    initial_snap = f"test/{test_name}/fire_ics.hdf5"
    m_init = get_particle_masses(initial_snap)
    m_final = get_particle_masses(final_snap)

    print("Particle type masses:")
    type_names = {0: "Gas", 1: "DM", 2: "DM-LowRes", 4: "Stars", 5: "BH"}
    for pt in sorted(set(list(m_init.keys()) + list(m_final.keys()))):
        mi = m_init.get(pt, 0)
        mf = m_final.get(pt, 0)
        name = type_names.get(pt, f"Type{pt}")
        print(f"  {name} (Type {pt}): initial={mi:.6g}, final={mf:.6g}, "
              f"delta={mf - mi:.6g}")

    # Dark matter mass must be conserved; the tolerance only has to absorb float32 summation over
    # ~1e6 masses, whose order follows the domain decomposition. On this IC's Type-2 population
    # (324853 particles, total 1.718e6, essentially all at the uniform 8.204 mass):
    #
    #   float32 ULP on the total        7.3e-08   the noise floor -- unavoidable
    #   losing one 8.204-mass particle  4.8e-06   what the gate must catch
    #
    # 1e-7 is ~1.4x above the floor and ~48x below a single lost particle.
    MASS_CONSERVATION_TOL = 1e-7
    assert m_init[1] == pytest.approx(m_final[1], rel=MASS_CONSERVATION_TOL), \
        f"DM mass changed: {m_init[1]} -> {m_final[1]}"
    if 2 in m_init and 2 in m_final:
        assert m_init[2] == pytest.approx(m_final[2], rel=MASS_CONSERVATION_TOL), \
            f"Low-res DM mass changed: {m_init[2]} -> {m_final[2]}"

    # Generate diagnostic plots
    test_dir = f"test/{test_name}"
    plot_density_temperature(final_snap, test_dir, label="(final)")
    plot_density_histogram(final_snap, test_dir, label="(final)")

    # Compare against reference solution
    if path.isfile(REFERENCE_FILE):
        m_ref = get_particle_masses(REFERENCE_FILE)
        for pt in (0, 4, 5):
            if pt in m_ref and pt in m_final:
                name = type_names.get(pt, f"Type{pt}")
                assert m_final[pt] == pytest.approx(m_ref[pt], rel=0.1), \
                    f"{name} mass differs from reference: " \
                    f"{m_final[pt]:.6g} vs {m_ref[pt]:.6g}"


TRACER_FLAGS = ("GALSF_FB_FIRE_AGE_TRACERS=16", "STARFORGE_FEEDBACK_TRACERS=2")


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(2),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_fire_metal_tracers(num_mpi_ranks, num_omp_threads):
    """Age tracers and STARFORGE feedback tracers in one build: each block is addressed at its own offset in
    Metallicity[]. The feedback tracers follow the age tracers, so a coupling that takes the last NUM_AGE_TRACERS
    slots as the age tracers misses the first bins; and the yield writers fill one tracer per channel (jets, winds,
    SNe), so three fields must exist whatever value the flag is given."""
    test_name = "fire"
    clean_test_outputs(test_name, TRACER_FLAGS)
    stash_baseline_output(test_name, TRACER_FLAGS)
    try:
        run_fire_test(test_name, num_mpi_ranks, num_omp_threads, TRACER_FLAGS)
    finally:
        finalize_variant_output(test_name, TRACER_FLAGS)

    def age_tracer_content(snapshot):
        """the gas content of each age-tracer bin, sum of mass x tracer, located by the snapshot's own keys"""
        with h5py.File(snapshot, "r") as F:
            zkey = F["Header"].attrs["Metals_Atomic_Number_Or_Key"]
            z = F["PartType0/Metallicity"][:].astype(np.float64)
            m = F["PartType0/Masses"][:].astype(np.float64)
        return zkey, m, z, (m[:, None] * z[:, zkey == -2]).sum(axis=0)

    _, _, _, initial = age_tracer_content(f"test/{test_name}/fire_ics.hdf5")  # the IC carries the 16 bins already
    zkey, m, z, final = age_tracer_content(get_final_snapshot(test_name, TRACER_FLAGS))
    gain = (final - initial) / np.where(initial > 0, initial, np.nan)  # the oldest bins start empty
    fb = np.flatnonzero(zkey == -3)
    print(f"Metallicity keys: {(zkey == -2).sum()} age tracers, {len(fb)} feedback tracers; age-tracer gas content "
          "gained over the run, per bin, relative to the IC:", np.array2string(gain, precision=3))
    assert (zkey == -2).sum() == 16 and len(fb) == 3, f"{(zkey == -2).sum()} age-tracer and {len(fb)} feedback-tracer fields"
    sne = z[:, fb[2]]
    print(f"SNe tracer: max {sne.max():.4g}, mass-weighted mean {np.sum(m * sne) / m.sum():.4g}")
    # stars 1-6 Myr old (bins 0-2) are present and deposit throughout the run; without deposits a bin only loses to
    # star formation
    assert np.all(gain[:3] > 0), "the first age-tracer bins received no deposits"
    assert 0 < sne.max() <= 1 + 1e-6, f"SNe tracer max {sne.max():.4g}: not a mass fraction"


DUST_FLAGS = ("GALSF_ISMDUSTCHEM_MODEL=(1+2+8)", "IO_DUST_NOT_IN_ICFILE")  # dust initialized from the parameters at flag 2
# a few steps (these are startup checks); TimeMax sets the timebase, so the step cap must shrink with it
DUST_STARTUP = {"Final_Simulation_Time": "0.25645", "Maximum_Timestep_Allowed": "1e-5"}


def run_fire_dust_startup(num_mpi_ranks, num_omp_threads, overrides):
    """Build the dust-chemistry variant and run it a few steps through run_test, which raises unless GIZMO reaches
    TimeMax; returns its stdout"""
    test_name = "fire"
    clean_test_outputs(test_name, DUST_FLAGS)
    build_gizmo_for_test(test_name, num_omp_threads, DUST_FLAGS)
    stash_baseline_output(test_name, DUST_FLAGS)
    cwd = getcwd()
    try:
        chdir(f"test/{test_name}")
        try:
            run_test(test_name, num_mpi_ranks, num_omp_threads, param_overrides=overrides, restart_flag=2)
        finally:
            out = open(f"test_{test_name}.out", errors="replace").read()
    finally:
        chdir(cwd)
        finalize_variant_output(test_name, DUST_FLAGS)
    return out


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(2),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_fire_ismdustchem_parameter_defaults(num_mpi_ranks, num_omp_threads):
    """Every ISM dust-chemistry scaling parameter has a default: fire.params sets none of them, and the run must start"""
    out = run_fire_dust_startup(num_mpi_ranks, num_omp_threads, DUST_STARTUP)
    assert "Tag AccretionTcutoffScaling" in out and "defaulting to one" in out


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(2),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_fire_ismdustchem_depletion_needs_ratio(num_mpi_ranks, num_omp_threads):
    """An initial dust depletion with no silicon-to-carbon ratio is refused at startup: the ratio (default 0) divides the
    initial carbonaceous depletion, which otherwise locks all the carbon into dust"""
    with pytest.raises(RuntimeError):
        run_fire_dust_startup(num_mpi_ranks, num_omp_threads, {**DUST_STARTUP, "Initial_ISMDustChem_Depletion": "0.5"})
    out = open("test/fire/test_fire.out", errors="replace").read()
    assert "Initial_ISMDustChem_Depletion > 0 requires Initial_ISMDustChem_SiltoCarbRatio > 0" in out


_TIMEBIN_ROW = re.compile(r"^ [X ]  bin=\s*(\d+)\s+\d+\s+\d+\s+\S+\s+\d+ . .\s+(\S+)\s+(\S+)%$")


def timebin_tables(logfile):
    """(bin, avg-time, cpu-frac %) rows of every 'Occupied timebins' report in a GIZMO stdout log"""
    tables, rows = [], None
    for line in open(logfile, errors="replace"):
        if line.startswith("Occupied timebins"):
            rows = []
        elif rows is not None:
            m = _TIMEBIN_ROW.match(line.rstrip("\n"))
            if m:
                rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
            else:
                tables.append(rows)
                rows = None
    return tables


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(2),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_fire_timebin_report(num_mpi_ranks, num_omp_threads):
    """The timebin report's cpu-frac column is each occupied bin's share of the CPU time, its average cost weighted by how
    often it is active relative to the highest occupied bin (2^(H-bin), halved below the top), for every occupied bin
    including those below an empty one. FIRE's timebin hierarchy has such gaps. Checked against the printed avg-time,
    whose 2-decimal rounding sets the bounds on each share."""
    test_name = "fire"
    clean_test_outputs(test_name)
    stash_baseline_output(test_name)
    try:
        run_fire_test(test_name, num_mpi_ranks, num_omp_threads)
    finally:
        finalize_variant_output(test_name)
    tables = [t for t in timebin_tables(f"test/{test_name}/test_{test_name}.out") if t]
    n_gap = n_bad = 0
    for t in tables:
        bins = np.array([r[0] for r in t])
        avg, frac = np.array([r[1] for r in t]), np.array([r[2] for r in t])
        n_gap += np.any(np.diff(bins) < -1)
        w = 2.0 ** (bins.max() - bins)
        w = np.where(w > 1, w / 2, w)
        lo_t, hi_t = w * np.maximum(avg - 0.005, 0), w * (avg + 0.005)
        lo = 100 * lo_t / np.maximum(lo_t + (hi_t.sum() - hi_t), 1e-300)
        hi = 100 * np.minimum(hi_t / np.maximum(hi_t + (lo_t.sum() - lo_t), 1e-300), 1)
        if np.any((frac < lo - 0.05) | (frac > hi + 0.05)):
            n_bad += 1
    print(f"{len(tables)} timebin reports, {n_gap} with an empty bin below the highest occupied one; "
          f"{n_bad} with a cpu-frac outside its share")
    assert n_gap > 0, "no timebin report has a gap: the check is not exercised"
    assert n_bad == 0, f"{n_bad} of {len(tables)} timebin reports have a cpu-frac inconsistent with their avg-time column"
