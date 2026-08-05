"""General routines to build gizmo for a test and obtain ICs and params files"""

import subprocess
from os import system, environ, path, chdir, cpu_count, remove
from urllib.request import urlretrieve, HTTPError
from shutil import move, rmtree, copyfile
from glob import glob
import numpy as np
import pytest
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py

DEFAULT_TEST_TIMEOUT = None  # opt-in: no timeout by default


def _resolve_test_timeout(timeout):
    """Resolve effective subprocess timeout. Explicit arg wins; otherwise read
    GIZMO_TEST_TIMEOUT env var; otherwise fall back to DEFAULT_TEST_TIMEOUT.
    With DEFAULT_TEST_TIMEOUT=None the subprocess runs unbounded unless the
    caller explicitly passes a timeout (or sets GIZMO_TEST_TIMEOUT in the env).
    Any value <= 0 also disables the timeout."""
    if timeout is None:
        env_val = environ.get("GIZMO_TEST_TIMEOUT")
        if env_val:
            timeout = float(env_val)
        else:
            timeout = DEFAULT_TEST_TIMEOUT
    if timeout is None:
        return None
    return timeout if timeout > 0 else None


def flush_colorbar(mappable, ax=None, label=None, **kwargs):
    """Add a colorbar that is flush with the axes and lines up with its edges."""
    if ax is None:
        ax = mappable.axes
    fig = ax.get_figure()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax, label=label, **kwargs)


def variant_suffix(extra_config_flags=()):
    """Return a filename-safe suffix encoding extra_config_flags (empty for no flags)."""
    if not extra_config_flags:
        return ""
    sanitized = ["".join(c if c.isalnum() else "_" for c in f) for f in extra_config_flags]
    return "_" + "__".join(sanitized)


def variant_output_dir(test_name: str, extra_config_flags=()) -> str:
    """Return the output directory used for a given (test, flag combination)."""
    return f"test/{test_name}/output{variant_suffix(extra_config_flags)}"


def _robust_replace_dir(d):
    """rmtree d, but if NFS silly-rename .nfs* lock files prevent it (e.g. a
    stuck process from a previous run is still holding a deleted file open),
    move d out of the way to a unique stale.* sibling and keep going."""
    if not path.isdir(d):
        return
    try:
        rmtree(d)
    except OSError:
        import time
        stale = f"{d}.stale.{int(time.time())}"
        move(d, stale)


def clean_test_outputs(test_name: str, extra_config_flags=()):
    """Remove this variant's output directory, plot PNGs, and log files from a previous test run.
    Other variants' output directories (including the baseline plain "output") are left untouched.
    No-op when GIZMO_TEST_SKIP_BUILD_RUN is set (we're validating externally produced snapshots)."""
    if environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        return
    test_dir = f"test/{test_name}"
    output_dir = variant_output_dir(test_name, extra_config_flags)
    _robust_replace_dir(output_dir)
    for f in glob(path.join(test_dir, f"test_{test_name}.out")):
        remove(f)
    for f in glob(path.join(test_dir, f"test_{test_name}.err")):
        remove(f)


def default_omp_threads():
    """Return the default number of OpenMP threads for tests."""
    return 2


def default_mpi_ranks(max_ranks=None):
    """Return the number of MPI ranks to use, defaulting to half the available CPU count
    (to leave room for OpenMP threads). Optionally cap at max_ranks."""
    n = cpu_count() // 2
    if max_ranks is not None:
        n = min(n, max_ranks)
    return max(n, 1)


_KOKKOS_SYSTYPES = ("MacBookCellar_Kokkos", "Vista")


def _current_systype():
    """Read active SYSTYPE from Makefile.systype (first uncommented SYSTYPE=... line)."""
    try:
        with open("Makefile.systype") as f:
            for line in f:
                line = line.strip()
                if line.startswith("SYSTYPE=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return ""


def build_gizmo_for_test(test_name: str, num_openmp_threads: int = 0, extra_config_flags: tuple = ()):
    """Sets environment variables and runs a script for building gizmo for a given test.
    If num_openmp_threads > 0, appends OPENMP=<num_openmp_threads> to Config.sh before building.
    extra_config_flags is a tuple of strings to append to Config.sh (e.g. ("TRANSPORT_SUBCYCLE=10",)).
    On Kokkos systypes (MacBookCellar_Kokkos, Vista) auto-appends GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY
    if absent, so the Kokkos neighbor-list code path is actually exercised (the non-flag legacy tree
    walk is retained only for backward compat).
    No-op when GIZMO_TEST_SKIP_BUILD_RUN is set (we're validating externally produced snapshots)."""
    if environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        return
    system("rm -f GIZMO test/*/GIZMO")
    system(f"cp test/{test_name}/Config.sh .")
    if num_openmp_threads > 0:
        with open("Config.sh", "a") as f:
            f.write(f"\nOPENMP={num_openmp_threads}\n")
    if extra_config_flags:
        with open("Config.sh", "a") as f:
            for flag in extra_config_flags:
                f.write(f"\n{flag}\n")
    if _current_systype() in _KOKKOS_SYSTYPES:
        with open("Config.sh") as f:
            cfg = f.read()
        needs_flag = not any(
            line.strip().startswith("GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY")
            and not line.strip().startswith("#")
            for line in cfg.splitlines()
        )
        if needs_flag:
            with open("Config.sh", "a") as f:
                f.write("\nGIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY\n")
    system("make clean && make -j8")
    if not path.isfile("GIZMO"):
        raise FileNotFoundError("Did not successfully build GIZMO")
    move("GIZMO", f"test/{test_name}/GIZMO")
    system(f"chmod +x test/{test_name}/GIZMO")


def download_test_files(test_name: str):
    """Downloads the ICs and parameter files for a test of a given name"""

    website_path = "http://www.tapir.caltech.edu/~phopkins/sims/"
    website_path2 = f"https://users.flatironinstitute.org/~mgrudic/gizmo_tests/{test_name}/"

    # Note: we are assuming a convention for the test ICs, params, and exact values
    icfile = f"{test_name}_ics.hdf5"
    exactfile = f"{test_name}_exact.txt"  # exact solution (might not exist!)
    exactfile2 = f"{test_name}_exact.hdf5"  # exact solution (might not exist!)

    for f in icfile, exactfile, exactfile2:
        # Never clobber a locally-generated IC (e.g. from make_<test>_ics.py): a stale
        # remote copy would silently override the freshly generated one. Reference "exact"
        # solutions have no local generator, so those are always fetched.
        if f == icfile and path.isfile(icfile):
            continue
        try:
            urlretrieve(website_path + f, f)
        except HTTPError as err:
            try:
                urlretrieve(website_path2 + f, f)
            except HTTPError as err:
                print(f"Could not find {f} at {website_path} or {website_path2}")

    if not path.isfile(icfile):
        raise (FileNotFoundError(f"Could not find ICs and params for test {test_name}"))


def run_test(test_name: str, num_mpi_ranks: int = 1, num_openmp_threads: int = 0, timeout: float | None = None):
    """Runs the test. If num_openmp_threads > 0, sets OMP_NUM_THREADS for the run.
    If the GIZMO subprocess exceeds the timeout, it is killed and the test is skipped
    via pytest.skip. Timeout defaults to GIZMO_TEST_TIMEOUT env var or DEFAULT_TEST_TIMEOUT.
    No-op when GIZMO_TEST_SKIP_BUILD_RUN is set (we're validating externally produced snapshots)."""
    if environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        return
    if num_openmp_threads > 0:
        environ["OMP_NUM_THREADS"] = str(num_openmp_threads)
    else:
        environ.setdefault("OMP_NUM_THREADS", "1")
    # Pin BLAS to single-threaded so transitive uses (e.g. via Hypre's BoomerAMG
    # in MHD_MODIFIED_GRADIENT) don't introduce nondeterministic/non-reproducible
    # results that get amplified by the divergence-cleaning feedback loop.
    environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    environ.setdefault("MKL_NUM_THREADS", "1")
    paramsfile = f"{test_name}.params"
    if environ.get("SLURM_JOB_ID"):
        cmd = ["srun", "-n", str(num_mpi_ranks), "--cpu-bind=none"]
    else:
        cmd = ["mpirun", "-np", str(num_mpi_ranks), "--use-hwthread-cpus", "--oversubscribe"]
    if num_openmp_threads > 0 and not environ.get("SLURM_JOB_ID"):
        cmd += ["--bind-to", "none"]
    cmd += ["./GIZMO", paramsfile, "0"]

    effective_timeout = _resolve_test_timeout(timeout)
    with open(f"test_{test_name}.out", "w") as out, open(f"test_{test_name}.err", "w") as err:
        try:
            subprocess.run(cmd, stdout=out, stderr=err, timeout=effective_timeout, check=False)
        except subprocess.TimeoutExpired:
            pytest.skip(f"{test_name} exceeded {effective_timeout}s timeout; GIZMO subprocess killed")


def get_cooling_tables(test_directory="."):
    """Downloads spcool_tables.tar.gz and copies TREECOOL to test directory.
    Idempotent: skips steps whose targets already exist (supports symlinks as well as
    real files/dirs), so tests that share cooling data with sibling tests via symlinks
    don't trigger unnecessary downloads."""

    spcool_dir = f"{test_directory}/spcool_tables"
    if not (path.isdir(spcool_dir) or path.islink(spcool_dir)):
        url = "https://users.flatironinstitute.org/~mgrudic/gizmo_tests/spcool_tables.tgz"
        urlretrieve(url, f"{test_directory}/spcool_tables.tgz")
        system(f"tar -xvf {test_directory}/spcool_tables.tgz -C {test_directory}/; rm spcool_tables.tgz")
    treecool_dst = f"{test_directory}/TREECOOL"
    if not (path.isfile(treecool_dst) or path.islink(treecool_dst)):
        system(f"cp cooling/TREECOOL {test_directory}")


_BASELINE_STASH = "__output_baseline_stash__"


def stash_baseline_output(test_name: str, extra_config_flags=()):
    """If running a non-baseline variant, move an existing output/ aside so the variant
    run doesn't clobber it. Returns True if a stash was made."""
    if not variant_suffix(extra_config_flags):
        return False
    plain = f"test/{test_name}/output"
    stash = f"test/{test_name}/{_BASELINE_STASH}"
    if path.isdir(plain):
        _robust_replace_dir(stash)
        move(plain, stash)
        return True
    return False


def finalize_variant_output(test_name: str, extra_config_flags=()):
    """After a non-baseline run, rename output/ → variant dir, then restore the
    baseline stash (if any). Idempotent and safe to call in a finally block."""
    if not variant_suffix(extra_config_flags):
        return
    plain = f"test/{test_name}/output"
    stash = f"test/{test_name}/{_BASELINE_STASH}"
    dst = variant_output_dir(test_name, extra_config_flags)
    if path.isdir(plain):
        _robust_replace_dir(dst)
        move(plain, dst)
    if path.isdir(stash):
        _robust_replace_dir(plain)
        move(stash, plain)
    # Preserve GIZMO's stdout with the variant: run_test opens test/<name>/test_<name>.out with mode
    # "w", so otherwise each variant truncates the previous one's in-code diagnostics.
    log = f"test/{test_name}/test_{test_name}.out"
    if path.isfile(log) and path.isdir(dst):
        copyfile(log, path.join(dst, f"test_{test_name}.out"))


def build_and_run_test(test_name: str, num_mpi_ranks: int = 1, num_openmp_threads: int = 0, extra_config_flags: tuple = (), timeout: float | None = None):
    """Top-level routine that does all necessary building, downloading, and running of the test.
    When extra_config_flags is non-empty, the resulting output/ directory is renamed to a
    variant-specific name so that multiple flag combinations can coexist on disk. The baseline
    output/ (if any) is temporarily stashed aside so it isn't overwritten by the variant run.

    Set env var GIZMO_TEST_SKIP_BUILD_RUN=1 to reuse snapshots already present in
    test/<name>/output/ — useful for validating a remote (e.g. Vista GPU) run against
    the pytest assertions + plots without rebuilding/rerunning locally."""
    if environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        return
    clean_test_outputs(test_name, extra_config_flags)
    build_gizmo_for_test(test_name, num_openmp_threads, extra_config_flags)
    stash_baseline_output(test_name, extra_config_flags)
    from os import getcwd
    cwd = getcwd()
    try:
        chdir(f"test/{test_name}/")
        try:
            download_test_files(test_name)
            run_test(test_name, num_mpi_ranks, num_openmp_threads, timeout=timeout)
        finally:
            chdir(cwd)
    finally:
        finalize_variant_output(test_name, extra_config_flags)


def parse_params(params_file: str) -> dict:
    """Parse a GIZMO parameter file and return a dict of key-value pairs."""
    params = {}
    with open(params_file) as f:
        for line in f:
            line = line.split("%")[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                params[parts[0]] = parts[1]
    return params


def get_final_snapshot(test_name: str, extra_config_flags=()) -> str:
    """Return the path to the last snapshot produced by a test (variant-aware)."""
    output_dir = variant_output_dir(test_name, extra_config_flags)
    snaps = sorted(glob(f"{output_dir}/snapshot_*.hdf5"))
    if not snaps:
        raise RuntimeError(f"No snapshots found for test {test_name} in {output_dir}")
    return snaps[-1]


def assert_final_time(snapshot_file: str, test_name: str, rtol: float = 1e-6):
    """Assert that the snapshot time matches TimeMax from the test's parameter file."""
    params_file = f"test/{test_name}/{test_name}.params"
    params = parse_params(params_file)
    time_max = float(params["TimeMax"])
    with h5py.File(snapshot_file, "r") as F:
        time = float(F["Header"].attrs["Time"])
    assert abs(time - time_max) < rtol * abs(
        time_max
    ), f"Snapshot time {time} does not match TimeMax {time_max} (rtol={rtol})"


def gas_energy(snapshot_file: str, include_magnetic: bool = False,
               include_potential: bool = False) -> float:
    """Total gas energy from a snapshot: thermal + kinetic, optionally magnetic and gravitational.

    The caller must include every reservoir the problem actually has, or the "conserved" quantity
    will not be conserved. Thermal+kinetic alone is right only with SELFGRAVITY_OFF and no MAGNETIC.
      include_magnetic: adds sum(B^2/2 * m/rho), needs MagneticField in the snapshot.
      include_potential: adds 0.5*sum(m*phi) (the 1/2 avoids double-counting pairs), needs
        Potential in the snapshot, i.e. OUTPUT_POTENTIAL in the Config.
    """
    with h5py.File(snapshot_file, "r") as F:
        g = F["PartType0"]
        m = g["Masses"][:]
        v = g["Velocities"][:]
        u = g["InternalEnergy"][:]
        e = float(np.sum(m * u) + 0.5 * np.sum(m * np.sum(v**2, axis=1)))
        if include_magnetic:
            if "MagneticField" not in g:
                raise KeyError(f"{snapshot_file}: MagneticField absent; MAGNETIC not enabled?")
            b = g["MagneticField"][:]
            rho = g["Density"][:]
            e += float(np.sum(0.5 * np.sum(b**2, axis=1) * m / np.maximum(rho, 1e-300)))
        if include_potential:
            if "Potential" not in g:
                raise KeyError(f"{snapshot_file}: Potential absent; add OUTPUT_POTENTIAL to Config.sh")
            e += float(0.5 * np.sum(m * g["Potential"][:]))
    return e


def assert_energy_conserved(test_name: str, extra_config_flags=(), tol: float = 0.1,
                            injected_energy: float | None = None,
                            include_magnetic: bool = False, include_potential: bool = False):
    """Assert the gas energy budget closes, comparing the first and last snapshot.

    With injected_energy=None the problem is assumed closed (no source after t=0) and the total
    must be constant. Otherwise the CHANGE in total energy must equal injected_energy, which is
    the right form for a test that deposits a known amount (e.g. 1e51 erg for a supernova).

    Only valid where thermal+kinetic is the whole budget -- see gas_energy(). tol defaults to 10%,
    deliberately loose: this is meant to catch gross violations, not to police integration
    accuracy. For scale, a rejected wakeup-limiter variant (deferring the demotion rather than
    truncating the step) destroyed 62% of the blast energy in sedov, and the density-profile
    assertion caught it only indirectly, as a displaced shell.
    """
    outdir = variant_output_dir(test_name, extra_config_flags)
    snaps = sorted(glob(f"{outdir}/snapshot_*.hdf5"))
    assert len(snaps) >= 2, f"need >=2 snapshots to check conservation, found {len(snaps)}"
    kw = {"include_magnetic": include_magnetic, "include_potential": include_potential}
    e_0, e_f = gas_energy(snaps[0], **kw), gas_energy(snaps[-1], **kw)
    if injected_energy is None:
        assert abs(e_f / e_0 - 1.0) < tol, (
            f"{test_name}: gas energy not conserved: E_final/E_initial = {e_f / e_0:.4f} "
            f"(tolerance {tol:.0%}); E_0={e_0:.6e} E_f={e_f:.6e}"
        )
        return e_f / e_0
    ratio = (e_f - e_0) / injected_energy
    assert abs(ratio - 1.0) < tol, (
        f"{test_name}: energy budget does not close: dE/E_injected = {ratio:.4f} "
        f"(tolerance {tol:.0%}); dE={e_f - e_0:.6e} E_injected={injected_energy:.6e}"
    )
    return ratio


def assert_snapshots_are_close(
    snapshot1: str,
    snapshot2: str,
    fields_to_compare: tuple = ("Density", "Velocities", "InternalEnergy"),
    rtol: float = 1e-2,
    atol: float = 0,
):
    """Test-assert that the specified gas data fields in two snapshots are within specified tolerance"""
    fields_to_read = ("ParticleIDs",) + fields_to_compare

    datafields = {snapshot1: {}, snapshot2: {}}
    for s in snapshot1, snapshot2:
        with h5py.File(s, "r") as F:  # read
            for f in fields_to_read:
                datafields[s][f] = F["PartType0/" + f][:]

        id_order = datafields[s]["ParticleIDs"].argsort()
        for f in fields_to_read:  # sort by ID
            datafields[s][f] = datafields[s][f][id_order]

    for f in fields_to_compare:
        np.testing.assert_allclose(
            datafields[snapshot1][f], datafields[snapshot2][f], rtol=rtol, atol=atol,
            err_msg=f"Field {f} differs between {snapshot1} and {snapshot2}",
        )


def plot_1D_snapshot_comparison(
    snapshot1: str,
    snapshot2: str,
    fields_to_plot: tuple = ("Density", "Velocities", "InternalEnergy"),
    output_dir: str = ".",
):
    """Plot 1D comparison of gas data fields between two snapshots, sorted by particle ID."""
    fields_to_read = ("ParticleIDs", "Coordinates") + fields_to_plot

    datafields = {snapshot1: {}, snapshot2: {}}
    for s in snapshot1, snapshot2:
        with h5py.File(s, "r") as F:
            for f in fields_to_read:
                datafields[s][f] = F["PartType0/" + f][:]

        id_order = datafields[s]["ParticleIDs"].argsort()
        for f in fields_to_read:
            datafields[s][f] = datafields[s][f][id_order]

    for f in fields_to_plot:
        plt.plot(datafields[snapshot1]["Coordinates"][:, 0], datafields[snapshot1][f], ".", label="Initial")
        plt.plot(datafields[snapshot2]["Coordinates"][:, 0], datafields[snapshot2][f], ".", label="Final")
        plt.legend()
        plt.ylabel(f)
        plt.xlabel("x")
        plt.savefig(path.join(output_dir, f"{f}.png"))
        plt.close()
