"""General routines to build gizmo for a test and obtain ICs and params files"""

import subprocess
from os import system, environ, path, chdir, cpu_count, remove, makedirs
from urllib.request import urlretrieve, HTTPError
from shutil import move, rmtree
from glob import glob
import pytest
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py


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


def clean_test_outputs(test_name: str, extra_config_flags=()):
    """Remove this variant's output directory, plot PNGs, and log files from a previous test run.
    Other variants' output directories (including the baseline plain "output") are left untouched.
    No-op when GIZMO_TEST_SKIP_BUILD_RUN is set (we're validating externally produced snapshots)."""
    if environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        return
    test_dir = f"test/{test_name}"
    output_dir = variant_output_dir(test_name, extra_config_flags)
    if path.isdir(output_dir):
        rmtree(output_dir)
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
        with open("src/Makefile.systype") as f:
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
    No-op when GIZMO_TEST_SKIP_BUILD_RUN is set (we're validating externally produced snapshots)."""
    if environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        return
    system("rm -f src/GIZMO test/*/GIZMO")
    system(f"cp test/{test_name}/Config.sh src/")
    if num_openmp_threads > 0:
        with open("src/Config.sh", "a") as f:
            f.write(f"\nOPENMP={num_openmp_threads}\n")
    if extra_config_flags:
        with open("src/Config.sh", "a") as f:
            for flag in extra_config_flags:
                f.write(f"\n{flag}\n")
    system("make clean && make -j8")
    if not path.isfile("src/GIZMO"):
        raise FileNotFoundError("Did not successfully build GIZMO")
    move("src/GIZMO", f"test/{test_name}/GIZMO")
    system(f"chmod +x test/{test_name}/GIZMO")


def download_test_files(test_name: str):
    """Downloads the ICs and parameter files for a test of a given name"""

    website_path = "http://www.tapir.caltech.edu/~phopkins/sims/"
    website_path2 = f"https://users.flatironinstitute.org/~mgrudic/gizmo_tests/{test_name}/"

    # The IC to fetch is whatever the parameter file asks for, which is NOT always
    # <test_name>_ics.hdf5: some tests use a different name (shocktube and aneos_shocktube
    # both want shocktube_ics_emass) and some borrow another test's (gravtree uses
    # ../evrard/evrard_ics). Keying the download off the test name alone silently skipped
    # those, and the test then failed for want of a file that was upstream all along.
    # Fall back to the convention when there is no params file or no InitCondFile in it.
    icfile = f"{test_name}_ics.hdf5"
    try:
        ic_param = parse_params(f"{test_name}.params").get("InitCondFile")
        if ic_param:
            icfile = ic_param if path.splitext(ic_param)[1] else ic_param + ".hdf5"
    except OSError:
        pass

    exactfile = f"{test_name}_exact.txt"  # exact solution (might not exist!)
    exactfile2 = f"{test_name}_exact.hdf5"  # exact solution (might not exist!)

    for f in icfile, exactfile, exactfile2:
        if path.isfile(f):
            continue
        # Both stores are flat, so request the basename even when the local destination
        # sits in another test's directory.
        remote = path.basename(f)
        dest_dir = path.dirname(f)
        if dest_dir:
            makedirs(dest_dir, exist_ok=True)
        try:
            urlretrieve(website_path + remote, f)
        except HTTPError as err:
            try:
                urlretrieve(website_path2 + remote, f)
            except HTTPError as err:
                print(f"Could not find {remote} at {website_path} or {website_path2}")

    if not path.isfile(icfile):
        raise (FileNotFoundError(f"Could not find ICs and params for test {test_name}"))


DEFAULT_TEST_TIMEOUT = None  # opt-in: no timeout by default

# Dropped in a run's output directory when the run did not reach TimeMax. finalize_variant_output
# moves that directory as-is, so without a marker a killed run is indistinguishable on disk from a
# complete one, and any cross-variant comparison silently compares different end times.
TRUNCATED_MARKER = "RUN_TRUNCATED"

# GIZMO prints this on the one path that exits the main loop having reached TimeMax (core/run.cc).
_GIZMO_FINISHED = "Final time="


def _resolve_test_timeout(timeout):
    """Resolve effective subprocess timeout. Explicit arg wins; otherwise read
    GIZMO_TEST_TIMEOUT from the environment; otherwise DEFAULT_TEST_TIMEOUT."""
    if timeout is not None:
        return timeout
    env = environ.get("GIZMO_TEST_TIMEOUT")
    if env:
        try:
            val = float(env)
        except ValueError:
            return DEFAULT_TEST_TIMEOUT
        return val if val > 0 else None
    return DEFAULT_TEST_TIMEOUT


def _run_output_dir(test_name: str, paramsfile: str) -> str:
    """A run's OutputDir, relative to the test directory (i.e. to run_test's cwd)."""
    try:
        return parse_params(paramsfile).get("OutputDir", "output")
    except OSError:
        return "output"


def mark_run_truncated(test_name: str, reason: str, paramsfile: str | None = None):
    """Record in the run's output directory that it did not reach TimeMax."""
    d = _run_output_dir(test_name, paramsfile or f"{test_name}.params")
    try:
        makedirs(d, exist_ok=True)
        with open(path.join(d, TRUNCATED_MARKER), "w") as f:
            f.write(reason + "\n")
    except OSError:
        pass  # best-effort: never mask the real failure with a bookkeeping error


def run_truncated_reason(test_name: str, extra_config_flags=()) -> str | None:
    """Why this variant's run did not finish, or None if it did. Call from the repo root."""
    marker = path.join(variant_output_dir(test_name, extra_config_flags), TRUNCATED_MARKER)
    if not path.isfile(marker):
        return None
    with open(marker) as f:
        return f.read().strip() or "run did not reach TimeMax"


def _log_tail(logfile: str, nbytes: int = 4000) -> str:
    try:
        with open(logfile, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - nbytes))
            return f.read().decode(errors="replace")
    except OSError:
        return ""


def _log_reached_timemax(logfile: str, nbytes: int = 262144) -> bool:
    """GIZMO announces reaching TimeMax near the end of stdout; scanning the tail is enough
    and keeps this cheap on the multi-GB logs that deep timestep hierarchies produce."""
    return _GIZMO_FINISHED in _log_tail(logfile, nbytes)


def _check_gizmo_exit(test_name: str, returncode: int, outfile: str, errfile: str, paramsfile: str):
    """Fail loudly if GIZMO did not run to completion.

    Exit status alone is not enough: GIZMO also returns 0 when it leaves the main loop early on
    a 'stop' file or its CPU-time limit, and when it gives up in domain decomposition. In those
    cases the only symptom is a short final snapshot, which a test that does not happen to call
    assert_final_time will report as a pass."""
    if returncode != 0:
        mark_run_truncated(test_name, f"GIZMO exited with status {returncode}", paramsfile)
        raise RuntimeError(
            f"GIZMO exited with status {returncode} for test '{test_name}'.\n"
            f"--- tail of {errfile} ---\n{_log_tail(errfile)}\n"
            f"--- tail of {outfile} ---\n{_log_tail(outfile)}"
        )
    if not _log_reached_timemax(outfile):
        mark_run_truncated(test_name, "GIZMO exited 0 without reaching TimeMax", paramsfile)
        raise RuntimeError(
            f"GIZMO exited 0 for test '{test_name}' but never reported reaching TimeMax, so it "
            "left the main loop early -- a 'stop' file in the output directory or the "
            "TimeLimitCPU cutoff both do this and both exit 0.\n"
            f"--- tail of {outfile} ---\n{_log_tail(outfile)}"
        )


def run_test(test_name: str, num_mpi_ranks: int = 1, num_openmp_threads: int = 0,
             timeout: float | None = None, param_overrides: dict | None = None,
             allow_nonzero_exit: bool = False):
    """Runs the test. If num_openmp_threads > 0, sets OMP_NUM_THREADS for the run.
    If the GIZMO subprocess exceeds the timeout, it is killed and the test is skipped
    via pytest.skip. Timeout defaults to GIZMO_TEST_TIMEOUT env var or DEFAULT_TEST_TIMEOUT.
    param_overrides replaces parameter values for this run only, via a sibling params file.
    Raises if GIZMO exits nonzero or stops before TimeMax, unless allow_nonzero_exit is set --
    for the rare test whose subject IS a failed run (read_ic_binary's sensitivity arm asserts on
    the end-of-file signature), where the exit check would otherwise fire before the test's own
    assertions can run.
    No-op when GIZMO_TEST_SKIP_BUILD_RUN is set (we're validating externally produced snapshots)."""
    if environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        return
    if num_openmp_threads > 0:
        environ["OMP_NUM_THREADS"] = str(num_openmp_threads)
    # Pin BLAS to single-threaded so transitive uses (e.g. via Hypre's BoomerAMG
    # in MHD_MODIFIED_GRADIENT) don't introduce nondeterministic/non-reproducible
    # results that get amplified by the divergence-cleaning feedback loop.
    environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    environ.setdefault("MKL_NUM_THREADS", "1")
    # Silence Kokkos "OMP_PROC_BIND not set" warnings.  `false` is the documented
    # unit-testing value; for production runs on a managed cluster the launcher
    # (e.g. ibrun on TACC) pins ranks to cpusets, so binding is already handled.
    environ.setdefault("OMP_PROC_BIND", "false")
    environ.setdefault("OMP_PLACES", "threads")
    paramsfile = f"{test_name}.params"
    if param_overrides:
        paramsfile = write_params_with_overrides(paramsfile, param_overrides)
    # Deliberately mpirun on every path, including under Slurm. The starforge_dev harness
    # prefers `srun` when SLURM_JOB_ID is set; that branch is not ported because every kokkos
    # run we have validated (build_kokkos_rusty.sh + run_merge_tests_genoa.sh, Genoa nodes)
    # went through mpirun, and switching launcher would be an untested change to the one
    # configuration the suite is currently green on.
    cmd = ["mpirun", "-np", str(num_mpi_ranks), "--use-hwthread-cpus"]
    if num_openmp_threads > 0:
        cmd += ["--bind-to", "none"]
    cmd += ["./GIZMO", paramsfile, "0"]

    effective_timeout = _resolve_test_timeout(timeout)
    outfile, errfile = f"test_{test_name}.out", f"test_{test_name}.err"
    with open(outfile, "w") as out, open(errfile, "w") as err:
        try:
            proc = subprocess.run(cmd, stdout=out, stderr=err, timeout=effective_timeout, check=False)
        except subprocess.TimeoutExpired:
            reason = f"{test_name} exceeded {effective_timeout}s timeout; GIZMO subprocess killed"
            mark_run_truncated(test_name, reason, paramsfile)
            pytest.skip(reason)
    if not allow_nonzero_exit:
        _check_gizmo_exit(test_name, proc.returncode, outfile, errfile, paramsfile)


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
        # Resolve the TREECOOL source against the repo root, not the cwd: callers
        # that chdir into the test dir before calling (e.g. isodisk) would otherwise
        # look for data/cooling/TREECOOL under the test dir. __file__ is
        # test/harness/gizmo/test.py, so the repo root is four dirnames up.
        repo_root = path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
        treecool_src = path.join(repo_root, "data", "cooling", "TREECOOL")
        system(f"cp {treecool_src} {test_directory}")


def write_params_with_overrides(paramsfile: str, overrides: dict) -> str:
    """Write a sibling parameter file with overrides applied, and return its name.

    Lets one variant of a test run with e.g. a shorter TimeMax without a duplicate params
    file drifting out of sync with the original."""
    out = paramsfile.replace(".params", "_override.params")
    remaining = dict(overrides)
    lines = []
    with open(paramsfile) as f:
        for line in f:
            key = line.split("%")[0].split()
            if key and key[0] in remaining:
                lines.append(f"{key[0]}    {remaining.pop(key[0])}\n")
            else:
                lines.append(line)
    lines += [f"{k}    {v}\n" for k, v in remaining.items()]
    with open(out, "w") as f:
        f.write(f"% generated from {paramsfile}; overrides: {overrides}\n")
        f.writelines(lines)
    return out


_BASELINE_STASH = "__output_baseline_stash__"


def _test_dir(test_name: str, caller: str):
    """Absolute path to test/<name>, resolved against the repo root rather than the caller's
    working directory.

    The stash/variant moves below must not depend on cwd: ~25 tests chdir into their own
    directory and call finalize_variant_output() from a finally block, and any raise inside
    that block skips the chdir back. run_test() raises exactly that way when GIZMO exceeds
    GIZMO_TEST_TIMEOUT, so a timed-out variant reached the finally from the test directory,
    where every relative path missed. That turned a clean skip into a RuntimeError and, worse,
    would have left the baseline stashed with the variant's output standing in for it.

    __file__ is test/harness/gizmo/test.py, so the repo root is four dirnames up (same
    derivation as get_cooling_tables)."""
    repo_root = path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    d = path.join(repo_root, "test", test_name)
    if not path.isdir(d):
        raise RuntimeError(f"{caller}('{test_name}'): no such test directory {d}")
    return d


def stash_baseline_output(test_name: str, extra_config_flags=()):
    """If running a non-baseline variant, move an existing output/ aside so the variant
    run doesn't clobber it. Returns True if a stash was made."""
    if not variant_suffix(extra_config_flags):
        return False
    tdir = _test_dir(test_name, "stash_baseline_output")
    plain = path.join(tdir, "output")
    stash = path.join(tdir, _BASELINE_STASH)
    if path.isdir(plain):
        if path.isdir(stash):
            rmtree(stash)
        move(plain, stash)
        return True
    return False


def finalize_variant_output(test_name: str, extra_config_flags=()):
    """After a non-baseline run, rename output/ → variant dir, then restore the
    baseline stash (if any). Idempotent and safe to call in a finally block."""
    if not variant_suffix(extra_config_flags):
        return
    tdir = _test_dir(test_name, "finalize_variant_output")
    plain = path.join(tdir, "output")
    stash = path.join(tdir, _BASELINE_STASH)
    dst = path.join(tdir, "output" + variant_suffix(extra_config_flags))
    if path.isdir(plain):
        if path.isdir(dst):
            rmtree(dst)
        move(plain, dst)
    if path.isdir(stash):
        if path.isdir(plain):
            rmtree(plain)
        move(stash, plain)


def build_and_run_test(test_name: str, num_mpi_ranks: int = 1, num_openmp_threads: int = 0, extra_config_flags: tuple = (),
                       timeout: float | None = None, param_overrides: dict | None = None,
                       allow_nonzero_exit: bool = False):
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
    try:
        chdir(f"test/{test_name}/")
        download_test_files(test_name)
        run_test(test_name, num_mpi_ranks, num_openmp_threads, timeout=timeout,
                 param_overrides=param_overrides, allow_nonzero_exit=allow_nonzero_exit)
        chdir("../../")
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


def assert_final_time(snapshot_file: str, test_name: str, rtol: float = 1e-6, time_max: float | None = None):
    """Assert that the snapshot time matches TimeMax from the test's parameter file, or the
    given time_max when the run overrode it (see run_test's param_overrides)."""
    if time_max is None:
        params_file = f"test/{test_name}/{test_name}.params"
        params = parse_params(params_file)
        time_max = float(params["TimeMax"])
    with h5py.File(snapshot_file, "r") as F:
        time = float(F["Header"].attrs["Time"])
    assert abs(time - time_max) < rtol * abs(
        time_max
    ), f"Snapshot time {time} does not match TimeMax {time_max} (rtol={rtol})"


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
