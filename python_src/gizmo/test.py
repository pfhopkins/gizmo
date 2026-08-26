"""General routines to build gizmo for a test and obtain ICs and params files"""

import subprocess
from os import system, environ, path, chdir, cpu_count, remove, getcwd, makedirs
from urllib.request import urlretrieve, HTTPError
import fcntl
from shutil import move, rmtree, copyfile
from glob import glob
import numpy as np
import pytest
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py

DEFAULT_TEST_TIMEOUT = None  # opt-in: no timeout by default

# Dropped in a run's output directory when the run did not reach TimeMax. finalize_variant_output
# moves that directory as-is, so without a marker a killed run is indistinguishable on disk from a
# complete one, and any cross-variant comparison silently compares different end times.
TRUNCATED_MARKER = "RUN_TRUNCATED"

# GIZMO prints this on the one path that exits the main loop having reached TimeMax (core/run.cc).
_GIZMO_FINISHED = "Final time="


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


# Anchored to the repo root via this file's location, NOT relative to cwd. A relative path
# would silently point at a different file if anything ever built from another directory, and
# two jobs locking different files is no lock at all -- a failure that shows up as a corrupted
# build rather than as an error.
_BUILD_LOCK = path.join(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))),
                        ".gizmo_build.lock")
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
    # Serialise the build against other processes sharing this tree. Everything from here to the
    # move is repo-root state -- Config.sh, GIZMO_config.h, every object file, and the GIZMO
    # binary itself -- so two builds at once corrupt each other. The lock covers the BUILD only,
    # so builds serialise (minutes) while runs overlap (tens of minutes).
    #
    # NECESSARY BUT NOT SUFFICIENT for concurrent jobs in one checkout. With this lock in place,
    # a run of 11 concurrent jobs still produced a binary whose translation units disagreed on
    # the layout of the All struct -- mymalloc_init read All.MaxMemSize at an offset holding a
    # double's bit pattern and aborted before any physics -- and the identical test passed the
    # moment it built alone. Whatever leaks past the lock was not identified. For parallel jobs,
    # give each its own checkout (git worktree, ~10 MB); keep this lock as the last line of
    # defence, not the guarantee.
    #
    # lockf, NOT flock. Measured on this filesystem with four jobs on four nodes: fcntl.flock
    # granted all four simultaneously -- it is honoured node-locally with no cluster
    # coordination -- while fcntl.lockf serialised them exactly (waits 0/20/40/60 s, no
    # overlapping intervals). They are different mechanisms with different cluster support; do
    # not "simplify" this back to flock.
    #
    # Chosen over a mkdir/O_EXCL lock because the kernel releases this one when the holder dies.
    # These jobs get killed and time out; a lock that survives its owner turns one dead job into
    # every later job hanging until someone clears it by hand.
    with open(_BUILD_LOCK, "w") as _lock:
        fcntl.lockf(_lock, fcntl.LOCK_EX)
        _build_gizmo_locked(test_name, num_openmp_threads, extra_config_flags)


def _build_gizmo_locked(test_name: str, num_openmp_threads: int, extra_config_flags: tuple):
    """The build itself. Caller must hold _BUILD_LOCK -- see build_gizmo_for_test."""
    # Only this test's binary, NOT test/*/GIZMO: removing other tests' binaries breaks a
    # concurrent job that has already built and is about to run, and serves no purpose here.
    system(f"rm -f GIZMO test/{test_name}/GIZMO")
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


def run_test(test_name: str, num_mpi_ranks: int = 1, num_openmp_threads: int = 0, timeout: float | None = None,
             param_overrides: dict | None = None):
    """Runs the test. If num_openmp_threads > 0, sets OMP_NUM_THREADS for the run.
    If the GIZMO subprocess exceeds the timeout, it is killed and the test is skipped
    via pytest.skip. Timeout defaults to GIZMO_TEST_TIMEOUT env var or DEFAULT_TEST_TIMEOUT.
    param_overrides replaces parameter values for this run only, via a sibling params file.
    Raises if GIZMO exits nonzero or stops before TimeMax.
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
    if param_overrides:
        paramsfile = write_params_with_overrides(paramsfile, param_overrides)
    if environ.get("SLURM_JOB_ID"):
        cmd = ["srun", "-n", str(num_mpi_ranks), "--cpu-bind=none"]
    else:
        cmd = ["mpirun", "-np", str(num_mpi_ranks), "--use-hwthread-cpus", "--oversubscribe"]
    if num_openmp_threads > 0 and not environ.get("SLURM_JOB_ID"):
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
        system(f"cp cooling/TREECOOL {test_directory}")


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


def _require_repo_root(test_name: str, caller: str):
    """The stash/variant moves below are all relative to the repo root. Called from the test
    directory instead, every path misses, nothing moves, and the baseline is left stashed with
    the variant's output standing in for it -- so refuse rather than silently do nothing."""
    if not path.isdir(f"test/{test_name}"):
        raise RuntimeError(
            f"{caller}('{test_name}') called from {getcwd()}, where 'test/{test_name}' does not "
            "exist. Restore the working directory before calling this (see build_and_run_test)."
        )


def stash_baseline_output(test_name: str, extra_config_flags=()):
    """If running a non-baseline variant, move an existing output/ aside so the variant
    run doesn't clobber it. Returns True if a stash was made."""
    if not variant_suffix(extra_config_flags):
        return False
    _require_repo_root(test_name, "stash_baseline_output")
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
    _require_repo_root(test_name, "finalize_variant_output")
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


def build_and_run_test(test_name: str, num_mpi_ranks: int = 1, num_openmp_threads: int = 0, extra_config_flags: tuple = (), timeout: float | None = None,
                       param_overrides: dict | None = None):
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
    cwd = getcwd()
    try:
        chdir(f"test/{test_name}/")
        try:
            download_test_files(test_name)
            run_test(test_name, num_mpi_ranks, num_openmp_threads, timeout=timeout,
                     param_overrides=param_overrides)
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
    """Return the path to the last snapshot produced by a test (variant-aware).

    Refuses to hand back a snapshot from a run that was killed or stopped short, so that a
    truncated directory cannot be mistaken for a finished one -- particularly by tests that
    compare variants against each other and would otherwise report a physics regression."""
    output_dir = variant_output_dir(test_name, extra_config_flags)
    reason = run_truncated_reason(test_name, extra_config_flags)
    if reason:
        raise RuntimeError(f"{output_dir} is from a run that did not finish: {reason}")
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
