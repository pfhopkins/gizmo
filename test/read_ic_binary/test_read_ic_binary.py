"""Regression test: GIZMO must read a GADGET-2 *binary* IC that carries no Temperature block.

blockpresent() answers "should this be written to snapshots?", and read_ic.cc used it to decide
what to READ from the ICs -- different questions, and for Temperature the answers differ.
OUTPUT_TEMPERATURE is implicit for any COOLING (non-Grackle) run, i.e. essentially every
FIRE/STARFORGE config, so read_ic demanded a Temperature block in the ICs. Gadget-binary ICs
have no block labels, so absence cannot be probed and the sequential fread ran off the end:

    reading block 78 (Temperature)...
    I/O error (fread) on task=0 has occured: end of file
    ENDRUN ... 'my_fread()' ... error level 778

That broke every legacy gadget-binary IC, including the MUSIC zoom ICs behind the FIRE suite.
HDF5 ICs were spared because that reader probes with H5Dopen and skips absent datasets. The fix
is therefore scoped to the binary formats only (ICFormat 1/2): HDF5 behaviour is deliberately
left bit-for-bit unchanged, and INPUT_READ_TEMPERATURE forces the read regardless of format.

Parametrized both directions so it cannot silently stop testing the bug:
  * default                -> must read the IC and run (the fix)
  * INPUT_READ_TEMPERATURE -> must fail with the end-of-file signature above, i.e. the pre-fix
                              behaviour. If this ever passes, the IC no longer reproduces the
                              condition and the first variant has stopped meaning anything.

The IC is generated locally (make_binary_ic.py), so this deliberately avoids
build_and_run_test(), which would try to fetch <name>_ics.hdf5.
"""

import glob
from os import chdir, getcwd, path

import pytest

from gizmo.test import (
    build_gizmo_for_test, clean_test_outputs, default_mpi_ranks, default_omp_threads,
    finalize_variant_output, run_test, stash_baseline_output, variant_output_dir,
)

from make_binary_ic import make_binary_ic

TEST_NAME = "read_ic_binary"
TEST_DIR = f"test/{TEST_NAME}"
IC_FILE = f"{TEST_DIR}/{TEST_NAME}_ics"
N_GAS = 512

_EOF_SIGNATURE = "end of file"


def _reads_temperature_block(log):
    """True if GIZMO tried to READ a Temperature block from the ICs.

    Must be line-scoped: with OUTPUT_TEMPERATURE set (as Config.sh does) the string
    "(Temperature)" also appears when the field is *written* to snapshots, so a bare
    substring test on the whole log matches the write path too.
    """
    return any(("reading block" in ln) and ("(Temperature)" in ln) for ln in log.splitlines())


def _build_and_run_local(num_mpi_ranks, num_omp_threads, extra_config_flags, **ic_kwargs):
    """build_and_run_test() minus download_test_files(): our IC is generated, not fetched."""
    clean_test_outputs(TEST_NAME, extra_config_flags)
    build_gizmo_for_test(TEST_NAME, num_omp_threads, extra_config_flags)
    stash_baseline_output(TEST_NAME, extra_config_flags)
    cwd = getcwd()
    try:
        chdir(TEST_DIR)
        try:
            make_binary_ic(f"{TEST_NAME}_ics", ngas=N_GAS, **ic_kwargs)
            run_test(TEST_NAME, num_mpi_ranks, num_omp_threads)
        finally:
            chdir(cwd)
    finally:
        finalize_variant_output(TEST_NAME, extra_config_flags)


def _run_log_text():
    p = f"{TEST_DIR}/test_{TEST_NAME}.out"
    return open(p, errors="replace").read() if path.isfile(p) else ""


@pytest.mark.parametrize("num_mpi_ranks", (min(default_mpi_ranks(), 2),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize(
    "extra_config_flags",
    [(), ("INPUT_READ_TEMPERATURE",)],
    ids=["no_temperature_block_is_fine", "sensitivity_demands_temperature"],
)
def test_read_ic_binary(num_mpi_ranks, num_omp_threads, extra_config_flags):
    _build_and_run_local(num_mpi_ranks, num_omp_threads, extra_config_flags)

    log = _run_log_text()
    snaps = sorted(glob.glob(variant_output_dir(TEST_NAME, extra_config_flags) + "/snapshot_*.hdf5"))
    demands_temperature = "INPUT_READ_TEMPERATURE" in extra_config_flags

    assert log, f"no run log at {TEST_DIR}/test_{TEST_NAME}.out"

    if demands_temperature:
        # Sensitivity check: asking for the block that the IC does not contain MUST fail,
        # with the same signature the bug produced. This keeps the other variant honest.
        assert _reads_temperature_block(log), (
            "expected GIZMO to attempt to READ the Temperature block with INPUT_READ_TEMPERATURE "
            "set; if it no longer does, this test has stopped reproducing the original condition"
        )
        assert _EOF_SIGNATURE in log, (
            "expected an end-of-file error when demanding a Temperature block from an IC that "
            "has none -- this test no longer reproduces the bug it is guarding against"
        )
        assert not snaps, "GIZMO should not have completed a snapshot after the IC read failed"
        return

    # The fix: no Temperature block in the IC is perfectly fine.
    assert _EOF_SIGNATURE not in log, (
        "GIZMO hit an end-of-file reading a gadget-binary IC. It is demanding a block the IC "
        "does not contain (historically Temperature -- see this file's docstring)."
    )
    assert not _reads_temperature_block(log), (
        "GIZMO tried to read a Temperature block from the ICs without INPUT_READ_TEMPERATURE; "
        "the IC-read path is keyed off output settings again"
    )
    assert snaps, "GIZMO produced no snapshots -- it failed to start from the binary IC"

    # Temperature must still be WRITTEN to snapshots (Config.sh sets OUTPUT_TEMPERATURE):
    # the fix must not have "worked" by disabling temperature output altogether.
    import h5py
    with h5py.File(snaps[-1], "r") as F:
        assert "Temperature" in F["PartType0"], (
            "Temperature missing from snapshot output despite OUTPUT_TEMPERATURE: the input-side "
            "fix must not disable output of the field"
        )


@pytest.mark.parametrize("num_mpi_ranks", (min(default_mpi_ranks(), 2),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize(
    "id_dtype, id_offset",
    [("uint32", 0), ("uint64", 2**33)],
    ids=["ids_32bit", "ids_64bit"],
)
def test_read_ic_binary_ids(num_mpi_ranks, num_omp_threads, id_dtype, id_offset):
    """ParticleIDs from a binary IC must arrive intact, whichever width the file stores them at.

    Legacy GADGET-2 binary ICs carry 32-bit IDs; MUSIC and LONGIDS-era files carry 64-bit ones.
    Reading 4-byte IDs but unpacking them at sizeof(MyIDType) fuses neighbouring pairs into huge
    values and fills the rest from stale buffer contents, silently (issue #79). The 64-bit case
    offsets the IDs past 2^32 so a truncating reader cannot pass it.
    """
    import h5py
    import numpy as np

    _build_and_run_local(num_mpi_ranks, num_omp_threads, (), id_dtype=np.dtype(id_dtype), id_offset=id_offset)
    snaps = sorted(glob.glob(variant_output_dir(TEST_NAME) + "/snapshot_*.hdf5"))
    assert snaps, f"GIZMO produced no snapshots from a binary IC with {id_dtype} IDs; see {TEST_DIR}/test_{TEST_NAME}.out"

    with h5py.File(snaps[0], "r") as F:
        ids = np.sort(F["PartType0/ParticleIDs"][:].astype(np.uint64))
    expected = np.arange(1, N_GAS + 1, dtype=np.uint64) + np.uint64(id_offset)
    n_bad = int(np.count_nonzero(ids != expected))
    assert n_bad == 0, (
        f"{n_bad}/{N_GAS} ParticleIDs differ from the IC's {id_dtype} IDs; "
        f"min/max read {ids.min()}/{ids.max()}, expected {expected[0]}/{expected[-1]}, "
        f"{N_GAS - np.unique(ids).size} duplicates"
    )
