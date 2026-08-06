"""Regression test for RANDOMIZE_GRAVTREE in the periodic TreePM regime.

Background: RANDOMIZE_GRAVTREE decorrelates tree-force errors between rebuilds
(Grudic+ 2020, arXiv:2010.11254), ported from AREPO. AREPO uses two mechanisms:
for NON-periodic gravity it randomly moves/enlarges the tree root node; for
PERIODIC gravity it instead translates all coordinates by a random vector mod
the box, keeping the box size fixed (Weinberger, Springel & Pakmor 2020, sec 3.1).

GIZMO's current implementation applies the non-periodic method (shift the domain
center and double the root node, `len *= 2` in domain_findExtent) to ALL cases.
In a periodic box that doubling throws away one bit of Peano-Hilbert resolution
per dimension: every particle lives in the central octant, the top-tree wastes
its first refinement level, and in a clumpy / zoom-in configuration the top-tree
can no longer refine finely enough inside the dense region to split its gravity
cost across MPI ranks -> severe domain load imbalance.

This test reuses the periodic PMGRID Zeldovich setup and turns on
RANDOMIZE_GRAVTREE. The primary, scale-independent assertion is the invariant
that the bug violates:

    in a periodic box, RANDOMIZE_GRAVTREE must NOT enlarge the root node,
    i.e. DomainLen must stay ~= BoxSize, not ~= 2 * BoxSize.

Against the CURRENT (unpatched) code this test is expected to FAIL (DomainLen is
~2x BoxSize). Once the periodic path is fixed to use a coordinate translation
instead of box-doubling, it should pass. The measured domain load imbalance is
reported as a secondary diagnostic (its magnitude at CI scale is not asserted,
since the pathology only becomes severe at many ranks / high dynamic range).
"""

import re
import glob
import pytest
from gizmo.test import (
    build_and_run_test, clean_test_outputs, default_mpi_ranks,
    default_omp_threads, variant_output_dir,
)

# We piggy-back on the zeldovich test directory so the IC download + params are
# already wired; RANDOMIZE_GRAVTREE is added purely as a compile flag.
_TEST_NAME = "zeldovich"
_FLAGS = ("RANDOMIZE_GRAVTREE",)
_RUN_LOG = f"test/{_TEST_NAME}/test_{_TEST_NAME}.out"

_DOMAINLEN_RE = re.compile(
    r"RANDOMIZE_GRAVTREE:\s*DomainLen=([-\d.eE+]+).*?BoxSize=([-\d.eE+]+)"
)
_IMBAL_RE = re.compile(r"imbalance:.*?load=([-\d.eE+]+)")


def _parse_log(path):
    """Return (list of (DomainLen, BoxSize), list of load-imbalance values)."""
    domainlens, imbalances = [], []
    with open(path) as f:
        for line in f:
            m = _DOMAINLEN_RE.search(line)
            if m:
                domainlens.append((float(m.group(1)), float(m.group(2))))
            m = _IMBAL_RE.search(line)
            if m:
                imbalances.append(float(m.group(1)))
    return domainlens, imbalances


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_zeldovich_randomize_gravtree(num_mpi_ranks, num_omp_threads):
    clean_test_outputs(_TEST_NAME, _FLAGS)
    build_and_run_test(_TEST_NAME, num_mpi_ranks, num_omp_threads,
                       extra_config_flags=_FLAGS)

    # sanity: the run actually produced output (didn't crash in decomposition)
    outputdir = variant_output_dir(_TEST_NAME, _FLAGS)
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    assert snaps, "GIZMO produced no snapshots with RANDOMIZE_GRAVTREE + periodic TreePM"

    domainlens, imbalances = _parse_log(_RUN_LOG)
    assert domainlens, (
        "no 'RANDOMIZE_GRAVTREE: DomainLen=...' diagnostic lines found in the run log; "
        "instrumentation in domain_findExtent may be missing"
    )

    # --- secondary diagnostic: domain load imbalance over the run ---
    if imbalances:
        imbalances.sort()
        med = imbalances[len(imbalances) // 2]
        print(f"[randomize_gravtree] load imbalance over run: "
              f"median={med:.3f} max={max(imbalances):.3f} (n={len(imbalances)})")

    # --- primary invariant: root node not enlarged in a periodic box ---
    # domain_findExtent sets len = 1.001 * (particle extent); for a full periodic
    # box the extent ~= BoxSize, so a correct periodic implementation gives
    # DomainLen ~= BoxSize. The buggy doubling gives DomainLen ~= 2 * BoxSize.
    worst_ratio = max(dl / box for dl, box in domainlens)
    assert worst_ratio < 1.1, (
        f"RANDOMIZE_GRAVTREE enlarged the root node in a periodic box "
        f"(DomainLen/BoxSize = {worst_ratio:.3f}, expected ~1.0). "
        f"Box-doubling discards SFC resolution and wrecks periodic-TreePM load balance; "
        f"the periodic path should use a coordinate translation instead."
    )
