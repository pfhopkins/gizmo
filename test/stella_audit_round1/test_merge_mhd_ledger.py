"""STL-428: a merged cell's kicks book both parents' MHD ledger rates (DtEnergyMHDNum), a split partitions it with the mass.

The merge/split driver of test/merge_split_conservation, linked against the Stella default build with the test hooks; its
merge_mhd_ledger and split_mhd_ledger checks (equal rates 7.5 + 7.5, bar 1e-12 relative) and every other check must pass.
Expected before the fix (origin/stella_dev objects, this driver): the merged rate is 7.5 (half of it missing), the split pair
holds twice the parent's.
The driver reads the parameter file of merge_split_conservation_stella_default in its own run directory; the hook build
(GALSF_RESOLVEDISM_ISOLATED_FB_TEST) also needs the test-star tags, set to no stars here."""

import os
import subprocess
from os.path import join

from audit_round1_common import HOOKS, REPO, TEST_DIR, binary, driver, write_params


def test_merge_mhd_ledger_rate():
    binary("round1_off", add=HOOKS, drivers=("merge_split_driver",))
    d = join(TEST_DIR, "run_mhd_ledger")
    os.makedirs(join(d, "output"), exist_ok=True)
    if not os.path.islink(join(d, "gizmo")):
        os.symlink(REPO, join(d, "gizmo"))
    write_params(d, {"TestNumStars": 0, "TestStarMass2": 0, "TestStarMass3": 0},
                 base=join(REPO, "test", "merge_split_conservation_stella_default", "merge_split_conservation_stella_default.params"))
    r = subprocess.run([driver("round1_off", "merge_split_driver"), "run.params"], capture_output=True, text=True, cwd=d, timeout=600)
    lines = r.stdout.splitlines()
    print("\n".join(l for l in lines if "mhd_ledger" in l or l.startswith("DONE")))
    assert [l for l in lines if l.startswith("DONE")], r.stdout[-3000:] + r.stderr[-3000:]
    assert any("merge_mhd_ledger:rate_is_the_sum" in l for l in lines), "the ledger case did not run (field not compiled?)"
    failed = [l for l in lines if l.startswith("FAIL")]
    assert not failed, "\n".join(failed)
