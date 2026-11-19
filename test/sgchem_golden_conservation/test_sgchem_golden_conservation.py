"""SGCHEM golden conservation audit (cooling/sgchem/cxx/golden/test_conservation.cc) for the production network: net 5 on the
production configuration Config_stella_default.sh, built from a copy of the tree by build_conservation.sh. Phase R (RHS parity
Fortran vs C++, 1e-12), phase C (element pools: overdraft and negatives after each completed call <= 1e-14 of the pool, no failed or
timed-out call), phase O (one-zone final T within 1 %), phase I and phase M, and the gate-3 table golden. Pass = exit status 0.
Needs SUNDIALS (SUNDIALS_DIR, default the fb_tests install) and the SGCHEM tables in the tree."""

import os
import subprocess
from os.path import abspath, dirname, join

import pytest

TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))


def test_sgchem_golden_conservation_net5():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds the golden binaries")
    out = join(TEST_DIR, "output", "net5")
    env = dict(os.environ, CANON_CONFIG=join(REPO, "Config_stella_default.sh"), JOBS=os.environ.get("JOBS", "4"))
    args = ["bash", join(REPO, "cooling", "sgchem", "cxx", "golden", "build_conservation.sh"), REPO, out, "5"]
    if os.environ.get("SUNDIALS_DIR"):
        args.append(os.environ["SUNDIALS_DIR"])
    r = subprocess.run(args, env=env, capture_output=True, text=True)
    txt = open(join(out, "conservation_net5.txt")).read() if os.path.exists(join(out, "conservation_net5.txt")) else ""
    print("\n".join(l for l in txt.splitlines() if "PHASE" in l or "PARITY" in l or "failures" in l or "timeouts" in l))
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
