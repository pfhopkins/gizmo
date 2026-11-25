"""SGChem solver failures and the dust temperature (stella_dev D-58 to D-61; STL-64, STL-164, STL-294), C++ backend, nets 5 and 17.

solver_failure: the WLM cells whose CVODE step hit the floor and the TNOCHEM hand-off grid, each within 1% of the converged
reference (rtol 1e-8, stella_dev), no floor hit, no D+ excess rejection, recoverable H+ rejections only in the known
calls; forced failures (TEST_SGCHEM_FORCED_SOLVER_FAILURE): retry, TNOCHEM finish, abort, dump replay, no stale CR
heating on flagged evaluations. dust_temperature: dense cold sequences without bisection, Newton <= 30 iterations from a
cold start and <= 20 from the previous root, slope against a central difference <= 1e-6.

Standalone: no GIZMO run. Needs mpicxx, SUNDIALS_DIR and the chemistry tables (SGCHEM_TABLES_DIR or get_stella_tables.sh).
"""

import subprocess
import sys
from os import environ, path, makedirs

import pytest

HERE = path.dirname(path.abspath(__file__))
sys.path.insert(0, path.join(path.dirname(HERE), "sgchem_onezone"))
from test_sgchem_onezone import chemistry_tables_dir  # noqa: E402


@pytest.mark.parametrize("net", (5, 17))
def test_sgchem_solver_failure(net):
    if not environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR")
    env = {**environ, "CHEMCXX_DATADIR": chemistry_tables_dir(net)}
    out = path.join(HERE, "output", f"net{net}")
    makedirs(out, exist_ok=True)
    r = subprocess.run(["bash", path.join(HERE, "build_and_run.sh"), str(net), out], capture_output=True, text=True, env=env)
    print(r.stdout)
    for b, f in (("default", "solver_failure"), ("default", "dust_temperature"), ("forced", "solver_failure")):
        p = path.join(out, b, f + ".txt")
        if path.exists(p):
            print(f"---- {b}/{f}\n" + open(p).read()[-6000:])
    assert r.returncode == 0, f"net {net}:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}"
