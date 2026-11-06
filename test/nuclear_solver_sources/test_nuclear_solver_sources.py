"""NUCLEAR_NETWORK_SOLVER=1 / =2 add the SkyNet / external-network source file to the build."""
import re
import subprocess
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.mark.parametrize("solver,source", [(1, "nuclear/nuclear_skynet.cc"), (2, "nuclear/nuclear_torch.cc")])
def test_solver_source_is_compiled(tmp_path, solver, source):
    config = tmp_path / "Config.sh"
    config.write_text(f"EOS_HELMHOLTZ\nNUCLEAR_NETWORK\nNUCLEAR_NETWORK_SOLVER={solver}\n")
    try:
        out = subprocess.run(["make", "-C", str(SRC), "-n", f"CONFIG={config}"], capture_output=True, text=True).stdout
    finally:
        for name in ("GIZMO_config.h", "compile_time_info.cc"):
            (SRC / name).unlink(missing_ok=True)
    assert source in re.findall(r"-c (\S+\.cc) -o", out)
