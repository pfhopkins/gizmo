"""stella_configs: the production configurations in scripts/stella read what they set. Static; no build, no run.
  test_every_setting_is_read: each flag of Config_stella_*.sh and the template appears in src/ outside Template_Config.sh, each
    parameter of stella_*.params is registered in begrun.cc (scripts/stella/check_config_reads.py, exit 0).
  test_checker_reports_unread: a flag and a parameter the code does not know are named and the exit code is 1.
  test_setups: default, cloud, galaxy and zoom start from STELLA_DEFAULTS; the zoom is comoving with TimeBetSnapshot > 1
    (run.cc stops otherwise); cloud, galaxy and tall box are not.
  test_gas_kernel_radius_uncapped: every tracked params file of a Stella build with adaptive gas softening sets
    MinGasKernelRadiusFractional, and the gas kernel radius floor it gives is below 1e-3 pc.
"""
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CFG = REPO / "scripts" / "stella"
CHECK = CFG / "check_config_reads.py"


def run_check(*files):
    return subprocess.run([sys.executable, str(CHECK), str(REPO / "src"), *map(str, files)], capture_output=True, text=True)


def test_every_setting_is_read():
    r = run_check(*sorted(CFG.glob("Config_stella_*.sh")), CFG / "Template_Config_stella.sh", *sorted(CFG.glob("*.params")))
    assert r.returncode == 0, r.stdout


def test_checker_reports_unread(tmp_path):
    (tmp_path / "Config_x.sh").write_text("SGCHEM\nNO_SUCH_FLAG_IN_THE_CODE=3\n")
    (tmp_path / "x.params").write_text("NoSuchParameterInTheCode 1\nTimeMax 1\n")
    r = run_check(tmp_path / "Config_x.sh", tmp_path / "x.params")
    assert r.returncode == 1
    assert "UNREAD flag: NO_SUCH_FLAG_IN_THE_CODE" in r.stdout and "UNREAD parameter: NoSuchParameterInTheCode" in r.stdout
    assert "SGCHEM" not in r.stdout and "TimeMax" not in r.stdout


def par(name):
    return {l.split()[0]: l.split()[1] for l in (CFG / name).read_text().splitlines() if l.strip()}


def test_setups():
    for c in ("default", "cloud", "galaxy", "zoom"):
        assert re.search(r"^STELLA_DEFAULTS$", (CFG / f"Config_stella_{c}.sh").read_text(), re.M), c
    z = par("stella_zoom.params")
    assert z["ComovingIntegrationOn"] == "1" and float(z["TimeBetSnapshot"]) > 1.0 and float(z["OmegaBaryon"]) > 0
    for p in ("stella_cloud.params", "stella_galaxy.params", "stella_tallbox.params"):
        assert par(p)["ComovingIntegrationOn"] == "0", p


KERNEL_FAC = 2.8  # MinKernelRadius = MinGasKernelRadiusFractional x ForceSoftening[0], ForceSoftening = 2.8 x SofteningGas (cubic spline)
FLOOR_MAX_PC = 1e-3


def active_flags(config):
    return {l.split("#")[0].strip().split("=")[0] for l in config.read_text().splitlines()} - {""}


def stella_params_files():
    """tracked params files of Stella builds (STELLA_DEFAULTS or GALSF_SINGLE_STARS_FROM_IMF) with adaptive gas softening"""
    files = subprocess.run(["git", "ls-files", "*.params"], cwd=REPO, capture_output=True, text=True).stdout.split()
    out = []
    for f in map(Path, files):
        config = REPO / f.parent / ("Config_" + f.stem + ".sh" if f.parent == Path("scripts/stella") else "Config.sh")
        if not config.exists():
            continue
        flags = active_flags(config)
        if flags & {"STELLA_DEFAULTS", "GALSF_SINGLE_STARS_FROM_IMF"} and flags & {"ADAPTIVE_GRAVSOFT_FORALL", "ADAPTIVE_GRAVSOFT_FORGAS"}:
            out.append(f)
    return out


@pytest.mark.parametrize("params", stella_params_files(), ids=str)
def test_gas_kernel_radius_uncapped(params):
    """without MinGasKernelRadiusFractional, adaptive gas softening defaults it to 1 (begrun.cc): h >= 2.8 x SofteningGas"""
    p = {l.split()[0]: l.split()[1] for l in (REPO / params).read_text().splitlines() if l.split() and not l.startswith(("%", "#"))}
    assert "MinGasKernelRadiusFractional" in p, f"{params}: MinGasKernelRadiusFractional not set (defaults to 1: h floored at 2.8 x SofteningGas)"
    comoving = p.get("ComovingIntegrationOn", "0") == "1"
    soft = float(p["SofteningGasMaxPhys"]) / float(p.get("HubbleParam", 1)) if comoving else float(p["SofteningGas"])
    floor_pc = float(p["MinGasKernelRadiusFractional"]) * KERNEL_FAC * soft * float(p.get("UnitLength_in_cm", 3.085678e21)) / 3.085678e18
    assert math.isfinite(floor_pc) and floor_pc < FLOOR_MAX_PC, f"{params}: gas kernel radius floored at {floor_pc:.3g} pc"
