"""stella_configs: the production configurations in scripts/stella read what they set. Static; no build, no run.
  test_every_setting_is_read: each flag of Config_stella_*.sh and the template appears in src/ outside Template_Config.sh, each
    parameter of stella_*.params is registered in begrun.cc (scripts/stella/check_config_reads.py, exit 0).
  test_checker_reports_unread: a flag and a parameter the code does not know are named and the exit code is 1.
  test_setups: default, cloud, galaxy and zoom start from STELLA_DEFAULTS; the zoom is comoving with TimeBetSnapshot > 1
    (run.cc stops otherwise); cloud, galaxy and tall box are not.
"""
import re
import subprocess
import sys
from pathlib import Path

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
