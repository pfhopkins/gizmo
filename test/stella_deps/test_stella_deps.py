"""stella_deps test: scripts/get_stella_deps.sh, the one setup command for the Stella dependencies outside this repository.
  1. a clean HOME: the setup installs the stella-tables release of the pinned commit into ~/.stella-deps/tables-<ref>, hash
     checked, and SUNDIALS at the version pinned in src/Makefile; the stamp holds no machine-local path; a rerun builds and
     downloads nothing;
  2. make reports the KETJU install it will use: under the cache, keyed by the pinned commits and by the Kokkos install;
     an explicit KETJU_PATH wins;
  3. a KETJU or SGChem build without the install stops in make, naming the setup command;
  4. (slow) the whole new-machine procedure, new_machine.sh: fresh clone, setup, build and run the cloud smoke test with and
     without KETJU, rerun, start-up stop without the cache.
Tests 1 and 4 need the network and are skipped without it."""
import os
import re
import subprocess
import time
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
SETUP = ROOT / "scripts" / "get_stella_deps.sh"
REF = re.search(r'#define STELLA_TABLES_REF "([0-9a-f]+)"', (ROOT / "src/galaxy_sf/stella_tables_check.h").read_text()).group(1)
SUNDIALS = re.search(r"^SUNDIALS_VERSION *= *(\S+)", (ROOT / "src/Makefile").read_text(), re.M).group(1)
KETJU_REF = re.search(r"^KETJU_REF *= *([0-9a-f]+)", (ROOT / "src/gravity/ketju/Makefile.ketju").read_text(), re.M)


def online():
    url = f"https://github.com/Ulli1991/stella-tables/releases/download/tables-{REF[:7]}/SHA256SUMS"
    return subprocess.run(["curl", "-fsSL", "-m", "20", "-o", os.devnull, url], capture_output=True).returncode == 0


needs_network = pytest.mark.skipif(not online(), reason="no network")


def env_with(**kw):
    env = {k: v for k, v in os.environ.items() if k not in ("STELLA_DEPS_PREFIX", "KETJU_PATH", "SUNDIALS_DIR")}
    env.update({k: str(v) for k, v in kw.items()})
    return env


@needs_network
def test_setup_tables_clean_home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    out = subprocess.run(["bash", str(SETUP)], capture_output=True, text=True, env=env_with(HOME=home))
    assert out.returncode == 0, out.stdout + out.stderr
    tables = home / ".stella-deps" / f"tables-{REF[:7]}"
    for f in ("stellar_tables_unified.hdf5", "stellar_wind_cumulative.hdf5", "energy-table-x.txt", "eh2-table.txt",
              "H2-cooling-ratios.dat", "wss_z_collis.txt"):
        assert (tables / f).is_file(), f
    stamp = (tables / "stella_tables.stamp").read_text()
    assert REF in stamp and f"release tables-{REF[:7]}" in stamp and str(tmp_path) not in stamp and "/tmp" not in stamp
    assert "all checks passed" in out.stdout
    assert (home / ".stella-deps" / f"sundials-{SUNDIALS}" / "include" / "cvode" / "cvode.h").is_file()
    assert sorted(p.name for p in (home / ".stella-deps").iterdir()) == [f"sundials-{SUNDIALS}", f"tables-{REF[:7]}"]   # no leftovers
    t0 = time.time()
    again = subprocess.run(["bash", str(SETUP)], capture_output=True, text=True, env=env_with(HOME=home))
    assert again.returncode == 0 and again.stdout.count("already installed") == 2
    assert "downloading" not in again.stdout and "building" not in again.stdout
    assert time.time() - t0 < 30


def ketju_path(*args, **kw):
    out = subprocess.run(["make", "-s", "--no-print-directory", "-C", str(ROOT / "src"), "CONFIGVARS=KETJU_REGULARIZATION", *args, "ketju-path"],
                         capture_output=True, text=True, env=env_with(**kw))
    assert out.returncode == 0, out.stdout + out.stderr
    return dict(re.findall(r"^(\w+)='(.*)'$", out.stdout, re.M))


def test_ketju_path_default(tmp_path):
    assert KETJU_REF, "KETJU_REF pin missing in Makefile.ketju"
    v = ketju_path(STELLA_DEPS_PREFIX=tmp_path)
    assert v["KETJU_PATH"].startswith(f"{tmp_path}/ketju-{KETJU_REF.group(1)[:7]}-{v['KETJU_C_REF'][:7]}-kokkos-")
    assert v["KETJU_KOKKOS"] and v["KETJU_OK"] == "0"
    other = ketju_path(f"KOKKOS_PATH={tmp_path}/another_kokkos", STELLA_DEPS_PREFIX=tmp_path)
    assert other["KETJU_PATH"] != v["KETJU_PATH"]   # keyed by the Kokkos install
    assert ketju_path(KETJU_PATH=tmp_path / "mine")["KETJU_PATH"] == str(tmp_path / "mine")


def test_missing_install_stops_make(tmp_path):
    out = subprocess.run(["make", "-n", "-C", str(ROOT / "src"), "CONFIGVARS=KETJU_REGULARIZATION"], capture_output=True, text=True,
                         env=env_with(STELLA_DEPS_PREFIX=tmp_path))
    assert out.returncode != 0 and "run scripts/get_stella_deps.sh --ketju" in out.stderr
    out = subprocess.run(["make", "-n", "-C", str(ROOT / "src"), "CONFIGVARS=SGCHEM"], capture_output=True, text=True,
                         env=env_with(STELLA_DEPS_PREFIX=tmp_path))
    assert out.returncode != 0 and f"SUNDIALS_DIR={tmp_path}/sundials-{SUNDIALS}; run scripts/get_stella_deps.sh" in out.stderr


@pytest.mark.slow
@needs_network
def test_new_machine(tmp_path):
    out = subprocess.run(["bash", str(Path(__file__).parent / "new_machine.sh"), str(tmp_path)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-4000:] + out.stderr[-4000:]
    assert "new_machine: all steps passed" in out.stdout
