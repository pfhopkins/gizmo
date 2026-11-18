"""Static check: every shipped Stella production config (cloud, default, tallbox, zoom) gets STARS_APPROACH_TIMESTEP from
declarations/precompiler_logic.h and compiles the files that use it (timestep, gravity tree); the opt-out
STARS_APPROACH_TIMESTEP_OFF removes it. Needs the build modules (mpicxx, gsl, hdf5, fftw) like any build."""
import os
import re
import shutil
import subprocess
import tempfile

import pytest

REPO = os.getcwd()
CONFIGS = ["Config_stella_cloud.sh", "Config_stella_default.sh", "Config_stella_tallbox.sh", "Config_stella_zoom.sh"]
SOURCES = ["core/timestep.cc", "gravity/forcetree.cc", "gravity/gravtree.cc"]


def prepare(cfg, extra):
    d = tempfile.mkdtemp(prefix="stella_cfg_")
    files = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=REPO, capture_output=True, check=True).stdout.split(b"\0")
    for f in files:
        f = f.decode()
        if not f or f.startswith("test/") or not os.path.isfile(os.path.join(REPO, f)):
            continue
        os.makedirs(os.path.dirname(os.path.join(d, f)) or d, exist_ok=True)
        shutil.copy2(os.path.join(REPO, f), os.path.join(d, f))
    shutil.copy(os.path.join(REPO, cfg), os.path.join(d, "Config.sh"))
    with open(os.path.join(d, "Config.sh"), "a") as fh:
        fh.write(extra)
    return d


def dry_compile(d, src):
    obj = src.replace(".cc", ".o")
    out = subprocess.run(["make", "-n", "-W", src, obj], cwd=d, capture_output=True, text=True, check=True).stdout
    cmd = [l for l in out.splitlines() if l.startswith("mpicxx") and " -c " in l][-1]
    return cmd


@pytest.mark.parametrize("cfg", CONFIGS)
def test_stella_config_has_approach_timestep(cfg):
    d = prepare(cfg, "")
    try:
        r = subprocess.run(["make", "GIZMO_config.h"], cwd=d, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-1500:]
        defs = subprocess.run(dry_compile(d, "core/timestep.cc").replace(" -c ", " -E -dM ").split(" -o ")[0], shell=True, cwd=d, capture_output=True, text=True, check=True).stdout
        assert re.search(r"^#define STARS_APPROACH_TIMESTEP\b", defs, re.M), cfg + ": STARS_APPROACH_TIMESTEP not defined"
        for src in SOURCES:
            cmd = dry_compile(d, src).replace(" -c ", " -fsyntax-only ").split(" -o ")[0]
            r = subprocess.run(cmd, shell=True, cwd=d, capture_output=True, text=True)
            assert r.returncode == 0, f"{cfg} {src}:\n" + r.stderr[-1500:]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_stella_approach_timestep_opt_out():
    d = prepare("Config_stella_cloud.sh", "STARS_APPROACH_TIMESTEP_OFF\n")
    try:
        r = subprocess.run(["make", "GIZMO_config.h"], cwd=d, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-1500:]
        defs = subprocess.run(dry_compile(d, "core/timestep.cc").replace(" -c ", " -E -dM ").split(" -o ")[0], shell=True, cwd=d, capture_output=True, text=True, check=True).stdout
        assert not re.search(r"^#define STARS_APPROACH_TIMESTEP\b", defs, re.M)
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.mark.parametrize("merge", ["", "KETJU_MERGE_STARS\n", "KETJU_MERGE_BH\n"])
def test_ketju_coupling_compiles_without_or_with_merge_flags(merge):
    """galaxy_sf/ketju_coupling.cc must compile with and without KETJU_MERGE_STARS / KETJU_MERGE_BH. Needs the KETJU headers:
    set KETJU_PATH to a KETJU install (scripts/get_stella_deps.sh) and KOKKOS_ROOT."""
    lib = os.environ.get("KETJU_PATH")
    if not lib or not os.path.isdir(os.path.join(lib, "include", "mstar")):
        pytest.skip("KETJU_PATH not set")
    d = prepare("Config_stella_cloud.sh", "KETJU_REGULARIZATION\nADAPTIVE_GRAVSOFT_FORALL=1\n" + merge)
    try:
        r = subprocess.run(["make", "GIZMO_config.h"], cwd=d, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-1500:]
        cmd = dry_compile(d, "galaxy_sf/ketju_coupling.cc").replace(" -c ", " -fsyntax-only ").split(" -o ")[0]
        r = subprocess.run(cmd, shell=True, cwd=d, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-1500:]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_stella_zoom_config_has_no_radiation_pressure():
    """The zoom reference configuration has no radiation pressure (Uli 2026-11-18): after precompiler_logic.h no star-side,
    tree or M1 radiation-pressure macro is defined, and the files that use the band fields still compile."""
    d = prepare("Config_stella_zoom.sh", "")
    try:
        r = subprocess.run(["make", "GIZMO_config.h"], cwd=d, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-1500:]
        defs = subprocess.run(dry_compile(d, "core/timestep.cc").replace(" -c ", " -E -dM ").split(" -o ")[0], shell=True, cwd=d, capture_output=True, text=True, check=True).stdout
        for m in ("GALSF_RESOLVEDISM_RADPRESSURE", "GALSF_RESOLVEDISM_RADPRESSURE_TREERAD", "RT_RAD_PRESSURE_FORCES"):
            assert not re.search(r"^#define " + m + r"\b", defs, re.M), "zoom config defines " + m
        for src in ["galaxy_sf/resolvedism_fb.cc", "galaxy_sf/resolvedism_fb_momentum.cc", "cooling/sgchem/chem_interface.cc", "gravity/forcetree.cc"]:
            cmd = dry_compile(d, src).replace(" -c ", " -fsyntax-only ").split(" -o ")[0]
            r = subprocess.run(cmd, shell=True, cwd=d, capture_output=True, text=True)
            assert r.returncode == 0, f"{src}:\n" + r.stderr[-1500:]
    finally:
        shutil.rmtree(d, ignore_errors=True)
