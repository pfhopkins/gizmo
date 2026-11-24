"""stella_configs: the production configurations in scripts/stella read what they set. Static; no build, no run.
  test_every_setting_is_read: each flag of Config_stella_*.sh and the template appears in src/ outside Template_Config.sh, each
    parameter of stella_*.params is registered in begrun.cc (scripts/stella/check_config_reads.py, exit 0).
  test_checker_reports_unread: a flag and a parameter the code does not know are named and the exit code is 1.
  test_setups: default, cloud, galaxy, zoom and tall box start from STELLA_DEFAULTS; the zoom is comoving with TimeBetSnapshot > 1 and has
    RESOLVED_ISM_FB_AFTER_CLOSING_KICK without RESOLVED_ISM_RECOMPUTE_HYDRO_POST_FB (as stella_dev);
    (run.cc stops otherwise); cloud, galaxy and tall box are not.
  test_gas_kernel_radius_uncapped: every tracked params file of a Stella build with adaptive gas softening sets
    MinGasKernelRadiusFractional, and the gas kernel radius floor it gives is below 1e-3 pc.
  test_radiation_pressure_per_config: Config through the preprocessor with precompiler_logic.h: cloud, default, galaxy and tall
    box set GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS themselves, the zoom build has no radiation pressure (Uli 2026-11-18).
  test_stella_defaults_without_radiation_pressure: STELLA_DEFAULTS alone sets no radiation pressure.
  test_stella_with_fire_or_starforge_stops: single stars (GALSF_SINGLE_STARS_FROM_IMF) with FIRE stellar evolution or with
    STARFORGE sinks stop the build with the message (Uli 2026-11-19: no hybrid yet; a star without a single-star record is no
    feedback source in a single-star build); FIRE-3 and STARFORGE defaults alone preprocess.
  test_no_environment_options_or_abort: the SGChem host code and the KETJU coupling read no option from the environment and stop
    through endrun, not MPI_Abort (#154; the standalone Fortran reference tool is not GIZMO code).
  test_no_sn_broadcast_by_default: STELLA_DEFAULTS alone and every production config build without SNE_BROADCAST_TIMESTEP
    (Uli 2026-11-19: opt-in only, never a default; the form trunk declined).
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
    for c in ("default", "cloud", "galaxy", "zoom", "tallbox"):
        assert re.search(r"^STELLA_DEFAULTS$", (CFG / f"Config_stella_{c}.sh").read_text(), re.M), c
    z = par("stella_zoom.params")
    assert z["ComovingIntegrationOn"] == "1" and float(z["TimeBetSnapshot"]) > 1.0 and float(z["OmegaBaryon"]) > 0
    for p in ("stella_cloud.params", "stella_galaxy.params", "stella_tallbox.params"):
        assert par(p)["ComovingIntegrationOn"] == "0", p
    for c in ("default", "galaxy", "tallbox", "zoom"):  # feedback after the closing kick is opt-in, in no Stella config (stella_dev 1694d0c79)
        cf = active_flags(CFG / f"Config_stella_{c}.sh")
        assert "RESOLVED_ISM_FB_AFTER_CLOSING_KICK" not in cf and "RESOLVED_ISM_RECOMPUTE_HYDRO_POST_FB" not in cf, c


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


RADP = ("GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS", "GALSF_FB_FIRE_RT_LOCALRP", "GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL",
        "GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE", "GALSF_SINGLE_STARS_RT_TABLE_BANDS")
WITH_RADP = ("cloud", "default", "galaxy", "tallbox")  # stella_dev: cloud, default, tallbox; the zoom reference has none (Uli 2026-11-18)


def macros(tmp_path, lines):
    """the macros a build of these Config.sh lines sees: GIZMO_config.h as file_io/prepare-config.perl writes it, then
    declarations/precompiler_logic.h through the C preprocessor"""
    hdr = ["#ifndef GIZMO_config_H", "#define GIZMO_config_H"]
    for l in lines:
        f = l.split()
        if f and f[0][0] != "#":
            k, _, v = f[0].partition("=")
            hdr.append(f"#define {k} {v}")
    (tmp_path / "GIZMO_config.h").write_text("\n".join(hdr + ["#endif"]) + "\n")
    (tmp_path / "probe.cc").write_text('#include "GIZMO_config.h"\n#include "declarations/precompiler_logic.h"\n')
    r = subprocess.run(["c++", "-std=c++20", "-E", "-dM", f"-I{tmp_path}", f"-I{REPO / 'src'}", str(tmp_path / "probe.cc")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return {l.split()[1].split("(")[0] for l in r.stdout.splitlines() if l.startswith("#define ")}


@pytest.mark.parametrize("c", ["cloud", "default", "galaxy", "tallbox", "zoom"])
def test_radiation_pressure_per_config(tmp_path, c):
    """radiation pressure is set by GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS in the configs that have it, not by STELLA_DEFAULTS:
    cloud, default, galaxy and tall box push the gas, the zoom build defines no radiation-pressure macro"""
    m = macros(tmp_path, (CFG / f"Config_stella_{c}.sh").read_text().splitlines())
    if c in WITH_RADP:
        assert "GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS" in m and "GALSF_FB_FIRE_RT_LOCALRP" in m
        assert re.search(r"^GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS\b", (CFG / f"Config_stella_{c}.sh").read_text(), re.M)
    else:
        assert not [k for k in RADP if k in m], [k for k in RADP if k in m]


def test_stella_defaults_without_radiation_pressure(tmp_path):
    """STELLA_DEFAULTS alone (the compile-suite set) defines no radiation-pressure macro"""
    m = macros(tmp_path, "SGCHEM CHEMISTRYNETWORK=5 METALS GALSF_RESOLVEDISM_TREECOL_PIXELS=12 GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES STELLA_DEFAULTS".split())
    assert "GALSF_SINGLE_STARS_FROM_IMF" in m and not [k for k in RADP if k in m], [k for k in RADP if k in m]


def test_zoom_cr_rate_scales_with_g0(tmp_path):
    """the zoom build scales the SGChem cosmic-ray rate with the stars' field, as stella_dev's Config_stella_zoom.sh
    (CR_SCALE_WITH_G0; GKS-26, Uli 2026-11-23: gku defaults as stella_dev's)"""
    m = macros(tmp_path, (CFG / "Config_stella_zoom.sh").read_text().splitlines())
    assert "CR_SCALE_WITH_G0" in m


@pytest.mark.parametrize("c", ["defaults_only", "cloud", "default", "galaxy", "tallbox", "zoom"])
def test_no_sn_broadcast_by_default(tmp_path, c):
    """SNE_BROADCAST_TIMESTEP is an opt-in option: neither STELLA_DEFAULTS nor a production config switches it on"""
    lines = ("SGCHEM CHEMISTRYNETWORK=5 METALS GALSF_RESOLVEDISM_TREECOL_PIXELS=12 GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES STELLA_DEFAULTS".split()
             if c == "defaults_only" else (CFG / f"Config_stella_{c}.sh").read_text().splitlines())
    m = macros(tmp_path, lines)
    assert "GALSF_SINGLE_STARS_FROM_IMF" in m and "SNE_BROADCAST_TIMESTEP" not in m


STELLA_SET = "SGCHEM CHEMISTRYNETWORK=5 METALS GALSF_RESOLVEDISM_TREECOL_PIXELS=12 GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES STELLA_DEFAULTS".split()
NO_HYBRID = "single stars (GALSF_SINGLE_STARS_FROM_IMF) are not combined with FIRE stellar evolution or STARFORGE sinks"


@pytest.mark.parametrize("lines", [STELLA_SET + ["FIRE_PHYSICS_DEFAULTS=3"], ["GALSF", "GALSF_SINGLE_STARS_FROM_IMF", "GALSF_FB_FIRE_STELLAREVOLUTION=3"],
                                   STELLA_SET + ["SINGLE_STAR_SINK_DYNAMICS"], ["SINGLE_STAR_STARFORGE_DEFAULTS", "GALSF_SINGLE_STARS_FROM_IMF"],
                                   ["FIRE_PHYSICS_DEFAULTS=3"], ["SINGLE_STAR_STARFORGE_DEFAULTS"]],
                         ids=["stella_fire3", "imf_fire_stellarevolution", "stella_sink_dynamics", "starforge_imf", "fire3_alone", "starforge_alone"])
def test_stella_with_fire_or_starforge_stops(tmp_path, lines):
    hdr = ["#ifndef GIZMO_config_H", "#define GIZMO_config_H"] + [f"#define {l.partition('=')[0]} {l.partition('=')[2]}" for l in lines] + ["#endif"]
    (tmp_path / "GIZMO_config.h").write_text("\n".join(hdr) + "\n")
    (tmp_path / "probe.cc").write_text('#include "GIZMO_config.h"\n#include "declarations/precompiler_logic.h"\n')
    r = subprocess.run(["c++", "-std=c++20", "-E", "-dM", f"-I{tmp_path}", f"-I{REPO / 'src'}", str(tmp_path / "probe.cc")], capture_output=True, text=True)
    if "GALSF_SINGLE_STARS_FROM_IMF" in lines or "STELLA_DEFAULTS" in lines:
        assert r.returncode != 0 and NO_HYBRID in r.stderr
    else:
        assert r.returncode == 0 and NO_HYBRID not in r.stderr


@pytest.mark.parametrize("tree", ["cooling/sgchem", "gravity/ketju"])
def test_no_environment_options_or_abort(tree):
    files = [f for f in (REPO / "src" / tree).rglob("*") if f.suffix in (".cc", ".h", ".c", ".cpp", ".hpp") and "fortran_reference" not in f.parts]
    assert files
    bad = [f"{f.relative_to(REPO)}:{i + 1}" for f in files for i, l in enumerate(f.read_text(errors="replace").splitlines())
           if re.search(r"\bgetenv\s*\(|\bMPI_Abort\s*\(", l.split("//")[0])]
    assert not bad, bad
