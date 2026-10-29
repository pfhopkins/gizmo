"""Grackle one-zone test (COOL_GRACKLE with COOL_GRACKLE_APIVERSION=3, i.e. Grackle 3.x).

OPTIONAL: Grackle is an external library, so this test is skipped unless Grackle (>= 3.0, installed with its CMake build so
that it provides grackle.pc) and its data tables are found:
    GRACKLE_PATH=<Grackle install prefix>  (or grackle.pc on PKG_CONFIG_PATH)
    GRACKLE_DATA_DIR=<directory with CloudyData_UVB=HM2012_shielded.h5>   (the input/ directory of the Grackle repository)
The GIZMO build picks up the same GRACKLE_PATH / pkg-config (see the COOL_GRACKLE notes in the documentation).

A uniform, static, periodic gas box is N identical one-zone problems. For each Grackle configuration the test builds GIZMO,
runs a hot box (nH=10 cm^-3, 1e6 K: collisional ionization, cooling to the photoheated UV-background equilibrium) and a dense
box (nH=1e4 cm^-3, 1000 K: H2 chemistry / metal and dust cooling), and compares every snapshot with a standalone one-zone
Grackle integration that uses the same Grackle settings (onezone_reference.cc, compiled here against the same grackle.pc).
That reference is integrated with 400 sub-steps per snapshot interval; GIZMO calls Grackle once per hydro step, so the runs
use MaxSizeTimestep 1e-6 to resolve the cooling transients.

Pass, at every snapshot:
  * the box stays uniform: max/min(u) - 1 < 1e-2;
  * specific internal energy and the snapshot Temperature agree with the reference within 2%;
  * the electron abundance and every species with mass fraction >= 1e-3 in the reference agree within 5%
    (smaller trace abundances, e.g. HeIII while it recombines, are not compared).
The GIZMO run must also report "Grackle Initialized: version 3" (i.e. the API-3 code path was built and used).
"""

import os
import subprocess
from os import chdir, getcwd, path
from shutil import rmtree, which

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, run_test, default_mpi_ranks, variant_suffix
from make_grackle_onezone_ics import make_grackle_onezone_ics

TEST_NAME = "grackle_onezone"
TEST_DIR = path.join("test", TEST_NAME)
TABLE = "CloudyData_UVB=HM2012_shielded.h5"
NUM_NGB = 32  # DesNumNgb in grackle_onezone.params
TOL_U_T, TOL_SPECIES, MIN_SPECIES_FRACTION, MAX_SPREAD = 0.02, 0.05, 1e-3, 1e-2

# Grackle configurations: tabular with metals (Cloudy tables, metal + photoelectric heating), primordial H2 network,
# full network (H, He, H2, D) with metals and dust. METALS needs GALSF so that InitMetallicity sets Z at start-up
# (with METALS alone GIZMO zeroes the metallicity of a fresh start).
VARIANTS = [
    ("COOL_GRACKLE_CHEMISTRY=0", "METALS", "GALSF"),
    ("COOL_GRACKLE_CHEMISTRY=2",),
    ("COOL_GRACKLE_CHEMISTRY=3", "METALS", "GALSF"),
]
#         name     nH      T      mu (only to set u)
CASES = [("hot",   10.0,   1e6,   0.6),
         ("dense", 1e4,    1000., 1.22)]
GALSF_PARAMS = {"CritPhysDensity": "1e30", "SfEffPerFreeFall": "1", "InitMetallicity": "1"}  # no star formation; Z = 1 x solar

SPECIES = {"GrackleHI": 3, "GrackleHII": 4, "GrackleHM": 5, "GrackleHeI": 6, "GrackleHeII": 7, "GrackleHeIII": 8,
           "GrackleH2I": 9, "GrackleH2II": 10, "GrackleDI": 11, "GrackleDII": 12, "GrackleHDI": 13}
SPECIES_LEVEL = {"GrackleH2I": 2, "GrackleH2II": 2, "GrackleDI": 3, "GrackleDII": 3, "GrackleHDI": 3}


def find_grackle():
    """Return (pkg-config spec for grackle.pc, full path of the data table), or skip the module with instructions."""
    how = ("Build Grackle >= 3 with its CMake build (cmake -B build -DCMAKE_INSTALL_PREFIX=<prefix> -DCMAKE_INSTALL_LIBDIR=lib; "
           "cmake --build build; cmake --install build), then set GRACKLE_PATH=<prefix> (or put grackle.pc on PKG_CONFIG_PATH) "
           f"and GRACKLE_DATA_DIR=<grackle repository>/input (the directory with {TABLE}).")
    if which("pkg-config") is None:
        pytest.skip(f"optional Grackle test: pkg-config not found. {how}", allow_module_level=True)
    spec = None
    prefix = os.environ.get("GRACKLE_PATH")
    if prefix:
        for sub in ("lib", "lib64"):
            pc = path.join(prefix, sub, "pkgconfig", "grackle.pc")
            if path.isfile(pc):
                spec = pc
                break
    if spec is None and subprocess.run(["pkg-config", "--exists", "grackle"]).returncode == 0:
        spec = "grackle"
    if spec is None:
        pytest.skip(f"optional Grackle test: no grackle.pc found (GRACKLE_PATH={prefix}). {how}", allow_module_level=True)
    version = subprocess.run(["pkg-config", "--modversion", spec], capture_output=True, text=True).stdout.strip()
    if not version or int(version.split(".")[0]) < 3:
        pytest.skip(f"optional Grackle test: COOL_GRACKLE_APIVERSION=3 needs Grackle >= 3.0, found '{version}'. {how}",
                    allow_module_level=True)
    table = path.join(os.environ.get("GRACKLE_DATA_DIR", ""), TABLE)
    if not path.isfile(table):
        pytest.skip(f"optional Grackle test: {TABLE} not found in GRACKLE_DATA_DIR={os.environ.get('GRACKLE_DATA_DIR')}. {how}",
                    allow_module_level=True)
    return spec, path.abspath(table)


GRACKLE_PC, DATA_FILE = find_grackle()


@pytest.fixture(scope="module")
def reference_driver():
    """Compile the standalone one-zone driver against grackle.pc: this already checks the Grackle installation."""
    exe = path.abspath(path.join(TEST_DIR, "onezone_reference"))
    flags = subprocess.run(["pkg-config", "--cflags", "--libs", GRACKLE_PC], capture_output=True, text=True, check=True).stdout.split()
    cxx = os.environ.get("CXX", "c++")
    res = subprocess.run([cxx, "-O2", "-std=c++17", path.join(TEST_DIR, "onezone_reference.cc"), "-o", exe] + flags,
                         capture_output=True, text=True)
    if res.returncode != 0:
        pytest.fail(f"Could not compile/link a program against Grackle ({GRACKLE_PC}); check the Grackle installation:\n{res.stderr}")
    return exe


def read_snapshots(outdir, chem, metals):
    rows = []
    snaps = sorted(f for f in os.listdir(outdir) if f.startswith("snapshot_") and f.endswith(".hdf5"))
    assert len(snaps) >= 3, f"only {len(snaps)} snapshots in {outdir}"
    for s in snaps:
        with h5py.File(path.join(outdir, s), "r") as f:
            g = f["PartType0"]
            d = {"name": s, "t": f["Header"].attrs["Time"], "u": g["InternalEnergy"][:], "T": g["Temperature"][:],
                 "rho": g["Density"][:], "h": g["KernelMaxRadius"][:], "Z": 0.02}
            if metals:
                z = g["Metallicity"][:]
                d["Z"] = float(np.median(z if z.ndim == 1 else z[:, 0]))
            if chem > 0:
                d["ne"] = g["ElectronAbundance"][:]
                for k in SPECIES:
                    if SPECIES_LEVEL.get(k, 1) <= chem:
                        d[k] = g[k][:]
        rows.append(d)
    return rows


def run_reference(exe, rows, chem, metals):
    first = rows[0]
    with h5py.File(DATA_FILE, "r") as t:  # same rule as InitGrackle
        ssm = 3 if ("UVBRates" in t and "CrossSections" in t["UVBRates"]) else 0
    shielding_length = np.median(first["h"]) / NUM_NGB ** (1.0 / 3.0)
    args = [exe, DATA_FILE, str(chem), str(metals), repr(float(np.median(first["rho"]))), repr(float(np.median(first["u"]))),
            repr(first["Z"]), repr(float(rows[-1]["t"])), str(len(rows) - 1), "400", repr(float(shielding_length)), str(ssm)]
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    return np.loadtxt(out.splitlines(), ndmin=2)


def compare(rows, ref, chem):
    """Return a list of failure messages (empty = pass)."""
    failures = []
    for d, r in zip(rows, ref):
        where = f"{d['name']} (t={d['t']:.4g})"
        if not np.isclose(r[0], d["t"], rtol=1e-3, atol=1e-12):
            failures.append(f"{where}: reference time {r[0]:.6g}")
        spread = d["u"].max() / d["u"].min() - 1
        if spread > MAX_SPREAD:
            failures.append(f"{where}: box not uniform, max/min(u)-1 = {spread:.3g}")
        for label, value, refval in (("u", np.median(d["u"]), r[1]), ("Temperature", np.median(d["T"]), r[14])):
            if abs(value / refval - 1) > TOL_U_T:
                failures.append(f"{where}: {label} {value:.6g} vs reference {refval:.6g}")
        if chem > 0:
            for key, col in [("ne", 2)] + [(k, c) for k, c in SPECIES.items() if k in d]:
                value, refval = np.median(d[key]), r[col]
                if refval >= MIN_SPECIES_FRACTION and abs(value / refval - 1) > TOL_SPECIES:
                    failures.append(f"{where}: {key} {value:.4g} vs reference {refval:.4g}")
    return failures


@pytest.mark.parametrize("flags", VARIANTS, ids=lambda f: "_".join(f).replace("COOL_GRACKLE_CHEMISTRY=", "chem"))
def test_grackle_onezone(flags, reference_driver):
    chem = int(next(f for f in flags if f.startswith("COOL_GRACKLE_CHEMISTRY=")).split("=")[1])
    metals = int("METALS" in flags)
    build_gizmo_for_test(TEST_NAME, extra_config_flags=flags)
    failures = []
    cwd = getcwd()
    try:
        chdir(TEST_DIR)
        for name, nH, temperature, mu in CASES:
            make_grackle_onezone_ics(f"{TEST_NAME}_{name}_ics.hdf5", nH=nH, temperature=temperature, mu=mu)
            outdir = f"output{variant_suffix(flags)}_{name}"
            if path.isdir(outdir):
                rmtree(outdir)
            overrides = {"InitCondFile": f"{TEST_NAME}_{name}_ics", "OutputDir": outdir, "GrackleDataFile": DATA_FILE}
            if "GALSF" in flags:
                overrides.update(GALSF_PARAMS)
            run_test(TEST_NAME, num_mpi_ranks=default_mpi_ranks(max_ranks=4), param_overrides=overrides)
            with open(f"test_{TEST_NAME}.out", errors="ignore") as f:
                assert "Grackle Initialized: version 3" in f.read(), "GIZMO did not initialize Grackle 3 (COOL_GRACKLE_APIVERSION=3)"
            rows = read_snapshots(outdir, chem, metals)
            ref = run_reference(reference_driver, rows, chem, metals)
            failures += [f"{name}: {m}" for m in compare(rows, ref, chem)]
    finally:
        chdir(cwd)
    assert not failures, f"{len(failures)} mismatches vs the standalone Grackle reference:\n" + "\n".join(failures[:40])
