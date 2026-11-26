"""Shared helpers of the audit round-1 tests (test/stella_audit_round1): binaries, boxes, runs, table references.

Binaries: STELLA_AUDIT_BIN_<KIND> names a prebuilt GIZMO (the round-1 Slurm job builds each in its own copy of the tree);
without it the test builds Config_stella_default.sh (or the named base config) plus HOOKS plus the arm's flags here.
Units: kpc, 1e10 Msun, km/s. Tables: the in-tree default paths (scripts/get_stella_tables.sh --into-tree)."""

import os
import re
import shutil
import subprocess
from os.path import abspath, dirname, join

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, link_test_driver, mpi_launch_command

TEST_NAME = "stella_audit_round1"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
TABLES = os.environ.get("STELLA_AUDIT_TABLES", join(REPO, "galaxy_sf", "stellar_tables"))
RUN_TIMEOUT = 14400
PC, M_P, K_B, MSUN, YR = 3.085678e18, 1.6726e-24, 1.3807e-16, 1.989e33, 3.15576e7
UNIT_T_YR = 3.085678e21 / 1.0e5 / YR  # code time unit [yr]
# test hooks shared by every run build: isolated feedback sites, energy and mass traces, double-precision snapshots
HOOKS = ("BOX_PERIODIC", "GRAVITY_NOT_PERIODIC", "GALSF_RESOLVEDISM_ISOLATED_FB_TEST", "OUTPUT_ENERGY_STEP_TRACE",
         "OUTPUT_IN_DOUBLEPRECISION", "INPUT_IN_DOUBLEPRECISION", "HYDRO_EXACT_PAIR_FLUX_EXCHANGE", "HYDRO_TIMESTEP_FROM_PAIR_FORCE")
ROUND1_FLAGS = ("GALSF_RESOLVEDISM_EJECTA_CARRY_BIRTH_COMPOSITION", "GALSF_RESOLVEDISM_TABLES_PHASE_MATCHED_LOOKUP",
                "GALSF_RESOLVEDISM_SNAPSHOT_KEEPS_LAST_WIND_AGE")


def config_text(base="Config_stella_default.sh", drop=(), add=()):
    lines = [l for l in open(join(REPO, base)).read().splitlines() if not (l.split() and l.split()[0].split("=")[0] in drop)]
    return "\n".join(lines) + "\n# audit round-1 test\n" + "\n".join(add) + "\n"


DRIVERS = ("phase_driver", "bill_driver", "merge_split_driver")


def binary(kind, base="Config_stella_default.sh", drop=(), add=(), drivers=()):
    """Prebuilt binary STELLA_AUDIT_BIN_<KIND>, else a build of base - drop + add; drivers (stems) are linked against it."""
    env = os.environ.get("STELLA_AUDIT_BIN_" + kind.upper())
    if env:
        return env
    if os.path.exists(join(TEST_DIR, "GIZMO_" + kind)) and all(os.path.exists(join(TEST_DIR, f"{kind}_{s}")) for s in drivers):
        return join(TEST_DIR, "GIZMO_" + kind)
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO")
    open(join(TEST_DIR, "Config.sh"), "w").write(config_text(base, drop, add))
    cwd = os.getcwd(); os.chdir(REPO)
    try:
        def post():
            for s in drivers:
                src = join(TEST_DIR, s + ".cc") if os.path.exists(join(TEST_DIR, s + ".cc")) else join(REPO, "test", "merge_split_conservation", s + ".cc")
                link_test_driver(src, join(TEST_DIR, f"{kind}_{s}"))
        build_gizmo_for_test(TEST_NAME, post_build=post)
        shutil.move(join(TEST_DIR, "GIZMO"), join(TEST_DIR, "GIZMO_" + kind))
    finally:
        os.chdir(cwd)
    return join(TEST_DIR, "GIZMO_" + kind)


def driver(kind, stem):
    """Driver stem linked against the build of this kind: next to a prebuilt binary (as <stem>), else TEST_DIR/<kind>_<stem>."""
    env = os.environ.get("STELLA_AUDIT_BIN_" + kind.upper())
    return join(dirname(env), stem) if env else join(TEST_DIR, f"{kind}_{stem}")


def make_box(fname, n_H=10.0, box_pc=16.0, m_cell=1.0, T=1.0e4, X_H=0.7375, tracabund=None, double=True):
    """Uniform periodic lattice of gas; the cell nearest the centre has ID 1 (the test star site). Returns the cell count."""
    m_gas = n_H * (box_pc * PC) ** 3 * M_P / X_H / MSUN
    ng = int(round((m_gas / m_cell) ** (1 / 3))); n = ng ** 3
    box = box_pc * 1e-3
    g = (np.arange(ng) + 0.5) * box / ng
    pos = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    ids = np.arange(n, dtype=np.uint32) + 10
    ids[np.argmin(np.sum((pos - 0.5 * box) ** 2, axis=1))] = 1
    u = 1.5 * K_B * T / (1.22 * M_P) / 1e10
    ft = np.float64 if double else np.float32
    with h5py.File(fname, "w") as f:
        h = f.create_group("Header")
        h.attrs.update(BoxSize=box, NumPart_ThisFile=np.array([n, 0, 0, 0, 0, 0], dtype=np.int32), NumPart_Total=np.array([n, 0, 0, 0, 0, 0], dtype=np.uint32),
                       NumPart_Total_HighWord=np.zeros(6, dtype=np.uint32), MassTable=np.zeros(6), Time=0.0, Redshift=0.0, NumFilesPerSnapshot=1,
                       Flag_Sfr=1, Flag_Cooling=1, Flag_StellarAge=1, Flag_Metals=0, Flag_Feedback=1, Flag_DoublePrecision=int(double), HubbleParam=1.0)
        p = f.create_group("PartType0")
        p["Coordinates"] = pos.astype(ft); p["Velocities"] = np.zeros((n, 3), dtype=ft)
        p["Masses"] = np.full(n, m_gas / n * MSUN / 1.989e43, dtype=ft); p["ParticleIDs"] = ids
        p["InternalEnergy"] = np.full(n, u, dtype=ft)
        b = np.zeros((n, 3), dtype=ft); b[:, 2] = 1e-8; p["MagneticField"] = b
        if tracabund is not None:
            p["SGChemTracAbund"] = np.tile(np.asarray(tracabund, dtype=ft), (n, 1))
    return n


def write_params(d, params, base=None):
    """run.params in d: the base parameter file with params set (value None removes a tag)."""
    base = base or join(TEST_DIR, "base.params")
    keys = set(params)
    lines = [l for l in open(base).read().splitlines() if not (l.split() and l.split()[0] in keys)]
    lines += [f"{k} {v}" for k, v in params.items() if v is not None]
    open(join(d, "run.params"), "w").write("\n".join(lines) + "\n")


def run(name, exe, params, nranks=2, restart_flag=0, base=None, expect_ok=True):
    """Run exe in TEST_DIR/run_<name>; returns (exit code, log text, output dir)."""
    d = join(TEST_DIR, "run_" + name); os.makedirs(d, exist_ok=True)
    if not os.path.islink(join(d, "gizmo")):
        os.symlink(REPO, join(d, "gizmo"))
    write_params(d, params, base)
    out = join(d, "output")
    if restart_flag == 0:
        shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    with open(join(d, "run.log"), "w") as log:
        r = subprocess.run(mpi_launch_command(nranks) + [exe, "run.params"] + ([str(restart_flag)] if restart_flag else []),
                           stdout=log, stderr=subprocess.STDOUT, cwd=d, timeout=RUN_TIMEOUT)
    text = open(join(d, "run.log"), errors="replace").read()
    if expect_ok:
        assert r.returncode == 0 and "Simulation ends." in text, f"{name}: GIZMO did not reach TimeMax ({r.returncode}): " + text[-2000:]
    return r.returncode, text, out


def kv(text, tag):
    """key=value rows of the lines that start with tag."""
    rows = []
    for l in text.splitlines():
        if l.startswith(tag + " "):
            rows.append({k: v for k, v in (t.split("=", 1) for t in l.split()[1:] if "=" in t)})
    return rows


# ---- table references, read independently of the code (same storage precision: the code reads the 3-D tables as float32) ----
class Tables:
    def __init__(self):
        f = h5py.File(join(TABLES, "stellar_tables_unified.hdf5"), "r")
        w = h5py.File(join(TABLES, "stellar_wind_cumulative.hdf5"), "r")
        self.lM, self.lZ, self.la = f["log_M_init"][:], f["log_Z"][:], f["log_age_yr"][:]
        nz, nm = len(self.lZ), len(self.lM)
        self.lt = f["lifetime_yr"][:].reshape(nz, nm); self.tpms = f["t_PMS"][:].reshape(nz, nm)
        self.rt = f["remnant_type"][:].reshape(nz, nm); self.rem = f["remnant_mass"][:].reshape(nz, nm)
        self.sn = f["elem_ej_SN_mass"][:].reshape(nz, nm, -1); self.agb = f["elem_ej_AGB_mass"][:].reshape(nz, nm, -1)
        self.X_init = f["X_init"][:].reshape(nz, -1)
        self.logQ = f["log_Q_ion"][:].astype(np.float32).astype(np.float64).reshape(nz, nm, -1)
        self.vw = f["v_wind"][:].astype(np.float32).astype(np.float64).reshape(nz, nm, -1)
        self.wcum = w["elem_ej_wind_cumulative"][:].astype(np.float64).reshape(nz, nm, len(self.la), -1)

    @staticmethod
    def bsearch(grid, v):
        if v <= grid[0]:
            return 0, 0.0
        if v >= grid[-1]:
            return len(grid) - 2, 1.0
        i = int(np.searchsorted(grid, v, side="right")) - 1
        return i, (v - grid[i]) / (grid[i + 1] - grid[i])

    def cell(self, M, Z):
        iz, fz = self.bsearch(self.lZ, np.log10(Z)); im, fm = self.bsearch(self.lM, np.log10(M))
        return iz, fz, im, fm

    def node(self, M, Z):
        iz = int(np.argmin(abs(self.lZ - np.log10(Z)))); im = int(np.argmin(abs(self.lM - np.log10(M))))
        return iz, im

    def lifetime(self, M, Z):
        """t_PMS + lifetime, bilinear as stellar_t_PMS + stellar_lifetime (years)."""
        iz, fz, im, fm = self.cell(M, Z)
        b = lambda a: (1 - fz) * ((1 - fm) * a[iz, im] + fm * a[iz, im + 1]) + fz * ((1 - fm) * a[iz + 1, im] + fm * a[iz + 1, im + 1])
        return b(self.tpms) + b(self.lt), b(self.lt)

    def wind_total_at_own_death(self, iz, im):
        """per element: a grid star's cumulative wind at its own lifetime (age interpolation as the code's per-corner clamp)."""
        i, f = self.bsearch(self.la, np.log10(max(self.lt[iz, im], 100.0)))
        return (1 - f) * self.wcum[iz, im, i] + f * self.wcum[iz, im, i + 1]

    def death_fraction(self, iz, im):
        """stbl_build_death_fractions with WINDS: e_k / (sum e + rem) of the SN (or AGB for WD) table at a grid star."""
        rt = self.rt[iz, im]
        ej = np.maximum(self.agb[iz, im] if rt == 0 else self.sn[iz, im], 0)
        D = ej.sum() + (0 if rt == 5 else max(self.rem[iz, im], 0))
        e = ej / D if D > 0 else 0 * ej
        s = e.sum()
        norm = (1 / s if s > 0 else 0) if (rt == 5 or s > 1) else 1
        return e * norm

    def interp_age(self, arr, iz, im, log_age, dead_at):
        """one grid star's value at log_age, linear between the bracketing age nodes, dead nodes (<= dead_at) dropped;
        returns (sum w v, sum w) for the caller's renormalisation."""
        i, f = self.bsearch(self.la, log_age)
        vs = ws = 0.0
        for v, w in ((arr[iz, im, i], 1 - f), (arr[iz, im, i + 1], f)):
            if v > dead_at:
                vs += w * v; ws += w
        return vs, ws

    def phase_matched(self, arr, M, Z, age, dead_at, dead_value):
        """the phase-matched lookup in closed form: each of the four grid stars at age t_life,c / t_life(M, Z)"""
        iz, fz, im, fm = self.cell(M, Z)
        lt = self.lifetime(M, Z)[1]
        vs = ws = 0.0
        for dz, wz in ((0, 1 - fz), (1, fz)):
            for dm, wm in ((0, 1 - fm), (1, fm)):
                a, b = self.interp_age(arr, iz + dz, im + dm, np.log10(max(age, 100.0)) + np.log10(self.lt[iz + dz, im + dm]) - np.log10(lt), dead_at)
                vs += wz * wm * a; ws += wz * wm * b
        return vs / ws if ws > 0 else dead_value


def snapshot_sum(fname, ptype, slot=None):
    """sum of m (or m X_slot) of particle type ptype in a snapshot, code units, float64"""
    with h5py.File(fname, "r") as f:
        g = f.get(f"PartType{ptype}")
        if g is None:
            return 0.0
        m = g["Masses"][:].astype(np.float64)
        if slot is None:
            return float(m.sum())
        return float((m * g["Metallicity"][:, slot].astype(np.float64)).sum())


def last_snapshot(out):
    s = sorted(f for f in os.listdir(out) if re.match(r"snapshot_\d+\.hdf5$", f))
    return join(out, s[-1])
