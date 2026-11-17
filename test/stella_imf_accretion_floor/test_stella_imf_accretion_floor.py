"""stella_imf_accretion_floor: star formation in gas drained to the IMF-accretion floor creates no mass.

Uniform n_H = 200 cm^-3 box of 1 Msun cells (12^3, 4 ranks, B = 2 uG along z), Stella production model with the forced-SF
density cutoff and SF cooldown 1e-4 t_ff (~1.6 steps). nHcutoffSF and CritPhysDensity are far below the box density, so the
cells form stars every other step and the box drains its gas down to the accretion floor (1% of the merge mass). Masses are
float32 as in Gadget ICs, so drained donors reach the floor exactly. TreeDomainUpdateFrequency 2 keeps the decomposition, and
with it cell merging, off as in a deep zoom sub-step, so the drained cells stay. Stars drawn there cannot collect their mass.

Every accretion call is checked (TEST_IMF_FAILED_DRAW_RESTORE_CHECK, IMF UNDO CHECK line):
  - each gas cell ends with its mass less what it gave to kept stars, its magnetic field less the matching share, its specific
    energy and velocity unchanged (1e-12); undone draws therefore restore their donors also in steps with kept stars;
  - the grants keyed to a star add up to the mass the star collected (1e-12), and star keys are unique;
  - every donor is blocked for SF and cooled down for the SF cooldown;
  - gas plus new stars keep mass and momentum (1e-13); no star gets mass from nowhere;
and the run: total baryonic mass conserved to 1e-14, every mass-trace checkpoint to 1e-12 (gizmo.conservation_trace),
every star holds its drawn IMF mass to within 1% (SFinfo.txt).

test_many_spawns_from_one_cell: one 64 Msun cell, SF efficiency high enough that it forms ~100 stars per accretion call
(sub-event split), so star keys must stay unique past the 31 values of the ID generation counter.
"""

import os
import re
import shutil
import subprocess
import tempfile
from os.path import abspath, dirname, join

import h5py
import numpy as np
import pytest

from gizmo.conservation_trace import check_conservation_traces
from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "stella_imf_accretion_floor"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
NRANKS = 4

PC, MSUN, M_P = 3.085678e18, 1.989e33, 1.6726e-24
X_H = 0.7065  # code HYDROGEN_MASSFRAC for net 5
N_H, NG, M_CELL, Z_SUN, T_IC, B_Z = 200.0, 12, 1.0, 0.1, 0.001, 2.0e-6

# solar element mass fractions C..Zn (Asplund+09, as resolvedism_solar_massfrac)
SOLAR = np.array([2.4765351227e-03, 7.2545495820e-04, 6.0028592850e-03, 5.2841265250e-07, 1.3157897125e-03,
                  3.0605839462e-05, 7.4124358876e-04, 5.8255839166e-05, 6.9620919184e-04, 6.0990559675e-06,
                  3.2376421296e-04, 8.5877929305e-06, 7.6870616805e-05, 3.2093705781e-06, 6.7169384016e-05,
                  4.8646655463e-08, 3.2681466346e-06, 3.3215539907e-07, 1.7387424601e-05, 1.1327606908e-05,
                  1.3528499188e-03, 4.4118818564e-06, 7.4619379442e-05, 7.5396909229e-07, 1.8184871682e-06])


def metallicity(zfrac):
    """Metallicity[28] = (Z, H, He, C..Zn), the init.cc solar-scaled pattern."""
    el = SOLAR * zfrac
    he = (1 - X_H) + (0.2485 - (1 - X_H)) * zfrac
    return np.concatenate([[el.sum(), 1 - el.sum() - he, he], el])


def make_ics(fname, heavy=None):
    """12^3 lattice of 1 Msun cells; heavy = mass [Msun] of the cell nearest the box centre."""
    L = (NG**3 * M_CELL * MSUN / (N_H * M_P / X_H)) ** (1 / 3) / PC
    dx = L / NG
    g = (np.arange(NG) + 0.5) * dx
    x = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    ng = len(x)
    met = metallicity(Z_SUN)
    m = np.full(ng, M_CELL)
    if heavy:
        m[np.argmin(np.sum((x - L / 2) ** 2, axis=1))] = heavy
    with h5py.File(fname, "w") as f:
        p0 = f.create_group("PartType0")
        p0["Coordinates"] = x * 1e-3
        p0["Velocities"] = np.zeros((ng, 3))
        p0["ParticleIDs"] = np.arange(1, ng + 1, dtype=np.uint64)
        p0["Masses"] = (m * MSUN / 1.989e43).astype(np.float32)  # float32 as in Gadget ICs: drained donors reach the floor exactly
        p0["InternalEnergy"] = np.full(ng, 1.0)  # ~100 K
        p0["KernelMaxRadius"] = np.full(ng, 2.0 * dx * 1e-3)
        p0["Metallicity"] = np.tile(met, (ng, 1))
        p0["ElementAbundance"] = np.tile(met[1:], (ng, 1))
        p0["SGChemTracAbund"] = np.tile([0.0, 1.0e-4, 0.0], (ng, 1))
        p0["MagneticField"] = np.tile([0.0, 0.0, B_Z], (ng, 1))  # [G]; a snapshot start (RestartFlag 2) reads B from here, not Bini*
        h = f.create_group("Header")
        npart = np.array([ng, 0, 0, 0, 0, 0])
        h.attrs["NumPart_ThisFile"] = npart.astype(np.int32)
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = T_IC
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = L * 1e-3
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_DoublePrecision"] = 0
        h.attrs["Flag_Metals"] = len(met)
        for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Feedback"):
            h.attrs[k] = 1
    return L


def mbary_trace(path):
    """(tag, M_bary, dM_bary) per logged checkpoint of mass_balance_steps.txt."""
    rows = [ln.split() for ln in open(path) if ln.strip() and not ln.startswith("#")]
    return [(r[1], float(r[3]), float(r[4])) for r in rows]


@pytest.fixture(scope="module")
def gizmo_binary():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO; nothing to validate from external snapshots")
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME, extra_config_flags=("TEST_IMF_FAILED_DRAW_RESTORE_CHECK",))
    finally:
        os.chdir(cwd)
    return join(TEST_DIR, "GIZMO")


def run_box(binary, label, heavy=None, params_override=()):
    """Run the box; returns (run.log text, output dir, returncode)."""
    run = tempfile.mkdtemp(prefix=f"{TEST_NAME}_{label}_")
    os.makedirs(join(run, "output"))
    os.symlink(REPO, join(run, "gizmo"))  # SGCHEM and the stellar tables open their data under gizmo/
    L = make_ics(join(run, f"{TEST_NAME}_ics.hdf5"), heavy=heavy)
    params = open(join(TEST_DIR, f"{TEST_NAME}.params")).read()
    for key, value in (("BoxSize", f"{L * 1e-3:.10g}"),) + tuple(params_override):
        params, n = re.subn(rf"(?m)^{key}\s.*$", f"{key} {value}", params)
        assert n == 1, key
    open(join(run, f"{TEST_NAME}.params"), "w").write(params)
    with open(join(run, "run.log"), "w") as log:
        r = subprocess.run(mpi_launch_command(NRANKS) + [binary, f"{TEST_NAME}.params", "2"],
                           stdout=log, stderr=subprocess.STDOUT, cwd=run)
    out = join(TEST_DIR, f"output_{label}")
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(join(run, "output"), out, ignore=shutil.ignore_patterns("snapshot_*", "restartfiles"))
    shutil.copy(join(run, "run.log"), join(out, "run.log"))
    shutil.rmtree(run, ignore_errors=True)
    return open(join(out, "run.log")).read(), out, r.returncode


def undo_checks(text):
    """One dict per accretion call from the IMF UNDO CHECK lines."""
    return [{k: float(v) for k, v in re.findall(r"(\w+)=(\S+)", l)} for l in text.splitlines() if l.startswith("IMF UNDO CHECK")]


def assert_every_call_restores(checks, out):
    assert checks, f"no IMF UNDO CHECK line ({out}/run.log)"
    worst = {k: max(abs(c[k]) for c in checks) for k in checks[0]}
    mixed = sum(1 for c in checks if c["kept"] > 0 and c["failed"] > 0)
    print(f"accretion calls checked {len(checks)} (with kept and failed draws {mixed}): "
          + " ".join(f"{k}={worst[k]:.3g}" for k in ("stars", "failed", "max_per_parent", "dup_keys", "orphan_grants", "cell_dM", "cell_du",
                                                      "cell_dv", "cell_dB", "B_max", "grant_sum", "tot_dM", "tot_dp", "donors", "not_blocked", "not_cooled")))
    assert worst["B_max"] > 0, "B = 0 in every cell: the field restore is not tested"
    assert worst["dup_keys"] == 0 and worst["orphan_grants"] == 0, "two stars share a key, or a grant has no star"
    assert worst["grant_sum"] <= 1e-12, "the grants keyed to a star do not add up to the mass it collected"
    for k in ("cell_dM", "cell_du", "cell_dv", "cell_dB"):
        assert worst[k] <= 1e-12, f"{k}: a gas cell does not end with what it had less its grants to kept stars"
    assert worst["tot_dM"] <= 1e-13 and worst["tot_dp"] <= 1e-13, "gas plus new stars changed mass or momentum"
    assert worst["not_blocked"] == 0 and worst["not_cooled"] == 0, "a donor was not blocked or not cooled down"
    return mixed


def assert_mass_trace(out):
    rep = check_conservation_traces(out, require_energy=False)
    print(rep.summary())
    assert rep.ok, rep.summary()


def test_stella_imf_accretion_floor(gizmo_binary):
    text, out, rc = run_box(gizmo_binary, "floor")
    n_fallback = len(re.findall(r"IMF accretion (failed for star|found no donor mass for star)", text))
    n_failed = sum(int(x) for x in re.findall(r"RESOLVEDISM IMF: (\d+) failed draws", text))
    n_spawn = sum(int(x) for x in re.findall(r"RESOLVEDISM SFR: spawned (\d+) stars", text))
    trace = mbary_trace(join(out, "mass_balance_steps.txt"))
    m0, m1 = trace[0][1], trace[-1][1]
    d_imf = sum(d for tag, _, d in trace if tag == "post_imf_accrete")
    sf = np.loadtxt(join(out, "SFinfo.txt"), usecols=(5, 6), ndmin=2)  # M_drawn, M_acc [Msun]
    heavy = int(np.sum(sf[:, 1] > sf[:, 0] + 1e-4))
    light = int(np.sum(sf[:, 1] < 0.99 * sf[:, 0] - 1e-4))
    print(f"stars spawned {n_spawn}, failed draws undone {n_failed}, fallback stars {n_fallback}; "
          f"M_bary {m0:.16e} -> {m1:.16e} (rel {(m1 - m0) / m0:+.3e}), sum dM at post_imf_accrete {d_imf:+.3e}; "
          f"SFinfo rows {len(sf)}, M_acc > M_drawn {heavy}, M_acc < 0.99 M_drawn {light}")
    assert rc == 0, f"GIZMO exited {rc}, see {out}/run.log"
    assert n_fallback == 0, f"{n_fallback} stars were given mass without donors"
    assert abs(m1 - m0) <= 1e-14 * m0, f"baryonic mass changed by {(m1 - m0) / m0:+.3e} (IMF accretion: {d_imf:+.3e} code)"
    assert heavy == 0 and light == 0, f"{heavy} stars heavier, {light} stars more than 1% lighter than their drawn IMF mass"
    assert n_failed > 0, "no failed draw: the drained-floor case was not reached"
    mixed = assert_every_call_restores(undo_checks(text), out)
    assert mixed > 0, "no accretion call with both kept and failed draws"
    assert_mass_trace(out)


def test_many_spawns_from_one_cell(gizmo_binary):
    text, out, rc = run_box(gizmo_binary, "many", heavy=64.0, params_override=(
        ("SfEffPerFreeFall", "1.0e8"),  # pfac = 1: the 64 Msun cell forms X = 64 Msun per event (~100 stars)
        ("FacSfThreshMJ", "1.0e8"),     # Jeans test passes: SF through the rate, not only the forced cutoff path
        ("TimeMax", "0.001002")))       # ~10 steps
    assert rc == 0, f"GIZMO exited {rc}, see {out}/run.log"
    checks = undo_checks(text)
    assert_every_call_restores(checks, out)
    assert max(c["max_per_parent"] for c in checks) >= 40, "no cell formed 40 stars in one accretion call"
    assert sum(c["failed"] for c in checks) > 0, "no failed draw"
    assert_mass_trace(out)
