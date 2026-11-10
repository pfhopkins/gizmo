"""single_stars_radpressure_cells: with band fluxes the direct radiation pressure goes per gas cell from the pixel fluxes; the IR boost stays at the star. One 40 Msun star in a uniform 16^3 lattice of 1 Msun
cells, chemistry on, gravity on (the fluxes come from the gravity walk), 2 kyr.
  test_thin: n_H = 1, box optically thin: momentum per step = sum over cells of sum_b sigma_b L_b DGR N_H(cell) / (4 pi r^2 c),
    r floored at the softening (3 %), plus the IR term at the star.
  test_thick: n_H = 1e4, photon cap active: momentum per step = L_emit / c + IR term (3 %). Disjoint bands and
    GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE (: LW and 6-8 eV counted twice in deposit and cap).
  test_snapshot_fields: OUTPUT_SGCHEM_RADIATION_FIELDS: G0_FUV = L(6-13.6 eV) / (4 pi c r^2) / Habing (3 %, median over cells beyond 3 softenings),
    fields in shape, star lums in log10 erg/s and the Lyman photon rate in log10 1/s from the tables (1e-3 dex).
  test_outward: the gas gains positive radial momentum about the star, at least half of what the log says (pixel centres, n_H = 1).
  test_no_extra_collectives: MPI collectives counted on rank 0 (test/ketju_quiet_steps interposer) equal those with the direct
    term at the star (GALSF_FB_RADPRESSURE_DIRECT_TERM_AT_STAR), with no star (quiet steps) and with the star shining.
  test_radial_shells_quiet_steps: no star: GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS adds no MPI collective.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command, mpirun_is_open_mpi

NAME = "single_stars_radpressure_cells"
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice, YR_PER_TIME_UNIT  # noqa: E402
from stella_table_bands import Tables  # noqa: E402

BUILDS = {}
M_STAR, F_LIFE, NGRID, Z = 40.0, 0.3, 16, 0.0142
MSUN, PC, MP, C = 1.989e33, 3.085678e18, 1.6726e-24, 2.9979e10
SIG = (2.0e-21, 1.3e-21, 0.5e-21)
HABING = 5.257e-14  # HABING_EGY_DENSITY_6_13P6EV_CGS


@pytest.fixture(scope="module")
def tables():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(HERE))
    return t


def build(tag, extra=()):
    if tag not in BUILDS:
        build_gizmo_for_test(NAME, extra_config_flags=tuple(extra))
        shutil.move(HERE / "GIZMO", HERE / f"GIZMO_{tag}")
        BUILDS[tag] = HERE / f"GIZMO_{tag}"
    return BUILDS[tag]


def run(exe, tag, tables, n_h, star_on=True, preload=None):
    rho = n_h * MP / 0.7065 * PC ** 3 / MSUN  # Msun / pc^3, HYDROGEN_MASSFRAC under SGChem
    dx = (1.0 / rho) ** (1.0 / 3.0); L = NGRID * dx
    tab = Tables(tables["StellarTablesFile"])
    lZ = np.log10(Z)
    age = tab._i2(tab.t_pms, np.log10(M_STAR), lZ) + F_LIFE * tab._i2(tab.life, np.log10(M_STAR), lZ)
    d = HERE / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    star = np.full(3, L / 2 + 0.31 * dx)
    make_treecol_pixels_ics(str(d / "output" / "snapshot_000.hdf5"), L, lattice(NGRID, L), [(M_STAR, age, star)] if star_on else [], hsml=2 * dx)
    for f in ("TREECOOL", "spcool_tables"):
        if (HERE / f).exists():
            os.symlink(HERE / f, d / f)
    T = tables["StellarTablesFile"]
    p = (HERE / f"{NAME}.params").read_text()
    p = p.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    over = {"BoxSize": L, "TimeMax": 2e-3, "MaxSizeTimestep": 1e-4, "DesNumNgb": 32}
    p = "\n".join([l for l in p.splitlines() if not (l.split() and l.split()[0] in over)] + [f"{k:27s} {v}" for k, v in over.items()]) + "\n"
    (d / f"{NAME}.params").write_text(p)
    with open(d / f"test_{NAME}.out", "w") as out:
        env = {**os.environ, "OMP_NUM_THREADS": "1", **({"LD_PRELOAD": str(preload)} if preload else {})}
        fwd = ["-x", "LD_PRELOAD"] if preload and mpirun_is_open_mpi() else []
        subprocess.run(mpi_launch_command(2) + fwd + [str(exe), f"{NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env=env, check=True)
    rows = np.atleast_2d(np.loadtxt(d / "output" / "MomWinds.txt"))
    return d, rows, tab, age, rho, L, star, dx


def per_time(rows):
    """momentum offered per code time over the steps after the first"""
    dt = np.diff(rows[:, 0]); ok = dt > 0
    return (rows[:-1][ok, 5] / dt[ok])[2:]


def bands(tab, age):
    return tab.bands(M_STAR, Z, age)  # PE, LW, 6-8, NUV 3.444-6, OPT


def sigma_lum(L, twice):
    return (SIG[0] * (L[0] + L[2] + (L[1] if twice else 0)), SIG[1] * (L[3] + (L[2] if twice else 0)), SIG[2] * L[4])


def ir_term(L, rho, twice):
    h = (3 * 64 / (4 * np.pi * rho)) ** (1.0 / 3.0) * PC
    rho_cgs = rho * MSUN / PC ** 3
    NH, Sigma, dgr = rho_cgs * 0.7065 / MP * h, rho_cgs * h, Z / 0.014
    f = [1 - np.exp(-s * dgr * NH) for s in SIG]
    L_abs = f[0] * (L[0] + L[2]) + f[1] * (L[3] + (L[2] if twice else 0)) + f[2] * L[4]
    return L_abs * 5.0 * dgr * Sigma / C * (PC / 1e5) / (MSUN * 1e5)


def to_mom(l_cgs):
    return l_cgs / C * (PC / 1e5) / (MSUN * 1e5)  # erg/s to Msun km/s per code time


def test_thin(tables):
    d, rows, tab, age, rho, L, star, dx = run(build("disjoint"), "thin", tables, 1.0)
    got = per_time(rows)
    p = lattice(NGRID, L); r2 = ((p - star) ** 2).sum(1)
    Lb = bands(tab, age); NH = MSUN * 0.7065 / MP; dgr = Z / 0.014
    want = to_mom(sum(sigma_lum(Lb, False)) * dgr * NH * (1 / (4 * np.pi * r2 * PC ** 2)).sum()) + ir_term(Lb, rho, False)
    print(f"thin: {np.median(got):.4g} expected {want:.4g}")
    assert abs(np.median(got) / want - 1) < 0.03


@pytest.mark.parametrize("tag,flags,twice", [("disjoint", (), False), ("twice", ("GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE",), True)])
def test_thick(tables, tag, flags, twice):
    d, rows, tab, age, rho, L, star, dx = run(build(tag, flags), f"thick_{tag}", tables, 1.0e4)
    Lb = bands(tab, age)
    L_emit = Lb[0] + Lb[2] + Lb[3] + Lb[4] + ((Lb[1] + Lb[2]) if twice else 0)
    got = np.median(per_time(rows)); want = to_mom(L_emit) + ir_term(Lb, rho, twice)
    print(f"thick {tag}: {got:.4g} expected {want:.4g}")
    assert abs(got / want - 1) < 0.03


def test_snapshot_fields(tables):
    d, rows, tab, age, rho, L, star, dx = run(build("fields", ("OUTPUT_SGCHEM_RADIATION_FIELDS",)), "fields", tables, 1.0)
    with h5py.File(sorted((d / "output").glob("snapshot_*.hdf5"))[-1]) as F:
        g = F["PartType0"]; s = F["PartType4"]; x = g["Coordinates"][:] - s["Coordinates"][0]
        fields = {k: g[k][:] for k in ("G0_FUV", "G0_LW", "G0_FUV_atten", "ShieldFac_Dust", "G0_NUV", "G0_OPT", "CR_IonizationRate")}
        luv, llw, lq = s["UV_Luminosity"][0], s["LW_Luminosity"][0], s["Lyman_PhotonRate"][0]; t = F["Header"].attrs["Time"]
    assert all(len(v) == len(x) for v in fields.values())
    r = np.sqrt((x ** 2).sum(1)); far = r > 3 * dx
    Lb = bands(tab, age + t * YR_PER_TIME_UNIT)
    want = (Lb[0] + Lb[2]) / (4 * np.pi * C * (r[far] * PC) ** 2) / HABING
    print(f"G0_FUV / analytic median {np.median(fields['G0_FUV'][far] / want):.4f}")
    assert abs(np.median(fields["G0_FUV"][far] / want) - 1) < 0.03
    assert abs(luv - np.log10(Lb[0] + Lb[2])) < 1e-3 and abs(llw - np.log10(Lb[1])) < 1e-3
    assert abs(lq - np.log10(tab.ionising(M_STAR, Z, age + t * YR_PER_TIME_UNIT)[1])) < 1e-3  # log10 photons/s from the tables


def test_outward(tables):
    d, rows, tab, age, rho, L, star, dx = run(build("disjoint"), "outward", tables, 1.0)
    with h5py.File(sorted((d / "output").glob("snapshot_*.hdf5"))[-1]) as F:
        g = F["PartType0"]; x = g["Coordinates"][:] - F["PartType4/Coordinates"][0]; v = g["Velocities"][:]; m = g["Masses"][:]
    x = (x + L / 2) % L - L / 2
    pr = (m * (v * x).sum(1) / np.sqrt((x ** 2).sum(1))).sum()
    print(f"radial momentum {pr:.4g}, logged {rows[:, 6].sum():.4g}")
    assert pr > 0.5 * rows[:, 6].sum(), "pushed away from the star"


def collectives(builds, star_on, tables):
    """(MPI collectives on rank 0, steps) per build, counted with the test/ketju_quiet_steps interposer"""
    so = HERE / "mpi_collective_count.so"
    subprocess.run(["mpicc", "-shared", "-fPIC", "-O1", str(HERE.parent / "ketju_quiet_steps" / "mpi_collective_count.c"), "-o", str(so)], check=True)
    got = {}
    for tag, flags in builds:
        d = run(build(tag, flags), f"collectives_{tag}_{int(star_on)}", tables, 1.0, star_on, so)[0]
        out = (d / f"test_{NAME}.out").read_text()
        got[tag] = (int(re.search(r"MPI_COLLECTIVES total (\d+)", out).group(1)), len(re.findall(r"^Sync-Point", out, re.M)))
    print(f"collectives (total, steps): {got}")
    return list(got.values())


@pytest.mark.parametrize("star_on", [False, True], ids=["quiet", "shining"])
def test_no_extra_collectives(tables, star_on):
    a, b = collectives((("disjoint", ()), ("at_star", ("GALSF_FB_RADPRESSURE_DIRECT_TERM_AT_STAR",))), star_on, tables)
    assert a == b


def test_radial_shells_quiet_steps(tables):
    a, b = collectives((("disjoint", ()), ("shells", ("GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS=12",))), False, tables)
    assert a == b
