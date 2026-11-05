"""SGChem in GIZMO on a uniform periodic gas box (8^3 cells, n_H = 100 cm^-3, T0 = 8000 K, solar metals).

test_sgchem_startup: the installed chemistry tables are checked at start-up; with the right tables the run reaches
TimeMax, with a changed table it stops with a message naming the install script.
test_sgchem_cell_state: SGCHEM network 5 adds 40 B to the gas cell (three tracked species, dust and gas temperature).
test_sgchem_snapshot_io: start values in snapshot 0; SGChemTracAbund and SGChemDustTemp equal the standalone run; a
restart from a snapshot whose datasets carry the old Chemcool* names reads them and continues as the standalone run.
test_sgchem_restart: a run stopped at half time and continued from its restart files ends identical to the uninterrupted run.
test_sgchem_habing_field: a uniform Habing-shaped field at G0 = 1, 10, 100 gives the inputs of stella_dev's production
path, and the cells equal the standalone run on them.
test_sgchem_ledger: ledger.txt rows (chemistry heating and cooling) at every snapshot close the box's thermal energy
budget and are identical with 1 and 4 threads; test_sgchem_restart also checks they survive a restart.
test_sgchem_cooling: the cells cool through SGChem: final energy, temperature, electrons and HI equal a standalone run of
the backend on the same inputs and timesteps, and the snapshots are identical with 1 and 4 OpenMP threads.
"""

import glob
import re
import shutil
import subprocess
import sys
from os import chdir, environ, path, makedirs

import h5py
import numpy as np
import pytest

from gizmo.test import (build_and_run_test, build_gizmo_for_test, run_test, get_cooling_tables, default_omp_threads,
                        mpi_launch_command, write_params_with_overrides)

HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(path.dirname(HERE))
NAME = "sgchem_box"


def chemistry_tables_dir():
    """SGCHEM_TABLES_DIR, else the install of scripts/get_stella_tables.sh (built from stella-tables, hash-checked)."""
    if environ.get("SGCHEM_TABLES_DIR"):
        return environ["SGCHEM_TABLES_DIR"]
    out = subprocess.run(["bash", path.join(ROOT, "scripts", "get_stella_tables.sh")], capture_output=True, text=True,
                         env={**environ, "PYTHON": environ.get("PYTHON", sys.executable)})
    assert out.returncode == 0, f"get_stella_tables.sh failed:\n{out.stdout}\n{out.stderr}"
    return [l for l in out.stdout.splitlines() if " holds " in l][-1].split()[1]


@pytest.fixture(scope="module")
def tables():
    if not environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    get_cooling_tables(HERE)
    sys.path.insert(0, HERE)
    from make_sgchem_box_ics import make_ics
    make_ics(path.join(HERE, "sgchem_box_ics.hdf5"))
    return chemistry_tables_dir()


def cell_size(log):
    return int(re.search(r"Size of hydro-cell structure\s+(\d+)", open(log).read()).group(1))


@pytest.fixture(scope="module")
def box(tables):
    """The same box built without SGCHEM (output_no_sgchem/, built first: every build removes test/*/GIZMO), then with it."""
    var = path.join(HERE, "output_no_sgchem")
    makedirs(var, exist_ok=True)
    with open(path.join(var, "Config.sh"), "w") as f:
        f.writelines(l for l in open(path.join(HERE, "Config.sh")) if not l.startswith(("SGCHEM", "CHEMISTRYNETWORK")))
    for f in ("sgchem_box_ics.hdf5", "sgchem_box.params", "TREECOOL", "spcool_tables"):
        if not path.exists(path.join(var, f)):
            shutil.copytree(path.join(HERE, f), path.join(var, f)) if path.isdir(path.join(HERE, f)) else shutil.copy(path.join(HERE, f), var)
    build_gizmo_for_test(f"{NAME}/output_no_sgchem", default_omp_threads())
    chdir(var)
    try:
        run_test("no_sgchem", 2, default_omp_threads(), params_name=NAME)
    finally:
        chdir(ROOT)
    build_and_run_test(NAME, 2, default_omp_threads(), param_overrides={"SGChemTablesPath": tables})
    return {"no_sgchem": path.join(var, "test_no_sgchem.out"), "sgchem": path.join(HERE, f"test_{NAME}.out")}


def test_sgchem_cell_state(box):
    grow = cell_size(box["sgchem"]) - cell_size(box["no_sgchem"])
    assert grow == 8 * (3 + 2), f"gas cell grows by {grow} B with SGCHEM network 5"


def standalone(tables, snap0, times, start=None, field=(0.0, 0.0, 0.0), stella_dev_inputs=False, over={}):
    """cell_standalone.cc built in the box build's src/ (same config), fed the cells' density and energy and the run's timesteps."""
    src = path.join(ROOT, "src")
    cc = subprocess.run("make -n -W cooling/sgchem/sgchem_cell.cc cooling/sgchem/sgchem_cell.o", shell=True, cwd=src,
                        capture_output=True, text=True).stdout
    cc = [l for l in cc.splitlines() if "sgchem_cell.cc" in l][-1]
    ln = [l for l in subprocess.run("make -n -W core/main.cc GIZMO", shell=True, cwd=src, capture_output=True,
                                    text=True).stdout.splitlines() if " -o GIZMO" in l][-1]
    exe = path.join(HERE, "output", "cell_standalone")
    obj = exe + ".o"
    cc = cc.replace("cooling/sgchem/sgchem_cell.cc", path.join(HERE, "cell_standalone.cc")).replace(
        "cooling/sgchem/sgchem_cell.o", obj) + " -Icooling/sgchem"
    subprocess.run(cc, shell=True, cwd=src, check=True)
    toks = ln.split()
    libs = " ".join(t for t in toks[1:] if not t.endswith(".o") and t not in ("-o", "GIZMO"))
    subprocess.run(f"{toks[0]} {obj} cooling/sgchem/cxx/*.o {libs} -o {exe}", shell=True, cwd=src, check=True)
    p = {k: float(v) for k, v in (l.split()[:2] for l in open(path.join(HERE, f"{NAME}.params")) if l.strip() and l[0] != "%"
                                  if len(l.split()) > 1 and l.split()[1][0] in "0123456789.-")}
    p.update(over)
    g = snap0["PartType0"]
    q = np.diff(times)[1:] / p["TimeMax"]  # no cooling at the first sync point (zero timestep); the steps are power-of-two
    dts = p["TimeMax"] * np.exp2(np.round(np.log2(q)))  # fractions of TimeMax: rebuild them exactly
    assert np.allclose(dts, np.diff(times)[1:], rtol=1e-12, atol=0)
    lines = [f"{p['UnitMass_in_g']!r} {p['UnitLength_in_cm']!r} {p['UnitVelocity_in_cm_per_s']!r} {p['HubbleParam']!r} "
             f"{p['SGChemCosmicRayIonRate']!r} {p['SGChemDustToGasRatio']!r} {p['SGChemDeuteriumAbundance']!r} {p['MinGasTemp']!r} "
             + " ".join(repr(float(x)) for x in field) + f" {int(stella_dev_inputs)} "
             f"{len(dts)} " + " ".join(repr(float(d)) for d in dts), " ".join(repr(float(z)) for z in g["Metallicity"][0])]
    if start is None:  # fresh start: the start values of the cells
        start = np.tile([10.0, 0.0, 1.0e-4, 0.0], (len(g["ParticleIDs"]), 1))
    lines += [f"{int(i)} {float(r)!r} {float(u)!r} " + " ".join(repr(float(x)) for x in s)
              for i, r, u, s in zip(g["ParticleIDs"][:], g["Density"][:], g["InternalEnergy"][:], start)]
    out = subprocess.run([exe, tables], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True).stdout
    res = {int(l.split()[0]): [float(x) for x in l.split()[1:]] for l in out.splitlines() if not l.startswith("FIELD")}
    assert len(res) == len(g["ParticleIDs"]), out[-2000:]
    res["field"] = [float(x) for x in out.splitlines()[0].split()[1:]]
    return res


def step_times(cpu_txt):
    return np.array([float(m) for m in re.findall(r"^Step \d+, Time: (\S+),", open(cpu_txt).read(), re.M)])


def test_sgchem_cooling(tables, box):
    out = path.join(HERE, "output")
    with h5py.File(path.join(out, "snapshot_000.hdf5")) as s0, h5py.File(path.join(out, "snapshot_002.hdf5")) as s2:
        ref = standalone(tables, s0, step_times(path.join(out, "cpu.txt")))
        g = s2["PartType0"]
        ids = [int(i) for i in g["ParticleIDs"][:]]
        got = {"u": g["InternalEnergy"][:], "T": g["Temperature"][:], "Ne": g["ElectronAbundance"][:], "HI": g["NeutralHydrogenAbundance"][:]}
    want = {"u": [ref[i][0] for i in ids], "T": [ref[i][1] for i in ids], "Ne": [ref[i][2] for i in ids], "HI": [1 - ref[i][4] for i in ids]}
    dev = {k: float(np.max(np.abs(got[k] / np.array(want[k]) - 1))) for k in got}
    print(f"box vs standalone, max relative deviation: {dev}; T: {np.min(got['T']):.6g}-{np.max(got['T']):.6g} K, "
          f"x(H2) = {ref[ids[0]][3]:.6g}, x(CO) = {ref[ids[0]][5]:.6g}, T_dust = {ref[ids[0]][6]:.6g} K")
    assert max(dev.values()) < 1e-8 and np.max(got["T"]) < 0.01 * 8000  # cell densities differ from snapshot 0 at 1e-10

    snaps = {}
    for nt in (1, 4):  # same executable and inputs, 1 and 4 OpenMP threads per rank
        chdir(HERE)
        try:
            run_test(NAME, 2, nt, param_overrides={"SGChemTablesPath": tables})
        finally:
            chdir(ROOT)
        with h5py.File(path.join(out, "snapshot_002.hdf5")) as f:
            o = np.argsort(f["PartType0/ParticleIDs"][:])
            snaps[nt] = {k: f["PartType0"][k][:][o] for k in ("InternalEnergy", "Temperature", "ElectronAbundance", "NeutralHydrogenAbundance")}
    for k in snaps[1]:
        assert np.array_equal(snaps[1][k], snaps[4][k]), f"{k} differs between 1 and 4 threads"


def test_sgchem_snapshot_io(tables, box):
    out = path.join(HERE, "output")
    with h5py.File(path.join(out, "snapshot_000.hdf5")) as s0:
        assert np.all(s0["PartType0/SGChemTracAbund"][:] == [0.0, 1.0e-4, 0.0]) and np.all(s0["PartType0/SGChemDustTemp"][:] == 10.0)
    with h5py.File(path.join(out, "snapshot_000.hdf5")) as s0, h5py.File(path.join(out, "snapshot_002.hdf5")) as s2:
        ref = standalone(tables, s0, step_times(path.join(out, "cpu.txt")))
        g = s2["PartType0"]
        want = np.array([[ref[int(i)][k] for k in (3, 4, 5, 6)] for i in g["ParticleIDs"][:]])
        got = np.column_stack([g["SGChemTracAbund"][:], g["SGChemDustTemp"][:]])
    dev = np.max(np.abs(got / want - 1), axis=0)
    print(f"x(H2), x(H+), x(CO), T_dust vs standalone, max relative deviation: {dev}")
    assert np.all(dev < 1e-8)

    # restart from snapshot 1 with the datasets under their old names: the state is read back
    rs = path.join(HERE, "output_restart")
    shutil.rmtree(rs, ignore_errors=True)
    makedirs(rs)
    snap1 = path.join(rs, "snapshot_001.hdf5")
    shutil.copy(path.join(out, "snapshot_001.hdf5"), snap1)
    with h5py.File(snap1, "r+") as f:
        for new, old in (("SGChemTracAbund", "ChemcoolTracAbund"), ("SGChemDustTemp", "ChemcoolDustTemp")):
            f["PartType0"].move(new, old)
    chdir(HERE)
    try:
        params = write_params_with_overrides(f"{NAME}.params", {"SGChemTablesPath": tables, "OutputDir": "output_restart"})
        with open(path.join(rs, "test.out"), "w") as o:
            subprocess.run(mpi_launch_command(2, default_omp_threads()) + ["./GIZMO", params, "2", "1"], stdout=o, stderr=o, check=True)
    finally:
        chdir(ROOT)
    assert "reading legacy-named 'ChemcoolTracAbund'" in open(path.join(rs, "test.out")).read()
    with h5py.File(snap1) as s1, h5py.File(path.join(rs, "snapshot_002.hdf5")) as s2:
        g1 = s1["PartType0"]
        st = dict(zip(g1["ParticleIDs"][:], np.column_stack([g1["ChemcoolDustTemp"][:], g1["ChemcoolTracAbund"][:]])))
        ref = standalone(tables, s1, step_times(path.join(rs, "cpu.txt")), [st[i] for i in g1["ParticleIDs"][:]])
        g = s2["PartType0"]
        want = np.array([[ref[int(i)][k] for k in (0, 3, 4, 5, 6)] for i in g["ParticleIDs"][:]])
        got = np.column_stack([g["InternalEnergy"][:], g["SGChemTracAbund"][:], g["SGChemDustTemp"][:]])
    dev = np.max(np.abs(got / want - 1), axis=0)
    print(f"restart from old names: u, x(H2), x(H+), x(CO), T_dust vs standalone, max relative deviation: {dev}")
    assert np.all(dev < 1e-8)


def test_sgchem_restart(tables, box):
    """stop at t = 0.064 (restart files written at TimeMax), continue to 0.128 from them: identical to the uninterrupted run"""
    shutil.rmtree(path.join(HERE, "output_restart1"), ignore_errors=True)
    chdir(HERE)
    try:
        run_test(NAME, 2, default_omp_threads(), param_overrides={"SGChemTablesPath": tables, "OutputDir": "output_restart1", "TimeMax": 0.064})
        params = write_params_with_overrides(f"{NAME}.params", {"SGChemTablesPath": tables, "OutputDir": "output_restart1"})
        with open(path.join(HERE, "output_restart1", "test.out"), "w") as o:
            subprocess.run(mpi_launch_command(2, default_omp_threads()) + ["./GIZMO", params, "1"], stdout=o, stderr=o, check=True)
    finally:
        chdir(ROOT)
    fields = ("InternalEnergy", "Temperature", "ElectronAbundance", "NeutralHydrogenAbundance", "SGChemTracAbund", "SGChemDustTemp")
    snaps = []
    for d in ("output", "output_restart1"):
        with h5py.File(path.join(HERE, d, "snapshot_002.hdf5")) as f:
            assert f["Header"].attrs["Time"] == 0.128
            o = np.argsort(f["PartType0/ParticleIDs"][:])
            snaps.append({k: f["PartType0"][k][:][o] for k in fields})
    for k in fields:
        assert np.array_equal(snaps[0][k], snaps[1][k]), f"{k} differs after the restart"
    led = [np.loadtxt(path.join(HERE, d, "ledger.txt"))[-1] for d in ("output", "output_restart1")]
    assert np.array_equal(led[0], led[1]), "ledger totals differ after the restart"


@pytest.mark.parametrize("g0", [1.0, 10.0, 100.0])
def test_sgchem_habing_field(tables, box, g0):
    """uniform Habing-shaped field: inputs as stella_dev's production path (12 pixels, floors, NUV/optical grain heating),
    and the box cells equal the standalone run on those inputs"""
    field = (1.7, g0, g0)  # SGChemBackgroundG0, SGChemG0, SGChemG0_LW
    out = path.join(HERE, f"output_g0_{g0:g}")
    shutil.rmtree(out, ignore_errors=True)
    chdir(HERE)
    try:
        run_test(NAME, 2, default_omp_threads(), param_overrides={"SGChemTablesPath": tables, "OutputDir": path.basename(out),
                 "SGChemBackgroundG0": field[0], "SGChemG0": field[1], "SGChemG0_LW": field[2]})
    finally:
        chdir(ROOT)
    with h5py.File(path.join(out, "snapshot_000.hdf5")) as s0, h5py.File(path.join(out, "snapshot_002.hdf5")) as s2:
        times = step_times(path.join(out, "cpu.txt"))
        ref = standalone(tables, s0, times, field=field)
        sd = standalone(tables, s0, times, field=field, stella_dev_inputs=True)
        g = s2["PartType0"]
        ids = [int(i) for i in g["ParticleIDs"][:]]
        got = np.column_stack([g["InternalEnergy"][:], g["Temperature"][:], g["SGChemTracAbund"][:], g["SGChemDustTemp"][:]])
        spread = g["Density"][:].max() / g["Density"][:].min() - 1  # the warm lattice moves: cell densities scatter
    want = np.array([[ref[i][k] for k in (0, 1, 3, 4, 5, 6)] for i in ids])
    one_zone = lambda r: np.array([r[ids[0]][k] for k in (0, 1, 3, 4, 5, 6)])
    f = np.array(ref["field"])
    din = np.abs(f[3:] / f[:3] - 1)
    dev = np.max(np.abs(got / want - 1), axis=0)
    dsd = np.abs(one_zone(ref) / one_zone(sd) - 1)
    print(f"G0 = {g0:g}: inputs (G0, G0_LW, G0_dust) stella_dev path {f[:3]}, this tree {f[3:]}, relative {din}; "
          f"cells vs standalone (u, T, H2, H+, CO, T_dust) {dev} (density spread {spread:.2g}); one-zone on this tree's vs stella_dev's inputs {dsd}; "
          f"T = {got[0, 1]:.6g} K, x(H2) = {got[0, 2]:.4g}, x(CO) = {got[0, 4]:.4g}, T_dust = {got[0, 5]:.4g} K")
    assert np.all(din < 1e-14) and np.all(dev < 1e-6) and np.all(dsd < 1e-5)


def test_sgchem_ledger(tables, box):
    """chemistry heating/cooling rows: one row per snapshot; they close the thermal energy budget of the box at rest;
    identical with 1 and 4 threads (exact per-thread sums)"""
    led = {}
    for nt in (1, 4):
        d = f"output_ledger_{nt}"
        shutil.rmtree(path.join(HERE, d), ignore_errors=True)
        chdir(HERE)
        try:
            run_test(NAME, 2, nt, param_overrides={"SGChemTablesPath": tables, "OutputDir": d, "SGChemG0": 10.0, "SGChemG0_LW": 10.0,
                                                   "SGChemBackgroundG0": 1.7})
        finally:
            chdir(ROOT)
        led[nt] = open(path.join(HERE, d, "ledger.txt")).read()
    assert led[1] == led[4], "ledger differs between 1 and 4 threads"
    rows = np.loadtxt(path.join(HERE, "output_ledger_4", "ledger.txt"))
    e = []
    for k in range(3):
        with h5py.File(path.join(HERE, "output_ledger_4", f"snapshot_00{k}.hdf5")) as f:
            assert f["Header"].attrs["Time"] == rows[k, 0]
            g = f["PartType0"]
            e.append(np.sum(g["Masses"][:] * g["InternalEnergy"][:]) * 1.989e33 * 1e10)  # erg (box units)
    assert np.all(rows[0, 1:] == 0)
    for k in (1, 2):
        net = rows[k, 1] - rows[k, 2]
        print(f"t = {rows[k, 0]:g}: heating {rows[k, 1]:.6e} erg, cooling {rows[k, 2]:.6e} erg, net {net:.6e}, "
              f"thermal energy change {e[k] - e[0]:.6e} erg, relative residual {(e[k] - e[0]) / net - 1:.2e}")
        assert abs((e[k] - e[0]) / net - 1) < 1e-6


def test_sgchem_startup(tables, box):
    log = open(box["sgchem"]).read()
    assert "SGCHEM: chemistry tables built" in log
    # the energy floor of MinGasTemp = 2.73 K: atomic gas with X_H = 0.7065 (ABHE = 0.1), 3/2 k T per particle
    xh, floor = re.search(r"SGCHEM: hydrogen mass fraction (\S+), MinEgySpec (\S+)", log).groups()
    mu = 4.0 / (1.0 + 3.0 * 0.7065)
    assert float(xh) == 0.7065 and float(floor) == pytest.approx(1.5 * 1.3806e-16 * 2.73 / (mu * 1.6726e-24) / 1.0e10, rel=1e-3)

    bad = path.join(HERE, "output_tables_changed")
    makedirs(bad, exist_ok=True)
    for f in ("energy-table-x.txt", "eh2-table.txt", "H2-cooling-ratios.dat", "wss_z_collis.txt"):
        shutil.copy(path.join(tables, f), bad)
    with open(path.join(bad, "wss_z_collis.txt"), "a") as f:
        f.write("\n")
    chdir(HERE)  # same executable, changed tables
    try:
        run_test(NAME, 2, default_omp_threads(), param_overrides={"SGChemTablesPath": bad}, allow_nonzero_exit=True)
    finally:
        chdir(ROOT)
    log = open(path.join(HERE, f"test_{NAME}.out")).read()
    assert "wss_z_collis.txt has hash" in log and "scripts/get_stella_tables.sh" in log
    assert "SGCHEM: chemistry tables built" not in log


def test_sgchem_floor_temperature(tables, box):
    """molecular gas held at the energy floor: the floor assumes atomic gas, so u stays above the solved energy; the snapshot
    holds the chemistry's temperature (one standalone step from the snapshot before), not u times the cached weight"""
    rs = path.join(HERE, "output_floor")
    shutil.rmtree(rs, ignore_errors=True)
    makedirs(rs)
    snap1 = path.join(rs, "snapshot_001.hdf5")
    shutil.copy(path.join(HERE, "output", "snapshot_001.hdf5"), snap1)
    with h5py.File(snap1, "r+") as f:
        x = f["PartType0/SGChemTracAbund"]
        a = x[:]
        a[:, 0] = 0.49
        x[...] = a
    floor = 60.0
    chdir(HERE)
    try:
        params = write_params_with_overrides(f"{NAME}.params", {"SGChemTablesPath": tables, "OutputDir": "output_floor", "MinGasTemp": floor,
                                                                "TimeMax": 0.08, "TimeBetSnapshot": 0.004})
        with open(path.join(rs, "test.out"), "w") as o:
            subprocess.run(mpi_launch_command(2, default_omp_threads()) + ["./GIZMO", params, "2", "1"], stdout=o, stderr=o, check=True)
    finally:
        chdir(ROOT)
    a, b = sorted(glob.glob(path.join(rs, "snapshot_*.hdf5")))[-2:]
    with h5py.File(a) as s0, h5py.File(b) as s1:
        g0, g1 = s0["PartType0"], s1["PartType0"]
        ta, tb = s0["Header"].attrs["Time"], s1["Header"].attrs["Time"]
        o0, o1 = np.argsort(g0["ParticleIDs"][:]), np.argsort(g1["ParticleIDs"][:])
        assert np.array_equal(g0["InternalEnergy"][:][o0], g1["InternalEnergy"][:][o1])  # held at the floor
        start = np.column_stack([g0["SGChemDustTemp"][:], g0["SGChemTracAbund"][:]])
        ref = standalone(tables, s0, np.array([ta, ta, tb]), start, over={"MinGasTemp": floor})
        got, want = g1["Temperature"][:], np.array([ref[int(i)][1] for i in g1["ParticleIDs"][:]])
    print(f"snapshot Temperature {got.min():.6g}-{got.max():.6g} K, chemistry {want.min():.6g}-{want.max():.6g} K")
    assert np.all(np.abs(got / want - 1) < 1e-8)


def test_sgchem_ledger_negative_total(tmp_path):
    """the exact sum of system/ledger.cc holds negative totals: printed as numbers, not NaN"""
    src = open(path.join(ROOT, "src", "system", "ledger.cc")).read()
    code = "\n".join(l for l in src.splitlines() if l.startswith(("#define LEDGER_NLIMB", "#define LEDGER_EXP0")))
    code += "\n" + re.search(r"^struct ledger_sum \{.*?\};\n", src, re.S | re.M).group(0)
    for f in ("ledger_normalize", "ledger_add_exact", "ledger_merge", "ledger_value"):
        code += re.search(rf"^static \w+ {f}\(.*?^}}\n", src, re.S | re.M).group(0)
    c = tmp_path / "ledger_value_unit.cc"
    c.write_text("#include <math.h>\n#include <stdint.h>\n#include <stdio.h>\n" + code + """
int main(void) {
    struct ledger_sum s = {{0}, 0};
    ledger_add_exact(&s, 1.5); ledger_add_exact(&s, -4.0); ledger_normalize(&s);
    printf("%.17g\\n", ledger_value(&s));
    return 0;
}
""")
    exe = tmp_path / "ledger_value_unit"
    subprocess.run(["g++", "-O1", str(c), "-o", str(exe), "-lm"], check=True)
    assert float(subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout) == -2.5
