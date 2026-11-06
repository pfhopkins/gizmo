"""Merger products of table stars inside a KETJU region (stella_dev stella_star_mergers.cc, ketju_coupling.cc contact radius).

Run "young": a 5 + 3 Msun pair at 60 % of the 5 Msun main sequence, head on. Bars, Hurley, Pols & Tout 2002 eq. 80 as on stella_dev:
the product is a main-sequence star of 8 Msun, birth Z mass-weighted, main-sequence fraction f = 0.1 sum_i M_i f_i / M, age
t_PMS(8 Msun) + t_MS_start + f (t_MS_end - t_MS_start), winds restart at that table age.

Run "evolved": a 3 Msun giant passes a 1 Msun star at 0.7 (R1 + R2) of the table radii, 3.2 x the contact distance of the
main-sequence fit, so it merges only with radii from the tables.
"""
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

HERE = Path(__file__).parent
NAME = "ketju_star_merger_product"
sys.path.insert(0, str(HERE.parent / "treecol_pixels"))
from stella_table_bands import Tables  # noqa: E402

AU, G, RSUN = 1.0 / 206264.806 / 1e3, 4.300710573e-6, 6.957e10
TIME_YR = 3.085678e21 / 1e5 / 3.15576e7          # code time [yr]
Z1, Z2, F_YOUNG = 0.0142, 0.004, 0.6


class Tab(Tables):
    def __init__(self, path):
        super().__init__(path)
        with h5py.File(path) as f:
            self.ms0, self.ms1, self.logR = f["phase_transitions/t_MS_start"][()], f["phase_transitions/t_MS_end"][()], f["logR_cm"][()].astype(np.float64)

    def f_ms(self, M, Z, age):
        lM, lZ = np.log10(M), np.log10(Z)
        t0, t1, tp = self._i2(self.ms0, lM, lZ), self._i2(self.ms1, lM, lZ), self._i2(self.t_pms, lM, lZ)
        return min(max((age - tp - t0) / (t1 - t0), 0.0), 1.0)

    def age_at(self, M, Z, f):
        lM, lZ = np.log10(M), np.log10(Z)
        t0, t1, tp = self._i2(self.ms0, lM, lZ), self._i2(self.ms1, lM, lZ), self._i2(self.t_pms, lM, lZ)
        return tp + t0 + f * (t1 - t0)

    def radius_rsun(self, M, Z, age):
        lM, lZ = np.log10(M), np.log10(Z)
        return 10 ** self._i3(self.logR, lM, lZ, np.log10(max(age - self._i2(self.t_pms, lM, lZ), 100.0))) / RSUN


def fit_rsun(m):
    return m ** 0.8 if m < 1 else m ** 0.57


def write_ics(name, m, x, v, m_drawn, z, tform):
    with h5py.File(name, "w") as f:
        h = f.create_group("Header"); n = len(m)
        h.attrs["NumPart_ThisFile"] = [0, 0, 0, 0, n, 0]; h.attrs["NumPart_Total"] = [0, 0, 0, 0, n, 0]
        h.attrs["NumPart_Total_HighWord"] = [0] * 6; h.attrs["MassTable"] = [0.0] * 6
        h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = 1.0; h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0; h.attrs["OmegaLambda"] = 0.0; h.attrs["HubbleParam"] = 1.0
        h.attrs["Flag_Sfr"] = 1; h.attrs["Flag_StellarAge"] = 1; h.attrs["Flag_Feedback"] = 1; h.attrs["Flag_Cooling"] = 0; h.attrs["Flag_Metals"] = 0
        g = f.create_group("PartType4")
        g["Coordinates"] = np.array(x) * AU; g["Velocities"] = np.array(v); g["Masses"] = np.array(m) / 1e10
        g["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint64); g["StellarFormationTime"] = np.array(tform)
        g["SingleStarDrawnMass"] = np.array(m_drawn); g["SingleStarBirthMetallicity"] = np.array(z)


def pair(m1, m2, rp_au, r0_au=30.0, vinf=5.0):
    mu = G * (m1 + m2); a = -mu / vinf ** 2; e = 1 - rp_au * AU / a
    h = np.sqrt(mu * a * (1 - e * e)); v0 = np.sqrt(vinf ** 2 + 2 * mu / (r0_au * AU))
    vt = h / (r0_au * AU); vr = -np.sqrt(max(v0 ** 2 - vt ** 2, 0.0))
    x = np.array([r0_au, 0, 0]); v = np.array([vr, vt, 0]); M = m1 + m2
    return [-m2 / M * x, m1 / M * x], [-m2 / M * v, m1 / M * v]


def run(tag, m, rp_au, ages, zs, tables):
    """Started from a snapshot (flag 2): a fresh start does not read stellar ages."""
    d = HERE / f"run_{tag}"
    shutil.rmtree(d, ignore_errors=True)
    (d / "output").mkdir(parents=True)
    x, v = pair(m[0], m[1], rp_au)
    write_ics(str(d / "output" / "snapshot_000.hdf5"), m, x, v, m, zs, [-a / TIME_YR for a in ages])
    p = (HERE / f"{NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"])
    (d / "run.params").write_text(p)
    with open(d / "run.out", "w") as o:
        subprocess.run(mpi_launch_command(1, 1) + [str(HERE / "GIZMO"), "run.params", "2", "0"], cwd=d, stdout=o, stderr=subprocess.STDOUT,
                       env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    snaps = sorted(glob.glob(str(d / "output" / "snapshot_*.hdf5")))
    assert len(snaps) >= 2
    with h5py.File(snaps[-1]) as f:
        g = f["PartType4"]; out = {k: g[k][()] for k in g.keys()}; out["t"] = f["Header"].attrs["Time"]
    out["log"] = (d / "run.out").read_text()
    return out


@pytest.fixture(scope="module")
def tab():
    t = get_stellar_evolution_tables()
    build_gizmo_for_test(NAME, 1)
    return t, Tab(t["StellarTablesFile"])


def test_product_state(tab):
    t, T = tab
    m1, m2 = 5.0, 3.0
    age = T.age_at(m1, Z1, F_YOUNG)
    o = run("young", [m1, m2], 0.0, [age, age], [Z1, Z2], t)
    assert "KETJU STAR COLLISION" in o["log"] and len(o["Masses"]) == 1
    M = m1 + m2; zb = (m1 * Z1 + m2 * Z2) / M
    f = 0.1 * (m1 * T.f_ms(m1, Z1, age) + m2 * T.f_ms(m2, Z2, age)) / M
    want = T.age_at(M, zb, f)
    got = (o["t"] - o["StellarFormationTime"][0]) * TIME_YR
    print(f"product drawn mass {o['SingleStarDrawnMass'][0]}, Z {o['SingleStarBirthMetallicity'][0]} / {zb}, f {f}, age {got} / {want}")
    assert o["SingleStarDrawnMass"][0] == pytest.approx(M, rel=1e-9)
    assert o["SingleStarBirthMetallicity"][0] == pytest.approx(zb, rel=1e-9)
    assert got == pytest.approx(want, rel=1e-3)
    assert o["SingleStarLastWindLogAge"][0] == pytest.approx(np.log10(want - T._i2(T.t_pms, np.log10(M), np.log10(zb))), abs=1e-3)


def test_contact_radius_from_tables(tab):
    t, T = tab
    m1, m2 = 3.0, 1.0
    life = T._i2(T.life, np.log10(m1), np.log10(Z1)) + T._i2(T.t_pms, np.log10(m1), np.log10(Z1))
    age = 0.985 * life
    r1, r2 = T.radius_rsun(m1, Z1, age), T.radius_rsun(m2, Z1, age)
    rp = 0.7 * (r1 + r2) * RSUN / 1.495978707e13
    fit = (fit_rsun(m1) + fit_rsun(m2)) * RSUN / 1.495978707e13
    print(f"radii {r1} {r2} Rsun, pericentre {rp} AU, fit contact {fit} AU")
    assert rp > 3 * fit, "the giant must be well above the main-sequence fit at this age"
    o = run("evolved", [m1, m2], rp, [age, age], [Z1, Z1], t)
    assert "KETJU STAR COLLISION" in o["log"] and len(o["Masses"]) == 1
