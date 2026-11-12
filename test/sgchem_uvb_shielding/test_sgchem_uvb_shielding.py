"""UV background shielding by the TreeCol columns (SGCHEM_UVB_TREECOL_SHIELDING), zoom configuration, net 17, FG20 at z = 6.5.
The driver evolves a real cell with prescribed TreeCol gas columns, in a build with and a build without the shielding:
zero columns give the unshielded result bit for bit; the band attenuation factors match the FG20 spectrum integral with Verner+96
cross sections; heating per ionisation rises behind a column; in a uniform sphere (centre column 1e20 cm^-2, n_H = 0.1-100) the
interior stays neutral and cold, and the ionisation front sits where photoionisation balances recombination for the shielded rate.
"""

import os
import shlex
import subprocess
import tempfile
from os.path import abspath, dirname, join

import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test

DRIVER_DIR = dirname(abspath(__file__))
REPO = abspath(join(DRIVER_DIR, "..", ".."))
DRIVER = "sgchem_uvb_shielding"
FLAG = "SGCHEM_UVB_TREECOL_SHIELDING"
ABHE = 0.079
RY_EV = 13.6057
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def _link_driver(test_dir, tag, repo=REPO):
    """Link the driver against the objects of the build that just finished, in place of core/main.o."""
    out = join(test_dir, "output")
    os.makedirs(out, exist_ok=True)
    dry = subprocess.run(["make", "-n", "-B", "GIZMO"], capture_output=True, text=True, cwd=repo).stdout.splitlines()
    compile_main = [l for l in dry if " -c core/main.cc" in l]
    link = [l for l in dry if "-o GIZMO" in l]
    assert compile_main and link, "could not read the compile/link commands from make -n"
    obj, exe = join(out, f"{DRIVER}_driver_{tag}.o"), join(out, f"{DRIVER}_driver_{tag}")
    cc = shlex.split(compile_main[-1])
    cc[cc.index("core/main.cc")] = join(DRIVER_DIR, f"{DRIVER}_driver.cc")
    cc[cc.index("core/main.o")] = obj
    cc.insert(1, "-I" + repo)
    ln = shlex.split(link[-1])
    ln[ln.index("core/main.o")] = obj
    ln[ln.index("GIZMO")] = exe
    for cmd in (cc, ln):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=repo)
        assert r.returncode == 0, "driver build failed:\n%s\n%s" % (" ".join(cmd), r.stderr[-4000:])
    return exe


def run_driver(exe, repo=REPO):
    run = tempfile.mkdtemp(prefix=f"{DRIVER}_")
    os.makedirs(join(run, "output"))
    os.symlink(repo, join(run, "gizmo"))  # SGCHEM and the stellar tables open their data under gizmo/
    os.symlink(join(repo, "cooling", "TREECOOL"), join(run, "TREECOOL"))  # FG20
    subprocess.run(["python3", join(DRIVER_DIR, f"make_{DRIVER}_ics.py"), join(run, f"{DRIVER}_ics.hdf5")], check=True)
    r = subprocess.run(["mpirun", "-np", "1", exe, join(DRIVER_DIR, f"{DRIVER}.params")], capture_output=True, text=True, cwd=run)
    assert "DONE" in r.stdout, "driver did not finish (exit %d):\n%s\n%s" % (r.returncode, r.stdout[-3000:], r.stderr[-3000:])
    return [l for l in r.stdout.splitlines() if l.split(" ")[0] in ("UVB", "FAC", "THIN", "SPH")]


def rows_of(lines, tag):
    return [{k: float(v) for k, v in (kv.split("=") for kv in l.split()[1:])} for l in lines if l.startswith(tag + " ")]


def verner(E, Eth, E0, s0, ya, P, yw, y0, y1):
    x = E / E0 - y0
    y = np.sqrt(x * x + y1 * y1)
    F = ((x - 1) ** 2 + yw ** 2) * y ** (0.5 * P - 5.5) * (1 + np.sqrt(y / ya)) ** (-P)
    return np.where(E >= Eth, s0 * F * 1e-18, 0.0)


THRESH = (13.6, 24.59, 54.42)
SIGMA = (lambda E: verner(E, 13.6, 0.4298, 5.475e4, 32.88, 2.963, 0, 0, 0),
         lambda E: verner(E, 24.59, 13.61, 949.2, 1.469, 3.188, 2.039, 0.4434, 2.136),
         lambda E: verner(E, 54.42, 1.720, 1.369e4, 32.88, 2.963, 0, 0, 0))


class FG20:
    """Exact attenuation of the HI, HeI photoionisation and heating rates behind columns (N_HI, N_HeI, N_HeII), z = 6.5 FG20 spectrum."""

    def __init__(self):
        d = np.loadtxt(join(DRIVER_DIR, "fg20_spectrum_z6_7.txt"))
        self.nu = np.logspace(0, np.log10(2000), 20000)
        J = np.exp(np.interp(np.log(self.nu), np.log(d[:, 0]), np.log(np.maximum(d[:, 2], 1e-300))))
        self.n = J / self.nu  # photons per unit nu, up to a constant
        self.E = self.nu * RY_EV
        self.S = np.array([s(self.E) for s in SIGMA])

    def factors(self, N):
        att = np.exp(-np.minimum((self.S * np.asarray(N)[:, None]).sum(0), 700))
        out = []
        for c in (0, 1):
            w = self.n * self.S[c]
            we = w * (self.E - THRESH[c])
            out.append((TRAPZ(w * att, self.nu) / TRAPZ(w, self.nu), TRAPZ(we * att, self.nu) / TRAPZ(we, self.nu)))
        return out


def check_factors(fac, ref):
    fails = []
    for d in fac:
        N = (d["NHI"], d["NHeI"], 0.0)
        if N[0] == 0:
            if any(d[k] != 1.0 for k in ("fgHI", "fgHeI", "fgHeII", "fqHI", "fqHeI", "fqHeII")):
                fails.append("zero column: factors not exactly 1: %s" % d)
            continue
        ex = ref.factors(N)
        for c, (kg, kq) in enumerate((("fgHI", "fqHI"), ("fgHeI", "fqHeI"))):
            for k, e in ((kg, ex[c][0]), (kq, ex[c][1])):
                tol = 0.10 if e > 0.05 else 0.15
                if e > 0.01 and abs(d[k] / e - 1) > tol:
                    fails.append("N_HI=%.3g N_HeI=%.3g %s=%.4g, FG20 spectrum %.4g" % (N[0], N[1], k, d[k], e))
    for he in (0, 1):  # heating per ionisation rises behind the column
        r = [d["fqHI"] / d["fgHI"] for d in sorted(fac, key=lambda d: d["NHI"]) if d["NHI"] > 0 and (d["NHeI"] > 0) == he and d["fgHI"] > 1e-6]
        if any(b < a * (1 - 1e-9) for a, b in zip(r, r[1:])) or r[-1] < 2.0:
            fails.append("HI heating per ionisation does not harden behind the column: %s" % r)
    return fails


def front_reference(ref, n, xhep, T):
    """Nmin of the sphere point where the pixel-mean exact rate balances recombination at x_HII = 0.5 (case B, n_e = n/2)."""
    alpha_b = 2.59e-13 * (T / 1e4) ** -0.7
    gamma0 = 2.911e-13  # FG20 Gamma_HI at z = 6.5 (TREECOOL)
    f_front = alpha_b * 0.5 * n / gamma0
    mus = np.repeat([2 / 3, 0, -2 / 3], 4)
    R = 1e20 / n
    best = None
    for ldepth in np.arange(15.0, 20.0, 0.01):
        r = R - 10 ** ldepth / n
        col = n * (-r * mus + np.sqrt(R * R - r * r * (1 - mus ** 2)))
        f = np.mean([ref.factors((0.5 * N, (ABHE - xhep) * N, 0.0))[0][0] for N in col])
        if f < f_front:
            best = col.min()
            break
    return best, f_front


def front_from_rows(rows):
    """Nmin where x_HII crosses 0.5 (log-interpolated), and the cell temperature there."""
    for a, b in zip(rows, rows[1:]):
        if a["xHp"] >= 0.5 > b["xHp"]:
            w = (a["xHp"] - 0.5) / (a["xHp"] - b["xHp"])
            return 10 ** ((1 - w) * np.log10(a["Nmin"]) + w * np.log10(b["Nmin"])), (1 - w) * a["T"] + w * b["T"], a["xHep"]
    return None, None, None


def check_sphere(on, off, ref):
    fails, notes = [], []
    for n in sorted({d["nH"] for d in on}):
        s_on = [d for d in on if d["nH"] == n]
        s_off = [d for d in off if d["nH"] == n]
        for d in s_on:
            if d["Nmin"] >= 1e19 and not (d["xHp"] < 1e-2 and d["T"] < 1e3):
                fails.append("n=%g Nmin=%.3g: shielded interior not neutral and cold (x_HII=%.3g, T=%.4g K)" % (n, d["Nmin"], d["xHp"], d["T"]))
        if n <= 1:
            for d in s_off:
                if d["xHp"] < 0.5:
                    fails.append("n=%g unshielded build: x_HII=%.3g < 0.5 at Nmin=%.3g (thin UVB should ionise)" % (n, d["xHp"], d["Nmin"]))
            if not min(d["xHp"] for d in s_off) > 0.8:
                continue  # thin UVB only partly ionises this density: no front to place
            Nf, Tf, xhep = front_from_rows(s_on)
            if Nf is None:
                fails.append("n=%g: no ionisation front inside the sphere" % n)
                continue
            Nref, f_front = front_reference(ref, n, xhep, Tf)
            notes.append("n=%g: front at Nmin=%.3g (T=%.0f K), reference %.3g (f=%.3g)" % (n, Nf, Tf, Nref, f_front))
            if Nref is None or abs(np.log10(Nf / Nref)) > 0.25:
                fails.append("n=%g: front at Nmin=%.3g cm^-2, photoionisation balance gives %s" % (n, Nf, Nref))
    return fails, notes


def test_sgchem_uvb_shielding():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO; nothing to validate from external snapshots")
    test_dir = join(REPO, "test", DRIVER)
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        exe = {}
        build_gizmo_for_test(DRIVER, extra_config_flags=(FLAG,), post_build=lambda: exe.__setitem__("on", _link_driver(test_dir, "on")))
        build_gizmo_for_test(DRIVER, post_build=lambda: exe.__setitem__("off", _link_driver(test_dir, "off")))
    finally:
        os.chdir(cwd)
    on, off = run_driver(exe["on"]), run_driver(exe["off"])
    thin_on, thin_off = [l for l in on if l.startswith("THIN")], [l for l in off if l.startswith("THIN")]
    assert len(thin_on) == 4 and thin_on == thin_off, "zero columns differ from the unshielded build:\n%s\n%s" % ("\n".join(thin_on), "\n".join(thin_off))
    ref = FG20()
    fails = check_factors(rows_of(on, "FAC"), ref)
    sph_fails, notes = check_sphere(rows_of(on, "SPH"), rows_of(off, "SPH"), ref)
    print("\n".join(notes))
    assert not fails + sph_fails, "\n".join(fails + sph_fails)
