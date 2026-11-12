"""Band table of SGCHEM_UVB_TREECOL_SHIELDING (chem_interface.cc) from the FG20 spectrum.
Usage: python3 make_uvb_shielding_bands.py <fg20_spec_nu.dat> [z]   (fg20_102720_rescaled.zip, galaxies.northwestern.edu/uvb-fg20)
Per band: absorber cross sections of HI, HeI, HeII averaged with the HI photoionisation weight J_nu/(h nu) sigma_HI (photon number
above 54.4 eV), and each band's fraction of the HI, HeI, HeII photoionisation (J/(h nu) sigma) and photoheating (x (h nu - h nu_th)) rates."""
import sys
import numpy as np

RY_EV = 13.6057
EDGES_EV = [13.6, 16.3, 19.7, 24.59, 29.9, 35.4, 40.8, 46.3, 51.0, 54.42, 27211.0]
THRESH = (13.6, 24.59, 54.42)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def verner(E, Eth, E0, s0, ya, P, yw, y0, y1):
    x = E / E0 - y0
    y = np.sqrt(x * x + y1 * y1)
    F = ((x - 1) ** 2 + yw ** 2) * y ** (0.5 * P - 5.5) * (1 + np.sqrt(y / ya)) ** (-P)
    return np.where(E >= Eth, s0 * F * 1e-18, 0.0)


SIGMA = (lambda E: verner(E, 13.6, 0.4298, 5.475e4, 32.88, 2.963, 0, 0, 0),
         lambda E: verner(E, 24.59, 13.61, 949.2, 1.469, 3.188, 2.039, 0.4434, 2.136),
         lambda E: verner(E, 54.42, 1.720, 1.369e4, 32.88, 2.963, 0, 0, 0))


def main(path, z=6.5):
    with open(path) as f:
        zs = np.array(f.readline().split(), float)
    d = np.loadtxt(path, skiprows=1)
    iz = int(np.argmin(abs(zs - z)))
    nu = np.logspace(0, np.log10(2000), 40000)
    nu = np.unique(np.concatenate([nu, [e / RY_EV * f for e in THRESH[1:] for f in (1, 1 + 1e-9)]]))
    n = np.exp(np.interp(np.log(nu), np.log(d[:, 0]), np.log(np.maximum(d[:, 1 + iz], 1e-300)))) / nu
    E = nu * RY_EV
    S = np.array([s(E) for s in SIGMA])
    edges = [e / RY_EV for e in EDGES_EV]
    edges[0] = 1.0
    sig, wg, wq = (np.zeros((len(edges) - 1, 3)) for _ in range(3))
    for k in range(len(edges) - 1):
        m = (nu >= edges[k]) & (nu < edges[k + 1])
        wt = n[m] * S[0][m] if S[0][m].any() else n[m]
        for s in range(3):
            sig[k, s] = TRAPZ(wt * S[s][m], nu[m]) / TRAPZ(wt, nu[m])
        for c in range(3):
            w = n * S[c]
            we = w * (E - THRESH[c])
            wg[k, c] = TRAPZ(np.where(m, w, 0), nu) / TRAPZ(w, nu)
            wq[k, c] = TRAPZ(np.where(m, we, 0), nu) / TRAPZ(we, nu)
    row = lambda a: "{" + ", ".join("%.4e" % x for x in a) + "}"
    print("/* z = %.2f */" % zs[iz])
    for name, t in (("sigma", sig), ("gamma", wg), ("heat", wq)):
        print("static const double uvb_band_%s[UVB_SHIELD_NBANDS][3] = {%s};" % (name, ", ".join(row(r) for r in t)))


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 6.5)
