/* Standalone check of cooling/jaco_composition.h against the standard cooling module's composition conventions:
   yhelium() for n_He/n_H and Get_Gas_Mean_Molecular_Weight_mu() for the mean molecular weight.
       make -C test/jaco_composition */
#include <cmath>
#include <cstdio>
#include "../../cooling/jaco_composition.h"

static const double HYDROGEN_MASSFRAC = 0.76, Z_SUN = 0.0142, Y_SUN = 0.2703; /* constants.h; SolarAbundances[0,1] */
static int failures = 0;

static void check(const char *what, double got, double expected, double rtol) {
    int ok = std::fabs(got - expected) <= rtol * std::fabs(expected);
    std::printf("%-58s %10.5g  expected %10.5g (rtol %g)  %s\n", what, got, expected, rtol, ok ? "ok" : "FAIL");
    failures += !ok;
}

/* yhelium(): 0.25 Y/(1-Y) with the tracked Y (COOL_METAL_LINES_BY_SPECIES), else (1-X)/(4X) at HYDROGEN_MASSFRAC */
static double gizmo_yhelium(int tracked, double Y) {
    if (tracked) { double y = (Y < 0.5) ? Y : 0.5; return 0.25 * y / (1. - y); }
    return (1. - HYDROGEN_MASSFRAC) / (4. * HYDROGEN_MASSFRAC);
}

/* Get_Gas_Mean_Molecular_Weight_mu() for neutral atomic gas (fmol = ne = 0) */
static double gizmo_mu_neutral(int tracked, double Zmet, double Y) {
    double X = HYDROGEN_MASSFRAC, Yv = 1. - X, Z = (Zmet < 0.25) ? Zmet : 0.25;
    if (tracked) {Yv = (Y < 0.35) ? Y : 0.35;}
    X = 1. - (Yv + Z);
    return 1. / (X + Yv / 4. + Z / 16.);
}

/* the network's mean molecular weight for neutral atomic gas: (mass per H)/(particles per H), metals at A = 16 */
static double jaco_mu_neutral(double X, double Y, double Z) {
    double y = 0.25 * Y / X, x_metals = Z / (16. * X);
    return (1. / X) / (1. + y + x_metals);
}

int main() {
    double Y, Z, X;
    /* METALS without a per-species array (e.g. isodisk_thermalfb): helium is not tracked */
    X = jaco_mass_fractions(Z_SUN, 0., 0, HYDROGEN_MASSFRAC, &Y, &Z);
    check("untracked He, Z = Zsun: Y", Y, 1. - HYDROGEN_MASSFRAC, 1e-14);
    check("untracked He, Z = Zsun: n_He/n_H vs yhelium()", 0.25 * Y / X, gizmo_yhelium(0, 0.), 0.03);
    check("untracked He, Z = Zsun: mu(neutral) vs GIZMO's", jaco_mu_neutral(X, Y, Z), gizmo_mu_neutral(0, Z_SUN, 0.), 1e-12);
    /* full species array: He tracked */
    X = jaco_mass_fractions(Z_SUN, Y_SUN, 1, HYDROGEN_MASSFRAC, &Y, &Z);
    check("tracked He, solar: X", X, 1. - Y_SUN - Z_SUN, 1e-14);
    check("tracked He, solar: n_He/n_H vs yhelium()", 0.25 * Y / X, gizmo_yhelium(1, Y_SUN), 0.03);
    check("tracked He, solar: mu(neutral) vs GIZMO's", jaco_mu_neutral(X, Y, Z), gizmo_mu_neutral(1, Z_SUN, Y_SUN), 1e-12);
    /* no METALS: GIZMO's defaults */
    X = jaco_mass_fractions(0., 0., 0, HYDROGEN_MASSFRAC, &Y, &Z);
    check("no metals: X", X, HYDROGEN_MASSFRAC, 1e-14);
    check("no metals: mu(neutral) vs GIZMO's", jaco_mu_neutral(X, Y, Z), gizmo_mu_neutral(0, 0., 0.), 1e-12);
    /* GIZMO's caps on extreme abundances */
    X = jaco_mass_fractions(0.4, 0.5, 1, HYDROGEN_MASSFRAC, &Y, &Z);
    check("capped Z = 0.25, Y = 0.35: X", X, 0.4, 1e-14);
    std::printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
