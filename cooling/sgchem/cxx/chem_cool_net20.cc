// chem_cool_net20.cc -- network 20 (TIGRESS-NCR): rates, equilibrium species, heating, cooling,
// dust temperature and the frozen per-cell radiation inputs.  Equation map: DEVIATIONS.md D-29.

#include "chem_net20.h"
#include "chem_net20_cie.h"
#include <algorithm>
#include <cmath>

namespace {

const double EV = CHEM_EV;
const double KB = CHEM_KBOLTZ;
const double CLIGHT = 2.99792458e10;
const double ARAD = 7.57e-15;
const double FOURPI = 4.0 * 3.14159265358979323846;

const double SIG_PE = 1.0e-21;                      // PE dust cross section per H at Z'_d = 1
const double J_PE_D = 1.8e-4, J_LW_D = 3.0e-5;     // standard-ISRF mean intensities (chi = 1) [erg cm^-2 s^-1 sr^-1]
const double ZETA_PD0 = 5.7e-11;                   // unshielded H2 photodissociation [s^-1]
const double ZETA_PIC0 = 3.5e-10;                  // unshielded C photoionization [s^-1]
const double SIG_PI_RATIO = 4.6 / 3.1;             // sigma_pi,H2 / sigma_pi,H
const double Q_PI_H2 = 4.4 * EV;
const double N0_CR = 9.35e20;
const double TT1 = 2e4, TT2 = 3.5e4;
const double ALPHA_GD = 3.2e-34;
const double B5 = 3.0;
const double TAU_OPT_RATIO = 1.87;
const double TFLOOR = 1.0;                          // floor for rate-coefficient evaluation only

inline double sq(double x) { return x * x; }

// H2 self-shielding (DB96, b5 = 3)
inline double f_shld_h2(double NH2)
{
    double x = std::max(NH2, 0.0) / 5e14, s = std::sqrt(1.0 + x);
    return 0.965 / sq(1.0 + x / B5) + 0.035 / s * std::exp(-8.5e-4 * s);
}

// C self-shielding with H2 cross-shielding (TH85 A6/A7)
inline double f_shld_c(double NC, double NH2)
{
    double y = 1.17e-8 * 1.2e-14 * 2.0 * std::max(NH2, 0.0);
    return std::exp(-1.6e-17 * std::max(NC, 0.0)) * std::exp(-y) / (1.0 + y);
}

// chi_H2 and chi_C from the frozen per-pixel LW weights and the live local H2 column
void shield(const ChemCell& c, double xH2, double* chi_h2, double* chi_c)
{
    const int pix = (c.iphoto == 6 && c.npix > 0);
    const int n = pix ? c.npix : 1;
    const double NH2_loc = xH2 * c.yn * c.dl;
    double h = 0.0, cc = 0.0;
    for (int k = 0; k < n; k++) {
        double NH2 = (pix ? c.col_h2[k] : 0.0) + NH2_loc;
        double NC  = pix ? c.col_c[k] : 0.0;
        h  += c.n20_wlw[k] * f_shld_h2(NH2);
        cc += c.n20_wlw[k] * f_shld_c(NC, NH2);
    }
    *chi_h2 = h; *chi_c = cc;
}

// temperature-only rate coefficients
struct TCoef {
    double T, lnT, sqT, T4;
    double a_rr;      // H+ case-B radiative recombination
    double k_ci;      // H + e collisional ionization
    double k_cpe;     // C+ + e (fit used by the reference implementation)
    double k_cph2;    // C+ + H2
    double gr_h, gr_c;  // T^C4 factors of the grain-assisted recombination fits
    double k1619;     // MH+ + e
};

void tcoef(double T, TCoef& k)
{
    k.T = T; k.lnT = std::log(T); k.sqT = std::sqrt(T); k.T4 = T * 1e-4;
    {
        double bb = 315614.0 / T, cc = 115188.0 / T;
        k.a_rr = 2.753e-14 * std::pow(bb, 1.5) * std::pow(1.0 + std::pow(cc, 0.407), -2.242);
    }
    if (T > 3e3) {
        double L = std::log(T * 8.6173e-5);
        k.k_ci = std::exp(-3.271396786e1 + L * (1.35365560e1 + L * (-5.73932875 + L * (1.56315498
                 + L * (-2.877056e-1 + L * (3.48255977e-2 + L * (-2.63197617e-3
                 + L * (1.11954395e-4 + L * (-2.03914985e-6)))))))));
    } else {
        k.k_ci = 0.0;
    }
    k.k_cpe = (T < 10.0) ? 9.982641225129824e-11 : std::exp(-0.7529152 * std::log(T) - 21.293937);
    k.k_cph2 = 3.3e-13 * std::pow(T, -1.3) * std::exp(-23.0 / T);
    k.gr_h = std::pow(T, 1.586e-2);
    k.gr_c = std::pow(T, 4.845e-2);
    k.k1619 = 1.0e-7 * std::pow(T * 1e-2, -0.5);
}

struct Ctx {
    double nH, xH2, xHp, xCtot, xOp, Zd, Zg, xi, zetaC, G0;
};

// grain-assisted recombination coefficient (fit form of the W01c Table 1 rates) [cm^3 s^-1]
inline double alpha_gr(double psi, double TC4, double lnT, const double* cf, double Zd)
{
    return 1e-14 * cf[0] / (1.0 + cf[1] * std::pow(psi, cf[2])
           * (1.0 + cf[3] * TC4 * std::pow(psi, -cf[5] - cf[6] * lnT))) * Zd;
}
const double CF_HP[7] = {12.25, 8.074e-6, 1.378, 5.087e2, 1.586e-2, 0.4723, 1.102e-5};
const double CF_CP[7] = {45.58, 6.089e-3, 1.128, 4.331e2, 4.845e-2, 0.8120, 1.333e-4};

// x_C+ in equilibrium at trial x_e
double xcp_of(double xe, const TCoef& k, const Ctx& q)
{
    if (q.xCtot <= 0.0 || q.zetaC <= 0.0) return 0.0;
    double ne = std::max(xe, 0.0) * q.nH;
    double agr = 0.0;
    if (q.Zd > 0.0) {
        double psi = q.G0 * k.sqT / (ne + 1e-50) + 1e-50;
        agr = alpha_gr(psi, k.gr_c, k.lnT, CF_CP, q.Zd);
    }
    double den = k.k_cpe * ne + agr * q.nH + k.k_cph2 * q.xH2 * q.nH + q.zetaC;
    return q.xCtot * q.zetaC / den;
}

// x_MH+ (molecular ions) at trial x_e, times 2 x_H2
double xmh_of(double xe, const TCoef& k, const Ctx& q)
{
    if (q.xH2 <= 0.0 || q.Zd <= 0.0 || q.xi <= 0.0) return 0.0;
    const double xe_max = 1.2006199779862501;
    double k1620 = 1e-14 * q.Zd, k1622 = 1e-14 * q.Zd, k1621 = 1e-9;
    double phi_s = (1.0 - xe / xe_max) * 0.67 / (1.0 + xe / 0.05);
    double xS = 5.3e-6 * q.Zg;
    double A = k.k1619 * (1.0 + k1621 / k1622 * xS), B = k1620 + k1621 * xS;
    double D = 4.0 * A * q.xi * (1.0 + phi_s) / q.nH;
    return 2.0 * q.xH2 * (D / (std::sqrt(B * B + D) + B)) / (2.0 * k.k1619);
}

// x_e = x_H+ + x_O+ + x_C+(x_e) + x_MH+(x_e): the right side falls with x_e, so the root is unique
double solve_xe(const TCoef& k, const Ctx& q, double guess, double* xcp, double* xmh)
{
    const double base = q.xHp + q.xOp;
    auto f = [&](double xe) { return base + xcp_of(xe, k, q) + xmh_of(xe, k, q) - xe; };
    double lo = base, hi = base + q.xCtot + xmh_of(base, k, q);
    double flo = f(lo), fhi = f(hi), x = lo;
    if (flo <= 0.0) x = lo;
    else if (fhi >= 0.0) x = hi;
    else {
        if (guess > lo && guess < hi) {
            double fg = f(guess);
            if (fg > 0.0) { lo = guess; flo = fg; } else { hi = guess; fhi = fg; }
        }
        int side = 0;
        for (int it = 0; it < 200; it++) {
            x = (lo * fhi - hi * flo) / (fhi - flo);
            double fx = f(x);
            if (std::fabs(fx) <= 1e-13 * x || (hi - lo) <= 1e-13 * hi) break;
            if (fx > 0.0) { lo = x; flo = fx; if (side == 1) fhi *= 0.5; side = 1; }
            else          { hi = x; fhi = fx; if (side == -1) flo *= 0.5; side = -1; }
        }
    }
    *xcp = xcp_of(x, k, q);
    *xmh = xmh_of(x, k, q);
    return x;
}

// CIE tables: He and metal cooling [per nH^2] and electrons per H
void cie(double T, double* lam_he, double* lam_met, double* xe_he, double* xe_met)
{
    if (T < 1e4) { *lam_he = *lam_met = *xe_he = *xe_met = 0.0; return; }
    const double* g = chem20_cie_log10T;
    double lt = std::min(std::log10(T), g[CHEM20_NCIE - 1]);
    int i = (int)(std::upper_bound(g, g + CHEM20_NCIE, lt) - g) - 1;
    i = std::max(0, std::min(i, CHEM20_NCIE - 2));
    double w = (lt - g[i]) / (g[i + 1] - g[i]);
    *lam_he  = std::pow(10.0, chem20_cie_log10_lam_he[i]  + w * (chem20_cie_log10_lam_he[i + 1]  - chem20_cie_log10_lam_he[i]));
    *lam_met = std::pow(10.0, chem20_cie_log10_lam_met[i] + w * (chem20_cie_log10_lam_met[i + 1] - chem20_cie_log10_lam_met[i]));
    *xe_he  = chem20_cie_xe_he[i]  + w * (chem20_cie_xe_he[i + 1]  - chem20_cie_xe_he[i]);
    *xe_met = chem20_cie_xe_met[i] + w * (chem20_cie_xe_met[i + 1] - chem20_cie_xe_met[i]);
}

// ---- cooling functions returning per-H rates [erg s^-1] ----
inline double cool2(double q01, double q10, double A10, double E10, double xs)
{
    return q01 / (q01 + q10 + A10) * A10 * E10 * xs;
}
inline double cool3(double q01, double q10, double q02, double q20, double q12, double q21,
                    double A10, double A20, double A21, double E10, double E20, double E21, double xs)
{
    double R10 = q10 + A10, R20 = q20 + A20, R21 = q21 + A21;
    double a0 = R10 * R20 + R10 * R21 + q12 * R20;
    double a1 = q01 * R20 + q01 * R21 + R21 * q02;
    double a2 = q02 * R10 + q02 * q12 + q12 * q01;
    double de = a0 + a1 + a2;
    return (a1 / de * A10 * E10 + a2 / de * (A20 * E20 + A21 * E21)) * xs;
}

const double FO = 0.75, FP = 0.25;   // H2 ortho / para

double cool_cii(double nH, double T, double xe, double xHI, double xH2, double xCp)
{
    double T2 = T * 1e-2;
    double k10e = 4.53e-8 * std::sqrt(1.0e4 / T);
    double k10HI = 7.58e-10 * std::pow(T2, 0.1281 + 0.0087 * std::log(T2));
    double k10o, k10p;
    if (T < 500.0) { k10o = (5.33 + 0.11 * T2) * 1e-10; k10p = (4.43 + 0.33 * T2) * 1e-10; }
    else { double t = std::pow(T, 0.07); k10o = 3.74757785025e-10 * t; k10p = 3.88997286356e-10 * t; }
    double q10 = nH * (k10e * xe + k10HI * xHI + (k10o * FO + k10p * FP) * xH2);
    const double A10 = 2.3e-6, E10 = 1.26e-14;
    double q01 = 2.0 * q10 * std::exp(-E10 / (KB * T));
    return cool2(q01, q10, A10, E10, xCp);
}

double cool_ci(double nH, double T, double xe, double xHI, double xH2, double xC)
{
    const double A10 = 7.880e-08, A20 = 1.810e-14, A21 = 2.650e-07;
    const double E10 = 3.261e-15, E20 = 8.624e-15, E21 = 5.363e-15;
    double T2 = T * 1e-2, lnT2 = std::log(T2), lnT = std::log(T);
    double fac = 8.629e-8 * std::sqrt(1.0e4 / T);
    double g10, g20, g21;
    if (T < 1.0e3) {
        g10 = (((-6.56325e-4 * lnT - 1.50892e-2) * lnT + 3.61184e-1) * lnT - 7.73782e-1) * lnT - 9.25141;
        g20 = (((0.705277e-2 * lnT - 0.111338) * lnT + 0.697638) * lnT - 1.30743) * lnT - 7.69735;
        g21 = (((2.35272e-3 * lnT - 4.18166e-2) * lnT + 0.358264) * lnT - 0.57443) * lnT - 7.4387;
    } else {
        g10 = (((1.0508e-1 * lnT - 3.47620) * lnT + 4.2595e1) * lnT - 2.27913e2) * lnT + 4.446e2;
        g20 = (((9.38138e-2 * lnT - 3.03283) * lnT + 3.61803e1) * lnT - 1.87474e2) * lnT + 3.50609e2;
        g21 = (((9.78573e-2 * lnT - 3.19268) * lnT + 3.85049e1) * lnT - 2.02193e2) * lnT + 3.86186e2;
    }
    double k10e = fac * std::exp(g10) / 3.0, k20e = fac * std::exp(g20) / 5.0, k21e = fac * std::exp(g21) / 5.0;
    double k10HI = 1.26e-10 * std::pow(T2, 0.115 + 0.057 * lnT2);
    double k20HI = 0.89e-10 * std::pow(T2, 0.228 + 0.046 * lnT2);
    double k21HI = 2.64e-10 * std::pow(T2, 0.231 + 0.046 * lnT2);
    double k10H2 = 0.67e-10 * std::pow(T2, -0.085 + 0.102 * lnT2) * FP + 0.71e-10 * std::pow(T2, -0.004 + 0.049 * lnT2) * FO;
    double k20H2 = 0.86e-10 * std::pow(T2, -0.010 + 0.048 * lnT2) * FP + 0.69e-10 * std::pow(T2, 0.169 + 0.038 * lnT2) * FO;
    double k21H2 = 1.75e-10 * std::pow(T2, 0.072 + 0.064 * lnT2) * FP + 1.48e-10 * std::pow(T2, 0.263 + 0.031 * lnT2) * FO;
    double q10 = nH * (k10HI * xHI + k10H2 * xH2 + k10e * xe);
    double q20 = nH * (k20HI * xHI + k20H2 * xH2 + k20e * xe);
    double q21 = nH * (k21HI * xHI + k21H2 * xH2 + k21e * xe);
    double q01 = 3.0 * q10 * std::exp(-E10 / (KB * T));
    double q02 = 5.0 * q20 * std::exp(-E20 / (KB * T));
    double q12 = (5.0 / 3.0) * q21 * std::exp(-E21 / (KB * T));
    return cool3(q01, q10, q02, q20, q12, q21, A10, A20, A21, E10, E20, E21, xC);
}

double cool_oi(double nH, double T, double xe, double xHI, double xH2, double xO)
{
    const double A10 = 8.910e-05, A20 = 1.340e-10, A21 = 1.750e-05;
    const double E10 = 3.144e-14, E20 = 4.509e-14, E21 = 1.365e-14;
    double T2 = T * 1e-2, lnT2 = std::log(T2);
    double k10HI = 3.57e-10 * std::pow(T2, 0.419 - 0.003 * lnT2);
    double k20HI = 3.19e-10 * std::pow(T2, 0.369 - 0.006 * lnT2);
    double k21HI = 4.34e-10 * std::pow(T2, 0.755 - 0.160 * lnT2);
    double k10H2 = 1.49e-10 * std::pow(T2, 0.264 + 0.025 * lnT2) * FP + 1.37e-10 * std::pow(T2, 0.296 + 0.043 * lnT2) * FO;
    double k20H2 = 1.90e-10 * std::pow(T2, 0.203 + 0.041 * lnT2) * FP + 2.23e-10 * std::pow(T2, 0.237 + 0.058 * lnT2) * FO;
    double k21H2 = 2.10e-12 * std::pow(T2, 0.889 + 0.043 * lnT2) * FP + 3.00e-12 * std::pow(T2, 1.198 + 0.525 * lnT2) * FO;
    double k10e = 5.12e-10 * std::pow(T, -0.075);
    double k20e = 4.86e-10 * std::pow(T, -0.026);
    double k21e = 1.08e-14 * std::pow(T, 0.926);
    double q10 = nH * (k10HI * xHI + k10H2 * xH2 + k10e * xe);
    double q20 = nH * (k20HI * xHI + k20H2 * xH2 + k20e * xe);
    double q21 = nH * (k21HI * xHI + k21H2 * xH2 + k21e * xe);
    double q01 = (3.0 / 5.0) * q10 * std::exp(-E10 / (KB * T));
    double q02 = (1.0 / 5.0) * q20 * std::exp(-E20 / (KB * T));
    double q12 = (1.0 / 3.0) * q21 * std::exp(-E21 / (KB * T));
    return cool3(q01, q10, q02, q20, q12, q21, A10, A20, A21, E10, E20, E21, xO);
}

// CO rotational cooling, LVG tables; T and log N_eff interpolated linearly
const int NT_CO = 11, NN_CO = 11;
const double TCO[NT_CO] = {10, 20, 30, 50, 80, 100, 300, 600, 1000, 1500, 2000};
const double NCO[NN_CO] = {14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0};
const double L0CO[NT_CO] = {24.77, 24.38, 24.21, 24.03, 23.89, 23.82,
                            23.34238089, 22.99832519, 22.75384686, 22.56640625, 22.43740866};
const double LLTECO[NN_CO * NT_CO] = {
    21.08, 20.35, 19.94, 19.45, 19.01, 18.80, 17.81, 17.23, 16.86, 16.66, 16.55,
    21.09, 20.35, 19.95, 19.45, 19.01, 18.80, 17.81, 17.23, 16.86, 16.66, 16.55,
    21.11, 20.37, 19.96, 19.46, 19.01, 18.80, 17.81, 17.23, 16.86, 16.66, 16.55,
    21.18, 20.40, 19.98, 19.47, 19.02, 18.81, 17.82, 17.23, 16.87, 16.66, 16.55,
    21.37, 20.51, 20.05, 19.52, 19.05, 18.83, 17.82, 17.23, 16.87, 16.66, 16.55,
    21.67, 20.73, 20.23, 19.64, 19.13, 18.90, 17.85, 17.25, 16.88, 16.67, 16.56,
    22.04, 21.05, 20.52, 19.87, 19.32, 19.06, 17.92, 17.28, 16.90, 16.69, 16.58,
    22.44, 21.42, 20.86, 20.19, 19.60, 19.33, 18.08, 17.38, 16.97, 16.75, 16.63,
    22.87, 21.82, 21.24, 20.55, 19.95, 19.66, 18.34, 17.59, 17.15, 16.91, 16.78,
    23.30, 22.23, 21.65, 20.94, 20.32, 20.03, 18.67, 17.89, 17.48, 17.26, 17.12,
    23.76, 22.66, 22.06, 21.35, 20.71, 20.42, 19.03, 18.26, 17.93, 17.74, 17.61};
const double NHALFCO[NN_CO * NT_CO] = {
    3.29, 3.49, 3.67, 3.97, 4.30, 4.46, 5.17, 5.47, 5.53, 5.30, 4.70,
    3.27, 3.48, 3.66, 3.96, 4.30, 4.45, 5.16, 5.47, 5.53, 5.30, 4.70,
    3.22, 3.45, 3.64, 3.94, 4.29, 4.45, 5.16, 5.47, 5.53, 5.30, 4.70,
    3.07, 3.34, 3.56, 3.89, 4.26, 4.42, 5.15, 5.46, 5.52, 5.30, 4.70,
    2.72, 3.09, 3.35, 3.74, 4.16, 4.34, 5.13, 5.45, 5.51, 5.29, 4.68,
    2.24, 2.65, 2.95, 3.42, 3.92, 4.14, 5.06, 5.41, 5.48, 5.26, 4.64,
    1.74, 2.15, 2.47, 2.95, 3.49, 3.74, 4.86, 5.30, 5.39, 5.17, 4.53,
    1.24, 1.65, 1.97, 2.45, 3.00, 3.25, 4.47, 5.02, 5.16, 4.94, 4.27,
    0.742, 1.15, 1.47, 1.95, 2.50, 2.75, 3.98, 4.57, 4.73, 4.52, 3.84,
    0.242, 0.652, 0.966, 1.45, 2.00, 2.25, 3.48, 4.07, 4.24, 4.03, 3.35,
    -0.258, 0.152, 0.466, 0.95, 1.50, 1.75, 2.98, 3.57, 3.74, 3.53, 2.85};
const double ALPHACO[NN_CO * NT_CO] = {
    0.439, 0.409, 0.392, 0.370, 0.361, 0.357, 0.385, 0.437, 0.428, 0.354, 0.322,
    0.436, 0.407, 0.391, 0.368, 0.359, 0.356, 0.385, 0.437, 0.427, 0.354, 0.322,
    0.428, 0.401, 0.385, 0.364, 0.356, 0.352, 0.383, 0.436, 0.427, 0.352, 0.320,
    0.416, 0.388, 0.373, 0.353, 0.347, 0.345, 0.380, 0.434, 0.425, 0.349, 0.316,
    0.416, 0.378, 0.360, 0.338, 0.332, 0.330, 0.371, 0.429, 0.421, 0.341, 0.307,
    0.450, 0.396, 0.367, 0.334, 0.322, 0.317, 0.355, 0.419, 0.414, 0.329, 0.292,
    0.492, 0.435, 0.403, 0.362, 0.339, 0.329, 0.343, 0.406, 0.401, 0.317, 0.276,
    0.529, 0.473, 0.441, 0.404, 0.381, 0.370, 0.362, 0.410, 0.392, 0.316, 0.272,
    0.555, 0.503, 0.473, 0.440, 0.423, 0.414, 0.418, 0.446, 0.404, 0.335, 0.289,
    0.582, 0.528, 0.499, 0.469, 0.457, 0.451, 0.470, 0.487, 0.432, 0.364, 0.310,
    0.596, 0.546, 0.519, 0.492, 0.483, 0.479, 0.510, 0.516, 0.448, 0.372, 0.313};

inline int seg(const double* x, int n, double v)
{
    if (v < x[0]) return 0;
    if (v > x[n - 1]) return n - 2;
    int i = 0;
    while (v > x[i]) i++;
    return std::max(i - 1, 0);
}
inline double lin(double x0, double x1, double y0, double y1, double x) { return y0 + (y1 - y0) / (x1 - x0) * (x - x0); }
inline double lin2(int iT, int iN, const double* d, double T, double N)
{
    double a = lin(TCO[iT], TCO[iT + 1], d[iN * NT_CO + iT], d[iN * NT_CO + iT + 1], T);
    double b = lin(TCO[iT], TCO[iT + 1], d[(iN + 1) * NT_CO + iT], d[(iN + 1) * NT_CO + iT + 1], T);
    return lin(NCO[iN], NCO[iN + 1], a, b, N);
}

double cool_co(double nH, double T, double xe, double xHI, double xH2, double xCO, double dvdr)
{
    if (xCO <= 0.0) return 0.0;
    const double mCO = 4.68e-23, Lmax = 3.086e20;
    double grad = std::max(dvdr, std::sqrt(2.0 * KB * T / mCO) / Lmax);
    double Neff = nH * xCO / grad;
    double T1 = std::min(T, 2000.0);
    double facT = std::pow(1.0 - std::exp(-T1), 1.0e3);
    double lN = std::log10(Neff * 1.0e5 + 1.0e13);
    double r4 = std::pow(T1, 0.25);
    double neff = nH * (xH2 + 1.75 * r4 * xHI + 680.1 / r4 * xe);
    int iT = seg(TCO, NT_CO, T1), iN = seg(NCO, NN_CO, lN);
    double L0 = std::pow(10.0, -lin(TCO[iT], TCO[iT + 1], L0CO[iT], L0CO[iT + 1], T1));
    double LLTE = std::pow(10.0, -lin2(iT, iN, LLTECO, T1, lN));
    double nhalf = std::pow(10.0, lin2(iT, iN, NHALFCO, T1, lN));
    double alpha = lin2(iT, iN, ALPHACO, T1, lN);
    double inv = 1.0 / L0 + neff / LLTE + 1.0 / L0 * std::pow(neff / nhalf, alpha) * (1.0 - nhalf * L0 / LLTE);
    return neff * xCO * facT / inv;
}

// H2 ro-vibrational cooling (analytic fit valid over density, T and f_H2)
double cool_h2rovib(double nH, double T, double xHI, double xH2)
{
    if (xH2 <= 0.0) return 0.0;
    double T3 = T * 1e-3, s3 = std::sqrt(T3);
    double nHI = xHI * nH, nH2 = xH2 * nH;
    double x1 = nHI + 5.0 * nH2, x2 = nHI + 4.5 * nH2, x3 = nHI + 0.75 * nH2, x4 = nHI + 0.05 * nH2;
    double f1 = 1.1e-25 * s3 * std::exp(-0.51 / T3) * (0.7 * x1 / (1.0 + x1 / 50.0) + 0.3 * x1 / (1.0 + x1 / 500.0));
    double f2 = 2.0e-25 * T3 * std::exp(-1.0 / T3) * (0.35 * x2 / (1.0 + x2 / 450.0) + 0.65 * x2 / (1.0 + x2 / 4500.0));
    double f3 = 2.4e-24 * s3 * T3 * std::exp(-2.0 / T3) * (x3 / (1.0 + x3 / 25.0));
    double f4 = 1.7e-23 * s3 * T3 * std::exp(-4.0 / T3) * (0.45 * x4 / (1.0 + x4 / 900.0) + 0.55 * x4 / (1.0 + x4 / 9000.0));
    return xH2 * (f1 + f2 + f3 + f4);
}

// H2 collisional dissociation rate per H2 [s^-1] (H and H2 partners, density dependent)
double zeta_cd_h2(double nH, double T, double xHI, double xH2)
{
    if (T <= 7.0e2) return 0.0;
    double Ti = 1.0 / T, l4 = std::log10(T * 1e-4);
    double k9l = 6.67e-12 * std::sqrt(T) * std::exp(-(1.0 + 63590.0 * Ti));
    double k9h = 3.52e-9 * std::exp(-43900.0 * Ti);
    double k10l = 5.996e-30 * std::pow(T, 4.1881) / std::pow(1.0 + 6.761e-6 * T, 5.6881) * std::exp(-54657.4 * Ti);
    double k10h = 1.3e-9 * std::exp(-53300.0 * Ti);
    double ncrH2 = std::pow(10.0, 4.845 - 1.3 * l4 + 1.62 * l4 * l4);
    double ncrHI = std::pow(10.0, 3.0 - 0.416 * l4 - 0.327 * l4 * l4);
    double n2 = nH * std::max(xHI / ncrHI + 2.0 * xH2 / ncrH2, 0.0);
    double kH = std::pow(10.0, std::log10(k9h) * n2 / (1.0 + n2) + std::log10(k9l) / (1.0 + n2));
    double kH2 = std::pow(10.0, std::log10(k10h) * n2 / (1.0 + n2) + std::log10(k10l) / (1.0 + n2));
    return kH2 * nH * xH2 + kH * nH * xHI;
}

// CR heating per primary ionization in molecular gas [eV]
double q_cr_h2(double nH)
{
    double l = std::log10(nH);
    if (l < 2.0) return 10.0;
    if (l < 4.0) return 10.0 + 3.0 * (l - 2.0) * 0.5;
    if (l < 7.0) return 13.0 + 4.0 * (l - 4.0) / 3.0;
    if (l < 10.0) return 17.0 + (l - 7.0) / 3.0;
    return 18.0;
}

// equilibrium dust temperature: UV + optical + blackbody + gas collisions = IR emission
double dust_temp(const ChemCell& c, double T, double Zd)
{
    if (Zd <= 0.0) return c.n20_tbb;
    const double C6 = 2.0e-25 * Zd / 100.0 * CLIGHT * ARAD;   // sigma_IR(T_d) c a T_d^4 = C6 T_d^6
    const double D = c.yn * ALPHA_GD * std::sqrt(T) * Zd;
    const double rhs = c.n20_gd_uv + c.n20_gd_ext + D * T;
    if (rhs <= 0.0) return 0.0;
    double td = std::pow(rhs / C6, 1.0 / 6.0);
    if (D > 0.0) td = std::min(td, rhs / D);
    for (int it = 0; it < 100; it++) {   // convex and increasing: Newton from above converges monotonically
        double t5 = std::pow(td, 5);
        double g = C6 * t5 * td + D * td - rhs, gp = 6.0 * C6 * t5 + D;
        double dt = g / gp;
        td -= dt;
        if (std::fabs(dt) <= 1e-10 * td) break;
    }
    return td;
}

}  // namespace

// ===========================================================================
void n20_cell_defaults(ChemCell& c)
{
    c.dvdr = 0.0;
    c.Zg = 1.0;
    for (int k = 0; k < CHEM_NPIX; k++) c.col_c[k] = 0.0;
    c.n20_kgr_const = 0;
    c.n20_tbb = 6.0;
    c.n20_sig_lw = 2.0e-21;
    c.n20_xe = -1.0;
}

// ===========================================================================
void cxx_calc_photo_net20(ChemCell& c)
{
    const double Zd = std::max(c.dust_to_gas_ratio, 0.0);
    const double chi_pe0 = std::max(c.G0, 0.0) / 1.7, chi_lw0 = std::max(c.G0_LW, 0.0) / 1.7;
    const int pix = (c.iphoto == 6 && c.npix > 0);
    const int n = pix ? c.npix : 1;
    const double N_add = c.AV_ext / c.AV_conversion_factor + c.yn * c.dl;
    double fpe = 0.0, flw = 0.0, fopt = 0.0;
    for (int k = 0; k < n; k++) {
        double w = pix ? c.fac_uv[k] : 1.0;
        double N = (pix ? c.col_tot[k] : 0.0) + N_add;
        double epe = std::exp(-SIG_PE * Zd * N), elw = std::exp(-c.n20_sig_lw * Zd * N);
        fpe  += w * epe;
        flw  += w * elw;
        fopt += w * std::exp(-SIG_PE * Zd * N / TAU_OPT_RATIO);
        c.n20_wlw[k] = chi_lw0 * w * elw;
    }
    for (int k = n; k < CHEM_NPIX; k++) c.n20_wlw[k] = 0.0;
    c.n20_chi_pe = chi_pe0 * fpe;
    c.n20_chi_lw = chi_lw0 * flw;
    c.n20_chi_fuv = (J_PE_D * c.n20_chi_pe + J_LW_D * c.n20_chi_lw) / (J_PE_D + J_LW_D);

    // CR attenuation by the effective column from the PE-band attenuation
    c.n20_xi = c.cosmic_ray_ion_rate;
    if (Zd > 0.0 && fpe > 0.0 && fpe < 1.0) {
        double Neff = -std::log(fpe) / (SIG_PE * Zd);
        if (Neff > N0_CR) c.n20_xi *= N0_CR / Neff;
    }

    // dust heating per H: local UV; optical = unattenuated FUV attenuated with tau_PE/1.87; blackbody
    c.n20_gd_uv = FOURPI * Zd * (SIG_PE * J_PE_D * c.n20_chi_pe + c.n20_sig_lw * J_LW_D * c.n20_chi_lw);
    double gd_fuv0 = FOURPI * Zd * (SIG_PE * J_PE_D * chi_pe0 + c.n20_sig_lw * J_LW_D * chi_lw0);
    double tb = c.n20_tbb;
    c.n20_gd_ext = gd_fuv0 * fopt + 2.0e-25 * Zd * sq(tb / 10.0) * CLIGHT * ARAD * sq(sq(tb));

    c.G_dust = 1.7 * c.n20_chi_fuv;
    c.chi_isrf = fpe;
    c.fac_shield_dust = fpe;
    c.fshield_CO = 1.0;
}

// ===========================================================================
int n20_solve_state(double xH2, double xHp, double energy, ChemCell& c, N20State& s)
{
    if (!(energy > 0.0) || !(c.yn > 0.0)) return 1;
    s.xH2 = std::min(std::max(xH2, 0.0), 0.5);
    s.xHp = std::min(std::max(xHp, 0.0), 1.0);
    s.xHI = std::max(1.0 - 2.0 * s.xH2 - s.xHp, 0.0);
    s.xCtot = std::max(c.abundc, 0.0);
    s.xOtot = std::max(c.abundo, 0.0);
    s.xOp = s.xOtot * s.xHp;

    Ctx q;
    q.nH = c.yn; q.xH2 = s.xH2; q.xHp = s.xHp; q.xCtot = s.xCtot; q.xOp = s.xOp;
    q.Zd = std::max(c.dust_to_gas_ratio, 0.0); q.Zg = c.Zg; q.xi = c.n20_xi;
    q.G0 = 1.7 * c.n20_chi_fuv;
    double chi_h2, chi_c;
    shield(c, s.xH2, &chi_h2, &chi_c);
    q.zetaC = ZETA_PIC0 * chi_c + 520.0 * 2.0 * s.xH2 * q.xi + 3.85 * q.xi;

    const double nbase = 1.0 + c.abhe - s.xH2;
    double xe = c.n20_xe;
    if (!(xe >= s.xHp)) xe = s.xHp + s.xCtot;
    double T = energy / (1.5 * KB * c.yn * (nbase + xe));
    double xcp = 0.0, xmh = 0.0;
    TCoef k;
    for (int it = 0; it < 60; it++) {
        tcoef(std::max(T, TFLOOR), k);
        double xe_new;
        if (T >= TT1) {
            double lhe, lme, xhe, xme;
            cie(T, &lhe, &lme, &xhe, &xme);
            xe_new = s.xHp + xhe + q.Zg * xme;
            xcp = xcp_of(xe_new, k, q);
            xmh = 0.0;
        } else {
            xe_new = solve_xe(k, q, xe, &xcp, &xmh);
        }
        double Tn = energy / (1.5 * KB * c.yn * (nbase + xe_new));
        int done = std::fabs(Tn - T) <= 1e-12 * T;
        T = Tn; xe = xe_new;
        if (done) break;
    }
    s.T = T; s.xe = xe; s.xCp = xcp; s.xMHp = xmh;

    // CO from the fit to equilibrium chemistry; C and O by conservation
    s.xCO = 0.0;
    if (s.xCtot > 0.0 && s.xH2 > 0.0 && q.Zd > 0.0) {
        double xi16 = std::max(q.xi * 1e16, 1e-6);
        double ncrit = std::pow(4e3 * q.Zd / (xi16 * xi16), std::cbrt(std::max(c.n20_chi_lw, 0.0)))
                     * (50.0 * xi16 / std::pow(q.Zd, 1.4));
        double fion = std::max(xcp / s.xCtot, (s.xOtot > 0.0) ? s.xOp / s.xOtot : 0.0);
        s.xCO = s.xCtot * 2.0 * s.xH2 * std::max(1.0 - fion, 0.0) / (1.0 + sq(ncrit / c.yn));
    }
    s.xC = std::max(s.xCtot - s.xCp - s.xCO, 0.0);
    s.xO = std::max(s.xOtot - s.xCO - s.xOp, 0.0);

    c.n20_xe = xe; c.n20_T = T;
    c.n20_xcp = s.xCp; c.n20_xc = s.xC; c.n20_xco = s.xCO; c.n20_xo = s.xO; c.n20_xop = s.xOp; c.n20_xmhp = s.xMHp;
    return 0;
}

// ===========================================================================
void n20_source_terms(const N20State& s, ChemCell& c, double* dxH2, double* dxHp, double* dedt)
{
    const double nH = c.yn, T = std::max(s.T, TFLOOR);
    TCoef k;
    tcoef(T, k);
    const double Zd = std::max(c.dust_to_gas_ratio, 0.0), Zg = c.Zg, xi = c.n20_xi;
    const double ne = s.xe * nH;
    const double G0 = 1.7 * c.n20_chi_fuv;
    double chi_h2, chi_c;
    shield(c, s.xH2, &chi_h2, &chi_c);
    c.fac_shield_h2 = (c.n20_chi_lw > 0.0) ? chi_h2 / c.n20_chi_lw : 1.0;

    // ---- H2 ----
    double kgr = 3.0e-17 * Zd;
    if (!c.n20_kgr_const) {
        double T2 = T * 1e-2, s2 = std::sqrt(T2);
        kgr *= 2.0 * s2 / (1.0 + 0.4 * s2 + 0.2 * T2 + 0.08 * T2 * T2);
    }
    const double zcr_h2 = 2.0 * xi * (2.3 * s.xH2 + 1.5 * s.xHI);
    const double zpd = ZETA_PD0 * chi_h2;
    const double zpi_h = c.rt_phot_HI, zpi_h2 = zpi_h * SIG_PI_RATIO;
    const double zcd = zeta_cd_h2(nH, T, s.xHI, s.xH2);
    *dxH2 = kgr * nH * s.xHI - (1.65 * zcr_h2 + zpd + zpi_h2 + zcd) * s.xH2;

    // ---- H+ ----
    const double zcr_h = 0.5 * zcr_h2;
    const double zci = k.k_ci * ne;
    double agr_h = 0.0;
    if (Zd > 0.0) {
        double psi_gr = G0 * k.sqT / (ne + 1e-50) + 1e-50;
        agr_h = alpha_gr(psi_gr, k.gr_h, k.lnT, CF_HP, Zd);
    }
    *dxHp = (zpi_h + zcr_h + zci) * s.xHI - (k.a_rr * ne + agr_h * nH) * s.xHp;

    // ---- heating per H [erg s^-1] ----
    const double psi = G0 * k.sqT / (ne + 1e-50) + 50.0;
    double g_pe = 0.0, l_grrec = 0.0;
    if (Zd > 0.0) {
        double eps = (5.22 + 2.25 * std::pow(T, 0.147))
                   / (1.0 + 0.04996 * std::pow(psi, 0.431) * (1.0 + 0.00430 * std::pow(psi, 0.692)));
        g_pe = 1e-26 * G0 * eps * Zd;
        double lp = std::log(psi);
        l_grrec = 1e-28 * Zd * s.xe * std::pow(T, 0.4535 + 2.234 / lp) * std::exp(-6.266 + (1.442 - 0.05089 * lp) * lp);
    }
    const double qH = (6.5 + 26.4 * std::sqrt(s.xe / (s.xe + 0.07))) * EV;
    const double g_cr = xi * (s.xHI * qH + 2.0 * s.xH2 * q_cr_h2(nH) * EV);
    const double g_pi = s.xHI * c.rt_heat_HI + s.xH2 * Q_PI_H2 * zpi_h2;
    double fcd = 0.0;
    {
        double t = 1.0 + T * 1e-3;
        double kH = std::pow(10.0, -11.06 + 0.0555 / t - 2.390 / (t * t));
        double kH2 = std::pow(10.0, -11.08 - 3.671 / t - 2.023 / (t * t));
        double dcol = (s.xHI * kH + s.xH2 * kH2) * nH;
        if (dcol > 0.0) fcd = 1.0 / (1.0 + (2.0e-7 + zpd) / dcol);
    }
    const double g_form = kgr * nH * s.xHI * (0.2 + 4.2 * fcd) * EV;
    const double g_pump = zpd * s.xH2 * 8.0 * 2.0 * fcd * EV;
    const double g_pd = zpd * s.xH2 * 0.4 * EV;

    // ---- cooling per nH^2 [erg cm^3 s^-1] ----
    const double T4 = k.T4;
    const double k12 = 5.31e-8 * std::pow(T4, 0.15) / (1.0 + std::pow(T4 / 5.0, 0.65)) * std::exp(-11.84 / T4);
    const double l_lya = s.xHI * s.xe * KB * 1.184e5 * k12;
    const double l_ci = 13.6 * EV * s.xHI * s.xe * k.k_ci;
    const double l_rovib = cool_h2rovib(nH, T, s.xHI, s.xH2) / nH;
    const double l_cd = 4.48 * EV * s.xH2 * zcd / nH;
    const double gff = 1.0 + 0.44 / (1.0 + 0.058 * sq(std::log(T / std::pow(10.0, 5.4))));
    const double l_ff = 1.422e-25 * gff * std::sqrt(T4) * s.xe * s.xHp;
    const double l_rr = (0.684 - 0.0416 * std::log(T4)) * KB * T * k.a_rr * s.xe * s.xHp;

    const double l_cii = cool_cii(nH, T, s.xe, s.xHI, s.xH2, s.xCp) / nH;
    const double l_c   = cool_ci(nH, T, s.xe, s.xHI, s.xH2, s.xC) / nH;
    const double l_o   = cool_oi(nH, T, s.xe, s.xHI, s.xH2, s.xO) / nH;
    const double l_co  = cool_co(nH, T, s.xe, s.xHI, s.xH2, s.xCO, c.dvdr) / nH;
    double l_neb = 0.0;
    {
        double L = std::log(T4);
        double lf = 6.92e-1 + L * (-5.86e-1 + L * (8.16e-1 + L * (-5.05e-1 + L * (1.18e-1 + L * (7.66e-3 + L * (-5.08e-3))))));
        double ne2 = ne * 1e-2;
        l_neb = Zg * s.xe * s.xHp * 3.68e-23 * std::exp(-3.86 / T4) * std::pow(10.0, lf)
              / (std::sqrt(T4) * (1.0 + 0.12 * std::pow(ne2, 0.38 - 0.12 * L)));
    }
    c.tdust = dust_temp(c, T, Zd);
    const double l_gd = ALPHA_GD * std::sqrt(T) * (T - c.tdust) * Zd;
    double lhe, lme, xhe, xme;
    cie(T, &lhe, &lme, &xhe, &xme);
    const double l_cie = lhe + Zg * lme;
    const double S = 1.0 / (1.0 + std::exp(-10.0 * (T - 0.5 * (TT1 + TT2)) / (TT2 - TT1)));

    // ---- diagnostics [erg cm^-3 s^-1]; heating negative ----
    const double n2 = nH * nH, w = 1.0 - S;
    double* L = c.lambda;
    double* LC = c.lambda_chem;
    for (int i = 0; i < CHEM_NRATES; i++) L[i] = 0.0;
    for (int i = 0; i < CHEM_NRATES_CHEM_MAX; i++) LC[i] = 0.0;
    L[0]  = w * l_gd * n2;
    L[1]  = l_rovib * n2;
    L[2]  = w * l_neb * n2;
    L[6]  = w * l_co * n2;
    L[10] = -g_cr * nH;
    L[11] = -g_pe * nH;
    L[12] = w * l_o * n2;
    L[13] = w * l_c * n2;
    L[15] = w * l_cii * n2;
    L[16] = w * l_grrec * n2;
    L[24] = l_lya * n2;
    L[25] = S * l_cie * n2;
    L[27] = l_ff * n2;
    LC[0] = l_cd * n2;
    LC[1] = -g_pd * nH;
    LC[2] = -g_pump * nH;
    LC[3] = -g_form * nH;
    LC[4] = l_ci * n2;
    LC[5] = l_rr * n2;
    LC[6] = -g_pi * nH;

    double sum = 0.0;
    for (int i = 0; i < CHEM_NRATES; i++) sum += L[i];
    for (int i = 0; i < C20_NRATES_CHEM; i++) sum += LC[i];
    *dedt = -sum;
}
