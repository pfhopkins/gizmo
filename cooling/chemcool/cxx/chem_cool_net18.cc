// chem_cool_net18.cc -- network 18: RTZ (mem_opt) ion rates, H2/CO, heating and cooling as a CVODE system.
// Species per H; neutral stages from the element pools and n_e from the ion charges, so every element and the
// charge are conserved by construction.  Provenance and changes: DEVIATIONS.md D-51.

#include "chem_net18.h"
#include "chem_net18_tables.h"
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sundials/sundials_context.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// RTZ constants.h
const double KB = 1.380649E-16, C_CGS = 2.9979246E10, H_PLANCK = 6.626196E-27, EV2ERG = 1.602E-12;
const double T_MIN = 2.725, T_MAX = 1E9;
const double MIN_METAL_COOL_DENS = 1E-10, MIN_COOL_ION = 1E-10;
// sub-ionizing (G0) photoionization of the neutral stage [s^-1 per G0] (rtz_main.c initialize_elements)
const double G0_PHOTO[N18_NEL] = {0, 0, 3.39E-10, 0, 0, 0, 6.59E-11, 4.47E-9, 1.13E-9, 4.71E-10};

inline int slot(int e, int j) { return n18_toff[e] + j; }

// ---------------------------------------------------------------- rates (T only)
double rad_rec(int iz, int in, double t)   // recombination.c cloudy_rad_rec
{
    double tt, rate;
    if ((((in <= 3 || in == 11) || (iz > 5 && iz < 9)) || iz == 10) || (iz == 26 && in > 11)) {
        tt = std::sqrt(t / n18_rnew[in - 1][iz - 1][2]);
        rate = n18_rnew[in - 1][iz - 1][0] / (tt * std::pow(tt + 1.0, 1.0 - n18_rnew[in - 1][iz - 1][1]) *
               std::pow(1.0 + std::sqrt(t / n18_rnew[in - 1][iz - 1][3]), 1.0 + n18_rnew[in - 1][iz - 1][1]));
    } else {
        tt = t * 1.0e-04;
        if (iz == 26 && in <= 13) rate = n18_rfe[in - 1][0] / std::pow(tt, n18_rfe[in - 1][1] + n18_rfe[in - 1][2] * std::log10(tt));
        else rate = n18_rrec[in - 1][iz - 1][0] / std::pow(tt, n18_rrec[in - 1][iz - 1][1]);
    }
    return rate;
}

double recomb(double T, double t32, int ion, int e)   // recombination.c recombination (ion = charge before recombining); t32 = T^-1.5
{
    const int Z = n18_Z[e];
    double rec = rad_rec(Z, Z - ion + 1, T);
    if (e >= 1) {
        const double* de = n18_dr_e[e - 1][ion];
        const double* dc = n18_dr_c[e - 1][ion];
        for (int i = 0; i < 9; i++) if (de[i] > 0.0) rec += t32 * dc[i] * std::exp(-de[i] / T);
    }
    return rec;
}

double collion(double T, int ion, int Z)   // collisional_ionization.c (Voronov 1997)
{
    double col = 0.0;
    const int nel = Z - ion;
    if (nel > 0) {
        const double* d = n18_ci[Z][nel];
        double U = d[0] / (T * 8.61732814974056E-05);
        if (U <= 80.0) col = d[2] * (1. + d[1] * std::sqrt(U)) * std::pow(U, d[4]) * std::exp(-U) / (d[3] + U);
    }
    return std::fmax(col, 1e-100);
}

double ct_rec(int ion, int Z, double T)   // charge_transfer.c: X^ion + H -> X^(ion-1) + H+
{
    if (ion == 0) return 0.0;
    const int ip = ion - 1;
    if (ion > 4) return 1.92E-9 * (double)(ip + 1);
    double tu = std::fmin(std::fmax(T, n18_ctrec[4][ip][Z]), n18_ctrec[5][ip][Z]) * 1E-4;
    tu = std::fmax(tu, 1E-10);
    double r = n18_ctrec[0][ip][Z] * 1E-9 * std::pow(tu, n18_ctrec[1][ip][Z]) * (1. + n18_ctrec[2][ip][Z] * std::exp(n18_ctrec[3][ip][Z] * tu));
    return std::fmax(r, 1E-100);
}

double ct_ion(int ion, int Z, double T)   // charge_transfer.c: X^ion + H+ -> X^(ion+1) + H
{
    const int ip = ion;
    if (ip > 1) return 0.0;
    double tu = std::fmin(std::fmax(T, n18_ction[4][ip][Z]), n18_ction[5][ip][Z]) * 1E-4;
    tu = std::fmax(tu, 1E-10);
    double r = n18_ction[0][ip][Z] * 1E-9 * std::pow(tu, n18_ction[1][ip][Z]) * (1. + n18_ction[2][ip][Z] * std::exp(n18_ction[3][ip][Z] * tu))
             * std::exp(-1.0 * n18_ction[6][ip][Z] / tu);
    return std::fmax(r, 1E-100);
}

const double* hit_table(int e)
{
    switch (n18_Z[e]) {
        case 6: return &n18_hit_6[0][0];   case 7: return &n18_hit_7[0][0];   case 8: return &n18_hit_8[0][0];
        case 10: return &n18_hit_10[0][0]; case 12: return &n18_hit_12[0][0]; case 14: return &n18_hit_14[0][0];
        case 16: return &n18_hit_16[0][0]; default: return &n18_hit_26[0][0];
    }
}

// ions whose high-T rate is blended with fine-structure cooling (cooling.c high_t_cooling_rates_tflag)
bool hit_tflag(int e, int j)
{
    const int Z = n18_Z[e];
    return (Z == 6 && j <= 1) || (Z == 7 && j == 1) || (Z == 8 && (j == 0 || j == 2)) || (Z == 10 && j == 1) ||
           (Z == 14 && j <= 1) || (Z == 16 && j == 0) || (Z == 26 && j <= 1);
}

// ---------------------------------------------------------------- fine structure (cooling.c)
// RTZ file order: CII158 CI609 CI230 CI370 NII205 NII76 NII122 OI63 OI44 OI145 NeII13 SiII35 OIII88 OIII33 OIII52
//                 FeI14 FeI24 FeI35 FeII26 FeII15 FeII35 SiI130 SiI45 SiI68 SI17 SI25 SI56
// colliders: 0 e, 1 H+, 2 He+, 3 He++, 4 H, 5 He, 6 o-H2, 7 p-H2
struct FSIon { int nlev; double g0, g1, g2, l10, l20, l21, A10, A20, A21; int q10, q20, q21, e, j, lam; };
const FSIon FS[11] = {
    {3, 1, 3, 5, 609.590, 230.352, 370.269, 7.930E-8, 1.000E-30, 2.650E-7, 1, 2, 3, 2, 0, 13},          // CI
    {2, 2, 4, 0, 157.636, 0, 0, 2.290E-6, 0, 0, 0, -1, -1, 2, 1, 15},                                    // CII
    {3, 1, 3, 5, 205.244, 76.4318, 121.767, 2.080E-6, 1.000E-30, 7.460E-6, 4, 5, 6, 3, 1, 9},           // NII
    {3, 5, 3, 1, 63.1679, 44.0453, 145.495, 8.910E-5, 1.340E-10, 1.750E-5, 7, 8, 9, 4, 0, 12},          // OI
    {3, 1, 3, 5, 88.3323, 32.6523, 51.8004, 2.597E-5, 3.170E-11, 9.760E-5, 12, 13, 14, 4, 2, 9},       // OIII
    {2, 4, 2, 0, 12.8101, 0, 0, 8.590E-3, 0, 0, 10, -1, -1, 5, 1, 9},                                    // NeII
    {3, 1, 3, 5, 129.641, 44.7993, 68.4548, 8.250E-6, 3.490E-10, 4.210E-5, 21, 22, 23, 7, 0, 14},      // SiI
    {2, 2, 4, 0, 34.8046, 0, 0, 2.131E-4, 0, 0, 11, -1, -1, 7, 1, 16},                                   // SiII
    {3, 5, 3, 1, 25.2421, 17.4278, 56.2957, 1.400E-3, 7.050E-8, 3.020E-4, 25, 24, 26, 8, 0, 9},        // SI (q10=25um, q20=17um)
    {3, 9, 7, 5, 24.0358, 14.2005, 34.7038, 2.510E-3, 1.000E-30, 1.560E-3, 16, 15, 17, 9, 0, 9},       // FeI (q10=24um, q20=14um)
    {3, 10, 8, 6, 25.9811, 14.9731, 35.3394, 2.050E-3, 1.000E-30, 1.560E-3, 18, 19, 20, 9, 1, 9},      // FeII
};

// collider column in the RTZ file; SiI and SI hold H in column 1 and H+ in column 0 (no electron rates)
inline int fs_col(int line, int coll)
{
    if (line >= 21) { if (coll == 4) return 1; if (coll == 1) return 0; if (coll == 0) return -1; }
    return coll;
}

// ---------------------------------------------------------------- T-only coefficient set
struct Coef {
    double T, z;
    double ci[N18_NTRAC], rec[N18_NTRAC], ctr[N18_NTRAC], cti[N18_NTRAC], hit[N18_NTRAC];
    double ciH, recH;
    int dr_on[N18_NEL]; double dr_sf, dr_lT, dr_sqrtT, dr_a2[N18_NEL], dr_a3[N18_NEL];
    double cHI, cHII, cHeI, cHeII, cHeIII, cbrem, ccomp;
    double h2f[4], k1, k2, k5, k13, k14, k15, adust, bH, bH2, be, bHe, ncrit;
    double f1, f2;
    double fsq[27][8];
    double fsx[11][3], fsB[11][3];    // exp(-E/T), CMB B_nu per transition
    double sqT, drc_beta, drc_pref, dgc, pe_t7;
    double co_a, co_vH2, co_vH;
};

// ionization, recombination and charge-exchange coefficients: exact, or log-log interpolated in a table
const int NTAB = 3441, NION = 4 * N18_NTRAC + 2;
const double LT0 = 0.4, DLT = 0.0025;   // log10 T grid 0.4 .. 9.0

void ion_exact(double T, double* v)   // v: ci[113] rec[113] ctr[113] cti[113] ciH recH
{
    const double t32 = std::pow(T, -1.5);
    for (int i = 0; i < NION; i++) v[i] = 0.0;
    for (int e = 1; e < N18_NEL; e++) {
        const int Z = n18_Z[e];
        for (int j = 0; j <= Z; j++) {
            const int s = slot(e, j);
            v[s]                 = (j < Z) ? collion(T, j, Z) : 0.0;
            v[N18_NTRAC + s]     = (j > 0) ? recomb(T, t32, j, e) : 0.0;
            v[2 * N18_NTRAC + s] = ct_rec(j, Z, T);
            v[3 * N18_NTRAC + s] = ct_ion(j, Z, T);
        }
    }
    v[4 * N18_NTRAC] = collion(T, 0, 1);
    v[4 * N18_NTRAC + 1] = recomb(T, t32, 1, 0);
}

const double* ion_table()   // log10 of the coefficients, built once (thread-safe static init)
{
    static const std::vector<double> tab = [] {
        std::vector<double> t((size_t)NTAB * NION);
        double v[NION];
        for (int i = 0; i < NTAB; i++) {
            ion_exact(std::pow(10.0, LT0 + i * DLT), v);
            for (int m = 0; m < NION; m++) t[(size_t)i * NION + m] = std::log10(std::fmax(v[m], 1e-300));
        }
        return t;
    }();
    return tab.data();
}

void coef_fill(Coef& k, double T, double z, int exact)
{
    k.T = T; k.z = z;
    double v[NION];
    if (exact) ion_exact(T, v);
    else {
        const double* tab = ion_table();
        double x = (std::log10(T) - LT0) / DLT;
        int i = std::min(std::max((int)std::floor(x), 0), NTAB - 2);
        double fh = x - i, fl = 1.0 - fh;
        const double* a = tab + (size_t)i * NION; const double* b = a + NION;
        for (int m = 0; m < NION; m++) {
            double l = fl * a[m] + fh * b[m];
            v[m] = (l < -299.0) ? 0.0 : std::exp(2.302585092994046 * l);
        }
    }
    std::memcpy(k.ci, v, sizeof(k.ci)); std::memcpy(k.rec, v + N18_NTRAC, sizeof(k.rec));
    std::memcpy(k.ctr, v + 2 * N18_NTRAC, sizeof(k.ctr)); std::memcpy(k.cti, v + 3 * N18_NTRAC, sizeof(k.cti));
    k.ciH = v[4 * N18_NTRAC]; k.recH = v[4 * N18_NTRAC + 1];
    for (int s = 0; s < N18_NTRAC; s++) k.hit[s] = 0.0;

    // grain recombination T parts (dust_recombination.c)
    k.dr_sf = (T > 1E4) ? 0.0 : (T > 1E3) ? std::exp(-1.0 * T / 1E3) / std::exp(-1.0) : 1.0;
    double lT = std::fmax(T, 10.0);
    k.dr_lT = lT; k.dr_sqrtT = std::sqrt(lT);
    for (int e = 0; e < N18_NEL; e++) {
        const double* d = n18_dustrec[n18_Z[e]];
        double rs = 0; for (int i = 0; i < 7; i++) rs += std::fabs(d[i]);
        k.dr_on[e] = (rs > 0.0);
        k.dr_a2[e] = d[3] * std::pow(lT, d[4]);
        k.dr_a3[e] = (-1.0 * d[5]) - (d[6] * std::log(lT));
    }

    // primordial cooling coefficients (cooling.c)
    const double sT = std::sqrt(T), q5 = 1.0 / (1.0 + std::sqrt(T / 1E5));
    k.cHI = std::fmax(1.27E-21 * sT * q5 * std::exp(-157809.1 / T), 1e-100) + std::fmax(4.13e-19 * std::exp(-117744.0 / T), 1e-100);
    { double l = 315614.0 / T; k.cHII = std::fmax(3.435E-30 * T * std::pow(l, 1.97) * std::pow(1.0 + std::pow(l / 2.25, 0.376), -3.72), 1e-100); }
    k.cHeI = std::fmax(9.38E-22 * sT * q5 * std::exp(-285335.4 / T), 1e-100);
    {
        double ci2 = std::fmax(4.95E-22 * sT * q5 * std::exp(-631515.0 / T), 1e-100);
        double ex2 = std::fmax(5.54E-17 * std::pow(T, -0.397) * q5 * std::exp(-473638.0 / T), 1e-100);
        double rc2 = std::fmax(KB * T * 1.26E-14 * std::pow(570670.0 / T, 0.75), 1e-100);
        double dr2 = std::fmax(1.24E-13 / (T * sT) * std::exp(-470000.0 / T) * (1.0 + 0.3 * std::exp(-94000.0 / T)), 1e-100);
        k.cHeII = ci2 + ex2 + rc2 + dr2;
    }
    { double l = 1263030.0 / T; k.cHeIII = std::fmax(8.0 * 3.435E-30 * T * std::pow(l, 1.97) * std::pow(1.0 + std::pow(l / 2.25, 0.376), -3.72), 1e-100); }
    k.cbrem = std::fmax(1.42E-27 * sT, 1e-100);
    { double a = 1.0 / (1.0 + z), ai = 2.727 / a; k.ccomp = 1.017E-37 * ai * ai * ai * ai * (T - ai); }

    // H2 (molecules.c, cooling.c H2_cooling)
    {
        double T3 = 1E-3 * T, T315 = T3 * std::sqrt(T3);
        k.h2f[0] = 1.1E-25 * std::sqrt(T3) * std::exp(-0.51 / T3);
        k.h2f[1] = 2.0E-25 * T3 * std::exp(-1.0 / T3);
        k.h2f[2] = 2.4E-24 * T315 * std::exp(-2.0 / T3);
        k.h2f[3] = 1.7E-23 * T315 * std::exp(-4.0 / T3);
        double logT = std::log10(T), Te = T * 8.621738E-5, lnTe = std::log(Te);
        if (T < 6000.0) k.k1 = std::pow(10.0, -17.845 + logT * (0.762 + logT * (0.1523 - 0.03274 * logT)));
        else { double l2 = logT * logT; k.k1 = std::pow(10.0, -16.420 + l2 * (0.1998 + l2 * (-5.447E-3 + 4.0415E-5 * l2))); }
        k.k2 = 4.0E-9 * std::pow(std::fmax(T, 300.0), -0.17);
        k.k5 = 2.4E-6 / sT * (1.0 + T / 20000.);
        double k13 = -1.801849334E1 + lnTe * (2.36085220E0 + lnTe * (-2.82744300E-1 + lnTe * (1.62331664E-2 + lnTe * (-3.36501203E-2 + lnTe * (1.17832978E-2 + lnTe * (-1.65619470E-3 + lnTe * (1.06827520E-4 - 2.63128581E-6 * lnTe)))))));
        k.k13 = std::exp(std::fmax(k13, -92.0));
        double k14 = 1.357772745525155 * 2.5634E-15 * std::pow(Te, 1.78186);
        if (T > 1160.0) {
            k14 = -3.388464953E1 + lnTe * (1.13944933E0 + lnTe * (-1.4210135E-1 + lnTe * (8.4644554E-3 + lnTe * (-1.4328641E-3 + lnTe * (2.0122503E-4 + lnTe * (8.6639632E-5 + lnTe * (-2.5850097E-5 + lnTe * (2.4555012E-6 - 8.0683825E-8 * lnTe))))))));
            k14 = std::exp(k14);
        }
        k.k14 = k14;
        k.k15 = (T > 8000.0) ? 9.6E-7 * std::pow(T, -0.90) : 6.9E-9 * std::pow(T, -0.35);
        k.adust = (3.5E-17) * std::sqrt(std::fmin(T / 100.0, 1e2));
        k.bH  = 6.67E-12 * sT * std::exp(-1.0 * (1.0 + (63593.0 / T)));
        k.bH2 = 5.996E-30 * std::pow(T, 4.1881) * std::pow(1.0 + (6.761E-6 * T), -5.6881) * std::exp(-54657.4 / T);
        k.be  = 4.38E-10 * std::pow(T, 0.35) * std::exp(-102000.0 / T);
        k.bHe = std::pow(10.0, -27.029 + (3.801 * logT) - (29487.0 / T));
        k.ncrit = 1.1E5 / std::sqrt(T / 1000.0);
    }

    // high-temperature ion cooling (cooling.c get_high_t_cooling_rates)
    k.f1 = 0.5 * (std::tanh((5E-3) * (T - 1.E4)) + 1.0);
    k.f2 = 0.5 * (std::tanh((5E-3) * ((-1.0 * T) + 1.E4)) + 1.0);
    {
        double locT = std::fmax(T, 1000.0), lgT = std::log10(locT);
        double tscale = std::exp(-1.0 * std::pow(2000.0 / locT, 5.0));
        if (lgT >= 9.0) lgT = 9.0 - 1e-9;
        int il = (int)std::floor((lgT - 3.0) / 0.05);
        double fh = (lgT - n18_hit_logT[il]) / (n18_hit_logT[il + 1] - n18_hit_logT[il]), fl = 1.0 - fh;
        for (int e = 2; e < N18_NEL; e++) {
            const double* tab = hit_table(e);
            for (int j = 0; j <= n18_Z[e]; j++) {
                double r = std::exp(2.302585092994046 * (fl * tab[j * 121 + il] + fh * tab[j * 121 + il + 1]));
                k.hit[slot(e, j)] = r * (hit_tflag(e, j) ? k.f1 : 1.0) * tscale;
            }
        }
    }

    // fine-structure collision rates (two_level/three_level interpolation) and level factors
    {
        double lg = std::fmin(std::fmax(std::log10(T), 1.0), 4.975 - 1e-9);
        int il = (int)std::floor((lg - 1.0) / 0.025);
        double fh = (lg - (1.0 + il * 0.025)) / 0.025, fl = 1.0 - fh;
        for (int L = 0; L < 27; L++)
            for (int cl = 0; cl < 8; cl++) {
                int col = fs_col(L, cl);
                k.fsq[L][cl] = (col < 0) ? 0.0 : n18_fs[L][il][col] * fl + n18_fs[L][il + 1][col] * fh;
            }
        const double Tcmb = 2.725 * (1.0 + z);
        for (int i = 0; i < 11; i++) {
            const double lam[3] = {FS[i].l10, FS[i].l20, FS[i].l21};
            for (int t = 0; t < 3; t++) {
                if (lam[t] <= 0) { k.fsx[i][t] = 0; k.fsB[i][t] = 0; continue; }
                double nu = C_CGS / (lam[t] * 1E-4), E = (H_PLANCK * C_CGS / (lam[t] * 1E-4)) / KB;
                k.fsx[i][t] = std::exp(-1.0 * E / T);
                k.fsB[i][t] = (2.0 * H_PLANCK * (nu * nu * nu) / (C_CGS * C_CGS)) / (std::exp(H_PLANCK * nu / (KB * Tcmb)) - 1.0);
            }
        }
    }

    // dust, PE, CO parts
    k.sqT = sT;
    k.drc_beta = 0.74 / std::pow(T, 0.068);
    k.drc_pref = (1.5 * 4.65E-30) * std::pow(T, 0.94);
    k.dgc = sT * (1.0 - (0.80 * std::exp(-75.0 / T)));
    k.pe_t7 = std::pow(T / 1E4, 0.7);
    {
        double Ts = T / 1.0e3, Tp = 3080.0;
        k.co_a = 3.3e6 * std::sqrt(Ts * std::sqrt(Ts));
        k.co_vH2 = Tp * KB * 4.3e-14 * T * std::exp(-std::cbrt(3.14e5 / T)) * std::exp(-Tp / T);
        k.co_vH = Tp * KB * 3.0e-12 * sT * std::exp(-std::pow(2000.0 / T, 3.43)) * std::exp(-Tp / T);
    }
}

// three_level / two_level (cooling.c), rates and level factors from Coef
double fs_cool(const Coef& k, int i, double n_ion, const double* nc /*8 collider densities*/)
{
    const FSIon& F = FS[i];
    double C10 = 0;
    for (int cl = 0; cl < 8; cl++) C10 += k.fsq[F.q10][cl] * nc[cl];
    const double E10 = (H_PLANCK * C_CGS / (F.l10 * 1E-4)) / KB, l10 = F.l10 * 1E-4;
    const double B01 = F.A10 * (l10 * l10 * l10) * (F.g1 / F.g0) / (2.0 * H_PLANCK * C_CGS), B10 = (F.g0 / F.g1) * B01;
    const double Bn10 = k.fsB[i][0];
    if (F.nlev == 2) {
        double C01 = C10 * (F.g1 / F.g0) * k.fsx[i][0];
        double t1 = (B01 * Bn10) + C01, t2 = F.A10 + (B10 * Bn10) + C10;
        double r = t1 / t2, n0 = 1.0 / (r + 1.0), n1 = 1.0 / ((1.0 / r) + 1.0);
        return (F.A10 + (B10 * Bn10)) * E10 * KB * n1 * n_ion - B01 * Bn10 * E10 * KB * n0 * n_ion;
    }
    double C20 = 0, C21 = 0;
    for (int cl = 0; cl < 8; cl++) { C20 += k.fsq[F.q20][cl] * nc[cl]; C21 += k.fsq[F.q21][cl] * nc[cl]; }
    const double E20 = (H_PLANCK * C_CGS / (F.l20 * 1E-4)) / KB, E21 = (H_PLANCK * C_CGS / (F.l21 * 1E-4)) / KB;
    const double l20 = F.l20 * 1E-4, l21 = F.l21 * 1E-4;
    const double B02 = F.A20 * (l20 * l20 * l20) * (F.g2 / F.g0) / (2.0 * H_PLANCK * C_CGS);
    const double B12 = F.A21 * (l21 * l21 * l21) * (F.g2 / F.g1) / (2.0 * H_PLANCK * C_CGS);
    const double B20 = (F.g0 / F.g2) * B02, B21 = (F.g1 / F.g2) * B12;
    const double Bn20 = k.fsB[i][1], Bn21 = k.fsB[i][2];
    double C01 = C10 * (F.g1 / F.g0) * k.fsx[i][0], C02 = C20 * (F.g2 / F.g0) * k.fsx[i][1], C12 = C21 * (F.g2 / F.g1) * k.fsx[i][2];
    C01 += B01 * Bn10; C02 += B02 * Bn20; C12 += B12 * Bn21;
    C10 += B10 * Bn10; C20 += B20 * Bn20; C21 += B21 * Bn21;
    double n2n1 = ((C12 * (C01 + C02)) + (C02 * (F.A10 + C10))) / (((F.A21 + C21 + C20) * (C01 + C02)) - (C20 * C02));
    double n1n0 = (((F.A21 + C21 + C20) * (C01 + C02)) - (C20 * C02)) / (((F.A21 + C21 + C20) * (F.A10 + C10)) + (C20 * C12));
    double n0 = 1.0 / (1.0 + n1n0 + (n2n1 * n1n0));
    double n1 = 1.0 / (1.0 + n2n1 + (1.0 / n1n0));
    double n2 = 1.0 / (1.0 + (1.0 / n2n1) + ((1.0 / n2n1) * (1.0 / n1n0)));
    double cool = (F.A10 + (B10 * Bn10)) * E10 * KB * n1 * n_ion + (F.A20 + (B20 * Bn20)) * E20 * KB * n2 * n_ion
                + (F.A21 + (B21 * Bn21)) * E21 * KB * n2 * n_ion;
    double heat = B01 * Bn10 * E10 * KB * n0 * n_ion + B02 * Bn20 * E20 * KB * n0 * n_ion + B12 * Bn21 * E21 * KB * n1 * n_ion;
    return cool - heat;
}

// charge-exchange heating terms (cooling.c CT_heat_cool): element, stage, 0 = with H (recombination) / 1 = with H+, dE [eV]
struct CTH { int e, j, ion; double dE; };
const CTH CTHEAT[] = {
    {1, 1, 0, 10.99},
    {2, 0, 1, 2.34}, {2, 1, 0, -2.34}, {2, 2, 0, 4.01}, {2, 3, 0, 5.73}, {2, 4, 0, 11.3},
    {3, 0, 1, -0.94}, {3, 1, 0, 0.94}, {3, 2, 0, 4.56}, {3, 3, 0, 6.40}, {3, 4, 0, 11.0},
    {4, 0, 1, -0.02}, {4, 1, 0, 0.02}, {4, 2, 0, 6.65}, {4, 3, 0, 5.00}, {4, 4, 0, 8.47},
    {5, 3, 0, 5.82}, {5, 4, 0, 8.60},
    {6, 0, 1, 1.52}, {6, 1, 1, -1.44}, {6, 2, 0, 1.44}, {6, 3, 0, 5.73}, {6, 4, 0, 8.60},
    {7, 0, 1, 0.12}, {7, 1, 1, -2.72}, {7, 2, 0, 2.72}, {7, 3, 0, 4.23}, {7, 4, 0, 7.49},
    {8, 1, 0, -3.24}, {8, 3, 0, 5.73}, {8, 4, 0, 8.60},
    {9, 1, 1, -2.56}, {9, 2, 0, 2.56}, {9, 3, 0, 6.30}, {9, 4, 0, 10.0},
};

// ---------------------------------------------------------------- per-thread state
thread_local Coef tl_k, tl_k2;
thread_local double tl_uvb_z = -1.0, tl_uvb_pi[N18_NTRAC + 1], tl_uvb_ph[N18_NTRAC + 1];   // H stage 0 at N18_NTRAC
thread_local long st_calls = 0, st_steps = 0, st_rhs = 0, st_jac = 0;
thread_local double tl_fsh = -1.0;   // H2 self-shielding held fixed while the Jacobian is built (< 0: live)

void uvb_update(double z)   // photoionization_UVB.c update_UVB
{
    if (z == tl_uvb_z) return;
    tl_uvb_z = z;
    for (int i = 0; i <= N18_NTRAC; i++) tl_uvb_pi[i] = tl_uvb_ph[i] = 0.0;
    if (z < n18_uvb_z[0] || z >= n18_uvb_z[59]) return;
    int i = 0; while (!(z >= n18_uvb_z[i] && z < n18_uvb_z[i + 1])) i++;
    double sh = (z - n18_uvb_z[i]) / (n18_uvb_z[i + 1] - n18_uvb_z[i]), sl = 1.0 - sh;
#define N18_UVB_EL(Z, e) for (int j = 0; j < Z; j++) { int s = (e == 0) ? N18_NTRAC : slot(e, j); \
        tl_uvb_pi[s] = sl * n18_uvb_pi_##Z[i][j] + sh * n18_uvb_pi_##Z[i + 1][j]; \
        tl_uvb_ph[s] = sl * n18_uvb_ph_##Z[i][j] + sh * n18_uvb_ph_##Z[i + 1][j]; }
    N18_UVB_EL(1, 0) N18_UVB_EL(2, 1) N18_UVB_EL(6, 2) N18_UVB_EL(7, 3) N18_UVB_EL(8, 4)
    N18_UVB_EL(10, 5) N18_UVB_EL(12, 6) N18_UVB_EL(14, 7) N18_UVB_EL(16, 8) N18_UVB_EL(26, 9)
#undef N18_UVB_EL
}

thread_local int tl_k_exact = -1;
const Coef& coef_at(double T, double z, int exact)
{
    if (tl_k.T != T || tl_k.z != z || tl_k_exact != exact) { coef_fill(tl_k, T, z, exact); tl_k_exact = exact; }
    return tl_k;
}

// ODE slot k of an element holds stage k (k < ref) or k + 1; the reference stage (most abundant at entry) follows
// from the pool, which keeps the cancellation error of the derived stage small
inline int stage_of(int k, int r) { return k < r ? k : k + 1; }
inline double elem_pool(const ChemCell& c, int e, double xco) { return (e == 1 ? c.abhe : c.n18_pool[e]) - ((e == 2 || e == 4) ? xco : 0.0); }

// H+ and HI per H; ODE slot 1 holds the less abundant of the two (n18_ref[0] = 0: H+, 1: HI)
inline void hydrogen(const double* y, const ChemCell& c, double* xHI, double* xHp)
{
    const double h2 = std::fmax(y[0], 0.0), v = std::fmin(std::fmax(y[1], 0.0), 1.0);
    if (c.n18_ref[0] == 0) { *xHp = v; *xHI = std::fmax(1.0 - 2.0 * h2 - v, 0.0); }
    else                   { *xHI = v; *xHp = std::fmax(1.0 - 2.0 * h2 - v, 0.0); }
}

// stage abundances per H of He..Fe (TracAbund slots), clipped at 0; returns x_e
double stages(const double* y, const ChemCell& c, double* s)
{
    double xHI_, xe;
    hydrogen(y, c, &xHI_, &xe);
    for (int e = 1; e < N18_NEL; e++) {
        const int Z = n18_Z[e], r = c.n18_ref[e];
        double sum = 0;
        for (int k = 0; k < Z; k++) { double v = y[n18_ooff[e] + k]; sum += v; s[slot(e, stage_of(k, r))] = std::fmax(v, 0.0); }
        s[slot(e, r)] = std::fmax(elem_pool(c, e, y[2]) - sum, 0.0);
        for (int j = 1; j <= Z; j++) xe += j * s[slot(e, j)];
    }
    return xe;
}

double particles(const double* y, const ChemCell& c, double* xe_out)
{
    double s[N18_NTRAC], xe = stages(y, c, s), met = 0;
    for (int e = 2; e < N18_NEL; e++) met += c.n18_pool[e];
    if (xe_out) *xe_out = xe;
    return 1.0 - std::fmax(y[0], 0.0) + c.abhe + met - std::fmax(y[2], 0.0) + xe;
}

// species and energy derivatives at state y, rates at temperature T (coefficients k)
void assemble(const double* y, double T, const Coef& k, ChemCell& c, double* f)
{
    const double nH = c.yn, abhe = c.abhe;
    const double a = 1.0 / (1.0 + c.redshift);
    for (int i = 0; i <= N18_NODE; i++) f[i] = 0.0;

    // state: stage abundances per H (TracAbund layout), neutral stages from the pools
    double s[N18_NTRAC];
    const double xH2 = std::fmax(y[0], 0.0), xCO = std::fmax(y[2], 0.0);
    double xHI, xHp;
    hydrogen(y, c, &xHI, &xHp);
    s[N18_T_H2] = xH2; s[N18_T_HP] = xHp; s[N18_T_CO] = xCO;
    const double xe = stages(y, c, s);
    (void)abhe;
    const double ne = xe * nH, nHI = xHI * nH, nHII = xHp * nH, nH2 = xH2 * nH;
    c.n18_xe = xe; c.n18_T = T;

    // radiation, cosmic rays, grains
    const double G = c.G0 * c.n18_df25;                        // dust-attenuated FUV [Habing]
    const double fsh = (c.iphoto == 0) ? 1.0 : (tl_fsh >= 0.0) ? tl_fsh : cxx_calc_H2_shielding_factor(T, xH2, c.dl * nH, c);
    const double kdiss = c.G0_LW * 5.68E-11 * c.fac_shield_dust * fsh;   // H2 photodissociation [s^-1]
    const double zeta = c.cosmic_ray_ion_rate;
    const double phis = (1.0 - (xe / 1.20)) * (0.670 / (1.0 + (xe / 0.05)));
    const double ztot = zeta * (1.0 + phis), zH2 = 2.0 * zeta * (1.0 + phis), crsc = zH2 / 1E-16;
    const double fdg = c.dust_to_gas_ratio, ndust = nH * fdg;
    const int uvb = (c.n18_flags & N18_UVB) != 0;
    const double ss = uvb ? std::exp(-1.0 * nH / 1E-2) : 0.0;

    // grain recombination of singly charged ions (dust_recombination.c), per element
    double drec[N18_NEL];
    for (int e = 0; e < N18_NEL; e++) {
        drec[e] = 0.0;
        if (!k.dr_on[e] || k.dr_sf == 0.0) continue;
        const double* d = n18_dustrec[n18_Z[e]];
        double phi = (G + 1E-8) * k.dr_sqrtT / (ne + 1E-10);
        double a1 = d[1] * std::pow(phi, d[2]);
        double r = 1E-14 * d[0] / (1.0 + (a1 * (1.0 + (k.dr_a2[e] * std::pow(phi, k.dr_a3[e])))));
        drec[e] = std::fmax(r * k.dr_sf * (3.3E-5 / 6.0E-5), 1E-100);
    }

    // ---- species ----
    double dHp = 0.0;
    {   // hydrogen
        double up = k.ciH * ne + (uvb ? tl_uvb_pi[N18_NTRAC] * ss : 0.0) + n18_cr[0][0] * ztot + c.rt_phot_HI;
        double dn = k.recH * ne + drec[0] * ndust;
        dHp += up * xHI - dn * xHp;
    }
    double ds[N18_NTRAC];
    for (int i = 0; i < N18_NTRAC; i++) ds[i] = 0.0;
    for (int e = 1; e < N18_NEL; e++) {
        const int Z = n18_Z[e];
        for (int j = 0; j <= Z; j++) {
            const int si = slot(e, j);
            const double x = s[si];
            if (j < Z) {
                double up = k.ci[si] * ne + n18_cr[e][j] * ztot + k.cti[si] * nHII;
                if (uvb) up += tl_uvb_pi[si] * ss;
                if (j == 0) up += G * G0_PHOTO[e] + n18_cr_uv[Z] * crsc;
                if (e == 1) up += (j == 0) ? c.rt_phot_HeI : c.rt_phot_HeII;
                double fl = up * x;
                ds[si] -= fl; ds[si + 1] += fl;
                dHp -= k.cti[si] * nHII * x;                 // X + H+ -> X+ + H
            }
            if (j > 0) {
                double dn = k.rec[si] * ne + k.ctr[si] * nHI;
                if (j == 1) dn += drec[e] * ndust;
                double fl = dn * x;
                ds[si] -= fl; ds[si - 1] += fl;
                dHp += k.ctr[si] * nHI * x;                  // X+ + H -> X + H+
            }
        }
    }
    // H2 (molecules.c; formation per H [s^-1], destruction rate [s^-1])
    double h2_form = 0.0, h2_dest = 0.0, hprim = 0.0;
    {
        double crp = (1.28E-13 * (zH2 / 1.E-16) + 5.9E-9 * G) / (nH + 1e-40);
        double coll = k.k2 * xHI + k.k5 * xHp + k.k13 * xe + k.k14 * xHI + k.k15 * xHp;
        hprim = k.k1 * k.k2 * xe * xHI / (coll + crp);
        h2_form = (fdg * k.adust + hprim) * xHI * nH;
        h2_dest = k.bH * nHI + k.bH2 * nH2 + k.be * ne + k.bHe * s[slot(1, 0)] * nH + kdiss + zH2;
        if (!(c.n18_flags & N18_NOH2)) f[0] = h2_form - h2_dest * xH2;
    }
    // CO (molecules.c alpha_CO / beta_CO; CHx branching with n_H)
    double co_form = 0.0, co_dest = 0.0;
    if (!(c.n18_flags & N18_NOCO) && (c.n18_pool[2] + c.n18_pool[4]) > 1e-10) {
        double nCII = s[slot(2, 1)] * nH, xO = s[slot(4, 0)];
        double gchx = (1.41E-10 * G) + 8.88E-15 * (zH2 / 1E-16);
        double beta = 5.E-10 * xO / (5.E-10 * xO + gchx / nH);
        co_form = std::fmax(5.E-16 * nCII * nH2 * beta, 1e-100) / nH;
        co_dest = std::fmax(2.43E-10 * G * c.fshield_CO + 4.62E-15 * (zH2 / 1E-16), 1e-100) * xCO;
        f[2] = co_form - co_dest;
        ds[slot(2, 1)] += co_dest - co_form;
    }
    f[1] = (c.n18_ref[0] == 0) ? dHp : -dHp - 2.0 * f[0];
    for (int e = 1; e < N18_NEL; e++) for (int k = 0; k < n18_Z[e]; k++) f[n18_ooff[e] + k] = ds[slot(e, stage_of(k, c.n18_ref[e]))];

    // ---- heating and cooling [erg cm^-3 s^-1] (cooling.c all_cooling) ----
    double* L = c.lambda; double* LC = c.lambda_chem;
    for (int i = 0; i < CHEM_NRATES; i++) L[i] = 0.0;
    for (int i = 0; i < CHEM_NRATES_CHEM_MAX; i++) LC[i] = 0.0;
    const double nHe0 = s[slot(1, 0)] * nH, nHe1 = s[slot(1, 1)] * nH, nHe2 = s[slot(1, 2)] * nH;
    const double nHtot = nHI + nHII + 2.0 * nH2;
    L[2]  = k.cHI * ne * nHI + k.cHII * ne * nHII + k.cHeI * ne * nHe0 + k.cHeII * ne * nHe1 + k.cHeIII * ne * nHe2;
    L[27] = k.cbrem * ne * (nHII + nHe1 + (4.0 * nHe2));
    L[17] = k.ccomp * ne;
    {   // H2 (Moseley+21)
        double x1 = nHI + 5.0 * nH2, x2 = nHI + 4.5 * nH2, x3 = nHI + 0.75 * nH2, x4 = nHI + 0.05 * nH2;
        double f1 = k.h2f[0] * (((0.7 * x1) / (1.0 + (x1 / 50.0))) + ((0.3 * x1) / (1.0 + (x1 / 500.0))));
        double f2 = k.h2f[1] * (((0.35 * x2) / (1.0 + (x2 / 450.0))) + ((0.65 * x2) / (1.0 + (x2 / 4500.0))));
        double f3 = k.h2f[2] * (x3 / (1.0 + (x3 / 25.0)));
        double f4 = k.h2f[3] * (((0.45 * x4) / (1.0 + (x4 / 900.0))) + ((0.55 * x4) / (1.0 + (x4 / 9000.0))));
        L[1] = std::fmax(nH2 * (f1 + f2 + f3 + f4), 1e-100);
    }
    {   // ion lines from the high-T tables
        double hT = 0.0;
        for (int e = 2; e < N18_NEL; e++) {
            double nel = c.n18_pool[e] * nH;
            if (!(nel > MIN_METAL_COOL_DENS)) continue;
            for (int j = 0; j <= n18_Z[e]; j++) hT += s[slot(e, j)] * nH * ne * k.hit[slot(e, j)];
        }
        L[25] = std::fmax(hT, 1e-100);
    }
    if (T < 1.1E4) {   // fine structure
        const double nc[8] = {ne, nHII, nHe1, nHe2, nHI, nHe0, 0.75 * nH2, 0.25 * nH2};
        for (int i = 0; i < 11; i++) {
            double n_ion = s[slot(FS[i].e, FS[i].j)] * nH;
            if (n_ion > 0.0) L[FS[i].lam] += fs_cool(k, i, n_ion, nc) * k.f2;
        }
    }
    // grains: recombination cooling and gas-grain exchange (T_d floor 2.725/a)
    if (ne > 0.0) L[19] = k.drc_pref * std::pow(G * k.sqT / (0.5 * ne), k.drc_beta) * ne * 0.5 * fdg * nH;
    {
        double hc = 1.0E-33 + (3.8E-33 - 1.0E-33) * std::fmax(std::fmin(2.0 * xH2, 1.0), 0.0);
        double Td = std::fmax(16.4 * std::pow(1.7 * G, 1.0 / 6.0), 2.725 / a);
        c.tdust = Td;
        L[0] = hc * k.dgc * (T - Td) * nH * nH * fdg;
    }
    {   // CO (Koyama & Inutsuka 2000), n = n_H
        double ta = k.co_a / nHtot, kbT = KB * T;
        double rot = (nHtot * xCO * nH * 4.0 * (kbT * kbT) * 9.7e-8) / (nHtot * 2.76 * KB * (1.0 + ta + 1.5 * std::sqrt(ta)));
        L[6] = std::fmax(rot + nH2 * xCO * nH * k.co_vH2 + nHI * xCO * nH * k.co_vH, 1e-100);
    }
    {   // photoelectric heating
        double fact = (G > 0.0) ? G * k.sqT / (ne * 0.5) : 0.0;
        double eps = std::fmax((4.9E-2 / (1.0 + 4E-3 * std::pow(fact, 0.73))) + (3.7E-2 * k.pe_t7 / (1.0 + 2E-4 * fact)), 1e-100);
        L[11] = -std::fmax(1.50 * 1.3E-24 * eps * G * fdg * nH, 1e-100);
    }
    {   // cosmic rays (Bialy 2019 q, metals with induced UV)
        double q = 6.43 * (1.0 + 4.06 * std::sqrt(xe / (0.07 + xe))) * EV2ERG;
        double h = nHI * zeta * q + nHe0 * 1.1 * zeta * q + nH2 * 2.0 * zeta * q + ne * zeta * 287.0 * EV2ERG;
        double uvs = 2.0 * zeta / 1E-16;
        for (int e = 2; e < N18_NEL; e++) {
            double nel = c.n18_pool[e] * nH; const int Z = n18_Z[e];
            if (nel < MIN_COOL_ION) continue;
            for (int j = 0; j < Z; j++) h += s[slot(e, j)] * nH * q * n18_cr[e][j] * ztot;
            h += s[slot(e, 0)] * nH * n18_cr_uv_heat[Z] * n18_cr_uv[Z] * EV2ERG * uvs;
        }
        L[10] = -h;
    }
    if (uvb) {   // HM12 photoheating
        double h = nHI * tl_uvb_ph[N18_NTRAC];
        for (int e = 1; e < N18_NEL; e++) {
            if (c.n18_pool[e] * nH < MIN_COOL_ION) continue;
            for (int j = 0; j < n18_Z[e]; j++) h += s[slot(e, j)] * nH * tl_uvb_ph[slot(e, j)];
        }
        LC[1] = -h * ss;
    }
    {   // sub-ionizing FUV photoheating of the neutral metals
        double h = 0.0;
        for (int e = 2; e < N18_NEL; e++) {
            if (c.n18_pool[e] * nH < MIN_COOL_ION) continue;
            h += s[slot(e, 0)] * nH * n18_g0_heat[n18_Z[e]] * G;
        }
        LC[2] = -h;
    }
    {   // H2 pumping, photodissociation, formation (Bialy 2018); D0 I_UV = 1.7 k_diss
        double ncf = 1.0 / (1.0 + (k.ncrit / nHtot)), DI = 1.7 * kdiss;
        double pump = 9.0 * DI * 1.12 * EV2ERG * nH2 * ncf, pd = DI * 0.4 * EV2ERG * nH2;
        double form = fdg * k.adust * (0.2 * EV2ERG + 4.48 * EV2ERG * ncf) * nHtot * nHtot * xHI
                    + hprim * (0.6 * EV2ERG + 3.13 * EV2ERG * ncf) * nHtot * nHtot * xHI;
        LC[0] = -std::fmax(pd + form + pump, 1e-100);
    }
    if (T <= 1E5) {   // charge exchange
        double h = 0.0;
        for (const CTH& t : CTHEAT) {
            int si = slot(t.e, t.j);
            h += (t.ion ? k.cti[si] * nHII : k.ctr[si] * nHI) * s[si] * nH * t.dE * EV2ERG;
        }
        LC[3] = -h;
    }
    LC[4] = -(c.rt_heat_HI * nHI + c.rt_heat_HeI * nHe0 + c.rt_heat_HeII * nHe1);   // M1 photoheating

    if (!(c.n18_flags & N18_FIXT)) {
        double net = 0.0;
        for (int i = 0; i < CHEM_NRATES; i++) net -= L[i];
        for (int i = 0; i < 5; i++) net -= LC[i];
        f[N18_NODE] = net;
    }
}

double temp_of(const double* y, const ChemCell& c, double* ntot_out)
{
    double ntot = particles(y, c, nullptr);
    if (ntot_out) *ntot_out = ntot;
    if (c.n18_flags & N18_FIXT) return c.n18_fixT;
    return y[N18_NODE] / (1.5 * KB * c.yn * ntot);
}

int bad_state(const double* y)
{
    for (int i = 0; i <= N18_NODE; i++) if (!std::isfinite(y[i])) return 1;
    return y[N18_NODE] <= 0.0;
}

// ---------------------------------------------------------------- CVODE glue
int cv_rhs(sunrealtype, N_Vector yy, N_Vector yp, void* ud)
{
    st_rhs++;
    return n18_rhs(N_VGetArrayPointer(yy), N_VGetArrayPointer(yp), *(ChemCell*)ud);
}

// Jacobian: species columns by differences at frozen rate coefficients, plus the T column through dT/dy
int cv_jac(sunrealtype, N_Vector yy, N_Vector fy, SUNMatrix J, void* ud, N_Vector, N_Vector, N_Vector)
{
    st_jac++;
    ChemCell& c = *(ChemCell*)ud;
    const int n = N18_NODE + 1;
    double* y = N_VGetArrayPointer(yy);
    const double* f0 = N_VGetArrayPointer(fy);
    double ntot, T0 = temp_of(y, c, &ntot);
    const double Tr = std::fmin(std::fmax(T0, T_MIN), T_MAX);
    const int exact = (c.n18_flags & N18_EXACT) != 0;
    const Coef& k = coef_at(Tr, c.redshift, exact);
    double lam0[CHEM_NRATES], lamc0[CHEM_NRATES_CHEM_MAX], td0 = c.tdust;
    std::memcpy(lam0, c.lambda, sizeof(lam0)); std::memcpy(lamc0, c.lambda_chem, sizeof(lamc0));
    double fa[N18_NODE + 1], fb[N18_NODE + 1];
    tl_fsh = (c.iphoto == 0) ? 1.0 : cxx_calc_H2_shielding_factor(Tr, std::fmax(y[0], 0.0), c.dl * c.yn, c);
    assemble(y, Tr, k, c, fa);   // f at frozen T (equals f0 up to the clamp)

    // d f / d T at fixed composition
    double vT[N18_NODE + 1];
    const int fixT = (c.n18_flags & N18_FIXT) != 0;
    if (!fixT) {
        double dT = 1e-6 * Tr;
        coef_fill(tl_k2, Tr + dT, c.redshift, exact);
        assemble(y, Tr + dT, tl_k2, c, fb);
        for (int i = 0; i < n; i++) vT[i] = (fb[i] - fa[i]) / dT;
    }
    double dTdy[N18_NODE + 1];
    for (int i = 0; i < n; i++) dTdy[i] = 0.0;
    if (!fixT && T0 > T_MIN && T0 < T_MAX) {
        // d n_tot / d y: H2 -1, H+ +1, CO -1 - (charge of the C and O reference stages), stage j of element e: j - ref
        if (c.n18_ref[0] == 0) { dTdy[0] = T0 / ntot; dTdy[1] = -T0 / ntot; }
        else                   { dTdy[0] = 3.0 * T0 / ntot; dTdy[1] = T0 / ntot; }
        dTdy[2] = (1.0 + c.n18_ref[2] + c.n18_ref[4]) * T0 / ntot;
        for (int e = 1; e < N18_NEL; e++) for (int k = 0; k < n18_Z[e]; k++) dTdy[n18_ooff[e] + k] = -(stage_of(k, c.n18_ref[e]) - c.n18_ref[e]) * T0 / ntot;
        dTdy[N18_NODE] = T0 / y[N18_NODE];
    }
    // species columns
    for (int col = 0; col < N18_NODE; col++) {
        int e = 0;
        if (col >= 5) { e = 2; while (e + 1 < N18_NEL && col >= n18_ooff[e + 1]) e++; }
        else if (col >= 3) e = 1;
        double pool = (col == 0) ? 0.5 : (col == 1) ? 1.0 : (col == 2) ? std::fmin(c.n18_pool[2], c.n18_pool[4]) : elem_pool(c, e, 0.0);
        double h = 1e-7 * std::fmax(std::fabs(y[col]), 1e-4 * pool);
        if (h <= 0) h = 1e-30;
        double s0 = y[col];
        y[col] = s0 + h;
        assemble(y, Tr, k, c, fb);
        y[col] = s0;
        double* Jc = SUNDenseMatrix_Column(J, col);
        for (int i = 0; i < n; i++) Jc[i] = (fb[i] - fa[i]) / h + (fixT ? 0.0 : vT[i] * dTdy[col]);
    }
    double* Je = SUNDenseMatrix_Column(J, N18_NODE);
    for (int i = 0; i < n; i++) Je[i] = fixT ? 0.0 : vT[i] * dTdy[N18_NODE];
    (void)f0;
    tl_fsh = -1.0;
    std::memcpy(c.lambda, lam0, sizeof(lam0)); std::memcpy(c.lambda_chem, lamc0, sizeof(lamc0)); c.tdust = td0;
    return 0;
}

void cv_err(int, const char*, const char*, char*, void*) {}

struct Work {
    SUNContext ctx = nullptr;
    void* mem = nullptr;
    N_Vector y = nullptr, atol = nullptr;
    SUNMatrix A = nullptr;
    SUNLinearSolver LS = nullptr;
};
thread_local Work tl_w;

Work* work_get(const double* y0)
{
    Work& w = tl_w;
    const int n = N18_NODE + 1;
    if (!w.mem) {
        if (SUNContext_Create(NULL, &w.ctx)) return nullptr;
        w.y = N_VNew_Serial(n, w.ctx); w.atol = N_VNew_Serial(n, w.ctx);
        for (int i = 0; i < n; i++) NV_Ith_S(w.y, i) = y0[i];
        w.mem = CVodeCreate(CV_BDF, w.ctx);
        if (!w.mem) return nullptr;
        CVodeSetErrHandlerFn(w.mem, cv_err, NULL);
        if (CVodeInit(w.mem, cv_rhs, 0.0, w.y) < 0) return nullptr;
        w.A = SUNDenseMatrix(n, n, w.ctx);
        w.LS = SUNLinSol_Dense(w.y, w.A, w.ctx);
        CVodeSetLinearSolver(w.mem, w.LS, w.A);
        CVodeSetJacFn(w.mem, cv_jac);
    }
    return &w;
}

// element-wise projection onto the pools (negative ions -> 0, oversubscribed elements scaled down)
void project(double* y, const ChemCell& c)
{
    y[0] = std::fmax(y[0], 0.0); y[1] = std::fmax(y[1], 0.0); y[2] = std::fmax(y[2], 0.0);
    if (2.0 * y[0] + y[1] > 1.0) { double f = 1.0 / (2.0 * y[0] + y[1]); y[0] *= f; y[1] *= f; }
    y[2] = std::fmin(y[2], std::fmin(c.n18_pool[2], c.n18_pool[4]));
    for (int e = 1; e < N18_NEL; e++) {
        double sum = 0;
        for (int k = 0; k < n18_Z[e]; k++) { double& v = y[n18_ooff[e] + k]; v = std::fmax(v, 0.0); sum += v; }
        double cap = elem_pool(c, e, y[2]);
        if (sum > cap && sum > 0) { double f = std::fmax(cap, 0.0) / sum; for (int k = 0; k < n18_Z[e]; k++) y[n18_ooff[e] + k] *= f; }
    }
}

}  // namespace

// ==================================================================== API
void n18_cell_defaults(ChemCell& c)
{
    for (int e = 0; e < N18_NEL; e++) { c.n18_pool[e] = 0.0; c.n18_ref[e] = 0; }
    c.n18_pool[0] = 1.0;
    c.n18_flags = 0; c.n18_fixT = 0.0;
    c.n18_rtol = 1e-4; c.n18_atol = 1e-10;
    c.n18_df25 = 1.0;
    c.n18_T = 0.0; c.n18_xe = 0.0;
    if (c.net == 18) c.fac_shield_dust = 1.0;   // calc_photo returns before setting it when G0 = 0
}

double n18_temperature(const double* y, const ChemCell& c) { return temp_of(y, c, nullptr); }

int n18_rhs(const double* y, double* ydot, ChemCell& c)
{
    for (int i = 0; i <= N18_NODE; i++) if (!std::isfinite(y[i])) { for (int j = 0; j <= N18_NODE; j++) ydot[j] = 0.0; return CHEM_RHS_RECOVERABLE; }
    double T = temp_of(y, c, nullptr);
    if (!(T > 0.0)) { for (int j = 0; j <= N18_NODE; j++) ydot[j] = 0.0; return CHEM_RHS_RECOVERABLE; }
    T = std::fmin(std::fmax(T, T_MIN), T_MAX);
    if (c.n18_flags & N18_UVB) uvb_update(c.redshift);
    assemble(y, T, coef_at(T, c.redshift, (c.n18_flags & N18_EXACT) != 0), c, ydot);
    for (int i = 0; i <= N18_NODE; i++)
        if (!std::isfinite(ydot[i])) {
            std::fprintf(stderr, "n18_rhs: non-finite ydot[%d] (ID %ld, T=%g yn=%g)\n", i, c.id_current, T, c.yn);
            for (int j = 0; j <= N18_NODE; j++) ydot[j] = 0.0;
            return CHEM_RHS_RECOVERABLE;
        }
    return CHEM_RHS_OK;
}

void n18_trac_to_ode(const double* tr, double energy, double* y, ChemCell& c)
{
    double xH2 = std::fmin(std::fmax(tr[N18_T_H2], 0.0), 0.5), xHp = std::fmin(std::fmax(tr[N18_T_HP], 0.0), 1.0);
    if (2.0 * xH2 + xHp > 1.0) { double f = 1.0 / (2.0 * xH2 + xHp); xH2 *= f; xHp *= f; }
    double xCO = std::fmin(std::fmax(tr[N18_T_CO], 0.0), std::fmin(c.n18_pool[2], c.n18_pool[4]));
    double xHI = std::fmax(1.0 - 2.0 * xH2 - xHp, 0.0);
    c.n18_ref[0] = (xHI >= xHp) ? 0 : 1;
    y[0] = xH2; y[1] = (c.n18_ref[0] == 0) ? xHp : xHI; y[2] = xCO;
    for (int e = 1; e < N18_NEL; e++) {
        double sum = 0;
        for (int j = 0; j <= n18_Z[e]; j++) sum += std::fmax(tr[slot(e, j)], 0.0);
        double f = (sum > 0.0) ? std::fmax(elem_pool(c, e, xCO), 0.0) / sum : 0.0;
        int r = 0;
        for (int j = 1; j <= n18_Z[e]; j++) if (tr[slot(e, j)] > tr[slot(e, r)]) r = j;
        c.n18_ref[e] = r;
        for (int k = 0; k < n18_Z[e]; k++) y[n18_ooff[e] + k] = std::fmax(tr[slot(e, stage_of(k, r))], 0.0) * f;
    }
    y[N18_NODE] = energy;
}

void n18_ode_to_trac(const double* y, double* tr, const ChemCell& c)
{
    double xHI, xHp;
    hydrogen(y, c, &xHI, &xHp);
    tr[N18_T_H2] = y[0]; tr[N18_T_HP] = xHp; tr[N18_T_CO] = y[2];
    for (int e = 1; e < N18_NEL; e++) {
        const int r = c.n18_ref[e];
        double sum = 0;
        for (int k = 0; k < n18_Z[e]; k++) { tr[slot(e, stage_of(k, r))] = y[n18_ooff[e] + k]; sum += y[n18_ooff[e] + k]; }
        tr[slot(e, r)] = std::fmax(elem_pool(c, e, y[2]) - sum, 0.0);
    }
}

int n18_evolve(double dt, double* trac, double* energy, ChemCell& c)
{
    const int n = N18_NODE + 1;
    st_calls++;
    c.n18_pool[1] = c.abhe;
    double y[N18_NODE + 1];
    n18_trac_to_ode(trac, *energy, y, c);
    if (dt <= 0.0) { n18_ode_to_trac(y, trac, c); double yd[N18_NODE + 1]; n18_rhs(y, yd, c); return 0; }
    if (c.n18_flags & N18_UVB) uvb_update(c.redshift);

    Work* w = work_get(y);
    if (!w) return 3;
    for (int i = 0; i < n; i++) NV_Ith_S(w->y, i) = y[i];
    double* at = N_VGetArrayPointer(w->atol);
    at[0] = 0.5 * c.n18_atol; at[1] = 1e-2 * c.n18_atol;
    at[2] = std::fmax(c.n18_atol * std::fmin(c.n18_pool[2], c.n18_pool[4]), 1e-30);
    for (int e = 1; e < N18_NEL; e++) for (int k = 0; k < n18_Z[e]; k++) at[n18_ooff[e] + k] = std::fmax(c.n18_atol * elem_pool(c, e, 0.0), 1e-30);
    at[N18_NODE] = 0.0;
    if (CVodeReInit(w->mem, 0.0, w->y) < 0) return 3;
    CVodeSVtolerances(w->mem, c.n18_rtol, w->atol);
    CVodeSetUserData(w->mem, &c);
    CVodeSetMaxNumSteps(w->mem, 200000);
    CVodeSetInitStep(w->mem, 0.0);
    CVodeSetMaxStep(w->mem, 0.0);

    int rc = 1;
    double t = 0.0;
    for (int attempt = 0; attempt < 8; attempt++) {
        double tret = t;
        int flag = CVode(w->mem, dt, w->y, &tret, CV_NORMAL);
        long ns = 0; CVodeGetNumSteps(w->mem, &ns); st_steps += ns;
        if (flag >= 0) { rc = 0; break; }
        rc = (flag == CV_TOO_MUCH_WORK) ? 1 : 100;
        // restart from the last accepted state (or the previous restart point) with a small first step
        double* yy = N_VGetArrayPointer(w->y);
        if (bad_state(yy) || !(tret > t)) { for (int i = 0; i < n; i++) yy[i] = y[i]; }
        else { project(yy, c); t = tret; for (int i = 0; i < n; i++) y[i] = yy[i]; }
        if (CVodeReInit(w->mem, t, w->y) < 0) return 3;
        CVodeSetInitStep(w->mem, (dt - t) * std::pow(1e-3, attempt + 1));
    }
    if (rc != 0) { std::fprintf(stderr, "n18_evolve: CVODE failed (ID %ld, yn=%g, T=%g)\n", c.id_current, c.yn, c.n18_T); return rc; }
    double* yy = N_VGetArrayPointer(w->y);
    for (int i = 0; i < n; i++) y[i] = yy[i];
    project(y, c);
    if (bad_state(y)) return 100;
    *energy = y[N18_NODE];
    n18_ode_to_trac(y, trac, c);
    double yd[N18_NODE + 1];
    n18_rhs(y, yd, c);   // diagnostics at the returned state
    return 0;
}

void n18_stats(long* calls, long* steps, long* rhs, long* jac)
{
    *calls = st_calls; *steps = st_steps; *rhs = st_rhs; *jac = st_jac;
}
