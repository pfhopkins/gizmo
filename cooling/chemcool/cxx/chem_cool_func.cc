// chem_cool_func.cc -- Gate-3 port of cool_func.F (all 28 radiative
// heating/cooling rates) and of get_coolrates_wssCIE (wss_z_collis.F).
//
// Convention (unchanged from the Fortran): Lambda is the net COOLING rate,
// positive, erg s^-1 cm^-3, so heating terms enter NEGATIVE.
//
// Faithfulness notes that are easy to get wrong and are load-bearing:
//  * `kboltz` (cool.h <- BOLTZMANN) is a Fortran DEFAULT REAL literal, i.e.
//    float32-rounded; `kb` in fs_data.h is spelled 1.38066d0-16 and is a true
//    double.  They are NOT the same number.  CHEM_KBOLTZ / CHEM_FS_KB below.
//    (Same trap as DEVIATIONS.md D-10.)
//  * cool_func's temperature index: `temp .gt. tmax -> itemp = nmd, dtemp = 0`
//    (no extrapolation above 1e8 K, D-39).
//  * rates(3) is unconditionally zeroed at cool_func.F:1223 and then (under
//    WSS_CIE_COOL) overwritten by the WSS metal-line CIE rate.
//  * integer powers use chem_fpowi (gfortran's association), real powers use
//    pow(), matching gfortran's lowering.

#include "../../../GIZMO_config.h"
#include "chem_types.h"
#include <cmath>
#include <cstdio>

// fs_data.h: kb is spelled 1.38066d-16 -- a genuine double, unlike kboltz.
#define CHEM_FS_KB   1.38066e-16
#define FS_OXA10     8.865e-5
#define FS_OXA20     1.275e-10
#define FS_OXA21     1.772e-5
#define FS_OXE10     (2.2771e2 * CHEM_FS_KB)
#define FS_OXE21     (9.886e1  * CHEM_FS_KB)
#define FS_OXE20     (FS_OXE10 + FS_OXE21)
#define FS_CIA10     7.932e-8
#define FS_CIA20     2.054e-14
#define FS_CIA21     2.654e-7
#define FS_CIE10     (2.360e1 * CHEM_FS_KB)
#define FS_CIE21     (3.884e1 * CHEM_FS_KB)
#define FS_CIE20     (FS_CIE10 + FS_CIE21)
#define FS_CIIA10    2.291e-6
#define FS_CIIE10    (9.125e1 * CHEM_FS_KB)
#define FS_SIIA10    8.4e-6
#define FS_SIIA20    2.4e-10
#define FS_SIIA21    4.2e-5
#define FS_SIIE10    (1.1e2 * CHEM_FS_KB)
#define FS_SIIE21    (2.1e2 * CHEM_FS_KB)
#define FS_SIIE20    (FS_SIIE10 + FS_SIIE21)
#define FS_SIIIA10   2.17e-4
#define FS_SIIIE10   (4.1224e2 * CHEM_FS_KB)

// cool_func.F:131-136.  log_fp_h2o is spelled `-0.602060` (no D exponent) and
// is therefore a default-REAL literal -- float32-rounded.  log_fo_h2o has a
// `d0` and is a true double.  Kept distinct on purpose.
static const double F_ORTHO_H2O = 0.75;
static const double LOG_FO_H2O  = -0.1249387;
static const double F_PARA_H2O  = 0.25;
static const double LOG_FP_H2O  = -0.602060f;

// ===========================================================================
// get_coolrates_wssCIE (wss_z_collis.F:66-168)
// ===========================================================================
double cxx_wss_cie_rate(const ChemTables& T, double temp, const double Zm[12])
{
    // wss_z_collis.F:91 spells `lT_range = 6.98185` with no D exponent, and
    // the nine solar reference abundances below are likewise default-REAL
    // literals.  float32(6.98185) is 2.1e-8 away from the decimal, which moves
    // the interpolation index enough to shift the rate by 5e-6 -- exactly the
    // deviation the Gate-3 golden measured on rates(3) (0/383880 passing).
    const double lT_range = (double)6.98185f;   // logT from 2.0 to 8.98185
    const int NlT = 351;                        // 352 points -> 351 intervals

    double idx_real = NlT * (std::log10(temp) - 2.0) / lT_range + 1;
    int idx = (int)idx_real;
    double x = idx_real - idx;

    if (idx < 1)          { idx = 1;       x = 0.0; }
    else if (idx > NlT-1) { idx = NlT - 1; x = 1.0; }

    double rC  = Zm[1] / (double)2.07e-3f;
    double rN  = Zm[7] / (double)8.36e-4f;
    double rO  = Zm[3] / (double)5.49e-3f;
    double rNe = Zm[8] / (double)1.41e-3f;
    double rMg = Zm[2] / (double)5.91e-4f;
    double rSi = Zm[5] / (double)6.83e-4f;
    double rS  = Zm[9] / (double)4.09e-4f;
    double rCa = Zm[10] / (double)6.44e-5f;
    double rFe = Zm[4] / (double)1.10e-3f;

    const double* t[9] = { T.wss_C, T.wss_N, T.wss_O, T.wss_Ne, T.wss_Mg,
                           T.wss_Si, T.wss_S, T.wss_Ca, T.wss_Fe };
    const double r[9]  = { rC, rN, rO, rNe, rMg, rSi, rS, rCa, rFe };

    double tot = 0.0;
    for (int k = 0; k < 9; k++)
        tot += std::pow(10.0, (1.0 - x) * t[k][idx-1] + x * t[k][idx]) * r[k];
    return tot;
}

// ===========================================================================
// cool_func (cool_func.F:10-1323)
// ===========================================================================
int cxx_cool_func(double temp, double yn, double dl, double divv, double G_dust,
                  const double abundances[16], double* ylam,
                  double rates[CHEM_NRATES], int iflag_cmb, ChemCell& c)
{
    const ChemTables& T = *c.tab;
    const double kboltz = CHEM_KBOLTZ;

    // Verify that the temperature is valid.  NB the Fortran RETURNS here
    // WITHOUT touching ylam or rates -- they keep their previous values.
    if (temp < 0.0) {
        std::printf("COOL_FUNC: initial temperature is unphysical,\n");
        std::printf(" temp = %12.5e\n", temp);
        return 0;
    }

    // ---- table read-out (cool_func.F:213-362) ----
    int itemp; double dtemp;
    if (temp <= T.tmin) {
        itemp = 1; dtemp = 0.0;
    } else if (temp > T.tmax) {
        itemp = CHEM_NMD; dtemp = 0.0;      // no extrapolation above tmax
    } else {
        itemp = (int)(std::log10(temp) / T.dtlog) + 1;
        if (itemp <= 0 || itemp > CHEM_NMD) {
            std::printf("Fatal error in cool_func %d %g\n", itemp, temp);
            c.cool_func_failed = 1;
            return 1;                       // ABORT(0)
        }
        dtemp = temp - T.temptab[itemp - 1];
    }
    const double* clrow = T.cltab[itemp - 1];
    const double* dclrow = T.dtcltab[itemp - 1];
    double cl[CHEM_NCLTAB + 1];             // 1-based, cl[k] == Fortran clk
    for (int k = 1; k <= CHEM_NCLTAB; k++)
        cl[k] = clrow[k - 1] + dtemp * dclrow[k - 1];

    double CMB_temp = 2.726 * (1.0 + c.redshift);

    // ---- abundances (cool_func.F:373-388) ----
    double abh2   = abundances[0];
    double abo    = abundances[1];
    double aboh   = abundances[2];
    double abh2o  = abundances[3];
    double abco   = abundances[4];
    double abcI   = abundances[5];
    double abcII  = abundances[6];
    double absiI  = abundances[7];
    double absiII = abundances[8];
    double abe    = abundances[9];
    double abhp   = abundances[10];
    double abhI   = abundances[11];
    double abhd   = abundances[12];
    double abheI  = abundances[13];
    double abheII = abundances[14];
    double abheIII= abundances[15];

#if defined(WSS_CIE_COOL) && (CHEMISTRYNETWORK == 4 || CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6 || CHEMISTRYNETWORK == 9 || CHEMISTRYNETWORK == 15)
    // He ionization state from the CIE table, only for nets that pass no He states (cool_func.F:392-416)
    if (temp <= 1e4) {
        abheI = c.abhe; abheII = 0.0; abheIII = 0.0;
    } else if (temp >= 1e7) {
        abheIII = c.abhe; abheI = 0.0; abheII = 0.0;
    } else {
        double idx_real = 50 * (std::log10(temp) - 4.0) / (3.0) + 1;
        int idx = (int)idx_real;
        double w = idx_real - idx;
        abheI  = c.abhe * ((1.0 - w) * T.HeI_tbl[idx-1]  + w * T.HeI_tbl[idx]);
        abheII = c.abhe * ((1.0 - w) * T.HeII_tbl[idx-1] + w * T.HeII_tbl[idx]);
        abheIII = c.abhe - abheI - abheII;
    }
#endif

    if (abh2 > (0.5 + CHEM_COOL_EPS) || abh2 < 0.0) {
        std::printf("COOL_FUNC: initial  H2 fraction is unphysical,\n");
        std::printf(" fraction = %12.5e\n", abh2);
        c.cool_func_failed = 1;
        return 2;                           // ABORT(2)
    }

    double ynh2 = abh2 * yn;
    double ynh  = abhI * yn;
    double yne  = abe  * yn;
    double ynhp = abhp * yn;
    double ynhd = abhd * yn;
    double ynhe = abheI * yn;

    // ---- effective columns for CO / H2O (cool_func.F:455-599) ----
    CoCool co = {0,0,0,0,0}, c13o = {0,0,0,0,0}, co18 = {0,0,0,0,0};
    if (abco > 1e-5 * c.abundo) {
        double maximum_CO_column_rot = T.co_colntab[CHEM_NCDCO - 1];
        double maximum_CO_column_vib = T.co_vib_colntab[CHEM_NCDCO_VIB - 1];
        double N_co_eff, N_c13o_eff, N_co18_eff;
        double N_co_eff_vib, N_c13o_eff_vib, N_co18_eff_vib;
        if (divv == 0.0) {
            N_co_eff = N_c13o_eff = N_co18_eff = maximum_CO_column_rot;
            N_co_eff_vib = N_c13o_eff_vib = N_co18_eff_vib = maximum_CO_column_vib;
        } else {
            double dv = 1e-5 * std::fabs(divv);
            N_co_eff   = std::log10(abco * yn / dv);
            N_c13o_eff = N_co_eff - 2.0;              // isotope ratios, NLM95
            N_co18_eff = N_co_eff - 2.69897;          // log10(2e-3)
            N_co_eff_vib   = N_co_eff;
            N_c13o_eff_vib = N_c13o_eff;
            N_co18_eff_vib = N_co18_eff;

            if (N_co18_eff > maximum_CO_column_rot) {
                N_co_eff = N_c13o_eff = N_co18_eff = maximum_CO_column_rot;
            } else if (N_c13o_eff > maximum_CO_column_rot) {
                N_co_eff = N_c13o_eff = maximum_CO_column_rot;
            } else if (N_co_eff > maximum_CO_column_rot) {
                N_co_eff = maximum_CO_column_rot;
            }
            if (N_co18_eff_vib > maximum_CO_column_vib) {
                N_co_eff_vib = N_c13o_eff_vib = N_co18_eff_vib = maximum_CO_column_vib;
            } else if (N_c13o_eff_vib > maximum_CO_column_vib) {
                N_co_eff_vib = N_c13o_eff_vib = maximum_CO_column_vib;
            } else if (N_co_eff_vib > maximum_CO_column_vib) {
                N_co_eff_vib = maximum_CO_column_vib;
            }
        }

        double temp_co = (temp <= T.co_temptab[0]) ? T.co_temptab[0] : temp;
        cxx_co_cool(T, temp_co, N_co_eff, N_co_eff_vib, &co);
        if (N_co_eff <= T.co_colntab[0]) {
            c13o = co; co18 = co;
        } else {
            cxx_co_cool(T, temp_co, N_c13o_eff, N_c13o_eff_vib, &c13o);
            cxx_co_cool(T, temp_co, N_co18_eff, N_co18_eff_vib, &co18);
        }
    }

    H2oCool h2o_o = {0,0,0,0}, h2o_p = {0,0,0,0};
    H2oCool h2o18_o = {0,0,0,0}, h2o18_p = {0,0,0,0};
    double h2o_vib_lte_rate = 0.0, h2o18_vib_lte_rate = 0.0;
    if (abh2o > 1e-5 * c.abundo) {
        double Np, No, N18p, N18o, Nv, N18v;
        if (std::fabs(divv) < 1.0) {
            Np = No = N18p = N18o = T.h2o_colntab[CHEM_NCDH2O - 1];
            Nv = N18v = T.h2o_vib_colntab[CHEM_NCDH2O_VIB - 1];
        } else {
            double dv = 1e-5 * std::fabs(divv);
            Nv = std::log10(abh2o * yn / dv);
            Np = Nv + LOG_FP_H2O;
            No = Nv + LOG_FO_H2O;
            N18p = Np - 2.69897;
            N18o = No - 2.69897;
            N18v = Nv - 2.69897;
            double maxrot = T.h2o_colntab[CHEM_NCDH2O - 1];
            double maxvib = T.h2o_vib_colntab[CHEM_NCDH2O_VIB - 1];
            if (Np   > maxrot) Np   = maxrot;
            if (No   > maxrot) No   = maxrot;
            if (Nv   > maxvib) Nv   = maxvib;
            if (N18p > maxrot) N18p = maxrot;
            if (N18o > maxrot) N18o = maxrot;
            if (N18v > maxvib) N18v = maxvib;
        }
        cxx_h2o_rot_cool(T, temp, Np,   0, &h2o_p);
        cxx_h2o_rot_cool(T, temp, No,   1, &h2o_o);
        cxx_h2o_rot_cool(T, temp, N18p, 0, &h2o18_p);
        cxx_h2o_rot_cool(T, temp, N18o, 1, &h2o18_o);
        h2o_vib_lte_rate   = cxx_h2o_vib_cool(T, temp, Nv);
        h2o18_vib_lte_rate = cxx_h2o_vib_cool(T, temp, N18v);
    }

    // ---- (R1) gas-grain cooling/heating ----
    rates[0] = 0.0;
    if (c.dust_to_gas_ratio > 0.0 && yn > 1e8 / std::pow(c.dust_to_gas_ratio, 1.5)) {
        if (iflag_cmb == 0) {
            // Cap at the grain-sublimation bound, consistent with calc_dust_temp.
            c.tdust = (temp < 1.5e3) ? temp : 1.5e3;
            rates[0] = 4.68e-31 * chem_fpowi(c.tdust, 6) * yn * c.dust_to_gas_ratio;
        }
    } else {
        if (iflag_cmb == 0)
            rates[0] = cl[14] * (temp - c.tdust) * (yn * yn) * c.dust_to_gas_ratio;
    }

    // ---- (R2) H2 (vr) cooling ----
    if (ynh2 == 0.0) {
        rates[1] = 0.0;
    } else {
        double h2var0 = ynh * cl[4] + ynh2 * cl[5] + ynhe * cl[64]
                      + ynhp * cl[65] + yne * cl[66];
        if (h2var0 < 1e-4 * cl[1]) rates[1] = h2var0 * ynh2;
        else                       rates[1] = ynh2 * cl[1] / (1.0 + cl[1] / h2var0);

        if (yn >= 1e8) {
            double vth = std::sqrt(kboltz * temp / CHEM_PROTONMASS);
            double N_H2_jeans = ynh2 / (vth / dl);
            double N_H2_LVG   = ynh2 / std::fabs(divv);
            double N_H2_eff   = (N_H2_LVG < N_H2_jeans) ? N_H2_LVG : N_H2_jeans;
            rates[1] *= cxx_compute_h2_opacity(T, temp, N_H2_eff);
        }
    }

    // ---- (R3, R25-R28) atomic / H-He plasma cooling ----
    rates[2]  = cl[6] * (yn * yn);
    rates[24] = cl[52] * ynh * yne;
    rates[25] = cl[59] * ynhe * yne + cl[60] * abheII * yn * (yne * yne);
    rates[26] = cl[61] * abheII * yn * yne;
    rates[27] = cl[62] * (abhp + abheII) * yn * yne + cl[63] * abheIII * yn * yne;
    if (temp < 100) rates[27] = 0.0;

    // ---- (R4, R23) H2O rotational cooling ----
    double neff = ynh2 + std::sqrt(2.0) * ynh + 0.5 * ynhe;
    if (abh2o <= 1e-5 * c.abundo || neff == 0.0) {
        rates[3] = 0.0; rates[22] = 0.0;
    } else {
        double inv;
        inv = (1.0 / h2o_o.L0) + (neff / h2o_o.lte)
            + (1.0 / h2o_o.L0) * (1.0 - h2o_o.n05 * h2o_o.L0 / h2o_o.lte)
            * std::pow(neff / h2o_o.n05, h2o_o.alpha);
        rates[3] = F_ORTHO_H2O * abh2o * neff * yn / inv;

        inv = (1.0 / h2o_p.L0) + (neff / h2o_p.lte)
            + (1.0 / h2o_p.L0) * (1.0 - h2o_p.n05 * h2o_p.L0 / h2o_p.lte)
            * std::pow(neff / h2o_p.n05, h2o_p.alpha);
        rates[3] += F_PARA_H2O * abh2o * neff * yn / inv;

        inv = (1.0 / h2o18_o.L0) + (neff / h2o18_o.lte)
            + (1.0 / h2o18_o.L0) * (1.0 - h2o18_o.n05 * h2o18_o.L0 / h2o18_o.lte)
            * std::pow(neff / h2o18_o.n05, h2o18_o.alpha);
        rates[22] = 2e-3 * F_ORTHO_H2O * abh2o * neff * yn / inv;

        inv = (1.0 / h2o18_p.L0) + (neff / h2o18_p.lte)
            + (1.0 / h2o18_p.L0) * (1.0 - h2o18_p.n05 * h2o18_p.L0 / h2o18_p.lte)
            * std::pow(neff / h2o18_p.n05, h2o18_p.alpha);
        rates[22] += 2e-3 * F_PARA_H2O * abh2o * neff * yn / inv;
    }

    // ---- (R5, R6) H2O vibrational cooling ----
    neff = ynh2 + ynh;
    if (abh2o <= 1e-5 * c.abundo || neff == 0.0 || temp < 1e2) {
        rates[4] = 0.0; rates[5] = 0.0;
    } else {
        double L0rate = cl[8] * abh2 + cl[9] * abhI;
        double inv = (1.0 / L0rate) + (neff / h2o_vib_lte_rate);
        rates[4] = abh2o * yn * neff * (1.0 / inv);
        inv = (1.0 / L0rate) + (neff / h2o18_vib_lte_rate);
        rates[5] = 2e-3 * abh2o * yn * neff * (1.0 / inv);
    }

    // ---- (R7, R21, R22) CO rotational cooling ----
    neff = ynh2 + std::sqrt(2.0) * ynh + 0.5 * ynhe;
    if (abco <= 1e-5 * c.abundo || neff == 0.0) {
        rates[6] = 0.0; rates[20] = 0.0; rates[21] = 0.0;
    } else {
        double inv;
        inv = (1.0 / co.L0) + (neff / co.lte)
            + (1.0 / co.L0) * (1.0 - co.n05 * co.L0 / co.lte)
            * std::pow(neff / co.n05, co.alpha);
        rates[6] = abco * neff * yn / inv;

        inv = (1.0 / c13o.L0) + (neff / c13o.lte)
            + (1.0 / c13o.L0) * (1.0 - c13o.n05 * c13o.L0 / c13o.lte)
            * std::pow(neff / c13o.n05, c13o.alpha);
        rates[20] = 1e-2 * abco * neff * yn / inv;

        inv = (1.0 / co18.L0) + (neff / co18.lte)
            + (1.0 / co18.L0) * (1.0 - co18.n05 * co18.L0 / co18.lte)
            * std::pow(neff / co18.n05, co18.alpha);
        rates[21] = 2e-3 * abco * neff * yn / inv;

        if (temp <= T.co_temptab[0]) {
            // No accurate data below 5 K; taper rather than switch off, which
            // would break the solver.
            double low_T_adjust = std::exp(-1e1 / temp) / std::exp(-1e1 / 5e0);
            rates[6]  *= low_T_adjust;
            rates[20] *= low_T_adjust;
            rates[21] *= low_T_adjust;
        }
    }
#ifdef NO_CO_COOL
    rates[6] = 0.0; rates[20] = 0.0; rates[21] = 0.0;
#endif

    // ---- (R8, R9, R24) CO vibrational cooling ----
    neff = ynh2 + ynh;
    if (abco <= 1e-5 * c.abundo || neff == 0.0 || temp < 1e2) {
        rates[7] = 0.0; rates[8] = 0.0; rates[23] = 0.0;
    } else {
        double L0rate = cl[12] * abh2 + cl[13] * abhI;
        double inv = (1.0 / L0rate) + (neff / co.vib_lte);
        rates[7] = abco * yn * neff * (1.0 / inv);
        inv = (1.0 / L0rate) + (neff / c13o.vib_lte);
        rates[8] = 1e-2 * abco * yn * neff * (1.0 / inv);
        inv = (1.0 / L0rate) + (neff / co18.vib_lte);
        rates[23] = 2e-3 * abco * yn * neff * (1.0 / inv);
    }
#ifdef NO_CO_COOL
    rates[7] = 0.0; rates[8] = 0.0; rates[23] = 0.0;
#endif

    // ---- (R10) OH cooling ----
    rates[9] = cl[15] * aboh * neff;

    // ---- (R11) cosmic-ray heating ----
    if (iflag_cmb == 0) {
#if CHEMISTRYNETWORK == 16 || defined(CHEMCOOL_CRHEAT_K23)
        {   // Gong+17 / Kim+2023 Eqs. 32-33: Draine (2011) for HI, Krumholz (2014) for H2, per H nucleus
            double qcr_HI = 6.5 + 26.4 * std::sqrt(abe / (abe + 0.07));
            double logn = std::log10(yn), qcr_H2;
            if (logn <= 2.0)      qcr_H2 = 10.0;
            else if (logn < 4.0)  qcr_H2 = 10.0 + 3.0 * (logn - 2.0) / 2.0;
            else if (logn < 7.0)  qcr_H2 = 13.0 + 4.0 * (logn - 4.0) / 3.0;
            else if (logn < 10.0) qcr_H2 = 17.0 + (logn - 7.0) / 3.0;
            else                  qcr_H2 = 18.0;
            double qcr = (abhI * qcr_HI + 2.0 * abh2 * qcr_H2) * 1.6e-12;
            rates[10] = -qcr * yn * c.cosmic_ray_ion_rate;
        }
#else
        rates[10] = -3.2e-11 * (ynh2 + ynh) * c.cosmic_ray_ion_rate;
#endif
#ifdef CR_HEATING_DM72
        rates[10] = -1e-11 * (ynh2 + ynh + yne) * c.cosmic_ray_ion_rate
                  * (1.0 + 4.1 * std::sqrt(abe / (abe + 7e-2)));
#endif
        c.cr_ion_heat = -rates[10]; // ionisation heating, integrated by the driver
        // Hadronic + Coulomb heating from CR proton losses (Guo & Oh 2008,
        // Mannheim & Schlickeiser 1994); CR_gas_heating() without its ionisation part, which the zeta term heats.
        if (c.cr_energy_density > 0.0) {
            rates[10] -= (0.87 * 6.37e-16 / 6.0
                        + 0.53 * 3.09e-16 * abe
                          / (1.0 + 4.0 * c.abhe))
                       * c.cr_energy_density * yn;
        }
    } else {
        rates[10] = 0.0;
        c.cr_ion_heat = 0.0;
    }

    // ---- (R12) photoelectric heating (Bakes & Tielens 94 / Wolfire+03) ----
    if (G_dust == 0.0 || c.dust_to_gas_ratio < 1e-8 || iflag_cmb == 1) {
        rates[11] = 0.0;
    } else if (yne < 1e-9 * G_dust * cl[16]) {
        rates[11] = 0.0;
    } else {
        double PEvar0 = G_dust * cl[16] / yne;
        double PEvar1 = 1.0 + 4e-3 * std::pow(PEvar0, (double)0.73f);
        double PEvar2 = 1.0 + 2e-4 * PEvar0;
        double eps = (4.9e-2 / PEvar1) + (cl[17] / PEvar2);
        rates[11] = -1.3e-24 * eps * G_dust * yn * c.dust_to_gas_ratio;
#ifndef CHEMCOOL_PE_K23
        // grain recombination cooling (SGChem rates(29)), kept in the PE rate
        rates[11] = rates[11] + c.dust_to_gas_ratio * cl[76]
                  * std::pow(PEvar0, cl[75]) * abe * (yn * yn);
#endif
#ifdef CHEMCOOL_PE_K23
        {   // Kim+2023 Eqs. 30-31 (psi + 50), plus WD01 Eq. 45 recombination cooling (R_V=3.1, b_C=4, ISRF)
            double psi = G_dust * std::sqrt(temp) / yne + 50.0;
            double epsk = (5.22 + 2.25 * std::pow(temp, 0.147))
                        / (1.0 + 0.04996 * std::pow(psi, 0.431) * (1.0 + 0.0043 * std::pow(psi, 0.692)));
            double lp = std::log(psi);
            rates[11] = (-1e-26 * epsk * G_dust
                         + 1e-28 * yne * std::pow(temp, 0.3959 + 2.380 / lp)
                           * std::exp(-6.554 + 1.575 * lp - 0.05674 * lp * lp))
                      * yn * c.dust_to_gas_ratio;
        }
#endif
    }

    // ---- (R13) OI fine-structure cooling ----
    {
        double oxc10 = cl[18] * ynh + cl[21] * ynh2 + cl[24] * yne + cl[27] * ynhp;
        double oxc20 = cl[19] * ynh + cl[22] * ynh2 + cl[25] * yne + cl[28] * ynhp;
        double oxc21 = cl[20] * ynh + cl[23] * ynh2 + cl[26] * yne + cl[29] * ynhp;

        if (abo <= 1e-5 * c.abundo) {
            rates[12] = 0.0;
        } else if (oxc10 == 0.0 && oxc20 == 0.0 && oxc21 == 0.0) {
            rates[12] = 0.0;
        } else {
            double oxa = std::exp(-FS_OXE10 / (kboltz * temp));
            double oxb = std::exp(-FS_OXE20 / (kboltz * temp));
            double oxcc= std::exp(-FS_OXE21 / (kboltz * temp));
            double oxc01 = 0.6 * oxc10 * oxa;
            double oxc02 = 0.2 * oxc20 * oxb;
            double oxc12 = (1.0 / 3.0) * oxc21 * oxcc;

            double oxb10 = cxx_compute_stim(FS_OXA10, FS_OXE10, CMB_temp);
            double oxb01 = 0.6 * oxb10;
            double oxb20 = cxx_compute_stim(FS_OXA20, FS_OXE20, CMB_temp);
            double oxb02 = 0.2 * oxb20;
            double oxb21 = cxx_compute_stim(FS_OXA21, FS_OXE21, CMB_temp);
            double oxb12 = (1.0 / 3.0) * oxb21;

            double R01 = oxc01 + oxb01, R02 = oxc02 + oxb02, R12 = oxc12 + oxb12;
            double R10 = oxc10 + FS_OXA10 + oxb10;
            double R20 = oxc20 + FS_OXA20 + oxb20;
            double R21 = oxc21 + FS_OXA21 + oxb21;

            double n0, n1, n2;
            if (R01 < 1e-40) { n0 = 1.0; n1 = 0.0; n2 = 0.0; }
            else cxx_three_level_pops(R01, R02, R12, R10, R20, R21, &n0, &n1, &n2);

            double lam1 = (FS_OXA10 + oxb10) * FS_OXE10 * n1
                        + ((FS_OXA20 + oxb20) * FS_OXE20
                         + (FS_OXA21 + oxb21) * FS_OXE21) * n2;
            double lam2 = (oxb01 * FS_OXE10 + oxb02 * FS_OXE20) * n0
                        + oxb12 * FS_OXE21 * n1;
            rates[12] = (lam1 - lam2) * abo * yn;
        }
    }

    // ---- (R14) CI fine-structure cooling ----
    {
        double cIc10 = cl[30] * ynh + cl[33] * ynh2 + cl[36] * yne + cl[39] * ynhp;
        double cIc20 = cl[31] * ynh + cl[34] * ynh2 + cl[37] * yne + cl[40] * ynhp;
        double cIc21 = cl[32] * ynh + cl[35] * ynh2 + cl[38] * yne + cl[41] * ynhp;

        if (abcI <= 1e-5 * c.abundc) {
            rates[13] = 0.0;
        } else if (cIc10 == 0.0 && cIc20 == 0.0 && cIc21 == 0.0) {
            rates[13] = 0.0;
        } else {
            double a = std::exp(-FS_CIE10 / (kboltz * temp));
            double b = std::exp(-FS_CIE20 / (kboltz * temp));
            double cc= std::exp(-FS_CIE21 / (kboltz * temp));
            double cIc01 = 3.0 * cIc10 * a;
            double cIc02 = 5.0 * cIc20 * b;
            double cIc12 = (5.0 / 3.0) * cIc21 * cc;

            double b10 = cxx_compute_stim(FS_CIA10, FS_CIE10, CMB_temp);
            double b01 = 3.0 * b10;
            double b20 = cxx_compute_stim(FS_CIA20, FS_CIE20, CMB_temp);
            double b02 = 5.0 * b20;
            double b21 = cxx_compute_stim(FS_CIA21, FS_CIE21, CMB_temp);
            double b12 = (5.0 / 3.0) * b21;

            double R01 = cIc01 + b01, R02 = cIc02 + b02, R12 = cIc12 + b12;
            double R10 = cIc10 + FS_CIA10 + b10;
            double R20 = cIc20 + FS_CIA20 + b20;
            double R21 = cIc21 + FS_CIA21 + b21;

            double n0, n1, n2;
            cxx_three_level_pops(R01, R02, R12, R10, R20, R21, &n0, &n1, &n2);

            double lam1 = (FS_CIA10 + b10) * FS_CIE10 * n1
                        + ((FS_CIA20 + b20) * FS_CIE20
                         + (FS_CIA21 + b21) * FS_CIE21) * n2;
            double lam2 = (b01 * FS_CIE10 + b02 * FS_CIE20) * n0
                        + b12 * FS_CIE21 * n1;
            rates[13] = (lam1 - lam2) * abcI * yn;
        }
    }

    // ---- (R15) SiI fine-structure cooling (proton rates from HM89) ----
    {
        double siIc10 = cl[42] * ynh + 7.2e-9 * ynhp;
        double siIc20 = cl[43] * ynh + 7.2e-9 * ynhp;
        double siIc21 = cl[44] * ynh + 2.2e-8 * ynhp;

        if (absiI <= 1e-5 * c.abundsi) {
            rates[14] = 0.0;
        } else if (siIc10 == 0.0 && siIc20 == 0.0 && siIc21 == 0.0) {
            rates[14] = 0.0;
        } else {
            double a = std::exp(-FS_SIIE10 / (kboltz * temp));
            double b = std::exp(-FS_SIIE20 / (kboltz * temp));
            double cc= std::exp(-FS_SIIE21 / (kboltz * temp));
            double c01 = 3.0 * siIc10 * a;
            double c02 = 5.0 * siIc20 * b;
            double c12 = (5.0 / 3.0) * siIc21 * cc;

            double b10 = cxx_compute_stim(FS_SIIA10, FS_SIIE10, CMB_temp);
            double b01 = 3.0 * b10;
            double b20 = cxx_compute_stim(FS_SIIA20, FS_SIIE20, CMB_temp);
            double b02 = 5.0 * b20;
            double b21 = cxx_compute_stim(FS_SIIA21, FS_SIIE21, CMB_temp);
            double b12 = (5.0 / 3.0) * b21;

            double R01 = c01 + b01, R02 = c02 + b02, R12 = c12 + b12;
            double R10 = siIc10 + FS_SIIA10 + b10;
            double R20 = siIc20 + FS_SIIA20 + b20;
            double R21 = siIc21 + FS_SIIA21 + b21;

            double n0, n1, n2;
            cxx_three_level_pops(R01, R02, R12, R10, R20, R21, &n0, &n1, &n2);

            double lam1 = (FS_SIIA10 + b10) * FS_SIIE10 * n1
                        + ((FS_SIIA20 + b20) * FS_SIIE20
                         + (FS_SIIA21 + b21) * FS_SIIE21) * n2;
            double lam2 = (b01 * FS_SIIE10 + b02 * FS_SIIE20) * n0
                        + b12 * FS_SIIE21 * n1;
            rates[14] = (lam1 - lam2) * absiI * yn;
        }
    }

    // ---- (R16) CII fine-structure cooling (LVG escape probability) ----
    {
        double cIIc10 = cl[45] * ynh + cl[46] * ynh2 + cl[47] * yne;
        double cIIc01 = cl[48] * cIIc10;

        if (cIIc10 == 0.0 || abcII <= 1e-5 * c.abundc) {
            rates[15] = 0.0;
        } else {
            double b10 = cxx_compute_stim(FS_CIIA10, FS_CIIE10, CMB_temp);
            double b01 = 2.0 * b10;

            double vth = std::sqrt(kboltz * temp / (6.0 * CHEM_PROTONMASS));
            double N_jeans = (yn * abcII) / (vth / dl);
            double N_LVG   = (yn * abcII) / std::fabs(divv);
            double col_eff = (N_jeans < N_LVG) ? N_jeans : N_LVG;

            double tau  = 7.2e-13 * col_eff;
            double beta = 1.0 / (1.0 + 3.0 * tau);
            double a10_eff = FS_CIIA10 * beta;

            double v0 = (cIIc10 + a10_eff + b10);
            double v1 = (cIIc01 + b01);
            double n0 = v0 / (v0 + v1);
            double n1 = v1 / (v0 + v1);
            double v2 = (a10_eff + b10) * FS_CIIE10 * n1;
            double v3 = b01 * FS_CIIE10 * n0;
            rates[15] = (v2 - v3) * abcII * yn;
        }
    }

    // ---- (R17) SiII fine-structure cooling ----
    {
        double c10 = cl[49] * ynh + cl[50] * yne;
        double c01 = cl[51] * c10;
        if (c10 == 0.0 || absiII <= 1e-5 * c.abundsi) {
            rates[16] = 0.0;
        } else {
            double b10 = cxx_compute_stim(FS_SIIIA10, FS_SIIIE10, CMB_temp);
            double b01 = 2.0 * b10;
            double v0 = (c10 + FS_SIIIA10 + b10);
            double v1 = (c01 + b01);
            double n0 = v0 / (v0 + v1);
            double n1 = v1 / (v0 + v1);
            double v2 = (FS_SIIIA10 + b10) * n1;
            double v3 = b01 * n0;
            rates[16] = (v2 - v3) * FS_SIIIE10 * absiII * yn;
        }
    }

    // ---- (R18) Compton cooling ----
    rates[17] = 1.017e-37 * chem_fpowi(CMB_temp, 4) * (temp - CMB_temp) * yne;

    // ---- (R19) HD cooling ----
    if (abhd > 0.0) {
        double r19scale = 1.0, lnh;
        if (yn >= 1e8)      { lnh = 8.0; }
        else if (ynh < 1.0) { lnh = 0.0; r19scale = ynh; }
        else                { lnh = std::log10(ynh); }
        rates[18] = std::pow(10.0, cl[53] + lnh * (cl[54] + lnh * (cl[55]
                    + lnh * (cl[56] + lnh * cl[57])))) * ynhd;
        rates[18] *= r19scale;
    } else {
        rates[18] = 0.0;
    }

    // ---- (R20) H2 CIE cooling, with the Turk opacity cutoff ----
    rates[19] = cl[58] * yn * (2.0 * ynh2);
    {
        double tau_CIE = std::pow(yn / 1.4e16, 2.8);
        if (tau_CIE < 1e-5) tau_CIE = 1e-5;
        double f = (1.0 - std::exp(-tau_CIE)) / tau_CIE;
        rates[19] *= (f < 1.0) ? f : 1.0;
    }

    // cool_func.F:1223 -- rates(3) is unconditionally zeroed here, then
    // replaced by the WSS CIE metal-line rate when WSS_CIE_COOL is on.
    rates[2] = 0.0;
#ifdef WSS_CIE_COOL
    rates[2] = cxx_wss_cie_rate(T, temp, c.Zmass) * (yn * yn);
#endif

    if (iflag_cmb == 0)   // the T_CMB floor call leaves the published rates alone
        for (int k = 0; k < CHEM_NRATES; k++) c.lambda[k] = rates[k];

    double s = 0.0;
    for (int k = 0; k < CHEM_NRATES; k++) s += rates[k];
    *ylam = s;
    return 0;
}
