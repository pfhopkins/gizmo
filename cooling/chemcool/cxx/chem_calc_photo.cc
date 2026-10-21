// chem_calc_photo.cc -- Gate-3 port of calc_photo.F.
//
//   chi                         (calc_photo.F:683-735)
//   calc_H2_shielding_factor    (calc_photo.F:745-823)  DB96 curve of growth
//   calc_HD_shielding_factor    (calc_photo.F:825-914)
//   calc_photo (+ wrapper)      (calc_photo.F:73-666)
//
// Effective-column convention (2026-07-28 variable-DGR work, preserved here):
//   * AV is built from AV_conversion_factor * dust_to_gas_ratio * N_H, i.e. the
//     DGR rides on the *conversion*, so a variable per-cell DGR changes AV but
//     NOT the column; and
//   * f_dust uses sigma_dust = 2e-21 * dust_to_gas_ratio applied to the RAW
//     column (plus the external NH_ext = AV_ext / AV_conversion_factor), which
//     is the same product written the other way round.  The two must stay
//     consistent -- writing f_dust = exp(-AV * 2e-21/AV_conversion_factor) in
//     the non-pixel branch is the SAME number, and that is exactly what the
//     Fortran does at line 203.  Reproduced literally, both branches.
//
// H2 self-shielding is applied ONCE, in the RHS (ykdis = phrates(1)*fshield),
// for nets != 1 -- NOT baked into phrates(1) here (h2_double_shield_bug,
// 2026-07-07).  phrates(1) below therefore carries only the dust factor.

#include "../../../GIZMO_config.h"
#include "chem_types.h"
#include "chem_net20.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

// shield_data.h parameters
#define NCO_SHIELD_MIN 1e12
#define NCO_SHIELD_MAX 1e20
#define NH2_SHIELD_MIN 1e13
#define NH2_SHIELD_MAX 1e22
#define DNSHIELD       0.05
#define IOFF_H2        259
#define IOFF_CO        239

// ===========================================================================
// chi(AV) -- ISRF attenuation factor (calc_photo.F:683-735)
// ===========================================================================
double cxx_chi(const ChemTables& T, double AV, int isrf_option)
{
    const double* tab = (isrf_option == 1) ? T.chi_tab_mathis : T.chi_tab_black;
    if (AV == 0.0) return 1.0;
    if (AV > 2e2)  return tab[199];          // constant above AV = 200

    int index = (int)AV;
    if (index == 200) {
        // The Fortran has a bare `stop` here (DEVIATIONS D-4: not ported --
        // a bare stop kills one MPI rank and deadlocks the job).  Unreachable:
        // AV > 200 returned above and AV == 200.0 exactly gives index 200 only
        // if AV == 200, which the `AV > 2e2` test lets through -- so guard.
        std::printf("Index = 200... Whoops.\n");
        return tab[199];
    }
    double dchi, dAV = AV - index;
    if (index == 0) { dchi = tab[0] - 1.0;              return 1.0 + dAV * dchi; }
    dchi = tab[index] - tab[index - 1];
    return tab[index - 1] + dAV * dchi;
}

// ===========================================================================
// calc_H2_shielding_factor (calc_photo.F:745-823)
// Evaluated at EVERY RHS call with the live abh2 and T (audit R5).
// ===========================================================================
double cxx_calc_H2_shielding_factor(double temp, double abh2, double NH,
                                    ChemCell& c)
{
    double b  = std::sqrt(CHEM_KBOLTZ * temp / CHEM_PROTONMASS);
#ifdef CHEMCOOL_H2SHIELD_B5
    b = CHEMCOOL_H2SHIELD_B5 * 1e5;
#endif
    double b5 = b / 1e5;
    double fshield = 0.0;
    double NH2_local = abh2 * NH;

    if (c.iphoto == 2) {
#ifndef RAYTRACE
        std::printf("CALC_PHOTO: invalid photodissociation option\n");
        std::printf("CALC_PHOTO: recompile with RAYTRACE enabled\n");
        c.cool_func_failed = 1;              // ABORT(121)
        return 1.0;
#endif
    } else if (c.iphoto == 6) {
#ifndef TREE_RAD_H2
        std::printf("CALC_PHOTO: invalid photodissociation option\n");
        std::printf("CALC_PHOTO: recompile with TREE_RAD_H2 enabled\n");
        c.cool_func_failed = 1;              // ABORT(122)
        return 1.0;
#else
        // nray MUST be NPIX for the pixel average -- this holds for the
        // standalone TREE_RAD_H2 (M1) path too (Uli, 2026-07-08).
        int nray = c.npix;
        for (int I = 0; I < nray; I++) {
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
            double fac_uv_pix = c.fac_uv[I];
#else
            double fac_uv_pix = 1.0 / nray;
#endif
            double h2dis0 = (NH2_local + c.col_h2[I]) / 5e14;
            double h2dis1 = std::sqrt(1.0 + h2dis0);
            fshield += fac_uv_pix * (9.65e-1 / ((1.0 + h2dis0 / b5)
                                              * (1.0 + h2dis0 / b5))
                     + 3.5e-2 * std::exp(-8.5e-4 * h2dis1) / h2dis1);
        }
        return fshield;
#endif
    } else if (c.iphoto > 6) {
        std::printf("CALC_PHOTO: invalid photodissociation option \n");
        c.cool_func_failed = 1;              // ABORT(123)
        return 1.0;
    } else {
        double h2dis0 = NH2_local / 5e14;
        double h2dis1 = std::sqrt(1.0 + h2dis0);
        fshield = 9.65e-1 / chem_fpowi(1.0 + h2dis0 / b5, 2)
                + 3.5e-2 * std::exp(-8.5e-4 * h2dis1) / h2dis1;
    }
    return fshield;
}

// ===========================================================================
// calc_HD_shielding_factor (calc_photo.F:825-914)
// ===========================================================================
double cxx_calc_HD_shielding_factor(double temp, double abh2, double abhd,
                                    double NH, ChemCell& c)
{
    double b  = std::sqrt(CHEM_KBOLTZ * temp / CHEM_PROTONMASS);
    double b5 = b / 1e5;
    double fshield_hd = 0.0;
    double NHD_local = abhd * NH;

    if (c.iphoto == 2) {
        /* no-op in the Fortran */
    } else if (c.iphoto == 6) {
#ifdef TREE_RAD_H2
        for (int I = 0; I < c.npix; I++) {
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
            double fac_uv_pix = c.fac_uv[I];
#else
            double fac_uv_pix = 1.0 / c.npix;
#endif
            double hddis0;
            if (abh2 == 0.0 || abhd == 0.0)
                hddis0 = (NHD_local + c.abundD * c.col_h2[I]) / 5e14;
            else
                hddis0 = (NHD_local + (abhd / abh2) * c.col_h2[I]) / 5e14;
            double hddis1 = std::sqrt(1.0 + hddis0);
            // NB the Fortran weights only the FIRST term by fac_uv_pix; the
            // second term is added unweighted (once per pixel).  Literal.
            if (hddis0 < 1e1) {
                fshield_hd += fac_uv_pix * 9.5e-1 / chem_fpowi(1.0 + hddis0 / b5, 2)
                            + 5.0e-2 * std::exp(-8.5e-4 * hddis1) / hddis1;
            } else if (hddis0 < 1e4) {
                fshield_hd += fac_uv_pix * 9.4e-1 / chem_fpowi(1.0 + hddis0 / b5, 2)
                            + 6.0e-2 / std::pow(hddis1, 1.5);
            } else {
                fshield_hd += fac_uv_pix * 9.85e-1 / chem_fpowi(1.0 + hddis0 / b5, 2)
                            + 4.0e-3 / std::pow(hddis1, 0.9);
            }
        }
        return fshield_hd;
#endif
    } else {
        double hddis0 = NHD_local / 5e14;
        double hddis1 = std::sqrt(1.0 + hddis0);
        if (hddis0 < 1e1) {
            fshield_hd += 9.5e-1 / chem_fpowi(1.0 + hddis0 / b5, 2)
                        + 5.0e-2 * std::exp(-8.5e-4 * hddis1) / hddis1;
        } else if (hddis0 < 1e4) {
            fshield_hd += 9.4e-1 / chem_fpowi(1.0 + hddis0 / b5, 2)
                        + 6.0e-2 / std::pow(hddis1, 1.5);
        } else {
            fshield_hd += 9.85e-1 / chem_fpowi(1.0 + hddis0 / b5, 2)
                        + 4.0e-3 / std::pow(hddis1, 0.9);
        }
    }
    return fshield_hd;
}

// helper: the L96 CO / H2 shielding-curve lookup shared by the two branches
static inline double co_shield_lookup(const double* tab, int imax, double N,
                                      double Nmin, double Nmax, int ioff,
                                      int zero_index_is_unity)
{
    if (N <= Nmin) return 1.0;
    if (N >= Nmax) return tab[imax - 1];
    int ipos = (int)(std::log10(N) / DNSHIELD) - ioff;
    if (zero_index_is_unity && ipos == 0) return 1.0;
    double dth   = tab[ipos] - tab[ipos - 1];
    double dNact = std::log10(N) - (ipos + ioff) * DNSHIELD;
    return tab[ipos - 1] + dth * (dNact / DNSHIELD);
}

// ===========================================================================
// calc_photo (calc_photo.F:73-666).  Fills c.phrates / c.G_dust / c.chi_isrf /
// c.fshield_CO.  Called once per cell call (audit R5).
// ===========================================================================
void cxx_calc_photo(double temp, double abh2, double abhd, double abco,
                    ChemCell& c)
{
    if (c.net == 20) { cxx_calc_photo_net20(c); return; }
    const ChemTables& T = *c.tab;
    const int npr = CHEM_NPHTAB;

    if (c.G0 == 0) {
        for (int i = 0; i < npr; i++) c.phrates[i] = 0.0;
        c.G_dust = 0.0;
        return;
    }

    double yn = c.yn, dl = c.dl;
    double localapprox = (c.iphoto != 7) ? dl * yn : 0.0;
    double AVlocal = c.AV_conversion_factor * c.dust_to_gas_ratio * localapprox;
    double NH;
    if (c.iphoto == 3)                         NH = dl * yn;
    else if (c.iphoto == 4 || c.iphoto == 5)   NH = c.column_est / 1.4 / CHEM_PROTONMASS
                                                  + localapprox;
    else                                       NH = localapprox;

    double fshield = (c.iphoto == 0)
                   ? 1.0
                   : cxx_calc_H2_shielding_factor(temp, abh2, NH, c);

    double fshield_HD = (c.iphoto == 0 || c.abundD == 0)
                      ? 1.0
                      : cxx_calc_HD_shielding_factor(temp, abh2, abhd, NH, c);

    double f_dust = 1.0, f_simp = 1.0, df05 = 1.0, df19 = 1.0;
    double df25 = 1.0, df30 = 1.0, AV = 0.0;
    double df15 = 1.0, df17 = 1.0;   // net 15 (NL99) only
    double df187 = 1.0;   // sigma_PE = 1e-21 Z' (Gong+17; Kim+2023 under CHEMCOOL_PE_K23)
#if CHEMISTRYNETWORK == 16
    double df418 = 1.0, df388 = 1.0, df376 = 1.0,
           df261 = 1.0, df212 = 1.0, df266 = 1.0;   // Gong+17 (SGChem)
#endif

    if ((c.iphoto != 2 && c.iphoto != 6) || c.dust_to_gas_ratio == 0.0) {
        if (c.iphoto == 0) AV = c.AV_ext;
        else               AV = c.AV_ext + c.AV_conversion_factor
                              * c.dust_to_gas_ratio * NH;
        if (AV > 0.0) {
            f_dust = std::exp(-AV * (2e-21 / c.AV_conversion_factor));
            c.chi_isrf = cxx_chi(T, AV, c.isrf_option);
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6
            f_simp = std::exp(-2.5 * AV);
#else
            f_simp = 1.0;
#endif
            df25 = std::exp(-2.5 * AV);
            df30 = std::exp(-3.0 * AV);
#if CHEMISTRYNETWORK != 4 && CHEMISTRYNETWORK != 5 && CHEMISTRYNETWORK != 6
            df05 = std::exp(-0.5 * AV);
            df19 = std::exp(-1.9 * AV);
#endif
#if CHEMISTRYNETWORK == 15
            df15 = std::exp(-1.5 * AV);
            df17 = std::exp(-1.5 * AV);   // 1.5 as in calc_photo.F (Glover code)
#endif
#if CHEMISTRYNETWORK == 16 || defined(CHEMCOOL_PE_K23)
            df187 = std::exp(-1.87 * AV);
#endif
#if CHEMISTRYNETWORK == 16
            df418 = std::exp(-4.18 * AV);
            df388 = std::exp(-3.88 * AV);
            df376 = std::exp(-3.76 * AV);
            df261 = std::exp(-2.61 * AV);
            df212 = std::exp(-2.12 * AV);
            df266 = std::exp(-2.66 * AV);
#endif
        } else {
            f_dust = 1.0; f_simp = 1.0;
            df05 = 1.0; df19 = 1.0; df25 = 1.0; df30 = 1.0;
            c.chi_isrf = 1.0;
        }
    } else {    // iphoto == 2 (six-ray) or 6 (TreeCol)
#if defined RAYTRACE || defined TREE_RAD || defined TREE_RAD_H2
        double sigma_dust = 2e-21 * c.dust_to_gas_ratio;
        f_dust = 0.0; f_simp = 0.0;
        df05 = 0.0; df19 = 0.0; df25 = 0.0; df30 = 0.0;
        df15 = 0.0; df17 = 0.0;
#if CHEMISTRYNETWORK == 16 || defined(CHEMCOOL_PE_K23)
        df187 = 0.0;
#endif
#if CHEMISTRYNETWORK == 16
        df418 = 0.0; df388 = 0.0; df376 = 0.0;
        df261 = 0.0; df212 = 0.0; df266 = 0.0;
#endif
        c.chi_isrf = 0.0;
        int nray = c.npix;
        for (int I = 0; I < nray; I++) {
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
            double fac_uv_pix = c.fac_uv[I];
#else
            double fac_uv_pix = 1.0 / nray;
#endif
            double NH_ext = c.AV_ext / c.AV_conversion_factor;
            f_dust += fac_uv_pix * std::exp(-sigma_dust * (c.col_tot[I] + NH_ext));
            AV = c.AV_ext + c.AV_conversion_factor * c.dust_to_gas_ratio
                          * c.col_tot[I];
            AV += AVlocal;
            c.chi_isrf += fac_uv_pix * cxx_chi(T, AV, c.isrf_option);
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6
            f_simp += fac_uv_pix * std::exp(-2.5 * AV);
#else
            f_simp = 1.0;
#endif
            df25 += fac_uv_pix * std::exp(-2.5 * AV);
            df30 += fac_uv_pix * std::exp(-3.0 * AV);
#if CHEMISTRYNETWORK != 4 && CHEMISTRYNETWORK != 5 && CHEMISTRYNETWORK != 6
            df05 += fac_uv_pix * std::exp(-0.5 * AV);
            df19 += fac_uv_pix * std::exp(-1.9 * AV);
#endif
#if CHEMISTRYNETWORK == 15
            df15 += fac_uv_pix * std::exp(-1.5 * AV);
            df17 += fac_uv_pix * std::exp(-1.5 * AV);
#endif
#if CHEMISTRYNETWORK == 16 || defined(CHEMCOOL_PE_K23)
            df187 += fac_uv_pix * std::exp(-1.87 * AV);
#endif
#if CHEMISTRYNETWORK == 16
            df418 += fac_uv_pix * std::exp(-4.18 * AV);
            df388 += fac_uv_pix * std::exp(-3.88 * AV);
            df376 += fac_uv_pix * std::exp(-3.76 * AV);
            df261 += fac_uv_pix * std::exp(-2.61 * AV);
            df212 += fac_uv_pix * std::exp(-2.12 * AV);
            df266 += fac_uv_pix * std::exp(-2.66 * AV);
#endif
        }
#endif
    }

    // M1: the RT solver already applied continuum dust attenuation.
#ifdef RADTRANSFER
    f_dust = 1.0; c.chi_isrf = 1.0; f_simp = 1.0; df25 = 1.0; df30 = 1.0;
    df05 = 1.0; df19 = 1.0; df15 = 1.0; df17 = 1.0;
#if CHEMISTRYNETWORK == 16 || defined(CHEMCOOL_PE_K23)
    df187 = 1.0;
#endif
#if CHEMISTRYNETWORK == 16
    df418 = 1.0; df388 = 1.0; df376 = 1.0;
    df261 = 1.0; df212 = 1.0; df266 = 1.0;
#endif
#endif
    // TreeRay attenuates the STELLAR bands inside the walk, but chi_isrf KEEPS
    // its computed AV attenuation (the isotropic ISRF background still needs
    // target-side shielding).
#ifdef TREE_RAY
    f_dust = 1.0; f_simp = 1.0; df25 = 1.0; df30 = 1.0;
    df05 = 1.0; df19 = 1.0; df15 = 1.0; df17 = 1.0;
#if CHEMISTRYNETWORK == 16 || defined(CHEMCOOL_PE_K23)
    df187 = 1.0;
#endif
#if CHEMISTRYNETWORK == 16
    df418 = 1.0; df388 = 1.0; df376 = 1.0;
    df261 = 1.0; df212 = 1.0; df266 = 1.0;
#endif
#endif

    // ---- CO self-shielding + shielding by H2 (L96) ----
    double fshield_CO = 1.0;
#if CHEMISTRYNETWORK != 1 && CHEMISTRYNETWORK != 4
#ifdef CO_SHIELDING
    if (c.iphoto != 2 && c.iphoto != 6) {
        double NCO = abco * NH;
        double theta_CO = co_shield_lookup(T.CO_self_shielding, CHEM_IMAX_COSS,
                                           NCO, NCO_SHIELD_MIN, NCO_SHIELD_MAX,
                                           IOFF_CO, 0);
        double NH2 = abh2 * NH;
        double theta_H2 = co_shield_lookup(T.CO_H2_shielding, CHEM_IMAX_COH2,
                                           NH2, NH2_SHIELD_MIN, NH2_SHIELD_MAX,
                                           IOFF_H2, 1);
        fshield_CO = theta_H2 * theta_CO;
    } else {
#if defined RAYTRACE || defined TREE_RAD_H2
        double theta_CO = 0.0, theta_H2 = 0.0;
        int nray = c.npix;
        for (int I = 0; I < nray; I++) {
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
            double fac_uv_pix = c.fac_uv[I];
#else
            double fac_uv_pix = 1.0 / nray;
#endif
            theta_CO += fac_uv_pix
                      * co_shield_lookup(T.CO_self_shielding, CHEM_IMAX_COSS,
                                         c.col_co[I], NCO_SHIELD_MIN,
                                         NCO_SHIELD_MAX, IOFF_CO, 1);
            theta_H2 += fac_uv_pix
                      * co_shield_lookup(T.CO_H2_shielding, CHEM_IMAX_COH2,
                                         c.col_h2[I], NH2_SHIELD_MIN,
                                         NH2_SHIELD_MAX, IOFF_H2, 1);
        }
        fshield_CO = theta_H2 * theta_CO;
#endif
    }
#endif /* CO_SHIELDING */
#endif /* NEEDS_CO_RATES */
    c.fshield_CO = fshield_CO;

    // ---- H2 photodissociation: Lyman-Werner band, dust factor only ----
#if CHEMISTRYNETWORK != 1
    c.phrates[0] = T.phtab[0] * c.G0_LW * f_dust;
#else
    c.phrates[0] = T.phtab[0] * c.G0_LW * fshield * f_dust;
#endif
    c.fac_shield_h2 = fshield;     // OUTPUT_SHIELD_FAC diagnostics (calc_photo.F)
    c.fac_shield_dust = f_dust;

    if (c.net == 4) goto done;

    if (c.net == 5 || c.net == 6) {
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6
#ifdef CO_SHIELDING
        c.phrates[9] = T.phtab[9] * c.G0 * f_simp * fshield_CO;
#else
        c.phrates[9] = T.phtab[9] * c.G0 * f_simp;
#endif
#endif
        // M1 RT: HI photoionization + heating for net 5/6 (2026-07-04).  The
        // shared HI block below is skipped by this early exit, so replicate it.
        c.phrates[37] = T.phtab[37] * c.G0 + c.rt_phot_HI;
        c.phrates[41] = T.phtab[41] * c.G0 + c.rt_heat_HI;
        goto done;
    }

#if CHEMISTRYNETWORK == 16
    // Gong+17 (SGChem) dust factors; rate_eq_gong17 reads phrates 1, 2, 3, 10, 48, 49
    c.phrates[0]  = T.phtab[0]  * c.G0_LW * df418;
    c.phrates[1]  = T.phtab[1]  * c.G0 * df376;
    c.phrates[2]  = T.phtab[2]  * c.G0 * df261;
    c.phrates[9]  = T.phtab[9]  * c.G0 * fshield_CO * df388;
    c.phrates[47] = T.phtab[47] * c.G0 * df212;
    c.phrates[48] = T.phtab[48] * c.G0 * df266;
    goto done;
#endif

    // ---- other networks: shared rates ----
    c.phrates[3] = T.phtab[3] * c.G0 * df05;
    c.phrates[4] = T.phtab[4] * c.G0 * df19;
    c.phrates[7] = T.phtab[7] * c.G0 * df05;
    c.phrates[8] = T.phtab[8] * c.G0 * df19;
    c.phrates[34] = T.phtab[34] * c.G0_LW * fshield_HD * f_dust;

    c.phrates[37] = T.phtab[37] * c.G0;   // HI
    c.phrates[38] = T.phtab[38] * c.G0;   // HeI
    c.phrates[39] = T.phtab[39] * c.G0;   // HeII
    c.phrates[40] = T.phtab[40] * c.G0;   // H2
    c.phrates[45] = T.phtab[45] * c.G0;   // OI
    c.phrates[46] = T.phtab[46] * c.G0;   // SiII
    c.phrates[41] = T.phtab[41] * c.G0;   // HI   heat
    c.phrates[42] = T.phtab[42] * c.G0;   // HeI  heat
    c.phrates[43] = T.phtab[43] * c.G0;   // HeII heat
    c.phrates[44] = T.phtab[44] * c.G0;   // H2   heat

    c.phrates[37] += c.rt_phot_HI;
    c.phrates[38] += c.rt_phot_HeI;
    c.phrates[39] += c.rt_phot_HeII;
    c.phrates[41] += c.rt_heat_HI;
    c.phrates[42] += c.rt_heat_HeI;
    c.phrates[43] += c.rt_heat_HeII;

    if (c.net == 1) goto done;

    c.phrates[1] = T.phtab[1] * c.G0 * df30;
    c.phrates[5] = T.phtab[5] * c.G0 * df30;
    if (c.net != 7 && c.net != 8 && c.net != 11) {
        c.phrates[2] = T.phtab[2] * c.G0 * df19;
        c.phrates[6] = T.phtab[6] * c.G0 * df19;
    }
    if (c.net == 2) goto done;

    // CO
    c.phrates[9] = T.phtab[9] * c.G0 * fshield_CO * df25;
    // Network 17 needs only phrates 1,4,5,8,9,10,35,38-45
    if (c.net == 17) goto done;

    // Network 15 (NL99) reads only phrates 1, 2, 10, 48-51 of the remaining set
    if (c.net == 15) {
        c.phrates[47] = T.phtab[47] * c.G0 * df15;
        c.phrates[48] = T.phtab[48] * c.G0 * df17;
        c.phrates[49] = T.phtab[49] * c.G0 * df25;
        c.phrates[50] = T.phtab[50] * c.G0 * df19;
        goto done;
    }

    // nets 3/7/8/11 not ported (DEVIATIONS D-14)
    std::printf("CHEMCOOL_CXX calc_photo: network %d not ported\n", c.net);

done:
    // f_rsc rescales the field strength to Habing units (photoinit sets it).
    (void)f_simp;   // used only under CHEMISTRYNETWORK 5/6 (UNUSED_PARAM in the .F)
    (void)df187;    // read only by net 16 or under CHEMCOOL_PE_K23
#if CHEMISTRYNETWORK == 16 || defined(CHEMCOOL_PE_K23)
    c.G_dust = c.f_rsc * c.G0 * df187;
#else
    c.G_dust = c.f_rsc * c.G0 * df25;
#endif
#ifdef CHEMCOOL_CR_ATTEN_K23
    {   // Kim+2023 Eq. 55: N_eff = tau_PE,eff / (1e-21 Z'_d), xi ~ 1/N_eff above N_0 = 9.35e20
#if CHEMISTRYNETWORK == 16 || defined(CHEMCOOL_PE_K23)
        double cr_tau  = -std::log(std::max(df187, 1e-300));
#else
        double cr_tau  = -std::log(std::max(df25, 1e-300));
#endif
        double cr_neff = cr_tau / (1e-21 * std::max(c.dust_to_gas_ratio, 1e-30));
        if (cr_neff > 9.35e20) c.cosmic_ray_ion_rate *= 9.35e20 / cr_neff;
    }
#endif
}
