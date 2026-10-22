// chem_cool_util.cc -- Gate-3 port of cool_util.F.
//
// Ported here, in the order they appear in the Fortran:
//   co_cool                 (cool_util.F:10-152)
//   h2o_rot_cool            (cool_util.F:168-288)
//   h2o_vib_cool            (cool_util.F:304-369)
//   three_level_pops        (cool_util.F:385-422)
//   compute_stim            (cool_util.F:439-456)
//   compute_h2_opacity      (cool_util.F:506-560)
//   compute_h2_heating_fraction (cool_util.F:574-680; live in net 15 only)
//   calc_dust_temp          (cool_util.F:802-1007, incl. the 2026-07 bounded
//                            Newton fix and the hot/dust-free tdust cap)
//
// NOT ported (see DEVIATIONS.md D-12): load_H2_table lives in
// chem_tables_build.cc (it is table construction, not a kernel);
// dma_calc is `#if defined DMA`,
// which no Stella build sets and whose body returns hard zeros anyway.
//
// Every integer index below is the Fortran 1-based index; the accessors
// chem_f2()/[i-1] do the translation, so the code reads like the .F source.

#include "../../../GIZMO_config.h"
#include "chem_types.h"
#include "chem_tables_data.h"
#include <cmath>
#include <cstdio>

// Fortran INT() truncates toward zero; C's (int) cast does the same.
static inline int f_int(double x) { return (int)x; }

// ===========================================================================
// co_cool (cool_util.F:10-152)
// ===========================================================================
void cxx_co_cool(const ChemTables& T, double temp, double N_co_eff,
                 double N_co_eff_vib, CoCool* out)
{
    const int nTco = CHEM_NTCO, ncdco = CHEM_NCDCO;
    const int nTco_vib = CHEM_NTCO_VIB, ncdco_vib = CHEM_NCDCO_VIB;
    int itemp_co, iN_co;
    double dtemp_co, dN_co;

    // ---- CO rotational cooling.  No CO cooling below 5 K (no good data). ----
    if (temp < T.co_temptab[0]) {
        out->L0 = 0.0; out->lte = 0.0; out->n05 = 1.0; out->alpha = 1.0;
        out->vib_lte = 0.0;
        return;
    } else if (temp == T.co_temptab[0]) {
        itemp_co = 1; dtemp_co = 0.0;
    } else if (temp >= T.co_temptab[nTco - 1]) {
        itemp_co = nTco; dtemp_co = 0.0;
    } else {
        itemp_co = f_int(temp) - 4;          // table starts at 5 K
        dtemp_co = temp - f_int(temp);
    }

    if (N_co_eff <= T.co_colntab[0]) {
        iN_co = 1; dN_co = 0.0;
    } else if (N_co_eff >= T.co_colntab[ncdco - 1]) {
        iN_co = ncdco; dN_co = 0.0;
    } else {
        iN_co = f_int((10 * N_co_eff) - 144);
        dN_co = (N_co_eff - T.co_colntab[iN_co - 1]) / 0.1;
    }

    double L0 = T.co_L0[itemp_co - 1] + dtemp_co * T.dTco_L0[itemp_co - 1];

    double lte1 = chem_f2(T.co_lte, ncdco, iN_co, itemp_co)
                + dtemp_co * chem_f2(T.dTco_lte, ncdco, iN_co, itemp_co);
    double alp1 = chem_f2(T.co_alp, ncdco, iN_co, itemp_co)
                + dtemp_co * chem_f2(T.dTco_alp, ncdco, iN_co, itemp_co);
    double n051 = chem_f2(T.co_n05, ncdco, iN_co, itemp_co)
                + dtemp_co * chem_f2(T.dTco_n05, ncdco, iN_co, itemp_co);

    double lte, alpha, n05;
    if (iN_co == ncdco) {
        lte = lte1; alpha = alp1; n05 = n051;
    } else {
        double lte2 = chem_f2(T.co_lte, ncdco, iN_co + 1, itemp_co)
                    + dtemp_co * chem_f2(T.dTco_lte, ncdco, iN_co + 1, itemp_co);
        double alp2 = chem_f2(T.co_alp, ncdco, iN_co + 1, itemp_co)
                    + dtemp_co * chem_f2(T.dTco_alp, ncdco, iN_co + 1, itemp_co);
        double n052 = chem_f2(T.co_n05, ncdco, iN_co + 1, itemp_co)
                    + dtemp_co * chem_f2(T.dTco_n05, ncdco, iN_co + 1, itemp_co);
        lte   = lte1 + (lte2 - lte1) * dN_co;
        alpha = alp1 + (alp2 - alp1) * dN_co;
        n05   = n051 + (n052 - n051) * dN_co;
    }

    out->L0    = std::pow(10.0, -L0);
    out->lte   = std::pow(10.0, -lte);
    out->n05   = std::pow(10.0,  n05);
    out->alpha = alpha;

    // ---- CO vibrational cooling ----
    if (temp <= T.co_vib_temptab[0]) {
        itemp_co = 1; dtemp_co = 0.0;
    } else if (temp >= T.co_vib_temptab[nTco_vib - 1]) {
        itemp_co = nTco_vib; dtemp_co = 0.0;
    } else {
        itemp_co = f_int(temp) - 99;         // table starts at 100 K
        dtemp_co = temp - f_int(temp);
    }

    if (N_co_eff_vib <= T.co_vib_colntab[0]) {
        iN_co = 1; dN_co = 0.0;
    } else if (N_co_eff_vib >= T.co_vib_colntab[ncdco_vib - 1]) {
        iN_co = ncdco_vib; dN_co = 0.0;
    } else {
        // NB: the Fortran indexes with N_co_eff (the ROTATIONAL column), not
        // N_co_eff_vib -- cool_util.F:131.  Reproduced literally; see
        // DEVIATIONS.md D-13.
        iN_co = f_int((10 * N_co_eff) - 129);
        dN_co = (N_co_eff_vib - T.co_vib_colntab[iN_co - 1]) / 0.1;
    }

    double v1 = chem_f2(T.co_vib_LTE_final, ncdco_vib, iN_co, itemp_co)
              + dtemp_co * chem_f2(T.dTco_vib_LTE, ncdco_vib, iN_co, itemp_co);
    double vout;
    if (iN_co == ncdco_vib) {
        vout = v1;
    } else {
        double v2 = chem_f2(T.co_vib_LTE_final, ncdco_vib, iN_co + 1, itemp_co)
                  + dtemp_co * chem_f2(T.dTco_vib_LTE, ncdco_vib, iN_co + 1, itemp_co);
        vout = v1 + (v2 - v1) * dN_co;
    }
    out->vib_lte = std::pow(10.0, -vout) * std::exp(-3.08e3 / temp);
}

// ===========================================================================
// h2o_rot_cool (cool_util.F:168-288).  iop_flag: 0 == para, 1 == ortho
// ===========================================================================
void cxx_h2o_rot_cool(const ChemTables& T, double temp, double N_h2o_eff,
                      int iop_flag, H2oCool* out)
{
    const int nTh2o = CHEM_NTH2O, ncdh2o = CHEM_NCDH2O;
    int itemp_h2o, iN_h2o;
    double dtemp_h2o, dN_h2o;

    if (temp <= T.h2o_temptab[0]) {
        itemp_h2o = 1; dtemp_h2o = 0.0;
    } else if (temp >= T.h2o_temptab[nTh2o - 1]) {
        itemp_h2o = nTh2o; dtemp_h2o = 0.0;
    } else {
        itemp_h2o = f_int(temp) - 9;         // table starts at 10 K
        dtemp_h2o = temp - f_int(temp);
    }

    if (N_h2o_eff <= T.h2o_colntab[0]) {
        iN_h2o = 1; dN_h2o = 0.0;
    } else if (N_h2o_eff >= T.h2o_colntab[ncdh2o - 1]) {
        iN_h2o = ncdh2o; dN_h2o = 0.0;
    } else {
        iN_h2o = f_int((10 * N_h2o_eff) - 144);
        dN_h2o = (N_h2o_eff - T.h2o_colntab[iN_h2o - 1]) / 0.1;
    }

    const double *L0t, *dL0t, *LTEt, *dLTEt, *ALPt, *dALPt, *N05t, *dN05t;
    if (iop_flag == 0) {
        L0t = T.h2o_L0_para;    dL0t  = T.dTh2o_L0_para;
        LTEt = T.h2o_LTE_para;  dLTEt = T.dTh2o_LTE_para;
        ALPt = T.h2o_alp_para;  dALPt = T.dTh2o_alp_para;
        N05t = T.h2o_n05_para;  dN05t = T.dTh2o_n05_para;
    } else {
        L0t = T.h2o_L0_ortho;   dL0t  = T.dTh2o_L0_ortho;
        LTEt = T.h2o_LTE_ortho; dLTEt = T.dTh2o_LTE_ortho;
        ALPt = T.h2o_alp_ortho; dALPt = T.dTh2o_alp_ortho;
        N05t = T.h2o_n05_ortho; dN05t = T.dTh2o_n05_ortho;
    }

    double L0 = L0t[itemp_h2o - 1] + dtemp_h2o * dL0t[itemp_h2o - 1];
    double lte1 = chem_f2(LTEt, ncdh2o, iN_h2o, itemp_h2o)
                + dtemp_h2o * chem_f2(dLTEt, ncdh2o, iN_h2o, itemp_h2o);
    double alp1 = chem_f2(ALPt, ncdh2o, iN_h2o, itemp_h2o)
                + dtemp_h2o * chem_f2(dALPt, ncdh2o, iN_h2o, itemp_h2o);
    double n051 = chem_f2(N05t, ncdh2o, iN_h2o, itemp_h2o)
                + dtemp_h2o * chem_f2(dN05t, ncdh2o, iN_h2o, itemp_h2o);

    double lte, alpha, n05;
    if (iN_h2o == ncdh2o) {
        lte = lte1; alpha = alp1; n05 = n051;
    } else {
        double lte2 = chem_f2(LTEt, ncdh2o, iN_h2o + 1, itemp_h2o)
                    + dtemp_h2o * chem_f2(dLTEt, ncdh2o, iN_h2o + 1, itemp_h2o);
        double alp2 = chem_f2(ALPt, ncdh2o, iN_h2o + 1, itemp_h2o)
                    + dtemp_h2o * chem_f2(dALPt, ncdh2o, iN_h2o + 1, itemp_h2o);
        double n052 = chem_f2(N05t, ncdh2o, iN_h2o + 1, itemp_h2o)
                    + dtemp_h2o * chem_f2(dN05t, ncdh2o, iN_h2o + 1, itemp_h2o);
        lte   = lte1 + (lte2 - lte1) * dN_h2o;
        alpha = alp1 + (alp2 - alp1) * dN_h2o;
        n05   = n051 + (n052 - n051) * dN_h2o;
    }

    out->L0    = std::pow(10.0, -L0);
    out->lte   = std::pow(10.0, -lte);
    out->n05   = std::pow(10.0,  n05);
    out->alpha = alpha;
}

// ===========================================================================
// h2o_vib_cool (cool_util.F:304-369)
// ===========================================================================
double cxx_h2o_vib_cool(const ChemTables& T, double temp, double N_h2o_eff)
{
    const int nTh2o_vib = CHEM_NTH2O_VIB, ncdh2o_vib = CHEM_NCDH2O_VIB;
    int itemp_h2o, iN_h2o;
    double dtemp_h2o, dN_h2o;

    if (temp <= T.h2o_vib_temptab[0]) {
        itemp_h2o = 1; dtemp_h2o = 0.0;
    } else if (temp >= T.h2o_vib_temptab[nTh2o_vib - 1]) {
        itemp_h2o = nTh2o_vib; dtemp_h2o = 0.0;
    } else {
        itemp_h2o = f_int(temp) - 99;
        dtemp_h2o = temp - f_int(temp);
    }

    if (N_h2o_eff <= T.h2o_vib_colntab[0]) {
        iN_h2o = 1; dN_h2o = 0.0;
    } else if (N_h2o_eff >= T.h2o_vib_colntab[ncdh2o_vib - 1]) {
        iN_h2o = ncdh2o_vib; dN_h2o = 0.0;
    } else {
        iN_h2o = f_int((10 * N_h2o_eff) - 129);
        dN_h2o = (N_h2o_eff - T.h2o_vib_colntab[iN_h2o - 1]) / 0.1;
    }

    double v1 = chem_f2(T.h2o_vib_LTE_final, ncdh2o_vib, iN_h2o, itemp_h2o)
              + dtemp_h2o * chem_f2(T.dTh2o_vib_LTE, ncdh2o_vib, iN_h2o, itemp_h2o);
    double vout;
    if (iN_h2o == ncdh2o_vib) {
        vout = v1;
    } else {
        double v2 = chem_f2(T.h2o_vib_LTE_final, ncdh2o_vib, iN_h2o + 1, itemp_h2o)
                  + dtemp_h2o * chem_f2(T.dTh2o_vib_LTE, ncdh2o_vib, iN_h2o + 1, itemp_h2o);
        vout = v1 + (v2 - v1) * dN_h2o;
    }
    return std::pow(10.0, -vout) * std::exp(-2.325e3 / temp);
}

// ===========================================================================
// three_level_pops (cool_util.F:385-422)
// ===========================================================================
void cxx_three_level_pops(double r01, double r02, double r12,
                          double r10, double r20, double r21,
                          double* n0, double* n1, double* n2)
{
    if (r01 == 0.0 && r02 == 0.0) { *n0 = 1.0; *n1 = 0.0; *n2 = 0.0; return; }

    double a1 = r01 + r02, a2 = -r10, a3 = -r20;
    double b1 = r01, b2 = -(r10 + r12), b3 = r21;

    *n2 = -a1 * (a1 * b2 - b1 * a2) / ((a1 - a2) * (a1 * b3 - b1 * a3)
                                     - (a1 - a3) * (a1 * b2 - b1 * a2));
    *n1 = (a1 / (a1 - a2)) - ((a1 - a3) / (a1 - a2)) * (*n2);
    *n0 = 1.0 - *n1 - *n2;
}

// ===========================================================================
// compute_stim (cool_util.F:439-456)
// ===========================================================================
double cxx_compute_stim(double a10, double e10, double rad_temp)
{
    double x = e10 / (CHEM_KBOLTZ * rad_temp);
    return (x < 5.0) ? a10 / (std::exp(x) - 1.0) : 0.0;
}

// ===========================================================================
// compute_h2_opacity (cool_util.F:506-560)
// ===========================================================================
double cxx_compute_h2_opacity(const ChemTables& T, double temp, double N_H2_eff)
{
    const int nh2op = CHEM_NH2OP;
    double column_min = std::pow(10.0, T.h2_opac_column[0]);
    double column_max = std::pow(10.0, T.h2_opac_column[nh2op - 1]);

    if (N_H2_eff <= column_min) return 1.0;
    if (N_H2_eff >= column_max) return 0.0;

    double logN = std::log10(N_H2_eff);
    int jj = 1 + f_int(10 * (logN - 17.0));
    double diff = T.h2_opac_column[jj] - T.h2_opac_column[jj - 1];
    double dN = (logN - T.h2_opac_column[jj - 1]) / diff;

    double logT = std::log10(temp);
    int ii; double dT;
    if (logT <= T.h2_opac_temp[0]) {
        ii = 1; dT = 0.0;
    } else if (logT >= T.h2_opac_temp[nh2op - 1]) {
        ii = nh2op; dT = 0.0;
    } else {
        ii = 1 + f_int((logT - 1.5) / (double)0.03f);   // default-REAL literal
        diff = T.h2_opac_temp[ii] - T.h2_opac_temp[ii - 1];
        dT = (logT - T.h2_opac_temp[ii - 1]) / diff;
    }

    double o1, o2;
    if (dT > 0.0) {
        o1 = chem_f2(T.h2_opac, nh2op, ii, jj)
           + dT * (chem_f2(T.h2_opac, nh2op, ii + 1, jj)
                 - chem_f2(T.h2_opac, nh2op, ii, jj));
        o2 = chem_f2(T.h2_opac, nh2op, ii, jj + 1)
           + dT * (chem_f2(T.h2_opac, nh2op, ii + 1, jj + 1)
                 - chem_f2(T.h2_opac, nh2op, ii, jj + 1));
    } else {
        o1 = chem_f2(T.h2_opac, nh2op, ii, jj);
        o2 = chem_f2(T.h2_opac, nh2op, ii, jj + 1);
    }
    return std::pow(10.0, o1 + dN * (o2 - o1));
}

// ===========================================================================
// compute_h2_heating_fraction (cool_util.F:574-680)
// ===========================================================================
double cxx_compute_h2_heating_fraction(double T, double n, double xh2)
{
    auto h2_rv_heat = [](int k) { return dat_h2_rv_heat[k - 1]; };   // Fortran 1-based; NOLINT(clang-analyzer-core.uninitialized.UndefReturn): analyzer cannot track dT == 0.0
    double logT = std::log10(T);
    double logn = std::log10(n);
    int idx1, idx2, idx3, offset;
    double dT, dn, nscale, dh2, f1, f2, f3, f4, fr1, fr2, frac1, frac2;

    if (logT < 1.0)      { idx1 = 1;  dT = 0.0; }
    else if (logT > 3.5) { idx1 = 11; dT = 0.0; }
    else {
        idx1 = 1 + f_int(4.0 * (logT - 1.0));
        dT = 4.0 * (logT - 1.0) - (idx1 - 1);
    }

    if (logn < 1.0)      { idx2 = 1;  dn = 0.0; nscale = (n / 1e1); }
    else if (logn > 8.0) { idx2 = 29; dn = 0.0; nscale = 1.0; }
    else {
        idx2 = 1 + f_int(4.0 * (logn - 1.0));
        dn = 4.0 * (logn - 1.0) - (idx2 - 1);
        nscale = 1.0;
    }

    xh2 = (xh2 < 0.0) ? 0.0 : (xh2 > 0.5) ? 0.5 : xh2;   // table covers xh2 in [0, 0.5]
    if (xh2 == 0.5) { idx3 = 6; dh2 = 0.0; }
    else {
        idx3 = 1 + f_int(xh2 / 1e-1);
        dh2  = xh2 / 1e-1 - (idx3 - 1);
        if (idx3 >= 6) { idx3 = 6; dh2 = 0.0; }               // last row: no idx3+1 read
    }

    offset = 319 * (idx3 - 1) + 29 * (idx1 - 1) + idx2;
    if (dT == 0.0) {
        if (dn == 0.0) {
            frac1 = nscale * h2_rv_heat(offset);
        } else {
            f1 = h2_rv_heat(offset);
            f2 = h2_rv_heat(offset + 1);
            frac1 = f1 + dn * (f2 - f1);
        }
    } else {
        if (dn == 0.0) {
            f1 = nscale * h2_rv_heat(offset);
            f2 = nscale * h2_rv_heat(offset + 29);
            frac1 = f1 + dT * (f2 - f1);
        } else {
            f1 = h2_rv_heat(offset);
            f2 = h2_rv_heat(offset + 1);
            f3 = h2_rv_heat(offset + 29);
            f4 = h2_rv_heat(offset + 30);
            fr1 = f1 + dn * (f2 - f1);
            fr2 = f3 + dn * (f4 - f3);
            frac1 = fr1 + dT * (fr2 - fr1);
        }
    }

    if (dh2 == 0.0) return frac1;

    offset = 319 * idx3 + 29 * (idx1 - 1) + idx2;
    if (dT == 0.0) {
        if (dn == 0.0) {
            frac2 = nscale * h2_rv_heat(offset);
        } else {
            f1 = h2_rv_heat(offset);
            f2 = h2_rv_heat(offset + 1);
            frac2 = f1 + dn * (f2 - f1);
        }
    } else {
        if (dn == 0.0) {
            f1 = nscale * h2_rv_heat(offset);
            f2 = nscale * h2_rv_heat(offset + 29);
            frac2 = f1 + dT * (f2 - f1);
        } else {
            f1 = h2_rv_heat(offset);
            f2 = h2_rv_heat(offset + 1);
            f3 = h2_rv_heat(offset + 29);
            f4 = h2_rv_heat(offset + 30);
            fr1 = f1 + dn * (f2 - f1);
            fr2 = f3 + dn * (f4 - f3);
            frac2 = fr1 + dT * (fr2 - fr1);
        }
    }
    return frac1 + dh2 * (frac2 - frac1);
}

// ===========================================================================
// calc_dust_temp (cool_util.F:802-1007)
//
// tdust is an in/out algebraic state (audit sec 3.1 item 2) and lives in
// c.tdust; the driver snapshots and restores it on substep retries.
// ===========================================================================
void cxx_calc_dust_temp(double yn, double chi_isrf, double temp, double abHI,
                        double* RH2, double* Rdust, int nocalc, ChemCell& c)
{
    const ChemTables& T = *c.tab;
    const int ndust_iter = 20000;
    double isrf_heat = 0.0, dust_cool = 0.0, gas_dust = 0.0, dust_h2_heat = 0.0;

    if (c.dust_to_gas_ratio > 1e-8 && temp < 1e5) {
        int itemp; double dtemp;
        if (temp <= T.tmin)      { itemp = 1;        dtemp = 0.0; }
        else if (temp >= T.tmax) { itemp = CHEM_NMD; dtemp = 0.0; }
        else {
            itemp = (int)(std::log10(temp) / T.dtlog) + 1;
            dtemp = temp - T.temptab[itemp - 1];
        }

        double ch7  = (T.chtab[itemp - 1][6]  + dtemp * T.dtchtab[itemp - 1][6]) * c.dust_to_gas_ratio; // table is per unit DGR
        double cl14 = T.cltab[itemp - 1][13] + dtemp * T.dtcltab[itemp - 1][13];

        double temp2, tdust2, fa, stick;
        if (nocalc == 1) {
            // Only recompute RH2/Rdust, do not update tdust
            temp2  = temp * 1e-2;
            tdust2 = c.tdust * 1e-2;
            fa     = 1.0 / (1.0 + 1e4 * std::exp(-6e2 / c.tdust));
            stick  = 1.0 / (1.0 + 0.4 * std::pow(temp2 + tdust2, 0.5)
                                + 0.2 * temp2 + 0.08 * std::pow(temp2, 2.0));
#ifdef CHEMCOOL_H2FORM_K23
            fa     = 1.0;   // Kim+2023 Eq. 16: no T_d dependence
            stick  = 2.0 / (1.0 + 0.4 * std::sqrt(temp2) + 0.2 * temp2 + 0.08 * temp2 * temp2);
#endif
            *Rdust = ch7 * fa * stick;
            *RH2   = *Rdust * abHI * (yn * yn);
            return;
        }

        // ISRF heating (FUV 6-13.6 eV only)
        // cool_util.F:857 spells `1.7` without a D exponent -> DEFAULT REAL.
        // CMB share of the ISRF coefficient removed at z = 0, CMB added at T_CMB(z)
        const double CMB_heat_z0 = 4.68e-31 * std::pow(2.726, 6.0);
        double field_strength = c.G0_dust / (double)1.7f;
        if (c.isrf_option == 1) isrf_heat = 5.8e-24  * chi_isrf - CMB_heat_z0;
        else                    isrf_heat = 5.73e-24 * chi_isrf - CMB_heat_z0;
        if (isrf_heat < 0.0) isrf_heat = 0.0;
        isrf_heat = isrf_heat * yn * field_strength;
        double CMB_temp = 2.726 * (1.0 + c.redshift);
        double CMB_heat = 4.68e-31 * chem_fpowi(CMB_temp, 6) * yn;
        isrf_heat = (isrf_heat + CMB_heat) * c.dust_to_gas_ratio;
#ifdef DIFFUSE_DUST_EMISSION
        isrf_heat += c.diffuse_dust_heat;
#endif

        double tdust_new = c.tdust;
        int converged = 0;
        for (int K = 1; K <= ndust_iter; K++) {
            c.tdust = tdust_new;

            dust_cool = 4.68e-31 * chem_fpowi(c.tdust, 6) * yn;
            dust_cool *= c.dust_to_gas_ratio;
#ifdef NO_GAS_DUST_HEAT
            gas_dust = 0.0;
#else
            gas_dust = cl14 * (temp - c.tdust) * (yn * yn);
            gas_dust *= c.dust_to_gas_ratio;
#endif
            double h2_form_dust = 1.0 - c.h2_form_ex - c.h2_form_kin;

            temp2  = temp * 1e-2;
            tdust2 = c.tdust * 1e-2;
            fa     = 1.0 / (1.0 + 1e4 * std::exp(-6e2 / c.tdust));
            stick  = 1.0 / (1.0 + 0.4 * std::pow(temp2 + tdust2, 0.5)
                                + 0.2 * temp2 + 0.08 * std::pow(temp2, 2.0));
#ifdef CHEMCOOL_H2FORM_K23
            fa     = 1.0;   // Kim+2023 Eq. 16: no T_d dependence
            stick  = 2.0 / (1.0 + 0.4 * std::sqrt(temp2) + 0.2 * temp2 + 0.08 * temp2 * temp2);
            *Rdust = ch7 * fa * stick;
            double Rdust_rat = 0.0;
#else
            double dfa = -fa * (1 - fa) * (6e2 / (c.tdust * c.tdust));
            *Rdust = ch7 * fa * stick;
            double Rdust_rat = (dfa / fa
                              - 2e-2 * stick / std::sqrt(c.tdust + temp));
#endif
            *RH2 = *Rdust * abHI * (yn * yn);

#ifdef NO_DUST_H2_HEAT
            dust_h2_heat = 0.0;
#else
            dust_h2_heat = (*RH2) * h2_form_dust * 4.48 * CHEM_EV;
#endif
            double F_nr, dF_nr;
            if (*Rdust == 0.0) {          // can happen if deff = 0
                F_nr = isrf_heat - dust_cool + gas_dust;
#ifdef NO_GAS_DUST_HEAT
                dF_nr = -(6.0 * dust_cool / c.tdust);
#else
                dF_nr = -(6.0 * dust_cool / c.tdust) - cl14 * (yn * yn);
#endif
            } else {
                F_nr = isrf_heat - dust_cool + gas_dust + dust_h2_heat;
#ifdef NO_GAS_DUST_HEAT
                dF_nr = -(6.0 * dust_cool / c.tdust) + dust_h2_heat * Rdust_rat;
#else
                dF_nr = -(6.0 * dust_cool / c.tdust) - cl14 * (yn * yn)
                      + dust_h2_heat * Rdust_rat;
#endif
            }

            tdust_new = c.tdust - F_nr / dF_nr;
            // Bounded Newton iterate (2026-07-02 fix): grains sublimate above
            // ~1500 K, and without the cap dust_cool ~ tdust^6 lets the solver
            // run to tdust >> 1e4 K and overstiffen the network.  NaN guard too.
            if (tdust_new != tdust_new) tdust_new = T.tmin;
            if (tdust_new < T.tmin)     tdust_new = T.tmin;
            if (tdust_new > 1.5e3)      tdust_new = 1.5e3;
            if (std::fabs((tdust_new - c.tdust) / c.tdust) < 1e-2) {
                converged = 1;
                break;
            }
        }
        if (!converged) {
            std::printf("Dust temp iteration failed! %g %g %g\n",
                        c.tdust, tdust_new, c.tdust - tdust_new);
            std::printf("Particle : %ld\n", c.id_current);
            std::printf("density, temp, %g %g\n", yn, temp);
            std::printf("%g %g %g %g\n", isrf_heat, dust_cool, gas_dust,
                        dust_h2_heat);
            std::printf("dust_to_gas_ratio = %g\n", c.dust_to_gas_ratio);
            std::printf("RH2 = %g\nch7 = %g\n", *RH2, ch7);
            // ABORT(10) in the Fortran.  The port keeps the failure LOUD but
            // routes it through the driver's recoverable path (DEVIATIONS D-3).
            c.dust_iter_failed = 1;
        }
    } else {
        *RH2 = 0.0; *Rdust = 0.0;
        // Hot / dust-free branch: grains never reach the gas temperature.
        c.tdust = (temp < 1.5e3) ? temp : 1.5e3;
    }

    if (c.tdust == 0.0) {
        std::printf("Zero dust temp!\n%g %g %g %g\n",
                    isrf_heat, dust_cool, gas_dust, dust_h2_heat);
    }
}
