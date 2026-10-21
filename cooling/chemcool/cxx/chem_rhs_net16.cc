// chem_rhs_net16.cc -- verbatim-in-structure port of rate_eq_gong17.F
// (CHEMISTRYNETWORK=16, Gong, Ostriker & Wolfire 2017; M is silicon).
//
// ODE vector (0-based): y[0]=H2, y[1]=H+, y[2]=C+, y[3]=CHx, y[4]=OHx, y[5]=CO,
// y[6]=HCO+, y[7]=He+, y[8]=Si+, y[9]=energy density.
// Deviations (DEVIATIONS.md D-25):
//  - validate_input on a local copy (R3); a flagged input returns ydot = 0 and
//    CHEM_RHS_RECOVERABLE, as the Fortran returns zero ydot with ipar(1) = 1.
//  - temp <= 0: ydot is still built at tmin (as the Fortran) and the call
//    returns CHEM_RHS_RECOVERABLE.
//  - NaN ydot: recoverable return (the Fortran has no validate_output here).
//  - default-REAL literals (PI, 1.4) are float32, as compiled.
//  - THERMAL_INFO_DUMP diagnostics not ported.

#include "chem_types.h"
#include "chem_rhs_common.h"
#include <algorithm>
#include <cstdio>

int chem_rhs_net16(double /*t*/, const double* y_in, double* ydot, ChemCell& c)
{
    const ChemTables& T = *c.tab;
    const int nspec = 10;
    const int ih2 = C15_IH2, ihp = C15_IHP, ic = C15_IC, ichx = C15_ICHX,
              iohx = C15_IOHX, ico = C15_ICO, ihcop = C15_IHCOP,
              ihep = C15_IHEP, imp = C15_IMP, itmp = 9;

    const double yn = c.yn, dl = c.dl, divv = c.divv;
    const double NH = 0.5 * yn * dl;                         // .F:94
    const double ynsq = yn * yn;

    // --- validate_input on a local copy (.F:99-106) ---
    double y[10];
    for (int i = 0; i < nspec; i++) { y[i] = y_in[i]; ydot[i] = 0.0; }
    if (chem_validate_input(y, nspec, itmp, ih2, c.id_current))
        return CHEM_RHS_RECOVERABLE;

    // --- tracked species (.F:110-118) ---
    double abh2   = (y[ih2]   > 0.0) ? y[ih2]   : 0.0;
    double abhp   = (y[ihp]   > 0.0) ? y[ihp]   : 0.0;
    double abcp   = (y[ic]    > 0.0) ? y[ic]    : 0.0;
    double abchx  = (y[ichx]  > 0.0) ? y[ichx]  : 0.0;
    double abohx  = (y[iohx]  > 0.0) ? y[iohx]  : 0.0;
    double abco   = (y[ico]   > 0.0) ? y[ico]   : 0.0;
    double abhcop = (y[ihcop] > 0.0) ? y[ihcop] : 0.0;
    double abhep  = (y[ihep]  > 0.0) ? y[ihep]  : 0.0;
    double abmp   = (y[imp]   > 0.0) ? y[imp]   : 0.0;

    // --- conservation-law species (.F:122-130) ---
    double abe   = abhp + abcp + abhcop + abhep + abmp;  if (abe < 0.0) abe = 0.0;
    double abHI  = 1.0 - 2.0 * abh2 - abhp - abchx - abohx - abhcop;  if (abHI < 0.0) abHI = 0.0;
    double abheI = c.abhe - abhep;  if (abheI < 0.0) abheI = 0.0;
    double abcI  = c.abundc - abcp - abchx - abco - abhcop;  if (abcI < 0.0) abcI = 0.0;
    double abmI  = c.abundM - abmp;  if (abmI < 0.0) abmI = 0.0;
    double abotot = c.abundo - abohx - abco - abhcop;  if (abotot < 0.0) abotot = 0.0;

    // --- energy, temperature (.F:134-143) ---
    double energy = y[itmp];
    double yntot = (1.0 + c.abhe - abh2 + abe) * yn;
    double ekn = energy / (yntot * CHEM_KBOLTZ);
    double temp, gamma;
    cxx_lookup_temp_gamma(abh2, c.abhe, ekn, &temp, &gamma, T);

    // --- temperature bounds / table index (.F:149-167) ---
    ChemTIdx ti = chem_temp_index(temp, T);
    int flagged = ti.flag_bad;                               // ipar(1) = 1, temp = tmin
    temp = ti.temp;

    // --- rate coefficients (.F:171-255) ---
    const double ch1   = chem_CH(T, 1,   ti);
    const double ch2   = chem_CH(T, 2,   ti);
    const double ch3   = chem_CH(T, 3,   ti);
    const double ch4   = chem_CH(T, 4,   ti);
    const double ch5   = chem_CH(T, 5,   ti);
    const double ch6   = chem_CH(T, 6,   ti);
    const double ch9   = chem_CH(T, 9,   ti);
    const double ch11  = chem_CH(T, 11,  ti);
    const double ch13  = chem_CH(T, 13,  ti);
    const double ch14  = chem_CH(T, 14,  ti);
    const double ch17  = chem_CH(T, 17,  ti);
    const double ch25  = chem_CH(T, 25,  ti);
    const double ch34  = chem_CH(T, 34,  ti);
    const double ch35  = chem_CH(T, 35,  ti);
    const double ch36  = chem_CH(T, 36,  ti);
    const double ch37  = chem_CH(T, 37,  ti);
    const double ch38  = chem_CH(T, 38,  ti);
    const double ch39  = chem_CH(T, 39,  ti);
    const double ch40  = chem_CH(T, 40,  ti);
    const double ch41  = chem_CH(T, 41,  ti);
    const double ch43  = chem_CH(T, 43,  ti);
    const double ch44  = chem_CH(T, 44,  ti);
    const double ch46  = chem_CH(T, 46,  ti);
    const double ch47  = chem_CH(T, 47,  ti);
    const double ch53  = chem_CH(T, 53,  ti);
    const double ch87  = chem_CH(T, 87,  ti);
    const double ch89  = chem_CH(T, 89,  ti);
    const double ch92  = chem_CH(T, 92,  ti);
    const double ch105 = chem_CH(T, 105, ti);
    const double ch106 = chem_CH(T, 106, ti);
    const double ch135 = chem_CH(T, 135, ti);
    const double ch136 = chem_CH(T, 136, ti);
    const double ch147 = chem_CH(T, 147, ti);
    const double ch174 = chem_CH(T, 174, ti);
    const double ch175 = chem_CH(T, 175, ti);
    const double ch176 = chem_CH(T, 176, ti);
    const double ch177 = chem_CH(T, 177, ti);
    const double ch178 = chem_CH(T, 178, ti);
    const double ch179 = chem_CH(T, 179, ti);
    const double ch180 = chem_CH(T, 180, ti);
    const double ch181 = chem_CH(T, 181, ti);

    const double cst1  = T.cst[0],  cst54 = T.cst[53], cst59 = T.cst[58],
                 cst75 = T.cst[74], cst85 = T.cst[84], cst92 = T.cst[91],
                 cst93 = T.cst[92], cst94 = T.cst[93];

    // --- branching ratios (.F:260-284) ---
    const double ch106a = (2.34 / 6.7) * ch106;
    const double ch87a  = 0.92 * ch87;
    const double ch87b  = 0.08 * ch87;
    const double h2op_branch = cst92 * abh2 / (cst92 * abh2 + ch175 * abe + 1e-30);
    const double ch174a = h2op_branch * ch174;
    const double cst93a = h2op_branch * cst93;
    const double ch176a = (2.31 / 3.3) * ch176;

    // --- H2 collisional dissociation, density-dependent (.F:289-300) ---
    double ykdh, ykdh2;
    if (temp < 5e2) {
        ykdh  = 0.0;
        ykdh2 = 0.0;
    } else {
        double ncrinv = 2.0 * abh2 * (ch6 - ch5) + ch5;
        double h2var0 = 1.0 / (1.0 + yn * ncrinv);
        double h2var1 = std::pow(ch3, h2var0);
        double h2var2 = std::pow(ch4, h2var0);
        ykdh  = ch1 * h2var1;
        ykdh2 = ch2 * h2var2;
    }

    // --- photorates (.F:304-310); shielding called for every iphoto ---
    double fshield = cxx_calc_H2_shielding_factor(temp, abh2, NH, c);
    double ykdis = c.phrates[0] * fshield;
    double ph2  = c.phrates[1];
    double ph3  = c.phrates[2];
    double ph10 = c.phrates[9];
    double ph48 = c.phrates[47];
    double ph49 = c.phrates[48];

    // --- grain recombination, WD01 x 0.6, phi floored at 100 (.F:316-374) ---
    double phi = (abe == 0.0) ? 1e20 : c.G_dust * std::sqrt(temp) / (yn * abe);
    phi = std::max(phi, 1e2);

    double hgrvar1  = 8.074e-6 * std::pow(phi, 1.378);
    double hgrvar2  = 1.0 + ch34 * std::pow(phi, ch35);
    double h_gr     = 1.225e-13 * c.dust_to_gas_ratio / (1.0 + hgrvar1 * hgrvar2);

    double hegrvar1 = 3.185e-7 * std::pow(phi, 1.512);
    double hegrvar2 = 1.0 + ch135 * std::pow(phi, ch136);
    double heii_gr  = 5.572e-14 * c.dust_to_gas_ratio / (1.0 + hegrvar1 * hegrvar2);

    double c2grvar1 = 6.089e-3 * std::pow(phi, 1.128);
    double c2grvar2 = 1.0 + ch36 * std::pow(phi, ch37);
    double cii_gr   = 4.558e-13 * c.dust_to_gas_ratio / (1.0 + c2grvar1 * c2grvar2);

    double si2grvar1 = 5.678e-8 * std::pow(phi, 1.874);
    double si2grvar2 = 1.0 + ch38 * std::pow(phi, ch39);
    double si2_gr    = 2.166e-14 * c.dust_to_gas_ratio / (1.0 + si2grvar1 * si2grvar2);

    h_gr    = h_gr    * std::exp(-temp / 3e4);
    heii_gr = heii_gr * std::exp(-temp / 3e4);
    cii_gr  = cii_gr  * std::exp(-temp / 3e4);
    si2_gr  = si2_gr  * std::exp(-temp / 3e4);

    h_gr    = 0.6 * h_gr;
    heii_gr = 0.6 * heii_gr;
    cii_gr  = 0.6 * cii_gr;
    si2_gr  = 0.6 * si2_gr;

    // --- dust temperature + H2 formation on grains (.F:380-389) ---
    double RH2, Rdust;
    if (c.dust_to_gas_ratio > 0.0) {
        if (yn > 1e8 / std::pow(c.dust_to_gas_ratio, 1.5)) {
            c.tdust = temp;
            cxx_calc_dust_temp(yn, c.chi_isrf, temp, abHI, &RH2, &Rdust, 1, c);
        } else {
            cxx_calc_dust_temp(yn, c.chi_isrf, temp, abHI, &RH2, &Rdust, 0, c);
        }
    } else {
        cxx_calc_dust_temp(yn, c.chi_isrf, temp, abHI, &RH2, &Rdust, 0, c);
    }

    // --- cosmic rays (.F:394-408); tables are per unit rate ---
    const double cri = c.cosmic_ray_ion_rate;
    double cr1  = T.crtab[0]  * cri * (2.3 * abh2 + 1.5 * abHI);
    double cr2  = T.crtab[1]  * cri * (2.3 * abh2 + 1.5 * abHI);
    double cr10 = T.crtab[9]  * cri;
    double cr11 = T.crtab[10] * cri;
    double cr6  = T.crtab[5]  * cri;
    double cr3  = T.crtab[2]  * cri;
    double cr9  = T.crtab[8]  * cri;
    double cp1  = 2.0 * abh2 * T.crphot[0]  * cri;
    double cp11 = 2.0 * abh2 * T.crphot[10] * cri;
    double cp13 = 2.0 * abh2 * T.crphot[12] * cri;

    // --- O+ equilibrium abundance (.F:420-430) ---
    double num   = ch47 * abotot * abhp + ch179 * abhep * abohx;
    double denom = cst93 * abh2 + ch46 * abHI + ch47 * abhp;
    double abop  = (denom > 0.0) ? num / denom : 0.0;
    if (abop > abotot) abop = abotot;   // O+ cannot exceed the free O pool
    double aboI  = abotot - abop;
    abe = abe + abop;

    // --- H3+ equilibrium abundance (.F:437-466) ---
    double fh3p;
    if (abh2 < 1e-10) {
        fh3p = 0.0;
    } else if (ch105 < 1e-40) {
        fh3p = 0.0;
    } else {
        fh3p = ch105 * abh2 / (ch105 * abh2 + cst1 * abHI);
    }

    num   = fh3p * ((cr2 / yn) * abh2 + cst75 * abhep * abh2);
    denom = ch180 * abcI + ch174 * aboI + cst59 * abco
          + ch106 * abe;
    double abh3p;
    if (num == 0.0) {
        abh3p = 0.0;
    } else {
        abh3p = (-denom + std::sqrt(denom * denom + 4.0 * ch106 * num))
              / (2.0 * ch106);
    }
    abHI = (abHI > 3.0 * abh3p) ? abHI - 3.0 * abh3p : 0.0;   // H3+ holds three H atoms of the HI pool; not capped
    abe = abe + abh3p;

    // --- protons (.F:470-479) ---
    ydot[ihp] = (ch11 * abe   * abHI
              -  ch13 * abhp  * abe
              -  h_gr * abhp  * 1.0
              + (1.0 - fh3p)  * cst75 * abhep * abh2
              +  ch46 * abop  * abHI
              -  ch47 * aboI  * abhp
              +  ch9  * abhep * abh2) * yn
              +  cr1  * abHI
              +  cr10 * abh2
              + (1.0 - fh3p) * cr2 * abh2;

    // --- molecular hydrogen (.F:483-501) ---
    ydot[ih2] = (-ch25  * abe  * abh2
              -   ykdh  * abHI * abh2
              -   ykdh2 * abh2 * abh2
              +   Rdust * abHI * 1.0
              -   ch176 * abh2 * abcp
              -   cst54 * abcI * abh2
              +   ch180 * abh3p * abcI
              +   ch174 * abh3p * aboI
              +   cst59 * abh3p * abco
              -     ch9 * abhep * abh2
              - 2.0 * fh3p * cst75 * abhep * abh2   // via H2+ + H2
              -   cst93 * abop  * abh2
              +  ch106a * abh3p * abe
              +    ch53 * abHI  * abohx
              +   ch178 * abHI  * abchx) * yn
              - 2.0 * fh3p * cr2 * abh2
              -   ykdis * abh2
              -    cr10 * abh2
              -    cr11 * abh2;

    // --- C+ (.F:505-514) ---
    ydot[ic] = ( ch92 * abhep * abco
             -   ch40 * abcp  * abe
             +   ch43 * abcI  * abe
             - cii_gr * abcp  * 1.0
             -  ch176 * abcp  * abh2
             -  ch177 * abcp  * abohx
             +  ch181 * abchx * abhep) * yn
             +    ph2 * abcI
             +    cr3 * abcI
             +    cp1 * abcI;

    // --- CHx (.F:518-525) ---
    ydot[ichx] = (  ch180 * abcI   * abh3p
               +   ch176a * abcp   * abh2
               +    cst54 * abcI   * abh2
               -    cst85 * aboI   * abchx
               -    ch181 * abchx  * abhep
               -    ch178 * abHI   * abchx) * yn
               -     ph48 * abchx;

    // --- OHx (.F:529-537) ---
    ydot[iohx] = ( ch174a * aboI  * abh3p
               +    ch87b * abhcop * abe     // HCO+ + e -> OH + C (Geppert+05)
               +   cst93a * abop  * abh2
               +    ch89  * abco  * abHI
               -    ch147 * abcI  * abohx
               -    ch177 * abcp  * abohx
               -    ch53  * abHI  * abohx
               -    cst94 * aboI  * abohx
               -    ch179 * abhep * abohx) * yn
               -     ph49 * abohx;

    // --- HCO+ (.F:541-544) ---
    ydot[ihcop] = (cst59 * abh3p  * abco
                +  ch177 * abcp   * abohx
                -   ch87 * abhcop * abe) * yn
                +    cr9 * abco;

    // --- CO (.F:548-556) ---
    ydot[ico] = (cst85 * abchx  * aboI
              +  ch147 * abohx  * abcI
              +  ch87a * abhcop * abe
              -  cst59 * abh3p  * abco
              -   ch92 * abhep  * abco
              -   ch89 * abHI   * abco) * yn
              -   ph10 * abco
              -    cr9 * abco
              -   cp11 * abco;

    // --- Si+ (.F:560-565) ---
    ydot[imp] = (0.0
              -   ch41 * abmp * abe
              - si2_gr * abmp * 1.0
              +   ch44 * abmI * abe) * yn
              +    ph3 * abmI
              +   cp13 * abmI;

    // --- He+ (.F:569-577) ---
    ydot[ihep] =  cr6  * abheI
               +  ch14 * abheI * abe * yn
               - ( ch9 * abhep * abh2
               +  ch92 * abhep * abco
               +  ch17 * abhep * abe
               + cst75 * abhep * abh2
               + heii_gr * abhep * 1.0
               + ch179 * abhep * abohx
               + ch181 * abhep * abchx) * yn;

    // --- chemical heating/cooling, rates_chem(1..6) (.F:581-628) ---
    double rc[6];
    rc[0] = 4.48 * CHEM_EV * (ykdh  * abHI * abh2 +
                              ykdh2 * abh2 * abh2 +
                              ch25  * abe  * abh2) * ynsq;
    rc[1] = -(4e-1 * CHEM_EV * abh2 * yn * ykdis);

    double cdex = ((double)1.4f * std::exp(-1.81e4 / (temp + 1200)) * abh2
                 + 1.0 * std::exp(-1e3 / temp) * abHI) * 1e-12 * std::sqrt(temp) * yn;
    double rdex = 2e-7;
    double fpheat = cdex / (cdex + rdex);
    rc[2] = -(2.0 * CHEM_EV * abh2 * yn * ykdis * 8.5 * fpheat);

    double f_ex;
    if (RH2 > 0.0) {
        f_ex = cxx_compute_h2_heating_fraction(temp, yn, abh2);
        f_ex = f_ex * c.h2_form_ex;
    } else {
        f_ex = 0.0;
    }
    double f_tot = c.h2_form_kin + f_ex;
    rc[3] = -(4.48 * CHEM_EV * f_tot * RH2);

    rc[4] = 13.6 * CHEM_EV * ch11 * abe * abHI * ynsq;
    rc[5] = CHEM_KBOLTZ * temp * abhp * (ch13 * abe + h_gr) * ynsq;

    double ylam_chem = rc[0] + rc[1] + rc[2] + rc[3] + rc[4] + rc[5];
    for (int i = 0; i < 6; i++) c.lambda_chem[i] = rc[i];
    for (int i = 6; i < CHEM_NRATES_CHEM_MAX; i++) c.lambda_chem[i] = 0.0;

    // --- cool_func abundance vector (.F:632-647) ---
    double abundances[16];
    for (int i = 0; i < 16; i++) abundances[i] = 0.0;
    abundances[0]  = abh2;
    abundances[1]  = aboI;
    abundances[4]  = abco;
    abundances[5]  = abcI;
    abundances[6]  = abcp;
    abundances[7]  = abmI;
    abundances[8]  = abmp;
    abundances[9]  = abe;
    abundances[10] = abhp;
    abundances[11] = abHI;
    abundances[13] = abheI;
    abundances[14] = abhep;

    // --- local Jeans length (.F:650-653) ---
    double rho_lj  = (1.0 + 4.0 * c.abhe) * CHEM_PROTONMASS * yn;
    double cs2_lj  = gamma * (gamma - 1.0) * energy / rho_lj;
    double L_jeans = std::sqrt(CHEM_PI * cs2_lj / (CHEM_GRAVITY * rho_lj));

    double ylam, rates[CHEM_NRATES];
    int cferr = cxx_cool_func(temp, yn, L_jeans, divv, c.G_dust, abundances,
                              &ylam, rates, 0, c);
    if (cferr) return CHEM_RHS_RECOVERABLE;

    ydot[itmp] = -(ylam + ylam_chem);

    for (int i = 0; i < nspec; i++) {
        if (ydot[i] != ydot[i]) {
            std::fprintf(stderr,
                "chem_rhs_net16: NaN ydot[%d] (ID %ld, T=%g yn=%g)\n",
                i, c.id_current, temp, yn);
            for (int j = 0; j < nspec; j++) ydot[j] = 0.0;
            return CHEM_RHS_RECOVERABLE;
        }
    }
    return flagged ? CHEM_RHS_RECOVERABLE : CHEM_RHS_OK;
}
