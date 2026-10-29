// chem_rhs_net15.cc -- verbatim-in-structure port of rate_eq_nl99.F
// (CHEMISTRYNETWORK=15, Nelson & Langer 1999 CO network; KOYAMA branch not ported).
//
// ODE vector (0-based): y[0]=H2, y[1]=H+, y[2]=C+, y[3]=CHx, y[4]=OHx, y[5]=CO,
// y[6]=HCO+, y[7]=He+, y[8]=M+, y[9]=energy density.
// Deviations (DEVIATIONS.md D-22):
//  - validate_input on a local copy (R3).  Unlike nets 5/17 the Fortran does
//    not return early on a flag: ydot is still built from the clipped copy, so
//    the port builds it too and returns CHEM_RHS_RECOVERABLE at the end (same
//    for the temp<=0 path).  NaN input returns at once with ydot = 0.
//  - NaN ydot: recoverable return (the Fortran has no validate_output here).
//  - default-REAL literals (PI, 1.4) are float32, as compiled.

#include "chem_types.h"
#include "chem_rhs_common.h"
#include <cstdio>

int chem_rhs_net15(double /*t*/, const double* y_in, double* ydot, ChemCell& c)
{
    const ChemTables& T = *c.tab;
    const int nspec = 10;
    const int ih2 = C15_IH2, ihp = C15_IHP, ic = C15_IC, ichx = C15_ICHX,
              iohx = C15_IOHX, ico = C15_ICO, ihcop = C15_IHCOP,
              ihep = C15_IHEP, imp = C15_IMP, itmp = 9;

    const double yn = c.yn, dl = c.dl, divv = c.divv;
    const double NH = 0.5 * yn * dl;                         // .F:77

    // --- validate_input on a local copy (R3) ---
    double y[10];
    for (int i = 0; i < nspec; i++) { y[i] = y_in[i]; ydot[i] = 0.0; }
    int flagged = chem_validate_input_full(y, nspec, itmp, ih2, c.id_current);
    if (flagged < 0) return CHEM_RHS_RECOVERABLE;

    // --- tracked species (.F:85-93) ---
    double abh2   = (y[ih2]   > 0.0) ? y[ih2]   : 0.0;
    double abhp   = (y[ihp]   > 0.0) ? y[ihp]   : 0.0;
    double abcp   = (y[ic]    > 0.0) ? y[ic]    : 0.0;
    double abchx  = (y[ichx]  > 0.0) ? y[ichx]  : 0.0;
    double abohx  = (y[iohx]  > 0.0) ? y[iohx]  : 0.0;
    double abco   = (y[ico]   > 0.0) ? y[ico]   : 0.0;
    double abhcop = (y[ihcop] > 0.0) ? y[ihcop] : 0.0;
    double abhep  = (y[ihep]  > 0.0) ? y[ihep]  : 0.0;
    double abmp   = (y[imp]   > 0.0) ? y[imp]   : 0.0;

    // --- conservation-law species (.F:97-102) ---
    double abe   = abhp + abcp + abhcop + abhep + abmp;  if (abe < 0.0) abe = 0.0;
    double abHI  = 1.0 - 2.0 * abh2 - abhp - abchx - abohx - abhcop;  if (abHI < 0.0) abHI = 0.0;
    double abheI = c.abhe - abhep;  if (abheI < 0.0) abheI = 0.0;
    double abcI  = c.abundc - abcp - abchx - abco - abhcop;  if (abcI < 0.0) abcI = 0.0;
    double aboI  = c.abundo - abohx - abco - abhcop;  if (aboI < 0.0) aboI = 0.0;
    double abmI  = c.abundM - abmp;  if (abmI < 0.0) abmI = 0.0;

    // --- energy, temperature (.F:106-115) ---
    double energy = y[itmp];
    double yntot = (1.0 + c.abhe - abh2 + abe) * yn;
    double ekn = energy / (yntot * CHEM_KBOLTZ);
    double temp, gamma;
    cxx_lookup_temp_gamma(abh2, c.abhe, ekn, &temp, &gamma, T);

    // --- temperature bounds / table index (.F:121-139) ---
    ChemTIdx ti = chem_temp_index(temp, T);
    if (ti.flag_bad) flagged = 1;                            // ipar(1) = 1, temp = tmin
    temp = ti.temp;

    // --- rate coefficients (.F:143-183) ---
    const double ch1   = chem_CH(T, 1,   ti);
    const double ch2   = chem_CH(T, 2,   ti);
    const double ch3   = chem_CH(T, 3,   ti);
    const double ch4   = chem_CH(T, 4,   ti);
    const double ch5   = chem_CH(T, 5,   ti);
    const double ch6   = chem_CH(T, 6,   ti);
    const double ch9   = chem_CH(T, 9,   ti);
    const double ch11  = chem_CH(T, 11,  ti);
    const double ch13  = chem_CH(T, 13,  ti);
    const double ch17  = chem_CH(T, 17,  ti);
    const double ch25  = chem_CH(T, 25,  ti);
    const double ch34  = chem_CH(T, 34,  ti);
    const double ch35  = chem_CH(T, 35,  ti);
    const double ch40  = chem_CH(T, 40,  ti);
    const double ch87  = chem_CH(T, 87,  ti);
    const double ch92  = chem_CH(T, 92,  ti);
    const double ch105 = chem_CH(T, 105, ti);
    const double ch106 = chem_CH(T, 106, ti);
    const double ch146 = chem_CH(T, 146, ti);
    const double ch147 = chem_CH(T, 147, ti);

    // H3+ dissociative recombination branch to H2 + H (.F:188)
    const double ch106a = (2.34 / 6.7) * ch106;
    const double ch87a  = 0.92 * ch87;      // HCO+ + e -> CO + H (Geppert+05)
    const double ch87b  = 0.08 * ch87;      // HCO+ + e -> OH + C

    // --- H2 collisional dissociation, density-dependent (.F:193-198) ---
    double ykdh = 0.0, ykdh2 = 0.0;          // negligible below 500 K
    if (temp >= 5e2) {
        double ncrinv = 2.0 * abh2 * (ch6 - ch5) + ch5;
        double h2var0 = 1.0 / (1.0 + yn * ncrinv);
        ykdh  = ch1 * std::pow(ch3, h2var0);
        ykdh2 = ch2 * std::pow(ch4, h2var0);
    }

    // --- photodissociation (.F:202-209); shielding called for every iphoto ---
    double fshield = cxx_calc_H2_shielding_factor(temp, abh2, NH, c);
    double ykdis = c.phrates[0] * fshield;
    double ph2  = c.phrates[1];
    double ph10 = c.phrates[9];
    double ph48 = c.phrates[47];
    double ph49 = c.phrates[48];
    double ph50 = c.phrates[49];
    double ph51 = c.phrates[50];

    // --- grain recombination: phi floored at 100, hot-gas suppression ---
    double h_gr = chem_h_gr_phi100(temp, yn, abe, c.G_dust, c.dust_to_gas_ratio, ch34, ch35);

    // --- dust temperature + H2 formation on grains (.F:240-249) ---
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

    // --- cosmic rays (.F:254-257); table is per unit rate ---
    double cr1 = T.crtab[0] * c.cosmic_ray_ion_rate;
    double cr2 = T.crtab[1] * c.cosmic_ray_ion_rate;
    double cr6 = T.crtab[5] * c.cosmic_ray_ion_rate;
    // C ionization, CO dissociation; CR-induced C photoionization and CO photodissociation
    double cr3  = T.crtab[2]  * c.cosmic_ray_ion_rate;
    double cr13 = T.crtab[12] * c.cosmic_ray_ion_rate;
    double cp1  = T.crphot[0] * c.cosmic_ray_ion_rate;
    double cp11 = T.crphot[10] * c.cosmic_ray_ion_rate * std::sqrt(temp) * abh2
                * std::sqrt(abco);

    const double cst1  = T.cst[0],  cst54 = T.cst[53], cst59 = T.cst[58],
                 cst75 = T.cst[74], cst81 = T.cst[80], cst82 = T.cst[81],
                 cst83 = T.cst[82], cst84 = T.cst[83], cst85 = T.cst[84],
                 cst86 = T.cst[85];

    // --- H3+ equilibrium abundance (.F:261-274) ---
    double fh3p_den = ch105 * abh2 + cst1 * abHI;
    double fh3p = (fh3p_den > 0.0) ? ch105 * abh2 / fh3p_den : 0.0;

    double num   = fh3p * ((cr2 / yn) * abh2 + cst75 * abhep * abh2);
    double denom = cst84 * abcI + cst86 * aboI + cst59 * abco
                 + ch106 * abe + cst81 * abmI;
    double abh3p = num / denom;
    abHI = (abHI > 3.0 * abh3p) ? abHI - 3.0 * abh3p : 0.0;   // H3+ holds three H atoms of the HI pool; not capped
    abe = abe + abh3p;

    // --- protons (.F:278-284) ---
    ydot[ihp] = (ch11 * abe   * abHI
              -  ch13 * abhp  * abe
              -  h_gr * abhp  * 1.0
              + (1.0 - fh3p)  * cst75 * abhep * abh2
              +  ch9  * abhep * abh2) * yn
              +  cr1  * abHI
              + (1.0 - fh3p) * cr2 * abh2
              +  ph50 * abhcop;

    // --- molecular hydrogen (.F:288-302) ---
    ydot[ih2] = (-ch25  * abe  * abh2
              -   ykdh  * abHI * abh2
              -   ykdh2 * abh2 * abh2
              +   Rdust * abHI * 1.0
              -   cst82 * abh2 * abcp
              -   cst54 * abcI * abh2
              +   cst84 * abh3p * abcI
              +   cst86 * abh3p * aboI
              +   cst59 * abh3p * abco
              -     ch9 * abhep * abh2
              - 2.0 * fh3p * cst75 * abhep * abh2   // via H2+ + H2
              +  ch106a * abh3p * abe
              +   cst81 * abh3p * abmI) * yn
              - 2.0 * fh3p * cr2 * abh2
              -   ykdis * abh2;

    // --- C+ (.F:306-310) ---
    ydot[ic] = ( ch92 * abhep * abco
             -   ch40 * abcp * abe
             -  cst82 * abcp * abh2
             -  cst83 * abcp * abohx) * yn
             +    ph2 * abcI
             +    cr3 * abcI
             +    cp1 * abcI;

    // --- CHx (.F:314-318) ---
    ydot[ichx] = (cst84 * abcI * abh3p
               +  cst82 * abcp * abh2
               +  cst54 * abcI * abh2
               -  cst85 * aboI * abchx) * yn
               -   ph48 * abchx;

    // --- OHx (.F:322-325) ---
    ydot[iohx] = (cst86 * aboI * abh3p
               +  ch87b * abhcop * abe
               -  ch147 * abcI * abohx
               -  cst83 * abcp * abohx) * yn
               -   ph49 * abohx;

    // --- HCO+ (.F:329-332) ---
    ydot[ihcop] = (cst59 * abh3p  * abco
                +  cst83 * abcp   * abohx
                -   ch87 * abhcop * abe) * yn
                -   ph50 * abhcop;

    // --- CO (.F:336-342) ---
    ydot[ico] = (cst85 * abchx  * aboI
              +  ch147 * abohx  * abcI
              +  ch87a * abhcop * abe
              -  cst59 * abh3p  * abco
              -   ch92 * abhep  * abco) * yn
              -   ph10 * abco
              +   ph50 * abhcop
              -   cr13 * abco
              -   cp11;

    // --- M+ (.F:346-348) ---
    ydot[imp] = (cst81 * abh3p * abmI
              -  ch146 * abmp  * abe) * yn
              +   ph51 * abmI;

    // --- He+ (.F:352-356) ---
    ydot[ihep] = cr6 * abheI
               - ( ch9 * abhep * abh2
               +  ch92 * abhep * abco
               +  ch17 * abhep * abe
               + cst75 * abhep * abh2) * yn;

    // --- chemical heating/cooling, rates_chem(1..6) (.F:360-410) ---
    const double ynsq = yn * yn;                             // yn**2
    double rc[6];
    rc[0] = 4.48 * CHEM_EV * (ykdh  * abHI * abh2 +
                              ykdh2 * abh2 * abh2 +
                              ch25  * abe  * abh2) * ynsq;
    rc[1] = -(4e-1 * CHEM_EV * abh2 * yn * ykdis);

    double cdex = ((double)1.4f * std::exp(-1.81e4 / (temp + 1200)) * abh2
                 + 1.0 * std::exp(-1e3 / temp) * abHI) * 1e-12 * std::sqrt(temp) * yn;
    double rdex = 2e-7;
    double fpheat = cdex / (cdex + rdex);
    rc[2] = -(2.0 * CHEM_EV * abh2 * yn * ykdis * 6.94 * fpheat);

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

    // --- cool_func abundance vector (.F:414-426) ---
    double abundances[16];
    for (int i = 0; i < 16; i++) abundances[i] = 0.0;
    abundances[0]  = abh2;
    abundances[1]  = aboI;
    abundances[4]  = abco;
    abundances[5]  = abcI;
    abundances[6]  = abcp;
    abundances[8]  = c.abundsi;
    abundances[9]  = abe;
    abundances[10] = abhp;
    abundances[11] = abHI;

    // --- local Jeans length (.F:429-432) ---
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
                "chem_rhs_net15: NaN ydot[%d] (ID %ld, T=%g yn=%g)\n",
                i, c.id_current, temp, yn);
            for (int j = 0; j < nspec; j++) ydot[j] = 0.0;
            return CHEM_RHS_RECOVERABLE;
        }
    }
    return flagged ? CHEM_RHS_RECOVERABLE : CHEM_RHS_OK;
}
