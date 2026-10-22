// chem_rhs_net5.cc -- verbatim-in-structure port of rate_eq_simple.F
// (CHEMISTRYNETWORK=5 branch only; nets 4/6 branches not ported).
// Source: /orion/ptmp/uli/orion/test/gizmo/cooling/chemcool/rate_eq_simple.F
// (canonical, read 2026-07-29, includes no net-5-relevant fixes pending).
//
// ODE vector (0-based): y[0]=H2, y[1]=H+, y[2]=CO, y[3]=energy density.
// Deviations from the Fortran (all per the audit port-semantics register):
//  - validate_input works on a local copy (R3); flagged input returns
//    CHEM_RHS_RECOVERABLE instead of zero-ydot+ipar.
//  - temp<=0 poisoned-input path returns CHEM_RHS_RECOVERABLE immediately
//    (Fortran computes a tmin-evaluated ydot AND rejects the substep after
//    the dvode call returns -- same net effect, documented in DEVIATIONS.md).
//  - validate_output NaN-ydot ABORT(1) is replaced by a loud recoverable
//    return (unified on the net-17 behavior; DEVIATIONS.md).
//  - the unreachable temp<0 bare `stop` (rate_eq_simple.F:505-511) is not
//    ported (audit R7 cleanup item).
//  - debug/compile-out switches BS19_XH2 / CHEM_EQ_H2 / NO_CO_FORM /
//    ADIABATIC_DENSITY_THRESHOLD / CONST_NCRIT_UVPUMP are OFF in production
//    and not ported.
//  - RT_INFRARED is OFF in the one-zone/production-subgrid config: the
//    calc_dust_temp call uses nocalc=0.

#include "../../../GIZMO_config.h"
#include "chem_types.h"
#include "chem_rhs_common.h"
#include <cstdio>

int chem_rhs_net5(double /*t*/, const double* y_in, double* ydot, ChemCell& c)
{
    const ChemTables& T = *c.tab;
    const int nspec = 4;
    const int ih2 = CIH2, ihp = CIHP, ico = CICO, itmp = 3;

    const double yn = c.yn, dl = c.dl, divv = c.divv;
    const double NH = yn * dl;                       // rate_eq_simple.F:73

    // --- validate_input on a local copy (R3) ---
    double y[4];
    for (int i = 0; i < nspec; i++) { y[i] = y_in[i]; ydot[i] = 0.0; }
    if (chem_validate_input(y, nspec, itmp, ih2, c.id_current))
        return CHEM_RHS_RECOVERABLE;

    // --- tracked species (floors as in .F:92-97) ---
    double abhp = (y[ihp] > 0.0) ? y[ihp] : 0.0;
    double abh2 = (y[ih2] > 0.0) ? y[ih2] : 0.0;
    double abco = (y[ico] > 0.0) ? y[ico] : 0.0;

    // --- conservation-law species (.F:106-113) ---
    double abHI = 1.0 - 2.0 * abh2 - abhp; if (abHI < 0.0) abHI = 0.0;
    // H+ above 1 (clipped by validate_input) still sees photoionization as a restoring sink
    const double abHI_ph = 1.0 - 2.0 * abh2 - ((y_in[ihp] > abhp) ? y_in[ihp] : abhp);
    const double abHI_ph_rate = (abHI_ph < 0.0) ? abHI_ph : abHI;
    double abcp = c.abundc - abco;         if (abcp < 0.0) abcp = 0.0;
    double abe  = abhp + abcp + c.abundsi;

    // --- energy, temperature (.F:117-126) ---
    double energy = y[itmp];
    double yntot = (1.0 + c.abhe - abh2 + abe) * yn;
    double ekn = energy / (yntot * CHEM_KBOLTZ);
    double temp, gamma;
    cxx_lookup_temp_gamma(abh2, c.abhe, ekn, &temp, &gamma, T);

    // --- temperature bounds / table index (.F:150-168) ---
    ChemTIdx ti = chem_temp_index(temp, T);
    if (ti.flag_bad) return CHEM_RHS_RECOVERABLE;    // temp<=0 poisoned input
    temp = ti.temp;

    // --- rate coefficients (.F:172-194) ---
    const double ch2  = chem_CH(T, 2,  ti);
    const double ch4  = chem_CH(T, 4,  ti);
    const double ch5  = chem_CH(T, 5,  ti);
    const double ch11 = chem_CH(T, 11, ti);
    const double ch13 = chem_CH(T, 13, ti);
    const double ch25 = chem_CH(T, 25, ti);
    const double ch34 = chem_CH(T, 34, ti);
    const double ch35 = chem_CH(T, 35, ti);
    const double ch137 = chem_CH(T, 137, ti), ch138 = chem_CH(T, 138, ti);
    const double ch139 = chem_CH(T, 139, ti);
    const double ch161 = chem_CH(T, 161, ti), ch162 = chem_CH(T, 162, ti);
    const double ch163 = chem_CH(T, 163, ti), ch164 = chem_CH(T, 164, ti);
    const double ch165 = chem_CH(T, 165, ti), ch166 = chem_CH(T, 166, ti);
    const double ch167 = chem_CH(T, 167, ti), ch168 = chem_CH(T, 168, ti);
    const double ch169 = chem_CH(T, 169, ti), ch170 = chem_CH(T, 170, ti);
    const double ch171 = chem_CH(T, 171, ti), ch172 = chem_CH(T, 172, ti);

    // --- H2 collisional dissociation as net 17: MSM96 CID + DT by H; He neutral ---
    const double msm_a = (double)(float)-1.164408f;   // single-precision literals in the .F
    const double msm_b = (double)(float)-1.174242f;
    double cdh2_h = (ch161 - (ch161 - ch163)
                  / (1.0 + std::pow(yn * abHI / ch165, ch166)))
                  + (ch162 - (ch162 - ch164)
                  / (1.0 + std::pow(yn * abHI / (std::pow(10.0, msm_a) * ch165), ch166)));
    cdh2_h = std::pow(10.0, cdh2_h);

    double cdh2_h_dt = (ch167 - (ch167 - ch169)
                     / (1.0 + std::pow(yn * abHI / ch171, ch172)))
                     + (ch168 - (ch168 - ch170)
                     / (1.0 + std::pow(yn * abHI / (std::pow(10.0, msm_b) * ch171), ch172)));
    cdh2_h_dt = std::pow(10.0, cdh2_h_dt);

    cdh2_h = cdh2_h + cdh2_h_dt;

    double ncrinv = ch5 + c.abhe * ch139;
    double h2var0 = 1.0 / (1.0 + yn * ncrinv);
    double h2var2 = std::pow(ch4, h2var0);
    double cdh2_h2 = ch2 * h2var2;
    double cdh2_he;
    if (ch138 < 1e-100) cdh2_he = 1e-100;
    else                cdh2_he = ch138 * std::pow(ch137 / ch138, h2var0);

    // --- photodissociation (.F:208-225) ---
    double fshield = (c.iphoto == 0) ? 1.0
                     : cxx_calc_H2_shielding_factor(temp, abh2, NH, c);
    double ykdis = c.phrates[0] * fshield;           // phrates(1)
    double gamma_co  = c.phrates[9];                 // phrates(10)
    double gamma_chx = 2.94e-10 * c.G_dust;
    double ph38 = c.phrates[37];
    double ph42 = c.phrates[41];

    // --- grain recombination (.F:232-253) ---
    double h_gr = chem_h_gr(temp, yn, abe, c.G_dust, c.dust_to_gas_ratio, ch34, ch35);

    // --- dust temperature + H2 formation on grains (.F:255-264) ---
    double RH2, Rdust;
    cxx_calc_dust_temp(yn, c.chi_isrf, temp, abHI, &RH2, &Rdust, 0, c);

    // --- cosmic rays (.F:269-272); table is per unit rate ---
    double cr1  = T.crtab[0] * c.cosmic_ray_ion_rate;
    double cr2  = T.crtab[1] * c.cosmic_ray_ion_rate;
    double cr10 = T.crtab[9] * c.cosmic_ray_ion_rate;
    double cr11 = T.crtab[10] * c.cosmic_ray_ion_rate;

    // --- protons (.F:276-281) ---
    ydot[ihp] = (ch11 * abe  * abHI
              -  ch13 * abhp * abe
              -  h_gr * abhp * 1.0) * yn
              +  cr1  * abHI
              +  cr10 * abh2
              +  ph38 * abHI_ph_rate;

    // --- molecular hydrogen (.F:285-292) ---
    ydot[ih2] = (-ch25  * abe  * abh2
              -   cdh2_h  * abHI * abh2
              -   cdh2_h2 * abh2 * abh2
              -   cdh2_he * c.abhe * abh2
              +   Rdust * abHI * 1.0) * yn
              -   ykdis * abh2
              -   0.5 * cr2 * abh2   // "fudge factor of two gives correct HI production"
              -   cr10 * abh2
              -   cr11 * abh2;

    // --- CO, NL97 (.F:302-324) ---
    const double k0 = 5e-16, k1 = 5e-10;
    double abo = c.abundo - abco; if (abo < 0.0) abo = 0.0;  // .F:309 max(0.0,...)
    double beta;
    if (abo == 0.0)               beta = 0.0;
    else if (gamma_chx < 1e-30)   beta = 1.0;
    else                          beta = k1 * abo / (k1 * abo + gamma_chx / yn);
    ydot[ico] = k0 * abcp * abh2 * beta * yn - gamma_co * abco;

    // --- chemical heating/cooling, rates_chem(1..6) (.F:354-418) ---
    double rates_chem[6];
    rates_chem[0] = 4.48 * CHEM_EV * (cdh2_h  * abHI * abh2 +
                                      cdh2_h2 * abh2 * abh2 +
                                      cdh2_he * c.abhe * abh2 +
                                      ch25  * abe  * abh2) * (yn * yn);
    rates_chem[1] = -4e-1 * CHEM_EV * abh2 * yn * ykdis;

    double cdex = ((double)1.4f * std::exp(-1.81e4 / (temp + 1200)) * abh2
                 + 1.0 * std::exp(-1e3 / temp) * abHI) * 1e-12 * std::sqrt(temp) * yn;
    double rdex = 2e-7;
    double fpheat = cdex / (cdex + rdex);
    rates_chem[2] = -2.0 * CHEM_EV * abh2 * yn * ykdis * 6.94 * fpheat;

    double f_ex = (RH2 > 0.0)
        ? cxx_compute_h2_heating_fraction(temp, yn, abh2) * c.h2_form_ex : 0.0;
    double f_tot = c.h2_form_kin + f_ex;
    rates_chem[3] = -4.48 * CHEM_EV * f_tot * RH2;

    rates_chem[4] = 13.6 * CHEM_EV * ch11 * abe * abHI * (yn * yn);

    rates_chem[5] = CHEM_KBOLTZ * temp * abhp * (ch13 * abe + h_gr) * (yn * yn);
    if (temp < 100.0) rates_chem[5] = 0.0;            // .F:412-414

    double ylam_chem = rates_chem[0] + rates_chem[1] + rates_chem[2]
                     + rates_chem[3] + rates_chem[4] + rates_chem[5]
                     - (ph42 * abHI) * yn;            // M1 photoheating add-on

    for (int i = 0; i < 6; i++) c.lambda_chem[i] = rates_chem[i];  // .F:422-427
    for (int i = 6; i < CHEM_NRATES_CHEM_MAX; i++) c.lambda_chem[i] = 0.0;

    // --- cool_func abundance vector (.F:446-462) ---
    double abundances[16];
    for (int i = 0; i < 16; i++) abundances[i] = 0.0;
    abundances[0]  = abh2;
    { double a2 = c.abundo - abco; if (a2 < 0.0) a2 = 0.0; abundances[1] = a2 * (1.0 - abhp); }  // O+/O = H+/H
    abundances[4]  = abco;
    abundances[6]  = abcp;
    abundances[8]  = c.abundsi;
    abundances[9]  = abe;
    abundances[10] = abhp;
    abundances[11] = abHI;

    // --- Jeans length (.F:473-477) ---
    double rho = (1.0 + 4.0 * c.abhe) * CHEM_PROTONMASS * yn;
    double cs2 = gamma * (gamma - 1.0) * energy / rho;
    if (cs2 < 0.0) cs2 = 0.0;
    double L_jeans = std::sqrt(CHEM_PI * cs2 / (CHEM_GRAVITY * rho));

    double ylam, rates[CHEM_NRATES];
    int cferr = cxx_cool_func(temp, yn, L_jeans, divv, c.G_dust, abundances,
                              &ylam, rates, 0, c);
    if (cferr) return CHEM_RHS_RECOVERABLE;           // Fortran ABORT(0)/(2) sites

    // T_CMB floor
    double ylam_floor, rates_floor[CHEM_NRATES];
    if (cxx_cool_func(2.726 * (1.0 + c.redshift), yn, L_jeans, divv, c.G_dust, abundances,
                      &ylam_floor, rates_floor, 1, c))
        return CHEM_RHS_RECOVERABLE;

    // pdv_term is hard-zeroed in production (chemcool.cc:721) -- not carried.
    ydot[itmp] = -(ylam + ylam_chem - ylam_floor);

    // --- validate_output: NaN ydot (.F:502; unified recoverable, DEVIATIONS) ---
    for (int i = 0; i < nspec; i++) {
        if (ydot[i] != ydot[i]) {
            std::fprintf(stderr,
                "chem_rhs_net5: NaN ydot[%d] (ID %ld, T=%g yn=%g)\n",
                i, c.id_current, temp, yn);
            for (int j = 0; j < nspec; j++) ydot[j] = 0.0;
            return CHEM_RHS_RECOVERABLE;
        }
    }
    return CHEM_RHS_OK;
}
