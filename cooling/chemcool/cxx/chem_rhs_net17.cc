// chem_rhs_net17.cc -- verbatim-in-structure port of rate_eq_ulli.F
// (CHEMISTRYNETWORK=17) + comp_equilib.
// Source: /orion/ptmp/uli/orion/test/gizmo/cooling/chemcool/rate_eq_ulli.F
// (canonical, read 2026-07-29 -- includes the 2026-07-28 fixes:
//  METAL-ELECTRON FIX: abe += abundsi (line 172);
//  UNINITIALIZED-NH FIX: NH assigned BEFORE the H2-shielding call).
//
// ODE vector (0-based): y[0]=H2, y[1]=H+, y[2]=CO, y[3]=He+, y[4]=He++,
// y[5]=D+, y[6]=HD, y[7]=energy density.
// Deviations (documented in DEVIATIONS.md):
//  - validate_input on a local copy; flagged input / NaN guards / temp<=0
//    return CHEM_RHS_RECOVERABLE (audit R3).
//  - GALSF_RESOLVEDISM_ISOLATED_FB_TEST debug print not ported.
//  - lambda_chem diagnostic copies are written unconditionally (in the
//    Fortran they are OUTPUT_INDIVIDUAL_COOLRATES-only; AMBIGUOUS A8 -- the
//    diagnostic-only choice does not affect the ODE).
//  - RT_INFRARED off: calc_dust_temp nocalc=0.

#include "chem_types.h"
#include "chem_rhs_common.h"
#include <cstdio>

// comp_equilib (rate_eq_ulli.F:1063-1093).  Returns -1 sentinel on
// creation-without-destruction or over-saturated equilibrium (callers clamp).
static double comp_equilib(double num, double denom, double max_abundance, int id)
{
    static thread_local int nwarn = 0;
    if (num == 0.0) return 0.0;
    if (denom == 0.0) {
        if (nwarn < 50) { std::fprintf(stderr,
            "comp_equilib: creation without destruction, species %d\n", id); nwarn++; }
        return -1.0;
    }
    double r = num / denom;
    if (r > max_abundance) {
        if (nwarn < 50) { std::fprintf(stderr,
            "comp_equilib: impossible abundance, species %d (%g/%g=%g)\n",
            id, num, denom, r); nwarn++; }
        return -1.0;
    }
    return r;
}

int chem_rhs_net17(double /*t*/, const double* y_in, double* ydot, ChemCell& c)
{
    const ChemTables& T = *c.tab;
    const int nspec = 8;
    const int ih2 = CIH2, ihp = CIHP, ico = CICO, ihep = CIHEP, ihepp = CIHEPP,
              idp = CIDP, ihd = CIHD, itmp = 7;
    const int niter = 25;
    const double eqtol = 1e-3;

    const double yn = c.yn, dl = c.dl, divv = c.divv;
    const double ynsq = yn * yn;

    // --- validate_input on a local copy (R3) ---
    double y[8];
    for (int i = 0; i < nspec; i++) { y[i] = y_in[i]; ydot[i] = 0.0; }
    if (chem_validate_input(y, nspec, itmp, ih2, c.id_current))
        return CHEM_RHS_RECOVERABLE;

    // --- species unpack (.F:145-177) ---
    double abhp   = y[ihp];
    double abh2   = y[ih2];
    double abhep  = y[ihep];
    double abhepp = y[ihepp];
    double abco   = (y[ico] > 0.0) ? y[ico] : 0.0;

    double abdp, abhd, abDI;
    if (c.no_dchem == 0 && c.abundD > 0.0) {
        abdp = y[idp];
        abhd = y[ihd];
        abDI = c.abundD - abdp - abhd; if (abDI < 0.0) abDI = 0.0;
    } else {
        abdp = 0.0; abhd = 0.0; abDI = 0.0;
    }

    double abcp = c.abundc - abco; if (abcp < 0.0) abcp = 0.0;
    // METAL-ELECTRON FIX (2026-07-28): + abundsi
    double abe  = abhp + abhep + 2.0 * abhepp + abcp + abdp + c.abundsi;
    double abHI = 1.0 - 2.0 * abh2 - abhp - abhd; if (abHI < 0.0) abHI = 0.0;
    double abheI = c.abhe - abhep - abhepp; if (abheI < 0.0) abheI = 0.0;

    // --- energy / temperature with NaN guards (.F:181-218) ---
    double energy = y[itmp];
    double yntot = (1.0 + c.abhe + abe - abh2) * yn;
    if (yntot <= 0.0 || yntot != yntot) {
        std::fprintf(stderr, "chem_rhs_net17: bad yntot=%g (ID %ld)\n",
                     yntot, c.id_current);
        return CHEM_RHS_RECOVERABLE;
    }
    double ekn = energy / (CHEM_KBOLTZ * yntot);
    if (ekn != ekn) return CHEM_RHS_RECOVERABLE;
    double temp, gamma;
    cxx_lookup_temp_gamma(abh2, c.abhe, ekn, &temp, &gamma, T);
    if (temp != temp || gamma != gamma) return CHEM_RHS_RECOVERABLE;

    // --- Jeans length EARLY, with cs2 clamp (.F:222-226) ---
    double rho = (1.0 + 4.0 * c.abhe) * 1.6726e-24 * yn;  // mh literal, .F:40
    double cs2 = gamma * (gamma - 1.0) * energy / rho;
    if (cs2 < 0.0) cs2 = 0.0;
    double L_jeans = std::sqrt(CHEM_PI * cs2 / (CHEM_GRAVITY * rho));

    double CMB_temp = 2.726 * (1.0 + c.redshift);

    // --- temperature bounds / index (.F:235-253) ---
    ChemTIdx ti = chem_temp_index(temp, T);
    if (ti.flag_bad) return CHEM_RHS_RECOVERABLE;
    temp = ti.temp;

    // --- rate coefficients (.F:257-417) ---
    const double ch1  = chem_CH(T, 1,  ti), ch2  = chem_CH(T, 2,  ti);
    const double ch3  = chem_CH(T, 3,  ti), ch4  = chem_CH(T, 4,  ti);
    const double ch5  = chem_CH(T, 5,  ti), ch6  = chem_CH(T, 6,  ti);
    const double ch8  = chem_CH(T, 8,  ti), ch9  = chem_CH(T, 9,  ti);
    const double ch10 = chem_CH(T, 10, ti), ch11 = chem_CH(T, 11, ti);
    const double ch13 = chem_CH(T, 13, ti), ch14 = chem_CH(T, 14, ti);
    const double ch15 = chem_CH(T, 15, ti), ch17 = chem_CH(T, 17, ti);
    double       ch18 = chem_CH(T, 18, ti);
    const double ch20 = chem_CH(T, 20, ti), ch21 = chem_CH(T, 21, ti);
    const double ch22 = chem_CH(T, 22, ti), ch23 = chem_CH(T, 23, ti);
    double       ch24 = chem_CH(T, 24, ti);
    const double ch25 = chem_CH(T, 25, ti), ch26 = chem_CH(T, 26, ti);
    const double ch27 = chem_CH(T, 27, ti), ch28 = chem_CH(T, 28, ti);
    const double ch29 = chem_CH(T, 29, ti), ch30 = chem_CH(T, 30, ti);
    const double ch31 = chem_CH(T, 31, ti), ch32 = chem_CH(T, 32, ti);
    const double ch33 = chem_CH(T, 33, ti), ch34 = chem_CH(T, 34, ti);
    const double ch35 = chem_CH(T, 35, ti), ch92 = chem_CH(T, 92, ti);
    const double ch98 = chem_CH(T, 98, ti), ch99 = chem_CH(T, 99, ti);
    const double ch100 = chem_CH(T, 100, ti), ch101 = chem_CH(T, 101, ti);
    const double ch102 = chem_CH(T, 102, ti), ch103 = chem_CH(T, 103, ti);
    const double ch108 = chem_CH(T, 108, ti), ch109 = chem_CH(T, 109, ti);
    const double ch110 = chem_CH(T, 110, ti), ch126 = chem_CH(T, 126, ti);
    const double ch127 = chem_CH(T, 127, ti), ch137 = chem_CH(T, 137, ti);
    double       ch138 = chem_CH(T, 138, ti);
    const double ch139 = chem_CH(T, 139, ti), ch144 = chem_CH(T, 144, ti);
    const double ch153 = chem_CH(T, 153, ti), ch154 = chem_CH(T, 154, ti);
    const double ch155 = chem_CH(T, 155, ti), ch156 = chem_CH(T, 156, ti);
    double       ch157 = chem_CH(T, 157, ti);
    const double ch158 = chem_CH(T, 158, ti), ch159 = chem_CH(T, 159, ti);
    double       ch160 = chem_CH(T, 160, ti);
    const double ch161 = chem_CH(T, 161, ti), ch162 = chem_CH(T, 162, ti);
    const double ch163 = chem_CH(T, 163, ti), ch164 = chem_CH(T, 164, ti);
    const double ch165 = chem_CH(T, 165, ti), ch166 = chem_CH(T, 166, ti);
    const double ch167 = chem_CH(T, 167, ti), ch168 = chem_CH(T, 168, ti);
    const double ch169 = chem_CH(T, 169, ti), ch170 = chem_CH(T, 170, ti);
    const double ch171 = chem_CH(T, 171, ti), ch172 = chem_CH(T, 172, ti);
    double       ch173 = chem_CH(T, 173, ti);

    // --- H2 CID: MSM96 two-branch log form (.F:424-462) ---
    // NB the magic constants are SINGLE-precision literals in the Fortran
    // (-1.164408e0, -1.174242e0): reproduce their float rounding exactly.
    const double msm_a = (double)(float)-1.164408f;
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

    double ncrinv = ch5 + abheI * ch139;              // .F:464 (net-17 blend!)
    double h2var0 = 1.0 / (1.0 + yn * ncrinv);
    double h2var2 = std::pow(ch4, h2var0);
    double cdh2_h2 = ch2 * h2var2;
    double cdh2_he;
    if (ch138 < 1e-100) cdh2_he = 1e-100;
    else                cdh2_he = ch138 * std::pow(ch137 / ch138, h2var0);

    // --- H2 + H+ charge transfer, density-interpolated (.F:476-483) ---
    if (std::fabs(ch24) < 1e-100) ch24 = 1e-100;
    if (ch173 < 1e-100) ch173 = 1e-100;
    double h2_ct_low = ch24;
    ch24 = ch173 * std::pow(h2_ct_low / ch173, h2var0);

    // --- H2+ critical density (.F:487-494) ---
    double ncrit_h2p_h = 4e2 * (1e4 / temp);
    double ncrit_h2p_e = 5e1;
    double ncrit_h2p_inv;
    if (abHI > 0.0 || abe > 0.0)
        ncrit_h2p_inv = abHI / ncrit_h2p_h + abe / ncrit_h2p_e;
    else
        ncrit_h2p_inv = 1.0 / 4e2;
    double h2p_alpha = 1.0 / (1.0 + yn * ncrit_h2p_inv);

    // --- H2+ dissociative recombination / CID by H (.F:498-508) ---
    if (ch157 < 1e-100) ch157 = 1e-100;
    double h2p_dr = ch157 * std::pow(ch30 / ch157, h2p_alpha);
    if (ch160 < 1e-100) ch160 = 1e-100;
    double h2p_cdh = ch160 * std::pow(ch159 / ch160, h2p_alpha);

    // --- photodissociation rates (.F:512-573) ---
    // local H2 shielding column over the full length dl, as net 5 and calc_photo
    double NH = dl * yn;
    double fshield = (c.iphoto == 0) ? 1.0
                     : cxx_calc_H2_shielding_factor(temp, abh2, NH, c);
    double ykdis = c.phrates[0] * fshield;

    double ph4 = c.phrates[3];
    // CMB H- photodetachment: thermal (GP98) + non-thermal (Coppola+11)
    ph4 = ph4 + 0.11 * std::pow(CMB_temp, 2.13) * std::exp(-8.823e3 / CMB_temp);
    ph4 = ph4 + 8.0e-8 * std::pow(CMB_temp, 1.3) * std::exp(-2.3e3 / CMB_temp);

    double ph5;
    if (c.G0 == 0.0) {
        ph5 = 0.0;
    } else {
        double h2p_pd_low = c.phrates[4];
        double h2p_pd_lte = 2.2e-11 * c.G0 * std::pow(temp, -0.22);
        if (temp < 3.15e3 && h2p_pd_lte > h2p_pd_low) h2p_pd_lte = h2p_pd_low;
        if (h2p_pd_lte < 1e-100) h2p_pd_lte = 1e-100;
        ph5 = h2p_pd_lte * std::pow(h2p_pd_low / h2p_pd_lte, h2p_alpha);
    }

    double ph8  = c.phrates[7];
    double ph9  = c.phrates[8];
    double ph35 = c.phrates[34];
    double ph38 = c.phrates[37];
    double ph39 = c.phrates[38];
    double ph40 = c.phrates[39];
    double ph41 = c.phrates[40];
    double ph42 = c.phrates[41];
    double ph43 = c.phrates[42];
    double ph44 = c.phrates[43];

    double gamma_co  = c.phrates[9];
    double gamma_chx = 2.94e-10 * c.G_dust;

    // --- grain recombination (.F:577-594) ---
    double h_gr = chem_h_gr(temp, yn, abe, c.G_dust, c.dust_to_gas_ratio, ch34, ch35);

    // --- dust temperature / H2 formation on grains (.F:596-605) ---
    double RH2, Rdust;
    cxx_calc_dust_temp(yn, c.chi_isrf, temp, abHI, &RH2, &Rdust, 0, c);

    // --- cosmic rays (.F:607-610); table is per unit rate ---
    double cr1  = T.crtab[0] * c.cosmic_ray_ion_rate;
    double cr1a = T.crtab[0] * c.cosmic_ray_ion_rate;
    double cr2  = T.crtab[1] * c.cosmic_ray_ion_rate;
    double cr6  = T.crtab[5] * c.cosmic_ray_ion_rate;

    // --- optical depth to He+-recombination photons (.F:614-618) ---
    NH = 0.5 * dl * yn;
    if (2.0 * NH * abHI < 1.6e17) ch18 = 0.0;

    // --- equilibrium species: H-, H2+, HeH+ (.F:622-713) ---
    double abhm_old = 0.0, abh2p_old = 0.0, abhehp_old = 0.0;
    double abhm = 0.0, abh2p = 0.0, abhehp = 0.0;
    double abM = 0.0;   // persists past the loop; ydot(ihp) uses the value from
                        // the LAST executed iteration (Fortran variable scoping)

    for (int I = 1; I <= niter; I++) {
        // H-
        double num   = ch21 * abHI * abe + ch126 * abh2 * abe;
        double denom = ch22 * abHI + ch26 * abe + ch27 * abHI + ch28 * abhp
                     + ch29 * abhp + (ph4 / yn) + ch8 * abhep
                     + ch31 * abheI + 2.0 * ch127 * abh2p;
        abhm = comp_equilib(num, denom, 1.0, 1);
        if (abhm < 0.0) abhm = 0.0;
        abe  = abe  - (abhm - abhm_old); if (abe  < 0.0) abe  = 0.0;
        abHI = abHI - (abhm - abhm_old); if (abHI < 0.0) abHI = 0.0;

        // H2+
        abM = abHI + abh2 / std::sqrt(2.0) + abheI / 2.0;
        num   = ch23 * abHI * abhp + ch24 * abh2 * abhp
              + ch29 * abhp * abhm + ch156 * abhehp * abHI
              + (cr2 * abh2 / yn) + T.cst[74] * abhep * abh2
              + (ph41 / yn) * abh2;
        denom = T.cst[0] * abHI + h2p_dr * abe + (ph5 / yn)
              + 2.0 * ch127 * abhm + ch154 * abheI
              + h2p_cdh * abM;
        abh2p = comp_equilib(num, denom, 0.5, 2);
        if (abh2p < 0.0) abh2p = 0.0;
        abe  = abe  + (abh2p - abh2p_old);        if (abe  < 0.0) abe  = 0.0;
        abHI = abHI - 2.0 * (abh2p - abh2p_old);  if (abHI < 0.0) abHI = 0.0;

        // HeH+
        num   = ch155 * abheI * abhp + ch154 * abheI * abh2p;
        denom = ch156 * abHI + ch153 * abe;
        abhehp = comp_equilib(num, denom, c.abhe, 3);
        if (abhehp < 0.0) abhehp = 0.0;           // WEDGEFIX 1
        abe   = abe   + (abhehp - abhehp_old); if (abe   < 0.0) abe   = 0.0;
        abHI  = abHI  - (abhehp - abhehp_old); if (abHI  < 0.0) abHI  = 0.0;
        abheI = abheI - (abhehp - abhehp_old); if (abheI < 0.0) abheI = 0.0;

        // convergence (.F:697-703)
        if (I >= 2) {
            if (std::fabs(abhm_old - abhm)     < eqtol * abhm &&
                std::fabs(abh2p_old - abh2p)   < eqtol * abh2p &&
                std::fabs(abhehp_old - abhehp) < eqtol * abhehp)
                break;
        }
        abhm_old  = abhm;
        abh2p_old = abh2p;
        abhehp_old = abhehp;                      // WEDGEFIX 2
    }

    // --- protons (.F:717-744) ---
    ydot[ihp] = (ch11 * abe   * abHI
              + ch144 * abHI  * abHI
              + ch158 * abheI * abHI
              + T.cst[0] * abh2p * abHI
              -  ch13 * abhp  * abe
              -  h_gr * abhp  * 1.0
              -  ch23 * abHI  * abhp
              -  ch24 * abh2  * abhp
              -  ch28 * abhp  * abhm
              -  ch29 * abhp  * abhm
              +  ch18 * abhep * abe
              +  ch32 * abhep * abHI
              -  ch33 * abheI * abhp
              - ch155 * abheI * abhp
              + h2p_cdh * abh2p * abM
              +  ch9  * abhep * abh2) * yn
              +  ph5  * abh2p
              +  cr1  * abHI
              +  ph38 * abHI;
    if (c.no_dchem == 0 && c.abundD > 0.0) {
        ydot[ihp] += (ch99  * abdp * abHI
                   -  ch98  * abhp * abDI
                   +  ch100 * abdp * abh2
                   -  ch101 * abhd * abhp) * yn;
    }

    // --- molecular hydrogen (.F:748-775) ---
    ydot[ih2] = (ch108 * abHI * abHI * abHI
              +  ch109 * abHI * abHI * abh2
              +  ch110 * abHI * abHI * abheI) * ynsq
              + (ch22   * abHI * abhm
              +  Rdust  * abHI * 1.0
              +  T.cst[0] * abHI * abh2p
              -  ch24   * abhp * abh2
              -  ch25   * abe  * abh2
              -  cdh2_h * abHI * abh2
              -  cdh2_h2* abh2 * abh2
              -  cdh2_he* abheI* abh2
              -  ch9    * abhep* abh2
              -  T.cst[74] * abhep * abh2
              -  ch126  * abh2 * abe
              +  ch127  * abhm * abh2p) * yn
              -  ykdis  * abh2
              -  cr2    * abh2
              -  ph41   * abh2;
    if (c.no_dchem == 0 && c.abundD > 0.0) {
        ydot[ih2] += (ch101 * abhd * abhp
                   -  ch100 * abh2 * abdp
                   -  ch102 * abh2 * abDI
                   +  ch103 * abhd * abHI) * yn;
    }

    // --- He+ (.F:779-813) ---
    if (c.abhe == 0.0) {
        ydot[ihep] = 0.0;
    } else {
        ydot[ihep] = (ch14 * abheI * abe
                   -  ch15 * abhep * abe
                   -  ch17 * abhep * abe
                   +  ch20 * abhepp * abe
                   -  ch32 * abhep * abHI
                   +  ch33 * abheI * abhp
                   -  ch8  * abhep * abhm
                   -  ch9  * abhep * abh2
                   -  T.cst[74] * abhep * abh2
                   -  ch92 * abhep * abco) * yn
                   +  cr6  * abheI
                   +  ph39 * abheI
                   -  ph40 * abhep;
        if (c.no_dchem == 0 && c.abundD > 0.0) {
            ydot[ihep] += (ch33 * abheI * abdp
                        -  ch32 * abhep * abDI) * yn;
        }
    }

    // --- He++ (.F:817-823) ---
    if (c.abhe == 0.0) {
        ydot[ihepp] = 0.0;
    } else {
        ydot[ihepp] = (ch15 * abhep * abe - ch20 * abhepp * abe) * yn
                    + ph40 * abhep;
    }

    // --- D+ (.F:827-840) ---
    if (c.abundD == 0.0 || c.no_dchem == 1) {
        ydot[idp] = 0.0;
    } else {
        ydot[idp] = (ch98  * abhp  * abDI
                  -  ch99  * abdp  * abHI
                  -  ch100 * abdp  * abh2
                  +  ch101 * abhd  * abhp
                  -  ch13  * abdp  * abe
                  +  ch11  * abDI  * abe
                  +  ch32  * abhep * abDI
                  -  ch33  * abheI * abdp) * yn
                  +  cr1a  * abDI
                  +  ph38  * abDI;
    }

    // --- HD (.F:848-860) ---
    if (c.abundD == 0.0 || c.no_dchem == 1) {
        ydot[ihd] = 0.0;
    } else {
        ydot[ihd] = (ch108 * abHI
                  +  ch109 * abh2
                  +  ch110 * abheI) * abHI * abDI * ynsq
                  + (ch100 * abh2 * abdp
                  -  ch101 * abhd * abhp
                  +  ch102 * abh2 * abDI
                  -  ch103 * abhd * abHI) * yn
                  -  ph35  * abhd
                  -  ph41  * abhd;
    }

    // --- CO: NL97 + He+ CT destruction (.F:864-878) ---
    const double k0 = 5e-16, k1 = 5e-10;
    double abo = c.abundo - abco; if (abo < 0.0) abo = 0.0;
    double beta;
    if (abo == 0.0)             beta = 0.0;
    else if (gamma_chx < 1e-30) beta = 1.0;
    else                        beta = k1 * abo / (k1 * abo + gamma_chx / yn);
    ydot[ico] = k0 * abcp * abh2 * beta * yn
              - ch92 * abhep * abco * yn
              - gamma_co * abco;

    // --- chemical heating/cooling, rates_chem(1..22) (.F:886-990) ---
    double rc[22];
    rc[0] = 4.48 * CHEM_EV * (cdh2_h * abHI * abh2 +
                              cdh2_h2 * abh2 * abh2 +
                              cdh2_he * abheI * abh2 +
                              ch25 * abe * abh2) * ynsq;
    rc[1] = 1.83 * CHEM_EV * ch24 * abh2 * abhp * ynsq;
    rc[2] = -4e-1 * CHEM_EV * abh2 * yn * ykdis;

    double cdex = ((double)1.4f * std::exp(-1.81e4 / (temp + 1200)) * abh2
                 + 1.0 * std::exp(-1e3 / temp) * abHI) * 1e-12 * std::sqrt(temp) * yn;
    double rdex = 2e-7;
    double fpheat = cdex / (cdex + rdex);
    rc[3] = -2.0 * CHEM_EV * abh2 * yn * ykdis * 6.94 * fpheat;

    rc[4] = -3.73 * CHEM_EV * ch22 * abHI * abhm * ynsq * (1.0 - h2var0);
    rc[5] = -1.83 * CHEM_EV * T.cst[0] * abHI * abh2p * ynsq * (1.0 - h2var0);
    rc[6] = -4.48 * CHEM_EV * (ch108 * abHI * abHI * abHI +
                               ch109 * abHI * abHI * abh2 +
                               ch110 * abHI * abHI * abheI) * ynsq * yn;
    rc[7] = 13.6 * CHEM_EV * (ch11 * abe + ch144 * abHI + ch158 * abheI)
          * abHI * ynsq;
    rc[8] = 24.6 * CHEM_EV * ch14 * abe * abheI * ynsq;
    rc[9] = 54.4 * CHEM_EV * ch15 * abe * abhep * ynsq;
    rc[10] = CHEM_KBOLTZ * temp * abhp * (ch13 * abe + h_gr) * ynsq;
    if (temp < 100.0) rc[10] = 0.0;
    rc[11] = 40.8 * CHEM_EV * ch10 * abhep * abe * ynsq;
    rc[12] = CHEM_KBOLTZ * temp * (ch17 - ch10) * abhep * abe * ynsq;
    rc[13] = CHEM_KBOLTZ * temp * abhepp * ch20 * abe * ynsq;
    rc[14] = -ph8 * abhm * yn;
    rc[15] = -ph9 * abh2p * yn;
    rc[16] = 0.0;
    rc[17] = 0.0;
    rc[18] = 0.0;
    rc[19] = -(ph42 * abHI + ph43 * abheI + ph44 * abhep) * yn;
    rc[20] = (0.75 * CHEM_EV + CHEM_KBOLTZ * temp) * ch21 * abHI * abe * ynsq;
    double f_ex = (RH2 > 0.0)
        ? cxx_compute_h2_heating_fraction(temp, yn, abh2) * c.h2_form_ex : 0.0;
    rc[21] = -4.48 * CHEM_EV * (c.h2_form_kin + f_ex) * RH2;

    double ylam_chem = 0.0;
    for (int i = 0; i < 22; i++) {
        ylam_chem += rc[i];
        c.lambda_chem[i] = rc[i];   // unconditional in port (A8 choice)
    }

    // --- cool_func abundance vector (.F:994-1007) ---
    double abundances[16];
    for (int i = 0; i < 16; i++) abundances[i] = 0.0;
    abundances[0]  = abh2;
    abundances[1]  = abo * (1.0 - abhp);   // "Assume O+/O = H+/H"
    abundances[4]  = abco;
    abundances[6]  = abcp;
    abundances[8]  = c.abundsi;
    abundances[9]  = abe;
    abundances[10] = abhp;
    abundances[11] = abHI;
    abundances[12] = abhd;
    abundances[13] = abheI;
    abundances[14] = abhep;
    abundances[15] = abhepp;

    double ylam, rates[CHEM_NRATES];
    int cferr = cxx_cool_func(temp, yn, L_jeans, divv, c.G_dust, abundances,
                              &ylam, rates, 0, c);
    if (cferr) return CHEM_RHS_RECOVERABLE;

    double ylam_cmb = 0.0;
    if (c.redshift > 0.0) {
        double rates_cmb[CHEM_NRATES];
        int e2 = cxx_cool_func(CMB_temp, yn, L_jeans, divv, c.G_dust, abundances,
                               &ylam_cmb, rates_cmb, 1, c);
        if (e2) return CHEM_RHS_RECOVERABLE;
    }

    ydot[itmp] = -(ylam - ylam_cmb + ylam_chem);

    // --- final NaN sweep (.F:1026-1044) ---
    for (int i = 0; i < nspec; i++) {
        if (ydot[i] != ydot[i]) {
            std::fprintf(stderr,
                "chem_rhs_net17: NaN ydot[%d] (ID %ld, T=%g yn=%g e=%g)\n",
                i, c.id_current, temp, yn, energy);
            for (int j = 0; j < nspec; j++) ydot[j] = 0.0;
            return CHEM_RHS_RECOVERABLE;
        }
    }
    return CHEM_RHS_OK;
}
