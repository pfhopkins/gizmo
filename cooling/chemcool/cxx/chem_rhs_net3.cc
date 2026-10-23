// chem_rhs_net3.cc -- verbatim-in-structure port of rate_eq_mol.F
// (CHEMISTRYNETWORK=3, Glover+10 molecular network) + comp_equilib.
//
// ODE vector (0-based, chemcool_consts.h): y[0]=H2, y[1]=H+, y[2]=C+, y[3]=Si+,
// y[4]=O+, y[5]=D+, y[6]=HD, y[7]=He+, y[8]=He++, y[9]=CO, y[10]=C2, y[11]=OH,
// y[12]=H2O, y[13]=O2, y[14]=HCO+, y[15]=CH, y[16]=CH2, y[17]=Si++, y[18]=CH3+,
// y[19]=energy density.
// Deviations (DEVIATIONS.md D-28):
//  - validate_input on a local copy; a flagged input returns ydot = 0 and
//    CHEM_RHS_RECOVERABLE.  temp <= 0 builds ydot at tmin and returns recoverable.
//  - comp_equilib / bad temperature index ABORTs return CHEM_RHS_FATAL with
//    c.fatal_code = the ABORT code; the driver returns it (chemcool_fatal path).
//  - NaN ydot returns recoverable (no validate_output in the Fortran).
//  - calc_photo runs inside every evaluation, as in the Fortran (writes c.phrates,
//    c.G_dust, c.chi_isrf, c.fshield_CO).
//  - not ported: ADIABATIC_DENSITY_THRESHOLD, FF_COLLAPSE_TEST, COLLECT_EQB_DATA,
//    DEBUG_RATE_EQ (all off in production).

#include "chem_types.h"
#include "chem_rhs_common.h"
#include <cstdio>
#include <cmath>

// comp_equilib (rate_eq_mol.F:1785-1819); returns the ABORT code, 0 if none
static int comp_equilib3(double num, double denom, double max_abundance, int id,
                         double* out)
{
    if (num == 0.0) { *out = 0.0; return 0; }
    if (num < 1e-30) { *out = 0.0; return 0; }
    if (denom == 0.0) {
        std::fprintf(stderr, "Destruction without creation in comp_equilib\n"
                             "Species = %d\n %g %g\n", id, num, denom);
        return 9751;
    }
    double r = num / denom;
    if (r > max_abundance) {
        std::fprintf(stderr, "Error: impossible abundance!\nSpecies %d\n %g %g %g\n",
                     id, num, denom, r);
        return 9752;
    }
    *out = r;
    return 0;
}

static inline double fmax0(double x) { return (x > 0.0) ? x : 0.0; }

int chem_rhs_net3(double /*t*/, const double* y_in, double* ydot, ChemCell& c)
{
    const ChemTables& T = *c.tab;
    const int nspec = 20;
    const int ih2 = C3_IH2, ihp = C3_IHP, ic = C3_IC, isi = C3_ISI, io = C3_IO,
              idp = C3_IDP, ihd = C3_IHD, ihep = C3_IHEP, ihepp = C3_IHEPP,
              ico = C3_ICO, ic2 = C3_IC2, ioh = C3_IOH, ih2o = C3_IH2O, io2 = C3_IO2,
              ihcop = C3_IHCOP, ich = C3_ICH, ich2 = C3_ICH2, isipp = C3_ISIPP,
              ich3p = C3_ICH3P, itmp = 19;
    const int niter = 10;
    const double eqtol = 1e-3;
    const double abhe = c.abhe, abundc = c.abundc, abundo = c.abundo,
                 abundsi = c.abundsi, abundD = c.abundD;

    const double yn = c.yn, dl = c.dl, divv = c.divv;
    // HI column within dl (.F:158-162)
    const double NH = (c.iphoto == 3) ? dl * yn : 0.5 * dl * yn;

    // --- validate_input on a local copy (.F:172-182) ---
    double y[20];
    for (int i = 0; i < nspec; i++) { y[i] = y_in[i]; ydot[i] = 0.0; }
    if (chem_validate_input_full(y, nspec, itmp, ih2, c.id_current) != 0)
        return CHEM_RHS_RECOVERABLE;
    int flagged = 0;

    // --- tracked species (.F:186-204) ---
    double abhp   = y[ihp];
    double abh2   = y[ih2];
    double abhd   = y[ihd];
    double abdp   = y[idp];
    double abhep  = y[ihep];
    double abhepp = y[ihepp];
    double abcp   = y[ic];
    double abop   = y[io];
    double aboh   = y[ioh];
    double abh2o  = y[ih2o];
    double abco   = y[ico];
    double absip  = y[isi];
    double abc2   = y[ic2];
    double abo2   = y[io2];
    double abhcop = y[ihcop];
    double abch   = y[ich];
    double abch2  = y[ich2];
    double abch3p = y[ich3p];
    double absipp = y[isipp];

    // --- conservation-law species (.F:206-233) ---
    double abe = abhp + abdp + abhep + abcp + abop + absip + abhcop
               + 2.0 * absipp + abch3p + 2.0 * abhepp;
    double abHI = 1.0 - 2.0 * abh2 - abhp - abhd - aboh
                - 2.0 * abh2o - abhcop - abch - 2.0 * abch2 - 3.0 * abch3p;
    if (abHI < 0.0) abHI = 0.0;
    // H+ above 1 (clipped by validate_input) still sees photoionization as a restoring sink (D-30)
    const double abHI_ph = 1.0 - 2.0 * abh2 - ((y_in[ihp] > abhp) ? y_in[ihp] : abhp) - abhd - aboh
                         - 2.0 * abh2o - abhcop - abch - 2.0 * abch2 - 3.0 * abch3p;
    double abDI = abundD - abdp - abhd;
    if (abDI < 0.0) abDI = 0.0;
    double abheI = abhe - abhep - abhepp;
    if (abheI < 0.0) abheI = 0.0;
    double abcI = abundc - abcp - abco - 2.0 * abc2 - abhcop - abch - abch2 - abch3p;
    if (abcI < 0.0) abcI = 0.0;
    double aboI = abundo - abop - aboh - abh2o - abco - 2.0 * abo2 - abhcop;
    if (aboI < 0.0) aboI = 0.0;
    double absiI = abundsi - absip - absipp;
    if (absiI < 0.0) absiI = 0.0;
    double ab3M = abHI + abh2 + abheI;

    // --- energy, temperature (.F:236-246) ---
    double energy = y[itmp];
    double yntot = (1.0 + abhe - abh2 + abe) * yn;
    double ekn = energy / (yntot * CHEM_KBOLTZ);
    double temp, gamma;
    cxx_lookup_temp_gamma(abh2, abhe, ekn, &temp, &gamma, T);
    const double ynsq = yn * yn;                             // yn**2
    const double yncb = chem_fpowi(yn, 3);                   // yn**3

    // --- bounds (.F:250-268); above tmax dtemp is NOT zeroed in this network ---
    ChemTIdx ti;
    ti.flag_bad = 0;
    if (temp <= 0.0) {
        ti.itemp = 1; ti.dtemp = 0.0; temp = T.tmin; flagged = 1;
    } else if (temp <= T.tmin) {
        ti.itemp = 1; ti.dtemp = 0.0;
    } else if (temp >= T.tmax) {
        ti.itemp = CHEM_NMD; ti.dtemp = temp - T.temptab[CHEM_NMD - 1];
    } else {
        ti.itemp = (int)(std::log10(temp) / T.dtlog) + 1;
        if (ti.itemp <= 0 || ti.itemp > CHEM_NMD) {
            std::fprintf(stderr, " Fatal error in rate_eq.F %d %g\n", ti.itemp, temp);
            c.fatal_code = 1;
            return CHEM_RHS_FATAL;                           // ABORT(1)
        }
        ti.dtemp = temp - T.temptab[ti.itemp - 1];
    }
    ti.temp = temp;

    // --- rate coefficients (.F:272-526) ---
#define CH(k) const double ch##k = chem_CH(T, k, ti)
    CH(1); CH(2); CH(3); CH(4); CH(5); CH(6); CH(8); CH(9); CH(10); CH(11);
    CH(13); CH(14); CH(15); CH(17); CH(20); CH(21); CH(22); CH(23); CH(24);
    CH(25); CH(26); CH(27); CH(28); CH(29); CH(30); CH(32); CH(33); CH(34);
    CH(35); CH(36); CH(37); CH(38); CH(39); CH(40); CH(41); CH(42); CH(43);
    CH(44); CH(45); CH(46); CH(47); CH(48); CH(49); CH(50); CH(51); CH(52);
    CH(53); CH(54); CH(55); CH(56); CH(57); CH(58); CH(59); CH(60); CH(61);
    CH(62); CH(63); CH(64); CH(65); CH(66); CH(67); CH(68); CH(69); CH(70);
    CH(71); CH(72); CH(73); CH(74); CH(77); CH(78); CH(79); CH(80); CH(81);
    CH(82); CH(83); CH(84); CH(85); CH(86); CH(87); CH(88); CH(89); CH(90);
    CH(92); CH(93); CH(94); CH(95); CH(96); CH(97); CH(98); CH(99); CH(100);
    CH(101); CH(102); CH(103); CH(104); CH(105); CH(106); CH(107); CH(108);
    CH(109); CH(110); CH(111); CH(112); CH(113); CH(114); CH(115); CH(116);
    CH(117); CH(118); CH(119); CH(121); CH(122); CH(124); CH(131); CH(132);
    CH(133); CH(134); CH(135); CH(136);
#undef CH
    double ch7  = chem_CH(T, 7, ti) * c.dust_to_gas_ratio;
    double ch18 = chem_CH(T, 18, ti);

    // H3+ dissociative recombination branch to H2 + H (.F:531)
    const double ch106a = (2.34 / 6.7) * ch106;

    // --- H2 collisional dissociation (.F:538-543) ---
    double ncrinv = 2.0 * abh2 * (ch6 - ch5) + ch5;
    double h2var0 = 1.0 / (1.0 + yn * ncrinv);
    double h2var1 = std::pow(ch3, h2var0);
    double h2var2 = std::pow(ch4, h2var0);
    double ykdh   = ch1 * h2var1;
    double ykdh2  = ch2 * h2var2;

    // --- photochemistry, evaluated here as in the Fortran (.F:547-611) ---
    cxx_calc_photo(temp, abh2, 0.0, abco, c);
    const double* ph = c.phrates;
    const double ykdis = ph[0];
    const double ph2 = ph[1], ph3 = ph[2], ph4 = ph[3], ph5 = ph[4], ph6 = ph[5],
                 ph7 = ph[6], ph8 = ph[7], ph9 = ph[8], ph10 = ph[9], ph11 = ph[10],
                 ph12 = ph[11], ph14 = ph[13], ph15 = ph[14], ph16 = ph[15],
                 ph18 = ph[17], ph19 = ph[18], ph20 = ph[19], ph21 = ph[20],
                 ph22 = ph[21], ph23 = ph[22], ph24 = ph[23], ph25 = ph[24],
                 ph27 = ph[26], ph28 = ph[27], ph29 = ph[28], ph30 = ph[29],
                 ph31 = ph[30], ph32 = ph[31], ph33 = ph[32], ph34 = ph[33],
                 ph35 = ph[34], ph36 = ph[35], ph37 = ph[36], ph38 = ph[37],
                 ph39 = ph[38], ph40 = ph[39], ph41 = ph[40], ph42 = ph[41],
                 ph43 = ph[42], ph44 = ph[43], ph45 = ph[44], ph46 = ph[45],
                 ph47 = ph[46];
    const double ph16a = 0.5 * ph16;
    const double ph16b = 0.5 * ph16;
    const double ph33a = (1.0 / 6.0) * ph33;   // H2+ + O
    const double ph33b = (1.0 / 6.0) * ph33;   // H+  + OH
    const double ph33c = (1.0 / 6.0) * ph33;   // O+  + H2
    const double ph33d = 0.5 * ph33;           // OH+ + H
    const double ph34a = (1.0 / 3.3) * ph34;   // H+   + H2O
    const double ph34b = (1.0 / 3.3) * ph34;   // H2+  + OH
    const double ph34c = (1.0 / 11.0) * ph34;  // H2O+ + H
    const double ph34d = (1.0 / 3.3) * ph34;   // OH+  + H2

    // --- cosmic rays (.F:613-622), after calc_photo ---
    const double zeta = c.cosmic_ray_ion_rate;
    const double cr1  = T.crtab[0] * zeta;
    const double cr1a = T.crtab[0] * zeta;
    const double cr2  = T.crtab[1] * zeta;
    const double cr3  = T.crtab[2] * zeta;
    const double cr4  = T.crtab[3] * zeta;
    const double cr5  = T.crtab[4] * zeta;
    const double cr6  = T.crtab[5] * zeta;
    const double cr7  = T.crtab[6] * zeta;
    const double cr9  = T.crtab[8] * zeta;

    // --- grain recombination, dust temperature, H2 on dust (.F:624-713) ---
    double h_gr, heii_gr, dii_gr, cii_gr, oii_gr, si2_gr, RH2, Rdust;
    if (c.dust_to_gas_ratio > 0.0) {
        double phi = (abe == 0.0) ? 1e20 : c.G_dust * std::sqrt(temp) / (yn * abe);
        if (phi < 1e-6) {
            h_gr = 1.225e-13 * c.dust_to_gas_ratio;
        } else {
            double hgrvar1 = 8.074e-6 * std::pow(phi, 1.378);
            double hgrvar2 = 1.0 + ch34 * std::pow(phi, ch35);
            h_gr = 1.225e-13 * c.dust_to_gas_ratio / (1.0 + hgrvar1 * hgrvar2);
        }
        if (phi < 1e-6) {
            heii_gr = 5.572e-14 * c.dust_to_gas_ratio;
        } else {
            double hegrvar1 = 3.185e-7 * std::pow(phi, 1.512);
            double hegrvar2 = 1.0 + ch135 * std::pow(phi, ch136);
            heii_gr = 5.572e-14 * c.dust_to_gas_ratio / (1.0 + hegrvar1 * hegrvar2);
        }
        dii_gr = h_gr / std::sqrt(2.0);
        oii_gr = h_gr / 4.0;
        if (phi < 1e-6) {
            cii_gr = 4.558e-13 * c.dust_to_gas_ratio;
        } else {
            double c2grvar1 = 6.089e-3 * std::pow(phi, 1.128);
            double c2grvar2 = 1.0 + ch36 * std::pow(phi, ch37);
            cii_gr = 4.558e-13 * c.dust_to_gas_ratio / (1.0 + c2grvar1 * c2grvar2);
        }
        if (phi < 1e-6) {
            si2_gr = 2.166e-14 * c.dust_to_gas_ratio;
        } else {
            double si2grvar1 = 5.678e-8 * std::pow(phi, 1.874);
            double si2grvar2 = 1.0 + ch38 * std::pow(phi, ch39);
            si2_gr = 2.166e-14 * c.dust_to_gas_ratio / (1.0 + si2grvar1 * si2grvar2);
        }
        cxx_calc_dust_temp(yn, c.chi_isrf, temp, abHI, &RH2, &Rdust, 0, c);
        double temp2  = temp * 1e-2;
        double tdust2 = c.tdust * 1e-2;
        double fa = 1.0 / (1.0 + 1e4 * std::exp(-6e2 / c.tdust));
#ifdef CHEMCOOL_H2FORM_K23
        (void)fa; (void)tdust2;
        ch7 = ch7 * 2.0 / (1.0 + 0.4 * std::sqrt(temp2) + 0.2 * temp2 + 0.08 * temp2 * temp2);
#else
        ch7 = ch7 * fa / (1.0 + 0.4 * std::sqrt(temp2 + tdust2) + 0.2 * temp2
                          + 0.08 * temp2 * temp2);
#endif
    } else {
        h_gr = 0.0; heii_gr = 0.0; dii_gr = 0.0; cii_gr = 0.0; oii_gr = 0.0;
        si2_gr = 0.0; ch7 = 0.0; RH2 = 0.0; Rdust = 0.0;
    }
    (void)Rdust;

    // optically thin to ionizing radiation (.F:718-720)
    if (2.0 * NH * abHI < 1.6e17) ch18 = 0.0;

    // branching ratios (.F:726-757)
    const double ch71a = 0.013 * ch71;   // O   + H2 + H
    const double ch71b = 0.597 * ch71;   // OH  + H  + H
    const double ch71c = 0.14 * ch71;    // OH  + H2
    const double ch71d = 0.25 * ch71;    // H2O + H
    const double ch73b = 0.09 * ch73;    // O  + H2
    const double ch73c = 0.20 * ch73;    // OH + H
    const double ch87a = 0.92 * ch87;    // CO + H
    const double ch87b = 0.08 * ch87;    // OH + C
    struct { const double* p; double operator[](int k) const { return p[k - 1]; } }
        cst = {T.cst};                   // 1-based: cst[k] == cst(k)
    const double cst64a = 0.376 * cst[64];

    // --- equilibrium species, fixed-point iteration (.F:765-1068) ---
    double abhm_old = 0.0, abh2p_old = 0.0, abh3p_old = 0.0, abchp_old = 0.0,
           abch2p_old = 0.0, abohp_old = 0.0, abh2op_old = 0.0, abh3op_old = 0.0,
           abcop_old = 0.0, absihp_old = 0.0, abhocp_old = 0.0, abom_old = 0.0,
           abcm_old = 0.0, abo2p_old = 0.0;
    double abhm = 0.0, abh2p = 0.0, abh3p = 0.0, abchp = 0.0, abch2p = 0.0,
           abohp = 0.0, abh2op = 0.0, abh3op = 0.0, abcop = 0.0, absihp = 0.0,
           abhocp = 0.0, abom = 0.0, abcm = 0.0, abo2p = 0.0;
    double num, denom;
#define EQB(var, mx, id) do { int ab_ = comp_equilib3(num, denom, (mx), (id), &(var)); \
                                if (ab_) { c.fatal_code = ab_; return CHEM_RHS_FATAL; } } while (0)

    for (int I = 1; I <= niter; I++) {
        // H-
        num   = ch21 * abHI * abe;
        denom = ch22 * abHI + ch26 * abe + ch27 * abHI + ch28 * abhp
              + ch29 * abhp + (ph4 / yn) + ch8 * abhep + cst[74] * aboh
              + cst[76] * abcI + cst[22] * aboI;
        EQB(abhm, 1.0, 1);
        abe  = fmax0(abe  - (abhm - abhm_old));
        abHI = fmax0(abHI - (abhm - abhm_old));

        // H2+
        num   = ch23 * abHI * abhp + ch24 * abh2 * abhp
              + ch29 * abhp * abhm + (cr2 * abh2 / yn)
              + ch107 * abHI * abh3p + (ph37 * abh3p / yn)
              + (ph33a * abh2op / yn) + (ph34b * abh3op / yn)
              + cst[75] * abhep * abh2 + (ph41 * abh2 / yn);
        denom = cst[1] * abHI + ch30 * abe + (ph5 / yn)
              + cst[13] * aboI + cst[77] * abcI
              + ch105 * abh2;
        EQB(abh2p, 0.5, 2);
        abe  = fmax0(abe  + (abh2p - abh2p_old));
        abHI = fmax0(abHI - 2.0 * (abh2p - abh2p_old));

        // H3+
        num   = ch105 * abh2 * abh2p + cst[51] * abh2 * abhp;
        denom = ch106 * abe + ch107 * abHI + ((ph36 + ph37) / yn)
              + cst[14] * aboI + cst[55] * abcI
              + cst[56] * aboh + cst[57] * abh2o
              + cst[58] * abco + cst[59] * abco
              + cst[60] * absiI;
        EQB(abh3p, (1.0 / 3.0), 3);
        abe  = fmax0(abe  + (abh3p - abh3p_old));
        abHI = fmax0(abHI - 3.0 * (abh3p - abh3p_old));

        if (abundc == 0.0) {
            abchp = 0.0; abch2p = 0.0; abcm = 0.0;
        } else {
            // CH+
            num = ch104 * abcp * abHI + ch60 * abcp * abh2
                + ch62 * abch2p * abHI + cst[42] * abhcop * abcI
                + (ph15 / yn) * abch2p + (ph16b / yn) * abch3p
                + (ph19 / yn) * abch + cst[55] * abcI * abh3p
                + cst[62] * abhp * abch2 + cst[65] * abch * abhp
                + cst[77] * abh2p * abcI;
            denom = cst[7] * abHI + cst[8] * abh2 + ch61 * abe
                  + cst[41] * aboI + (ph14 / yn);
            EQB(abchp, abundc, 4);
            abe  = fmax0(abe  + (abchp - abchp_old));
            abHI = fmax0(abHI - (abchp - abchp_old));
            abcI = fmax0(abcI - (abchp - abchp_old));

            // CH2+
            num = ch59 * abcp * abh2 + cst[8] * abchp * abh2
                + ch66 * abch3p * abHI + (ph16a / yn) * abch3p
                + (ph21 / yn) * abch2 + cst[66] * abch2 * abhp;
            denom = cst[9] * abh2 + ch62 * abHI + ch63 * abe
                  + ch64 * abe + ch65 * abe + (ph15 / yn)
                  + cst[53] * aboI + cst[67] * abo2;
            EQB(abch2p, abundc, 5);
            abe  = fmax0(abe  + (abch2p - abch2p_old));
            abHI = fmax0(abHI - 2.0 * (abch2p - abch2p_old));
            abcI = fmax0(abcI - (abch2p - abch2p_old));

            // C-
            num = cst[47] * abcI * abe;
            denom = cst[48] * aboI + ch88 * abhp + cst[49] * abHI
                  + (ph29 / yn) + cst[61] * abh2;
            EQB(abcm, abundc, 6);
            abe  = fmax0(abe  - (abcm - abcm_old));
            abcI = fmax0(abcI - (abcm - abcm_old));
        }

        if (abundo == 0.0) {
            abohp = 0.0; abh2op = 0.0; abh3op = 0.0; abom = 0.0; abo2p = 0.0;
        } else {
            // OH+
            num = cst[13] * aboI * abh2p + cst[14] * aboI * abh3p
                + cst[15] * abop * abh2 + cst[25] * aboh * abhp
                + cst[29] * abh2o * abhep + (ph23 / yn) * aboh
                + (ph33d / yn) * abh2op + (ph34d / yn) * abh3op;
            denom = cst[16] * abh2 + ch72 * abe + (ph22 / yn);
            EQB(abohp, abundo, 7);
            abe  = fmax0(abe  + (abohp - abohp_old));
            abHI = fmax0(abHI - (abohp - abohp_old));
            aboI = fmax0(aboI - (abohp - abohp_old));

            // H2O+
            num = cst[16] * abohp * abh2 + cst[27] * abh2o * abhp
                + cst[30] * abh2o * abhep + (ph24 / yn) * abh2o
                + (ph34c / yn) * abh3op + cst[56] * aboh * abh3p;
            denom = cst[17] * abh2 + ch73 * abe + (ph33 / yn);
            EQB(abh2op, abundo, 8);
            abe  = fmax0(abe  + (abh2op - abh2op_old));
            abHI = fmax0(abHI - 2.0 * (abh2op - abh2op_old));
            aboI = fmax0(aboI - (abh2op - abh2op_old));

            // H3O+
            num = cst[17] * abh2op * abh2 + cst[44] * abhcop * abh2o
                + cst[57] * abh3p * abh2o + cst[70] * abh2o * absihp;
            denom = ch71 * abe + cst[21] * absiI + (ph34 / yn)
                  + cst[78] * abcI;
            EQB(abh3op, abundo, 9);
            abe  = fmax0(abe  + (abh3op - abh3op_old));
            abHI = fmax0(abHI - 3.0 * (abh3op - abh3op_old));
            aboI = fmax0(aboI - (abh3op - abh3op_old));

            // O-
            num = cst[23] * aboI * abe;
            denom = cst[24] * abHI + cst[50] * abcI + (ph30 / yn)
                  + cst[63] * abh2 + ch88 * abhp;
            EQB(abom, abundo, 10);
            abe  = fmax0(abe  - (abom - abom_old));
            aboI = fmax0(aboI - (abom - abom_old));

            // O2+
            num = cst[35] * abo2 * abhp + cst[36] * abo2 * abhep
                + (ph28 / yn) * abo2;
            denom = cst[38] * abcI + cst[39] * abcI + ch84 * abe;
            EQB(abo2p, 0.5 * abundo, 11);
            abe  = fmax0(abe  + (abo2p - abo2p_old));
            aboI = fmax0(aboI - 2.0 * (abo2p - abo2p_old));
        }

        if (abundc == 0.0 || abundo == 0.0) {
            abcop = 0.0; abhocp = 0.0;
        } else {
            const double mco = (abundc < abundo) ? abundc : abundo;
            // CO+
            num = cst[3] * abcp * aboh + cst[31] * abc2 * abop
                + cst[33] * abo2 * abcp + cst[38] * abo2p * abcI
                + cst[41] * abchp * aboI + ch112 * abcI * abop
                * ab3M * yn + ch112 * abcp * aboI * ab3M * yn
                + ch124 * abcp * aboI + (cr9 / yn) * abco;
            denom = cst[40] * abHI + ch119 * abe + cst[80] * abh2;
            EQB(abcop, mco, 12);
            abe  = fmax0(abe  + (abcop - abcop_old));
            abcI = fmax0(abcI - (abcop - abcop_old));
            aboI = fmax0(aboI - (abcop - abcop_old));

            // HOC+
            num   = cst[5] * abcp * abh2o + cst[58] * abh3p * abco;
            denom = cst[45] * abco + cst[46] * abh2 + ch56 * abe;
            EQB(abhocp, mco, 13);
            abe  = fmax0(abe  + (abhocp - abhocp_old));
            abHI = fmax0(abHI - (abhocp - abhocp_old));
            abcI = fmax0(abcI - (abhocp - abhocp_old));
            aboI = fmax0(aboI - (abhocp - abhocp_old));
        }

        if (abundsi == 0.0) {
            absihp = 0.0;
        } else {
            // SiH+
            num = ch122 * absip * abHI
                + cst[21] * abh3op * absiI
                + cst[43] * abhcop * absiI
                + cst[60] * abh3p * absiI;
            denom = cst[20] * abHI + ch74 * abe + (ph31 / yn)
                  + (ph32 / yn) + cst[70] * abh2o;
            EQB(absihp, abundsi, 14);
            abe   = fmax0(abe   + (absihp - absihp_old));
            abHI  = fmax0(abHI  - (absihp - absihp_old));
            absiI = fmax0(absiI - (absihp - absihp_old));
        }

        // convergence (.F:1034-1051)
        if (I >= 2) {
            if ((std::fabs(abhm_old - abhm)     < eqtol * abhm)   &&
                (std::fabs(abh2p_old - abh2p)   < eqtol * abh2p)  &&
                (std::fabs(abh3p_old - abh3p)   < eqtol * abh3p)  &&
                (std::fabs(abchp_old - abchp)   < eqtol * abchp)  &&
                (std::fabs(abch2p_old - abch2p) < eqtol * abch2p) &&
                (std::fabs(abohp_old - abohp)   < eqtol * abohp)  &&
                (std::fabs(abh2op_old - abh2op) < eqtol * abh2op) &&
                (std::fabs(abh3op_old - abh3op) < eqtol * abh3op) &&
                (std::fabs(abcop_old - abcop)   < eqtol * abcop)  &&
                (std::fabs(absihp_old - absihp) < eqtol * absihp) &&
                (std::fabs(abhocp_old - abhocp) < eqtol * abhocp) &&
                (std::fabs(abom_old - abom)     < eqtol * abom)   &&
                (std::fabs(abcm_old - abcm)     < eqtol * abcm)   &&
                (std::fabs(abo2p_old - abo2p)   < eqtol * abo2p))
                break;
        }
        abhm_old = abhm; abh2p_old = abh2p; abh3p_old = abh3p; abchp_old = abchp;
        abch2p_old = abch2p; abohp_old = abohp; abh2op_old = abh2op;
        abh3op_old = abh3op; abcop_old = abcop; absihp_old = absihp;
        abhocp_old = abhocp; abom_old = abom; abcm_old = abcm; abo2p_old = abo2p;
    }
#undef EQB

    // --- protons (.F:1072-1114) ---
    ydot[ihp] = (ch11 * abe   * abHI
             + cst[1] * abh2p * abHI
             -   ch13 * abhp  * abe
             -   ch23 * abHI  * abhp
             -   ch24 * abh2  * abhp
             -   ch28 * abhp  * abhm
             -   ch29 * abhp  * abhm
             -   h_gr * abhp  * 1.0
             +   ch46 * abop  * abHI
             -   ch47 * aboI  * abhp
             -   ch48 * abcI  * abhp
             -   ch49 * absiI * abhp
             -   ch98 * abhp  * abDI
             +   ch99 * abdp  * abHI
             +  ch100 * abdp  * abh2
             -  ch101 * abhd  * abhp
             - cst[25] * aboh  * abhp
             - cst[27] * abh2o * abhp
             + cst[28] * abh2o * abhep
             - cst[35] * abo2  * abhp
             + cst[40] * abcop * abHI
             - cst[51] * abh2  * abhp
             - cst[62] * abch2 * abhp
             - cst[65] * abch  * abhp
             - cst[66] * abch2 * abhp
             -   ch88 * abcm  * abhp
             +   ch58 * abcp  * abHI
             +   ch18 * abhep * abe
             +   ch32 * abhep * abHI
             -   ch33 * abheI * abhp
             +   ch95 * absipp * abHI
             -   ch96 * absip * abhp
             -   ch88 * abom  * abhp
             +   ch9  * abhep * abh2) * yn
             +   ph5  * abh2p
             +   cr1  * abHI
             +   ph14 * abchp
             +   ph22 * abohp
             +   ph32 * absihp
             +  ph33b * abh2op
             +  ph34a * abh3op
             +  ph36  * abh3p
             +  ph38  * ((abHI_ph < 0.0) ? abHI_ph : abHI);

    // --- molecular hydrogen (.F:1118-1185) ---
    ydot[ih2] = (ch108 * chem_fpowi(abHI, 3)
              + ch109 * chem_fpowi(abHI, 2) * abh2
              + ch110 * chem_fpowi(abHI, 2) * abheI) * ynsq
              + (ch22   * abHI   * abhm
              + cst[1]  * abHI   * abh2p
              -  ch24   * abhp   * abh2
              -  ch25   * abe    * abh2
              -  ykdh   * abHI   * abh2
              -  ykdh2  * abh2   * abh2
              +  ch7    * abHI   * 1.0
              - ch100   * abh2   * abdp
              + ch101   * abhd   * abhp
              - ch102   * abh2   * abDI
              + ch103   * abhd   * abHI
              - ch52    * aboI   * abh2
              + ch53    * aboh   * abHI
              - ch54    * aboh   * abh2
              + ch55    * abh2o  * abHI
              - ch59    * abcp   * abh2
              - ch60    * abcp   * abh2
              + ch62    * abch2p * abHI
              + ch65    * abch2p * abe
              + ch66    * abch3p * abHI
              + ch68    * abch3p * abe
              - ch70    * abch   * abh2
              + ch71a   * abh3op * abe
              + ch71c   * abh3op * abe
              + ch73b   * abh2op * abe
              - ch78    * abcI   * abh2
              - ch86    * abo2   * abh2
              - ch105   * abh2p  * abh2
              + ch106a  * abh3p  * abe
              + ch107   * abh3p  * abHI
              + cst[7]  * abchp  * abHI
              - cst[8]  * abchp  * abh2
              - cst[9]  * abch2p * abh2
              + ch80    * abch   * abHI
              + cst[11] * abch2  * abHI
              + cst[14] * aboI   * abh3p
              - cst[15] * abop   * abh2
              - cst[16] * abohp  * abh2
              - cst[17] * abh2op * abh2
              - cst[51] * abh2   * abhp
              + cst[20] * absihp * abHI
              - cst[54] * abcI   * abh2
              + cst[55] * abcI   * abh3p
              + cst[56] * aboh   * abh3p
              + cst[57] * abh2o  * abh3p
              + cst[58] * abco   * abh3p
              + cst[59] * abco   * abh3p
              + cst[60] * absiI  * abh3p
              - cst[61] * abcm   * abh2
              - cst[63] * abom   * abh2
              + cst64a  * abch2  * aboI
              + cst[68] * abhep  * abch2
              + cst[69] * abch3p * aboI
              + cst[62] * abhp   * abch2
              + cst[78] * abh3op * abcI
              - cst[75] * abhep  * abh2
              - cst[80] * abcop  * abh2
              - ch9     * abhep  * abh2) * yn
              -  ykdis  * abh2
              -  cr2    * abh2
              +  ph33c  * abh2op
              +  ph34d  * abh3op
              +  ph16b  * abch3p
              +  ph36   * abh3p
              -  ph41   * abh2;

    // --- D+ (.F:1189-1203) ---
    if (abundD == 0.0) {
        ydot[idp] = 0.0;
    } else {
        ydot[idp] = (ch98  * abhp  * abDI
                  - ch99  * abdp  * abHI
                  - ch100 * abdp  * abh2
                  + ch101 * abhd  * abhp
                  - ch13  * abdp  * abe
                  - dii_gr * abdp * 1.0
                  + ch11  * abDI  * abe
                  + ch32  * abhep * abDI
                  - ch33  * abheI * abdp) * yn
                  + cr1a  * abDI
                  + ph38  * abDI;
    }

    // --- HD (.F:1211-1220) ---
    if (abundD == 0.0) {
        ydot[ihd] = 0.0;
    } else {
        ydot[ihd] = (ch100 * abh2 * abdp
                  - ch101 * abhd * abhp
                  + ch102 * abh2 * abDI
                  - ch103 * abhd * abHI) * yn
                  - ph35  * abhd
                  - ph41  * abhd;
    }

    // --- He+ (.F:1224-1255) ---
    if (abhe == 0.0) {
        ydot[ihep] = 0.0;
    } else {
        ydot[ihep] = (-ch8    * abhm  * abhep
                   -  ch9     * abh2  * abhep
                   +  ch14    * abheI * abe
                   -  ch15    * abhep * abe
                   -  heii_gr * abhep * 1.0
                   -  ch17    * abhep * abe
                   +  ch20    * abe   * abhepp
                   -  ch32    * abhep * abHI
                   -  ch32    * abhep * abDI
                   +  ch33    * abheI * abhp
                   +  ch33    * abheI * abdp
                   -  ch92    * abco  * abhep
                   -  ch93    * abco  * abhep
                   -  ch94    * abcI  * abhep
                   -  ch97    * aboI  * abhep
                   -  cst[26] * aboh  * abhep
                   -  cst[28] * abh2o * abhep
                   -  cst[29] * abh2o * abhep
                   -  cst[30] * abh2o * abhep
                   -  cst[32] * abc2  * abhep
                   -  cst[36] * abo2  * abhep
                   -  cst[37] * abo2  * abhep
                   -  cst[52] * absiI * abhep
                   -  cst[75] * abh2  * abhep
                   -  cst[68] * abhep * abch2) * yn
                   +  cr6     * abheI
                   +  ph39    * abheI
                   -  ph40    * abhep;
    }

    // --- He++ (.F:1263-1269) ---
    if (abhe == 0.0) {
        ydot[ihepp] = 0.0;
    } else {
        ydot[ihepp] = (ch15 * abhep  * abe
                    -  ch20 * abhepp * abe) * yn
                    +  ph40 * abhep;
    }

    // --- C+ (.F:1273-1297) ---
    ydot[ic] = (-ch104   * abcp  * abHI
             -  ch59    * abcp  * abh2
             -  ch60    * abcp  * abh2
             +  cst[7]  * abchp * abHI
             -  ch58    * abcp  * abHI
             -  ch40    * abcp  * abe
             +  ch43    * abcI  * abe
             +  ch48    * abcI  * abhp
             -  cii_gr  * abcp  * 1.0
             -  cst[2]  * abcp  * absiI
             -  cst[3]  * abcp  * aboh
             -  cst[4]  * abcp  * abh2o
             -  cst[5]  * abcp  * abh2o
             +  cst[32] * abc2  * abhep
             -  cst[33] * abo2  * abcp
             -  cst[34] * abo2  * abcp
             +  cst[39] * abo2p * abcI
             +  cst[68] * abhep * abch2
             +  ch92    * abco  * abhep
             +  ch94    * abcI  * abhep
             -  ch124   * abcp  * aboI) * yn
             -  ch112   * abcp  * aboI * ab3M * ynsq
             +  ph2     * abcI
             +  ph14    * abchp
             +  cr3     * abcI;

    // --- O+ (.F:1301-1316) ---
    ydot[io] = (ch45    * aboI * abe
             +  ch47    * aboI * abhp
             -  ch46    * abop * abHI
             -  ch42    * abop * abe
             -  oii_gr  * abop * 1.0
             -  cst[15] * abop * abh2
             +  cst[26] * aboh * abhep
             -  cst[31] * abc2 * abop
             +  cst[34] * abo2 * abcp
             +  cst[37] * abo2 * abhep
             +  ch93    * abco * abhep
             +  ch97    * aboI * abhep) * yn
             -  ch112   * abop * abcI * ab3M * ynsq
             +  cr4     * aboI
             +  ph33c   * abh2op
             +  ph46    * aboI;

    // --- OH (.F:1320-1355) ---
    ydot[ioh] = (ch71b   * abh3op * abe
              +  ch71c   * abh3op * abe
              +  ch73c   * abh2op * abe
              +  cst[22] * aboI   * abhm
              +  cst[24] * abom   * abHI
              +  ch50    * aboI   * abHI
              -  ch51    * aboh   * abHI
              +  ch52    * aboI   * abh2
              -  ch53    * aboh   * abHI
              -  ch54    * aboh   * abh2
              +  ch55    * abh2o  * abHI
              -  cst[71] * abcI   * aboh
              -  cst[3]  * abcp   * aboh
              -  cst[25] * aboh   * abhp
              -  cst[26] * aboh   * abhep
              +  cst[28] * abh2o  * abhep
              -  ch81    * aboI   * aboh
              +  ch85    * abo2   * abHI
              +  2.0 * ch86 * abo2 * abh2
              +  ch87b   * abhcop * abe
              +  ch89    * abco   * abHI
              -  cst[56] * aboh   * abh3p
              +  cst[67] * abch2p * abo2
              -  cst[74] * abhm   * aboh
              -  2.0 * ch118 * aboh * aboh
              -  ch131   * aboh   * abHI
              -  ch132   * aboh   * abh2
              +  ch133   * abh2o  * abHI
              +  ch134   * abh2o  * abh2) * yn
              +  ch113   * aboI   * abHI * ab3M * ynsq
              -  ch114   * aboh   * abHI * ab3M * ynsq
              -  ph11    * aboh
              -  ph23    * aboh
              +  ph12    * abh2o
              +  ph33b   * abh2op
              +  ph34b   * abh3op;

    // --- H2O (.F:1359-1381) ---
    ydot[ih2o] = (ch71d  * abh3op * abe
               - cst[4]  * abcp   * abh2o
               - cst[5]  * abcp   * abh2o
               + cst[21] * abh3op * absiI
               + ch51    * aboh   * abHI
               + ch54    * aboh   * abh2
               - ch55    * abh2o  * abHI
               - cst[27] * abh2o  * abhp
               - cst[28] * abh2o  * abhep
               - cst[29] * abh2o  * abhep
               - cst[30] * abh2o  * abhep
               - cst[44] * abhcop * abh2o
               - cst[57] * abh3p  * abh2o
               + cst[63] * abom   * abh2
               - cst[70] * abh2o  * absihp
               + cst[74] * abhm   * aboh
               + ch118   * aboh   * aboh
               - ch133   * abh2o  * abHI
               - ch134   * abh2o  * abh2) * yn
               + ch114   * aboh   * abHI * ab3M * ynsq
               - ph12    * abh2o
               - ph24    * abh2o
               + ph34a   * abh3op;

    // --- CO (.F:1385-1407) ---
    ydot[ico] = (ch57    * abcI   * aboI
              + cst[71] * abcI   * aboh
              + ch77    * abch   * aboI
              + cst[34] * abo2   * abcp
              + ch83    * abo2   * abcI
              + cst[40] * abcop  * abHI
              + cst[42] * abhcop * abcI
              + cst[43] * abhcop * absiI
              + cst[44] * abhcop * abh2o
              + ch87a   * abhcop * abe
              + ch56    * abhocp * abe
              + cst[48] * abcm   * aboI
              + cst[50] * abom   * abcI
              - cst[58] * abh3p  * abco
              - cst[59] * abh3p  * abco
              + cst[64] * abch2  * aboI
              - ch89    * abco   * abHI
              + ch90    * abc2   * aboI
              - ch92    * abco   * abhep
              - ch93    * abco   * abhep) * yn
              + ch111   * abcI   * aboI * ab3M * ynsq
              - ph10    * abco
              - cr9     * abco;

    // --- C2 (.F:1411-1417) ---
    ydot[ic2] = (ch79    * abcI * abcI
              + cst[10] * abcI * abch
              - cst[31] * abc2 * abop
              - cst[32] * abc2 * abhep
              - ch90    * abc2 * aboI) * yn
              + ch115   * abcI * abcI * ab3M * ynsq
              - ph25    * abc2;

    // --- O2 (.F:1421-1435) ---
    ydot[io2] = (ch81    * aboI  * aboh
              + ch82    * aboI  * aboI
              - cst[33] * abo2  * abcp
              - cst[34] * abo2  * abcp
              - ch83    * abo2  * abcI
              - cst[35] * abo2  * abhp
              - cst[36] * abo2  * abhep
              - cst[37] * abo2  * abhep
              + cst[39] * abo2p * abcI
              - cst[67] * abo2  * abch2p
              - ch85    * abo2  * abHI
              - ch86    * abo2  * abh2) * yn
              + ch116   * aboI  * aboI * ab3M * ynsq
              - ph27    * abo2
              - ph28    * abo2;

    // --- HCO+ (.F:1439-1452) ---
    ydot[ihcop] = (cst[4]  * abcp   * abh2o
                + cst[45] * abhocp * abco
                + cst[46] * abhocp * abh2
                + ch117   * abch   * aboI
                + cst[53] * abch2p * aboI
                + cst[59] * abh3p  * abco
                - cst[42] * abcI   * abhcop
                - cst[43] * absiI  * abhcop
                - cst[44] * abh2o  * abhcop
                + cst[67] * abch2p * abo2
                + cst[69] * abch3p * aboI
                + cst[78] * abh3op * abcI
                + cst[80] * abcop  * abh2
                - ch87    * abe    * abhcop) * yn;

    // --- CH (.F:1456-1472) ---
    ydot[ich] = (ch63    * abch2p * abe
              + ch68    * abch3p * abe
              + ch69    * abch3p * abe
              + cst[11] * abch2  * abHI
              + cst[12] * abcI   * abHI
              + ch78    * abcI   * abh2
              + cst[49] * abcm   * abHI
              - ch70    * abh2   * abch
              - ch80    * abHI   * abch
              - ch77    * aboI   * abch
              - cst[10] * abcI   * abch
              - ch117   * aboI   * abch
              - cst[65] * abhp   * abch
              + cst[76] * abhm   * abcI) * yn
              - ph18    * abch
              - ph19    * abch
              + ph20    * abch2;

    // --- CH2 (.F:1476-1486) ---
    ydot[ich2] = (ch67    * abch3p * abe
               + ch70    * abch   * abh2
               + cst[54] * abcI   * abh2
               + cst[61] * abcm   * abh2
               - cst[11] * abHI   * abch2
               - cst[62] * abhp   * abch2
               - cst[64] * aboI   * abch2
               - cst[66] * abhp   * abch2
               - cst[68] * abhep  * abch2) * yn
               - ph20    * abch2
               - ph21    * abch2;

    // --- CH3+ (.F:1490-1496) ---
    ydot[ich3p] = (cst[9]  * abch2p * abh2
                - ch66    * abHI   * abch3p
                - ch67    * abe    * abch3p
                - ch68    * abe    * abch3p
                - ch69    * abe    * abch3p
                - cst[69] * aboI   * abch3p) * yn
                - ph16 * abch3p;

    // --- Si+, Si++ (.F:1498-1529) ---
    if (abundsi > 0.0) {
        ydot[isi] = (-ch41    * absip  * abe
                  +  cst[2]  * abcp   * absiI
                  +  ch49    * absiI  * abhp
                  +  ch44    * absiI  * abe
                  -  si2_gr  * absip  * 1.0
                  -  ch122   * absip  * abHI
                  +  cst[20] * absihp * abHI
                  +  cst[52] * absiI  * abhep
                  +  ch95    * absipp * abHI
                  -  ch96    * absip  * abhp
                  +  ch121   * absipp * abe) * yn
                  +  ph3     * absiI
                  +  cr5     * absiI
                  +  ph31    * absihp
                  -  cr7     * absip
                  -  ph47    * absip;
        ydot[isipp] = (ch96  * absip  * abhp
                    -  ch95  * absipp * abHI
                    -  ch121 * absipp * abe) * yn
                    +  cr7   * absip
                    +  ph47  * absip;
    } else {
        ydot[isi]   = 0.0;
        ydot[isipp] = 0.0;
    }

    if (abundc == 0.0) {
        ydot[ic] = 0.0; ydot[ico] = 0.0; ydot[ic2] = 0.0; ydot[ihcop] = 0.0;
        ydot[ich] = 0.0; ydot[ich2] = 0.0; ydot[ich3p] = 0.0;
    }
    if (abundo == 0.0) {
        ydot[io] = 0.0; ydot[ioh] = 0.0; ydot[ih2o] = 0.0; ydot[ico] = 0.0;
        ydot[ihcop] = 0.0; ydot[io2] = 0.0;
    }

    // --- chemical heating & cooling, rates_chem(1..23) (.F:1554-1687) ---
    double rc[23];
    rc[0] = 4.48 * CHEM_EV * (ykdh  * abHI * abh2 +
                              ykdh2 * abh2 * abh2 +
                              ch25  * abe  * abh2) * ynsq;
    rc[1] = 1.83 * CHEM_EV * ch24 * abh2 * abhp * ynsq;
    rc[2] = -(4e-1 * CHEM_EV * abh2 * yn * ykdis);
    double cdex = ((double)1.4f * std::exp(-1.81e4 / (temp + 1200)) * abh2
                 + 1.0 * std::exp(-1e3 / temp) * abHI) * 1e-12 * std::sqrt(temp) * yn;
    double rdex = 2e-7;
    double fpheat = cdex / (cdex + rdex);
#ifdef CONST_NCRIT_UVPUMP
    fpheat = 1.0 / (1.0 + 1e4 / yn);
#endif
    rc[3] = -(2.0 * CHEM_EV * abh2 * yn * ykdis * 8.5 * fpheat);
    double f_ex;
    if (RH2 > 0.0) {
        f_ex = cxx_compute_h2_heating_fraction(temp, yn, abh2);
        f_ex = f_ex * c.h2_form_ex;
    } else {
        f_ex = 0.0;
    }
    double f_tot = c.h2_form_kin + f_ex;
    rc[4]  = -(4.48 * CHEM_EV * f_tot * RH2);
    rc[5]  = -(3.53 * CHEM_EV * ch22 * abHI * abhm * ynsq * (1.0 - h2var0));
    rc[6]  = -(1.83 * CHEM_EV * cst[1] * abHI * abh2p * ynsq * (1.0 - h2var0));
    rc[7]  = -(4.48 * CHEM_EV * abHI * abHI * (ch108 * abHI
               + ch109 * abh2 + ch110 * abheI) * yncb);
    rc[8]  = 13.6 * CHEM_EV * ch11 * abe * abHI * ynsq;
    rc[9]  = 24.6 * CHEM_EV * ch14 * abe * abheI * ynsq;
    rc[10] = 0.0;
    rc[11] = CHEM_KBOLTZ * temp * abhp * (ch13 * abe + h_gr) * ynsq;
    rc[12] = 40.8 * CHEM_EV * ch10 * abhep * abe * ynsq;
    rc[13] = CHEM_KBOLTZ * temp * ((ch17 - ch10) * abe + heii_gr) * abhep * ynsq;
    rc[14] = CHEM_KBOLTZ * temp * abhepp * ch20 * abe * ynsq;
    rc[15] = -(ph6 * abcI * yn);
    rc[16] = -(ph7 * absiI * yn);
    rc[17] = -(ph8 * abhm * yn);
    rc[18] = -(ph9 * abh2p * yn);
    rc[19] = -(ph42 * (abHI + abDI) * yn);
    rc[20] = -(ph43 * abheI * yn);
    rc[21] = -(ph44 * abhepp * yn);
    rc[22] = -(ph45 * (abh2 + abhd) * yn);

    double ylam_chem = 0.0;
    for (int i = 0; i < 23; i++) {
        ylam_chem = ylam_chem + rc[i];
        c.lambda_chem[i] = rc[i];
    }

    // --- cool_func abundance vector (.F:1691-1710) ---
    double abundances[16];
    abundances[0]  = abh2;
    abundances[1]  = aboI;
    abundances[2]  = aboh;
    abundances[3]  = abh2o;
    abundances[4]  = abco;
    abundances[5]  = abcI;
    abundances[6]  = abcp;
    abundances[7]  = absiI;
    abundances[8]  = absip;
    abundances[9]  = abe;
    abundances[10] = abhp;
    abundances[11] = abHI;
    abundances[12] = abhd;
    abundances[13] = abheI;
    abundances[14] = abhep;
    abundances[15] = abhepp;

    // --- local Jeans length (.F:1722-1725) ---
    double rho_lj  = (1.0 + 4.0 * abhe) * CHEM_PROTONMASS * yn;
    double cs2_lj  = gamma * (gamma - 1.0) * energy / rho_lj;
    double L_jeans = std::sqrt(CHEM_PI * cs2_lj / (CHEM_GRAVITY * rho_lj));

    double ylam, rates[CHEM_NRATES];
    int cferr = cxx_cool_func(temp, yn, L_jeans, divv, c.G_dust, abundances,
                              &ylam, rates, 0, c);
    if (cferr) return CHEM_RHS_RECOVERABLE;

    double ylam_cmb = 0.0;
    if (c.redshift > 0.0) {
        double CMB_temp = 2.726 * (1.0 + c.redshift);
        double rates_cmb[CHEM_NRATES];
        int e2 = cxx_cool_func(CMB_temp, yn, L_jeans, divv, c.G_dust, abundances,
                               &ylam_cmb, rates_cmb, 1, c);
        if (e2) return CHEM_RHS_RECOVERABLE;
    }

    ydot[itmp] = -(ylam - ylam_cmb + ylam_chem);

    for (int i = 0; i < nspec; i++) {
        if (ydot[i] != ydot[i]) {
            std::fprintf(stderr, "chem_rhs_net3: NaN ydot[%d] (ID %ld, T=%g yn=%g)\n",
                         i, c.id_current, temp, yn);
            for (int j = 0; j < nspec; j++) ydot[j] = 0.0;
            return CHEM_RHS_RECOVERABLE;
        }
    }
    return flagged ? CHEM_RHS_RECOVERABLE : CHEM_RHS_OK;
}
