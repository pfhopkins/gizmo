// chem_tables_build.cc -- Gate-3 port of the CHEMCOOL table layer:
//
//   coolinmo.F              -> cltab / dtcltab, the CO+H2O spline tables,
//                              the cl6 atomic-cooling spline, HeI/HeII tables
//   cheminmo.F              -> chtab / dtchtab, crtab, crphot, the CO/H2
//                              shielding curves; calls photoinit + const_rates
//   photoinit_ism.F         -> phtab, f_rsc
//   const_rates.F           -> cst
//   wss_z_collis.F (reader) -> the 9 WSS CIE metal tables (SAME data file)
//   cool_util.F load_H2_table        -> h2_opac* (SAME data file)
//   calc_temp.F init_temperature_lookup -> temp/gamma/eh2 (SAME data files)
//
// The 346 analytic rate coefficients themselves are NOT re-typed here: they
// are the machine-extracted rates_gen.cc functions from Gate 1, which are
// golden-tested 346/346 at 1e-12 against the Fortran-built tables.  This file
// only reproduces the *table assembly* -- the temperature grid, the spline
// chains, the residual/zeroing loops and the finite-difference derivatives.
//
// EXCEPTION: cl52, cl59, cl60, cl61 have no rates_gen entry.  RATE_AUDIT.md
// classified them kind=TABLE (they appear in coolinmo's residual zeroing loop)
// so the generator skipped them, and DEVIATIONS.md D-9 then established that in
// the production config (iflag_atom=2 + WSS_CIE_COOL) they are NOT zeroed and
// keep their ANALYTIC main-loop values.  D-9's parenthetical claim that they
// are "covered by the Gate-1 rate goldens" was wrong -- they were skipped by
// the generator, so nothing covered them.  They are written out by hand below
// from coolinmo.F:1028-1071 and are covered by the Gate-3 table golden.

#include "../../../GIZMO_config.h"
#include "chem_types.h"
#include "chem_tables_data.h"
#include "rates_gen.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// from chem_spline.cc (spline.F port)
void chem_spline_eval(int nval, const double* positions, const double* values,
                      int nnew, const double* new_positions, double* new_values);
void chem_build_cl_tables(const double* temptab, int nmd,
                          double* cl1, double* cl2, double* cl3, double* cl6,
                          int iflag_atom);

// registry of the machine-generated rate functions
struct RateEntry { const char* name; const char* kind; int idx;
                   double (*fn)(double, const ChemGlobals&); int flags; };
static const RateEntry chem_rate_registry[] = {
#include "rates_registry.inc"
};
static const int chem_rate_registry_n =
    (int)(sizeof(chem_rate_registry) / sizeof(chem_rate_registry[0]));
#if CHEMISTRYNETWORK == 16
// entries whose CHEMISTRYNETWORK == 16 branch differs or is net-16 only
static const RateEntry chem_rate_registry16[] = {
#include "rates_registry16.inc"
};
static const int chem_rate_registry16_n =
    (int)(sizeof(chem_rate_registry16) / sizeof(chem_rate_registry16[0]));
#endif

// ---------------------------------------------------------------------------
// owned storage
// ---------------------------------------------------------------------------
struct ChemTableStore {
    std::vector<double> temptab, cltab, dtcltab, chtab, dtchtab;
    std::vector<double> crtab, crphot, phtab, cst;
    std::vector<double> co_temptab, co_colntab, co_L0, dTco_L0;
    std::vector<double> co_lte, co_n05, co_alp, dTco_lte, dTco_n05, dTco_alp;
    std::vector<double> co_vib_temptab, co_vib_colntab, co_vib_LTE, dTco_vib_LTE;
    std::vector<double> h2o_temptab, h2o_colntab;
    std::vector<double> h2o_L0_o, dTh2o_L0_o, h2o_L0_p, dTh2o_L0_p;
    std::vector<double> h2o_LTE_o, h2o_n05_o, h2o_alp_o;
    std::vector<double> h2o_LTE_p, h2o_n05_p, h2o_alp_p;
    std::vector<double> dTh2o_LTE_o, dTh2o_n05_o, dTh2o_alp_o;
    std::vector<double> dTh2o_LTE_p, dTh2o_n05_p, dTh2o_alp_p;
    std::vector<double> h2o_vib_temptab, h2o_vib_colntab;
    std::vector<double> h2o_vib_LTE, dTh2o_vib_LTE;
    std::vector<double> h2_opac_temp, h2_opac_column, h2_opac;
    std::vector<double> wss[9];
    std::vector<double> HeI, HeII;
    std::vector<double> CO_ss, CO_H2;
    std::vector<double> chi_mathis, chi_black;
    std::vector<double> temp_table, gamma_table, eh2_table;
    std::vector<double> hng;
};

// Fortran a(i,j) with i fast, leading dimension ld -> writable reference
static inline double& F2(std::vector<double>& v, int ld, int i1, int j1)
{ return v[(size_t)(j1 - 1) * ld + (i1 - 1)]; }

// ---------------------------------------------------------------------------
// 2-D spline chain shared by the CO / H2O tables (coolinmo.F:1184-1249 etc.):
// spline in T for each raw column, then spline in N for each output T.
// ---------------------------------------------------------------------------
static void build_2d_table(int nT_raw, const double* T_raw,
                           int nN_raw, const double* N_raw,
                           const double* data_raw,   // [nN_raw*nT_raw], T fast
                           int nT, const double* Ttab,
                           int nN, const double* Ntab,
                           std::vector<double>& out)   // [nT*nN], N fast
{
    std::vector<double> smalltab((size_t)nN_raw * nT);   // (i_Nraw, j_T)
    std::vector<double> raw(nT_raw), fit(nT);
    for (int i = 1; i <= nN_raw; i++) {
        for (int j = 1; j <= nT_raw; j++) raw[j-1] = data_raw[nT_raw*(i-1) + (j-1)];
        chem_spline_eval(nT_raw, T_raw, raw.data(), nT, Ttab, fit.data());
        for (int j = 1; j <= nT; j++) smalltab[(size_t)(j-1)*nN_raw + (i-1)] = fit[j-1];
    }
    out.assign((size_t)nT * nN, 0.0);
    std::vector<double> fxT(nN_raw), fit2(nN);
    for (int j = 1; j <= nT; j++) {
        for (int i = 1; i <= nN_raw; i++) fxT[i-1] = smalltab[(size_t)(j-1)*nN_raw + (i-1)];
        chem_spline_eval(nN_raw, N_raw, fxT.data(), nN, Ntab, fit2.data());
        for (int i = 1; i <= nN; i++) out[(size_t)(j-1)*nN + (i-1)] = fit2[i-1];
    }
}

// forward difference in T, last row copied (coolinmo.F:1235-1249 pattern)
static void tderiv_2d(const std::vector<double>& v, int nT, int nN,
                      std::vector<double>& d)
{
    d.assign((size_t)nT * nN, 0.0);
    for (int j = 1; j <= nT - 1; j++)
        for (int i = 1; i <= nN; i++)
            d[(size_t)(j-1)*nN + (i-1)] = v[(size_t)j*nN + (i-1)]
                                        - v[(size_t)(j-1)*nN + (i-1)];
    for (int i = 1; i <= nN; i++)
        d[(size_t)(nT-1)*nN + (i-1)] = d[(size_t)(nT-2)*nN + (i-1)];
}

static void tderiv_1d(const std::vector<double>& v, int nT, std::vector<double>& d)
{
    d.assign(nT, 0.0);
    for (int j = 0; j < nT - 1; j++) d[j] = v[j+1] - v[j];
    d[nT-1] = d[nT-2];
}

// ---------------------------------------------------------------------------
// file readers -- the SAME data files the Fortran opened
// ---------------------------------------------------------------------------
static int read_wss(const std::string& dir, ChemTableStore& S)
{
    std::string path = dir + "/wss_z_collis.txt";
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) { std::printf("CHEMCOOL_CXX: cannot open %s\n", path.c_str()); return 1; }
    for (int k = 0; k < 9; k++) S.wss[k].assign(CHEM_NWSS, 0.0);
    for (int I = 0; I < CHEM_NWSS; I++) {
        double v[25];
        for (int k = 0; k < 25; k++)
            if (std::fscanf(f, "%lf", &v[k]) != 1) {
                std::printf("CHEMCOOL_CXX: short read in %s at row %d\n",
                            path.c_str(), I); std::fclose(f); return 1;
            }
        // columns: Temp, (l,n)x7, metal_only, C, N, O, Ne, Mg, Si, S, Ca, Fe
        double Temp = v[0];
        // exp(-(1e4/Temp)**2): 1e4 is a default-REAL literal but 10000.0 is
        // exactly representable, so this is plain double arithmetic.
        double fac = std::exp(-chem_fpowi(1e4 / Temp, 2));
        // `1e-20` in wss_z_collis.F:45-47 is a Fortran DEFAULT-REAL literal
        // (no D exponent) -> float32(1e-20) = 1.00000002004e-20.  Using the
        // exact decimal shifts every FLOORED row's log10 by 1.378e-8, which is
        // precisely the offset the Gate-3 golden measured on 59/352 rows.
        // Same trap as DEVIATIONS.md D-10.
        const double one_em20 = (double)1e-20f;
        if (fac < one_em20) fac = one_em20;
        for (int k = 0; k < 9; k++)
            S.wss[k][I] = std::log10(std::fabs(v[16 + k] * fac));
    }
    std::fclose(f);
    return 0;
}

static int read_h2_opac(const std::string& dir, ChemTableStore& S)
{
    std::string path = dir + "/H2-cooling-ratios.dat";
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) { std::printf("CHEMCOOL_CXX: cannot open %s\n", path.c_str()); return 1; }
    const int n = CHEM_NH2OP;
    S.h2_opac_temp.assign(n, 0.0);
    S.h2_opac_column.assign(n, 0.0);
    S.h2_opac.assign((size_t)n * n, 0.0);
    for (int I = 1; I <= n; I++)
        for (int J = 1; J <= n; J++) {
            double a, b, cval;
            if (std::fscanf(f, "%lf %lf %lf", &a, &b, &cval) != 3) {
                std::printf("CHEMCOOL_CXX: short read in %s\n", path.c_str());
                std::fclose(f); return 1;
            }
            S.h2_opac_temp[I-1] = a;
            S.h2_opac_column[J-1] = b;
            F2(S.h2_opac, n, I, J) = cval;
        }
    std::fclose(f);
    return 0;
}

static int read_temp_gamma(const std::string& dir, ChemTableStore& S)
{
    S.temp_table.assign(CHEM_ISIZE, 0.0);
    S.gamma_table.assign(CHEM_ISIZE, 0.0);
    S.eh2_table.assign(CHEM_ISIZE2, 0.0);

    std::string p1 = dir + "/energy-table-x.txt";
    FILE* f = std::fopen(p1.c_str(), "r");
    if (!f) { std::printf("CHEMCOOL_CXX: cannot open %s\n", p1.c_str()); return 1; }
    for (int I = 0; I < CHEM_ISIZE; I++) {
        double foo;
        if (std::fscanf(f, "%lf %lf %lf", &foo, &S.temp_table[I],
                        &S.gamma_table[I]) != 3) {
            std::printf("CHEMCOOL_CXX: short read in %s at %d\n", p1.c_str(), I);
            std::fclose(f); return 1;
        }
    }
    std::fclose(f);

    std::string p2 = dir + "/eh2-table.txt";
    f = std::fopen(p2.c_str(), "r");
    if (!f) { std::printf("CHEMCOOL_CXX: cannot open %s\n", p2.c_str()); return 1; }
    for (int I = 0; I < CHEM_ISIZE2; I++) {
        double foo;
        if (std::fscanf(f, "%lf %lf", &foo, &S.eh2_table[I]) != 2) {
            std::printf("CHEMCOOL_CXX: short read in %s at %d\n", p2.c_str(), I);
            std::fclose(f); return 1;
        }
    }
    std::fclose(f);
    return 0;
}

// ---------------------------------------------------------------------------
// photoinit_ism.F (phtab, f_rsc)
// ---------------------------------------------------------------------------
static void build_phtab(std::vector<double>& p)
{
    p.assign(CHEM_NPHTAB, 0.0);
    p[0]  = 3.3e-11;    // H2 photodissociation (DB96 table 2, 3:1 o:p)
    p[1]  = 1.765e-10;  // CI ionization
    p[2]  = 2.647e-9;   // SiI ionization
    p[3]  = 4.2e-7;     // H- photodissociation
    p[4]  = 6.5e-10;    // H2+ photodissociation
    p[5]  = 0.0; p[6] = 0.0; p[7] = 0.0; p[8] = 0.0;   // heating rates: unset
    p[9]  = 1.411e-10;  // CO photodissociation (Heays+17)
    p[10] = 2.235e-10;  // OH  -> O + H
    p[11] = 3.47e-10;   // H2O -> OH + H
    p[13] = 1.47e-10;   // CH+ photodissociation
    p[14] = 2.71e-11;   // CH2+ -> CH+ + H
    p[15] = 5.88e-10;   // CH3+ -> CH2+ + H
    p[17] = 5.06e-10;   // CH -> C + H
    p[18] = 4.47e-10;   // CH -> CH+ + e-
    p[19] = 4.235e-10;  // CH2 -> CH + H
    p[20] = 5.88e-10;   // CH2 -> CH2+ + e-
    p[21] = 5.88e-13;   // OH+ -> O + H+
    p[22] = 9.41e-13;   // OH -> OH+ + e-
    p[23] = 1.94e-11;   // H2O -> H2O+ + e-
    p[24] = 8.82e-11;   // C2 -> C + C
    p[26] = 4.06e-10;   // O2 -> O + O
    p[27] = 3.29e-11;   // O2 -> O2+ + e-
    p[28] = 1.41e-7;    // C- photodetachment
    p[29] = 1.41e-7;    // O- photodetachment
    p[30] = 1.21e-9;    // SiH+ -> Si+ + H
    p[31] = 1.279e-10;  // SiH+ -> Si  + H+
    p[32] = 1.765e-10;  // H2O+ photodissociation
    p[33] = 4.85e-11;   // H3O+ photodissociation
    p[34] = 3.3e-11;    // HD photodissociation
    p[35] = 5.88e-13;   // H3+ photodissociation (total)
    // p[37..44] HI/HeI/HeII/H2 ionization + heating: set by external code
    // p[45], p[46] OI / SiI photoionization: zero
    p[47] = 1e-9;       // CHx  [NL99]
    p[48] = 5e-10;      // OHx  [NL99]
    p[49] = 1.5e-10;    // HCO+ [NL99]
    p[50] = 2.0e-10;    // M    [NL99]
    p[51] = 1.353e-10;  // N2 photodissociation [vD88]
    p[52] = 2.53e-10;   // NO photodissociation
    p[53] = 1.53e-10;   // NO photoionization
#if CHEMISTRYNETWORK == 16
    p[1]  = 2.059e-10;  // CI ionization (Heays+17)
    p[47] = 5.35e-10;   // CHx
    p[48] = 2.235e-10;  // OHx
    p[49] = 0.0;        // HCO+
#endif
}

// ---------------------------------------------------------------------------
// cheminmo.F:1922-1948 -- the CO self-shielding / CO-by-H2 shielding curves
// ---------------------------------------------------------------------------
static void build_co_shielding(ChemTableStore& S)
{
    const double NCO_shield_min = 1e12, NH2_shield_min = 1e13;
    S.CO_ss.assign(CHEM_IMAX_COSS, 0.0);
    S.CO_H2.assign(CHEM_IMAX_COH2, 0.0);

    for (int I = 1; I <= CHEM_IMAX_COSS; I++) {
        double NCO = std::log10(NCO_shield_min) + 0.05 * (I - 1);
        int J = 51;
        for (J = 51; J >= 1; J--) if (NCO >= dat_CO_shield_columns[J-1]) break;
        if (J > 50) J = 50;
        // The Fortran DO loop leaves J = 0 if the test never fires; the
        // subsequent code then indexes CO_shield_columns(0).  Unreachable for
        // this grid (NCO starts at 12.0 == column(1)), and we clamp to J = 1.
        if (J < 1) J = 1;
        double dN = (NCO - dat_CO_shield_columns[J-1])
                  / (dat_CO_shield_columns[J] - dat_CO_shield_columns[J-1]);
        S.CO_ss[I-1] = dN * dat_CO_shield_factors[J]
                     + (1.0 - dN) * dat_CO_shield_factors[J-1];
    }
    for (int I = 1; I <= CHEM_IMAX_COH2; I++) {
        double NH2 = std::log10(NH2_shield_min) + 0.05 * (I - 1);
        int J = 43;
        for (J = 43; J >= 1; J--) if (NH2 >= dat_H2_shield_columns[J-1]) break;
        if (J > 42) J = 42;
        if (J < 1) J = 1;
        double dN = (NH2 - dat_H2_shield_columns[J-1])
                  / (dat_H2_shield_columns[J] - dat_H2_shield_columns[J-1]);
        S.CO_H2[I-1] = dN * dat_H2_shield_factors[J]
                     + (1.0 - dN) * dat_H2_shield_factors[J-1];
    }
}

// ---------------------------------------------------------------------------
// coolinmo.F:1028-1071 -- the four cl entries the Gate-1 generator skipped.
// ---------------------------------------------------------------------------
static double cl52_of(double temp)
{
    if (temp < 1e3) return 0.0;
    return 7.5e-19 * std::exp(-1.18348e5 / temp) / (1.0 + std::sqrt(temp / 1e5));
}
static double cl59_of(double temp)
{
    if (temp < 2e3) return 0.0;
    return 1.1e-19 * std::pow(temp, 0.082) * std::exp(-2.3e5 / temp);
}
static double cl60_of(double temp)
{
    if (temp < 1e2) return 0.0;
    return 9.1e-27 * std::exp(-1.3179e4 / temp)
         / (std::pow(temp, 0.1687) * (1.0 + std::sqrt(temp / 1e5)));
}
static double cl61_of(double temp)
{
    if (temp < 4e3) return 0.0;
    return 5.54e-17 * std::exp(-4.73638e5 / temp)
         / (std::pow(temp, 0.397) * (1.0 + std::sqrt(temp / 1e5)));
}

// evaluate one generated rate into its table slot
static void store_rate(ChemTableStore& S, const RateEntry& e, const ChemGlobals& g,
                       int nmd)
{
    if (!e.fn) return;
    if (std::strcmp(e.kind, "cl") == 0) {
        for (int i = 0; i < nmd; i++)
            S.cltab[(size_t)i * CHEM_NCLTAB + (e.idx - 1)] =
                e.fn(S.temptab[i], g);
    } else if (std::strcmp(e.kind, "ch") == 0) {
        for (int i = 0; i < nmd; i++)
            S.chtab[(size_t)i * CHEM_NCHTAB + (e.idx - 1)] =
                e.fn(S.temptab[i], g);
    } else if (std::strcmp(e.kind, "cr") == 0) {
        S.crtab[e.idx - 1] = e.fn(0.0, g);
    } else if (std::strcmp(e.kind, "crphot") == 0) {
        S.crphot[e.idx - 1] = e.fn(0.0, g);
    } else if (std::strcmp(e.kind, "cst") == 0) {
        S.cst[e.idx - 1] = e.fn(0.0, g);
    }
}

// ===========================================================================
// chem_tables_build
// ===========================================================================
int chem_tables_build(ChemTables& T, const ChemBuildConfig& cfg,
                      const char* datadir)
{
    ChemTableStore* S = new ChemTableStore();
    std::string dir(datadir);

    ChemGlobals g;
    g.iflag_mn = cfg.iflag_mn; g.iflag_ad = cfg.iflag_ad;
    g.iflag_atom = cfg.iflag_atom;
    g.iflag_3bh2a = cfg.iflag_3bh2a; g.iflag_3bh2b = cfg.iflag_3bh2b;
    g.iflag_h3pra = cfg.iflag_h3pra;
    g.deff = cfg.deff; g.dust_to_gas_ratio = cfg.dust_to_gas_ratio;
    g.phi_pah = cfg.phi_pah; g.cosmic_ray_ion_rate = cfg.cosmic_ray_ion_rate;
    g.abhe = cfg.abhe;

    // ---- temperature grid (coolinmo.F:209-215) ----
    const int nmd = CHEM_NMD;
    double tmin = 1.0, tmax = 1e8;
    double dtlog = std::log10(tmax) / (nmd - 1);
    g.dtlog = dtlog; g.tmin = tmin; g.tmax = tmax;
    S->temptab.assign(nmd, 0.0);
    for (int i = 0; i < nmd; i++) S->temptab[i] = std::pow(10.0, i * dtlog);

    // ---- cltab / chtab / crtab / crphot / cst from the generated rates ----
    S->cltab.assign((size_t)nmd * CHEM_NCLTAB, 0.0);
    S->chtab.assign((size_t)nmd * CHEM_NCHTAB, 0.0);
    S->crtab.assign(CHEM_NCRTAB, 0.0);
    S->crphot.assign(CHEM_NCRPHOT, 0.0);
    S->cst.assign(CHEM_NCONST, 0.0);

    for (int r = 0; r < chem_rate_registry_n; r++)
        store_rate(*S, chem_rate_registry[r], g, nmd);
#if CHEMISTRYNETWORK == 16
    for (int r = 0; r < chem_rate_registry16_n; r++)
        store_rate(*S, chem_rate_registry16[r], g, nmd);
#endif
#if CHEMISTRYNETWORK == 3 || CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 15 || CHEMISTRYNETWORK == 17
    S->crtab[1] = 1.0; // H2 -> H2+ + e-: zeta per H2 as SGChem (cheminmo.F crtab(2))
#endif
    // the four entries with no generated function (see file header)
    for (int i = 0; i < nmd; i++) {
        double t = S->temptab[i];
        S->cltab[(size_t)i * CHEM_NCLTAB + 51] = cl52_of(t);
        S->cltab[(size_t)i * CHEM_NCLTAB + 58] = cl59_of(t);
        S->cltab[(size_t)i * CHEM_NCLTAB + 59] = cl60_of(t);
        S->cltab[(size_t)i * CHEM_NCLTAB + 60] = cl61_of(t);
#ifdef CHEMCOOL_GASDUST_K23
        S->cltab[(size_t)i * CHEM_NCLTAB + 13] = 3.2e-34 * std::sqrt(t);   // Kim+2023 Eq. 45
#endif
    }

    // ---- cl1, cl2, cl3, cl6 (spline entries; chem_spline.cc) ----
    {
        std::vector<double> c1(nmd), c2(nmd), c3(nmd), c6(nmd);
        chem_build_cl_tables(S->temptab.data(), nmd, c1.data(), c2.data(),
                             c3.data(), c6.data(), cfg.iflag_atom);
        for (int i = 0; i < nmd; i++) {
            S->cltab[(size_t)i * CHEM_NCLTAB + 0] = c1[i];
            S->cltab[(size_t)i * CHEM_NCLTAB + 1] = c2[i];
            S->cltab[(size_t)i * CHEM_NCLTAB + 2] = c3[i];
            S->cltab[(size_t)i * CHEM_NCLTAB + 5] = c6[i];
        }
    }

    // ---- residual zeroing loops (coolinmo.F:1523-1592) ----
    if (cfg.iflag_atom == 1) {
        // cl6 already includes HI/HeI/HeII excitation cooling
        for (int i = 0; i < nmd; i++) {
            S->cltab[(size_t)i * CHEM_NCLTAB + 51] = 0.0;
            S->cltab[(size_t)i * CHEM_NCLTAB + 58] = 0.0;
            S->cltab[(size_t)i * CHEM_NCLTAB + 59] = 0.0;
            S->cltab[(size_t)i * CHEM_NCLTAB + 60] = 0.0;
        }
    } else if (cfg.iflag_atom == 2 && !cfg.wss_cie) {
        // #ifndef WSS_CIE_COOL -- HeI/HeII excitation + bremsstrahlung are
        // already inside the ca2 fit
        for (int i = 0; i < nmd; i++) {
            S->cltab[(size_t)i * CHEM_NCLTAB + 58] = 0.0;
            S->cltab[(size_t)i * CHEM_NCLTAB + 59] = 0.0;
            S->cltab[(size_t)i * CHEM_NCLTAB + 60] = 0.0;
            S->cltab[(size_t)i * CHEM_NCLTAB + 61] = 0.0;
            S->cltab[(size_t)i * CHEM_NCLTAB + 62] = 0.0;
        }
    }

    // ---- HeI / HeII ionization-fraction tables (coolinmo.F:1595-1619) ----
    S->HeI.assign(dat_HeI_tbl, dat_HeI_tbl + CHEM_NHE_TBL);
    S->HeII.assign(dat_HeII_tbl, dat_HeII_tbl + CHEM_NHE_TBL);

    // ---- approximate temperature derivatives (coolinmo.F:1625-1639) ----
    S->dtcltab.assign((size_t)nmd * CHEM_NCLTAB, 0.0);
    for (int i = 0; i < nmd - 1; i++) {
        double dt = 1.0 / (S->temptab[i+1] - S->temptab[i]);
        for (int j = 0; j < CHEM_NCLTAB; j++)
            S->dtcltab[(size_t)i*CHEM_NCLTAB + j] =
                (S->cltab[(size_t)(i+1)*CHEM_NCLTAB + j]
               - S->cltab[(size_t)i*CHEM_NCLTAB + j]) * dt;
    }
    for (int j = 0; j < CHEM_NCLTAB; j++)
        S->dtcltab[(size_t)(nmd-1)*CHEM_NCLTAB + j] =
            S->dtcltab[(size_t)(nmd-2)*CHEM_NCLTAB + j];

    S->dtchtab.assign((size_t)nmd * CHEM_NCHTAB, 0.0);
    for (int i = 0; i < nmd - 1; i++) {
        double dt = 1.0 / (S->temptab[i+1] - S->temptab[i]);
        for (int k = 0; k < CHEM_NCHTAB; k++)
            S->dtchtab[(size_t)i*CHEM_NCHTAB + k] =
                (S->chtab[(size_t)(i+1)*CHEM_NCHTAB + k]
               - S->chtab[(size_t)i*CHEM_NCHTAB + k]) * dt;
    }
    for (int k = 0; k < CHEM_NCHTAB; k++)
        S->dtchtab[(size_t)(nmd-1)*CHEM_NCHTAB + k] =
            S->dtchtab[(size_t)(nmd-2)*CHEM_NCHTAB + k];

    // ======================================================================
    // CO rotational cooling (coolinmo.F:1172-1249)
    // ======================================================================
    S->co_temptab.assign(CHEM_NTCO, 0.0);
    for (int I = 1; I <= CHEM_NTCO; I++) S->co_temptab[I-1] = 4.0 + 1.0 * I;
    S->co_colntab.assign(CHEM_NCDCO, 0.0);
    for (int I = 1; I <= CHEM_NCDCO; I++) S->co_colntab[I-1] = 14.4 + 0.1 * I;

    S->co_L0.assign(CHEM_NTCO, 0.0);
    chem_spline_eval(12, dat_co_temp, dat_co_data_L0, CHEM_NTCO,
                     S->co_temptab.data(), S->co_L0.data());
    build_2d_table(12, dat_co_temp, 10, dat_co_column, dat_co_data_LTE,
                   CHEM_NTCO, S->co_temptab.data(), CHEM_NCDCO,
                   S->co_colntab.data(), S->co_lte);
    build_2d_table(12, dat_co_temp, 10, dat_co_column, dat_co_data_n05,
                   CHEM_NTCO, S->co_temptab.data(), CHEM_NCDCO,
                   S->co_colntab.data(), S->co_n05);
    build_2d_table(12, dat_co_temp, 10, dat_co_column, dat_co_data_alp,
                   CHEM_NTCO, S->co_temptab.data(), CHEM_NCDCO,
                   S->co_colntab.data(), S->co_alp);
    tderiv_1d(S->co_L0, CHEM_NTCO, S->dTco_L0);
    tderiv_2d(S->co_lte, CHEM_NTCO, CHEM_NCDCO, S->dTco_lte);
    tderiv_2d(S->co_n05, CHEM_NTCO, CHEM_NCDCO, S->dTco_n05);
    tderiv_2d(S->co_alp, CHEM_NTCO, CHEM_NCDCO, S->dTco_alp);

    // ---- CO vibrational cooling (coolinmo.F:1257-1302) ----
    S->co_vib_temptab.assign(CHEM_NTCO_VIB, 0.0);
    for (int I = 1; I <= CHEM_NTCO_VIB; I++) S->co_vib_temptab[I-1] = 99.0 + 1.0 * I;
    S->co_vib_colntab.assign(CHEM_NCDCO_VIB, 0.0);
    for (int I = 1; I <= CHEM_NCDCO_VIB; I++) S->co_vib_colntab[I-1] = 12.9 + 0.1 * I;
    build_2d_table(6, dat_co_vib_temp, 8, dat_co_vib_column, dat_co_vib_LTE,
                   CHEM_NTCO_VIB, S->co_vib_temptab.data(), CHEM_NCDCO_VIB,
                   S->co_vib_colntab.data(), S->co_vib_LTE);
    tderiv_2d(S->co_vib_LTE, CHEM_NTCO_VIB, CHEM_NCDCO_VIB, S->dTco_vib_LTE);

    // ======================================================================
    // H2O rotational cooling (coolinmo.F:1304-1464)
    // ======================================================================
    S->h2o_temptab.assign(CHEM_NTH2O, 0.0);
    for (int I = 1; I <= CHEM_NTH2O; I++) S->h2o_temptab[I-1] = 9.0 + 1.0 * I;
    S->h2o_colntab.assign(CHEM_NCDH2O, 0.0);
    for (int I = 1; I <= CHEM_NCDH2O; I++) S->h2o_colntab[I-1] = 9.9 + 0.1 * I;

    const int nlow = 6, nhigh = 5, nh2o_temp = 11, nh2o_col = 10;
    double h2o_temp[11], L0_o[11], L0_p[11];
    for (int I = 1; I <= nlow; I++) {
        L0_o[I-1] = dat_h2o_L0_low_ortho[I-1];
        L0_p[I-1] = dat_h2o_L0_low_para[I-1];
        h2o_temp[I-1] = dat_h2o_temp_low[I-1];
    }
    for (int I = nlow + 1; I <= nh2o_temp; I++) {
        L0_o[I-1] = dat_h2o_L0_high[I - nlow - 1];
        L0_p[I-1] = dat_h2o_L0_high[I - nlow - 1];
        h2o_temp[I-1] = dat_h2o_temp_high[I - nlow - 1];
    }

    // merge low/high into the (nh2o_temp x nh2o_column) raw blocks
    std::vector<double> LTE_o(110), LTE_p(110), ALP_o(110), ALP_p(110),
                        N05_o(110), N05_p(110);
    for (int J = 1; J <= nh2o_col; J++)
        for (int I = 1; I <= nh2o_temp; I++) {
            int idx1 = I + (J - 1) * nh2o_temp;
            if (I <= nlow) {
                int idx2 = I + (J - 1) * nlow;
                LTE_o[idx1-1] = dat_h2o_LTE_low_ortho[idx2-1];
                LTE_p[idx1-1] = dat_h2o_LTE_low_para[idx2-1];
                ALP_o[idx1-1] = dat_h2o_alp_low_ortho[idx2-1];
                ALP_p[idx1-1] = dat_h2o_alp_low_para[idx2-1];
                N05_o[idx1-1] = dat_h2o_n05_low_ortho[idx2-1];
                N05_p[idx1-1] = dat_h2o_n05_low_para[idx2-1];
            } else {
                int idx2 = (I - nlow) + (J - 1) * nhigh;
                LTE_o[idx1-1] = dat_h2o_LTE_high[idx2-1];
                LTE_p[idx1-1] = dat_h2o_LTE_high[idx2-1];
                ALP_o[idx1-1] = dat_h2o_alp_high[idx2-1];
                ALP_p[idx1-1] = dat_h2o_alp_high[idx2-1];
                N05_o[idx1-1] = dat_h2o_n05_high[idx2-1];
                N05_p[idx1-1] = dat_h2o_n05_high[idx2-1];
            }
        }

    S->h2o_L0_o.assign(CHEM_NTH2O, 0.0);
    S->h2o_L0_p.assign(CHEM_NTH2O, 0.0);
    chem_spline_eval(nh2o_temp, h2o_temp, L0_o, CHEM_NTH2O,
                     S->h2o_temptab.data(), S->h2o_L0_o.data());
    chem_spline_eval(nh2o_temp, h2o_temp, L0_p, CHEM_NTH2O,
                     S->h2o_temptab.data(), S->h2o_L0_p.data());

    build_2d_table(nh2o_temp, h2o_temp, nh2o_col, dat_h2o_column, LTE_o.data(),
                   CHEM_NTH2O, S->h2o_temptab.data(), CHEM_NCDH2O,
                   S->h2o_colntab.data(), S->h2o_LTE_o);
    build_2d_table(nh2o_temp, h2o_temp, nh2o_col, dat_h2o_column, N05_o.data(),
                   CHEM_NTH2O, S->h2o_temptab.data(), CHEM_NCDH2O,
                   S->h2o_colntab.data(), S->h2o_n05_o);
    build_2d_table(nh2o_temp, h2o_temp, nh2o_col, dat_h2o_column, ALP_o.data(),
                   CHEM_NTH2O, S->h2o_temptab.data(), CHEM_NCDH2O,
                   S->h2o_colntab.data(), S->h2o_alp_o);
    build_2d_table(nh2o_temp, h2o_temp, nh2o_col, dat_h2o_column, LTE_p.data(),
                   CHEM_NTH2O, S->h2o_temptab.data(), CHEM_NCDH2O,
                   S->h2o_colntab.data(), S->h2o_LTE_p);
    build_2d_table(nh2o_temp, h2o_temp, nh2o_col, dat_h2o_column, N05_p.data(),
                   CHEM_NTH2O, S->h2o_temptab.data(), CHEM_NCDH2O,
                   S->h2o_colntab.data(), S->h2o_n05_p);
    build_2d_table(nh2o_temp, h2o_temp, nh2o_col, dat_h2o_column, ALP_p.data(),
                   CHEM_NTH2O, S->h2o_temptab.data(), CHEM_NCDH2O,
                   S->h2o_colntab.data(), S->h2o_alp_p);

    tderiv_1d(S->h2o_L0_o, CHEM_NTH2O, S->dTh2o_L0_o);
    tderiv_1d(S->h2o_L0_p, CHEM_NTH2O, S->dTh2o_L0_p);
    tderiv_2d(S->h2o_LTE_o, CHEM_NTH2O, CHEM_NCDH2O, S->dTh2o_LTE_o);
    tderiv_2d(S->h2o_n05_o, CHEM_NTH2O, CHEM_NCDH2O, S->dTh2o_n05_o);
    tderiv_2d(S->h2o_alp_o, CHEM_NTH2O, CHEM_NCDH2O, S->dTh2o_alp_o);
    tderiv_2d(S->h2o_LTE_p, CHEM_NTH2O, CHEM_NCDH2O, S->dTh2o_LTE_p);
    tderiv_2d(S->h2o_n05_p, CHEM_NTH2O, CHEM_NCDH2O, S->dTh2o_n05_p);
    tderiv_2d(S->h2o_alp_p, CHEM_NTH2O, CHEM_NCDH2O, S->dTh2o_alp_p);

    // ---- H2O vibrational cooling (coolinmo.F:1466-1517) ----
    S->h2o_vib_temptab.assign(CHEM_NTH2O_VIB, 0.0);
    for (int I = 1; I <= CHEM_NTH2O_VIB; I++) S->h2o_vib_temptab[I-1] = 99.0 + 1.0 * I;
    S->h2o_vib_colntab.assign(CHEM_NCDH2O_VIB, 0.0);
    for (int I = 1; I <= CHEM_NCDH2O_VIB; I++) S->h2o_vib_colntab[I-1] = 12.9 + 0.1 * I;
    build_2d_table(6, dat_h2o_vib_temp, 8, dat_h2o_vib_column, dat_h2o_vib_LTE,
                   CHEM_NTH2O_VIB, S->h2o_vib_temptab.data(), CHEM_NCDH2O_VIB,
                   S->h2o_vib_colntab.data(), S->h2o_vib_LTE);
    tderiv_2d(S->h2o_vib_LTE, CHEM_NTH2O_VIB, CHEM_NCDH2O_VIB, S->dTh2o_vib_LTE);

    // ---- photoinit + CO shielding + ISRF + hng ----
    build_phtab(S->phtab);
    build_co_shielding(*S);
    S->chi_mathis.assign(dat_chi_tab_mathis, dat_chi_tab_mathis + CHEM_NCHI);
    S->chi_black .assign(dat_chi_tab_black,  dat_chi_tab_black  + CHEM_NCHI);
    S->hng.assign(dat_hng, dat_hng + CHEM_NHNG);

    // ---- data files ----
    if (cfg.wss_cie && read_wss(dir, *S))    { delete S; return 1; }
    if (read_h2_opac(dir, *S))               { delete S; return 1; }
    if (read_temp_gamma(dir, *S))            { delete S; return 1; }
    if (!cfg.wss_cie) for (int k = 0; k < 9; k++) S->wss[k].assign(CHEM_NWSS, 0.0);

    // ---- publish ----
    std::memset(&T, 0, sizeof(T));
    T.owner = S;
    T.temptab = S->temptab.data();
    T.cltab   = reinterpret_cast<const double (*)[CHEM_NCLTAB]>(S->cltab.data());
    T.dtcltab = reinterpret_cast<const double (*)[CHEM_NCLTAB]>(S->dtcltab.data());
    T.chtab   = reinterpret_cast<const double (*)[CHEM_NCHTAB]>(S->chtab.data());
    T.dtchtab = reinterpret_cast<const double (*)[CHEM_NCHTAB]>(S->dtchtab.data());
    T.crtab = S->crtab.data(); T.crphot = S->crphot.data();
    T.phtab = S->phtab.data(); T.cst = S->cst.data();
    T.dtlog = dtlog; T.tmin = tmin; T.tmax = tmax;

    T.co_temptab = S->co_temptab.data(); T.co_colntab = S->co_colntab.data();
    T.co_L0 = S->co_L0.data();           T.dTco_L0 = S->dTco_L0.data();
    T.co_lte = S->co_lte.data();         T.dTco_lte = S->dTco_lte.data();
    T.co_n05 = S->co_n05.data();         T.dTco_n05 = S->dTco_n05.data();
    T.co_alp = S->co_alp.data();         T.dTco_alp = S->dTco_alp.data();
    T.co_vib_temptab = S->co_vib_temptab.data();
    T.co_vib_colntab = S->co_vib_colntab.data();
    T.co_vib_LTE_final = S->co_vib_LTE.data();
    T.dTco_vib_LTE = S->dTco_vib_LTE.data();

    T.h2o_temptab = S->h2o_temptab.data(); T.h2o_colntab = S->h2o_colntab.data();
    T.h2o_L0_ortho = S->h2o_L0_o.data();   T.dTh2o_L0_ortho = S->dTh2o_L0_o.data();
    T.h2o_L0_para  = S->h2o_L0_p.data();   T.dTh2o_L0_para  = S->dTh2o_L0_p.data();
    T.h2o_LTE_ortho = S->h2o_LTE_o.data(); T.dTh2o_LTE_ortho = S->dTh2o_LTE_o.data();
    T.h2o_n05_ortho = S->h2o_n05_o.data(); T.dTh2o_n05_ortho = S->dTh2o_n05_o.data();
    T.h2o_alp_ortho = S->h2o_alp_o.data(); T.dTh2o_alp_ortho = S->dTh2o_alp_o.data();
    T.h2o_LTE_para  = S->h2o_LTE_p.data(); T.dTh2o_LTE_para  = S->dTh2o_LTE_p.data();
    T.h2o_n05_para  = S->h2o_n05_p.data(); T.dTh2o_n05_para  = S->dTh2o_n05_p.data();
    T.h2o_alp_para  = S->h2o_alp_p.data(); T.dTh2o_alp_para  = S->dTh2o_alp_p.data();
    T.h2o_vib_temptab = S->h2o_vib_temptab.data();
    T.h2o_vib_colntab = S->h2o_vib_colntab.data();
    T.h2o_vib_LTE_final = S->h2o_vib_LTE.data();
    T.dTh2o_vib_LTE = S->dTh2o_vib_LTE.data();

    T.h2_opac_temp = S->h2_opac_temp.data();
    T.h2_opac_column = S->h2_opac_column.data();
    T.h2_opac = S->h2_opac.data();

    T.wss_C = S->wss[0].data(); T.wss_N = S->wss[1].data();
    T.wss_O = S->wss[2].data(); T.wss_Ne = S->wss[3].data();
    T.wss_Mg = S->wss[4].data(); T.wss_Si = S->wss[5].data();
    T.wss_S = S->wss[6].data(); T.wss_Ca = S->wss[7].data();
    T.wss_Fe = S->wss[8].data();
    T.HeI_tbl = S->HeI.data(); T.HeII_tbl = S->HeII.data();
    T.CO_self_shielding = S->CO_ss.data();
    T.CO_H2_shielding = S->CO_H2.data();
    T.chi_tab_mathis = S->chi_mathis.data();
    T.chi_tab_black = S->chi_black.data();
    T.temp_table = S->temp_table.data();
    T.gamma_table = S->gamma_table.data();
    T.eh2_table = S->eh2_table.data();
    T.hng = S->hng.data();
    return 0;
}

void chem_tables_free(ChemTables& T)
{
    if (T.owner) { delete static_cast<ChemTableStore*>(T.owner); }
    std::memset(&T, 0, sizeof(T));
}
