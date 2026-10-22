// test_gate3.cc -- golden test for the Gate-3 leaf-kernel and table ports.
//
// Links the canonical Fortran objects and the C++ port into ONE binary, so
// both implementations see the same inputs, and drives them over dense
// parameter grids that deliberately include every guard boundary.
//
//   PHASE A  table builder: chem_tables_build() vs the arrays COOLINMO /
//            CHEMINMO / PHOTOINIT / CONST_RATES / read_wss_collis_table /
//            LOAD_H2_TABLE / INIT_TEMPERATURE_LOOKUP actually built.
//   PHASE B  leaf kernels, both sides reading the FORTRAN tables -- isolates a
//            kernel deviation from a table deviation.
//   PHASE C  the same kernels with the C++-built tables (end-to-end).
//
// Tolerance 1e-12 relative throughout (same rule as golden/build_splines.sh).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

#include "chem_types.h"

extern "C" {
#include "GIZMO_config.h"
#include "cooling/chemcool/f2c.h"
#include "cooling/chemcool/chemcool_consts.h"
#include "declarations/treecol_config.h"

#include "../../chem_commons.h"


extern CHEM_TLS_VARCOMP struct { double abhe; double X_H_chem; } CHEM_COMP;
extern CHEM_TLS struct { double G_dust, chi_isrf, phrates[NPHTAB]; } current_photo_;

#if defined(RT_TREECOL_DUST_WEIGHTED) || defined(RT_TREECOL_H2_CO)
extern CHEM_TLS struct {
    double diffuse_dust_heat;
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
    double fac_uv[NPIX];
#endif
    double column_density_projection[NPIX];
    double column_density_projection_h2[NPIX];
    double column_density_projection_co[NPIX];
} PROJECT;
#endif

void COOLINMO(void);
void CHEMINMO(void);

void LOAD_H2_TABLE(void);
void INIT_TEMPERATURE_LOOKUP(void);
void calc_photo_(double* temp, double* rpar, double* abh2, double* abhd,
                 double* abco, double* phrates, double* G_dust, double* chi_isrf);
void compute_gamma_(double* abh2, double* ekn, double* gamma);
void calc_spec_(double* temp, double* en);
void chemcool_fatal_(int* code) { std::printf("FATAL %d\n", *code); std::exit(1); }
}

// reference shims (chem_backend_fortran.cc)
void chem_bind_tables(ChemTables& T);
void chem_bind_cell(ChemCell& c, const ChemTables& T, int net);
void fort_lookup_temp_gamma(double abh2, double ekn, double* temp, double* gamma);
double fort_calc_H2_shielding_factor(double temp, double abh2, double NH);
void fort_calc_dust_temp(double yn, double chi_isrf, double temp, double abHI,
                         double* RH2, double* Rdust, int nocalc, ChemCell& c);
int fort_cool_func(double temp, double yn, double dl, double divv, double G_dust,
                   const double abundances[16], double* ylam,
                   double rates[CHEM_NRATES], int iflag_cmb, ChemCell& c);

// ---------------------------------------------------------------------------
static const double TOL = 1e-12;

struct Acc {
    const char* name; long n = 0, npass = 0; double worst = 0.0;
    double wa = 0, wb = 0;
    explicit Acc(const char* nm) : name(nm) {}
    // For the finite-difference derivative tables the meaningful criterion is
    // the difference relative to the SCALE OF THE OPERANDS being differenced,
    // not to the (catastrophically cancelled) difference itself: dT tables are
    // v[j+1]-v[j] of numbers that agree to <1e-15 relative, so a 1e-8 relative
    // error in the difference is pure cancellation, not a port deviation.
    void add_scaled(double a, double b, double scale) {
        n++;
        double d = std::fabs(a - b);
        double rel = (scale > 0) ? d / scale : d;
        if (!std::isfinite(a) || !std::isfinite(b)) rel = 1e300;
        if (rel <= TOL) npass++;
        if (rel > worst) { worst = rel; wa = a; wb = b; }
    }
    void add(double a, double b) {
        n++;
        double rel;
        if (a == b) rel = 0.0;
        else if (std::isnan(a) && std::isnan(b)) rel = 0.0;
        else if (!std::isfinite(a) || !std::isfinite(b)) rel = 1e300;
        else {
            double d = std::fabs(a - b);
            double s = std::fabs(a) > std::fabs(b) ? std::fabs(a) : std::fabs(b);
            rel = (s > 0) ? d / s : d;
        }
        if (rel <= TOL) npass++;
        if (rel > worst) { worst = rel; wa = a; wb = b; }
    }
};

static int report(const Acc& r)
{
    int ok = (r.npass == r.n);
    std::printf("  %-28s %8ld/%-8ld %s   worst rel = %.3e",
                r.name, r.npass, r.n, ok ? "PASS" : "FAIL", r.worst);
    if (!ok) std::printf("   (F=%.17g  C=%.17g)", r.wa, r.wb);
    std::printf("\n");
    return ok;
}

// ---------------------------------------------------------------------------
static double Zs = 0.1;

static void init_commons(double G0, double zeta, int iphoto)
{
    COOLI.iflag_mn = 4; COOLI.iflag_ad = 6; COOLI.iflag_atom = 2;
    COOLI.iflag_3bh2a = 5; COOLI.iflag_3bh2b = 4; COOLI.iflag_h3pra = 1;
    COOLI.iphoto = iphoto;
    COOLIC.irad_heat = 0; COOLI.isrf_option = 1; COOLIC.no_chem = 0;
    COOLIC.id_current = 1; COOLIC.index_current = 0;
    COOLI.iflag_h2opc = 0; COOLI.idma_mass_option = 1;
    COOLR.h2_form_ex = 0.84; COOLR.h2_form_kin = 0.12;
    COOLR.deff = 1.0; COOLR.phi_pah = 0.5;
    COOLR.AV_conversion_factor = 5.348e-22;
    COOLR.AV_ext = 0.0; COOLR.dm_density = 0.0; COOLRC.pdv_term = 0.0;
    COOLRC.redshift = 0.0;
    COOLRC.abundD = 2.6e-5;
    COOLRC.G0 = G0; COOLRC.G0_LW = G0; COOLRC.G0_dust = G0;
    COOLRC.cosmic_ray_ion_rate = zeta;
    COOLRC.cr_energy_density = 0.0;
    COOLRC.dust_to_gas_ratio = Zs;
    COOLRC.tdust = 10.0;
    COOLRC.rt_phot_HI = COOLRC.rt_phot_HeI = COOLRC.rt_phot_HeII = 0;
    COOLRC.rt_heat_HI = COOLRC.rt_heat_HeI = COOLRC.rt_heat_HeII = 0;
    COOLRC.abundc  = 1.4e-4 * Zs;
    COOLRC.abundo  = 3.2e-4 * Zs;
    COOLRC.abundsi = 1.5e-5 * Zs;
    COOLRC.abundM  = 1.5e-5 * Zs;
    COOLRC.abundN  = 7.6e-5 * Zs;
#ifdef WSS_CIE_COOL
    double zsolar[12] = {0.7065, 2.07e-3, 5.91e-4, 5.49e-3, 1.10e-3, 6.83e-4,
                         0.2806, 8.36e-4, 1.41e-3, 4.09e-4, 6.44e-5, 1.10e-3};
    for (int k = 0; k < 12; k++) COOLRC.Zmass[k] = zsolar[k];
    for (int k = 0; k < 12; k++) if (k != 0 && k != 6) COOLRC.Zmass[k] *= Zs;
#endif
}

// ===========================================================================
// PHASE A -- tables
// ===========================================================================
static int phase_A(const ChemTables& C)
{
    int ok = 1;
    std::printf("PHASE A -- table construction (C++ build vs COOLINMO/CHEMINMO)\n");

    { Acc r("temptab");
      for (int i = 0; i < NMD; i++) r.add(COOLR.temptab[i], C.temptab[i]);
      ok &= report(r); }
    { Acc r("cltab");
      const double* F = &COOLR.cltab[0][0];
      for (int i = 0; i < NMD; i++) for (int k = 0; k < NCLTAB; k++)
          r.add(F[(size_t)i*NCLTAB + k], C.cltab[i][k]);
      ok &= report(r); }
    { Acc r("dtcltab (scaled)");
      const double* F = &COOLR.dtcltab[0][0];
      const double* V = &COOLR.cltab[0][0];
      for (int i = 0; i < NMD; i++) for (int k = 0; k < NCLTAB; k++) {
          double a1 = std::fabs(V[(size_t)i*NCLTAB + k]);
          double a2 = (i+1 < NMD) ? std::fabs(V[(size_t)(i+1)*NCLTAB + k]) : a1;
          double dt = (i+1 < NMD) ? 1.0/(COOLR.temptab[i+1]-COOLR.temptab[i]) : 1.0;
          r.add_scaled(F[(size_t)i*NCLTAB + k], C.dtcltab[i][k],
                       (a1 > a2 ? a1 : a2) * dt);
      }
      ok &= report(r); }
    { Acc r("chtab");
      const double* F = &COOLR.chtab[0][0];
      for (int i = 0; i < NMD; i++) for (int k = 0; k < NCHTAB; k++)
          r.add(F[(size_t)i*NCHTAB + k], C.chtab[i][k]);
      ok &= report(r); }
    { Acc r("dtchtab (scaled)");
      const double* F = &COOLR.dtchtab[0][0];
      const double* V = &COOLR.chtab[0][0];
      for (int i = 0; i < NMD; i++) for (int k = 0; k < NCHTAB; k++) {
          double a1 = std::fabs(V[(size_t)i*NCHTAB + k]);
          double a2 = (i+1 < NMD) ? std::fabs(V[(size_t)(i+1)*NCHTAB + k]) : a1;
          double dt = (i+1 < NMD) ? 1.0/(COOLR.temptab[i+1]-COOLR.temptab[i]) : 1.0;
          r.add_scaled(F[(size_t)i*NCHTAB + k], C.dtchtab[i][k],
                       (a1 > a2 ? a1 : a2) * dt);
      }
      ok &= report(r); }
    { Acc r("crtab");   for (int k=0;k<NCRTAB;k++)  r.add(COOLR.crtab[k],  C.crtab[k]);  ok &= report(r); }
    { Acc r("crphot");  for (int k=0;k<NCRPHOT;k++) r.add(COOLR.crphot[k], C.crphot[k]); ok &= report(r); }
    { Acc r("phtab");   for (int k=0;k<NPHTAB;k++)  r.add(COOLR.phtab[k],  C.phtab[k]);  ok &= report(r); }
    { Acc r("cst");     for (int k=0;k<NCONST;k++)  r.add(COOLR.cst[k],    C.cst[k]);    ok &= report(r); }

    // molecular-coolant tables, via the Fortran-bound view
    ChemTables F; chem_bind_tables(F);
    struct Pair { const char* nm; const double* f; const double* c; size_t n;
                  const double* vf; int nN; };
    Pair pairs[] = {
        {"co_temptab",      F.co_temptab,      C.co_temptab,      CHEM_NTCO, 0, 0},
        {"co_colntab",      F.co_colntab,      C.co_colntab,      CHEM_NCDCO, 0, 0},
        {"co_L0",           F.co_L0,           C.co_L0,           CHEM_NTCO, 0, 0},
        {"dTco_L0",         F.dTco_L0,         C.dTco_L0,         CHEM_NTCO, F.co_L0, 0},
        {"co_lte",          F.co_lte,          C.co_lte,   (size_t)CHEM_NTCO*CHEM_NCDCO, 0, 0},
        {"co_n05",          F.co_n05,          C.co_n05,   (size_t)CHEM_NTCO*CHEM_NCDCO, 0, 0},
        {"co_alp",          F.co_alp,          C.co_alp,   (size_t)CHEM_NTCO*CHEM_NCDCO, 0, 0},
        {"dTco_lte",        F.dTco_lte,        C.dTco_lte, (size_t)CHEM_NTCO*CHEM_NCDCO, F.co_lte, 0},
        {"dTco_n05",        F.dTco_n05,        C.dTco_n05, (size_t)CHEM_NTCO*CHEM_NCDCO, F.co_n05, 0},
        {"dTco_alp",        F.dTco_alp,        C.dTco_alp, (size_t)CHEM_NTCO*CHEM_NCDCO, F.co_alp, 0},
        {"co_vib_temptab",  F.co_vib_temptab,  C.co_vib_temptab,  CHEM_NTCO_VIB, 0, 0},
        {"co_vib_colntab",  F.co_vib_colntab,  C.co_vib_colntab,  CHEM_NCDCO_VIB, 0, 0},
        {"co_vib_LTE",      F.co_vib_LTE_final,C.co_vib_LTE_final,(size_t)CHEM_NTCO_VIB*CHEM_NCDCO_VIB, 0, 0},
        {"dTco_vib_LTE",    F.dTco_vib_LTE,    C.dTco_vib_LTE,    (size_t)CHEM_NTCO_VIB*CHEM_NCDCO_VIB, F.co_vib_LTE_final, 0},
        {"h2o_temptab",     F.h2o_temptab,     C.h2o_temptab,     CHEM_NTH2O, 0, 0},
        {"h2o_colntab",     F.h2o_colntab,     C.h2o_colntab,     CHEM_NCDH2O, 0, 0},
        {"h2o_L0_ortho",    F.h2o_L0_ortho,    C.h2o_L0_ortho,    CHEM_NTH2O, 0, 0},
        {"h2o_L0_para",     F.h2o_L0_para,     C.h2o_L0_para,     CHEM_NTH2O, 0, 0},
        {"dTh2o_L0_ortho",  F.dTh2o_L0_ortho,  C.dTh2o_L0_ortho,  CHEM_NTH2O, F.h2o_L0_ortho, 0},
        {"dTh2o_L0_para",   F.dTh2o_L0_para,   C.dTh2o_L0_para,   CHEM_NTH2O, F.h2o_L0_para, 0},
        {"h2o_LTE_ortho",   F.h2o_LTE_ortho,   C.h2o_LTE_ortho,   (size_t)CHEM_NTH2O*CHEM_NCDH2O, 0, 0},
        {"h2o_n05_ortho",   F.h2o_n05_ortho,   C.h2o_n05_ortho,   (size_t)CHEM_NTH2O*CHEM_NCDH2O, 0, 0},
        {"h2o_alp_ortho",   F.h2o_alp_ortho,   C.h2o_alp_ortho,   (size_t)CHEM_NTH2O*CHEM_NCDH2O, 0, 0},
        {"h2o_LTE_para",    F.h2o_LTE_para,    C.h2o_LTE_para,    (size_t)CHEM_NTH2O*CHEM_NCDH2O, 0, 0},
        {"h2o_n05_para",    F.h2o_n05_para,    C.h2o_n05_para,    (size_t)CHEM_NTH2O*CHEM_NCDH2O, 0, 0},
        {"h2o_alp_para",    F.h2o_alp_para,    C.h2o_alp_para,    (size_t)CHEM_NTH2O*CHEM_NCDH2O, 0, 0},
        {"dTh2o_LTE_ortho", F.dTh2o_LTE_ortho, C.dTh2o_LTE_ortho, (size_t)CHEM_NTH2O*CHEM_NCDH2O, F.h2o_LTE_ortho, 0},
        {"dTh2o_n05_ortho", F.dTh2o_n05_ortho, C.dTh2o_n05_ortho, (size_t)CHEM_NTH2O*CHEM_NCDH2O, F.h2o_n05_ortho, 0},
        {"dTh2o_alp_ortho", F.dTh2o_alp_ortho, C.dTh2o_alp_ortho, (size_t)CHEM_NTH2O*CHEM_NCDH2O, F.h2o_alp_ortho, 0},
        {"dTh2o_LTE_para",  F.dTh2o_LTE_para,  C.dTh2o_LTE_para,  (size_t)CHEM_NTH2O*CHEM_NCDH2O, F.h2o_LTE_para, 0},
        {"dTh2o_n05_para",  F.dTh2o_n05_para,  C.dTh2o_n05_para,  (size_t)CHEM_NTH2O*CHEM_NCDH2O, F.h2o_n05_para, 0},
        {"dTh2o_alp_para",  F.dTh2o_alp_para,  C.dTh2o_alp_para,  (size_t)CHEM_NTH2O*CHEM_NCDH2O, F.h2o_alp_para, 0},
        {"h2o_vib_temptab", F.h2o_vib_temptab, C.h2o_vib_temptab, CHEM_NTH2O_VIB, 0, 0},
        {"h2o_vib_colntab", F.h2o_vib_colntab, C.h2o_vib_colntab, CHEM_NCDH2O_VIB, 0, 0},
        {"h2o_vib_LTE",     F.h2o_vib_LTE_final,C.h2o_vib_LTE_final,(size_t)CHEM_NTH2O_VIB*CHEM_NCDH2O_VIB, 0, 0},
        {"dTh2o_vib_LTE",   F.dTh2o_vib_LTE,   C.dTh2o_vib_LTE,   (size_t)CHEM_NTH2O_VIB*CHEM_NCDH2O_VIB, F.h2o_vib_LTE_final, 0},
        {"h2_opac_temp",    F.h2_opac_temp,    C.h2_opac_temp,    CHEM_NH2OP, 0, 0},
        {"h2_opac_column",  F.h2_opac_column,  C.h2_opac_column,  CHEM_NH2OP, 0, 0},
        {"h2_opac",         F.h2_opac,         C.h2_opac,         (size_t)CHEM_NH2OP*CHEM_NH2OP, 0, 0},
        {"CO_self_shielding", F.CO_self_shielding, C.CO_self_shielding, CHEM_IMAX_COSS, 0, 0},
        {"CO_H2_shielding",   F.CO_H2_shielding,   C.CO_H2_shielding,   CHEM_IMAX_COH2, 0, 0},
        {"temp_table",      F.temp_table,      C.temp_table,      CHEM_ISIZE, 0, 0},
        {"gamma_table",     F.gamma_table,     C.gamma_table,     CHEM_ISIZE, 0, 0},
        {"eh2_table",       F.eh2_table,       C.eh2_table,       CHEM_ISIZE2, 0, 0},
#ifdef WSS_CIE_COOL
        {"wss_C",  F.wss_C,  C.wss_C,  CHEM_NWSS, 0, 0},
        {"wss_N",  F.wss_N,  C.wss_N,  CHEM_NWSS, 0, 0},
        {"wss_O",  F.wss_O,  C.wss_O,  CHEM_NWSS, 0, 0},
        {"wss_Ne", F.wss_Ne, C.wss_Ne, CHEM_NWSS, 0, 0},
        {"wss_Mg", F.wss_Mg, C.wss_Mg, CHEM_NWSS, 0, 0},
        {"wss_Si", F.wss_Si, C.wss_Si, CHEM_NWSS, 0, 0},
        {"wss_S",  F.wss_S,  C.wss_S,  CHEM_NWSS, 0, 0},
        {"wss_Ca", F.wss_Ca, C.wss_Ca, CHEM_NWSS, 0, 0},
        {"wss_Fe", F.wss_Fe, C.wss_Fe, CHEM_NWSS, 0, 0},
        {"HeI_tbl",  F.HeI_tbl,  C.HeI_tbl,  CHEM_NHE_TBL, 0, 0},
        {"HeII_tbl", F.HeII_tbl, C.HeII_tbl, CHEM_NHE_TBL, 0, 0},
#endif
    };
    for (size_t p = 0; p < sizeof(pairs)/sizeof(pairs[0]); p++) {
        Acc r(pairs[p].nm);
        for (size_t i = 0; i < pairs[p].n; i++) {
            if (pairs[p].vf) {
                // dT(i,j) = v(i,j+1) - v(i,j); v varies smoothly in j, so
                // |v(i,j)| is the operand scale.
                r.add_scaled(pairs[p].f[i], pairs[p].c[i],
                             std::fabs(pairs[p].vf[i]));
            } else {
                r.add(pairs[p].f[i], pairs[p].c[i]);
            }
        }
        ok &= report(r);
    }
    return ok;
}

// ===========================================================================
// parameter grids
// ===========================================================================
static std::vector<double> temp_grid()
{
    std::vector<double> t;
    // every guard boundary that appears in cool_func / cool_util / coolinmo
    const double guards[] = {
        1.0, 1.0000001, 4.999999, 5.0, 5.000001, 9.999999, 10.0, 10.000001,
        49.9999, 50.0, 50.0001, 99.9999, 100.0, 100.0001, 193.999, 194.0,
        194.001, 249.999, 250.0, 250.001, 510.999, 511.0, 511.001, 699.999,
        700.0, 700.001, 999.999, 1000.0, 1000.001, 1999.99, 2000.0, 2000.01,
        2089.99, 2090.0, 2090.01, 3685.99, 3686.0, 3686.01, 3999.99, 4000.0,
        4000.01, 5000.0, 7509.99, 7510.0, 7510.01, 9999.99, 10000.0, 10000.01,
        19999.9, 20000.0, 20000.1, 1.9952623e4, 1e5, 3.2e5 - 1, 3.2e5,
        3.2e5 + 1, 1e6, 1.28e6 - 1, 1.28e6, 1.28e6 + 1, 1e7 - 1, 1e7, 1e7 + 1,
        1e8 - 1, 1e8, 1e8 + 1, 2e8, 1e9
    };
    for (size_t i = 0; i < sizeof(guards)/sizeof(guards[0]); i++) t.push_back(guards[i]);
    // dense log sweep 1 K -> 1e9 K
    for (int i = 0; i <= 900; i++) t.push_back(std::pow(10.0, 9.0 * i / 900.0));
    // dense linear sweep in the 1-K-bin CO/H2O table region (tests int(temp))
    for (int i = 0; i < 400; i++) t.push_back(5.0 + 9.97 * i);
    return t;
}

struct AbSet { double a[16]; double divv, dl; };

static std::vector<AbSet> ab_grid()
{
    std::vector<AbSet> v;
    // molecular / atomic / ionized / trace mixes, with and without divv
    const double divvs[] = {0.0, 1e-13, 1e-11, -3.7e-12, 1e-8};
    const double dls[]   = {1e17, 3e19};
    double base[6][16] = {
        // H2   O      OH     H2O    CO     CI     CII    SiI    SiII   e-     H+     HI     HD     HeI    HeII   HeIII
        {0.49,  3.0e-5,1e-9,  1e-9,  1e-6,  1e-6,  1e-5,  1e-7,  1e-6,  1e-5,  1e-6,  0.02,  1e-9,  0.1,   0.0,   0.0},
        {1e-6,  3.2e-5,1e-12, 1e-12, 1e-12, 1e-8,  1.4e-5,1e-9,  1.5e-6,1.4e-5,1e-4,  0.999, 1e-12, 0.1,   0.0,   0.0},
        {0.0,   1e-9,  0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   1.1,   1.0,   1e-9,  0.0,   0.0,   0.0,   0.1},
        {0.25,  1.6e-5,1e-7,  1e-6,  1.4e-5,1e-7,  1e-7,  1e-8,  1e-8,  1e-7,  1e-8,  0.5,   2.6e-5,0.1,   0.0,   0.0},
        {0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   1.0,   0.0,   0.1,   0.0,   0.0},
        {0.1,   3.2e-4,1e-8,  1e-7,  3.0e-4,1.4e-4,1e-9,  1.5e-5,1e-10, 1e-9,  1e-10, 0.8,   1e-6,  0.05,  0.03,  0.02},
    };
    for (int b = 0; b < 6; b++)
        for (size_t d = 0; d < sizeof(divvs)/sizeof(divvs[0]); d++)
            for (size_t l = 0; l < sizeof(dls)/sizeof(dls[0]); l++) {
                AbSet s; std::memcpy(s.a, base[b], sizeof(s.a));
                s.divv = divvs[d]; s.dl = dls[l];
                v.push_back(s);
            }
    return v;
}

// ===========================================================================
// PHASE B / C -- leaf kernels
// ===========================================================================
static int kernels(const ChemTables& TT, const char* tag, int net)
{
    int ok = 1;
    std::vector<double> T = temp_grid();
    std::vector<AbSet> A = ab_grid();

    // ---------------- cool_func ----------------
    {
        Acc rylam("cool_func ylam");
        Acc rrate("cool_func rates[28]");
        Acc rtd  ("cool_func tdust out");
        static char rnames[CHEM_NRATES][24];
        std::vector<Acc> per;
        for (int k = 0; k < CHEM_NRATES; k++) {
            std::snprintf(rnames[k], sizeof(rnames[k]), "  rates(%d)", k + 1);
            per.push_back(Acc(rnames[k]));
        }
        const double yns[] = {1e-3, 1.0, 1e2, 1e4, 1e6, 1e9, 1e12};
        for (size_t ai = 0; ai < A.size(); ai++) {
            for (size_t yi = 0; yi < sizeof(yns)/sizeof(yns[0]); yi++) {
                for (size_t ti = 0; ti < T.size(); ti += 3) {
                    double temp = T[ti];
                    for (int cmb = 0; cmb < 2; cmb++) {
                        ChemCell cF, cC;
                        chem_bind_cell(cF, TT, net); cF.tdust = 23.7;
                        chem_bind_cell(cC, TT, net); cC.tdust = 23.7;
                        double lamF = 0, lamC = 0;
                        double rF[CHEM_NRATES], rC[CHEM_NRATES];
                        for (int k = 0; k < CHEM_NRATES; k++) rF[k] = rC[k] = -12345.0;
                        double Gd = 1.7;
                        fort_cool_func(temp, yns[yi], A[ai].dl, A[ai].divv, Gd,
                                       A[ai].a, &lamF, rF, cmb, cF);
                        cxx_cool_func(temp, yns[yi], A[ai].dl, A[ai].divv, Gd,
                                      A[ai].a, &lamC, rC, cmb, cC);
                        rylam.add(lamF, lamC);
                        rtd.add(cF.tdust, cC.tdust);
                        for (int k = 0; k < CHEM_NRATES; k++) {
                            rrate.add(rF[k], rC[k]);
                            per[k].add(rF[k], rC[k]);
                        }
                    }
                }
            }
        }
        std::printf("PHASE %s -- cool_func\n", tag);
        ok &= report(rylam); ok &= report(rrate); ok &= report(rtd);
        for (int k = 0; k < CHEM_NRATES; k++)
            if (per[k].npass != per[k].n) report(per[k]);
    }

    // ---------------- calc_dust_temp ----------------
    {
        Acc rtd("calc_dust_temp tdust");
        Acc rrh("calc_dust_temp RH2");
        Acc rrd("calc_dust_temp Rdust");
        const double yns[] = {1e-2, 1.0, 1e3, 1e6, 1e9};
        const double chis[] = {1e-4, 0.1, 1.0};
        const double dgrs[] = {0.0, 1e-9, 1e-8, 0.1, 1.0};
        const double td0s[] = {5.0, 10.0, 300.0, 1500.0};
        for (size_t yi = 0; yi < sizeof(yns)/sizeof(yns[0]); yi++)
        for (size_t xi = 0; xi < sizeof(chis)/sizeof(chis[0]); xi++)
        for (size_t di = 0; di < sizeof(dgrs)/sizeof(dgrs[0]); di++)
        for (size_t t0 = 0; t0 < sizeof(td0s)/sizeof(td0s[0]); t0++)
        for (size_t ti = 0; ti < T.size(); ti += 7)
        for (int nocalc = 0; nocalc < 2; nocalc++) {
            double temp = T[ti];
            ChemCell cF, cC;
            chem_bind_cell(cF, TT, net); chem_bind_cell(cC, TT, net);
            cF.dust_to_gas_ratio = cC.dust_to_gas_ratio = dgrs[di];
            COOLRC.dust_to_gas_ratio = dgrs[di];
            cF.tdust = cC.tdust = td0s[t0];
            double RH2f = 0, Rdf = 0, RH2c = 0, Rdc = 0;
            fort_calc_dust_temp(yns[yi], chis[xi], temp, 0.7, &RH2f, &Rdf, nocalc, cF);
            cxx_calc_dust_temp (yns[yi], chis[xi], temp, 0.7, &RH2c, &Rdc, nocalc, cC);
            rtd.add(cF.tdust, cC.tdust);
            rrh.add(RH2f, RH2c);
            rrd.add(Rdf, Rdc);
        }
        COOLRC.dust_to_gas_ratio = Zs;
        std::printf("PHASE %s -- calc_dust_temp\n", tag);
        ok &= report(rtd); ok &= report(rrh); ok &= report(rrd);
    }

    // ---------------- calc_H2_shielding_factor ----------------
    {
        Acc r("calc_H2_shielding");
        const double NHs[] = {0.0, 1e10, 1e14, 5e14, 1e17, 1e19, 1e21, 1e23, 1e25};
        const double h2s[] = {0.0, 1e-8, 1e-4, 0.1, 0.25, 0.5};
        for (size_t ti = 0; ti < T.size(); ti += 5)
        for (size_t ni = 0; ni < sizeof(NHs)/sizeof(NHs[0]); ni++)
        for (size_t hi = 0; hi < sizeof(h2s)/sizeof(h2s[0]); hi++) {
            ChemCell c; chem_bind_cell(c, TT, net);
            double a = fort_calc_H2_shielding_factor(T[ti], h2s[hi], NHs[ni]);
            double b = cxx_calc_H2_shielding_factor(T[ti], h2s[hi], NHs[ni], c);
            r.add(a, b);
        }
        std::printf("PHASE %s -- calc_H2_shielding_factor\n", tag);
        ok &= report(r);
    }

    // ---------------- lookup_temp_gamma ----------------
    {
        Acc rt("lookup_temp_gamma temp");
        Acc rg("lookup_temp_gamma gamma");
        const double h2s[] = {0.0, 1e-6, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4,
                              0.4999999, 0.5};
        static char hnames[10][24];
        std::vector<Acc> per;
        for (size_t hi = 0; hi < sizeof(h2s)/sizeof(h2s[0]); hi++) {
            std::snprintf(hnames[hi], sizeof(hnames[hi]), "  abh2=%g", h2s[hi]);
            per.push_back(Acc(hnames[hi]));
        }
        for (size_t hi = 0; hi < sizeof(h2s)/sizeof(h2s[0]); hi++)
        for (int i = 0; i <= 4000; i++) {
            double ekn = 0.5 + i * 12.5;         // spans below 1 and above 5e4
            double tF, gF, tC, gC;
            fort_lookup_temp_gamma(h2s[hi], ekn, &tF, &gF);
            cxx_lookup_temp_gamma(h2s[hi], CHEM_COMP.abhe, ekn, &tC, &gC, TT);
            rt.add(tF, tC); rg.add(gF, gC); per[hi].add(tF, tC);
        }
        std::printf("PHASE %s -- lookup_temp_gamma\n", tag);
        ok &= report(rt); ok &= report(rg);
        for (size_t hi = 0; hi < per.size(); hi++)
            if (per[hi].npass != per[hi].n) report(per[hi]);
    }

    // ---------------- calc_spec / compute_gamma ----------------
    {
        Acc rs("calc_spec");
        for (int i = 0; i <= 20000; i++) {
            double temp = 0.5 + i * 3.0;
            double en = 0; calc_spec_(&temp, &en);
            rs.add(en, cxx_calc_spec(TT, temp));
        }
        Acc rg("compute_gamma");
        const double h2s[] = {0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                              0.45, 0.5};
        for (size_t hi = 0; hi < sizeof(h2s)/sizeof(h2s[0]); hi++)
        for (int i = 0; i <= 2000; i++) {
            double ekn = std::pow(10.0, 1.0 + 4.0 * i / 2000.0);
            double a = 0, h = h2s[hi], e = ekn;
            compute_gamma_(&h, &e, &a);
            rg.add(a, cxx_compute_gamma(TT, h2s[hi], ekn));
        }
        std::printf("PHASE %s -- calc_spec / compute_gamma\n", tag);
        ok &= report(rs); ok &= report(rg);
    }

    // ---------------- calc_photo ----------------
    {
        Acc rp("calc_photo phrates");
        static char pn[NPHTAB][24];
        std::vector<Acc> per;
        for (int k = 0; k < NPHTAB; k++) {
            std::snprintf(pn[k], sizeof(pn[k]), "  phrates(%d)", k + 1);
            per.push_back(Acc(pn[k]));
        }
        Acc rgd("calc_photo G_dust");
        Acc rci("calc_photo chi_isrf");
        const double G0s[]  = {0.0, 1e-3, 1.7, 100.0};
        const double dgrs[] = {0.0, 1e-4, 0.1};
        const double NHcol[]= {0.0, 1e18, 1e21, 1e23};
        const int    iph[]  = {0, 6};
        const double h2s[]  = {0.0, 1e-4, 0.25, 0.5};
        const double cos_[] = {0.0, 1e-8, 1e-5};
        for (size_t gi = 0; gi < sizeof(G0s)/sizeof(G0s[0]); gi++)
        for (size_t di = 0; di < sizeof(dgrs)/sizeof(dgrs[0]); di++)
        for (size_t ci = 0; ci < sizeof(NHcol)/sizeof(NHcol[0]); ci++)
        for (size_t pi = 0; pi < sizeof(iph)/sizeof(iph[0]); pi++)
        for (size_t hi = 0; hi < sizeof(h2s)/sizeof(h2s[0]); hi++)
        for (size_t oi = 0; oi < sizeof(cos_)/sizeof(cos_[0]); oi++)
        for (size_t ti = 0; ti < T.size(); ti += 61) {
            double temp = T[ti], abh2 = h2s[hi], abhd = 1e-9, abco = cos_[oi];
            double dl = 1e18, yn = 1e3;
            COOLRC.G0 = COOLRC.G0_LW = COOLRC.G0_dust = G0s[gi];
            COOLRC.dust_to_gas_ratio = dgrs[di];
            COOLI.iphoto = iph[pi];
#if defined(RT_TREECOL_DUST_WEIGHTED) || defined(RT_TREECOL_H2_CO)
            for (int k = 0; k < NPIX; k++) {
                PROJECT.column_density_projection[k]    = NHcol[ci];
                PROJECT.column_density_projection_h2[k] = abh2 * NHcol[ci];
                PROJECT.column_density_projection_co[k] = abco * NHcol[ci];
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
                PROJECT.fac_uv[k] = 1.0 / NPIX;
#endif
            }
#endif
            double rpar[NRPAR]; std::memset(rpar, 0, sizeof(rpar));
            rpar[0] = yn; rpar[1] = dl;
            double phF[NPHTAB], GdF = 0, chF = 0;
            for (int k = 0; k < NPHTAB; k++) phF[k] = 0.0;
            calc_photo_(&temp, rpar, &abh2, &abhd, &abco, phF, &GdF, &chF);

            ChemCell c; chem_bind_cell(c, TT, net);
            c.yn = yn; c.dl = dl;
            for (int k = 0; k < CHEM_NPHTAB; k++) c.phrates[k] = 0.0;
            c.G_dust = 0.0; c.chi_isrf = 0.0;
            cxx_calc_photo(temp, abh2, abhd, abco, c);

            for (int k = 0; k < NPHTAB; k++) { rp.add(phF[k], c.phrates[k]);
                                               per[k].add(phF[k], c.phrates[k]); }
            rgd.add(GdF, c.G_dust);
            rci.add(chF, c.chi_isrf);
        }
        COOLRC.G0 = COOLRC.G0_LW = COOLRC.G0_dust = 1.7;
        COOLRC.dust_to_gas_ratio = Zs;
        std::printf("PHASE %s -- calc_photo\n", tag);
        ok &= report(rp); ok &= report(rgd); ok &= report(rci);
        for (int k = 0; k < NPHTAB; k++)
            if (per[k].npass != per[k].n) report(per[k]);
    }
    return ok;
}

// ===========================================================================
int main(int argc, char** argv)
{
    const char* datadir = (argc > 1) ? argv[1] : "gizmo/cooling/chemcool";
    int net = CHEMISTRYNETWORK;
    std::printf("=== GATE 3 GOLDEN -- network %d, tol %.0e ===\n", net, TOL);

    init_commons(1.7, 3e-17, 6);
    COOLINMO(); CHEMINMO();
    LOAD_H2_TABLE(); INIT_TEMPERATURE_LOOKUP();

    ChemBuildConfig cfg;
    cfg.net = net;
    cfg.iflag_atom = COOLI.iflag_atom;
    cfg.iflag_mn = COOLI.iflag_mn; cfg.iflag_ad = COOLI.iflag_ad;
    cfg.iflag_3bh2a = COOLI.iflag_3bh2a; cfg.iflag_3bh2b = COOLI.iflag_3bh2b;
    cfg.iflag_h3pra = COOLI.iflag_h3pra;
#ifdef WSS_CIE_COOL
    cfg.wss_cie = 1;
#else
    cfg.wss_cie = 0;
#endif
    cfg.deff = COOLR.deff; cfg.phi_pah = COOLR.phi_pah;
    cfg.dust_to_gas_ratio = COOLRC.dust_to_gas_ratio;
    cfg.cosmic_ray_ion_rate = COOLRC.cosmic_ray_ion_rate;
    cfg.abhe = CHEM_COMP.abhe;

    ChemTables Cx;
    if (chem_tables_build(Cx, cfg, datadir)) {
        std::printf("FATAL: chem_tables_build failed\n"); return 1;
    }

    int ok = 1;
    ok &= phase_A(Cx);

    ChemTables Ft; chem_bind_tables(Ft);
    std::printf("\n=== PHASE B: kernels on the FORTRAN tables ===\n");
    ok &= kernels(Ft, "B", net);

    std::printf("\n=== PHASE C: kernels on the C++-BUILT tables ===\n");
    ok &= kernels(Cx, "C", net);

    // ---- PHASE D: the iflag_atom = 1 cl6 branch (coolatom, 81 points) -----
    // Production uses iflag_atom = 2, so this branch was ported but never
    // golden-tested (DEVIATIONS "Open / deferred").  Run it LAST because
    // COOLINMO overwrites the Fortran cltab in place.
    std::printf("\n=== PHASE D: iflag_atom = 1 (coolatom spline + residual zeroing) ===\n");
    {
        COOLI.iflag_atom = 1;
        COOLINMO();
        ChemBuildConfig c1 = cfg; c1.iflag_atom = 1;
        ChemTables D1;
        if (chem_tables_build(D1, c1, datadir)) {
            std::printf("  FATAL: chem_tables_build(iflag_atom=1) failed\n");
            ok = 0;
        } else {
            Acc r6("cltab (iflag_atom=1)");
            const double* F = &COOLR.cltab[0][0];
            for (int i = 0; i < NMD; i++) for (int k = 0; k < NCLTAB; k++)
                r6.add(F[(size_t)i*NCLTAB + k], D1.cltab[i][k]);
            ok &= report(r6);
            chem_tables_free(D1);
        }
        COOLI.iflag_atom = 2;
    }

    std::printf("\n=== GATE 3 GOLDEN: %s ===\n", ok ? "ALL PASS" : "FAILURES PRESENT");
    chem_tables_free(Cx);
    return ok ? 0 : 1;
}
