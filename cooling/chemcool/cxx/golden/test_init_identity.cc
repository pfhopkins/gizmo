// test_init_identity.cc -- CHEMCOOL_CXX init vs the Fortran init it replaced.
//
// Runs the Fortran init exactly as chemcool_init did before (COMMON defaults,
// COOLINMO, CHEMINMO, INIT_TOLERANCES, LOAD_H2_TABLE, INIT_TEMPERATURE_LOOKUP),
// then the Fortran-free chem_cxx_tables_init(), and checks bitwise:
//   1. build config: COMMON-derived (old) vs ChemInitParams-derived (new)
//   2. cell template g_tmpl: old COMMON-read assembly vs new
//   3. C++ tables built from the old config vs the new GIZMO tables
//   4. init-derived scalars (abhe, X_H_chem, f_rsc) and untouched COMMON scalars
// and reports Fortran-built vs C++-built tables (bitwise count, max rel diff).
// Build: build_init_identity.sh.  Needs a GIZMO_config.h with CHEMCOOL_CXX.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>

// the backend's statics (g_tmpl) become visible in this translation unit
#include "cooling/chemcool/cxx/chem_gizmo_backend.cc"

extern "C" {
#include "cooling/chemcool/chem_commons.h"
extern CHEM_TLS_VARCOMP struct { double abhe; double X_H_chem; } CHEM_COMP;
void COOLINMO(void);
void CHEMINMO(void);
void INIT_TOLERANCES(void);
void LOAD_H2_TABLE(void);
void INIT_TEMPERATURE_LOOKUP(void);
void chemcool_fatal_(int* code) { std::printf("chemcool_fatal %d\n", *code); std::exit(2); }
}
void chem_bind_tables(ChemTables& T);                          // chem_backend_fortran.cc
void chem_bind_build_config(ChemBuildConfig& cfg, int net);    // chem_backend_fortran.cc

static int g_fail = 0;

static bool bits_eq(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

// identical bit patterns required
static void must_match(const char* name, const double* a, const double* b, size_t n)
{
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) if (!bits_eq(a[i], b[i])) bad++;
    std::printf("  %-22s n=%-8zu %s\n", name, n, bad ? "MISMATCH" : "bitwise identical");
    if (bad) { std::printf("    %zu differing entries\n", bad); g_fail = 1; }
}

// Fortran vs C++ table: bitwise count and max relative difference
static void report(const char* name, const double* f, const double* c, size_t n)
{
    size_t same = 0; double maxrel = 0, maxabs = 0; size_t imax = 0;
    for (size_t i = 0; i < n; i++) {
        if (bits_eq(f[i], c[i])) { same++; continue; }
        double d = std::fabs(f[i] - c[i]);
        double s = std::fmax(std::fabs(f[i]), std::fabs(c[i]));
        double r = s > 0 ? d / s : d;
        if (r > maxrel) { maxrel = r; imax = i; }
        if (d > maxabs) maxabs = d;
    }
    std::printf("  %-22s n=%-8zu bitwise=%-8zu maxrel=%.3e maxabs=%.3e", name, n, same, maxrel, maxabs);
    if (same != n) std::printf("  (worst i=%zu F=%.17g C=%.17g)", imax, f[imax], c[imax]);
    std::printf("\n");
}

#define TAB_LIST(X) \
    X(temptab, CHEM_NMD) X(crtab, CHEM_NCRTAB) X(crphot, CHEM_NCRPHOT) X(phtab, CHEM_NPHTAB) X(cst, CHEM_NCONST) \
    X(co_temptab, CHEM_NTCO) X(co_colntab, CHEM_NCDCO) X(co_L0, CHEM_NTCO) X(dTco_L0, CHEM_NTCO) \
    X(co_lte, CHEM_NTCO * CHEM_NCDCO) X(co_n05, CHEM_NTCO * CHEM_NCDCO) X(co_alp, CHEM_NTCO * CHEM_NCDCO) \
    X(dTco_lte, CHEM_NTCO * CHEM_NCDCO) X(dTco_n05, CHEM_NTCO * CHEM_NCDCO) X(dTco_alp, CHEM_NTCO * CHEM_NCDCO) \
    X(co_vib_temptab, CHEM_NTCO_VIB) X(co_vib_colntab, CHEM_NCDCO_VIB) \
    X(co_vib_LTE_final, CHEM_NTCO_VIB * CHEM_NCDCO_VIB) X(dTco_vib_LTE, CHEM_NTCO_VIB * CHEM_NCDCO_VIB) \
    X(h2o_temptab, CHEM_NTH2O) X(h2o_colntab, CHEM_NCDH2O) \
    X(h2o_L0_ortho, CHEM_NTH2O) X(dTh2o_L0_ortho, CHEM_NTH2O) X(h2o_L0_para, CHEM_NTH2O) X(dTh2o_L0_para, CHEM_NTH2O) \
    X(h2o_LTE_ortho, CHEM_NTH2O * CHEM_NCDH2O) X(dTh2o_LTE_ortho, CHEM_NTH2O * CHEM_NCDH2O) \
    X(h2o_n05_ortho, CHEM_NTH2O * CHEM_NCDH2O) X(dTh2o_n05_ortho, CHEM_NTH2O * CHEM_NCDH2O) \
    X(h2o_alp_ortho, CHEM_NTH2O * CHEM_NCDH2O) X(dTh2o_alp_ortho, CHEM_NTH2O * CHEM_NCDH2O) \
    X(h2o_LTE_para, CHEM_NTH2O * CHEM_NCDH2O) X(dTh2o_LTE_para, CHEM_NTH2O * CHEM_NCDH2O) \
    X(h2o_n05_para, CHEM_NTH2O * CHEM_NCDH2O) X(dTh2o_n05_para, CHEM_NTH2O * CHEM_NCDH2O) \
    X(h2o_alp_para, CHEM_NTH2O * CHEM_NCDH2O) X(dTh2o_alp_para, CHEM_NTH2O * CHEM_NCDH2O) \
    X(h2o_vib_temptab, CHEM_NTH2O_VIB) X(h2o_vib_colntab, CHEM_NCDH2O_VIB) \
    X(h2o_vib_LTE_final, CHEM_NTH2O_VIB * CHEM_NCDH2O_VIB) X(dTh2o_vib_LTE, CHEM_NTH2O_VIB * CHEM_NCDH2O_VIB) \
    X(h2_opac_temp, CHEM_NH2OP) X(h2_opac_column, CHEM_NH2OP) X(h2_opac, CHEM_NH2OP * CHEM_NH2OP) \
    X(CO_self_shielding, CHEM_IMAX_COSS) X(CO_H2_shielding, CHEM_IMAX_COH2) \
    X(chi_tab_mathis, CHEM_NCHI) X(chi_tab_black, CHEM_NCHI) \
    X(temp_table, CHEM_ISIZE) X(gamma_table, CHEM_ISIZE) X(eh2_table, CHEM_ISIZE2) X(hng, CHEM_NHNG)
#define WSS_LIST(X) \
    X(wss_C, CHEM_NWSS) X(wss_N, CHEM_NWSS) X(wss_O, CHEM_NWSS) X(wss_Ne, CHEM_NWSS) X(wss_Mg, CHEM_NWSS) \
    X(wss_Si, CHEM_NWSS) X(wss_S, CHEM_NWSS) X(wss_Ca, CHEM_NWSS) X(wss_Fe, CHEM_NWSS) \
    X(HeI_tbl, CHEM_NHE_TBL) X(HeII_tbl, CHEM_NHE_TBL)

static void compare_tables(const ChemTables& A, const ChemTables& B, bool strict)
{
    auto cmp = [&](const char* n, const double* a, const double* b, size_t k) {
        if (strict) must_match(n, a, b, k); else report(n, a, b, k);
    };
#define X(f, n) cmp(#f, A.f, B.f, (size_t)(n));
    TAB_LIST(X)
#ifdef WSS_CIE_COOL
    WSS_LIST(X)
#endif
#undef X
    cmp("chtab", &A.chtab[0][0], &B.chtab[0][0], (size_t)CHEM_NMD * CHEM_NCHTAB);
    cmp("dtchtab", &A.dtchtab[0][0], &B.dtchtab[0][0], (size_t)CHEM_NMD * CHEM_NCHTAB);
    cmp("cltab", &A.cltab[0][0], &B.cltab[0][0], (size_t)CHEM_NMD * CHEM_NCLTAB);
    cmp("dtcltab", &A.dtcltab[0][0], &B.dtcltab[0][0], (size_t)CHEM_NMD * CHEM_NCLTAB);
    cmp("dtlog", &A.dtlog, &B.dtlog, 1);
    cmp("tmin", &A.tmin, &B.tmin, 1);
    cmp("tmax", &A.tmax, &B.tmax, 1);
}

int main()
{
    const int net = CHEMISTRYNETWORK;
    std::printf("CHEMISTRYNETWORK=%d\n", net);

    // chemcool_init parameter set; DGR and zeta overridable via TEST_DGR / TEST_ZETA
    ChemInitParams p;
    p.iflag_mn = 4; p.iflag_ad = 6; p.iflag_atom = 2;
    p.iflag_3bh2a = 5; p.iflag_3bh2b = 4; p.iflag_h3pra = 1;
    p.iphoto = 6; p.irad_heat = 0; p.isrf_option = 1; p.no_chem = 0;
    p.h2_form_ex = 0.84; p.h2_form_kin = 0.12; p.deff = 1.0; p.phi_pah = 0.5;
    p.AV_conversion_factor = 5.348e-22; p.AV_ext = 0.0; p.dm_density = 0.0;
    p.pdv_term = 0.0; p.redshift = 0.0;
    p.abundD = 2.6e-5; p.G0 = 1.7; p.G0_LW = 1.7;
    p.cosmic_ray_ion_rate = 3e-17; p.cr_energy_density = 0.0; p.dust_to_gas_ratio = 1.0;
    if (const char* e = std::getenv("TEST_DGR")) p.dust_to_gas_ratio = std::atof(e);
    if (const char* e = std::getenv("TEST_ZETA")) p.cosmic_ray_ion_rate = std::atof(e);

    // ---- old: Fortran init exactly as the pre-port chemcool_init ----
    COOLI.iflag_mn = p.iflag_mn; COOLI.iflag_ad = p.iflag_ad; COOLI.iflag_atom = p.iflag_atom;
    COOLI.iflag_3bh2a = p.iflag_3bh2a; COOLI.iflag_3bh2b = p.iflag_3bh2b; COOLI.iflag_h3pra = p.iflag_h3pra;
    COOLI.iphoto = p.iphoto; COOLIC.irad_heat = p.irad_heat; COOLI.isrf_option = p.isrf_option; COOLIC.no_chem = p.no_chem;
    COOLR.h2_form_ex = p.h2_form_ex; COOLR.h2_form_kin = p.h2_form_kin; COOLR.deff = p.deff; COOLR.phi_pah = p.phi_pah;
    COOLR.AV_conversion_factor = p.AV_conversion_factor; COOLR.AV_ext = p.AV_ext; COOLR.dm_density = p.dm_density;
    COOLRC.pdv_term = p.pdv_term; COOLRC.redshift = p.redshift; COOLRC.abundD = p.abundD;
    COOLRC.G0 = p.G0; COOLRC.G0_LW = p.G0_LW; COOLRC.cosmic_ray_ion_rate = p.cosmic_ray_ion_rate;
    COOLRC.cr_energy_density = p.cr_energy_density; COOLRC.dust_to_gas_ratio = p.dust_to_gas_ratio;
    COOLINMO(); CHEMINMO(); INIT_TOLERANCES(); LOAD_H2_TABLE(); INIT_TEMPERATURE_LOOKUP();

    ChemBuildConfig cfg_old;
    chem_bind_build_config(cfg_old, net);
    ChemTables T_old;
    if (chem_tables_build(T_old, cfg_old, "gizmo/cooling/chemcool")) { std::printf("old table build failed\n"); return 2; }

    // ---- new: the Fortran-free init ----
    chem_cxx_tables_init(&p);
    const ChemTables& T_new = chem_cxx_tables();

    // old cell template, assembled from the COMMONs exactly as at 63cfeb88
    ChemCell o;
    chem_cxx_cell_defaults(o, chem_cxx_tables(), net);
    o.abhe = CHEM_COMP.abhe;
    o.iphoto = COOLI.iphoto; o.isrf_option = COOLI.isrf_option;
    o.iflag_mn = COOLI.iflag_mn; o.iflag_ad = COOLI.iflag_ad; o.iflag_atom = COOLI.iflag_atom;
    o.iflag_3bh2a = COOLI.iflag_3bh2a; o.iflag_3bh2b = COOLI.iflag_3bh2b;
    o.iflag_h3pra = COOLI.iflag_h3pra; o.iflag_h2opc = COOLI.iflag_h2opc;
    o.deff = COOLR.deff; o.phi_pah = COOLR.phi_pah;
    o.h2_form_ex = COOLR.h2_form_ex; o.h2_form_kin = COOLR.h2_form_kin;
    o.f_rsc = COOLR.f_rsc; o.AV_conversion_factor = COOLR.AV_conversion_factor; o.AV_ext = COOLR.AV_ext;
    o.abundN = COOLRC.abundN; o.G0_dust = COOLRC.G0_dust; o.cr_energy_density = COOLRC.cr_energy_density;
#ifdef WSS_CIE_COOL
    for (int k = 0; k < 12; k++) o.Zmass[k] = COOLRC.Zmass[k];
#endif
#if defined(RT_TREECOL_DUST_WEIGHTED) || defined(RT_TREECOL_H2_CO)
    o.npix = NPIX;
    for (int k = 0; k < NPIX; k++) o.fac_uv[k] = 1.0 / NPIX;
#else
    o.npix = 0;
#endif

    std::printf("[1] build config (old from COMMON, new from ChemInitParams)\n");
    ChemBuildConfig cfg_new;   // as chem_cxx_tables_init assembles it
    std::memset(&cfg_new, 0, sizeof(cfg_new));
    cfg_new.net = net; cfg_new.iflag_atom = p.iflag_atom; cfg_new.iflag_mn = p.iflag_mn; cfg_new.iflag_ad = p.iflag_ad;
    cfg_new.iflag_3bh2a = p.iflag_3bh2a; cfg_new.iflag_3bh2b = p.iflag_3bh2b; cfg_new.iflag_h3pra = p.iflag_h3pra;
#ifdef WSS_CIE_COOL
    cfg_new.wss_cie = 1;
#endif
    cfg_new.deff = p.deff; cfg_new.phi_pah = p.phi_pah; cfg_new.dust_to_gas_ratio = p.dust_to_gas_ratio;
    cfg_new.cosmic_ray_ion_rate = p.cosmic_ray_ion_rate; cfg_new.abhe = g_abhe;
    {
        int ok = cfg_old.net == cfg_new.net && cfg_old.iflag_atom == cfg_new.iflag_atom && cfg_old.iflag_mn == cfg_new.iflag_mn
              && cfg_old.iflag_ad == cfg_new.iflag_ad && cfg_old.iflag_3bh2a == cfg_new.iflag_3bh2a
              && cfg_old.iflag_3bh2b == cfg_new.iflag_3bh2b && cfg_old.iflag_h3pra == cfg_new.iflag_h3pra
              && cfg_old.wss_cie == cfg_new.wss_cie && bits_eq(cfg_old.deff, cfg_new.deff) && bits_eq(cfg_old.phi_pah, cfg_new.phi_pah)
              && bits_eq(cfg_old.dust_to_gas_ratio, cfg_new.dust_to_gas_ratio)
              && bits_eq(cfg_old.cosmic_ray_ion_rate, cfg_new.cosmic_ray_ion_rate) && bits_eq(cfg_old.abhe, cfg_new.abhe);
        std::printf("  config fields          %s (abhe old=%.17g new=%.17g)\n", ok ? "bitwise identical" : "MISMATCH", cfg_old.abhe, cfg_new.abhe);
        if (!ok) g_fail = 1;
    }

    std::printf("[2] cell template g_tmpl (old COMMON assembly vs new), %zu bytes\n", sizeof(ChemCell));
    {
        const unsigned char* a = reinterpret_cast<const unsigned char*>(&o);
        const unsigned char* b = reinterpret_cast<const unsigned char*>(&g_tmpl);
        size_t bad = 0;
        for (size_t i = 0; i < sizeof(ChemCell); i++) if (a[i] != b[i]) bad++;
        std::printf("  ChemCell bytes         %s (%zu differing bytes)\n", bad ? "MISMATCH" : "bitwise identical", bad);
        if (bad) g_fail = 1;
    }

    std::printf("[3] C++ tables: old-config build vs new GIZMO init\n");
    compare_tables(T_old, T_new, true);

    std::printf("[4] init-derived scalars (Fortran COMMON after init vs C++)\n");
    must_match("abhe", &CHEM_COMP.abhe, &g_tmpl.abhe, 1);
    { double x = chem_cxx_xh_default(); must_match("X_H_chem", &CHEM_COMP.X_H_chem, &x, 1); }
    { double a = chem_cxx_abhe_default(); must_match("abhe (fill_inputs)", &CHEM_COMP.abhe, &a, 1); }
    must_match("f_rsc", &COOLR.f_rsc, &g_tmpl.f_rsc, 1);
    must_match("dtlog", &COOLR.dtlog, &T_new.dtlog, 1);
    must_match("tmin", &COOLR.tmin, &T_new.tmin, 1);
    must_match("tmax", &COOLR.tmax, &T_new.tmax, 1);
    {
        int ok = COOLI.iflag_h2opc == 0 && COOLI.iphoto == p.iphoto && COOLI.isrf_option == p.isrf_option
              && bits_eq(COOLRC.abundN, 0.0) && bits_eq(COOLRC.G0_dust, 0.0) && bits_eq(COOLRC.cr_energy_density, p.cr_energy_density)
              && bits_eq(COOLR.deff, p.deff) && bits_eq(COOLR.phi_pah, p.phi_pah) && bits_eq(COOLR.AV_ext, p.AV_ext)
              && bits_eq(COOLR.h2_form_ex, p.h2_form_ex) && bits_eq(COOLR.h2_form_kin, p.h2_form_kin)
              && bits_eq(COOLRC.dust_to_gas_ratio, p.dust_to_gas_ratio) && bits_eq(COOLRC.cosmic_ray_ion_rate, p.cosmic_ray_ion_rate);
#ifdef WSS_CIE_COOL
        for (int k = 0; k < 12; k++) ok = ok && bits_eq(COOLRC.Zmass[k], 0.0);
#endif
        std::printf("  COMMON scalars after init == chemcool_init inputs: %s\n", ok ? "yes" : "NO");
        if (!ok) g_fail = 1;
    }

    std::printf("[5] Fortran-built (COOLINMO/CHEMINMO/...) vs C++-built tables\n");
    ChemTables T_f;
    chem_bind_tables(T_f);
    compare_tables(T_f, T_new, false);

    std::printf("%s\n", g_fail ? "RESULT: FAIL" : "RESULT: PASS (old and new CHEMCOOL_CXX init bitwise identical)");
    return g_fail;
}
