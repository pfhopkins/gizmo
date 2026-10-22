// test_onezone_ab.cc -- Fortran vs C++ one-zone golden for networks 3, 15 and 16.
//
// Links the canonical Fortran (rate_eq, EVOLVE_ABUNDANCES/DVODE, calc_photo,
// COOLINMO/CHEMINMO tables) and the C++ path GIZMO runs under CHEMCOOL_CXX
// (chem_tables_build, cxx_calc_photo, chem_rhs_net3/15/16, cxx_evolve_abundances
// on CVODE) into ONE binary.
//
//   PHASE R  RHS: rate_eq vs chem_rhs_netN at identical states, identical frozen
//            photorates and identical tdust.  R1 on the Fortran tables (isolates
//            the RHS port), R2 on the C++-built tables.  Tolerance 1e-12.
//   PHASE O  one-zone evolutions over an (nH, T0, G0, Z) grid, 80 log-spaced
//            steps to 300 Myr, each backend refreshing its own photorates and
//            carrying its own tdust.  Reports per-step T and final T/species.
//            Gate: final T within 1% (GATE2/GATE3 criterion (a)/(c)).
//
// Usage: test_onezone_ab [traj.csv]     (run where gizmo/cooling/chemcool exists)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <csetjmp>
#include <vector>
#include <algorithm>

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
void INIT_TOLERANCES(void);
void LOAD_H2_TABLE(void);
void INIT_TEMPERATURE_LOOKUP(void);
void EVOLVE_ABUNDANCES(double* timestep, double* dl, double* yn, double* divv,
                       double* energy, double* abundances, double* column_est);
void CALC_PHOTO_WRAPPER(double* temp, double* rpar, double* abh2, double* abhd,
                        double* abco);
void RATE_EQ(int* neq, double* time, double* y, double* ydot, double* rpar,
             int* ipar);

static jmp_buf g_fatal_jmp;
static int g_fatal_armed = 0;
void chemcool_fatal_(int* code)
{
    std::printf("  [Fortran chemcool_fatal %d]\n", *code);
    if (g_fatal_armed) longjmp(g_fatal_jmp, 1);
    std::exit(1);
}
}

#if CHEMISTRYNETWORK != 3 && CHEMISTRYNETWORK != 15 && CHEMISTRYNETWORK != 16
#error "test_onezone_ab covers networks 3, 15 and 16"
#endif

void chem_bind_tables(ChemTables& T);
void chem_bind_cell(ChemCell& c, const ChemTables& T, int net);

static const int NET = CHEMISTRYNETWORK;
static const double SEC_PER_MYR = 3.156e13;
static const double DL_SHIELD = 3.086e19;   // 10 pc TreeCol column length
#if CHEMISTRYNETWORK == 3
static const char* SPNAME[NSPEC] = {"H2", "H+", "C+", "Si+", "O+", "D+", "HD", "He+", "He++",
                                    "CO", "C2", "OH", "H2O", "O2", "HCO+", "CH", "CH2", "Si++",
                                    "CH3+", "e"};
#else
static const char* SPNAME[NSPEC] = {"H2", "H+", "C+", "CHx", "OHx", "CO", "HCO+",
                                    "He+", "M+", "e"};
#endif

// elemental pool of each tracked species (fractions in the mixes below are of these)
static void species_pools(double* pool)
{
    const double C = COOLRC.abundc, O = COOLRC.abundo, CO = std::min(C, O);
#if CHEMISTRYNETWORK == 3
    const double Si = COOLRC.abundsi, D = COOLRC.abundD, He = CHEM_COMP.abhe;
    const double p[TRAC_NUM] = {0.5, 1.0, C, Si, O, D, D, He, He, CO, 0.5 * C, O, O,
                                0.5 * O, CO, C, C, Si, C};
#else
    const double p[TRAC_NUM] = {0.5, 1.0, C, C, O, CO, CO, CHEM_COMP.abhe, COOLRC.abundM};
#endif
    for (int i = 0; i < TRAC_NUM; i++) pool[i] = p[i];
}

// ---------------------------------------------------------------------------
static double g_zeta = 2e-16;

static void set_cell_commons(double G0, double Zs)
{
    COOLRC.G0 = G0; COOLRC.G0_LW = G0; COOLRC.G0_dust = G0;
    COOLRC.cosmic_ray_ion_rate = g_zeta;
    COOLRC.cr_energy_density = 0.0;
    COOLRC.dust_to_gas_ratio = Zs;
    COOLRC.redshift = 0.0; COOLRC.pdv_term = 0.0;
    COOLRC.abundD  = 2.6e-5;
    COOLRC.abundc  = 1.4e-4 * Zs;
    COOLRC.abundo  = 3.2e-4 * Zs;
    COOLRC.abundsi = 1.5e-5 * Zs;
    COOLRC.abundM  = COOLRC.abundsi;
    COOLRC.abundN  = 7.6e-5 * Zs;
    COOLRC.rt_phot_HI = COOLRC.rt_phot_HeI = COOLRC.rt_phot_HeII = 0;
    COOLRC.rt_heat_HI = COOLRC.rt_heat_HeI = COOLRC.rt_heat_HeII = 0;
#ifdef WSS_CIE_COOL
    const double zsolar[12] = {0.7065, 2.07e-3, 5.91e-4, 5.49e-3, 1.10e-3, 6.83e-4,
                               0.2806, 8.36e-4, 1.41e-3, 4.09e-4, 6.44e-5, 1.10e-3};
    for (int k = 0; k < 12; k++) COOLRC.Zmass[k] = zsolar[k];
    for (int k = 0; k < 12; k++) if (k != 0 && k != 6) COOLRC.Zmass[k] *= Zs;
#endif
}

static void init_commons()
{
    COOLI.iflag_mn = 4; COOLI.iflag_ad = 6; COOLI.iflag_atom = 2;
    COOLI.iflag_3bh2a = 5; COOLI.iflag_3bh2b = 4; COOLI.iflag_h3pra = 1;
    COOLI.iphoto = 6;
    COOLIC.irad_heat = 0; COOLI.isrf_option = 1; COOLIC.no_chem = 0;
    COOLIC.id_current = 1; COOLIC.index_current = 0;
    COOLI.iflag_h2opc = 0; COOLI.idma_mass_option = 1;
    COOLR.h2_form_ex = 0.84; COOLR.h2_form_kin = 0.12;
    COOLR.deff = 1.0; COOLR.phi_pah = 0.5;
    COOLR.AV_conversion_factor = 5.348e-22;
    COOLR.AV_ext = 0.0; COOLR.dm_density = 0.0;
    COOLRC.tdust = 10.0;
    set_cell_commons(1.7, 0.1);
}

static void set_columns(double yn, double abh2, double abco)
{
#if defined(RT_TREECOL_DUST_WEIGHTED) || defined(RT_TREECOL_H2_CO)
    double NH = yn * DL_SHIELD;
    for (int k = 0; k < NPIX; k++) {
        PROJECT.column_density_projection[k]    = NH;
        PROJECT.column_density_projection_h2[k] = abh2 * NH;
        PROJECT.column_density_projection_co[k] = abco * NH;
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
        PROJECT.fac_uv[k] = 1.0 / NPIX;
#endif
    }
#else
    (void)yn; (void)abh2; (void)abco;
#endif
}

// electrons and temperature exactly as the RHS (before O+/H3+)
static double abe_of(const double* y)
{
#if CHEMISTRYNETWORK == 3
    double e = y[IHP] + y[IDP] + y[IHEP] + y[IC] + y[IO] + y[ISi] + y[IHCOP]
             + 2.0 * y[ISIPP] + y[ICH3P] + 2.0 * y[IHEPP];
#else
    double e = y[IHP] + y[IC] + y[IHCOP] + y[IHEP] + y[IMP];
#endif
    return e > 0.0 ? e : 0.0;
}
static double temp_of(const ChemTables& T, double yn, double energy, const double* y)
{
    double yntot = (1.0 + CHEM_COMP.abhe - y[IH2] + abe_of(y)) * yn;
    double t, g;
    cxx_lookup_temp_gamma(y[IH2], CHEM_COMP.abhe, energy / (yntot * CHEM_KBOLTZ),
                          &t, &g, T);
    return t;
}
static double energy_at(const ChemTables& T, double yn, double T0, const double* y)
{
    double yntot = (1.0 + CHEM_COMP.abhe - y[IH2] + abe_of(y)) * yn;
    double e = 1.5 * yntot * CHEM_KBOLTZ * T0;
    for (int k = 0; k < 40; k++) {
        double t = temp_of(T, yn, e, y);
        if (std::fabs(t - T0) < 1e-10 * T0) break;
        e *= T0 / t;
    }
    return e;
}

// ===========================================================================
// PHASE R -- RHS golden
// ===========================================================================
struct Acc {
    const char* name; long n = 0, npass = 0; double worst = 0.0, wa = 0, wb = 0;
    explicit Acc(const char* nm) : name(nm) {}
    void add(double a, double b, double tol) {
        n++;
        double rel;
        if (a == b) rel = 0.0;
        else if (!std::isfinite(a) || !std::isfinite(b)) rel = 1e300;
        else {
            double s = std::max(std::fabs(a), std::fabs(b));
            rel = std::fabs(a - b) / s;
        }
        if (rel <= tol) npass++;
        if (rel > worst) { worst = rel; wa = a; wb = b; }
    }
};
static int report(const Acc& r)
{
    int ok = (r.npass == r.n);
    std::printf("  %-26s %8ld/%-8ld %s   worst rel = %.3e", r.name, r.npass, r.n,
                ok ? "PASS" : "FAIL", r.worst);
    if (!ok) std::printf("   (F=%.17g  C=%.17g)", r.wa, r.wb);
    std::printf("\n");
    return ok;
}

static int phase_R(const ChemTables& TT, const char* tag)
{
    const double TOL = 1e-12;
    std::vector<Acc> acc;
    static char nm[NSPEC][32];
    for (int i = 0; i < NSPEC; i++) {
        std::snprintf(nm[i], sizeof(nm[i]), "ydot(%s)", i == ITMP ? "energy" : SPNAME[i]);
        acc.push_back(Acc(nm[i]));
    }
    Acc rlc("lambda_chem(all)"), rtd("tdust after RHS"), rfl("flag (0/1)");

    const double yns[]  = {1e-2, 1.0, 30.0, 1e3, 1e5, 1e7};
    const double ekns[] = {3.0, 15.0, 75.0, 400.0, 1.5e3, 7e3, 1.5e4, 6e4, 3e5, 1.5e6, 2e7};
#if CHEMISTRYNETWORK == 3
    // G0 = 0 not gated: calc_photo returns before setting chi_isrf, which rate_eq_mol passes on uninitialized
    const double G0s[]  = {1e-4, 1.7, 1e3};
#else
    const double G0s[]  = {0.0, 1.7, 1e3};
#endif
    const double Zss[]  = {0.1, 1.0};
#if CHEMISTRYNETWORK == 3
    // fractions of species_pools(): H2 H+ C+ Si+ O+ D+ HD He+ He++ CO C2 OH H2O O2 HCO+ CH CH2 Si++ CH3+
    const double mix[][TRAC_NUM] = {
        {2e-6, 1e-4, 1.0, 1.0, 1e-4, 1e-4, 0.0, 1e-3, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},      // atomic
        {0.99, 1e-8, 1e-3, 0.05, 1e-8, 1e-8, 0.5, 1e-9, 0.0, 0.9, 1e-4, 1e-3, 1e-2, 1e-3,
         1e-6, 1e-4, 1e-5, 0.0, 1e-7},                                                         // molecular
        {0.5, 1e-6, 0.4, 0.5, 1e-6, 0.01, 0.3, 1e-7, 0.0, 0.3, 1e-3, 1e-2, 1e-2, 1e-3,
         1e-5, 1e-3, 1e-3, 1e-3, 1e-5},                                                        // mixed
        {0.0, 0.999, 1.0, 0.5, 0.9, 0.99, 0.0, 0.9, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0},     // ionized
        {0.2, 0.01, 0.1, 0.3, 0.05, 0.1, 0.1, 0.1, 0.01, 0.1, 0.05, 0.2, 0.2, 0.05,
         0.05, 0.1, 0.1, 0.1, 0.05},                                                           // minor-species heavy
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},                             // all neutral
        {1.0001, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0},                      // H2 clip
        {1.0004, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0},                      // H2 flag
        {0.5, 1.0002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},                      // H+ flag
        {0.3, -1e-9, 0.5, 0.5, -1e-12, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0, -1e-14, 0.01, 0.0,
         0.0, 1e-3, 0.0, 0.0, 0.0},                                                            // small negatives
    };
#else
    // fractions of the pools: H2/0.5, H+, C+/C, CHx/C, OHx/O, CO/C, HCO+/C, He+/He, M+/M
    const double mix[][9] = {
        {2e-6, 1e-4,   1.0,   0.0,  0.0,  0.0,   0.0,  0.0,  1.0},   // atomic, C+ / M+
        {0.99, 1e-8,   1e-3,  1e-4, 1e-3, 0.95,  1e-6, 1e-9, 0.05},  // molecular, CO-rich
        {0.5,  1e-6,   0.4,   1e-3, 1e-2, 0.3,   1e-5, 1e-7, 0.5},   // H2 / C+ / CO mix
        {0.0,  0.999,  1.0,   0.0,  0.0,  0.0,   0.0,  0.9,  1.0},   // ionized
        {0.2,  0.01,   0.1,   0.2,  0.3,  0.1,   0.05, 0.1,  0.3},   // CHx/OHx/HCO+ heavy
        {0.0,  0.0,    0.0,   0.0,  0.0,  0.0,   0.0,  0.0,  0.0},   // all neutral, no ions
        {1.0001,0.0,   0.0,   0.0,  0.0,  1.0,   0.0,  0.0,  0.0},   // H2 slightly > 0.5 (clip)
        {1.0004,0.0,   0.0,   0.0,  0.0,  1.0,   0.0,  0.0,  0.0},   // H2 > 0.5 + eps_max (flag)
        {0.5,  1.0002, 0.0,   0.0,  0.0,  0.0,   0.0,  0.0,  0.0},   // H+ excess -> flagged
        {0.3, -1e-9,   0.5,  -1e-12, 0.0, 0.2,   0.0,  0.0,  0.5},   // small negatives (clip)
    };
#endif
    const int nmix = sizeof(mix) / sizeof(mix[0]);
    long nflag = 0, nabort = 0;

    for (size_t zi = 0; zi < sizeof(Zss)/sizeof(Zss[0]); zi++)
    for (size_t gi = 0; gi < sizeof(G0s)/sizeof(G0s[0]); gi++) {
        set_cell_commons(G0s[gi], Zss[zi]);
        double pool[TRAC_NUM];
        species_pools(pool);
#if CHEMISTRYNETWORK != 3
        pool[5] = pool[6] = COOLRC.abundc;   // phase R grid as before: CO, HCO+ fractions of C
#endif
        for (size_t ni = 0; ni < sizeof(yns)/sizeof(yns[0]); ni++)
        for (int mi = 0; mi < nmix; mi++)
        for (size_t ei = 0; ei < sizeof(ekns)/sizeof(ekns[0]); ei++) {
            double yn = yns[ni];
            double y[NSPEC];
            for (int i = 0; i < TRAC_NUM; i++) y[i] = mix[mi][i] * pool[i];
            double yntot = (1.0 + CHEM_COMP.abhe - y[IH2] + abe_of(y)) * yn;
            y[ITMP] = ekns[ei] * yntot * CHEM_KBOLTZ;

            // frozen photo state from the Fortran calc_photo (both sides read it)
            double tguess = temp_of(TT, yn, y[ITMP], y);
            set_columns(yn, std::max(y[IH2], 0.0), std::max(y[ICO], 0.0));
            double rpar[NRPAR]; std::memset(rpar, 0, sizeof(rpar));
            rpar[0] = yn; rpar[1] = 0.0; rpar[2] = 0.0;
            double a2 = std::max(y[IH2], 0.0), ahd = 0.0, aco = std::max(y[ICO], 0.0);
            COOLRC.cosmic_ray_ion_rate = g_zeta;
            CALC_PHOTO_WRAPPER(&tguess, rpar, &a2, &ahd, &aco);

            ChemCell c; chem_bind_cell(c, TT, NET);
            c.yn = yn; c.dl = 0.0; c.divv = 0.0;
            const double td0 = 17.3;
            c.tdust = td0; COOLRC.tdust = td0;

            double yF[NSPEC], ydF[NSPEC], ydC[NSPEC];
            for (int i = 0; i < NSPEC; i++) { yF[i] = y[i]; ydF[i] = -7.0; ydC[i] = -9.0; }
            int neq = NSPEC, ipar[NIPAR] = {0, 0};
            double t0 = 0.0;
            for (int k = 0; k < NRATES_CHEM; k++) COOLRC.lambda_chem[k] = 0.0;   // flagged evals leave it unwritten
            int aborted = 0;
            g_fatal_armed = 1;
            if (setjmp(g_fatal_jmp) == 0) RATE_EQ(&neq, &t0, yF, ydF, rpar, ipar);
            else aborted = 1;                  // comp_equilib / index ABORT
            g_fatal_armed = 0;
            int rc = NET == 3 ? chem_rhs_net3(0.0, y, ydC, c)
                   : NET == 16 ? chem_rhs_net16(0.0, y, ydC, c) : chem_rhs_net15(0.0, y, ydC, c);

            int fF = (ipar[0] != 0) || aborted, fC = rc != 0;
            rfl.add(fF, fC, 0.0);
            nflag += fF; nabort += aborted;
            if (aborted) continue;             // Fortran ydot unspecified after an ABORT
            for (int i = 0; i < NSPEC; i++) acc[i].add(ydF[i], ydC[i], TOL);
            for (int k = 0; k < NRATES_CHEM; k++) rlc.add(COOLRC.lambda_chem[k], c.lambda_chem[k], TOL);
            rtd.add(COOLRC.tdust, c.tdust, TOL);
        }
    }
    std::printf("PHASE %s -- RHS (rate_eq vs chem_rhs_net%d), %ld flagged states (%ld Fortran ABORTs)\n",
                tag, NET, nflag, nabort);
    int ok = 1;
    for (int i = 0; i < NSPEC; i++) ok &= report(acc[i]);
    ok &= report(rlc); ok &= report(rtd); ok &= report(rfl);
    return ok;
}

// ===========================================================================
// PHASE O -- one-zone evolutions
// ===========================================================================
struct Point { double nH, T0, G0, Zs; int start; };   // start 0 atomic, 1 molecular

static void initial_state(int start, double* y)
{
    for (int i = 0; i < TRAC_NUM; i++) y[i] = 0.0;
#if CHEMISTRYNETWORK == 3
    if (start == 0) {
        y[IH2] = 1e-6; y[IHP] = 1e-4;
        y[IC] = COOLRC.abundc; y[ISi] = COOLRC.abundsi;
    } else {
        y[IH2] = 0.49; y[IHP] = 1e-6;
        y[IC] = 0.01 * COOLRC.abundc; y[ICO] = 0.5 * COOLRC.abundc;
        y[ISi] = 0.01 * COOLRC.abundsi; y[IHD] = 0.5 * COOLRC.abundD;
    }
#else
    if (start == 0) {
        y[IH2] = 1e-6; y[IHP] = 1e-4;
        y[IC] = COOLRC.abundc; y[IMP] = COOLRC.abundM;
    } else {
        y[IH2] = 0.49; y[IHP] = 1e-6;
        y[IC] = 0.01 * COOLRC.abundc; y[ICO] = 0.5 * COOLRC.abundc;
        y[IMP] = 0.01 * COOLRC.abundM;
    }
#endif
}

int main(int argc, char** argv)
{
    const char* trajf = (argc > 1) ? argv[1] : "onezone_ab_traj.csv";
    std::printf("=== ONE-ZONE A/B GOLDEN -- network %d (Fortran DVODE vs C++ CVODE) ===\n", NET);

    init_commons();
    COOLINMO(); CHEMINMO(); INIT_TOLERANCES();
    LOAD_H2_TABLE(); INIT_TEMPERATURE_LOOKUP();
#if defined(RT_TREECOL_DUST_WEIGHTED) || defined(RT_TREECOL_H2_CO)
    std::memset(&PROJECT, 0, sizeof(PROJECT));
#endif

    ChemBuildConfig cfg;
    cfg.net = NET;
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
    if (chem_tables_build(Cx, cfg, "gizmo/cooling/chemcool")) {
        std::printf("FATAL: chem_tables_build failed\n"); return 1;
    }
    ChemTables Ft; chem_bind_tables(Ft);

    int ok = 1;
    ok &= phase_R(Ft, "R1 (Fortran tables)");
    ok &= phase_R(Cx, "R2 (C++-built tables)");

    // ---------------- PHASE O ----------------
    std::vector<Point> pts;
    const double nHs[] = {0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5};
    const double G0s[] = {0.1, 1.7, 17.0};
    const double Zss[] = {0.1, 1.0};
    for (double Z : Zss) for (double G0 : G0s) for (double nH : nHs) {
        pts.push_back({nH, 1e4, G0, Z, 0});
        pts.push_back({nH, 50.0, G0, Z, 1});
    }
    pts.push_back({1.0, 1e6, 1.7, 1.0, 0});
    pts.push_back({100.0, 1e6, 1.7, 1.0, 0});
    pts.push_back({1e3, 3e5, 1.7, 0.1, 0});

    const int NSTEP_OZ = 80;
    const double t0 = 1e3 * 3.156e7, t1 = 300.0 * SEC_PER_MYR;
    std::vector<double> tgrid(NSTEP_OZ + 1, 0.0);
    for (int k = 1; k <= NSTEP_OZ; k++) tgrid[k] = t0 * std::pow(t1 / t0, (k - 1.0) / (NSTEP_OZ - 1.0));

    FILE* tf = std::fopen(trajf, "w");
    if (tf) {
        std::fprintf(tf, "point,nH,T0,G0,Z,start,step,t_Myr,backend,T,energy");
        for (int i = 0; i < TRAC_NUM; i++) std::fprintf(tf, ",%s", SPNAME[i]);
        std::fprintf(tf, ",tdust\n");
    }

    double worst_T_final = 0.0, worst_T_traj = 0.0;
    int worst_pf = -1, worst_pt = -1, nfail_gate = 0, nrunfail = 0;
    double worst_sp[TRAC_NUM], worst_sp_raw[TRAC_NUM];
    int worst_sp_pt[TRAC_NUM];
    for (int i = 0; i < TRAC_NUM; i++) { worst_sp[i] = 0; worst_sp_raw[i] = 0; worst_sp_pt[i] = -1; }
#if CHEMISTRYNETWORK == 3
    const double atolv[TRAC_NUM] = {1e-7, 1e-12, 1e-16, 1e-16, 1e-16, 1e-14, 1e-12, 1e-14, 1e-14,
                                    1e-14, 1e-14, 1e-14, 1e-14, 1e-14, 1e-18, 1e-18, 1e-18, 1e-14, 1e-18};
#else
    const double atolv[TRAC_NUM] = {1e-7, 1e-12, 1e-16, 1e-14, 1e-14, 1e-14, 1e-18, 1e-14, 1e-14};
#endif

    std::printf("\nPHASE O -- one-zone evolutions, %zu points x %d steps to %.0f Myr, zeta = %.1e, TreeCol L = 10 pc\n",
                pts.size(), NSTEP_OZ, t1 / SEC_PER_MYR, g_zeta);
    std::printf("  %3s %8s %7s %5s %4s %2s | %11s %11s %9s | %9s %9s | rc F/C\n",
                "pt", "nH", "T0", "G0", "Z", "st", "T_F", "T_C", "dT/T", "max traj", "T_drift");

    for (size_t p = 0; p < pts.size(); p++) {
        const Point& P = pts[p];
        set_cell_commons(P.G0, P.Zs);
        double pool[TRAC_NUM];
        species_pools(pool);
        double yn = P.nH;
        double yF[TRAC_NUM], yC[TRAC_NUM];
        initial_state(P.start, yF);
        for (int i = 0; i < TRAC_NUM; i++) yC[i] = yF[i];
        double eF = energy_at(Cx, yn, P.T0, yF), eC = eF;
        double tdF = 10.0, tdC = 10.0;
        double traj = 0.0, TFprev = P.T0;
        int rcF = 0, rcC = 0;
        double TF = P.T0, TC = P.T0;

        for (int k = 1; k <= NSTEP_OZ; k++) {
            double dt = tgrid[k] - tgrid[k - 1];
            if (k == 1) dt = tgrid[1];
            // ---- Fortran ----
            if (!rcF) {
                double Tf = temp_of(Cx, yn, eF, yF);
                set_columns(yn, yF[IH2], yF[ICO]);
                double rpar[NRPAR]; std::memset(rpar, 0, sizeof(rpar));
                rpar[0] = yn;
                double a2 = yF[IH2], ahd = 0.0, aco = yF[ICO];
                COOLRC.cosmic_ray_ion_rate = g_zeta;
                COOLRC.tdust = tdF;
                CALC_PHOTO_WRAPPER(&Tf, rpar, &a2, &ahd, &aco);
                double dts = dt, dl = 0.0, dv = 0.0, ce = 0.0;
                g_fatal_armed = 1;
                if (setjmp(g_fatal_jmp) == 0) {
                    EVOLVE_ABUNDANCES(&dts, &dl, &yn, &dv, &eF, yF, &ce);
                } else {
                    rcF = 1;
                }
                g_fatal_armed = 0;
                tdF = COOLRC.tdust;
            }
            // ---- C++ ----
            if (!rcC) {
                double Tc = temp_of(Cx, yn, eC, yC);
                set_columns(yn, yC[IH2], yC[ICO]);
                COOLRC.cosmic_ray_ion_rate = g_zeta;
                ChemCell c; chem_bind_cell(c, Cx, NET);
                c.yn = yn; c.dl = 0.0; c.divv = 0.0; c.tdust = tdC;
                cxx_calc_photo(Tc, yC[IH2], 0.0, yC[ICO], c);
                rcC = cxx_evolve_abundances(dt, 0.0, yn, 0.0, &eC, yC, c);
                tdC = c.tdust;
            }
            TFprev = TF;
            TF = temp_of(Cx, yn, eF, yF);
            TC = temp_of(Cx, yn, eC, yC);
            double d = std::fabs(TC - TF) / TF;
            if (d > traj) traj = d;
            if (tf) {
                for (int b = 0; b < 2; b++) {
                    const double* yy = b ? yC : yF;
                    std::fprintf(tf, "%zu,%g,%g,%g,%g,%d,%d,%.6e,%s,%.9e,%.9e", p, P.nH, P.T0,
                                 P.G0, P.Zs, P.start, k, tgrid[k] / SEC_PER_MYR,
                                 b ? "cxx" : "f77", b ? TC : TF, b ? eC : eF);
                    for (int i = 0; i < TRAC_NUM; i++) std::fprintf(tf, ",%.9e", yy[i]);
                    std::fprintf(tf, ",%.6e\n", b ? tdC : tdF);
                }
            }
        }
        double dT = std::fabs(TC - TF) / TF;
        double drift = std::fabs(TF - TFprev) / TF;
        std::printf("  %3zu %8.1e %7.0e %5.1f %4.1f %2d | %11.5g %11.5g %9.2e | %9.2e %9.2e | %d/%d\n",
                    p, P.nH, P.T0, P.G0, P.Zs, P.start, TF, TC, dT, traj, drift, rcF, rcC);
        if (rcF || rcC) { nrunfail++; continue; }
        if (dT > 0.01) nfail_gate++;
        if (dT > worst_T_final) { worst_T_final = dT; worst_pf = (int)p; }
        if (traj > worst_T_traj) { worst_T_traj = traj; worst_pt = (int)p; }
        for (int i = 0; i < TRAC_NUM; i++) {
            double fl = std::max(atolv[i], 1e-6 * pool[i]);
            double a = std::fabs(yF[i]), b = std::fabs(yC[i]);
            double r = std::fabs(yF[i] - yC[i]) / std::max(std::max(a, b), fl);
            double rr = std::fabs(yF[i] - yC[i]) / std::max(std::max(a, b), 1e-300);
            if (r > worst_sp[i]) { worst_sp[i] = r; worst_sp_pt[i] = (int)p; }
            if (rr > worst_sp_raw[i]) worst_sp_raw[i] = rr;
        }
    }
    if (tf) std::fclose(tf);

    std::printf("\n  final T: worst |dT|/T = %.3e (pt %d); trajectory worst = %.3e (pt %d)\n",
                worst_T_final, worst_pf, worst_T_traj, worst_pt);
    std::printf("  final species (rel. above max(atol, 1e-6*pool) | raw rel):\n");
    for (int i = 0; i < TRAC_NUM; i++)
        std::printf("    %-5s %.3e (pt %d) | %.3e\n", SPNAME[i], worst_sp[i], worst_sp_pt[i],
                    worst_sp_raw[i]);
    std::printf("  GATE final T within 1%%: %s (%d over, %d run failures)\n",
                (nfail_gate == 0 && nrunfail == 0) ? "PASS" : "FAIL", nfail_gate, nrunfail);
    ok &= (nfail_gate == 0 && nrunfail == 0);

    std::printf("\n=== ONE-ZONE A/B GOLDEN: %s ===\n", ok ? "ALL PASS" : "FAILURES PRESENT");
    chem_tables_free(Cx);
    return ok ? 0 : 1;
}
