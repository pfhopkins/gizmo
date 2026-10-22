// test_conservation.cc -- element/charge conservation audit + Fortran/C++ parity,
// networks 3, 5, 15, 16, 17 (one binary per net, both backends linked).
//
//   PHASE R  RHS parity: rate_eq vs chem_rhs_netN on a state grid, frozen
//            photorates and tdust, tolerance 1e-12 (as test_onezone_ab).
//   PHASE C  conservation: each grid state (nH, T, G0, Z, start, dt) is passed
//            through the chemcool.cc pre-solve clamps (net-17 molecule zeroing,
//            chemcool_clamp_pools) and one driver call per backend
//            (EVOLVE_ABUNDANCES / cxx_evolve_abundances).  Per element, with
//            TRUE species content and no floors:
//              over = max(0, sum_i n_i y_i - pool) / pool   (pool overdraft)
//              neg  = sum_i n_i max(0, -y_i) / pool         (negative species)
//            before and after each call, plus the mass each pre-solve clamp
//            removes.  Probes (only in the instrumented build copy, see
//            build_conservation.sh) count every derived-species floor in the
//            C++ RHS, every check_abundances clip, and the QSS H3+ stock.
//   PHASE O  one-zone trajectory parity (as test_onezone_ab PHASE O): final T
//            within 1% between the backends.
//   PHASE I  chemcool_set_ionized with a per-cell abhe != ABHE.
//
// Usage: test_conservation [conservation.csv]   (run where gizmo/cooling/chemcool exists)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <csetjmp>
#include <vector>
#include <map>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
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

// the chemcool.cc pre-solve clamps, compiled from the real header
#define CHEM_INTERFACE_H
struct ChemInputs { double abundc, abundo, abundsi, abundD, abhe, abundM; };
#include "cooling/chemcool/chemcool_network.h"

#if CHEMISTRYNETWORK != 3 && CHEMISTRYNETWORK != 5 && CHEMISTRYNETWORK != 15 && CHEMISTRYNETWORK != 16 && CHEMISTRYNETWORK != 17
#error "test_conservation covers networks 3, 5, 15, 16, 17"
#endif

void chem_bind_tables(ChemTables& T);
void chem_bind_cell(ChemCell& c, const ChemTables& T, int net);

static const int NET = CHEMISTRYNETWORK;
static const double SEC_PER_MYR = 3.156e13;
static const double DL_SHIELD = 3.086e19;
static double g_zeta = 2e-16;

// ---------------------------------------------------------------------------
// species content: elements H He C O M D, charge q
// ---------------------------------------------------------------------------
enum { EH = 0, EHE, EC, EO, EM, ED, NEL };
static const char* ELNAME[NEL] = {"H", "He", "C", "O", "M", "D"};
struct Sp { const char* name; int n[NEL]; int q; };
#if CHEMISTRYNETWORK == 5
static const Sp SP[TRAC_NUM] = {{"H2", {2, 0, 0, 0, 0, 0}, 0}, {"H+", {1, 0, 0, 0, 0, 0}, 1},
                                {"CO", {0, 0, 1, 1, 0, 0}, 0}};
#elif CHEMISTRYNETWORK == 17
static const Sp SP[TRAC_NUM] = {{"H2", {2, 0, 0, 0, 0, 0}, 0}, {"H+", {1, 0, 0, 0, 0, 0}, 1},
                                {"CO", {0, 0, 1, 1, 0, 0}, 0}, {"He+", {0, 1, 0, 0, 0, 0}, 1},
                                {"He++", {0, 1, 0, 0, 0, 0}, 2}, {"D+", {0, 0, 0, 0, 0, 1}, 1},
                                {"HD", {1, 0, 0, 0, 0, 1}, 0}};
#elif CHEMISTRYNETWORK == 3   /* M = Si */
static const Sp SP[TRAC_NUM] = {{"H2", {2, 0, 0, 0, 0, 0}, 0}, {"H+", {1, 0, 0, 0, 0, 0}, 1},
                                {"C+", {0, 0, 1, 0, 0, 0}, 1}, {"Si+", {0, 0, 0, 0, 1, 0}, 1},
                                {"O+", {0, 0, 0, 1, 0, 0}, 1}, {"D+", {0, 0, 0, 0, 0, 1}, 1},
                                {"HD", {1, 0, 0, 0, 0, 1}, 0}, {"He+", {0, 1, 0, 0, 0, 0}, 1},
                                {"He++", {0, 1, 0, 0, 0, 0}, 2}, {"CO", {0, 0, 1, 1, 0, 0}, 0},
                                {"C2", {0, 0, 2, 0, 0, 0}, 0}, {"OH", {1, 0, 0, 1, 0, 0}, 0},
                                {"H2O", {2, 0, 0, 1, 0, 0}, 0}, {"O2", {0, 0, 0, 2, 0, 0}, 0},
                                {"HCO+", {1, 0, 1, 1, 0, 0}, 1}, {"CH", {1, 0, 1, 0, 0, 0}, 0},
                                {"CH2", {2, 0, 1, 0, 0, 0}, 0}, {"Si++", {0, 0, 0, 0, 1, 0}, 2},
                                {"CH3+", {3, 0, 1, 0, 0, 0}, 1}};
#else
static const Sp SP[TRAC_NUM] = {{"H2", {2, 0, 0, 0, 0, 0}, 0}, {"H+", {1, 0, 0, 0, 0, 0}, 1},
                                {"C+", {0, 0, 1, 0, 0, 0}, 1}, {"CHx", {1, 0, 1, 0, 0, 0}, 0},
                                {"OHx", {1, 0, 0, 1, 0, 0}, 0}, {"CO", {0, 0, 1, 1, 0, 0}, 0},
                                {"HCO+", {1, 0, 1, 1, 0, 0}, 1}, {"He+", {0, 1, 0, 0, 0, 0}, 1},
                                {"M+", {0, 0, 0, 0, 1, 0}, 1}};
#endif

static void pools_of(double P[NEL])
{
    P[EH] = 1.0; P[EHE] = CHEM_COMP.abhe; P[EC] = COOLRC.abundc; P[EO] = COOLRC.abundo;
#if CHEMISTRYNETWORK == 3
    P[EM] = COOLRC.abundsi; P[ED] = COOLRC.abundD;
#else
    P[EM] = COOLRC.abundM; P[ED] = COOLRC.abundD;
#endif
}
// an element is audited when some tracked species carries it
static bool el_active(int e) { for (int i = 0; i < TRAC_NUM; i++) if (SP[i].n[e]) return true; return false; }

struct Viol { double over[NEL], neg[NEL]; };
static Viol measure(const double* y)
{
    Viol v; double P[NEL]; pools_of(P);
    for (int e = 0; e < NEL; e++) {
        double s = 0.0, ng = 0.0;
        for (int i = 0; i < TRAC_NUM; i++) { s += SP[i].n[e] * y[i]; if (y[i] < 0) ng -= SP[i].n[e] * y[i]; }
        v.over[e] = (P[e] > 0) ? std::max(0.0, s - P[e]) / P[e] : 0.0;
        v.neg[e]  = (P[e] > 0) ? ng / P[e] : 0.0;
    }
    return v;
}

// ---------------------------------------------------------------------------
// probes, called from the instrumented build copies (weak no-ops otherwise)
// ---------------------------------------------------------------------------
struct FloorStat { long n = 0; double maxrel = 0.0; long bin[4] = {0, 0, 0, 0}; char var[16] = ""; };
static std::map<int, FloorStat> g_floor;     // key = fileid*100000 + line
static long g_rhs_calls = 0;
struct ClipStat { long n_neg = 0, n_max = 0; double dmass_neg[NEL] = {0}, dmass_max[NEL] = {0};
                  double max_over_after[NEL] = {0}; long n_accept = 0, n_reject = 0; };
static ClipStat g_clip;
static double g_h3p_max = 0.0, g_h3p_rel_max = 0.0;
static long g_h3p_cap = 0, g_h3p_capped = 0;

static char g_where[200] = "init";
// pool a floored derived variable is measured against
static double floor_pool(const char* var)
{
    std::string v(var);
    if (v == "abheI") return CHEM_COMP.abhe;
    if (v == "abcI" || v == "abcp") return COOLRC.abundc;
    if (v == "aboI" || v == "abo" || v == "abotot") return COOLRC.abundo;
    if (v == "abmI") return COOLRC.abundM;
    if (v == "absiI") return COOLRC.abundsi;
    if (v == "abDI") return COOLRC.abundD;
    return 1.0;                                  // abHI, abe: per H nucleus
}
extern "C" void probe_floor(int fileid, int line, const char* var, double v)
{
    FloorStat& s = g_floor[fileid * 100000 + line];
    std::snprintf(s.var, sizeof(s.var), "%s", var);
    double P = floor_pool(var), r = (P > 0) ? std::fabs(v) / P : 0.0;
    s.n++; if (r > s.maxrel) s.maxrel = r;
    s.bin[r < 1e-12 ? 0 : r < 1e-6 ? 1 : r < 1e-3 ? 2 : 3]++;
    static const char* tr = std::getenv("CONS_FLOOR_TRACE");
    if (tr && s.n <= std::atol(tr)) std::printf("  FLOORTRACE %d:%d %s v=%.3e rel=%.3e at %s\n", fileid, line, var, v, r, g_where);
}
extern "C" void probe_rhs() { g_rhs_calls++; }
extern "C" void probe_h3p(double abh3p, double abHI)
{
    if (3.0 * abh3p > g_h3p_max) g_h3p_max = 3.0 * abh3p;
    if (abHI > 0 && 3.0 * abh3p / abHI > g_h3p_rel_max) g_h3p_rel_max = 3.0 * abh3p / abHI;
    if (3.0 * abh3p > abHI) g_h3p_cap++;              // pre-fix: stock exceeds the HI pool
    if (abHI == 0.0 && abh3p > 0.0) g_h3p_capped++;  // post-fix: cap active
}
extern "C" void probe_check(const double* yin, const double* yout, int rc)
{
    if (rc) { g_clip.n_reject++; return; }
    g_clip.n_accept++;
    double P[NEL]; pools_of(P);
    for (int i = 0; i < TRAC_NUM; i++) {
        double d = yout[i] - yin[i];
        if (d == 0.0) continue;
        if (d > 0) g_clip.n_neg++; else g_clip.n_max++;
        for (int e = 0; e < NEL; e++) {
            if (!SP[i].n[e] || P[e] <= 0) continue;
            double r = SP[i].n[e] * d / P[e];
            if (d > 0) g_clip.dmass_neg[e] = std::max(g_clip.dmass_neg[e], r);
            else       g_clip.dmass_max[e] = std::max(g_clip.dmass_max[e], -r);
        }
    }
    Viol v = measure(yout);
    for (int e = 0; e < NEL; e++) g_clip.max_over_after[e] = std::max(g_clip.max_over_after[e], v.over[e]);
}

// ---------------------------------------------------------------------------
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
    COOLI.iflag_mn = 1; COOLI.iflag_ad = 1; COOLI.iflag_atom = 2;
    COOLI.iflag_3bh2a = 1; COOLI.iflag_3bh2b = 1; COOLI.iflag_h3pra = 1;
    COOLI.iphoto = 6;
    COOLIC.irad_heat = 0; COOLI.isrf_option = 1; COOLIC.no_chem = 0;
    COOLIC.id_current = 1; COOLIC.index_current = 0;
    COOLI.iflag_h2opc = 0; COOLI.idma_mass_option = 1;
    COOLR.h2_form_ex = 0.84; COOLR.h2_form_kin = 0.12;
    COOLR.deff = 1.0; COOLR.phi_pah = 1.0;
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
        PROJECT.column_density_projection_h2[k] = std::max(abh2, 0.0) * NH;
        PROJECT.column_density_projection_co[k] = std::max(abco, 0.0) * NH;
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
        PROJECT.fac_uv[k] = 1.0 / NPIX;
#endif
    }
#else
    (void)yn; (void)abh2; (void)abco;
#endif
}

// electrons as the RHS builds them before any QSS species
static double abe_of(const double* y)
{
#if CHEMISTRYNETWORK == 5
    double cp = std::max(COOLRC.abundc - y[ICO], 0.0);
    return y[IHP] + cp + COOLRC.abundsi;
#elif CHEMISTRYNETWORK == 17
    double cp = std::max(COOLRC.abundc - y[ICO], 0.0);
    return y[IHP] + y[IHEP] + 2.0 * y[IHEPP] + cp + y[IDP] + COOLRC.abundsi;
#elif CHEMISTRYNETWORK == 3
    return y[IHP] + y[IDP] + y[IHEP] + y[IC] + y[IO] + y[ISi] + y[IHCOP]
         + 2.0 * y[ISIPP] + y[ICH3P] + 2.0 * y[IHEPP];
#else
    double e = y[IHP] + y[IC] + y[IHCOP] + y[IHEP] + y[IMP];
    return e > 0.0 ? e : 0.0;
#endif
}
static double temp_of(const ChemTables& T, double yn, double energy, const double* y)
{
    double yntot = (1.0 + CHEM_COMP.abhe - y[IH2] + abe_of(y)) * yn;
    double t, g;
    cxx_lookup_temp_gamma(y[IH2], CHEM_COMP.abhe, energy / (yntot * CHEM_KBOLTZ), &t, &g, T);
    return t;
}
static double energy_at(const ChemTables& T, double yn, double T0, const double* y)
{
    double yntot = (1.0 + CHEM_COMP.abhe - y[IH2] + abe_of(y)) * yn;
    double e = 1.5 * yntot * CHEM_KBOLTZ * T0;
    for (int k = 0; k < 60; k++) {
        double t = temp_of(T, yn, e, y);
        if (std::fabs(t - T0) < 1e-10 * T0) break;
        e *= T0 / t;
    }
    return e;
}

static ChemInputs inputs_now()
{
    ChemInputs in;
    in.abundc = COOLRC.abundc; in.abundo = COOLRC.abundo; in.abundsi = COOLRC.abundsi;
    in.abundD = COOLRC.abundD; in.abhe = CHEM_COMP.abhe; in.abundM = COOLRC.abundM;
    return in;
}

// start states: 0 atomic, 1 molecular, 2 ionized, 3 mixed, 4 overdrawn (5% over every pool)
static const int NSTART = 5;
static ChemTables* g_cx = nullptr;
#include <chrono>
static double wall() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void initial_state(int start, double* y)
{
    const double C = COOLRC.abundc, O = COOLRC.abundo, He = CHEM_COMP.abhe, D = COOLRC.abundD;
    const double CO = std::min(C, O);
    for (int i = 0; i < TRAC_NUM; i++) y[i] = 0.0;
    (void)He; (void)D; (void)O;
    switch (start) {
    case 0: y[IH2] = 1e-6; y[IHP] = 1e-4; break;
    case 1: y[IH2] = 0.49;  y[IHP] = 1e-6; y[ICO] = 0.5 * CO; break;
    case 2: y[IH2] = 0.0;   y[IHP] = 0.999; break;
    case 3: y[IH2] = 0.25;  y[IHP] = 0.3;  y[ICO] = 0.3 * CO; break;
    case 4: y[IH2] = 0.30;  y[IHP] = 0.42; y[ICO] = 1.05 * CO; break;
    }
#if CHEMISTRYNETWORK == 17
    switch (start) {
    case 0: y[IDP] = 1e-4 * D; break;
    case 1: y[IHD] = 0.5 * D; y[IDP] = 1e-6 * D; break;
    case 2: y[IHEP] = 0.5 * He; y[IHEPP] = 0.4 * He; y[IDP] = D; break;
    case 3: y[IHEP] = 0.5 * He; y[IHEPP] = 0.5 * He; y[IDP] = 0.5 * D; y[IHD] = 0.5 * D; break;
    case 4: y[IHEP] = 0.6 * He; y[IHEPP] = 0.45 * He; y[IDP] = 0.55 * D; y[IHD] = 0.5 * D; break;
    }
#endif
#if CHEMISTRYNETWORK == 3
    const double Si = COOLRC.abundsi;
    switch (start) {
    case 0: y[IC] = C; y[ISi] = Si; y[IDP] = 1e-4 * D; break;
    case 1: y[IC] = 0.01 * C; y[ISi] = 0.01 * Si; y[IHD] = 0.5 * D; y[IDP] = 1e-6 * D;
            y[IOH] = 1e-3 * O; y[IH2O] = 1e-3 * O; y[IHCOP] = 1e-6 * C; y[ICH] = 1e-4 * C; break;
    case 2: y[IC] = C; y[IO] = 0.9 * O; y[ISi] = 0.5 * Si; y[ISIPP] = 0.5 * Si; y[IHEP] = 0.5 * He;
            y[IHEPP] = 0.4 * He; y[IDP] = D; break;
    case 3: y[IC] = 0.1 * C; y[IO] = 0.05 * O; y[ISi] = 0.3 * Si; y[ISIPP] = 0.1 * Si; y[IDP] = 0.5 * D;
            y[IHD] = 0.5 * D; y[IHEP] = 0.1 * He; y[IHEPP] = 0.01 * He; y[IC2] = 0.05 * C; y[IOH] = 0.2 * O;
            y[IH2O] = 0.1 * O; y[IO2] = 0.02 * O; y[IHCOP] = 0.05 * CO; y[ICH] = 0.1 * C; y[ICH2] = 0.1 * C;
            y[ICH3P] = 0.02 * C; break;
    case 4: y[IC] = 0.5 * C; y[IO] = 0.3 * O; y[ISi] = 0.6 * Si; y[ISIPP] = 0.45 * Si; y[IDP] = 0.55 * D;
            y[IHD] = 0.5 * D; y[IHEP] = 0.6 * He; y[IHEPP] = 0.45 * He; y[IC2] = 0.05 * C; y[IOH] = 0.3 * O;
            y[IH2O] = 0.2 * O; y[IO2] = 0.05 * O; y[IHCOP] = 0.05 * CO; y[ICH] = 0.1 * C; y[ICH2] = 0.1 * C;
            y[ICH3P] = 0.05 * C; break;
    }
#endif
#if CHEMISTRYNETWORK == 15 || CHEMISTRYNETWORK == 16
    const double M = COOLRC.abundM;
    switch (start) {
    case 0: y[IC] = C; y[IMP] = M; break;
    case 1: y[IC] = 0.01 * C; y[ICHX] = 1e-3 * C; y[IOHX] = 1e-3 * O; y[IHCOP] = 1e-6 * C; y[IMP] = 0.01 * M; break;
    case 2: y[IC] = C; y[IHEP] = 0.9 * He; y[IMP] = M; break;
    case 3: y[IC] = 0.1 * C; y[ICHX] = 0.2 * C; y[IOHX] = 0.3 * O; y[IHCOP] = 0.05 * CO; y[IHEP] = 0.1 * He; y[IMP] = 0.3 * M; y[ICO] = 0.1 * CO; break;
    case 4: y[IC] = 0.5 * C; y[ICHX] = 0.3 * C; y[IOHX] = 0.6 * O; y[IHCOP] = 0.05 * CO; y[IHEP] = 1.05 * He; y[IMP] = 1.05 * M; y[ICO] = 0.25 * CO;
            y[IH2] = 0.3; y[IHP] = 0.3; break;
    }
#endif
}

// chemcool.cc pre-solve path (molecule zeroing, then chemcool_clamp_pools)
static void presolve(double* y, double removed[NEL])
{
    double y0[TRAC_NUM]; for (int i = 0; i < TRAC_NUM; i++) y0[i] = y[i];
#if CHEMISTRYNETWORK == 17
    if (y[IHP] > 0.99) { y[IH2] = 0.0; y[ICO] = 0.0; y[IHD] = 0.0; }
#endif
    ChemInputs in = inputs_now();
    chemcool_clamp_pools(y, &in);
    double P[NEL]; pools_of(P);
    for (int e = 0; e < NEL; e++) {
        double d = 0.0;
        for (int i = 0; i < TRAC_NUM; i++) d += SP[i].n[e] * (y0[i] - y[i]);
        removed[e] = (P[e] > 0) ? d / P[e] : 0.0;
    }
}

static int run_fortran(double dt, double yn, double* e, double* y, double* td)
{
    double Tf = temp_of(*g_cx, yn, *e, y);   // lookup tables are Gate-3 identical
    set_columns(yn, y[IH2], y[ICO]);
    double rpar[NRPAR]; std::memset(rpar, 0, sizeof(rpar));
    rpar[0] = yn;
#if CHEMISTRYNETWORK == 17
    double a2 = y[IH2], ahd = y[IHD], aco = y[ICO];
#else
    double a2 = y[IH2], ahd = 0.0, aco = y[ICO];
#endif
    COOLRC.cosmic_ray_ion_rate = g_zeta;
    COOLRC.tdust = *td;
    CALC_PHOTO_WRAPPER(&Tf, rpar, &a2, &ahd, &aco);
    double dts = dt, dl = 0.0, dv = 0.0, ce = 0.0;
    int rc = 0;
    g_fatal_armed = 1;
    if (setjmp(g_fatal_jmp) == 0) EVOLVE_ABUNDANCES(&dts, &dl, &yn, &dv, e, y, &ce);
    else rc = 1;
    g_fatal_armed = 0;
    *td = COOLRC.tdust;
    return rc;
}

static int run_cxx(double dt, double yn, double* e, double* y, double* td)
{
    double Tc = temp_of(*g_cx, yn, *e, y);
    set_columns(yn, y[IH2], y[ICO]);
    COOLRC.cosmic_ray_ion_rate = g_zeta;
    ChemCell c; chem_bind_cell(c, *g_cx, NET);
    c.yn = yn; c.dl = 0.0; c.divv = 0.0; c.tdust = *td;
#if CHEMISTRYNETWORK == 17
    cxx_calc_photo(Tc, y[IH2], y[IHD], y[ICO], c);
#else
    cxx_calc_photo(Tc, y[IH2], 0.0, y[ICO], c);
#endif
    int rc = cxx_evolve_abundances(dt, 0.0, yn, 0.0, e, y, c);
    *td = c.tdust;
    return rc;
}

// ===========================================================================
// PHASE R -- RHS parity
// ===========================================================================
struct Acc {
    long n = 0, npass = 0; double worst = 0.0;
    void add(double a, double b, double tol) {
        n++; double rel;
        if (a == b) rel = 0.0;
        else if (!std::isfinite(a) || !std::isfinite(b)) rel = 1e300;
        else rel = std::fabs(a - b) / std::max(std::fabs(a), std::fabs(b));
        if (rel <= tol) npass++;
        if (rel > worst) worst = rel;
    }
};

static int rhs_cxx(const double* y, double* yd, ChemCell& c)
{
#if CHEMISTRYNETWORK == 3
    return chem_rhs_net3(0.0, y, yd, c);
#elif CHEMISTRYNETWORK == 5
    return chem_rhs_net5(0.0, y, yd, c);
#elif CHEMISTRYNETWORK == 15
    return chem_rhs_net15(0.0, y, yd, c);
#elif CHEMISTRYNETWORK == 16
    return chem_rhs_net16(0.0, y, yd, c);
#else
    return chem_rhs_net17(0.0, y, yd, c);
#endif
}

static int phase_R(const ChemTables& TT, const char* tag)
{
    const double TOL = 1e-12;
    std::vector<Acc> acc(NSPEC);
    Acc rlc, rtd, rfl;
    const double yns[]  = {1e-2, 1.0, 30.0, 1e3, 1e5, 1e7};
    const double Ts[]   = {10.0, 50.0, 300.0, 3e3, 1e4, 5e4, 3e5, 3e6, 3e7};
#if CHEMISTRYNETWORK == 3
    // G0 = 0 not gated: calc_photo returns before setting chi_isrf, which rate_eq_mol passes on uninitialized
    const double G0s[]  = {1e-4, 1.7, 1e3};
#else
    const double G0s[]  = {0.0, 1.7, 1e3};
#endif
    const double Zss[]  = {0.1, 1.0, 3.0};
    long nflag = 0, nabort = 0, nbad_g0[3] = {0, 0, 0}, nst_g0[3] = {0, 0, 0};
    for (double Z : Zss) for (double G0 : G0s) {
        const int gi = (G0 < 1e-3) ? 0 : (G0 < 10.0) ? 1 : 2;
        set_cell_commons(G0, Z);
        for (double yn : yns) for (int st = 0; st < NSTART; st++) for (double T0 : Ts) {
            std::snprintf(g_where, sizeof(g_where), "R nH=%g T=%g G0=%g Z=%g start=%d", yn, T0, G0, Z, st);
            double y[NSPEC];
            initial_state(st, y);
            y[ITMP] = energy_at(TT, yn, T0, y);
            double tguess = temp_of(TT, yn, y[ITMP], y);
            set_columns(yn, y[IH2], y[ICO]);
            double rpar[NRPAR]; std::memset(rpar, 0, sizeof(rpar));
            rpar[0] = yn;
#if CHEMISTRYNETWORK == 17
            double a2 = std::max(y[IH2], 0.0), ahd = std::max(y[IHD], 0.0), aco = std::max(y[ICO], 0.0);
#else
            double a2 = std::max(y[IH2], 0.0), ahd = 0.0, aco = std::max(y[ICO], 0.0);
#endif
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
            for (int k = 0; k < NRATES_CHEM; k++) COOLRC.lambda_chem[k] = 0.0;
            int aborted = 0;
            g_fatal_armed = 1;
            if (setjmp(g_fatal_jmp) == 0) RATE_EQ(&neq, &t0, yF, ydF, rpar, ipar);
            else aborted = 1;          // comp_equilib / index ABORT
            g_fatal_armed = 0;
            int rc = rhs_cxx(y, ydC, c);
            int fF = (ipar[0] != 0) || aborted, fC = rc != 0;
            rfl.add(fF, fC, 0.0);
            nflag += fF; nabort += aborted;
            if (fF && fC) continue;   // both flagged: Fortran ydot zeroed, C++ ydot unspecified
            long before = 0, after = 0;
            for (int i = 0; i < NSPEC; i++) before += acc[i].n - acc[i].npass;
            for (int i = 0; i < NSPEC; i++) acc[i].add(ydF[i], ydC[i], TOL);
            for (int i = 0; i < NSPEC; i++) after += acc[i].n - acc[i].npass;
            nst_g0[gi]++; if (after > before) nbad_g0[gi]++;
            for (int k = 0; k < NRATES_CHEM; k++) rlc.add(COOLRC.lambda_chem[k], c.lambda_chem[k], TOL);
            rtd.add(COOLRC.tdust, c.tdust, TOL);
        }
    }
    std::printf("PHASE %s -- RHS parity (rate_eq vs chem_rhs_net%d), %ld flagged states (%ld Fortran ABORTs), tol %.0e\n",
                tag, NET, nflag, nabort, TOL);
    std::printf("  states with a ydot mismatch, by G0 (lowest / 1.7 / 1e3): %ld/%ld  %ld/%ld  %ld/%ld\n",
                nbad_g0[0], nst_g0[0], nbad_g0[1], nst_g0[1], nbad_g0[2], nst_g0[2]);
    int ok = 1;
    for (int i = 0; i < NSPEC; i++) {
        int pass = acc[i].npass == acc[i].n; ok &= pass;
        std::printf("  ydot(%-6s) %7ld/%-7ld %s worst rel = %.3e\n", i == ITMP ? "energy" : SP[i].name,
                    acc[i].npass, acc[i].n, pass ? "PASS" : "FAIL", acc[i].worst);
    }
    const char* nm[3] = {"lambda_chem(all)", "tdust after RHS", "flag (0/1)"};
    Acc* a3[3] = {&rlc, &rtd, &rfl};
    for (int k = 0; k < 3; k++) {
        int pass = a3[k]->npass == a3[k]->n; ok &= pass;
        std::printf("  %-17s %7ld/%-7ld %s worst rel = %.3e\n", nm[k], a3[k]->npass, a3[k]->n,
                    pass ? "PASS" : "FAIL", a3[k]->worst);
    }
    return ok;
}

// ===========================================================================
// PHASE C -- conservation grid
// Each (state, backend) call runs in a forked child with a wall-clock limit
// (CONS_TIMEOUT s, default 120); the child returns the state and its probe
// deltas through a pipe.  Solver output of the child goes to /dev/null.
// ===========================================================================
struct FloorRec { int key; long n; double maxrel; long bin[4]; char var[16]; };
struct ChildResult {
    int rc; double wsec, en; double y[TRAC_NUM];
    ClipStat clip; double h3p_max, h3p_rel_max; long h3p_cap, h3p_capped, rhs;
    int nfl; FloorRec fl[64];
};
static void merge_child(const ChildResult& r)
{
    g_rhs_calls += r.rhs;
    g_clip.n_neg += r.clip.n_neg; g_clip.n_max += r.clip.n_max;
    g_clip.n_accept += r.clip.n_accept; g_clip.n_reject += r.clip.n_reject;
    for (int e = 0; e < NEL; e++) {
        g_clip.dmass_neg[e] = std::max(g_clip.dmass_neg[e], r.clip.dmass_neg[e]);
        g_clip.dmass_max[e] = std::max(g_clip.dmass_max[e], r.clip.dmass_max[e]);
        g_clip.max_over_after[e] = std::max(g_clip.max_over_after[e], r.clip.max_over_after[e]);
    }
    g_h3p_max = std::max(g_h3p_max, r.h3p_max); g_h3p_rel_max = std::max(g_h3p_rel_max, r.h3p_rel_max);
    g_h3p_cap += r.h3p_cap; g_h3p_capped += r.h3p_capped;
    for (int i = 0; i < r.nfl; i++) {
        FloorStat& s = g_floor[r.fl[i].key];
        s.n += r.fl[i].n; s.maxrel = std::max(s.maxrel, r.fl[i].maxrel);
        for (int k = 0; k < 4; k++) s.bin[k] += r.fl[i].bin[k];
        std::snprintf(s.var, sizeof(s.var), "%s", r.fl[i].var);
    }
}
static int run_forked(int backend, double dt, double nH, double* en, double* y, double* td, double* wsec)
{
    static const char* to = std::getenv("CONS_TIMEOUT");
    const double tmax = to ? std::atof(to) : 120.0;
    int fd[2];
    if (pipe(fd)) { std::perror("pipe"); std::exit(1); }
    std::fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        close(fd[0]);
        int dn = open("/dev/null", O_WRONLY); dup2(dn, 1); dup2(dn, 2);
        g_floor.clear(); g_clip = ClipStat(); g_rhs_calls = 0;
        g_h3p_max = g_h3p_rel_max = 0.0; g_h3p_cap = g_h3p_capped = 0;
        ChildResult r = ChildResult();
        double w0 = wall();
        r.rc = backend ? run_cxx(dt, nH, en, y, td) : run_fortran(dt, nH, en, y, td);
        r.wsec = wall() - w0; r.en = *en;
        for (int i = 0; i < TRAC_NUM; i++) r.y[i] = y[i];
        r.clip = g_clip; r.h3p_max = g_h3p_max; r.h3p_rel_max = g_h3p_rel_max;
        r.h3p_cap = g_h3p_cap; r.h3p_capped = g_h3p_capped; r.rhs = g_rhs_calls;
        for (auto& kv : g_floor) {
            if (r.nfl >= 64) break;
            FloorRec& f = r.fl[r.nfl++];
            f.key = kv.first; f.n = kv.second.n; f.maxrel = kv.second.maxrel;
            for (int k = 0; k < 4; k++) f.bin[k] = kv.second.bin[k];
            std::memcpy(f.var, kv.second.var, sizeof(f.var));
        }
        const char* p = (const char*)&r; size_t left = sizeof(r);
        while (left) { ssize_t w = write(fd[1], p, left); if (w <= 0) break; p += w; left -= w; }
        _exit(0);
    }
    close(fd[1]);
    ChildResult r; char* p = (char*)&r; size_t got = 0;
    double w0 = wall();
    int timed_out = 0;
    while (got < sizeof(r)) {
        double left = tmax - (wall() - w0);
        if (left <= 0) { timed_out = 1; break; }
        struct pollfd pf = {fd[0], POLLIN, 0};
        int pr = poll(&pf, 1, (int)(left * 1000) + 1);
        if (pr == 0) { timed_out = 1; break; }
        if (pr < 0) continue;
        ssize_t n = read(fd[0], p + got, sizeof(r) - got);
        if (n <= 0) break;
        got += n;
    }
    close(fd[0]);
    if (timed_out) kill(pid, SIGKILL);
    int st; waitpid(pid, &st, 0);
    if (timed_out) { *wsec = tmax; return -9; }
    if (got < sizeof(r)) { *wsec = wall() - w0; return -8; }     // child died
    merge_child(r);
    *en = r.en; for (int i = 0; i < TRAC_NUM; i++) y[i] = r.y[i];
    *wsec = r.wsec;
    return r.rc;
}
struct Worst { double v = 0.0; char where[160] = ""; };
static void upd(Worst& w, double v, const char* where) { if (v > w.v) { w.v = v; std::snprintf(w.where, sizeof(w.where), "%s", where); } }

static void phase_C(const char* csvname)
{
    const double nHs[] = {1e-2, 1.0, 1e2, 1e4, 1e6};
    const double Ts[]  = {10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7};
    const double G0s[] = {0.0, 1.7, 1e2, 1e4};
    const double Zss[] = {0.1, 1.0, 3.0};
    const double dts[] = {3.156e9, 3.156e14};      // 100 yr, 10 Myr
    FILE* f = std::fopen(csvname, "w");
    if (f) {
        std::fprintf(f, "nH,T0,G0,Z,start,dt,backend,rc,wall_s");
        for (int e = 0; e < NEL; e++) std::fprintf(f, ",pre_removed_%s,over_in_%s,over_out_%s,neg_out_%s",
                                                   ELNAME[e], ELNAME[e], ELNAME[e], ELNAME[e]);
        for (int i = 0; i < TRAC_NUM; i++) std::fprintf(f, ",%s", SP[i].name);
        std::fprintf(f, "\n");
    }
    Worst w_pre[NEL], w_in[NEL], w_out[2][NEL], w_neg[2][NEL];
    long ncall[2] = {0, 0}, nfail[2] = {0, 0}, n_over_out[2][NEL] = {{0}}, n_pre[NEL] = {0};
    long ntot = 0;
    double wmax[2] = {0, 0}; char wwhere[2][160] = {"", ""}; long ntimeout[2] = {0, 0};
    const char* fz = std::getenv("CONS_ONLY_Z"); const char* fg = std::getenv("CONS_ONLY_G0");
    const char* fn = std::getenv("CONS_ONLY_NH"); const char* ft = std::getenv("CONS_ONLY_T");
    for (double Z : Zss) for (double G0 : G0s) {
        if ((fz && std::atof(fz) != Z) || (fg && std::atof(fg) != G0)) continue;
        set_cell_commons(G0, Z);
        for (double nH : nHs) for (double T0 : Ts) for (int st = 0; st < NSTART; st++) for (double dt : dts) {
            if ((fn && std::atof(fn) != nH) || (ft && std::atof(ft) != T0)) continue;
            char where[160];
            std::snprintf(where, sizeof(where), "nH=%g T=%g G0=%g Z=%g start=%d dt=%g", nH, T0, G0, Z, st, dt);
            std::snprintf(g_where, sizeof(g_where), "C %s", where);
            double y0[TRAC_NUM], removed[NEL];
            initial_state(st, y0);
            presolve(y0, removed);
            Viol vin = measure(y0);
            for (int e = 0; e < NEL; e++) {
                if (removed[e] > 0) n_pre[e]++;
                upd(w_pre[e], removed[e], where);
                upd(w_in[e], vin.over[e], where);
            }
            const double e0 = energy_at(*g_cx, nH, T0, y0);
            ntot++;
            for (int b = 0; b < 2; b++) {
                double y[TRAC_NUM]; for (int i = 0; i < TRAC_NUM; i++) y[i] = y0[i];
                double en = e0, td = 10.0;
                double wsec = 0.0;
                int rc = run_forked(b, dt, nH, &en, y, &td, &wsec);
                if (rc == -9) ntimeout[b]++;
                if (wsec > wmax[b]) { wmax[b] = wsec; std::snprintf(wwhere[b], sizeof(wwhere[b]), "%s", where); }
                ncall[b]++;
                if (rc) nfail[b]++;
                Viol v = measure(y);
                for (int e = 0; e < NEL; e++) {
                    if (!rc) { upd(w_out[b][e], v.over[e], where); upd(w_neg[b][e], v.neg[e], where); }
                    if (!rc && v.over[e] > 1e-14) n_over_out[b][e]++;
                }
                if (f) {
                    std::fprintf(f, "%g,%g,%g,%g,%d,%g,%s,%d,%.3f", nH, T0, G0, Z, st, dt, b ? "cxx" : "f77", rc, wsec);
                    for (int e = 0; e < NEL; e++) std::fprintf(f, ",%.3e,%.3e,%.3e,%.3e", removed[e], vin.over[e], v.over[e], v.neg[e]);
                    for (int i = 0; i < TRAC_NUM; i++) std::fprintf(f, ",%.9e", y[i]);
                    std::fprintf(f, "\n");
                }
            }
        }
    }
    if (f) std::fclose(f);
    std::printf("\nPHASE C -- conservation, %ld states x 2 backends (failures: f77 %ld, cxx %ld)\n",
                ntot, nfail[0], nfail[1]);
    std::printf("  wall-clock timeouts (excluded from the statistics below): f77 %ld, cxx %ld\n", ntimeout[0], ntimeout[1]);
    std::printf("  slowest call: f77 %.1f s (%s), cxx %.1f s (%s)\n", wmax[0], wwhere[0], wmax[1], wwhere[1]);
    std::printf("  element | presolve clamp removed (max, #states) | overdraft into solver | "
                "overdraft after call f77 / cxx (#states > 1e-14) | negative after call f77 / cxx\n");
    for (int e = 0; e < NEL; e++) {
        if (!el_active(e)) continue;
        std::printf("  %-3s  pre %.3e (%ld)  in %.3e  out f77 %.3e (%ld) cxx %.3e (%ld)  neg f77 %.3e cxx %.3e\n",
                    ELNAME[e], w_pre[e].v, n_pre[e], w_in[e].v,
                    w_out[0][e].v, n_over_out[0][e], w_out[1][e].v, n_over_out[1][e],
                    w_neg[0][e].v, w_neg[1][e].v);
        if (w_out[0][e].v > 0) std::printf("        worst f77 at %s\n", w_out[0][e].where);
        if (w_out[1][e].v > 0) std::printf("        worst cxx at %s\n", w_out[1][e].where);
        if (w_in[e].v > 0)     std::printf("        worst into-solver at %s\n", w_in[e].where);
    }
}

// ===========================================================================
// PHASE O -- trajectory parity (test_onezone_ab PHASE O grid)
// ===========================================================================
static int phase_O()
{
    struct Point { double nH, T0, G0, Zs; int start; };
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
    double worst = 0.0, worst_traj = 0.0; int nfail_gate = 0, nrunfail = 0, wp = -1;
    Viol vmax[2]; for (int b = 0; b < 2; b++) for (int e = 0; e < NEL; e++) vmax[b].over[e] = vmax[b].neg[e] = 0.0;
    for (size_t p = 0; p < pts.size(); p++) {
        const auto& P = pts[p];
        std::snprintf(g_where, sizeof(g_where), "O pt %zu", p);
        set_cell_commons(P.G0, P.Zs);
        double yF[TRAC_NUM], yC[TRAC_NUM];
        initial_state(P.start, yF);
        for (int i = 0; i < TRAC_NUM; i++) yC[i] = yF[i];
        double eF = energy_at(*g_cx, P.nH, P.T0, yF), eC = eF, tdF = 10.0, tdC = 10.0;
        int rcF = 0, rcC = 0; double traj = 0.0, TF = P.T0, TC = P.T0;
        for (int k = 1; k <= NSTEP_OZ; k++) {
            double dt = (k == 1) ? tgrid[1] : tgrid[k] - tgrid[k - 1];
            if (!rcF) rcF = run_fortran(dt, P.nH, &eF, yF, &tdF);
            if (!rcC) rcC = run_cxx(dt, P.nH, &eC, yC, &tdC);
            TF = temp_of(*g_cx, P.nH, eF, yF); TC = temp_of(*g_cx, P.nH, eC, yC);
            traj = std::max(traj, std::fabs(TC - TF) / TF);
            Viol a = measure(yF), b = measure(yC);
            for (int e = 0; e < NEL; e++) {
                vmax[0].over[e] = std::max(vmax[0].over[e], a.over[e]); vmax[1].over[e] = std::max(vmax[1].over[e], b.over[e]);
                vmax[0].neg[e] = std::max(vmax[0].neg[e], a.neg[e]);   vmax[1].neg[e] = std::max(vmax[1].neg[e], b.neg[e]);
            }
        }
        double dT = std::fabs(TC - TF) / TF;
        std::printf("  %3zu nH=%7.1e T0=%6.0e G0=%5.1f Z=%3.1f st=%d | T_F %11.5g T_C %11.5g dT/T %9.2e traj %9.2e | %d/%d\n",
                    p, P.nH, P.T0, P.G0, P.Zs, P.start, TF, TC, dT, traj, rcF, rcC);
        if (rcF || rcC) { nrunfail++; continue; }
        if (dT > 0.01) nfail_gate++;
        if (dT > worst) { worst = dT; wp = (int)p; }
        worst_traj = std::max(worst_traj, traj);
    }
    std::printf("  final T worst |dT|/T = %.3e (pt %d), trajectory worst %.3e\n", worst, wp, worst_traj);
    std::printf("  overdraft along trajectories (max over 80 steps x %zu points):\n", pts.size());
    for (int e = 0; e < NEL; e++) if (el_active(e))
        std::printf("    %-3s f77 %.3e cxx %.3e | negative f77 %.3e cxx %.3e\n", ELNAME[e],
                    vmax[0].over[e], vmax[1].over[e], vmax[0].neg[e], vmax[1].neg[e]);
    int ok = (nfail_gate == 0 && nrunfail == 0);
    std::printf("  GATE final T within 1%%: %s (%d over, %d run failures)\n", ok ? "PASS" : "FAIL", nfail_gate, nrunfail);
    return ok;
}

// ===========================================================================
// PHASE I -- chemcool_set_ionized pool check with abhe != ABHE
// ===========================================================================
static void phase_I()
{
    std::printf("\nPHASE I -- chemcool_set_ionized element pools (per-cell abhe 0.07 and 0.12, ABHE = %g)\n", (double)ABHE);
    const double ahes[] = {0.07, 0.12};
    for (double ahe : ahes) {
        set_cell_commons(1.7, 1.0);
        double save = CHEM_COMP.abhe; CHEM_COMP.abhe = ahe;
        ChemInputs in = inputs_now();
        double y[TRAC_NUM]; initial_state(1, y);
        chemcool_set_ionized(y, &in);
        Viol v = measure(y);
        double P[NEL]; pools_of(P);
        for (int e = 0; e < NEL; e++) if (el_active(e)) {
            double s = 0; for (int i = 0; i < TRAC_NUM; i++) s += SP[i].n[e] * y[i];
            std::printf("  abhe=%.3f %-3s tracked/pool = %.6f  overdraft %.3e\n", ahe, ELNAME[e], P[e] > 0 ? s / P[e] : 0.0, v.over[e]);
        }
        CHEM_COMP.abhe = save;
    }
}

static void report_probes()
{
    std::printf("\nPROBES (C++ backend, all phases)\n");
    std::printf("  RHS evaluations probed: %ld\n", g_rhs_calls);
    std::printf("  derived-species floors that fired at RHS evaluations (incl. CVODE trial iterates):\n");
    std::printf("    file:line var  count | |v|/pool bins <1e-12 / 1e-12..1e-6 / 1e-6..1e-3 / >1e-3 | max |v|/pool\n");
    if (g_floor.empty()) std::printf("    none\n");
    for (auto& kv : g_floor) {
        int fid = kv.first / 100000, line = kv.first % 100000;
        const char* fn = fid == 1 ? "chem_rhs_net5.cc" : fid == 2 ? "chem_rhs_net15.cc" :
                         fid == 3 ? "chem_rhs_net16.cc" : fid == 4 ? "chem_rhs_net17.cc" :
                         fid == 5 ? "chem_rhs_net3.cc" : "?";
        const FloorStat& s = kv.second;
        std::printf("    %s:%d %-6s %ld | %ld / %ld / %ld / %ld | %.3e\n", fn, line, s.var, s.n,
                    s.bin[0], s.bin[1], s.bin[2], s.bin[3], s.maxrel);
    }
    std::printf("  check_abundances: %ld accepted, %ld rejected substeps; clipped species: %ld negative->0, %ld capped\n",
                g_clip.n_accept, g_clip.n_reject, g_clip.n_neg, g_clip.n_max);
    for (int e = 0; e < NEL; e++) if (el_active(e))
        std::printf("    %-3s max mass added by negative clip %.3e, removed by cap %.3e, overdraft after check %.3e (rel. to pool)\n",
                    ELNAME[e], g_clip.dmass_neg[e], g_clip.dmass_max[e], g_clip.max_over_after[e]);
#if CHEMISTRYNETWORK == 15 || CHEMISTRYNETWORK == 16
    std::printf("  QSS H3+: max 3*x(H3+) = %.3e, max 3*x(H3+)/x(HI) = %.3e, evaluations with 3*x(H3+) > x(HI): %ld, with HI = 0 and H3+ > 0: %ld\n",
                g_h3p_max, g_h3p_rel_max, g_h3p_cap, g_h3p_capped);
#endif
}

int main(int argc, char** argv)
{
    const char* csv = (argc > 1) ? argv[1] : "conservation.csv";
    std::printf("=== CONSERVATION AUDIT -- network %d (Fortran DVODE vs C++ CVODE) ===\n", NET);
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
    static ChemTables Cx;
    if (chem_tables_build(Cx, cfg, "gizmo/cooling/chemcool")) { std::printf("FATAL: chem_tables_build failed\n"); return 1; }
    g_cx = &Cx;
    ChemTables Ft; chem_bind_tables(Ft);

    const char* only = std::getenv("CONS_PHASES");   // e.g. "RCOI"
    auto want = [&](char p) { return !only || std::strchr(only, p); };
    int ok = 1;
    if (want('R')) { ok &= phase_R(Ft, "R1 (Fortran tables)"); ok &= phase_R(Cx, "R2 (C++-built tables)"); }
    if (want('C')) phase_C(csv);
    if (want('O')) { std::printf("\nPHASE O -- one-zone trajectories to 300 Myr\n"); ok &= phase_O(); }
    if (want('I')) phase_I();
    report_probes();
    std::printf("\n=== PARITY: %s ===\n", ok ? "ALL PASS" : "FAILURES PRESENT");
    chem_tables_free(Cx);
    return ok ? 0 : 1;
}
