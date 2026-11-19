// test_solver_failure.cc -- C++ chemistry solver failures, nets 5 and 17 (WLM runs 2026-11-17; D-58, D-59).
//
// Hot collisionally ionized gas sits at x_H+ = 1 (the TNOCHEM hand-off writes exactly 1); before D-58 CVODE trial
// states above 1 + EPS_MAX were rejected, the step collapsed to the floor and the substep loop could run out of budget.
//   PHASE A  the cells whose CVODE step hit the floor in the run (golden/net<N>_wlm_floor_cells.txt, from run.log),
//            replayed from the logged substep state over the rest of the call, C++ CVODE and Fortran DVODE
//            (states logged at %.9e, pools at %g; abhe = 0.1 in every row, the runs' fixed ABHE).
//   PHASE H  the hand-off state (hot-CIE ions) at bypass T 4.5e5..5e5 K, nH 1e-2..10, call dt 1e10..3e12 s.
//   PASS A/H: every call rc = 0 on both backends, no validate_input excess rejection, no CVODE floor hit, net 17: no
//            substep rejected for a D+ excess (the D+ kink, D-60), and the
//            final T of EACH backend within 1% of the converged solution (Uli 2026-11-19: against the converged
//            reference, not backend vs backend; production rtol unchanged). The reference is committed with the
//            test, golden/net<N>_converged_reference.txt: both backends at rtol 1e-8, written by this test built with
//            -DTEST_WRITE_CONVERGED_REFERENCE (one line per A/H call: tag id dt nH T_C T_F).
//            Every A/H result goes to the CSV (argument 1) at %.17g for a bitwise comparison.
//            Known harmless case (Uli's decision 2026-11-19): net 5 keeps Arepo's EPS_MAX = 1e-4, so in gas at
//            x_H+ ~ 0.9999+ CVODE's error-tolerated H+ overshoot can cross it; the recoverable validate_input rejections of
//            these four net-5 calls are recorded, not failed: logged cells 6796375 and 1080890, hand-off (cell 1080890's
//            inputs) nH 10, bypass T 4.99e5 K, dt 7.7044e11 and 3e12 s. rc, floor hits and the 1% T bar stay hard.
//   PHASE R  (TEST_SGCHEM_FORCED_SOLVER_FAILURE) failures forced per attempt (1 = substep budget of one,
//            100 = every substep rejected), D-59:
//            R1/R2  first attempt fails with code 1 / 100 on the logged cells: the retry rescues (rc 0, T within 1% of
//                   the converged reference); the retry starts from the last accepted state, bit for bit (an unforced solve over the
//                   accepted interval gives the same state; code 100: the entry state)
//            R3a    first attempt rejected (cold), the retry heats the cell above TNOCHEM and fails: TNOCHEM finish
//            R3b    first attempt heats the cell above TNOCHEM and fails, the CVODE retry fails: TNOCHEM finish
//            R4/R5  retry fails in a 100 K cell (code 100 / 1): abort with that code
//            R6     the dumps of R3a/R3b/R4 rebuild a NaN-filled cell (chem_cell_parse) and replay it bit for bit
//            R7     a flagged evaluation (CR heating set to 1e30) books no stale CR ionisation heating: at a substep
//                   start (a), at the final evaluation (b), at the failure point before the retry (c), in the TNOCHEM
//                   subcycle (d, nothing applied, nothing booked); (a)-(c) within 50% of the unflagged solve,
//                   (d) at most 1.5 x the unflagged finish
//
// Usage: test_solver_failure [csv]      (run where gizmo/cooling/sgchem holds the tables)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <csetjmp>
#include <string>
#include <vector>
#include <algorithm>

#include "chem_types.h"

extern "C" {
#include "GIZMO_config.h"
#include "cooling/sgchem/f2c.h"
#include "cooling/sgchem/sgchem_consts.h"
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

static jmp_buf g_fatal_jmp;
static int g_fatal_armed = 0;
static int g_fatal_code = 0;
void sgchem_fatal_(int* code)
{
    g_fatal_code = *code;
    if (g_fatal_armed) longjmp(g_fatal_jmp, 1);
    std::printf("  [unarmed sgchem_fatal %d]\n", *code);
    std::exit(1);
}
}

#if CHEMISTRYNETWORK != 17 && CHEMISTRYNETWORK != 5
#error "test_solver_failure covers networks 5 and 17"
#endif

void chem_bind_tables(ChemTables& T);
void chem_bind_cell(ChemCell& c, const ChemTables& T, int net);
extern thread_local long cxx_stat_substeps, cxx_stat_retries, cxx_stat_floor_hits;
extern thread_local long chem_excess_states;
#ifdef TEST_PARENT_DRIVER
static long chem_clip_excess_fail[CHEM_NSPEC_MAX];
#else
extern thread_local long chem_clip_excess_fail[CHEM_NSPEC_MAX];
#endif
#ifdef TEST_PARENT_DRIVER   // bitwise comparison against the driver before D-59
static long cxx_stat_failure_retries = 0, cxx_stat_tnochem_finishes = 0;
#else
extern thread_local long cxx_stat_failure_retries, cxx_stat_tnochem_finishes;
#endif
#ifdef TEST_SGCHEM_FORCED_SOLVER_FAILURE
extern thread_local int chem_test_force_failure[2];
extern thread_local double chem_test_retry_start_y[CHEM_NSPEC_MAX];
extern thread_local double chem_test_retry_start_t;
extern thread_local std::string chem_test_last_dump;
extern thread_local long chem_test_flag_rhs[4];
#endif

static const int NET = CHEMISTRYNETWORK;
static const double SGCHEM_TNOCHEM_TEST = 5e5;   // cvode_driver.cc default
static const double DL_SHIELD = 3.086e19;
static double g_rt_heat = 0.0;   // photoheating for the phase-R hot cases (net 17: He+, net 5: HI)

// the solver inputs the run logged for a cell
struct LoggedCell {
    long id; double dt, yn, dl, divv;
    double y[NSPEC];   // species..., energy
    double tdust, abundc, abundo, abundsi, abundD, dgr, G0, G0_LW, G0_dust, crir;
};

// net<N>_wlm_floor_cells.txt: id t_call t_rem yn dl divv y[NSPEC] tdust abhe abundc abundo abundsi abundD dgr G0 G0_LW G0_dust crir ph1
static std::vector<LoggedCell> load_cells(const char* path, std::vector<double>& ph1)
{
    std::vector<LoggedCell> v;
    FILE* f = std::fopen(path, "r");
    if (!f) return v;
    char line[4096];
    while (std::fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        LoggedCell L; double tcall, abhe, p1;
        char* p = line; char* end;
        double x[64]; int n = 0;
        while (n < 64) { x[n] = std::strtod(p, &end); if (end == p) break; n++; p = end; }
        if (n != 6 + NSPEC + 12) continue;
        int k = 0;
        L.id = (long)x[k++]; tcall = x[k++]; L.dt = x[k++]; L.yn = x[k++]; L.dl = x[k++]; L.divv = x[k++];
        for (int i = 0; i < NSPEC; i++) L.y[i] = x[k++];
        L.tdust = x[k++]; abhe = x[k++]; L.abundc = x[k++]; L.abundo = x[k++]; L.abundsi = x[k++]; L.abundD = x[k++];
        L.dgr = x[k++]; L.G0 = x[k++]; L.G0_LW = x[k++]; L.G0_dust = x[k++]; L.crir = x[k++]; p1 = x[k++];
        (void)tcall; (void)abhe;
        v.push_back(L); ph1.push_back(p1);
    }
    std::fclose(f);
    return v;
}

static void set_cell_commons(const LoggedCell& L)
{
    COOLRC.G0 = L.G0; COOLRC.G0_LW = L.G0_LW; COOLRC.G0_dust = L.G0_dust;
    COOLRC.cosmic_ray_ion_rate = L.crir;
    COOLRC.cr_energy_density = 0.0;
    COOLRC.dust_to_gas_ratio = L.dgr;
    COOLRC.redshift = 0.0; COOLRC.pdv_term = 0.0;
    COOLRC.abundD  = L.abundD;
    COOLRC.abundc  = L.abundc;
    COOLRC.abundo  = L.abundo;
    COOLRC.abundsi = L.abundsi;
    COOLRC.abundM  = L.abundsi;
    COOLRC.abundN  = 7.6e-5 * L.abundo / 3.2e-4;
    COOLRC.rt_phot_HI = COOLRC.rt_phot_HeI = COOLRC.rt_phot_HeII = 0;
    COOLRC.rt_heat_HI = COOLRC.rt_heat_HeI = COOLRC.rt_heat_HeII = 0;
#if CHEMISTRYNETWORK == 17
    COOLRC.rt_heat_HeII = g_rt_heat;
#else
    COOLRC.rt_heat_HI = g_rt_heat;
#endif
#ifdef WSS_CIE_COOL
    // chem_interface.cc order: He C Mg O Fe Si H N Ne S Ca Zn (mass fractions), metals scaled by the cell's O
    const double zsolar[12] = {0.2806, 2.07e-3, 5.91e-4, 5.49e-3, 1.10e-3, 6.83e-4,
                               0.7065, 8.36e-4, 1.41e-3, 4.09e-4, 6.44e-5, 1.10e-3};
    const double Zs = L.abundo / 3.2e-4;
    for (int k = 0; k < 12; k++) COOLRC.Zmass[k] = (k == 0 || k == 6) ? zsolar[k] : zsolar[k] * Zs;
#endif
    COOLIC.id_current = (int)L.id;
}

// sgchem.cc ChemInitParams values
static void init_commons(const LoggedCell& L)
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
    set_cell_commons(L);
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

// electrons and temperature as rate_eq_ulli / rate_eq_simple build them
static double abe_of(const double* y)
{
    double cp = std::max(COOLRC.abundc - y[ICO], 0.0);
#if CHEMISTRYNETWORK == 17
    return y[IHP] + y[IHEP] + 2.0 * y[IHEPP] + cp + y[IDP] + COOLRC.abundsi;
#else
    return y[IHP] + cp + COOLRC.abundsi;
#endif
}
static double temp_of(const ChemTables& T, double yn, double energy, const double* y)
{
    double yntot = (1.0 + CHEM_COMP.abhe - y[IH2] + abe_of(y)) * yn;
    double t, g;
    cxx_lookup_temp_gamma(y[IH2], CHEM_COMP.abhe, energy / (yntot * CHEM_KBOLTZ), &t, &g, T);
    return t;
}
#ifdef TEST_SGCHEM_FORCED_SOLVER_FAILURE
static double energy_at(const ChemTables& T, double yn, double T0, const double* y)
{
    double yntot = (1.0 + CHEM_COMP.abhe - y[IH2] + abe_of(y)) * yn;
    double e = 1.5 * yntot * CHEM_KBOLTZ * T0;
    for (int k = 0; k < 60; k++) {
        double t = temp_of(T, yn, e, y);
        if (std::fabs(t - T0) < 1e-12 * T0) break;
        e *= T0 / t;
    }
    return e;
}
#endif
// TNOCHEM bypass temperature (H electrons only, cvode_driver.cc) and the energy that gives Tb
static double bypass_T(const ChemTables& T, double yn, double e, const double* y)
{
    double yntot = (1.0 + CHEM_COMP.abhe + y[IHP] - y[IH2]) * yn;
    double t, g;
    cxx_lookup_temp_gamma(y[IH2], CHEM_COMP.abhe, e / (yntot * CHEM_KBOLTZ), &t, &g, T);
    return t;
}
static double energy_bypass_T(const ChemTables& T, double yn, double Tb, const double* y)
{
    double yntot = (1.0 + CHEM_COMP.abhe + y[IHP] - y[IH2]) * yn;
    double e = 1.5 * yntot * CHEM_KBOLTZ * Tb;
    for (int k = 0; k < 60; k++) {
        double t = bypass_T(T, yn, e, y);
        if (std::fabs(t - Tb) < 1e-12 * Tb) break;
        e *= Tb / t;
    }
    return e;
}
// hot-CIE ion set of the TNOCHEM path (cvode_driver.cc set_hot_cie)
static void hot_cie(double* y, double abundD)
{
    for (int i = 0; i < TRAC_NUM; i++) y[i] = 0.0;
    y[IHP] = 1.0;
#if CHEMISTRYNETWORK == 17
    y[IHEPP] = CHEM_COMP.abhe; y[IDP] = abundD;
#else
    (void)abundD;
#endif
}

static ChemTables* g_cx = nullptr;

struct Res { int rc; long substeps, retries, floors, excess, fretries, tfinish, dpfail; double e, T, crh; double y[NSPEC]; };

#ifndef TEST_SGCHEM_FORCED_SOLVER_FAILURE
static Res run_fortran(const LoggedCell& L, double dt, double yn, double e0, const double* y0)
{
    Res r; double y[NSPEC]; for (int i = 0; i < NSPEC; i++) y[i] = y0[i];
    double e = e0;
    set_cell_commons(L);
    double Tf = temp_of(*g_cx, yn, e, y);
    set_columns(yn, y[IH2], y[ICO]);
    double rpar[NRPAR]; std::memset(rpar, 0, sizeof(rpar));
    rpar[0] = yn;
#if CHEMISTRYNETWORK == 17
    double a2 = y[IH2], ahd = y[IHD], aco = y[ICO];
#else
    double a2 = y[IH2], ahd = 0.0, aco = y[ICO];
#endif
    COOLRC.tdust = L.tdust;
    CALC_PHOTO_WRAPPER(&Tf, rpar, &a2, &ahd, &aco);
    double dts = dt, dl = L.dl, dv = L.divv, ce = 0.0;
    r.rc = 0; g_fatal_code = 0;
    g_fatal_armed = 1;
    if (setjmp(g_fatal_jmp) == 0) EVOLVE_ABUNDANCES(&dts, &dl, &yn, &dv, &e, y, &ce);
    else r.rc = g_fatal_code ? g_fatal_code : -1;
    g_fatal_armed = 0;
    r.substeps = r.retries = r.floors = r.excess = r.fretries = r.tfinish = r.dpfail = -1; r.crh = 0.0;
    r.e = e; for (int i = 0; i < NSPEC; i++) r.y[i] = y[i];
    r.y[ITMP] = e;
    r.T = temp_of(*g_cx, yn, e, y);
    return r;
}

#endif
// the C++ cell as GIZMO builds it (chem_bind_cell + cxx_calc_photo)
static void make_cell(ChemCell& c, const LoggedCell& L, double yn, const double* y, double e)
{
    set_cell_commons(L);
    double Tc = temp_of(*g_cx, yn, e, y);
    set_columns(yn, y[IH2], y[ICO]);
    chem_bind_cell(c, *g_cx, NET);
    c.yn = yn; c.dl = L.dl; c.divv = L.divv; c.tdust = L.tdust;
#if CHEMISTRYNETWORK == 17
    cxx_calc_photo(Tc, y[IH2], y[IHD], y[ICO], c);
#else
    cxx_calc_photo(Tc, y[IH2], 0.0, y[ICO], c);
#endif
}

static Res solve_cell(ChemCell& c, double dt, double yn, double e0, const double* y0)
{
    Res r; double y[NSPEC]; for (int i = 0; i < NSPEC; i++) y[i] = y0[i];
    double e = e0;
    const long s0 = cxx_stat_substeps, r0 = cxx_stat_retries, f0 = cxx_stat_floor_hits;
    const long x0 = chem_excess_states, q0 = cxx_stat_failure_retries, t0 = cxx_stat_tnochem_finishes;
#if CHEMISTRYNETWORK == 17
    const long d0 = chem_clip_excess_fail[IDP];
#endif
    r.rc = cxx_evolve_abundances(dt, c.dl, yn, c.divv, &e, y, c);
    r.substeps = cxx_stat_substeps - s0; r.retries = cxx_stat_retries - r0; r.floors = cxx_stat_floor_hits - f0;
    r.excess = chem_excess_states - x0; r.fretries = cxx_stat_failure_retries - q0; r.tfinish = cxx_stat_tnochem_finishes - t0;
    r.crh = c.cr_ion_heat_int;
#if CHEMISTRYNETWORK == 17
    r.dpfail = chem_clip_excess_fail[IDP] - d0;
#else
    r.dpfail = 0;
#endif
    r.e = e; for (int i = 0; i < NSPEC; i++) r.y[i] = y[i];
    r.y[ITMP] = e;
    r.T = temp_of(*g_cx, yn, e, y);
    return r;
}

static Res run_cxx(const LoggedCell& L, double dt, double yn, double e0, const double* y0, double* ph1)
{
    ChemCell c; make_cell(c, L, yn, y0, e0);
    if (ph1) *ph1 = c.phrates[0];
    return solve_cell(c, dt, yn, e0, y0);
}

static FILE* g_csv = nullptr;
static std::vector<double> g_refT;   // converged final T per A/H call, in call order
#ifndef TEST_SGCHEM_FORCED_SOLVER_FAILURE
static size_t g_row = 0;
#endif

#ifndef TEST_WRITE_CONVERGED_REFERENCE
static int load_reference(const char* path)
{
    FILE* f = std::fopen(path, "r");
    if (!f) return 0;
    char line[512], tag[64];
    long id; double dt, yn, TC, TF;
    while (std::fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        if (std::sscanf(line, "%63s %ld %lf %lf %lf %lf", tag, &id, &dt, &yn, &TC, &TF) == 6) g_refT.push_back(TC);
    }
    std::fclose(f);
    return (int)g_refT.size();
}

#endif
#ifndef TEST_SGCHEM_FORCED_SOLVER_FAILURE
// one C++/DVODE pair against the A/H gate; returns 1 if it fails
static int check_pair(const char* tag, const LoggedCell& L, double dt, double yn, double e0, const double* y0)
{
    double ph1 = 0.0;
    Res c = run_cxx(L, dt, yn, e0, y0, &ph1);
    Res f = run_fortran(L, dt, yn, e0, y0);
    const size_t k = g_row++;
#ifdef TEST_WRITE_CONVERGED_REFERENCE
    if (g_csv) std::fprintf(g_csv, "%s %ld %.17g %.17g %.17g %.17g\n", tag, L.id, dt, yn, c.T, f.T);
    const double Tref = c.T;
    (void)k;
#else
    const double Tref = (k < g_refT.size()) ? g_refT[k] : NAN;
#endif
    double relC = std::fabs(c.T - Tref) / Tref, relF = std::fabs(f.T - Tref) / Tref;
    // net 5 calls with known recoverable H+ rejections (Uli 2026-11-19, see the header)
    int known = 0;
#if CHEMISTRYNETWORK == 5
    known = (std::strcmp(tag, "logged") == 0 && (L.id == 6796375 || L.id == 1080890))
         || (std::strcmp(tag, "hand-off") == 0 && L.id == 1080890 && yn == 10.0 && (dt == 7.7044e11 || dt == 3e12)
             && std::fabs(bypass_T(*g_cx, yn, e0, y0) - 4.99e5) < 1.0);
#endif
    int bad = (c.rc != 0) || (f.rc != 0) || (c.excess != 0 && !known) || (c.floors != 0) || (c.dpfail != 0)
           || !(relC <= 0.01) || !(relF <= 0.01);
    std::printf("  %-10s %8ld %10.4g %9.4g %9.4g | %4d %6ld %6ld %7ld %5ld %10.4g %8.1e | %4d %10.4g %8.1e | %10.4g%s\n",
                tag, L.id, dt, yn, temp_of(*g_cx, yn, e0, y0), c.rc, c.substeps, c.retries, c.excess, c.floors, c.T, relC,
                f.rc, f.T, relF, Tref, bad ? "  <-- FAIL" : (known && c.excess ? "  (known: recoverable H+ rejections)" : ""));
    if (c.dpfail) std::printf("      %ld substeps rejected for a D+ excess\n", c.dpfail);
    if (bad)
        for (const Res* r : {&c, &f}) {
            std::printf("      %s y:", r == &c ? "C++    " : "Fortran");
            for (int i = 0; i < NSPEC; i++) std::printf(" %.9e", r->y[i]);
            std::printf("\n");
        }
#ifndef TEST_WRITE_CONVERGED_REFERENCE
    if (g_csv) {
        std::fprintf(g_csv, "%s,%ld,%.17g,%.17g,%d", tag, L.id, dt, yn, c.rc);
        for (int i = 0; i < NSPEC; i++) std::fprintf(g_csv, ",%.17g", c.y[i]);
        std::fprintf(g_csv, ",%.17g\n", c.crh);
    }
#endif
    std::fflush(stdout);
    return bad;
}
#endif

int main(int argc, char** argv)
{
    std::printf("=== SOLVER FAILURE, NETWORK %d (WLM runs; D-58, D-59) ===\n", NET);
    if (argc > 1) g_csv = std::fopen(argv[1], "w");
    char cellfile[256];
    std::snprintf(cellfile, sizeof(cellfile), "gizmo/cooling/sgchem/cxx/golden/net%d_wlm_floor_cells.txt", NET);
    std::vector<double> ph1_log;
    std::vector<LoggedCell> cells = load_cells(cellfile, ph1_log);
    if (cells.empty()) { std::printf("FATAL: no logged cells in %s\n", cellfile); return 1; }
    char reffile[256];
    std::snprintf(reffile, sizeof(reffile), "gizmo/cooling/sgchem/cxx/golden/net%d_converged_reference.txt", NET);
#ifndef TEST_WRITE_CONVERGED_REFERENCE
    std::printf("converged reference: %d calls from %s\n", load_reference(reffile), reffile);
#else
    (void)reffile;
#endif
    init_commons(cells[0]);
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
    cfg.dust_to_gas_ratio = 1.0;
    cfg.cosmic_ray_ion_rate = 3e-17;
    cfg.abhe = CHEM_COMP.abhe;
    ChemTables Cx;
    if (chem_tables_build(Cx, cfg, "gizmo/cooling/sgchem")) { std::printf("FATAL: chem_tables_build\n"); return 1; }
    g_cx = &Cx;
    const LoggedCell& L0 = cells.back();

#ifdef TEST_SGCHEM_FORCED_SOLVER_FAILURE
    auto force = [](int a0, int a1) { chem_test_force_failure[0] = a0; chem_test_force_failure[1] = a1; };
    int nbadR = 0;
    // ---------------- R1/R2: first attempt fails, the retry rescues and starts from the accepted state ----------------
    for (int mode : {1, 100}) {
        const int tagR = (mode == 1) ? 1 : 2;
        std::printf("PHASE R%d -- first attempt fails with code %d, retry free, %zu logged cells\n", tagR, mode, cells.size());
        long nretry = 0; int nbad = 0, nstart = 0, nstart_checked = 0, nstart_later = 0;
        for (size_t kc = 0; kc < cells.size(); kc++) {
            const LoggedCell& L = cells[kc];
            force(mode, 0);
            chem_test_retry_start_t = -1.0;
            Res c = run_cxx(L, L.dt, L.yn, L.y[ITMP], L.y, nullptr);
            force(0, 0);
            const double Tref = (kc < g_refT.size()) ? g_refT[kc] : NAN;
            double relT = std::fabs(c.T - Tref) / Tref;
            int bad = (c.rc != 0) || (c.tfinish != 0) || !(relT <= 0.01);
            // the retry's start state against an independent unforced solve over the accepted interval
            int start_bad = 0;
            if (c.fretries == 1) {
                nstart_checked++;
                const double ts = chem_test_retry_start_t;
                double yref[NSPEC];
                if (ts == 0.0) {
                    for (int i = 0; i < NSPEC; i++) yref[i] = L.y[i];
                } else {
                    Res u = run_cxx(L, ts, L.yn, L.y[ITMP], L.y, nullptr);
                    for (int i = 0; i < NSPEC; i++) yref[i] = u.y[i];
                    nstart_later++;
                }
                for (int i = 0; i < NSPEC; i++) start_bad |= (chem_test_retry_start_y[i] != yref[i]);
                start_bad |= (mode == 100) && (ts != 0.0);   // code 100 rejects every substep: retry from the entry state
                nstart += start_bad;
            }
            nretry += c.fretries; nbad += bad;
            if (bad || start_bad)
                std::printf("  %8ld rc=%d retries=%ld tnochem=%ld T_C=%.6g T_ref=%.6g relT=%.1e retry start t=%.17g %s  <-- FAIL\n",
                            L.id, c.rc, c.fretries, c.tfinish, c.T, Tref, relT, chem_test_retry_start_t,
                            start_bad ? "(start state differs)" : "");
        }
        int ok = (nbad == 0) && (nstart == 0) && (nretry > 0) && (nstart_checked == nretry)
              && (mode != 100 || nretry == (long)cells.size()) && (mode != 1 || nstart_later > 0);
        std::printf("  GATE R%d: %d cells fail, %ld retries (%d from t > 0), %d of %d retry start states differ: %s\n",
                    tagR, nbad, nretry, nstart_later, nstart, nstart_checked, ok ? "PASS" : "FAIL");
        nbadR += !ok;
    }

    // ---------------- R3: TNOCHEM finish after heating across TNOCHEM ----------------
    // net 17: He in He+ with He+ photoheating; net 5: half-neutral H with HI photoheating
    std::vector<std::string> dumps; std::vector<std::pair<int, int>> dump_force;
    std::vector<double> dump_e; std::vector<std::vector<double>> dump_y; std::vector<int> dump_rc;
    for (int variant = 0; variant < 2; variant++) {
        double y[NSPEC] = {0};
        hot_cie(y, L0.abundD);
#if CHEMISTRYNETWORK == 17
        y[IHEPP] = 0.0; y[IHEP] = CHEM_COMP.abhe;
        g_rt_heat = 1e-17;
#else
        y[IHP] = 0.5;
        g_rt_heat = 1e-15;
#endif
        const double yn = 1.0, dt = 1e11;
        y[ITMP] = energy_bypass_T(Cx, yn, 4.97e5, y);
        const int a0 = (variant == 0) ? 100 : 1;
        force(a0, 1);
        chem_test_retry_start_t = -1.0; chem_test_last_dump.clear();
        Res c = run_cxx(L0, dt, yn, y[ITMP], y, nullptr);
        force(0, 0);
        double yh[NSPEC]; hot_cie(yh, L0.abundD);
        int hotset = 1;
        for (int i = 0; i < TRAC_NUM; i++) hotset &= (c.y[i] == yh[i]);
        double rs_bypass_T = -1.0;
        if (chem_test_retry_start_t >= 0.0)
            rs_bypass_T = bypass_T(Cx, yn, chem_test_retry_start_y[ITMP], chem_test_retry_start_y);
        // R3b: attempt 0 heated the cell above TNOCHEM before failing, the retry still ran CVODE from there
        int path_ok = (variant == 0) ? (chem_test_retry_start_t == 0.0 && rs_bypass_T < SGCHEM_TNOCHEM_TEST)
                                     : (chem_test_retry_start_t > 0.0 && rs_bypass_T > SGCHEM_TNOCHEM_TEST);
        int ok = (c.rc == 0) && (c.fretries == 1) && (c.tfinish == 1) && hotset && path_ok && std::isfinite(c.e)
              && (c.e > y[ITMP]) && !chem_test_last_dump.empty();
        std::printf("PHASE R3%c -- %s: rc=%d retries=%ld tnochem=%ld hot-CIE set %d retry start t=%.4g bypass T %.4g "
                    "e_end/e0=%.4f dump %zu B: %s\n", variant == 0 ? 'a' : 'b',
                    variant == 0 ? "first attempt rejected, retry heats across TNOCHEM and fails"
                                 : "first attempt heats across TNOCHEM and fails, CVODE retry fails",
                    c.rc, c.fretries, c.tfinish, hotset, chem_test_retry_start_t, rs_bypass_T, c.e / y[ITMP],
                    chem_test_last_dump.size(), ok ? "PASS" : "FAIL");
        nbadR += !ok;
        dumps.push_back(chem_test_last_dump); dump_force.push_back({a0, 1}); dump_rc.push_back(c.rc);
        dump_e.push_back(c.e); dump_y.push_back(std::vector<double>(c.y, c.y + NSPEC));
        g_rt_heat = 0.0;
    }

    // ---------------- R4/R5: retry fails in a 100 K cell: abort ----------------
    {
        LoggedCell Lc = L0;
        Lc.dgr = 0.1; Lc.tdust = 10.0;
        double y[NSPEC] = {0};
        y[IH2] = 1e-3; y[IHP] = 1e-4;
#if CHEMISTRYNETWORK == 17
        y[IDP] = 1e-4 * Lc.abundD;
#endif
        const double yn = 100.0, dt = 3e12;
        y[ITMP] = energy_at(Cx, yn, 100.0, y);
        Res u = run_cxx(Lc, dt, yn, y[ITMP], y, nullptr);
        std::printf("PHASE R4/R5 -- 100 K cell, nH 100, dt 3e12 s: unforced rc=%d in %ld substeps, T_end %.4g\n",
                    u.rc, u.substeps, u.T);
        for (int mode : {100, 1}) {
            force(mode, mode); chem_test_last_dump.clear();
            Res c = run_cxx(Lc, dt, yn, y[ITMP], y, nullptr);
            force(0, 0);
            int ok = (u.rc == 0) && (u.substeps > 1) && (c.rc == mode) && (c.fretries == 1) && (c.tfinish == 0)
                  && !chem_test_last_dump.empty();
            std::printf("PHASE R%d -- retry fails with code %d at 100 K: rc=%d retries=%ld tnochem=%ld dump %zu B: %s\n",
                        mode == 100 ? 4 : 5, mode, c.rc, c.fretries, c.tfinish, chem_test_last_dump.size(), ok ? "PASS" : "FAIL");
            nbadR += !ok;
            if (mode == 100) {
                dumps.push_back(chem_test_last_dump); dump_force.push_back({100, 100}); dump_rc.push_back(c.rc);
                dump_e.push_back(c.e); dump_y.push_back(std::vector<double>(c.y, c.y + NSPEC));
            }
        }
    }

    // ---------------- R6: rebuild each dumped cell from its dump alone and replay it ----------------
    for (size_t k = 0; k < dumps.size(); k++) {
        ChemCell c;
        std::memset(static_cast<void*>(&c), 0xff, sizeof(c));   // every field the dump misses is NaN (or -1)
        ChemCellEntry e;
        const int missing = chem_cell_parse(dumps[k].c_str(), c, e);
        c.tab = &Cx;
        double y[NSPEC];
        for (int i = 0; i < NSPEC; i++) y[i] = e.y_entry[i];
        force(dump_force[k].first, dump_force[k].second); chem_test_last_dump.clear();
        double en = y[ITMP];
        double yy[NSPEC]; for (int i = 0; i < NSPEC; i++) yy[i] = y[i];
        const int rc = cxx_evolve_abundances(e.time, c.dl, c.yn, c.divv, &en, yy, c);
        force(0, 0);
        // the dump's first line carries the per-thread running count; the cell state starts on the second line
        auto body = [](const std::string& d) { const size_t nl = d.find('\n'); return nl == std::string::npos ? d : d.substr(nl); };
        int same = (rc == dump_rc[k]) && !dumps[k].empty() && (body(chem_test_last_dump) == body(dumps[k]));
        if (rc == 0) { same &= (en == dump_e[k]); for (int i = 0; i < TRAC_NUM; i++) same &= (yy[i] == dump_y[k][i]); }
        int ok = (missing == 0) && same;
        std::printf("PHASE R6 -- replay of dump %zu from the dump alone: %d fields missing, rc %d, dump and result %s: %s\n",
                    k, missing, rc, same ? "identical" : "DIFFER", ok ? "PASS" : "FAIL");
        nbadR += !ok;
    }
    // ---------------- R7: flagged evaluations book no stale CR heating ----------------
    {
        const LoggedCell* Lm = &cells[0];
        Res best{}; best.substeps = 0;
        for (const LoggedCell& L : cells) {   // the logged cell with the most substeps
            Res u = run_cxx(L, L.dt, L.yn, L.y[ITMP], L.y, nullptr);
            if (u.substeps > best.substeps) { best = u; Lm = &L; }
        }
        for (int site = 0; site < 2; site++) {
            if (site == 0) { chem_test_flag_rhs[0] = 2; chem_test_flag_rhs[1] = 0; }
            else           { chem_test_flag_rhs[0] = 0; chem_test_flag_rhs[1] = 1; }
            Res c = run_cxx(*Lm, Lm->dt, Lm->yn, Lm->y[ITMP], Lm->y, nullptr);
            chem_test_flag_rhs[0] = chem_test_flag_rhs[1] = 0;
            const double ratio = c.crh / best.crh;
            int ok = (best.substeps >= 2) && (best.crh > 0.0) && std::isfinite(c.crh) && (ratio > 0.5) && (ratio < 1.5)
                  && (c.rc == 0);
            std::printf("PHASE R7%c -- flagged %s, CR heating 1e30 (cell %ld, %ld substeps): integral %.6g vs unflagged %.6g "
                        "(ratio %.4g): %s\n", site == 0 ? 'a' : 'b', site == 0 ? "evaluation at the start of substep 2"
                        : "final evaluation", Lm->id, best.substeps, c.crh, best.crh, ratio, ok ? "PASS" : "FAIL");
            nbadR += !ok;
        }
    }
    // R7c: flagged evaluation at the failure point with an accepted substep pending (retry starts at t > 0), then the retry
    {
        const LoggedCell* Lm = nullptr;
        long most = 0;
        for (const LoggedCell& L : cells) {
            force(1, 0); chem_test_retry_start_t = -1.0;
            Res u = run_cxx(L, L.dt, L.yn, L.y[ITMP], L.y, nullptr);
            force(0, 0);
            if (chem_test_retry_start_t > 0.0 && u.substeps > most) { most = u.substeps; Lm = &L; }
        }
        if (!Lm) { std::printf("PHASE R7c -- no logged cell retries from t > 0: FAIL\n"); nbadR++; Lm = &cells[0]; }
        force(1, 0);
        Res ref = run_cxx(*Lm, Lm->dt, Lm->yn, Lm->y[ITMP], Lm->y, nullptr);
        chem_test_flag_rhs[2] = 1;
        Res c = run_cxx(*Lm, Lm->dt, Lm->yn, Lm->y[ITMP], Lm->y, nullptr);
        chem_test_flag_rhs[2] = 0;
        force(0, 0);
        const double ratio = c.crh / ref.crh;
        int ok = (ref.fretries == 1) && (ref.crh > 0.0) && std::isfinite(c.crh) && (ratio > 0.5) && (ratio < 1.5) && (c.rc == 0);
        std::printf("PHASE R7c -- flagged evaluation at the failure point, CR heating 1e30 (cell %ld): integral %.6g vs unflagged "
                    "%.6g (ratio %.4g): %s\n", Lm->id, c.crh, ref.crh, ratio, ok ? "PASS" : "FAIL");
        nbadR += !ok;
    }
    // R7d: flagged first evaluation of the TNOCHEM subcycle after a failed retry (the R3a cell)
    {
        double y[NSPEC] = {0};
        hot_cie(y, L0.abundD);
#if CHEMISTRYNETWORK == 17
        y[IHEPP] = 0.0; y[IHEP] = CHEM_COMP.abhe;
        g_rt_heat = 1e-17;
#else
        y[IHP] = 0.5;
        g_rt_heat = 1e-15;
#endif
        y[ITMP] = energy_bypass_T(Cx, 1.0, 4.97e5, y);
        force(100, 1);
        Res ref = run_cxx(L0, 1e11, 1.0, y[ITMP], y, nullptr);
        chem_test_flag_rhs[3] = 1;
        Res c = run_cxx(L0, 1e11, 1.0, y[ITMP], y, nullptr);
        chem_test_flag_rhs[3] = 0;
        force(0, 0);
        g_rt_heat = 0.0;
        int ok = (ref.tfinish == 1) && (c.tfinish == 1) && std::isfinite(c.crh) && (c.crh >= 0.0) && (c.crh <= 1.5 * ref.crh)
              && (c.rc == 0);
        std::printf("PHASE R7d -- flagged first TNOCHEM subcycle evaluation, CR heating 1e30: integral %.6g vs unflagged %.6g: %s\n",
                    c.crh, ref.crh, ok ? "PASS" : "FAIL");
        nbadR += !ok;
    }
    const int ok = (nbadR == 0);
    std::printf("  GATE R: %s\n", ok ? "PASS" : "FAIL");
#else
    const char* head = "  %-10s %8s %10s %9s %9s | %4s %6s %6s %7s %5s %10s %8s | %4s %10s %8s | %10s\n";
    // ---------------- PHASE A ----------------
    std::printf("PHASE A -- %zu logged floor cells, replayed over the rest of the call\n", cells.size());
    std::printf(head, "", "ID", "dt", "yn", "T0", "rcC", "sub", "ret", "excess", "floor", "T_C", "relC", "rcF", "T_F", "relF", "T_ref");
    int nbadA = 0, nph1 = 0;
    for (size_t k = 0; k < cells.size(); k++) {
        const LoggedCell& L = cells[k];
        double ph1 = 0.0;
        nbadA += check_pair("logged", L, L.dt, L.yn, L.y[ITMP], L.y);
        (void)run_cxx(L, 1.0, L.yn, L.y[ITMP], L.y, &ph1);
        nph1 += (std::fabs(ph1 / ph1_log[k] - 1.0) > 1e-4);   // inputs reproduce the run's H2 photodissociation rate
    }
    std::printf("  GATE A: %d of %zu cells fail; %d cells with ph1 off the logged value\n", nbadA, cells.size(), nph1);

    // ---------------- PHASE H ----------------
    std::printf("PHASE H -- TNOCHEM hand-off state (hot-CIE ions) over T, nH, dt\n");
    std::printf(head, "", "ID", "dt", "yn", "T0", "rcC", "sub", "ret", "excess", "floor", "T_C", "relC", "rcF", "T_F", "relF", "T_ref");
    int nbadH = 0;
    for (double yn : {1e-2, 0.0639667, 0.3, 1.0, 10.0})
    for (double Tb : {4.5e5, 4.8e5, 4.99e5})
    for (double dt : {1e10, 1e11, 7.7044e11, 3e12}) {
        double y[NSPEC];
        hot_cie(y, L0.abundD);
        y[ITMP] = energy_bypass_T(Cx, yn, Tb, y);
        nbadH += check_pair("hand-off", L0, dt, yn, y[ITMP], y);
    }
    std::printf("  GATE H: %d of 60 calls fail\n", nbadH);
    const int ok = (nbadA == 0) && (nph1 == 0) && (nbadH == 0);
    std::printf("  GATE rc = 0, no validate_input excess rejection (net 5: four known calls recorded), no CVODE floor hit, "
                "no D+ excess rejection, each backend's T within 1%% of the converged reference: %s\n", ok ? "PASS" : "FAIL");
#endif
    if (g_csv) std::fclose(g_csv);
    chem_tables_free(Cx);
    std::printf("=== SOLVER FAILURE NET %d: %s ===\n", NET, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
