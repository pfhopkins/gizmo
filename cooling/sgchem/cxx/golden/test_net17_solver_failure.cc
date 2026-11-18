// test_net17_solver_failure.cc -- net 17 solver failure of the WLM run (2026-11-17, ID 4819441 "nstep exceeded").
//
// Hot collisionally ionized gas sits at x_H+ = 1 (the TNOCHEM hand-off writes exactly 1); CVODE trial states
// above 1 + EPS_MAX are rejected by validate_input, the step collapses to the floor and the substep loop
// can run out of budget.
//   PHASE A  the 66 cells whose CVODE step hit the floor in the run (net17_wlm_floor_cells.txt, from run.log),
//            replayed from the logged substep state over the rest of the call, C++ CVODE and Fortran DVODE.
//   PHASE H  the hand-off state (H+ = 1, He++ = abhe, D+ = abundD) at bypass T 4.5e5..5e5 K,
//            nH 1e-2..10, call dt 1e10..3e12 s, with the run's pools and rates.
//   PASS: every call rc = 0, no validate_input excess rejection, no CVODE floor hit, T within 1% of DVODE.
//   PHASE R  (built with -DCHEM_NSTEP=1, and SGCHEM_SOLVER_FAILURE_ENERGY_ONLY_COOLING to pass) every phase-A cell
//            with a forced solver failure: rc = 0, species within their pools, finite positive energy.
//
// Usage: test_net17_solver_failure      (run where gizmo/cooling/sgchem holds the tables)

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

#if CHEMISTRYNETWORK != 17
#error "test_net17_solver_failure covers network 17"
#endif

void chem_bind_tables(ChemTables& T);
void chem_bind_cell(ChemCell& c, const ChemTables& T, int net);
extern thread_local long cxx_stat_substeps, cxx_stat_retries, cxx_stat_floor_hits;
#ifdef SGCHEM_SOLVER_FAILURE_ENERGY_ONLY_COOLING
extern thread_local long cxx_stat_recoveries;
#else
static long cxx_stat_recoveries = 0;
#endif
extern thread_local long chem_excess_states;

static const int NET = 17;
static const double DL_SHIELD = 3.086e19;

// the solver inputs chem_print_cell_context logged for a cell
struct LoggedCell {
    long id; double dt, yn, dl, divv;
    double y[NSPEC];   // species..., energy
    double tdust, abundc, abundo, abundsi, abundD, dgr, G0, G0_LW, G0_dust, crir;
};

// net17_wlm_floor_cells.txt: id t_call t_rem yn dl divv y[8] tdust abhe abundc abundo abundsi abundD dgr G0 G0_LW G0_dust crir ph1
static std::vector<LoggedCell> load_cells(const char* path, std::vector<double>& ph1)
{
    std::vector<LoggedCell> v;
    FILE* f = std::fopen(path, "r");
    if (!f) return v;
    char line[4096];
    while (std::fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        LoggedCell L; double tcall, abhe, p1;
        int n = std::sscanf(line, "%ld %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                            &L.id, &tcall, &L.dt, &L.yn, &L.dl, &L.divv, &L.y[0], &L.y[1], &L.y[2], &L.y[3], &L.y[4], &L.y[5],
                            &L.y[6], &L.y[7], &L.tdust, &abhe, &L.abundc, &L.abundo, &L.abundsi, &L.abundD, &L.dgr,
                            &L.G0, &L.G0_LW, &L.G0_dust, &L.crir, &p1);
        if (n != 26) continue;
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

// electrons and temperature as rate_eq_ulli builds them
static double abe_of(const double* y)
{
    double cp = std::max(COOLRC.abundc - y[ICO], 0.0);
    return y[IHP] + y[IHEP] + 2.0 * y[IHEPP] + cp + y[IDP] + COOLRC.abundsi;
}
static double temp_of(const ChemTables& T, double yn, double energy, const double* y)
{
    double yntot = (1.0 + CHEM_COMP.abhe - y[IH2] + abe_of(y)) * yn;
    double t, g;
    cxx_lookup_temp_gamma(y[IH2], CHEM_COMP.abhe, energy / (yntot * CHEM_KBOLTZ), &t, &g, T);
    return t;
}
// energy at which the TNOCHEM bypass temperature (H electrons only, cvode_driver.cc) equals Tb
static double energy_bypass_T(const ChemTables& T, double yn, double Tb, const double* y)
{
    double yntot = (1.0 + CHEM_COMP.abhe + y[IHP] - y[IH2]) * yn;
    double e = 1.5 * yntot * CHEM_KBOLTZ * Tb;
    for (int k = 0; k < 60; k++) {
        double t, g;
        cxx_lookup_temp_gamma(y[IH2], CHEM_COMP.abhe, e / (yntot * CHEM_KBOLTZ), &t, &g, T);
        if (std::fabs(t - Tb) < 1e-12 * Tb) break;
        e *= Tb / t;
    }
    return e;
}

static ChemTables* g_cx = nullptr;

struct Res { int rc; long substeps, retries, floors, excess, recoveries; double e, T; double y[NSPEC]; };

static Res run_fortran(const LoggedCell& L, double dt, double yn, double e0, const double* y0)
{
    Res r; double y[NSPEC]; for (int i = 0; i < NSPEC; i++) y[i] = y0[i];
    double e = e0;
    set_cell_commons(L);
    double Tf = temp_of(*g_cx, yn, e, y);
    set_columns(yn, y[IH2], y[ICO]);
    double rpar[NRPAR]; std::memset(rpar, 0, sizeof(rpar));
    rpar[0] = yn;
    double a2 = y[IH2], ahd = y[IHD], aco = y[ICO];
    COOLRC.tdust = L.tdust;
    CALC_PHOTO_WRAPPER(&Tf, rpar, &a2, &ahd, &aco);
    double dts = dt, dl = L.dl, dv = L.divv, ce = 0.0;
    r.rc = 0; g_fatal_code = 0;
    g_fatal_armed = 1;
    if (setjmp(g_fatal_jmp) == 0) EVOLVE_ABUNDANCES(&dts, &dl, &yn, &dv, &e, y, &ce);
    else r.rc = g_fatal_code ? g_fatal_code : -1;
    g_fatal_armed = 0;
    r.substeps = r.retries = r.floors = r.excess = r.recoveries = -1;
    r.e = e; for (int i = 0; i < NSPEC; i++) r.y[i] = y[i];
    r.T = temp_of(*g_cx, yn, e, y);
    return r;
}

static Res run_cxx(const LoggedCell& L, double dt, double yn, double e0, const double* y0, double* ph1)
{
    Res r; double y[NSPEC]; for (int i = 0; i < NSPEC; i++) y[i] = y0[i];
    double e = e0;
    set_cell_commons(L);
    double Tc = temp_of(*g_cx, yn, e, y);
    set_columns(yn, y[IH2], y[ICO]);
    ChemCell c; chem_bind_cell(c, *g_cx, NET);
    c.yn = yn; c.dl = L.dl; c.divv = L.divv; c.tdust = L.tdust;
    cxx_calc_photo(Tc, y[IH2], y[IHD], y[ICO], c);
    if (ph1) *ph1 = c.phrates[0];
    const long s0 = cxx_stat_substeps, r0 = cxx_stat_retries, f0 = cxx_stat_floor_hits;
    const long x0 = chem_excess_states, v0 = cxx_stat_recoveries;
    r.rc = cxx_evolve_abundances(dt, L.dl, yn, L.divv, &e, y, c);
    r.substeps = cxx_stat_substeps - s0; r.retries = cxx_stat_retries - r0; r.floors = cxx_stat_floor_hits - f0;
    r.excess = chem_excess_states - x0; r.recoveries = cxx_stat_recoveries - v0;
    r.e = e; for (int i = 0; i < NSPEC; i++) r.y[i] = y[i];
    r.T = temp_of(*g_cx, yn, e, y);
    return r;
}

// one C++/DVODE pair against the gate; returns 1 if it fails
static int check_pair(const char* tag, const LoggedCell& L, double dt, double yn, double e0, const double* y0)
{
    double ph1 = 0.0;
    Res c = run_cxx(L, dt, yn, e0, y0, &ph1);
    Res f = run_fortran(L, dt, yn, e0, y0);
    double relT = (f.rc == 0) ? std::fabs(c.T - f.T) / f.T : 0.0;
    int bad = (c.rc != 0) || (c.excess != 0) || (c.floors != 0) || (relT > 0.01);
    std::printf("  %-10s %8ld %10.4g %9.4g %9.4g | %4d %6ld %6ld %7ld %5ld %10.4g | %4d %10.4g %8.1e%s\n",
                tag, L.id, dt, yn, temp_of(*g_cx, yn, e0, y0), c.rc, c.substeps, c.retries, c.excess, c.floors, c.T,
                f.rc, f.T, relT, bad ? "  <-- FAIL" : "");
    if (bad)
        for (const Res* r : {&c, &f}) {
            std::printf("      %s y:", r == &c ? "C++    " : "Fortran");
            for (int i = 0; i < NSPEC; i++) std::printf(" %.9e", r->y[i]);
            std::printf("\n");
        }
    std::fflush(stdout);
    return bad;
}

int main()
{
    std::printf("=== NET 17 SOLVER FAILURE (WLM run, hot ionized gas at x_H+ = 1) ===\n");
    std::vector<double> ph1_log;
    std::vector<LoggedCell> cells = load_cells("gizmo/cooling/sgchem/cxx/golden/net17_wlm_floor_cells.txt", ph1_log);
    if (cells.empty()) { std::printf("FATAL: no logged cells\n"); return 1; }
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

#if defined(CHEM_NSTEP) && CHEM_NSTEP == 1
    // ---------------- PHASE R ----------------
    std::printf("PHASE R -- forced solver failure (substep budget 1) on the %zu logged cells\n", cells.size());
    int nbadR = 0; long nrec = 0;
    for (size_t k = 0; k < cells.size(); k++) {
        const LoggedCell& L = cells[k];
        Res c = run_cxx(L, L.dt, L.yn, L.y[ITMP], L.y, nullptr);
        nrec += c.recoveries;
        int bad = (c.rc != 0) || !(c.e > 0.0) || !std::isfinite(c.e);
        for (int i = 0; i < TRAC_NUM; i++) bad |= !(c.y[i] >= 0.0 && c.y[i] <= 1.0);
        bad |= (c.y[IHEP] + c.y[IHEPP] > CHEM_COMP.abhe * (1.0 + 1e-4)) || (c.y[IDP] + c.y[IHD] > L.abundD * (1.0 + 1e-4));
        nbadR += bad;
        std::printf("  %8ld rc=%d recoveries=%ld T0=%.4g T_end=%.4g e_end/e0=%.6f%s\n", L.id, c.rc, c.recoveries,
                    temp_of(Cx, L.yn, L.y[ITMP], L.y), c.T, c.e / L.y[ITMP], bad ? "  <-- FAIL" : "");
    }
    const int ok = (nbadR == 0) && (nrec > 0);
    std::printf("  GATE recovered: rc = 0, species within their pools, finite positive energy, recoveries %ld > 0: %s\n",
                nrec, ok ? "PASS" : "FAIL");
#else
    const char* head = "  %-10s %8s %10s %9s %9s | %4s %6s %6s %7s %5s %10s | %4s %10s %8s\n";
    // ---------------- PHASE A ----------------
    std::printf("PHASE A -- %zu logged floor cells, replayed over the rest of the call\n", cells.size());
    std::printf(head, "", "ID", "dt", "yn", "T0", "rcC", "sub", "ret", "excess", "floor", "T_C", "rcF", "T_F", "relT");
    int nbadA = 0, nph1 = 0;
    for (size_t k = 0; k < cells.size(); k++) {
        const LoggedCell& L = cells[k];
        double ph1 = 0.0;
        set_cell_commons(L);
        nbadA += check_pair("logged", L, L.dt, L.yn, L.y[ITMP], L.y);
        (void)run_cxx(L, 1.0, L.yn, L.y[ITMP], L.y, &ph1);
        nph1 += (std::fabs(ph1 / ph1_log[k] - 1.0) > 1e-4);   // inputs reproduce the run's H2 photodissociation rate
    }
    std::printf("  GATE A: %d of %zu cells fail; %d cells with ph1 off the logged value\n", nbadA, cells.size(), nph1);

    // ---------------- PHASE H ----------------
    std::printf("PHASE H -- TNOCHEM hand-off state (H+ = 1, He++ = abhe, D+ = abundD) over T, nH, dt\n");
    std::printf(head, "", "ID", "dt", "yn", "T0", "rcC", "sub", "ret", "excess", "floor", "T_C", "rcF", "T_F", "relT");
    const LoggedCell& L0 = cells.back();
    int nbadH = 0;
    for (double yn : {1e-2, 0.0639667, 0.3, 1.0, 10.0})
    for (double Tb : {4.5e5, 4.8e5, 4.99e5})
    for (double dt : {1e10, 1e11, 7.7044e11, 3e12}) {
        double y[NSPEC] = {0};
        y[IHP] = 1.0; y[IHEPP] = CHEM_COMP.abhe; y[IDP] = L0.abundD;
        y[ITMP] = energy_bypass_T(Cx, yn, Tb, y);
        nbadH += check_pair("hand-off", L0, dt, yn, y[ITMP], y);
    }
    std::printf("  GATE H: %d of 60 calls fail\n", nbadH);
    const int ok = (nbadA == 0) && (nph1 == 0) && (nbadH == 0);
    std::printf("  GATE rc = 0, no validate_input excess rejection, no CVODE floor hit, T within 1%% of DVODE: %s\n",
                ok ? "PASS" : "FAIL");
#endif
    chem_tables_free(Cx);
    std::printf("=== NET 17 SOLVER FAILURE: %s ===\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
