// solver_failure.cc -- C++ chemistry solver failures, nets 5 and 17 (stella_dev golden/test_solver_failure.cc, C++ only;
// gku has no Fortran backend, so the DVODE columns and the DVODE waiver are dropped).
//
// Hot collisionally ionised gas sits at x_H+ = 1 (the TNOCHEM hand-off writes exactly 1); before D-58 CVODE trial states
// above 1 + EPS_MAX were rejected, the step collapsed to the floor and the substep loop could run out of budget.
//   PHASE A  the cells whose CVODE step hit the floor in the WLM runs (net<N>_wlm_floor_cells.txt), replayed from the
//            logged substep state over the rest of the call.
//   PHASE H  the hand-off state (hot-CIE ions) at bypass T 4.5e5..5e5 K, nH 1e-2..10, call dt 1e10..3e12 s.
//   PASS A/H: every call rc = 0, no validate_input excess rejection, no CVODE floor hit, net 17: no substep rejected for a
//            D+ excess, final T within 1% of the converged reference (net<N>_converged_reference.txt, column T_C: the
//            C++ solve at rtol 1e-8 on stella_dev, Uli 2026-11-19). Known cases (Uli 2026-11-19/20): net 5 logged cells
//            6796375 (<= 1 rejection) and 1080890 (<= 2), hand-off nH 10, bypass T 4.99e5 K, dt 7.7044e11 and 3e12 s
//            (<= 2 each); net 17 hand-off nH 10, bypass T 4.5e5 and 4.8e5 K, dt 7.7044e11 and 3e12 s (<= 1 each); all H+.
//            Phase A also checks that the inputs reproduce the run's logged H2 photodissociation rate (1e-4).
//   PHASE R  (TEST_SGCHEM_FORCED_SOLVER_FAILURE) failures forced per attempt (1 = substep budget of one, 100 = every
//            substep rejected), D-59: R1/R2 retry rescues and starts from the accepted state bit for bit; R3a/R3b TNOCHEM
//            finish; R4/R5 abort in a 100 K cell; R6 dump replay into 0xff- and 0x00-filled cells bit for bit; R7a-f
//            flagged evaluations book no stale CR heating (bars as stella_dev).
// Usage: solver_failure <chemistry tables dir> <directory of the cell and reference files> [csv]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#include "chem_types.h"
#include "sgchem_consts.h"

extern "C" void sgchem_fatal_(int* code) { std::printf("  [sgchem_fatal %d]\n", *code); std::exit(1); }

#if CHEMISTRYNETWORK != 17 && CHEMISTRYNETWORK != 5
#error "solver_failure covers networks 5 and 17"
#endif

extern thread_local long cxx_stat_substeps, cxx_stat_retries, cxx_stat_floor_hits;
extern thread_local long chem_excess_states;
extern thread_local long chem_clip_excess_fail[CHEM_NSPEC_MAX];
extern thread_local long chem_excess_species[CHEM_NSPEC_MAX];
extern thread_local long cxx_stat_failure_retries, cxx_stat_tnochem_finishes;
#ifdef TEST_SGCHEM_FORCED_SOLVER_FAILURE
extern thread_local int chem_test_force_failure[2];
extern thread_local double chem_test_retry_start_y[CHEM_NSPEC_MAX];
extern thread_local double chem_test_retry_start_t;
extern thread_local double chem_test_retry_start_tdust;
extern thread_local std::string chem_test_last_dump;
extern thread_local long chem_test_flag_rhs[5];
#endif

static const int NET = CHEMISTRYNETWORK;
static const double SGCHEM_TNOCHEM_TEST = 5e5;   // cvode_driver.cc default
static const double DL_SHIELD = 3.086e19;
static const double ABHE_F = (double)0.1f;       // the Fortran /chem_comp/ abhe the references were made with
static double g_rt_heat = 0.0;                   // photoheating for the phase-R hot cases (net 17: He+, net 5: HI)
static double g_abundc = 0.0, g_abundsi = 0.0;   // pools of the last cell set up (electrons in temp_of)

struct LoggedCell {
    long id; double dt, yn, dl, divv;
    double y[NSPEC];
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

static ChemTables* g_cx = nullptr;

// the cell inputs as stella_dev's test hands them to the chemistry (sgchem.cc ChemInitParams values; no pixel attenuation flag)
static void set_inputs(ChemCell& c, const LoggedCell& L)
{
    chem_cxx_cell_defaults(c, *g_cx, NET);
    c.abhe = ABHE_F;
    c.iphoto = 6; c.irad_heat = 0; c.isrf_option = 1;
    for (int k = 0; k < CHEM_NPIX; k++) { c.pix_att[k] = 0.0; c.fac_uv[k] = 1.0 / CHEM_NPIX; }
    c.G0 = L.G0; c.G0_LW = L.G0_LW; c.G0_dust = L.G0_dust;
    c.cosmic_ray_ion_rate = L.crir; c.cr_energy_density = 0.0;
    c.dust_to_gas_ratio = L.dgr; c.redshift = 0.0;
    c.abundD = L.abundD; c.abundc = L.abundc; c.abundo = L.abundo; c.abundsi = L.abundsi; c.abundM = L.abundsi;
    c.abundN = 7.6e-5 * L.abundo / 3.2e-4;
    c.rt_phot_HI = c.rt_phot_HeI = c.rt_phot_HeII = 0.0;
    c.rt_heat_HI = c.rt_heat_HeI = c.rt_heat_HeII = 0.0;
#if CHEMISTRYNETWORK == 17
    c.rt_heat_HeII = g_rt_heat;
#else
    c.rt_heat_HI = g_rt_heat;
#endif
    // sgchem_cell.cc order: He C Mg O Fe Si H N Ne S Ca Zn (mass fractions), metals scaled by the cell's O
    const double zsolar[12] = {0.2806, 2.07e-3, 5.91e-4, 5.49e-3, 1.10e-3, 6.83e-4,
                               0.7065, 8.36e-4, 1.41e-3, 4.09e-4, 6.44e-5, 1.10e-3};
    const double Zs = L.abundo / 3.2e-4;
    for (int k = 0; k < 12; k++) c.Zmass[k] = (k == 0 || k == 6) ? zsolar[k] : zsolar[k] * Zs;
    c.id_current = L.id;
    g_abundc = L.abundc; g_abundsi = L.abundsi;
}

static double abe_of(const double* y)
{
    double cp = std::max(g_abundc - y[ICO], 0.0);
#if CHEMISTRYNETWORK == 17
    return y[IHP] + y[IHEP] + 2.0 * y[IHEPP] + cp + y[IDP] + g_abundsi;
#else
    return y[IHP] + cp + g_abundsi;
#endif
}
static double temp_of(const ChemTables& T, double yn, double energy, const double* y)
{
    double yntot = (1.0 + ABHE_F - y[IH2] + abe_of(y)) * yn;
    double t, g;
    cxx_lookup_temp_gamma(y[IH2], ABHE_F, energy / (yntot * CHEM_KBOLTZ), &t, &g, T);
    return t;
}
#ifdef TEST_SGCHEM_FORCED_SOLVER_FAILURE
static double energy_at(const ChemTables& T, double yn, double T0, const double* y)
{
    double yntot = (1.0 + ABHE_F - y[IH2] + abe_of(y)) * yn;
    double e = 1.5 * yntot * CHEM_KBOLTZ * T0;
    for (int k = 0; k < 60; k++) {
        double t = temp_of(T, yn, e, y);
        if (std::fabs(t - T0) < 1e-12 * T0) break;
        e *= T0 / t;
    }
    return e;
}
#endif
static double bypass_T(const ChemTables& T, double yn, double e, const double* y)
{
    double yntot = (1.0 + ABHE_F + y[IHP] - y[IH2]) * yn;
    double t, g;
    cxx_lookup_temp_gamma(y[IH2], ABHE_F, e / (yntot * CHEM_KBOLTZ), &t, &g, T);
    return t;
}
static double energy_bypass_T(const ChemTables& T, double yn, double Tb, const double* y)
{
    double yntot = (1.0 + ABHE_F + y[IHP] - y[IH2]) * yn;
    double e = 1.5 * yntot * CHEM_KBOLTZ * Tb;
    for (int k = 0; k < 60; k++) {
        double t = bypass_T(T, yn, e, y);
        if (std::fabs(t - Tb) < 1e-12 * Tb) break;
        e *= Tb / t;
    }
    return e;
}
static void hot_cie(double* y, double abundD)
{
    for (int i = 0; i < TRAC_NUM; i++) y[i] = 0.0;
    y[IHP] = 1.0;
#if CHEMISTRYNETWORK == 17
    y[IHEPP] = ABHE_F; y[IDP] = abundD;
#else
    (void)abundD;
#endif
}

struct Res { int rc; long substeps, retries, floors, excess, exc_hp, fretries, tfinish, dpfail; double e, T, crh; double y[NSPEC]; };

// the C++ cell as stella_dev's test builds it: inputs, uniform columns over 12 pixels, cxx_calc_photo
static void make_cell(ChemCell& c, const LoggedCell& L, double yn, const double* y, double e)
{
    set_inputs(c, L);
    double Tc = temp_of(*g_cx, yn, e, y);
    const double NH = yn * DL_SHIELD;
    for (int k = 0; k < CHEM_NPIX; k++) {
        c.col_tot[k] = NH; c.col_h2[k] = std::max(y[IH2], 0.0) * NH; c.col_co[k] = std::max(y[ICO], 0.0) * NH;
    }
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
    const long x0 = chem_excess_states, q0 = cxx_stat_failure_retries, t0 = cxx_stat_tnochem_finishes, h0 = chem_excess_species[IHP];
#if CHEMISTRYNETWORK == 17
    const long d0 = chem_clip_excess_fail[IDP];
#endif
    r.rc = cxx_evolve_abundances(dt, c.dl, yn, c.divv, &e, y, c);
    r.substeps = cxx_stat_substeps - s0; r.retries = cxx_stat_retries - r0; r.floors = cxx_stat_floor_hits - f0;
    r.excess = chem_excess_states - x0; r.fretries = cxx_stat_failure_retries - q0; r.tfinish = cxx_stat_tnochem_finishes - t0;
    r.crh = c.cr_ion_heat_int;
    r.exc_hp = chem_excess_species[IHP] - h0;
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

#ifndef TEST_SGCHEM_FORCED_SOLVER_FAILURE
static size_t g_row = 0;
// one C++ call against the A/H gate; returns 1 if it fails
static int check_call(const char* tag, const LoggedCell& L, double dt, double yn, double e0, const double* y0)
{
    Res c = run_cxx(L, dt, yn, e0, y0, nullptr);
    const size_t k = g_row++;
    const double Tref = (k < g_refT.size()) ? g_refT[k] : NAN;
    const double relC = std::fabs(c.T - Tref) / Tref;
    long known = 0;
    const int handoff10 = (std::strcmp(tag, "hand-off") == 0 && yn == 10.0 && (dt == 7.7044e11 || dt == 3e12));
    const double Tb = bypass_T(*g_cx, yn, e0, y0);
#if CHEMISTRYNETWORK == 5
    if (std::strcmp(tag, "logged") == 0 && L.id == 6796375) known = 1;
    if (std::strcmp(tag, "logged") == 0 && L.id == 1080890) known = 2;
    if (handoff10 && L.id == 1080890 && std::fabs(Tb - 4.99e5) < 1.0) known = 2;
#else
    if (handoff10 && L.id == 3018810 && (std::fabs(Tb - 4.5e5) < 1.0 || std::fabs(Tb - 4.8e5) < 1.0)) known = 1;
#endif
    const int excess_ok = (c.excess == 0) || (c.excess <= known && c.exc_hp == c.excess);
    int bad = (c.rc != 0) || !excess_ok || (c.floors != 0) || (c.dpfail != 0) || !(relC <= 0.01);
    std::printf("  %-10s %8ld %10.4g %9.4g %9.4g | %4d %6ld %6ld %7ld %5ld %10.4g %8.1e | %10.4g%s\n",
                tag, L.id, dt, yn, temp_of(*g_cx, yn, e0, y0), c.rc, c.substeps, c.retries, c.excess, c.floors, c.T, relC,
                Tref, bad ? "  <-- FAIL" : (known && c.excess ? "  (known: recoverable H+ rejections)" : ""));
    if (known && c.excess == 0) std::printf("      known case did not occur\n");
    if (c.excess && c.exc_hp != c.excess) std::printf("      %ld of %ld rejections are not H+\n", c.excess - c.exc_hp, c.excess);
    if (c.dpfail) std::printf("      %ld substeps rejected for a D+ excess\n", c.dpfail);
    if (bad) {
        std::printf("      C++ y:");
        for (int i = 0; i < NSPEC; i++) std::printf(" %.9e", c.y[i]);
        std::printf("\n");
    }
    if (g_csv) {
        std::fprintf(g_csv, "%s,%ld,%.17g,%.17g,%d", tag, L.id, dt, yn, c.rc);
        for (int i = 0; i < NSPEC; i++) std::fprintf(g_csv, ",%.17g", c.y[i]);
        std::fprintf(g_csv, ",%.17g\n", c.crh);
    }
    std::fflush(stdout);
    return bad;
}
#endif

int main(int argc, char** argv)
{
    if (argc < 3) { std::printf("usage: %s <tables dir> <data dir> [csv]\n", argv[0]); return 2; }
    std::printf("=== SOLVER FAILURE, NETWORK %d (WLM runs; D-58, D-59, D-60), C++ ===\n", NET);
    if (argc > 3) g_csv = std::fopen(argv[3], "w");
    char cellfile[1024], reffile[1024];
    std::snprintf(cellfile, sizeof(cellfile), "%s/net%d_wlm_floor_cells.txt", argv[2], NET);
    std::snprintf(reffile, sizeof(reffile), "%s/net%d_converged_reference.txt", argv[2], NET);
    std::vector<double> ph1_log;
    std::vector<LoggedCell> cells = load_cells(cellfile, ph1_log);
    if (cells.empty()) { std::printf("FATAL: no logged cells in %s\n", cellfile); return 1; }
    std::printf("converged reference: %d calls from %s\n", load_reference(reffile), reffile);

    ChemBuildConfig cfg;
    cfg.net = NET;
    cfg.iflag_atom = 2; cfg.iflag_mn = 4; cfg.iflag_ad = 6;
    cfg.iflag_3bh2a = 5; cfg.iflag_3bh2b = 4; cfg.iflag_h3pra = 1;
    cfg.wss_cie = 1;
    cfg.deff = 1.0; cfg.phi_pah = 0.5;
    cfg.dust_to_gas_ratio = 1.0;
    cfg.cosmic_ray_ion_rate = 3e-17;
    cfg.abhe = ABHE_F;
    ChemTables Cx;
    if (chem_tables_build(Cx, cfg, argv[1])) { std::printf("FATAL: chem_tables_build\n"); return 1; }
    g_cx = &Cx;
    const LoggedCell& L0 = cells.back();
    { ChemCell tmp; set_inputs(tmp, cells[0]); }   // pools of the first cell, as stella_dev's init_commons

#ifdef TEST_SGCHEM_FORCED_SOLVER_FAILURE
    auto force = [](int a0, int a1) { chem_test_force_failure[0] = a0; chem_test_force_failure[1] = a1; };
    int nbadR = 0;
    // R1/R2: first attempt fails, the retry rescues and starts from the accepted state
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
                start_bad |= (mode == 100) && (ts != 0.0);
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

    // R3: TNOCHEM finish after heating across TNOCHEM (net 17: He in He+ with He+ photoheating; net 5: half-neutral H, HI photoheating)
    std::vector<std::string> dumps; std::vector<std::pair<int, int>> dump_force;
    std::vector<double> dump_e; std::vector<std::vector<double>> dump_y; std::vector<int> dump_rc;
    for (int variant = 0; variant < 2; variant++) {
        double y[NSPEC] = {0};
        hot_cie(y, L0.abundD);
#if CHEMISTRYNETWORK == 17
        y[IHEPP] = 0.0; y[IHEP] = ABHE_F;
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

    // R4/R5: retry fails in a 100 K cell: abort
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

    // R6: rebuild each dumped cell from its dump alone and replay it
    for (size_t k = 0; k < 2 * dumps.size(); k++) {
        const size_t kd = k % dumps.size();
        const int fill = (k < dumps.size()) ? 0xff : 0x00;
        ChemCell c;
        std::memset(static_cast<void*>(&c), fill, sizeof(c));
        ChemCellEntry e;
        const int missing = chem_cell_parse(dumps[kd].c_str(), c, e);
        c.tab = &Cx;
        double y[NSPEC];
        for (int i = 0; i < NSPEC; i++) y[i] = e.y_entry[i];
        force(dump_force[kd].first, dump_force[kd].second); chem_test_last_dump.clear();
        double en = y[ITMP];
        double yy[NSPEC]; for (int i = 0; i < NSPEC; i++) yy[i] = y[i];
        const int rc = cxx_evolve_abundances(e.time, c.dl, c.yn, c.divv, &en, yy, c);
        force(0, 0);
        auto body = [](const std::string& d) { const size_t nl = d.find('\n'); return nl == std::string::npos ? d : d.substr(nl); };
        int same = (rc == dump_rc[kd]) && !dumps[kd].empty() && (body(chem_test_last_dump) == body(dumps[kd]));
        if (rc == 0) { same &= (en == dump_e[kd]); for (int i = 0; i < TRAC_NUM; i++) same &= (yy[i] == dump_y[kd][i]); }
        int ok = (missing == 0) && same;
        std::printf("PHASE R6 -- replay of dump %zu (fill 0x%02x) from the dump alone: %d fields missing, rc %d, dump and result %s: %s\n",
                    kd, fill, missing, rc, same ? "identical" : "DIFFER", ok ? "PASS" : "FAIL");
        nbadR += !ok;
    }
    // R7a/R7b: flagged evaluations book no stale CR heating
    {
        const LoggedCell* Lm = &cells[0];
        Res best{}; best.substeps = 0;
        for (const LoggedCell& L : cells) {
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
    // R7e/R7f: the first one / two substep-start evaluations flagged, on a steady cell (constant CR heating)
    {
        double y[NSPEC] = {0};
        y[IH2] = 1e-8; y[IHP] = 1e-2;
#if CHEMISTRYNETWORK == 17
        y[IDP] = 1e-2 * L0.abundD;
#endif
        const double yn = 1.0, dt = 1e10;
        LoggedCell Lw = L0; Lw.tdust = 20.0;
        y[ITMP] = energy_at(Cx, yn, 8000.0, y);
        Res uf = run_cxx(Lw, dt, yn, y[ITMP], y, nullptr);
        for (int nflag = 1; nflag <= 2; nflag++) {
            chem_test_flag_rhs[4] = nflag;
            Res c = run_cxx(Lw, dt, yn, y[ITMP], y, nullptr);
            chem_test_flag_rhs[4] = 0;
            const double ratio = c.crh / uf.crh;
            int ok = (uf.crh > 0.0) && std::isfinite(c.crh) && (ratio > 0.98) && (ratio < 1.02) && (c.rc == 0);
            std::printf("PHASE R7%c -- first %d substep-start evaluation(s) flagged, steady cell (8000 K, nH 1, %ld substeps unflagged, "
                        "%ld flagged): integral %.6g vs unflagged %.6g (ratio %.4g): %s\n", nflag == 1 ? 'e' : 'f', nflag, uf.substeps,
                        c.substeps, c.crh, uf.crh, ratio, ok ? "PASS" : "FAIL");
            nbadR += !ok;
        }
    }
    // R7c: flagged evaluation at the failure point with an accepted substep pending (retry starts at t > 0)
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
        chem_test_retry_start_tdust = -1.0;
        Res ref = run_cxx(*Lm, Lm->dt, Lm->yn, Lm->y[ITMP], Lm->y, nullptr);
        const double td_ref = chem_test_retry_start_tdust;
        chem_test_flag_rhs[2] = 1;
        chem_test_retry_start_tdust = -1.0;
        Res c = run_cxx(*Lm, Lm->dt, Lm->yn, Lm->y[ITMP], Lm->y, nullptr);
        const double td_flag = chem_test_retry_start_tdust;
        chem_test_flag_rhs[2] = 0;
        force(0, 0);
        const double ratio = c.crh / ref.crh;
        int same_state = (c.e == ref.e);
        for (int i = 0; i < TRAC_NUM; i++) same_state &= (c.y[i] == ref.y[i]);
        int ok = (ref.fretries == 1) && (ref.crh > 0.0) && std::isfinite(c.crh) && (ratio > 0.5) && (ratio < 1.5) && (c.rc == 0)
              && same_state && (c.fretries == 1) && (td_ref > 0.0) && (td_flag == td_ref) && (td_flag != 1e3);
        std::printf("PHASE R7c -- flagged evaluation at the failure point, CR heating 1e30, tdust moved to 1e3 (cell %ld): integral "
                    "%.6g vs unflagged %.6g (ratio %.4g), retry starts at tdust %.6g (unflagged %.6g), final state %s: %s\n", Lm->id,
                    c.crh, ref.crh, ratio, td_flag, td_ref, same_state ? "identical" : "DIFFERS", ok ? "PASS" : "FAIL");
        nbadR += !ok;
    }
    // R7d: flagged first evaluation of the TNOCHEM subcycle after a failed retry (the R3a cell)
    {
        double y[NSPEC] = {0};
        hot_cie(y, L0.abundD);
#if CHEMISTRYNETWORK == 17
        y[IHEPP] = 0.0; y[IHEP] = ABHE_F;
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
    const char* head = "  %-10s %8s %10s %9s %9s | %4s %6s %6s %7s %5s %10s %8s | %10s\n";
    std::printf("PHASE A -- %zu logged floor cells, replayed over the rest of the call\n", cells.size());
    std::printf(head, "", "ID", "dt", "yn", "T0", "rcC", "sub", "ret", "excess", "floor", "T_C", "relC", "T_ref");
    int nbadA = 0, nph1 = 0;
    for (size_t k = 0; k < cells.size(); k++) {
        const LoggedCell& L = cells[k];
        double ph1 = 0.0;
        nbadA += check_call("logged", L, L.dt, L.yn, L.y[ITMP], L.y);
        (void)run_cxx(L, 1.0, L.yn, L.y[ITMP], L.y, &ph1);
        if (std::fabs(ph1 / ph1_log[k] - 1.0) > 1e-4) {
            nph1++;
            std::printf("      cell %ld: ph1 %.9e vs logged %.9e\n", L.id, ph1, ph1_log[k]);
        }
    }
    std::printf("  GATE A: %d of %zu cells fail; %d cells with ph1 off the logged value\n", nbadA, cells.size(), nph1);

    std::printf("PHASE H -- TNOCHEM hand-off state (hot-CIE ions) over T, nH, dt\n");
    std::printf(head, "", "ID", "dt", "yn", "T0", "rcC", "sub", "ret", "excess", "floor", "T_C", "relC", "T_ref");
    int nbadH = 0;
    for (double yn : {1e-2, 0.0639667, 0.3, 1.0, 10.0})
    for (double Tb : {4.5e5, 4.8e5, 4.99e5})
    for (double dt : {1e10, 1e11, 7.7044e11, 3e12}) {
        double y[NSPEC];
        hot_cie(y, L0.abundD);
        y[ITMP] = energy_bypass_T(Cx, yn, Tb, y);
        nbadH += check_call("hand-off", L0, dt, yn, y[ITMP], y);
    }
    std::printf("  GATE H: %d of 60 calls fail\n", nbadH);
    const int ok = (nbadA == 0) && (nph1 == 0) && (nbadH == 0) && (g_row == g_refT.size());
    std::printf("  GATE rc = 0, no validate_input excess rejection (known calls recorded), no CVODE floor hit, no D+ excess rejection, "
                "T within 1%% of the converged reference (%zu calls, %zu reference rows): %s\n", g_row, g_refT.size(), ok ? "PASS" : "FAIL");
#endif
    if (g_csv) std::fclose(g_csv);
    chem_tables_free(Cx);
    std::printf("=== SOLVER FAILURE NET %d: %s ===\n", NET, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
