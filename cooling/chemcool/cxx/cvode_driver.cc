// cvode_driver.cc -- port of evolve_abundances.F (driver loop, TNOCHEM bypass,
// check_abundances, init_tolerances) on SUNDIALS CVODE (BDF + dense internal
// difference-quotient Jacobian == DVODE mf=22).
//
// Port-semantics decisions (audit §7 register; full rationale in DEVIATIONS.md):
//  R1/R4 KEEP: the OUTER substep loop is kept literally (safety factors 0.1
//    species / 0.3 energy, ctrl_floor pool gating, dt = min(dt_est, remaining)).
//    One CVode instance per cell call; successful substeps CONTINUE the
//    integrator (no reinit) == DVODE istate=2, so check_abundances clips do
//    NOT feed back into the integrator trajectory (clip-no-feedback KEPT).
//    Failures restore y/tdust, halve dt, CVodeReInit == istate=1.
//  R3 REDESIGN: RHS input violations act on a LOCAL COPY of y and signal via a
//    positive (recoverable) return instead of the ipar(1)+zero-ydot convention;
//    CVODE's internal recoverable retry substitutes for some driver halvings.
//    If CVODE cannot recover it returns a negative flag and the driver's
//    restore/halve path runs.
//  A10 KEEP: atol(energy) = 0 (pure relative control).  Legal in CVODE;
//    safe because the energy component is strictly positive (floored at
//    rho*MinEgySpec before entry and physically bounded below by CMB/floor).
//  R6 KEEP: TNOCHEM hot bypass ported wholesale (molecule zero + hot-CIE ion
//    set + explicit 10%-per-step energy Euler subcycle + mid-step handback +
//    1e-10 relative completion tolerance + ipar-clear == ignore RHS retval).
//  R7 KEEP: failure taxonomy: dt < 1e-8 s -> return 100 ("ABORT(100)");
//    substep budget 9e6 -> return 1 ("ABORT(0)"); negative dt -> return 2.
//    The one-zone harness maps nonzero to the chemcool_fatal exit path.

#include "chem_types.h"
#include "chem_net20.h"
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sundials/sundials_context.h>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#define DTCOOL_SCALE_FACTOR 0.3
#define CHEM_RTOL 1e-4
#define CHEM_NSTEP 9000000
#ifndef CHEMCOOL_TNOCHEM
#define CHEMCOOL_TNOCHEM 5e5
#endif

// solver statistics, exposed to the harness for the DVODE/CVODE comparison
// thread_local for the same reason as the workspace: the counters would
// otherwise be a data race.  chemcxx_stats() therefore reports the CALLING
// thread's totals; single-threaded runs (the Gate-2/3 harnesses) are unchanged.
thread_local long cxx_stat_calls = 0, cxx_stat_substeps = 0, cxx_stat_retries = 0;
thread_local long cxx_stat_rhs_evals = 0, cxx_stat_cvode_steps = 0;
thread_local long chem_nan_states = 0;

// CVODE step floor: 10 x the double resolution of the time coordinate at the substep end. Below it
// t + h no longer advances t; CVODE's own first-step lower bound is 100 x that. A failure at the floor
// is logged with the cell context and rejected like any failed substep (halve and retry, fatal 100 below 1e-8 s).
static const double CHEM_HMIN_EPS = 10.0 * DBL_EPSILON;

// full context of the cell handed to the solver, for the fatal paths
static void chem_print_cell_context(const char* why, double time, double yn, double dl, double divv,
                                    const double* y, int nspec, const ChemCell& c)
{
    std::fprintf(stderr, "cxx_evolve: %s (ID %ld): dt=%g yn=%g dl=%g divv=%g net=%d\n",
                 why, c.id_current, time, yn, dl, divv, c.net);
    std::fprintf(stderr, "  state (species..., energy):");
    for (int i = 0; i < nspec; i++) std::fprintf(stderr, " %.9e", y[i]);
    double temp = -1, gamma = 0, ntot = (1.0 + c.abhe - y[CIH2]) * yn;
    int finite = (yn > 0 && std::isfinite(yn) && std::isfinite(y[nspec - 1]) && std::isfinite(y[CIH2]));
    if (finite) cxx_lookup_temp_gamma(y[CIH2], c.abhe, y[nspec - 1] / (ntot * CHEM_KBOLTZ), &temp, &gamma, *c.tab);
    std::fprintf(stderr, "\n  T=%g (from energy, H2 and abhe; -1 = not computable) tdust=%g\n", temp, c.tdust);
    std::fprintf(stderr, "  pools abhe=%g abundc=%g abundo=%g abundsi=%g abundD=%g dgr=%g\n",
                 c.abhe, c.abundc, c.abundo, c.abundsi, c.abundD, c.dust_to_gas_ratio);
    std::fprintf(stderr, "  rates G0=%g G0_LW=%g G0_dust=%g crir=%g rt_phot=%g %g %g rt_heat=%g %g %g ph1=%g ph38=%g ph42=%g\n",
                 c.G0, c.G0_LW, c.G0_dust, c.cosmic_ray_ion_rate, c.rt_phot_HI, c.rt_phot_HeI, c.rt_phot_HeII,
                 c.rt_heat_HI, c.rt_heat_HeI, c.rt_heat_HeII, c.phrates[0], c.phrates[37], c.phrates[41]);
    std::fflush(stderr);
}

// audit R10 switch: 1 = keep the Fortran "last RHS evaluation wins" semantics
static int chem_legacy_lambda()
{
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("CHEMCXX_LEGACY_LAMBDA");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached;
}

static inline int rhs_dispatch(double t, const double* y, double* ydot, ChemCell& c)
{
    if (c.net == 3)  return chem_rhs_net3(t, y, ydot, c);
    if (c.net == 15) return chem_rhs_net15(t, y, ydot, c);
    if (c.net == 16) return chem_rhs_net16(t, y, ydot, c);
    if (c.net == 20) return chem_rhs_net20(t, y, ydot, c);
    return (c.net == 5) ? chem_rhs_net5(t, y, ydot, c)
                        : chem_rhs_net17(t, y, ydot, c);
}

// CVODE RHS trampoline
static int cvode_rhs(sunrealtype t, N_Vector yy, N_Vector yp, void* user_data)
{
    ChemCell* c = (ChemCell*)user_data;
    const double* y = N_VGetArrayPointer(yy);
    double* ydot = N_VGetArrayPointer(yp);
    cxx_stat_rhs_evals++;
    return rhs_dispatch(t, y, ydot, *c);   // 0 ok, +1 recoverable, -1 fatal (c->fatal_code)
}

// DVODE is silent about recoverable trouble; keep CVODE equally quiet so the
// benchmark logs stay readable.  Set CHEMCXX_VERBOSE=1 to restore the chatter.
static void cvode_err_handler(int error_code, const char* module,
                              const char* function, char* msg, void* /*eh_data*/)
{
    static int verbose = -1;
    if (verbose < 0) { const char* e = std::getenv("CHEMCXX_VERBOSE"); verbose = (e && *e != '0'); }
    if (verbose && error_code < 0)
        std::fprintf(stderr, "[CVODE] %s::%s: %s\n", module, function, msg);
}

// init_tolerances port (evolve_abundances.F:757-869, nets 3/5/15/16/17 branches)
static void chem_atol(int net, double* atol)
{
    if (net == 3) {
        atol[C3_IH2] = 1e-7;    atol[C3_IHP] = 1e-12;   atol[C3_IDP] = 1e-14;
        atol[C3_IHD] = 1e-12;   atol[C3_IHEP] = 1e-14;  atol[C3_IHEPP] = 1e-14;
        atol[C3_IC] = 1e-16;    atol[C3_ISI] = 1e-16;   atol[C3_IO] = 1e-16;
        atol[C3_ISIPP] = 1e-14; atol[C3_ICO] = 1e-14;   atol[C3_IC2] = 1e-14;
        atol[C3_IOH] = 1e-14;   atol[C3_IH2O] = 1e-14;  atol[C3_IO2] = 1e-14;
        atol[C3_IHCOP] = 1e-18; atol[C3_ICH] = 1e-18;   atol[C3_ICH2] = 1e-18;
        atol[C3_ICH3P] = 1e-18;
        atol[19] = 0.0;
    } else if (chem_net_nl99(net)) {
        atol[C15_IH2] = 1e-7;   atol[C15_IHP] = 1e-12;  atol[C15_IC] = 1e-16;
        atol[C15_ICHX] = 1e-14; atol[C15_IOHX] = 1e-14; atol[C15_ICO] = 1e-14;
        atol[C15_IHCOP] = 1e-18; atol[C15_IHEP] = 1e-14; atol[C15_IMP] = 1e-14;
        atol[9] = 0.0;
    } else if (net == 5) {
        atol[CIH2] = 1e-7;  atol[CIHP] = 1e-12; atol[CICO] = 1e-14;
        atol[3] = 0.0;                       // ATOL_TMP = 0d0
    } else if (net == 20) {
        atol[C20_IH2] = 1e-10; atol[C20_IHP] = 1e-12;
        atol[2] = 0.0;
    } else {
        atol[CIH2] = 1e-7;  atol[CIHP] = 1e-12; atol[CICO] = 1e-14;
        atol[CIHEP] = 1e-14; atol[CIHEPP] = 1e-14;
        atol[CIDP] = 1e-14;  atol[CIHD] = 1e-12;
        atol[7] = 0.0;
    }
}

// clamp_pool_group port: scale species k[0..n-1] (w[j] atoms each) down
// proportionally so that together they hold at most pool of the element
static void chem_clamp_pool_group(double* y, const int* k, const double* w, int n, double pool)
{
    double s = 0.0;
    for (int j = 0; j < n; j++) s += w[j] * y[k[j]];
    if (s > pool && s > 0.0) {
        double f = ((pool > 0.0) ? pool : 0.0) / s;
        for (int j = 0; j < n; j++) y[k[j]] = f * y[k[j]];
    }
}

// check_abundances port (evolve_abundances.F:499-741, nets 3/5/15/16/17 branches).
// Returns 0 ok (y possibly clipped), 1 reject.
static int chem_check_abundances(double* y, ChemCell& c, const double* atol)
{
    const int nspec = c.nspec;
    const int itmp = nspec - 1;
    double max_abundance[CHEM_NSPEC_MAX];
    if (c.net == 3) {
        const double co_max = (c.abundo < c.abundc) ? c.abundo : c.abundc;
        max_abundance[C3_IH2]   = 0.5;
        max_abundance[C3_IHP]   = 1.0;
        max_abundance[C3_IDP]   = c.abundD;
        max_abundance[C3_IHD]   = c.abundD;
        max_abundance[C3_IHEP]  = c.abhe;
        max_abundance[C3_IHEPP] = c.abhe;
        max_abundance[C3_IO]    = c.abundo;
        max_abundance[C3_IC]    = c.abundc;
        max_abundance[C3_ISI]   = c.abundsi;
        max_abundance[C3_ISIPP] = c.abundsi;
        max_abundance[C3_ICO]   = co_max;
        max_abundance[C3_IC2]   = 0.5 * c.abundc;
        max_abundance[C3_IOH]   = c.abundo;
        max_abundance[C3_IH2O]  = c.abundo;
        max_abundance[C3_IO2]   = 0.5 * c.abundo;
        max_abundance[C3_IHCOP] = co_max;
        max_abundance[C3_ICH]   = c.abundc;
        max_abundance[C3_ICH2]  = c.abundc;
        max_abundance[C3_ICH3P] = c.abundc;
    } else if (chem_net_nl99(c.net)) {
        const double co_max = (c.abundo < c.abundc) ? c.abundo : c.abundc;
        max_abundance[C15_IH2]   = 0.5;
        max_abundance[C15_IHP]   = 1.0;
        max_abundance[C15_IC]    = c.abundc;
        max_abundance[C15_ICHX]  = c.abundc;
        max_abundance[C15_IOHX]  = c.abundo;
        max_abundance[C15_ICO]   = co_max;
        max_abundance[C15_IHCOP] = co_max;
        max_abundance[C15_IHEP]  = c.abhe;
        max_abundance[C15_IMP]   = c.abundM;
    } else if (c.net == 20) {
        max_abundance[C20_IH2] = 0.5;
        max_abundance[C20_IHP] = 1.0;
    } else {
        max_abundance[CIH2] = 0.5;
        max_abundance[CIHP] = 1.0;
        max_abundance[CICO] = (c.abundc < c.abundo) ? c.abundc : c.abundo;
    }
    if (c.net == 17) {
        max_abundance[CIHEP] = c.abhe;
        max_abundance[CIHEPP] = c.abhe;
        max_abundance[CIDP] = c.abundD;
        max_abundance[CIHD] = c.abundD;
    }
    max_abundance[itmp] = 0.0;   // unused (J != itmp below)

    for (int j = 0; j < nspec; j++)
        if (y[j] != y[j]) return 1;                         // NaN

    for (int j = 0; j < nspec; j++) {
        if (y[j] < 0.0) {
            if (y[j] < -atol[j]) return 1;                  // large negative
            y[j] = 0.0;                                     // small: clip
        }
    }
    for (int j = 0; j < nspec; j++) {
        if (j != itmp && y[j] > max_abundance[j]) {
            double diff = y[j] - max_abundance[j];
            if (diff < CHEM_RTOL * y[j]) y[j] = max_abundance[j];
            else return 1;                                  // large excess
        }
    }
    // the caps bound each species singly, not their sum: project over-subscribed
    // element groups (H last, as the metal groups only lower H-bearing species)
    if (c.net == 3) {
        { const int k[] = {C3_IC, C3_ICO, C3_IC2, C3_IHCOP, C3_ICH, C3_ICH2, C3_ICH3P};
          const double w[] = {1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0};
          chem_clamp_pool_group(y, k, w, 7, c.abundc); }
        { const int k[] = {C3_IO, C3_ICO, C3_IOH, C3_IH2O, C3_IO2, C3_IHCOP};
          const double w[] = {1.0, 1.0, 1.0, 1.0, 2.0, 1.0};
          chem_clamp_pool_group(y, k, w, 6, c.abundo); }
        { const int k[] = {C3_ISI, C3_ISIPP}; const double w[] = {1.0, 1.0};
          chem_clamp_pool_group(y, k, w, 2, c.abundsi); }
        { const int k[] = {C3_IHEP, C3_IHEPP}; const double w[] = {1.0, 1.0};
          chem_clamp_pool_group(y, k, w, 2, c.abhe); }
        { const int k[] = {C3_IDP, C3_IHD}; const double w[] = {1.0, 1.0};
          chem_clamp_pool_group(y, k, w, 2, c.abundD); }
        { const int k[] = {C3_IH2, C3_IHP, C3_IHD, C3_IOH, C3_IH2O, C3_IHCOP, C3_ICH, C3_ICH2, C3_ICH3P};
          const double w[] = {2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0};
          chem_clamp_pool_group(y, k, w, 9, 1.0); }
    } else if (chem_net_nl99(c.net)) {
        { const int k[] = {C15_IC, C15_ICHX, C15_ICO, C15_IHCOP}; const double w[] = {1.0, 1.0, 1.0, 1.0};
          chem_clamp_pool_group(y, k, w, 4, c.abundc); }
        { const int k[] = {C15_IOHX, C15_ICO, C15_IHCOP}; const double w[] = {1.0, 1.0, 1.0};
          chem_clamp_pool_group(y, k, w, 3, c.abundo); }
        { const int k[] = {C15_IH2, C15_IHP, C15_ICHX, C15_IOHX, C15_IHCOP}; const double w[] = {2.0, 1.0, 1.0, 1.0, 1.0};
          chem_clamp_pool_group(y, k, w, 5, 1.0); }
    } else if (c.net == 5 || c.net == 20) {
        { const int k[] = {CIH2, CIHP}; const double w[] = {2.0, 1.0};
          chem_clamp_pool_group(y, k, w, 2, 1.0); }
    } else {
        { const int k[] = {CIHEP, CIHEPP}; const double w[] = {1.0, 1.0};
          chem_clamp_pool_group(y, k, w, 2, c.abhe); }
        { const int k[] = {CIDP, CIHD}; const double w[] = {1.0, 1.0};
          chem_clamp_pool_group(y, k, w, 2, c.abundD); }
        { const int k[] = {CIH2, CIHP, CIHD}; const double w[] = {2.0, 1.0, 1.0};
          chem_clamp_pool_group(y, k, w, 3, 1.0); }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Persistent CVODE workspace.
// DVODE reuses a static rwork/iwork across calls and re-initializes with
// istate=1; allocating a fresh SUNContext + CVode memory per cell call would
// charge the port a malloc/free cost DVODE never pays and would make the
// wall-time comparison meaningless.  CVodeReInit on a cached instance is the
// exact istate=1 equivalent.  NOT thread-safe (neither is the Fortran side --
// audit R2); one workspace per nspec.
// ---------------------------------------------------------------------------
struct CvWork {
    SUNContext ctx = nullptr;
    void* mem = nullptr;
    N_Vector yv = nullptr, atolv = nullptr;
    SUNMatrix A = nullptr;
    SUNLinearSolver LS = nullptr;
    int nspec = 0;
};
// Gate 3 / audit R2: ONE workspace cache PER THREAD.  DVODE reuses a static
// rwork/iwork and re-initialises with istate=1; caching the CVODE equivalent
// keeps the wall-time comparison fair (DEVIATIONS D-7), and making the cache
// thread_local is what makes the C++ path OpenMP-safe -- a shared CVode memory
// object cannot be entered concurrently.  Cost: one workspace per (thread,net).
static thread_local CvWork g_work[5];   // [0] nspec 4 (net 5), [1] nspec 8 (net 17), [2] nspec 10 (nets 15/16), [3] nspec 20 (net 3), [4] nspec 3 (net 20)

static CvWork* cv_get(int nspec, const double* atol, double t0, const double* y)
{
    CvWork* w = &g_work[(nspec == 4) ? 0 : (nspec == 10) ? 2 : (nspec == 20) ? 3 : (nspec == 3) ? 4 : 1];
    if (w->mem == nullptr) {
        if (SUNContext_Create(NULL, &w->ctx)) return nullptr;
        w->nspec = nspec;
        w->yv = N_VNew_Serial(nspec, w->ctx);
        w->atolv = N_VNew_Serial(nspec, w->ctx);
        for (int i = 0; i < nspec; i++) {
            NV_Ith_S(w->yv, i) = y[i];
            NV_Ith_S(w->atolv, i) = atol[i];
        }
        w->mem = CVodeCreate(CV_BDF, w->ctx);
        if (!w->mem) return nullptr;
        CVodeSetErrHandlerFn(w->mem, cvode_err_handler, NULL);
        if (CVodeInit(w->mem, cvode_rhs, t0, w->yv) < 0) return nullptr;
        w->A = SUNDenseMatrix(nspec, nspec, w->ctx);
        w->LS = SUNLinSol_Dense(w->yv, w->A, w->ctx);
        CVodeSetLinearSolver(w->mem, w->LS, w->A);   // internal DQ dense Jac (mf=22)
    } else {
        for (int i = 0; i < nspec; i++) {
            NV_Ith_S(w->yv, i) = y[i];
            NV_Ith_S(w->atolv, i) = atol[i];
        }
        if (CVodeReInit(w->mem, t0, w->yv) < 0) return nullptr;
    }
    CVodeSVtolerances(w->mem, CHEM_RTOL, w->atolv);
    CVodeSetMaxNumSteps(w->mem, 200000);             // iwork(6) = 200000
    return w;
}

// evolve_abundances port.
// non_eq_abundances has c.trac_num entries; energy in/out [erg cm^-3].
// Returns 0 ok; 100 = dt collapse (ABORT(100)); 1 = nstep exceeded (ABORT(0));
// 2 = negative/zero dt (ABORT(1)/(2)); 3 = CVODE setup failure.
// 101 = non-finite input state.
// Net 3: an RHS ABORT (comp_equilib 9751/9752, index 1) returns that code.
int cxx_evolve_abundances(double time, double dl, double yn, double divv,
                          double* energy, double* non_eq_abundances, ChemCell& c)
{
    const int nspec = c.nspec;
    const int itmp = nspec - 1;
    const int ih2 = CIH2, ihp = CIHP;
    const int ico = chem_net_nl99(c.net) ? C15_ICO : (c.net == 3) ? C3_ICO : CICO;   // unused for net 20 (no CO in the ODE)

    if (time < 0.0) return 0;                                // .F:72-74

    cxx_stat_calls++;
    c.yn = yn; c.dl = dl; c.divv = divv;
    c.fatal_code = 0;

    double y[CHEM_NSPEC_MAX], y_old[CHEM_NSPEC_MAX], ydot[CHEM_NSPEC_MAX];
    for (int i = 0; i < c.trac_num; i++) y[i] = non_eq_abundances[i];
    y[itmp] = *energy;
    for (int i = 0; i < nspec; i++) { y_old[i] = y[i]; ydot[i] = 0.0; }

    // a non-finite state or density handed in by the caller is a bug upstream: fail now, with context
    {
        int bad = !std::isfinite(yn) || !std::isfinite(time);
        for (int i = 0; i < nspec; i++) bad |= !std::isfinite(y[i]);
        if (bad) {
            chem_print_cell_context("FATAL non-finite input state (code 101)", time, yn, dl, divv, y, nspec, c);
            return 101;
        }
    }
    const long nan_states0 = chem_nan_states;   // NaN trial states rejected during this solve
    double nan_h = 0.0, nan_t = 0.0, nan_dt = 0.0;
    long floor_hits = 0;

    double atol[CHEM_NSPEC_MAX];
    chem_atol(c.net, atol);

    // safety factors + ctrl_floor (.F:35-56, 166-195)
    double safety[CHEM_NSPEC_MAX];
    int used[CHEM_NSPEC_MAX];
    for (int i = 0; i < nspec; i++) { safety[i] = 1e-1; used[i] = 1; }
    safety[itmp] = DTCOOL_SCALE_FACTOR;
    if (chem_net_nl99(c.net)) {   // Glover's control: H2, C+, CO and energy only
        static const int used15[9] = {1, 0, 1, 0, 0, 1, 0, 0, 0};
        for (int i = 0; i < 9; i++) { used[i] = used15[i]; safety[i] = used15[i] ? 1e-1 : 0.0; }
    }
    if (c.net == 3) {             // Glover's control: H2, H+ and energy only
        for (int i = 0; i < 19; i++) { used[i] = (i <= 1); safety[i] = (i <= 1) ? 1e-1 : 0.0; }
    }
    const double eps_ctrl = 1e-6;
    double ctrl_floor[CHEM_NSPEC_MAX];
    for (int j = 0; j < nspec; j++) ctrl_floor[j] = atol[j];
    if (c.net == 3) {
        double co_pool = (c.abundc < c.abundo) ? c.abundc : c.abundo;
        for (int J = 0; J < nspec; J++) {
            if (J == C3_IC || J == C3_IC2 || J == C3_ICH || J == C3_ICH2 || J == C3_ICH3P)
                ctrl_floor[J] = std::fmax(atol[J], eps_ctrl * c.abundc);
            if (J == C3_IO || J == C3_IOH || J == C3_IH2O || J == C3_IO2)
                ctrl_floor[J] = std::fmax(atol[J], eps_ctrl * c.abundo);
            if (J == C3_ICO || J == C3_IHCOP)
                ctrl_floor[J] = std::fmax(atol[J], eps_ctrl * co_pool);
            if (J == C3_ISI || J == C3_ISIPP)
                ctrl_floor[J] = std::fmax(atol[J], eps_ctrl * c.abundsi);
            if (J == C3_IDP || J == C3_IHD)
                ctrl_floor[J] = std::fmax(atol[J], eps_ctrl * c.abundD);
            if (J == C3_IHEP || J == C3_IHEPP)
                ctrl_floor[J] = std::fmax(atol[J], eps_ctrl * c.abhe);
        }
    } else if (chem_net_nl99(c.net)) {
        double co_pool = (c.abundc < c.abundo) ? c.abundc : c.abundo;
        ctrl_floor[C15_IC]    = std::fmax(atol[C15_IC],    eps_ctrl * c.abundc);
        ctrl_floor[C15_ICHX]  = std::fmax(atol[C15_ICHX],  eps_ctrl * c.abundc);
        ctrl_floor[C15_IOHX]  = std::fmax(atol[C15_IOHX],  eps_ctrl * c.abundo);
        ctrl_floor[C15_ICO]   = std::fmax(atol[C15_ICO],   eps_ctrl * co_pool);
        ctrl_floor[C15_IHCOP] = std::fmax(atol[C15_IHCOP], eps_ctrl * co_pool);
        ctrl_floor[C15_IHEP]  = std::fmax(atol[C15_IHEP],  eps_ctrl * c.abhe);
        ctrl_floor[C15_IMP]   = std::fmax(atol[C15_IMP],   eps_ctrl * c.abundM);
    } else if (c.net != 20) {
        double co_pool = (c.abundc < c.abundo) ? c.abundc : c.abundo;
        double f = eps_ctrl * co_pool;
        if (f > ctrl_floor[ico]) ctrl_floor[ico] = f;
        if (c.net == 17) {
            f = eps_ctrl * c.abundD;
            if (f > ctrl_floor[CIDP]) ctrl_floor[CIDP] = f;
            if (f > ctrl_floor[CIHD]) ctrl_floor[CIHD] = f;
            f = eps_ctrl * c.abhe;
            if (f > ctrl_floor[CIHEP])  ctrl_floor[CIHEP]  = f;
            if (f > ctrl_floor[CIHEPP]) ctrl_floor[CIHEPP] = f;
        }
    }

    double t_start = 0.0, t_end = time;

    // ---- TNOCHEM hot bypass (.F:200-260) ----
    {
        auto bypass_temp = [&]() {
            if (c.net == 20) {
                N20State s20;
                return n20_solve_state(y[ih2], y[ihp], y[itmp], c, s20) ? 0.0 : s20.T;
            }
            double yntot_byp = (1.0 + c.abhe + y[ihp] - y[ih2]) * yn;
            double ekn_byp = y[itmp] / (CHEM_KBOLTZ * yntot_byp);
            double temp_byp, gam_byp;
            cxx_lookup_temp_gamma(y[ih2], c.abhe, ekn_byp, &temp_byp, &gam_byp, *c.tab);
            return temp_byp;
        };
        double temp_byp = bypass_temp();
        if (temp_byp > CHEMCOOL_TNOCHEM) {
            y[ih2] = 0.0;
            if (c.net != 20) y[ico] = 0.0;
            if (c.net == 17) y[CIHD] = 0.0;
            // HOTIONFIX hot-CIE ion set (.F:223-228)
            y[ihp] = 1.0;
            if (c.net == 17) {
                y[CIHEP] = 0.0;
                y[CIHEPP] = c.abhe;
                y[CIDP] = c.abundD;
            }
            if (chem_net_nl99(c.net)) {   // highest tracked stages (.F:232-240)
                y[C15_ICHX]  = 0.0;
                y[C15_IOHX]  = 0.0;
                y[C15_IHCOP] = 0.0;
                y[C15_IC]    = c.abundc;
                y[C15_IHEP]  = c.abhe;
                y[C15_IMP]   = c.abundM;
            }
            if (c.net == 3) {   // .F:280-297
                y[C3_IHD]   = 0.0;
                y[C3_IC2]   = 0.0;
                y[C3_IOH]   = 0.0;
                y[C3_IH2O]  = 0.0;
                y[C3_IO2]   = 0.0;
                y[C3_IHCOP] = 0.0;
                y[C3_ICH]   = 0.0;
                y[C3_ICH2]  = 0.0;
                y[C3_ICH3P] = 0.0;
                y[C3_IC]    = c.abundc;
                y[C3_IO]    = c.abundo;
                y[C3_ISI]   = 0.0;
                y[C3_ISIPP] = c.abundsi;
                y[C3_IHEP]  = 0.0;
                y[C3_IHEPP] = c.abhe;
                y[C3_IDP]   = c.abundD;
            }
            int done = 0;
            for (long nbyp = 1; nbyp <= 100000; nbyp++) {
                // ipar-clear semantics: RHS retval ignored; a flagged eval
                // left ydot all-zero, exactly like the Fortran zero-ydot path
                // (net 15 builds the full ydot, as rate_eq_nl99 does).
                rhs_dispatch(t_start, y, ydot, c);
                if (c.fatal_code) return c.fatal_code;      // Fortran ABORT inside rate_eq
                double dt_byp;
                if (std::fabs(ydot[itmp]) > 0.0)
                    dt_byp = std::fmin(t_end - t_start,
                                       1e-1 * std::fabs(y[itmp] / ydot[itmp]));
                else
                    dt_byp = t_end - t_start;
                y[itmp] += dt_byp * ydot[itmp];
                t_start += dt_byp;
                if (t_start >= t_end - 1e-10 * t_end) { done = 1; break; }  // WEDGEFIX 3
                temp_byp = bypass_temp();
                if (temp_byp <= CHEMCOOL_TNOCHEM) break;    // hand off to stiff path
            }
            // budget exhausted while hot: accept state (.F:257-259)
            if (done || t_start >= t_end - 1e-10 * t_end) {
                for (int i = 0; i < c.trac_num; i++) non_eq_abundances[i] = y[i];
                *energy = y[itmp];
                return 0;
            }
            // else: fall through to the stiff path for the remaining interval
        }
    }

    // ---- CVODE setup (cached workspace; ReInit == istate=1) ----
    CvWork* w = cv_get(nspec, atol, t_start, y);
    if (!w) return 3;
    CVodeSetUserData(w->mem, &c);

    int new_step = 1;
    int need_reinit = 0;         // istate=1 equivalent pending
    double dt = 0.0, dt_est = 0.0;
    int retcode = 1;             // if the loop exhausts: "nstep exceeded"
    double tdust_old = c.tdust;

    for (long I = 1; I <= CHEM_NSTEP; I++) {
        if (new_step == 1) {
            for (int j = 0; j < nspec; j++) y_old[j] = y[j];
            tdust_old = c.tdust;

            int rerr = rhs_dispatch(t_start, y, ydot, c);
            if (c.fatal_code) { retcode = c.fatal_code; goto out; }
            dt_est = 1e18;
            for (int j = 0; j < nspec; j++) {
                if (used[j] && ydot[j] != 0.0 && safety[j] != 0.0 && y[j] > ctrl_floor[j]) {
                    double e = std::fabs(safety[j] * y[j] / ydot[j]);
                    if (e < dt_est) dt_est = e;
                }
            }
            dt = std::fmin(dt_est, t_end - t_start);
            if (rerr) {
                // Driver-level RHS eval flagged.  In the Fortran, ipar(1)
                // persists through the dvode call and the substep is rejected
                // afterwards; here we take the failure path directly.
                c.tdust = tdust_old;
                dt = dt / 2.0;
                if (dt < 1e-8) { retcode = 100; goto out; }
                cxx_stat_retries++;
                new_step = 0; need_reinit = 1;
                continue;
            }
        }

        if (dt < 0.0) { retcode = 2; goto out; }
        if (dt == 0.0) {
            if (t_end == t_start) { retcode = 0; goto out_nocopy; }
            retcode = 2; goto out;
        }

        {
            if (need_reinit) {
                for (int i = 0; i < nspec; i++) NV_Ith_S(w->yv, i) = y[i];
                CVodeReInit(w->mem, t_start, w->yv);         // istate = 1
                need_reinit = 0;
            }
            double tret = t_start;
            const double hmin = CHEM_HMIN_EPS * (t_start + dt);
            CVodeSetMinStep(w->mem, hmin);
            const long nan_before = chem_nan_states;
            int flag = CVode(w->mem, t_start + dt, w->yv, &tret, CV_NORMAL);
            cxx_stat_substeps++;
            if (c.fatal_code) { retcode = c.fatal_code; goto out; }
            if (chem_nan_states > nan_before && nan_states0 == nan_before) {
                CVodeGetCurrentStep(w->mem, &nan_h);
                nan_t = t_start; nan_dt = dt;
            }
            if (flag < 0) {
                double hcur = 0.0;
                CVodeGetCurrentStep(w->mem, &hcur);
                if (std::fabs(hcur) <= hmin * (1.0 + 1e-6) && floor_hits++ == 0) {
                    std::fprintf(stderr, "cxx_evolve: CVODE step reached the floor h=%g (time resolution at t=%g, substep dt=%g, flag %d) of %g; substep rejected\n",
                                 hcur, t_start + dt, dt, flag, t_end);
                    chem_print_cell_context("CVODE step-size floor", time, yn, dl, divv, y_old, nspec, c);
                }
            }

            int fail = (flag < 0);
            if (!fail) {
                for (int i = 0; i < nspec; i++) y[i] = NV_Ith_S(w->yv, i);
                fail = chem_check_abundances(y, c, atol);
            }

            if (fail) {
                for (int j = 0; j < nspec; j++) y[j] = y_old[j];
                c.tdust = tdust_old;
                dt = dt / 2.0;
                if (dt < 1e-8) {
                    std::fprintf(stderr,
                        "cxx_evolve: excessively small timestep! dt=%g dt_est=%g "
                        "t=[%g,%g] ID=%ld yn=%g\n",
                        dt, dt_est, t_start, t_end, c.id_current, yn);
                    retcode = 100; goto out;
                }
                cxx_stat_retries++;
                new_step = 0;
                need_reinit = 1;                          // istate = 1
            } else {
                t_start = t_start + dt;
                new_step = 1;                             // istate stays 2:
                // NB check_abundances clips modified OUR y (final answer /
                // future y_old) but were NOT pushed into CVODE -- continuation
                // resumes from CVODE's internal (unclipped) trajectory,
                // exactly the DVODE clip-no-feedback semantics.
            }
        }

        if (t_start >= t_end - 1e-10 * t_end) { retcode = 0; goto out; }
    }
    std::fprintf(stderr, "cxx_evolve: nstep exceeded (ID %ld)\n", c.id_current);
    retcode = 1;

out:
    if (retcode == 0 || retcode == 1) {
        for (int i = 0; i < c.trac_num; i++) non_eq_abundances[i] = y[i];
        *energy = y[itmp];
        // ---- audit R10: state-consistent lambda diagnostics ----
        // The Fortran leaves lambda/lambda_chem at whatever the LAST RHS
        // evaluation wrote.  With a multistep integrator that evaluation is at
        // an internal stage or, worse, at a REJECTED trial point, so the
        // published cooling-rate breakdown does not correspond to the state
        // that was actually returned -- and it is the breakdown that lands in
        // snapshots and in every "why did this cell cool" post-mortem.
        // One extra RHS evaluation at the ACCEPTED (t_end, y) fixes that.
        // Cost: one RHS per cell call, ~0.4% of the ~240 evaluations a typical
        // call already makes.  Set CHEMCXX_LEGACY_LAMBDA=1 to restore the
        // Fortran semantics for a bit-comparable A/B (DEVIATIONS D-19).
        if (!chem_legacy_lambda()) {
            double ydot_final[CHEM_NSPEC_MAX];
            (void)rhs_dispatch(t_start, y, ydot_final, c);
        }
    }
out_nocopy:
    if (floor_hits > 0)
        std::fprintf(stderr, "cxx_evolve: CVODE step floor reached in %ld substeps (ID %ld); solve rc=%d\n",
                     floor_hits, c.id_current, retcode);
    if (chem_nan_states > nan_states0) {
        std::fprintf(stderr, "cxx_evolve: %ld NaN trial states rejected by CVODE (ID %ld; first in substep t=%g dt=%g, CVODE h=%g); solve rc=%d\n",
                     chem_nan_states - nan_states0, c.id_current, nan_t, nan_dt, nan_h, retcode);
    }
    {
        long ns = 0;
        if (w && w->mem && CVodeGetNumSteps(w->mem, &ns) == CV_SUCCESS)
            cxx_stat_cvode_steps += ns;
    }
    return retcode;
}
