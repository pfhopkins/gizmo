/* C++ implementation: Ulrich Steinwandel (uli@mpa-garching.mpg.de).
   Based on the SGChem Fortran networks by Simon Glover (GNU GPL, as distributed in Arepo). */
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
//    100 and 1 retry once (CVODE, budget CHEM_NSTEP_RETRY) from the last accepted state with rtol/atol and
//    substep safety x0.1; if the retry fails too, hot gas (bypass T > TNOCHEM) finishes via the TNOCHEM
//    path, otherwise the code is returned (abort) after the full cell dump (D-59).

#include "chem_types.h"
#include "chem_net20.h"
#include "chem_net18.h"
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sundials/sundials_context.h>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>

#define DTCOOL_SCALE_FACTOR 0.3
#ifndef CHEM_RTOL
#define CHEM_RTOL 1e-4
#endif
#ifndef CHEM_NSTEP
#define CHEM_NSTEP 9000000
#endif
#ifndef CHEM_NSTEP_RETRY
#define CHEM_NSTEP_RETRY 1000000   /* substep budget of the D-59 retry */
#endif
#ifndef SGCHEM_TNOCHEM
#define SGCHEM_TNOCHEM 5e5
#endif

// solver statistics, exposed to the harness for the DVODE/CVODE comparison
// thread_local for the same reason as the workspace: the counters would
// otherwise be a data race.  chemcxx_stats() therefore reports the CALLING
// thread's totals; single-threaded runs (the Gate-2/3 harnesses) are unchanged.
thread_local long cxx_stat_calls = 0, cxx_stat_substeps = 0, cxx_stat_retries = 0;
thread_local long cxx_stat_rhs_evals = 0, cxx_stat_cvode_steps = 0;
thread_local long chem_nan_states = 0;
thread_local long chem_excess_states = 0;
thread_local long cxx_stat_failure_retries = 0;    // solver failures (1, 100) retried with tighter tolerances (D-59)
thread_local long cxx_stat_tnochem_finishes = 0;   // failed retries above TNOCHEM finished via the TNOCHEM path (D-59)
thread_local long cxx_stat_floor_hits = 0;         // CVODE substeps rejected at the step-size floor
thread_local long chem_clip_excess_fail[CHEM_NSPEC_MAX];   // substeps check_abundances rejected for a large excess, per species
#ifdef TEST_SGCHEM_FORCED_SOLVER_FAILURE
thread_local int chem_test_force_failure[2] = {0, 0}; // per attempt: 1 = substep budget of one, 100 = every substep rejected
thread_local double chem_test_retry_start_y[CHEM_NSPEC_MAX];   // state and time the retry started from
thread_local double chem_test_retry_start_t = -1.0;
thread_local std::string chem_test_last_dump;                   // last full cell dump
thread_local long chem_test_flag_rhs[4] = {0, 0, 0, 0};   // flag: substep-start evaluation number [0], final [1], at recover [2], first subcycle [3]
#endif

// CVODE step floor: 10 x the double resolution of the time coordinate at the substep end. Below it
// t + h no longer advances t; CVODE's own first-step lower bound is 100 x that. A failure at the floor
// is logged with the cell context and rejected like any failed substep (halve and retry, fatal 100 below 1e-8 s).
static const double CHEM_HMIN_EPS = 10.0 * DBL_EPSILON;

// every ChemCell input the solve reads (tdust: its call-entry value is dumped separately)
#ifdef SGCHEM_HMINUS_PHOTODETACHMENT_FROM_STELLAR_BANDS
#define CHEM_CELL_HMINUS(X) X(hm_detach_stellar) X(G0_hm_isrf)
#else
#define CHEM_CELL_HMINUS(X)
#endif
#define CHEM_CELL_SCALARS(X) \
    X(net) X(trac_num) X(nspec) X(abhe) X(yn) X(dl) X(divv) X(column_est) \
    X(abundc) X(abundo) X(abundsi) X(abundN) X(abundD) X(abundM) X(G0) X(G0_LW) X(G0_dust) CHEM_CELL_HMINUS(X) \
    X(dust_to_gas_ratio) X(cosmic_ray_ion_rate) X(cr_energy_density) X(redshift) X(AV_ext) \
    X(deff) X(phi_pah) X(h2_form_ex) X(h2_form_kin) X(f_rsc) X(AV_conversion_factor) \
    X(iphoto) X(isrf_option) X(no_dchem) X(iflag_mn) X(iflag_ad) X(iflag_atom) X(iflag_3bh2a) X(iflag_3bh2b) \
    X(iflag_h3pra) X(iflag_h2opc) X(irad_heat) X(G_dust) X(chi_isrf) X(fshield_CO) X(npix) X(NH_cell) \
    X(diffuse_dust_heat) X(rt_phot_HI) X(rt_phot_HeI) X(rt_phot_HeII) X(rt_heat_HI) X(rt_heat_HeI) X(rt_heat_HeII) \
    X(rt_phot_H2) X(rt_heat_H2) X(dvdr) X(Zg) X(n20_kgr_const) X(n20_tbb) X(n20_sig_lw) X(n20_chi_pe) X(n20_chi_lw) \
    X(n20_chi_fuv) X(n20_xi) X(n20_gd_uv) X(n20_gd_ext) X(n20_xe)
#define CHEM_CELL_ARRAYS(X) \
    X(Zmass, 12) X(phrates, CHEM_NPHTAB) X(col_h2, CHEM_NPIX) X(col_tot, CHEM_NPIX) X(col_co, CHEM_NPIX) \
    X(fac_uv, CHEM_NPIX) X(pix_att, CHEM_NPIX) X(col_c, CHEM_NPIX) X(n20_wlw, CHEM_NPIX)

static void dump_add(std::string& s, const char* key, const double* v, int n)
{
    char b[40];
    s += ' '; s += key; s += '=';
    for (int i = 0; i < n; i++) {
        std::snprintf(b, sizeof(b), (i ? ",%.17g" : "%.17g"), v[i]);
        s += b;
    }
}

// The full cell state at %.17g, written with one fprintf (threads do not interleave); chem_cell_parse reads it back.
static void chem_dump_cell(const char* why, const ChemCell& c, const ChemCellEntry& e)
{
    char head[256];
    std::snprintf(head, sizeof(head), "cxx_evolve: %s (ID %ld, net %d); full cell state:\n", why, c.id_current, c.net);
    std::string s = head;
    std::snprintf(head, sizeof(head), "  cell %ld entry:", c.id_current);
    s += head;
    const double sc[] = {e.time, e.t_start, e.t_end, e.tdust_entry, (double)e.attempt, (double)e.code, e.tdust_now, e.n20_xe_entry};
    const char* sk[] = {"dt", "t_start", "t_end", "tdust_entry", "attempt", "code", "tdust_now", "n20_xe_entry"};
    for (int i = 0; i < 8; i++) dump_add(s, sk[i], &sc[i], 1);
    dump_add(s, "y_entry", e.y_entry, c.nspec);
    dump_add(s, "y_now", e.y_now, c.nspec);
    std::snprintf(head, sizeof(head), "\n  cell %ld inputs:", c.id_current);
    s += head;
#define DUMP_S(f) { const double v = (double)c.f; dump_add(s, #f, &v, 1); }
#define DUMP_A(f, n) { double v[n]; for (int i = 0; i < (n); i++) v[i] = (double)c.f[i]; dump_add(s, #f, v, n); }
    CHEM_CELL_SCALARS(DUMP_S)
    std::snprintf(head, sizeof(head), " id_current=%ld", c.id_current);
    s += head;
    std::snprintf(head, sizeof(head), "\n  cell %ld arrays:", c.id_current);
    s += head;
    CHEM_CELL_ARRAYS(DUMP_A)
#undef DUMP_S
#undef DUMP_A
    std::snprintf(head, sizeof(head), "\n  cell %ld end\n", c.id_current);
    s += head;
    std::fprintf(stderr, "%s", s.c_str());
    std::fflush(stderr);
#ifdef TEST_SGCHEM_FORCED_SOLVER_FAILURE
    chem_test_last_dump = s;
#endif
}

// values of key in text ("key=v1,v2,..."): returns how many were read (0 if the key is absent)
static int parse_key(const char* text, const char* key, double* v, int nmax)
{
    const size_t kl = std::strlen(key);
    for (const char* p = std::strstr(text, key); p; p = std::strstr(p + 1, key)) {
        if ((p != text && p[-1] != ' ') || p[kl] != '=') continue;
        const char* q = p + kl + 1;
        int n = 0;
        while (n < nmax) {
            char* end;
            v[n] = std::strtod(q, &end);
            if (end == q) break;
            n++;
            if (*end != ',') break;
            q = end + 1;
        }
        return n;
    }
    return 0;
}

// Rebuild a cell and its call from a chem_dump_cell text; c.tab is left to the caller. Returns 0 ok, else the
// number of fields missing or short.
int chem_cell_parse(const char* text, ChemCell& c, ChemCellEntry& e)
{
    int bad = 0;
    double v[CHEM_NPHTAB + CHEM_NPIX + CHEM_NSPEC_MAX];   // longer than any dumped array
#define PARSE_S(f) { if (parse_key(text, #f, v, 1) == 1) c.f = (decltype(c.f))v[0]; else bad++; }
#define PARSE_A(f, n) { if (parse_key(text, #f, v, n) == (n)) { for (int i = 0; i < (n); i++) c.f[i] = v[i]; } else bad++; }
    CHEM_CELL_SCALARS(PARSE_S)
    CHEM_CELL_ARRAYS(PARSE_A)
#undef PARSE_S
#undef PARSE_A
    const char* sk[] = {"dt", "t_start", "t_end", "tdust_entry", "attempt", "code", "tdust_now", "n20_xe_entry"};
    double sc[8];
    for (int i = 0; i < 8; i++) { if (parse_key(text, sk[i], v, 1) == 1) sc[i] = v[0]; else { sc[i] = 0; bad++; } }
    e.time = sc[0]; e.t_start = sc[1]; e.t_end = sc[2]; e.tdust_entry = sc[3];
    e.attempt = (int)sc[4]; e.code = (int)sc[5]; e.tdust_now = sc[6]; e.n20_xe_entry = sc[7];
    c.n20_xe = e.n20_xe_entry;
    {
        const char* p = std::strstr(text, " id_current=");
        if (p) c.id_current = std::strtol(p + 12, nullptr, 10); else bad++;
    }
    if (parse_key(text, "y_entry", e.y_entry, c.nspec) != c.nspec) bad++;
    if (parse_key(text, "y_now", e.y_now, c.nspec) != c.nspec) bad++;
    c.tdust = e.tdust_entry;
    return bad;
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
            else { chem_clip_excess_fail[j]++; return 1; }  // large excess
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

static CvWork* cv_get(int nspec, const double* atol, double t0, const double* y, double tol_scale)
{
    CvWork* w = &g_work[(nspec == 4) ? 0 : (nspec == 10) ? 2 : (nspec == 20) ? 3 : (nspec == 3) ? 4 : 1];
    if (w->mem == nullptr) {
        if (SUNContext_Create(NULL, &w->ctx)) return nullptr;
        w->nspec = nspec;
        w->yv = N_VNew_Serial(nspec, w->ctx);
        w->atolv = N_VNew_Serial(nspec, w->ctx);
        for (int i = 0; i < nspec; i++) {
            NV_Ith_S(w->yv, i) = y[i];
            NV_Ith_S(w->atolv, i) = tol_scale * atol[i];
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
            NV_Ith_S(w->atolv, i) = tol_scale * atol[i];
        }
        if (CVodeReInit(w->mem, t0, w->yv) < 0) return nullptr;
    }
    CVodeSVtolerances(w->mem, tol_scale * CHEM_RTOL, w->atolv);
    CVodeSetMaxNumSteps(w->mem, 200000);             // iwork(6) = 200000
    return w;
}

// Explicit energy-only subcycle at frozen composition (TNOCHEM bypass, .F:296-320): 10% of e/|edot| per
// step, RHS retval ignored (a flagged evaluation leaves ydot zero, as the Fortran ipar-clear path).
// Returns 1 at t_end, 0 when stop() asks to hand off, 2 when the budget is spent, -1 on an RHS ABORT.
template <class Stop>
static int energy_only_subcycle(double& t_start, double t_end, double* y, double* ydot, int itmp,
                                ChemCell& c, Stop stop)
{
    for (long n = 1; n <= 100000; n++) {
        int rerr = rhs_dispatch(t_start, y, ydot, c);
        if (c.fatal_code) return -1;
#ifdef TEST_SGCHEM_FORCED_SOLVER_FAILURE
        if (chem_test_flag_rhs[3] && n == 1) { rerr = 1; c.cr_ion_heat = 1e30; for (int i = 0; i <= itmp; i++) ydot[i] = 0.0; }
#endif
        if (rerr && ydot[itmp] == 0.0) c.cr_ion_heat = 0.0;   // flagged evaluation: nothing applied, nothing booked
        double dt_byp;
        if (std::fabs(ydot[itmp]) > 0.0)
            dt_byp = std::fmin(t_end - t_start, 1e-1 * std::fabs(y[itmp] / ydot[itmp]));
        else
            dt_byp = t_end - t_start;
        c.cr_ion_heat_int += c.cr_ion_heat * dt_byp; // explicit step: exactly the applied heating
        y[itmp] += dt_byp * ydot[itmp];
        t_start += dt_byp;
        if (t_start >= t_end - 1e-10 * t_end) return 1;  // WEDGEFIX 3
        if (stop()) return 0;
    }
    return 2;
}

// evolve_abundances port.
// non_eq_abundances has c.trac_num entries; energy in/out [erg cm^-3].
// Returns 0 ok; 100 = dt collapse (ABORT(100)) and 1 = nstep exceeded (ABORT(0)) after a failed retry below
// TNOCHEM (R7, D-59); 2 = negative/zero dt (ABORT(1)/(2)); 3 = CVODE setup failure.
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
    if (c.net == 18) {   // own driver: whole interval in one CVODE solve, no hot bypass
        c.yn = yn; c.dl = dl; c.divv = divv;
        return n18_evolve(time, non_eq_abundances, energy, c);
    }

    cxx_stat_calls++;
    c.yn = yn; c.dl = dl; c.divv = divv;
    c.fatal_code = 0;
    c.cr_ion_heat_int = 0.0;

    double y[CHEM_NSPEC_MAX], y_old[CHEM_NSPEC_MAX], ydot[CHEM_NSPEC_MAX];
    for (int i = 0; i < c.trac_num; i++) y[i] = non_eq_abundances[i];
    y[itmp] = *energy;
    for (int i = 0; i < nspec; i++) { y_old[i] = y[i]; ydot[i] = 0.0; }

    int attempt = 0;             // 1 = retry after a solver failure (tolerances and substep safety x0.1)
    int dumped = 0;              // full cell state already printed for this call
    ChemCellEntry ent;           // call-entry state for chem_dump_cell
    ent.time = time; ent.t_start = 0.0; ent.t_end = time; ent.tdust_entry = c.tdust; ent.tdust_now = c.tdust;
    ent.attempt = 0; ent.code = 0; ent.n20_xe_entry = c.n20_xe;
    for (int i = 0; i < nspec; i++) { ent.y_entry[i] = y[i]; ent.y_now[i] = y[i]; }
    auto dump = [&](const char* why, int code, double ts, const double* ynow) {
        ent.code = code; ent.t_start = ts; ent.attempt = attempt; ent.tdust_now = c.tdust;
        for (int i = 0; i < nspec; i++) ent.y_now[i] = ynow[i];
        chem_dump_cell(why, c, ent);
        dumped = 1;
    };

    // a non-finite state or density handed in by the caller is a bug upstream: fail now, with context
    {
        int bad = !std::isfinite(yn) || !std::isfinite(time);
        for (int i = 0; i < nspec; i++) bad |= !std::isfinite(y[i]);
        if (bad) {
            dump("FATAL non-finite input state (code 101)", 101, 0.0, y);
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
    auto set_hot_cie = [&]() {   // molecules zero, HOTIONFIX hot-CIE ion set (.F:216-297)
        y[ih2] = 0.0;
        if (c.net != 20) y[ico] = 0.0;
        if (c.net == 17) y[CIHD] = 0.0;
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
    };

    double tol_scale = 1.0;
    CvWork* w = nullptr;
    int retcode = 1;             // if the loop exhausts: "nstep exceeded"
    double crh_prev = 0.0, crh_dt = 0.0; // CR ionisation heating at the last accepted state, pending accepted interval
    int crh_valid = 0;           // crh_prev holds an unflagged evaluation (else the pending interval is booked as a rectangle)

attempt_start:
#ifdef TEST_SGCHEM_FORCED_SOLVER_FAILURE
    if (attempt == 1) { for (int i = 0; i < nspec; i++) chem_test_retry_start_y[i] = y[i]; chem_test_retry_start_t = t_start; }
#endif
    // ---- TNOCHEM hot bypass (.F:200-260); the retry is always a CVODE retry (D-59) ----
    if (attempt == 0 && bypass_temp() > SGCHEM_TNOCHEM) {
        set_hot_cie();
        // net 15 builds the full ydot, as rate_eq_nl99 does
        int done = (energy_only_subcycle(t_start, t_end, y, ydot, itmp, c,
                        [&]() { return bypass_temp() <= SGCHEM_TNOCHEM; }) == 1);   // hand off to stiff path
        if (c.fatal_code) {                             // Fortran ABORT inside rate_eq
            dump("RHS abort in the TNOCHEM bypass", c.fatal_code, t_start, y);
            return c.fatal_code;
        }
        // budget exhausted while hot: accept state (.F:257-259)
        if (done || t_start >= t_end - 1e-10 * t_end) {
            for (int i = 0; i < c.trac_num; i++) non_eq_abundances[i] = y[i];
            *energy = y[itmp];
            return 0;
        }
        // else: fall through to the stiff path for the remaining interval
    }

    {
    // ---- CVODE setup (cached workspace; ReInit == istate=1) ----
    w = cv_get(nspec, atol, t_start, y, tol_scale);
    if (!w) { dump("CVODE setup failure (code 3)", 3, t_start, y); return 3; }
    CVodeSetUserData(w->mem, &c);

    int new_step = 1;
    int need_reinit = 0;         // istate=1 equivalent pending
    double dt = 0.0, dt_est = 0.0;
    double tdust_old = c.tdust;
    retcode = 1;
#ifdef TEST_SGCHEM_FORCED_SOLVER_FAILURE
    const int force = chem_test_force_failure[attempt];
    const long nstep_max = (force == 1) ? 1 : (attempt ? CHEM_NSTEP_RETRY : CHEM_NSTEP);
#else
    const int force = 0;
    const long nstep_max = attempt ? CHEM_NSTEP_RETRY : CHEM_NSTEP;
#endif

    long nsub_start = 0;         // substep-start evaluations of this attempt (test hook)
    (void)nsub_start;
    for (long I = 1; I <= nstep_max; I++) {
        if (new_step == 1) {
            nsub_start++;
            for (int j = 0; j < nspec; j++) y_old[j] = y[j];
            tdust_old = c.tdust;

            int rerr = rhs_dispatch(t_start, y, ydot, c);
            if (c.fatal_code) { retcode = c.fatal_code; goto out; }
#ifdef TEST_SGCHEM_FORCED_SOLVER_FAILURE
            if (chem_test_flag_rhs[0] == nsub_start) { rerr = 1; c.cr_ion_heat = 1e30; }   // test: flagged, stale CR heating
#endif
            if (rerr) c.cr_ion_heat = crh_prev;          // flagged evaluation: no fresh CR heating
            c.cr_ion_heat_int += 0.5 * ((crh_valid ? crh_prev : c.cr_ion_heat) + c.cr_ion_heat) * crh_dt; // trapezoid over the accepted substep
            crh_valid |= !rerr; crh_prev = c.cr_ion_heat; crh_dt = 0.0;
            dt_est = 1e18;
            for (int j = 0; j < nspec; j++) {
                if (used[j] && ydot[j] != 0.0 && safety[j] != 0.0 && y[j] > ctrl_floor[j]) {
                    double e = std::fabs(tol_scale * safety[j] * y[j] / ydot[j]);
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
                if (dt < 1e-8) { retcode = 100; goto recover; }
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
            if (c.net == 17) CVodeSetStopTime(w->mem, t_start + dt);   // net 17: no step past the substep end (D-60)
            int flag = CVode(w->mem, t_start + dt, w->yv, &tret, CV_NORMAL);
            if (force == 100) flag = CV_ERR_FAILURE;          // test: every substep rejected
            cxx_stat_substeps++;
            if (c.fatal_code) { retcode = c.fatal_code; goto out; }
            if (chem_nan_states > nan_before && nan_states0 == nan_before) {
                CVodeGetCurrentStep(w->mem, &nan_h);
                nan_t = t_start; nan_dt = dt;
            }
            if (flag < 0) {
                double hcur = 0.0;
                CVodeGetCurrentStep(w->mem, &hcur);
                const int at_floor = (std::fabs(hcur) <= hmin * (1.0 + 1e-6));
                cxx_stat_floor_hits += at_floor;
                if (at_floor && floor_hits++ == 0) {
                    std::fprintf(stderr, "cxx_evolve: CVODE step reached the floor h=%g (time resolution at t=%g, substep dt=%g, flag %d) of %g; substep rejected\n",
                                 hcur, t_start + dt, dt, flag, t_end);
                    dump("CVODE step-size floor (state at the substep start)", flag, t_start, y_old);
                    dumped = 0;                           // informational; a later failure still prints
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
                    retcode = 100; goto recover;
                }
                cxx_stat_retries++;
                new_step = 0;
                need_reinit = 1;                          // istate = 1
            } else {
                t_start = t_start + dt;
                crh_dt = dt;
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
    }

recover:
    // solver failure (1, 100), y = state at the last accepted time t_start (D-59)
    {
        const double tdust_acc = c.tdust;                     // the evaluation must not move the accepted dust temperature
        int rerr = rhs_dispatch(t_start, y, ydot, c);         // CR heating at the accepted state closes the pending trapezoid
        c.tdust = tdust_acc;
        if (c.fatal_code) { retcode = c.fatal_code; goto out; }
#ifdef TEST_SGCHEM_FORCED_SOLVER_FAILURE
        if (chem_test_flag_rhs[2]) { rerr = 1; c.cr_ion_heat = 1e30; }   // test: flagged, stale CR heating
#endif
        if (rerr) c.cr_ion_heat = crh_prev;
        c.cr_ion_heat_int += 0.5 * ((crh_valid ? crh_prev : c.cr_ion_heat) + c.cr_ion_heat) * crh_dt;
        crh_valid |= !rerr; crh_prev = c.cr_ion_heat; crh_dt = 0.0;
        long ns = 0;
        if (w && w->mem && CVodeGetNumSteps(w->mem, &ns) == CV_SUCCESS) cxx_stat_cvode_steps += ns;
        if (attempt == 0) {
            cxx_stat_failure_retries++;
            std::fprintf(stderr, "cxx_evolve: solver failure code %d at t=%.17g of %.17g s (ID %ld); retry from the last "
                         "accepted state with rtol/atol and substep safety x0.1 (retry %ld on this thread)\n",
                         retcode, t_start, t_end, c.id_current, cxx_stat_failure_retries);
            attempt = 1; tol_scale = 0.1; w = nullptr;
            goto attempt_start;
        }
        const double Tb = bypass_temp();
        if (Tb > SGCHEM_TNOCHEM) {
            cxx_stat_tnochem_finishes++;
            char why[256];
            std::snprintf(why, sizeof(why), "retry failed with code %d in hot gas (bypass T %g > %g); rest of the step via "
                          "the TNOCHEM path (energy only, hot-CIE ions; %ld on this thread)",
                          retcode, Tb, (double)SGCHEM_TNOCHEM, cxx_stat_tnochem_finishes);
            dump(why, retcode, t_start, y);
            set_hot_cie();
            const int s = energy_only_subcycle(t_start, t_end, y, ydot, itmp, c, []() { return false; });
            if (c.fatal_code) { retcode = c.fatal_code; dumped = 0; goto out; }
            if (s == 2)
                std::fprintf(stderr, "cxx_evolve: energy-only budget spent at t=%g of %g s (ID %ld); state accepted\n",
                             t_start, t_end, c.id_current);
            retcode = 0;
            w = nullptr;
            goto out;
        }
        char why[256];
        std::snprintf(why, sizeof(why), "retry failed with code %d, bypass T %g <= %g: abort", retcode, Tb, (double)SGCHEM_TNOCHEM);
        dump(why, retcode, t_start, y);
        w = nullptr;
    }

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
            int rerr_final = rhs_dispatch(t_start, y, ydot_final, c);
#ifdef TEST_SGCHEM_FORCED_SOLVER_FAILURE
            if (chem_test_flag_rhs[1]) { rerr_final = 1; c.cr_ion_heat = 1e30; }   // test: flagged, stale CR heating
#endif
            if (rerr_final) c.cr_ion_heat = crh_prev;    // flagged: no fresh CR heating
        }
        const double crh_end = chem_legacy_lambda() ? crh_prev : c.cr_ion_heat;
        c.cr_ion_heat_int += 0.5 * ((crh_valid ? crh_prev : crh_end) + crh_end) * crh_dt; // last accepted substep
    }
out_nocopy:
    if (retcode != 0 && !dumped) dump("solver failure returned to the caller", retcode, t_start, y);
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
