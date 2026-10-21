// chem_backend_c_api.cc -- plain-C entry points so a C driver (onezone3.c)
// can drive the C++/CVODE chemistry backend.
//
// The ChemCell context is refreshed from the COMMON-block input scalars the
// DRIVER itself populates, on every chemcxx_init() call, which the harness
// issues immediately after CALC_PHOTO_WRAPPER -- i.e. the frozen photorate
// contract of audit §1.5 is reproduced exactly: phrates/G_dust/chi_isrf are
// snapshotted once per cell call and never refreshed during the solve.
//
// GATE 3.  The *tables* the C++ side uses are now BUILT IN C++
// (chem_tables_build.cc), not viewed from /coolr/, and the leaf kernels are
// unconditionally the C++ ones (cxx_cool_func, cxx_calc_dust_temp,
// cxx_calc_H2_shielding_factor, cxx_lookup_temp_gamma now live in
// chem_cool_func.cc / chem_cool_util.cc / chem_calc_photo.cc /
// chem_calc_temp.cc; chem_backend_fortran.cc only supplies the fort_*
// REFERENCE wrappers the goldens drive side by side).  So in this harness the
// `cxx` backend differs from the `f77` backend in EVERY layer -- tables,
// kernels, RHS, integrator.  Set
//     CHEMCXX_TABLES=fortran
// to restore the Gate-2 behaviour (view the Fortran tables), which is the way
// to bisect an A/B difference into "table build" vs "solver".

#include "chem_types.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>

extern thread_local long cxx_stat_calls, cxx_stat_substeps, cxx_stat_retries,
                         cxx_stat_rhs_evals, cxx_stat_cvode_steps;

// chem_backend_fortran.cc: reads the harness's COMMON-block input scalars
void chem_bind_build_config(ChemBuildConfig& cfg, int net);

static ChemTables g_tab;          // C++-built (default) or Fortran-bound
static thread_local ChemCell g_cell;   // per OpenMP thread; g_tab is shared, read-only after init
static int        g_bound = 0;

// tdust is a CARRIED per-cell state, not a per-call input.  In GIZMO it lives
// in CellP[i].DustTemp: chem_interface.cc:471/703 seeds COOLRC.tdust from it and
// :736 reads it back, so chem_bind_cell()'s `c.tdust = COOLRC.tdust` is exactly
// right there.  The one-zone harness, however, sets COOLRC.tdust = 10 K ONCE at
// init and never writes the C++ result back -- so without the carry below the
// C++ cell restarts its dust-temperature Newton solve from 10 K on every hydro
// step while the Fortran path carries its own COOLRC.tdust forward.  At Gate 2
// this could not happen: the leaf shim wrote c.tdust into COOLRC.tdust, so both
// backends shared the global.  Now that the C++ owns tdust in ChemCell, the
// harness needs this to compare like with like.  Measured effect: the six
// nH = 1e6 certification zones move from 1.1-1.3% Teq disagreement to <1e-3.
// Production (GIZMO) is unaffected -- the caller carries tdust there.
static thread_local double g_carry_tdust = 0.0;
static thread_local int    g_have_tdust  = 0;

extern "C" {

int chemcxx_init(int net)
{
    if (net != 5 && net != 15 && net != 16 && net != 17) return 1;
    if (!g_bound) {
        const char* mode = std::getenv("CHEMCXX_TABLES");
        if (mode && std::strcmp(mode, "fortran") == 0) {
            chem_bind_tables(g_tab);
            std::fprintf(stderr, "[chemcxx] tables: FORTRAN /coolr/ view\n");
        } else {
            ChemBuildConfig cfg;
            chem_bind_build_config(cfg, net);
            const char* dd = std::getenv("CHEMCXX_DATADIR");
            if (chem_tables_build(g_tab, cfg, dd ? dd : "gizmo/cooling/chemcool"))
                return 1;
            std::fprintf(stderr, "[chemcxx] tables: built in C++ (Gate 3)\n");
        }
        g_bound = 1;
    }
    chem_bind_cell(g_cell, g_tab, net);
    if (g_have_tdust) g_cell.tdust = g_carry_tdust;
    return 0;
}

int chemcxx_evolve(double timestep, double dl, double yn, double divv,
                   double* energy, double* abundances)
{
    if (!g_bound) return 1;
    int rc = cxx_evolve_abundances(timestep, dl, yn, divv, energy, abundances,
                                   g_cell);
    g_carry_tdust = g_cell.tdust; g_have_tdust = 1;
    return rc;
}

void chemcxx_stats(long* calls, long* substeps, long* retries,
                   long* rhs_evals, long* cvode_steps)
{
    *calls = cxx_stat_calls;   *substeps = cxx_stat_substeps;
    *retries = cxx_stat_retries; *rhs_evals = cxx_stat_rhs_evals;
    *cvode_steps = cxx_stat_cvode_steps;
}

// expose the post-solve dust temperature so the caller can carry it (the
// Fortran path carries it in COOLRC.tdust automatically)
double chemcxx_tdust(void) { return g_cell.tdust; }
void chemcxx_set_tdust(double t) { g_cell.tdust = t;
                                   g_carry_tdust = t; g_have_tdust = 1; }

// last solve's per-process cooling rates (cool_func lambda) and chemistry rates, for COOLR publication
void chemcxx_get_lambda(double* lam, int nlam, double* lamchem, int nchem)
{
    for (int k = 0; k < nlam && k < CHEM_NRATES; k++) lam[k] = g_cell.lambda[k];
    for (int k = 0; k < nchem && k < CHEM_NRATES_CHEM_MAX; k++) lamchem[k] = g_cell.lambda_chem[k];
}

}  // extern "C"
