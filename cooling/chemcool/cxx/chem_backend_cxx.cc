// chem_backend_cxx.cc -- the PURE C++ CHEMCOOL backend (Gate 3).
//
// This translation unit is what replaces chem_backend_fortran.cc on the
// CHEMCOOL_CXX path.  It links NO Fortran: the tables are built by
// chem_tables_build.cc, the kernels are chem_cool_func.cc / chem_cool_util.cc /
// chem_calc_photo.cc / chem_calc_temp.cc, the RHS is chem_rhs_net{5,17}.cc and
// the driver is cvode_driver.cc.  There is no /coolr/, no COMMON view and no
// libgfortran dependency anywhere in that set.
//
// THREAD SAFETY (audit R2).  Two kinds of state exist and they are separated:
//   * ChemTables -- built once in chem_cxx_tables_init(), read-only thereafter,
//     safe to share across threads.  Every kernel takes it by const reference.
//   * ChemCell   -- one per cell, carrying yn/dl/divv, the frozen photorates,
//     tdust and the lambda diagnostics.  Nothing is global: the C++ API takes
//     it by reference, and the plain-C convenience API keeps ONE PER THREAD
//     (thread_local), so an OpenMP loop over cells is race-free.
// The remaining shared-mutable object is the CVODE workspace cache in
// cvode_driver.cc, which is likewise thread_local (see DEVIATIONS D-7/D-15).

#include "chem_types.h"
#include "chem_net20.h"
#include <cstring>
#include <cstdio>

// ---------------------------------------------------------------------------
// process-global, immutable-after-init table set
// ---------------------------------------------------------------------------
static ChemTables g_tab;
static int        g_tab_ready = 0;

const ChemTables& chem_cxx_tables() { return g_tab; }
int chem_cxx_tables_ready() { return g_tab_ready; }

int chem_cxx_tables_init(const ChemBuildConfig& cfg, const char* datadir)
{
    if (g_tab_ready) return 0;
    if (chem_tables_build(g_tab, cfg, datadir)) return 1;
    g_tab_ready = 1;
    return 0;
}

void chem_cxx_tables_shutdown()
{
    if (g_tab_ready) { chem_tables_free(g_tab); g_tab_ready = 0; }
}

// ---------------------------------------------------------------------------
// per-cell context.  Pure function: no globals are read or written, so it is
// safe to call concurrently.  The caller then overwrites the per-cell physical
// inputs (yn, dl, divv, G0, columns, ...) before calling calc_photo/evolve.
// ---------------------------------------------------------------------------
void chem_cxx_cell_defaults(ChemCell& c, const ChemTables& T, int net)
{
    std::memset(&c, 0, sizeof(ChemCell));
    c.net = net;
    c.trac_num = chem_trac_num(net);
    c.nspec = c.trac_num + 1;
    c.abhe = 0.1;                            // chemcool_consts.h ABHE (nets 3/5/15/16/17)
    c.no_dchem = 0;                          // audit sec 3.1 item 4: explicit
    c.iflag_mn = 1; c.iflag_ad = 1; c.iflag_atom = 2;
    c.iflag_3bh2a = 1; c.iflag_3bh2b = 1; c.iflag_h3pra = 1;
    c.iflag_h2opc = 0;
    c.isrf_option = 1;
    c.iphoto = 0;
    c.deff = 1.0; c.phi_pah = 0.5;
    c.h2_form_ex = 0.84; c.h2_form_kin = 0.12;
    c.f_rsc = 1.0;                           // photoinit_ism.F:310
    c.AV_conversion_factor = 5.348e-22;
    c.tdust = 10.0;
    c.fshield_CO = 1.0;
    c.npix = CHEM_NPIX;
    for (int k = 0; k < CHEM_NPIX; k++) { c.fac_uv[k] = 1.0 / CHEM_NPIX; c.pix_att[k] = 1.0; }
    n20_cell_defaults(c);
    c.tab = &T;
}

// ---------------------------------------------------------------------------
// plain-C convenience API.  One ChemCell per THREAD (audit R2): an OpenMP
// loop over cells can call these without any cross-thread interference.
// A caller that wants full control should use the C++ API above and own its
// ChemCell explicitly.
// ---------------------------------------------------------------------------
static thread_local ChemCell tl_cell;
static thread_local int      tl_ready = 0;

extern "C" {

int chemcxx_pure_init(int net, int iflag_atom, int wss_cie,
                      double deff, double phi_pah, double dust_to_gas_ratio,
                      double cosmic_ray_ion_rate, const char* datadir)
{
    ChemBuildConfig cfg;
    cfg.net = net;
    cfg.iflag_atom = iflag_atom;
    cfg.iflag_mn = 1; cfg.iflag_ad = 1;
    cfg.iflag_3bh2a = 1; cfg.iflag_3bh2b = 1; cfg.iflag_h3pra = 1;
    cfg.wss_cie = wss_cie;
    cfg.deff = deff; cfg.phi_pah = phi_pah;
    cfg.dust_to_gas_ratio = dust_to_gas_ratio;
    cfg.cosmic_ray_ion_rate = cosmic_ray_ion_rate;
    cfg.abhe = (double)0.1f;                 // ABHE: a default-REAL literal in coolinmo
    if (chem_cxx_tables_init(cfg, datadir)) return 1;
    return 0;
}

ChemCell* chemcxx_pure_cell(int net)
{
    if (!g_tab_ready) return 0;
    if (!tl_ready) { chem_cxx_cell_defaults(tl_cell, g_tab, net); tl_ready = 1; }
    return &tl_cell;
}

void chemcxx_pure_reset_cell(int net)
{
    chem_cxx_cell_defaults(tl_cell, g_tab, net);
    tl_ready = 1;
}

/* calc_photo_wrapper equivalent -- freezes phrates/G_dust/chi_isrf for the
   whole cell call (audit R5). */
void chemcxx_pure_calc_photo(double temp, double abh2, double abhd, double abco)
{
    cxx_calc_photo(temp, abh2, abhd, abco, tl_cell);
}

/* CALC_TEMP equivalent */
double chemcxx_pure_calc_temp(double abh2, double ekn)
{
    double t, g;
    cxx_lookup_temp_gamma(abh2, tl_cell.abhe, ekn, &t, &g, g_tab);
    return t;
}

double chemcxx_pure_calc_gamma(double abh2, double ekn)
{
    double t, g;
    cxx_lookup_temp_gamma(abh2, tl_cell.abhe, ekn, &t, &g, g_tab);
    return g;
}

int chemcxx_pure_evolve(double timestep, double dl, double yn, double divv,
                        double* energy, double* abundances)
{
    if (!g_tab_ready || !tl_ready) return 1;
    return cxx_evolve_abundances(timestep, dl, yn, divv, energy, abundances,
                                 tl_cell);
}

double chemcxx_pure_tdust(void) { return tl_cell.tdust; }
void   chemcxx_pure_set_tdust(double t) { tl_cell.tdust = t; }
const double* chemcxx_pure_lambda(void) { return tl_cell.lambda; }
const double* chemcxx_pure_lambda_chem(void) { return tl_cell.lambda_chem; }

}  // extern "C"
