/* C++ implementation: Ulrich Steinwandel (uli@mpa-garching.mpg.de).
   Based on the SGChem Fortran networks by Simon Glover (GNU GPL, as distributed in Arepo). */
// chem_gizmo_backend.cc -- the C++ backend entry points that GIZMO calls.
//
// ChemCell is filled straight from struct ChemInputs; init, tables, calc_photo,
// temperature lookup and solve are C++.  No Fortran is called or linked.
//
// THREAD SAFETY: tables are immutable after chem_cxx_tables_init(); the cell,
// the last-solve rates and the CVODE workspace are thread_local.

#include "GIZMO_config.h"

#if defined(SGCHEM) && !defined(SGCHEM_FORTRAN)

#include "chem_types.h"
#include "chem_net20.h"
#include "chem_net18.h"
#include "cooling/sgchem/chem_interface.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>

extern "C" void sgchem_fatal_(int* code);

#if CHEMISTRYNETWORK != 3 && CHEMISTRYNETWORK != 5 && CHEMISTRYNETWORK != 15 && CHEMISTRYNETWORK != 16 && CHEMISTRYNETWORK != 17 && CHEMISTRYNETWORK != 18 && CHEMISTRYNETWORK != 20
#error "The SGCHEM C++ backend currently supports CHEMISTRYNETWORK 3, 5, 15, 16, 17, 18 and 20 only. \
Build with SGCHEM_FORTRAN (the legacy Fortran backend) for the other networks."
#endif


#if defined(RT_TREECOL_DUST_WEIGHTED) || defined(RT_TREECOL_H2_CO)
static_assert(NPIX <= CHEM_NPIX, "ChemCell pixel arrays are sized for RT_USE_TREECOL_FOR_NH=12");
#endif

static ChemCell g_tmpl;                                   // init-time constants of every cell
static thread_local ChemCell tl_cell;                     // the cell being solved on this thread
static thread_local double tl_lambda[28], tl_lambda_chem[NRATES_CHEM];  // last solved cell (Fortran COMMON semantics)

static_assert(28 <= CHEM_NRATES && NRATES_CHEM <= CHEM_NRATES_CHEM_MAX, "rate array sizes");

// coolinmo: abhe = ABHE stores a default-REAL literal; X_H_chem = 1d0/(1d0+4d0*ABHE)
static const double g_abhe = (double)(float)ABHE;
double chem_cxx_abhe_default(void) { return g_abhe; }
double chem_cxx_xh_default(void) { return 1.0 / (1.0 + 4.0 * g_abhe); }

void chem_cxx_tables_init(const struct ChemInitParams* p)
{
    ChemBuildConfig cfg;
    cfg.net = CHEMISTRYNETWORK;
    cfg.iflag_atom = p->iflag_atom; cfg.iflag_mn = p->iflag_mn; cfg.iflag_ad = p->iflag_ad;
    cfg.iflag_3bh2a = p->iflag_3bh2a; cfg.iflag_3bh2b = p->iflag_3bh2b; cfg.iflag_h3pra = p->iflag_h3pra;
#ifdef WSS_CIE_COOL
    cfg.wss_cie = 1;
#else
    cfg.wss_cie = 0;
#endif
    cfg.deff = p->deff; cfg.phi_pah = p->phi_pah;
    cfg.dust_to_gas_ratio = p->dust_to_gas_ratio;
    cfg.cosmic_ray_ion_rate = p->cosmic_ray_ion_rate;
    cfg.abhe = g_abhe;
    const char* dd = std::getenv("CHEMCXX_DATADIR");
    if (!dd) dd = (p->tables_dir && *p->tables_dir) ? p->tables_dir : "gizmo/cooling/sgchem";
    if (chem_cxx_tables_init(cfg, dd)) { int c = 3; sgchem_fatal_(&c); }
#if CHEMISTRYNETWORK == 18
    if (n18_tables_load(dd)) { int c = 3; sgchem_fatal_(&c); }
#endif

    // init-time constants; per-cell fields are overwritten in chem_cxx_begin_cell
    ChemCell& c = g_tmpl;
    chem_cxx_cell_defaults(c, chem_cxx_tables(), CHEMISTRYNETWORK);
    c.abhe = g_abhe;
    c.iphoto = p->iphoto; c.isrf_option = p->isrf_option;
    c.iflag_mn = p->iflag_mn; c.iflag_ad = p->iflag_ad; c.iflag_atom = p->iflag_atom;
    c.iflag_3bh2a = p->iflag_3bh2a; c.iflag_3bh2b = p->iflag_3bh2b;
    c.iflag_h3pra = p->iflag_h3pra; c.iflag_h2opc = 0;
    c.deff = p->deff; c.phi_pah = p->phi_pah;
    c.h2_form_ex = p->h2_form_ex; c.h2_form_kin = p->h2_form_kin;
    c.f_rsc = 1.0; c.AV_conversion_factor = p->AV_conversion_factor; c.AV_ext = p->AV_ext;  // f_rsc as photoinit
    c.abundN = 0.0; c.G0_dust = 0.0; c.cr_energy_density = p->cr_energy_density;
#ifdef WSS_CIE_COOL
    for (int k = 0; k < 12; k++) c.Zmass[k] = 0.0;
#endif
#if defined(RT_TREECOL_DUST_WEIGHTED) || defined(RT_TREECOL_H2_CO)
    c.npix = NPIX;
    for (int k = 0; k < NPIX; k++) c.fac_uv[k] = 1.0 / NPIX;
#else
    c.npix = 0;
#endif
}

double chem_cxx_calc_temp(double abh2, double abhe, double ekn)
{
#if CHEMISTRYNETWORK == 20
    (void)abh2; (void)abhe;
    return ekn / 1.5;   // gamma = 5/3; the solve recomputes T with the equilibrium x_e
#endif
    double temp, gamma;
    cxx_lookup_temp_gamma(abh2, abhe, ekn, &temp, &gamma, chem_cxx_tables());
    return temp;
}

// ChemInputs -> ChemCell (same fields and guards as chem_commit_to_fortran), then calc_photo
void chem_cxx_begin_cell(const struct ChemInputs* in, double temp, double abh2, double abhd, double abco)
{
    ChemCell& c = tl_cell;
    c = g_tmpl;
    c.yn = in->rpar[0]; c.dl = in->rpar[1]; c.divv = in->rpar[2];
    c.redshift = in->redshift;
    c.abundc = in->abundc; c.abundo = in->abundo; c.abundsi = in->abundsi;
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    c.abundN = in->abundN;
#ifdef WSS_CIE_COOL
    for (int k = 0; k < 12; k++) c.Zmass[k] = in->Zmass[k];
#endif
#endif
    c.abundD = in->abundD; c.abundM = in->abundM;
    c.G0 = in->G0; c.G0_LW = in->G0_LW;
#if defined(GALSF_RESOLVEDISM_G0_VARIABLE) || defined(GHOSDT_G0_CR) || defined(RADTRANSFER)
    c.G0_dust = in->G0_dust;
#ifdef SGCHEM_HMINUS_PHOTODETACHMENT_FROM_STELLAR_BANDS
    c.hm_detach_stellar = in->hm_detach_stellar; c.G0_hm_isrf = in->G0_hm_isrf;
#endif
#endif
    c.cosmic_ray_ion_rate = in->cosmic_ray_ion_rate;
#if defined(COSMIC_RAY_FLUID) && (defined(GALSF_RESOLVEDISM_G0_VARIABLE) || defined(GHOSDT_G0_CR) || defined(RADTRANSFER))
    c.cr_energy_density = in->cr_energy_density;
#endif
    c.rt_phot_HI = in->rt_phot_HI; c.rt_phot_HeI = in->rt_phot_HeI; c.rt_phot_HeII = in->rt_phot_HeII;
    c.rt_heat_HI = in->rt_heat_HI; c.rt_heat_HeI = in->rt_heat_HeI; c.rt_heat_HeII = in->rt_heat_HeII;
    c.rt_phot_H2 = in->rt_phot_H2; c.rt_heat_H2 = in->rt_heat_H2;
    c.dust_to_gas_ratio = in->dust_to_gas_ratio;
    c.tdust = in->tdust;
    c.id_current = in->id_current;
    c.irad_heat = in->irad_heat;
#ifdef GALSF_SGCHEM_VARIABLE_XH_AND_ABHE
    c.abhe = in->abhe;
#endif
#if defined(RT_TREECOL_DUST_WEIGHTED) || defined(RT_TREECOL_H2_CO)
    for (int k = 0; k < NPIX; k++) c.col_tot[k] = in->column_density_projection[k];
#if defined(GALSF_RESOLVEDISM_G0_VARIABLE) || defined(RT_TREECOL_STELLAR_PIXELS_PREATTENUATED)
    for (int k = 0; k < NPIX; k++) c.fac_uv[k] = in->fac_uv[k];
#endif
#endif
#ifdef RT_TREECOL_STELLAR_PIXELS_PREATTENUATED
    for (int k = 0; k < NPIX; k++) c.pix_att[k] = in->pix_att[k];
    c.NH_cell = in->NH_cell;
#endif
#ifdef RT_TREECOL_H2_CO
    for (int k = 0; k < NPIX; k++) { c.col_h2[k] = in->column_density_projection_h2[k]; c.col_co[k] = in->column_density_projection_co[k]; }
#endif
#if CHEMISTRYNETWORK == 18
    {
        const double pool[N18_NEL] = {1.0, c.abhe, in->abundc, in->abundN, in->abundo, in->abundNe, in->abundMg, in->abundsi, in->abundS, in->abundFe};
        for (int e = 0; e < N18_NEL; e++) c.n18_pool[e] = std::max(pool[e], 0.0);
        if (in->n18_uvb) c.n18_flags |= N18_UVB;
    }
#endif
#if CHEMISTRYNETWORK == 20
    c.Zg = c.abundo / 3.2e-4;
    c.dvdr = in->dvdr_cgs;
    c.n20_tbb = std::max(6.0, 2.73 * (1.0 + c.redshift));
#endif
    cxx_calc_photo(temp, abh2, abhd, abco, c);
}

void chem_cxx_evolve_cell(double timestep, double dl, double yn, double divv, double* energy, double* abundances)
{
    ChemCell& c = tl_cell;
    int rc = cxx_evolve_abundances(timestep, dl, yn, divv, energy, abundances, c);
    std::memcpy(tl_lambda, c.lambda, sizeof(tl_lambda));
    std::memcpy(tl_lambda_chem, c.lambda_chem, sizeof(tl_lambda_chem));
    if (rc != 0) sgchem_fatal_(&rc);  // same funnel as the Fortran driver
}

// unsolved (PHOTOION-ionized) cells keep the last solved cell's rates, as the Fortran path does
void chem_cxx_read_outputs(struct ChemOutputs* out)
{
    const ChemCell& c = tl_cell;
    out->tdust = c.tdust;
    out->cr_ion_heat = c.cr_ion_heat_int;
    std::memcpy(out->lambda, tl_lambda, sizeof(tl_lambda));
    std::memcpy(out->lambda_chem, tl_lambda_chem, sizeof(tl_lambda_chem));
#ifdef OUTPUT_SHIELD_FAC
    out->fac_shield_h2 = c.fac_shield_h2;
    out->fac_shield_dust = c.fac_shield_dust;
#endif
#ifdef OUTPUT_CHEM_PHOTORATES
    out->photorates[0] = c.phrates[0] * c.fac_shield_h2;
    out->photorates[1] = c.phrates[9];
    out->photorates[2] = c.G_dust;
    out->photorates[3] = c.G0_dust * c.chi_isrf / 1.7;
    out->photorates[4] = c.fac_shield_h2;
    out->photorates[5] = c.fshield_CO;
#endif
}

#if CHEMISTRYNETWORK == 18
double chem_cxx_net18_temp(const double* abundances, double energy, double yn, double abhe)
{
    return energy / (1.5 * 1.380649E-16 * yn * n18_particles_trac(abundances, abhe));
}

void chem_cxx_net18_finish(const double* abundances, double energy, double* temp)
{
    *temp = chem_cxx_net18_temp(abundances, energy, tl_cell.yn, tl_cell.abhe);
}
#endif

#if CHEMISTRYNETWORK == 20
// final temperature, equilibrium x_e and CO for the (evolved or forced) state of this thread's cell
void chem_cxx_net20_finish(double* abundances, double energy, double* temp)
{
    ChemCell& c = tl_cell;
    N20State s;
    if (n20_solve_state(abundances[C20_IH2], abundances[C20_IHP], energy, c, s)) { int rc = 4; sgchem_fatal_(&rc); }
    abundances[ICO] = s.xCO;
    *temp = s.T;
}

double chem_cxx_net20_electrons(void) { return tl_cell.n20_xe; }
#endif

#endif /* SGCHEM && !SGCHEM_FORTRAN */
