// chem_gizmo_backend.cc -- the CHEMCOOL_CXX entry points that GIZMO calls.
//
// ChemCell is filled straight from struct ChemInputs; calc_photo, temperature
// lookup and solve are C++.  No per-cell COMMON block is read or written.  The
// Fortran init (COOLINMO etc.) still runs once; its scalar settings seed the
// cell template so both backends start from identical constants.
//
// THREAD SAFETY: tables are immutable after chem_cxx_tables_init(); the cell,
// the last-solve rates and the CVODE workspace are thread_local.

#include "GIZMO_config.h"

#if defined(CHEMCOOL) && defined(CHEMCOOL_CXX)

#include "chem_types.h"
#include "cooling/chemcool/chem_interface.h"
#include <cstdlib>
#include <cstring>

extern "C" void chemcool_fatal_(int* code);

#if CHEMISTRYNETWORK != 5 && CHEMISTRYNETWORK != 15 && CHEMISTRYNETWORK != 16 && CHEMISTRYNETWORK != 17
#error "CHEMCOOL_CXX currently supports CHEMISTRYNETWORK 5, 15, 16 and 17 only. \
Build without CHEMCOOL_CXX (the Fortran path) for the other networks."
#endif

void chem_bind_build_config(ChemBuildConfig& cfg, int net);  // chem_backend_fortran.cc

#if defined(TREE_RAD) || defined(TREE_RAD_H2)
static_assert(NPIX <= CHEM_NPIX, "ChemCell pixel arrays are sized for NSIDE=1");
#endif

static ChemCell g_tmpl;                                   // init-time constants of every cell
static thread_local ChemCell tl_cell;                     // the cell being solved on this thread
static thread_local double tl_lambda[28], tl_lambda_chem[NRATES_CHEM];  // last solved cell (Fortran COMMON semantics)

static_assert(28 <= CHEM_NRATES && NRATES_CHEM <= CHEM_NRATES_CHEM_MAX, "rate array sizes");

void chem_cxx_tables_init(void)
{
    ChemBuildConfig cfg;
    chem_bind_build_config(cfg, CHEMISTRYNETWORK);
    const char* dd = std::getenv("CHEMCXX_DATADIR");
    if (chem_cxx_tables_init(cfg, dd ? dd : "gizmo/cooling/chemcool")) { int c = 3; chemcool_fatal_(&c); }

    // constants as chemcool_init / COOLINMO left them; per-cell fields keep the init
    // values wherever chem_commit_to_fortran would not overwrite them
    ChemCell& c = g_tmpl;
    chem_cxx_cell_defaults(c, chem_cxx_tables(), CHEMISTRYNETWORK);
    c.abhe = CHEM_COMP.abhe;
    c.iphoto = COOLI.iphoto; c.isrf_option = COOLI.isrf_option;
    c.iflag_mn = COOLI.iflag_mn; c.iflag_ad = COOLI.iflag_ad; c.iflag_atom = COOLI.iflag_atom;
    c.iflag_3bh2a = COOLI.iflag_3bh2a; c.iflag_3bh2b = COOLI.iflag_3bh2b;
    c.iflag_h3pra = COOLI.iflag_h3pra; c.iflag_h2opc = COOLI.iflag_h2opc;
    c.deff = COOLR.deff; c.phi_pah = COOLR.phi_pah;
    c.h2_form_ex = COOLR.h2_form_ex; c.h2_form_kin = COOLR.h2_form_kin;
    c.f_rsc = COOLR.f_rsc; c.AV_conversion_factor = COOLR.AV_conversion_factor; c.AV_ext = COOLR.AV_ext;
    c.abundN = COOLRC.abundN; c.G0_dust = COOLRC.G0_dust; c.cr_energy_density = COOLRC.cr_energy_density;
#ifdef WSS_CIE_COOL
    for (int k = 0; k < 12; k++) c.Zmass[k] = COOLRC.Zmass[k];
#endif
#if defined(TREE_RAD) || defined(TREE_RAD_H2)
    c.npix = NPIX;
    for (int k = 0; k < NPIX; k++) c.fac_uv[k] = 1.0 / NPIX;
#else
    c.npix = 0;
#endif
}

double chem_cxx_calc_temp(double abh2, double abhe, double ekn)
{
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
#if defined(GALSF_RESOLVEDISM_G0_VARIABLE) || defined(RADTRANSFER)
    c.G0_dust = in->G0_dust;
#endif
    c.cosmic_ray_ion_rate = in->cosmic_ray_ion_rate;
#if defined(COSMIC_RAY_FLUID) && (defined(GALSF_RESOLVEDISM_G0_VARIABLE) || defined(GALSF_RESOLVEDISM_G0_SCALE_SFR) || defined(RADTRANSFER))
    c.cr_energy_density = in->cr_energy_density;
#endif
    c.rt_phot_HI = in->rt_phot_HI; c.rt_phot_HeI = in->rt_phot_HeI; c.rt_phot_HeII = in->rt_phot_HeII;
    c.rt_heat_HI = in->rt_heat_HI; c.rt_heat_HeI = in->rt_heat_HeI; c.rt_heat_HeII = in->rt_heat_HeII;
    c.dust_to_gas_ratio = in->dust_to_gas_ratio;
    c.tdust = in->tdust;
    c.id_current = in->id_current;
    c.irad_heat = in->irad_heat;
#ifdef GALSF_CHEMCOOL_VARIABLE_XH_AND_ABHE
    c.abhe = in->abhe;
#endif
#if defined(TREE_RAD) || defined(TREE_RAD_H2)
    for (int k = 0; k < NPIX; k++) c.col_tot[k] = in->column_density_projection[k];
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
    for (int k = 0; k < NPIX; k++) c.fac_uv[k] = in->fac_uv[k];
#endif
#endif
#ifdef TREE_RAD_H2
    for (int k = 0; k < NPIX; k++) { c.col_h2[k] = in->column_density_projection_h2[k]; c.col_co[k] = in->column_density_projection_co[k]; }
#endif
    cxx_calc_photo(temp, abh2, abhd, abco, c);
}

void chem_cxx_evolve_cell(double timestep, double dl, double yn, double divv, double* energy, double* abundances)
{
    ChemCell& c = tl_cell;
    int rc = cxx_evolve_abundances(timestep, dl, yn, divv, energy, abundances, c);
    std::memcpy(tl_lambda, c.lambda, sizeof(tl_lambda));
    std::memcpy(tl_lambda_chem, c.lambda_chem, sizeof(tl_lambda_chem));
    if (rc != 0) chemcool_fatal_(&rc);  // same funnel as the Fortran driver
}

// unsolved (PHOTOION-ionized) cells keep the last solved cell's rates, as the Fortran path does
void chem_cxx_read_outputs(struct ChemOutputs* out)
{
    const ChemCell& c = tl_cell;
    out->tdust = c.tdust;
    std::memcpy(out->lambda, tl_lambda, sizeof(tl_lambda));
    std::memcpy(out->lambda_chem, tl_lambda_chem, sizeof(tl_lambda_chem));
#ifdef OUTPUT_SHIELD_FAC
    out->fac_shield_h2 = c.fac_shield_h2;
    out->fac_shield_dust = c.fac_shield_dust;
#endif
}

#endif /* CHEMCOOL && CHEMCOOL_CXX */
