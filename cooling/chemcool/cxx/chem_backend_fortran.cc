// chem_backend_fortran.cc -- Gate-2 leaf-kernel + table binding layer.
//
// See DEVIATIONS.md D-0 for the scope decision.  Short version: Gate 2 ports
// the SOLVER layer (driver + RHS + tolerance/step/retry policy) to C++/CVODE.
// Everything strictly BELOW the RHS -- cool_func, calc_dust_temp,
// calc_H2_shielding_factor, lookup_temp_gamma -- and the table builders
// (coolinmo/cheminmo) remain the canonical Fortran, reached through the shims
// here.  That is what makes the one-zone benchmark a clean A/B of the ported
// layer: identical tables, identical cooling function, different integrator
// and different RHS assembly.
//
// The Fortran COMMON blocks are still the transport for the leaf calls (they
// read /coolr/ directly), so chem_bind_cell() copies COMMON -> ChemCell for the
// C++ RHS, and the shims copy the two in/out quantities (tdust) back and forth.
// In the one-zone harness there is exactly one cell, so no leakage is possible;
// in production (Gate 4) the leaves must be ported for thread safety (audit R2).

#include "chem_types.h"
#include <cstring>
#include <cstdio>

extern "C" {

#include "GIZMO_config.h"
#include "cooling/chemcool/f2c.h"
#include "cooling/chemcool/chemcool_consts.h"

#ifndef NSIDE
#define NSIDE 1
#endif
#ifndef NPIX
#define NPIX (12 * NSIDE * NSIDE)
#endif

// ---- /coolr/ : layout copied verbatim from onezone2.c (which is itself the
// ---- cool.h common-block order); must stay byte-compatible.
#include "cooling/chemcool/chem_commons.h"


extern CHEM_TLS_VARCOMP struct { double abhe; double X_H_chem; } CHEM_COMP;

// /current_photo/ -- the frozen photorate cache (audit §1.5)
extern CHEM_TLS struct { double G_dust, chi_isrf, phrates[NPHTAB]; } current_photo_;

#if defined(TREE_RAD) || defined(TREE_RAD_H2)
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

// ---- /co_data/, /h2o_data/, /h2_opac_data/, /co_shield/, /gamma_data/ ----
// Layout copied verbatim from cool.h:291-341 / mol_data.h:60-63 /
// shield_data.h:96 / gamma.h.  Fortran a(i,j) has i FAST, so a C array
// declared [j][i] has the same storage.
#define NTCO       1996
#define NCDCO      46
#define NTCO_VIB   3901
#define NCDCO_VIB  61
#define NTH2O      3991
#define NCDH2O     91
#define NTH2O_VIB  3901
#define NCDH2O_VIB 61
#define NH2OP      101

extern struct {
    double co_temptab[NTCO], co_colntab[NCDCO];
    double co_L0[NTCO], dTco_L0[NTCO];
    double co_lte[NTCO][NCDCO], co_n05[NTCO][NCDCO], co_alp[NTCO][NCDCO];
    double dTco_lte[NTCO][NCDCO], dTco_n05[NTCO][NCDCO], dTco_alp[NTCO][NCDCO];
    double co_vib_temptab[NTCO_VIB], co_vib_colntab[NCDCO_VIB];
    double co_vib_LTE_final[NTCO_VIB][NCDCO_VIB];
    double dTco_vib_LTE[NTCO_VIB][NCDCO_VIB];
} co_data_;

extern struct {
    double h2o_temptab[NTH2O], h2o_colntab[NCDH2O];
    double h2o_L0_ortho[NTH2O], dTh2o_L0_ortho[NTH2O];
    double h2o_L0_para[NTH2O],  dTh2o_L0_para[NTH2O];
    double h2o_LTE_ortho[NTH2O][NCDH2O], h2o_n05_ortho[NTH2O][NCDH2O],
           h2o_alp_ortho[NTH2O][NCDH2O];
    double h2o_LTE_para[NTH2O][NCDH2O],  h2o_n05_para[NTH2O][NCDH2O],
           h2o_alp_para[NTH2O][NCDH2O];
    double dTh2o_LTE_ortho[NTH2O][NCDH2O], dTh2o_n05_ortho[NTH2O][NCDH2O],
           dTh2o_alp_ortho[NTH2O][NCDH2O];
    double dTh2o_LTE_para[NTH2O][NCDH2O],  dTh2o_n05_para[NTH2O][NCDH2O],
           dTh2o_alp_para[NTH2O][NCDH2O];
    double h2o_vib_temptab[NTH2O_VIB], h2o_vib_colntab[NCDH2O_VIB];
    double h2o_vib_LTE_final[NTH2O_VIB][NCDH2O_VIB];
    double dTh2o_vib_LTE[NTH2O_VIB][NCDH2O_VIB];
} h2o_data_;

extern struct {
    double h2_opac_temp[NH2OP], h2_opac_column[NH2OP];
    double h2_opac[NH2OP][NH2OP];
} h2_opac_data_;

extern struct {
    double CO_self_shielding[161], CO_H2_shielding[181];
} co_shield_;

extern struct {
    double temp_table[CHEM_ISIZE], gamma_table[CHEM_ISIZE], eh2_table[CHEM_ISIZE2];   /* gamma.h isize, isize2 */
} gamma_data_;

// ---- Fortran leaf kernels (all arguments by reference) ----
void cool_func_(double* temp, double* yn, double* dl, double* divv,
                double* G_dust, double* abundances, double* ylam,
                double* rates, int* iflag_cmb, double* energy);
void calc_dust_temp_(double* yn, double* chi_isrf, double* temp, double* abHI,
                     double* RH2, double* Rdust, int* nocalc);
void calc_h2_shielding_factor_(double* fshield, double* temp, double* abh2,
                               double* NH, double* rpar);
void lookup_temp_gamma_(double* abh2, double* ekn, double* temp, double* gamma);

}  // extern "C"

#include "chem_tables_data.h"

// ===========================================================================
// binding
// ===========================================================================

void chem_bind_tables(ChemTables& T)
{
    T.temptab = COOLR.temptab;
    // cool.h:244,255 declare chtab(nchtab,nmd) => species index FAST.  The C
    // declaration above spells the same storage as [NCHTAB][NMD]; both are
    // NCHTAB*NMD doubles, and the correct element (k,itemp) sits at
    // (itemp-1)*NCHTAB + (k-1).  Reinterpret with the row stride that makes
    // chem_CH()'s [itemp-1][k-1] indexing hit that address.
    T.chtab   = reinterpret_cast<const double (*)[NCHTAB]>(&COOLR.chtab[0][0]);
    T.dtchtab = reinterpret_cast<const double (*)[NCHTAB]>(&COOLR.dtchtab[0][0]);
    T.cltab   = reinterpret_cast<const double (*)[NCLTAB]>(&COOLR.cltab[0][0]);
    T.dtcltab = reinterpret_cast<const double (*)[NCLTAB]>(&COOLR.dtcltab[0][0]);
    T.crtab   = COOLR.crtab;
    T.phtab   = COOLR.phtab;
    T.cst     = COOLR.cst;
    T.crphot  = COOLR.crphot;
    T.dtlog   = COOLR.dtlog;
    T.tmin    = COOLR.tmin;
    T.tmax    = COOLR.tmax;

    // ---- Gate 3: the leaf-kernel tables, viewed from their COMMONs ----
    // Used ONLY by the Gate-3 golden harness, so a kernel deviation is not
    // confounded with a table-builder deviation.  Production
    // (chem_tables_build.cc) owns its own copies and links no Fortran.
    T.co_temptab = co_data_.co_temptab;
    T.co_colntab = co_data_.co_colntab;
    T.co_L0      = co_data_.co_L0;
    T.dTco_L0    = co_data_.dTco_L0;
    T.co_lte     = &co_data_.co_lte[0][0];
    T.co_n05     = &co_data_.co_n05[0][0];
    T.co_alp     = &co_data_.co_alp[0][0];
    T.dTco_lte   = &co_data_.dTco_lte[0][0];
    T.dTco_n05   = &co_data_.dTco_n05[0][0];
    T.dTco_alp   = &co_data_.dTco_alp[0][0];
    T.co_vib_temptab   = co_data_.co_vib_temptab;
    T.co_vib_colntab   = co_data_.co_vib_colntab;
    T.co_vib_LTE_final = &co_data_.co_vib_LTE_final[0][0];
    T.dTco_vib_LTE     = &co_data_.dTco_vib_LTE[0][0];

    T.h2o_temptab = h2o_data_.h2o_temptab;
    T.h2o_colntab = h2o_data_.h2o_colntab;
    T.h2o_L0_ortho = h2o_data_.h2o_L0_ortho;
    T.dTh2o_L0_ortho = h2o_data_.dTh2o_L0_ortho;
    T.h2o_L0_para = h2o_data_.h2o_L0_para;
    T.dTh2o_L0_para = h2o_data_.dTh2o_L0_para;
    T.h2o_LTE_ortho = &h2o_data_.h2o_LTE_ortho[0][0];
    T.h2o_n05_ortho = &h2o_data_.h2o_n05_ortho[0][0];
    T.h2o_alp_ortho = &h2o_data_.h2o_alp_ortho[0][0];
    T.h2o_LTE_para  = &h2o_data_.h2o_LTE_para[0][0];
    T.h2o_n05_para  = &h2o_data_.h2o_n05_para[0][0];
    T.h2o_alp_para  = &h2o_data_.h2o_alp_para[0][0];
    T.dTh2o_LTE_ortho = &h2o_data_.dTh2o_LTE_ortho[0][0];
    T.dTh2o_n05_ortho = &h2o_data_.dTh2o_n05_ortho[0][0];
    T.dTh2o_alp_ortho = &h2o_data_.dTh2o_alp_ortho[0][0];
    T.dTh2o_LTE_para  = &h2o_data_.dTh2o_LTE_para[0][0];
    T.dTh2o_n05_para  = &h2o_data_.dTh2o_n05_para[0][0];
    T.dTh2o_alp_para  = &h2o_data_.dTh2o_alp_para[0][0];
    T.h2o_vib_temptab   = h2o_data_.h2o_vib_temptab;
    T.h2o_vib_colntab   = h2o_data_.h2o_vib_colntab;
    T.h2o_vib_LTE_final = &h2o_data_.h2o_vib_LTE_final[0][0];
    T.dTh2o_vib_LTE     = &h2o_data_.dTh2o_vib_LTE[0][0];

    T.h2_opac_temp   = h2_opac_data_.h2_opac_temp;
    T.h2_opac_column = h2_opac_data_.h2_opac_column;
    T.h2_opac        = &h2_opac_data_.h2_opac[0][0];

#ifdef WSS_CIE_COOL
    T.wss_C = COOLR.C_tbl;  T.wss_N = COOLR.N_tbl;  T.wss_O = COOLR.O_tbl;
    T.wss_Ne = COOLR.Ne_tbl; T.wss_Mg = COOLR.Mg_tbl; T.wss_Si = COOLR.Si_tbl;
    T.wss_S = COOLR.S_tbl;  T.wss_Ca = COOLR.Ca_tbl; T.wss_Fe = COOLR.Fe_tbl;
    T.HeI_tbl = COOLR.HeI_tbl; T.HeII_tbl = COOLR.HeII_tbl;
#endif
    T.CO_self_shielding = co_shield_.CO_self_shielding;
    T.CO_H2_shielding   = co_shield_.CO_H2_shielding;
    T.temp_table  = gamma_data_.temp_table;
    T.gamma_table = gamma_data_.gamma_table;
    T.eh2_table   = gamma_data_.eh2_table;

    // isrf.h chi_tab_* and highn_gamma.h hng are function-local Fortran DATA
    // arrays -- not reachable through any COMMON.  The machine-extracted
    // copies are used in both modes and validated end-to-end by the calc_photo
    // (chi_isrf) and compute_gamma goldens.
    T.chi_tab_mathis = dat_chi_tab_mathis;
    T.chi_tab_black  = dat_chi_tab_black;
    T.hng            = dat_hng;
    T.owner = 0;
}

// Gate 3: hand the harness's own configuration scalars to the C++ table
// builder, so the `cxx` backend builds its tables instead of viewing /coolr/.
void chem_bind_build_config(ChemBuildConfig& cfg, int net)
{
    cfg.net = net;
    cfg.iflag_atom = COOLI.iflag_atom;
    cfg.iflag_mn = COOLI.iflag_mn;
    cfg.iflag_ad = COOLI.iflag_ad;
    cfg.iflag_3bh2a = COOLI.iflag_3bh2a;
    cfg.iflag_3bh2b = COOLI.iflag_3bh2b;
    cfg.iflag_h3pra = COOLI.iflag_h3pra;
#ifdef WSS_CIE_COOL
    cfg.wss_cie = 1;
#else
    cfg.wss_cie = 0;
#endif
    cfg.deff = COOLR.deff;
    cfg.phi_pah = COOLR.phi_pah;
    cfg.dust_to_gas_ratio = COOLRC.dust_to_gas_ratio;
    cfg.cosmic_ray_ion_rate = COOLRC.cosmic_ray_ion_rate;
}

void chem_bind_cell(ChemCell& c, const ChemTables& T, int net)
{
    std::memset(&c, 0, sizeof(ChemCell));
    c.net = net;
    c.trac_num = (net == 5) ? 3 : chem_net_nl99(net) ? 9 : 7;
    c.nspec = c.trac_num + 1;
    c.abhe = CHEM_COMP.abhe;

    c.abundc = COOLRC.abundc;   c.abundo = COOLRC.abundo;
    c.abundsi = COOLRC.abundsi; c.abundN = COOLRC.abundN;
    c.abundD = COOLRC.abundD;
    c.abundM = COOLRC.abundM;

    c.G0 = COOLRC.G0; c.G0_LW = COOLRC.G0_LW; c.G0_dust = COOLRC.G0_dust;
    c.dust_to_gas_ratio = COOLRC.dust_to_gas_ratio;
    c.cosmic_ray_ion_rate = COOLRC.cosmic_ray_ion_rate;
    c.cr_energy_density = COOLRC.cr_energy_density;
    c.redshift = COOLRC.redshift;
    c.AV_ext = COOLR.AV_ext;

    c.deff = COOLR.deff; c.phi_pah = COOLR.phi_pah;
    c.h2_form_ex = COOLR.h2_form_ex; c.h2_form_kin = COOLR.h2_form_kin;
    c.f_rsc = COOLR.f_rsc;
    c.AV_conversion_factor = COOLR.AV_conversion_factor;

    c.iphoto = COOLI.iphoto; c.isrf_option = COOLI.isrf_option;
    c.iflag_mn = COOLI.iflag_mn; c.iflag_ad = COOLI.iflag_ad;
    c.iflag_atom = COOLI.iflag_atom;
    c.iflag_3bh2a = COOLI.iflag_3bh2a; c.iflag_3bh2b = COOLI.iflag_3bh2b;
    c.iflag_h3pra = COOLI.iflag_h3pra; c.iflag_h2opc = COOLI.iflag_h2opc;
    c.irad_heat = COOLIC.irad_heat;
    c.id_current = COOLIC.id_current;
    // audit §3.1 item 4: no_dchem is never written for net 17 and works only by
    // Fortran BSS zero-init.  Made explicit here.
    c.no_dchem = 0;

#ifdef WSS_CIE_COOL
    for (int k = 0; k < 12; k++) c.Zmass[k] = COOLRC.Zmass[k];
#endif

    // frozen photo state (must be bound AFTER CALC_PHOTO_WRAPPER)
    for (int k = 0; k < NPHTAB && k < CHEM_NPHTAB; k++)
        c.phrates[k] = current_photo_.phrates[k];
    c.G_dust = current_photo_.G_dust;
    c.chi_isrf = current_photo_.chi_isrf;

    c.fshield_CO = 1.0;
    c.npix = 0;
#if defined(TREE_RAD) || defined(TREE_RAD_H2)
    c.npix = NPIX;
    for (int k = 0; k < NPIX && k < CHEM_NPIX; k++) {
        c.col_h2[k]  = PROJECT.column_density_projection_h2[k];
        c.col_tot[k] = PROJECT.column_density_projection[k];
        c.col_co[k]  = PROJECT.column_density_projection_co[k];
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
        c.fac_uv[k] = PROJECT.fac_uv[k];
#else
        c.fac_uv[k] = 1.0 / NPIX;
#endif
    }
#ifdef DIFFUSE_DUST_EMISSION
    c.diffuse_dust_heat = PROJECT.diffuse_dust_heat;
#endif
#endif
    c.rt_phot_HI  = COOLRC.rt_phot_HI;
    c.rt_phot_HeI = COOLRC.rt_phot_HeI;
    c.rt_phot_HeII= COOLRC.rt_phot_HeII;
    c.rt_heat_HI  = COOLRC.rt_heat_HI;
    c.rt_heat_HeI = COOLRC.rt_heat_HeI;
    c.rt_heat_HeII= COOLRC.rt_heat_HeII;
    c.column_est = 0.0;   // rpar(5); D-5: the iphoto 4/5 paths are always 0

    c.tdust = COOLRC.tdust;     // in/out algebraic state (audit §3.1 item 2)
    c.tab = &T;
}

// ===========================================================================
// leaf-kernel REFERENCE shims (Gate-3 golden harness only)
// The production C++ kernels live in chem_cool_func.cc / chem_cool_util.cc /
// chem_calc_photo.cc / chem_calc_temp.cc and are named cxx_*.  These fort_*
// wrappers exist so the goldens can drive both implementations side by side.
// ===========================================================================

void fort_lookup_temp_gamma(double abh2, double ekn, double* temp, double* gamma)
{
    double a = abh2, e = ekn;
    lookup_temp_gamma_(&a, &e, temp, gamma);
}

double fort_calc_H2_shielding_factor(double temp, double abh2, double NH)
{
    double fshield = 1.0, t = temp, a = abh2, n = NH;
    double rpar[NRPAR];
    std::memset(rpar, 0, sizeof(rpar));
    calc_h2_shielding_factor_(&fshield, &t, &a, &n, rpar);
    return fshield;
}

void fort_calc_dust_temp(double yn, double chi_isrf, double temp, double abHI,
                         double* RH2, double* Rdust, int nocalc, ChemCell& c)
{
    // tdust is an in/out algebraic state carried by the driver's snapshot
    // (audit §3.1 item 2): push the driver's copy in, take the Newton result out.
    COOLRC.tdust = c.tdust;
    double y = yn, x = chi_isrf, t = temp, h = abHI;
    int nc = nocalc;
    calc_dust_temp_(&y, &x, &t, &h, RH2, Rdust, &nc);
    c.tdust = COOLRC.tdust;
}

int fort_cool_func(double temp, double yn, double dl, double divv, double G_dust,
                   const double abundances[16], double* ylam,
                   double rates[CHEM_NRATES], int iflag_cmb, ChemCell& c)
{
    COOLRC.tdust = c.tdust;         // cool_func reads tdust for gas-grain cooling
    double t = temp, y = yn, d = dl, dv = divv, g = G_dust, en = 0.0;
    double ab[16];
    for (int i = 0; i < 16; i++) ab[i] = abundances[i];
    int fc = iflag_cmb;
    cool_func_(&t, &y, &d, &dv, &g, ab, ylam, rates, &fc, &en);
    // cool_func WRITES tdust in the yn*DGR > 1e10 branch (cool_func.F:615)
    c.tdust = COOLRC.tdust;
    for (int i = 0; i < CHEM_NRATES && i < 28; i++) c.lambda[i] = COOLRC.lambda[i];
    if (*ylam != *ylam) return 1;
    return 0;
}
