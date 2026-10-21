// chem_types.h -- Gate-2 shared interface for the CHEMCOOL C++/CVODE port.
//
// Design contract (see /orion/ptmp/uli/orion/stella/CHEMCOOL_CONTROL_FLOW_AUDIT.md §7):
//  - R2 REDESIGN: all per-cell COMMON state becomes the explicit ChemCell struct,
//    passed to every kernel.  Immutable global tables live in ChemTables (built
//    once at init; thread-safe to share).
//  - Index conventions: species use the 0-based C macros (IH2, IHP, ...); the ODE
//    vector is y[0..TRAC_NUM-1] species + y[TRAC_NUM] = internal energy DENSITY
//    [erg cm^-3].  Rate-table ids keep the Fortran 1-based numbering through the
//    CH(k)/CL(k) accessors so the port reads like the .F source.
//  - All values are double (Fortran REAL*8 everywhere).
#ifndef CHEM_TYPES_H
#define CHEM_TYPES_H

#include <cmath>
#include "../../../GIZMO_config.h"   /* CHEMISTRYNETWORK sets the chtab/crphot/cst widths below */

// ---- dimensions (mirror cooling/chemcool/chemcool_consts.h) ----
#define CHEM_NMD     10000
#define CHEM_NCLTAB  67
#if CHEMISTRYNETWORK == 16  /* Gong+17 adds chtab 174-181, crphot 13, cst 92-94 */
#define CHEM_NCHTAB  181
#define CHEM_NCRPHOT 13
#define CHEM_NCONST  94
#else
#define CHEM_NCHTAB  173
#define CHEM_NCRPHOT 12
#define CHEM_NCONST  91
#endif
#define CHEM_NCRTAB  12
#define CHEM_NPHTAB  54
#define CHEM_NRATES  28      /* cool_func radiative rates */
#define CHEM_NPIX    12      /* NSIDE=1 HEALPix, as one-zone/production */
#define CHEM_NETAB_X 21      /* energy table blocks in x = abh2/(1+abhe-abh2) (gamma.h netab_x) */
#define CHEM_ISIZE   (50000 * CHEM_NETAB_X)  /* energy/gamma table rows (gamma.h) */
#define CHEM_ISIZE2  50000   /* eh2 table rows */
#define CHEM_NABN    16      /* ABUND_NUM: cool_func abundance vector */

/* ---- Gate 3: molecular-coolant table dimensions (cool.h:267-341) ---- */
#define CHEM_NTCO        1996
#define CHEM_NCDCO       46
#define CHEM_NTCO_VIB    3901
#define CHEM_NCDCO_VIB   61
#define CHEM_NTH2O       3991
#define CHEM_NCDH2O      91
#define CHEM_NTH2O_VIB   3901
#define CHEM_NCDH2O_VIB  61
#define CHEM_NH2OP       101    /* H2-cooling-ratios.dat is 101 x 101 */
#define CHEM_NWSS        352    /* wss_z_collis.txt rows */
#define CHEM_NHE_TBL     51     /* HeI_tbl / HeII_tbl */
#define CHEM_IMAX_COSS   161
#define CHEM_IMAX_COH2   181
#define CHEM_NCHI        200    /* isrf.h chi tables */
#define CHEM_NHNG        125    /* highn_gamma.h */

// ---- physical constants ----
// NOT the decimal values in chemcool_consts.h: those macros are spelled
// `1.38066e-16` etc., i.e. Fortran DEFAULT REAL (single-precision) literals.
// cool.h does `REAL kboltz; parameter (kboltz = BOLTZMANN)`, so the constant is
// rounded to float32 and only then widened to real*8 -- the build carries no
// -fdefault-real-8.  What the running Fortran holds is therefore the float32
// value, up to 5e-8 relative away from the decimal.  Using the exact decimal
// here would silently "improve" the port and show up later as unexplained A/B
// drift; the values below are float32(decimal) widened to double, which is
// bit-exactly what the Fortran uses.  Found 2026-07-29 via the cl1-3 spline
// goldens (worst relerr was exactly float32(kboltz) = 1.264e-09).
// Decimal originals: 1.60219e-12, 1.38066e-16, 1.6726e-24, 6.672e-8.
#define CHEM_EV         1.6021899518908755e-12
#define CHEM_KBOLTZ     1.3806599982553653e-16
#define CHEM_PROTONMASS 1.6726000836652178e-24
#define CHEM_GRAVITY    6.6719998415010195e-08
#define CHEM_PI        ((double)3.14159265358979323846f)  /* Fortran PI is a default-REAL literal */
#define CHEM_COOL_EPS  1e-6
#define CHEM_EPS_MAX   1e-4     /* validate_input reject threshold */

// network-dependent compile-time constants are passed at runtime instead:
// net 5: abhe=0.10, TRAC_NUM=3, NRATES_CHEM=6
// net 17: abhe=0.10, TRAC_NUM=7, NRATES_CHEM=22
// nets 15/16: abhe=0.10, TRAC_NUM=9, NRATES_CHEM=6
#define CHEM_NRATES_CHEM_MAX 22

// species slots (0-based, chemcool_consts.h nets 5/17)
#define CIH2   0
#define CIHP   1
#define CICO   2
#define CIHEP  3   /* net 17 only */
#define CIHEPP 4
#define CIDP   5
#define CIHD   6

// nets 15 (NL99) and 16 (Gong+17) species slots (0-based); energy is slot 9
#define C15_IH2   0
#define C15_IHP   1
#define C15_IC    2   /* C+ */
#define C15_ICHX  3
#define C15_IOHX  4
#define C15_ICO   5
#define C15_IHCOP 6
#define C15_IHEP  7
#define C15_IMP   8   /* M+ (net 16: Si+) */
#define CHEM_NSPEC_MAX 10

// nets 15 and 16 share species slots, driver policy and pools
static inline int chem_net_nl99(int net) { return net == 15 || net == 16; }

// ---------------------------------------------------------------------------
// Immutable table state.  This is a *view* over the rate/cooling tables, NOT a
// copy: at Gate 2 the tables are the ones COOLINMO/CHEMINMO built in the
// Fortran /coolr/ COMMON (see chem_backend_fortran.cc, chem_bind_tables()).
//
// LAYOUT (verified against cool.h:244,255):
//     Fortran  chtab(nchtab, nmd)   -- species index FAST
//  => C        chtab[itemp-1][k-1]  with row stride CHEM_NCHTAB
// The Gate-2 scaffold originally declared chtab[NCHTAB][NMD] and indexed
// [k-1][itemp-1], i.e. TRANSPOSED; fixed here (see DEVIATIONS.md D-8).
// ---------------------------------------------------------------------------
struct ChemTables {
    const double* temptab;                       // [CHEM_NMD]
    const double (*chtab )[CHEM_NCHTAB];         // [CHEM_NMD][CHEM_NCHTAB]
    const double (*dtchtab)[CHEM_NCHTAB];
    const double (*cltab )[CHEM_NCLTAB];         // [CHEM_NMD][CHEM_NCLTAB]
    const double (*dtcltab)[CHEM_NCLTAB];
    const double* crtab;                         // [CHEM_NCRTAB]
    const double* crphot;                        // [CHEM_NCRPHOT]
    const double* phtab;                         // [CHEM_NPHTAB]
    const double* cst;                           // [CHEM_NCONST]
    double dtlog, tmin, tmax;

    // ---- Gate 3: everything the leaf kernels used to read from COMMON ----
    // CO rotational cooling (/co_data/).  Fortran co_lte(ncdco,nTco) => column
    // index FAST: element (iN,iT) lives at (iT-1)*CHEM_NCDCO + (iN-1).
    const double* co_temptab;      // [CHEM_NTCO]
    const double* co_colntab;      // [CHEM_NCDCO]
    const double* co_L0;           // [CHEM_NTCO]
    const double* dTco_L0;         // [CHEM_NTCO]
    const double* co_lte;          // [CHEM_NTCO * CHEM_NCDCO]
    const double* co_n05;
    const double* co_alp;
    const double* dTco_lte;
    const double* dTco_n05;
    const double* dTco_alp;
    // CO vibrational cooling
    const double* co_vib_temptab;  // [CHEM_NTCO_VIB]
    const double* co_vib_colntab;  // [CHEM_NCDCO_VIB]
    const double* co_vib_LTE_final;// [CHEM_NTCO_VIB * CHEM_NCDCO_VIB]
    const double* dTco_vib_LTE;
    // H2O rotational cooling (/h2o_data/)
    const double* h2o_temptab;     // [CHEM_NTH2O]
    const double* h2o_colntab;     // [CHEM_NCDH2O]
    const double* h2o_L0_ortho;    const double* dTh2o_L0_ortho;
    const double* h2o_L0_para;     const double* dTh2o_L0_para;
    const double* h2o_LTE_ortho;   const double* dTh2o_LTE_ortho;
    const double* h2o_n05_ortho;   const double* dTh2o_n05_ortho;
    const double* h2o_alp_ortho;   const double* dTh2o_alp_ortho;
    const double* h2o_LTE_para;    const double* dTh2o_LTE_para;
    const double* h2o_n05_para;    const double* dTh2o_n05_para;
    const double* h2o_alp_para;    const double* dTh2o_alp_para;
    // H2O vibrational cooling
    const double* h2o_vib_temptab; // [CHEM_NTH2O_VIB]
    const double* h2o_vib_colntab; // [CHEM_NCDH2O_VIB]
    const double* h2o_vib_LTE_final;
    const double* dTh2o_vib_LTE;
    // H2 opacity table (/h2_opac_data/, loaded from H2-cooling-ratios.dat).
    // Fortran h2_opac(nh2op,nh2op): first index FAST.
    const double* h2_opac_temp;    // [CHEM_NH2OP]
    const double* h2_opac_column;  // [CHEM_NH2OP]
    const double* h2_opac;         // [CHEM_NH2OP * CHEM_NH2OP]
    // WSS CIE metal-line cooling (wss_z_collis.txt, log10 of the rates)
    const double* wss_C;  const double* wss_N;  const double* wss_O;
    const double* wss_Ne; const double* wss_Mg; const double* wss_Si;
    const double* wss_S;  const double* wss_Ca; const double* wss_Fe;
    const double* HeI_tbl;   // [CHEM_NHE_TBL]
    const double* HeII_tbl;  // [CHEM_NHE_TBL]
    // CO / H2 shielding curves (/co_shield/, built in cheminmo)
    const double* CO_self_shielding;  // [CHEM_IMAX_COSS]
    const double* CO_H2_shielding;    // [CHEM_IMAX_COH2]
    // ISRF attenuation curves (isrf.h)
    const double* chi_tab_mathis;     // [CHEM_NCHI]
    const double* chi_tab_black;      // [CHEM_NCHI]
    // temperature / gamma lookup (gamma.h, from energy-table.txt, eh2-table.txt)
    const double* temp_table;         // [CHEM_ISIZE]
    const double* gamma_table;        // [CHEM_ISIZE]
    const double* eh2_table;          // [CHEM_ISIZE2]
    // high-n gamma (highn_gamma.h)
    const double* hng;                // [CHEM_NHNG]

    // opaque owner (non-null only for C++-built tables; see chem_tables_build.cc)
    void* owner;
};

// Fortran (i,j) with the FIRST index fast, leading dimension ld.
static inline double chem_f2(const double* a, int ld, int i1, int j1)
{ return a[(size_t)(j1 - 1) * ld + (i1 - 1)]; }

// REAL**INTEGER with EXACTLY gfortran's expansion (gfc_conv_cst_int_power):
// powi(n) = powi(n/2)*powi(n-n/2).  Same helper as rates_gen.cc; kept here so
// the hand-written kernels associate their integer powers identically.
static inline double chem_fpowi(double x, int n)
{
    if (n == 0) return 1.0;
    if (n < 0)  return 1.0 / chem_fpowi(x, -n);
    if (n == 1) return x;
    double a = chem_fpowi(x, n / 2);
    double b = chem_fpowi(x, n - n / 2);
    return a * b;
}

// temperature index into the rate tables -- EXACT Fortran semantics
// (rate_eq_simple.F:150-168 / rate_eq_ulli.F:230-248).  flag_bad=1 replicates
// the temp<=0 -> ipar(1)=1 poisoned-input path (temp is then treated at tmin).
struct ChemTIdx {
    int itemp;      // 1-based, [1..CHEM_NMD]
    double dtemp;
    double temp;    // clamped temperature actually used
    int flag_bad;
};

inline ChemTIdx chem_temp_index(double temp, const ChemTables& T)
{
    ChemTIdx r; r.flag_bad = 0;
    if (temp <= 0.0)          { r.itemp = 1; r.dtemp = 0.0; r.temp = T.tmin; r.flag_bad = 1; }
    else if (temp <= T.tmin)  { r.itemp = 1; r.dtemp = 0.0; r.temp = temp; }
    else if (temp >= T.tmax)  { r.itemp = CHEM_NMD; r.dtemp = 0.0; r.temp = temp; }
    else {
        r.itemp = (int)(std::log10(temp) / T.dtlog) + 1;
        if (r.itemp <= 0) r.itemp = 1;               // unreachable; ABORT(1) in Fortran
        if (r.itemp > CHEM_NMD) r.itemp = CHEM_NMD;
        r.dtemp = temp - T.temptab[r.itemp - 1];
        r.temp = temp;
    }
    return r;
}

// Fortran-style 1-based accessors: ch(k) = chtab(k,itemp) + dtemp*dtchtab(k,itemp)
inline double chem_CH(const ChemTables& T, int k, const ChemTIdx& ti)
{ return T.chtab[ti.itemp-1][k-1] + ti.dtemp * T.dtchtab[ti.itemp-1][k-1]; }
inline double chem_CL(const ChemTables& T, int k, const ChemTIdx& ti)
{ return T.cltab[ti.itemp-1][k-1] + ti.dtemp * T.dtcltab[ti.itemp-1][k-1]; }

// ---------------------------------------------------------------------------
// Per-cell context: everything do_chemcool_step writes into the COMMONs,
// as one explicit struct (audit R2).  Passed as CVODE user_data.
// ---------------------------------------------------------------------------
struct ChemCell {
    int net;               // 5, 15, 16 or 17
    int trac_num;          // 3, 9 (15/16) or 7
    int nspec;             // trac_num + 1
    double abhe;           // 0.10 (nets 5/15/16/17)

    // rpar equivalents
    double yn;             // H-nuclei number density [cm^-3]
    double dl;             // local shielding length [cm] (0 in production)
    double divv;           // velocity divergence [s^-1]
    double column_est;     // rpar(5): external column estimate (iphoto 4/5)

    // elemental pools (number abundances rel. to H nuclei)
    double abundc, abundo, abundsi, abundN, abundD;
    double abundM;         // NL99 low-IP metal "M" (net 15)

    // radiation / grains / CR (per-cell inputs)
    double G0, G0_LW, G0_dust;
    double dust_to_gas_ratio;
    double cosmic_ray_ion_rate;
    double cr_energy_density;
    double redshift;
    double AV_ext;

    // constants normally set once by chemcool_init (kept per-cell for simplicity)
    double deff, phi_pah, h2_form_ex, h2_form_kin, f_rsc, AV_conversion_factor;
    int iphoto, isrf_option, no_dchem;
    int iflag_mn, iflag_ad, iflag_atom, iflag_3bh2a, iflag_3bh2b, iflag_h3pra, iflag_h2opc;
    int irad_heat;
    long id_current;

    // WSS metal mass fractions (chemcool.cc Zmass mapping)
    double Zmass[12];

    // frozen photo state (calc_photo output; audit R5)
    double phrates[CHEM_NPHTAB];
    double G_dust, chi_isrf;
    double fshield_CO;           // calc_photo output, frozen for the cell call
    double fac_shield_h2, fac_shield_dust;  // calc_photo diagnostics (OUTPUT_SHIELD_FAC)

    // TREECOL pixel columns for the LIVE H2 self-shielding evaluated inside
    // the RHS (calc_H2_shielding_factor); frozen for the cell call.
    int    npix;                 // 12*NSIDE^2 (0 => no pixel data)
    double col_h2[CHEM_NPIX];    // per-pixel N_H2 [cm^-2]
    double col_tot[CHEM_NPIX];   // per-pixel N_H  [cm^-2] (dust attenuation)
    double col_co[CHEM_NPIX];    // per-pixel N_CO [cm^-2]
    double fac_uv[CHEM_NPIX];    // per-pixel UV weights (sum 1)

    // heating from the surrounding dust (DIFFUSE_DUST_EMISSION; /project/)
    double diffuse_dust_heat;

    // M1 per-cell photoionization / photoheating rates (0 unless RADTRANSFER)
    double rt_phot_HI, rt_phot_HeI, rt_phot_HeII;
    double rt_heat_HI, rt_heat_HeI, rt_heat_HeII;

    // algebraic state riding outside the ODE vector (audit §3.1 item 2):
    // mutated by calc_dust_temp inside every RHS eval; snapshot/restored by
    // the driver on substep retries.
    double tdust;
    int dust_iter_failed;   // calc_dust_temp Newton budget exhausted (was ABORT(10))
    int cool_func_failed;   // cool_func hit an ABORT case (bad T index / H2 > 0.5)

    // net-17 equilibrium-species previous-iterate memory (rate_eq_ulli locals;
    // persist across RHS evals within a cell call in the Fortran because they
    // are recomputed each call from scratch -- kept here only for diagnostics)
    // (not needed for correctness: the fixed-point loop reinitializes each eval)

    // diagnostics: last-RHS-eval rate breakdowns (audit R10; the port keeps the
    // Fortran "last evaluation wins" semantics for A/B comparability)
    double lambda[CHEM_NRATES];
    double lambda_chem[CHEM_NRATES_CHEM_MAX];

    const ChemTables* tab;
};

// ---------------------------------------------------------------------------
// Kernel interfaces
// ---------------------------------------------------------------------------

// cool_func port (cool_func.F).  abundances[16] slot mapping (1-based Fortran
// slots 1..16 -> C 0..15): 0=H2,1=O,2=OH,3=H2O,4=CO,5=CI,6=C+,7=SiI,8=Si+,
// 9=e-,10=H+,11=HI,12=HD,13=HeI,14=HeII,15=HeIII.
// Returns 0 ok; nonzero = the Fortran ABORT cases (bad T index / H2>0.5+eps),
// which the caller maps to a recoverable RHS failure.
int cxx_cool_func(double temp, double yn, double dl, double divv, double G_dust,
                  const double abundances[16], double* ylam, double rates[CHEM_NRATES],
                  int iflag_cmb, ChemCell& c);

// calc_dust_temp port (cool_util.F:802-1007).  Mutates c.tdust; outputs the H2
// grain-formation rates.  nocalc=1 skips the Newton solve (RT_INFRARED path).
void cxx_calc_dust_temp(double yn, double chi_isrf, double temp, double abHI,
                        double* RH2, double* Rdust, int nocalc, ChemCell& c);

// DB96 H2 self-shielding with live abh2/temp (calc_photo.F:745-823, iphoto=6
// TreeCol path; iphoto=0 handled by caller with fshield=1).
double cxx_calc_H2_shielding_factor(double temp, double abh2, double NH, ChemCell& c);

// ---- cool_util.F kernels (chem_cool_util.cc) ------------------------------
struct CoCool  { double L0, lte, alpha, n05, vib_lte; };
struct H2oCool { double L0, lte, alpha, n05; };
void cxx_co_cool (const ChemTables& T, double temp, double N_co_eff,
                  double N_co_eff_vib, CoCool* out);
void cxx_h2o_rot_cool(const ChemTables& T, double temp, double N_h2o_eff,
                      int iop_flag, H2oCool* out);
double cxx_h2o_vib_cool(const ChemTables& T, double temp, double N_h2o_eff);
void cxx_three_level_pops(double r01, double r02, double r12,
                          double r10, double r20, double r21,
                          double* n0, double* n1, double* n2);
double cxx_compute_stim(double a10, double e10, double rad_temp);
double cxx_compute_h2_opacity(const ChemTables& T, double temp, double N_H2_eff);
double cxx_compute_h2_heating_fraction(double T, double n, double xh2);

// WSS CIE metal-line cooling (wss_z_collis.F get_coolrates_wssCIE port)
double cxx_wss_cie_rate(const ChemTables& T, double temp, const double Zm[12]);

// ---- calc_photo.F (chem_calc_photo.cc) -----------------------------------
double cxx_chi(const ChemTables& T, double AV, int isrf_option);
double cxx_calc_HD_shielding_factor(double temp, double abh2, double abhd,
                                    double NH, ChemCell& c);
// Fills c.phrates / c.G_dust / c.chi_isrf / c.fshield_CO (audit R5: once per
// cell call).  yn and dl come from c.
void cxx_calc_photo(double temp, double abh2, double abhd, double abco,
                    ChemCell& c);

// ---- calc_temp.F / compute_gamma.F (chem_calc_temp.cc) -------------------
void cxx_lookup_temp_gamma(double abh2, double abhe, double ekn, double* temp, double* gamma,
                           const ChemTables& T);
double cxx_calc_spec(const ChemTables& T, double temp);
double cxx_calc_gamma_temp(const ChemTables& T, double abh2, double temp,
                           double abhe);
double cxx_compute_gamma(const ChemTables& T, double abh2, double ekn);

// ---- table construction (chem_tables_build.cc) ---------------------------
// Builds every table above in C++ (coolinmo + cheminmo + photoinit +
// const_rates + wss reader + H2 opacity loader + temp/gamma loader).
// `datadir` is the directory the Fortran opened relative to CWD, i.e.
// "gizmo/cooling/chemcool".  Returns 0 on success.
struct ChemBuildConfig {
    int net;                 // CHEMISTRYNETWORK (5 or 17)
    int iflag_atom, iflag_mn, iflag_ad;
    int iflag_3bh2a, iflag_3bh2b, iflag_h3pra;
    int wss_cie;             // WSS_CIE_COOL
    double deff, phi_pah, dust_to_gas_ratio, cosmic_ray_ion_rate;
};
int  chem_tables_build(ChemTables& T, const ChemBuildConfig& cfg,
                       const char* datadir);
void chem_tables_free(ChemTables& T);

// ---- pure-C++ backend (chem_backend_cxx.cc) ------------------------------
// Process-global immutable tables + a per-cell context helper.  No COMMON, no
// Fortran, no shared mutable state: see the file header for the R2 argument.
const ChemTables& chem_cxx_tables();
int  chem_cxx_tables_ready();
int  chem_cxx_tables_init(const ChemBuildConfig& cfg, const char* datadir);
void chem_cxx_tables_shutdown();
void chem_cxx_cell_defaults(ChemCell& c, const ChemTables& T, int net);

// RHS routines.  y has c.nspec entries (species + energy density LAST).
// Contract (audit R3 REDESIGN): y is CONST -- validate_input clips act on an
// internal local copy; gross violations return CHEM_RHS_RECOVERABLE instead of
// mutating the integrator state.
//   return 0                    -> ydot valid
//   return CHEM_RHS_RECOVERABLE -> recoverable bad input (CVODE retries)
#define CHEM_RHS_OK 0
#define CHEM_RHS_RECOVERABLE 1
int chem_rhs_net5 (double t, const double* y, double* ydot, ChemCell& c);
int chem_rhs_net17(double t, const double* y, double* ydot, ChemCell& c);
int chem_rhs_net15(double t, const double* y, double* ydot, ChemCell& c);
int chem_rhs_net16(double t, const double* y, double* ydot, ChemCell& c);

// ---------------------------------------------------------------------------
// Gate-2 backend binding (chem_backend_fortran.cc).
//
// SCOPE DECISION (DEVIATIONS.md D-0): Gate 2 ports the SOLVER LAYER -- the
// driver (evolve_abundances.F), the RHS assemblies (rate_eq_*.F) and the
// tolerance/step/retry policy.  The *leaf kernels* below the RHS
// (cool_func, calc_dust_temp, calc_H2_shielding_factor, lookup_temp_gamma)
// and the table builders (coolinmo/cheminmo) are still the canonical Fortran,
// reached through these shims and through the /coolr/ COMMON.  This is
// deliberate: it makes the benchmark an APPLES-TO-APPLES test of the ported
// layer (any deviation is attributable to CVODE/RHS/driver, not to a
// half-ported cooling function).  Gate 3 replaces the leaves.
// ---------------------------------------------------------------------------

// Point a ChemTables at the Fortran /coolr/ tables (call AFTER COOLINMO/CHEMINMO).
void chem_bind_tables(ChemTables& T);

// Fill the per-cell context from the current /coolr/, /cooli/, /chem_comp/ and
// /current_photo/ COMMON contents (call after CALC_PHOTO_WRAPPER, before the
// solve).  `net` is 5 or 17.
void chem_bind_cell(ChemCell& c, const ChemTables& T, int net);

// driver entry point (cvode_driver.cc) -- signature mirrors EVOLVE_ABUNDANCES
int cxx_evolve_abundances(double time, double dl, double yn, double divv,
                          double* energy, double* non_eq_abundances, ChemCell& c);

#endif
