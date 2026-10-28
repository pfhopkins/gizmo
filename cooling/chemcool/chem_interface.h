/* cooling/chemcool/chem_interface.h
 *
 * GATE 0 of the CHEMCOOL C++/CVODE migration (strangler plan; see
 * /orion/ptmp/uli/orion/stella/CHEMCOOL_CXX_CVODE_ASSESSMENT.md, section 5):
 * freeze the chemistry seam between GIZMO and the Fortran chemistry.
 *
 * ChemInputs holds EVERY per-cell quantity that do_chemcool_step() used to
 * write directly into the Fortran COMMON blocks (COOLR / COOLI / CHEM_COMP /
 * PROJECT), plus the per-cell arguments handed to the Fortran entry points
 * (EVOLVE_ABUNDANCES / CALC_PHOTO_WRAPPER / CALC_TEMP).  ChemOutputs holds
 * what is read back from the COMMON blocks after the Fortran solve.
 *
 * Contract (per cell, per call):
 *   1. caller seeds in->dt_code, in->dl_code, in->mode, in->u_code;
 *   2. chem_fill_inputs(target, in)  -- ALL per-cell gathering logic (moved
 *      VERBATIM from do_chemcool_step; provenance comments in
 *      chem_interface.cc reference the pre-refactor chemcool.cc line
 *      numbers).  No COMMON block is written here.  The only Fortran call
 *      made here is CALC_TEMP, which is a pure lookup (reads only the
 *      init-time temp/gamma tables loaded by INIT_TEMPERATURE_LOOKUP and its
 *      explicit arguments -- verified calc_temp.F/lookup_temp_gamma).
 *      Cell-state side effects that the original fill path performed
 *      (UV_flux/LW_flux floor clamps, CellP.G0/G0_LW/CR_ionization_rate
 *      stores) are preserved verbatim inside this function.
 *   3. chem_commit_to_fortran(in)    -- the ONLY place per-cell COMMON-block
 *      state is written.  (chemcool_init() still writes the one-time global
 *      initialization defaults at startup; that is init-time, not per-cell.)
 *   4. Fortran calls (CALC_PHOTO_WRAPPER, EVOLVE_ABUNDANCES) from
 *      do_chemcool_step, using the struct fields as arguments;
 *   5. chem_read_outputs(out)        -- read-back side of the seam.
 *
 * energy and the abundance vector are in/out THROUGH the EVOLVE_ABUNDANCES
 * argument list itself (locals seeded from in->energy_cgs / in->abundances),
 * not through ChemOutputs.
 *
 * This is a pure refactor: bit-identical by construction.  Do NOT add, merge,
 * or reorder numerical operations here without a full bit-identity re-run.
 */
#ifndef CHEM_INTERFACE_H
#define CHEM_INTERFACE_H

#ifndef CHEMCOOL_ABUNDM_SOLAR
#define CHEMCOOL_ABUNDM_SOLAR 1.0e-7  /* NL99 "M" per H at solar Z (Glover & Clark 2012; SGChem MAbund default) */
#endif

#include "../../declarations/allvars.h"
#include "f2c.h"
#include "chemcool_consts.h"

#ifdef CHEMCOOL


/* ------------------------------------------------------------------------
 * Fortran COMMON block declarations - these map to the Fortran COMMON
 * /coolr/, /cooli/, /chem_comp/ and /project/ blocks (Fortran side declared
 * in cool.h).  MOVED VERBATIM from chemcool.cc:13-81 (pre-Gate-0).  The
 * conditional members MUST track the Fortran cool.h layout exactly.
 * ---------------------------------------------------------------------- */
extern "C" {
#include "chem_commons.h"


    /* Mirrors Fortran common /chem_comp/ in cool.h. Defaults initialized
     * in coolinmo from ABHE macro. With GALSF_CHEMCOOL_VARIABLE_XH_AND_ABHE,
     * overwritten per-particle (chem_commit_to_fortran) before each
     * EVOLVE_ABUNDANCES call. */
    extern CHEM_TLS_VARCOMP struct {
        double abhe;
        double X_H_chem;
    } CHEM_COMP;

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

    /* Fortran function declarations */
    void COOLINMO(void);
    void CHEMINMO(void);
    void INIT_TOLERANCES(void);
    void LOAD_H2_TABLE(void);
    void INIT_TEMPERATURE_LOOKUP(void);
    void CALC_TEMP(double *abh2, double *ekn, double *temp);
    void CALC_PHOTO_WRAPPER(double *temp, double *rpar, double *abh2, double *abhd, double *abco);
    void EVOLVE_ABUNDANCES(double *timestep, double *dl, double *yn, double *divv,
                           double *energy, double *abundances, double *column_est);
}


/* ------------------------------------------------------------------------
 * ChemInputs: the complete per-cell input state of one chemistry call.
 * Types match the COMMON-block members they are committed to (double/int).
 * Fields marked "[commit: <guard>]" are only written to the COMMON blocks
 * under that compile guard, exactly mirroring the pre-refactor per-cell
 * write set; under other configs the field is either a read-only mirror of
 * the Fortran-owned value (abhe/X_H_chem) or left untouched (never read).
 * ---------------------------------------------------------------------- */
struct ChemInputs {
    /* --- caller-seeded (arguments of do_chemcool_step) ------------------ */
    double dt_code;            /* dt   [code units], seeded by caller        */
    double dl_code;            /* dl   [code units], seeded by caller        */
    int    mode;               /* do_chemcool_step mode, seeded by caller    */
    double u_code;             /* specific internal energy [code], seeded by caller */

    /* --- hydro/thermo state (orig chemcool.cc:173-181,577-602,604-621) -- */
    double redshift;           /* -> COOLRC.redshift                          */
    double rho_cgs;            /* density [g cm^-3] (post orig:584)          */
    double timestep_cgs;       /* dt [s]            (post orig:585)          */
    double divv_cgs;           /* velocity divergence [s^-1] (post orig:588) */
    double energy_cgs;         /* internal energy density [erg cm^-3],
                                  MinEgySpec-floored (orig:578-581,586)      */
    double dl_cgs;             /* cell size [cm]    (post orig:587)          */
    double yn;                 /* H nuclei number density [cm^-3] (orig:598/601) */
    double ekn;                /* energy/(k_B n) argument of CALC_TEMP (orig:620) */
    double temp;               /* CALC_TEMP result on the ENTRY state (orig:621) */

    /* --- solver species state (orig chemcool.cc:589-618) ---------------- */
    double abundances[TRAC_NUM]; /* working abundance vector (from TracAbund);
                                    mutated by the PHOTOION/pool-clamp blocks in
                                    do_chemcool_step, then passed to
                                    EVOLVE_ABUNDANCES (in/out)               */
    double abh2, abhd, abco;   /* CALC_PHOTO_WRAPPER arguments (orig:607-617)*/
    double abe;                /* = abundances[IHP] on entry (orig:618)      */
    double rpar[NRPAR];        /* {yn, dl_cgs, divv_cgs, 0...} (orig:604-606)*/
    double column_est;         /* EVOLVE_ABUNDANCES in/out arg (orig:719)    */

    /* --- element pools (orig chemcool.cc:183-267) ----------------------- */
    double abundc;             /* -> COOLRC.abundc  (gas-phase C/H by number) */
    double abundo;             /* -> COOLRC.abundo                            */
    double abundsi;            /* -> COOLRC.abundsi                           */
    double abundN;             /* -> COOLRC.abundN [commit: GALSF_RESOLVEDISM_METALS_INDIVIDUAL] */
    double abundM;             /* -> COOLRC.abundM: NL99 low-IP metal "M" (network 15) */
    double abundD;             /* -> COOLRC.abundD                            */
    double abundNe, abundMg, abundS, abundFe;  /* gas-phase Ne, Mg, S, Fe per H (network 18) */
    double Zmass[12];          /* -> COOLRC.Zmass  [commit: WSS_CIE_COOL && METALS_INDIVIDUAL]
                                  gadget_tnt_new ordering:
                                  He,C,Mg,O,Fe,Si,H,N,Ne,S,Ca,Zn (orig:220-231) */

    /* --- dust (orig chemcool.cc:274-308,568-575,624-634) ---------------- */
    double dust_to_gas_ratio;  /* -> COOLRC.dust_to_gas_ratio [solar units];
                                  includes the 10x-solar certified-domain cap
                                  AND the T>1e5 K hot-gas sputter zero        */
    double tdust;              /* -> COOLRC.tdust (cell DustTemp or M1
                                  Dust_Temperature)                          */

    /* --- radiation field ------------------------------------------------- */
    double G0;                 /* -> COOLRC.G0    [Habing]                    */
    double G0_LW;              /* -> COOLRC.G0_LW [Habing]                    */
    double G0_dust;            /* -> COOLRC.G0_dust [commit: G0_VARIABLE || GHOSDT_G0_CR || RADTRANSFER] */
    double rt_phot_HI;         /* -> COOLRC.rt_phot_HI   [s^-1]               */
    double rt_phot_HeI;        /* -> COOLRC.rt_phot_HeI  [s^-1]               */
    double rt_phot_HeII;       /* -> COOLRC.rt_phot_HeII [s^-1]               */
    double rt_heat_HI;         /* -> COOLRC.rt_heat_HI   [erg s^-1 /atom]     */
    double rt_heat_HeI;        /* -> COOLRC.rt_heat_HeI                       */
    double rt_heat_HeII;       /* -> COOLRC.rt_heat_HeII                      */

    /* --- cosmic rays ----------------------------------------------------- */
    double cosmic_ray_ion_rate; /* -> COOLRC.cosmic_ray_ion_rate [s^-1]       */
    double cr_energy_density;   /* -> COOLRC.cr_energy_density [erg cm^-3] [commit: COSMIC_RAY_FLUID && (G0_VARIABLE || GHOSDT_G0_CR || RADTRANSFER)] */

    /* --- composition (orig chemcool.cc:592-602) ------------------------- */
    double abhe;               /* -> CHEM_COMP.abhe [commit: GALSF_CHEMCOOL_VARIABLE_XH_AND_ABHE];
                                  otherwise read-only mirror of the Fortran-
                                  owned constant (used by the He/D pool clamps) */
    double X_H_chem;           /* -> CHEM_COMP.X_H_chem [commit: as abhe]    */

    /* --- misc flags/terms ----------------------------------------------- */
    double pdv_term;           /* -> COOLRC.pdv_term (orig:721, always 0)     */
    int    id_current;         /* -> COOLIC.id_current (orig:272)             */
    int    no_chem;            /* -> COOLIC.no_chem    (orig:717, always 0)   */
    int    irad_heat;          /* -> COOLIC.irad_heat (orig:565: 1 if cosmological,
                                  else keeps the init value 0)               */

    /* --- per-pixel TREECOL columns -> PROJECT (orig chemcool.cc:637-713) - */
#if defined(RT_TREECOL_DUST_WEIGHTED) || defined(RT_TREECOL_H2_CO)
    double column_density_projection[NPIX];   /* total-H (or effective-dust) column [cm^-2] */
#if defined(GALSF_RESOLVEDISM_G0_VARIABLE) || defined(RT_TREECOL_STELLAR_PIXELS_PREATTENUATED)
    double fac_uv[NPIX];       /* per-pixel FUV flux fraction [dust column && G0_VARIABLE] */
#endif
#endif
#ifdef RT_TREECOL_STELLAR_PIXELS_PREATTENUATED
    double pix_att[NPIX];      /* 1 = floor pixel (chemistry applies the column), 0 = transported stellar flux */
    double NH_cell;            /* M1 unresolved in-cell column [cm^-2]; 0 for TreeRay */
#endif
#ifdef RT_TREECOL_H2_CO
    double column_density_projection_h2[NPIX]; /* N_H2 [cm^-2] */
    double column_density_projection_co[NPIX]; /* N_CO [cm^-2] */
#endif
#if CHEMISTRYNETWORK == 20
    double dvdr_cgs;           /* mean |dv_i/dx_i| [s^-1]: CO LVG column (net 20) */
#endif
#if CHEMISTRYNETWORK == 18
    int    n18_uvb;            /* HM12 UV background on (cosmological runs) */
#endif
};


/* ------------------------------------------------------------------------
 * ChemOutputs: what do_chemcool_step reads back from the COMMON blocks
 * after CALC_PHOTO_WRAPPER / EVOLVE_ABUNDANCES.  (energy + abundances come
 * back through the EVOLVE_ABUNDANCES argument list, not through here.)
 * NOTE: when EVOLVE_ABUNDANCES is skipped (PHOTOION-ionized cells), lambda/
 * lambda_chem intentionally hold whatever the LAST solved cell left in the
 * COMMON block -- identical to pre-refactor behavior (bit-identity).
 * ---------------------------------------------------------------------- */
struct ChemOutputs {
    double tdust;                    /* <- COOLRC.tdust (updated by the solve) */
    double lambda[28];               /* <- COOLRC.lambda: individual cooling
                                        rates [erg s^-1 cm^-3]               */
    double lambda_chem[NRATES_CHEM]; /* <- COOLRC.lambda_chem                 */
    double cr_ion_heat;              /* CR ionisation heating applied over the solve [erg cm^-3] (C++ backend; 0 otherwise) */
#ifdef OUTPUT_SHIELD_FAC
    double fac_shield_h2;            /* <- COOLRC.fac_shield_h2 (calc_photo)  */
    double fac_shield_dust;          /* <- COOLRC.fac_shield_dust             */
#endif
#ifdef OUTPUT_CHEM_PHOTORATES
    double photorates[6];            /* k_H2 (x fshield) [s^-1], k_CO [s^-1], G_dust (PE), G0_dust chi_isrf/1.7 (dust heating), fshield_H2, fshield_CO */
#endif
};


void chem_fill_inputs(int target, struct ChemInputs *in);
void chem_fill_gas_abundances(int target, struct ChemInputs *in);
void chem_commit_to_fortran(const struct ChemInputs *in);
void chem_percell_snapshot(void);
#ifdef CHEMCOOL_CXX
void chem_cxx_tables_init(void);
double chem_cxx_calc_temp(double abh2, double abhe, double ekn);
void chem_cxx_begin_cell(const struct ChemInputs *in, double temp, double abh2, double abhd, double abco);
void chem_cxx_evolve_cell(double timestep, double dl, double yn, double divv, double *energy, double *abundances);
void chem_cxx_read_outputs(struct ChemOutputs *out);
#if CHEMISTRYNETWORK == 18
double chem_cxx_net18_temp(const double *abundances, double energy, double yn, double abhe);
void chem_cxx_net18_finish(const double *abundances, double energy, double *temp);
#endif
#if CHEMISTRYNETWORK == 20
void chem_cxx_net20_finish(double *abundances, double energy, double *temp);
double chem_cxx_net20_electrons(void);
#endif
#endif
void chem_read_outputs(struct ChemOutputs *out);


#endif /* CHEMCOOL */
#endif /* CHEM_INTERFACE_H */
