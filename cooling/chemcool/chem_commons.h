/* C layout of the Fortran COMMON blocks in cool.h: /coolr/ (tables, shared), /coolrc/ (per cell, per thread),
 * /cooli/ (flags, shared), /coolic/ (per cell, per thread). Must match cool.h member for member. */
#ifndef CHEM_COMMONS_H
#define CHEM_COMMONS_H

#ifdef _OPENMP
#define CHEM_TLS __thread   /* threadprivate COMMONs are plain TLS symbols under -fopenmp */
#else
#define CHEM_TLS
#endif
#if defined(_OPENMP) && defined(GALSF_CHEMCOOL_VARIABLE_XH_AND_ABHE)
#define CHEM_TLS_VARCOMP __thread   /* /chem_comp/ is per cell only with variable composition */
#else
#define CHEM_TLS_VARCOMP
#endif

extern "C" {
extern struct {
    double temptab[NMD], cltab[NCLTAB][NMD], chtab[NCHTAB][NMD],
           dtcltab[NCLTAB][NMD], dtchtab[NCHTAB][NMD],
           crtab[NCRTAB], crphot[NCRPHOT],
           phtab[NPHTAB], cst[NCONST], dtlog, tmax, tmin,
           deff, f_rsc, phi_pah,
           AV_conversion_factor, AV_ext,
           h2_form_ex, h2_form_kin,
#ifdef WSS_CIE_COOL
           C_tbl[352], N_tbl[352], O_tbl[352],
           Ne_tbl[352], Mg_tbl[352], Si_tbl[352],
           S_tbl[352], Ca_tbl[352], Fe_tbl[352],
           HeI_tbl[51], HeII_tbl[51],
#endif
           dm_density;
} COOLR;

struct chem_coolrc_t {
    double tdust, abundc, abundo, abundsi, abundD,
           abundM, abundN, G0, G0_LW, G0_dust,
           dust_to_gas_ratio, cosmic_ray_ion_rate,
           redshift, pdv_term,
           lambda[28], lambda_chem[NRATES_CHEM],
#ifdef OUTPUT_SHIELD_FAC
           fac_shield_h2, fac_shield_dust,
#endif
#ifdef WSS_CIE_COOL
           Zmass[12],
#endif
           rt_phot_HI, rt_phot_HeI, rt_phot_HeII,
           rt_heat_HI, rt_heat_HeI, rt_heat_HeII,
           cr_energy_density;
};
extern CHEM_TLS struct chem_coolrc_t COOLRC;

extern struct {
    int iphoto, iflag_mn, iflag_ad, iflag_atom,
        iflag_3bh2a, iflag_3bh2b, iflag_h3pra,
        iflag_h2opc,
        idma_mass_option,
        isrf_option;
#ifdef TEST
    int itest;
#endif
} COOLI;

struct chem_coolic_t { int id_current, index_current, no_chem, irad_heat; };
extern CHEM_TLS struct chem_coolic_t COOLIC;
}

#endif
