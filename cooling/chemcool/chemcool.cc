#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "../../declarations/allvars.h"
#include "../../core/proto.h"
#include "f2c.h"
#include "chemcool_consts.h"

#ifdef CHEMCOOL

/* Fortran common block declarations - these map to the Fortran COMMON /coolr/ and /cooli/ blocks */
extern "C" {
    extern struct {
        double temptab[NMD], cltab[NCLTAB][NMD], chtab[NCHTAB][NMD],
               dtcltab[NCLTAB][NMD], dtchtab[NCHTAB][NMD],
               crtab[NCRTAB], crphot[NCRPHOT],
               phtab[NPHTAB], cst[NCONST], dtlog, tdust, tmax, tmin,
               deff, abundc, abundo, abundsi, abundD,
               abundM, abundN, G0, G0_LW, f_rsc, phi_pah,
               dust_to_gas_ratio, AV_conversion_factor,
               cosmic_ray_ion_rate, redshift, AV_ext,
               pdv_term, h2_form_ex, h2_form_kin,
               lambda[28], lambda_chem[NRATES_CHEM],
#ifdef OUTPUT_SHIELD_FAC
               fac_shield_h2, fac_shield_dust,
#endif
#ifdef WSS_CIE_COOL
               Zmass[12],
               C_tbl[352], N_tbl[352], O_tbl[352],
               Ne_tbl[352], Mg_tbl[352], Si_tbl[352],
               S_tbl[352], Ca_tbl[352], Fe_tbl[352],
               HeI_tbl[51], HeII_tbl[51],
#endif
               dm_density;
    } COOLR;

    extern struct {
        int iphoto, iflag_mn, iflag_ad, iflag_atom,
            iflag_3bh2a, iflag_3bh2b, iflag_h3pra,
            iflag_h2opc, id_current, index_current,
            idma_mass_option, no_chem, irad_heat,
            isrf_option;
    } COOLI;

#if defined(TREE_RAD) || defined(TREE_RAD_H2)
    extern struct {
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


/* Initialization of chemcool */
void chemcool_init(void)
{
    if(ThisTask == 0) {
        printf("initialize SG cooling and chemistry...\n");
        fflush(stdout);
    }

    COOLINMO();
    CHEMINMO();
    INIT_TOLERANCES();
    LOAD_H2_TABLE();
    INIT_TEMPERATURE_LOOKUP();

    if(ThisTask == 0) {
        printf("initialization of SG cooling and chemistry finished.\n");
        fflush(stdout);
    }
}


/* Compute new entropy and abundances at end of timestep dt.
 * Mode = 0 ==> update TracAbund, Entropy, Gamma, DustTemp
 * Mode = 1 ==> no update, return cooling rate
 * Mode = 2 ==> no update, return temperature
 * Mode = 3 ==> no update, return final energy
 */
double do_chemcool_step(int target, double dt, double dl, int mode)
{
    double rho, timestep, divv, energy, ekn;
    double temp;
    double yn, abh2, abhd, abco, abe;
    double abundances[TRAC_NUM], column_est;
    double rpar[NRPAR];

#if defined(TREE_RAD) || defined(TREE_RAD_H2)
    double columni;
#endif
#ifdef TREE_RAD
    double NH;
#endif
#ifdef TREE_RAD_H2
    double NH2, NCO;
#endif
    int i;


    if(All.ComovingIntegrationOn) {
        /* comoving case placeholder - not currently used */
        COOLR.redshift = 0.;
        rho      = CellP[target].Density * All.cf_a3inv;
        timestep = dt;
        divv     = CellP[target].Gradients.Velocity[0][0] + CellP[target].Gradients.Velocity[1][1] + CellP[target].Gradients.Velocity[2][2];
    } else {
        rho      = CellP[target].Density * All.cf_a3inv;
        COOLR.redshift = 0.;
        timestep = dt;
        divv     = CellP[target].Gradients.Velocity[0][0] + CellP[target].Gradients.Velocity[1][1] + CellP[target].Gradients.Velocity[2][2];
    }

    /* Per-particle element abundances for cooling rates */
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    {
        /* Number abundances n_X/n_H = (X_mass_frac / A_X) / (X_H / A_H) where A_X = atomic weight */
        double X_H = DMAX(P[target].ElementAbundance[ELEM_H], 1e-10);
        COOLR.abundc  = (P[target].ElementAbundance[ELEM_C]  / 12.0) / (X_H / 1.0);
        COOLR.abundo  = (P[target].ElementAbundance[ELEM_O]  / 16.0) / (X_H / 1.0);
        COOLR.abundsi = (P[target].ElementAbundance[ELEM_Si] / 28.0) / (X_H / 1.0);
        COOLR.abundN  = (P[target].ElementAbundance[ELEM_N]  / 14.0) / (X_H / 1.0);

#ifdef WSS_CIE_COOL
        /* Map ElementAbundance[15] -> Zmass[12] (mass fractions, gadget_tnt_new ordering) */
        COOLR.Zmass[0]  = P[target].ElementAbundance[ELEM_He];  /* He */
        COOLR.Zmass[1]  = P[target].ElementAbundance[ELEM_C];   /* C  */
        COOLR.Zmass[2]  = P[target].ElementAbundance[ELEM_Mg];  /* Mg */
        COOLR.Zmass[3]  = P[target].ElementAbundance[ELEM_O];   /* O  */
        COOLR.Zmass[4]  = P[target].ElementAbundance[ELEM_Fe];  /* Fe */
        COOLR.Zmass[5]  = P[target].ElementAbundance[ELEM_Si];  /* Si */
        COOLR.Zmass[6]  = P[target].ElementAbundance[ELEM_H];   /* H  */
        COOLR.Zmass[7]  = P[target].ElementAbundance[ELEM_N];   /* N  */
        COOLR.Zmass[8]  = P[target].ElementAbundance[ELEM_Ne];  /* Ne */
        COOLR.Zmass[9]  = P[target].ElementAbundance[ELEM_S];   /* S  */
        COOLR.Zmass[10] = P[target].ElementAbundance[ELEM_Ca];  /* Ca */
        COOLR.Zmass[11] = 0; /* unused slot */
#endif
    }
#else
    /* Fallback: WNM values (Sembach+ 2000) scaled by global metallicity */
    COOLR.abundc  = All.InitialMetallicity * 1.4e-4;
    COOLR.abundo  = All.InitialMetallicity * 3.2e-4;
    COOLR.abundsi = All.InitialMetallicity * 1.5e-5;
#endif
    COOLR.abundD  = All.DeutAbund;
    COOLR.G0      = All.G0;
    COOLR.G0_LW   = All.G0; /* default: LW = full FUV, overridden below if G0_VARIABLE */
    COOLR.cosmic_ray_ion_rate = All.CosmicRayIonRate;

    COOLI.id_current = P[target].ID;

    COOLR.dust_to_gas_ratio = All.DGRnormalized;
#ifdef DGR_SCALE_WITH_Z
    COOLR.dust_to_gas_ratio = All.DGRnormalized * All.InitialMetallicity;
#endif


#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
    double u_Habing = 5.29e-14; /* Habing field, in erg cm^-3 */
    double fac_flux2habing = 1.0 / (4.*M_PI*C_LIGHT_CGS * pow(UNIT_LENGTH_IN_CGS, 2)) / u_Habing;

    double UV_flux_tot = 0.0;
    double LW_flux_tot = 0.0;
    double UV_flux_min_pix = 0.324e-2/NPIX / fac_flux2habing;
    for(i = 0; i < NPIX; i++) {
        CellP[target].UV_flux[i] = DMAX(CellP[target].UV_flux[i], UV_flux_min_pix);
        UV_flux_tot += CellP[target].UV_flux[i];
        LW_flux_tot += CellP[target].LW_flux[i]; /* LW has no floor — can be zero in pristine regions */
    }

    double G0_tot = UV_flux_tot * fac_flux2habing * All.G0;
    COOLR.G0 = G0_tot;

    /* Lyman-Werner G0 for H2 photodissociation (11.2-13.6 eV only) */
    double G0_LW = LW_flux_tot * fac_flux2habing * All.G0;
    COOLR.G0_LW = G0_LW;

#ifdef CR_SCALE_WITH_G0
    COOLR.cosmic_ray_ion_rate = (COOLR.G0 / 1.7) * All.CosmicRayIonRate;
#endif
#endif /* GALSF_RESOLVEDISM_G0_VARIABLE */


#ifdef GALSF_RESOLVEDISM_G0_SCALE_SFR
    double g0 = All.FactorG0 * All.G0;
    if(g0 < 0.324e-2) g0 = 0.324e-2;
    COOLR.G0 = g0;
    COOLR.G0_LW = g0; /* no separate LW in SFR-scaled mode */
#ifdef CR_SCALE_WITH_G0
    COOLR.cosmic_ray_ion_rate = All.FactorG0 * All.CosmicRayIonRate;
#endif
#endif


    /* Set correct dust temperature in coolr common block */
    COOLR.tdust = CellP[target].DustTemp;

    /* 'energy' is internal energy density, NOT specific internal energy [in code units] */
    energy = rho * CellP[target].InternalEnergy;
    if(energy < rho * All.MinEgySpec) {
        energy = rho * All.MinEgySpec;
    }

    /* Convert to cgs units */
    rho      *= UNIT_DENSITY_IN_CGS;
    timestep *= UNIT_TIME_IN_CGS;
    energy   *= UNIT_ENERGY_IN_CGS / pow(UNIT_LENGTH_IN_CGS, 3);
    dl       *= UNIT_LENGTH_IN_CGS;
    divv     *= UNIT_VEL_IN_CGS / UNIT_LENGTH_IN_CGS;
    for(i = 0; i < TRAC_NUM; i++) {
        abundances[i] = CellP[target].TracAbund[i];
    }
    yn = rho / ((1.0 + 4.0 * ABHE) * PROTONMASS_CGS); /* number density of hydrogen only */

    rpar[0] = yn;
    rpar[1] = dl;
    rpar[2] = divv;
    abh2 = abundances[IH2];
#if CHEMISTRYNETWORK == 1 || CHEMISTRYNETWORK == 17
    abhd = abundances[IHD];
#else
    abhd = 0.0;
#endif
#if CHEMISTRYNETWORK != 4
    abco = abundances[ICO];
#else
    abco = 0.0;
#endif
    abe = abundances[IHP];

    ekn = energy / (BOLTZMANN_CGS * (1.0 + ABHE - abh2 + abe) * yn);
    CALC_TEMP(&abh2, &ekn, &temp);
    if(mode == 2) return temp;


#ifdef TREE_RAD
    for(i = 0; i < NPIX; i++) {
        columni = CellP[target].Projection[i] * UNIT_DENSITY_IN_CGS * UNIT_LENGTH_IN_CGS;
        NH = columni / ((1.0 + 4.0 * ABHE) * PROTONMASS_CGS);
        PROJECT.column_density_projection[i] = NH;
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
        PROJECT.fac_uv[i] = CellP[target].UV_flux[i] / UV_flux_tot;
#endif
    }
#endif

#ifdef TREE_RAD_H2
    for(i = 0; i < NPIX; i++) {
        columni = CellP[target].ProjectionH2[i] * UNIT_DENSITY_IN_CGS * UNIT_LENGTH_IN_CGS;
        NH2 = columni / (2.0 * PROTONMASS_CGS);
        PROJECT.column_density_projection_h2[i] = NH2;
    }

#if CHEMISTRYNETWORK != 1 && CHEMISTRYNETWORK != 4
    for(i = 0; i < NPIX; i++) {
        columni = CellP[target].ProjectionCO[i] * UNIT_DENSITY_IN_CGS * UNIT_LENGTH_IN_CGS;
        NCO = columni / (28.0 * PROTONMASS_CGS);
        PROJECT.column_density_projection_co[i] = NCO;
    }
#else
    for(i = 0; i < NPIX; i++) {
        PROJECT.column_density_projection_co[i] = 0.0;
    }
#endif
#endif /* TREE_RAD_H2 */


    /* Switch off chemistry for high-density particles */
    COOLI.no_chem = 0;

    column_est = 0.0;

    COOLR.pdv_term = 0.;

    CALC_PHOTO_WRAPPER(&temp, rpar, &abh2, &abhd, &abco);

    int skip_evolve_abundances = 0;

#ifdef GALSF_RESOLVEDISM_PHOTOION
    double temp_HII = 1e4;
    /* abe_HII = H+ + He+ + D+ + C+ ≈ 1 + abhe + abundD + abundc */
    double abe_HII = 1.0 + ABHE + COOLR.abundD + COOLR.abundc;
    double energy_HII = temp_HII * 1.5 * BOLTZMANN_CGS * yn * (1.0 + ABHE + abe_HII);
    if(CellP[target].Ionized == 1) {
        skip_evolve_abundances = 1;
        abundances[IH2] = 0.0;
        abundances[IHP] = 0.9998;
#if CHEMISTRYNETWORK != 4
        abundances[ICO] = 0.0;
#endif
#if CHEMISTRYNETWORK == 1 || CHEMISTRYNETWORK == 17
        /* In HII regions: He singly ionized, D fully ionized, HD destroyed */
        abundances[IHEP]  = ABHE;        /* He -> He+ (13.6-54.4 eV photons) */
        abundances[IHEPP] = 0.0;         /* He++ negligible for most O stars */
        abundances[IDP]   = COOLR.abundD; /* D -> D+ (same IP as H) */
        abundances[IHD]   = 0.0;         /* HD dissociated */
#endif
        if(energy < energy_HII) energy = energy_HII;
    }
#endif

    if(mode == 1 || mode == 2) timestep = 0.0;

    /* Evolve abundances */
    if(skip_evolve_abundances == 0)
        EVOLVE_ABUNDANCES(&timestep, &dl, &yn, &divv, &energy, abundances, &column_est);

    /* Compute cooling rate */
    double cooling_rate = 0.;
    for(i = 0; i < 28; i++) {
#ifdef OUTPUT_INDIVIDUAL_COOLRATES
        CellP[target].Lambda[i] = COOLR.lambda[i] / yn / yn;
#endif
        cooling_rate += COOLR.lambda[i];
    }
    for(i = 0; i < NRATES_CHEM; i++) {
#ifdef OUTPUT_INDIVIDUAL_COOLRATES
        if(i < 6) CellP[target].LambdaChem[i] = COOLR.lambda_chem[i] / yn / yn;
#endif
        cooling_rate += COOLR.lambda_chem[i];
    }
    cooling_rate *= (UNIT_LENGTH_IN_CGS * pow(UNIT_TIME_IN_CGS, 3) / UNIT_MASS_IN_CGS);

    abh2 = abundances[IH2];
    abe  = abundances[IHP];

    /* Compute final temperature */
    ekn = energy / (BOLTZMANN_CGS * (1.0 + ABHE - abh2 + abe) * yn);
    CALC_TEMP(&abh2, &ekn, &temp);

    /* Convert back to code units from cgs */
    energy *= pow(UNIT_LENGTH_IN_CGS, 3) / UNIT_ENERGY_IN_CGS;
    rho    /= UNIT_DENSITY_IN_CGS;

    if(mode == 0) {
        for(i = 0; i < TRAC_NUM; i++) {
            CellP[target].TracAbund[i] = abundances[i];
        }
        CellP[target].InternalEnergy = energy / rho;
        CellP[target].InternalEnergyPred = CellP[target].InternalEnergy;
        CellP[target].Temp = temp;
        CellP[target].DustTemp = COOLR.tdust;
        return CellP[target].InternalEnergy;
    } else if(mode == 1) {
        return cooling_rate;
    } else if(mode == 2) {
        return temp;
    } else if(mode == 3) {
        return CellP[target].InternalEnergy;
    } else {
        printf("Unknown mode: %d!\n", mode);
        endrun(101);
    }
    return CellP[target].InternalEnergy;
}


#endif /* CHEMCOOL */
