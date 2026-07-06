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
               abundM, abundN, G0, G0_LW, G0_dust,
               f_rsc, phi_pah,
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
               dm_density,
               rt_phot_HI, rt_phot_HeI, rt_phot_HeII,
               rt_heat_HI, rt_heat_HeI, rt_heat_HeII,
               cr_energy_density;
    } COOLR;

    extern struct {
        int iphoto, iflag_mn, iflag_ad, iflag_atom,
            iflag_3bh2a, iflag_3bh2b, iflag_h3pra,
            iflag_h2opc, id_current, index_current,
            idma_mass_option, no_chem, irad_heat,
            isrf_option;
    } COOLI;

    /* Mirrors Fortran common /chem_comp/ in cool.h. Defaults initialized
     * in coolinmo from ABHE macro. With GALSF_CHEMCOOL_VARIABLE_XH_AND_ABHE,
     * overwritten per-particle below before each EVOLVE_ABUNDANCES call. */
    extern struct {
        double abhe;
        double X_H_chem;
    } CHEM_COMP;

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
/* Target of the Fortran ABORT() macro (cool.h): clean, LOUD abort.
 * A bare Fortran 'stop' killed one rank silently and deadlocked the rest in
 * the next MPI collective (zombie jobs; pisn200 2026-07-04).  endrun() ->
 * MPI_Abort kills the WHOLE job immediately and visibly. */
extern "C" void chemcool_fatal_(int *code)
{
    printf("CHEMCOOL FATAL: solver failure code %d (particle id_current=%d, task=%d) -- clean abort.\n",
           *code, COOLI.id_current, ThisTask);
    fflush(stdout);
    endrun(78100 + *code);
}

void chemcool_init(void)
{
    if(ThisTask == 0) {
        printf("initialize SG cooling and chemistry...\n");
        fflush(stdout);
    }

    /* Set COOLI flags before COOLINMO/CHEMINMO (they read these via common block).
       Standard Glover & Clark values, matching gadget_tnt_new defaults. */
    COOLI.iflag_mn   = 1;  /* GP98 mutual neutralization rate */
    COOLI.iflag_ad   = 1;  /* GP98 H- associative detachment rate */
    COOLI.iflag_atom = 2;  /* SD93 combined atomic cooling table */
    COOLI.iflag_3bh2a = 1; /* three-body H2 formation channel a (PSS83) */
    COOLI.iflag_3bh2b = 1; /* three-body H2 formation channel b (PSS83) */
    COOLI.iflag_h3pra = 1; /* H3+ dissociative recombination (GP98) */
    COOLI.iphoto     = 6;  /* TREECOL column densities (overridden per-particle) */
    COOLI.irad_heat  = 0;  /* no external radiative heating (handled via G0) */
    COOLI.isrf_option = 1; /* Draine (1978) ISRF spectral shape */
    COOLI.no_chem    = 0;

    /* Set COOLR defaults before COOLINMO (matching gadget_tnt_new standard values) */
    COOLR.h2_form_ex  = 0.84;       /* fraction of H2 binding energy → vibrational excitation */
    COOLR.h2_form_kin = 0.12;       /* fraction → kinetic energy of H2 molecule */
    COOLR.deff        = 1.0;        /* H2 formation on dust efficiency factor */
    COOLR.phi_pah     = 1.0;        /* PAH photoelectric heating efficiency */
    COOLR.AV_conversion_factor = 5.348e-22; /* N_H to A_V conversion (Bohlin+ 1978) */
    COOLR.AV_ext      = 0.0;        /* no external dust extinction */
    COOLR.dm_density  = 0.0;        /* no dark matter annihilation heating */
    COOLR.pdv_term    = 0.0;
    COOLR.redshift    = 0.0;        /* isolated galaxy, non-cosmological */
    COOLR.abundD      = All.DeutAbund;
    COOLR.G0          = All.G0;
    COOLR.G0_LW       = All.G0;
    COOLR.cosmic_ray_ion_rate = All.CosmicRayIonRate;
    COOLR.cr_energy_density = 0.0;  /* [erg/cm^3], overwritten per-cell if COSMIC_RAY_FLUID active */
    COOLR.dust_to_gas_ratio = All.DGRnormalized;

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
        COOLR.redshift = 1.0 / All.Time - 1.0;
    } else {
        COOLR.redshift = 0.;
    }
    rho      = CellP[target].Density * All.cf_a3inv;
    timestep = dt;
    divv     = (CellP[target].Gradients.Velocity[0][0] + CellP[target].Gradients.Velocity[1][1] + CellP[target].Gradients.Velocity[2][2]) * All.cf_a2inv;
    if(All.ComovingIntegrationOn) divv += 3.0 * hubble_function(All.Time); /* Hubble flow contribution to physical divergence */

    /* Per-particle element abundances for cooling rates.  28-slot layout:
     *   Met[0]=total Z, [1]=H (explicit), [2]=He, [MET_OF(k)] = element k for k>=ELEM_C.
     *   Σ Met[1..27] = 1 by construction. */
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    {
        /* Number abundances n_X/n_H = (X_mass_frac / A_X) / X_H.
         * X_H MUST match the hydrogen reference used for yn (see ~line 425/428), else
         * n_X = abundX * yn is internally inconsistent:
         *   variable mode -> per-cell H mass fraction   (yn = rho * X_H_cell / m_p)
         *   fixed mode    -> 1/(1+4*ABHE)               (yn = rho / ((1+4*ABHE)*m_p))
         * Using per-cell H in fixed mode inflates metal number abundances by
         * X_H_fixed/X_H_cell in H-depleted enriched cells (spurious ~6x in the
         * 55 Zsun tail -> abundc ~132x instead of ~20x -> excess net-17 stiffness).
         * (Pre-fix this matched gadget_tnt_new, which gets away with per-cell H only
         *  because its mild enrichment keeps X_H_cell ~ 1/(1+4*ABHE).) */
#ifdef GALSF_CHEMCOOL_VARIABLE_XH_AND_ABHE
        double X_H = DMAX(P[target].Metallicity[MET_OF(ELEM_H)], 1e-10);
#else
        double X_H = 1.0 / (1.0 + 4.0 * ABHE);
#endif
#ifdef GALSF_RESOLVEDISM_DUST
        /* Gas-phase abundances: total metal minus metal locked in dust */
        double C_gas  = DMAX(P[target].Metallicity[MET_OF(ELEM_C)]  - CellP[target].Dust[0], 0);
        double O_gas  = DMAX(P[target].Metallicity[MET_OF(ELEM_O)]  - CellP[target].Dust[1], 0);
        double Si_gas = DMAX(P[target].Metallicity[MET_OF(ELEM_Si)] - CellP[target].Dust[3], 0);
        COOLR.abundc  = (C_gas  / 12.0) / X_H;
        COOLR.abundo  = (O_gas  / 16.0) / X_H;
        COOLR.abundsi = (Si_gas / 28.0) / X_H;
#else
        COOLR.abundc  = (P[target].Metallicity[MET_OF(ELEM_C)]  / 12.0) / X_H;
        COOLR.abundo  = (P[target].Metallicity[MET_OF(ELEM_O)]  / 16.0) / X_H;
        COOLR.abundsi = (P[target].Metallicity[MET_OF(ELEM_Si)] / 28.0) / X_H;
#endif
        COOLR.abundN  = (P[target].Metallicity[MET_OF(ELEM_N)]  / 14.0) / X_H;

#ifdef WSS_CIE_COOL
        /* Map Metallicity[28] -> Zmass[12] (gadget_tnt_new ordering) */
        COOLR.Zmass[0]  = P[target].Metallicity[MET_OF(ELEM_He)]; /* He */
        COOLR.Zmass[1]  = P[target].Metallicity[MET_OF(ELEM_C)];  /* C  */
        COOLR.Zmass[2]  = P[target].Metallicity[MET_OF(ELEM_Mg)]; /* Mg */
        COOLR.Zmass[3]  = P[target].Metallicity[MET_OF(ELEM_O)];  /* O  */
        COOLR.Zmass[4]  = P[target].Metallicity[MET_OF(ELEM_Fe)]; /* Fe */
        COOLR.Zmass[5]  = P[target].Metallicity[MET_OF(ELEM_Si)]; /* Si */
        COOLR.Zmass[6]  = P[target].Metallicity[MET_OF(ELEM_H)];  /* H  (stored explicitly at Met[1]) */
        COOLR.Zmass[7]  = P[target].Metallicity[MET_OF(ELEM_N)];  /* N  */
        COOLR.Zmass[8]  = P[target].Metallicity[MET_OF(ELEM_Ne)]; /* Ne */
        COOLR.Zmass[9]  = P[target].Metallicity[MET_OF(ELEM_S)];  /* S  */
        COOLR.Zmass[10] = P[target].Metallicity[MET_OF(ELEM_Ca)]; /* Ca */
        COOLR.Zmass[11] = P[target].Metallicity[MET_OF(ELEM_Zn)]; /* Zn */
#endif

        /* --- Chemistry-input metallicity cap ---------------------------------------
         * Decouple the chemistry/cooling solver from FB over-concentration. Clustered
         * single-star feedback piles several SNe of ejecta into individual cells,
         * spiking a handful (~tens of 1e7) to >>10 Zsun (max seen ~55 Zsun). The network
         * (esp. net-17 trace species) was never validated there and goes stiff -> wedge.
         * Cap ONLY the abundances handed to the solver at CHEM_ZMAX_SOLAR * solar;
         * the stored Metallicity[]/Dust[] are untouched (mass conservation & enrichment
         * bookkeeping unaffected). Uniform scale preserves abundance ratios (incl. CO=min(C,O)). */
        {
            const double Z_solar_chem = 0.0134;
            const double Zmax_chem    = 10.0 * Z_solar_chem;   /* CHEM_ZMAX_SOLAR = 10 */
            double Zcell = P[target].Metallicity[0];           /* total metal mass fraction */
            if(Zcell > Zmax_chem) {
                double s = Zmax_chem / Zcell;
                COOLR.abundc  *= s;
                COOLR.abundo  *= s;
                COOLR.abundsi *= s;
                COOLR.abundN  *= s;
#ifdef WSS_CIE_COOL
                COOLR.Zmass[1] *= s; COOLR.Zmass[2] *= s; COOLR.Zmass[3]  *= s;
                COOLR.Zmass[4] *= s; COOLR.Zmass[5] *= s; COOLR.Zmass[7]  *= s;
                COOLR.Zmass[8] *= s; COOLR.Zmass[9] *= s; COOLR.Zmass[10] *= s;
                COOLR.Zmass[11] *= s;   /* scale metals only; leave He(0) and H(6) */
#endif
            }
        }
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

#ifdef GALSF_RESOLVEDISM_DUST
    {
        /* Per-particle DGR from evolved dust fields, normalized to solar (DGR_solar ~ 0.01) */
        double DGR_actual = 0;
        for(int kd = 0; kd < NUM_RESOLVEDISM_DUST; kd++) DGR_actual += CellP[target].Dust[kd];
        COOLR.dust_to_gas_ratio = DGR_actual / 0.01;
    }
#else
    /* No evolved dust model: assume fixed dust-to-metal ratio, scale DGR with local metallicity.
       DGRnormalized is the dust-to-metal efficiency (1.0 = MW-like ~40% depletion).
       dust_to_gas_ratio is in solar units: 1.0 = DGR_solar ~ 0.01 */
#ifdef METALS
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    COOLR.dust_to_gas_ratio = All.DGRnormalized * P[target].Metallicity[0] / All.SolarAbundances[0];
#else
    COOLR.dust_to_gas_ratio = All.DGRnormalized * P[target].Metallicity[0] / All.SolarAbundances[0];
#endif
#else
    COOLR.dust_to_gas_ratio = All.DGRnormalized;
#endif
#endif
    /* --- Chemistry-input DGR cap (2026-07-04) -----------------------------------
     * Companion to the CHEM_ZMAX metal-abundance cap above, which protected
     * abundc/abundo/abundsi/abundN from FB over-concentration but FORGOT the DGR.
     * In the no-dust-model (production) branch DGR scales with the RAW cell Z:
     * post-SN/PISN ejecta cells hit Z~0.5-0.7 (unmixed ejecta) -> DGR ~ 40-50x
     * solar (absolute ~0.5!) -> the dust-coupled terms (h_gr, gas_dust, dust
     * cooling, all prop. to DGR) overstiffen -> the recurring net-17 DVODE
     * 'nstep exceeded' wedge class; metal diffusion then spreads the spike
     * neighborhood-wide.  Cap the SOLVER INPUT at 10x solar = exactly the domain
     * certified by the one-zone sweeps (Z<=10 Zsun, DGR<=10, 1080/1080 clean on
     * net5+net17).  Stored Metallicity[]/Dust[] untouched; the DGR~Z proxy is
     * unphysical in fresh ejecta anyway (no condensation yet). Applies to the
     * evolved-dust branch too (same certified-domain argument). */
    if(COOLR.dust_to_gas_ratio > 10.0) {COOLR.dust_to_gas_ratio = 10.0;}



#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
    /* Habing/ISRF energy densities [erg/cm^3] (Draine 1978, Mathis+ 1983).
       FUV (6-13.6 eV):  5.29e-14  (Habing 1968, used for tree-ray PE heating)
       LW  (11.2-13.6):  2.11e-14  (H2 photodissociation band, from Draine spectrum)
       NUV and OPT: computed on tree but not used for heating currently. */
    double u_Hab_FUV = 5.29e-14;
    double u_Hab_LW  = 2.11e-14;
    double fac_flux2energy = All.cf_a2inv / (4.*M_PI*C_LIGHT_CGS * pow(UNIT_LENGTH_IN_CGS, 2)); /* luminosity flux to energy density */

    double UV_flux_tot = 0.0;
    double LW_flux_tot = 0.0;
    double UV_flux_min_pix = 0.324e-2/NPIX / (fac_flux2energy / u_Hab_FUV);
    double LW_flux_min_pix = 0.1 * UV_flux_min_pix;
    for(i = 0; i < NPIX; i++) {
        CellP[target].UV_flux[i] = DMAX(CellP[target].UV_flux[i], UV_flux_min_pix);
        CellP[target].LW_flux[i] = DMAX(CellP[target].LW_flux[i], LW_flux_min_pix);
        UV_flux_tot += CellP[target].UV_flux[i];
        LW_flux_tot += CellP[target].LW_flux[i];
    }

    double G0_tot = UV_flux_tot * fac_flux2energy / u_Hab_FUV * All.G0;
    double G0_LW  = LW_flux_tot * fac_flux2energy / u_Hab_LW  * All.G0;

    /* For cosmological runs: add metagalactic FUV background floor from TREECOOL */
    if(All.ComovingIntegrationOn) {
        double uvb_gJH0 = 0;
        get_uvb_rates(&uvb_gJH0, NULL, NULL, NULL, NULL, NULL);
        double G0_uvb = uvb_gJH0 / 2.29e-10; /* convert Gamma_HI to G0 in Habing (FIRE convention) */
        if(G0_tot < G0_uvb) G0_tot = G0_uvb;
        if(G0_LW < 0.1 * G0_uvb) G0_LW = 0.1 * G0_uvb; /* LW ~ 10% of FUV for metagalactic spectrum */
    }

    COOLR.G0 = G0_tot;
    CellP[target].G0 = G0_tot;
    COOLR.G0_LW = G0_LW;
    CellP[target].G0_LW = G0_LW;
    COOLR.G0_dust = G0_tot;

#ifdef COSMIC_RAY_FLUID
    { /* compute zeta from local CR energy density (Brugaletta+ 2024, Cummings+ 2016) */
        double ecr_cgs = Get_CosmicRayEnergyDensity_cgs(target); /* [erg cm^-3] */
        double ecr_eVcm3 = ecr_cgs / 1.602e-12; /* convert to [eV cm^-3] */
        COOLR.cosmic_ray_ion_rate = 3e-17 * ecr_eVcm3;
        COOLR.cr_energy_density = ecr_cgs; /* [erg cm^-3], for hadronic+Coulomb heating in cool_func */
    }
#elif defined(CR_SCALE_WITH_G0)
    COOLR.cosmic_ray_ion_rate = DMAX(1e-21, DMIN(2e-16, (COOLR.G0 / 1.7) * All.CosmicRayIonRate));
#endif
    CellP[target].CR_ionization_rate = COOLR.cosmic_ray_ion_rate;
#endif /* GALSF_RESOLVEDISM_G0_VARIABLE */


#ifdef GALSF_RESOLVEDISM_G0_SCALE_SFR
    double g0 = All.FactorG0 * All.G0;
    if(g0 < 0.324e-2) g0 = 0.324e-2;
    COOLR.G0 = g0;
    COOLR.G0_LW = g0; /* no separate LW in SFR-scaled mode */
#if defined(COSMIC_RAY_FLUID)
    { /* CR fluid active: zeta from the LIVE local CR energy density (same normalization as the
         G0_VARIABLE and M1 branches: 3e-17 s^-1 per eV/cm^3, Brugaletta+24/Cummings+16) — the
         SFR-scaled subgrid guess must never override the transported CR field. */
      double ecr_cgs = Get_CosmicRayEnergyDensity_cgs(target); /* [erg cm^-3] */
      COOLR.cosmic_ray_ion_rate = 3e-17 * (ecr_cgs / 1.602e-12);
      COOLR.cr_energy_density = ecr_cgs;
    }
#elif defined(CR_SCALE_WITH_G0)
    COOLR.cosmic_ray_ion_rate = All.FactorG0 * All.CosmicRayIonRate;
#endif
#endif

    /* Default RT photoionization rates to zero */
    COOLR.rt_phot_HI = COOLR.rt_phot_HeI = COOLR.rt_phot_HeII = 0;
    COOLR.rt_heat_HI = COOLR.rt_heat_HeI = COOLR.rt_heat_HeII = 0;

#if defined(RADTRANSFER)
    { /* M1 RT active: compute G0, G0_LW, and photoionization rates from radiation field */
        double rho_code = CellP[target].Density * All.cf_a3inv;
        /* Habing normalizations for M1 RT bands [erg/cm^3].
           FUV = 8-13.6 eV (narrower than tree-ray 6-13.6), LW = 11.2-13.6 eV */
        double u_Hab_FUV_M1 = 5.03e-14, u_Hab_LW = 2.11e-14, u_Hab_NUV = 3.85e-14; /* NUV(3.4-8eV): std ISRF = 0.024 eV/cm^3 */

        /* Convert Rad_E_gamma (extensive: total energy per particle) to physical energy density (erg/cm^3):
           u_rad = Rad_E_gamma / V_i = Rad_E_gamma * Density / Mass  [code energy/volume]
           u_rad_cgs = u_rad * UNIT_PRESSURE_IN_CGS */
        double fac_to_cgs = (rho_code / P[target].Mass) * UNIT_PRESSURE_IN_CGS;

#if defined(RT_PHOTOELECTRIC)
        double u_PE_cgs  = CellP[target].Rad_E_gamma[RT_FREQ_BIN_PHOTOELECTRIC] * fac_to_cgs;
#else
        double u_PE_cgs  = 0;
#endif
#if defined(RT_LYMAN_WERNER)
        double u_LW_cgs  = CellP[target].Rad_E_gamma[RT_FREQ_BIN_LYMAN_WERNER] * fac_to_cgs;
#else
        double u_LW_cgs  = 0;
#endif
#if defined(RT_NUV)
        double u_NUV_cgs = CellP[target].Rad_E_gamma[RT_FREQ_BIN_NUV] * fac_to_cgs;
#else
        double u_NUV_cgs = 0;
#endif
        /* G0: FUV field (8-13.6 eV, already includes LW) in Habing units — drives the
           molecular photochemistry (H2/CO dissociation, C ionization).
           FLOOR (2026-07-04, Uli): use the SAME floor as the TREE_RAD/G0_VARIABLE path
           (0.324e-2 * All.G0 = 5.508e-3 for All.G0=1.7; see UV_flux_min_pix above), so
           M1 and subgrid share one far-field floor. RT_ISRF_BACKGROUND must be OFF —
           the diffuse field is represented by this floor, identically in both schemes. */
        double G0_floor_treerad = 0.324e-2 * All.G0;
        COOLR.G0    = DMAX(u_PE_cgs / u_Hab_FUV_M1, G0_floor_treerad);
        /* G0_LW: LW field (11.2-13.6 eV) in its own Habing units; TREE_RAD floors LW at 10% of the FUV floor */
        COOLR.G0_LW = DMAX(u_LW_cgs / u_Hab_LW, 0.1 * G0_floor_treerad);
        /* G0_dust: GRAIN photoelectric heating field (Bakes & Tielens gas PE heating in
           cool_func) = FUV(8-13.6) + NUV(3.4-8), each with its own Habing conversion.
           NUV photons above the grain work function eject grain photoelectrons -> heat gas.
           Optical (0.4-3.4 eV) is below the work function (no PE heating); its grain-heating
           role is carried by the M1 IR/dust solver, not here. */
        COOLR.G0_dust = DMAX(u_PE_cgs / u_Hab_FUV_M1 + u_NUV_cgs / u_Hab_NUV, G0_floor_treerad); /* same TREE_RAD floor (G0_dust=G0_tot there) */

        /* Store the M1 FUV/LW field back to the cell so it is OUTPUT and inspectable,
           mirroring the subgrid G0_VARIABLE path (CellP.G0/.G0_LW at lines 345/347).
           Without this the M1 G0/G0_LW were computed, fed to the chemistry, and thrown
           away — invisible in snapshots and impossible to validate against TREE_RAD
           (2026-07-04). Fields live under G0_VARIABLE||RADTRANSFER (cell_data.h). */
        CellP[target].G0    = COOLR.G0;
        CellP[target].G0_LW = COOLR.G0_LW;

#if defined(RT_CHEM_PHOTOION)
        /* Compute per-band photoionization rates [s^-1] and heating rates [erg s^-1 per atom]
           Gamma_X = c * (u_rad / E_photon) * sigma_X   where all quantities in CGS */
        double fac_sigma = UNIT_LENGTH_IN_CGS * UNIT_LENGTH_IN_CGS; /* convert code sigma to cm^2 */
        { /* HI band */
            double u_cgs = CellP[target].Rad_E_gamma[RT_FREQ_BIN_H0] * fac_to_cgs;
            double E_phot = rt_nu_eff_eV[RT_FREQ_BIN_H0] * ELECTRONVOLT_IN_ERGS;
            double sigma  = rt_ion_sigma_HI[RT_FREQ_BIN_H0] * fac_sigma;
            double Gamma  = C_LIGHT_CGS * (u_cgs / E_phot) * sigma;
            COOLR.rt_phot_HI = Gamma;
            COOLR.rt_heat_HI = Gamma * (rt_nu_eff_eV[RT_FREQ_BIN_H0] - 13.6) * ELECTRONVOLT_IN_ERGS;
        }
#if defined(RT_PHOTOION_MULTIFREQUENCY)
        { /* HeI band */
            double u_cgs = CellP[target].Rad_E_gamma[RT_FREQ_BIN_He0] * fac_to_cgs;
            double E_phot = rt_nu_eff_eV[RT_FREQ_BIN_He0] * ELECTRONVOLT_IN_ERGS;
            double sigma  = rt_ion_sigma_HeI[RT_FREQ_BIN_He0] * fac_sigma;
            double Gamma  = C_LIGHT_CGS * (u_cgs / E_phot) * sigma;
            COOLR.rt_phot_HeI = Gamma;
            COOLR.rt_heat_HeI = Gamma * (rt_nu_eff_eV[RT_FREQ_BIN_He0] - 24.6) * ELECTRONVOLT_IN_ERGS;
        }
        { /* HeII band */
            double u_cgs = CellP[target].Rad_E_gamma[RT_FREQ_BIN_He1] * fac_to_cgs;
            double E_phot = rt_nu_eff_eV[RT_FREQ_BIN_He1] * ELECTRONVOLT_IN_ERGS;
            double sigma  = rt_ion_sigma_HeII[RT_FREQ_BIN_He1] * fac_sigma;
            double Gamma  = C_LIGHT_CGS * (u_cgs / E_phot) * sigma;
            COOLR.rt_phot_HeII = Gamma;
            COOLR.rt_heat_HeII = Gamma * (rt_nu_eff_eV[RT_FREQ_BIN_He1] - 54.4) * ELECTRONVOLT_IN_ERGS;
        }
#endif /* RT_PHOTOION_MULTIFREQUENCY */
#endif /* RT_CHEM_PHOTOION */

        /* CR ionization in M1 mode: the GALSF_RESOLVEDISM_G0_VARIABLE block that
           normally couples the CR fluid to the chemistry is compiled OUT under
           RADTRANSFER (mutually exclusive), so it MUST be replicated here — otherwise
           the network silently falls back to the fixed default CosmicRayIonRate. */
#if defined(COSMIC_RAY_FLUID)
        { double ecr_cgs = Get_CosmicRayEnergyDensity_cgs(target); /* [erg cm^-3] */
          COOLR.cosmic_ray_ion_rate = 3e-17 * (ecr_cgs / 1.602e-12); /* zeta from e_CR [eV cm^-3] */
          COOLR.cr_energy_density = ecr_cgs; }
#elif defined(CR_SCALE_WITH_G0)
        COOLR.cosmic_ray_ion_rate = DMAX(1e-21, DMIN(2e-16, (COOLR.G0 / 1.7) * All.CosmicRayIonRate));
#endif
        CellP[target].CR_ionization_rate = COOLR.cosmic_ray_ion_rate;
    }
#endif /* RADTRANSFER */

    /* Cosmological UVB floor: applied AFTER RT and G0_VARIABLE so it acts as a minimum.
       Adds metagalactic photoionization + photoheating rates from TREECOOL (FG2020).
       Compatible with RT (additive) and G0_VARIABLE (floor). */
    if(All.ComovingIntegrationOn) {
        double uvb_gJH0=0, uvb_gJHe0=0, uvb_gJHep=0, uvb_epsH0=0, uvb_epsHe0=0, uvb_epsHep=0;
        get_uvb_rates(&uvb_gJH0, &uvb_gJHe0, &uvb_gJHep, &uvb_epsH0, &uvb_epsHe0, &uvb_epsHep);
        /* Photoionization: add UVB as floor (RT may have set higher local values) */
        if(COOLR.rt_phot_HI   < uvb_gJH0)  COOLR.rt_phot_HI   = uvb_gJH0;
        if(COOLR.rt_phot_HeI  < uvb_gJHe0) COOLR.rt_phot_HeI  = uvb_gJHe0;
        if(COOLR.rt_phot_HeII < uvb_gJHep) COOLR.rt_phot_HeII = uvb_gJHep;
        if(COOLR.rt_heat_HI   < uvb_epsH0)  COOLR.rt_heat_HI   = uvb_epsH0;
        if(COOLR.rt_heat_HeI  < uvb_epsHe0) COOLR.rt_heat_HeI  = uvb_epsHe0;
        if(COOLR.rt_heat_HeII < uvb_epsHep) COOLR.rt_heat_HeII = uvb_epsHep;
        COOLI.irad_heat = 1;
    }

    /* Set correct dust temperature in coolr common block */
#if defined(RT_INFRARED)
    /* M1 solver owns the dust temperature — feed it into CHEMCOOL so gas-dust
       coupling (R1) and H2 formation on dust use the transported value */
    COOLR.tdust = CellP[target].Dust_Temperature;
#else
    COOLR.tdust = CellP[target].DustTemp;
#endif

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
#ifdef GALSF_CHEMCOOL_VARIABLE_XH_AND_ABHE
    {
        double X_H_local  = DMAX(P[target].Metallicity[MET_OF(ELEM_H)],  1e-10);
        double X_He_local = DMAX(P[target].Metallicity[MET_OF(ELEM_He)], 0.0);
        CHEM_COMP.X_H_chem = X_H_local;
        CHEM_COMP.abhe     = X_He_local / (4.0 * X_H_local);
        yn = rho * X_H_local / PROTONMASS_CGS;
    }
#else
    yn = rho / ((1.0 + 4.0 * ABHE) * PROTONMASS_CGS); /* number density of hydrogen only */
#endif

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

    /* --- Hot-gas dust zero (2026-07-04, Uli) --------------------------------
     * The DGR~Z proxy is an ISM (high-column) relation and must not hand the
     * network dust in HOT gas: fresh SN/PISN ejecta has no condensed dust yet,
     * and in the hot phase grains are sputtered/sublimated away.  Zero the
     * chemistry-input DGR for T > 1e5 K (matches the network's own temp<1d5
     * dust-block gate).  NOTE deliberately a GAS-temperature threshold in the
     * sputtering regime, NOT T_sub~1.5e3 K: WNM at ~8e3 K carries COLD dust
     * (T_dust~15 K; sublimation is a dust-temperature criterion), and its
     * grain-assisted recombination must survive.  Below 1e5 K the 10x-MW
     * certified-domain cap above still applies.  Stored fields untouched. */
    if(temp > 1.0e5) {COOLR.dust_to_gas_ratio = 0;}


#ifdef TREE_RAD
    for(i = 0; i < NPIX; i++) {
        columni = CellP[target].Projection[i] * UNIT_DENSITY_IN_CGS * UNIT_LENGTH_IN_CGS * All.cf_a2inv;
#ifdef GALSF_CHEMCOOL_VARIABLE_XH_AND_ABHE
        NH = columni * CHEM_COMP.X_H_chem / PROTONMASS_CGS;
#else
        NH = columni / ((1.0 + 4.0 * ABHE) * PROTONMASS_CGS);
#endif
#if defined(RADTRANSFER)
        /* M1 RT: the radiation transport already applies dust attenuation to the FUV/LW
           bands, so the field handed to the chemistry (Rad_E_gamma -> G0) is ALREADY
           dust-attenuated. The total-H column drives ONLY the dust f_dust in calc_photo
           (iphoto=6, lines 304/307); zero it here so dust is not double-counted. The H2
           and CO self-shielding columns (populated below) are KEPT — M1 continuum
           transport does not capture line self-shielding, so the chemistry must still do it. */
        PROJECT.column_density_projection[i] = 0.0;
#else
        PROJECT.column_density_projection[i] = NH;
#endif
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
        PROJECT.fac_uv[i] = CellP[target].UV_flux[i] / UV_flux_tot;
#endif
    }
#endif

#ifdef TREE_RAD_H2
    for(i = 0; i < NPIX; i++) {
        columni = CellP[target].ProjectionH2[i] * UNIT_DENSITY_IN_CGS * UNIT_LENGTH_IN_CGS * All.cf_a2inv;
        NH2 = columni / (2.0 * PROTONMASS_CGS);
        PROJECT.column_density_projection_h2[i] = NH2;
    }

#if CHEMISTRYNETWORK != 1 && CHEMISTRYNETWORK != 4
    for(i = 0; i < NPIX; i++) {
        columni = CellP[target].ProjectionCO[i] * UNIT_DENSITY_IN_CGS * UNIT_LENGTH_IN_CGS * All.cf_a2inv;
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

    /* Clamp molecular abundances to zero in fully ionized gas (xH+ > 0.99).
     * H2/HD/CO cannot survive in ionized gas — forcing them to zero avoids
     * extreme stiffness in DVODE where destruction rates exceed formation
     * rates by >10 orders of magnitude, which can produce NaN.
     * Only needed for network 17 which tracks HD and has more stiff couplings. */
#if CHEMISTRYNETWORK == 17
    if(abundances[IHP] > 0.99 && skip_evolve_abundances == 0) {
        abundances[IH2] = 0.0;
        abundances[ICO] = 0.0;
        abundances[IHD] = 0.0;
    }
#endif

    if(mode == 1 || mode == 2) timestep = 0.0;

    /* Evolve abundances */
    if(skip_evolve_abundances == 0) {
        /* DEBUG: check if H+ is already >1 before entering the Fortran solver */
        if(abundances[IHP] > 1.0) {
            printf("CHEMCOOL_DEBUG: H+ > 1 ON ENTRY: ID=%llu H+=%.10e H2=%.10e CO=%.10e abundc=%.6e abundo=%.6e yn=%.4e\n",
                (unsigned long long)P[target].ID, abundances[IHP], abundances[IH2],
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6
                abundances[ICO],
#else
                0.0,
#endif
                COOLR.abundc, COOLR.abundo, yn);
        }
#ifdef GALSF_CHEMCOOL_DEBUG_INPUT
        /* Entry validation: catch unphysical compositions handed TO the solver,
         * i.e. species exceeding their elemental pools.  These arise upstream
         * (FB mass injection changes n_H without diluting per-H TracAbund
         * ratios; diffusion moves elements while ratios stay; dust locks
         * gas-phase metals) and are the root cause of DVODE wedging — catch
         * them here with full context instead of inside a 9e6-step retry loop. */
        {
            int viol = 0; double rtol = 1.0e-3;
            double co_pool = (COOLR.abundc < COOLR.abundo) ? COOLR.abundc : COOLR.abundo;
#if CHEMISTRYNETWORK != 4
            if(abundances[ICO] > co_pool * (1.0 + rtol) + 1e-20) viol |= 1;
#endif
            if(2.0*abundances[IH2] + abundances[IHP] > 1.0 + rtol) viol |= 2;
#if CHEMISTRYNETWORK == 1 || CHEMISTRYNETWORK == 17
            if(abundances[IHEP] + abundances[IHEPP] > CHEM_COMP.abhe * (1.0 + rtol) + 1e-20) viol |= 4;
            if(abundances[IDP] + abundances[IHD] > COOLR.abundD * (1.0 + rtol) + 1e-20) viol |= 8;
#endif
            if(viol) {
                int ionized_flag = -1;
#ifdef GALSF_RESOLVEDISM_PHOTOION
                ionized_flag = (int)CellP[target].Ionized;
#endif
                printf("CHEMCOOL_INPUT_VIOLATION: ID=%llu viol=%d yn=%.4e T=%.4e Ionized=%d\n",
                    (unsigned long long)P[target].ID, viol, yn, temp, ionized_flag);
                printf("CHEMCOOL_INPUT_VIOLATION: pools abundc=%.6e abundo=%.6e abhe=%.6e abundD=%.6e dgr=%.4e tdust=%.4e\n",
                    COOLR.abundc, COOLR.abundo, CHEM_COMP.abhe, COOLR.abundD,
                    COOLR.dust_to_gas_ratio, COOLR.tdust);
                printf("CHEMCOOL_INPUT_VIOLATION: y H2=%.6e H+=%.6e CO=%.6e",
                    abundances[IH2], abundances[IHP],
#if CHEMISTRYNETWORK != 4
                    abundances[ICO]
#else
                    0.0
#endif
                );
#if CHEMISTRYNETWORK == 1 || CHEMISTRYNETWORK == 17
                printf(" He+=%.6e He++=%.6e D+=%.6e HD=%.6e",
                    abundances[IHEP], abundances[IHEPP], abundances[IDP], abundances[IHD]);
#endif
                printf("\n");
                fflush(stdout);
            }
        }
#endif

        /* ---- Enforce physical species-vs-pool consistency on solver input ----
         * Molecular/ionic species cannot exceed the elemental reservoir they
         * draw from.  Upstream operations (FB mass injection changing n_H
         * without diluting per-H ratios; turbulent diffusion of elements but
         * not ratios; dust grain condensation locking gas-phase metals) can
         * leave a cell with e.g. CO > available gas-phase C, which the rate
         * equation cannot relax -> DVODE rejects every step and wedges (one
         * rank stop()s -> silent MPI hang).  Clamp here so the solver always
         * receives a consistent composition.  Amounts clamped are tiny
         * (trace species near their floors) so conservation is unaffected. */
#if CHEMISTRYNETWORK != 4
        {
            double co_pool = (COOLR.abundc < COOLR.abundo) ? COOLR.abundc : COOLR.abundo;
            if(co_pool < 0) co_pool = 0;
            if(abundances[ICO] > co_pool) abundances[ICO] = co_pool;  /* CO <= min(C_gas, O_gas) */
        }
#endif
#if CHEMISTRYNETWORK == 1 || CHEMISTRYNETWORK == 17
        {
            /* He+ + He++ <= abhe (He pool); D+ + HD <= abundD (D pool).
             * Rescale the pair down proportionally if the sum overshoots
             * (e.g. PI HII reset loads He+ exactly at abhe, then DVODE
             * tolerance drift tips the sum ~0.1% over -> check_abundances veto). */
            double abhe_pool = CHEM_COMP.abhe;
            double he_sum = abundances[IHEP] + abundances[IHEPP];
            if(he_sum > abhe_pool && he_sum > 0) {
                double f = abhe_pool / he_sum;
                abundances[IHEP]  *= f;
                abundances[IHEPP] *= f;
            }
            double d_sum = abundances[IDP] + abundances[IHD];
            if(d_sum > COOLR.abundD && d_sum > 0) {
                double f = COOLR.abundD / d_sum;
                abundances[IDP] *= f;
                abundances[IHD] *= f;
            }
        }
#endif
        EVOLVE_ABUNDANCES(&timestep, &dl, &yn, &divv, &energy, abundances, &column_est);
    }

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
#if defined(RADTRANSFER)
        /* Write the H ionization state + electron abundance from CHEMCOOL back to
         * the cell (2026-07-04). Two reasons, both M1-critical:
         *  (a) the M1 ionizing-band OPACITY reads CellP[].HI (rt_absorption_rate,
         *      rt_utilities.cc:151) -- if it stays fully-neutral, the ionizing
         *      photons are always absorbed at the neutral rate and the HII region
         *      can never break through (ionization capped, sn20_rmhd);
         *  (b) snapshots (NeutralHydrogenAbundance/ElectronAbundance) otherwise
         *      show stale defaults, disconnected from the chemistry.
         * Under subgrid GALSF_RESOLVEDISM_PHOTOION the PI module sets these; under
         * M1 nothing did, so CHEMCOOL (the chemistry solver) must. */
        {
            double abh2_c = abundances[IH2], abhp_c = abundances[IHP];
            double abHI_c = 1.0 - 2.0*abh2_c - abhp_c; if(abHI_c < 0) abHI_c = 0;
            CellP[target].HI  = abHI_c;                              /* neutral H fraction n_HI/n_H */
            CellP[target].HII = abhp_c;                             /* ionized H fraction n_HII/n_H */
#if CHEMISTRYNETWORK == 1 || CHEMISTRYNETWORK == 17
            /* nets with explicit He states: electrons include He+ and 2x He++ */
            CellP[target].Ne  = abhp_c + abundances[IHEP] + 2.0*abundances[IHEPP]
                                + COOLR.abundc + COOLR.abundsi;
#else
            CellP[target].Ne  = abhp_c + COOLR.abundc + COOLR.abundsi; /* electrons: H+ plus fixed metal pools */
#endif
        }
#endif
        CellP[target].InternalEnergy = energy / rho;
        CellP[target].InternalEnergyPred = CellP[target].InternalEnergy;
        CellP[target].Temp = temp;
        CellP[target].DustTemp = COOLR.tdust;
#if defined(GALSF_RESOLVEDISM_G0_VARIABLE) || defined(RADTRANSFER)
        /* Attenuated FUV field the chemistry actually used = G0 x f_dust (2026-07-04).
         * M1: f_dust=1 (dust column zeroed, transport did the attenuation) -> G0_atten=G0.
         * TREE_RAD: G0 is UNATTENUATED and f_dust=exp(-AV...) applies the dust attenuation,
         * so G0_atten is the effective FUV -> directly comparable to M1's G0. */
#ifdef OUTPUT_SHIELD_FAC
        CellP[target].ShieldFacDust = COOLR.fac_shield_dust;
        CellP[target].G0_atten      = CellP[target].G0 * COOLR.fac_shield_dust;
#else
        CellP[target].ShieldFacDust = 1.0;
        CellP[target].G0_atten      = CellP[target].G0;
#endif
#endif
#ifdef OUTPUT_COOLRATE
        CellP[target].CoolingRate_CHEMCOOL = cooling_rate;
#endif

#if defined(RADTRANSFER)
        /* --- Distribute CHEMCOOL cooling rates into RT bins ---
           Mirrors the standard cooling solver (cooling.cc ~lines 1295-1333).
           COOLR.lambda[0..27] contains individual rates in erg/s/cm^3 (= nH^2 * Lambda).
           Lambda_RadiativeCooling_toRHDBins uses the same nH^2-scaled convention. */
        {
            double nHcgs_loc = HYDROGEN_MASSFRAC * UNIT_DENSITY_IN_CGS * CellP[target].Density * All.cf_a3inv / PROTONMASS_CGS;
            double nH2_loc = nHcgs_loc * nHcgs_loc;

            /* Guard against NaN/Inf in individual CHEMCOOL cooling terms before they are
             * redistributed into RT bins (2026-07-04). A single NaN term (e.g. net-17
             * HeI/HeII / WSS_CIE_COOL tables returning NaN at some T) poisons Rad_Je[NUV]
             * -> ef energy NaN -> hydro Riemann NaN -> crash at step 1. Zero any bad term
             * (it did not contribute to the gas energy update either) and trace it once. */
            for(int jl = 0; jl < 28; jl++) {
                if(!isfinite(COOLR.lambda[jl])) {
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
                    static int lnan = 0;
                    if(lnan < 8) { lnan++;
                        printf("LAMNAN lambda[%d]=%g temp=%g nH=%g abh2=%g abe=%g\n",
                               jl, (double)COOLR.lambda[jl], temp, nHcgs_loc, abh2, abe); fflush(stdout); }
#endif
                    COOLR.lambda[jl] = 0.0;
                }
            }

            /* Molecular + fine-structure → IR:
               R2=H2, R4=H2O rot, R5=H2O vib, R6=H2O18 vib,
               R7=CO rot, R8=CO vib, R9=13CO vib, R10=OH, R13=OI, R14=CI,
               R15=SiI, R16=CII, R17=SiII, R18=Compton, R19=HD, R20=H2 CIE,
               R21=13CO rot, R22=CO18 rot, R23=H2O18 rot, R24=CO18 vib
               NOTE: R1 (gas-dust, index 0) excluded — rt_ir_lambdadust handles it.
               NOTE: R11 (CR heating, index 10) and R12 (PE heating, index 11) excluded —
               these are not radiative cooling, they are heating from CRs and the PE band. */
            double Lambda_IR = 0;
            int ir_idx[] = {1,3,4,5,6,7,8,9,12,13,14,15,16,17,18,19,20,21,22,23};
            for(int j=0; j<20; j++) {Lambda_IR += COOLR.lambda[ir_idx[j]];}

            /* Atomic/ionic excitation → NUV:
               R3=atomic (index 2), R25=Ly-alpha (24), R26=HeI (25), R27=HeII (26) */
            double Lambda_NUV = COOLR.lambda[2] + COOLR.lambda[24] + COOLR.lambda[25] + COOLR.lambda[26];

            /* Bremsstrahlung R28 (index 27): NUV if hot, IR if cold */
            if(temp >= 1e5) {Lambda_NUV += COOLR.lambda[27];} else {Lambda_IR += COOLR.lambda[27];}

#ifdef RT_INFRARED
            CellP[target].Lambda_RadiativeCooling_toRHDBins[RT_FREQ_BIN_INFRARED] = Lambda_IR / nH2_loc;
#endif
#ifdef RT_NUV
#ifdef RT_INFRARED
            /* If Radiation_Temperature > 1e4, redirect NUV → IR (same as standard solver).
               (Radiation_Temperature is an RT_INFRARED-only field; guard added 2026-07-04 —
               without the IR band there is no redirect target, NUV keeps its own cooling.) */
            if(CellP[target].Radiation_Temperature > 1.e4) {
                CellP[target].Lambda_RadiativeCooling_toRHDBins[RT_FREQ_BIN_INFRARED] += Lambda_NUV / nH2_loc;
                CellP[target].Lambda_RadiativeCooling_toRHDBins[RT_FREQ_BIN_NUV] = 0;
            } else {
                CellP[target].Lambda_RadiativeCooling_toRHDBins[RT_FREQ_BIN_NUV] = Lambda_NUV / nH2_loc;
            }
#else
            CellP[target].Lambda_RadiativeCooling_toRHDBins[RT_FREQ_BIN_NUV] = Lambda_NUV / nH2_loc;
#endif
#elif defined(RT_INFRARED)
            CellP[target].Lambda_RadiativeCooling_toRHDBins[RT_FREQ_BIN_INFRARED] += Lambda_NUV / nH2_loc;
#endif

#ifdef RT_INFRARED
            /* Solve for equilibrium dust temperature. rt_ir_lambdadust root-finds Tdust
               balancing: dust absorption (all bands) + gas-dust collisional exchange = dust emission.
               It updates Dust_Temperature and Lambda_RadiativeCooling_toRHDBins[IR].
               CHEMCOOL already applied gas-dust coupling (R1) to the gas energy, so we do NOT
               apply the returned LambdaDust to the gas — only use it for Tdust + IR bookkeeping. */
            rt_ir_lambdadust(target, temp);
            CellP[target].DustTemp = CellP[target].Dust_Temperature;
#endif
        }
#endif /* RADTRANSFER */

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
