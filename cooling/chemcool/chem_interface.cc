/* cooling/chemcool/chem_interface.cc
 *
 * GATE 0 of the CHEMCOOL C++/CVODE migration: the frozen chemistry seam.
 * See chem_interface.h for the contract.
 *
 * chem_fill_inputs():      ALL per-cell gathering logic, MOVED VERBATIM from
 *                          do_chemcool_step() in chemcool.cc.  Provenance
 *                          comments "[orig chemcool.cc:N]" refer to the
 *                          pre-refactor line numbers of chemcool.cc as of
 *                          2026-07-28 (branch uli-singlestar-galaxy, Gate-0
 *                          baseline).  Every numerical expression and its
 *                          order of operations is unchanged.
 * chem_commit_to_fortran(): the ONLY place the per-cell COMMON-block state is
 *                          written (COOLR/COOLI/CHEM_COMP/PROJECT).
 * chem_read_outputs():     read-back of the COMMON fields the solve updated.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "../../declarations/allvars.h"
#include "../../core/proto.h"
#include "chem_interface.h"
#include "../../declarations/resolvedism_dgr.h"
#include "../../galaxy_sf/resolvedism_selfcheck.h"

#ifdef CHEMCOOL

/* =========================================================================
 * chem_fill_inputs: gather the complete per-cell chemistry input state.
 * Caller must have seeded in->dt_code, in->dl_code, in->mode, in->u_code.
 * No COMMON block is written here.  The only Fortran call is CALC_TEMP,
 * a pure table lookup (init-time temp/gamma tables + explicit args only).
 * ========================================================================= */
void chem_fill_inputs(int target, struct ChemInputs *in)
{
    /* [orig chemcool.cc:155-170] local working variables (same roles) */
    double rho, timestep, divv, energy, ekn;
    double temp;
    double yn, abh2, abhd, abco, abe;
    double dl = in->dl_code;

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

    /* [orig chemcool.cc:173-177] cosmological redshift for the solver */
    if(All.ComovingIntegrationOn) {
        in->redshift = 1.0 / All.Time - 1.0;
    } else {
        in->redshift = 0.;
    }
    /* [orig chemcool.cc:178-181] density, timestep, physical velocity divergence */
    rho      = CellP[target].Density * All.cf_a3inv;
    timestep = in->dt_code;
    divv     = (CellP[target].Gradients.Velocity[0][0] + CellP[target].Gradients.Velocity[1][1] + CellP[target].Gradients.Velocity[2][2]) * All.cf_a2inv;
    if(All.ComovingIntegrationOn) divv += 3.0 * hubble_function(All.Time); /* Hubble flow contribution to physical divergence */
#if CHEMISTRYNETWORK == 20
    {   /* mean |dv_i/dx_i| (Hubble flow included) for the CO LVG column, physical cgs */
        double hf = All.ComovingIntegrationOn ? hubble_function(All.Time) : 0.0;
        double g = 0; int k;
        for(k = 0; k < 3; k++) {g += fabs(CellP[target].Gradients.Velocity[k][k] * All.cf_a2inv + hf);}
        in->dvdr_cgs = g / 3.0 * UNIT_VEL_IN_CGS / UNIT_LENGTH_IN_CGS;
    }
#endif

    /* Per-particle element abundances for cooling rates.  28-slot layout:
     *   Met[0]=total Z, [1]=H (explicit), [2]=He, [MET_OF(k)] = element k for k>=ELEM_C.
     *   Σ Met[1..27] = 1 by construction. */
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    /* [orig chemcool.cc:186-260] */
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
        /* [orig chemcool.cc:203-210] Gas-phase abundances: total metal minus metal locked in dust */
        double C_gas  = DMAX(P[target].Metallicity[MET_OF(ELEM_C)]  - CellP[target].Dust[0], 0);
        double O_gas  = DMAX(P[target].Metallicity[MET_OF(ELEM_O)]  - CellP[target].Dust[1], 0);
        double Si_gas = DMAX(P[target].Metallicity[MET_OF(ELEM_Si)] - CellP[target].Dust[3], 0);
        in->abundc  = (C_gas  / 12.0) / X_H;
        in->abundo  = (O_gas  / 16.0) / X_H;
        in->abundsi = (Si_gas / 28.0) / X_H;
#else
        /* [orig chemcool.cc:212-214] */
        in->abundc  = (P[target].Metallicity[MET_OF(ELEM_C)]  / 12.0) / X_H;
        in->abundo  = (P[target].Metallicity[MET_OF(ELEM_O)]  / 16.0) / X_H;
        in->abundsi = (P[target].Metallicity[MET_OF(ELEM_Si)] / 28.0) / X_H;
#endif
        in->abundN  = (P[target].Metallicity[MET_OF(ELEM_N)]  / 14.0) / X_H;   /* [orig chemcool.cc:216] */

#ifdef WSS_CIE_COOL
        /* [orig chemcool.cc:218-231] Map Metallicity[28] -> Zmass[12] (gadget_tnt_new ordering) */
        in->Zmass[0]  = P[target].Metallicity[MET_OF(ELEM_He)]; /* He */
        in->Zmass[1]  = P[target].Metallicity[MET_OF(ELEM_C)];  /* C  */
        in->Zmass[2]  = P[target].Metallicity[MET_OF(ELEM_Mg)]; /* Mg */
        in->Zmass[3]  = P[target].Metallicity[MET_OF(ELEM_O)];  /* O  */
        in->Zmass[4]  = P[target].Metallicity[MET_OF(ELEM_Fe)]; /* Fe */
        in->Zmass[5]  = P[target].Metallicity[MET_OF(ELEM_Si)]; /* Si */
        in->Zmass[6]  = P[target].Metallicity[MET_OF(ELEM_H)];  /* H  (stored explicitly at Met[1]) */
        in->Zmass[7]  = P[target].Metallicity[MET_OF(ELEM_N)];  /* N  */
        in->Zmass[8]  = P[target].Metallicity[MET_OF(ELEM_Ne)]; /* Ne */
        in->Zmass[9]  = P[target].Metallicity[MET_OF(ELEM_S)];  /* S  */
        in->Zmass[10] = P[target].Metallicity[MET_OF(ELEM_Ca)]; /* Ca */
        in->Zmass[11] = P[target].Metallicity[MET_OF(ELEM_Zn)]; /* Zn */
#endif

        /* --- Chemistry-input metallicity cap ---------------------------------------
         * [orig chemcool.cc:234-259]
         * Decouple the chemistry/cooling solver from FB over-concentration. Clustered
         * single-star feedback piles several SNe of ejecta into individual cells,
         * spiking a handful (~tens of 1e7) to >>10 Zsun (max seen ~55 Zsun). The network
         * (esp. net-17 trace species) was never validated there and goes stiff -> wedge.
         * Cap ONLY the abundances handed to the solver at CHEM_ZMAX_SOLAR * solar;
         * the stored Metallicity[]/Dust[] are untouched (mass conservation & enrichment
         * bookkeeping unaffected). Uniform scale preserves abundance ratios (incl. CO=min(C,O)). */
        {
            const double Z_solar_chem = All.SolarAbundances[0];
            const double Zmax_chem    = 10.0 * Z_solar_chem;   /* CHEM_ZMAX_SOLAR = 10 */
            double Zcell = P[target].Metallicity[0];           /* total metal mass fraction */
            if(Zcell > Zmax_chem) {
                double s = Zmax_chem / Zcell;
                in->abundc  *= s;
                in->abundo  *= s;
                in->abundsi *= s;
                in->abundN  *= s;
#ifdef WSS_CIE_COOL
                in->Zmass[1] *= s; in->Zmass[2] *= s; in->Zmass[3]  *= s;
                in->Zmass[4] *= s; in->Zmass[5] *= s; in->Zmass[7]  *= s;
                in->Zmass[8] *= s; in->Zmass[9] *= s; in->Zmass[10] *= s;
                in->Zmass[11] *= s;   /* scale metals only; leave He(0) and H(6) */
#endif
            }
            /* NL99 "M": depleted low-IP metals, CHEMCOOL_ABUNDM_SOLAR per H at solar Z, scaled with the (capped) cell Z */
            in->abundM = CHEMCOOL_ABUNDM_SOLAR * DMIN(Zcell, Zmax_chem) / Z_solar_chem;
#if CHEMISTRYNETWORK == 16
            in->abundM = in->abundsi; /* Gong+17: M is silicon */
#endif
        }
    }
#else
    /* [orig chemcool.cc:262-265] Fallback: WNM values (Sembach+ 2000) scaled by global metallicity */
    in->abundc  = All.InitialMetallicity * 1.4e-4;
    in->abundo  = All.InitialMetallicity * 3.2e-4;
    in->abundsi = All.InitialMetallicity * 1.5e-5;
    in->abundM  = All.InitialMetallicity * CHEMCOOL_ABUNDM_SOLAR;
#endif
    /* [orig chemcool.cc:267-270] */
    in->abundD  = All.DeutAbund;
    in->G0      = All.G0;
    in->G0_LW   = All.G0; /* default: LW = full FUV, overridden below if G0_VARIABLE */
    in->cosmic_ray_ion_rate = All.CosmicRayIonRate;

    in->id_current = P[target].ID;   /* [orig chemcool.cc:272] (same int truncation as COOLIC.id_current) */

#ifdef GALSF_RESOLVEDISM_DUST
    /* [orig chemcool.cc:274-280] */
    {
        /* Per-particle DGR from evolved dust fields, normalized to solar (DGR_solar ~ 0.01) */
        double DGR_actual = 0;
        for(int kd = 0; kd < NUM_RESOLVEDISM_DUST; kd++) DGR_actual += CellP[target].Dust[kd];
        in->dust_to_gas_ratio = DGR_actual / 0.01;
    }
#else
    /* [orig chemcool.cc:282-294]
       No evolved dust model: assume fixed dust-to-metal ratio, scale DGR with local metallicity.
       DGRnormalized is the dust-to-metal efficiency (1.0 = MW-like ~40% depletion).
       dust_to_gas_ratio is in solar units: 1.0 = DGR_solar ~ 0.01 */
#ifdef METALS
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    in->dust_to_gas_ratio = All.DGRnormalized * P[target].Metallicity[0] / All.SolarAbundances[0];
#else
    in->dust_to_gas_ratio = All.DGRnormalized * P[target].Metallicity[0] / All.SolarAbundances[0];
#endif
#else
    in->dust_to_gas_ratio = All.DGRnormalized;
#endif
#endif
    /* --- Chemistry-input DGR cap (2026-07-04) -----------------------------------
     * [orig chemcool.cc:295-308]
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
    if(in->dust_to_gas_ratio > 10.0) {in->dust_to_gas_ratio = 10.0;}



#ifdef GALSF_RESOLVEDISM_GHOSTD_G0_CR
    { /* GHOSTD baseline: uniform G0 and zeta scaled by the global young-star SFR; dynamic fields below override */
        double g0 = DMAX(All.FactorG0 * All.G0, 0.324e-2);
        in->G0 = g0; in->G0_LW = g0; in->G0_dust = g0;
#if defined(COSMIC_RAY_FLUID)
        double ecr_cgs = Get_CosmicRayEnergyDensity_cgs(target); /* transported CRs win over the SFR scaling */
        in->cosmic_ray_ion_rate = 3e-17 * (ecr_cgs / 1.602e-12);
        in->cr_energy_density = ecr_cgs;
#elif defined(CR_SCALE_WITH_G0)
        in->cosmic_ray_ion_rate = All.FactorG0 * All.CosmicRayIonRate;
#endif
    }
#endif

#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
    /* Habing energy densities [erg/cm^3]: FUV 6-13.6 eV 5.29e-14, LW 11.2-13.6 eV 2.11e-14 */
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

    in->G0 = G0_tot;
    CellP[target].G0 = G0_tot;
    in->G0_LW = G0_LW;
    CellP[target].G0_LW = G0_LW;
    in->G0_dust = G0_tot;

#ifdef COSMIC_RAY_FLUID
    { /* compute zeta from local CR energy density (Brugaletta+ 2024, Cummings+ 2016) */
        double ecr_cgs = Get_CosmicRayEnergyDensity_cgs(target); /* [erg cm^-3] */
        double ecr_eVcm3 = ecr_cgs / 1.602e-12; /* convert to [eV cm^-3] */
        in->cosmic_ray_ion_rate = 3e-17 * ecr_eVcm3;
        in->cr_energy_density = ecr_cgs; /* [erg cm^-3], for hadronic+Coulomb heating in cool_func */
    }
#elif defined(CR_SCALE_WITH_G0)
    in->cosmic_ray_ion_rate = DMAX(1e-21, DMIN(2e-16, (in->G0 / 1.7) * All.CosmicRayIonRate));
#endif
    CellP[target].CR_ionization_rate = in->cosmic_ray_ion_rate;
#endif /* GALSF_RESOLVEDISM_G0_VARIABLE */

    /* [orig chemcool.cc:382-384] Default RT photoionization rates to zero */
    in->rt_phot_HI = in->rt_phot_HeI = in->rt_phot_HeII = 0;
    in->rt_heat_HI = in->rt_heat_HeI = in->rt_heat_HeII = 0;
    /* Mirror of the init-time COOLIC.irad_heat default (chemcool_init sets 0; the
     * pre-refactor per-cell path only ever wrote it in the cosmological branch
     * below, so in non-cosmological runs the committed 0 == the value the COMMON
     * block held anyway — bit-identical). */
    in->irad_heat = 0;

#if defined(RADTRANSFER)
    /* [orig chemcool.cc:386-483] */
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
        in->G0    = DMAX(u_PE_cgs / u_Hab_FUV_M1, G0_floor_treerad);
        /* G0_LW: LW field (11.2-13.6 eV) in its own Habing units; TREE_RAD floors LW at 10% of the FUV floor */
        in->G0_LW = DMAX(u_LW_cgs / u_Hab_LW, 0.1 * G0_floor_treerad);
        /* G0_dust: GRAIN photoelectric heating field (Bakes & Tielens gas PE heating in
           cool_func) = FUV(8-13.6) + NUV(3.4-8), each with its own Habing conversion.
           NUV photons above the grain work function eject grain photoelectrons -> heat gas.
           Optical (0.4-3.4 eV) is below the work function (no PE heating); its grain-heating
           role is carried by the M1 IR/dust solver, not here. */
        in->G0_dust = DMAX(u_PE_cgs / u_Hab_FUV_M1 + u_NUV_cgs / u_Hab_NUV, G0_floor_treerad); /* same TREE_RAD floor (G0_dust=G0_tot there) */

        /* Store the M1 FUV/LW field back to the cell so it is OUTPUT and inspectable,
           mirroring the subgrid G0_VARIABLE path (CellP.G0/.G0_LW at orig lines 345/347).
           Without this the M1 G0/G0_LW were computed, fed to the chemistry, and thrown
           away — invisible in snapshots and impossible to validate against TREE_RAD
           (2026-07-04). Fields live under G0_VARIABLE||RADTRANSFER (cell_data.h). */
        CellP[target].G0    = in->G0;
        CellP[target].G0_LW = in->G0_LW;

#if defined(RT_CHEM_PHOTOION)
        /* Compute per-band photoionization rates [s^-1] and heating rates [erg s^-1 per atom]
           Gamma_X = c * (u_rad / E_photon) * sigma_X   where all quantities in CGS */
        double fac_sigma = UNIT_LENGTH_IN_CGS * UNIT_LENGTH_IN_CGS; /* convert code sigma to cm^2 */
        { /* HI band */
            double u_cgs = CellP[target].Rad_E_gamma[RT_FREQ_BIN_H0] * fac_to_cgs;
            double E_phot = rt_nu_eff_eV[RT_FREQ_BIN_H0] * ELECTRONVOLT_IN_ERGS;
            double sigma  = rt_ion_sigma_HI[RT_FREQ_BIN_H0] * fac_sigma;
            double Gamma  = C_LIGHT_CGS * (u_cgs / E_phot) * sigma;
            in->rt_phot_HI = Gamma;
            in->rt_heat_HI = Gamma * (rt_nu_eff_eV[RT_FREQ_BIN_H0] - 13.6) * ELECTRONVOLT_IN_ERGS;
        }
#if defined(RT_PHOTOION_MULTIFREQUENCY)
        { /* HeI band */
            double u_cgs = CellP[target].Rad_E_gamma[RT_FREQ_BIN_He0] * fac_to_cgs;
            double E_phot = rt_nu_eff_eV[RT_FREQ_BIN_He0] * ELECTRONVOLT_IN_ERGS;
            double sigma  = rt_ion_sigma_HeI[RT_FREQ_BIN_He0] * fac_sigma;
            double Gamma  = C_LIGHT_CGS * (u_cgs / E_phot) * sigma;
            in->rt_phot_HeI = Gamma;
            in->rt_heat_HeI = Gamma * (rt_nu_eff_eV[RT_FREQ_BIN_He0] - 24.6) * ELECTRONVOLT_IN_ERGS;
        }
        { /* HeII band */
            double u_cgs = CellP[target].Rad_E_gamma[RT_FREQ_BIN_He1] * fac_to_cgs;
            double E_phot = rt_nu_eff_eV[RT_FREQ_BIN_He1] * ELECTRONVOLT_IN_ERGS;
            double sigma  = rt_ion_sigma_HeII[RT_FREQ_BIN_He1] * fac_sigma;
            double Gamma  = C_LIGHT_CGS * (u_cgs / E_phot) * sigma;
            in->rt_phot_HeII = Gamma;
            in->rt_heat_HeII = Gamma * (rt_nu_eff_eV[RT_FREQ_BIN_He1] - 54.4) * ELECTRONVOLT_IN_ERGS;
        }
#endif /* RT_PHOTOION_MULTIFREQUENCY */
#endif /* RT_CHEM_PHOTOION */

        /* CR ionization in M1 mode: the GALSF_RESOLVEDISM_G0_VARIABLE block that
           normally couples the CR fluid to the chemistry is compiled OUT under
           RADTRANSFER (mutually exclusive), so it MUST be replicated here — otherwise
           the network silently falls back to the fixed default CosmicRayIonRate. */
#if defined(COSMIC_RAY_FLUID)
        { double ecr_cgs = Get_CosmicRayEnergyDensity_cgs(target); /* [erg cm^-3] */
          in->cosmic_ray_ion_rate = 3e-17 * (ecr_cgs / 1.602e-12); /* zeta from e_CR [eV cm^-3] */
          in->cr_energy_density = ecr_cgs; }
#elif defined(CR_SCALE_WITH_G0)
        in->cosmic_ray_ion_rate = DMAX(1e-21, DMIN(2e-16, (in->G0 / 1.7) * All.CosmicRayIonRate));
#endif
        CellP[target].CR_ionization_rate = in->cosmic_ray_ion_rate;
    }
#endif /* RADTRANSFER */

#ifdef TREE_RAY_PI
    /* [orig chemcool.cc:485-550]
     * TreeRay-PI: photoionization rates from the tree-transported ionizing bands.
     * Mirrors the M1 consumption above: Gamma_X = sigma_X * F_b / E_b per band,
     * summed over bands; heating = excess energy above threshold per ionization.
     * F_b = Sum_pix Ion_flux (stored as L[erg/s]/r_code^2) / (4pi * UNIT_LENGTH^2)
     * -> erg/s/cm^2 (a2inv for comoving). The cosmological UVB floor below still
     * applies on top, exactly as for M1. */
    {
        int b_pi, p_pi;
        double gHI = 0, hHI = 0;
#ifdef TREERAY_PI_HE
        double gHeI = 0, hHeI = 0, gHeII = 0, hHeII = 0;
#endif
        if(TRPI_bands_ready) {
            double fconv = All.cf_a2inv / (4.0 * M_PI * UNIT_LENGTH_IN_CGS * UNIT_LENGTH_IN_CGS);
            /* photon-conserving limiter (Abel & Wandelt 99): a tau>>1 cell may not ionize
             * more atoms than the photons entering it. The tree path has no source
             * depletion, so without this an over-ionized front cell strips its own
             * shielding and the front runs away (measured tau_cell ~ 400 at 1.4 pc). */
            double rho_c  = CellP[target].Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS;
            double L_c    = pow(P[target].Mass * UNIT_MASS_IN_CGS / rho_c, 1.0/3.0);
            double nH_c   = HYDROGEN_MASSFRAC * rho_c / PROTONMASS_CGS;
            double nHI_c  = DMAX(0.0, 1.0 - CellP[target].TracAbund[IHP]) * nH_c;
#ifdef TREERAY_PI_HE
            double nHeII_c  = CellP[target].TracAbund[IHEP]  * nH_c;
            double nHeIII_c = CellP[target].TracAbund[IHEPP] * nH_c;
            double nHe_c    = (1.0 - HYDROGEN_MASSFRAC - P[target].Metallicity[0]) * rho_c / (4.0 * PROTONMASS_CGS);
            double nHeI_c   = DMAX(0.0, nHe_c - nHeII_c - nHeIII_c);
#endif
            for(b_pi = 0; b_pi < N_TRPI_BANDS; b_pi++) {
                double F_b = 0; for(p_pi = 0; p_pi < NPIX; p_pi++) {F_b += CellP[target].Ion_flux[b_pi*NPIX + p_pi];}
                F_b *= fconv;                                   /* erg/s/cm^2 */
                if(F_b <= 0) continue;
                double tau_c = TRPI_sigma_HI[b_pi] * nHI_c * L_c;
#ifdef TREERAY_PI_HE
                tau_c += (TRPI_sigma_HeI[b_pi] * nHeI_c + TRPI_sigma_HeII[b_pi] * nHeII_c) * L_c;
#endif
                double fpc = (tau_c > 1.0e-4) ? (1.0 - exp(-tau_c)) / tau_c : 1.0;
                double E_b = TRPI_Eev[b_pi] * ELECTRONVOLT_IN_ERGS;
                double G_b = fpc * TRPI_sigma_HI[b_pi] * F_b / E_b;   /* s^-1 per HI atom */
                gHI += G_b; hHI += G_b * DMAX(TRPI_Eev[b_pi] - 13.6, 0.0) * ELECTRONVOLT_IN_ERGS;
#ifdef TREERAY_PI_HE
                double GheI = fpc * TRPI_sigma_HeI[b_pi] * F_b / E_b;
                gHeI += GheI; hHeI += GheI * DMAX(TRPI_Eev[b_pi] - 24.6, 0.0) * ELECTRONVOLT_IN_ERGS;
                double GheII = fpc * TRPI_sigma_HeII[b_pi] * F_b / E_b;
                gHeII += GheII; hHeII += GheII * DMAX(TRPI_Eev[b_pi] - 54.4, 0.0) * ELECTRONVOLT_IN_ERGS;
#endif
            }
        }
        in->rt_phot_HI = gHI; in->rt_heat_HI = hHI;
#ifdef TREERAY_PI_HE
        in->rt_phot_HeI = gHeI; in->rt_heat_HeI = hHeI;
        in->rt_phot_HeII = gHeII; in->rt_heat_HeII = hHeII;
#endif
#if defined(GALSF_RESOLVEDISM_ISOLATED_FB_TEST)
        /* single-cell time series: the definitive He ratchet-vs-reset probe */
        if(P[target].ID == 61225) {
            printf("TRPICELL t=%.6e mode=%d gHI=%.3e gHeI=%.3e xHP_in=%.6e xHeP_in=%.6e\n",
                   All.Time, in->mode, gHI,
#ifdef TREERAY_PI_HE
                   gHeI,
#else
                   0.0,
#endif
                   (double)CellP[target].TracAbund[IHP], (double)CellP[target].TracAbund[IHEP]); fflush(stdout);
        }
        static int trpidbg_n = 0;
        if(gHI > 1e-13 && trpidbg_n < 250 && (trpidbg_n++, 1)) {  /* cells INSIDE the ionized zone only */
            double F0=0,F1=0; for(p_pi=0;p_pi<NPIX;p_pi++){F0+=CellP[target].Ion_flux[0*NPIX+p_pi];
#if (N_TRPI_BANDS > 1)
                F1+=CellP[target].Ion_flux[1*NPIX+p_pi];
#endif
            }
            printf("TRPIDBG id=%llu pos=%.5f,%.5f,%.5f gHI=%.3e gHeI=%.3e F0=%.3e F1=%.3e ready=%d\n",
                   (unsigned long long)P[target].ID, P[target].Pos[0], P[target].Pos[1], P[target].Pos[2], gHI,
#ifdef TREERAY_PI_HE
                   gHeI,
#else
                   0.0,
#endif
                   F0, F1, TRPI_bands_ready); fflush(stdout);
        }
#endif
    }
#endif /* TREE_RAY_PI */

    /* [orig chemcool.cc:552-566]
       Cosmological UVB floor: applied AFTER RT and G0_VARIABLE so it acts as a minimum.
       Adds metagalactic photoionization + photoheating rates from TREECOOL (FG2020).
       Compatible with RT (additive) and G0_VARIABLE (floor). */
    if(All.ComovingIntegrationOn) {
        double uvb_gJH0=0, uvb_gJHe0=0, uvb_gJHep=0, uvb_epsH0=0, uvb_epsHe0=0, uvb_epsHep=0;
        get_uvb_rates(&uvb_gJH0, &uvb_gJHe0, &uvb_gJHep, &uvb_epsH0, &uvb_epsHe0, &uvb_epsHep);
        /* Photoionization: add UVB as floor (RT may have set higher local values) */
        if(in->rt_phot_HI   < uvb_gJH0)  in->rt_phot_HI   = uvb_gJH0;
        if(in->rt_phot_HeI  < uvb_gJHe0) in->rt_phot_HeI  = uvb_gJHe0;
        if(in->rt_phot_HeII < uvb_gJHep) in->rt_phot_HeII = uvb_gJHep;
        if(in->rt_heat_HI   < uvb_epsH0)  in->rt_heat_HI   = uvb_epsH0;
        if(in->rt_heat_HeI  < uvb_epsHe0) in->rt_heat_HeI  = uvb_epsHe0;
        if(in->rt_heat_HeII < uvb_epsHep) in->rt_heat_HeII = uvb_epsHep;
        in->irad_heat = 1;
    }

    /* [orig chemcool.cc:568-575] Set correct dust temperature in coolr common block */
#if defined(RT_INFRARED)
    /* M1 solver owns the dust temperature — feed it into CHEMCOOL so gas-dust
       coupling (R1) and H2 formation on dust use the transported value */
    in->tdust = CellP[target].Dust_Temperature;
#else
    in->tdust = CellP[target].DustTemp;
#endif

    /* [orig chemcool.cc:577-581]
       'energy' is internal energy density, NOT specific internal energy [in code units] */
    energy = rho * in->u_code;
    if(energy < rho * All.MinEgySpec) {
        energy = rho * All.MinEgySpec;
    }

    /* [orig chemcool.cc:583-591] Convert to cgs units */
    rho      *= UNIT_DENSITY_IN_CGS;
    timestep *= UNIT_TIME_IN_CGS;
    energy   *= UNIT_ENERGY_IN_CGS / pow(UNIT_LENGTH_IN_CGS, 3);
    dl       *= UNIT_LENGTH_IN_CGS;
    divv     *= UNIT_VEL_IN_CGS / UNIT_LENGTH_IN_CGS;
    for(i = 0; i < TRAC_NUM; i++) {
        in->abundances[i] = CellP[target].TracAbund[i];
    }
#ifdef GALSF_CHEMCOOL_VARIABLE_XH_AND_ABHE
    /* [orig chemcool.cc:592-599] */
    {
        double X_H_local  = DMAX(P[target].Metallicity[MET_OF(ELEM_H)],  1e-10);
        double X_He_local = DMAX(P[target].Metallicity[MET_OF(ELEM_He)], 0.0);
        in->X_H_chem = X_H_local;
        in->abhe     = X_He_local / (4.0 * X_H_local);
        yn = rho * X_H_local / PROTONMASS_CGS;
    }
#else
    /* [orig chemcool.cc:601] fixed composition: CHEM_COMP is Fortran-owned
     * (coolinmo default from the ABHE macro).  Mirror it READ-ONLY into the
     * struct for the He/D pool clamps in do_chemcool_step; commit does NOT
     * write it back in this configuration. */
    in->abhe     = CHEM_COMP.abhe;
    in->X_H_chem = CHEM_COMP.X_H_chem;
    yn = rho / ((1.0 + 4.0 * ABHE) * PROTONMASS_CGS); /* number density of hydrogen only */
#endif

    /* [orig chemcool.cc:604-618]  NOTE: rpar[3..6] are deliberately left
     * unset, exactly as the pre-refactor stack array was — with iphoto=6 the
     * Fortran loads rpar(5) into column_est/NH2 locals but never uses them
     * (calc_photo.F:140,391; only iphoto=4/5 or H2_FORM_TEST consume it). */
    in->rpar[0] = yn;
    in->rpar[1] = dl;
    in->rpar[2] = divv;
    abh2 = in->abundances[IH2];
#if CHEMISTRYNETWORK == 1 || CHEMISTRYNETWORK == 17
    abhd = in->abundances[IHD];
#else
    abhd = 0.0;
#endif
#if CHEMISTRYNETWORK != 4
    abco = in->abundances[ICO];
#else
    abco = 0.0;
#endif
    abe = in->abundances[IHP];

    /* [orig chemcool.cc:620-621] entry temperature (CALC_TEMP is a pure
     * lookup: init-time temp/gamma tables + explicit arguments only) */
    ekn = energy / (BOLTZMANN_CGS * (1.0 + ABHE - abh2 + abe) * yn);
#ifdef CHEMCOOL_CXX
    temp = chem_cxx_calc_temp(abh2, in->abhe, ekn);  /* this cell's abhe (Fortran reads the last committed one) */
#else
    CALC_TEMP(&abh2, &ekn, &temp);
#endif
    /* (the mode==2 early return [orig chemcool.cc:622] is taken by
     *  do_chemcool_step right after this function returns; the remaining
     *  gathering below only fills the struct — no observable state — so
     *  computing it unconditionally is bit-identical) */

    /* --- Hot-gas dust zero (2026-07-04, Uli) --------------------------------
     * [orig chemcool.cc:624-634]
     * The DGR~Z proxy is an ISM (high-column) relation and must not hand the
     * network dust in HOT gas: fresh SN/PISN ejecta has no condensed dust yet,
     * and in the hot phase grains are sputtered/sublimated away.  Zero the
     * chemistry-input DGR for T > 1e5 K (matches the network's own temp<1d5
     * dust-block gate).  NOTE deliberately a GAS-temperature threshold in the
     * sputtering regime, NOT T_sub~1.5e3 K: WNM at ~8e3 K carries COLD dust
     * (T_dust~15 K; sublimation is a dust-temperature criterion), and its
     * grain-assisted recombination must survive.  Below 1e5 K the 10x-MW
     * certified-domain cap above still applies.  Stored fields untouched. */
    if(temp > 1.0e5) {in->dust_to_gas_ratio = 0;}
    STELLA_CHECK("dgr_consistent", !(temp > 1.0e5 && CellP[target].Temp > 1.0e5) || resolvedism_cell_DGR_solar(target) == 0,
                 "ID=%llu T=%g Temp_stored=%g DGR_shared=%g", (unsigned long long)P[target].ID, temp, (double)CellP[target].Temp, resolvedism_cell_DGR_solar(target));
#ifdef STELLA_SELFCHECK
    if(temp > 1.0e5 && CellP[target].Temp > 1.0e5) {static int n_dgrchk = 0; if(++n_dgrchk == 1 || n_dgrchk % 1000 == 0) STELLA_CHECK_RAN("dgr_consistent", 1);}
#endif


#ifdef TREE_RAD
    /* [orig chemcool.cc:637-684] */
    for(i = 0; i < NPIX; i++) {
        columni = CellP[target].Projection[i] * UNIT_DENSITY_IN_CGS * UNIT_LENGTH_IN_CGS * All.cf_a2inv;
#ifdef GALSF_CHEMCOOL_VARIABLE_XH_AND_ABHE
        NH = columni * in->X_H_chem / PROTONMASS_CGS;
#else
        NH = columni / ((1.0 + 4.0 * ABHE) * PROTONMASS_CGS);
#endif
#if defined(RADTRANSFER)
        /* M1 RT: the radiation transport already applies dust attenuation to the FUV/LW
           bands, so the field handed to the chemistry (Rad_E_gamma -> G0) is ALREADY
           dust-attenuated. The total-H column drives ONLY the dust f_dust in calc_photo
           (iphoto=6, lines 304/307); zero it here so dust is not double-counted. The H2
           and CO self-shielding columns (populated below) are KEPT — M1 continuum
           transport does not capture line self-shielding, so the chemistry must still do it.
           NOTE (TREE_RAY lesson, 2026-07-07): TreeRay does NOT zero these columns — it
           overrides only the per-rate dust factors inside calc_photo (f_dust/f_simp/df25/
           df30 -> 1) and KEEPS chi_isrf(AV): the tree transports stellar sources only,
           so the ISRF background must stay target-side attenuated. Zeroing the columns
           under TREE_RAY broke net-17 chemistry from step zero. */
        in->column_density_projection[i] = 0.0;
#else
        /* VARIABLE-DGR FIX (Uli 2026-07-28): calc_photo computes AV = conv x
         * dust_to_gas_ratio x N -- but the LOCAL DGR cannot multiply the PATH
         * column when DGR varies along the ray (it stays inside the integral).
         * The tree now accumulates the DGR-weighted column (ProjectionDust,
         * solar-DGR units, same per-cell DGR recipe as this file via
         * resolvedism_cell_DGR_solar); passing N_eff = N_dust/DGR_local makes
         * the Fortran product exactly conv x N_dust. When the T>1e5 sputter
         * gate (or zero dust) disables dust terms, the value is inert. */
        if(in->dust_to_gas_ratio > 1.0e-10) {
            double coldust_ = CellP[target].ProjectionDust[i] * UNIT_DENSITY_IN_CGS * UNIT_LENGTH_IN_CGS * All.cf_a2inv;
            double NH_dust_;
#ifdef GALSF_CHEMCOOL_VARIABLE_XH_AND_ABHE
            NH_dust_ = coldust_ * in->X_H_chem / PROTONMASS_CGS;
#else
            NH_dust_ = coldust_ / ((1.0 + 4.0 * ABHE) * PROTONMASS_CGS);
#endif
            in->column_density_projection[i] = NH_dust_ / in->dust_to_gas_ratio;
        } else {
            in->column_density_projection[i] = NH;
        }
#endif
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
        in->fac_uv[i] = CellP[target].UV_flux[i] / UV_flux_tot;
#endif
    }
#endif

#if defined(TREE_RAD_H2) && !defined(TREE_RAD)
    /* [orig chemcool.cc:686-693]
       M1 (RADTRANSFER) + standalone H2/CO line self-shielding: there is NO total-H
       TreeCol column (TREE_RAD off -- the FUV-G0 dust walk is M1's job, and M1 already
       dust-attenuated the continuum). Zero the dust column so calc_photo's f_dust and
       the df* dust factors compute at AV=0 (no double-counting). The H2/CO line-shield
       columns are populated below and DO drive the target-side self-shielding. */
    for(i = 0; i < NPIX; i++) { in->column_density_projection[i] = 0.0; }
#endif

#ifdef TREE_RAD_H2
    /* [orig chemcool.cc:695-713] */
    for(i = 0; i < NPIX; i++) {
        columni = CellP[target].ProjectionH2[i] * UNIT_DENSITY_IN_CGS * UNIT_LENGTH_IN_CGS * All.cf_a2inv;
        NH2 = columni / (2.0 * PROTONMASS_CGS);
        in->column_density_projection_h2[i] = NH2;
    }

#if CHEMISTRYNETWORK != 1 && CHEMISTRYNETWORK != 4
    for(i = 0; i < NPIX; i++) {
        columni = CellP[target].ProjectionCO[i] * UNIT_DENSITY_IN_CGS * UNIT_LENGTH_IN_CGS * All.cf_a2inv;
        NCO = columni / (28.0 * PROTONMASS_CGS);
        in->column_density_projection_co[i] = NCO;
    }
#else
    for(i = 0; i < NPIX; i++) {
        in->column_density_projection_co[i] = 0.0;
    }
#endif
#endif /* TREE_RAD_H2 */


    /* [orig chemcool.cc:716-721] Switch off chemistry for high-density particles */
    in->no_chem = 0;

    in->column_est = 0.0;

    in->pdv_term = 0.;

    /* ---- store the remaining per-cell scalars into the seam struct ---- */
    in->rho_cgs      = rho;
    in->timestep_cgs = timestep;
    in->divv_cgs     = divv;
    in->energy_cgs   = energy;
    in->dl_cgs       = dl;
    in->yn           = yn;
    in->ekn          = ekn;
    in->temp         = temp;
    in->abh2         = abh2;
    in->abhd         = abhd;
    in->abco         = abco;
    in->abe          = abe;
}


/* =========================================================================
 * chem_commit_to_fortran: the ONLY writer of per-cell COMMON-block state.
 * Field-by-field copy of the final values computed by chem_fill_inputs;
 * the compile guards reproduce EXACTLY the set of COMMON members the
 * pre-refactor per-cell path wrote under each configuration (fields the
 * original never wrote — e.g. CHEM_COMP without VARIABLE_XH_AND_ABHE, or
 * G0_dust without G0_VARIABLE/RADTRANSFER — stay Fortran-owned).
 * ========================================================================= */
/* per-cell COMMON values after chemcool_init; every cell starts from them on every thread (no copyin needed) */
static struct chem_coolrc_t chem_coolrc_init;
static struct chem_coolic_t chem_coolic_init;

void chem_percell_snapshot(void) {chem_coolrc_init = COOLRC; chem_coolic_init = COOLIC;}

void chem_commit_to_fortran(const struct ChemInputs *in)
{
    int i;

    COOLRC = chem_coolrc_init;
    COOLIC = chem_coolic_init;

    COOLRC.redshift = in->redshift;                       /* [orig chemcool.cc:174/176] */

    COOLRC.abundc  = in->abundc;                          /* [orig chemcool.cc:208-214/263-265] */
    COOLRC.abundo  = in->abundo;
    COOLRC.abundsi = in->abundsi;
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    COOLRC.abundN  = in->abundN;                          /* [orig chemcool.cc:216] */
#ifdef WSS_CIE_COOL
    for(i = 0; i < 12; i++) {COOLRC.Zmass[i] = in->Zmass[i];}  /* [orig chemcool.cc:220-231,253-256] */
#endif
#endif
    COOLRC.abundD = in->abundD;                           /* [orig chemcool.cc:267] */
    COOLRC.abundM = in->abundM;

    COOLRC.G0    = in->G0;
    COOLRC.G0_LW = in->G0_LW;
#if defined(GALSF_RESOLVEDISM_G0_VARIABLE) || defined(GALSF_RESOLVEDISM_GHOSTD_G0_CR) || defined(RADTRANSFER)
    COOLRC.G0_dust = in->G0_dust;
#endif

    COOLRC.cosmic_ray_ion_rate = in->cosmic_ray_ion_rate;
#if defined(COSMIC_RAY_FLUID) && (defined(GALSF_RESOLVEDISM_G0_VARIABLE) || defined(GALSF_RESOLVEDISM_GHOSTD_G0_CR) || defined(RADTRANSFER))
    COOLRC.cr_energy_density = in->cr_energy_density;
#endif

    COOLRC.rt_phot_HI   = in->rt_phot_HI;                 /* [orig chemcool.cc:383-384,447-465,515-518,559-564] */
    COOLRC.rt_phot_HeI  = in->rt_phot_HeI;
    COOLRC.rt_phot_HeII = in->rt_phot_HeII;
    COOLRC.rt_heat_HI   = in->rt_heat_HI;
    COOLRC.rt_heat_HeI  = in->rt_heat_HeI;
    COOLRC.rt_heat_HeII = in->rt_heat_HeII;

    COOLRC.dust_to_gas_ratio = in->dust_to_gas_ratio;     /* [orig chemcool.cc:279-308,634] */
    COOLRC.tdust             = in->tdust;                 /* [orig chemcool.cc:572/574] */
    COOLRC.pdv_term          = in->pdv_term;              /* [orig chemcool.cc:721] */

    COOLIC.id_current = in->id_current;                   /* [orig chemcool.cc:272] */
    COOLIC.no_chem    = in->no_chem;                      /* [orig chemcool.cc:717] */
    COOLIC.irad_heat  = in->irad_heat;                    /* [orig chemcool.cc:565 + init default] */

#ifdef GALSF_CHEMCOOL_VARIABLE_XH_AND_ABHE
    CHEM_COMP.X_H_chem = in->X_H_chem;                   /* [orig chemcool.cc:596-597] */
    CHEM_COMP.abhe     = in->abhe;
#endif

#if defined(TREE_RAD) || defined(TREE_RAD_H2)
    for(i = 0; i < NPIX; i++) {PROJECT.column_density_projection[i] = in->column_density_projection[i];}  /* [orig chemcool.cc:657/675/677/692] */
#if defined(TREE_RAD) && defined(GALSF_RESOLVEDISM_G0_VARIABLE)
    for(i = 0; i < NPIX; i++) {PROJECT.fac_uv[i] = in->fac_uv[i];}                                        /* [orig chemcool.cc:681] */
#endif
#endif
#ifdef TREE_RAD_H2
    for(i = 0; i < NPIX; i++) {PROJECT.column_density_projection_h2[i] = in->column_density_projection_h2[i];}  /* [orig chemcool.cc:699] */
    for(i = 0; i < NPIX; i++) {PROJECT.column_density_projection_co[i] = in->column_density_projection_co[i];}  /* [orig chemcool.cc:706/710] */
#endif
}


/* =========================================================================
 * chem_read_outputs: read back the COMMON fields updated by the Fortran
 * solve (CALC_PHOTO_WRAPPER / EVOLVE_ABUNDANCES).  energy + abundances come
 * back through the EVOLVE_ABUNDANCES argument list, not through here.
 * ========================================================================= */
void chem_read_outputs(struct ChemOutputs *out)
{
    int i;
    out->tdust = COOLRC.tdust;                            /* [orig chemcool.cc:924] */
    for(i = 0; i < 28; i++) {out->lambda[i] = COOLRC.lambda[i];}                  /* [orig chemcool.cc:869-871,956-985] */
    for(i = 0; i < NRATES_CHEM; i++) {out->lambda_chem[i] = COOLRC.lambda_chem[i];}  /* [orig chemcool.cc:875-877] */
#ifdef OUTPUT_SHIELD_FAC
    out->fac_shield_h2   = COOLRC.fac_shield_h2;
    out->fac_shield_dust = COOLRC.fac_shield_dust;        /* [orig chemcool.cc:931] */
#endif
}


#ifdef GALSF_RESOLVEDISM_GHOSTD_G0_CR
/* GHOSTD: FactorG0 = SFR over the last 30 Myr / (Mgas0 / 4 Gyr) */
void ghostd_update_factor_g0(void)
{
    const double t_window_Gyr = 0.03, t_depletion_Gyr = 4.0;
    double m_young = 0, m_young_tot = 0;
    for(int i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 4 || P[i].Mass <= 0) {continue;}
        double m_birth = P[i].Mass;
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
        if(P[i].sampled != 1) {continue;} /* only finalized IMF-sampled stars */
        if(P[i].MstarSampleIMF[4] > 0) {m_birth = P[i].MstarSampleIMF[4];}
        else if(P[i].MstarSampleIMF[0] > 0) {m_birth = P[i].MstarSampleIMF[0] / UNIT_MASS_IN_SOLAR;} /* stars read without a birth-mass slot */
#endif
        if(evaluate_stellar_age_Gyr(i) < t_window_Gyr) {m_young += m_birth;}
    }
    MPI_Allreduce(&m_young, &m_young_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    All.FactorG0 = (All.Mgas0 > 0) ? (m_young_tot / t_window_Gyr) / (All.Mgas0 / t_depletion_Gyr) : 0;
    if(ThisTask == 0) {printf("GHOSTD: Time=%g M_young=%g Msun Mgas0=%g Msun FactorG0=%g\n", All.Time, m_young_tot * UNIT_MASS_IN_SOLAR, All.Mgas0 * UNIT_MASS_IN_SOLAR, All.FactorG0);}
}

/* GHOSTD: reference gas mass from the initial conditions, then the initial FactorG0 */
void ghostd_set_reference_gas_mass(void)
{
    double m_gas = 0;
    for(int i = 0; i < NumPart; i++) {if(P[i].Type == 0) {m_gas += P[i].Mass;}}
    MPI_Allreduce(&m_gas, &All.Mgas0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    ghostd_update_factor_g0();
}
#endif

#endif /* CHEMCOOL */
