#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "../../declarations/allvars.h"
#include "../../core/proto.h"
#include "f2c.h"
#include "chemcool_consts.h"
/* GATE 0 seam (2026-07-28): the Fortran COMMON-block externs, the typed
 * per-cell ChemInputs/ChemOutputs structs, and the fill/commit/read functions
 * now live in chem_interface.h/.cc.  chem_commit_to_fortran() is the ONLY
 * per-cell writer of COOLR/COOLI/CHEM_COMP/PROJECT; chemcool_init() below
 * retains the one-time init-time defaults. */
#include "chem_interface.h"
#include "chemcool_network.h"

#ifdef CHEMCOOL


/* Initialization of chemcool */
/* Target of the Fortran ABORT() macro (cool.h): clean, LOUD abort.
 * A bare Fortran 'stop' killed one rank silently and deadlocked the rest in
 * the next MPI collective (zombie jobs; pisn200 2026-07-04).  endrun() ->
 * MPI_Abort kills the WHOLE job immediately and visibly. */
extern "C" void chemcool_fatal_(int *code)
{
    printf("CHEMCOOL FATAL: solver failure code %d (particle id_current=%d, task=%d) -- clean abort.\n",
           *code, COOLIC.id_current, ThisTask);
    fflush(stdout);
    endrun(78100 + *code);
}

void chemcool_init(void)
{
    if(ThisTask == 0) {
        printf("initialize SG cooling and chemistry...\n");
        fflush(stdout);
    }

    /* Set COOLI flags before COOLINMO/CHEMINMO (they read these via common block); rate options as SGChem */
    COOLI.iflag_mn   = 4;  /* H- + H+ mutual neutralization (CDG99) */
    COOLI.iflag_ad   = 6;  /* H- + H associative detachment */
    COOLI.iflag_atom = 2;  /* SD93 combined atomic cooling table */
#if CHEMISTRYNETWORK == 1
    COOLI.iflag_3bh2a = 6; /* H2+H: three-body formation (Forrey 2013) */
#else
    COOLI.iflag_3bh2a = 5; /* H2+H: high-density CID (MSM96), three-body formation (G07) */
#endif
    COOLI.iflag_3bh2b = 4; /* H2+H2: one-eighth of the H2+H rates */
    COOLI.iflag_h3pra = 1; /* H3+ dissociative recombination (GP98) */
    COOLI.iphoto     = 6;  /* TREECOL column densities (overridden per-particle) */
    COOLIC.irad_heat  = 0;  /* no external radiative heating (handled via G0) */
    COOLI.isrf_option = 1; /* Draine (1978) ISRF spectral shape */
    COOLIC.no_chem    = 0;

    /* Set COOLR defaults before COOLINMO (matching gadget_tnt_new standard values) */
    COOLR.h2_form_ex  = 0.84;       /* fraction of H2 binding energy → vibrational excitation */
    COOLR.h2_form_kin = 0.12;       /* fraction → kinetic energy of H2 molecule */
    COOLR.deff        = 1.0;        /* H2 formation on dust efficiency factor */
    COOLR.phi_pah     = 0.5;        /* PAH charging parameter in the PE and grain-recombination fits */
    COOLR.AV_conversion_factor = 5.348e-22; /* N_H to A_V conversion (Bohlin+ 1978) */
    COOLR.AV_ext      = 0.0;        /* no external dust extinction */
    COOLR.dm_density  = 0.0;        /* no dark matter annihilation heating */
    COOLRC.pdv_term    = 0.0;
    COOLRC.redshift    = 0.0;        /* isolated galaxy, non-cosmological */
    COOLRC.abundD      = All.DeutAbund;
    COOLRC.G0          = All.G0;
    COOLRC.G0_LW       = All.G0;
    COOLRC.cosmic_ray_ion_rate = All.CosmicRayIonRate;
    COOLRC.cr_energy_density = 0.0;  /* [erg/cm^3], overwritten per-cell if COSMIC_RAY_FLUID active */
    COOLRC.dust_to_gas_ratio = All.DGRnormalized;

    COOLINMO();
    CHEMINMO();
    INIT_TOLERANCES();
    LOAD_H2_TABLE();
    INIT_TEMPERATURE_LOOKUP();
#ifdef CHEMCOOL_CXX
    chem_cxx_tables_init();
#endif
    chem_percell_snapshot(); /* per-thread COMMON copies start every cell from these values */

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
static double chemcool_step_core(int target, double dt, double dl, int mode, double u_code, int write_energy);

/* legacy entry: starts from CellP.InternalEnergy and writes the result back */
double do_chemcool_step(int target, double dt, double dl, int mode)
{
    return chemcool_step_core(target, dt, dl, mode, CellP[target].InternalEnergy, 1);
}

/* DoCooling backend: starts from u_old, returns the new specific energy; the caller owns InternalEnergy */
double chemcool_cool_cell(int target, double u_old, double dt)
{
    return chemcool_step_core(target, dt, 0, 0, u_old, 0);
}

static double chemcool_step_core(int target, double dt, double dl, int mode, double u_code, int write_energy)
{
    /* GATE 0 seam: fill -> commit -> Fortran -> read outputs.
     * All per-cell gathering logic (pre-refactor lines 173-721) was MOVED
     * VERBATIM into chem_fill_inputs() (chem_interface.cc, with line
     * provenance comments); chem_commit_to_fortran() is the only writer of
     * the COOLR/COOLI/CHEM_COMP/PROJECT COMMON blocks. */
    struct ChemInputs  chem_in;
    struct ChemOutputs chem_out;
    int i, chem_solved = 0;

    chem_in.dt_code = dt;
    chem_in.dl_code = dl;
    chem_in.mode    = mode;
    chem_in.u_code  = u_code;
    chem_fill_inputs(target, &chem_in);

    /* [orig chemcool.cc:620-622] temperature-only query returns before any
     * Fortran solve (and, now, before any COMMON write) */
    if(mode == 2) return chem_in.temp;

#ifdef CHEMCOOL_CXX
    COOLIC.id_current = chem_in.id_current;  /* only for chemcool_fatal messages */
#else
    chem_commit_to_fortran(&chem_in);
#endif

    /* Local working variables seeded from the seam struct — identical roles
     * (and values) to the pre-refactor locals of the same names. */
    double rho      = chem_in.rho_cgs;
    double timestep = chem_in.timestep_cgs;
    double divv     = chem_in.divv_cgs;
    double energy   = chem_in.energy_cgs;
    double dl_cgs   = chem_in.dl_cgs;
    double yn       = chem_in.yn;
    double temp     = chem_in.temp;
    double abh2 = chem_in.abh2, abhd = chem_in.abhd, abco = chem_in.abco, abe = chem_in.abe;
    double *abundances = chem_in.abundances;  /* working buffer; EVOLVE_ABUNDANCES in/out */
    double column_est  = chem_in.column_est;
    double ekn, u_new = u_code;


#ifdef CHEMCOOL_CXX
    chem_cxx_begin_cell(&chem_in, temp, abh2, abhd, abco);
#else
    CALC_PHOTO_WRAPPER(&temp, chem_in.rpar, &abh2, &abhd, &abco);
#endif

    int skip_evolve_abundances = 0;

#ifdef GALSF_RESOLVEDISM_PHOTOION
    double temp_HII = 1e4;
    /* abe_HII = H+ + He+ + D+ + C+ ≈ 1 + abhe + abundD + abundc */
    double abe_HII = 1.0 + ABHE + chem_in.abundD + chem_in.abundc;
    double energy_HII = temp_HII * 1.5 * BOLTZMANN_CGS * yn * (1.0 + ABHE + abe_HII);
    if(CellP[target].Ionized == 1) {
        skip_evolve_abundances = 1;
        chemcool_set_ionized(abundances, &chem_in);
        if(energy < energy_HII) {
            if(mode == 0) { /* PI heating ledger: floor raise [erg] */
                double dE_PI = (energy_HII - energy) / rho * P[target].Mass * UNIT_MASS_IN_CGS;
                #pragma omp atomic
                ChemcoolPIHeatErg += dE_PI;
                #pragma omp atomic
                ChemcoolPIHeatCells += 1;
            }
            energy = energy_HII;
        }
    }
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
    /* StarBench BENCHMARK-MODE thermodynamics (TestStarBenchIso > 0, 2026-07-31):
     * two-temperature medium exactly as the StarBench codes and the Smith+21/
     * Hu+17 idealized tests assume — ionized gas pinned at T_II, neutral gas
     * held isothermal at the given ambient T, NO radiative losses in between.
     * Rationale (measured): with the full thermochemistry on, mixed cells at
     * the ionized/neutral interface radiate the D-type driving work away at
     * high n (t_cool ~ kyr at n=100) and EVERY method — subgrid, M1, TreeRay —
     * undershoots the benchmark by ~50%%; the analytic solutions exclude such
     * losses by construction. Test-tag only; no effect in production. */
    if(All.TestStarBenchIso > 0) {
#if !defined(TREE_RAY_PI) && !defined(RADTRANSFER)
        skip_evolve_abundances = 1;
        if(CellP[target].Ionized == 1) {
            energy = energy_HII;                       /* exact T_II pin */
        } else {
            /* neutral atomic H+He, no electrons: hold isothermal */
            energy = All.TestStarBenchIso * 1.5 * BOLTZMANN_CGS * yn * (1.0 + ABHE);
        }
#endif
    }
#endif
#endif

#if defined(GALSF_RESOLVEDISM_ISOLATED_FB_TEST) && (defined(TREE_RAY_PI) || defined(RADTRANSFER))
    /* RT benchmark mode: chemistry ionizes the gas; neutral gas pinned isothermal after the solve */
    int sb_iso_neutral_pin = (All.TestStarBenchIso > 0) ? 1 : 0;
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
                chem_in.abundc, chem_in.abundo, yn);
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
            double co_pool = (chem_in.abundc < chem_in.abundo) ? chem_in.abundc : chem_in.abundo;
#if CHEMISTRYNETWORK != 4
            if(abundances[ICO] > co_pool * (1.0 + rtol) + 1e-20) viol |= 1;
#endif
            if(2.0*abundances[IH2] + abundances[IHP] > 1.0 + rtol) viol |= 2;
#if CHEMISTRYNETWORK == 1 || CHEMISTRYNETWORK == 17
            if(abundances[IHEP] + abundances[IHEPP] > chem_in.abhe * (1.0 + rtol) + 1e-20) viol |= 4;
            if(abundances[IDP] + abundances[IHD] > chem_in.abundD * (1.0 + rtol) + 1e-20) viol |= 8;
#endif
            if(viol) {
                int ionized_flag = -1;
#ifdef GALSF_RESOLVEDISM_PHOTOION
                ionized_flag = (int)CellP[target].Ionized;
#endif
                printf("CHEMCOOL_INPUT_VIOLATION: ID=%llu viol=%d yn=%.4e T=%.4e Ionized=%d\n",
                    (unsigned long long)P[target].ID, viol, yn, temp, ionized_flag);
                printf("CHEMCOOL_INPUT_VIOLATION: pools abundc=%.6e abundo=%.6e abhe=%.6e abundD=%.6e dgr=%.4e tdust=%.4e\n",
                    chem_in.abundc, chem_in.abundo, chem_in.abhe, chem_in.abundD,
                    chem_in.dust_to_gas_ratio, chem_in.tdust);
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
        chemcool_clamp_pools(abundances, &chem_in);
#ifdef CHEMCOOL_CXX
        chem_cxx_evolve_cell(timestep, dl_cgs, yn, divv, &energy, abundances);
        chem_solved = 1;
#else
        EVOLVE_ABUNDANCES(&timestep, &dl_cgs, &yn, &divv, &energy, abundances, &column_est);
#endif
    }

    /* GATE 0 seam: read back the COMMON fields the solve updated.  When the
     * solve was skipped (PHOTOION-ionized cell) this reads the same
     * (stale-lambda) COMMON state the pre-refactor code read -- identical. */
#ifdef CHEMCOOL_CXX
    chem_cxx_read_outputs(&chem_out);
    if(!chem_solved) {chem_out.cr_ion_heat = 0;} /* skipped solve: no CR ionisation heating applied */
#else
    chem_read_outputs(&chem_out);
#endif

    /* Compute cooling rate */
    double cooling_rate = 0.;
    for(i = 0; i < 28; i++) {
#ifdef OUTPUT_INDIVIDUAL_COOLRATES
        CellP[target].Lambda[i] = chem_out.lambda[i] / yn / yn;
#endif
        cooling_rate += chem_out.lambda[i];
    }
    for(i = 0; i < NRATES_CHEM; i++) {
#ifdef OUTPUT_INDIVIDUAL_COOLRATES
        if(i < 6) CellP[target].LambdaChem[i] = chem_out.lambda_chem[i] / yn / yn;
#endif
        cooling_rate += chem_out.lambda_chem[i];
    }
    cooling_rate *= (UNIT_LENGTH_IN_CGS * pow(UNIT_TIME_IN_CGS, 3) / UNIT_MASS_IN_CGS);

    abh2 = abundances[IH2];
    abe  = abundances[IHP];

    /* Compute final temperature */
#if CHEMISTRYNETWORK == 18
    (void)ekn; (void)abe;
    chem_cxx_net18_finish(abundances, energy, &temp);  /* T from all ions and electrons */
#elif CHEMISTRYNETWORK == 20
    (void)ekn; (void)abe;
    chem_cxx_net20_finish(abundances, energy, &temp);  /* T with the equilibrium x_e; writes CO */
#elif defined(CHEMCOOL_CXX)
    ekn = energy / (BOLTZMANN_CGS * (1.0 + ABHE - abh2 + abe) * yn);
    temp = chem_cxx_calc_temp(abh2, chem_in.abhe, ekn);
#else
    ekn = energy / (BOLTZMANN_CGS * (1.0 + ABHE - abh2 + abe) * yn);
    CALC_TEMP(&abh2, &ekn, &temp);
#endif

    /* Enforce the global temperature floor (All.MinGasTemp) on the CHEMCOOL path.
     * The parameter was accepted but silently IGNORED here (ignored-param class,
     * cf. the f_CR lesson) -- every other cooling backend honours it. Needed e.g.
     * for the StarBench D-type benchmark's fixed 1e3 K ambient. Thermal energy is
     * rescaled at fixed composition. (2026-07-30) */
    if(temp < All.MinGasTemp && temp > 0) {
        energy *= All.MinGasTemp / temp;
        temp = All.MinGasTemp;
    }

#if defined(GALSF_RESOLVEDISM_ISOLATED_FB_TEST) && (defined(TREE_RAY_PI) || defined(RADTRANSFER))
    if(sb_iso_neutral_pin) {
        /* StarBench two-temperature: T(x) = T_n + (1e4 - T_n) x (Bisbas+15) — removes <hnu> heating sensitivity */
        double x_pin = DMIN(DMAX(abundances[IHP], 0.0), 1.0);
        double T_pin = All.TestStarBenchIso + (1.0e4 - All.TestStarBenchIso) * x_pin;
        energy = T_pin * 1.5 * BOLTZMANN_CGS * yn * (1.0 + ABHE + x_pin);
        temp = T_pin;
    }
#endif

    /* Convert back to code units from cgs */
    energy *= pow(UNIT_LENGTH_IN_CGS, 3) / UNIT_ENERGY_IN_CGS;
    rho    /= UNIT_DENSITY_IN_CGS;

    if(mode == 0) {
        for(i = 0; i < TRAC_NUM; i++) {
            CellP[target].TracAbund[i] = abundances[i];
        }
#if defined(COSMIC_RAY_FLUID) && defined(CHEMCOOL_CXX)
        { /* the CR ionisation heating the solve applied, minus the radioactive share of zeta, comes out of e_CR (booked in E_CR_loss_cumul) */
            double zeta_cr = CR_IONRATE_SOLCIRC * (chem_in.cr_energy_density / 1.602e-12) / CR_ENDEN_SOLCIRC;
            double f_cr = (chem_in.cosmic_ray_ion_rate > 0) ? DMIN(zeta_cr / chem_in.cosmic_ray_ion_rate, 1.) : 0;
            double vol_cgs = P[target].Mass / (CellP[target].Density * All.cf_a3inv) * pow(UNIT_LENGTH_IN_CGS, 3);
            double de_code = f_cr * DMAX(chem_out.cr_ion_heat, 0) * vol_cgs / UNIT_ENERGY_IN_CGS, ecr_tot = 0;
            for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {ecr_tot += CellP[target].CosmicRayEnergy[k];}
            if(de_code > 0 && ecr_tot > 0) {
                static long n_capped = 0;
                if(de_code > ecr_tot) {if(n_capped++ == 0) {printf("CHEMCOOL: CR ionisation heating %g exceeds e_CR %g (ID %llu); capped at e_CR (first of possibly more)\n", de_code, ecr_tot, (unsigned long long)P[target].ID);} de_code = ecr_tot;}
                double q = 1. - de_code / ecr_tot;
                for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {CellP[target].CosmicRayEnergy[k] *= q; CellP[target].CosmicRayEnergyPred[k] *= q; CellP[target].CosmicRayFlux[k] *= q; CellP[target].CosmicRayFluxPred[k] *= q;}
            }
        }
#endif
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
            CellP[target].Ne  = chemcool_electrons(abundances, &chem_in);
        }
#endif
        u_new = energy / rho;
        if(write_energy) {CellP[target].InternalEnergy = u_new; CellP[target].InternalEnergyPred = u_new;}
        else {CellP[target].Ne = chemcool_electrons(abundances, &chem_in);} /* DoCooling backend: the chemistry's electrons per H */
        CellP[target].Temp = temp;
        CellP[target].DustTemp = chem_out.tdust;
#if defined(GALSF_RESOLVEDISM_G0_VARIABLE) || defined(RADTRANSFER)
        /* G0 x f_dust: the dust-attenuated field the chemistry used (Stella M1/TreeRay: f_dust covers floor pixels and the in-cell column) */
#ifdef OUTPUT_SHIELD_FAC
        CellP[target].ShieldFacDust = chem_out.fac_shield_dust;
        CellP[target].G0_atten      = CellP[target].G0 * chem_out.fac_shield_dust;
#else
        CellP[target].ShieldFacDust = 1.0;
        CellP[target].G0_atten      = CellP[target].G0;
#endif
#endif
#ifdef OUTPUT_CHEM_PHOTORATES
        for(i = 0; i < 5; i++) {CellP[target].ChemPhotoRates[i] = chem_out.photorates[i];}
#endif
#ifdef OUTPUT_COOLRATE
        CellP[target].CoolingRate_CHEMCOOL = cooling_rate;
#endif

#if defined(RADTRANSFER)
        /* --- Distribute CHEMCOOL cooling rates into RT bins ---
           Mirrors the standard cooling solver (cooling.cc ~lines 1295-1333).
           chem_out.lambda[0..27] (read from COOLRC.lambda) contains individual rates in erg/s/cm^3 (= nH^2 * Lambda).
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
                if(!isfinite(chem_out.lambda[jl])) {
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
                    static int lnan = 0;
                    if(lnan < 8) { lnan++;
                        printf("LAMNAN lambda[%d]=%g temp=%g nH=%g abh2=%g abe=%g\n",
                               jl, (double)chem_out.lambda[jl], temp, nHcgs_loc, abh2, abe); fflush(stdout); }
#endif
                    chem_out.lambda[jl] = 0.0;
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
            for(int j=0; j<20; j++) {Lambda_IR += chem_out.lambda[ir_idx[j]];}

            /* Atomic/ionic excitation → NUV:
               R3=atomic (index 2), R25=Ly-alpha (24), R26=HeI (25), R27=HeII (26) */
            double Lambda_NUV = chem_out.lambda[2] + chem_out.lambda[24] + chem_out.lambda[25] + chem_out.lambda[26];

            /* Bremsstrahlung R28 (index 27): NUV if hot, IR if cold */
            if(temp >= 1e5) {Lambda_NUV += chem_out.lambda[27];} else {Lambda_IR += chem_out.lambda[27];}

#ifdef RT_INFRARED
            CellP[target].Lambda_RadiativeCooling_toRHDBins[RT_FREQ_BIN_INFRARED] = Lambda_IR / nH2_loc;
#endif
#if defined(RT_NUV) && !defined(RT_NUV_BAND_6_8EV) /* Stella 6-8 eV band feeds G0: line cooling goes to IR below */
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
            rt_ir_lambdadust(target, temp, P, CellP);
            CellP[target].DustTemp = CellP[target].Dust_Temperature;
#endif
        }
#endif /* RADTRANSFER */

        return u_new;
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


/* electrons, H+ and HI per H from the cell's current chemistry abundances (startup ionisation state) */
double chemcool_startup_electrons(int target, double *x_HII, double *x_HI)
{
    struct ChemInputs cin; memset(&cin, 0, sizeof(cin));
    chem_fill_gas_abundances(target, &cin);
    double ab[TRAC_NUM]; for(int k = 0; k < TRAC_NUM; k++) {ab[k] = CellP[target].TracAbund[k];}
    *x_HII = ab[IHP]; *x_HI = DMAX(1.0 - 2.0*ab[IH2] - ab[IHP], 0.0);
#if CHEMISTRYNETWORK == 20
    return ab[IHP]; /* net 20 electrons come from the last solve */
#else
    return chemcool_electrons(ab, &cin);
#endif
}


#endif /* CHEMCOOL */
