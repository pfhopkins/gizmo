/* This structure holds all the information that is stored for each particle of the simulation. */

extern ALIGN(32) struct particle_data
{
    short int Type;                 /*!< flags particle type.  0=gas, 1=halo/high-res dm, 2=alt dm/disk/collisionless, 3=pic/dust/bulge/alt dm, 4=new stars, 5=sink */
    short int TimeBin;
    MyIDType ID;                    /*! < unique ID of particle (assigned at beginning of the simulation) */
    MyIDType ID_child_number;       /*! < child number for particles 'split' from main (retain ID, get new child number) */
#ifndef SINK_WIND_SPAWN
    int ID_generation;              /*! < generation (need to track for particle-splitting to ensure each 'child' gets a unique child number */
#else
    MyIDType ID_generation;
#endif
    
    integertime Ti_begstep;         /*!< marks start of current timestep of particle on integer timeline */
    integertime Ti_current;         /*!< current time of the particle */
    
    ALIGN(32) MyDouble Pos[3];      /*!< particle position at its current time */
    MyDouble Mass;                  /*!< particle mass */
    
    MyDouble Vel[3];                /*!< particle velocity at its current time */
    MyDouble dp[3];
    MyFloat Particle_DivVel;        /*!< velocity divergence of neighbors (for predict step) */
    
    MyDouble GravAccel[3];          /*!< particle acceleration due to gravity */
#ifdef PMGRID
    MyFloat GravPM[3];                /*!< particle acceleration due to long-range PM gravity force */
#endif
    MyFloat OldAcc;                    /*!< magnitude of old gravitational force. Used in relative opening criterion */
#ifdef SPECIAL_POINT_MOTION
    MyFloat Acc_Total_PrevStep[3];  /*!< old total acceleration on a given cell/particle */
#endif
#ifdef HERMITE_INTEGRATION
    MyFloat Hermite_OldAcc[3];
    MyFloat OldPos[3];
    MyFloat OldVel[3];
    MyFloat OldJerk[3];
    short int AccretedThisTimestep;     /*!< flag to decide whether to stick with the KDK step for stability reasons, e.g. when actively accreting */
#ifdef KETJU_REGULARIZATION
    short int HermiteHistoryStale;      /*!< 1 when particle just exited a KETJU chain — fall back to KDK for one step until OldX is reseeded */
#endif
#endif
#ifdef COUNT_MASS_IN_GRAVTREE
    MyFloat TreeMass;  /*!< Mass seen by the particle as it sums up the gravitational force from the tree - should be equal to total mass, a useful debug diagnostic  */
#endif
#if defined(EVALPOTENTIAL) || defined(COMPUTE_POTENTIAL_ENERGY) || defined(OUTPUT_POTENTIAL)
    MyFloat Potential;        /*!< gravitational potential */
#if defined(EVALPOTENTIAL) && defined(PMGRID)
    MyFloat PM_Potential;
#endif
#endif
#if defined(GALSF_SFR_TIDAL_HILL_CRITERION) || defined(TIDAL_TIMESTEP_CRITERION) || defined(COMPUTE_JERK_IN_GRAVTREE) || defined(OUTPUT_TIDAL_TENSOR) || (defined(SINGLE_STAR_TIMESTEPPING) && (SINGLE_STAR_TIMESTEPPING > 0)) || defined(ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION)
#define COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
    double tidal_tensorps[3][3];                        /*!< tidal tensor (=second derivatives of grav. potential) */
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    double tidal_tensor_mag_prev;                       /*!< saved frobenius norm of the tidal tensor, from the previous timestep >*/
    double tidal_tensorps_prevstep[3][3];               /*!< save the entire tensor if this is active >*/
    double tidal_zeta;                                  /*!< also need to calculate an analog of the ags zeta variable here >*/
#endif
#ifdef PMGRID
    double tidal_tensorpsPM[3][3];                      /*!< for TreePM simulations, long range tidal field */
#endif
#endif
    
#ifdef ADAPTIVE_TREEFORCE_UPDATE
    MyFloat time_since_last_treeforce;
    MyFloat tdyn_step_for_treeforce;
#ifndef COMPUTE_JERK_IN_GRAVTREE
#define COMPUTE_JERK_IN_GRAVTREE
#endif
#endif
    
#ifdef COMPUTE_JERK_IN_GRAVTREE
    double GravJerk[3];
#endif
    
#ifdef GALSF
    MyFloat StellarAge;        /*!< formation time of star particle */
#endif
#ifdef METALS
    MyFloat Metallicity[NUM_METAL_SPECIES]; /*!< metallicity (species-by-species) of gas or star particle */
#endif
#ifdef GALSF_SFR_IMF_VARIATION
    MyFloat IMF_Mturnover; /*!< IMF turnover mass [in solar] (or any other parameter which conveniently describes the IMF) */
    MyFloat IMF_FormProps[N_IMF_FORMPROPS]; /*!< formation properties of star particles to record for output */
#endif
#ifdef GALSF_SFR_IMF_SAMPLING
    MyFloat IMF_NumMassiveStars; /*!< number of massive stars to associate with this star particle (for feedback) */
#ifdef GALSF_SFR_IMF_SAMPLING_DISTRIBUTE_SF
    MyFloat TimeDistribOfStarFormation; /*!< free-fall time at the moment of star formation, which defines for this particle the delay distribution for forming the relevant O-stars */
    MyFloat IMF_WeightedMeanStellarFormationTime; /*!< weighted mean stellar formation time, to use instead of the normal stellarage parameter on-the-fly */
#endif
#endif
    
    MyFloat KernelRadius;           /*!< search radius around particle for neighbors/interactions */
    MyFloat NumNgb;                 /*!< neighbor number around particle */
    MyFloat DrkernNgbFactor;        /*!< correction factor needed for varying kernel lengths */
#ifdef DO_DENSITY_AROUND_NONGAS_PARTICLES
    MyFloat DensityAroundParticle;         /*!< gas density in the neighborhood of the collisionless particle (evaluated from neighbors) */
#endif
#if defined(DO_DENSITY_AROUND_NONGAS_PARTICLES) || defined(COOLING)
    MyFloat GradRho[3];             /*!< gas density gradient evaluated simply from the neighboring particles, for collisionless centers */
#endif
#ifdef RT_USE_TREECOL_FOR_NH
    MyFloat ColumnDensityBins[RT_USE_TREECOL_FOR_NH];     /*!< angular bins for column density */
    MyFloat SigmaEff;              /*!< effective column density -log(avg(exp(-sigma))) averaged over column density bins from the gravity tree (does not include the self-contribution) */
#endif
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
    MyFloat UV_luminosity;                /*!< FUV luminosity [erg/s], 6-13.6 eV (tree-ray) or 8-13.6 eV (M1 RT) */
    MyFloat LW_luminosity;                /*!< Lyman-Werner luminosity [erg/s], 11.2-13.6 eV (for H2 photodissociation) */
#ifdef GALSF_RESOLVEDISM_NUV_VARIABLE
    MyFloat NUV_luminosity;               /*!< near-UV luminosity [erg/s], 3.4-8 eV */
#endif
#ifdef GALSF_RESOLVEDISM_OPT_VARIABLE
    MyFloat OPT_luminosity;               /*!< optical+NIR luminosity [erg/s], 0.4-3.4 eV */
#endif
#ifdef GALSF_RESOLVEDISM_PHOTOION
    MyFloat Lyman_photons_per_sec;        /*!< ionizing photon rate [sec^-1] */
#endif
#endif
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
    int sampled;                          /*!< flag: 1 if IMF has been sampled for this star */
    MyFloat MstarSampleIMF[N_STELLAR_MASS]; /*!< individual stellar masses from IMF sampling */
#endif
#if defined(GALSF_RESOLVEDISM_SAMPLE_IMF) || defined(GALSF_RESOLVEDISM_STOCHASTIC_IMF)
    MyFloat FormationDensity;             /*!< physical gas density at time of star formation (code units * cf_a3inv at formation) */
#endif
#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
    MyFloat BirthMetallicity;             /*!< total metallicity Z at birth, for table lookups */
#endif
/* GALSF_RESOLVEDISM_METALS_INDIVIDUAL: per-species mass fractions now live in
 * P[].Metallicity[NUM_METAL_SPECIES=27] (FIRE pattern extended).  Layout:
 *   Metallicity[0] = total Z
 *   Metallicity[1] = He
 *   Metallicity[2..26] = 25 individual metals indexed by StellarElement enum
 *   X_H is implicit: 1 − Metallicity[0] − Metallicity[1]
 * No separate ElementAbundance[] array is stored. */
#if defined(CHEMCOOL) && (CHEMISTRYNETWORK == 17)
    MyFloat DeuteriumAbundance;       /*!< total D mass fraction (free + D+ + HD), for diffusion of D pool in network 17 */
#endif
#ifdef GALSF_RESOLVEDISM_WINDS
    MyFloat WindMassAccum;       /*!< accumulated wind mass since last injection [Msun] (used for momentum) */
    MyFloat WindMomentumAccum;   /*!< accumulated wind momentum magnitude [Msun * km/s] */
    MyFloat M_current_old;       /*!< M_current at start of previous timestep [Msun] */
    MyFloat last_wind_log_age;   /*!< log10(age in yr) at last wind injection, for telescoping cumulative-table lookup */
#endif
#ifdef GALSF_RESOLVEDISM_FB
    /* Σ_j (Mass_j × kernel.wk) measured by a FB weighting pre-pass.  Used as the
     * bit-exact normalizer in the FB injection pass (replacing stale
     * P[i].DensityAroundParticle): wk_j_normalized = Mass_j × kernel.wk /
     * FB_Area_weighted_sum → Σwk = 1 by construction → all per-event sums
     * (Mej, Esne, p_ejecta, yields[k]) conserve exactly.  STARFORGE pattern
     * adapted from mechanical_fb.cc:282.  Zeroed before each weighting pass. */
    MyDouble FB_Area_weighted_sum;
    /* Particle mass [Msun] snapshotted in resolvedism_determine_SNe() at the
     * moment SNe_ThisTimeStep is set (death detection).  The FB walks later
     * drain P[i].Mass down to ~rem_mass, so the bookkeeping loop in
     * resolvedism_inject_fb_energy() must use this snapshot value to log the
     * actual ejecta mass (M_at_SN_trigger - rem_mass) rather than the post-walk
     * residual (~0).  Reset to -1 when bookkeeping is done. */
    MyFloat M_at_SN_trigger;
#ifdef GALSF_RESOLVEDISM_SN_SPAWN
    /* SN ejecta-spawn prototype (2026-07-05): when the FB kernel holds less gas than
     * ~f*Mej (f = flag value), the SN is NOT kernel-deposited; instead the payload is
     * stored here and a new gas cell (mass+THERMAL energy+yields; v=v_star) is created
     * in merge_split_routine's safe phase; the split/merge machinery then regularizes
     * it to local resolution and hydro drives the blast. Inert without the flag. */
    int SN_SpawnPending;                              /*!< 1 = payload waiting to spawn */
    MyFloat SpawnEjMass;                              /*!< code-mass ejecta */
    MyFloat SpawnEjEnergy;                            /*!< code thermal energy (1e51 erg) */
    MyFloat SpawnEjZ[NUM_RESOLVEDISM_ELEMENTS];       /*!< code-mass yields per element */
    MyFloat FB_KernelGasMass;                         /*!< gas mass measured in FB weight walk */
#endif
#ifdef GALSF_RESOLVEDISM_FB_HEALPIX
    /* GRIFFIN-style solid-angle-uniform FB injection (Lahen+23; 2026-07-05): the
     * weighting walk counts gas neighbors per 12 HEALPix base pixels around the star;
     * the injection walk then weights each receiver 1/(N_occupied_pix * N_in_pixel)
     * instead of kernel wk — isotropic deposition, no funnelling into the nearest
     * dense clump (Lahen+23 fig. 4). Sum of weights = 1 exactly by construction. */
    MyFloat FB_HpxCount[12];
#endif
#endif
#ifdef GALSF_RESOLVEDISM_TYPE_IA
    MyFloat M_drawn_Ia;          /*!< original drawn mass for Type Ia DTD [Msun], >0 = WD eligible */
#endif
#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
    MyFloat Mstar;                        /*!< single massive-star mass assigned to particle */
#endif
#if defined(RT_SOURCE_INJECTION)
    MyFloat KernelSum_Around_RT_Source; /*!< kernel summation around sources for radiation injection (save so can be different from 'density') */
#endif
    
#ifdef GALSF_RESOLVEDISM_FB
    MyFloat InternalEnergyAroundParticle; /*!< mass-weighted ambient internal energy around star (from density loop) */
#endif
#ifdef GALSF_RESOLVEDISM_DUST
    MyFloat DGR_around; /*!< kernel-weighted local dust-to-gas ratio around star (from density loop) */
#endif
#if defined(GALSF_FB_MECHANICAL) || defined(GALSF_FB_THERMAL) || defined(GALSF_RESOLVEDISM_FB)
    MyFloat SNe_ThisTimeStep; /* flag that indicated number of SNe for the particle in the timestep */
#ifdef GALSF_FB_FIRE_STELLAREVOLUTION
    MyFloat MassReturn_ThisTimeStep; /* gas return from stellar winds */
#ifdef GALSF_FB_FIRE_RPROCESS
    MyFloat RProcessEvent_ThisTimeStep; /* R-process event tracker */
#endif
#ifdef GALSF_FB_FIRE_AGE_TRACERS
    MyFloat AgeDeposition_ThisTimeStep; /* age-tracer deposition */
#endif
#endif
#endif
#ifdef GALSF_FB_MECHANICAL
#define AREA_WEIGHTED_SUM_ELEMENTS 12 /* number of weights needed for full momentum-and-energy conserving system */
    MyFloat Area_weighted_sum[AREA_WEIGHTED_SUM_ELEMENTS]; /* normalized weights for particles in kernel weighted by area, not mass */
#endif
#ifdef GALSF_FB_FIRE_RT_LOCALRP
    MyFloat NewStar_Momentum_For_JetFeedback; /* amount of momentum to return from protostellar jet sub-grid model */
#endif
    
#if defined(GRAIN_FLUID)
    MyFloat Grain_Size;
    MyFloat Gas_Density;
    MyFloat Gas_InternalEnergy;
    MyFloat Gas_Velocity[3];
    MyFloat Grain_AccelTimeMin;
#if defined(GRAIN_BACKREACTION)
    MyFloat Grain_DeltaMomentum[3];
#endif
#if defined(GRAIN_LORENTZFORCE)
    MyFloat Gas_B[3];
#endif
#endif
#if defined(PIC_MHD)
    short int MHD_PIC_SubType;
#endif
    
#if defined(SINK_PARTICLES)
    MyIDType SwallowID;
    int IndexMapToTempStruc;   /*!< allows for mapping to SinkTempInfo struc */
    short int SinkSubType;     /*!< 0=SMBH (from IC or seeded), 1=stellar-mass BH (from dead star) */
#ifdef SINK_WIND_SPAWN
    MyFloat unspawned_wind_mass;    /*!< tabulates the wind mass which has not yet been spawned */
#endif
#ifdef SINK_COUNTPROGS
    int Sink_CountProgs;
#endif
    MyFloat Sink_Mass;
    MyFloat Sink_Formation_Mass; /* initial mass of sink (total particle) when it formed */
#ifdef SINK_RIAF_SUBEDDINGTON_MODEL
    MyFloat Sink_Mdot_ROI;
    MyFloat Sink_ROI;
#endif
#if defined(SINK_GRAVCAPTURE_FIXEDSINKRADIUS)
    MyFloat SinkRadius;
#endif
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
    MyFloat dt_since_last_gas_search; /* keep track of time since the sink's last neighbor search and gas interaction (for feedback/accretion) */
    short int do_gas_search_this_timestep; /* flag for deciding whether to do gas stuff for a given timestep */
#endif
#ifdef GRAIN_FLUID
    MyFloat Sink_Dust_Mass;
#endif
#ifdef RT_REINJECT_ACCRETED_PHOTONS
    MyFloat Sink_accreted_photon_energy;
#endif
#ifdef SINGLE_STAR_SINK_DYNAMICS
    MyFloat SwallowTime; /* freefall time of a particle onto a sink particle  */
#endif
#if defined(SINGLE_STAR_TIMESTEPPING)
    MyFloat Sink_SurroundingGasVel; /* Relative speed of sink to surrounding gas  */
#endif
#if (SINGLE_STAR_SINK_FORMATION & 8)
    int Sink_Ngb_Flag; /* whether or not the gas lives in a sink's hydro stencil */
#endif
#ifdef SINK_ALPHADISK_ACCRETION
    MyFloat Sink_Mass_Reservoir;
#endif
#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
    MyFloat Sink_AccretionDeficit; /* difference between continuously-accreted and discretely-accreted masses, needs to be evolved to ensure exact conservation with some modules */
#endif
#ifdef SINK_FOLLOW_ACCRETED_ANGMOM
    MyFloat Sink_Specific_AngMom[3];
#endif
#ifdef SINK_RETURN_BFLUX
    MyDouble B[3];
#endif
#ifdef JET_DIRECTION_FROM_KERNEL_AND_SINK
    MyFloat Mgas_in_Kernel;
    MyFloat Jgas_in_Kernel[3];
#endif
    MyFloat Sink_Mdot;
    int Sink_TimeBinGasNeighbor;
#if defined(SINGLE_STAR_TIMESTEPPING)
    MyFloat Sink_dr_to_NearestGasNeighbor;
#endif
#ifdef SINK_REPOSITION_ON_POTMIN
    MyFloat Sink_PotentialMinimumOfNeighborsPos[3];
    MyFloat Sink_PotentialMinimumOfNeighbors;
#endif
#endif  /* if defined(SINK_PARTICLES) */
#ifdef SINK_SEED_FROM_LOCALGAS_TOTALMENCCRITERIA
    MyFloat MencInRcrit;
#endif
    
    
#ifdef SINK_CALC_DISTANCES
    MyFloat Min_Distance_to_Sink;
    MyFloat Min_xyz_to_Sink[3];
#ifdef SPECIAL_POINT_MOTION
    MyFloat vel_of_nearest_special[3];
    MyFloat acc_of_nearest_special[3];
#ifdef SPECIAL_POINT_WEIGHTED_MOTION
    MyFloat weight_sum_for_special_point_smoothing;
#endif
#endif
#if defined(SINGLE_STAR_FIND_BINARIES) || (SINGLE_STAR_TIMESTEPPING > 0)
    MyDouble Min_Sink_OrbitalTime; //orbital time for binary
    MyDouble comp_dx[3]; //position offset of binary companion - this will be evolved in the Kepler solution while we use the Pos attribute to track the binary COM
    MyDouble comp_dv[3]; //velocity offset of binary companion - this will be evolved in the Kepler solution while we use the Vel attribute to track the binary COM velocity
    MyDouble comp_Mass; //mass of binary companion
    int is_in_a_binary; // flag whether star is in a binary or not
#endif
#ifdef SINGLE_STAR_TIMESTEPPING
    MyFloat Min_Sink_Freefall_time;
    MyFloat Min_Sink_Approach_Time;
#if (SINGLE_STAR_TIMESTEPPING > 0)
    int SuperTimestepFlag; // >=2 if allowed to super-timestep (increases with each drift/kick), 1 if a candidate for super-timestepping, 0 otherwise
    MyDouble COM_dt_tidal; //timescale from tidal tensor evaluated at the center of mass without contribution from the companion
    MyDouble COM_GravAccel[3]; //gravitational acceleration evaluated at the center of mass without contribution from the companion
#endif
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
    MyFloat MaxFeedbackVel; // maximum signal velocity of any feedback mechanism emanating from the star
    MyFloat Min_Sink_FeedbackTime;  // minimum time for feedback to arrive from a star
#endif
#endif
#endif
    
    
#ifdef SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION
    MyFloat ProtoStellarAge; /*!< record the proto-stellar age instead of age */
    MyFloat ProtoStellarRadius_inSolar; /*!< protostellar radius (also tracks evolution from protostar to ZAMS star) */
    int ProtoStellarStage; /* Track the stage of protostellar evolution, 0: pre collapse, 1: no burning, 2: fixed Tc burning, 3: variable Tc burning, 4: shell burning, 5: main sequence, 6: supernova, see Offner 2009 Appendix B*/ //IO flag IO_STAGE_PROTOSTAR
    MyFloat Mass_D; /* Mass of gas in the protostar that still contains D to burn */ // IO flag IO_MASS_D_PROTOSTAR
    MyFloat StarLuminosity_Solar; /* the total luminosity of the star in L_solar units*/ //IO flag IO_LUM_SINGLESTAR
    MyFloat ZAMS_Mass; /* The mass the star has when reaching the main sequence */ //IO flag IO_ZAMS_MASS
#ifdef SINGLE_STAR_FB_WINDS
    MyFloat Wind_direction[6]; // direction of wind launches, to reduce anisotropy launches go along a random axis then a random perpendicular one, then one perpendicular to both.
    int wind_mode; // tells what kind of wind model to use, 1 for particle spawning and 2 for using the FIRE wind module
#endif
#ifdef  SINGLE_STAR_FB_SNE
    MyFloat Mass_final; //final mass of the star before going SN (Since this is not saved to snapshots, hard restarts in the middle of spawning an SN will do weird things)
#endif
#endif
    
#ifdef KETJU_REGULARIZATION
    MyDouble KetjuFinalVel[3];  /* velocity swapped into P.Vel for the host KDK after drift (carries the pre-applied -neg2 half-kick) */
    MyDouble KetjuTrueVel[3];   /* TRUE physical end-of-step (t_sync) velocity straight from MSTAR, before any host-KDK half-kick bookkeeping — use this for energy diagnostics / analysis, NOT P.Vel (which is left half-a-kick off for chain members) */
    MyDouble KetjuTruePos[3];   /* TRUE physical end-of-step (t_sync) position straight from MSTAR — synchronized with KetjuTrueVel. P.Pos is reconstructed by the velocity-trick drift and can be slightly phase-desynced from the velocity at pericenter; use this pair for diagnostics. */
    MyDouble KetjuExtAccel[3];  /* EXTERNAL gravitational acceleration on this chain member. With member-member pairs excluded in the tree walk (see KetjuRegionTag), the member's tree GravAccel IS the external field, so this is just a copy of GravAccel. The host KDK applies only this external field to chain members; MSTAR integrates the internal forces. =0 for an isolated region, reducing exactly to pure-MSTAR. */
    int KetjuRegionTag;         /* nonzero region identifier (region index+1) used by the gravity tree to EXCLUDE member<->member interactions: two particles with the same nonzero tag skip their mutual force so each member's GravAccel is the exact external (non-member) field. 0 = not a chain member. Set in ketju_limit_timesteps and ketju_find_regions before each gravity_tree call; region ordering is globally consistent (merge over gathered centers), so the tag agrees across MPI tasks. */
    short int KetjuIntegrated;  /* 1 if this particle was KETJU-integrated this step */
    short int KetjuReleaseBlock; /* DURABLE Hermite block after a KETJU release. Decremented once per kick1 (do_the_kick mode==0); while >0, eligible_for_hermite returns 0. Unlike the consume-on-check HermiteHistoryStale (which the prediction pass eats before the correction pass runs), this survives arbitrary eligibility queries across sub-syncs, so it reliably blocks a released star's Hermite corrector from firing on its poisoned kick1 Old* (Old* captured before find_regions still holds a close companion's un-excluded force → 1e4 km/s on the IMF born-sub-softening pair). Cleared naturally once a clean full step has stored consistent Old*. */
    short int KetjuFreshScatter; /* 1 only between scatter_results (MSTAR results written this step) and ketju_set_final_velocities. Guards the velocity swap: a freshly CAPTURED member whose region was NOT integrated this step (members still inactive on the capture sync, run_integration skipped it) has KetjuIntegrated=1 but STALE KetjuFinalVel — swapping that in destroyed the state (one-step dE=+194 in the episodic test). */
    integertime KetjuTrickUntil; /* integer time at which this member's velocity-trick drift CHORD completes (= Ti at scatter + reg.ti_step). The final-velocity swap must NOT happen before this: ketju_set_final_velocities runs after EVERY sync, and when other particles create an intermediate sync mid-chord, a premature swap replaces the trick velocity with the physical one for the REMAINDER of the drift — the member then misses MSTAR's end position entirely (trace-verified: a 15 AU hard binary was teleported to 230 AU separation, +0.92 total energy in one step). Always-on configs never hit this (all members share the global-min bin = no intermediate syncs); episodic/cluster configs do. */
    integertime KetjuRegionTiStep; /* region external timestep (TNT get_region_max_timestep); 0 = not a region member. Applied in get_timestep so chain members take the external/region step, not the tiny internal-orbit one. */
#if defined(SINK_PARTICLES)
    MyDouble KetjuSpin[3];     /* BH spin angular momentum vector S [length*mass*velocity units] */
#endif
#ifdef KETJU_PN_REMNANT_TAG
    signed char RemnantType;   /* StellarRemnantType code; -1 = still a live star, set at SN time */
#endif
#endif

#if defined(DM_SIDM)
    double dtime_sidm; /*!< timestep used if self-interaction probabilities greater than 0.2 are found */
    long unsigned int NInteractions; /*!< Total number of interactions */
#endif
    
#if defined(SUBFIND)
    int GrNr;
    int SubNr;
    int DM_NumNgb;
    unsigned short targettask, origintask2;
    int origintask, submark, origindex;
    MyFloat DM_KernelRadius;
    union
    {
        MyFloat DM_Density;
        MyFloat DM_Potential;
    } u;
    union
    {
        MyFloat DM_VelDisp;
        MyFloat DM_BindingEnergy;
    } v;
#ifdef FOF_DENSITY_SPLIT_TYPES
    union
    {
        MyFloat int_energy;
        MyFloat density_sum;
    } w;
#endif
#endif
    
    float GravCost[GRAVCOSTLEVELS];   /*!< weight factor used for balancing the work-load */
    
#ifdef WAKEUP
    integertime dt_step;
#endif
    
#if defined(FIRE_SUPERLAGRANGIAN_JEANS_REFINEMENT) || defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM)
    MyFloat Time_Of_Last_MergeSplit;
#endif
    
#ifdef SPECIAL_POINT_WEIGHTED_MOTION
    MyFloat Time_Of_Last_SmoothedVelUpdate;
#endif
    
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE)
    MyFloat AGS_zeta;               /*!< correction term for adaptive gravitational softening lengths */
#endif
    
#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
    MyDouble AGS_KernelRadius;          /*!< smoothing length (for gravitational forces) */
    MyDouble AGS_vsig;          /*!< signal velocity of particle approach, to properly time-step */
#endif
    
#if defined(WAKEUP)
    short int wakeup;                     /*!< flag to wake up particle */
#endif
    
#ifdef GALSF_MERGER_STARCLUSTER_PARTICLES
    MyFloat StarParticleEffectiveSize;   /*!< effective 'size' of a star particle at formation */
#endif
    
#ifdef DM_FUZZY
    MyFloat AGS_Density;                /*!< density calculated corresponding to AGS routine (over interacting DM neighbors) */
    MyFloat AGS_Gradients_Density[3];   /*!< density gradient calculated corresponding to AGS routine (over interacting DM neighbors) */
    MyFloat AGS_Gradients2_Density[3][3];   /*!< density gradient calculated corresponding to AGS routine (over interacting DM neighbors) */
    MyFloat AGS_Numerical_QuantumPotential; /*!< additional potential terms 'generated' by un-resolved compression [numerical diffusivity] */
    MyFloat AGS_Dt_Numerical_QuantumPotential; /*!< time derivative of the above */
#if (DM_FUZZY > 0)
    MyFloat AGS_Psi_Re;
    MyFloat AGS_Psi_Re_Pred;
    MyFloat AGS_Dt_Psi_Re;
    MyFloat AGS_Gradients_Psi_Re[3];
    MyFloat AGS_Gradients2_Psi_Re[3][3];
    MyFloat AGS_Psi_Im;
    MyFloat AGS_Psi_Im_Pred;
    MyFloat AGS_Dt_Psi_Im;
    MyFloat AGS_Gradients_Psi_Im[3];
    MyFloat AGS_Gradients2_Psi_Im[3][3];
    MyFloat AGS_Dt_Psi_Mass;
#endif
#endif
#if defined(AGS_FACE_CALCULATION_IS_ACTIVE)
    MyDouble NV_T[3][3];                                           /*!< holds the tensor used for gradient estimation */
#endif
#ifdef CBE_INTEGRATOR
    double CBE_basis_moments[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];         /* moments per basis function */
    double CBE_basis_moments_dt[CBE_INTEGRATOR_NBASIS][CBE_INTEGRATOR_NMOMENTS];      /* time-derivative of moments per basis function */
#ifdef CBE_INTEGRATOR_WITHGRADIENTS
    CBE_basis_moments_Gradients[CBE_INTEGRATOR_NBASIS][3]; /* gradients of the scalar weight of each basis function */
#endif
#endif
}
*P,                /*!< holds particle data on local processor */
*DomainPartBuf;        /*!< buffer for particle data used in domain decomposition */
