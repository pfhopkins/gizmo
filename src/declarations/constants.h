/* values of various constants to be parsed as part of global definitions */

/* Minimum number of active particles for GPU offload to be worthwhile.
 * Below this threshold, the kernel launch + memory transfer overhead exceeds
 * the GPU compute benefit, so we fall back to the CPU (OpenMP) path.
 * Applies to cooling, nuclear burning, and any future Kokkos-parallelized loops. */
#ifndef GPU_MIN_PARTICLES_FOR_OFFLOAD
#define GPU_MIN_PARTICLES_FOR_OFFLOAD 4096
#endif
/* DEBUG BISECTION: override to force CPU fallback for cooling/nuclear/turb.
   Uncomment to disable GPU kernels while keeping GPU kernels active
   (All_dev sync, __managed__ CoolTables, #pragma omp declare target, etc.) */
//#undef GPU_MIN_PARTICLES_FOR_OFFLOAD
//#define GPU_MIN_PARTICLES_FOR_OFFLOAD 999999999

#if (SLOPE_LIMITER_TOLERANCE > 0)
#define WAKEUP   4.1            /* allows 2 timestep bins within kernel */
#else
#define WAKEUP   2.1            /* allows only 1-separated timestep bins within kernel */
#endif

#define ARENA_SHARE_OF_TASK_MEMORY          0.90 /* most of a task's memory share the working pool may take when the code has to bring its own size down to fit the machine; the rest is for the particle arrays and the tree, which are not allocated from the pool */
#define REDUC_FAC_FOR_MEMORY_IN_DOMAIN      0.98 /* used to pad memory in domain decomposition structures, should be slightly less than unity */

#define TREE_ALLOC_FACTOR_START             0.45 /* tree nodes allocated per particle at startup, before the run ratchets it upward to fit the particle distribution.  Named here because the startup memory projection has to use it before init() assigns it */

/* Cushion on the particle storage a rank holds for the coming epoch, as a fraction of the particles
 * per rank.  It covers the amount by which the next epoch's worst ghost import may exceed the one
 * just measured; on a galaxy-scale zoom that overshoot is a few percent of the per-rank count at the
 * median and around an eighth of it at the ninetieth percentile, so this covers the common case with
 * room to spare.  The rare import that exceeds it is met by growing the storage when it arrives. */
#define DOMAIN_CAPACITY_SAFETY_FRACTION     0.15

/* Releasing storage means migrating every particle array, so it is only worth doing for a saving that
 * is a real proportion of the capacity AND a real amount of memory.  The second test is expressed
 * against the particles per rank rather than as a slot count, so that it means the same thing on a
 * small problem as on a large one; a fixed count would silently forbid ever releasing anything on a
 * problem whose whole per-rank capacity is smaller than that count. */
#define DOMAIN_CAPACITY_SHRINK_FRACTION     0.10
#define DOMAIN_CAPACITY_SHRINK_MIN_FRAC     0.10

/* these are tolerances for the slope-limiters. we define them here, because the gradient constraint routine needs to be sure to use the -same- values in both the gradients and reimann solver routines */
#if MHD_CONSTRAINED_GRADIENT
#if (MHD_CONSTRAINED_GRADIENT > 1)
#define MHD_CONSTRAINED_GRADIENT_FAC_MINMAX 7.5
#define MHD_CONSTRAINED_GRADIENT_FAC_MEDDEV 5.0
#define MHD_CONSTRAINED_GRADIENT_FAC_MED_PM 0.25
#define MHD_CONSTRAINED_GRADIENT_FAC_MAX_PM 0.25
#else
#define MHD_CONSTRAINED_GRADIENT_FAC_MINMAX 7.5
#define MHD_CONSTRAINED_GRADIENT_FAC_MEDDEV 1.5
#define MHD_CONSTRAINED_GRADIENT_FAC_MED_PM 0.2
#define MHD_CONSTRAINED_GRADIENT_FAC_MAX_PM 0.2
#endif
#else
#define MHD_CONSTRAINED_GRADIENT_FAC_MINMAX 2.0
#define MHD_CONSTRAINED_GRADIENT_FAC_MEDDEV 1.0
#define MHD_CONSTRAINED_GRADIENT_FAC_MED_PM 0.20
#define MHD_CONSTRAINED_GRADIENT_FAC_MAX_PM 0.125
#endif

#ifndef  MULTIPLEDOMAINS
#define  MULTIPLEDOMAINS     8    /* fallback only: precompiler_logic.h is included before this header and already sets the effective default to 32; this 8 takes effect only if that define is removed or the user sets MULTIPLEDOMAINS in Config.sh */
#endif

#ifndef  TOPNODEFACTOR
#ifdef SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM
#define  TOPNODEFACTOR       8.0
#else
#define  TOPNODEFACTOR       4.0
#endif
#endif

#ifndef  GRAVCOSTLEVELS
#define  GRAVCOSTLEVELS      20
#endif

#define  NUMBER_OF_MEASUREMENTS_TO_RECORD  6  /* this is the number of past executions of a timebin that the reported average CPU-times average over */

#define  NODELISTLENGTH      8

#define  GRAVITY_LET_DETECTOR_ENTRIES  4096  /* how many targets the gravity walk can record as not covered by the locally-built tree. Recording even one stops the run, so this only has to be large enough to report what happened, with room for every thread to record at once */

#define  EPSILON_FOR_TREERND_SUBNODE_SPLITTING (1.0e-4) /* define some number << 1; particles with less than this separation will trigger randomized sub-node splitting in the tree. we set it to a global value here so that other sub-routines will know not to force particle separations below this */

#if !defined(EOS_GAMMA)
#define EOS_GAMMA (5.0/3.0) /*!< adiabatic index of simulated gas */
#endif

#ifdef GAMMA_ENFORCE_ADIABAT
#define EOS_ENFORCE_ADIABAT (GAMMA_ENFORCE_ADIABAT) /* this allows for either term to be defined, for backwards-compatibility */
#endif


#if !defined(RT_HYDROGEN_GAS_ONLY) || defined(RT_CHEM_PHOTOION_HE)
#define  HYDROGEN_MASSFRAC 0.76 /*!< mass fraction of hydrogen, relevant only for radiative cooling */
#else
#define  HYDROGEN_MASSFRAC 1.0  /*!< mass fraction of hydrogen, relevant only for radiative cooling */
#endif


#define  MAX_REAL_NUMBER  1e56
#define  MIN_REAL_NUMBER  1e-56

#if (defined(MAGNETIC) && !defined(COOLING)) || defined(EOS_ELASTIC)
#define  CONDITION_NUMBER_DANGER  1.0e7 /*!< condition number above which we will not trust matrix-based gradients */
#else
#define  CONDITION_NUMBER_DANGER  1.0e3 /*!< condition number above which we will not trust matrix-based gradients */
#endif

/* ... often used physical constants (cgs units). note many of these are defined to better precision with different units in e.g. the GSL package, these are purely for user convenience */
#define  GRAVITY_G_CGS      (6.672e-8)
#define  SOLAR_MASS_CGS     (1.989e33)
#define  SOLAR_LUM_CGS      (3.826e33)
#define  SOLAR_RADIUS_CGS   (6.957e10)
#define  BOLTZMANN_CGS      (1.38066e-16)
#define  C_LIGHT_CGS        (2.9979e10)
#define  PROTONMASS_CGS     (1.6726e-24)
#define  ELECTRONMASS_CGS   (9.10953e-28)
#define  THOMPSON_CX_CGS    (6.65245e-25)
#define  ELECTRONCHARGE_CGS (4.8032e-10)
#define  SECONDS_PER_YEAR   (3.155e7)
#define  HUBBLE_H100_CGS    (3.2407789e-18)    /* in h/sec */
#define  ELECTRONVOLT_IN_ERGS (1.60217733e-12)
#define HABING_FLUX_CGS      (1.6e-3)
#define DRAINE_FLUX_CGS      (1.7 * HABING_FLUX_CGS)


/* and a bunch of useful unit-conversion macros pre-bundled here, to help keep the 'h' terms and other correct */
#define UNIT_MASS_IN_CGS        ((All.UnitMass_in_g/All.HubbleParam))
#define UNIT_VEL_IN_CGS         ((All.UnitVelocity_in_cm_per_s))
#define UNIT_LENGTH_IN_CGS      ((All.UnitLength_in_cm/All.HubbleParam))
#define UNIT_TIME_IN_CGS        (((UNIT_LENGTH_IN_CGS)/(UNIT_VEL_IN_CGS)))
#define UNIT_ENERGY_IN_CGS      (((UNIT_MASS_IN_CGS)*(UNIT_VEL_IN_CGS)*(UNIT_VEL_IN_CGS)))
#define UNIT_PRESSURE_IN_CGS    (((UNIT_ENERGY_IN_CGS)/(UNIT_LENGTH_IN_CGS*UNIT_LENGTH_IN_CGS*UNIT_LENGTH_IN_CGS)))
#define UNIT_DENSITY_IN_CGS     (((UNIT_MASS_IN_CGS)/(UNIT_LENGTH_IN_CGS*UNIT_LENGTH_IN_CGS*UNIT_LENGTH_IN_CGS)))
#define UNIT_SPECEGY_IN_CGS     (((UNIT_PRESSURE_IN_CGS)/(UNIT_DENSITY_IN_CGS)))
#define UNIT_SURFDEN_IN_CGS     (((UNIT_DENSITY_IN_CGS)*(UNIT_LENGTH_IN_CGS)))
#define UNIT_FLUX_IN_CGS        (((UNIT_PRESSURE_IN_CGS)*(UNIT_VEL_IN_CGS)))
#define UNIT_LUM_IN_CGS         (((UNIT_ENERGY_IN_CGS)/(UNIT_TIME_IN_CGS)))
#define UNIT_B_IN_GAUSS         ((sqrt(4.*M_PI*UNIT_PRESSURE_IN_CGS)))
#define UNIT_MASS_IN_SOLAR      (((UNIT_MASS_IN_CGS)/SOLAR_MASS_CGS))
#define UNIT_DENSITY_IN_NHCGS   (((UNIT_DENSITY_IN_CGS)/PROTONMASS_CGS))
#define UNIT_TIME_IN_YR         (((UNIT_TIME_IN_CGS)/(SECONDS_PER_YEAR)))
#define UNIT_TIME_IN_MYR        (((UNIT_TIME_IN_CGS)/(1.e6*SECONDS_PER_YEAR)))
#define UNIT_TIME_IN_GYR        (((UNIT_TIME_IN_CGS)/(1.e9*SECONDS_PER_YEAR)))
#define UNIT_LENGTH_IN_SOLAR    (((UNIT_LENGTH_IN_CGS)/SOLAR_RADIUS_CGS))
#define UNIT_LENGTH_IN_AU       (((UNIT_LENGTH_IN_CGS)/1.496e13))
#define UNIT_LENGTH_IN_PC       (((UNIT_LENGTH_IN_CGS)/3.085678e18))
#define UNIT_LENGTH_IN_KPC      (((UNIT_LENGTH_IN_CGS)/3.085678e21))
#define UNIT_PRESSURE_IN_EV     (((UNIT_PRESSURE_IN_CGS)/ELECTRONVOLT_IN_ERGS))
#define UNIT_VEL_IN_KMS         (((UNIT_VEL_IN_CGS)/1.e5))
#define UNIT_LUM_IN_SOLAR       (((UNIT_LUM_IN_CGS)/SOLAR_LUM_CGS))
#define UNIT_FLUX_IN_HABING     (((UNIT_FLUX_IN_CGS)/HABING_FLUX_CGS))
#define UNIT_EGY_DENSITY_IN_HABING ((UNIT_PRESSURE_IN_CGS)/(HABING_FLUX_CGS / C_LIGHT_CGS))
#define U_TO_TEMP_UNITS         ((PROTONMASS_CGS/BOLTZMANN_CGS)*((UNIT_ENERGY_IN_CGS)/(UNIT_MASS_IN_CGS))) /* units to convert specific internal energy to temperature. needs to be multiplied by dimensionless factor=mean_molec_weight_in_amu*(gamma_eos-1) */
#define MEAN_MOLECULAR_WEIGHT_IONIZED (0.59) /* mean molecular weight of fully-ionized primordial gas, in units of the proton mass. this is the composition assumed wherever no chemistry is solved: it is the fallback the cached MeanMolecularWeight is set to, and the value the non-gas helpers use */
#ifndef C_LIGHT_CODE
#define C_LIGHT_CODE            ((C_LIGHT_CGS/UNIT_VEL_IN_CGS)) /* pure convenience function, speed-of-light in code units */
#endif
#define C_LIGHT_CODE_REDUCED    (((RT_SPEEDOFLIGHT_REDUCTION)*(C_LIGHT_CODE))) /* reduced speed-of-light in code units, again here as a convenience function, but just returns constant */
#define H0_CGS                  ((All.HubbleParam*HUBBLE_H100_CGS)) /* actual value of H0 in cgs */
#define COSMIC_BARYON_DENSITY_CGS ((All.OmegaBaryon*(H0_CGS)*(H0_CGS)*(3./(8.*M_PI*GRAVITY_G_CGS))*All.cf_a3inv)) /* cosmic mean baryon density [scale-factor-dependent] in cgs units */



#ifdef RT_COMOVING
#define RSOL_CORRECTION_FACTOR_FOR_VELOCITY_TERMS (0) /* this prefactor goes in front of various terms which vanish in the comoving frame RHD equations */
#else
#define RSOL_CORRECTION_FACTOR_FOR_VELOCITY_TERMS ((C_LIGHT_CODE_REDUCED)/(C_LIGHT_CODE)) /* these terms in the mixed-frame equations need to be multiplied by c_reduced/c */
#endif


#ifdef GALSF_FB_FIRE_RT_HIIHEATING
#define HIIRegion_Temp (1.0e4) /* temperature (in K) of heated gas */
#endif


#if defined(COOLING) || defined(RT_INFRARED)
#define MAX_DUST_TEMP 1.0e4 // maximum dust temperature for which we expect to call opacity or dust-to-metals ratio functions
#endif

/* some flags for the field "flag_ic_info" in the file header */
#define FLAG_ZELDOVICH_ICS     1
#define FLAG_SECOND_ORDER_ICS  2
#define FLAG_EVOLVED_ZELDOVICH 3
#define FLAG_EVOLVED_2LPT      4
#define FLAG_NORMALICS_2LPT    5


#ifndef PM_ASMTH
#define PM_ASMTH (1.25) /*! PM_ASMTH gives the scale of the short-range/long-range force split in units of FFT-mesh cells */
#endif
#ifndef PM_RCUT
#define PM_RCUT (4.5) /*! PM_RCUT gives the maximum distance (in units of the scale used for the force split) out to which short-range forces are evaluated in the short-range tree walk. */
#endif
#define MAXLEN_OUTPUTLIST 1201    /*!< maxmimum number of entries in output list */
#define DRIFT_TABLE_LENGTH 1000    /*!< length of the lookup table used to hold the drift and kick factors */
#define MAXITER 150

#ifndef LINKLENGTH
#define LINKLENGTH (0.2)
#endif
#ifndef FOF_GROUP_MIN_SIZE
#ifdef FOF_GROUP_MIN_LEN
#define FOF_GROUP_MIN_SIZE FOF_GROUP_MIN_LEN
#else
#define FOF_GROUP_MIN_SIZE 32
#endif
#endif
#ifndef SUBFIND_ADDIO_NUMOVERDEN
#define SUBFIND_ADDIO_NUMOVERDEN 1
#endif




#define CPU_ALL            0
#define CPU_TREEWALK1      1
#define CPU_TREEWALK2      2
#define CPU_TREEWAIT1      3
#define CPU_TREEWAIT2      4
#define CPU_TREESEND       5
#define CPU_TREERECV       6
#define CPU_TREEMISC       7
#define CPU_TREEBUILD      8
#define CPU_TREEHMAXUPDATE 9
#define CPU_DOMAIN         10
#define CPU_DENSCOMPUTE    11
/* Slot 12: holds the GRADIENTS bracket (printed as "gradients"); the name is
 * kept so slot numbers, and the meaning of columns in old logs, do not shift. */
#define CPU_DENSWAIT       12
#define CPU_DENSCOMM       13
/* Slot 14: "misc_hydro" catch-all; also receives the drift charged by the
 * shared ghost-import helpers, whose callers include non-hydro loops. */
#define CPU_DENSMISC       14
#define CPU_HYDCOMPUTE     15
#define CPU_HYDWAIT        16
#define CPU_HYDCOMM        17
#define CPU_HYDMISC        18
#define CPU_DRIFT          19
/* Slot 20: was CPU_TIMELINE (mislabeled "kicks" in cpu.txt). It's actually
 * populated only by find_timesteps() in core/timestep.cc, so renamed to
 * reflect its real meaning. No "kicks" bucket exists in legacy CPU_Step;
 * kick wall time falls into CPU_MISC by default. */
#define CPU_FIND_TIMESTEPS 20
/* Slot 21: RETIRED. The potential is computed in-tree together with the force,
 * so no separate potential phase exists and nothing charges this bucket. The
 * slot number is kept so old logs and parsers keep their column meaning. */
#define CPU_POTENTIAL      21
#define CPU_MESH           22
#define CPU_PEANO          23
#define CPU_COOLINGSFR     24
#define CPU_SNAPSHOT       25
#define CPU_FOF            26
#define CPU_SINKS     27
#define CPU_MISC           28
#define CPU_DRAGFORCE      29
#define CPU_SNIIHEATING    30
#define CPU_HIIHEATING     31
#define CPU_LOCALWIND      32
#define CPU_COOLSFRIMBAL   33
#define CPU_AGSDENSCOMPUTE 34
#define CPU_AGSDENSWAIT    35
#define CPU_AGSDENSCOMM    36
#define CPU_AGSDENSMISC    37
#define CPU_DYNDIFFMISC       38
#define CPU_DYNDIFFCOMPUTE    39
#define CPU_DYNDIFFWAIT       40
#define CPU_DYNDIFFCOMM       41
#define CPU_IMPROVDIFFMISC    42
#define CPU_IMPROVDIFFCOMPUTE 43
#define CPU_IMPROVDIFFWAIT    44
#define CPU_IMPROVDIFFCOMM    45
#define CPU_RTNONFLUXOPS  46
/* Categories carved from the original CPU_DUMMY00..10 slots. One double
 * increment per call site — the same cost as any other CPU_Step bucket.
 * Slots marked "reserved" have no writer; they are kept so slot numbers, and
 * therefore the meaning of columns in existing logs, never shift. */
#define CPU_NGB_BUILD         47  /* neighbor-list (CSR) construction, all callers */
#define CPU_SIDX_REFRESH      48  /* post-drift spatial-index refresh (tile bboxes, BVH refit, device staging) */
#define CPU_GHOSTIMPORT_SYMM  49  /* hydro-corridor SYMMETRIC ghost import (sub-row of ghost import) */
#define CPU_GRAV_PRECOMP      50  /* reserved: never charged */
#define CPU_FORCE_UPDATE_TREE 51  /* dynamic tree update on reused-tree steps (node drift + top-node kick exchange) */
#define CPU_SINK_ENV          52  /* sink_environment GPU kernel + scatter */
#define CPU_SINK_FEEDSWK      53  /* sink_feed + sink_swallow_and_kick combined */
#define CPU_PAIR_KERNEL       54  /* neighbor-loop pair-kernel launch + fence (runner only) */
#define CPU_GHOSTIMPORT       55  /* imported-ghost exchange, all callers (hydro, gravity, sinks, feedback) */
#define CPU_KICKS             56  /* first+second half-step kicks, Hermite predict/correct */
#define CPU_LOGMSG            57  /* per-step statistics + log-file writes (a barrier-sync absorber) */

#define CPU_PARTS          58  /* this gives the number of parts above (must be last) */

#define CPU_STRING_LEN 120


#if (BOX_SPATIAL_DIMENSION==1) || defined(ONEDIM)
#define NUMDIMS 1           /* define number of dimensions and volume normalization */
#define VOLUME_NORM_COEFF_FOR_NDIMS 2.0
#elif (BOX_SPATIAL_DIMENSION==2) || defined(TWODIMS)
#define NUMDIMS 2
#define VOLUME_NORM_COEFF_FOR_NDIMS M_PI
#else
#define VOLUME_NORM_COEFF_FOR_NDIMS 4.188790204786  /* 4pi/3 */
#define NUMDIMS 3
#endif


#define CUBE_EDGEFACTOR_1 0.366025403785    /* CUBE_EDGEFACTOR_1 = 0.5 * (sqrt(3)-1) */
#define CUBE_EDGEFACTOR_2 0.86602540        /* CUBE_EDGEFACTOR_2 = 0.5 * sqrt(3) */
