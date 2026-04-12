
/*! \file allvars.h
 *  \brief declares global variables.
 *
 *  This file declares all global variables. Further variables should be added here, and declared as
 *  'extern'. The actual existence of these variables is provided by the file 'allvars.c'. To produce
 *  'allvars.c' from 'allvars.h', do the following:
 *
 *     - Erase all #define statements
 *     - add #include "../declarations/allvars.h"
 *     - delete all keywords 'extern'
 *     - delete all struct definitions enclosed in {...}, e.g.
 *        "extern struct global_data_all_processes {....} All;"
 *        becomes "struct global_data_all_processes All;"
 */

/*!
 * This file was originally part of the GADGET3 code developed by
 * Volker Springel. The code has been modified extensively
 * by Phil Hopkins (phopkins@caltech.edu) for GIZMO (many new variables,
 * structures, and different naming conventions for some old variables).
 * The declarations have also been divided into a new scheme which improves
 * read-ability and ease of use and separates all of the macros written for
 * GIZMO.
 */


#ifndef ALLVARS_H
#define ALLVARS_H

#define GIZMO_VERSION     2026  /*!< code version (should be an int corresponding to the year) */

#include <mpi.h>
#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include <hdf5.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../GIZMO_config.h"

#include "precompiler_logic.h"

#include "constants.h"

#include "../eos/eos.h"

#ifdef COOL_GRACKLE
#include <grackle.h>
#endif

#include "../system/tags.h"

#include <assert.h>
#include <algorithm>
#include <type_traits>
#include <vector>

#include "macros.h"

#include "typedefs.h"
#include "../math_types/vec3.h"
#include "../math_types/mat3.h"
#include "../math_types/symmetric_tensor2.h"

#ifdef CHIMES
#include "../cooling/chimes/chimes_proto.h"
#endif

/*********************************************************/
/*  Global variables                                     */
/*********************************************************/

#ifdef BOX_PERIODIC
extern MyDouble boxSize, boxHalf;
#else
#define boxSize (All.BoxSize)
#define boxHalf (0.5*All.BoxSize)
#endif
#ifdef BOX_LONG_X
extern MyDouble boxSize_X, boxHalf_X;
#else
#define boxSize_X boxSize
#define boxHalf_X boxHalf
#endif
#ifdef BOX_LONG_Y
extern MyDouble boxSize_Y, boxHalf_Y;
#else
#define boxSize_Y boxSize
#define boxHalf_Y boxHalf
#endif
#ifdef BOX_LONG_Z
extern MyDouble boxSize_Z, boxHalf_Z;
#else
#define boxSize_Z boxSize
#define boxHalf_Z boxHalf
#endif

#ifdef BOX_SHEARING
extern MyDouble Shearing_Box_Vel_Offset;
extern MyDouble Shearing_Box_Pos_Offset;
#endif

#if defined(BOX_REFLECT_X) || defined(BOX_REFLECT_Y) || defined(BOX_REFLECT_Z) || defined(BOX_OUTFLOW_X) || defined(BOX_OUTFLOW_Y) || defined(BOX_OUTFLOW_Z)
extern short int special_boundary_condition_xyz_def_reflect[3];
extern short int special_boundary_condition_xyz_def_outflow[3];
#endif

template<typename T> GIZMO_GPU_FUNCTION inline void nearest_xyz(Vec3<T>& v, int sign=1) { NEAREST_XYZ(v[0], v[1], v[2], sign); }


#ifdef CHIMES
extern struct gasVariables *ChimesGasVars;
extern struct globalVariables ChimesGlobalVars;
extern char ChimesDataPath[256];
extern char ChimesEqAbundanceTable[196];
extern char ChimesPhotoIonTable[196];
extern double chimes_rad_field_norm_factor;
extern double shielding_length_factor;
extern double cr_rate;
extern int ChimesEqmMode;
extern int ChimesUVBMode;
extern int ChimesInitIonState;
extern int Chimes_incl_full_output;
extern int N_chimes_full_output_freq;
#endif // CHIMES
#ifdef CHIMES_METAL_DEPLETION
#define DEPL_N_ELEM 17
struct Chimes_depletion_data_structure
{
    double SolarAbund[DEPL_N_ELEM];
    double DeplPars[DEPL_N_ELEM][3];
    double DustToGasSaturated;
    double ChimesDepletionFactors[7];
    double ChimesDustRatio;
};
extern struct Chimes_depletion_data_structure *ChimesDepletionData;
#endif // CHIMES_METAL_DEPLETION


extern std::vector<int> ActiveParticleList;
extern unsigned char *ProcessedFlag;
extern int TimeBinCount[TIMEBINS];
extern int TimeBinCountGas[TIMEBINS];
extern int TimeBinActive[TIMEBINS];
extern int FirstInTimeBin[TIMEBINS];
extern int LastInTimeBin[TIMEBINS];
extern std::vector<int> NextInTimeBin;
extern std::vector<int> PrevInTimeBin;
#ifdef GALSF
extern double TimeBinSfr[TIMEBINS];
#endif

#ifdef SINK_PARTICLES
extern double TimeBin_Sink_mass[TIMEBINS];
extern double TimeBin_Sink_dynamicalmass[TIMEBINS];
extern double TimeBin_Sink_Mdot[TIMEBINS];
extern double TimeBin_Sink_Medd[TIMEBINS];
#endif


extern int ThisTask;		/*!< the number of the local processor  */
extern int NTask;		/*!< number of processors */
extern int PTask;		/*!< note: NTask = 2^PTask */
extern double CPUThisRun;	/*!< Sums CPU time of current process */
extern int NumForceUpdate;	/*!< number of active particles on local processor in current timestep  */
extern long long GlobNumForceUpdate;
extern int NumGasUpdate;	/*!< number of active gas cells on local processor in current timestep  */
extern int MaxTopNodes;	        /*!< Maximum number of nodes in the top-level tree used for domain decomposition */
extern int RestartFlag;		/*!< taken from command line used to start code. 0 is normal start-up from initial conditions, 1 is resuming a run from a set of restart files, while 2 marks a restart from a snapshot file. */
extern int RestartSnapNum;
extern int SelRnd;
extern int TakeLevel;
extern int *Exportflag;	        /*!< Buffer used for flagging whether a particle needs to be exported to another process */
extern int *Exportnodecount;
extern int *Exportindex;
extern int *Send_offset, *Send_count, *Recv_count, *Recv_offset;
extern size_t AllocatedBytes;
extern size_t HighMarkBytes;
extern size_t FreeBytes;
extern double CPU_Step[CPU_PARTS];
extern char CPU_Symbol[CPU_PARTS];
extern char CPU_SymbolImbalance[CPU_PARTS];
extern char CPU_String[CPU_STRING_LEN + 1];
extern double WallclockTime;    /*!< This holds the last wallclock time measurement for timings measurements */
extern int Flag_FullStep;	/*!< Flag used to signal that the current step involves all particles */
extern size_t HighMark_run,  HighMark_domain, HighMark_gravtree, HighMark_pmperiodic,
  HighMark_pmnonperiodic,  HighMark_gasdensity, HighMark_hydro, HighMark_GasGrad;

#ifdef TURB_DRIVING
extern size_t HighMark_turbpower;
#endif
extern int TreeReconstructFlag;
extern int TreeMomentsStaleFlag; /*!< flag to refresh tree node moments without a full tree rebuild, e.g. after star formation or sink mass changes */
extern int GlobFlag;
extern char DumpFlag;
extern int NeedToWakeupParticles;
extern int NeedToWakeupParticles_local;

extern int NumPart;		/*!< number of particles on the LOCAL processor */
extern int N_gas;		/*!< number of gas particles on the LOCAL processor  */
#ifdef SINK_WIND_SPAWN
extern double Max_Unspawned_MassUnits_fromSink;
#endif

extern long long Ntype[6];	/*!< total number of particles of each type */
extern int NtypeLocal[6];	/*!< local number of particles of each type */
extern gsl_rng *random_generator;	/*!< the random number generator used */
extern int Gas_split;           /*!< current number of newly-spawned gas particles outside block */
#ifdef GALSF
extern int Stars_converted;	/*!< current number of star particles in gas particle block */
#endif

extern double TimeOfLastTreeConstruction;	/*!< holds what it says */
extern std::vector<int> Ngblist;		/*!< Buffer to hold indices of neighbours retrieved by the neighbour search routines */
extern double *R2ngblist;
extern double DomainCorner[3], DomainCenter[3], DomainLen, DomainFac;
extern int *DomainStartList, *DomainEndList;
extern double *DomainWork;
extern int *DomainCount;
extern int *DomainCountGas;
extern int *DomainTask;
extern int *DomainNodeIndex;
extern int *DomainList, DomainNumChanged;
extern peanokey *Key, *KeySorted;


#ifdef HERMITE_INTEGRATION
extern int HermiteOnlyFlag;     /*!< flag to only do Hermite integration for applicable particles (ie. stars) in the gravity routine - set =1 on the first prediction pass and =2 on the second correction pass */
#endif

/* RT_CHEM_PHOTOION arrays and CRFLUID_EVOLVE_SPECTRUM arrays are now in the All struct
   (global_data_all_struct.h) for GPU visibility via __managed__ All. */


extern int MaxNodes;        /*!< maximum allowed number of internal nodes */
extern int Numnodestree;    /*!< number of (internal) nodes in each tree */

extern int *Nextnode;        /*!< gives next node in tree walk  (nodes array) */
extern int *Father;        /*!< gives parent node in tree (Prenodes array) */

extern int maxThreads;

#ifdef TURB_DRIVING // other global variables for forcing field (need to be carried through all timesteps)
extern double* StOUPhases; // random fluctuating component of the amplitudes
extern double* StAmpl; // relative amplitude for each k
extern double* StAka; // phases (real part)
extern double* StAkb; // phases (imag part)
extern double* StMode; // k vectors
extern int StNModes; // total number of modes
extern integertime StTPrev; // time of last update (to determine when next will be)
extern gsl_rng* StRng; // random number generator key
#endif


#if defined(DM_SIDM)
#define GEOFACTOR_TABLE_LENGTH 1000    /*!< length of the table used for the geometric factor spline */
extern MyDouble GeoFactorTable[GEOFACTOR_TABLE_LENGTH];
#endif
extern int NTopnodes, NTopleaves;

/* variables for input/output , usually only used on process 0 */
extern char ParameterFile[100];    /*!< file name of parameterfile used for starting the simulation */
extern FILE
#ifdef OUTPUT_ADDITIONAL_RUNINFO
*FdTimebin,    /*!< file handle for timebin.txt log-file. */
*FdInfo,       /*!< file handle for info.txt log-file. */
*FdEnergy,     /*!< file handle for energy.txt log-file. */
*FdTimings,    /*!< file handle for timings.txt log-file. */
*FdBalance,    /*!< file handle for balance.txt log-file. */
#ifdef RT_CHEM_PHOTOION
*FdPhotoIonChemStats,        /*!< file handle for photoionization chemistry log-file. */
#endif
#ifdef TURB_DRIVING
*FdTurb,       /*!< file handle for turbulent driving log-file */
#endif
#ifdef GR_TABULATED_COSMOLOGY
*FdDE,         /*!< file handle for darkenergy.txt log-file. */
#endif
#endif
*FdCPU;        /*!< file handle for cpu.txt log-file. */
#ifdef GALSF
extern FILE *FdSfr;        /*!< file handle for star formation log-file. */
#endif
#ifdef GALSF_FB_FIRE_RT_LOCALRP
extern FILE *FdMomWinds;    /*!< file handle for local photon-momentum subgrid model log-file */
#endif
#ifdef GALSF_FB_FIRE_RT_HIIHEATING
extern FILE *FdHIIHeating;    /*!< file handle for local HII model log-file */
#endif
#ifdef GALSF_FB_MECHANICAL
extern FILE *FdSNeFBLogFile;    /*!< file handle for mechanical feedback log-file */
#endif
#ifdef SINK_PARTICLES
extern FILE *FdSinks;    /*!< file handle for sinks.txt log-file. */
#endif
#ifdef OUTPUT_SINK_ACCRETION_HIST
extern FILE *FdSinkSwallowDetails;
#endif
#ifdef OUTPUT_SINK_FORMATION_PROPS
extern FILE *FdSinkFormationDetails;
#endif
#if (defined(OUTPUT_ADDITIONAL_RUNINFO) || defined(SINK_OUTPUT_MOREINFO)) && defined(SINK_PARTICLES)
extern FILE *FdSinksDetails;
#ifdef SINK_OUTPUT_MOREINFO
extern FILE *FdSinkMergerDetails;
#endif
#ifdef SINK_WIND_KICK
extern FILE *FdSinkWindDetails;
#endif
#endif


extern double DriftTable[DRIFT_TABLE_LENGTH]; /*! table for the cosmological drift factors */
extern double GravKickTable[DRIFT_TABLE_LENGTH]; /*! table for the cosmological kick factor for gravitational forces */
extern void *CommBuffer;    /*!< points to communication buffer, which is used at a few places */


#ifdef SUBFIND
extern int GrNr;
extern int NumPartGroup;
#endif

extern peanokey *DomainKeyBuf;

/* Tree variables */
extern long Nexport, Nimport;
extern int BufferCollisionFlag;
extern int BufferFullFlag;
extern int NextParticle;
extern int NextJ;
extern int TimerFlag;




extern struct topnode_data
{
  peanokey Size;
  peanokey StartKey;
  long long Count;
  MyFloat GravCost;
  int Daughter;
  int Pstart;
  int Blocks;
  int Leaf;
} *TopNodes;



#ifdef SUBFIND
extern struct Subfind_DensityOtherPropsEval_data_out
{
    MyOutputFloat M200, R200;
#ifdef SUBFIND_ADDIO_VELDISP
    MyOutputFloat V200[3], Disp200;
#endif
#ifdef SUBFIND_ADDIO_BARYONS
    MyOutputFloat gas_mass, star_mass, temp, xlum;
#endif
}
*Subfind_DensityOtherPropsEval_DataResult, *Subfind_DensityOtherPropsEval_DataOut, *Subfind_DensityOtherPropsEval_GlobalPasser;
#endif



/* Struct definition lives in its own minimal header so GPU-only TUs can include
   it without pulling in MPI, GSL, or HDF5.  allvars.h includes it here so
   all existing code that includes allvars.h sees the type as before. */
#include "global_data_all_struct.h"

/* In GPU TUs (cooling.cc, eos.cc), All is #defined to All_dev (a static
 * __managed__ variable).  Suppress the extern declaration so the macro doesn't
 * expand it to "extern ... All_dev" which would conflict with the static defn.
 * The guard uses 'All' as a defined-macro test rather than __CUDA_ARCH__
 * (which nvcc_wrapper does not reliably define). */
#ifndef All
extern struct global_data_all_processes All;
#endif



#include "particle_data.h"
#include "cell_data.h"

static_assert(std::is_trivially_copyable<particle_data>::value, "particle_data must be trivially copyable for MPI");
static_assert(std::is_trivially_copyable<gas_cell_data>::value, "gas_cell_data must be trivially copyable for MPI");




/* global state of system for checking conservation, etc. */
extern struct state_of_system
{
  double Mass,
    EnergyKin,
    EnergyPot,
    EnergyInt,
    EnergyTot,
    Momentum[4],
    AngMomentum[4],
    CenterOfMass[4],
    MassComp[6],
    EnergyKinComp[6],
    EnergyPotComp[6],
    EnergyIntComp[6],
    EnergyTotComp[6],
    MomentumComp[6][4],
    AngMomentumComp[6][4],
    CenterOfMassComp[6][4];
}
SysState, SysStateAtStart, SysStateAtEnd;


/* Various structures for communication during the gravity computation. */

extern struct data_index
{
  int Task;
  int Index;
  int IndexGet;
}
 *DataIndexTable;		/*!< the particles to be exported are grouped by task-number. This table allows the results to be disentangled again and to be assigned to the correct particle */


extern struct data_nodelist
{
  int NodeList[NODELISTLENGTH];
}
*DataNodeList;


extern struct gravdata_in
{
    int Type;
    Vec3<MyFloat> Pos;
    MyFloat Soft;
    MyFloat Mass;
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(COMPUTE_JERK_IN_GRAVTREE) || defined(SINK_DYNFRICTION_FROMTREE)
    Vec3<MyFloat> Vel;
#endif
#if defined(SINK_DYNFRICTION_FROMTREE)
    MyFloat Sink_Mass;
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL)
    MyFloat AGS_zeta;
#endif
#ifdef SINGLE_STAR_FIND_BINARIES
    MyFloat Min_Sink_OrbitalTime;   /*!<orbital time for binary */
    Vec3<MyDouble> comp_dx;        /*!< position of binary companion */
    Vec3<MyDouble> comp_dv;        /*!< velocity of binary companion */
    MyDouble comp_Mass;         /*!< mass of binary companion */
    int is_in_a_binary;
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    SymmetricTensor2<MyFloat> tidal_tensorps_prevstep;
#endif
#if (SINGLE_STAR_TIMESTEPPING > 0)
    int SuperTimestepFlag;  /*!< 2 if allowed to super-timestep, 1 if a candidate for super-timestepping, 0 otherwise */
#endif
    MyFloat OldAcc;
    int NodeList[NODELISTLENGTH];
}
 *GravDataIn,			/*!< holds particle data to be exported to other processors */
 *GravDataGet;			/*!< holds particle data imported from other processors */


extern struct gravdata_out
{
    Vec3<MyDouble> Acc;
#ifdef RT_USE_TREECOL_FOR_NH
    MyDouble ColumnDensityBins[RT_USE_TREECOL_FOR_NH];
#endif
#ifdef COUNT_MASS_IN_GRAVTREE
    MyDouble TreeMass;
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    MyDouble SubGrid_CosmicRayEnergyDensity;
#endif
#ifdef RT_OTVET
    SymmetricTensor2<MyDouble> ET[N_RT_FREQ_BINS];
#endif
#ifdef GALSF_FB_FIRE_RT_LONGRANGE
    MyDouble Rad_Flux_UV;
    MyDouble Rad_Flux_EUV;
#endif
#if defined(RT_USE_GRAVTREE_SAVE_RAD_ENERGY)
    MyDouble Rad_E_gamma[N_RT_FREQ_BINS];
#endif
#if defined(RT_USE_GRAVTREE_SAVE_RAD_FLUX)
    Vec3<MyDouble> Rad_Flux[N_RT_FREQ_BINS];
#endif
#ifdef CHIMES_STELLAR_FLUXES
    double Chimes_G0[CHIMES_LOCAL_UV_NBINS];
    double Chimes_fluxPhotIon[CHIMES_LOCAL_UV_NBINS];
#endif
#ifdef SINK_COMPTON_HEATING
    MyDouble Rad_Flux_AGN;
#endif
#ifdef SINK_SEED_FROM_LOCALGAS_TOTALMENCCRITERIA
    MyDouble MencInRcrit;
#endif
#ifdef EVALPOTENTIAL
    MyDouble Potential;
#endif
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
    SymmetricTensor2<MyFloat> tidal_tensorps;
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    MyFloat tidal_zeta;
#endif
#endif
#ifdef COMPUTE_JERK_IN_GRAVTREE
    Vec3<MyDouble> GravJerk;
#endif
#ifdef SINK_CALC_DISTANCES
    MyFloat Min_Distance_to_Sink;
    Vec3<MyFloat> Min_xyz_to_Sink;
#ifdef SPECIAL_POINT_MOTION
    Vec3<MyFloat> vel_of_nearest_special;
    Vec3<MyFloat> acc_of_nearest_special;
#ifdef SPECIAL_POINT_WEIGHTED_MOTION
    MyFloat weight_sum_for_special_point_smoothing;
#endif
#endif
#ifdef SINGLE_STAR_FIND_BINARIES
    MyFloat Min_Sink_OrbitalTime; //orbital time for binary
    Vec3<MyDouble> comp_dx; //position of binary companion
    Vec3<MyDouble> comp_dv; //velocity of binary companion
    MyDouble comp_Mass; //mass of binary companion
    int is_in_a_binary; // 1 if star is in a binary, 0 otherwise
#endif
#ifdef SINGLE_STAR_TIMESTEPPING
    MyFloat Min_Sink_Freefall_time;    // minimum value of sqrt(R^3 / G(M_SINK + M_particle)) as calculated from the tree-walk
    MyFloat Min_Sink_Approach_Time; // smallest approach time t_a = |v_radial|/r
#if (SINGLE_STAR_TIMESTEPPING > 0)
    SymmetricTensor2<MyFloat> COM_tidal_tensorps; //tidal tensor evaluated at the center of mass without contribution from the companion
    MyDouble COM_GravAccel[3]; //gravitational acceleration evaluated at the center of mass without contribution from the companion
    int COM_calc_flag; //flag that tells whether this was only a rerun to get the acceleration ad the tidal tenor at the center of mass of a binary
    int SuperTimestepFlag; // 2 if allowed to super-timestep, 1 if a candidate for super-timestepping, 0 otherwise
#endif
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
    MyFloat Min_Sink_FeedbackTime; // minimum time for feedback to arrive from a star
#endif    
#endif
#endif
}
 *GravDataResult,		/*!< holds the partial results computed for imported particles. Note: We use GravDataResult = GravDataGet, such that the result replaces the imported data */
 *GravDataOut;			/*!< holds partial results received from other processors. This will overwrite the GravDataIn array */


extern struct potdata_out
{
  MyDouble Potential;
}
 *PotDataResult,		/*!< holds the partial results computed for imported particles. Note: We use GravDataResult = GravDataGet, such that the result replaces the imported data */
 *PotDataOut;			/*!< holds partial results received from other processors. This will overwrite the GravDataIn array */


extern struct info_block
{
  char label[4];
  char type[8];
  int ndim;
  int is_present[6];
}
*InfoBlock;



/* this structure needs to be defined here, because routines for feedback event rates, etc, are shared among files */
extern struct addFB_evaluate_data_in_
{
    Vec3<MyDouble> Pos, Vel; MyDouble Msne, unit_mom_SNe;
    MyFloat KernelRadius, V_i, SNe_v_ejecta;
#ifdef GALSF_FB_MECHANICAL
    MyFloat Area_weighted_sum[AREA_WEIGHTED_SUM_ELEMENTS];
#endif
#ifdef METALS
    MyDouble yields[NUM_METAL_SPECIES];
#endif
    int NodeList[NODELISTLENGTH];
}
*addFB_evaluate_DataIn_, *addFB_evaluate_DataGet_;





/* Header for the standard file format */
extern struct io_header
{
  int npart[6];			    /*!< number of particles of each type in this file */
  double mass[6];           /*!< mass of particles of each type. If 0, then the masses are explicitly stored in the mass-block of the snapshot file, otherwise they are omitted */
  double time;			    /*!< time of snapshot file */
  double redshift;		    /*!< redshift of snapshot file */
  int flag_sfr;			    /*!< flags whether the simulation was including star formation */
  int flag_feedback;		/*!< flags whether feedback was included (obsolete) */
  unsigned int npartTotal[6];   /*!< total number of particles of each type in this snapshot. This can be different from npart if one is dealing with a multi-file snapshot. */
  int flag_cooling;		    /*!< flags whether cooling was included  */
  int num_files;		    /*!< number of files in multi-file snapshot */
  double BoxSize;		    /*!< box-size of simulation in case periodic boundaries were used */
  double OmegaMatter;       /*!< matter density in units of critical density */
  double OmegaLambda;		/*!< cosmological constant parameter */
  double HubbleParam;		/*!< Hubble parameter in units of 100 km/sec/Mpc */
  int flag_stellarage;		/*!< flags whether the file contains formation times of star particles */
  int flag_metals;		    /*!< flags whether the file contains metallicity values for gas and star particles */

  unsigned int npartTotalHighWord[6];   /*!< High word of the total number of particles of each type (needed to combine with npartTotal to allow >2^31 particles of a given type) */
  int flag_entropy_instead_u; /*!< flag here strictly for historical compatibility with unformatted binary files from GADGET-3 era formats, which expect this flag to exist. this does nothing in gizmo */
  int flag_doubleprecision; /*!< flags that snapshot contains double-precision instead of single precision */

  int flag_ic_info;             /*!< flag to inform whether IC files are generated with ordinary Zeldovich approximation,
                                     or whether they ocontains 2nd order lagrangian perturbation theory initial conditions.
                                     For snapshots files, the value informs whether the simulation was evolved from
                                     Zeldoch or 2lpt ICs. Encoding is as follows:
                                        FLAG_ZELDOVICH_ICS     (1)   - IC file based on Zeldovich
                                        FLAG_SECOND_ORDER_ICS  (2)   - Special IC-file containing 2lpt masses
                                        FLAG_EVOLVED_ZELDOVICH (3)   - snapshot evolved from Zeldovich ICs
                                        FLAG_EVOLVED_2LPT      (4)   - snapshot evolved from 2lpt ICs
                                        FLAG_NORMALICS_2LPT    (5)   - standard gadget file format with 2lpt ICs
                                     All other values, including 0 are interpreted as "don't know" for backwards compatability.
                                 */
  float lpt_scalingfactor;      /*!< scaling factor for 2lpt initial conditions */

  char fill[18];		        /*!< fills to 256 Bytes */
  char names[15][2];
}
header;				/*!< holds header for snapshot files */







enum iofields
{ IO_POS,
  IO_VEL,
  IO_ID,
  IO_CHILD_ID,
  IO_GENERATION_ID,
  IO_MASS,
  IO_U,
  IO_RHO,
  IO_NE,
  IO_NH,
  IO_KERNELRADIUS,
  IO_SFR,
  IO_AGE,
  IO_GRAINSIZE,
  IO_DUST_TO_GAS, 
  IO_GRAINTYPE,
  IO_HSMS,
  IO_Z,
  IO_DUSTCHEMZMET,
  IO_DUSTCHEMSPECIESMET,
  IO_ISMDUSTCHEMMOL,
  IO_MACHNUM,
  IO_DUSTCHEMGRAINBINNUMBERS,
  IO_DUSTCHEMGRAINBINSLOPES,
  IO_DUSTCHEMGRAINBINMASS,
  IO_DUSTCHEM_COAG_MASSRATE,
  IO_DUSTCHEM_SHAT_MASSRATE,
  IO_SINKMASS,
  IO_SINKMASSALPHA,
  IO_SINK_ANGMOM,
  IO_SINKMDOT,
  IO_SINKDUSTMASSACC,
  IO_R_PROTOSTAR,
  IO_MASS_D_PROTOSTAR,
  IO_ZAMS_MASS,
  IO_STAGE_PROTOSTAR,
  IO_AGE_PROTOSTAR,
  IO_LUM_SINGLESTAR,
  IO_SINKPROGS,
  IO_SINK_DIST,
  IO_ACRB,
  IO_SINKRAD,
  IO_SINK_FORM_MASS,
  IO_POT,
  IO_ACCEL,
  IO_HYDROACCEL,
  IO_HII,
  IO_HeI,
  IO_HeII,
  IO_UNSPMASS,
  IO_CRATE,
  IO_HRATE,
  IO_NHRATE,
  IO_HHRATE,
  IO_MCRATE,
  IO_DCRATE,
  IO_PHRATE,
  IO_DTENTR,
  IO_TSTP,
  IO_BFLD,
  IO_AMBIPOLAR,
  IO_OHMIC,
  IO_HALL,
  IO_IMF,
  IO_COSMICRAY_ENERGY,
  IO_COSMICRAY_KAPPA,
  IO_COSMICRAY_ALFVEN,
  IO_COSMICRAY_SLOPES,
  IO_DIVB,
  IO_ABVC,
  IO_AMDC,
  IO_PHI,
  IO_GRADPHI,
  IO_GRADRHO,
  IO_GRADVEL,
  IO_GRADMAG,
  IO_COOLRATE,
  IO_TIDALTENSORPS,
  IO_EOSTEMP,
  IO_EOSABAR,
  IO_EOSYE,
#ifdef NUCLEAR_NETWORK
  IO_NUCLEAR_COMPOSITION,
#endif
  IO_PRESSURE,
  IO_EOSCS,
  IO_EOS_STRESS_TENSOR,
  IO_CBE_MOMENTS,
  IO_EOSCOMP,
  IO_EOSPHASE,
  IO_PARTVEL,
  IO_RADGAMMA,
  IO_RAD_TEMP,
  IO_RAD_OPACITY,
  IO_DUST_TEMP,
  IO_RAD_ACCEL,
  IO_RAD_FLUX,
  IO_EDDINGTON_TENSOR,
  IO_VDIV,
  IO_VORT,
  IO_DELAYTIME,
  IO_SOFT,
  IO_AGS_HKERN,
  IO_AGS_RHO,
  IO_AGS_QPT,
  IO_AGS_PSI_RE,
  IO_AGS_PSI_IM,
  IO_AGS_ZETA,
  IO_VSTURB_DISS,
  IO_VSTURB_DRIVE,
  IO_grHI,
  IO_grHII,
  IO_grHM,
  IO_grHeI,
  IO_grHeII,
  IO_grHeIII,
  IO_grH2I,
  IO_grH2II,
  IO_grDI,
  IO_grDII,
  IO_grHDI,
  IO_OSTAR,
  IO_DTOSTAR,
  IO_TURB_DYNAMIC_COEFF,
  IO_TURB_DIFF_COEFF,
  IO_DYNERROR,
  IO_DYNERRORDEFAULT,
  IO_CHIMES_ABUNDANCES,
  IO_CHIMES_MU,
  IO_CHIMES_REDUCED,
  IO_CHIMES_NH,
  IO_CHIMES_STAR_SIGMA,
  IO_CHIMES_FLUX_G0,
  IO_CHIMES_FLUX_ION,
  IO_DENS_AROUND_STAR,
  IO_DELAY_TIME_HII,
  IO_MOLECULARFRACTION,
  IO_SHOCKMACHNUM,
  IO_LASTENTRY			/* This should be kept - it signals the end of the list */
};


enum siofields
{ SIO_GLEN,
  SIO_GOFF,
  SIO_MTOT,
  SIO_GPOS,
  SIO_DELTA_MSUB,
  SIO_DELTA_RSUB,
  SIO_DELTA_DISPSUB,
  SIO_DELTA_MGASSUB,
  SIO_DELTA_MSTSUB,
  SIO_DELTA_TEMPSUB,
  SIO_DELTA_LXSUB,
  SIO_NCON,
  SIO_MCON,
  SIO_BGPOS,
  SIO_BGMTOP,
  SIO_BGRTOP,
  SIO_NSUB,
  SIO_FSUB,
  SIO_SLEN,
  SIO_SOFF,
  SIO_PFOF,
  SIO_MSUB,
  SIO_SPOS,
  SIO_SVEL,
  SIO_SCM,
  SIO_SPIN,
  SIO_DSUB,
  SIO_VMAX,
  SIO_RVMAX,
  SIO_RHMS,
  SIO_MBID,
  SIO_GRNR,
  SIO_SMST,
  SIO_SLUM,
  SIO_SLATT,
  SIO_SLOBS,
  SIO_HALODUST,
  SIO_SAGE,
  SIO_SZ,
  SIO_SSFR,
  SIO_PPOS,
  SIO_PVEL,
  SIO_PTYP,
  SIO_PMAS,
  SIO_PID,

  SIO_LASTENTRY
};


/* Variables for Tree */
/* Tree Node structure (note, the ALIGN(32) directive will effectively pad the structure size to a multiple of 32 bytes) */
extern ALIGN(32) struct NODE
{
  Vec3<MyFloat> center;		/*!< geometrical center of node */
  MyFloat len;			/*!< sidelength of treenode */

  union
  {
    int suns[8];		/*!< temporary pointers to daughter nodes */
    struct
    {
      Vec3<MyFloat> s;		/*!< center of mass of node */
      MyFloat mass;		/*!< mass of node */
      unsigned int bitflags;	/*!< flags certain node properties */
      int sibling;		/*!< this gives the next node in the walk in case the current node can be used */
      int nextnode;		/*!< this gives the next node in case the current node needs to be opened */
      int father;		/*!< this gives the parent node of each node (or -1 if we have the root node) */
    }
    d;
  }
  u;

    double GravCost;
    integertime Ti_current;
    long N_part;   /*!< number of particles+cells in the tree node */
    MyFloat maxsoft;        /*!< hold the maximum gravitational softening of particle in the node */
#if defined(GRAVTREE_CALCULATE_GAS_MASS_IN_NODE)
  MyFloat gasmass;
#endif
#ifdef RT_USE_GRAVTREE
  MyFloat stellar_lum[N_RT_FREQ_BINS]; /*!< luminosity in the node*/
#ifdef CHIMES_STELLAR_FLUXES
  double chimes_stellar_lum_G0[CHIMES_LOCAL_UV_NBINS];
  double chimes_stellar_lum_ion[CHIMES_LOCAL_UV_NBINS];
#endif
#endif

#ifdef RT_SEPARATELY_TRACK_LUMPOS
    Vec3<MyFloat> rt_source_lum_s; /*!< luminosity-weighted position of RT sources in the node */
#endif
#ifdef SINK_PHOTONMOMENTUM
    MyFloat sink_lum;		    /*!< luminosity of BHs in the node */
    Vec3<MyFloat> sink_lum_grad;	/*!< gradient vector for gas around sink (for angular dependence) */
#endif
    
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    MyFloat cr_injection;
#endif

#ifdef SINK_CALC_DISTANCES
  MyFloat sink_mass;      /*!< holds the sink mass in the node.  Used for calculating tree based dist to closest sink */
  Vec3<MyFloat> sink_pos;    /*!< holds the mass-weighted position of the the actual sink particles within the node */
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SPECIAL_POINT_MOTION)
    Vec3<MyFloat> sink_vel;    /*!< holds the mass-weighted avg. velocity of sink particles in the node */
#endif
#if defined(SPECIAL_POINT_MOTION)
    Vec3<MyFloat> sink_acc; /*!< holds the mass-weighted avg. acceleration of sink particles in the node */
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SPECIAL_POINT_MOTION)
  int N_SINK;             /*!< holds the number of sink particles in the node. Used for refinement/search criteria */
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
    MyFloat MaxFeedbackVel;
#endif
#endif


#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    SymmetricTensor2<MyFloat> tidal_tensorps_prevstep;
#endif
#ifdef DM_SCALARFIELD_SCREENING
  Vec3<MyFloat> s_dm;
  MyFloat mass_dm;
#endif
}
 *Nodes_base,			/*!< points to the actual memory allocated for the nodes */
 *Nodes;			/*!< this is a pointer used to access the nodes which is shifted such that Nodes[All.MaxPart] gives the first allocated node */


extern struct extNODE
{
  Vec3<MyDouble> dp;
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    Vec3<MyDouble> rt_source_lum_dp;
    Vec3<MyFloat> rt_source_lum_vs;
#endif
#ifdef DM_SCALARFIELD_SCREENING
  Vec3<MyDouble> dp_dm;
  Vec3<MyFloat> vs_dm;
#endif
  Vec3<MyFloat> vs;
  MyFloat vmax;
  MyFloat hmax;			/*!< maximum gas kernel length in node. Only used for gas particles */
  MyFloat divVmax;
  integertime Ti_lastkicked;
  int Flag;
}
 *Extnodes, *Extnodes_base;



#endif  /* ALLVARS_H  - please do not put anything below this line */
