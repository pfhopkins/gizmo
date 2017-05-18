#!/bin/bash            # this line only there to enable syntax highlighting in this file

####################################################################################################
#  Enable/Disable compile-time options as needed: this is where you determine how the code will act
#  From the list below, please activate/deactivate the
#       options that apply to your run. If you modify any of these options,
#       make sure that you recompile the whole code by typing "make clean; make".
#
# This file was originally part of the GADGET3 code developed by
#   Volker Springel (volker.springel@h-its.org). The code has been modified
#   substantially by Phil Hopkins (phopkins@caltech.edu) for GIZMO (to add new modules and clean
#   up the naming conventions and changed many of them to match the new GIZMO conventions)
#
####################################################################################################



####################################################################################################
# --------------------------------------- Boundary Conditions & Dimensions
####################################################################################################
PERIODIC                        # Use this if periodic boundaries are needed (otherwise open boundaries are assumed)
#BND_PARTICLES                  # particles with ID=0 are forced in place (their accelerations are set =0):
                                # use for special boundary conditions where these particles represent fixed "walls"
#LONG_X=140                     # modify box dimensions (non-square periodic box): multiply X (PERIODIC and NOGRAVITY required)
#LONG_Y=1                       # modify box dimensions (non-square periodic box): multiply Y
#LONG_Z=1                       # modify box dimensions (non-square periodic box): multiply Z
#REFLECT_BND_X                  # make the x-boundary reflecting (assumes a box 0<x<1, unless PERIODIC is set)
#REFLECT_BND_Y                  # make the y-boundary reflecting (assumes a box 0<y<1, unless PERIODIC is set)
#REFLECT_BND_Z                  # make the z-boundary reflecting (assumes a box 0<z<1, unless PERIODIC is set)
#SHEARING_BOX=1                 # shearing box boundaries: 1=r-z sheet (r,z,phi coordinates), 2=r-phi sheet (r,phi,z), 3=r-phi-z box, 4=as 3, with vertical gravity
#SHEARING_BOX_Q=(3./2.)         # shearing box q=-dlnOmega/dlnr; will default to 3/2 (Keplerian) if not set
#ONEDIM                         # Switch for 1D test problems: code only follows the x-line. requires NOGRAVITY, and all y=z=0
#TWODIMS                        # Switch for 2D test problems: code only follows the xy-plane. requires NOGRAVITY, and all z=0.
####################################################################################################



####################################################################################################
# --------------------------------------- Hydro solver method
####################################################################################################
HYDRO_MESHLESS_FINITE_MASS      # Lagrangian (constant-mass) finite-volume Godunov method
#HYDRO_MESHLESS_FINITE_VOLUME   # Moving (quasi-Lagrangian) finite-volume Godunov method
## -----------------------------------------------------------------------------------------------------
# --------------------------------------- SPH methods:
#SPHEQ_DENSITY_INDEPENDENT_SPH  # force SPH to use the 'pressure-sph' formulation ("modern" SPH)
#SPHEQ_TRADITIONAL_SPH          # force SPH to use the 'density-sph' (GADGET-2 & GASOLINE SPH)
# --------------------------------------- SPH artificial diffusion options (use with SPH; not relevant for Godunov/Mesh modes)
#SPHAV_DISABLE_CD10_ARTVISC     # Disable Cullen & Dehnen 2010 'inviscid sph' (viscosity suppression outside shocks); just use Balsara switch
#SPHAV_DISABLE_PM_CONDUCTIVITY  # Disable mixing entropy (J.Read's improved Price-Monaghan conductivity with Cullen-Dehnen switches)
## -----------------------------------------------------------------------------------------------------
# --------------------------------------- Kernel Options
#KERNEL_FUNCTION=3              # Choose the kernel function (2=quadratic peak, 3=cubic spline [default], 4=quartic spline, 5=quintic spline, 6=Wendland C2, 7=Wendland C4, 8=2-part quadratic)
####################################################################################################



####################################################################################################
# --------------------------------------- Additional Fluid Physics
####################################################################################################
##-----------------------------------------------------------------------------------------------------
#---------------------------------------- Gas Equations-of-State
#EOS_GAMMA=(5.0/3.0)            # Polytropic Index of Gas (for an ideal gas law): if not set and no other (more complex) EOS set, defaults to GAMMA=5/3
#EOS_HELMHOLTZ                  # Use Timmes & Swesty 2000 EOS (for e.g. stellar or degenerate equations of state)
## -----------------------------------------------------------------------------------------------------
# --------------------------------- Magneto-Hydrodynamics
# ---------------------------------  these modules are public, but if used, the user should also cite the MHD-specific GIZMO methods paper
# ---------------------------------  (Hopkins 2015: 'Accurate, Meshless Methods for Magneto-Hydrodynamics') as well as the standard GIZMO paper
#MAGNETIC                       # master switch for MHD, regardless of which Hydro solver is used
#B_SET_IN_PARAMS                # set initial fields (Bx,By,Bz) in parameter file
#MHD_NON_IDEAL                  # enable non-ideal MHD terms: Ohmic resistivity, Hall effect, and ambipolar diffusion (solved explicitly)
#CONSTRAINED_GRADIENT_MHD=1     # use CG method (in addition to cleaning, optional!) to maintain low divB: set this value to control how aggressive the div-reduction is:
                                # 0=minimal (safest), 1=intermediate (recommended), 2=aggressive (less stable), 3+=very aggressive (less stable+more expensive)
##-----------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------
#--------------------------------------- Conduction
#CONDUCTION                     # Thermal conduction solved *explicitly*: isotropic if MAGNETIC off, otherwise anisotropic
#CONDUCTION_SPITZER             # Spitzer conductivity accounting for saturation: otherwise conduction coefficient is constant
##-----------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------
#--------------------------------------- Viscosity
#VISCOSITY                      # Navier-stokes equations solved *explicitly*: isotropic coefficients if MAGNETIC off, otherwise anisotropic
#VISCOSITY_BRAGINSKII           # Braginskii viscosity tensor for ideal MHD
##-----------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------
#--------------------------------------- Radiative Cooling physics (mostly geared towards galactic/extragalactic cooling)
#--------------------------- These modules were originally developed for a combination of -proprietary- physics modules. they can only be used with
#--------------------------- permission from the authors. email P. Hopkins to obtain the relevant permissions for the cooling routines of interest.
#COOLING                        # enables radiative cooling and heating: if GALSF, also external UV background read from file "TREECOOL"
#COOL_LOW_TEMPERATURES          # allow fine-structure and molecular cooling to ~10 K; account for optical thickness and line-trapping effects with proper opacities
#COOL_METAL_LINES_BY_SPECIES    # use full multi-species-dependent cooling tables ( https://dl.dropbox.com/u/16659252/spcool_tables.tgz )
#GRACKLE                        # enable GRACKLE: cooling+chemistry package (requires COOLING above; https://grackle.readthedocs.org/en/latest )
#GRACKLE_CHEMISTRY=1            # choose GRACKLE cooling chemistry: (0)=tabular, (1)=Atomic, (2)=(1)+H2+H2I+H2II, (3)=(2)+DI+DII+HD
##-----------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------
#--------------------------------------- Smagorinsky Turbulent Eddy Diffusion Model
#---------------------------------------- (this is developed by P. Hopkins as part of the FIRE package: the same FIRE authorship & approval policies apply, see below)
#TURB_DIFF_METALS               # turbulent diffusion of metals (passive scalars)
#TURB_DIFF_ENERGY               # turbulent diffusion of internal energy (conduction with effective turbulent coefficients)
#TURB_DIFF_VELOCITY             # turbulent diffusion of momentum (viscosity with effective turbulent coefficients)
##-----------------------------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------
# --------------------------------------- Aerodynamic Particles
# --------------------------------------- (this is developed by P. Hopkins as part of the FIRE package: the same FIRE authorship & approval policies apply, see below)
#GRAIN_FLUID                    # aerodynamically-coupled grains (particle type 3 are grains); default is Stokes drag
#GRAIN_EPSTEIN=1                # uses the cross section for molecular hydrogen (times this number) to calculate Epstein drag
#GRAIN_BACKREACTION             # account for momentum of grains pushing back on gas (from drag terms)
#GRAIN_LORENTZFORCE             # charged grains feel Lorentz forces (requires MAGNETIC)
#GRAIN_COLLISIONS               # model collisions between grains (super-particles; so this is stochastic)
##-----------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------
#---------------------------------------- Cosmic Rays
#---------------------------------------- (this is developed by P. Hopkins as part of the FIRE package: the same FIRE authorship & approval policies apply, see below)
#COSMIC_RAYS                    # two-fluid medium with CRs as an ultrarelativistic fluid: heating/cooling, anisotropic diffusion, streaming, injection by SNe
#COSMIC_RAYS_M1=(500.)          # solve the CR transport in the M1 limit [second-order expansion of the collisionless boltzmann eqn]; value here is the streaming speed in code units
#COSMIC_RAYS_DISABLE_STREAMING  # turn off CR streaming (propagation is purely advective+diffusion; warning: this can severely under-estimate CR losses to Alfven waves)
#COSMIC_RAYS_DISABLE_DIFFUSION  # turn off CR diffusion (leaves streaming intact, simply disables 'microscopic+turbulent' CR diffusion terms)
#COSMIC_RAYS_DISABLE_COOLING    # turn off CR heating/cooling interactions with gas (catastrophic losses, hadronic interactions, etc; only adiabatic PdV work terms remain)
#COSMIC_RAYS_DIFFUSION_CONSTANT # replaces physical CR diffusion with constant coefficient (equal to value of CosmicRayDiffusionCoeff in code units); turn off streaming to make this the ONLY transport
##-----------------------------------------------------------------------------------------------------
####################################################################################################



####################################################################################################
# ------------------------------------- Driven turbulence (for turbulence tests, large-eddy sims)
# ------------------------------- users of these routines should cite Bauer & Springel 2012, MNRAS, 423, 3102, and thank A. Bauer for providing the core algorithms
####################################################################################################
#TURB_DRIVING                   # turns on turbulent driving/stirring. see begrun for parameters that must be set
#POWERSPEC_GRID=128             # activates on-the-fly calculation of the turbulent velocity, vorticity, and smoothed-velocity power spectra
#ADJ_BOX_POWERSPEC              # compiles in a code module that allows via restart-flag 6 the calculation of a gas velocity power spectrum of a snapshot with an adjustable box (user defined center and size)
####################################################################################################



####################################################################################################
## ------------------------ Gravity & Cosmological Integration Options ---------------------------------
####################################################################################################
# --------------------------------------- TreePM Options (recommended for cosmological sims)
#PMGRID=512                     # COSMO enable: resolution of particle-mesh grid
#PM_PLACEHIGHRESREGION=1+2+16   # COSMO enable: particle types to place high-res PMGRID around
#PM_HIRES_REGION_CLIPPING=1000  # for stability: clips particles that escape the hires region in zoom/isolated sims
#PM_HIRES_REGION_CLIPDM         # split low-res DM particles that enter high-res region (completely surrounded by high-res)
#MULTIPLEDOMAINS=64             # Multi-Domain option for the top-tree level: iso=16,COSMO=64-128
## -----------------------------------------------------------------------------------------------------
# ---------------------------------------- Adaptive Grav. Softening (including Lagrangian conservation terms!)
#ADAPTIVE_GRAVSOFT_FORGAS       # allows variable softening length (=Hsml) for gas particles
#ADAPTIVE_GRAVSOFT_FORALL=100   # enable adaptive gravitational softening lengths for all particle types
                                # (ADAPTIVE_GRAVSOFT_FORGAS should be disabled). the softening is set to the distance
                                # enclosing a neighbor number set in the parameter file. baryons search for other baryons,
                                # dm for dm, sidm for sidm, etc. If set to numerical value, the maximum softening is this times All.ForceSoftening[for appropriate particle type]
## -----------------------------------------------------------------------------------------------------
#NOGRAVITY                      # turn off self-gravity (compatible with analytic_gravity)
#GRAVITY_NOT_PERIODIC           # self-gravity is not periodic, even though the rest of the box is periodic
## -----------------------------------------------------------------------------------------------------
#ANALYTIC_GRAVITY               # Specific analytic gravitational force to use instead of/with self-gravity. If set to a numerical value
                                #  > 0 (e.g. =1), then BH_CALC_DISTANCES will be enabled, and it will use the nearest BH particle as the center for analytic gravity computations
                                #  (edit "gravity/analytic_gravity.h" to actually assign the analytic gravitational forces)
##-----------------------------------------------------------------------------------------------------
#--------------------------------------- Self-Interacting DM (Rocha et al. 2012)
#-------------------------------- use of these routines requires explicit pre-approval by developers
#--------------------------------    P. Hopkins & J. Bullock or M. Boylan-Kolchin (acting for M. Rocha)
#SIDM=2                         # Self-interacting particle types (specify the particle types which are self-interacting DM
                                # with a bit mask, as for PM_PLACEHIGHRESREGION above (see description)
                                # (previous "DMDISK_INTERACTIONS" is identical to setting SIDM=2+4)
##-----------------------------------------------------------------------------------------------------
#--------------------------------------- SCF (potential expansion in basis functions)
#-------------------------------- use of these routines requires explicit pre-approval by developers M. Vogelsberger & L. Hernquist
#SCFPOTENTIAL                   # turn SCF on/off
#SCF_HYBRID=1                   # =1:tree:stars<->stars,DM<->DM,stars->DM/SCF:stars<-DM =2:tree:stars<->stars,stars->DM/SCF:stars<-DM,DM<->DM
#SCF_HQ_MASS=95.2401            # mass of halo of expansion basis
#SCF_HQ_A=29.7754               # scale length of halo of expansion basis
#SCF_NEFF=5000000               # effective particle number in halo
#SCF_NMAX=1                     # maximum n for expansion cut-off
#SCF_LMAX=1                     # maximum l for expansion cut-off
#SCF_SCALEFAC                   # readin scale factors for coefficients from file scf_scalefac.dat
##-----------------------------------------------------------------------------------------------------
#SCALARFIELD                    # Gravity is mediated by a long-range scalar field instead of DE or DM
##-----------------------------------------------------------------------------------------------------
#--------------------------------------- Dark energy (flags to allow complicated equations of state)
#DARKENERGY                     # enables Dark Energy with non-trivial equations-of-state (master switch)
#TIMEDEPDE                      # read w(z) from a DE file (pre-tabulated)
#EXTERNALHUBBLE                 # reads the hubble function from the DE file (pre-tabulated)
#TIMEDEPGRAV                    # rescales H and G according to DE model (pre-tabulated)
##-----------------------------------------------------------------------------------------------------
#------------------------------- Fine-grained phase space structure analysis (M. Vogelsberger)
#-------------------------------- use of these routines requires explicit pre-approval by developer M. Vogelsberger
#DISTORTIONTENSORPS             # main switch: integrate phase-space distortion tensor
#OUTPUT_DISTORTIONTENSORPS      # write phase-space distortion tensor to snapshot
#OUTPUT_TIDALTENSORPS           # write configuration-space tidal tensor to snapshot
#OUTPUT_LAST_CAUSTIC            # write info on last passed caustic to snapshot
#GDE_TYPES=2+4+8+16+32          # track GDE for these types
#GDE_READIC                     # read initial sheet orientation/initial density/initial caustic count from ICs
#GDE_LEAN                       # lean version of GDE
##-----------------------------------------------------------------------------------------------------
#EOS_TRUELOVE_PRESSURE          # adds artificial pressure floor force Jeans length above resolution scale (means you will get the wrong answer, but things will look smooth)
####################################################################################################



####################################################################################################
# --------------------------------------- On the fly FOF groupfinder
# ----------------- This is originally developed as part of GADGET-3 by V. Springel
####################################################################################################
##-----------------------------------------------------------------------------------------------------
#-------------------------------------- Friends-of-friends on-the-fly finder options (source in fof.c)
##-----------------------------------------------------------------------------------------------------
#FOF                                # enable FoF searching on-the-fly and outputs (set parameter LINKLENGTH=x to control LinkingLength; default=0.2)
#FOF_PRIMARY_LINK_TYPES=2           # 2^type for the primary dark matter type
#FOF_SECONDARY_LINK_TYPES=1+16+32   # 2^type for the types linked to nearest primaries
#DENSITY_SPLIT_BY_TYPE=1+2+16+32    # 2^type for whch the densities should be calculated seperately
#FOF_GROUP_MIN_LEN=32               # default is 32
####################################################################################################



####################################################################################################-
#------------------ Galaxy formation / Star formation / Supermassive BH Models (with feedback)
####################################################################################################-
##-----------------------------------------------------------------------------------------------------
#-------------------------------------- Galaxy formation and galactic star formation
##-----------------------------------------------------------------------------------------------------
#---- basic/master switches ---- #
#GALSF                          # master switch for galactic star formation model: enables SF, stellar ages, metals, generations, etc.
#METALS                         # enable metallicities (with multiple species optional) for gas and stars
##GALSF_GENERATIONS=1           # the number of stars a gas particle may spawn (defaults to 1, set otherwise)
##-----------------------------------------------------------------------------------------------------------------------------
#----- old sub-grid models (for large-volume simulations) ---- #
#--------- these are all ultimately variations of the Springel & Hernquist 2005 sub-grid models for the ISM, star formation,
#--------- and stellar winds. their use follows the GADGET-3 use policies. If you are not sure whether you have permission to use them,
#--------- you should contact those authors (Lars Hernquist & Volker Springel)
#------------------------------------------------------------------------------------------------------------------------------
#GALSF_EFFECTIVE_EQS            # 'effective equation of state' model for the ISM and star formation
#GALSF_SUBGRID_WINDS            # sub-grid winds ('kicks' as in Oppenheimer+Dave,Springel+Hernquist,Boothe+Schaye,etc)
#GALSF_SUBGRID_VARIABLEVELOCITY # winds with velocity scaling based on halo properties (Oppenheimer+Dave); req.GALSF_SUBGRID_WINDS
#GALSF_SUBGRID_DMDISPERSION     # wind velocity scaling based on MV 13 paper, as used in Illustris. req.GALSF_SUBGRID_WINDS
#GALSF_WINDS_ISOTROPIC          # forces winds to have a random orientation (works with both subgrid+explicit winds)
#GALSF_WINDS_POLAR              # forces winds to have polar orientation (works for sub-grid winds)
#GALSF_TURNOFF_COOLING_WINDS    # turn off cooling for SNe-heated particles (as Stinson+ GASOLINE model; requires GALSF_FB_SNE_HEATING; use by permission of developer P. Hopkins)
#GALSF_GASOLINE_RADHEATING      # heat gas with luminosity from young stars (as Stinson+ 2013 GASOLINE model; requires GALSF_FB_SNE_HEATING; use by permission of developer P. Hopkins)
##-----------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------
#------ PFH physical models for star formation and feedback: these are the FIRE simulation modules (Hopkins et al. 2014) ------ ##
#--------- their use follows the FIRE authorship policy. Any new project using these physics must first be agreed to by all of the
#--------- core development team of the FIRE simulations: P. Hopkins, E. Quataert, D. Keres, C.A. Faucher-Giguere.
#--------- Papers using these modules must offer co-authorship to the members of the FIRE development team.
##-----------------------------------------------------------------------------------------------------
#FIRE_PHYSICS_DEFAULTS           # enable default set of FIRE physics packages (see details below)
#---- star formation law ---- #
#GALSF_SFR_MOLECULAR_CRITERION	 # estimates molecular fraction in SF-ing gas, only SF from that is allowed
#GALSF_SFR_VIRIAL_SF_CRITERION=0 # only allow star formation in virialized sub-regions (alpha<1) (0/no value='default'; 1=0+Jeans criterion; 2=1+'strict' (zero sf if not bound))
#GALSF_SFR_IMF_VARIATION         # determines the stellar IMF for each particle from the Guszejnov/Hopkins/Hennebelle/Chabrier/Padoan theory
#GALSF_SFR_IMF_SAMPLING          # discretely sample the IMF: simplified model with quantized number of massive stars
#----- physical stellar feedback mechanisms ---- #
#GALSF_FB_GASRETURN              # Paul Torrey's addition for stochastic gas return (modified for continuous return)
#GALSF_FB_HII_HEATING            # gas within HII regions around young stars is photo-heated to 10^4 K
#GALSF_FB_SNE_HEATING=1          # time-dependent explosions from SNe (I & II) in shockwave radii around stars (values: 0=force-gridded in xyz (WRONG-for testing only!); 1=tensor-symmetrized (momentum-conserving; USE ME); 2=no tensor re-normalization [non-conservative!]
#GALSF_FB_RPROCESS_ENRICHMENT=4  # tracks a set of 'dummy' species from neutron-star mergers (set to number: 4=extended model)
#GALSF_FB_RT_PHOTONMOMENTUM      # continuous acceleration from starlight (uses luminosity tree)
#GALSF_FB_LOCAL_UV_HEATING       # use local estimate of spectral information for photoionization and photoelectric heating
#GALSF_FB_RPWIND_LOCAL           # turn on local radiation pressure coupling to gas
##-----------------------------------------------------------------------------------------------------
#----------- deprecated options (most have been combined or optimized into the functions above, here for legacy)
##GALSF_FB_RPWIND_FROMCLUMPS	# clump-find to for wind angles (deprecated; now use density routine to find)
##GALSF_FB_RPWIND_CONTINUOUS	# wind accel term is continuous (more expensive and introduces more artificial dissipation)
##GALSF_FB_RPWIND_DO_IN_SFCALC	# do IR wind loop in SFR routine (allows fof clump-finding, useful for very IR-thick, but slow)
##GALSF_FB_RPWIND_FROMSFR       # drive radiation pressure with gas SFR (instead of default, which is nearby young stars)
##FIRE_UNPROTECT_FROZEN         # by default, FIRE-2 code version is 'frozen' so it cannot be changed by any code updates. this removes the protection, so you will use whatever the newest algorithms in GIZMO are, but use it with CAUTION since the algorithm may NOT agree with the other FIRE runs
##-----------------------------------------------------------------------------------------------------



####################################################################################################-
##-----------------------------------------------------------------------------------------------------
#-------------------------------------- SMBH/AGN stuff; also heavily expanded with PFH models
##-----------------------------------------------------------------------------------------------------
#------ PFH physical models for black hole growth and feedback: these are the FIRE simulation modules, their use follows the same FIRE policy above
#------ The original GADGET-3 BH model (only: BLACK_HOLES,BH_SWALLOWGAS,BH_BONDI,BH_DRAG) follow the GADGET-3 Springel & Hernquist policy above
##-----------------------------------------------------------------------------------------------------
#BLACK_HOLES                    # enables Black-Holes (master switch)
#------ seed models
#BH_HOST_TO_SEED_RATIO=1000     # Min FOF stellar mass for seeding is BH_HOST_TO_SEED_RATIO * All.SeedBlackHoleMass
                                #-  Requires FOF with linking type including star particles (MinFoFMassForNewSeed and massDMpart in the param file are ignored)
#BH_SEED_FROM_STAR_PARTICLE     # star particle on FOF potential minimum gets converted into a BH (default is densest gas particle)
#BH_POPIII_SEEDS                # BHs seeded on-the-fly from dense, low-metallicity gas
#------ accretion models/options
#BH_SWALLOWGAS                  # enables stochastic accretion of gas particles consistent with growth rate of hole
#BH_ALPHADISK_ACCRETION         # gas accreted into 'virtual' alpha-disk, and from there onto the BH
#BH_GRAVCAPTURE_GAS             # accretion determined only by resolved gravitational capture by the BH (for gas particles)
#BH_GRAVCAPTURE_NONGAS          # as BH_GRAVCAPTURE_GAS, but applies to non-gas particles (can be enabled with other accretion models for gas)
#BH_GRAVACCRETION=0             # Gravitational torque accretion estimator from Hopkins & Quataert (2011):
                                #-  [=0] for kinematic B/D decomposition as in Angles-Alcazar et al. (default) and [=1] for approximate f_disk evaluation
##BH_BONDI=0                    # Bondi-Hoyle style accretion model: 0=default (with velocity); 1=dont use gas velocity with sound speed; 2=variable-alpha tweak (Booth & Schaye 2009)
#BH_SUBGRIDBHVARIABILITY        # model variability below resolved dynamical time for BH
#BH_CALC_DISTANCES              # calculate distances for all particles to closest BH for, e.g., refinement
#------ feedback models/options
#BH_BAL_WINDS                   # particles within the BH kernel are given mass, momentum, and energy continuously as high-vel BAL winds
#BH_PHOTONMOMENTUM              # continuous long-range IR radiation pressure acceleration from BH (needs GALSF_FB_RT_PHOTONMOMENTUM)
#BH_HII_HEATING                 # photo-ionization feedback from BH (needs GALSF_FB_HII_HEATING)
#BH_COMPTON_HEATING             # enable Compton heating/cooling from BHs in cooling function (needs BH_PHOTONMOMENTUM)
##BH_THERMALFEEDBACK            # couple a fraction of the BH luminosity into surrounding gas as thermal energy (DiMatteo/Springel/Hernquist model)
##BH_BAL_KICK                   # do BAL winds with stochastic particle kicks at specified velocity (instead of continuous wind solution - requires BH_SWALLOWGAS - )
##BH_BAL_KICK_COLLIMATED        # winds follow the direction of angular momentum within Kernel (only for BH_BAL_KICK winds)
##BH_BAL_KICK_MOMENTUM_FLUX=10  # increase the effective mass-loading of BAL winds to reach the desired momentum flux in units of L_bol/c (needs BH_BAL_KICK)
##BH_WIND_SPAWN
#------------ use the BH_DRAG options only in cosmological cases where M_BH is not >> other particle masses
#BH_DYNFRICTION=0               # apply dynamical friction force to the BHs when m_bh not >> other particle mass: 0=[DM+stars+gas] (default); 1=[DM+stars]; 2=[stars]
##BH_DYNFRICTION_INCREASE=10    # artificially increase dynamic friction by this factor (requires BH_DYNFRICTION)
#BH_INCREASE_DYNAMIC_MASS=100   # Increase the dynamic particle mass by this factor at the time of FOF seeding
##BH_DRAG=1                     # Drag on black-holes due to accretion (w real mdot); set =2 to boost as if BH is accreting at eddington
#------ output options
#BH_OUTPUT_MOREINFO             # output additional info to "blackhole_details"
##-----------------------------------------------------------------------------------------------------
#------------ deprecated or de-bugging options (most have been combined or optimized into the functions above, here for legacy)
#BH_REPOSITION_ON_POTMIN=0      # reposition black hole on potential minimum (requires EVALPOTENTIAL). [set =1 to "jump" onto STARS only]
##DETACH_BLACK_HOLES            # Insert an independent data structure for BHs (currently explicitly depends on SEPARATE_STELLARDOMAINDECOMP)
##BH_SEED_STAR_MASS_FRACTION=0.02 # minimum star mass fraction for BH seeding
##BH_SEED_GROWTH_TESTS          # Currently testing options for BH seeding 
##-----------------------------------------------------------------------------------------------------
#-------------------------------------- AGN-Bubble feedback (D. Sijacki)
#-------------------------------- use of these routines requires explicit pre-approval by developer D. Sijacki
#BUBBLES                        # generation of hot bubbles in an isolated halo or the the biggest halo in the run
#MULTI_BUBBLES                  # hot bubbles in all haloes above certain mass threshold (works only with FOF and without BUBBLES)
#EBUB_PROPTO_BHAR               # Energy content of the bubbles with cosmic time evolves as an integrated BHAR(z) over a Salpeter time (Di Matteo 2003 eq. [11])
#BH_BUBBLES                     # calculate bubble energy directly from the black hole accretion rate
#UNIFIED_FEEDBACK               # activates BH_THERMALFEEDBACK at high Mdot and BH_BUBBLES FEEDBACK al low Mdot
##-----------------------------------------------------------------------------------------------------
####################################################################################################-



############################################################################################################################-
##-----------------------------------------------------------------------------------------------------
#-------------------------------------- Star formation with -individual- stars [sink particles]: from PFH [proprietary development with Matt Orr and David Guszejnov; modules not to be used without authors permission]
##-----------------------------------------------------------------------------------------------------
#SINGLE_STAR_FORMATION          # master switch for single star formation model: sink particles representing -individual- stars
#SINGLE_STAR_ACCRETION=3        # proto-stellar accretion: 0=grav capture only; 1+=alpha-disk accretion onto protostar; 2+=bondi accretion of diffuse gas; 3+=sub-grid variability
#SINGLE_STAR_FB_HEATING         # proto-stellar heating: luminosity determined by BlackHoleRadiativeEfficiency (typical ~5e-7)
#SINGLE_STAR_FB_JETS            # protostellar jets: outflow rate+velocity set by BAL_f_accretion+BAL_v_outflow
#SINGLE_STAR_PROMOTION          # proto-stars become ZAMS stars at end of pre-main sequence lifetime. FIRE feedback modules kick in, but using appropriate luminosities and temperatures for each
############################################################################################################################-



############################################################################################################################-
#--------------------------------------- Radiative Transfer & Radiation Hydrodynamics:
#--------------------------------------------- modules developed by PFH & David Khatami: not for use without authors permission
############################################################################################################################-
#--------------------- methods for calculating photon propagation (one, and only one, of these MUST be on for RT)
#RT_LEBRON                              # RT solved using the LEBRON approximation (locally-extincted background radiation in optically-thin networks; default in the FIRE simulations)
#RT_OTVET                               # RT solved using the OTVET approximation (optically thin Eddington tensor, but interpolated to thick when appropriate)
#RT_M1                                  # RT solved using the M1 approximation (solve fluxes and tensors with M1 closure; gives better shadowing; currently only compatible with explicit diffusion solver)
#RT_FLUXLIMITEDDIFFUSION                # RT solved using the flux-limited diffusion approximation (constant, always-isotropic Eddington tensor)
#--------------------- solvers (numerical) --------------------------------------------------------
#RT_SPEEDOFLIGHT_REDUCTION=1            # set to a number <1 to use the 'reduced speed of light' approximation for photon propagation (C_eff=C_true*RT_SPEEDOFLIGHT_REDUCTION)
#RT_DIFFUSION_IMPLICIT                  # solve the diffusion part of the RT equations (if needed) implicitly with Conjugate Gradient iteration (Petkova+Springel): less accurate and only works with some methods, but allows larger timesteps [otherwise more accurate explicit used]
#--------------------- physics: wavelengths+coupled RT-chemistry networks -----------------------------------
#RT_SOURCES=1+16+32                     # source list for ionizing photons given by bitflag (1=2^0=gas,16=2^4=new stars,32=2^5=BH)
#RT_CHEM_PHOTOION=2                     # ionizing photons: 1=H-only [single-band], 2=H+He [four-band]
#RT_XRAY=3                              # x-rays: 1=soft (0.5-2 keV), 2=hard (>2 keV), 3=soft+hard; used for Compton-heating
#RT_INFRARED                            # infrared: photons absorbed in other bands are down-graded to IR: IR radiation + dust + gas temperatures evolved independently
#RT_PHOTOELECTRIC                       # far-uv (8-13.6eV): track photo-electric heating photons + their dust interactions
#RT_LYMAN_WERNER                        # specific lyman-werner [narrow H2 dissociating] band
#RT_OPTICAL_NIR                         # optical+near-ir: 3600 Angstrom-3 micron (where direct stellar emission dominates)
#RT_INFRARED                            # IR radiative transfer: follows IR as a gray body with individually evolved dust, gas, and radiation temperatures
#--------------------- radiation pressure options -------------------------------------------------
#RT_DISABLE_RAD_PRESSURE                # turn off radiation pressure forces (included by default)
#RT_RAD_PRESSURE_OUTPUT                 # print radiation pressure to file (requires some extra variables to save it)
##-----------------------------------------------------------------------------------------------------
#------------ test-problem, deprecated, or de-bugging functions
##-----------------------------------------------------------------------------------------------------
#RT_NOGRAVITY                           # turn off gravity: if using an RT method that needs the gravity tree (FIRE, OTVET), use this -instead- of NOGRAVITY to safely turn off gravitational forces
#RT_DIFFUSION_CG_MODIFY_EDDINGTON_TENSOR # when RT_DIFFUSION_CG is enabled, modifies the Eddington tensor to the fully anisotropic version (less stable CG iteration)
#RT_SEPARATELY_TRACK_LUMPOS             # keep luminosity vs. mass positions separate in tree (useful if running in tree-only mode)
#RT_DISABLE_FLUXLIMITER                 # removes the flux-limiter from the diffusion operations (default is to include it when using the relevant approximations)
#RT_HYDROGEN_GAS_ONLY                   # sets hydrogen fraction to 1.0 (used for certain idealized chemistry calculations)
#RT_COOLING_PHOTOHEATING_OLDFORMAT      # includes photoheating and cooling (using RT information), doing just the photo-heating [for more general cooling physics, enable COOLING]
#RT_FIRE_FIX_SPECTRAL_SHAPE             # enable with GALSF_FB_RT_PHOTONMOMENTUM to use a fixed SED shape set in parameterfile for all incident fluxes
#RT_DISABLE_UV_BACKGROUND               # disable extenal UV background in cooling functions (to isolate pure effects of local RT, or if simulating the background directly)
#RT_INJECT_PHOTONS_DISCRETELY           # do photon injection in discrete packets, instead of sharing a continuous source function. works better with adaptive timestepping (default with GALSF)
####################################################################################################-



####################################################################################################
# --------------------------------------- Multi-Threading (parallelization) options
####################################################################################################
#OPENMP=2                       # Masterswitch for explicit OpenMP implementation
#PTHREADS_NUM_THREADS=4         # custom PTHREADs implementation (don't enable with OPENMP)
####################################################################################################



####################################################################################################
# --------------------------------------- Output/Input options
####################################################################################################
HAVE_HDF5						# needed when HDF5 I/O support is desired
#OUTPUT_IN_DOUBLEPRECISION      # snapshot files will be written in double precision
#INPUT_IN_DOUBLEPRECISION       # input files assumed to be in double precision (otherwise float is assumed)
#OUTPUT_POSITIONS_IN_DOUBLE     # input/output files in single, but positions in double (used in hires, hi-dynamic range sims when positions differ by < float accuracy)
#INPUT_POSITIONS_IN_DOUBLE      # as above, but specific to the ICs file
#OUTPUTPOTENTIAL                # forces code to compute+output potentials in snapshots
#OUTPUTACCELERATION             # output physical acceleration of each particle in snapshots
#OUTPUTCHANGEOFENERGY           # outputs rate-of-change of internal energy of gas particles in snapshots
#OUTPUT_VORTICITY				# outputs the vorticity vector
#OUTPUTTIMESTEP                 # outputs timesteps for each particle
#OUTPUTCOOLRATE					# outputs cooling rate, and conduction rate if enabled
#OUTPUTLINEOFSIGHT				# enables on-the-fly output of Ly-alpha absorption spectra
#OUTPUTLINEOFSIGHT_SPECTRUM
#OUTPUTLINEOFSIGHT_PARTICLES
#POWERSPEC_ON_OUTPUT            # compute and output power spectra (not used)
#RECOMPUTE_POTENTIAL_ON_OUTPUT	# update potential every output even it EVALPOTENTIAL is set
#TWOPOINT_FUNCTION_COMPUTATION_ENABLED #calculate mass 2point function by enabling and setting restartflag=5
#OUTPUT_ADDITIONAL_RUNINFO      # enables extended simulation output data (can slow down machines significantly in massively-parallel runs)
####################################################################################################



####################################################################################################
# -------------------------------------------- De-Bugging & special (usually test-problem only) behaviors
####################################################################################################
#DEVELOPER_MODE                 # allows you to modify various numerical parameters (courant factor, etc) at run-time
#EOS_ENFORCE_ADIABAT=(1.0)      # if set, this forces gas to lie -exactly- along the adiabat P=EOS_ENFORCE_ADIABAT*(rho^GAMMA)
#SLOPE_LIMITER_TOLERANCE=1      # sets the slope-limiters used. higher=more aggressive (less diffusive, but less stable). 1=default. 0=conservative. use on problems where sharp density contrasts in poor particle arrangement may cause errors. 2=same as AGGRESSIVE_SLOPE_LIMITERS below
#AGGRESSIVE_SLOPE_LIMITERS      # use the original GIZMO paper (more aggressive) slope-limiters. more accurate for smooth problems, but
                                # these can introduce numerical instability in problems with poorly-resolved large noise or density contrasts (e.g. multi-phase, self-gravitating flows)
#ENERGY_ENTROPY_SWITCH_IS_ACTIVE # enable energy-entropy switch as described in GIZMO methods paper. This can greatly improve performance on some problems where the
                                # the flow is very cold and highly super-sonic. it can cause problems in multi-phase flows with strong cooling, though, and is not compatible with non-barytropic equations of state
#FORCE_ENTROPIC_EOS_BELOW=(0.01) # set (manually) the alternative energy-entropy switch which is enabled by default in MFM/MFV: if relative velocities are below this threshold, it uses the entropic EOS
#TEST_FOR_IDUNIQUENESS          # explicitly check if particles have unique id numbers (only use for special behaviors)
#LONGIDS                        # use long ints for IDs (needed for super-large simulations)
#ASSIGN_NEW_IDS                 # assign IDs on startup instead of reading from ICs
#NO_CHILD_IDS_IN_ICS            # IC file does not have child IDs: do not read them (used for compatibility with snapshot restarts from old versions of the code)
#READ_HSML                      # reads hsml from IC file
#PREVENT_PARTICLE_MERGE_SPLIT   # don't allow gas particle splitting/merging operations
#COOLING_OPERATOR_SPLIT         # do the hydro heating/cooling in operator-split fashion from chemical/radiative. slightly more accurate when tcool >> tdyn, but much noisier when tcool << tdyn
#PARTICLE_EXCISION              # enable dynamical excision (remove particles within some radius)
#MERGESPLIT_HARDCODE_MAX_MASS=(1.0e-6)   # manually set maximum mass for particle merge-split operations (in code units): useful for snapshot restarts and other special circumstances
#MERGESPLIT_HARDCODE_MIN_MASS=(1.0e-7)   # manually set minimum mass for particle merge-split operations (in code units): useful for snapshot restarts and other special circumstances


#USE_MPI_IN_PLACE               # MPI debugging: makes AllGatherV compatible with MPI_IN_PLACE definitions in some MPI libraries
#NO_ISEND_IRECV_IN_DOMAIN       # MPI debugging: slower, but fixes memory errors during exchange in the domain decomposition (ANY RUN with >2e9 particles MUST SET THIS OR FAIL!)
#FIX_PATHSCALE_MPI_STATUS_IGNORE_BUG # MPI debugging
#MPISENDRECV_SIZELIMIT=100      # MPI debugging
#MPISENDRECV_CHECKSUM           # MPI debugging
#DONOTUSENODELIST               # MPI debugging
#NOTYPEPREFIX_FFTW              # FFTW debugging (fftw-header/libraries accessed without type prefix, adopting whatever was
                                #   chosen as default at compile of fftw). Otherwise, the type prefix 'd' for double is used.
#DOUBLEPRECISION_FFTW           # FFTW in double precision to match libraries
#DEBUG                          # enables core-dumps and FPU exceptions
#STOP_WHEN_BELOW_MINTIMESTEP    # forces code to quit when stepsize wants to go below MinSizeTimestep specified in the parameterfile
#SEPARATE_STELLARDOMAINDECOMP   # separate stars (ptype=4) and other non-gas particles in domain decomposition (may help load-balancing)
#DISABLE_SPH_PARTICLE_WAKEUP    # don't let gas particles move to lower timesteps based on neighbor activity (use for debugging)
#EVALPOTENTIAL                  # computes gravitational potential
#MHD_ALTERNATIVE_LEAPFROG_SCHEME # use alternative leapfrog where magnetic fields are treated like potential/positions (per Federico Stasyszyn's suggestion): still testing
#FREEZE_HYDRO                   # zeros all fluxes from RP and doesn't let particles move (for testing additional physics layers)
#SUPER_TIMESTEP_DIFFUSION       # use super-timestepping to accelerate integration of diffusion operators [for testing or if there are stability concerns]
#ALLOW_IMBALANCED_GASPARTICLELOAD # increases All.MaxPartSph to All.MaxPart: can allow better load-balancing in some cases, but uses more memory. But use me if you run into errors where it can't fit the domain (where you would increase PartAllocFac, but can't for some reason)
####################################################################################################





####################################################################################################-
####################################################################################################-
##-
##- LEGACY CODE & PARTIALLY-IMPLEMENTED FEATURES: BEWARE EVERYTHING BELOW THIS LINE !!!
##-
####################################################################################################-
####################################################################################################-


####################################################################################################-
#---------------------------------------- Subhalo on-the-fly finder options (needs "subfind" source code)
#------------------ This is originally developed as part of GADGET-3 (SUBFIND) by V. Springel
#------------------ Use of these modules follows the GADGET-3 policies described above
####################################################################################################-
#SUBFIND                            #- enables substructure finder
#MAX_NGB_CHECK=3                    #- Max numbers of neighbours for sattlepoint detection (default = 2)
#SAVE_MASS_TAB                      #- Saves the an additional array with the masses of the different components
#SUBFIND_SAVE_PARTICLELISTS         #- Saves also phase-space and type variables parallel to IDs
#SO_VEL_DISPERSIONS                 #- computes velocity dispersions for as part of FOF SO-properties
#ONLY_PRODUCE_HSML_FILES            #- only carries out density estimate
#SAVE_HSML_IN_IC_ORDER              #- will store the hsml-values in the order of the particles in the IC file
#KEEP_HSML_AS_GUESS                 #- keep using hsml for gas particles in subfind_density
#NO_GAS_CLOUDS                      #- Do not accept pure gaseous substructures
#WRITE_SUB_IN_SNAP_FORMAT           #- Save subfind results in snap format
#DUSTATT=11                         #- Includes dust attenuation into the luminosity calculation (using 11 radial bins)
#OBSERVER_FRAME                     #- If defined, use CB07 Observer Frame Luminosities, otherwise CB07 Rest Frame Luminosities
#SO_BAR_INFO                        #- Adds temperature, Lx, bfrac, etc to Groups
#SUBFIND_COUNT_BIG_HALOS=1e4        #- Adds extra blocks for Halos with M_TopHat > SUBFIND_COUNT_BIG_HALOS
#KD_CHOOSE_PSUBFIND_LIMIT           #- Increases the limit for the parallel subfind to the maximum possible
#KD_ALTERNATIVE_GROUP_SORT          #- Alternative way to sort the Groups/SubGroupe before writing
#SUBFIND_READ_FOF                   #-
#SUBFIND_COLLECTIVE_STAGE1          #-
#SUBFIND_COLLECTIVE_STAGE2          #-
#SUBFIND_ALTERNATIVE_COLLECTIVE     #-
#SUBFIND_RESHUFFLE_CATALOGUE        #-
#SUBFIND_RESHUFFLE_AND_POTENTIAL    #- needs -DSUBFIND_RESHUFFLE_CATALOGUE and COMPUTE_POTENTIAL_ENERGY
#SUBFIND_DENSITY_AND_POTENTIAL      #- only calculated density and potential and write them into snapshot
####################################################################################################-



####################################################################################################-
#--------------------------------------- nuclear network
#-------------------------------- (these are currently non-functional and should not be used)
####################################################################################################-
#NUCLEAR_NETWORK
#NEGLECT_DTDY_TERMS
#NETWORK_OUTPUT
#NETWORK_OUTPUT_BINARY
####################################################################################################-






