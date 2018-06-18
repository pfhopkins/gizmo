#!/bin/bash            # this line only there to enable syntax highlighting in this file

####################################################################################################
#  Enable/Disable compile-time options as needed: this is where you determine how the code will act
#  From the list below, please activate/deactivate the
#       options that apply to your run. If you modify any of these options,
#       make sure that you recompile the whole code by typing "make clean; make".
#
#  Consult the User Guide before enabling any option. Some modules are proprietary -- access to them
#    must be granted separately by the code authors (just having the code does NOT grant permission).
#    Even public modules have citations which must be included if the module is used for published work,
#    these are all given in the User Guide.
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
#BOX_PERIODIC               # Use this if periodic boundaries are needed (otherwise open boundaries are assumed)
#BOX_BND_PARTICLES          # particles with ID=0 are forced in place (their accelerations are set =0): use for special boundary conditions where these particles represent fixed "walls"
#BOX_LONG_X=140             # modify box dimensions (non-square periodic box): multiply X (BOX_PERIODIC and SELFGRAVITY_OFF required)
#BOX_LONG_Y=1               # modify box dimensions (non-square periodic box): multiply Y
#BOX_LONG_Z=1               # modify box dimensions (non-square periodic box): multiply Z
#BOX_REFLECT_X              # make the x-boundary reflecting (assumes a box 0<x<1, unless BOX_PERIODIC is set)
#BOX_REFLECT_Y              # make the y-boundary reflecting (assumes a box 0<y<1, unless BOX_PERIODIC is set)
#BOX_REFLECT_Z              # make the z-boundary reflecting (assumes a box 0<z<1, unless BOX_PERIODIC is set)
#BOX_SHEARING=1             # shearing box boundaries: 1=r-z sheet (r,z,phi coordinates), 2=r-phi sheet (r,phi,z), 3=r-phi-z box, 4=as 3, with vertical gravity
#BOX_SHEARING_Q=(3./2.)     # shearing box q=-dlnOmega/dlnr; will default to 3/2 (Keplerian) if not set
#BOX_SPATIAL_DIMENSION=3    # sets number of spatial dimensions evolved (default=3D). Switch for 1D/2D test problems: if =1, code only follows the x-line (all y=z=0), if =2, only xy-plane (all z=0). requires SELFGRAVITY_OFF
####################################################################################################



####################################################################################################
# --------------------------------------- Hydro solver method
####################################################################################################
# --------------------------------------- Finite-volume Godunov methods (choose one, or SPH)
#HYDRO_MESHLESS_FINITE_MASS     # solve hydro using the mesh-free Lagrangian (fixed-mass) finite-volume Godunov method
#HYDRO_MESHLESS_FINITE_VOLUME   # solve hydro using the mesh-free (quasi-Lagrangian) finite-volume Godunov method (control mesh motion with HYDRO_FIX_MESH_MOTION)
#HYDRO_REGULAR_GRID             # solve hydro equations on a regular (recti-linear) Cartesian mesh (grid) with a finite-volume Godunov method
## -----------------------------------------------------------------------------------------------------
# --------------------------------------- Options to explicitly control the mesh motion (for use with the MFV or grid solvers): only set for non-standard behavior
#HYDRO_FIX_MESH_MOTION=0        # mesh with arbitrarily-defined mesh-generating velocities: (0=non-moving, 1=fixed-v [set in ICs] cartesian, 2=fixed-v [ICs] cylindrical, 3=fixed-v [ICs] spherical, 4=analytic function, 5=smoothed-Lagrangian, 6=glass-generating, 7=fully-Lagrangian)
#HYDRO_GENERATE_TARGET_MESH     # use for IC generation (can be used with -any- hydro method: MFM/MFV/SPH/grid): this allows you to specify in the functions 'return_user_desired_target_density' and 'return_user_desired_target_pressure' (in eos.c) the desired initial density/pressure profile, and the code will try to evolve towards this.
## -----------------------------------------------------------------------------------------------------
# --------------------------------------- SPH methods (enable one of these flags to use SPH):
#HYDRO_PRESSURE_SPH             # solve hydro using SPH with the 'pressure-sph' formulation ('P-SPH')
#HYDRO_DENSITY_SPH              # solve hydro using SPH with the 'density-sph' formulation (GADGET-2 & GASOLINE SPH)
# --------------------------------------- SPH artificial diffusion options (use with SPH; not relevant for Godunov/Mesh modes)
#SPH_DISABLE_CD10_ARTVISC       # for SPH only: Disable Cullen & Dehnen 2010 'inviscid sph' (viscosity suppression outside shocks); just use Balsara switch
#SPH_DISABLE_PM_CONDUCTIVITY    # for SPH only: Disable mixing entropy (J.Read's improved Price-Monaghan conductivity with Cullen-Dehnen switches)
## -----------------------------------------------------------------------------------------------------
# --------------------------------------- Kernel Options
#KERNEL_FUNCTION=3              # Choose the kernel function (2=quadratic peak, 3=cubic spline [default], 4=quartic spline, 5=quintic spline, 6=Wendland C2, 7=Wendland C4, 8=2-part quadratic)
#KERNEL_CRK_FACES               # Use the consistent reproducing kernel [higher-order tensor corrections to kernel above, compared to our usual matrix formalism] from Frontiere, Raskin, and Owen to define the faces in MFM/MFV methods. can give more accurate closure, potentially improved accuracy in MHD problems. remains experimental for now.
####################################################################################################



####################################################################################################
# --------------------------------------- Additional Fluid Physics
####################################################################################################
## ----------------------------------------------------------------------------------------------------
# --------------------------------------- Gas (or Material) Equations-of-State
#EOS_GAMMA=(5.0/3.0)            # Polytropic Index of Gas (for an ideal gas law): if not set and no other (more complex) EOS set, defaults to GAMMA=5/3
#EOS_HELMHOLTZ                  # Use Timmes & Swesty 2000 EOS (for e.g. stellar or degenerate equations of state); if additional tables needed, download at http://www.tapir.caltech.edu/~phopkins/public/helm_table.dat (or the BitBucket site)
#EOS_TILLOTSON                  # Use Tillotson (1962) EOS (for solid/liquid+vapor bodies, impacts); custom EOS params can be specified or pre-computed materials used. see User Guide and Deng et al., arXiv:1711.04589
#EOS_ELASTIC                    # treat fluid as elastic or plastic (or visco-elastic) material, obeying Hooke's law with full stress terms and von Mises yield model. custom EOS params can be specified or pre-computed materials used.
## -----------------------------------------------------------------------------------------------------
# --------------------------------- Magneto-Hydrodynamics
# ---------------------------------  these modules are public, but if used, the user should also cite the MHD-specific GIZMO methods paper
# ---------------------------------  (Hopkins 2015: 'Accurate, Meshless Methods for Magneto-Hydrodynamics') as well as the standard GIZMO paper
#MAGNETIC                       # master switch for MHD, regardless of which Hydro solver is used
#MHD_B_SET_IN_PARAMS            # set initial fields (Bx,By,Bz) in parameter file
#MHD_NON_IDEAL                  # enable non-ideal MHD terms: Ohmic resistivity, Hall effect, and ambipolar diffusion (solved explicitly); Users should cite Hopkins 2017, MNRAS, 466, 3387, in addition to the MHD paper
#MHD_CONSTRAINED_GRADIENT=1     # use CG method (in addition to cleaning, optional!) to maintain low divB: set this value to control how aggressive the div-reduction is:
                                # 0=minimal (safest), 1=intermediate (recommended), 2=aggressive (less stable), 3+=very aggressive (less stable+more expensive). [Please cite Hopkins, MNRAS, 2016, 462, 576]
## ----------------------------------------------------------------------------------------------------
# -------------------------------------- Conduction
# ----------------------------------------- [Please cite and read the methods paper Hopkins 2017, MNRAS, 466, 3387]
#CONDUCTION                     # Thermal conduction solved *explicitly*: isotropic if MAGNETIC off, otherwise anisotropic
#CONDUCTION_SPITZER             # Spitzer conductivity accounting for saturation: otherwise conduction coefficient is constant  [cite Su et al., 2017, MNRAS, 471, 144, in addition to the conduction methods paper above]
## ----------------------------------------------------------------------------------------------------
# -------------------------------------- Viscosity
# ----------------------------------------- [Please cite and read the methods paper Hopkins 2017, MNRAS, 466, 3387]
#VISCOSITY                      # Navier-stokes equations solved *explicitly*: isotropic coefficients if MAGNETIC off, otherwise anisotropic
#VISCOSITY_BRAGINSKII           # Braginskii viscosity tensor for ideal MHD    [cite Su et al., 2017, MNRAS, 471, 144, in addition to the viscosity methods paper above]
## ----------------------------------------------------------------------------------------------------
# -------------------------------------- Radiative Cooling physics (mostly geared towards galactic/extragalactic cooling)
# -------------------------- These modules were originally developed for a combination of proprietary physics modules. However they are now written in
# --------------------------   a form which allows them to be modular (and public). Users are free to use the Grackle modules and standard 'COOLING' flags,
# --------------------------   provided proper credit/citations are provided to the relevant methods papers given in the Users Guide ---
# --------------------------   but all users should cite Hopkins et al. 2017 (arXiv:1702.06148), where Appendix B details the cooling physics
#COOLING                        # enables radiative cooling and heating: if GALSF, also external UV background read from file "TREECOOL" (included in the cooling folder)
#COOL_LOW_TEMPERATURES          # allow fine-structure and molecular cooling to ~10 K; account for optical thickness and line-trapping effects with proper opacities
#COOL_METAL_LINES_BY_SPECIES    # use full multi-species-dependent cooling tables ( http://www.tapir.caltech.edu/~phopkins/public/spcool_tables.tgz, or the Bitbucket site); requires METALS on; cite Wiersma et al. 2009 (MNRAS, 393, 99) in addition to Hopkins et al. 2017 (arXiv:1702.06148)
#COOL_GRACKLE                   # enable Grackle: cooling+chemistry package (requires COOLING above; https://grackle.readthedocs.org/en/latest ); see Grackle code for their required citations
#COOL_GRACKLE_CHEMISTRY=1       # choose Grackle cooling chemistry: (0)=tabular, (1)=Atomic, (2)=(1)+H2+H2I+H2II, (3)=(2)+DI+DII+HD
#METALS                         # enable metallicities (with multiple species optional) for gas and stars [must be included in ICs or injected via dynamical feedback; needed for some routines]
## ----------------------------------------------------------------------------------------------------
# -------------------------------------- Smagorinsky Turbulent Eddy Diffusion Model
# --------------------------------------- Users of these modules should cite Hopkins et al. 2017 (arXiv:1702.06148) and Colbrook et al. (arXiv:1610.06590)
#TURB_DIFF_METALS               # turbulent diffusion of metals (passive scalars); requires METALS
#TURB_DIFF_ENERGY               # turbulent diffusion of internal energy (conduction with effective turbulent coefficients)
#TURB_DIFF_VELOCITY             # turbulent diffusion of momentum (viscosity with effective turbulent coefficients)
## ----------------------------------------------------------------------------------------------------
# --------------------------------------- Aerodynamic Particles
# ----------------------------- This is developed by P. Hopkins, who requests that you inform him of planned projects with these modules
# ------------------------------  because he is supervising several students using them as well, and there are some components still in active development.
# ------------------------------  Users should cite: Hopkins & Lee 2016, MNRAS, 456, 4174, and Lee, Hopkins, & Squire 2017, MNRAS, 469, 3532, for the numerical methods
#GRAIN_FLUID                    # aerodynamically-coupled grains (particle type 3 are grains); default is Epstein drag
#GRAIN_EPSTEIN_STOKES=1         # uses the cross section for molecular hydrogen (times this number) to calculate Epstein-Stokes drag (will use calculate which applies and use appropriate value); if used with GRAIN_LORENTZFORCE, will also compute Coulomb drag
#GRAIN_BACKREACTION             # account for momentum of grains pushing back on gas (from drag terms)
#GRAIN_LORENTZFORCE             # charged grains feel Lorentz forces (requires MAGNETIC); if used with GRAIN_EPSTEIN_STOKES flag, will also compute Coulomb drag (grain charges self-consistently computed from gas properties)
#GRAIN_COLLISIONS               # model collisions between grains (super-particles; so this is stochastic)
## ----------------------------------------------------------------------------------------------------
#---------------------------------------- Cosmic Rays
#---------------------------------------- (this is developed by P. Hopkins as part of the FIRE package: the same FIRE authorship & approval policies apply, see below)
#COSMIC_RAYS                    # two-fluid medium with CRs as an ultrarelativistic fluid: heating/cooling, anisotropic diffusion, streaming, injection by SNe
#COSMIC_RAYS_ALFVEN=(500.)      # solve CR transport based on Alfven-limited scattering from Thomas+Pfrommer 18, evolves CRs+resonant Alfven population; value here is maximum free-streaming speed in code units
#COSMIC_RAYS_M1=(500.)          # solve the CR transport in the M1 limit [second-order expansion of the collisionless boltzmann eqn]; value here is the streaming speed in code units
#COSMIC_RAYS_DISABLE_STREAMING  # turn off CR streaming (propagation is purely advective+diffusion; warning: this can severely under-estimate CR losses to Alfven waves)
#COSMIC_RAYS_DISABLE_DIFFUSION  # turn off CR diffusion (leaves streaming intact, simply disables 'microscopic+turbulent' CR diffusion terms)
#COSMIC_RAYS_DISABLE_COOLING    # turn off CR heating/cooling interactions with gas (catastrophic losses, hadronic interactions, etc; only adiabatic PdV work terms remain)
#COSMIC_RAYS_DIFFUSION_CONSTANT # replaces physical CR diffusion with constant coefficient (equal to value of CosmicRayDiffusionCoeff in code units); turn off streaming to make this the ONLY transport
## ----------------------------------------------------------------------------------------------------
####################################################################################################



####################################################################################################
# ------------------------------------- Driven turbulence (for turbulence tests, large-eddy sims)
# ------------------------------- users of these routines should cite Bauer & Springel 2012, MNRAS, 423, 3102. Thanks to A. Bauer for providing the core algorithms
####################################################################################################
#TURB_DRIVING                   # turns on turbulent driving/stirring. see begrun for parameters that must be set
#TURB_DRIVING_SPECTRUMGRID=128  # activates on-the-fly calculation of the turbulent velocity, vorticity, and smoothed-velocity power spectra, evaluated on a grid of linear-size TURB_DRIVING_SPECTRUMGRID elements
#TURB_DRIVING_DUMPSPECTRUM      # compiles in a code module that allows via restart-flag 6 the calculation (+immediate output) of a gas velocity power spectrum of a snapshot with an adjustable box (user defined center and size)
####################################################################################################



####################################################################################################
## ------------------------ Gravity & Cosmological Integration Options ---------------------------------
####################################################################################################
# --------------------------------------- TreePM Options (recommended for cosmological sims)
#PMGRID=512                     # adds Particle-Mesh grid for faster (but less accurate) long-range gravitational forces: value sets resolution (e.g. a PMGRID^3 grid will overlay the box, as the 'top level' grid)
#PM_PLACEHIGHRESREGION=1+2+16   # adds a second-level (nested) PM grid before the tree: value denotes particle types (via bit-mask) to place high-res PMGRID around. Requires PMGRID.
#PM_HIRES_REGION_CLIPPING=1000  # optional additional criterion for boundaries in 'zoom-in' type simulations: clips gas particles that escape the hires region in zoom/isolated sims, specifically those whose nearest-neighbor distance exceeds this value (in code units)
#PM_HIRES_REGION_CLIPDM         # split low-res DM particles that enter high-res region (completely surrounded by high-res)
## -----------------------------------------------------------------------------------------------------
# ---------------------------------------- Adaptive Grav. Softening (including Lagrangian conservation terms!)
#ADAPTIVE_GRAVSOFT_FORGAS       # allows variable softening length for gas particles (scaled with local inter-element separation), so gravity traces same density field seen by hydro
#ADAPTIVE_GRAVSOFT_FORALL=100   # enable adaptive gravitational softening lengths for all particle types (ADAPTIVE_GRAVSOFT_FORGAS should be disabled). the softening is set to the distance
                                # enclosing a neighbor number set in the parameter file. baryons search for other baryons, dm for dm, sidm for sidm, etc. If set to numerical value, the maximum softening is this times All.ForceSoftening[for appropriate particle type]. cite Hopkins et al., arXiv:1702.06148
## -----------------------------------------------------------------------------------------------------
#SELFGRAVITY_OFF                # turn off self-gravity (compatible with GRAVITY_ANALYTIC); setting NOGRAVITY gives identical functionality
#GRAVITY_NOT_PERIODIC           # self-gravity is not periodic, even though the rest of the box is periodic
## -----------------------------------------------------------------------------------------------------
#GRAVITY_ANALYTIC               # specific analytic gravitational force to use instead of or with self-gravity. If set to a numerical value
                                #  > 0 (e.g. =1), then BH_CALC_DISTANCES will be enabled, and it will use the nearest BH particle as the center for analytic gravity computations
                                #  (edit "gravity/analytic_gravity.h" to actually assign the analytic gravitational forces). 'ANALYTIC_GRAVITY' gives same functionality
## ----------------------------------------------------------------------------------------------------
#--------------------------------------- Self-Interacting DM (Rocha et al. 2012) and Scalar-field DM and Fuzzy DM
#--------------------------------    use of these routines requires explicit pre-approval by developers J. Bullock or M. Boylan-Kolchin (acting for M. Rocha); approved users please cite Rocha et al., MNRAS 2013, 430, 81 and Robles et al, 2017 (arXiv:1706.07514).
#DM_SIDM=2                      # self-interacting particle types (specify the particle types which are self-interacting DM with a bit mask, as for PM_PLACEHIGHRESREGION above (see description); previous "DMDISK_INTERACTIONS" is identical to setting DM_SIDM=2+4
#DM_SCALARFIELD_SCREENING       # gravity is mediated by a long-range scalar field, with dynamical screening (primarily alternative DE models)
#DM_FUZZY                       #- DM particles (Type=1) are described by Bose-Einstein Condensate: within gravity kernel (adaptive), solves quantum pressure tensor for non-linear terms arising from Schroedinger equation for a given particle mass. IN TESTING PLEASE DO NOT USE!!!
## ----------------------------------------------------------------------------------------------------
# -------------------------------------- arbitrary time-dependent dark energy equations-of-state, expansion histories, or gravitational constants
#GR_TABULATED_COSMOLOGY         # enable reading tabulated cosmological/gravitational parameters (master switch)
#GR_TABULATED_COSMOLOGY_W       # read pre-tabulated dark energy equation-of-state w(z)
#GR_TABULATED_COSMOLOGY_H       # read pre-tabulated hubble function (expansion history) H(z)
#GR_TABULATED_COSMOLOGY_G       # read pre-tabulated gravitational constant G(z) [also rescales H(z) appropriately]
## ----------------------------------------------------------------------------------------------------
#EOS_TRUELOVE_PRESSURE          # adds artificial pressure floor force Jeans length above resolution scale (means you can get the wrong answer, but things will look smooth).  cite Robertson & Kravtsov 2008, ApJ, 680, 1083
####################################################################################################



####################################################################################################
# --------------------------------------- On the fly FOF groupfinder
# ----------------- This is originally developed as part of GADGET-3 by V. Springel
# ----------------- Users of any of these modules should cite Springel et al., MNRAS, 2001, 328, 726 for the numerical methods.
####################################################################################################
## ----------------------------------------------------------------------------------------------------
# ------------------------------------- Friends-of-friends on-the-fly finder options (source in fof.c)
# -----------------------------------------------------------------------------------------------------
#FOF                                # enable FoF searching on-the-fly and outputs (set parameter LINKLENGTH=x to control LinkingLength; default=0.2)
#FOF_PRIMARY_LINK_TYPES=2           # 2^type for the primary dark matter type
#FOF_SECONDARY_LINK_TYPES=1+16+32   # 2^type for the types linked to nearest primaries
#FOF_DENSITY_SPLIT_TYPES=1+2+16+32  # 2^type for whch the densities should be calculated seperately
#FOF_GROUP_MIN_LEN=32               # default is 32
####################################################################################################



####################################################################################################
# ----------------- Galaxy formation & Galactic Star formation
####################################################################################################
## ---------------------------------------------------------------------------------------------------
#GALSF                           # master switch for galactic star formation model: enables SF, stellar ages, generations, etc. [cite Springel+Hernquist 2003, MNRAS, 339, 289]
## ----------------------------------------------------------------------------------------------------
# --- star formation law/particle spawning (additional options: otherwise all star particles will reflect IMF-averaged populations and form strictly based on a density criterion) ---- #
## ----------------------------------------------------------------------------------------------------
#GALSF_SFR_MOLECULAR_CRITERION   # estimates molecular/self-shielded fraction in SF-ing gas, only SF from that is allowed. Cite Krumholz & Gnedin (ApJ 2011 729 36) and Hopkins et al., 2017a, arXiv:1702.06148
#GALSF_SFR_VIRIAL_SF_CRITERION=0 # only allow star formation in virialized sub-regions (alpha<1) (0/no value='default'; 1=0+Jeans criterion; 2=1+'strict' (zero sf if not bound)), 3=2+converging-flow+time-smoothed. Cite Hopkins, Narayanan, & Murray 2013 (MNRAS, 432, 2647) and Hopkins et al., 2017a, arXiv:1702.06148; (or Grudic et al. arXiv:1708.09065 for option=3)
#GALSF_SFR_IMF_VARIATION         # determines the stellar IMF for each particle from the Guszejnov/Hopkins/Hennebelle/Chabrier/Padoan theory. Cite Guszejnov, Hopkins, & Ma 2017, MNRAS, 472, 2107
#GALSF_SFR_IMF_SAMPLING          # discretely sample the IMF: simplified model with quantized number of massive stars. Cite Kung-Yi Su, Hopkins, et al., Hayward, et al., 2017, "Discrete Effects in Stellar Feedback: Individual Supernovae, Hypernovae, and IMF Sampling in Dwarf Galaxies". 
#GALSF_GENERATIONS=1             # the number of star particles a gas particle may spawn (defaults to 1, set otherwise)
## ----------------------------------------------------------------------------------------------------------------------------
# ---- sub-grid models (for large-volume simulations or modest/low resolution galaxy simulations) -----------------------------
# -------- these are all ultimately variations of the Springel & Hernquist 2005 sub-grid models for the ISM, star formation, and winds.
# -------- Volker has granted permissions for their use, provided users properly cite the sources for the relevant models and scalings (described below)
#GALSF_EFFECTIVE_EQS            # Springel-Hernquist 'effective equation of state' model for the ISM and star formation [cite Springel & Hernquist, MNRAS, 2003, 339, 289]
#GALSF_SUBGRID_WINDS            # sub-grid winds ('kicks' as in Oppenheimer+Dave,Springel+Hernquist,Boothe+Schaye,etc): enable this master switch for basic functionality [cite Springel & Hernquist, MNRAS, 2003, 339, 289]
#GALSF_SUBGRID_WIND_SCALING=0   # set wind velocity scaling: 0 (default)=constant v [and mass-loading]; 1=velocity scales with halo mass (cite Oppenheimer & Dave, 2006, MNRAS, 373, 1265), requires FOF modules; 2=scale with local DM dispersion as Vogelsberger 13 (cite Zhu & Li, ApJ, 2016, 831, 52)
#GALSF_WINDS_ORIENTATION=0      # directs wind orientation [0=isotropic/random, 1=polar, 2=along density gradient]
## ----------------------------------------------------------------------------------------------------------------------------
# ---- pure thermal/scalar stellar feedback models (simplified energy injection; tend to severely over-cool owing to lack of mechanical/kinetic treatment at finite resolution -----------------------------
#GALSF_FB_THERMAL               # simple 'pure thermal energy dump' feedback: mass, metals, and thermal energy are injected locally in simple kernel-weighted fashion around young stars. currently follows AGORA feedback scheme. cite Kim et al., 2016, ApJ, 833, 202 if used.
#GALSF_FB_TURNOFF_COOLING       # turn off cooling for SNe-heated particles (as Stinson+ 2006 GASOLINE model, cite it); requires GALSF_FB_SNE_HEATING or GALSF_FB_THERMAL; use by permission of developer P. Hopkins)
#GALSF_GASOLINE_RADHEATING      # heat gas with luminosity from young stars (as Stinson+ 2013 GASOLINE model, cite it); requires GALSF_FB_SNE_HEATING; use by permission of developer P. Hopkins)
## ----------------------------------------------------------------------------------------------------
#------ PFH physical models for star formation and feedback: these are the FIRE simulation modules (Hopkins et al. 2014, Hopkins et al., 2017a, arXiv:1702.06148) ------ ##
#--------- Their use follows the FIRE authorship policy. Modules are NOT to be used without authors permission (including P. Hopkins, E. Quataert, D. Keres, and C.A. Faucher-Giguere),
#---------    even if you are already using the development GIZMO code. (PFH does not have sole authority to grant permission for the modules)
#--------- New projects using these modules must FIRST be PRE-APPROVED by the collaboration (after permission to use the modules has been explicitly granted),
#--------- and subsequently are required to follow the collaboration's paper approval and submission policies
##-----------------------------------------------------------------------------------------------------
#FIRE_PHYSICS_DEFAULTS           # enable default set of FIRE physics packages (see details below); note use policy above
#----------- physical stellar feedback mechanisms (sub-modules of the FIRE_PHYSICS_DEFAULTS; these cannot be turned off and on individually without extreme care, they have complicated inter-dependencies) ---- #
##----------------------GALSF_FB_GASRETURN              # Paul Torrey's addition for stochastic gas return (modified for continuous return)
##----------------------GALSF_FB_HII_HEATING            # gas within HII regions around young stars is photo-heated to 10^4 K
##----------------------GALSF_FB_SNE_HEATING=1          # time-dependent explosions from SNe (I & II) in shockwave radii around stars (values: 0=force-gridded in xyz (WRONG-for testing only!); 1=tensor-symmetrized (momentum-conserving; USE ME); 2=no tensor re-normalization [non-conservative!]
##----------------------GALSF_FB_RPROCESS_ENRICHMENT=4  # tracks a set of 'dummy' species from neutron-star mergers (set to number: 4=extended model)
##----------------------GALSF_FB_RT_PHOTONMOMENTUM      # continuous acceleration from starlight (uses luminosity tree)
##----------------------GALSF_FB_LOCAL_UV_HEATING       # use local estimate of spectral information for photoionization and photoelectric heating
##----------------------GALSF_FB_RPWIND_LOCAL           # turn on local radiation pressure coupling to gas
##-----------------------------------------------------------------------------------------------------
#----------- deprecated options (most have been combined or optimized into the functions above, here for legacy purposes only)
##----------------------GALSF_FB_RPWIND_FROMCLUMPS	    # clump-find to for wind angles (deprecated; now use density routine to find)
##----------------------GALSF_FB_RPWIND_CONTINUOUS	    # wind accel term is continuous (more expensive and introduces more artificial dissipation)
##----------------------GALSF_FB_RPWIND_DO_IN_SFCALC	# do IR wind loop in SFR routine (allows fof clump-finding, useful for very IR-thick, but slow)
##----------------------GALSF_FB_RPWIND_FROMSFR         # drive radiation pressure with gas SFR (instead of default, which is nearby young stars)
##----------------------FIRE_UNPROTECT_FROZEN           # by default, FIRE-2 code version is 'frozen' so it cannot be changed by any code updates. this removes the protection, so you will use whatever the newest algorithms in GIZMO are, but use it with CAUTION since the algorithm may NOT agree with the other FIRE runs
## ----------------------------------------------------------------------------------------------------
############################################################################################################################



############################################################################################################################
## ----------------------------------------------------------------------------------------------------
# ------------------------------------- Star formation with -individual- stars [sink particles]: from PFH [proprietary development with Mike Grudic and David Guszejnov; modules not to be used without authors permission]
## ----------------------------------------------------------------------------------------------------
#SINGLE_STAR_FORMATION          # master switch for single star formation model: sink particles representing -individual- stars
#SINGLE_STAR_ACCRETION=3        # proto-stellar accretion: 0=grav capture only; 1+=alpha-disk accretion onto protostar; 2+=bondi accretion of diffuse gas; 3+=sub-grid variability
#SINGLE_STAR_FB_HEATING         # proto-stellar heating: luminosity determined by BlackHoleRadiativeEfficiency (typical ~5e-7)
#SINGLE_STAR_FB_JETS            # protostellar jets: outflow rate+velocity set by BAL_f_accretion+BAL_v_outflow
#SINGLE_STAR_PROMOTION          # proto-stars become ZAMS stars at end of pre-main sequence lifetime. FIRE feedback modules kick in, but using appropriate luminosities and temperatures for each
############################################################################################################################



####################################################################################################
# ---------------- Black Holes (Sink-Particles with Accretion and Feedback)
####################################################################################################
#BLACK_HOLES                    # master switch to enable BHs
## ----------------------------------------------------------------------------------------------------
# ----- seeding / BH-particle spawning
## ----------------------------------------------------------------------------------------------------
#BH_SEED_FROM_FOF=0             # use FOF on-the-fly to seed BHs in massive FOF halos; =0 uses DM groups, =1 uses stellar [type=4] groups; requires FOF with linking type including relevant particles (cite Angles-Alcazar et al., MNRAS, 2017, arXiv:1707.03832)
#BH_SEED_FROM_LOCALGAS          # BHs seeded on-the-fly from dense, low-metallicity gas (no FOF), like star formation; criteria define-able in modular fashion in sfr_eff.c (function return_probability_of_this_forming_bh_from_seed_model). cite Grudic et al. (arXiv:1612.05635) and Lamberts et al. (MNRAS, 2016, 463, L31)
#BH_INCREASE_DYNAMIC_MASS=100   # increase the particle dynamical mass by this factor at the time of BH seeding
## ----------------------------------------------------------------------------------------------------
# ----- dynamics (when BH mass is not >> other particle masses, it will artificially get kicked and not sink via dynamical friction; these 're-anchor' the BH for low-res sims)
## ----------------------------------------------------------------------------------------------------
#BH_DYNFRICTION=0               # apply explicit dynamical friction force to the BHs when m_bh not >> other particle mass: 0=[DM+stars+gas]; 1=[DM+stars]; =2[stars]; >2 simply multiplies the DF force by this number (cite Tremmel, Governato, Volonteri, & Quinn,2015, MNRAS, 451, 1868)
#BH_DRAG=1                      # drag force on BH due to accretion; =1 uses actual mdot, =2 boost as if BH is accreting at eddington. cite Springel, Di Matteo, and Hernquist, 2005, MNRAS, 361, 776
#BH_REPOSITION_ON_POTMIN=0      # reposition black hole on potential minimum (requires EVALPOTENTIAL). [set =1 to "jump" onto STARS only]
## ----------------------------------------------------------------------------------------------------
# ----- accretion models (modules for gas or other particle accretion)
## ----------------------------------------------------------------------------------------------------
#BH_SWALLOWGAS                  # 'master switch' for accretion (should always be enabled if accretion is on). enables BH to actually eliminate gas particles and take their mass.
#BH_ALPHADISK_ACCRETION         # gas accreted goes into a 'virtual' alpha-disk (mass reservoir), which then accretes onto the BH at the viscous rate (determining luminosity, etc). cite GIZMO methods (or PFH private communication)
#BH_SUBGRIDBHVARIABILITY        # model variability below resolved dynamical time for BH (convolve accretion rate with a uniform power spectrum of fluctuations on timescales below the minimum resolved dynamical time). cite Hopkins & Quataert 2011, MNRAS, 415, 1027
#BH_GRAVCAPTURE_NONGAS          # accretion determined only by resolved gravitational capture by the BH, for non-gas particles (can be enabled with other accretion models for gas). cite Hopkins et al., 2016, MNRAS, 458, 816
## ----
#BH_GRAVCAPTURE_GAS             # accretion determined only by resolved gravitational capture by the BH (for gas particles). cite Hopkins et al., 2016, MNRAS, 458, 816
#BH_GRAVACCRETION=0             # Gravitational torque accretion estimator from Hopkins & Quataert (2011): [=0] B/D decomposition from Angles-Alcazar et al. (default), [=1] approximate f_disk, [=2] gravito-turbulent scaling, [=3] fixed accretion per dynamical time. cite Hopkins & Quataert 2011, MNRAS, 415, 1027 and Angles-Alcazar et al. 2017, MNRAS, 464, 2840
#BH_BONDI=0                     # Bondi-Hoyle style accretion model: 0=default (with velocity); 1=dont use gas velocity with sound speed; 2=variable-alpha tweak (Booth & Schaye 2009). cite Springel, Di Matteo, and Hernquist, 2005, MNRAS, 361, 776
## ----------------------------------------------------------------------------------------------------
# ----- feedback models/options
## ----------------------------------------------------------------------------------------------------
# -- thermal (pure thermal energy injection around BH particle, proportional to BH accretion rate)
#BH_THERMALFEEDBACK             # constant fraction of luminosity coupled in kernel around BH. cite Springel, Di Matteo, and Hernquist, 2005, MNRAS, 361, 776
#--- mechanical (wind from accretion disk/BH with specified mass/momentum/energy-loading relative to accretion rate)
#BH_WIND_CONTINUOUS=0           # gas in kernel given continuous wind flux (energy/momentum/etc). =0 for isotropic, =1 for collimated. cite Hopkins et al., 2016, MNRAS, 458, 816
#BH_WIND_KICK=1                 # gas in kernel given stochastic 'kicks' at fixed velocity. (>0=isotropic, <0=collimated, absolute value sets momentum-loading in L/c units). cite Angles-Alcazar et al., 2017, MNRAS, 464, 2840
#BH_WIND_SPAWN                  # spawn virtual 'wind' particles to carry BH winds out [in development by Paul Torrey]. use requires permissions from P. Torrey and PFH (requires permission, this module is not fully de-bugged and methods not published)
#--- radiative: [FIRE] these currently are built on the architecture of the FIRE stellar FB modules, and require some of those be active. their use therefore follows FIRE policies (see details above).
#BH_COMPTON_HEATING             # enable Compton heating/cooling from BHs in cooling function (needs BH_PHOTONMOMENTUM). cite Hopkins et al., 2016, MNRAS, 458, 816
#BH_HII_HEATING                 # photo-ionization feedback from BH (needs GALSF_FB_HII_HEATING). cite Hopkins et al., arXiv:1702.06148
#BH_PHOTONMOMENTUM              # continuous long-range IR radiation pressure acceleration from BH (needs GALSF_FB_RT_PHOTONMOMENTUM). cite Hopkins et al., arXiv:1702.06148
## ----------------------------------------------------------------------------------------------------
# ----- output options
## ----------------------------------------------------------------------------------------------------
#BH_OUTPUT_MOREINFO             # output additional info to "blackhole_details" on timestep-level, following Angles-Alcazar et al. 2017, MNRAS 472, 109 (use caution: files can get very large if many BHs exist)
#BH_CALC_DISTANCES              # calculate distances for all particles to closest BH for, e.g., refinement, external potentials, etc. cite Garrison-Kimmel et al., MNRAS, 2017, 471, 1709
## ----------------------------------------------------------------------------------------------------
#------------ deprecated or de-bugging options (most have been combined or optimized into the functions above, here for legacy)
##---------------------BH_SEED_GROWTH_TESTS             #- Currently testing options for BH seeding
####################################################################################################





############################################################################################################################-
#--------------------------------------- Radiative Transfer & Radiation Hydrodynamics:
#--------------------------------------------- modules developed by PFH with David Khatami, Mike Grudic, and Nathan Butcher
#---------------------------------------------  (special thanks to Alessandro Lupi): not for use without authors permission [these are proprietary because still in development before public release]
############################################################################################################################-
#--------------------- methods for calculating photon propagation (one, and only one, of these MUST be on for RT)
#RT_LEBRON                              # RT solved using the ray-based LEBRON approximation (locally-extincted background radiation in optically-thin networks; default in the FIRE simulations)
#RT_M1                                  # RT solved using the moments-based 1st-order M1 approximation (solve fluxes and tensors with M1 closure; gives better shadowing; currently only compatible with explicit diffusion solver)
#RT_OTVET                               # RT solved using the moments-based 0th-order OTVET approximation (optically thin Eddington tensor, but interpolated to thick when appropriate)
#RT_FLUXLIMITEDDIFFUSION                # RT solved using the moments-based 0th-order flux-limited diffusion approximation (constant, always-isotropic Eddington tensor)
#RT_LOCALRAYGRID=1                      # RT solved using the exact method of Jiang et al. (each cell carries a mesh in phase space of the intensity directions, rays directly solved over the 6+1D direction-space-frequency-time mesh [value=number of polar angles per octant: N_rays=4*value*(value+1)]
#--------------------- solvers (numerical) --------------------------------------------------------
#RT_SPEEDOFLIGHT_REDUCTION=1            # set to a number <1 to use the 'reduced speed of light' approximation for photon propagation (C_eff=C_true*RT_SPEEDOFLIGHT_REDUCTION)
#RT_DIFFUSION_IMPLICIT                  # solve the diffusion part of the RT equations (if needed) implicitly with Conjugate Gradient iteration (Petkova+Springel): less accurate and only works with some methods, but allows larger timesteps [otherwise more accurate explicit used]
#--------------------- physics: wavelengths+coupled RT-chemistry networks -----------------------------------
#RT_SOURCES=1+16+32                     # source list for ionizing photons given by bitflag (1=2^0=gas,16=2^4=new stars,32=2^5=BH)
#RT_XRAY=3                              # x-rays: 1=soft (0.5-2 keV), 2=hard (>2 keV), 3=soft+hard; used for Compton-heating
#RT_CHEM_PHOTOION=2                     # ionizing photons: 1=H-only [single-band], 2=H+He [four-band]
#RT_LYMAN_WERNER                        # specific lyman-werner [narrow H2 dissociating] band
#RT_PHOTOELECTRIC                       # far-uv (8-13.6eV): track photo-electric heating photons + their dust interactions
#RT_NUV                                 # near-UV: 1550-3600 Angstrom (where direct stellar emission dominates)
#RT_OPTICAL_NIR                         # optical+near-ir: 3600 Angstrom-3 micron (where direct stellar emission dominates)
#RT_INFRARED                            # infrared: photons absorbed in other bands are down-graded to IR: IR radiation + dust + gas temperatures evolved independently
#--------------------- radiation pressure options -------------------------------------------------
#RT_DISABLE_RAD_PRESSURE                # turn off radiation pressure forces (included by default)
#RT_RAD_PRESSURE_OUTPUT                 # print radiation pressure to file (requires some extra variables to save it)
#RT_DISABLE_R15_GRADIENTFIX             # for moments [FLD/OTVET/M1]: turn off the Rosdahl+ 2015 approximate 'fix' (on by default) for gradients under-estimating flux when under-resolved by replacing it with E_nu*c
##-----------------------------------------------------------------------------------------------------
#------------ test-problem, deprecated, or de-bugging functions
##-----------------------------------------------------------------------------------------------------
#RT_SELFGRAVITY_OFF                           # turn off gravity: if using an RT method that needs the gravity tree (FIRE, OTVET), use this -instead- of SELFGRAVITY_OFF to safely turn off gravitational forces
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
# --------------------------------------- Multi-Threading and Parallelization options
####################################################################################################
#OPENMP=2                       # Masterswitch for explicit OpenMP implementation
#PTHREADS_NUM_THREADS=4         # custom PTHREADs implementation (don't enable with OPENMP)
#MULTIPLEDOMAINS=16             # Multi-Domain option for the top-tree level (alters load-balancing)
####################################################################################################



####################################################################################################
# --------------------------------------- Input/Output options
####################################################################################################
#OUTPUT_ADDITIONAL_RUNINFO      # enables extended simulation output data (can slow down machines significantly in massively-parallel runs)
#OUTPUT_IN_DOUBLEPRECISION      # snapshot files will be written in double precision
#INPUT_IN_DOUBLEPRECISION       # input files assumed to be in double precision (otherwise float is assumed)
#OUTPUT_POSITIONS_IN_DOUBLE     # input/output files in single, but positions in double (used in hires, hi-dynamic range sims when positions differ by < float accuracy)
#INPUT_POSITIONS_IN_DOUBLE      # as above, but specific to the ICs file
#OUTPUT_POTENTIAL               # forces code to compute+output potentials in snapshots
#OUTPUT_ACCELERATION            # output physical acceleration of each particle in snapshots
#OUTPUT_CHANGEOFENERGY          # outputs rate-of-change of internal energy of gas particles in snapshots
#OUTPUT_VORTICITY               # outputs the vorticity vector
#OUTPUT_TIMESTEP                # outputs timesteps for each particle
#OUTPUT_COOLRATE                # outputs cooling rate, and conduction rate if enabled
#OUTPUT_LINEOFSIGHT				# enables on-the-fly output of Ly-alpha absorption spectra
#OUTPUT_LINEOFSIGHT_SPECTRUM    # computes power spectrum of these (requires additional code integration)
#OUTPUT_LINEOFSIGHT_PARTICLES   # computes power spectrum of these (requires additional code integration)
#OUTPUT_POWERSPEC               # compute and output power spectra (not used)
#OUTPUT_RECOMPUTE_POTENTIAL     # update potential every output even it EVALPOTENTIAL is set
#INPUT_READ_HSML                # force reading hsml from IC file (instead of re-computing them; in general this is redundant but useful if special guesses needed)
#OUTPUT_TWOPOINT_ENABLED        # allows user to calculate mass 2-point function by enabling and setting restartflag=5
#IO_DISABLE_HDF5                # disable HDF5 I/O support (for both reading/writing; use only if HDF5 not install-able)
#IO_COMPRESS_HDF5     		# write HDF5 in compressed form (will slow down snapshot I/O and may cause issues on old machines, but reduce snapshots 2x) 
####################################################################################################



####################################################################################################
# -------------------------------------------- De-Bugging & special (usually test-problem only) behaviors
####################################################################################################
# --------------------
# ----- General De-Bugging and Special Behaviors
#DEVELOPER_MODE                 # allows you to modify various numerical parameters (courant factor, etc) at run-time
#STOP_WHEN_BELOW_MINTIMESTEP    # forces code to quit when stepsize wants to go below MinSizeTimestep specified in the parameterfile
#DEBUG                          # enables core-dumps and FPU exceptions
# --------------------
# ----- Hydrodynamics
#FREEZE_HYDRO                   # zeros all fluxes from RP and doesn't let particles move (for testing additional physics layers)
#EOS_ENFORCE_ADIABAT=(1.0)      # if set, this forces gas to lie -exactly- along the adiabat P=EOS_ENFORCE_ADIABAT*(rho^GAMMA)
#SLOPE_LIMITER_TOLERANCE=1      # sets the slope-limiters used. higher=more aggressive (less diffusive, but less stable). 1=default. 0=conservative. use on problems where sharp density contrasts in poor particle arrangement may cause errors. 2=same as AGGRESSIVE_SLOPE_LIMITERS below
#AGGRESSIVE_SLOPE_LIMITERS      # use the original GIZMO paper (more aggressive) slope-limiters. more accurate for smooth problems, but
                                # these can introduce numerical instability in problems with poorly-resolved large noise or density contrasts (e.g. multi-phase, self-gravitating flows)
#ENERGY_ENTROPY_SWITCH_IS_ACTIVE # enable energy-entropy switch as described in GIZMO methods paper. This can greatly improve performance on some problems where the
                                # the flow is very cold and highly super-sonic. it can cause problems in multi-phase flows with strong cooling, though, and is not compatible with non-barytropic equations of state
#FORCE_ENTROPIC_EOS_BELOW=(0.01) # set (manually) the alternative energy-entropy switch which is enabled by default in MFM/MFV: if relative velocities are below this threshold, it uses the entropic EOS
#DISABLE_SPH_PARTICLE_WAKEUP    # don't let gas particles move to lower timesteps based on neighbor activity (use for debugging)
#DO_UPWIND_TIME_CENTERING       # this (and DO_HALFSTEP_FOR_MESHLESS_METHODS) use alternative methods for up-winding the fluxes in the MFM/MFV schemes. this up-weighting can be more accurate in hydrostatic problems with a large sound-speed discontinuity -if- the pressure gradient is steady-state, but if they are moving or unstable, it is less accurate (and can suppress mixing)
# --------------------
# ----- Additional Fluid Physics and Gravity
#COOLING_OPERATOR_SPLIT         # do the hydro heating/cooling in operator-split fashion from chemical/radiative. slightly more accurate when tcool >> tdyn, but much noisier when tcool << tdyn
#MHD_ALTERNATIVE_LEAPFROG_SCHEME # use alternative leapfrog where magnetic fields are treated like potential/positions (per Federico Stasyszyn's suggestion): still testing
#SUPER_TIMESTEP_DIFFUSION       # use super-timestepping to accelerate integration of diffusion operators [for testing or if there are stability concerns]
#EVALPOTENTIAL                  # computes gravitational potential
# --------------------
# ----- Particle IDs
#TEST_FOR_IDUNIQUENESS          # explicitly check if particles have unique id numbers (only use for special behaviors)
#LONGIDS                        # use long ints for IDs (needed for super-large simulations)
#ASSIGN_NEW_IDS                 # assign IDs on startup instead of reading from ICs
#NO_CHILD_IDS_IN_ICS            # IC file does not have child IDs: do not read them (used for compatibility with snapshot restarts from old versions of the code)
# --------------------
# ----- Particle Merging/Splitting/Deletion/Boundaries
#PREVENT_PARTICLE_MERGE_SPLIT   # don't allow gas particle splitting/merging operations
#PARTICLE_EXCISION              # enable dynamical excision (remove particles within some radius)
#MERGESPLIT_HARDCODE_MAX_MASS=(1.0e-6)   # manually set maximum mass for particle merge-split operations (in code units): useful for snapshot restarts and other special circumstances
#MERGESPLIT_HARDCODE_MIN_MASS=(1.0e-7)   # manually set minimum mass for particle merge-split operations (in code units): useful for snapshot restarts and other special circumstances
# --------------------
# ----- MPI & Parallel-FFTW De-Bugging
#USE_MPI_IN_PLACE               # MPI debugging: makes AllGatherV compatible with MPI_IN_PLACE definitions in some MPI libraries
#NO_ISEND_IRECV_IN_DOMAIN       # MPI debugging: slower, but fixes memory errors during exchange in the domain decomposition (ANY RUN with >2e9 particles MUST SET THIS OR FAIL!)
#FIX_PATHSCALE_MPI_STATUS_IGNORE_BUG # MPI debugging
#MPISENDRECV_SIZELIMIT=100      # MPI debugging
#MPISENDRECV_CHECKSUM           # MPI debugging
#DONOTUSENODELIST               # MPI debugging
#NOTYPEPREFIX_FFTW              # FFTW debugging (fftw-header/libraries accessed without type prefix, adopting whatever was
                                #   chosen as default at compile of fftw). Otherwise, the type prefix 'd' for double is used.
#DOUBLEPRECISION_FFTW           # FFTW in double precision to match libraries
# --------------------
# ----- Load-Balancing
#ALLOW_IMBALANCED_GASPARTICLELOAD # increases All.MaxPartSph to All.MaxPart: can allow better load-balancing in some cases, but uses more memory. But use me if you run into errors where it can't fit the domain (where you would increase PartAllocFac, but can't for some reason)
#SEPARATE_STELLARDOMAINDECOMP   # separate stars (ptype=4) and other non-gas particles in domain decomposition (may help load-balancing)
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
#------------------ Use of these modules follows the GADGET-3 permissions; if you are not sure, contact Volker
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
#NUCLEARNET_NEGLECT_DTDY_TERMS
#NUCLEARNET_OUTPUT_TIMEEVOLUTION
####################################################################################################-



####################################################################################################-
##----------------------------------------------------------------------------------------------------
#--------------------------------------- on the fly analysis of phase structure and potentials in DM simulations
#---------------------------------------  (this code works, but requires additional modules not part of GIZMO but from GADGET-3,
#---------------------------------------   where they were developed. users who wish to incorporate a new version/fix this are welcome)
##----------------------------------------------------------------------------------------------------
#------------------------------- Fine-grained phase space structure analysis (M. Vogelsberger)
#-------------------------------- use of these routines requires explicit pre-approval by developer M. Vogelsberger
#GDE_DISTORTIONTENSOR           #- main switch: integrate phase-space distortion tensor
#GDE_TYPES=2+4+8+16+32          #- track GDE for these types
#GDE_READIC                     #- read initial sheet orientation/initial density/initial caustic count from ICs
#GDE_LEAN                       #- lean version of GDE
#OUTPUT_DISTORTIONTENSOR        #- write phase-space distortion tensor to snapshot
#OUTPUT_TIDALTENSORPS           #- write configuration-space tidal tensor to snapshot
#OUTPUT_LAST_CAUSTIC            #- write info on last passed caustic to snapshot
##-----------------------------------------------------------------------------------------------------
#--------------------------------------- SCF (potential expansion in basis functions)
#-------------------------------- use of these routines requires explicit pre-approval by developers M. Vogelsberger & L. Hernquist
#SCFPOTENTIAL                   #- turn SCF on/off
#SCF_HYBRID=1                   #- =1:tree:stars<->stars,DM<->DM,stars->DM/SCF:stars<-DM =2:tree:stars<->stars,stars->DM/SCF:stars<-DM,DM<->DM
#SCF_HQ_MASS=95.2401            #- mass of halo of expansion basis
#SCF_HQ_A=29.7754               #- scale length of halo of expansion basis
#SCF_NEFF=5000000               #- effective particle number in halo
#SCF_NMAX=1                     #- maximum n for expansion cut-off
#SCF_LMAX=1                     #- maximum l for expansion cut-off
#SCF_SCALEFAC                   #- read-in scale factors for coefficients from file scf_scalefac.dat (must be created by user)
####################################################################################################-



