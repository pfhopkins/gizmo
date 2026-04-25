/* gpu_gravtree.cc — Step 13 Phase 4 Tier 1a / Phase 2-A
 *
 * GPU gravity walk (mode=0 primary-tree path).  Tier 1a–1c: core walk with
 * PMGRID, ADAPTIVE_GRAVSOFT_FORALL, SYMMETRIZE, EVALPOTENTIAL.  Phase 2-A:
 * RT cluster payloads (RT_USE_GRAVTREE, GALSF_FB_FIRE_RT_LONGRANGE,
 * CHIMES_STELLAR_FLUXES, RT_USE_TREECOL_FOR_NH).
 *
 * rt_get_source_luminosity() is not GPU-callable; it is pre-computed on CPU
 * for all local particles into a SharedSpace array (d_src_lum) before kernel
 * launch.  Node stellar luminosities come from the Phase 2-I SoA extension.
 * rt_kappa() is KOKKOS_INLINE_FUNCTION and runs on device for RT_LEBRON
 * fac_stellum initialisation.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef OPENMP_GPU_OFFLOAD
#include <Kokkos_Core.hpp>
#endif

#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../system/gpu_particles_arena.h"
#include "gpu_gravity_tree.h"
#include "gpu_gravtree.h"
#include "forcetree.h"

#include "../mesh/kernel.h"

#if defined(GIZMO_GPU_GRAVTREE) && defined(OPENMP_GPU_OFFLOAD)

/* Globals that live at file-scope in gravtree.cc without a header declaration. */
extern int Ewald_iter;
extern double Costtotal;

#ifdef PMGRID
/* Short-range tables live as file-scope (non-static) globals in forcetree.cc.
 * NTAB matches the #define there. */
#define GIZMO_GPU_GRAVTREE_NTAB 1000
extern float shortrange_table[GIZMO_GPU_GRAVTREE_NTAB];
extern float shortrange_table_potential[GIZMO_GPU_GRAVTREE_NTAB];
#endif

#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL) || defined(GALSF_MERGER_STARCLUSTER_PARTICLES)
/* GPU-callable mirror of ForceSoftening_KernelRadius(). */
static KOKKOS_INLINE_FUNCTION
double gpu_force_softening_kernel_radius(const struct particle_data *Pp, int p)
{
#ifdef GALSF_MERGER_STARCLUSTER_PARTICLES
    if(Pp[p].Type == 4) {return Pp[p].StarParticleEffectiveSize;}
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORALL)
    if((1 << Pp[p].Type) & (ADAPTIVE_GRAVSOFT_FORALL)) {return Pp[p].AGS_KernelRadius;}
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(SELFGRAVITY_OFF)
    if(Pp[p].Type == 0) {
#if defined(ADAPTIVE_GRAVSOFT_MAX_SOFT_HARD_LIMIT)
        double cap = ADAPTIVE_GRAVSOFT_MAX_SOFT_HARD_LIMIT / All.cf_atime;
        return (Pp[p].KernelRadius < cap) ? Pp[p].KernelRadius : cap;
#else
        return Pp[p].KernelRadius;
#endif
    }
#endif
    return All.ForceSoftening[Pp[p].Type];
}

static KOKKOS_INLINE_FUNCTION
double gpu_get_ags_zeta(const struct particle_data *Pp, int p)
{
    return Pp[p].AGS_zeta;
}
#endif

#if defined(ADAPTIVE_GRAVSOFT_FORALL)
/* Device-callable replica of ags_gravity_kernel_shared_BITFLAG (ags_rkern.cc). */
static KOKKOS_INLINE_FUNCTION int
gpu_ags_kernel_shared_BITFLAG(int ptype)
{
    if(!((1 << ptype) & (ADAPTIVE_GRAVSOFT_FORALL))) {return 0;}
    if(ptype == 0) {return 1;}
#if (ADAPTIVE_GRAVSOFT_FORALL & 32) && defined(SINK_PARTICLES)
    if(ptype == 5) {return 1;}
#endif
#if defined(GALSF) && ((ADAPTIVE_GRAVSOFT_FORALL & 16) || (ADAPTIVE_GRAVSOFT_FORALL & 8) || (ADAPTIVE_GRAVSOFT_FORALL & 4))
    if(All.ComovingIntegrationOn) {
        if(ptype == 4) {return 17;}
    } else {
        if(ptype == 4 || ptype == 2 || ptype == 3) {return 29;}
    }
#endif
#ifdef DM_SIDM
    if((1 << ptype) & (DM_SIDM)) {return DM_SIDM;}
#endif
    return (1 << ptype);
}
#endif

/* Phase 2-A: RT payload data passed to the GPU walk kernel.
 * src_lum[p * N_RT_FREQ_BINS + kf] = per-particle luminosity precomputed on
 * CPU via rt_get_source_luminosity().  Only populated when RT_USE_GRAVTREE is
 * active.  Sized for [NumPart * N_RT_FREQ_BINS] in SharedSpace. */
#ifdef RT_USE_GRAVTREE
#include "../radiation/rt_functions.h"
struct gpu_rt_walk_data_t {
    MyFloat *src_lum;             /* [NumPart * N_RT_FREQ_BINS] */
#ifdef CHIMES_STELLAR_FLUXES
    double  *src_lum_G0;          /* [NumPart * CHIMES_LOCAL_UV_NBINS] */
    double  *src_lum_ion;         /* [NumPart * CHIMES_LOCAL_UV_NBINS] */
#endif
};
#endif

/* Phase 2-C: sink radiation payload.  Pre-computed on CPU before the kernel
 * launches because sink_lum_bol() (and, under SINGLE_STAR_SINK_DYNAMICS,
 * calculate_individual_stellar_luminosity()) are not GPU-callable.
 * bh_lum[p]   = sink_lum_bol(P[p].Sink_Mdot, P[p].Sink_Mass, p) when P[p]
 *               is a valid type-5 sink with Mdot>0, else 0.
 * bh_angle[p] = P[p].Sink_Specific_AngMom (if SINK_FOLLOW_ACCRETED_ANGMOM)
 *               or P[p].GradRho otherwise.  Used for angle-weighted
 *               luminosity at particle leafs.  Node-level sink_lum /
 *               sink_lum_grad already live in the SoA (Phase 2-I). */
/* Phase 2-D: COSMIC_RAY_SUBGRID_LEBRON payload.  cr_get_source_injection_rate
 * is not GPU-callable so per-particle injection is precomputed on host.
 * t_max_cr = DMIN(1., evaluate_time_since_t_initial_in_Gyr(All.TimeBegin))/
 * UNIT_TIME_IN_GYR, passed as scalar (independent of target). */
#ifdef COSMIC_RAY_SUBGRID_LEBRON
struct gpu_cr_walk_data_t {
    MyFloat *cr_inject;       /* [NumPart] */
    double   t_max_cr;        /* scalar, in code time units */
};
#endif

#ifdef SINK_PHOTONMOMENTUM
struct gpu_sink_walk_data_t {
    MyFloat       *bh_lum;    /* [NumPart] */
    Vec3<MyFloat> *bh_angle;  /* [NumPart] */
};

/* Device-callable replica of sink_fb_angleweight (sinks/sink.cc:199). */
static KOKKOS_INLINE_FUNCTION double
gpu_sink_fb_angleweight(double sink_lum_input, Vec3<MyFloat> sink_angle,
                        double dx, double dy, double dz)
{
#ifdef SINGLE_STAR_SINK_DYNAMICS
    return sink_lum_input;
#else
    if(sink_lum_input <= 0) return 0.0;
    double r2 = dx*dx + dy*dy + dz*dz;
    if(r2 <= 0) return 0.0;
    double cf = All.cf_atime;
    if(r2 * (UNIT_LENGTH_IN_PC*UNIT_LENGTH_IN_PC) * (cf*cf) < 1.0) return 0.0; /* no force at < 1pc */
#if defined(SINK_FB_COLLIMATED)
    double sa_ns = (double)sink_angle[0]*sink_angle[0] + (double)sink_angle[1]*sink_angle[1] + (double)sink_angle[2]*sink_angle[2];
    double cos_theta;
    if(sa_ns > 0) {
        cos_theta = fabs((dx*(double)sink_angle[0] + dy*(double)sink_angle[1] + dz*(double)sink_angle[2]) / sqrt(r2 * sa_ns));
        if(!isfinite(cos_theta)) {cos_theta = 1.0;}
    } else {
        cos_theta = 1.0;
    }
    double wt_normalized = 0.0847655 * exp(4.5*cos_theta*cos_theta);
    return sink_lum_input * wt_normalized;
#else
    return sink_lum_input;
#endif
#endif /* SINGLE_STAR_SINK_DYNAMICS */
}
#endif /* SINK_PHOTONMOMENTUM */

/* -------------------------------------------------------------------------
 * Compile-time payload gates.
 * Tier 1c + Phase 2-A flags are now unlocked.  Everything else that hasn't
 * been ported remains #error'd so wrong-physics on those configs is caught
 * at compile time rather than producing silent incorrect results.
 * ---------------------------------------------------------------------- */
#if defined(ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION)
#error "GIZMO_GPU_GRAVTREE does not yet support ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION (Tier 3)."
#endif
#if defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM)
#error "GIZMO_GPU_GRAVTREE does not yet support SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM type-4 softening override (defer)."
#endif
/* GALSF_MERGER_STARCLUSTER_PARTICLES (type-4 star cluster softening via
 * StarParticleEffectiveSize) handled inline in gpu_force_softening_kernel_radius
 * below — enabled for Phase 2-D. */
/* ADAPTIVE_GRAVSOFT_MAX_SOFT_HARD_LIMIT (type-0 softening cap) handled inline
 * in gpu_force_softening_kernel_radius below — enabled for Phase 2-C. */
/* COMPUTE_TIDAL_TENSOR_IN_GRAVTREE + COMPUTE_JERK_IN_GRAVTREE: ported in Phase 5
 * (ATFU). Tidal tensor accumulation + jerk both mirror forcetree.cc:2081-2292.
 * Three sub-cases remain out-of-scope and are #error'd below:
 *   - PMGRID short-range tidal table (shortrange_table_tidal not mirrored)
 *   - AGS_SYMMETRIZE_FORCE_BY_AVERAGING tidal averaging branch
 *   - GRAVITY_SPHERICAL_SYMMETRY + ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION corners */
#if defined(COMPUTE_TIDAL_TENSOR_IN_GRAVTREE) && defined(PMGRID)
#error "GIZMO_GPU_GRAVTREE + COMPUTE_TIDAL_TENSOR + PMGRID: shortrange_table_tidal mirror not implemented (Phase 5 scope)."
#endif
#if defined(COMPUTE_TIDAL_TENSOR_IN_GRAVTREE) && defined(ADAPTIVE_GRAVSOFT_SYMMETRIZE_FORCE_BY_AVERAGING)
#error "GIZMO_GPU_GRAVTREE + COMPUTE_TIDAL_TENSOR + AGS_SYMMETRIZE_BY_AVERAGING: tidal averaging branch not implemented (Phase 5 scope)."
#endif
#if defined(COMPUTE_TIDAL_TENSOR_IN_GRAVTREE) && (defined(ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION) || defined(GRAVITY_SPHERICAL_SYMMETRY))
#error "GIZMO_GPU_GRAVTREE + COMPUTE_TIDAL_TENSOR + TIDAL_CRITERION/SPHERICAL: specialized branches not implemented (Phase 5 scope)."
#endif
/* SINK_PHOTONMOMENTUM, SINK_COMPTON_HEATING, SINK_DYNFRICTION_FROMTREE:
 * ported in Phase 2-C. */
#if defined(SPECIAL_POINT_MOTION) || defined(SPECIAL_POINT_WEIGHTED_MOTION)
#error "GIZMO_GPU_GRAVTREE does not yet support SPECIAL_POINT_MOTION payloads (deferred)."
#endif
/* SINK_CALC_DISTANCES, SINGLE_STAR_SINK_DYNAMICS, SINGLE_STAR_TIMESTEPPING,
 * SINGLE_STAR_FIND_BINARIES, SINGLE_STAR_FB_TIMESTEPLIMIT, SINGLE_STAR_STARFORGE_DEFAULTS:
 * ported in Phase 2-B. */
/* COSMIC_RAY_SUBGRID_LEBRON: ported in Phase 2-D. Reads SoA cr_injection at
 * node accepts + per-particle precomputed d_cr_inject at leaf nodes (since
 * cr_get_source_injection_rate is not GPU-callable). */
#if defined(DM_SCALARFIELD_SCREENING) || defined(GRAVITY_SPHERICAL_SYMMETRY) || defined(COUNT_MASS_IN_GRAVTREE)
#error "GIZMO_GPU_GRAVTREE does not yet support DM/spherical/mass-count payloads (Tier 3)."
#endif
#if defined(HERMITE_INTEGRATION) || defined(NEIGHBORS_MUST_BE_COMPUTED_EXPLICITLY_IN_FORCETREE)
#error "GIZMO_GPU_GRAVTREE does not yet support HERMITE / NEIGHBORS_MUST_BE_COMPUTED (Tier 3)."
#endif
/* ADAPTIVE_TREEFORCE_UPDATE: Phase 5.  Pre-walk skip-flag filtering in the
 * dispatcher + jerk accumulation in the walk kernel.  Skip-flag particles
 * (needs_new_treeforce()==0) bypass the GPU walk and use the CPU post-loop
 * jerk extrapolation path at gravtree.cc:512-520 (GravAccel += GravJerk*dt). */
/* Periodic boundary handling:
 *   BOX_PERIODIC + PMGRID   → TreePM.  Long-range forces come from PM; the tree
 *                             walk is short-range-only (rcut-truncated via the
 *                             shortrange-force tables already wired in).  The
 *                             CPU never calls force_treeevaluate_ewald_correction
 *                             in this case (see gravtree.cc:734 gate).  The GPU
 *                             primary walk is therefore already correct.
 *   BOX_PERIODIC + !PMGRID  → pure-tree periodic.  Requires the second Ewald-
 *                             correction walk; the GPU port of that walk lives
 *                             in gpu_ewald_walk_primary (dispatched from
 *                             gravtree.cc after the primary walk).
 *   GRAVITY_NOT_PERIODIC    → non-periodic box, Ewald not relevant. */
/* Pure-tree periodic gravity (BOX_PERIODIC && !GRAVITY_NOT_PERIODIC && !PMGRID)
 * is supported via the GPU Ewald walk (gpu_ewald_walk_primary), dispatched
 * from gravtree.cc on the Ewald_iter==1 pass. Implementation later in this
 * file. */
#if defined(SELFGRAVITY_OFF)
#error "GIZMO_GPU_GRAVTREE requires self-gravity to be enabled."
#endif


/* -------------------------------------------------------------------------
 * gpu_gravtree_walk_one — device-side walk for a single target particle.
 *
 * Returns 1 on success (acc written), 0 on failure (pseudo-particle hit;
 * host must run CPU walk for this target).  Mirrors force_treeevaluate()
 * mode=0 with SINK/CR/DM/tidal payload branches stripped (gated above).
 * Phase 2-A RT payloads (RT_USE_GRAVTREE, treecol, CHIMES, FIRE longrange)
 * are included here; they accumulate into CellP_dev[target] which the host
 * scatter loop copies back to CellP[].
 * ---------------------------------------------------------------------- */
static KOKKOS_INLINE_FUNCTION int
gpu_gravtree_walk_one(int target,
                      int maxPart, int maxNodes,
                      struct particle_data *P_dev,
                      struct gas_cell_data *CellP_dev,
                      const struct gpu_gravity_tree_soa_t *s,
#ifdef PMGRID
                      double rcut, double rcut2, double asmthfac,
                      const float *shortrange_tab, const float *shortrange_pot_tab,
#endif
#ifdef RT_USE_GRAVTREE
                      const struct gpu_rt_walk_data_t *rt_data,
#endif
#ifdef SINK_PHOTONMOMENTUM
                      const struct gpu_sink_walk_data_t *sink_data,
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
                      const struct gpu_cr_walk_data_t *cr_data,
#endif
                      Vec3<double> &acc_out,
                      int &ninter_out,
                      double &pot_out)
{
    Vec3<double> pos = P_dev[target].Pos;
    int ptype = P_dev[target].Type;
    double pmass = P_dev[target].Mass;
    if(pmass <= 0) {acc_out = Vec3<double>{0,0,0}; ninter_out = 0; pot_out = 0.0; return 1;}

#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL) || defined(GALSF_MERGER_STARCLUSTER_PARTICLES)
    double soft = gpu_force_softening_kernel_radius(P_dev, target);
#else
    double soft = All.ForceSoftening[ptype];
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL)
    double zeta = 0.0;
#if defined(ADAPTIVE_GRAVSOFT_FORGAS)
    if(ptype == 0) {
        if(soft > All.ForceSoftening[ptype]) { zeta = gpu_get_ags_zeta(P_dev, target); }
        else { soft = All.ForceSoftening[ptype]; zeta = 0; }
    }
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORALL)
    if(soft > All.ForceSoftening[ptype]) { zeta = gpu_get_ags_zeta(P_dev, target); }
    else { soft = All.ForceSoftening[ptype]; zeta = 0; }
#endif
#endif
    double aold = All.ErrTolForceAcc * P_dev[target].OldAcc;

#if defined(ADAPTIVE_GRAVSOFT_FORALL)
    const int ags_bitflag_primary = gpu_ags_kernel_shared_BITFLAG(ptype);
#endif

    /* ------------------------------------------------------------------ *
     * Phase 2-B: SINK_CALC_DISTANCES + SINGLE_STAR_* local accumulators.  *
     * Mirrors forcetree.cc:1509-1526.                                     *
     * ------------------------------------------------------------------ */
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SINK_DYNFRICTION_FROMTREE) || defined(COMPUTE_JERK_IN_GRAVTREE)
    Vec3<double> vel = P_dev[target].Vel;
#endif
#ifdef COMPUTE_JERK_IN_GRAVTREE
    Vec3<double> jerk_acc = {0,0,0};
#endif
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
    SymmetricTensor2<double> tidal_acc = {};
#endif
#ifdef SINK_DYNFRICTION_FROMTREE
    double target_sink_mass = (ptype == 5) ? P_dev[target].Sink_Mass : 0.0;
#endif
#ifdef SINK_COMPTON_HEATING
    double incident_flux_agn = 0.0;
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    double SubGrid_CosmicRayEnergyDensity = 0.0;
    double cr_injection = 0.0; /* per-interaction; reset in leaf/node blocks */
#endif
#ifdef SINK_CALC_DISTANCES
    double Min_Distance_to_Sink2 = MAX_REAL_NUMBER;
    Vec3<double> Min_xyz_to_Sink = {MAX_REAL_NUMBER, MAX_REAL_NUMBER, MAX_REAL_NUMBER};
#endif
#ifdef SINGLE_STAR_TIMESTEPPING
    double Min_Sink_Approach_Time = MAX_REAL_NUMBER;
    double Min_Sink_Freefall_time = MAX_REAL_NUMBER;
#endif
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
    double Min_Sink_FeedbackTime = MAX_REAL_NUMBER;
#endif
#ifdef SINGLE_STAR_FIND_BINARIES
    double Min_Sink_OrbitalTime = MAX_REAL_NUMBER;
    double comp_Mass_local = 0.0;
    Vec3<double> comp_dx_local = {0,0,0}, comp_dv_local = {0,0,0};
#endif

    /* ------------------------------------------------------------------ *
     * RT cluster local accumulators (Phase 2-A).  All gated by the same   *
     * #ifdefs as the CPU walk in forcetree.cc.                             *
     * ------------------------------------------------------------------ */
#ifdef RT_USE_TREECOL_FOR_NH
    const double angular_bin_size = 4.0 * M_PI / RT_USE_TREECOL_FOR_NH;
    double treecol_angular_bins[RT_USE_TREECOL_FOR_NH];
    {int kb; for(kb=0; kb<RT_USE_TREECOL_FOR_NH; kb++) {treecol_angular_bins[kb]=0.0;}}
#endif

#ifdef RT_USE_GRAVTREE
    double mass_stellarlum[N_RT_FREQ_BINS];
    {int kf; for(kf=0; kf<N_RT_FREQ_BINS; kf++) {mass_stellarlum[kf]=0.0;}}
#ifdef CHIMES_STELLAR_FLUXES
    double chimes_mass_stellarlum_G0[CHIMES_LOCAL_UV_NBINS];
    double chimes_mass_stellarlum_ion[CHIMES_LOCAL_UV_NBINS];
    double chimes_flux_G0[CHIMES_LOCAL_UV_NBINS];
    double chimes_flux_ion[CHIMES_LOCAL_UV_NBINS];
    {int kc; for(kc=0; kc<CHIMES_LOCAL_UV_NBINS; kc++) {chimes_mass_stellarlum_G0[kc]=0; chimes_mass_stellarlum_ion[kc]=0; chimes_flux_G0[kc]=0; chimes_flux_ion[kc]=0;}}
#endif
    Vec3<double> d_stellarlum = {};
#ifdef SINK_PHOTONMOMENTUM
    double mass_sinklumwt_forradfb = 0.0; /* per-interaction; reset in leaf/node blocks */
#endif
    /* Mirror CPU's valid_gas_particle_for_rt gate (forcetree.cc:1595) */
    volatile int valid_gas_particle_for_rt = (ptype == 0 && soft > 0 && pmass > 0) ? 1 : 0; /* volatile: nvc++ constant-propagates this to 0 inside the walk loop otherwise */
#ifdef RT_OTVET
    SymmetricTensor2<double> RT_ET[N_RT_FREQ_BINS];
    {int kf; for(kf=0; kf<N_RT_FREQ_BINS; kf++) {RT_ET[kf] = {};}}
#endif
#endif /* RT_USE_GRAVTREE */

#ifdef GALSF_FB_FIRE_RT_LONGRANGE
    double incident_flux_uv = 0.0, incident_flux_euv = 0.0;
#endif
#if defined(RT_USE_GRAVTREE_SAVE_RAD_ENERGY)
    double Rad_E_gamma[N_RT_FREQ_BINS];
    {int kf; for(kf=0; kf<N_RT_FREQ_BINS; kf++) {Rad_E_gamma[kf]=0.0;}}
#endif
#if defined(RT_USE_GRAVTREE_SAVE_RAD_FLUX)
    Vec3<double> Rad_Flux[N_RT_FREQ_BINS];
    {int kf; for(kf=0; kf<N_RT_FREQ_BINS; kf++) {Rad_Flux[kf]={};}}
#endif

    /* RT_LEBRON radiation-pressure coupling factor (forcetree.cc:1596-1604).
     * Pre-computed once per target before the walk loop because it only
     * depends on the target's properties.  Skipped when save-flux mode is
     * active (flux is stored and converted to RP after the walk by the caller). */
#if defined(RT_USE_GRAVTREE) && defined(RT_LEBRON) && !defined(RT_USE_GRAVTREE_SAVE_RAD_FLUX)
    double fac_stellum[N_RT_FREQ_BINS];
    {int kf; for(kf=0; kf<N_RT_FREQ_BINS; kf++) {fac_stellum[kf]=0.0;}}
    if(valid_gas_particle_for_rt) {
        double h_eff_phys = soft * pow(VOLUME_NORM_COEFF_FOR_NDIMS / (double)All.DesNumNgb, 1.0/NUMDIMS) * All.cf_atime;
        double sigma_particle = pmass / (h_eff_phys * h_eff_phys);
        double fac_stellum_0 = -All.PhotonMomentum_Coupled_Fraction / (4.0*M_PI * C_LIGHT_CODE_REDUCED * sigma_particle * All.G);
        int kf; for(kf=0; kf<N_RT_FREQ_BINS; kf++) {
            fac_stellum[kf] = fac_stellum_0 * (1.0 - exp(-rt_kappa(-1, kf, P_dev, CellP_dev) * sigma_particle));
        }
    }
#endif

    Vec3<double> acc = {0,0,0};
    int ninter = 0;
    double pot = 0.0;

    int no = maxPart;   /* root */

    while(no >= 0)
    {
        double h = soft, h_p = -1.0;
        Vec3<double> dr;
        double r2, mass;
#if defined(SINK_DYNFRICTION_FROMTREE) || defined(COMPUTE_JERK_IN_GRAVTREE)
        Vec3<double> dv = {0,0,0};
#endif
#ifdef SINK_DYNFRICTION_FROMTREE
        double m_j_eff_for_df = 0.0;
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL)
        int ptype_sec = -1;
        double zeta_sec = 0.0;
#endif
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
        double gasmass = 0.0;
#endif

        if(no < maxPart) /* particle leaf */
        {
            dr = P_dev[no].Pos - pos;
            r2 = dr.norm_sq();
            mass = P_dev[no].Mass;
#if defined(SINK_DYNFRICTION_FROMTREE) || defined(COMPUTE_JERK_IN_GRAVTREE)
            dv = P_dev[no].Vel - vel;
#endif
#ifdef SINK_DYNFRICTION_FROMTREE
            m_j_eff_for_df = mass;
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL) || defined(GALSF_MERGER_STARCLUSTER_PARTICLES)
            h_p = gpu_force_softening_kernel_radius(P_dev, no);
            ptype_sec = P_dev[no].Type;
#if defined(ADAPTIVE_GRAVSOFT_FORGAS)
            if(ptype_sec == 0) {zeta_sec = gpu_get_ags_zeta(P_dev, no);}
#elif defined(ADAPTIVE_GRAVSOFT_FORALL)
            zeta_sec = gpu_get_ags_zeta(P_dev, no);
#endif
#endif
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
            gasmass = (P_dev[no].Type == 0) ? P_dev[no].Mass : 0.0;
#endif
#ifdef RT_USE_GRAVTREE
            /* Load leaf luminosity unconditionally (matching node path; valid_gas gate in force kernel). */
            {
                d_stellarlum = dr;
                int kf; for(kf=0; kf<N_RT_FREQ_BINS; kf++) {
                    mass_stellarlum[kf] = rt_data->src_lum[(long)no * N_RT_FREQ_BINS + kf];
                }
#ifdef CHIMES_STELLAR_FLUXES
                for(kf=0; kf<CHIMES_LOCAL_UV_NBINS; kf++) {
                    chimes_mass_stellarlum_G0[kf] = rt_data->src_lum_G0[(long)no * CHIMES_LOCAL_UV_NBINS + kf];
                    chimes_mass_stellarlum_ion[kf] = rt_data->src_lum_ion[(long)no * CHIMES_LOCAL_UV_NBINS + kf];
                }
#endif
#ifdef SINK_PHOTONMOMENTUM
                /* Mirror forcetree.cc:1756-1767: per-sink-leaf angle-weighted luminosity. */
                mass_sinklumwt_forradfb = 0.0;
                if(P_dev[no].Type == 5) {
                    double bhlum_t = (double) sink_data->bh_lum[no];
                    mass_sinklumwt_forradfb = gpu_sink_fb_angleweight(bhlum_t, sink_data->bh_angle[no],
                                                                     dr[0], dr[1], dr[2]);
                }
#endif
            }
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
            /* Mirror forcetree.cc:1734-1736 leaf CR source injection. */
            cr_injection = (double) cr_data->cr_inject[no];
#endif
            /* Phase 2-B: sink-distance + timestepping tracking on particle leafs.
             * Mirrors forcetree.cc:1669-1732. Only fires when source is a sink
             * particle (Type == SPECIAL_POINT_TYPE_FOR_NODE_DISTANCES). */
#ifdef SINK_CALC_DISTANCES
            if((r2 > 0) && (mass > 0) && (P_dev[no].Type == SPECIAL_POINT_TYPE_FOR_NODE_DISTANCES))
            {
                if(r2 < Min_Distance_to_Sink2) {
                    Min_Distance_to_Sink2 = r2;
                    Min_xyz_to_Sink = dr;
                }
#ifdef SINGLE_STAR_TIMESTEPPING
                Vec3<double> sink_dv = P_dev[no].Vel - vel;
                double vSqr = sink_dv.norm_sq();
                double M_total = P_dev[no].Mass + pmass;
                double r2soft = SinkParticle_GravityKernelRadius;
                if(r2soft < soft) {r2soft = soft;}
                r2soft *= KERNEL_FAC_FROM_FORCESOFT_TO_PLUMMER;
                r2soft = r2 + r2soft * r2soft;
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
                if(ptype == 0) {
                    double tSqr_fb = r2soft / (P_dev[no].MaxFeedbackVel * P_dev[no].MaxFeedbackVel + MIN_REAL_NUMBER);
                    if(tSqr_fb < Min_Sink_FeedbackTime) {Min_Sink_FeedbackTime = tSqr_fb;}
                }
#endif
                double tSqr = r2soft / (vSqr + MIN_REAL_NUMBER);
                double tff4 = r2soft * r2soft * r2soft / (M_total * M_total);
                if(tSqr < Min_Sink_Approach_Time) {Min_Sink_Approach_Time = tSqr;}
                if(tff4 < Min_Sink_Freefall_time) {Min_Sink_Freefall_time = tff4;}
#ifdef SINGLE_STAR_FIND_BINARIES
                if(ptype == 5) {
                    double r_p5 = sqrt(r2);
                    double specific_energy = 0.5 * vSqr - All.G * M_total / r_p5;
                    if(r2 < SinkParticle_GravityKernelRadius * SinkParticle_GravityKernelRadius) {
                        double hinv_p5 = 1.0 / SinkParticle_GravityKernelRadius;
                        specific_energy = 0.5 * vSqr + All.G * M_total *
                            kernel_gravity(r_p5*hinv_p5, hinv_p5, hinv_p5*hinv_p5*hinv_p5, -1);
                    }
                    if(specific_energy < 0) {
                        double semimajor_axis = -All.G * M_total / (2.0 * specific_energy);
                        double t_orbital = 2.0 * M_PI *
                            sqrt(semimajor_axis*semimajor_axis*semimajor_axis / (All.G * M_total));
                        if(t_orbital < Min_Sink_OrbitalTime) {
                            Min_Sink_OrbitalTime = t_orbital;
                            comp_Mass_local = P_dev[no].Mass;
                            comp_dx_local = dr;
                            comp_dv_local = sink_dv;
                        }
                    }
                }
#endif /* SINGLE_STAR_FIND_BINARIES */
#endif /* SINGLE_STAR_TIMESTEPPING */
            }
#endif /* SINK_CALC_DISTANCES */
        }
        else if(no >= maxPart + maxNodes) /* pseudo-particle — remote */
        {
            return 0; /* host runs CPU walk for this target */
        }
        else /* tree node */
        {
            int idx = no - maxPart;
            Vec3<MyFloat> s_node = Vec3<MyFloat>{(MyFloat)s->s[idx][0], (MyFloat)s->s[idx][1], (MyFloat)s->s[idx][2]};
            MyFloat len_node = s->len[idx];
            MyFloat msoft_node = s->maxsoft[idx];
            MyFloat mass_node = s->mass[idx];
            Vec3<MyFloat> center_node = s->center[idx];

            dr[0] = s_node[0] - pos[0];
            dr[1] = s_node[1] - pos[1];
            dr[2] = s_node[2] - pos[2];
            r2 = dr.norm_sq();

#ifdef PMGRID
            if(r2 > rcut2)
            {
                double eff_dist = rcut + 0.5 * len_node;
                double dcx = fabs(center_node[0] - pos[0]);
                double dcy = fabs(center_node[1] - pos[1]);
                double dcz = fabs(center_node[2] - pos[2]);
                if(dcx > eff_dist || dcy > eff_dist || dcz > eff_dist) {
                    no = s->sibling[idx]; continue;
                }
            }
#endif

            if(h < msoft_node) {
                if(r2 < msoft_node * msoft_node) {
                    no = s->nextnode[idx]; continue;
                }
            }

            if(All.ErrTolTheta)
            {
                if(len_node * len_node > r2 * All.ErrTolTheta * All.ErrTolTheta) {
                    no = s->nextnode[idx]; continue;
                }
            }
            else
            {
                if((r2 < (soft + 0.6*len_node)*(soft + 0.6*len_node)) ||
                   (r2 < (msoft_node + 0.6*len_node)*(msoft_node + 0.6*len_node))) {
                    no = s->nextnode[idx]; continue;
                }
                if(mass_node * len_node * len_node > r2 * r2 * aold) {
                    no = s->nextnode[idx]; continue;
                }
                double dcx = fabs(center_node[0] - pos[0]);
                double dcy = fabs(center_node[1] - pos[1]);
                double dcz = fabs(center_node[2] - pos[2]);
                if(dcx < 0.60 * len_node && dcy < 0.60 * len_node && dcz < 0.60 * len_node) {
                    no = s->nextnode[idx]; continue;
                }
#if (defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES)) && defined(SINGLE_STAR_DIRECT_GRAVITY_RADIUS)
                /* Force star-star nodes to open inside the direct-gravity radius. */
                if(ptype == 5) {
                    double r_direct = (double)SINGLE_STAR_DIRECT_GRAVITY_RADIUS / UNIT_LENGTH_IN_AU + 0.6*len_node;
                    if((s->N_SINK[idx] > 0) && (r2 < r_direct * r_direct)) {
                        no = s->nextnode[idx]; continue;
                    }
                }
#endif
            }

            /* Node accepted — load payload fields */
            h_p = msoft_node;
            mass = mass_node;
#if defined(SINK_DYNFRICTION_FROMTREE) || defined(COMPUTE_JERK_IN_GRAVTREE)
            dv[0] = (double) s->node_vs[idx][0] - vel[0];
            dv[1] = (double) s->node_vs[idx][1] - vel[1];
            dv[2] = (double) s->node_vs[idx][2] - vel[2];
#endif
#ifdef SINK_DYNFRICTION_FROMTREE
            {
                long np = s->N_part[idx];
                m_j_eff_for_df = (np > 0) ? (mass / (double)np) : 0.0;
            }
#endif
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
            gasmass = s->gasmass[idx];
#endif
#ifdef RT_USE_GRAVTREE
            /* Load node stellar luminosity unconditionally (CPU does the same — no valid_gas gate here).
             * The nvc++ compiler miscompiles if(const int) in device code, so we also dropped 'const'
             * on valid_gas_particle_for_rt.  RT accumulation is still gated in the force kernel below. */
            {
                int kf; for(kf=0; kf<N_RT_FREQ_BINS; kf++) {
                    mass_stellarlum[kf] = s->stellar_lum[idx * N_RT_FREQ_BINS + kf];
                }
#ifdef CHIMES_STELLAR_FLUXES
                for(kf=0; kf<CHIMES_LOCAL_UV_NBINS; kf++) {
                    chimes_mass_stellarlum_G0[kf] = s->chimes_stellar_lum_G0[(long)idx * CHIMES_LOCAL_UV_NBINS + kf];
                    chimes_mass_stellarlum_ion[kf] = s->chimes_stellar_lum_ion[(long)idx * CHIMES_LOCAL_UV_NBINS + kf];
                }
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
                d_stellarlum[0] = s->rt_source_lum_s[idx][0] - pos[0];
                d_stellarlum[1] = s->rt_source_lum_s[idx][1] - pos[1];
                d_stellarlum[2] = s->rt_source_lum_s[idx][2] - pos[2];
#else
                d_stellarlum = dr;
#endif
#ifdef SINK_PHOTONMOMENTUM
                /* Mirror forcetree.cc:1977-1979: node-aggregated sink angle-weighted luminosity. */
                mass_sinklumwt_forradfb = gpu_sink_fb_angleweight((double) s->sink_lum[idx], s->sink_lum_grad[idx],
                                                                 d_stellarlum[0], d_stellarlum[1], d_stellarlum[2]);
#endif
            }
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
            /* Mirror forcetree.cc:1956-1957 node-aggregated CR injection. */
            cr_injection = (double) s->cr_injection[idx];
#endif
            /* Phase 2-B: node-side sink distance + timestepping accumulators.
             * Mirrors forcetree.cc:1993-2050. Runs only when the closed node
             * has non-zero sink mass (i.e., contains at least one sink). */
#ifdef SINK_CALC_DISTANCES
            if(s->sink_mass[idx] > 0)
            {
                Vec3<double> sink_dr;
                sink_dr[0] = s->sink_pos[idx][0] - pos[0];
                sink_dr[1] = s->sink_pos[idx][1] - pos[1];
                sink_dr[2] = s->sink_pos[idx][2] - pos[2];
                double sink_r2 = sink_dr.norm_sq();
                if(sink_r2 < Min_Distance_to_Sink2) {
                    Min_Distance_to_Sink2 = sink_r2;
                    Min_xyz_to_Sink = sink_dr;
                }
#ifdef SINGLE_STAR_TIMESTEPPING
                Vec3<double> sink_dv = Vec3<double>{(double)s->sink_vel[idx][0] - vel[0],
                                                    (double)s->sink_vel[idx][1] - vel[1],
                                                    (double)s->sink_vel[idx][2] - vel[2]};
                double vSqr = sink_dv.norm_sq();
                double M_total = s->sink_mass[idx] + pmass;
                double r2soft = SinkParticle_GravityKernelRadius;
                if(r2soft < soft) {r2soft = soft;}
                r2soft *= KERNEL_FAC_FROM_FORCESOFT_TO_PLUMMER;
                r2soft = r2 + r2soft * r2soft;
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
                if(ptype == 0) {
                    double tSqr_fb = r2soft / (s->MaxFeedbackVel[idx] * s->MaxFeedbackVel[idx] + MIN_REAL_NUMBER);
                    if(tSqr_fb < Min_Sink_FeedbackTime) {Min_Sink_FeedbackTime = tSqr_fb;}
                }
#endif
                double tSqr = r2soft / (vSqr + MIN_REAL_NUMBER);
                double tff4 = r2soft * r2soft * r2soft / (M_total * M_total);
                if(tSqr < Min_Sink_Approach_Time) {Min_Sink_Approach_Time = tSqr;}
                if(tff4 < Min_Sink_Freefall_time) {Min_Sink_Freefall_time = tff4;}
#ifdef SINGLE_STAR_FIND_BINARIES
                if(ptype == 5 && s->N_SINK[idx] == 1) {
                    double specific_energy = 0.5 * vSqr - All.G * M_total / sqrt(r2);
                    if(specific_energy < 0) {
                        double semimajor_axis = -All.G * M_total / (2.0 * specific_energy);
                        double t_orbital = 2.0 * M_PI *
                            sqrt(semimajor_axis*semimajor_axis*semimajor_axis / (All.G * M_total));
                        if(t_orbital < Min_Sink_OrbitalTime) {
                            Min_Sink_OrbitalTime = t_orbital;
                            comp_Mass_local = s->sink_mass[idx];
                            comp_dx_local = sink_dr;
                            comp_dv_local = sink_dv;
                        }
                    }
                }
#endif /* SINGLE_STAR_FIND_BINARIES */
#endif /* SINGLE_STAR_TIMESTEPPING */
            }
#endif /* SINK_CALC_DISTANCES */
        }

        /* Force kernel — common path for accepted particles and closed nodes. */
        if((r2 > 0.0) && (mass > 0.0))
        {
            double r = sqrt(r2);
            double fac_accel;
#ifdef EVALPOTENTIAL
            double fac_pot;
#endif
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
            double fac_tidal = 0.0, fac2_tidal = 0.0; /* mirrors forcetree.cc:1489; populated in branches below */
#endif
            if((r >= h) && (r >= h_p)) {
                fac_accel = mass / (r2 * r);
#ifdef EVALPOTENTIAL
                fac_pot   = -mass / r;
#endif
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
                /* forcetree.cc:2082 Newtonian branch */
                fac_tidal = fac_accel; fac2_tidal = 3.0 * mass / (r2 * r2 * r);
#endif
            } else {
#if defined(ADAPTIVE_GRAVSOFT_SYMMETRIZE_FORCE_BY_AVERAGING)
                double h_inv  = 1.0 / h;
                double h3_inv = h_inv * h_inv * h_inv;
                double u      = r * h_inv;
                fac_accel = mass * kernel_gravity(u, h_inv, h3_inv,  1);
#ifdef EVALPOTENTIAL
                fac_pot   = mass * kernel_gravity(u, h_inv, h3_inv, -1);
#endif
                if(h_p > 0) {
                    int symmetrize_by_averaging = 0;
#if defined(ADAPTIVE_GRAVSOFT_FORALL)
                    if(ptype_sec >= 0 && ((1 << ptype_sec) & ags_bitflag_primary)) {
                        symmetrize_by_averaging = 1;
                    }
#endif
#ifdef SINGLE_STAR_SINK_DYNAMICS
                    /* sinks: only gas-gas keeps averaging (forcetree.cc:2093-2095) */
                    if((ptype != 0) || (ptype_sec != 0)) {symmetrize_by_averaging = 0;}
#endif
                    double prefac_corr_p    = 1.0;
                    double prefac_corr_orig = 1.0;
                    if(symmetrize_by_averaging == 0) {prefac_corr_p = 2.0; prefac_corr_orig = 0.0;}
                    if((symmetrize_by_averaging == 1) || (h_p > h)) {
                        double h_p_inv  = 1.0 / h_p;
                        double h_p3_inv = h_p_inv * h_p_inv * h_p_inv;
                        double u_p      = r * h_p_inv;
                        double fac_p    = mass * kernel_gravity(u_p, h_p_inv, h_p3_inv, 1);
                        fac_accel = 0.5 * (prefac_corr_orig * fac_accel + prefac_corr_p * fac_p);
#ifdef EVALPOTENTIAL
                        double fac_pot_p = mass * kernel_gravity(u_p, h_p_inv, h_p3_inv, -1);
                        fac_pot          = 0.5 * (prefac_corr_orig * fac_pot   + prefac_corr_p * fac_pot_p);
#endif
                    }
                }
#else  /* MAX-symmetrize (non-AVERAGING) */
                double h_grav = h;
                if(h_p > h_grav) {h_grav = h_p;}
                double h_inv  = 1.0 / h_grav;
                double h3_inv = h_inv * h_inv * h_inv;
                double u      = r * h_inv;
                fac_accel = mass * kernel_gravity(u, h_inv, h3_inv,  1);
#ifdef EVALPOTENTIAL
                fac_pot   = mass * kernel_gravity(u, h_inv, h3_inv, -1);
#endif
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
                /* forcetree.cc:2097 softened branch */
                fac_tidal = fac_accel; fac2_tidal = mass * kernel_gravity(u, h_inv, h3_inv, 2);
#endif
#endif  /* SYMMETRIZE_FORCE_BY_AVERAGING */
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL)
                double fac_corr = 0.0;
                int add_primary = 1, add_secondary = 1;
                double u_p = (h_p > 0) ? (r / h_p) : 0.0;
#if defined(ADAPTIVE_GRAVSOFT_SYMMETRIZE_FORCE_BY_AVERAGING)
                double h_inv_for_zeta = 1.0 / h, h3_inv_for_zeta = h_inv_for_zeta*h_inv_for_zeta*h_inv_for_zeta;
                double u_for_zeta = r * h_inv_for_zeta;
#else
                double h_inv_for_zeta = 1.0 / h_grav, h3_inv_for_zeta = h_inv_for_zeta*h_inv_for_zeta*h_inv_for_zeta;
                double u_for_zeta = u;
#endif
                if(r <= 0.0 || pmass <= 0.0 || mass <= 0.0 || ptype_sec < 0) {
                    add_primary = 0; add_secondary = 0;
                }
                if(zeta == 0.0 || u_for_zeta >= 1.0 || h <= 0.0) {add_primary = 0;}
                if(zeta_sec == 0.0 || u_p >= 1.0 || h_p <= 0.0) {add_secondary = 0;}
                if(ptype != 0 || ptype_sec != 0) {
#if defined(ADAPTIVE_GRAVSOFT_FORALL)
                    int bm_sec = gpu_ags_kernel_shared_BITFLAG(ptype_sec);
                    if(!((1 << ptype)     & (ADAPTIVE_GRAVSOFT_FORALL)) ||
                       !((1 << ptype_sec) & ags_bitflag_primary)) {add_primary = 0;}
                    if(!((1 << ptype_sec) & (ADAPTIVE_GRAVSOFT_FORALL)) ||
                       !((1 << ptype)     & bm_sec)) {add_secondary = 0;}
#else
                    add_primary = 0; add_secondary = 0;
#endif
                }
                if(add_primary) {
                    double dWdr, wp;
                    kernel_main(u_for_zeta, h3_inv_for_zeta, h3_inv_for_zeta * h_inv_for_zeta, &wp, &dWdr, 1);
                    fac_corr += -(zeta / pmass) * dWdr / r;
                }
                if(add_secondary) {
                    double dWdr, wp;
                    double h_p_inv  = 1.0 / h_p;
                    double h_p3_inv = h_p_inv * h_p_inv * h_p_inv;
                    kernel_main(u_p, h_p3_inv, h_p3_inv * h_p_inv, &wp, &dWdr, 1);
                    fac_corr += -(zeta_sec / pmass) * dWdr / r;
                }
                if(!isnan(fac_corr)) {fac_accel += fac_corr;}
#endif
            }
#ifdef PMGRID
            int tabindex = (int) (asmthfac * r);
            if(tabindex >= 0 && tabindex < GIZMO_GPU_GRAVTREE_NTAB) {
                fac_accel *= shortrange_tab[tabindex];
#ifdef EVALPOTENTIAL
                fac_pot   *= shortrange_pot_tab[tabindex];
#endif
            } else {
                fac_accel = 0.0;
#ifdef EVALPOTENTIAL
                fac_pot   = 0.0;
#endif
            }
#endif
            acc[0] += fac_accel * dr[0];
            acc[1] += fac_accel * dr[1];
            acc[2] += fac_accel * dr[2];
#ifdef EVALPOTENTIAL
            pot    += fac_pot;
#endif
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
            /* forcetree.cc:2276-2281 non-PMGRID tidal tensor accumulation. */
            tidal_acc[0][0] += (-fac_tidal + dr[0] * dr[0] * fac2_tidal);
            tidal_acc[0][1] += ( dr[0] * dr[1] * fac2_tidal);
            tidal_acc[0][2] += ( dr[0] * dr[2] * fac2_tidal);
            tidal_acc[1][1] += (-fac_tidal + dr[1] * dr[1] * fac2_tidal);
            tidal_acc[1][2] += ( dr[1] * dr[2] * fac2_tidal);
            tidal_acc[2][2] += (-fac_tidal + dr[2] * dr[2] * fac2_tidal);
#endif
#ifdef COMPUTE_JERK_IN_GRAVTREE
            /* forcetree.cc:2289-2290.  Note: under ATFU the CPU skips the
             * `if(ptype>0)` gate to include gas in jerk accumulation. */
            {
                double dv_dot_dr = dv[0]*dr[0] + dv[1]*dr[1] + dv[2]*dr[2];
#ifndef ADAPTIVE_TREEFORCE_UPDATE
                if(ptype > 0)
#endif
                {
                    jerk_acc[0] += fac_accel * dv[0] - dv_dot_dr * fac2_tidal * dr[0];
                    jerk_acc[1] += fac_accel * dv[1] - dv_dot_dr * fac2_tidal * dr[1];
                    jerk_acc[2] += fac_accel * dv[2] - dv_dot_dr * fac2_tidal * dr[2];
                }
            }
#endif
#ifdef SINK_DYNFRICTION_FROMTREE
            /* Mirror forcetree.cc:2167-2190. Dynamical-friction correction
             * applied only to type-5 sinks acting on gravity sources. */
            if((fac_accel > MIN_REAL_NUMBER) && (ptype == 5) && (mass > MIN_REAL_NUMBER)) {
                double dv2 = dv.norm_sq();
                if((dv2 > MIN_REAL_NUMBER) && (target_sink_mass > MIN_REAL_NUMBER)) {
                    double dv0 = sqrt(dv2);
                    Vec3<double> dv_h = dv / dv0;
                    double rdotvhat = dot(dr, dv_h);
                    Vec3<double> b_im = dr - rdotvhat * dv_h;
                    double b_impact = b_im.norm();
                    double a_im = (b_impact * All.cf_atime) * (dv2 * All.cf_a2inv) / (All.G * target_sink_mass);
                    double fac_df = fac_accel * b_impact * a_im / (1.0 + a_im * a_im);
                    {
                        double m_j = m_j_eff_for_df;
                        if((m_j > 0) && (target_sink_mass > 14.251 * m_j)) {
                            double corr = (-1.0 + 3.0 / log10(target_sink_mass / m_j)) / 1.6;
                            if(corr < 0) corr = 0;
                            if(corr > 1) corr = 1;
                            fac_df *= corr;
                        }
                    }
                    if((m_j_eff_for_df <= MIN_REAL_NUMBER) || (b_impact <= MIN_REAL_NUMBER) || (dv2 <= MIN_REAL_NUMBER)) {
                        fac_df = 0;
                    }
                    /* parallel deflection component */
                    acc[0] += fac_df * dv_h[0];
                    acc[1] += fac_df * dv_h[1];
                    acc[2] += fac_df * dv_h[2];
                    /* perpendicular deflection component (residual after subtracting homogeneous) */
                    double fac_df_p = -fac_df / (b_impact * a_im + MIN_REAL_NUMBER);
                    if(fabs(fac_df_p) < MAX_REAL_NUMBER && isfinite(fac_df_p)) {
                        acc[0] += fac_df_p * b_im[0];
                        acc[1] += fac_df_p * b_im[1];
                        acc[2] += fac_df_p * b_im[2];
                    }
                }
            }
#endif /* SINK_DYNFRICTION_FROMTREE */
            ninter++;

            /* ------------------------------------------------------------ *
             * RT cluster payloads (Phase 2-A).  Structure mirrors           *
             * forcetree.cc:2290-2392 (outside the PMGRID tabindex gate,     *
             * so fac_accel is already 0 when tabindex >= NTAB — correct for  *
             * treecol; RT_USE_GRAVTREE computes its own fac_rt from          *
             * d_stellarlum independently).                                   *
             * ------------------------------------------------------------ */
#ifdef RT_USE_TREECOL_FOR_NH
            if(gasmass > 0.0)
            {
                int bin;
                if((fabs(dr[0]) > fabs(dr[1])) && (fabs(dr[0]) > fabs(dr[2]))) {
                    bin = (dr[0] > 0) ? 0 : 1;
                } else if(fabs(dr[1]) > fabs(dr[2])) {
                    bin = (dr[1] > 0) ? 2 : 3;
                } else {
                    bin = (dr[2] > 0) ? 4 : 5;
                }
                treecol_angular_bins[bin] += fac_accel * gasmass * r / (angular_bin_size * mass);
            }
#endif

#ifdef COSMIC_RAY_SUBGRID_LEBRON
            /* Mirror forcetree.cc:2300-2311. CR sub-grid LEBRON energy density
             * accumulation at the post-opening stage. All.Time>All.TimeBegin
             * and t_max_cr>0 are host-side checks; cr_data->t_max_cr is 0 at
             * t=TimeBegin which makes r_max=0 and the exp() factor zero, so
             * the block is a no-op on the first step even without the gate. */
            if(ptype == 0 && r > 0 && cr_injection > 0 && cr_data->t_max_cr > 0)
            {
                double kappa_0 = All.CosmicRay_Subgrid_Kappa_0;
                double vst_0   = All.CosmicRay_Subgrid_Vstream_0;
                double r_phys  = sqrt(r*r + soft*soft/4.0) * All.cf_atime;
                double t_max   = cr_data->t_max_cr;
                double r_max   = 0.5 * t_max * vst_0 * (1.0 + sqrt(1.0 + 16.0*kappa_0/(vst_0*vst_0*t_max)));
#ifdef PMGRID
                double r_max_pm = 0.5 * rcut * All.cf_atime;
                if(r_max_pm < r_max) {r_max = r_max_pm;}
#endif
                double denom = 4.0 * M_PI * r_phys * (kappa_0 + vst_0*r_phys);
                double expo  = r_phys*r_phys / (1.e-6*r_phys*r_phys + r_max*r_max);
                if(expo > 50.0) {expo = 50.0;}
                double fac_cr_distance = exp(-expo) / denom;
                if(fac_cr_distance > 0) {
                    SubGrid_CosmicRayEnergyDensity += fac_cr_distance * cr_injection / All.cf_a3inv;
                }
            }
#endif

#ifdef RT_USE_GRAVTREE
            if(valid_gas_particle_for_rt)
            {
                /* Compute fac_rt from d_stellarlum (may differ from dr when
                 * RT_SEPARATELY_TRACK_LUMPOS; otherwise d_stellarlum == dr). */
                double r2_rt = d_stellarlum.norm_sq(), r_rt = sqrt(r2_rt);
                double fac_rt;
                if(r_rt >= soft) {
                    fac_rt = 1.0 / (r2_rt * r_rt);
                } else {
                    double h_inv_rt = 1.0/soft, h3_inv_rt = h_inv_rt*h_inv_rt*h_inv_rt;
                    double u_rt = r_rt * h_inv_rt;
                    fac_rt = kernel_gravity(u_rt, h_inv_rt, h3_inv_rt, 1);
                }
                if((soft > r_rt) && (soft > 0)) {fac_rt *= (r2_rt / (soft * soft));}
                double fac_intensity = fac_rt * r_rt * All.cf_a2inv / (4.0 * M_PI);

#if defined(RT_USE_GRAVTREE_SAVE_RAD_ENERGY)
                {int kf; for(kf=0; kf<N_RT_FREQ_BINS; kf++) {Rad_E_gamma[kf] += fac_intensity * mass_stellarlum[kf];}}
#ifdef SINK_PHOTONMOMENTUM
                Rad_E_gamma[RT_FREQ_BIN_FIRE_IR] += fac_intensity * mass_sinklumwt_forradfb;
#endif
#endif
#ifdef SINK_COMPTON_HEATING
                incident_flux_agn += fac_intensity * mass_sinklumwt_forradfb; /* L/(4pi r^2) analog */
#endif

#ifdef CHIMES_STELLAR_FLUXES
                {
                    double chimes_fac = fac_intensity / (UNIT_LENGTH_IN_CGS * UNIT_LENGTH_IN_CGS);
                    int ck; for(ck=0; ck<CHIMES_LOCAL_UV_NBINS; ck++) {
                        chimes_flux_G0[ck]  += chimes_fac * chimes_mass_stellarlum_G0[ck];
                        chimes_flux_ion[ck] += chimes_fac * chimes_mass_stellarlum_ion[ck];
                    }
                }
#endif

#ifdef GALSF_FB_FIRE_RT_LONGRANGE
                incident_flux_uv += fac_intensity * mass_stellarlum[RT_FREQ_BIN_FIRE_UV];
                if((mass_stellarlum[RT_FREQ_BIN_FIRE_IR] < mass_stellarlum[RT_FREQ_BIN_FIRE_UV]) &&
                   (mass_stellarlum[RT_FREQ_BIN_FIRE_IR] > 0))
                {
                    incident_flux_euv += fac_intensity * mass_stellarlum[RT_FREQ_BIN_FIRE_UV] *
                        (All.PhotonMomentum_fUV + (1 - All.PhotonMomentum_fUV) *
                         ((mass_stellarlum[RT_FREQ_BIN_FIRE_UV] + mass_stellarlum[RT_FREQ_BIN_FIRE_IR]) /
                          (mass_stellarlum[RT_FREQ_BIN_FIRE_UV] + 2042.6 * mass_stellarlum[RT_FREQ_BIN_FIRE_IR])));
                } else {
                    double m_lum_total = 0;
                    int ks_q; for(ks_q=0; ks_q<N_RT_FREQ_BINS; ks_q++) {m_lum_total += mass_stellarlum[ks_q];}
                    incident_flux_euv += All.PhotonMomentum_fUV * fac_intensity * m_lum_total;
                }
#endif

#ifdef RT_OTVET
                if(r_rt > 0)
                {
                    int kf_rt; for(kf_rt=0; kf_rt<N_RT_FREQ_BINS; kf_rt++) {
                        double fac_otvet = mass_stellarlum[kf_rt] * fac_rt / (1.0e-37 + r_rt);
                        RT_ET[kf_rt] += fac_otvet * outer_product(d_stellarlum);
                    }
                }
#endif

#ifdef RT_LEBRON
#ifdef GALSF_FB_FIRE_RT_LONGRANGE
                if(r_rt * UNIT_LENGTH_IN_KPC * All.cf_atime > 50.0) {fac_rt = 0.0;}
#endif
                {
                    int kf_rt; double lum_force_fac = 0.0;
#if defined(RT_USE_GRAVTREE_SAVE_RAD_FLUX)
                    double fac_flux = -fac_rt * All.cf_a2inv / (4.0 * M_PI);
                    for(kf_rt=0; kf_rt<N_RT_FREQ_BINS; kf_rt++) {Rad_Flux[kf_rt] += mass_stellarlum[kf_rt] * fac_flux * d_stellarlum;}
#ifdef SINK_PHOTONMOMENTUM
                    Rad_Flux[RT_FREQ_BIN_FIRE_IR] += mass_sinklumwt_forradfb * fac_flux * d_stellarlum;
#endif
#else
                    for(kf_rt=0; kf_rt<N_RT_FREQ_BINS; kf_rt++) {lum_force_fac += mass_stellarlum[kf_rt] * fac_stellum[kf_rt];}
#if defined(SINK_PHOTONMOMENTUM) && !defined(RT_DISABLE_RAD_PRESSURE)
                    lum_force_fac += (All.Sink_Rad_MomentumFactor / (MIN_REAL_NUMBER + All.PhotonMomentum_Coupled_Fraction))
                                     * mass_sinklumwt_forradfb * fac_stellum[N_RT_FREQ_BINS-1];
#endif
#endif
                    if(lum_force_fac > 0) {acc += (fac_rt * lum_force_fac) * d_stellarlum;}
                }
#endif /* RT_LEBRON */

            } /* if(valid_gas_particle_for_rt) */
#endif /* RT_USE_GRAVTREE */

        } /* if((r2>0)&&(mass>0)) */

        if(no < maxPart) {
            no = s->nextnode_aux[no];
        } else {
            no = s->sibling[no - maxPart];
        }
    } /* while(no >= 0) */

    /* ------------------------------------------------------------------ *
     * Post-walk: write RT outputs to CellP_dev / P_dev.  The host scatter  *
     * loop in gpu_gravtree_walk_primary copies these to CellP[] / P[].    *
     * ------------------------------------------------------------------ */
#ifdef RT_USE_TREECOL_FOR_NH
    {int k; for(k=0; k<RT_USE_TREECOL_FOR_NH; k++) {P_dev[target].ColumnDensityBins[k] = treecol_angular_bins[k];}}
#endif
#ifdef RT_USE_GRAVTREE
    /* Use valid_gas_particle_for_rt (non-const int) throughout: nvc++ miscompiles raw boolean expressions in device code */
#ifdef RT_OTVET
    if(valid_gas_particle_for_rt) {
        int k; for(k=0; k<N_RT_FREQ_BINS; k++) {CellP_dev[target].ET[k] = RT_ET[k];}
    } else if(ptype == 0) {
        int k; for(k=0; k<N_RT_FREQ_BINS; k++) {CellP_dev[target].ET[k] = {};}
    }
#endif
#ifdef GALSF_FB_FIRE_RT_LONGRANGE
    if(valid_gas_particle_for_rt) {
        CellP_dev[target].Rad_Flux_UV  = incident_flux_uv;
        CellP_dev[target].Rad_Flux_EUV = incident_flux_euv;
    }
#endif
#ifdef CHIMES_STELLAR_FLUXES
    if(valid_gas_particle_for_rt) {
        int kc; for(kc=0; kc<CHIMES_LOCAL_UV_NBINS; kc++) {
            CellP_dev[target].Chimes_G0[kc]          = chimes_flux_G0[kc];
            CellP_dev[target].Chimes_fluxPhotIon[kc] = chimes_flux_ion[kc];
        }
    }
#endif
#if defined(RT_USE_GRAVTREE_SAVE_RAD_ENERGY)
    if(valid_gas_particle_for_rt) {
        int kf; for(kf=0; kf<N_RT_FREQ_BINS; kf++) {CellP_dev[target].Rad_E_gamma[kf] = Rad_E_gamma[kf];}
    }
#endif
#ifdef SINK_COMPTON_HEATING
    if(valid_gas_particle_for_rt) {
        CellP_dev[target].Rad_Flux_AGN = incident_flux_agn;
    }
#endif
#if defined(RT_USE_GRAVTREE_SAVE_RAD_FLUX)
    if(valid_gas_particle_for_rt) {
        int kf; for(kf=0; kf<N_RT_FREQ_BINS; kf++) {CellP_dev[target].Rad_Flux[kf] = Rad_Flux[kf];}
    }
#endif
#endif /* RT_USE_GRAVTREE */

#ifdef COSMIC_RAY_SUBGRID_LEBRON
    /* Mirror forcetree.cc:2478-2480. */
    if(ptype == 0) {CellP_dev[target].SubGrid_CosmicRayEnergyDensity = SubGrid_CosmicRayEnergyDensity;}
#endif

    /* Phase 2-B: sink-distance / single-star timestepping outputs.
     * Mirrors forcetree.cc:2499-2526 (mode=0). Scatter from P_dev back to P[]
     * happens in the host post-walk loop (primary driver). */
#ifdef SINK_CALC_DISTANCES
    P_dev[target].Min_Distance_to_Sink = sqrt(Min_Distance_to_Sink2);
    P_dev[target].Min_xyz_to_Sink = Min_xyz_to_Sink;
#ifdef SINGLE_STAR_FIND_BINARIES
    P_dev[target].is_in_a_binary = 0;
    P_dev[target].Min_Sink_OrbitalTime = Min_Sink_OrbitalTime;
    if(Min_Sink_OrbitalTime < MAX_REAL_NUMBER) {
        P_dev[target].is_in_a_binary = 1;
        P_dev[target].comp_Mass = comp_Mass_local;
        P_dev[target].comp_dx = comp_dx_local;
        P_dev[target].comp_dv = comp_dv_local;
    }
#endif
#ifdef SINGLE_STAR_TIMESTEPPING
    P_dev[target].Min_Sink_Approach_Time = sqrt(Min_Sink_Approach_Time);
    P_dev[target].Min_Sink_Freefall_time = sqrt(sqrt(Min_Sink_Freefall_time) / All.G);
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
    P_dev[target].Min_Sink_FeedbackTime = sqrt(Min_Sink_FeedbackTime);
#endif
#endif
#endif /* SINK_CALC_DISTANCES */

#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
    P_dev[target].tidal_tensorps = tidal_acc;
#endif
#ifdef COMPUTE_JERK_IN_GRAVTREE
    P_dev[target].GravJerk = jerk_acc;
#endif

    acc_out = acc;
    ninter_out = ninter;
    pot_out = pot;
    return 1;
}


extern "C" int gpu_gravtree_walk_primary(void)
{
    GIZMO_GPU_ENSURE_ALL_FRESH(gravtree);
    if(TakeLevel >= 0) {return 0;}
    if(Ewald_iter > 0) {return 0;}

    int num_active_total = (int) ActiveParticleList.size();
    if(num_active_total <= 0) {return 0;}

    /* The CPU walk (forcetree.cc) JIT-drifts particles and nodes whose
     * Ti_current is stale, at the point of encounter in the walk. Without
     * this, inactive particles (outside the currently active timebin) hold
     * stale positions since GIZMO only drifts active bins per sync-point
     * (run.cc:629) and only rebuilds the tree occasionally. The GPU walk
     * cannot call these host-only helpers from inside the Kokkos kernel,
     * so we apply the drift once up-front here: drift all particles, then
     * drift all nodes whose Ti_current lags All.Ti_Current. After this the
     * GPU SoA mirror is invalidated so the walk reads fresh positions and
     * moments. Cost is O(NumPart + Numnodestree) arithmetic per call. */
    move_particles(All.Ti_Current); /* drifts all P[], invalidates arena */
    /* Phase 6.0: per-node dirty-mark instead of unconditional invalidate. Only
     * nodes whose Ti_current lagged (and thus got mutated by force_drift_node)
     * need their SoA slot re-copied. The tree-rebuild / moment-refresh paths
     * separately mark everything dirty via force_treebuild +
     * force_refresh_node_moments hooks. On ATFU substeps where only a tiny
     * subset of nodes drifted, this avoids the O(MaxNodes) copy entirely. */
    for(int no = All.MaxPart; no < All.MaxPart + Numnodestree; no++) {
        if(Nodes[no].Ti_current != All.Ti_Current) {
            force_drift_node(no, All.Ti_Current);
            gpu_gravity_tree_mark_dirty(no);
        }
    }

    int *idx_host = (int *) mymalloc("gpu_grav_idx", num_active_total * sizeof(int));
    int num_active = 0;
    for(int a = 0; a < num_active_total; a++) {
        int i = ActiveParticleList[a];
        if(ProcessedFlag[i]) {continue;}
#ifdef ADAPTIVE_TREEFORCE_UPDATE
        /* Phase 5: exclude particles whose cached GravAccel+Jerk will be
         * extrapolated in the CPU post-loop (gravtree.cc:512-520).  The
         * CPU primary walk's line 733 check also skips these, so leaving
         * ProcessedFlag unset here is correct. */
        if(!needs_new_treeforce(i)) {continue;}
#endif
        idx_host[num_active++] = i;
    }
    if(num_active <= 0) {myfree(idx_host); return 0;}

    /* Acquire Phase 1 arena (P_dev + CellP_dev in SharedSpace) */
    gpu_particles_arena_acquire(NumPart, P, CellP);
    struct particle_data    *P_dev    = gpu_particles_arena_P();
    struct gas_cell_data    *CellP_dev = gpu_particles_arena_CellP();

    int min_nodes = MaxNodes + 1;
    gpu_gravity_tree_acquire(min_nodes, Nodes_base, Extnodes_base);
    gpu_gravity_tree_set_nextnode(All.MaxPart + NTopnodes, Nextnode);
    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!P_dev || !CellP_dev || !soa) {
        printf("gpu_gravtree_walk_primary: failed to acquire arena or tree SoA\n");
        endrun(913200);
    }

    /* ------------------------------------------------------------------ *
     * Phase 2-A: pre-compute per-particle source luminosities on CPU.     *
     * rt_get_source_luminosity() is not device-callable; this loop        *
     * mirrors what the CPU walk does per leaf-particle interaction, but    *
     * amortised to once per particle before the GPU kernel launch.         *
     * ------------------------------------------------------------------  */
#ifdef RT_USE_GRAVTREE
    MyFloat *d_src_lum = NULL;
#ifdef CHIMES_STELLAR_FLUXES
    double *d_src_lum_G0 = NULL, *d_src_lum_ion = NULL;
#endif
    {
        long sz = (long)NumPart * N_RT_FREQ_BINS * sizeof(MyFloat);
        d_src_lum = (MyFloat *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sz);
        if(!d_src_lum) {printf("gpu_gravtree_walk_primary: d_src_lum alloc failed\n"); endrun(913202);}
        memset(d_src_lum, 0, sz);
#ifdef CHIMES_STELLAR_FLUXES
        long szc = (long)NumPart * CHIMES_LOCAL_UV_NBINS * sizeof(double);
        d_src_lum_G0  = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(szc);
        d_src_lum_ion = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(szc);
        if(!d_src_lum_G0 || !d_src_lum_ion) {printf("gpu_gravtree_walk_primary: CHIMES lum alloc failed\n"); endrun(913203);}
        memset(d_src_lum_G0,  0, szc);
        memset(d_src_lum_ion, 0, szc);
#endif
        for(int p = 0; p < NumPart; p++) {
            if(P[p].Mass <= 0) {continue;}
            double lum[N_RT_FREQ_BINS];
#ifdef CHIMES_STELLAR_FLUXES
            double lum_G0[CHIMES_LOCAL_UV_NBINS], lum_ion[CHIMES_LOCAL_UV_NBINS];
            int active_check = rt_get_source_luminosity_chimes(p, 1, lum, lum_G0, lum_ion, P, CellP);
#else
            int active_check = rt_get_source_luminosity(p, 1, lum, P, CellP);
#endif
            if(active_check) {
                int kf; for(kf=0; kf<N_RT_FREQ_BINS; kf++) {
                    d_src_lum[(long)p * N_RT_FREQ_BINS + kf] = (MyFloat)lum[kf];
                }
#ifdef CHIMES_STELLAR_FLUXES
                for(kf=0; kf<CHIMES_LOCAL_UV_NBINS; kf++) {
                    d_src_lum_G0[(long)p * CHIMES_LOCAL_UV_NBINS + kf]  = lum_G0[kf];
                    d_src_lum_ion[(long)p * CHIMES_LOCAL_UV_NBINS + kf] = lum_ion[kf];
                }
#endif
            }
        }
    }
    struct gpu_rt_walk_data_t rt_data_snap;
    rt_data_snap.src_lum = d_src_lum;
#ifdef CHIMES_STELLAR_FLUXES
    rt_data_snap.src_lum_G0  = d_src_lum_G0;
    rt_data_snap.src_lum_ion = d_src_lum_ion;
#endif
#endif /* RT_USE_GRAVTREE */

    /* Phase 2-C: precompute per-particle bolometric sink luminosity and
     * angle vector for leaf-level SINK_PHOTONMOMENTUM contributions.
     * sink_lum_bol() (and, for SINGLE_STAR_SINK_DYNAMICS,
     * calculate_individual_stellar_luminosity()) are not GPU-callable. */
#ifdef SINK_PHOTONMOMENTUM
    MyFloat       *d_bh_lum   = NULL;
    Vec3<MyFloat> *d_bh_angle = NULL;
    {
        long sz_lum  = (long)NumPart * sizeof(MyFloat);
        long sz_ang  = (long)NumPart * sizeof(Vec3<MyFloat>);
        d_bh_lum   = (MyFloat *)       Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sz_lum);
        d_bh_angle = (Vec3<MyFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sz_ang);
        if(!d_bh_lum || !d_bh_angle) {printf("gpu_gravtree_walk_primary: bh_lum alloc failed\n"); endrun(913210);}
        memset(d_bh_lum,   0, sz_lum);
        memset(d_bh_angle, 0, sz_ang);
        for(int p = 0; p < NumPart; p++) {
            if(P[p].Type != 5 || P[p].Mass <= 0) continue;
            if(P[p].DensityAroundParticle <= 0 || P[p].Sink_Mdot <= 0) continue;
            double bhlum = sink_lum_bol(P[p].Sink_Mdot, P[p].Sink_Mass, p);
            d_bh_lum[p] = (MyFloat) bhlum;
#if defined(SINK_FOLLOW_ACCRETED_ANGMOM)
            d_bh_angle[p] = P[p].Sink_Specific_AngMom;
#else
            d_bh_angle[p] = P[p].GradRho;
#endif
        }
    }
    struct gpu_sink_walk_data_t sink_data_snap;
    sink_data_snap.bh_lum   = d_bh_lum;
    sink_data_snap.bh_angle = d_bh_angle;
#endif /* SINK_PHOTONMOMENTUM */

    /* Phase 2-D: precompute per-particle CR source injection rate.
     * cr_get_source_injection_rate() is CPU-only. */
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    MyFloat *d_cr_inject = NULL;
    double   t_max_cr    = 0.0;
    {
        long sz = (long)NumPart * sizeof(MyFloat);
        d_cr_inject = (MyFloat *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sz);
        if(!d_cr_inject) {printf("gpu_gravtree_walk_primary: cr_inject alloc failed\n"); endrun(913211);}
        memset(d_cr_inject, 0, sz);
        if(All.Time > All.TimeBegin) {
            double t_gyr = evaluate_time_since_t_initial_in_Gyr(All.TimeBegin);
            if(t_gyr > 1.0) {t_gyr = 1.0;}
            t_max_cr = t_gyr / UNIT_TIME_IN_GYR;
        }
        for(int p = 0; p < NumPart; p++) {
            if(P[p].Type != 0 || P[p].Mass <= 0) continue;
            double rate = cr_get_source_injection_rate(p, P, CellP);
            d_cr_inject[p] = (MyFloat) rate;
        }
    }
    struct gpu_cr_walk_data_t cr_data_snap;
    cr_data_snap.cr_inject = d_cr_inject;
    cr_data_snap.t_max_cr  = t_max_cr;
#endif /* COSMIC_RAY_SUBGRID_LEBRON */

    /* Scratch arrays for per-target results */
    int *d_idx    = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(int));
    int *d_failed = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(int));
    Vec3<double> *d_acc = (Vec3<double> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(Vec3<double>));
    int *d_ninter = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(int));
    double *d_pot = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(double));
    if(!d_idx || !d_failed || !d_acc || !d_ninter || !d_pot) {
        printf("gpu_gravtree_walk_primary: kokkos_malloc failed\n");
        endrun(913201);
    }
    memcpy(d_idx, idx_host, num_active * sizeof(int));
    memset(d_failed, 0, num_active * sizeof(int));

    int maxPart = All.MaxPart;
    int maxNodes_snap = MaxNodes;
    const struct gpu_gravity_tree_soa_t soa_snap = *soa;

#ifdef PMGRID
    double rcut_snap     = All.Rcut[0];
    double rcut2_snap    = rcut_snap * rcut_snap;
    double asmthfac_snap = 0.5 / All.Asmth[0] * (GIZMO_GPU_GRAVTREE_NTAB / 3.0);
    /* shortrange_table is a host global (forcetree.cc); copy to SharedSpace so
     * the CUDA kernel can read it from device code. */
    float *d_st  = (float *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(GIZMO_GPU_GRAVTREE_NTAB * sizeof(float));
    float *d_sp  = (float *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(GIZMO_GPU_GRAVTREE_NTAB * sizeof(float));
    if(!d_st || !d_sp) {printf("gpu_gravtree_walk_primary: shortrange table alloc failed\n"); endrun(913205);}
    memcpy(d_st, shortrange_table,           GIZMO_GPU_GRAVTREE_NTAB * sizeof(float));
    memcpy(d_sp, shortrange_table_potential, GIZMO_GPU_GRAVTREE_NTAB * sizeof(float));
#endif

#ifdef RT_USE_GRAVTREE
    const struct gpu_rt_walk_data_t rt_data_dev = rt_data_snap;
#endif
#ifdef SINK_PHOTONMOMENTUM
    const struct gpu_sink_walk_data_t sink_data_dev = sink_data_snap;
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    const struct gpu_cr_walk_data_t cr_data_dev = cr_data_snap;
#endif

    Kokkos::parallel_for("gravtree_walk_primary", num_active, KOKKOS_LAMBDA(int a) {
        int target = d_idx[a];
        Vec3<double> acc;
        int ninter;
        double pot;
        int ok = gpu_gravtree_walk_one(target, maxPart, maxNodes_snap,
                                        P_dev, CellP_dev, &soa_snap,
#ifdef PMGRID
                                        rcut_snap, rcut2_snap, asmthfac_snap, d_st, d_sp,
#endif
#ifdef RT_USE_GRAVTREE
                                        &rt_data_dev,
#endif
#ifdef SINK_PHOTONMOMENTUM
                                        &sink_data_dev,
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
                                        &cr_data_dev,
#endif
                                        acc, ninter, pot);
        if(ok) {
            d_acc[a] = acc;
            d_ninter[a] = ninter;
            d_pot[a] = pot;
            d_failed[a] = 0;
        } else {
            d_failed[a] = 1;
        }
    });
    Kokkos::fence();

    /* Scatter successes back to host; copy RT CellP fields from device mirror */
    int nsucceeded = 0;
    double costtotal_added = 0;
    for(int a = 0; a < num_active; a++) {
        int i = d_idx[a];
        if(!d_failed[a]) {
            P[i].GravAccel = d_acc[a];
#ifdef EVALPOTENTIAL
            P[i].Potential = d_pot[a];
#endif

            /* RT scatter-back: copy outputs written into the SharedSpace
             * device mirrors back to host P[]/CellP[].  On UVM systems this
             * is effectively a same-pointer copy (no-op performance-wise),
             * but kept explicit for correctness on non-UVM targets. */
#ifdef RT_USE_TREECOL_FOR_NH
            {int k; for(k=0; k<RT_USE_TREECOL_FOR_NH; k++) {P[i].ColumnDensityBins[k] = P_dev[i].ColumnDensityBins[k];}}
#endif
#ifdef RT_USE_GRAVTREE
#ifdef RT_OTVET
            if(P[i].Type == 0) {
                int k; for(k=0; k<N_RT_FREQ_BINS; k++) {CellP[i].ET[k] = CellP_dev[i].ET[k];}
            }
#endif
#ifdef GALSF_FB_FIRE_RT_LONGRANGE
            if(P[i].Type == 0 && P[i].Mass > 0) {
                CellP[i].Rad_Flux_UV  = CellP_dev[i].Rad_Flux_UV;
                CellP[i].Rad_Flux_EUV = CellP_dev[i].Rad_Flux_EUV;
            }
#endif
#ifdef CHIMES_STELLAR_FLUXES
            if(P[i].Type == 0 && P[i].Mass > 0) {
                int kc; for(kc=0; kc<CHIMES_LOCAL_UV_NBINS; kc++) {
                    CellP[i].Chimes_G0[kc]          = CellP_dev[i].Chimes_G0[kc];
                    CellP[i].Chimes_fluxPhotIon[kc]  = CellP_dev[i].Chimes_fluxPhotIon[kc];
                }
            }
#endif
#if defined(RT_USE_GRAVTREE_SAVE_RAD_ENERGY)
            if(P[i].Type == 0 && P[i].Mass > 0) {
                int kf; for(kf=0; kf<N_RT_FREQ_BINS; kf++) {CellP[i].Rad_E_gamma[kf] = CellP_dev[i].Rad_E_gamma[kf];}
            }
#endif
#if defined(RT_USE_GRAVTREE_SAVE_RAD_FLUX)
            if(P[i].Type == 0 && P[i].Mass > 0) {
                int kf; for(kf=0; kf<N_RT_FREQ_BINS; kf++) {CellP[i].Rad_Flux[kf] = CellP_dev[i].Rad_Flux[kf];}
            }
#endif
#ifdef SINK_COMPTON_HEATING
            if(P[i].Type == 0 && P[i].Mass > 0) {
                CellP[i].Rad_Flux_AGN = CellP_dev[i].Rad_Flux_AGN;
            }
#endif
#endif /* RT_USE_GRAVTREE */
#ifdef COSMIC_RAY_SUBGRID_LEBRON
            if(P[i].Type == 0 && P[i].Mass > 0) {
                CellP[i].SubGrid_CosmicRayEnergyDensity = CellP_dev[i].SubGrid_CosmicRayEnergyDensity;
            }
#endif

            /* Phase 5: tidal tensor + GravJerk scatter-back (ATFU/jerk path). */
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
            P[i].tidal_tensorps = P_dev[i].tidal_tensorps;
#endif
#ifdef COMPUTE_JERK_IN_GRAVTREE
            P[i].GravJerk = P_dev[i].GravJerk;
#endif

            /* Phase 2-B: sink-distance / single-star timestepping scatter-back */
#ifdef SINK_CALC_DISTANCES
            P[i].Min_Distance_to_Sink = P_dev[i].Min_Distance_to_Sink;
            P[i].Min_xyz_to_Sink      = P_dev[i].Min_xyz_to_Sink;
#ifdef SINGLE_STAR_FIND_BINARIES
            P[i].is_in_a_binary       = P_dev[i].is_in_a_binary;
            P[i].Min_Sink_OrbitalTime = P_dev[i].Min_Sink_OrbitalTime;
            if(P[i].is_in_a_binary) {
                P[i].comp_Mass = P_dev[i].comp_Mass;
                P[i].comp_dx   = P_dev[i].comp_dx;
                P[i].comp_dv   = P_dev[i].comp_dv;
            }
#endif
#ifdef SINGLE_STAR_TIMESTEPPING
            P[i].Min_Sink_Approach_Time = P_dev[i].Min_Sink_Approach_Time;
            P[i].Min_Sink_Freefall_time = P_dev[i].Min_Sink_Freefall_time;
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
            P[i].Min_Sink_FeedbackTime  = P_dev[i].Min_Sink_FeedbackTime;
#endif
#endif
#endif /* SINK_CALC_DISTANCES */

            ProcessedFlag[i] = 1;
            costtotal_added += d_ninter[a];
            nsucceeded++;
        }
    }
    Costtotal += costtotal_added;

    gpu_particles_arena_invalidate();
    gpu_gravity_tree_invalidate();

    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_pot);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_ninter);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_acc);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_failed);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_idx);

#ifdef RT_USE_GRAVTREE
#ifdef CHIMES_STELLAR_FLUXES
    if(d_src_lum_ion) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_src_lum_ion);}
    if(d_src_lum_G0)  {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_src_lum_G0);}
#endif
    if(d_src_lum) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_src_lum);}
#endif
#ifdef SINK_PHOTONMOMENTUM
    if(d_bh_angle) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_bh_angle);}
    if(d_bh_lum)   {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_bh_lum);}
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    if(d_cr_inject){Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_cr_inject);}
#endif
#ifdef PMGRID
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_sp);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_st);
#endif

    myfree(idx_host);

    return nsucceeded;
}

GPU_ALL_SYNC_FUNC(gravtree)


/* ========================================================================= *
 * Phase 2-E: GPU Ewald-correction walk (pure-tree periodic gravity).         *
 *                                                                            *
 * Mirrors force_treeevaluate_ewald_correction() mode=0 (forcetree.cc:2631).  *
 * Runs as a second pass over active targets when Ewald_iter==1, after the    *
 * primary walk has completed its Ewald_iter==0 pass. Adds the periodic-image *
 * correction to P[target].GravAccel via trilinear interpolation of the Ewald *
 * lookup tables (fcorrx/y/z), with the same opening criteria and            *
 * nearest-image wrapping as the CPU walk. No optional payloads (monopole     *
 * only), no softening kernel.                                                *
 * ========================================================================= */
#if defined(BOX_PERIODIC) && !defined(GRAVITY_NOT_PERIODIC)

/* SharedSpace mirrors of the four Ewald correction tables. Seeded once from
 * the CPU-side static tables in forcetree.cc via gizmo_get_ewald_tables().
 * Flat layout: index [i*(EN+1)^2 + j*(EN+1) + k] with EN = GIZMO_EWALD_EN. */
static MyFloat *g_d_fcorrx   = NULL;
static MyFloat *g_d_fcorry   = NULL;
static MyFloat *g_d_fcorrz   = NULL;
static MyFloat *g_d_potcorr  = NULL;
static double   g_ewald_fac_intp = 0.0;
static int      g_ewald_tables_ready = 0;

static void gpu_ewald_tables_acquire(void)
{
    if(g_ewald_tables_ready) return;
    const MyFloat *fx, *fy, *fz, *fp;
    double fi;
    gizmo_get_ewald_tables(&fx, &fy, &fz, &fp, &fi);
    long n  = (long)(GIZMO_EWALD_EN + 1) * (GIZMO_EWALD_EN + 1) * (GIZMO_EWALD_EN + 1);
    long sz = n * sizeof(MyFloat);
    g_d_fcorrx  = (MyFloat *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sz);
    g_d_fcorry  = (MyFloat *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sz);
    g_d_fcorrz  = (MyFloat *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sz);
    g_d_potcorr = (MyFloat *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sz);
    if(!g_d_fcorrx || !g_d_fcorry || !g_d_fcorrz || !g_d_potcorr) {
        printf("gpu_ewald_tables_acquire: kokkos_malloc failed\n");
        endrun(914101);
    }
    memcpy(g_d_fcorrx,  fx, sz);
    memcpy(g_d_fcorry,  fy, sz);
    memcpy(g_d_fcorrz,  fz, sz);
    memcpy(g_d_potcorr, fp, sz);
    g_ewald_fac_intp = fi;
    g_ewald_tables_ready = 1;
}

/* Device-side Ewald walk for a single target. Returns 1 on success (acc
 * written), 0 if a pseudo-particle was encountered (defer to CPU). */
static KOKKOS_INLINE_FUNCTION int
gpu_ewald_walk_one(int target,
                   int maxPart, int maxNodes,
                   struct particle_data *P_dev,
                   const struct gpu_gravity_tree_soa_t *s,
                   const MyFloat *fcorrx, const MyFloat *fcorry, const MyFloat *fcorrz,
                   double fac_intp, double boxsize, double boxhalf,
                   double errtoltheta, double errtolforceacc,
                   Vec3<double> &acc_out)
{
    Vec3<double> pos = P_dev[target].Pos;
    double aold = errtolforceacc * P_dev[target].OldAcc;
    Vec3<double> acc = {0.0, 0.0, 0.0};
    const int EN       = GIZMO_EWALD_EN;
    const int stride1  = EN + 1;
    const int stride2  = stride1 * stride1;

    int no = maxPart; /* root node */
    while(no >= 0)
    {
        double mass = 0.0;
        Vec3<double> dr = {0.0, 0.0, 0.0};
        int is_leaf = 0;
        int idx = 0;
        if(no < maxPart) /* particle leaf */
        {
            dr[0] = P_dev[no].Pos[0] - pos[0];
            dr[1] = P_dev[no].Pos[1] - pos[1];
            dr[2] = P_dev[no].Pos[2] - pos[2];
            mass  = P_dev[no].Mass;
            is_leaf = 1;
        }
        else if(no >= maxPart + maxNodes) /* pseudo-particle — defer to CPU */
        {
            return 0;
        }
        else /* internal node */
        {
            idx = no - maxPart;
            /* skip single-particle node (open it to its daughter chain) */
            if(!(s->bitflags[idx] & (1 << BITFLAG_MULTIPLEPARTICLES))) {
                no = s->nextnode[idx];
                continue;
            }
            mass  = s->mass[idx];
            dr[0] = s->s[idx][0] - pos[0];
            dr[1] = s->s[idx][1] - pos[1];
            dr[2] = s->s[idx][2] - pos[2];
        }

        /* nearest-image wrap on the displacement (mirrors GRAVITY_NEAREST_XYZ) */
        if(dr[0] >  boxhalf) dr[0] -= boxsize; else if(dr[0] < -boxhalf) dr[0] += boxsize;
        if(dr[1] >  boxhalf) dr[1] -= boxsize; else if(dr[1] < -boxhalf) dr[1] += boxsize;
        if(dr[2] >  boxhalf) dr[2] -= boxsize; else if(dr[2] < -boxhalf) dr[2] += boxsize;

        if(is_leaf) {
            no = s->nextnode_aux[no];
        } else {
            /* Opening check + periodic-boundary skip (mirrors forcetree.cc:2769-2842) */
            double r2  = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
            if(r2 <= 0) r2 = 1e-300;
            double len = s->len[idx];
            int openflag = 0;
            if(errtoltheta) {
                if(len * len > r2 * errtoltheta * errtoltheta) openflag = 1;
            } else {
                if(mass * len * len > r2 * r2 * aold) {
                    openflag = 1;
                } else {
                    double cx = s->center[idx][0] - pos[0]; if(cx >  boxhalf) cx -= boxsize; else if(cx < -boxhalf) cx += boxsize;
                    double cy = s->center[idx][1] - pos[1]; if(cy >  boxhalf) cy -= boxsize; else if(cy < -boxhalf) cy += boxsize;
                    double cz = s->center[idx][2] - pos[2]; if(cz >  boxhalf) cz -= boxsize; else if(cz < -boxhalf) cz += boxsize;
                    double adx = (cx < 0) ? -cx : cx;
                    double ady = (cy < 0) ? -cy : cy;
                    double adz = (cz < 0) ? -cz : cz;
                    if(adx < 0.60*len && ady < 0.60*len && adz < 0.60*len) openflag = 1;
                }
            }
            if(openflag) {
                /* short-cut: if the node is entirely on one side of the periodic
                 * boundary along any axis, we can safely skip it without opening */
                double ux = s->center[idx][0] - pos[0]; if(ux >  boxhalf) ux -= boxsize; else if(ux < -boxhalf) ux += boxsize;
                if(((ux < 0) ? -ux : ux) > 0.5*(boxsize - len)) { no = s->nextnode[idx]; continue; }
                double uy = s->center[idx][1] - pos[1]; if(uy >  boxhalf) uy -= boxsize; else if(uy < -boxhalf) uy += boxsize;
                if(((uy < 0) ? -uy : uy) > 0.5*(boxsize - len)) { no = s->nextnode[idx]; continue; }
                double uz = s->center[idx][2] - pos[2]; if(uz >  boxhalf) uz -= boxsize; else if(uz < -boxhalf) uz += boxsize;
                if(((uz < 0) ? -uz : uz) > 0.5*(boxsize - len)) { no = s->nextnode[idx]; continue; }
                /* cell too large → must refine */
                if(len > 0.20 * boxsize) { no = s->nextnode[idx]; continue; }
            }
            no = s->sibling[idx];
        }

        /* Trilinear interpolation of the Ewald correction table. */
        double signx, signy, signz, adrx, adry, adrz;
        if(dr[0] < 0) { adrx = -dr[0]; signx = +1.0; } else { adrx = dr[0]; signx = -1.0; }
        if(dr[1] < 0) { adry = -dr[1]; signy = +1.0; } else { adry = dr[1]; signy = -1.0; }
        if(dr[2] < 0) { adrz = -dr[2]; signz = +1.0; } else { adrz = dr[2]; signz = -1.0; }
        double u = adrx * fac_intp; int i = (int) u; if(i >= EN) i = EN - 1; u -= i;
        double v = adry * fac_intp; int j = (int) v; if(j >= EN) j = EN - 1; v -= j;
        double w = adrz * fac_intp; int k = (int) w; if(k >= EN) k = EN - 1; w -= k;
        double f1 = (1-u)*(1-v)*(1-w);
        double f2 = (1-u)*(1-v)*(w);
        double f3 = (1-u)*(v)  *(1-w);
        double f4 = (1-u)*(v)  *(w);
        double f5 = (u)  *(1-v)*(1-w);
        double f6 = (u)  *(1-v)*(w);
        double f7 = (u)  *(v)  *(1-w);
        double f8 = (u)  *(v)  *(w);
#define GIZMO_EW_IDX3(ii,jj,kk) ((ii)*stride2 + (jj)*stride1 + (kk))
        acc[0] += mass * signx * (fcorrx[GIZMO_EW_IDX3(i,  j,  k  )] * f1 +
                                  fcorrx[GIZMO_EW_IDX3(i,  j,  k+1)] * f2 +
                                  fcorrx[GIZMO_EW_IDX3(i,  j+1,k  )] * f3 +
                                  fcorrx[GIZMO_EW_IDX3(i,  j+1,k+1)] * f4 +
                                  fcorrx[GIZMO_EW_IDX3(i+1,j,  k  )] * f5 +
                                  fcorrx[GIZMO_EW_IDX3(i+1,j,  k+1)] * f6 +
                                  fcorrx[GIZMO_EW_IDX3(i+1,j+1,k  )] * f7 +
                                  fcorrx[GIZMO_EW_IDX3(i+1,j+1,k+1)] * f8);
        acc[1] += mass * signy * (fcorry[GIZMO_EW_IDX3(i,  j,  k  )] * f1 +
                                  fcorry[GIZMO_EW_IDX3(i,  j,  k+1)] * f2 +
                                  fcorry[GIZMO_EW_IDX3(i,  j+1,k  )] * f3 +
                                  fcorry[GIZMO_EW_IDX3(i,  j+1,k+1)] * f4 +
                                  fcorry[GIZMO_EW_IDX3(i+1,j,  k  )] * f5 +
                                  fcorry[GIZMO_EW_IDX3(i+1,j,  k+1)] * f6 +
                                  fcorry[GIZMO_EW_IDX3(i+1,j+1,k  )] * f7 +
                                  fcorry[GIZMO_EW_IDX3(i+1,j+1,k+1)] * f8);
        acc[2] += mass * signz * (fcorrz[GIZMO_EW_IDX3(i,  j,  k  )] * f1 +
                                  fcorrz[GIZMO_EW_IDX3(i,  j,  k+1)] * f2 +
                                  fcorrz[GIZMO_EW_IDX3(i,  j+1,k  )] * f3 +
                                  fcorrz[GIZMO_EW_IDX3(i,  j+1,k+1)] * f4 +
                                  fcorrz[GIZMO_EW_IDX3(i+1,j,  k  )] * f5 +
                                  fcorrz[GIZMO_EW_IDX3(i+1,j,  k+1)] * f6 +
                                  fcorrz[GIZMO_EW_IDX3(i+1,j+1,k  )] * f7 +
                                  fcorrz[GIZMO_EW_IDX3(i+1,j+1,k+1)] * f8);
#undef GIZMO_EW_IDX3
    }

    acc_out = acc;
    return 1;
}

extern "C" int gpu_ewald_walk_primary(void)
{
    GIZMO_GPU_ENSURE_ALL_FRESH(gravtree);
    if(TakeLevel >= 0) {return 0;}
    if(Ewald_iter == 0) {return 0;}
#ifdef PMGRID
    return 0; /* Ewald walk not needed under TreePM (gravtree.cc:734 gate) */
#else
    int num_active_total = (int) ActiveParticleList.size();
    if(num_active_total <= 0) {return 0;}

    /* Particles have already been drifted by the primary walk (Ewald_iter==0);
     * the tree SoA mirror is still valid. Just re-acquire (cache-hit). */
    int min_nodes = MaxNodes + 1;
    gpu_gravity_tree_acquire(min_nodes, Nodes_base, Extnodes_base);
    gpu_gravity_tree_set_nextnode(All.MaxPart + NTopnodes, Nextnode);
    const struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa) {return 0;}

    gpu_particles_arena_acquire(NumPart, P, CellP);
    struct particle_data *P_dev = gpu_particles_arena_P();
    if(!P_dev) {return 0;}

    gpu_ewald_tables_acquire();

    int *idx_host = (int *) mymalloc("gpu_ewald_idx", num_active_total * sizeof(int));
    int num_active = 0;
    for(int a = 0; a < num_active_total; a++) {
        int i = ActiveParticleList[a];
        if(P[i].Mass <= 0) continue;
        if(ProcessedFlag[i]) continue;
#ifdef ADAPTIVE_TREEFORCE_UPDATE
        if(!needs_new_treeforce(i)) continue;
#endif
        idx_host[num_active++] = i;
    }
    if(num_active == 0) { myfree(idx_host); return 0; }

    int *d_idx    = (int *)          Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(int));
    int *d_failed = (int *)          Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(int));
    Vec3<double> *d_acc = (Vec3<double> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(Vec3<double>));
    if(!d_idx || !d_failed || !d_acc) {printf("gpu_ewald_walk_primary: kokkos_malloc failed\n"); endrun(914102);}
    memcpy(d_idx, idx_host, num_active * sizeof(int));
    memset(d_failed, 0, num_active * sizeof(int));

    /* snapshot scalars */
    const int maxPart        = All.MaxPart;
    const int maxNodes_snap  = MaxNodes;
    const double boxsize     = All.BoxSize;
    const double boxhalf     = 0.5 * All.BoxSize;
    const double fac_intp    = g_ewald_fac_intp;
    const double errtoltheta = All.ErrTolTheta;
    const double errtolforceacc = All.ErrTolForceAcc;
    const MyFloat *fcorrx_dev = g_d_fcorrx;
    const MyFloat *fcorry_dev = g_d_fcorry;
    const MyFloat *fcorrz_dev = g_d_fcorrz;
    const struct gpu_gravity_tree_soa_t soa_snap = *soa;

    Kokkos::parallel_for("gpu_ewald_walk_primary", num_active, KOKKOS_LAMBDA(int a) {
        int target = d_idx[a];
        Vec3<double> acc;
        int ok = gpu_ewald_walk_one(target, maxPart, maxNodes_snap,
                                     P_dev, &soa_snap,
                                     fcorrx_dev, fcorry_dev, fcorrz_dev,
                                     fac_intp, boxsize, boxhalf,
                                     errtoltheta, errtolforceacc,
                                     acc);
        if(ok) {d_acc[a] = acc; d_failed[a] = 0;}
        else   {d_failed[a] = 1;}
    });
    Kokkos::fence();

    int nsucceeded = 0;
    for(int a = 0; a < num_active; a++) {
        if(!d_failed[a]) {
            int target = idx_host[a];
            P[target].GravAccel[0] += d_acc[a][0];
            P[target].GravAccel[1] += d_acc[a][1];
            P[target].GravAccel[2] += d_acc[a][2];
            ProcessedFlag[target] = 1;
            nsucceeded++;
        }
    }

    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_acc);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_failed);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_idx);
    myfree(idx_host);
    return nsucceeded;
#endif /* !PMGRID */
}

#else /* !(BOX_PERIODIC && !GRAVITY_NOT_PERIODIC) */

extern "C" int gpu_ewald_walk_primary(void) {return 0;}

#endif


#else /* !GIZMO_GPU_GRAVTREE || !OPENMP_GPU_OFFLOAD */

extern "C" int gpu_gravtree_walk_primary(void) {return 0;}
extern "C" int gpu_ewald_walk_primary(void)   {return 0;}

#endif
