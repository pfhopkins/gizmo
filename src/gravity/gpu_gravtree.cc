/* gpu_gravtree.cc
 *
 * GPU gravity walk (mode=0 primary-tree path).  Core walk with
 * PMGRID, ADAPTIVE_GRAVSOFT_FORALL, SYMMETRIZE, EVALPOTENTIAL, plus
 * RT cluster payloads (RT_USE_GRAVTREE, GALSF_FB_FIRE_RT_LONGRANGE,
 * CHIMES_STELLAR_FLUXES, RT_USE_TREECOL_FOR_NH).
 *
 * rt_get_source_luminosity() is not GPU-callable; it is pre-computed on CPU
 * for all local particles into a SharedSpace array (d_src_lum) before kernel
 * launch.  Node stellar luminosities come from the SoA extension.
 * rt_kappa() is KOKKOS_INLINE_FUNCTION and runs on device for RT_LEBRON
 * fac_stellum initialisation.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
/* This TU alone exposes the device-callable gravtree source-payload helpers
 * (rt_get_source_luminosity / sink_lum_bol_core / cr_get_source_injection_rate and
 * their stellar-evolution/cosmology leaves), so the source payload can be evaluated
 * on-device at each local particle-open. The compile-time capability predicate
 * GRAVTREE_SOURCE_LAZY_SUPPORTED gates body visibility; the runtime eager/lazy choice
 * is a separate active-count threshold. Must precede the source-helper header includes. */
#ifdef GRAVTREE_SOURCE_LAZY_SUPPORTED
#define GRAVTREE_SOURCE_DEVICE_TU
#endif
#include "../core/step_phases.h"
#include "../system/gpu_particles_arena.h"
#include "../declarations/gpu_error_check.h"
#include "gpu_gravity_tree.h"
#include "gpu_gravtree.h"
#include "forcetree.h"
#include "gravity_box_distance.h"   /* shared CPU/GPU gravity box-distance SSOT */
#include "gravtree_opening.h"       /* shared CPU/GPU primary-walk acceptance-geometry predicate (SSOT) */

#include "../mesh/kernel.h"
#include "gravtree_force_kernel.h"  /* shared CPU/GPU accepted-source contribution physics (SSOT) */
#include "gravtree_ewald.h"         /* shared CPU/GPU Ewald image-correction trilinear interp (SSOT) */
#include "pm_highres_region.h"      /* pmforce_is_particle_high_res SSOT (device-callable) */
/* gravtree_moment_sources.h (the SSOT source-input fill helper) is included further
 * below, AFTER the device-callable source cores, so that in this DEVICE_TU the helper
 * binds its RT/sink/CR calls to the inline device bodies rather than the proto.h host
 * decls. See the source-core include block after the walk-data struct definitions. */


/* Single gate for the Ewald periodic-image POTENTIAL correction added in the
 * primary walk (item #11): pure-tree periodic gravity with potentials requested.
 * Mirrors the CPU gate at forcetree.cc:2299.  Defined once so the four-flag
 * condition lives in one place and is referenced (not re-spelled) at the
 * table-acquire site, the host helper, and the walk body. */
#if defined(EVALPOTENTIAL) && defined(BOX_PERIODIC) && !defined(GRAVITY_NOT_PERIODIC) && !defined(PMGRID)
#define GIZMO_GPU_EWALD_POT_CORRECTION
#endif

/* Globals that live at file-scope in gravtree.cc without a header declaration. */
extern int Ewald_iter;
extern double Costtotal;

#ifdef PMGRID
/* Short-range tables live as file-scope (non-static) globals in forcetree.cc.
 * Table length owned by gravtree_force_kernel.h (shared with the CPU walk). */
#define GIZMO_GPU_GRAVTREE_NTAB GRAVTREE_SHORTRANGE_NTAB
extern float shortrange_table[GIZMO_GPU_GRAVTREE_NTAB];
extern float shortrange_table_potential[GIZMO_GPU_GRAVTREE_NTAB];
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
extern float shortrange_table_tidal[GIZMO_GPU_GRAVTREE_NTAB];
#endif
#endif

/* GPU-callable accessor for the cached force-softening kernel radius. Always available
 * (Pp[p].ForceSoftening is populated for every build by compute_all_force_softening()), so the
 * walk can load a leaf's secondary softening unconditionally -- mirrors the CPU
 * ForceSoftening_KernelRadius(). The actual computation lives in
 * compute_force_softening_kernel_radius(p) in forcetree.cc; new softening physics goes there
 * and is picked up here with no GPU-side change. Pp[p].ForceSoftening is the single source of truth. */
static KOKKOS_INLINE_FUNCTION
double gpu_force_softening_kernel_radius(const struct particle_data *Pp, int p)
{
    return Pp[p].ForceSoftening;
}

/* AGS_zeta field is gated (particle_data.h:331) on
 * ADAPTIVE_GRAVSOFT_FORGAS || AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
 * (the latter auto-defines under FORALL/CBE_INTEGRATOR/DM_FUZZY/SIDM).
 * The accessor must match the field gate — NOT include GALSF_MERGER_STARCLUSTER_PARTICLES
 * alone, which doesn't enable AGS_zeta. */
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE)
static KOKKOS_INLINE_FUNCTION
double gpu_get_ags_zeta(const struct particle_data *Pp, int p)
{
    return Pp[p].AGS_zeta;
}
#endif

/* Permanent invariant guard (always enforced).  A tree-node multipole must never stand in
 * for a particle leaf where any enabled leaf interaction (AGS softening/zeta, ...) distinguishes
 * them.  In the one-shot LET that risk is a foreign TERMINAL node the shared opening predicate wanted
 * to OPEN but cannot descend (nextnode<0): if it is a tagged real single-particle leaf it is routed
 * through particle-leaf secondary semantics below (legal); otherwise it is an unopenable aggregate
 * that would be silently downgraded to a multipole, which is ILLEGAL until an owner-continuation path
 * exists.  The host hard-surfaces (controlled stop) when g_inv_fterm_aggregate>0 after the walk. */
/* Device-written (Kokkos::atomic_add inside the walk) + host-read (report/reset/controlled-stop).
 * On a GPU compiler the storage MUST be device-addressable, so use the same `__managed__` idiom as
 * gpu_device_error_sentinel.h; on the Mac OpenMP build (no GPU compiler) a plain host static suffices
 * (the Kokkos lambda runs on host threads).  Plain `static long long` here built on Mac but failed the
 * Vista nvcc compile (#20096-D: address of a host variable in device code). */
#if defined(GIZMO_GPU_COMPILER)
static __managed__ long long g_inv_fterm_aggregate = 0;   /* predicate-OPEN foreign terminal, NOT a leaf -> illegal */
#else
static long long g_inv_fterm_aggregate = 0;
#endif

/* The device replicas of weight_function_for_weighted_motion_smoothing and
 * ags_gravity_kernel_shared_BITFLAG were collapsed into gravtree_force_kernel.h
 * (grav_weight_function_for_weighted_motion_smoothing / gravtree_ags_kernel_shared_bitflag),
 * shared verbatim with the CPU walk. */

/* RT payload data passed to the GPU walk kernel.
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

/* Sink radiation payload.  Pre-computed on CPU before the kernel
 * launches because sink_lum_bol() (and, under SINGLE_STAR_SINK_DYNAMICS,
 * calculate_individual_stellar_luminosity()) are not GPU-callable.
 * bh_lum[p]   = sink_lum_bol(P[p].Sink_Mdot, P[p].Sink_Mass, p) when P[p]
 *               is a valid type-5 sink with Mdot>0, else 0.
 * bh_angle[p] = P[p].Sink_Specific_AngMom (if SINK_FOLLOW_ACCRETED_ANGMOM)
 *               or P[p].GradRho otherwise.  Used for angle-weighted
 *               luminosity at particle leafs.  Node-level sink_lum /
 *               sink_lum_grad already live in the SoA. */
/* COSMIC_RAY_SUBGRID_LEBRON payload.  cr_get_source_injection_rate
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

/* The device replica of sink_fb_angleweight was collapsed into gravtree_force_kernel.h
 * (grav_sink_fb_angleweight, component args), shared verbatim with the host function. */
#endif /* SINK_PHOTONMOMENTUM */

/* ---- Device-callable source cores for the lazy per-open source evaluation ----
 * This TU opens GRAVTREE_SOURCE_DEVICE_TU, so gravtree_moment_sources.h's fill helper is
 * KOKKOS_INLINE here and its RT/sink/CR calls must see the inline device bodies. Pull the
 * enabled families' *_functions.h cores in FIRST (rt_functions.h is already included above
 * under RT_USE_GRAVTREE and transitively carries the stellar-evolution/cosmology + sink
 * leaves; sink/CR are added here for the non-RT source configs). Host/default TUs compile
 * the helper as static-inline against the proto.h host wrappers and skip this block. */
#if defined(GRAVTREE_SOURCE_DEVICE_TU)
#ifdef SINK_PHOTONMOMENTUM
#include "../sinks/sink_functions.h"                       /* sink_lum_bol_core */
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
#include "../eos/cosmic_ray_fluid/cosmic_ray_functions.h"  /* cr_get_source_injection_rate */
#endif
#endif
#include "gravtree_moment_sources.h" /* SSOT per-particle source-input fill helper (static-inline host / KOKKOS_INLINE device) */

/* Ewald periodic-image POTENTIAL correction for the primary walk.  Under
 * pure-tree periodic gravity with EVALPOTENTIAL the CPU walk (forcetree.cc)
 * adds mass*ewald_pot_corr(dr) to the potential of every accepted interaction;
 * the GPU primary walk previously added only the short-range potential and left
 * the periodic-image term out (the seeded g_d_potcorr table was never read).
 * This POD carries the device mirror of that table + its interpolation scale
 * into the primary walk.  It is passed UNCONDITIONALLY (one struct, optional
 * fields gated once here) rather than as a stacked-#ifdef parameter; the walk
 * reads it only inside the matching compile gate.  In a healthy run 'active' is
 * always 1: an acquire failure hard-stops the primary walk (the build requires
 * the correction).  'active' is 0 only as a NULL-guard for the graceful drain
 * that follows that endrun. */
struct gpu_ewald_pot_data_t {
    const MyFloat *potcorr;   /* flat [(EN+1)^3] Ewald potential-correction table, or NULL */
    double         fac_intp;  /* table interpolation scale (= g_ewald_fac_intp) */
    int            active;    /* 1 iff potcorr is a valid acquired table */
};

/* Host: acquire the Ewald tables (idempotent) and fill the potential POD.
 * Returns 0 on success (out->active=1), nonzero if the tables are not ready
 * (out->active=0); the caller treats a nonzero return as a hard stop. */
#ifdef GIZMO_GPU_EWALD_POT_CORRECTION
static int gpu_ewald_acquire_pot_data(struct gpu_ewald_pot_data_t *out);
#endif

/* -------------------------------------------------------------------------
 * Compile-time payload gates.
 * Everything not yet ported remains #error'd so wrong-physics on those
 * configs is caught at compile time rather than producing silent incorrect
 * results.
 * ---------------------------------------------------------------------- */
/* ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION: ported.
 * - Softening lookup: handled by P[i].ForceSoftening cache (single source of truth in
 *   compute_force_softening_kernel_radius()). Sphere-box opening criterion (mirrors
 *   forcetree.cc:2122-2130) handles NEIGHBORS_MUST_BE_COMPUTED auto-defined by this flag.
 * - Tree-node previous-step tidal tensor: SoA tidal_tensorps field (gpu_pseudo_update +
 *   gpu_moment_refresh + let_pack already populate it).
 * - Walk accumulators: tidal_zeta (scalar) + per-pair acc_corr_zeta correction (folded
 *   into acc).  Mirrors forcetree.cc:2481-2526. */
/* SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM: ported via the P[i].ForceSoftening cache.
 * The type-4 mass-based softening override lives in compute_force_softening_kernel_radius()
 * (forcetree.cc:67-69) and is read on GPU as Pp[p].ForceSoftening — single source of truth. */
/* GALSF_MERGER_STARCLUSTER_PARTICLES (type-4 star cluster softening via
 * StarParticleEffectiveSize) handled inline in gpu_force_softening_kernel_radius
 * below. */
/* ADAPTIVE_GRAVSOFT_MAX_SOFT_HARD_LIMIT (type-0 softening cap) handled inline
 * in gpu_force_softening_kernel_radius below. */
/* COMPUTE_TIDAL_TENSOR_IN_GRAVTREE + COMPUTE_JERK_IN_GRAVTREE: ported (ATFU).
 * Tidal tensor accumulation + jerk both mirror forcetree.cc:2081-2292.
 * All sub-cases are ported; see the entries below for details. */
/* COMPUTE_TIDAL_TENSOR + PMGRID: ported.  shortrange_table_tidal is mirrored
 * to SharedSpace at the top of gpu_gravtree_walk_primary() and consumed in the
 * tidal accumulation block (mirrors forcetree.cc:2538-2549). */
/* COMPUTE_TIDAL_TENSOR_IN_GRAVTREE + ADAPTIVE_GRAVSOFT_SYMMETRIZE_FORCE_BY_AVERAGING: ported.
 * The averaging branch in the inside-softening force-kernel section now sets fac_tidal
 * and averages fac2_tidal alongside fac_accel/fac_pot (mirrors forcetree.cc:2393-2394). */
/* COMPUTE_TIDAL_TENSOR_IN_GRAVTREE + ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION: ported via
 * the same walk-side block that handles ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION above. */
/* COMPUTE_TIDAL_TENSOR_IN_GRAVTREE + GRAVITY_SPHERICAL_SYMMETRY: ported.
 * The shell-theorem fac2_tidal override at the top of the tidal accumulation
 * block handles this case alongside the standard non-spherical formula. */
/* SINK_PHOTONMOMENTUM, SINK_COMPTON_HEATING, SINK_DYNFRICTION_FROMTREE:
 * ported. */
/* SPECIAL_POINT_MOTION + SPECIAL_POINT_WEIGHTED_MOTION: ported.
 * Walk-side accumulation of nearest-special-particle vel/acc lives in the
 * SINK_CALC_DISTANCES branches (leaf-particle and node paths). The Acc_Total_PrevStep
 * field is a member of particle_data and is automatically mirrored in P_dev.
 * Tree-node sink_acc lives in the GravitySoA + populated by gpu_pseudo_update +
 * gpu_moment_refresh.  The weighted variant uses the shared
 * grav_weight_function_for_weighted_motion_smoothing() (gravtree_force_kernel.h). */
/* SINK_CALC_DISTANCES, SINGLE_STAR_SINK_DYNAMICS, SINGLE_STAR_TIMESTEPPING,
 * SINGLE_STAR_FIND_BINARIES, SINGLE_STAR_FB_TIMESTEPLIMIT, SINGLE_STAR_STARFORGE_DEFAULTS:
 * ported. */
/* COSMIC_RAY_SUBGRID_LEBRON: ported. Reads SoA cr_injection at
 * node accepts + per-particle precomputed d_cr_inject at leaf nodes (since
 * cr_get_source_injection_rate is not GPU-callable). */
/* COUNT_MASS_IN_GRAVTREE: ported. tree_mass accumulator declared at function entry,
 * accumulated once per ACCEPTED interaction in the force kernel (r2>0, mass>0;
 * mirrors forcetree.cc -- excludes the target's own leaf), written to
 * P_dev[target].TreeMass at end of walk, scattered back to P[i].TreeMass.  The
 * post-loop +=P[i].Mass in gravtree.cc adds the target's own mass to finalize. */
/* DM_SCALARFIELD_SCREENING: ported. SoA tree-node fields mass_dm + s_dm are populated by
 * gpu_pseudo_update + gpu_moment_refresh + let_pack (already wired). The walk sets per-
 * interaction d_dm and mass_dm_local in both leaf and node branches, then accumulates the
 * Yukawa-screened scalar-field force on non-gas targets after the main force kernel. */
/* GRAVITY_SPHERICAL_SYMMETRY: ported. Box-center sph_center + r_target are computed
 * once at function entry; r_source is set per-interaction (leaf and node branches)
 * to the source distance from the box center. The shell-theorem force law overrides
 * fac_accel + dr right before the acc accumulation, and fac2_tidal at the start of
 * the tidal block (mirrors forcetree.cc:2446-2449 and :2534-2536). */
/* HERMITE_INTEGRATION + NEIGHBORS_MUST_BE_COMPUTED_EXPLICITLY_IN_FORCETREE: ported.
 * NEIGHBORS_MUST_BE_COMPUTED activates the sphere-box intersection opening criterion
 * in the walk loop above (mirrors forcetree.cc:2122-2130). HERMITE_INTEGRATION
 * additionally requires COMPUTE_JERK_IN_GRAVTREE which is auto-defined and already
 * ported: GravJerk written at line ~1155 and scattered back at line ~1478,
 * read by the Hermite predictor in core/kicks.cc:212 and gravtree.cc:571. */
/* ADAPTIVE_TREEFORCE_UPDATE: pre-walk skip-flag filtering in the
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
/* SELFGRAVITY_OFF: gravtree.cc wraps the entire tree dispatch (including GPU
 * dispatch) in #ifndef SELFGRAVITY_OFF, so this TU is never compiled into the
 * walk when gravity is disabled.  No guard needed here. */


/* -------------------------------------------------------------------------
 * gpu_gravtree_walk_one — device-side walk for a single target particle.
 *
 * Returns 1 on success (acc written), 0 on failure (pseudo-particle hit;
 * host must run CPU walk for this target).  Mirrors force_treeevaluate()
 * mode=0 with SINK/CR/DM/tidal payload branches stripped (gated above).
 * RT payloads (RT_USE_GRAVTREE, treecol, CHIMES, FIRE longrange)
 * are included here; they accumulate into CellP_dev[target] which the host
 * scatter loop copies back to CellP[].
 * ---------------------------------------------------------------------- */
static KOKKOS_INLINE_FUNCTION int
gpu_gravtree_walk_one(int target,
                      int maxPart, int maxNodes, int maxForeignNodes,    /* foreign-node range size; pseudos start at maxPart+maxNodes+maxForeignNodes */
                      struct particle_data *P_dev,
                      struct gas_cell_data *CellP_dev,
                      const struct gpu_gravity_tree_soa_t *tree_soa,
#ifdef GRAVITY_HYBRID_OPENING_CRIT
                      int is_first_step,   /* hybrid opening: relative criterion applies only after step 0 */
#endif
                      grav_pm_shortrange_t pm,   /* PM short-range config by value (empty when !PMGRID); per-target PLACEHIGHRESREGION override below mutates this local copy */
#ifdef RT_USE_GRAVTREE
                      const struct gpu_rt_walk_data_t *rt_data,
#endif
#ifdef SINK_PHOTONMOMENTUM
                      const struct gpu_sink_walk_data_t *sink_data,
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
                      const struct gpu_cr_walk_data_t *cr_data,
#endif
                      bool use_lazy_source,   /* evaluate the source payload on-device at each local open (no dense eager arrays; d_src_lum/d_bh_lum/d_cr_inject are NULL) */
                      const struct gpu_ewald_pot_data_t *ewald_pot,  /* periodic-image potential correction (unconditional; read only under the pure-tree-periodic EVALPOTENTIAL gate) */
                      Vec3<double> &acc_out,
                      int &ninter_out,
                      double &pot_out,
                      int &n_foreign_out)   /* diagnostic: #foreign node visits */
{
    Vec3<double> pos = P_dev[target].Pos;
    int ptype = P_dev[target].Type;
    double pmass = P_dev[target].Mass;
    if(pmass <= 0) {acc_out = Vec3<double>{0,0,0}; ninter_out = 0; pot_out = 0.0; n_foreign_out = 0; return 1;}

#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL) || defined(GALSF_MERGER_STARCLUSTER_PARTICLES)
    double soft = gpu_force_softening_kernel_radius(P_dev, target);
#else
    double soft = All.ForceSoftening[ptype];
#endif
    double zeta = 0.0;    /* unconditional (matches CPU walk); passed to the shared pair kernel, consumed there only under #if AGS */
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL)
    grav_target_select_soft_and_zeta(ptype, gpu_get_ags_zeta(P_dev, target), soft, zeta);
#endif
    double aold = All.ErrTolForceAcc * P_dev[target].OldAcc;

#if defined(PMGRID) && defined(PM_PLACEHIGHRESREGION)
    /* high-res zoom particles use the finer short-range PM cutoff (mirrors forcetree.cc target
     * prologue). The dispatcher passes the coarse-mesh rcut/asmthfac; override per target here. */
    if(pmforce_is_particle_high_res(ptype, pos)) {
        pm.rcut = All.Rcut[1]; pm.rcut2 = pm.rcut * pm.rcut; pm.asmthfac = grav_pm_asmthfac(All.Asmth[1]);
    }
#endif

    /* fed unconditionally to the shared pair kernel (consumed there only under the
     * symmetrize-by-averaging #if); matches the CPU walk's unconditional precompute. */
    const int ags_bitflag_primary = gravtree_ags_kernel_shared_bitflag(ptype);

    /* ------------------------------------------------------------------ *
     * SINK_CALC_DISTANCES + SINGLE_STAR_* local accumulators.              *
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
#ifdef COUNT_MASS_IN_GRAVTREE
    /* Diagnostic: total mass seen by this target during the walk, summed only
     * over accepted interactions (mirrors forcetree.cc). The walk excludes the
     * target's own leaf (r2==0); the post-loop +=P[i].Mass in gravtree.cc
     * finalizes the sum. */
    double tree_mass = 0.0;
#endif
#ifdef DM_SCALARFIELD_SCREENING
    /* Per-interaction DM-scalar-field state: d_dm is the displacement to the
     * dark-matter mass center (= total CoM only when source is a pure-DM leaf
     * particle), mass_dm_local is the mass at that center. Both reset each
     * leaf/node iteration. Mirrors forcetree.cc:646-647 declarations. */
    Vec3<double> d_dm = {0,0,0};
    double mass_dm_local = 0.0;
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    /* Walk-side accumulators for the tidal-criterion adaptive softening
     * correction (mirrors forcetree.cc:1822 + 2491 + 2519-2524). tidal_zeta is a
     * scalar accumulator analogous to AGS zeta; the per-pair acc_corr_zeta gets
     * folded directly into acc as we go. The primary's previous-step tidal
     * tensor is loaded once from P_dev[target] for use in the per-pair sum. */
    double tidal_zeta = 0.0;
    SymmetricTensor2<double> i_zeta_tidal_tt;
    SymmetricTensor2<double> j_zeta_tidal_tt; /* set per-interaction in leaf/node branches */
    {
        SymmetricTensor2<MyFloat> tmp = P_dev[target].tidal_tensorps_prevstep;
        for(int kk = 0; kk < 6; kk++) i_zeta_tidal_tt.data[kk] = (double) tmp.data[kk];
    }
#endif
#ifdef GRAVITY_SPHERICAL_SYMMETRY
    /* Shell-theorem gravity: forces from any source at r_source > r_target
     * vanish; forces from r_source < r_target use a 1/r^3 enclosed-mass formula
     * pointed toward the box center. Mirrors forcetree.cc:1745-1750 + 2446-2449. */
    double sph_center[3] = {0.0, 0.0, 0.0};
#ifdef BOX_PERIODIC
    sph_center[0] = 0.5 * boxSize_X;
    sph_center[1] = 0.5 * boxSize_Y;
    sph_center[2] = 0.5 * boxSize_Z;
#endif
    double r_target = 0.0; /* set per-interaction inside the PM short-range gate (mirrors the CPU walk) */
    double r_source = 0.0; /* set per-interaction in leaf/node branches */
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
    grav_sink_prox_accum_t sink_prox; grav_sink_prox_accum_init(sink_prox); /* nearest-sink + single-star timestep/binary accumulators (gravtree_force_kernel.h, shared with the CPU walk) */
#endif

    /* ------------------------------------------------------------------ *
     * RT cluster local accumulators.  All gated by the same               *
     * #ifdefs as the CPU walk in forcetree.cc.                             *
     * ------------------------------------------------------------------ */
#ifdef RT_USE_TREECOL_FOR_NH
    const double angular_bin_size = 4.0 * M_PI / RT_USE_TREECOL_FOR_NH;
    double treecol_angular_bins[RT_USE_TREECOL_FOR_NH];
    {int kb; for(kb=0; kb<RT_USE_TREECOL_FOR_NH; kb++) {treecol_angular_bins[kb]=0.0;}}
#endif

#ifdef SINK_SEED_FROM_LOCALGAS_TOTALMENCCRITERIA
    double m_enc_in_rcrit = 0.0, r_for_total_menclosed = grav_target_menc_radius(soft); /* baseline Rcrit_min applied in the helper */
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    /* per-target CR gate (the host precompute leaves t_max_cr=0 unless All.Time>All.TimeBegin,
     * mirroring the CPU walk's gate) */
    int cr_active_gate = (cr_data->t_max_cr > 0) ? 1 : 0;
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
    /* valid-gas RT gate via the shared helper */
    volatile int valid_gas_particle_for_rt = grav_target_valid_gas_for_rt(ptype, soft, pmass); /* volatile: nvc++ constant-propagates this to 0 inside the walk loop otherwise */
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
        double kappa_eff[N_RT_FREQ_BINS]; int kf; for(kf=0; kf<N_RT_FREQ_BINS; kf++) {kappa_eff[kf] = rt_kappa(-1, kf, P_dev, CellP_dev);}
        grav_target_rt_fac_stellum(soft, pmass, kappa_eff, fac_stellum);
    }
#endif

    Vec3<double> acc = {0,0,0};
    int ninter = 0;
    double pot = 0.0;
    int n_foreign = 0;  /* diagnostic */

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
        int ptype_sec = -1;   /* unconditional, matching the CPU walk: consumed by the shared pair kernel */
        double zeta_sec = 0.0;   /* unconditional (matches CPU walk); assigned below only under #if AGS */
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
        double gasmass = 0.0;
#endif

        if(no < maxPart) /* particle leaf */
        {
            dr = P_dev[no].Pos - pos;
            gravity_box_nearest_image(dr[0], dr[1], dr[2], -1);
            r2 = dr.norm_sq();
            mass = P_dev[no].Mass;
#if defined(GRAVTREE_SOURCE_DEVICE_TU)
            /* Lazy source payload for this local leaf: evaluate the SSOT helper on-device
             * (identical gates/formula to the eager prefill) instead of reading the dense
             * arrays. Computed once here; consumed by the RT/sink/CR blocks below. Only for
             * local particle leaves (no<maxPart) -- foreign leaves and nodes stay
             * moment-backed and never reach this branch. */
            struct gravtree_source_inputs_t lazy_src;
            if(use_lazy_source) { gravtree_fill_particle_source_inputs(no, P_dev, CellP_dev, &lazy_src); }
#endif
#ifdef DM_SCALARFIELD_SCREENING
            /* Set per-interaction DM state for this leaf particle (mirrors forcetree.cc:2055). */
            if(ptype != 0 && P_dev[no].Type == 1) { d_dm = dr; mass_dm_local = mass; }
            else { d_dm = Vec3<double>{0,0,0}; mass_dm_local = 0; }
#endif
#ifdef GRAVITY_SPHERICAL_SYMMETRY
            r_source = grav_spherical_symmetry_r_from_center(P_dev[no].Pos[0],P_dev[no].Pos[1],P_dev[no].Pos[2],sph_center[0],sph_center[1],sph_center[2]);
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
            /* Load secondary's previous-step tidal tensor (mirrors forcetree.cc:1945). */
            {
                SymmetricTensor2<MyFloat> tmp = P_dev[no].tidal_tensorps_prevstep;
                for(int kk = 0; kk < 6; kk++) j_zeta_tidal_tt.data[kk] = (double) tmp.data[kk];
            }
#endif
#if defined(SINK_DYNFRICTION_FROMTREE) || defined(COMPUTE_JERK_IN_GRAVTREE)
            dv = P_dev[no].Vel - vel;
#endif
#ifdef SINK_DYNFRICTION_FROMTREE
            m_j_eff_for_df = mass;
#endif
            /* secondary (leaf) softening, loaded unconditionally so a pair whose source softening
             * exceeds the target's gets the symmetrized max(h,h_p) force (mirrors forcetree.cc:2093).
             * ptype_sec/zeta_sec stay gated -- only the adaptive symmetrize-by-averaging path uses them. */
            h_p = gpu_force_softening_kernel_radius(P_dev, no);
            ptype_sec = P_dev[no].Type;
#if defined(ADAPTIVE_GRAVSOFT_FORGAS)
            if(ptype_sec == 0) {zeta_sec = gpu_get_ags_zeta(P_dev, no);}
#elif defined(ADAPTIVE_GRAVSOFT_FORALL)
            zeta_sec = gpu_get_ags_zeta(P_dev, no);
#endif
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
            gasmass = (P_dev[no].Type == 0) ? P_dev[no].Mass : 0.0;
#if defined(SINK_ALPHADISK_ACCRETION) && defined(RT_USE_TREECOL_FOR_NH)
            /* gas at the inner edge of a sink's alpha-disk should not see a hole due to
             * the sink (mirrors forcetree.cc leaf branch + the node-moment kernel). */
            if(P_dev[no].Type == 5) {gasmass = (double) P_dev[no].Sink_Mass_Reservoir;}
#endif
#endif
#ifdef RT_USE_GRAVTREE
            /* Load leaf luminosity only for valid gas targets (mirrors forcetree.cc; the
             * RT accumulation below is gated the same way, so non-gas targets never read
             * these and skipping the loads avoids the wasted per-leaf table traffic). */
            if(valid_gas_particle_for_rt)
            {
                d_stellarlum = dr;
                int kf; for(kf=0; kf<N_RT_FREQ_BINS; kf++) {
#if defined(GRAVTREE_SOURCE_DEVICE_TU)
                    mass_stellarlum[kf] = use_lazy_source ? (lazy_src.rt_active ? lazy_src.src_lum[kf] : (MyFloat)0)
                                                          : rt_data->src_lum[(long)no * N_RT_FREQ_BINS + kf];
#else
                    mass_stellarlum[kf] = rt_data->src_lum[(long)no * N_RT_FREQ_BINS + kf];
#endif
                }
#ifdef CHIMES_STELLAR_FLUXES
                for(kf=0; kf<CHIMES_LOCAL_UV_NBINS; kf++) {
#if defined(GRAVTREE_SOURCE_DEVICE_TU)
                    chimes_mass_stellarlum_G0[kf]  = use_lazy_source ? (lazy_src.rt_active ? lazy_src.src_lum_G0[kf]  : 0.0) : rt_data->src_lum_G0[(long)no * CHIMES_LOCAL_UV_NBINS + kf];
                    chimes_mass_stellarlum_ion[kf] = use_lazy_source ? (lazy_src.rt_active ? lazy_src.src_lum_ion[kf] : 0.0) : rt_data->src_lum_ion[(long)no * CHIMES_LOCAL_UV_NBINS + kf];
#else
                    chimes_mass_stellarlum_G0[kf] = rt_data->src_lum_G0[(long)no * CHIMES_LOCAL_UV_NBINS + kf];
                    chimes_mass_stellarlum_ion[kf] = rt_data->src_lum_ion[(long)no * CHIMES_LOCAL_UV_NBINS + kf];
#endif
                }
#endif
#ifdef SINK_PHOTONMOMENTUM
                /* per-sink-leaf angle-weighted luminosity (shared formula helper) */
                mass_sinklumwt_forradfb = 0.0;
                if(P_dev[no].Type == 5) {
                    double bhlum_t, bha0, bha1, bha2;
#if defined(GRAVTREE_SOURCE_DEVICE_TU)
                    if(use_lazy_source) {
                        bhlum_t = lazy_src.bh_active ? (double)lazy_src.bh_lum : 0.0;
                        bha0 = lazy_src.bh_active ? (double)lazy_src.bh_angle[0] : 0.0;
                        bha1 = lazy_src.bh_active ? (double)lazy_src.bh_angle[1] : 0.0;
                        bha2 = lazy_src.bh_active ? (double)lazy_src.bh_angle[2] : 0.0;
                    } else
#endif
                    {
                        bhlum_t = (double) sink_data->bh_lum[no];
                        bha0 = (double) sink_data->bh_angle[no][0]; bha1 = (double) sink_data->bh_angle[no][1]; bha2 = (double) sink_data->bh_angle[no][2];
                    }
                    mass_sinklumwt_forradfb = grav_sink_fb_angleweight(bhlum_t, bha0, bha1, bha2, dr[0], dr[1], dr[2]);
                }
#endif
            }
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
            /* Mirror forcetree.cc:1734-1736 leaf CR source injection. */
#if defined(GRAVTREE_SOURCE_DEVICE_TU)
            cr_injection = use_lazy_source ? (double) lazy_src.cr_inject : (double) cr_data->cr_inject[no];
#else
            cr_injection = (double) cr_data->cr_inject[no];
#endif
#endif
            /* Sink-distance + single-star timestepping tracking on particle leafs via the
             * shared helper (gravtree_force_kernel.h) — CPU-walk semantics verbatim. */
#ifdef SINK_CALC_DISTANCES
            if((r2 > 0) && (mass > 0))
            {
                grav_sink_prox_target_t prox_target = {}; prox_target.ptype = ptype; prox_target.pmass = pmass; prox_target.soft = soft;
#if defined(SINGLE_STAR_TIMESTEPPING)
                prox_target.vel = vel;
#endif
                grav_sink_prox_leaf_src_t prox_src = {}; prox_src.src_type = P_dev[no].Type; prox_src.src_mass = P_dev[no].Mass; prox_src.motion.vel = P_dev[no].Vel;
#if defined(SPECIAL_POINT_MOTION) || defined(SPECIAL_POINT_WEIGHTED_MOTION)
                prox_src.motion.acc = P_dev[no].Acc_Total_PrevStep;
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
                prox_src.motion.max_feedback_vel = P_dev[no].MaxFeedbackVel;
#endif
                grav_sink_prox_leaf_accumulate(r2, dr, prox_target, prox_src, sink_prox);
            }
#endif /* SINK_CALC_DISTANCES */
        }
        else if(no >= maxPart + maxNodes + maxForeignNodes) /* pseudo-particle — remote (foreign-node range below pseudos; foreign nodes treated as internal in the else branch below) */
        {
            return 0; /* host runs CPU walk for this target */
        }
        else /* tree node */
        {
            if(no >= maxPart + maxNodes && no < maxPart + maxNodes + maxForeignNodes) n_foreign++;  /* diagnostic */
            int idx = no - maxPart;
            Vec3<MyFloat> s_node = Vec3<MyFloat>{(MyFloat)tree_soa->s[idx][0], (MyFloat)tree_soa->s[idx][1], (MyFloat)tree_soa->s[idx][2]};
            MyFloat len_node = tree_soa->len[idx];
            MyFloat msoft_node = tree_soa->maxsoft[idx];
            MyFloat mass_node = tree_soa->mass[idx];
            Vec3<MyFloat> center_node = tree_soa->center[idx];

            dr[0] = s_node[0] - pos[0];
            dr[1] = s_node[1] - pos[1];
            dr[2] = s_node[2] - pos[2];
            gravity_box_nearest_image(dr[0], dr[1], dr[2], -1);
            r2 = dr.norm_sq();

            /* LET guard (mirrors forcetree.cc): if a foreign node has nextnode < 0
             * (unreplaced -1 sentinel from unpack), opening it would immediately exit
             * the while(no >= 0) walk, skipping this node's force contribution.
             * Force multipole treatment instead. */
            int in_foreign_n = (no >= maxPart + maxNodes);
            int foreign_force_multipole = (in_foreign_n && (tree_soa->nextnode[idx] < 0));

            /* Foreign-leaf identity lookup.  foreign_slot = no-(MaxPart+MaxNodes) = idx-maxNodes
             * -- the foreign-only sidecar index, EXPLICIT and bounds-checked so it can never be
             * confused with the per-node SoA index idx (= no-MaxPart). */
            int    fl_tag = 0, fl_type = -1;
            double fl_zeta = 0.0, fl_soft = 0.0;
            if(in_foreign_n && tree_soa->foreign_leaf_tag) {
                int fs = idx - maxNodes;
                if(fs >= 0 && fs < tree_soa->foreign_leaf_cap) {
                    fl_tag  = tree_soa->foreign_leaf_tag[fs];
                    fl_type = tree_soa->foreign_leaf_type[fs];
                    fl_zeta = (double) tree_soa->foreign_leaf_zeta[fs];
                    fl_soft = (double) tree_soa->foreign_leaf_soft[fs];
                }
            }

            /* empty-node skip (mirrors forcetree.cc:2165): a zero-mass node contributes no
             * force -> advance to the sibling. Not gated on foreign_force_multipole (a skip
             * is never converted to a forced multipole); also avoids descending empty nodes. */
            if(mass_node <= 0) { no = tree_soa->sibling[idx]; continue; }

            /* single-particle node -> open to its leaf for an exact force (mirrors
             * forcetree.cc:2171). A foreign single-particle node with nextnode<0 must instead
             * be used as a multipole (opening it would exit the walk and drop its contribution),
             * so gate on foreign_force_multipole, matching the LET-sentinel guard above. */
            if(!(tree_soa->bitflags[idx] & (1 << BITFLAG_MULTIPLEPARTICLES))) {
                if(!foreign_force_multipole) { no = tree_soa->nextnode[idx]; continue; }
            }

            /* Acceptance geometry via the shared predicate (gravtree_opening.h), the single home for
             * the node opening decision. The caller owns the wrapped dr/r2 (also used below for the
             * accepted-node force) and the foreign-multipole policy; the predicate is foreign-blind
             * geometry. PM short-range cull, neighbour sphere-box / softening-open, the angular and
             * relative opening criteria, and the sink-direct gate all live in the predicate. */
            {
                double cen0 = (double)center_node[0] - pos[0];
                double cen1 = (double)center_node[1] - pos[1];
                double cen2 = (double)center_node[2] - pos[2];
#ifdef PMGRID
                double pred_rcut = pm.rcut, pred_rcut2 = pm.rcut2;
#else
                double pred_rcut = 0.0, pred_rcut2 = 0.0;
#endif
#ifdef GRAVITY_HYBRID_OPENING_CRIT
                int pred_is_first_step = is_first_step;
#else
                int pred_is_first_step = 0;
#endif
#if (defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES)) && defined(SINGLE_STAR_DIRECT_GRAVITY_RADIUS)
                int pred_n_sink = (int)tree_soa->N_SINK[idx];
#else
                int pred_n_sink = 0;
#endif
                gravtree_open_t pred = gravtree_open_decision_from_distances(
                    r2, cen0, cen1, cen2, soft, h, aold, ptype,
                    (double)len_node, (double)mass_node, (double)msoft_node,
                    pred_rcut, pred_rcut2, pred_n_sink, pred_is_first_step);
                /* Foreign LET policy.  A foreign tagged leaf (fl_tag==1) is a TERMINAL leaf source:
                 * its nextnode is the DFS CONTINUATION after the leaf, NOT a child to descend.  So a
                 * predicate OPEN on it cannot mean "descend" (there is nowhere to descend) -- it must
                 * mean "accept this already-leaf source with leaf semantics" (the identity lookup
                 * above supplies the leaf identity at the payload load below).  foreign_force_multipole (the nextnode<0 sentinel
                 * case) is the other terminal that must be accepted, not descended.  Both fall through to
                 * the accept path; only a genuine descendable node takes nextnode.  (Pre-fix the descend
                 * guard keyed on foreign_force_multipole alone, so a tagged leaf whose continuation
                 * resolved to a valid >=0 slot was OPENED -> advanced to its continuation -> its mass
                 * was never summed; the dropped foreign-leaf mass is rank-asymmetric, breaking force
                 * reciprocity / momentum conservation at np>=2 under adaptive softening.) */
                int foreign_real_leaf = (in_foreign_n && fl_tag == 1);
                int must_accept_foreign_terminal = foreign_force_multipole || foreign_real_leaf;
                if(pred == GRAV_SKIP_NODE) { no = tree_soa->sibling[idx]; continue; }
                if(pred == GRAV_OPEN_NODE && !must_accept_foreign_terminal) { no = tree_soa->nextnode[idx]; continue; }
                /* Permanent invariant guard (predicate-keyed): an OPEN that must_accept a foreign
                 * terminal is LEGAL when the terminal is a tagged real leaf (fl_tag==1) -- routed
                 * through particle-leaf semantics at the payload load below.  The ILLEGAL case is an
                 * untagged forced multipole (foreign_force_multipole && !fl_tag): a non-particle foreign
                 * terminal accepted as a multipole in leaf-sensitive support -- a silent physics
                 * downgrade the host controlled-stops on after the walk. */
                if(pred == GRAV_OPEN_NODE && must_accept_foreign_terminal && fl_tag != 1) {
                    Kokkos::atomic_add(&g_inv_fterm_aggregate, 1LL);
                }
            }

            /* Node accepted — load payload fields */
            h_p = msoft_node;
            mass = mass_node;
            /* A tagged real foreign single-particle leaf must be consumed with particle-leaf
             * secondary-source semantics.  The node payload above already supplied mass/h_p and the
             * synthesized RT/sink/CR/tidal (singleton-aggregate == particle value); restore the two
             * leaf-identity fields the node moment cannot carry (Type + AGS_zeta) via the shared seam
             * so grav_force_pair applies AGS symmetrization/zeta exactly as on the source's home rank. */
            if(fl_tag == 1) {
                grav_apply_foreign_leaf_identity(fl_tag, fl_type, fl_zeta, fl_soft, &ptype_sec, &zeta_sec, &h_p);
            }
#ifdef DM_SCALARFIELD_SCREENING
            /* Set per-interaction DM state for this accepted node (mirrors forcetree.cc:2272).
             * d_dm uses the DM CoM s_dm, NOT the total CoM (s_node). */
            if(ptype != 0) {
                d_dm[0] = (double)tree_soa->s_dm[idx][0] - pos[0];
                d_dm[1] = (double)tree_soa->s_dm[idx][1] - pos[1];
                d_dm[2] = (double)tree_soa->s_dm[idx][2] - pos[2];
                mass_dm_local = (double)tree_soa->mass_dm[idx];
            } else { d_dm = Vec3<double>{0,0,0}; mass_dm_local = 0; }
#endif
#ifdef GRAVITY_SPHERICAL_SYMMETRY
            r_source = grav_spherical_symmetry_r_from_center(s_node[0],s_node[1],s_node[2],sph_center[0],sph_center[1],sph_center[2]);
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
            /* Load node's previous-step tidal tensor from SoA (mirrors forcetree.cc:2278). */
            for(int kk = 0; kk < 6; kk++) {
                j_zeta_tidal_tt.data[kk] = (double) tree_soa->tidal_tensorps[(long)idx * 6 + kk];
            }
#endif
#if defined(SINK_DYNFRICTION_FROMTREE) || defined(COMPUTE_JERK_IN_GRAVTREE)
            dv[0] = (double) tree_soa->node_vs[idx][0] - vel[0];
            dv[1] = (double) tree_soa->node_vs[idx][1] - vel[1];
            dv[2] = (double) tree_soa->node_vs[idx][2] - vel[2];
#endif
#ifdef SINK_DYNFRICTION_FROMTREE
            {
                long np = tree_soa->N_part[idx];
                m_j_eff_for_df = (np > 0) ? (mass / (double)np) : 0.0;
            }
#endif
#ifdef GRAVTREE_CALCULATE_GAS_MASS_IN_NODE
            gasmass = tree_soa->gasmass[idx];
#endif
#ifdef RT_USE_GRAVTREE
            /* Load node stellar luminosity only for valid gas targets (mirrors
             * forcetree.cc; the RT accumulation below is gated the same way).
             * valid_gas_particle_for_rt is volatile int -- nvc++ constant-propagates
             * plain int gates in device code otherwise. */
            if(valid_gas_particle_for_rt)
            {
                int kf; for(kf=0; kf<N_RT_FREQ_BINS; kf++) {
                    mass_stellarlum[kf] = tree_soa->stellar_lum[idx * N_RT_FREQ_BINS + kf];
                }
#ifdef CHIMES_STELLAR_FLUXES
                for(kf=0; kf<CHIMES_LOCAL_UV_NBINS; kf++) {
                    chimes_mass_stellarlum_G0[kf] = tree_soa->chimes_stellar_lum_G0[(long)idx * CHIMES_LOCAL_UV_NBINS + kf];
                    chimes_mass_stellarlum_ion[kf] = tree_soa->chimes_stellar_lum_ion[(long)idx * CHIMES_LOCAL_UV_NBINS + kf];
                }
#endif
#ifdef RT_SEPARATELY_TRACK_LUMPOS
                d_stellarlum[0] = tree_soa->rt_source_lum_s[idx][0] - pos[0];
                d_stellarlum[1] = tree_soa->rt_source_lum_s[idx][1] - pos[1];
                d_stellarlum[2] = tree_soa->rt_source_lum_s[idx][2] - pos[2];
                gravity_box_nearest_image(d_stellarlum[0], d_stellarlum[1], d_stellarlum[2], -1);
#else
                d_stellarlum = dr;
#endif
#ifdef SINK_PHOTONMOMENTUM
                /* node-aggregated sink angle-weighted luminosity (shared formula helper) */
                mass_sinklumwt_forradfb = grav_sink_fb_angleweight((double) tree_soa->sink_lum[idx],
                                                                   (double) tree_soa->sink_lum_grad[idx][0], (double) tree_soa->sink_lum_grad[idx][1], (double) tree_soa->sink_lum_grad[idx][2],
                                                                   d_stellarlum[0], d_stellarlum[1], d_stellarlum[2]);
#endif
            }
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
            /* Mirror forcetree.cc:1956-1957 node-aggregated CR injection. */
            cr_injection = (double) tree_soa->cr_injection[idx];
#endif
            /* Node-side sink distance + timestepping accumulators via the shared helper
             * (gravtree_force_kernel.h) — CPU-walk semantics verbatim. The sink_vel/sink_acc
             * SoA fields are populated from Nodes[] by gpu_pseudo_update + gpu_moment_refresh. */
#ifdef SINK_CALC_DISTANCES
#ifdef SPECIAL_POINT_WEIGHTED_MOTION
            {
                Vec3<double> node_vs = Vec3<double>{(double)tree_soa->node_vs[idx][0], (double)tree_soa->node_vs[idx][1], (double)tree_soa->node_vs[idx][2]};
                grav_sink_prox_node_specialweighted(r2, node_vs, ptype, sink_prox);
            }
#endif
            if(tree_soa->sink_mass[idx] > 0)
            {
                Vec3<double> sink_dr;
                sink_dr[0] = tree_soa->sink_pos[idx][0] - pos[0];
                sink_dr[1] = tree_soa->sink_pos[idx][1] - pos[1];
                sink_dr[2] = tree_soa->sink_pos[idx][2] - pos[2];
                gravity_box_nearest_image(sink_dr[0], sink_dr[1], sink_dr[2], -1);
                grav_sink_prox_target_t prox_target = {}; prox_target.ptype = ptype; prox_target.pmass = pmass; prox_target.soft = soft;
#if defined(SINGLE_STAR_TIMESTEPPING)
                prox_target.vel = vel;
#endif
                grav_sink_prox_node_src_t prox_src = {}; prox_src.sink_mass = (double) tree_soa->sink_mass[idx];
#if defined(SINGLE_STAR_FIND_BINARIES)
                prox_src.n_sink = (int) tree_soa->N_SINK[idx];
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) || defined(SPECIAL_POINT_MOTION)
                prox_src.motion.vel = tree_soa->sink_vel[idx];
#endif
#if defined(SPECIAL_POINT_MOTION)
                prox_src.motion.acc = tree_soa->sink_acc[idx];
#endif
#if defined(SINGLE_STAR_TIMESTEPPING) && defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
                prox_src.motion.max_feedback_vel = tree_soa->MaxFeedbackVel[idx];
#endif
                grav_sink_prox_node_accumulate(r2, sink_dr, prox_src, prox_target, sink_prox);
            }
#endif /* SINK_CALC_DISTANCES */
        }

        /* Force kernel — common path for accepted particles and closed nodes. */
        if((r2 > 0.0) && (mass > 0.0))
        {
            double r = sqrt(r2);
            double fac_accel;
            double fac_pot = 0;   /* unconditional; a dead 0 when !EVALPOTENTIAL (consumed only under that gate) */
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
            double fac_tidal = 0.0, fac2_tidal = 0.0; /* mirrors forcetree.cc:1489; populated in branches below */
#endif
            /* pair-wise gravity terms (Newtonian/softened selection, softening symmetrization,
             * AGS zeta corrections) via the shared contribution kernel (gravtree_force_kernel.h),
             * the single home for the pair physics on both walks. */
            {
                grav_force_pair_t pair_out = grav_force_pair(r, r2, mass, h, h_p, ptype, ptype_sec, pmass,
                                                             zeta, zeta_sec, ags_bitflag_primary);
                fac_accel = pair_out.fac_accel;
#ifdef EVALPOTENTIAL
                fac_pot = pair_out.fac_pot;
#endif
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
                fac_tidal = pair_out.fac_tidal; fac2_tidal = pair_out.fac2_tidal;
#endif
            }
            int tabindex = 0;   /* unconditional; computed + consumed only under PMGRID */
#ifdef PMGRID
            tabindex = grav_pm_shortrange_tabindex(pm.asmthfac, r);
            /* PM short-range gate (mirrors forcetree.cc): wraps the acceleration,
             * potential, dynamical-friction, adaptive-tidal-correction, tidal-tensor
             * and jerk contributions ONLY. A source beyond the table range contributes
             * nothing to those, but fac_accel stays UN-truncated for the payload blocks
             * below the gate (TREECOL column estimate) -- there is no PM-side completion
             * for those integrals, so truncating or zeroing them would be wrong. */
            if(grav_pm_shortrange_in_range(tabindex))
#endif
            {
#ifdef PMGRID
            grav_force_apply_pm_truncation(pm, tabindex, fac_pot, fac_accel);
#endif
#ifdef GRAVITY_SPHERICAL_SYMMETRY
            /* Shell-theorem override via the shared helper; pot above is unmodified, matching CPU sequencing. */
            r_target = grav_spherical_symmetry_r_from_center(pos[0],pos[1],pos[2],sph_center[0],sph_center[1],sph_center[2]);
            grav_spherical_symmetry_force_override(r_source, r_target, h, mass, sph_center[0],sph_center[1],sph_center[2], pos[0],pos[1],pos[2], dr, fac_accel);
#endif
            acc += fac_accel * dr;
#ifdef EVALPOTENTIAL
            pot    += fac_pot;
#ifdef GIZMO_GPU_EWALD_POT_CORRECTION
            /* Ewald periodic-image potential correction (mirrors forcetree.cc:2300).
             * Pure-tree periodic only; under PMGRID the long-range potential comes
             * from the PM solver.  active is 1 in a healthy run (acquire failure
             * hard-stops the caller); the guard only covers the post-endrun drain. */
            if(ewald_pot->active) {
                grav_ewald_interp_weights ew = grav_ewald_interp_setup(dr[0], dr[1], dr[2], ewald_pot->fac_intp);
                pot += mass * grav_ewald_interp_apply(ewald_pot->potcorr, ew);
            }
#endif
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
            /* Adaptive softening 'tidal' correction terms via the shared helper
             * (gravtree_force_kernel.h); GRAVITY_SPHERICAL_SYMMETRY override of
             * fac2_tidal happens later, in the tidal accumulation block. */
            grav_ags_tidal_criterion_accumulate(r, r2, dr, mass, h, h_p, ptype, ptype_sec, fac_tidal, fac2_tidal,
                                                i_zeta_tidal_tt, j_zeta_tidal_tt, tidal_zeta, acc);
#endif
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
#ifdef GRAVITY_SPHERICAL_SYMMETRY
            fac2_tidal = grav_spherical_symmetry_fac2_tidal_override(r_source, r_target, h, mass);
#endif
            /* tidal-tensor accumulation via the shared helper (PM-truncated or bare;
             * tabindex is in range here -- this call sits inside the PM short-range gate) */
            grav_tidal_tensor_accumulate(dr, fac_tidal, fac2_tidal, pm, tabindex, tidal_acc);
#endif
#ifdef COMPUTE_JERK_IN_GRAVTREE
            grav_jerk_accumulate(dv, dr, fac_accel, fac2_tidal, ptype, jerk_acc);
#endif
#ifdef SINK_DYNFRICTION_FROMTREE
            /* dynamical-friction deflection for type-5 sink targets (shared helper) */
            grav_sink_dynfriction_accumulate(dr, dv, fac_accel, mass, target_sink_mass, m_j_eff_for_df, ptype, acc);
#endif /* SINK_DYNFRICTION_FROMTREE */
            } /* closes the PM short-range gate (tabindex in range; mirrors forcetree.cc) */
            ninter++;
#ifdef COUNT_MASS_IN_GRAVTREE
            /* counted only for accepted interactions (r2>0, mass>0), mirroring
             * forcetree.cc -- the walk excludes the target's own (r2==0) leaf;
             * the post-loop += P[i].Mass in gravtree.cc adds it back exactly once. */
            tree_mass += mass;
#endif

            /* ------------------------------------------------------------ *
             * RT cluster payloads.  Structure mirrors                      *
             * forcetree.cc: OUTSIDE the PM short-range gate, so for an      *
             * out-of-range source fac_accel is the raw un-truncated value   *
             * here (used by the TREECOL column estimate, which has no       *
             * PM-side completion); RT_USE_GRAVTREE computes its own fac_rt  *
             * from d_stellarlum independently.                              *
             * ------------------------------------------------------------ */
#ifdef RT_USE_TREECOL_FOR_NH
            grav_treecol_accumulate(dr, r, fac_accel, gasmass, mass, angular_bin_size, treecol_angular_bins);
#endif

#ifdef SINK_SEED_FROM_LOCALGAS_TOTALMENCCRITERIA
            /* Mirror forcetree.cc. Per-interaction mass accumulation, where
             * each visited node contributes its multipole mass when within Rcrit. */
            if(r < r_for_total_menclosed) {m_enc_in_rcrit += mass;}
#endif

#ifdef COSMIC_RAY_SUBGRID_LEBRON
            grav_cr_lebron_accumulate(ptype, r, soft, cr_injection, cr_active_gate, cr_data->t_max_cr, pm, SubGrid_CosmicRayEnergyDensity);
#endif

#ifdef RT_USE_GRAVTREE
            if(valid_gas_particle_for_rt)
            {
                /* payload formulas in the shared helper; fac_rt computed there from d_stellarlum
                 * (may differ from dr when RT_SEPARATELY_TRACK_LUMPOS; otherwise d_stellarlum == dr) */
                grav_rt_src_t rt_src = {}; rt_src.d_stellarlum = d_stellarlum; rt_src.soft = soft; rt_src.mass_stellarlum = mass_stellarlum;
#ifdef CHIMES_STELLAR_FLUXES
                rt_src.chimes_mass_stellarlum_G0 = chimes_mass_stellarlum_G0; rt_src.chimes_mass_stellarlum_ion = chimes_mass_stellarlum_ion;
#endif
#ifdef SINK_PHOTONMOMENTUM
                rt_src.mass_sinklumwt_forradfb = mass_sinklumwt_forradfb;
#endif
#if defined(RT_LEBRON) && !defined(RT_USE_GRAVTREE_SAVE_RAD_FLUX)
                rt_src.fac_stellum = fac_stellum;
#endif
                grav_rt_accum_t rt_accum = {};
#if defined(RT_USE_GRAVTREE_SAVE_RAD_ENERGY)
                rt_accum.Rad_E_gamma = Rad_E_gamma;
#endif
#ifdef CHIMES_STELLAR_FLUXES
                rt_accum.chimes_flux_G0 = chimes_flux_G0; rt_accum.chimes_flux_ion = chimes_flux_ion;
#endif
#ifdef GALSF_FB_FIRE_RT_LONGRANGE
                rt_accum.incident_flux_uv = &incident_flux_uv; rt_accum.incident_flux_euv = &incident_flux_euv;
#endif
#ifdef SINK_COMPTON_HEATING
                rt_accum.incident_flux_agn = &incident_flux_agn;
#endif
#ifdef RT_OTVET
                rt_accum.RT_ET = RT_ET;
#endif
#if defined(RT_USE_GRAVTREE_SAVE_RAD_FLUX)
                rt_accum.Rad_Flux = Rad_Flux;
#endif
                grav_rt_payload_accumulate(rt_src, rt_accum, acc);
            } /* if(valid_gas_particle_for_rt) */
#endif /* RT_USE_GRAVTREE */

#ifdef DM_SCALARFIELD_SCREENING
            /* Yukawa-screened scalar-field force on non-gas targets (shared helper;
             * own table gate keyed on the dm-center distance, outside the main PM gate) */
            if(ptype != 0)
            {
                grav_dm_scalarfield_accumulate(d_dm, mass_dm_local, h, pm, acc);
            }
#endif /* DM_SCALARFIELD_SCREENING */

        } /* if((r2>0)&&(mass>0)) */

        if(no < maxPart) {
            no = tree_soa->nextnode_aux[no];
        } else {
            no = tree_soa->sibling[no - maxPart];
        }
    } /* while(no >= 0) */

    /* ------------------------------------------------------------------ *
     * Post-walk: write RT outputs to CellP_dev / P_dev.  The host scatter  *
     * loop in gpu_gravtree_walk_primary copies these to CellP[] / P[].    *
     * ------------------------------------------------------------------ */
#ifdef RT_USE_TREECOL_FOR_NH
    {int k; for(k=0; k<RT_USE_TREECOL_FOR_NH; k++) {P_dev[target].ColumnDensityBins[k] = treecol_angular_bins[k];}}
#endif
#ifdef SINK_SEED_FROM_LOCALGAS_TOTALMENCCRITERIA
    P_dev[target].MencInRcrit = m_enc_in_rcrit;
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

    /* Sink-distance / single-star timestepping outputs.
     * Mirrors forcetree.cc:2499-2526 (mode=0). Scatter from P_dev back to P[]
     * happens in the host post-walk loop (primary driver). */
#ifdef SINK_CALC_DISTANCES
    P_dev[target].Min_Distance_to_Sink = sqrt(sink_prox.Min_Distance_to_Sink2);
    P_dev[target].Min_xyz_to_Sink = sink_prox.Min_xyz_to_Sink;
#ifdef SINGLE_STAR_FIND_BINARIES
    P_dev[target].is_in_a_binary = 0;
    P_dev[target].Min_Sink_OrbitalTime = sink_prox.Min_Sink_OrbitalTime;
    if(sink_prox.Min_Sink_OrbitalTime < MAX_REAL_NUMBER) {
        P_dev[target].is_in_a_binary = 1;
        P_dev[target].comp_Mass = sink_prox.comp_Mass;
        P_dev[target].comp_dx = sink_prox.comp_dx;
        P_dev[target].comp_dv = sink_prox.comp_dv;
    }
#endif
#ifdef SINGLE_STAR_TIMESTEPPING
    P_dev[target].Min_Sink_Approach_Time = sqrt(sink_prox.Min_Sink_Approach_Time);
    P_dev[target].Min_Sink_Freefall_time = sqrt(sqrt(sink_prox.Min_Sink_Freefall_time) / All.G);
#ifdef SINGLE_STAR_FB_TIMESTEPLIMIT
    P_dev[target].Min_Sink_FeedbackTime = sqrt(sink_prox.Min_Sink_FeedbackTime);
#endif
#endif
#endif /* SINK_CALC_DISTANCES */

#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
    P_dev[target].tidal_tensorps = tidal_acc;
#endif
#ifdef COMPUTE_JERK_IN_GRAVTREE
    P_dev[target].GravJerk = jerk_acc;
#endif
#ifdef COUNT_MASS_IN_GRAVTREE
    P_dev[target].TreeMass = tree_mass;
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
    P_dev[target].tidal_zeta = (MyFloat) tidal_zeta;
#endif
#ifdef SPECIAL_POINT_MOTION
    P_dev[target].vel_of_nearest_special = Vec3<MyFloat>{(MyFloat)sink_prox.vel_of_nearest_special[0],
                                                         (MyFloat)sink_prox.vel_of_nearest_special[1],
                                                         (MyFloat)sink_prox.vel_of_nearest_special[2]};
    P_dev[target].acc_of_nearest_special = Vec3<MyFloat>{(MyFloat)sink_prox.acc_of_nearest_special[0],
                                                         (MyFloat)sink_prox.acc_of_nearest_special[1],
                                                         (MyFloat)sink_prox.acc_of_nearest_special[2]};
#ifdef SPECIAL_POINT_WEIGHTED_MOTION
    P_dev[target].weight_sum_for_special_point_smoothing = (MyFloat) sink_prox.weight_sum_for_special_point_smoothing;
#endif
#endif

    acc_out = acc;
    ninter_out = ninter;
    pot_out = pot;
    n_foreign_out = n_foreign;
    return 1;
}


extern "C" int gpu_gravtree_walk_primary(void)
{
    GIZMO_GPU_ENSURE_ALL_FRESH();
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
     * drift all nodes whose Ti_current lags All.Ti_Current.  The node drift
     * loop is a single GPU kernel that mutates UVM Nodes/Extnodes AND the
     * SoA mirror in one pass — no host loop, no AoS->SoA reseed afterwards.
     * Cost is O(active drifted nodes) with GPU parallelism over
     * Numnodestree (early-out when Ti_current matches). */
    /* Sub-bucket timing — env-gated; no-op when GIZMO_VERBOSE_DIAG off. */
    double t_grv_start = my_second();
    /* Host-side wrapper in the GPU TU must use the out-of-line host accessor
     * `gizmo_host_ti_current()` (defined in core/predict.cc) rather than a
     * bare All.Ti_Current read, so the host-snapshot intent at this call
     * site stays correct even when the device-pass redirect is active. */
    integertime ti_curr_host = gizmo_host_ti_current();
    move_particles(ti_curr_host); /* drifts all P[], invalidates arena */
    double t_grv_mp = my_second();
    /* SoA must exist before the drift kernel — it writes mirror fields. */
    gpu_gravity_tree_acquire(MaxNodes + 1, Nodes_base, Extnodes_base);
    if(gpu_force_drift_nodes(ti_curr_host) != 0) {
        endrun(929702);
        return 1;   /* soft bad-stop: skip walk on un-drifted nodes (idx_host not yet alloc'd); drains at next poll */
    }
    double t_grv_drift_nodes = my_second();

    int *idx_host = (int *) mymalloc("gpu_grav_idx", num_active_total * sizeof(int));
    int num_active = 0;
    for(int a = 0; a < num_active_total; a++) {
        int i = ActiveParticleList[a];
        if(ProcessedFlag[i]) {continue;}
        /* SSOT pre-walk candidacy (Mass>0 + Hermite eligibility + needs_new_treeforce):
         * the GPU pre-pass must use the same candidate set as the CPU primary walk +
         * finalization, or it leaves a non-candidate's GravAccel fresh-written but raw.
         * ProcessedFlag is intentionally left unset on a candidacy skip here (matches
         * the CPU primary walk: a cached/extrapolated particle is finalized later). */
        if(!gravity_treewalk_candidate_prewalk(i)) {continue;}
        idx_host[num_active++] = i;
    }
    if(num_active <= 0) {myfree(idx_host); return 0;}
    double t_grv_active_list = my_second();

    /* Acquire the arena (P_dev + CellP_dev in SharedSpace) */
    gpu_particles_arena_set_site("gpu_gravtree_walk_primary");
    gpu_particles_arena_acquire(NumPart, P, CellP);
    struct particle_data    *P_dev    = gpu_particles_arena_P();
    struct gas_cell_data    *CellP_dev = gpu_particles_arena_CellP();
    double t_grv_arena = my_second();

    int min_nodes = MaxNodes + 1;
    gpu_gravity_tree_acquire(min_nodes, Nodes_base, Extnodes_base);
    /* soa->nextnode_aux aliases UVM Nextnode[] (set by
     * force_treeallocate); no per-walk memcpy needed. */
    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    /* CellP is legitimately NULL on a gas-free (DM-only) problem (TotN_gas==0).
     * Every CellP_dev use in this walk + post-walk scatter is gas-gated
     * (device: valid_gas_particle_for_rt / ptype==0; host scatter: Type==0),
     * so a null CellP_dev is safe there. Only require it when gas exists. */
    const bool need_cellp = (All.TotN_gas > 0);   /* host read; bare All.* is safe in GPU-TU host code (all-mirror) */
    if(!P_dev || !soa || (need_cellp && !CellP_dev)) {
        printf("gpu_gravtree_walk_primary: failed to acquire arena or tree SoA\n");
        endrun(913200);
        myfree(idx_host);   /* LIFO mymalloc cleanup before drain */
        return 1;
    }

    /* Per-particle gravity source inputs (RT luminosity / sink bolometric luminosity /
     * CR injection), evaluated via the shared SSOT helper gravtree_fill_particle_source_
     * inputs(). Two modes, same physics:
     *   EAGER: one host pass over all NumPart fills dense SharedSpace arrays the kernel
     *          reads by particle id.
     *   LAZY:  no dense arrays and no O(NumPart) pass -- the kernel evaluates the same
     *          helper on-device at each local particle-open, so the cost scales with the
     *          active set rather than NumPart.
     * Selected by the active-count threshold below; the result is identical either way. */
    bool use_lazy_source = false;
#if defined(GRAVTREE_SOURCE_LAZY_SUPPORTED) && (defined(RT_USE_GRAVTREE) || defined(SINK_PHOTONMOMENTUM) || defined(COSMIC_RAY_SUBGRID_LEBRON))
    /* Use lazy when the step touches few particles -- either an absolute count or a small
     * fraction of the local pool. Hard-coded (no env var, no parameter). */
    {
        const long   GRAVTREE_SOURCE_LAZY_CAP  = 256;
        const double GRAVTREE_SOURCE_LAZY_FRAC = 0.01;
        use_lazy_source = ((long)num_active <= GRAVTREE_SOURCE_LAZY_CAP) ||
                          ((double)num_active < GRAVTREE_SOURCE_LAZY_FRAC * (double)NumPart);
    }
#endif

#ifdef RT_USE_GRAVTREE
    MyFloat *d_src_lum = NULL;
#ifdef CHIMES_STELLAR_FLUXES
    double *d_src_lum_G0 = NULL, *d_src_lum_ion = NULL;
#endif
    if(!use_lazy_source) {
        long sz = (long)NumPart * N_RT_FREQ_BINS * sizeof(MyFloat);
        d_src_lum = (MyFloat *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sz);
        if(!d_src_lum) {printf("gpu_gravtree_walk_primary: d_src_lum alloc failed\n"); endrun(913202); myfree(idx_host); return 1;}
        memset(d_src_lum, 0, sz);
#ifdef CHIMES_STELLAR_FLUXES
        long szc = (long)NumPart * CHIMES_LOCAL_UV_NBINS * sizeof(double);
        d_src_lum_G0  = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(szc);
        d_src_lum_ion = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(szc);
        if(!d_src_lum_G0 || !d_src_lum_ion) {printf("gpu_gravtree_walk_primary: CHIMES lum alloc failed\n"); endrun(913203); myfree(idx_host); return 1;}
        memset(d_src_lum_G0,  0, szc);
        memset(d_src_lum_ion, 0, szc);
#endif
    }
#endif /* RT_USE_GRAVTREE */

#ifdef SINK_PHOTONMOMENTUM
    MyFloat       *d_bh_lum   = NULL;
    Vec3<MyFloat> *d_bh_angle = NULL;
    if(!use_lazy_source) {
        long sz_lum  = (long)NumPart * sizeof(MyFloat);
        long sz_ang  = (long)NumPart * sizeof(Vec3<MyFloat>);
        d_bh_lum   = (MyFloat *)       Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sz_lum);
        d_bh_angle = (Vec3<MyFloat> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sz_ang);
        if(!d_bh_lum || !d_bh_angle) {printf("gpu_gravtree_walk_primary: bh_lum alloc failed\n"); endrun(913210); myfree(idx_host); return 1;}
        memset(d_bh_lum,   0, sz_lum);
        memset(d_bh_angle, 0, sz_ang);
    }
#endif /* SINK_PHOTONMOMENTUM */

#ifdef COSMIC_RAY_SUBGRID_LEBRON
    MyFloat *d_cr_inject = NULL;
    /* per-step CR-age scalar: needed by the walk's CR gate in BOTH modes (cr_active_gate
     * + grav_cr_lebron_accumulate), so it is computed unconditionally, NOT inside the
     * eager-only dense-array block. Lazy skips only the d_cr_inject ARRAY. */
    double   t_max_cr    = 0.0;
    if(All.Time > All.TimeBegin) {
        double t_gyr = evaluate_time_since_t_initial_in_Gyr(All.TimeBegin);
        if(t_gyr > 1.0) {t_gyr = 1.0;}
        t_max_cr = t_gyr / UNIT_TIME_IN_GYR;     /* per-step scalar; computed once, not per-particle */
    }
    if(!use_lazy_source) {
        long sz = (long)NumPart * sizeof(MyFloat);
        d_cr_inject = (MyFloat *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(sz);
        if(!d_cr_inject) {printf("gpu_gravtree_walk_primary: cr_inject alloc failed\n"); endrun(913211); myfree(idx_host); return 1;}
        memset(d_cr_inject, 0, sz);
    }
#endif /* COSMIC_RAY_SUBGRID_LEBRON */

    /* EAGER only: single host pass over all NumPart, gated physics in the shared SSOT
     * helper, copying ONLY active entries into the bulk-zeroed SharedSpace arrays. In
     * LAZY mode this whole O(NumPart) pass is skipped -- the kernel evaluates the same
     * helper on-device at each local particle-open instead. */
#if defined(RT_USE_GRAVTREE) || defined(SINK_PHOTONMOMENTUM) || defined(COSMIC_RAY_SUBGRID_LEBRON)
    if(!use_lazy_source)
    for(int p = 0; p < NumPart; p++) {
        struct gravtree_source_inputs_t in;
        gravtree_fill_particle_source_inputs(p, P, CellP, &in);
#ifdef RT_USE_GRAVTREE
        if(in.rt_active) {
            int kf;
            for(kf = 0; kf < N_RT_FREQ_BINS; kf++) {d_src_lum[(long)p * N_RT_FREQ_BINS + kf] = in.src_lum[kf];}
#ifdef CHIMES_STELLAR_FLUXES
            for(kf = 0; kf < CHIMES_LOCAL_UV_NBINS; kf++) {
                d_src_lum_G0[(long)p * CHIMES_LOCAL_UV_NBINS + kf]  = in.src_lum_G0[kf];
                d_src_lum_ion[(long)p * CHIMES_LOCAL_UV_NBINS + kf] = in.src_lum_ion[kf];
            }
#endif
        }
#endif
#ifdef SINK_PHOTONMOMENTUM
        if(in.bh_active) {
            d_bh_lum[p]   = in.bh_lum;
            d_bh_angle[p] = in.bh_angle;
        }
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
        if(in.cr_inject != 0) {d_cr_inject[p] = in.cr_inject;}
#endif
    }
#endif

#ifdef RT_USE_GRAVTREE
    struct gpu_rt_walk_data_t rt_data_snap;
    rt_data_snap.src_lum = d_src_lum;
#ifdef CHIMES_STELLAR_FLUXES
    rt_data_snap.src_lum_G0  = d_src_lum_G0;
    rt_data_snap.src_lum_ion = d_src_lum_ion;
#endif
#endif /* RT_USE_GRAVTREE */
#ifdef SINK_PHOTONMOMENTUM
    struct gpu_sink_walk_data_t sink_data_snap;
    sink_data_snap.bh_lum   = d_bh_lum;
    sink_data_snap.bh_angle = d_bh_angle;
#endif /* SINK_PHOTONMOMENTUM */
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    struct gpu_cr_walk_data_t cr_data_snap;
    cr_data_snap.cr_inject = d_cr_inject;
    cr_data_snap.t_max_cr  = t_max_cr;
#endif /* COSMIC_RAY_SUBGRID_LEBRON */

    double t_grv_precompute = my_second();
    /* Scratch arrays for per-target results */
    int *d_idx    = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(int));
    int *d_failed = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(int));
    Vec3<double> *d_acc = (Vec3<double> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(Vec3<double>));
    int *d_ninter = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(int));
    double *d_pot = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(double));
    int *d_foreign = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(int));  /* diagnostic */
    if(!d_idx || !d_failed || !d_acc || !d_ninter || !d_pot || !d_foreign) {
        printf("gpu_gravtree_walk_primary: kokkos_malloc failed\n");
        endrun(913201);
        myfree(idx_host);   /* LIFO mymalloc cleanup before drain */
        return 1;
    }
    memcpy(d_idx, idx_host, num_active * sizeof(int));
    memset(d_failed, 0, num_active * sizeof(int));

    int maxPart = All.MaxPart;
    int maxNodes_snap = MaxNodes;
    int maxForeignNodes_snap = MaxForeignNodes;    /* LET */
    const struct gpu_gravity_tree_soa_t soa_snap = *soa;
#ifdef GRAVITY_HYBRID_OPENING_CRIT
    /* host-evaluate the first-step predicate once; captured by value into the device walk */
    int is_first_step_snap = (All.Ti_Current == 0 && RestartFlag != 1);
#endif

#ifdef PMGRID
    double rcut_snap     = All.Rcut[0];
    double rcut2_snap    = rcut_snap * rcut_snap;
    double asmthfac_snap = 0.5 / All.Asmth[0] * (GIZMO_GPU_GRAVTREE_NTAB / 3.0);
    /* shortrange_table is a host global (forcetree.cc); copy to SharedSpace so
     * the CUDA kernel can read it from device code. */
    float *d_st  = (float *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(GIZMO_GPU_GRAVTREE_NTAB * sizeof(float));
    float *d_sp  = (float *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(GIZMO_GPU_GRAVTREE_NTAB * sizeof(float));
    if(!d_st || !d_sp) {printf("gpu_gravtree_walk_primary: shortrange table alloc failed\n"); endrun(913205); myfree(idx_host); return 1;}
    memcpy(d_st, shortrange_table,           GIZMO_GPU_GRAVTREE_NTAB * sizeof(float));
    memcpy(d_sp, shortrange_table_potential, GIZMO_GPU_GRAVTREE_NTAB * sizeof(float));
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
    float *d_stid = (float *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(GIZMO_GPU_GRAVTREE_NTAB * sizeof(float));
    if(!d_stid) {printf("gpu_gravtree_walk_primary: shortrange tidal table alloc failed\n"); endrun(913206); myfree(idx_host); return 1;}
    memcpy(d_stid, shortrange_table_tidal, GIZMO_GPU_GRAVTREE_NTAB * sizeof(float));
#endif
#endif
    /* read-only PM short-range config captured by value into the device walk (empty when
     * !PMGRID; per-target PLACEHIGHRESREGION override happens inside the walk on its copy). */
    grav_pm_shortrange_t pm_snap{};
#ifdef PMGRID
    pm_snap.rcut = rcut_snap; pm_snap.rcut2 = rcut2_snap; pm_snap.asmthfac = asmthfac_snap;
    pm_snap.shortrange_tab = d_st;
#ifdef EVALPOTENTIAL
    pm_snap.shortrange_pot_tab = d_sp;
#endif
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
    pm_snap.shortrange_tidal_tab = d_stid;
#endif
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

    /* Ewald periodic-image potential correction (pure-tree periodic + EVALPOTENTIAL).
     * Acquire the table once (idempotent). This build requires the correction, so a
     * missing table is a hard stop that aborts the primary walk -- never a silent
     * skip of the term. */
    struct gpu_ewald_pot_data_t ewald_pot_snap;
    ewald_pot_snap.potcorr = NULL; ewald_pot_snap.fac_intp = 0.0; ewald_pot_snap.active = 0;
#ifdef GIZMO_GPU_EWALD_POT_CORRECTION
    if(gpu_ewald_acquire_pot_data(&ewald_pot_snap) != 0) {
        printf("gpu_gravtree_walk_primary: Ewald potential-correction table unavailable; EVALPOTENTIAL periodic build requires it\n");
        endrun(913212);
        myfree(idx_host);   /* LIFO mymalloc cleanup; do not launch the walk with the term disabled */
        return 1;
    }
#endif
    const struct gpu_ewald_pot_data_t ewald_pot_dev = ewald_pot_snap;

    /* Invariant guard: reset the per-walk counter. */
    g_inv_fterm_aggregate = 0;

    double t_grv_pre_kernel = my_second();
    Kokkos::parallel_for("gravtree_walk_primary", num_active, KOKKOS_LAMBDA(int a) {
        int target = d_idx[a];
        Vec3<double> acc;
        int ninter;
        double pot;
        int nforeign;
        int ok = gpu_gravtree_walk_one(target, maxPart, maxNodes_snap, maxForeignNodes_snap,
                                        P_dev, CellP_dev, &soa_snap,
#ifdef GRAVITY_HYBRID_OPENING_CRIT
                                        is_first_step_snap,
#endif
                                        pm_snap,
#ifdef RT_USE_GRAVTREE
                                        &rt_data_dev,
#endif
#ifdef SINK_PHOTONMOMENTUM
                                        &sink_data_dev,
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
                                        &cr_data_dev,
#endif
                                        use_lazy_source,
                                        &ewald_pot_dev,
                                        acc, ninter, pot, nforeign);
        if(ok) {
            d_acc[a] = acc;
            d_ninter[a] = ninter;
            d_pot[a] = pot;
            d_foreign[a] = nforeign;
            d_failed[a] = 0;
        } else {
            d_failed[a] = 1;
        }
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("gravtree_walk_primary", num_active);
    /* Permanent invariant guard.  An untagged predicate-OPEN foreign terminal is an unopenable
     * aggregate that would silently downgrade leaf-sensitive physics to a multipole; until an
     * owner-continuation path exists this hard-surfaces as a controlled stop.
     *
     * The CLEAN case is reported too, not only the violation: a run that confirms the count is
     * zero is how the foreign-leaf import path is shown to be behaving, and that confirmation is
     * unavailable if only the failure prints. */
    if(ThisTask == 0 && gizmo_verbose_diag()) {
        printf("[gravtree invariant] unopenable foreign-terminal aggregates accepted = %lld\n",
               g_inv_fterm_aggregate);
        fflush(stdout);
    }
    if(g_inv_fterm_aggregate > 0) {
        printf("[GRAV-INVARIANT VIOLATION rank=%d] %lld predicate-OPEN foreign-terminal nodes accepted "
               "as multipoles but NOT tagged real leaves (unopenable aggregates in leaf-sensitive "
               "support) -- silently downgrades adaptive-softening/zeta physics; C2/owner-continuation "
               "required. Stopping.\n", ThisTask, g_inv_fterm_aggregate);
        fflush(stdout);
        endrun(90000087);
    }
    double t_grv_post_kernel = my_second();

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
#ifdef SINK_SEED_FROM_LOCALGAS_TOTALMENCCRITERIA
            P[i].MencInRcrit = P_dev[i].MencInRcrit;
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

            /* Tidal tensor + GravJerk scatter-back (ATFU/jerk path). */
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
            P[i].tidal_tensorps = P_dev[i].tidal_tensorps;
#endif
#ifdef COMPUTE_JERK_IN_GRAVTREE
            P[i].GravJerk = P_dev[i].GravJerk;
#endif
#ifdef COUNT_MASS_IN_GRAVTREE
            /* Direct assignment: GPU walk visited the entire tree for this target
             * (no MPI export/import partition).  The post-loop += P[i].Mass at
             * gravtree.cc:605 then adds the target's own mass for the diagnostic. */
            P[i].TreeMass = P_dev[i].TreeMass;
#endif
#ifdef ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION
            P[i].tidal_zeta = P_dev[i].tidal_zeta;
#endif
#ifdef SPECIAL_POINT_MOTION
            P[i].vel_of_nearest_special = P_dev[i].vel_of_nearest_special;
            P[i].acc_of_nearest_special = P_dev[i].acc_of_nearest_special;
#ifdef SPECIAL_POINT_WEIGHTED_MOTION
            P[i].weight_sum_for_special_point_smoothing = P_dev[i].weight_sum_for_special_point_smoothing;
#endif
#endif

            /* Sink-distance / single-star timestepping scatter-back */
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
            if(TakeLevel >= 0) {P[i].GravCost[TakeLevel] = d_ninter[a];}

            /* No arena mirror-update here: under UVM-canonical
             * P_dev = arena_P aliases host P[], so the
             * struct copy P_dev[i] = P[i] would be self-assignment. */

            nsucceeded++;
        }
    }
    Costtotal += costtotal_added;

    /* Diagnostic: print GPU walk summary + first 10 particles for LET vs no-LET comparison */
    if(ThisTask == 0 && gizmo_verbose_diag()) {
        long long tot_foreign = 0;
        for(int a = 0; a < num_active; a++) { if(!d_failed[a]) tot_foreign += d_foreign[a]; }
        printf("GPU_WALK_SUMMARY[t=0 LET=%d]: nsucceeded=%d/%d total_ninter=%.0f total_foreign=%lld avg_ninter=%.1f maxForeignNodes=%d Numforeignnodes=%d\n",
               (maxForeignNodes_snap > 0) ? 1 : 0, nsucceeded, num_active, costtotal_added,
               tot_foreign, (nsucceeded > 0) ? costtotal_added / nsucceeded : 0.0,
               maxForeignNodes_snap, Numforeignnodes);
        int nprinted = 0;
        for(int a = 0; a < num_active && nprinted < 10; a++) {
            int i = d_idx[a];
            if(!d_failed[a]) {
                Vec3<double> av = d_acc[a];
                double amag = sqrt(av[0]*av[0] + av[1]*av[1] + av[2]*av[2]);
                printf("GPU_WALK_PART[t=0 a=%d ID=%llu]: |acc|=%.8g acc=(%.6g,%.6g,%.6g) ninter=%d foreign=%d\n",
                       i, (unsigned long long)P[i].ID, amag, av[0], av[1], av[2], d_ninter[a], d_foreign[a]);
                nprinted++;
            }
        }
        fflush(stdout);
    }

    /* mark_clean (not invalidate): the per-active-i
     * P_dev[i]=P[i] mirror in the scatter loop above keeps arena coherent
     * for the touched indices; untouched i's were unchanged from acquire-time
     * (kernel output went to d_* buffers, not arena). Verified safe with
     * GIZMO_GPU_ARENA_DEBUG=1. */
    gpu_particles_arena_mark_clean_after_scatter("gpu_gravtree_walk_primary");
    /* No SoA invalidate: the next force_treebuild fully repopulates the SoA
     * via the build pipeline; the next pre-walk drift mutates only the
     * stale-Ti_current nodes via gpu_force_drift_nodes (UVM AoS + SoA in one
     * kernel).  No host-side reseed scaffolding remains. */

    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_foreign);
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
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_stid);
#endif
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_sp);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_st);
#endif

    myfree(idx_host);

    /* Sub-bucket timing — env-gated; no-op when GIZMO_VERBOSE_DIAG off.
     * grav_tree_walk cost can look suspiciously similar to other
     * full-NumPart taxes; this breakdown isolates which is to blame. */
    {
        double t_grv_done = my_second();
        gizmo_step_phase_record("grav_move_particles",  timediff(t_grv_start,        t_grv_mp));
        gizmo_step_phase_record("grav_drift_nodes",     timediff(t_grv_mp,           t_grv_drift_nodes));
        gizmo_step_phase_record("grav_active_list",     timediff(t_grv_drift_nodes,  t_grv_active_list));
        gizmo_step_phase_record("grav_arena_acquire",   timediff(t_grv_active_list,  t_grv_arena));
        gizmo_step_phase_record("grav_full_precompute", timediff(t_grv_arena,        t_grv_precompute));
        gizmo_step_phase_record("grav_scratch_alloc",   timediff(t_grv_precompute,   t_grv_pre_kernel));
        gizmo_step_phase_record("grav_kernel",          timediff(t_grv_pre_kernel,   t_grv_post_kernel));
        gizmo_step_phase_record("grav_postwalk",        timediff(t_grv_post_kernel,  t_grv_done));
        /* Interaction counter from inside the function */
        gizmo_step_phase_record("grav_num_active_dbl",  (double)num_active);
    }

    return nsucceeded;
}



/* ========================================================================= *
 * GPU Ewald-correction walk (pure-tree periodic gravity).                   *
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

static int gpu_ewald_tables_acquire(void)
{
    if(g_ewald_tables_ready) return 0;
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
        return 1;   /* soft bad-stop: caller skips the Ewald walk on not-ready tables */
    }
    memcpy(g_d_fcorrx,  fx, sz);
    memcpy(g_d_fcorry,  fy, sz);
    memcpy(g_d_fcorrz,  fz, sz);
    memcpy(g_d_potcorr, fp, sz);
    g_ewald_fac_intp = fi;
    g_ewald_tables_ready = 1;
    return 0;
}

#ifdef GIZMO_GPU_EWALD_POT_CORRECTION
static int gpu_ewald_acquire_pot_data(struct gpu_ewald_pot_data_t *out)
{
    out->potcorr = NULL; out->fac_intp = 0.0; out->active = 0;
    if(gpu_ewald_tables_acquire() != 0) return 1;   /* table acquire failed; caller hard-stops */
    out->potcorr  = g_d_potcorr;
    out->fac_intp = g_ewald_fac_intp;
    out->active   = 1;
    return 0;
}
#endif

/* Device-side Ewald walk for a single target. Returns 1 on success (acc
 * written), 0 if a pseudo-particle was encountered (defer to CPU). */
static KOKKOS_INLINE_FUNCTION int
gpu_ewald_walk_one(int target,
                   int maxPart, int maxNodes, int maxForeignNodes,    /* LET */
                   struct particle_data *P_dev,
                   const struct gpu_gravity_tree_soa_t *tree_soa,
#ifdef GRAVITY_HYBRID_OPENING_CRIT
                   int is_first_step,   /* hybrid opening: relative criterion applies only after step 0 */
#endif
                   const MyFloat *fcorrx, const MyFloat *fcorry, const MyFloat *fcorrz,
                   double fac_intp, double boxsize, double boxhalf,
                   double errtoltheta, double errtolforceacc,
                   Vec3<double> &acc_out)
{
    Vec3<double> pos = P_dev[target].Pos;
    double aold = errtolforceacc * P_dev[target].OldAcc;
    Vec3<double> acc = {0.0, 0.0, 0.0};

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
        else if(no >= maxPart + maxNodes + maxForeignNodes) /* pseudo-particle — defer to CPU (foreign-node range below) */
        {
            return 0;
        }
        else /* internal node */
        {
            idx = no - maxPart;
            /* skip single-particle node (open it to its daughter chain) */
            if(!(tree_soa->bitflags[idx] & (1 << BITFLAG_MULTIPLEPARTICLES))) {
                no = tree_soa->nextnode[idx];
                continue;
            }
            mass  = tree_soa->mass[idx];
            dr[0] = tree_soa->s[idx][0] - pos[0];
            dr[1] = tree_soa->s[idx][1] - pos[1];
            dr[2] = tree_soa->s[idx][2] - pos[2];
        }

        /* nearest-image wrap on the displacement (shared SSOT helper) */
        gravity_box_nearest_image(dr[0], dr[1], dr[2], -1);

        if(is_leaf) {
            no = tree_soa->nextnode_aux[no];
        } else {
            /* Opening check + periodic-boundary skip (mirrors forcetree.cc:2769-2842) */
            double r2  = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
            if(r2 <= 0) r2 = 1e-300;
            double len = tree_soa->len[idx];
            int openflag = 0;
            if(errtoltheta) {
                if(len * len > r2 * errtoltheta * errtoltheta) openflag = 1;
            }
#ifndef GRAVITY_HYBRID_OPENING_CRIT
            else {
#else
            /* hybrid: relative criterion only after step 0 (mirrors forcetree.cc:3489-3493) */
            if(!is_first_step) {
#endif
                if(mass * len * len > r2 * r2 * aold) {
                    openflag = 1;
                } else {
                    double ad0 = tree_soa->center[idx][0] - pos[0], ad1 = tree_soa->center[idx][1] - pos[1], ad2 = tree_soa->center[idx][2] - pos[2];
                    double adx = gravity_box_long_abs_x(ad0, ad1, ad2, -1);
                    double ady = gravity_box_long_abs_y(ad0, ad1, ad2, -1);
                    double adz = gravity_box_long_abs_z(ad0, ad1, ad2, -1);
                    if(adx < 0.60*len && ady < 0.60*len && adz < 0.60*len) openflag = 1;
                }
            }
            if(openflag) {
                /* short-cut: if the node is entirely on one side of the periodic
                 * boundary along any axis, we can safely skip it without opening */
                double ux = tree_soa->center[idx][0] - pos[0]; if(ux >  boxhalf) ux -= boxsize; else if(ux < -boxhalf) ux += boxsize;
                if(((ux < 0) ? -ux : ux) > 0.5*(boxsize - len)) { no = tree_soa->nextnode[idx]; continue; }
                double uy = tree_soa->center[idx][1] - pos[1]; if(uy >  boxhalf) uy -= boxsize; else if(uy < -boxhalf) uy += boxsize;
                if(((uy < 0) ? -uy : uy) > 0.5*(boxsize - len)) { no = tree_soa->nextnode[idx]; continue; }
                double uz = tree_soa->center[idx][2] - pos[2]; if(uz >  boxhalf) uz -= boxsize; else if(uz < -boxhalf) uz += boxsize;
                if(((uz < 0) ? -uz : uz) > 0.5*(boxsize - len)) { no = tree_soa->nextnode[idx]; continue; }
                /* cell too large → must refine */
                if(len > 0.20 * boxsize) { no = tree_soa->nextnode[idx]; continue; }
            }
            no = tree_soa->sibling[idx];
        }

        /* Trilinear interp of the Ewald force octant tables via the shared SSOT helper
         * (gravtree_ewald.h): weights from |dr| once, applied to all three tables; the
         * odd-force per-component signs stay here. */
        double signx = (dr[0] < 0) ? +1.0 : -1.0;
        double signy = (dr[1] < 0) ? +1.0 : -1.0;
        double signz = (dr[2] < 0) ? +1.0 : -1.0;
        grav_ewald_interp_weights ew = grav_ewald_interp_setup(dr[0], dr[1], dr[2], fac_intp);
        acc[0] += mass * signx * grav_ewald_interp_apply(fcorrx, ew);
        acc[1] += mass * signy * grav_ewald_interp_apply(fcorry, ew);
        acc[2] += mass * signz * grav_ewald_interp_apply(fcorrz, ew);
    }

    acc_out = acc;
    return 1;
}

extern "C" int gpu_ewald_walk_primary(void)
{
    GIZMO_GPU_ENSURE_ALL_FRESH();
    if(Ewald_iter == 0) {return 0;}
#ifdef PMGRID
    return 0; /* Ewald walk not needed under TreePM (gravtree.cc:734 gate) */
#else
    int num_active_total = (int) ActiveParticleList.size();
    if(num_active_total <= 0) {return 0;}

    /* Particles have already been drifted by the primary walk (Ewald_iter==0);
     * the tree SoA mirror is still valid. Just re-acquire (cache-hit).
     * soa->nextnode_aux aliases UVM Nextnode[]; no per-walk memcpy. */
    int min_nodes = MaxNodes + 1;
    gpu_gravity_tree_acquire(min_nodes, Nodes_base, Extnodes_base);
    const struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa) {return 0;}

    gpu_particles_arena_set_site("gpu_gravtree_walk_ewald");
    gpu_particles_arena_acquire(NumPart, P, CellP);
    struct particle_data *P_dev = gpu_particles_arena_P();
    if(!P_dev) {return 0;}

    if(gpu_ewald_tables_acquire() != 0) {return 1;}   /* soft bad-stop: tables not ready, skip walk (idx_host not yet alloc'd); drains at next poll */

    int *idx_host = (int *) mymalloc("gpu_ewald_idx", num_active_total * sizeof(int));
    int num_active = 0;
    for(int a = 0; a < num_active_total; a++) {
        int i = ActiveParticleList[a];
        if(ProcessedFlag[i]) continue;
        /* SSOT pre-walk candidacy (Mass>0 + Hermite eligibility + needs_new_treeforce):
         * parity with the CPU primary loop + finalization (see primary walk). */
        if(!gravity_treewalk_candidate_prewalk(i)) continue;
        idx_host[num_active++] = i;
    }
    if(num_active == 0) { myfree(idx_host); return 0; }

    int *d_idx    = (int *)          Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(int));
    int *d_failed = (int *)          Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(int));
    Vec3<double> *d_acc = (Vec3<double> *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_active * sizeof(Vec3<double>));
    if(!d_idx || !d_failed || !d_acc) {printf("gpu_ewald_walk_primary: kokkos_malloc failed\n"); endrun(914102); myfree(idx_host); return 1;}
    memcpy(d_idx, idx_host, num_active * sizeof(int));
    memset(d_failed, 0, num_active * sizeof(int));

    /* snapshot scalars */
    const int maxPart            = All.MaxPart;
    const int maxNodes_snap      = MaxNodes;
    const int maxForeignNodes_sn = MaxForeignNodes;    /* LET */
    const double boxsize     = All.BoxSize;
    const double boxhalf     = 0.5 * All.BoxSize;
    const double fac_intp    = g_ewald_fac_intp;
    const double errtoltheta = All.ErrTolTheta;
    const double errtolforceacc = All.ErrTolForceAcc;
    const MyFloat *fcorrx_dev = g_d_fcorrx;
    const MyFloat *fcorry_dev = g_d_fcorry;
    const MyFloat *fcorrz_dev = g_d_fcorrz;
    const struct gpu_gravity_tree_soa_t soa_snap = *soa;
#ifdef GRAVITY_HYBRID_OPENING_CRIT
    /* host-evaluate the first-step predicate once; captured by value into the device walk */
    const int is_first_step_snap = (All.Ti_Current == 0 && RestartFlag != 1);
#endif

    Kokkos::parallel_for("gpu_ewald_walk_primary", num_active, KOKKOS_LAMBDA(int a) {
        int target = d_idx[a];
        Vec3<double> acc;
        int ok = gpu_ewald_walk_one(target, maxPart, maxNodes_snap, maxForeignNodes_sn,
                                     P_dev, &soa_snap,
#ifdef GRAVITY_HYBRID_OPENING_CRIT
                                     is_first_step_snap,
#endif
                                     fcorrx_dev, fcorry_dev, fcorrz_dev,
                                     fac_intp, boxsize, boxhalf,
                                     errtoltheta, errtolforceacc,
                                     acc);
        if(ok) {d_acc[a] = acc; d_failed[a] = 0;}
        else   {d_failed[a] = 1;}
    });
    Kokkos::fence();
    gizmo_gpu_check_last_error("gpu_ewald_walk_primary", num_active);

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


