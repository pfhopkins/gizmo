/* gpu_gravtree.cc — Step 13 Phase 4 Tier 1a
 *
 * Core gravity walk on GPU, no optional payloads. See gpu_gravtree.h for
 * architectural notes (speculative GPU + CPU fallback on pseudo-particle).
 *
 * Walk structure extracted verbatim from force_treeevaluate() (mode=0 path)
 * in forcetree.cc:1435-2603, stripped of all #ifdef payload branches. When
 * a thread encounters a pseudo-particle (no >= MaxPart + MaxNodes), it sets
 * failed[i]=1 and exits without writing P[i].GravAccel; the host leaves
 * ProcessedFlag[i] unset so the existing CPU primary loop + MPI export
 * machinery handles that particle unchanged.
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

#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL)
/* GPU-callable mirror of ForceSoftening_KernelRadius() that takes a
 * particle_data pointer (so we can use P_dev in the kernel without touching
 * the global P[]).  Tier 1b scope: only the FORGAS / FORALL branches are
 * implemented — the GALSF/SINGLE_STAR/MAX_SOFT_HARD_LIMIT/TIDAL paths are
 * gated off above and will land with later tiers. */
static KOKKOS_INLINE_FUNCTION
double gpu_force_softening_kernel_radius(const struct particle_data *Pp, int p)
{
#if defined(ADAPTIVE_GRAVSOFT_FORALL)
    if((1 << Pp[p].Type) & (ADAPTIVE_GRAVSOFT_FORALL)) {return Pp[p].AGS_KernelRadius;}
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS)
    if(Pp[p].Type == 0) {return Pp[p].KernelRadius;}
#endif
    return All.ForceSoftening[Pp[p].Type];
}

/* Source-particle AGS_zeta access — exists with FORGAS or FORALL.  When
 * neither flag enables AGS for the type, the field still exists but the
 * caller will gate the use via add_ags_zeta_terms_secondary. */
static KOKKOS_INLINE_FUNCTION
double gpu_get_ags_zeta(const struct particle_data *Pp, int p)
{
    return Pp[p].AGS_zeta;
}
#endif

#if defined(ADAPTIVE_GRAVSOFT_FORALL)
/* Forward decl of the (file-scope, non-static) helper in ags_rkern.cc.
 * On Kokkos OMP this resolves at link time and the kernel runs on host
 * threads, so direct call is fine.  CUDA offload would require either
 * porting to a header-defined inline or a __device__ duplicate; flagged
 * for a Phase 4 follow-up. */
extern int ags_gravity_kernel_shared_BITFLAG(short int particle_type_primary);
#endif

/* Compile-time gating: Tier 1a supports ONLY the core walk. Any of these
 * payloads requires the walk kernel to compute extra terms that haven't been
 * ported yet — they land in Tier 1b, 2, or 3. Gating here keeps the kernel
 * code readable and prevents silent wrong-physics on configs we haven't
 * validated. */
/* PMGRID: short-range cutoff (rcut2) + tabulated short-range factor are
 * supported in Tier 1b.  shortrange_table / shortrange_table_potential are
 * declared extern below; on Kokkos OMP they live in host memory and are
 * directly accessible from the kernel.  True device offload (CUDA) will
 * require mirroring the tables — flagged for a Phase 4 follow-up. */
/* ADAPTIVE_GRAVSOFT_FORGAS is supported in Tier 1b.3.  The walk kernel
 * also has the FORALL code paths in place, BUT precompiler_logic.h:152-156
 * auto-defines ADAPTIVE_GRAVSOFT_SYMMETRIZE_FORCE_BY_AVERAGING whenever
 * FORALL is set, and SYMMETRIZE is not yet ported (defers to Tier 1c).
 * So the FORALL branch is unreachable in practice — the SYMMETRIZE gate
 * below catches it.  ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION pairs with
 * COMPUTE_TIDAL_TENSOR_IN_GRAVTREE (Tier 3) and is gated separately. */
#if defined(ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION)
#error "GIZMO_GPU_GRAVTREE does not yet support ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION (lands with Tier 3 tidal tensor)."
#endif
#if defined(ADAPTIVE_GRAVSOFT_SYMMETRIZE_FORCE_BY_AVERAGING)
#error "GIZMO_GPU_GRAVTREE does not yet support ADAPTIVE_GRAVSOFT_SYMMETRIZE_FORCE_BY_AVERAGING (auto-defined by ADAPTIVE_GRAVSOFT_FORALL — port in Tier 1c)."
#endif
#if defined(GALSF_MERGER_STARCLUSTER_PARTICLES) || defined(ADAPTIVE_GRAVSOFT_MAX_SOFT_HARD_LIMIT) || defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM)
#error "GIZMO_GPU_GRAVTREE does not yet support type-specific softening overrides used by ForceSoftening_KernelRadius (defer)."
#endif
/* OUTPUT_POTENTIAL alone (without EVALPOTENTIAL) routes through compute_potential()
 * in core/run.cc, which is a separate CPU-only walk and therefore unaffected by
 * GIZMO_GPU_GRAVTREE. EVALPOTENTIAL adds inline-with-the-walk potential
 * accumulation, ported in this tier. */
#if defined(COMPUTE_TIDAL_TENSOR_IN_GRAVTREE) || defined(COMPUTE_JERK_IN_GRAVTREE)
#error "GIZMO_GPU_GRAVTREE Tier 1a does not support TIDAL_TENSOR / JERK in gravtree (Tier 3)."
#endif
#if defined(RT_USE_GRAVTREE) || defined(RT_USE_TREECOL_FOR_NH)
#error "GIZMO_GPU_GRAVTREE Tier 1a does not support RT payloads in gravtree (Tier 2)."
#endif
#if defined(SINK_CALC_DISTANCES) || defined(SINK_PHOTONMOMENTUM) || defined(SINK_DYNFRICTION_FROMTREE) || defined(SINK_COMPTON_HEATING)
#error "GIZMO_GPU_GRAVTREE Tier 1a does not support SINK_* payloads in gravtree (Tier 2)."
#endif
#if defined(SINGLE_STAR_STARFORGE_DEFAULTS) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_TIMESTEPPING) || defined(SINGLE_STAR_FIND_BINARIES) || defined(SINGLE_STAR_FB_TIMESTEPLIMIT)
#error "GIZMO_GPU_GRAVTREE Tier 1a does not support SINGLE_STAR_* payloads (Tier 2)."
#endif
#if defined(COSMIC_RAY_SUBGRID_LEBRON) || defined(GALSF_FB_FIRE_RT_LONGRANGE) || defined(CHIMES_STELLAR_FLUXES)
#error "GIZMO_GPU_GRAVTREE Tier 1a does not support CR/FIRE/CHIMES payloads (Tier 2/3)."
#endif
#if defined(DM_SCALARFIELD_SCREENING) || defined(GRAVITY_SPHERICAL_SYMMETRY) || defined(COUNT_MASS_IN_GRAVTREE) || defined(GRAVTREE_CALCULATE_GAS_MASS_IN_NODE)
#error "GIZMO_GPU_GRAVTREE Tier 1a does not support DM/spherical/mass-count payloads (Tier 3)."
#endif
#if defined(HERMITE_INTEGRATION) || defined(ADAPTIVE_TREEFORCE_UPDATE) || defined(NEIGHBORS_MUST_BE_COMPUTED_EXPLICITLY_IN_FORCETREE)
#error "GIZMO_GPU_GRAVTREE Tier 1a does not support HERMITE / ATFU / NEIGHBORS_MUST_BE_COMPUTED (Phase 5 / Tier 3)."
#endif
#if defined(BOX_PERIODIC) && !defined(GRAVITY_NOT_PERIODIC)
#error "GIZMO_GPU_GRAVTREE Tier 1a requires non-periodic gravity (BOX_PERIODIC + GRAVITY_NOT_PERIODIC, or no BOX_PERIODIC). Periodic Ewald walk ports in a later tier."
#endif
#if defined(SELFGRAVITY_OFF)
#error "GIZMO_GPU_GRAVTREE requires self-gravity to be enabled (SELFGRAVITY_OFF disables the walk entirely)."
#endif


/* Device-side walk for a single target particle. Returns 1 on success
 * (acc written), 0 on failure (hit pseudo-particle — host must run CPU walk
 * for this target). Closely mirrors force_treeevaluate() mode=0 with all
 * payload branches stripped.
 *
 * pot_out: always populated with the raw potential sum.  When EVALPOTENTIAL
 * is not defined, the host scatter-back ignores it and the value is
 * effectively dead — left in the signature so the kernel body stays the
 * same regardless of compile-time gating. */
static KOKKOS_INLINE_FUNCTION int
gpu_gravtree_walk_one(int target,
                      int maxPart, int maxNodes,
                      struct particle_data *P_dev,
                      const struct gpu_gravity_tree_soa_t *s,
#ifdef PMGRID
                      double rcut, double rcut2, double asmthfac,
#endif
                      Vec3<double> &acc_out,
                      int &ninter_out,
                      double &pot_out)
{
    Vec3<double> pos = P_dev[target].Pos;
    int ptype = P_dev[target].Type;
    double pmass = P_dev[target].Mass;
    if(pmass <= 0) {acc_out = Vec3<double>{0,0,0}; ninter_out = 0; pot_out = 0.0; return 1;}

    /* Target softening: starts as the per-type floor, replaced by the
     * adaptive kernel radius if AGS is active for this type.  The CPU walk
     * (forcetree.cc:1545,1556-1571) does the same: look up the kernel
     * radius, and if it's larger than the floor, take the AGS_zeta
     * correction; otherwise clamp to the floor and skip zeta. */
    double soft = All.ForceSoftening[ptype];
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL)
    double zeta = 0.0;
    {
        double soft_adapt = gpu_force_softening_kernel_radius(P_dev, target);
        if(soft_adapt > soft) {
            soft = soft_adapt;
#if defined(ADAPTIVE_GRAVSOFT_FORGAS)
            if(ptype == 0) {zeta = gpu_get_ags_zeta(P_dev, target);}
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORALL)
            if((1 << ptype) & (ADAPTIVE_GRAVSOFT_FORALL)) {zeta = gpu_get_ags_zeta(P_dev, target);}
#endif
        }
    }
#endif
    double aold = All.ErrTolForceAcc * P_dev[target].OldAcc;

    Vec3<double> acc = {0,0,0};
    int ninter = 0;
    double pot = 0.0;

    int no = maxPart;   /* root */

    while(no >= 0)
    {
        double h = soft, h_p = -1.0;
        Vec3<double> dr;
        double r2, mass;
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL)
        int ptype_sec = -1;     /* secondary (source) type; -1 means "node" */
        double zeta_sec = 0.0;  /* secondary AGS_zeta */
#endif

        if(no < maxPart) /* particle leaf */
        {
            dr = P_dev[no].Pos - pos;
            /* GRAVITY_NEAREST_XYZ is a no-op under GRAVITY_NOT_PERIODIC (Tier-1a gate). */
            r2 = dr.norm_sq();
            mass = P_dev[no].Mass;
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL)
            /* Source softening is now adaptive (per-particle).  Mirrors the
             * particle-leaf block in forcetree.cc:1773-1779. */
            h_p = gpu_force_softening_kernel_radius(P_dev, no);
            ptype_sec = P_dev[no].Type;
#if defined(ADAPTIVE_GRAVSOFT_FORGAS)
            if(ptype_sec == 0) {zeta_sec = gpu_get_ags_zeta(P_dev, no);}
#elif defined(ADAPTIVE_GRAVSOFT_FORALL)
            zeta_sec = gpu_get_ags_zeta(P_dev, no);
#endif
#endif
        }
        else if(no >= maxPart + maxNodes) /* pseudo-particle — remote */
        {
            return 0; /* host runs CPU walk for this target */
        }
        else /* tree node */
        {
            int idx = no - maxPart;
            Vec3<MyFloat> s_node = s->s[idx];
            MyFloat len_node = s->len[idx];
            MyFloat msoft_node = s->maxsoft[idx];
            MyFloat mass_node = s->mass[idx];
            Vec3<MyFloat> center_node = s->center[idx];

            dr[0] = s_node[0] - pos[0];
            dr[1] = s_node[1] - pos[1];
            dr[2] = s_node[2] - pos[2];
            r2 = dr.norm_sq();

#ifdef PMGRID
            /* TreePM: if the node center-of-mass lies beyond rcut, check the
             * geometric center too (corner-of-cell vs softened force radius).
             * If the entire cell is outside rcut + 0.5*len, accept and skip
             * to sibling without ever computing the short-range force.
             * Mirrors forcetree.cc:1873-1882 with periodic wrap omitted
             * (Tier 1b gate: GRAVITY_NOT_PERIODIC). */
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

            /* Basic softening-radius check (non-ADAPTIVE branch from forcetree.cc:1888). */
            if(h < msoft_node) {
                if(r2 < msoft_node * msoft_node) {no = s->nextnode[idx]; continue;}
            }

            /* Opening criterion — Barnes-Hut OR relative, matching CPU walk. */
            if(All.ErrTolTheta)
            {
                if(len_node * len_node > r2 * All.ErrTolTheta * All.ErrTolTheta) {
                    no = s->nextnode[idx]; continue;
                }
            }
            else
            {
                /* inside softening */
                if((r2 < (soft + 0.6*len_node)*(soft + 0.6*len_node)) ||
                   (r2 < (msoft_node + 0.6*len_node)*(msoft_node + 0.6*len_node))) {
                    no = s->nextnode[idx]; continue;
                }
                /* relative acc-based check */
                if(mass_node * len_node * len_node > r2 * r2 * aold) {
                    no = s->nextnode[idx]; continue;
                }
                /* inside-the-cell check (non-periodic: absolute values) */
                double dcx = fabs(center_node[0] - pos[0]);
                double dcy = fabs(center_node[1] - pos[1]);
                double dcz = fabs(center_node[2] - pos[2]);
                if(dcx < 0.60 * len_node && dcy < 0.60 * len_node && dcz < 0.60 * len_node) {
                    no = s->nextnode[idx]; continue;
                }
            }

            /* Node accepted for force accumulation. */
            h_p = msoft_node;
            mass = mass_node;
        }

        /* Force kernel — common path for accepted particles and closed nodes. */
        if((r2 > 0.0) && (mass > 0.0))
        {
            double r = sqrt(r2);
            double fac_accel;
#ifdef EVALPOTENTIAL
            double fac_pot;
#endif
            if((r >= h) && (r >= h_p)) {
                fac_accel = mass / (r2 * r);
#ifdef EVALPOTENTIAL
                fac_pot   = -mass / r;
#endif
            } else {
                double h_grav = h;
                if(h_p > h_grav) {h_grav = h_p;} /* MAX-symmetrize (non-AVERAGING branch) */
                double h_inv = 1.0 / h_grav;
                double h3_inv = h_inv * h_inv * h_inv;
                double u = r * h_inv;
                fac_accel = mass * kernel_gravity(u, h_inv, h3_inv,  1);
#ifdef EVALPOTENTIAL
                fac_pot   = mass * kernel_gravity(u, h_inv, h3_inv, -1);
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL)
                /* AGS zeta correction term: ensures conservative forces with
                 * adaptive gravitational softening (Price & Monaghan 2007 style).
                 * Mirrors forcetree.cc:2109-2140.  Both 'primary' (target zeta)
                 * and 'secondary' (source zeta) contributions are added when
                 * the kernel-shared BITFLAG check passes. */
                double fac_corr = 0.0;
                int add_primary = 1, add_secondary = 1;
                double u_p = (h_p > 0) ? (r / h_p) : 0.0;
                if(r <= 0.0 || pmass <= 0.0 || mass <= 0.0 || ptype_sec < 0) {
                    add_primary = 0; add_secondary = 0;
                }
                if(zeta == 0.0 || u >= 1.0 || h <= 0.0) {add_primary = 0;}
                if(zeta_sec == 0.0 || u_p >= 1.0 || h_p <= 0.0) {add_secondary = 0;}
                if(ptype != 0 || ptype_sec != 0) {
#if defined(ADAPTIVE_GRAVSOFT_FORALL)
                    int bm_pri = ags_gravity_kernel_shared_BITFLAG((short int)ptype);
                    int bm_sec = ags_gravity_kernel_shared_BITFLAG((short int)ptype_sec);
                    if(!((1 << ptype)     & (ADAPTIVE_GRAVSOFT_FORALL)) ||
                       !((1 << ptype_sec) & bm_pri)) {add_primary = 0;}
                    if(!((1 << ptype_sec) & (ADAPTIVE_GRAVSOFT_FORALL)) ||
                       !((1 << ptype)     & bm_sec)) {add_secondary = 0;}
#else
                    add_primary = 0; add_secondary = 0;  /* FORGAS only: gas-gas only */
#endif
                }
                if(add_primary) {
                    double dWdr, wp;
                    kernel_main(u, h3_inv, h3_inv * h_inv, &wp, &dWdr, 1);
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
            /* TreePM short-range: tabulated error-function complementary
             * factor multiplies both the force and (if EVALPOTENTIAL) the
             * potential.  Samples past tabindex=NTAB represent r > rcut
             * where the long-range (PM) force takes over and the tree walk
             * contributes nothing. */
            int tabindex = (int) (asmthfac * r);
            if(tabindex >= 0 && tabindex < GIZMO_GPU_GRAVTREE_NTAB) {
                fac_accel *= shortrange_table[tabindex];
#ifdef EVALPOTENTIAL
                fac_pot   *= shortrange_table_potential[tabindex];
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
            ninter++;
        }

        /* Advance. GravCost updates (if TakeLevel >= 0) are skipped on GPU —
         * host dispatcher falls back to CPU walk when TakeLevel >= 0, so
         * skipping here is safe. */
        if(no < maxPart) {
            no = s->nextnode_aux[no];
        } else {
            no = s->sibling[no - maxPart];
        }
    }

    acc_out = acc;
    ninter_out = ninter;
    pot_out = pot;
    return 1;
}


extern "C" int gpu_gravtree_walk_primary(void)
{
    GIZMO_GPU_ENSURE_ALL_FRESH(gravtree);
    /* Skip when the tree is in cost-measurement mode — GravCost accumulation
     * needs atomic ops we haven't wired yet. The CPU walk handles those
     * iterations. TakeLevel / ActiveParticleList / ProcessedFlag come from
     * allvars.h; Ewald_iter / Costtotal from the file-scope externs above. */
    if(TakeLevel >= 0) {return 0;}

    /* Ewald-iter iterations handle periodic corrections — guarded off at
     * compile time above, but belt-and-suspenders: skip when Ewald_iter>0. */
    if(Ewald_iter > 0) {return 0;}

    int num_active_total = (int) ActiveParticleList.size();
    if(num_active_total <= 0) {return 0;}

    /* Build local-work index list: only particles we haven't processed yet.
     * (In the first pass this is all of them, but gravity_tree() may re-enter
     * after a buffer-fill; ProcessedFlag respects that.) */
    int *idx_host = (int *) mymalloc("gpu_grav_idx", num_active_total * sizeof(int));
    int num_active = 0;
    for(int a = 0; a < num_active_total; a++) {
        int i = ActiveParticleList[a];
        if(!ProcessedFlag[i]) {idx_host[num_active++] = i;}
    }
    if(num_active <= 0) {myfree(idx_host); return 0;}

    /* Acquire Phase 1 arena (P_dev in SharedSpace) + Phase 3 tree SoA. */
    gpu_particles_arena_acquire(NumPart, P, CellP);
    struct particle_data *P_dev = gpu_particles_arena_P();

    int min_nodes = MaxNodes + 1;
    gpu_gravity_tree_acquire(min_nodes, Nodes_base, Extnodes_base);
    /* Nextnode[] is sized (MaxPart + NTopnodes) in force_treeallocate. */
    gpu_gravity_tree_set_nextnode(All.MaxPart + NTopnodes, Nextnode);
    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!P_dev || !soa) {
        printf("gpu_gravtree_walk_primary: failed to acquire arena or tree SoA\n");
        endrun(913200);
    }

    /* Scratch: failed flag + result arrays in SharedSpace. */
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
#endif

    Kokkos::parallel_for("gravtree_walk_primary", num_active, KOKKOS_LAMBDA(int a) {
        int target = d_idx[a];
        Vec3<double> acc;
        int ninter;
        double pot;
        int ok = gpu_gravtree_walk_one(target, maxPart, maxNodes_snap,
                                        P_dev, &soa_snap,
#ifdef PMGRID
                                        rcut_snap, rcut2_snap, asmthfac_snap,
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

    /* Scatter successes back to host; mark ProcessedFlag. Failed particles
     * are left untouched so the CPU primary loop picks them up. */
    int nsucceeded = 0;
    double costtotal_added = 0;
    for(int a = 0; a < num_active; a++) {
        int i = d_idx[a];
        if(!d_failed[a]) {
            /* Write raw force sum (no G factor): the post-walk loop in
             * gravity_tree() does P[i].GravAccel *= All.G unconditionally
             * for every active particle, so applying G here would
             * double-multiply (silent in tests with GravityConstantInternal=1
             * but wrong for any G != 1). */
            P[i].GravAccel = d_acc[a];
#ifdef EVALPOTENTIAL
            /* Same convention: write raw potential sum.  Post-walk loop in
             * gravity_tree() does P[i].Potential *= All.G and adds the
             * cosmological/PM corrections. */
            P[i].Potential = d_pot[a];
#endif
            ProcessedFlag[i] = 1;
            costtotal_added += d_ninter[a];
            nsucceeded++;
        }
    }
    /* Feed ninteractions into Costtotal so diagnostics stay consistent. */
    Costtotal += costtotal_added;

    /* Mark the particle arena stale: we wrote GravAccel into host P[],
     * so the arena no longer mirrors host P[]. Next acquire() will re-copy. */
    gpu_particles_arena_invalidate();

    /* Mark the tree SoA stale: the CPU walk that follows may call
     * force_drift_node() which updates Nodes[no].u.d.s and .mass in-place
     * (Nodes_base is the live array).  Without invalidation here the SoA
     * would serve frozen (pre-drift) node COM positions on the next timestep.
     * This forces gpu_gravity_tree_acquire() to reseed from the then-current
     * Nodes_base on the next call (analogous to the particle arena pattern). */
    gpu_gravity_tree_invalidate();

    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_pot);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_ninter);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_acc);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_failed);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_idx);
    myfree(idx_host);

    return nsucceeded;
}

GPU_ALL_SYNC_FUNC(gravtree)

#else /* !GIZMO_GPU_GRAVTREE || !OPENMP_GPU_OFFLOAD */

/* No-op stub when the flag is not set. Keeps the caller in gravity_tree()
 * simple — it can unconditionally call gpu_gravtree_walk_primary(). */
extern "C" int gpu_gravtree_walk_primary(void) {return 0;}

#endif
