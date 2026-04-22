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

/* Compile-time gating: Tier 1a supports ONLY the core walk. Any of these
 * payloads requires the walk kernel to compute extra terms that haven't been
 * ported yet — they land in Tier 1b, 2, or 3. Gating here keeps the kernel
 * code readable and prevents silent wrong-physics on configs we haven't
 * validated. */
#if defined(PMGRID)
#error "GIZMO_GPU_GRAVTREE Tier 1a does not support PMGRID (added in Tier 1b)."
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL) || defined(ADAPTIVE_GRAVSOFT_FROM_TIDAL_CRITERION)
#error "GIZMO_GPU_GRAVTREE Tier 1a does not support ADAPTIVE_GRAVSOFT_* (added in Tier 1b)."
#endif
#if defined(EVALPOTENTIAL) || defined(OUTPUT_POTENTIAL)
#error "GIZMO_GPU_GRAVTREE Tier 1a does not support EVALPOTENTIAL / OUTPUT_POTENTIAL (added in Tier 1b)."
#endif
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
 * payload branches stripped. */
static KOKKOS_INLINE_FUNCTION int
gpu_gravtree_walk_one(int target,
                      int maxPart, int maxNodes,
                      struct particle_data *P_dev,
                      const struct gpu_gravity_tree_soa_t *s,
                      Vec3<double> &acc_out,
                      int &ninter_out)
{
    Vec3<double> pos = P_dev[target].Pos;
    int ptype = P_dev[target].Type;
    double pmass = P_dev[target].Mass;
    if(pmass <= 0) {acc_out = Vec3<double>{0,0,0}; ninter_out = 0; return 1;}

    /* Tier-1a softening: reduces to All.ForceSoftening[ptype] under the
     * compile-time gating above (no ADAPTIVE_GRAVSOFT_*). Using All. */
    double soft = All.ForceSoftening[ptype];
    double aold = All.ErrTolForceAcc * P_dev[target].OldAcc;

    Vec3<double> acc = {0,0,0};
    int ninter = 0;

    int no = maxPart;   /* root */

    while(no >= 0)
    {
        double h = soft, h_p = -1.0;
        Vec3<double> dr;
        double r2, mass;

        if(no < maxPart) /* particle leaf */
        {
            dr = P_dev[no].Pos - pos;
            /* GRAVITY_NEAREST_XYZ is a no-op under GRAVITY_NOT_PERIODIC (Tier-1a gate). */
            r2 = dr.norm_sq();
            mass = P_dev[no].Mass;
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
            if((r >= h) && (r >= h_p)) {
                fac_accel = mass / (r2 * r);
            } else {
                double h_grav = h;
                if(h_p > h_grav) {h_grav = h_p;} /* MAX-symmetrize (non-AVERAGING branch) */
                double h_inv = 1.0 / h_grav;
                double h3_inv = h_inv * h_inv * h_inv;
                double u = r * h_inv;
                fac_accel = mass * kernel_gravity(u, h_inv, h3_inv, 1);
            }
            acc[0] += fac_accel * dr[0];
            acc[1] += fac_accel * dr[1];
            acc[2] += fac_accel * dr[2];
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
    if(!d_idx || !d_failed || !d_acc || !d_ninter) {
        printf("gpu_gravtree_walk_primary: kokkos_malloc failed\n");
        endrun(913201);
    }
    memcpy(d_idx, idx_host, num_active * sizeof(int));
    memset(d_failed, 0, num_active * sizeof(int));

    int maxPart = All.MaxPart;
    int maxNodes_snap = MaxNodes;
    const struct gpu_gravity_tree_soa_t soa_snap = *soa;

    Kokkos::parallel_for("gravtree_walk_primary", num_active, KOKKOS_LAMBDA(int a) {
        int target = d_idx[a];
        Vec3<double> acc;
        int ninter;
        int ok = gpu_gravtree_walk_one(target, maxPart, maxNodes_snap,
                                        P_dev, &soa_snap,
                                        acc, ninter);
        if(ok) {
            d_acc[a] = acc;
            d_ninter[a] = ninter;
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
