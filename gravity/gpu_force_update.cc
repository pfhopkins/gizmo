/* gpu_force_update.cc — Step 13 Phase 7.c
 *
 * GPU replacement for force_update_tree().
 * Propagates per-particle momentum kicks (P[i].dp) through the tree via three
 * GPU-accelerated stages + host-side MPI:
 *
 *   Stage 1  gpu_force_drift_nodes()    — reuse Phase 7.a kernel; drifts all
 *                                         stale nodes to All.Ti_Current in one pass.
 *   Stage 2  gpu_force_kick_kernel      — per-active-particle Father-chain walk;
 *                                         atomic-accumulates dp into Extnodes[no].dp,
 *                                         atomic-maxes vmax, sets NODEHASBEENKICKED,
 *                                         fills UVM DomainList buffer.
 *   Stage 3  force_finish_kick_nodes()  — unchanged CPU code; does MPI Allgatherv
 *                                         of changed domain nodes and applies received
 *                                         kicks to ancestor chain (all UVM, CPU-safe).
 *
 * RT_SEPARATELY_TRACK_LUMPOS: rt_get_source_luminosity() is not GPU-callable;
 * rt_source_lum_dp is pre-computed on host into a UVM buffer before kernel launch.
 * DM_SCALARFIELD_SCREENING: dp_dm is computed in-kernel (Type != 0 check).
 *
 * DomainList is allocated as SharedSpace (UVM) so the GPU kernel writes and
 * force_finish_kick_nodes reads without copies.  The global DomainList pointer
 * is temporarily redirected to the UVM buffer for the duration of the call.
 *
 * P[i].dp zeroing: the GPU kernel zeros Pp[i].dp (the arena copy).  On UVM
 * systems Pp==P so this is sufficient.  On non-UVM systems (e.g., Mac CPU
 * Kokkos where SharedSpace != P memory) the kernel zero does not propagate to
 * P[].  A host loop after Kokkos::fence() explicitly zeros P[i].dp for all
 * active particles; this is a no-op on UVM and the necessary fix on non-UVM.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../core/step_phases.h"
#include "../system/gpu_particles_arena.h"
#include "gpu_gravity_tree.h"
#include "forcetree.h"


/* Atomic max for MyFloat via 64-bit CAS (MyFloat = double in GIZMO typedefs). */
static_assert(sizeof(MyFloat) == sizeof(uint64_t),
              "gpu_atomic_max_float: MyFloat must be 64-bit");
KOKKOS_INLINE_FUNCTION static void
gpu_atomic_max_myfloat(MyFloat* addr, MyFloat val)
{
    uint64_t val_bits, old_bits;
    memcpy(&val_bits, &val, sizeof(MyFloat));
    MyFloat old = *addr;
    while(val > old) {
        memcpy(&old_bits, &old, sizeof(MyFloat));
        uint64_t prev = Kokkos::atomic_compare_exchange(
            reinterpret_cast<uint64_t*>(addr), old_bits, val_bits);
        if(prev == old_bits) break;
        memcpy(&old, &prev, sizeof(MyFloat));
    }
}

/* =========================================================================
 * gpu_force_update_tree — drop-in GPU replacement for force_update_tree().
 * ========================================================================= */
extern "C" void gpu_force_update_tree(void)
{
    PRINT_STATUS("Kick-subroutine will prepare for dynamic update of tree (GPU)");

    /* Phase 7 Round A2: sub-bucket the 1.12s force_update_tree top-level cost. */
    double t_fut_start = my_second();

    GlobFlag++;
    DomainNumChanged = 0;

    /* Allocate UVM buffers for DomainList and its atomic count. */
    const int ntop = (NTopleaves > 0) ? NTopleaves : 1;
    int *domain_list_dev = (int *)Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
                                ntop * sizeof(int));
    int *domain_count_dev = (int *)Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
                                sizeof(int));
    domain_count_dev[0] = 0;
    DomainList = domain_list_dev;   /* redirect global ptr to UVM buffer */

    /* Re-acquire particles arena (invalidated at end of gpu_gravtree_walk_primary).
     * On UVM systems this is a same-pointer re-registration (cheap). */
    gpu_particles_arena_set_site("gpu_force_update_domainlist");
    gpu_particles_arena_acquire(NumPart, P, CellP);
    double t_fut_arena = my_second();

    /* Stage 1: drift all stale nodes to Ti_Current (reuse Phase 7.a kernel).
     * Codex 2026-05-12: out-of-line host accessor; see
     * feedback_all_dev_trap_host_side.md. */
    if(gpu_force_drift_nodes(gizmo_host_ti_current()) != 0) { endrun(929703); }
    double t_fut_drift_nodes = my_second();

    const int num_active = (int)ActiveParticleList.size();
    if(num_active <= 0) {
        DomainNumChanged = 0;
        goto finish_mpi;
    }

    {
        /* Copy active-particle index list to a UVM buffer. */
        int *active_dev = (int *)Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
                                    num_active * sizeof(int));
        memcpy(active_dev, ActiveParticleList.data(), num_active * sizeof(int));

        struct particle_data *Pp   = gpu_particles_arena_P();
        int                  *Fa   = Father;    /* UVM since Phase 6.6 */
        struct NODE          *No   = Nodes;     /* UVM shifted ptr since Phase 6.8d */
        struct extNODE       *Ex   = Extnodes;  /* UVM since Phase 6.8d */
        int                   gflag = GlobFlag;
        /* Codex 2026-05-12: out-of-line host accessor (used host-side
         * here, captured by-value into the device lambda). See
         * feedback_all_dev_trap_host_side.md. */
        integertime           ti_cur = gizmo_host_ti_current();

#ifdef RT_SEPARATELY_TRACK_LUMPOS
        /* Pre-compute rt_source_lum_dp per active particle on CPU (not GPU-callable). */
        MyDouble *rt_lum_dp_dev = (MyDouble *)Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
                                       num_active * 3 * sizeof(MyDouble));
        for(int idx = 0; idx < num_active; idx++) {
            int i = ActiveParticleList[idx];
            double lum[N_RT_FREQ_BINS];
            int active_check = rt_get_source_luminosity(i, -1, lum, P, CellP);
            Vec3<MyDouble> dp_i = P[i].dp;
            Vec3<MyDouble> rt_dp = active_check ? dp_i : Vec3<MyDouble>{};
            rt_lum_dp_dev[idx*3+0] = rt_dp[0];
            rt_lum_dp_dev[idx*3+1] = rt_dp[1];
            rt_lum_dp_dev[idx*3+2] = rt_dp[2];
        }
#endif

        /* Stage 2: GPU kick kernel — per-active-particle Father-chain walk. */
        Kokkos::parallel_for("gpu_force_kick", num_active,
            KOKKOS_LAMBDA(const int idx) {
                const int i = active_dev[idx];

                /* Read and zero P[i].dp (arena copy) — host zero below handles non-UVM. */
                Vec3<MyDouble> dp = Pp[i].dp;
                Pp[i].dp = Vec3<MyDouble>{};

                /* Compute velocity magnitude for vmax update. */
                MyFloat vmax = 0;
                for(int k = 0; k < 3; k++) {
                    MyFloat v = (MyFloat)fabs((double)Pp[i].Vel[k]);
                    if(v > vmax) vmax = v;
                }

#ifdef RT_SEPARATELY_TRACK_LUMPOS
                Vec3<MyDouble> rt_dp = { rt_lum_dp_dev[idx*3+0],
                                          rt_lum_dp_dev[idx*3+1],
                                          rt_lum_dp_dev[idx*3+2] };
#endif
#ifdef DM_SCALARFIELD_SCREENING
                Vec3<MyDouble> dp_dm = (Pp[i].Type != 0) ? dp : Vec3<MyDouble>{};
#endif

                /* Walk Father chain, accumulating kicks. */
                int no = Fa[i];
                while(no >= 0) {
                    /* dp accumulation (atomic since multiple particles share ancestors). */
                    for(int k = 0; k < 3; k++) {
                        Kokkos::atomic_add(&Ex[no].dp[k], dp[k]);
                    }
#ifdef RT_SEPARATELY_TRACK_LUMPOS
                    for(int k = 0; k < 3; k++) {
                        Kokkos::atomic_add(&Ex[no].rt_source_lum_dp[k], rt_dp[k]);
                    }
#endif
#ifdef DM_SCALARFIELD_SCREENING
                    for(int k = 0; k < 3; k++) {
                        Kokkos::atomic_add(&Ex[no].dp_dm[k], dp_dm[k]);
                    }
#endif
                    gpu_atomic_max_myfloat(&Ex[no].vmax, vmax);
                    Kokkos::atomic_fetch_or(&No[no].u.d.bitflags,
                                            (unsigned int)(1 << BITFLAG_NODEHASBEENKICKED));
                    Ex[no].Ti_lastkicked = ti_cur;

                    if(No[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) {
                        /* Deduplicate: claim with atomic_exchange on Flag. */
                        int old_flag = Kokkos::atomic_exchange(&Ex[no].Flag, gflag);
                        if(old_flag != gflag) {
                            int slot = Kokkos::atomic_fetch_add(domain_count_dev, 1);
                            if(slot < ntop) { domain_list_dev[slot] = no; }
                        }
                        break;
                    }
                    no = No[no].u.d.father;
                }
            });
        Kokkos::fence();
        double t_fut_kernel_done = my_second();
        gizmo_step_phase_record("fut_kernel", timediff(t_fut_drift_nodes, t_fut_kernel_done));

        /* Zero P[i].dp on the host.  On UVM systems Pp==P so the kernel zero above
         * already did this; on non-UVM (Mac CPU Kokkos) the arena is a separate
         * SharedSpace buffer and this host loop is the authoritative zero. */
        for(int idx = 0; idx < num_active; idx++) {
            P[ActiveParticleList[idx]].dp = {};
        }

        DomainNumChanged = domain_count_dev[0];

        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(active_dev);
#ifdef RT_SEPARATELY_TRACK_LUMPOS
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(rt_lum_dp_dev);
#endif
    }

finish_mpi:
    /* Stage 3: host-side MPI Allgatherv + ancestor apply.
     * force_finish_kick_nodes reads DomainList (now UVM), DomainNumChanged, and
     * writes to Extnodes/Nodes (UVM) — all coherent after Kokkos::fence() above. */
    double t_fut_pre_mpi = my_second();
    force_finish_kick_nodes();
    double t_fut_post_mpi = my_second();

    /* Restore global DomainList to NULL and free UVM buffers. */
    DomainList = NULL;
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(domain_count_dev);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(domain_list_dev);

    PRINT_STATUS(" ..Tree has been updated dynamically (GPU)");

    /* Phase 7 Round A2 sub-buckets — env-gated; no-op when off. */
    {
        double t_fut_done = my_second();
        gizmo_step_phase_record("fut_arena_acquire", timediff(t_fut_start,    t_fut_arena));
        gizmo_step_phase_record("fut_drift_nodes",   timediff(t_fut_arena,    t_fut_drift_nodes));
        gizmo_step_phase_record("fut_mpi",           timediff(t_fut_pre_mpi,  t_fut_post_mpi));
        gizmo_step_phase_record("fut_total",         timediff(t_fut_start,    t_fut_done));
    }
}

GPU_ALL_SYNC_FUNC(force_update)

