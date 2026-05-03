/* gpu_particles_arena.cc — Step 13 Phase 1
 *
 * See gpu_particles_arena.h for design notes.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <Kokkos_Core.hpp>

/* GPU All mirror: must precede allvars.h so nvc++ sees `All` (=All_dev) when it
 * eagerly parses templates in declarations/allvars.h that reference it. Matches
 * the include order in hydro/density_gpu.cc and other GPU TUs. */
#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../core/step_phases.h"
#include "gpu_particles_arena.h"
#include "../mesh/gpu_neighbor_list.h"


static struct particle_data *arena_P     = NULL;
static struct gas_cell_data *arena_CellP = NULL;
static int arena_capacity_ = 0;
static int arena_valid_    = 0;

/* DIAGNOSTIC: per-acquire sequence counter and call-site tag.
 * Call gpu_particles_arena_set_site("tag") just before acquire to label it. */
static int  g_acquire_serial  = 0;
static int  g_valid_memcpy_serial = 0;  /* serial# of last acquire that did memcpy */
static const char *g_arena_site = "(unknown)";  /* set by caller before acquire */

/* Phase 8a Round 1: track last call site that asserted "I just mirror-updated
 * the arena" — for diagnostic output when the debug byte-compare guard fails. */
static const char *g_arena_last_clean_site = "(none)";

/* Phase 8a Round 1.5: also track who did the most-recent slow-path memcpy
 * (that made arena valid) and who most recently invalidated. The acquire
 * trio (memcpy, invalidate, current) names the suspects when the guard fires. */
static const char *g_arena_last_memcpy_site    = "(none)";
static const char *g_arena_last_invalidate_site = "(none)";

extern "C" void gpu_particles_arena_set_site(const char *site) { g_arena_site = site; }

/* Phase 8a Round 1: runtime-gated arena debug. The helpers below are always
 * compiled in; the actual byte-compare guard inside gpu_particles_arena_acquire
 * is enabled by GIZMO_GPU_ARENA_DEBUG=1 env var (cached on first lookup).
 * This lets us toggle the heavy O(N) check without rebuilding. */
static int g_arena_debug_init = 0;
static int g_arena_debug_on   = 0;
static inline int arena_debug_enabled_(void)
{
    if(!g_arena_debug_init) {
        g_arena_debug_init = 1;
        const char *q = getenv("GIZMO_GPU_ARENA_DEBUG");
        g_arena_debug_on = (q && q[0] && q[0] != '0') ? 1 : 0;
    }
    return g_arena_debug_on;
}

/* One-time print of key field byte-offsets within particle_data and gas_cell_data,
 * so the arena_report_first_diff_ output can be mapped to field names. */
static int g_offsets_printed = 0;
static void arena_print_struct_offsets_(void)
{
    if(g_offsets_printed) {return;}
    g_offsets_printed = 1;
    printf("ARENA_STRUCT_OFFSETS: sizeof(particle_data)=%d  sizeof(gas_cell_data)=%d\n",
           (int)sizeof(struct particle_data), (int)sizeof(struct gas_cell_data));
    /* key particle_data fields */
#ifdef EVALPOTENTIAL
    printf("  P: GravAccel=%d  Potential=%d  OldAcc=%d  GravCost=%d\n",
           (int)offsetof(struct particle_data, GravAccel),
           (int)offsetof(struct particle_data, Potential),
           (int)offsetof(struct particle_data, OldAcc),
           (int)offsetof(struct particle_data, GravCost));
#else
    printf("  P: GravAccel=%d  OldAcc=%d  GravCost=%d\n",
           (int)offsetof(struct particle_data, GravAccel),
           (int)offsetof(struct particle_data, OldAcc),
           (int)offsetof(struct particle_data, GravCost));
#endif
    printf("  P: Pos=%d  Vel=%d  Mass=%d  Type=%d  ID=%d\n",
           (int)offsetof(struct particle_data, Pos),
           (int)offsetof(struct particle_data, Vel),
           (int)offsetof(struct particle_data, Mass),
           (int)offsetof(struct particle_data, Type),
           (int)offsetof(struct particle_data, ID));
#ifdef SINK_CALC_DISTANCES
    printf("  P: Min_Distance_to_Sink=%d\n",
           (int)offsetof(struct particle_data, Min_Distance_to_Sink));
#endif
#if defined(GALSF) && defined(RT_USE_GRAVTREE)
    printf("  P: KernelRadius=%d  DensityAroundParticle=%d\n",
           (int)offsetof(struct particle_data, KernelRadius),
           (int)offsetof(struct particle_data, DensityAroundParticle));
#endif
    /* gas_cell_data fields relevant to RT */
#if defined(RT_USE_GRAVTREE) || defined(RADTRANSFER)
    printf("  CellP: Mass=%d  Density=%d\n",
           (int)offsetof(struct gas_cell_data, Mass),
           (int)offsetof(struct gas_cell_data, Density));
#ifdef RT_INFRARED
    printf("  CellP: Radiation_Temperature=%d  Dust_Temperature=%d\n",
           (int)offsetof(struct gas_cell_data, Radiation_Temperature),
           (int)offsetof(struct gas_cell_data, Dust_Temperature));
#endif
#endif
    fflush(stdout);
}

/* Print the byte offset of the first mismatch and attempt to show nearby bytes. */
static void arena_report_first_diff_(const unsigned char *arena_buf,
                                     const unsigned char *host_buf,
                                     size_t total_bytes,
                                     size_t stride,   /* sizeof(struct) */
                                     const char *label,
                                     int serial)
{
    for(size_t b = 0; b < total_bytes; b++) {
        if(arena_buf[b] != host_buf[b]) {
            int ptcl = (int)(b / stride);
            int foff = (int)(b % stride);
            printf("  ARENA_DIFF(%s) serial=%d: first diff in %s[] at particle %d, "
                   "byte_in_struct=%d (struct_size=%d)\n",
                   g_arena_site, serial, label, ptcl, foff, (int)stride);
            printf("  arena[%d]+%d = 0x%02x  host[%d]+%d = 0x%02x\n",
                   ptcl, foff, (unsigned)arena_buf[b],
                   ptcl, foff, (unsigned)host_buf[b]);
            /* Show 16 bytes of context around the diff. */
            int lo = (foff > 8) ? foff - 8 : 0;
            int hi = (foff + 8 < (int)stride) ? foff + 8 : (int)stride - 1;
            printf("  arena context [%d..%d]:", lo, hi);
            for(int k = lo; k <= hi; k++) {
                printf(" %02x", (unsigned)(arena_buf[ptcl*stride + k]));
            }
            printf("\n  host  context [%d..%d]:", lo, hi);
            for(int k = lo; k <= hi; k++) {
                printf(" %02x", (unsigned)(host_buf[ptcl*stride + k]));
            }
            printf("\n");
            fflush(stdout);
            return;
        }
    }
}

extern "C" void gpu_particles_arena_acquire(int min_capacity,
                                            struct particle_data *P_host,
                                            struct gas_cell_data *CellP_host)
{
    if(min_capacity <= 0) {min_capacity = 1;}
    g_acquire_serial++;
    int my_serial = g_acquire_serial;

    /* Phase 7 Round A4: arena state diagnostics. env-gated; no-op when off. */
    gizmo_step_phase_record("arena_acquire_calls", 1.0);

    if(arena_P && arena_CellP && arena_capacity_ >= min_capacity) {
        if(arena_valid_) {
            if(arena_debug_enabled_()) {
                /* Phase 8a Round 1: runtime debug guard. arena claims to be in sync
                 * with host; verify by byte-compare. If a host mutation site forgot
                 * to call gpu_particles_arena_invalidate(), or a Phase-8 mirror-update
                 * is incomplete, this aborts at the offending acquire rather than
                 * yielding silent stale-data corruption downstream. */
                int p_diff = (memcmp(arena_P, P_host, min_capacity * sizeof(struct particle_data)) != 0);
                int c_diff = (CellP_host && memcmp(arena_CellP, CellP_host, min_capacity * sizeof(struct gas_cell_data)) != 0);
                if(p_diff || c_diff) {
                    arena_print_struct_offsets_();
                    printf("gpu_particles_arena_acquire: arena_valid_==1 but host data differs from arena.\n"
                           "  current_site='%s' serial=%d (last_memcpy_serial=%d)\n"
                           "  last_memcpy_site='%s'   (= site that did the most-recent slow-path memcpy)\n"
                           "  last_invalidate_site='%s' (= site that most recently set arena_valid_=0)\n"
                           "  last_mark_clean_site='%s' (= site that most recently asserted arena coherent)\n"
                           "  ==> kernel between {memcpy or mark_clean} and now wrote arena[i].field while host[i].field stayed unchanged.\n"
                           "  Capacity = %d. P_diff=%d CellP_diff=%d. Aborting.\n",
                           g_arena_site, my_serial, g_valid_memcpy_serial,
                           g_arena_last_memcpy_site,
                           g_arena_last_invalidate_site,
                           g_arena_last_clean_site,
                           min_capacity, p_diff, c_diff);
                    if(p_diff) {
                        arena_report_first_diff_((const unsigned char *)arena_P,
                                                (const unsigned char *)P_host,
                                                (size_t)min_capacity * sizeof(struct particle_data),
                                                sizeof(struct particle_data), "P", my_serial);
                    }
                    if(c_diff) {
                        arena_report_first_diff_((const unsigned char *)arena_CellP,
                                                (const unsigned char *)CellP_host,
                                                (size_t)min_capacity * sizeof(struct gas_cell_data),
                                                sizeof(struct gas_cell_data), "CellP", my_serial);
                    }
                    fflush(stdout);
                    endrun(913002);
                }
            }
            /* Fast path: arena holds the latest host state already (no invalidate
             * fired since the previous acquire). Skip memcpy entirely — the win
             * compounds across kernels in a single timestep. */
            gizmo_step_phase_record("arena_acquire_fastpath", 1.0);
            return;
        }
        /* Slow path: arena is stale (some host mutation site invalidated us).
         * Re-seed from host. UVM keeps pages device-resident if they weren't
         * dirtied on the device side since last access. CellP_host may be NULL
         * for N-body callers (no gas); arena_CellP storage exists but is unused. */
        gizmo_step_phase_record("arena_acquire_slowpath", 1.0);
        double t_acq_start = my_second();
        memcpy(arena_P,     P_host,     min_capacity * sizeof(struct particle_data));
        if(CellP_host) {memcpy(arena_CellP, CellP_host, min_capacity * sizeof(struct gas_cell_data));}
        gizmo_step_phase_record("arena_acquire_memcpy_time", timediff(t_acq_start, my_second()));
        g_valid_memcpy_serial = my_serial;
        g_arena_last_memcpy_site = g_arena_site; /* Phase 8a R1.5: who fired the slow-path? */
        arena_valid_ = 1;
        /* DO NOT mark compact_xyzh dirty here.  Within a step, the only
         * within-step mutators of P[].KernelRadius are density's inflate /
         * restore / iter-sync (which call gpu_compact_xyzh_mark_h_dirty()
         * explicitly).  Other within-step host mutations (mech_fb / hii_fb /
         * radfb_g per-source mass-loss apply, etc.) touch Mass/Vel/dp/Z but
         * NOT KernelRadius, so arena re-seed brings new field values that
         * don't affect compact_xyzh.h.  Cross-step mutators (predict.cc drift,
         * sink_swallow accretion, merge_split refinement) all happen AFTER
         * the last in-step ngb_build and BEFORE the next step's drift, which
         * invalidates the SIDX entirely (sidx_cached=0 → full rebuild seeds
         * compact_xyzh from current h, marking dirty=0 fresh).  Ghost-exchange
         * h-import is the only multi-rank case that updates ghost-slot h
         * mid-step; if/when that becomes hot, mark_h_dirty inside the ghost
         * import path rather than blanket-marking on every arena re-seed. */
        return;
    }

    /* Need fresh allocation: capacity grew or first-ever acquire. */
    if(arena_CellP) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(arena_CellP); arena_CellP = NULL;}
    if(arena_P)     {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(arena_P);     arena_P     = NULL;}

    arena_P     = (struct particle_data *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(min_capacity * sizeof(struct particle_data));
    arena_CellP = (struct gas_cell_data *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(min_capacity * sizeof(struct gas_cell_data));
    if(!arena_P || !arena_CellP) {
        printf("gpu_particles_arena_acquire: kokkos_malloc failed for capacity=%d\n", min_capacity);
        endrun(913001);
    }
    memcpy(arena_P,     P_host,     min_capacity * sizeof(struct particle_data));
    if(CellP_host) {memcpy(arena_CellP, CellP_host, min_capacity * sizeof(struct gas_cell_data));}
    arena_capacity_ = min_capacity;
    g_valid_memcpy_serial = my_serial;
    g_arena_last_memcpy_site = g_arena_site; /* fresh-alloc memcpy also counts */
    arena_valid_    = 1;
    /* Fresh arena alloc: any prior SIDX's compact_xyzh is stale wrt current host h. */
    gpu_compact_xyzh_mark_h_dirty();
}

extern "C" void gpu_particles_arena_invalidate(void)
{
    /* Phase 7 Round A4: count invalidate calls per step. env-gated; no-op when off. */
    gizmo_step_phase_record("arena_invalidate_calls", 1.0);
    /* Phase 8a R1.5: track who most recently invalidated, using whatever site
     * was last set via gpu_particles_arena_set_site (callers that don't set
     * a site keep showing the previous setter, which is also informative —
     * implies the invalidate happened deep in code that doesn't own the site). */
    g_arena_last_invalidate_site = g_arena_site;
    arena_valid_ = 0;
}

/* Phase 8a Round 1: contract API for mirror-update sites.
 *
 * A wrapper that mutates host P/CellP fields after its GPU kernel returns
 * MUST either (a) call gpu_particles_arena_invalidate(), or (b) write the
 * same fields to the arena and call this function instead.  Calling this
 * function is a CONTRACT that the arena now holds host-equivalent data
 * for the modified indices/fields; the next acquire can fast-path.
 *
 * The optional debug guard (GIZMO_GPU_ARENA_DEBUG=1) byte-compares on the
 * next acquire and aborts if the contract was violated, naming the most
 * recent site that asserted clean.  Counter recorded in STEP_PHASES so we
 * can correlate "arena_mark_clean_calls" with "arena_acquire_fastpath" —
 * a properly-mirrored step should have these match closely.
 */
extern "C" void gpu_particles_arena_mark_clean_after_scatter(const char *site)
{
    g_arena_last_clean_site = (site ? site : "(unnamed)");
    arena_valid_ = 1;
    g_valid_memcpy_serial = g_acquire_serial; /* logical sync point */
    gizmo_step_phase_record("arena_mark_clean_calls", 1.0);
}

extern "C" void gpu_particles_arena_release(void)
{
    if(arena_CellP) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(arena_CellP); arena_CellP = NULL;}
    if(arena_P)     {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(arena_P);     arena_P     = NULL;}
    arena_capacity_ = 0;
    arena_valid_    = 0;
}

extern "C" struct particle_data *gpu_particles_arena_P(void)     {return arena_valid_ ? arena_P     : NULL;}
extern "C" struct gas_cell_data *gpu_particles_arena_CellP(void) {return arena_valid_ ? arena_CellP : NULL;}
extern "C" int gpu_particles_arena_capacity(void)                {return arena_capacity_;}
extern "C" int gpu_particles_arena_valid(void)                   {return arena_valid_;}

