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
#include "gpu_particles_arena.h"

#ifdef OPENMP_GPU_OFFLOAD

static struct particle_data *arena_P     = NULL;
static struct gas_cell_data *arena_CellP = NULL;
static int arena_capacity_ = 0;
static int arena_valid_    = 0;

/* DIAGNOSTIC: per-acquire sequence counter and call-site tag.
 * Call gpu_particles_arena_set_site("tag") just before acquire to label it. */
static int  g_acquire_serial  = 0;
static int  g_valid_memcpy_serial = 0;  /* serial# of last acquire that did memcpy */
static const char *g_arena_site = "(unknown)";  /* set by caller before acquire */

extern "C" void gpu_particles_arena_set_site(const char *site) { g_arena_site = site; }

#ifdef GIZMO_GPU_ARENA_DEBUG
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
#ifdef OPENMP_GPU_OFFLOAD
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
#endif /* GIZMO_GPU_ARENA_DEBUG */

extern "C" void gpu_particles_arena_acquire(int min_capacity,
                                            struct particle_data *P_host,
                                            struct gas_cell_data *CellP_host)
{
    if(min_capacity <= 0) {min_capacity = 1;}
    g_acquire_serial++;
    int my_serial = g_acquire_serial;

    if(arena_P && arena_CellP && arena_capacity_ >= min_capacity) {
        if(arena_valid_) {
#ifdef GIZMO_GPU_ARENA_DEBUG
            /* Debug guard: arena claims to be in sync with host; verify by byte-compare.
             * If a host mutation site forgot to call gpu_particles_arena_invalidate(),
             * this aborts at the offending kernel call rather than yielding silent
             * stale-data corruption downstream. */
            int p_diff = (memcmp(arena_P, P_host, min_capacity * sizeof(struct particle_data)) != 0);
            int c_diff = (CellP_host && memcmp(arena_CellP, CellP_host, min_capacity * sizeof(struct gas_cell_data)) != 0);
            if(p_diff || c_diff) {
                arena_print_struct_offsets_();
                printf("gpu_particles_arena_acquire: arena_valid_==1 but host data differs from arena.\n"
                       "  site='%s' serial=%d (last_memcpy_serial=%d)\n"
                       "  Some host mutation site missed calling gpu_particles_arena_invalidate().\n"
                       "  Capacity = %d. P_diff=%d CellP_diff=%d. Aborting.\n",
                       g_arena_site, my_serial, g_valid_memcpy_serial,
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
#endif
            /* Fast path: arena holds the latest host state already (no invalidate
             * fired since the previous acquire). Skip memcpy entirely — the win
             * compounds across kernels in a single timestep. */
            return;
        }
        /* Slow path: arena is stale (some host mutation site invalidated us).
         * Re-seed from host. UVM keeps pages device-resident if they weren't
         * dirtied on the device side since last access. CellP_host may be NULL
         * for N-body callers (no gas); arena_CellP storage exists but is unused. */
        memcpy(arena_P,     P_host,     min_capacity * sizeof(struct particle_data));
        if(CellP_host) {memcpy(arena_CellP, CellP_host, min_capacity * sizeof(struct gas_cell_data));}
        g_valid_memcpy_serial = my_serial;
        arena_valid_ = 1;
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
    arena_valid_    = 1;
}

extern "C" void gpu_particles_arena_invalidate(void)
{
    arena_valid_ = 0;
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

#endif /* OPENMP_GPU_OFFLOAD */
