/* gpu_particles_arena.cc
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
#include <exception>
#include <vector>
#if defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>   /* hipMemAdvise, for the particle-storage placement policy below */
#elif defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>      /* cudaMemAdvise, for the particle-storage placement policy below */
#endif

/* GPU All mirror: must precede allvars.h so nvc++ sees `All` (=All_dev) when it
 * eagerly parses templates in declarations/allvars.h that reference it. Matches
 * the include order in hydro/density_gpu.cc and other GPU TUs. */
#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../declarations/lifecycle_counters.h"
#include "../core/proto.h"
#include "../core/timestep_functions.h"
#include "gpu_particles_arena.h"
#include "../mesh/gpu_neighbor_list.h"
#include "../mesh/ghost_writeback.h"   /* ghost_get_num_ghosts */


/* Under UVM-canonical particles, P[] and CellP[] live in
 * Kokkos::SharedSpace and the arena is a pure pointer alias. Acquire copies
 * pointers; invalidate / mark_clean / refresh / set_site are pure no-ops kept
 * as stable API for callers. The prior debug-guard infrastructure
 * (per-acquire serial counters, per-call-site tracking strings) was removed
 * since the byte-compare guard it served is unreachable under the alias
 * scheme. */
static struct particle_data *arena_P     = NULL;
static struct gas_cell_data *arena_CellP = NULL;
static int arena_capacity_ = 0;
static int arena_valid_    = 0;

extern "C" void gpu_particles_arena_set_site(const char *site) { (void)site; }

extern "C" void gpu_particles_arena_acquire(int min_capacity,
                                            struct particle_data *P_host,
                                            struct gas_cell_data *CellP_host)
{
    /* Tiny-N corridor counter: increments on API entry. Mode B paths in
     * run_neighbor_loop must NOT enter this function. See
     * declarations/lifecycle_counters.h. */
    g_gpu_arena_acquire_counter++;

    if(min_capacity <= 0) {min_capacity = 1;}
    arena_P         = P_host;
    arena_CellP     = CellP_host;
    arena_capacity_ = min_capacity;
    arena_valid_    = 1;
}

extern "C" void gpu_particles_arena_invalidate(void) {}

/* ---- bulk prefetch of the live particle storage (contract in gpu_particles_arena.h) ---- */
static int    prefetch_enabled_ = -1;   /* -1: not decided yet */
static int    prefetch_side_    = 0;    /* where the last bulk prefetch went: 1 device, 2 host */
static double prefetch_seconds_ = 0, prefetch_bytes_dev_ = 0, prefetch_bytes_host_ = 0;
static long long prefetch_calls_ = 0;
static uint64_t tohost_arena_counter_ = 0;   /* g_gpu_arena_acquire_counter at the last bulk move of P and CellP to the host */
#if defined(KOKKOS_ENABLE_CUDA)
static int cuda_managed_memory_is_host_coherent(void);
#endif

static int particles_prefetch_enabled(void)
{
    if(prefetch_enabled_ < 0) {
        prefetch_enabled_ = 0;
#if defined(KOKKOS_ENABLE_CUDA) && defined(GPU_PCIE_OPTIMIZATIONS)
        if(std::is_same<GIZMO_KOKKOS_SHARED_SPACE, Kokkos::CudaUVMSpace>::value && !cuda_managed_memory_is_host_coherent()) {prefetch_enabled_ = 1;}
        const char *env = getenv("GIZMO_UVM_PREFETCH");
        if(env && env[0]) {prefetch_enabled_ = (atoi(env) != 0) && std::is_same<GIZMO_KOKKOS_SHARED_SPACE, Kokkos::CudaUVMSpace>::value;}
#endif
#if defined(GPU_PCIE_OPTIMIZATIONS)
        if(ThisTask == 0) {printf("Particle storage: bulk prefetch between host and device phases is %s.\n", prefetch_enabled_ ? "on" : "off"); fflush(stdout);}
#endif
    }
    return prefetch_enabled_;
}

static int owned_kept_on_device_ = 0;   /* see gizmo_particles_keep_owned_on_device */
static int ghostdev_base_ = -1, ghostdev_count_ = 0;   /* see gizmo_ghostdev_set_live */

/* Device-resident tree-to-density span (contract in gpu_particles_arena.h). */
static int treehost_armed_ = 0;   /* between gizmo_treehost_begin and _disarm */
static int treehost_span_  = 0;   /* P left on the device by the tree build, density pass not reached yet */
static int treehost_cells_ = 0;   /* within the span: the cells were moved to the device too */

/* The ghost range is device-resident (gizmo_ghostdev_set_live) and is still the live pool's. */
static int ghostdev_range_live(void)
{
    return ghostdev_count_ > 0 && ghost_get_num_ghosts() == ghostdev_count_ && ghost_get_num_local() == ghostdev_base_;
}

static void particles_prefetch(int to_device, const char *site, int with_cells = 1, int n_owned = -1, int with_particles = 1)
{
    (void) site;
    if(!particles_prefetch_enabled()) {return;}
    owned_kept_on_device_ = 0;
    if(!to_device || (with_cells && with_particles)) {treehost_span_ = 0; treehost_cells_ = 0;}   /* P goes to the host, or both arrays end up on the device */
#if defined(KOKKOS_ENABLE_CUDA)
    /* A to-host move leaves a device-resident ghost range on the device. */
    const int keep_ghosts = !to_device && n_owned < 0 && ghostdev_range_live();
    const int n_live = keep_ghosts ? ghostdev_base_ : NumPart;
    /* n_owned >= 0: only the rank's own particles [0, n_owned) and their cells, leaving imported ghosts where they are. */
    const int n_part = (n_owned >= 0 && n_owned < NumPart) ? n_owned : n_live;
    const size_t bytes_p = (with_particles && P && n_part > 0) ? (size_t) n_part * sizeof(struct particle_data) : 0;
    /* Imported ghosts sit at the end of [0, NumPart) in both arrays, past N_gas: cover them while the
       corridor holds any (unused cell slots between N_gas and the ghosts only cost the copy). */
    const int n_cells = (n_owned >= 0 || keep_ghosts) ? ((n_part < N_gas) ? n_part : N_gas)
                      : (ghost_get_num_ghosts() > 0) ? ((NumPart < All.MaxPartGas) ? NumPart : All.MaxPartGas) : N_gas;
    const size_t bytes_c = (with_cells && CellP && n_cells > 0) ? (size_t) n_cells * sizeof(struct gas_cell_data) : 0;
    if(bytes_p + bytes_c == 0) {return;}
    const double t0 = my_second();
    int dev = 0;
    if(cudaGetDevice(&dev) != cudaSuccess) {return;}
    const int dst = to_device ? dev : cudaCpuDeviceId;
    /* Ordered after all device work already queued (the device must be done with the pages
       before they go to the host), and complete before return (the host phase that follows
       touches them at once; a kernel launched after this finds them resident). */
    Kokkos::fence("gizmo_particles_prefetch");
    if(bytes_p) {cudaMemPrefetchAsync(P, bytes_p, dst, 0);}
    if(bytes_c) {cudaMemPrefetchAsync(CellP, bytes_c, dst, 0);}
    cudaError_t rc = cudaStreamSynchronize(0);
    if(rc != cudaSuccess) {(void) cudaGetLastError(); prefetch_enabled_ = 0; if(ThisTask == 0) {printf("Particle storage: bulk prefetch failed at %s (%s); switched off.\n", site, cudaGetErrorString(rc)); fflush(stdout);}}
    prefetch_side_ = prefetch_enabled_ ? (to_device ? 1 : 2) : 0;
    if(!to_device && with_cells && n_owned < 0) {tohost_arena_counter_ = g_gpu_arena_acquire_counter;}
    prefetch_seconds_ += timediff(t0, my_second());
    if(to_device) {prefetch_bytes_dev_ += (double) (bytes_p + bytes_c);} else {prefetch_bytes_host_ += (double) (bytes_p + bytes_c);}
    prefetch_calls_++;
#else
    (void) to_device;
#endif
}

extern "C" void gizmo_managed_prefetch_to_host(const void *ptr, size_t bytes, const char *site)
{
    if(!ptr || bytes == 0 || !particles_prefetch_enabled()) {return;}
#if defined(KOKKOS_ENABLE_CUDA)
    cudaPointerAttributes attr;
    if(cudaPointerGetAttributes(&attr, ptr) != cudaSuccess) {(void) cudaGetLastError(); return;}
    if(attr.type != cudaMemoryTypeManaged) {return;}   /* ordinary host or device memory: nothing to move */
    const double t0 = my_second();
    Kokkos::fence("gizmo_managed_prefetch_to_host");
    cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId, 0);
    cudaError_t rc = cudaStreamSynchronize(0);
    if(rc != cudaSuccess) {(void) cudaGetLastError(); if(ThisTask == 0) {printf("Particle storage: buffer prefetch failed at %s (%s).\n", site, cudaGetErrorString(rc)); fflush(stdout);}}
    prefetch_seconds_ += timediff(t0, my_second());
    prefetch_bytes_host_ += (double) bytes;
    prefetch_calls_++;
#else
    (void) site;
#endif
}

extern "C" void gizmo_managed_prefetch_to_device(const void *ptr, size_t bytes, const char *site)
{
    if(!ptr || bytes == 0 || !particles_prefetch_enabled()) {return;}
#if defined(KOKKOS_ENABLE_CUDA)
    cudaPointerAttributes attr;
    if(cudaPointerGetAttributes(&attr, ptr) != cudaSuccess) {(void) cudaGetLastError(); return;}
    if(attr.type != cudaMemoryTypeManaged) {return;}   /* ordinary host or device memory: nothing to move */
    int dev_ = 0;
    if(cudaGetDevice(&dev_) != cudaSuccess) {(void) cudaGetLastError(); return;}
    const double t0 = my_second();
    Kokkos::fence("gizmo_managed_prefetch_to_device");
    cudaMemPrefetchAsync(ptr, bytes, dev_, 0);
    cudaError_t rc = cudaStreamSynchronize(0);
    if(rc != cudaSuccess) {(void) cudaGetLastError(); if(ThisTask == 0) {printf("Particle storage: buffer prefetch failed at %s (%s).\n", site, cudaGetErrorString(rc)); fflush(stdout);}}
    prefetch_seconds_ += timediff(t0, my_second());
    prefetch_bytes_dev_ += (double) bytes;
    prefetch_calls_++;
#else
    (void) site;
#endif
}

/* GIZMO_UVM_KEEP_DEVICE: 0 off, 1 the corridor span (default). */
static int keep_device_level(void)
{
    static int level = -1;
    if(level < 0) {
        level = 1;
        const char *env = getenv("GIZMO_UVM_KEEP_DEVICE");
        if(env && env[0]) {level = atoi(env); if(level < 0) {level = 0;}}
        if(ThisTask == 0 && particles_prefetch_enabled()) {printf("Particle storage: keeping particles on the device between device passes: level %d.\n", level); fflush(stdout);}
    }
    return level;
}

extern "C" int gizmo_particles_keep_owned_on_device(const char *site)
{
    (void) site;
    if(keep_device_level() < 1 || !particles_prefetch_enabled() || prefetch_side_ != 1) {return 0;}
    owned_kept_on_device_ = 1;
    return 1;
}
extern "C" int gizmo_particles_owned_kept_on_device(void) {return owned_kept_on_device_;}

extern "C" void gizmo_particles_prefetch_ghosts_to_host(int n_owned, const char *site)
{
    if(!particles_prefetch_enabled() || !owned_kept_on_device_ || n_owned < 0 || n_owned >= NumPart || !P) {return;}
    if(ghostdev_range_live()) {return;}   /* a device-resident ghost range stays on the device */
#if defined(KOKKOS_ENABLE_CUDA)
    const size_t bytes_p = (size_t) (NumPart - n_owned) * sizeof(struct particle_data);
    const int c_end = (NumPart < All.MaxPartGas) ? NumPart : All.MaxPartGas;
    const size_t bytes_c = (CellP && c_end > n_owned) ? (size_t) (c_end - n_owned) * sizeof(struct gas_cell_data) : 0;
    const double t0 = my_second();
    Kokkos::fence("gizmo_particles_prefetch_ghosts_to_host");
    cudaMemPrefetchAsync(P + n_owned, bytes_p, cudaCpuDeviceId, 0);
    if(bytes_c) {cudaMemPrefetchAsync(CellP + n_owned, bytes_c, cudaCpuDeviceId, 0);}
    cudaError_t rc = cudaStreamSynchronize(0);
    if(rc != cudaSuccess) {   /* as particles_prefetch: report, switch prefetching off, leave nothing kept */
        (void) cudaGetLastError(); prefetch_enabled_ = 0; prefetch_side_ = 0; owned_kept_on_device_ = 0;
        if(ThisTask == 0) {printf("Particle storage: bulk prefetch failed at %s (%s); switched off.\n", site, cudaGetErrorString(rc)); fflush(stdout);}
        return;
    }
    prefetch_seconds_ += timediff(t0, my_second());
    prefetch_bytes_host_ += (double) (bytes_p + bytes_c);
    prefetch_calls_++;
#else
    (void) site;
#endif
}

/* Grow-only device scratch for the helpers below (released at exit with the context). */
static void *kept_scratch_ = NULL;
static size_t kept_scratch_bytes_ = 0;
static void *kept_scratch(size_t bytes)
{
    if(bytes > kept_scratch_bytes_) {
        if(kept_scratch_) {gizmo_gpu_free_device(kept_scratch_);}
        kept_scratch_bytes_ = bytes + bytes / 4;
        kept_scratch_ = gizmo_gpu_alloc_device(kept_scratch_bytes_, "kept_device_scratch");
        if(!kept_scratch_) {kept_scratch_bytes_ = 0;}
    }
    return kept_scratch_;
}

extern "C" long long gizmo_particles_count_not_at_time(const int *idx, int n, integertime t_now)
{
    if(n <= 0) {return 0;}
    int *d_idx = (int *) kept_scratch((size_t) n * sizeof(int));
    if(!d_idx) {   /* no scratch: answer on the host (correct, only slower) */
        long long c = 0;
        for(int k = 0; k < n; k++) {const int j = idx[k]; if(j >= 0 && j < NumPart && P[j].Ti_current != t_now) {c++;}}
        return c;
    }
    Kokkos::View<const int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h(idx, (size_t) n);
    Kokkos::View<int *, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> d(d_idx, (size_t) n);
    Kokkos::deep_copy(d, h);
    const struct particle_data *Pd = P;
    const int npart = NumPart;
    long long count = 0;
    Kokkos::parallel_reduce("gizmo_count_not_at_time", n, KOKKOS_LAMBDA(int k, long long &c) {
        const int j = d_idx[k];
        if(j >= 0 && j < npart && Pd[j].Ti_current != t_now) {c++;}
    }, count);
    Kokkos::fence("gizmo_count_not_at_time");
    return count;
}

/* Copies whole records as words; padding included, which a host struct copy may leave alone (never read). */
template <typename W>
KOKKOS_INLINE_FUNCTION void kept_copy_words(W *dst, const W *src, int nw) {for(int w = 0; w < nw; w++) {dst[w] = src[w];}}
template <typename W>
KOKKOS_INLINE_FUNCTION void kept_zero_words(W *dst, int nw) {for(int w = 0; w < nw; w++) {dst[w] = 0;}}

extern "C" void gizmo_particles_gather_send_records(const int *idx, int n, struct particle_data *dst_P, struct gas_cell_data *dst_CellP)
{
    if(n <= 0) {return;}
    static_assert(sizeof(struct particle_data) % sizeof(unsigned int) == 0 && sizeof(struct gas_cell_data) % sizeof(unsigned int) == 0,
                  "particle records are copied as 4-byte words");
    typedef unsigned int W;
    const int wp = (int) (sizeof(struct particle_data) / sizeof(W)), wc = (int) (sizeof(struct gas_cell_data) / sizeof(W));
    const size_t idx_bytes = ((size_t) n * sizeof(int) + 255) & ~(size_t) 255;
    const size_t p_bytes = (size_t) n * sizeof(struct particle_data), c_bytes = (size_t) n * sizeof(struct gas_cell_data);
    char *base = (char *) kept_scratch(idx_bytes + p_bytes + c_bytes);
    if(!base) {   /* no scratch: pack on the host, as the import does (correct, only slower) */
        for(int k = 0; k < n; k++) {
            const int j = idx[k];
            if(j < 0) {memset(&dst_P[k], 0, sizeof(struct particle_data)); memset(&dst_CellP[k], 0, sizeof(struct gas_cell_data)); continue;}
            dst_P[k] = P[j];
            if(P[j].Type == 0 && j < N_gas) {dst_CellP[k] = CellP[j];} else {memset(&dst_CellP[k], 0, sizeof(struct gas_cell_data));}
        }
        return;
    }
    int *d_idx = (int *) base;
    W *d_p = (W *) (base + idx_bytes);
    W *d_c = (W *) (base + idx_bytes + p_bytes);
    {
        Kokkos::View<const int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h(idx, (size_t) n);
        Kokkos::View<int *, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> d(d_idx, (size_t) n);
        Kokkos::deep_copy(d, h);
    }
    const W *Pw = (const W *) P;
    const W *Cw = (const W *) CellP;
    const struct particle_data *Pd = P;
    const int ngas = N_gas;
    Kokkos::parallel_for("gizmo_gather_send_records", n, KOKKOS_LAMBDA(int k) {
        const int j = d_idx[k];
        W *op = d_p + (size_t) k * wp;
        W *oc = d_c + (size_t) k * wc;
        if(j < 0) {kept_zero_words(op, wp); kept_zero_words(oc, wc); return;}
        kept_copy_words(op, Pw + (size_t) j * wp, wp);
        if(Pd[j].Type == 0 && j < ngas) {kept_copy_words(oc, Cw + (size_t) j * wc, wc);} else {kept_zero_words(oc, wc);}
    });
    Kokkos::fence("gizmo_gather_send_records");
    Kokkos::View<W *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hp((W *) dst_P, (size_t) n * wp), hc((W *) dst_CellP, (size_t) n * wc);
    Kokkos::View<const W *, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dp(d_p, (size_t) n * wp), dc(d_c, (size_t) n * wc);
    Kokkos::deep_copy(hp, dp);
    Kokkos::deep_copy(hc, dc);
}

static char *mirror_buf_ = NULL;
static size_t mirror_cap_ = 0, mirror_off_c_ = 0;
static int mirror_n_part_ = 0, mirror_n_cells_ = 0;

extern "C" int gizmo_particles_mirror_fill(int n_part, struct particle_data **P_dev, struct gas_cell_data **CellP_dev, const char *site)
{
    static int enabled = -1;
    if(enabled < 0) {
        const char *env = getenv("GIZMO_UVM_MIRROR");
        enabled = (env && env[0]) ? (atoi(env) != 0) : 1;
        if(ThisTask == 0 && particles_prefetch_enabled()) {printf("Particle storage: device copy for passes that do not write the particles is %s.\n", enabled ? "on" : "off"); fflush(stdout);}
    }
    /* Only from host-resident storage: an explicit host-to-device copy reads managed pages in place (a
       cudaMemcpyDefault copy migrates them to the device, one fault at a time). */
    if(!enabled || !particles_prefetch_enabled() || n_part <= 0 || !P) {return 0;}
    if(prefetch_side_ != 2 || owned_kept_on_device_ || ghostdev_range_live()) {return 0;}   /* the copy reads host-resident storage only */
#if defined(KOKKOS_ENABLE_CUDA)
    const int n_cells = (ghost_get_num_ghosts() > 0) ? ((n_part < All.MaxPartGas) ? n_part : All.MaxPartGas) : ((N_gas < n_part) ? N_gas : n_part);
    const size_t bytes_p = (size_t) n_part * sizeof(struct particle_data);
    const size_t bytes_c = (CellP && n_cells > 0) ? (size_t) n_cells * sizeof(struct gas_cell_data) : 0;
    const size_t off_c = (bytes_p + 255) & ~(size_t) 255;
    if(off_c + bytes_c > mirror_cap_) {
        if(mirror_buf_) {gizmo_gpu_free_device(mirror_buf_);}
        mirror_cap_ = (off_c + bytes_c) + (off_c + bytes_c) / 4;
        mirror_buf_ = (char *) gizmo_gpu_alloc_device(mirror_cap_, "particle_mirror");
        if(!mirror_buf_) {mirror_cap_ = 0; return 0;}
    }
    const double t0 = my_second();
    Kokkos::fence("gizmo_particles_mirror_fill");
    cudaMemcpyAsync(mirror_buf_, P, bytes_p, cudaMemcpyHostToDevice, 0);
    if(bytes_c) {cudaMemcpyAsync(mirror_buf_ + off_c, CellP, bytes_c, cudaMemcpyHostToDevice, 0);}
    cudaError_t rc = cudaStreamSynchronize(0);
    if(rc != cudaSuccess) {(void) cudaGetLastError(); if(ThisTask == 0) {printf("Particle storage: device copy failed at %s (%s).\n", site, cudaGetErrorString(rc)); fflush(stdout);} return 0;}
    prefetch_seconds_ += timediff(t0, my_second());
    prefetch_bytes_dev_ += (double) (bytes_p + bytes_c);
    prefetch_calls_++;
    mirror_off_c_ = off_c; mirror_n_part_ = n_part; mirror_n_cells_ = bytes_c ? n_cells : 0;
    *P_dev = (struct particle_data *) mirror_buf_;
    *CellP_dev = bytes_c ? (struct gas_cell_data *) (mirror_buf_ + off_c) : NULL;
    return 1;
#else
    (void) P_dev; (void) CellP_dev; (void) site;
    return 0;
#endif
}

extern "C" void gizmo_particles_mirror_copy_back_writes(void)
{
#if defined(KOKKOS_ENABLE_CUDA)
    const int n = mirror_n_part_;
    if(n <= 0 || !mirror_buf_) {return;}
    std::vector<int> w((size_t) n);
    int *d_w = (int *) kept_scratch((size_t) n * sizeof(int));
    const struct particle_data *Pm = (const struct particle_data *) mirror_buf_;
    if(d_w) {
        Kokkos::parallel_for("gizmo_mirror_wakeup", n, KOKKOS_LAMBDA(int j) {d_w[j] = Pm[j].wakeup;});
        Kokkos::fence("gizmo_mirror_wakeup");
        Kokkos::View<int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h(w.data(), (size_t) n);
        Kokkos::View<const int *, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> d(d_w, (size_t) n);
        Kokkos::deep_copy(h, d);
    } else {
        for(int j = 0; j < n; j++) {cudaMemcpy(&w[(size_t) j], &Pm[j].wakeup, sizeof(int), cudaMemcpyDeviceToHost);}
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n >= 65536)
#endif
    for(int j = 0; j < n; j++) {P[j].wakeup = w[(size_t) j];}
#if defined(HYDRO_MESHLESS_FINITE_VOLUME)
    const int nc = mirror_n_cells_;
    if(nc > 0) {
        std::vector<MyDouble> dm((size_t) nc);
        MyDouble *d_m = (MyDouble *) kept_scratch((size_t) nc * sizeof(MyDouble));
        const struct gas_cell_data *Cm = (const struct gas_cell_data *) (mirror_buf_ + mirror_off_c_);
        if(d_m) {
            Kokkos::parallel_for("gizmo_mirror_dmass", nc, KOKKOS_LAMBDA(int j) {d_m[j] = Cm[j].dMass;});
            Kokkos::fence("gizmo_mirror_dmass");
            Kokkos::View<MyDouble *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h(dm.data(), (size_t) nc);
            Kokkos::View<const MyDouble *, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> d(d_m, (size_t) nc);
            Kokkos::deep_copy(h, d);
        } else {
            for(int j = 0; j < nc; j++) {cudaMemcpy(&dm[(size_t) j], &Cm[j].dMass, sizeof(MyDouble), cudaMemcpyDeviceToHost);}
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(nc >= 65536)
#endif
        for(int j = 0; j < nc; j++) {CellP[j].dMass = dm[(size_t) j];}
    }
#endif
    mirror_n_part_ = 0; mirror_n_cells_ = 0;
#endif
}

/* ---- imported ghosts kept on the device (contract in gpu_particles_arena.h) ---- */

#include "gpu_ghostdev_config.h"
#if defined(GHOSTDEV_COMPILED) && defined(KOKKOS_ENABLE_CUDA)
#define GHOSTDEV_BUILT 1
#if defined(GHOSTDEV_AUDITED_CONFIG)
#define GHOSTDEV_DEFAULT_ON 1
#else
#define GHOSTDEV_DEFAULT_ON 0
#endif
#endif

static double ghostdev_seconds_ = 0, ghostdev_bytes_ = 0;
static long long ghostdev_calls_ = 0;

extern "C" int gizmo_ghostdev_enabled(void)
{
    static int enabled = -1;
    if(enabled < 0) {
        enabled = 0;
#if defined(GHOSTDEV_BUILT)
        enabled = GHOSTDEV_DEFAULT_ON;
        const char *env = getenv("GIZMO_GPU_GHOSTDEV");
        if(env && env[0]) {enabled = (atoi(env) != 0);}
        enabled = enabled && particles_prefetch_enabled() && NTask > 1;
        if(ThisTask == 0 && particles_prefetch_enabled()) {printf("Particle storage: imported hydro ghosts delivered to and kept on the device: %s.\n", enabled ? "on" : "off"); fflush(stdout);}
#endif
    }
    return enabled;
}

extern "C" void gizmo_ghostdev_set_live(int base, int n)
{
    if(n > 0) {ghostdev_base_ = base; ghostdev_count_ = n;}
    else {ghostdev_base_ = -1; ghostdev_count_ = 0;}
}
extern "C" int gizmo_ghostdev_live(void) {return ghostdev_range_live();}

extern "C" int gizmo_ghost_range_copy_to_device(int base, int n, const struct particle_data *src_P, const struct gas_cell_data *src_CellP)
{
    if(n <= 0) {return 0;}
#if defined(KOKKOS_ENABLE_CUDA)
    int dev = 0;
    if(cudaGetDevice(&dev) != cudaSuccess) {(void) cudaGetLastError(); return 1;}
    const size_t bytes_p = (size_t) n * sizeof(struct particle_data);
    const size_t bytes_c = (src_CellP && CellP) ? (size_t) n * sizeof(struct gas_cell_data) : 0;
    const double t0 = my_second();
    Kokkos::fence("gizmo_ghost_range_copy_to_device");   /* nothing still reads the slots being overwritten */
    /* A copy into host-resident managed pages migrates them a fault at a time; resident ones make this a no-op. The
       range shares its first page with the last owned record: synchronized below before anything else touches it. */
    cudaMemPrefetchAsync(P + base, bytes_p, dev, 0);
    if(bytes_c) {cudaMemPrefetchAsync(CellP + base, bytes_c, dev, 0);}
    cudaMemcpyAsync(P + base, src_P, bytes_p, cudaMemcpyHostToDevice, 0);
    if(bytes_c) {cudaMemcpyAsync(CellP + base, src_CellP, bytes_c, cudaMemcpyHostToDevice, 0);}
    const cudaError_t rc = cudaStreamSynchronize(0);
    if(rc != cudaSuccess) {
        (void) cudaGetLastError();
        printf("Particle storage: ghost copy to the device failed on task %d (%s); copied on the host.\n", ThisTask, cudaGetErrorString(rc));
        fflush(stdout);
        return 1;
    }
    ghostdev_seconds_ += timediff(t0, my_second());
    ghostdev_bytes_ += (double) (bytes_p + bytes_c);
    ghostdev_calls_++;
    return 0;
#else
    (void) base; (void) src_P; (void) src_CellP;
    return 1;
#endif
}

extern "C" void gizmo_ghostdev_stats(double *seconds, double *bytes, long long *calls)
{
    if(seconds) {*seconds = ghostdev_seconds_;}
    if(bytes) {*bytes = ghostdev_bytes_;}
    if(calls) {*calls = ghostdev_calls_;}
}

#define GHOST_PINNED_SLOTS 6
static void *ghost_pinned_[GHOST_PINNED_SLOTS] = {};
static size_t ghost_pinned_bytes_[GHOST_PINNED_SLOTS] = {};
extern "C" void *gizmo_ghost_pinned_buffer(int slot, size_t bytes)
{
    static const char *labels[GHOST_PINNED_SLOTS] = {"ghost_recv_P_pinned", "ghost_recv_CellP_pinned", "ghost_send_P_pinned",
                                                     "ghost_send_CellP_pinned", "ghost_send_trim_pinned", "ghost_recv_trim_pinned"};
    if(slot < 0 || slot >= GHOST_PINNED_SLOTS) {return NULL;}
    if(bytes == 0) {bytes = 1;}
    if(bytes > ghost_pinned_bytes_[slot]) {
        if(ghost_pinned_[slot]) {gizmo_gpu_free_pinned_host(ghost_pinned_[slot]);}
        const size_t want = bytes + bytes / 4;
        ghost_pinned_[slot] = gizmo_gpu_alloc_pinned_host(want, labels[slot]);
        ghost_pinned_bytes_[slot] = ghost_pinned_[slot] ? want : 0;
    }
    return ghost_pinned_[slot];
}

extern "C" void gizmo_ghost_range_zero_wakeup_on_device(int base, int n)
{
    if(n <= 0) {return;}
    struct particle_data *Pd = P;
    Kokkos::parallel_for("gizmo_ghost_zero_wakeup", Kokkos::RangePolicy<>(base, base + n), KOKKOS_LAMBDA(int j) {Pd[j].wakeup = 0;});
    Kokkos::fence("gizmo_ghost_zero_wakeup");
}

extern "C" void gizmo_ghost_range_gather_wakeup(int base, int n, int *dst)
{
    if(n <= 0) {return;}
    int *d_w = (int *) kept_scratch((size_t) n * sizeof(int));
    if(!d_w) {for(int g = 0; g < n; g++) {dst[g] = P[base + g].wakeup;} return;}   /* no scratch: read on the host (correct, only slower) */
    const struct particle_data *Pd = P;
    Kokkos::parallel_for("gizmo_ghost_gather_wakeup", n, KOKKOS_LAMBDA(int g) {d_w[g] = Pd[base + g].wakeup;});
    Kokkos::fence("gizmo_ghost_gather_wakeup");
    Kokkos::View<int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h(dst, (size_t) n);
    Kokkos::View<const int *, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> d(d_w, (size_t) n);
    Kokkos::deep_copy(h, d);
}

struct GhostScatterSegments {int n, cell[8], obj_off[8], rec_off[8], words[8];};
extern "C" int gizmo_ghost_range_scatter_records(int base, int n, const void *records, int rec_bytes, int nseg,
                                                 const int *cell, const int *obj_off, const int *rec_off, const int *words)
{
    if(n <= 0 || nseg <= 0) {return 0;}
    typedef unsigned int W;
    const size_t bytes = (size_t) n * (size_t) rec_bytes;
    char *d_rec = (nseg <= 8 && rec_bytes % (int) sizeof(W) == 0) ? (char *) kept_scratch(bytes) : NULL;
    if(!d_rec) {   /* no scratch: scatter on the host (correct, only slower) */
        for(int g = 0; g < n; g++) {
            const char *rec = (const char *) records + (size_t) g * rec_bytes;
            for(int s = 0; s < nseg; s++) {
                char *obj = cell[s] ? (char *) &CellP[base + g] : (char *) &P[base + g];
                memcpy(obj + obj_off[s], rec + rec_off[s], (size_t) words[s] * sizeof(W));
            }
        }
        return 1;
    }
    struct GhostScatterSegments seg = {};
    seg.n = nseg;
    for(int s = 0; s < nseg; s++) {seg.cell[s] = cell[s]; seg.obj_off[s] = obj_off[s] / (int) sizeof(W); seg.rec_off[s] = rec_off[s] / (int) sizeof(W); seg.words[s] = words[s];}
    {
        Kokkos::View<const char *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h((const char *) records, bytes);
        Kokkos::View<char *, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> d(d_rec, bytes);
        Kokkos::deep_copy(d, h);
    }
    W *Pw = (W *) (P + base);
    W *Cw = CellP ? (W *) (CellP + base) : NULL;
    const int wp = (int) (sizeof(struct particle_data) / sizeof(W)), wc = (int) (sizeof(struct gas_cell_data) / sizeof(W));
    const int wr = rec_bytes / (int) sizeof(W);
    const W *Rw = (const W *) d_rec;
    Kokkos::parallel_for("gizmo_ghost_scatter_records", n, KOKKOS_LAMBDA(int g) {
        const W *rec = Rw + (size_t) g * wr;
        for(int s = 0; s < seg.n; s++) {
            W *obj = seg.cell[s] ? (Cw + (size_t) g * wc) : (Pw + (size_t) g * wp);
            for(int w = 0; w < seg.words[s]; w++) {obj[seg.obj_off[s] + w] = rec[seg.rec_off[s] + w];}
        }
    });
    Kokkos::fence("gizmo_ghost_scatter_records");
    return 0;
}

extern "C" int gizmo_particles_gather_send_segments(const int *idx, int n, void *dst, int rec_bytes, int nseg,
                                                    const int *cell, const int *obj_off, const int *rec_off, const int *words)
{
    if(n <= 0) {return 0;}
    typedef unsigned int W;
    const size_t idx_bytes = ((size_t) n * sizeof(int) + 255) & ~(size_t) 255;
    const size_t rec_total = (size_t) n * (size_t) rec_bytes;
    char *base = (nseg <= 8 && rec_bytes % (int) sizeof(W) == 0) ? (char *) kept_scratch(idx_bytes + rec_total) : NULL;
    if(!base) {   /* no scratch: pack on the host (correct, only slower) */
        for(int k = 0; k < n; k++) {
            char *rec = (char *) dst + (size_t) k * rec_bytes;
            memset(rec, 0, (size_t) rec_bytes);
            const int j = idx[k];
            if(j < 0) {continue;}
            const int gas = (P[j].Type == 0 && j < N_gas);
            for(int s = 0; s < nseg; s++) {
                if(cell[s] && !gas) {continue;}
                const char *obj = cell[s] ? (const char *) &CellP[j] : (const char *) &P[j];
                memcpy(rec + rec_off[s], obj + obj_off[s], (size_t) words[s] * sizeof(W));
            }
        }
        return 1;
    }
    int *d_idx = (int *) base;
    W *d_rec = (W *) (base + idx_bytes);
    {
        Kokkos::View<const int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h(idx, (size_t) n);
        Kokkos::View<int *, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> d(d_idx, (size_t) n);
        Kokkos::deep_copy(d, h);
    }
    struct GhostScatterSegments seg = {};
    seg.n = nseg;
    for(int s = 0; s < nseg; s++) {seg.cell[s] = cell[s]; seg.obj_off[s] = obj_off[s] / (int) sizeof(W); seg.rec_off[s] = rec_off[s] / (int) sizeof(W); seg.words[s] = words[s];}
    const W *Pw = (const W *) P;
    const W *Cw = (const W *) CellP;
    const struct particle_data *Pd = P;
    const int ngas = N_gas;
    const int wp = (int) (sizeof(struct particle_data) / sizeof(W)), wc = (int) (sizeof(struct gas_cell_data) / sizeof(W));
    const int wr = rec_bytes / (int) sizeof(W);
    Kokkos::parallel_for("gizmo_gather_send_segments", n, KOKKOS_LAMBDA(int k) {
        W *rec = d_rec + (size_t) k * wr;
        for(int w = 0; w < wr; w++) {rec[w] = 0;}
        const int j = d_idx[k];
        if(j < 0) {return;}
        const int gas = (Pd[j].Type == 0 && j < ngas);
        for(int s = 0; s < seg.n; s++) {
            if(seg.cell[s] && !gas) {continue;}
            const W *obj = seg.cell[s] ? (Cw + (size_t) j * wc) : (Pw + (size_t) j * wp);
            for(int w = 0; w < seg.words[s]; w++) {rec[seg.rec_off[s] + w] = obj[seg.obj_off[s] + w];}
        }
    });
    Kokkos::fence("gizmo_gather_send_segments");
    Kokkos::View<char *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h((char *) dst, rec_total);
    Kokkos::View<const char *, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> d((const char *) d_rec, rec_total);
    Kokkos::deep_copy(h, d);
    return 0;
}

extern "C" void gizmo_ghost_range_gather_records(int base, int n, void *records, int rec_bytes, int nseg,
                                                 const int *cell, const int *obj_off, const int *rec_off, const int *bytes)
{
    if(n <= 0 || nseg <= 0) {return;}
    const size_t total = (size_t) n * (size_t) rec_bytes;
    char *d_rec = (nseg <= 8) ? (char *) kept_scratch(total) : NULL;
    if(!d_rec) {   /* no scratch: read on the host (correct, only slower) */
        for(int g = 0; g < n; g++) {
            char *rec = (char *) records + (size_t) g * rec_bytes;
            for(int s = 0; s < nseg; s++) {
                const char *obj = cell[s] ? (const char *) &CellP[base + g] : (const char *) &P[base + g];
                memcpy(rec + rec_off[s], obj + obj_off[s], (size_t) bytes[s]);
            }
        }
        return;
    }
    struct GhostScatterSegments seg = {};   /* words[] holds bytes here */
    seg.n = nseg;
    for(int s = 0; s < nseg; s++) {seg.cell[s] = cell[s]; seg.obj_off[s] = obj_off[s]; seg.rec_off[s] = rec_off[s]; seg.words[s] = bytes[s];}
    const char *Pb = (const char *) (P + base);
    const char *Cb = CellP ? (const char *) (CellP + base) : NULL;
    const size_t sp = sizeof(struct particle_data), sc = sizeof(struct gas_cell_data);
    const int rb = rec_bytes;
    Kokkos::parallel_for("gizmo_ghost_gather_records", n, KOKKOS_LAMBDA(int g) {
        char *rec = d_rec + (size_t) g * rb;
        for(int s = 0; s < seg.n; s++) {
            const char *obj = seg.cell[s] ? (Cb + (size_t) g * sc) : (Pb + (size_t) g * sp);
            for(int b = 0; b < seg.words[s]; b++) {rec[seg.rec_off[s] + b] = obj[seg.obj_off[s] + b];}
        }
    });
    Kokkos::fence("gizmo_ghost_gather_records");
    Kokkos::View<char *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h((char *) records, total);
    Kokkos::View<const char *, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> d(d_rec, total);
    Kokkos::deep_copy(h, d);
}

extern "C" int gizmo_particles_prefetch_side(void) {return prefetch_side_;}
extern "C" int gizmo_particles_back_to_host_wanted(long long n)
{
    if(n * 8 >= (long long) NumPart) {return 1;}
    static int always = -1;
    if(always < 0) {const char *env = getenv("GIZMO_UVM_TOHOST_ALWAYS"); always = (env && env[0]) ? (atoi(env) != 0) : 1;}
    return always && g_gpu_arena_acquire_counter != tohost_arena_counter_;
}
extern "C" void gizmo_particles_prefetch_to_device(const char *site) {particles_prefetch(1, site);}
extern "C" void gizmo_particles_prefetch_P_to_device(const char *site) {particles_prefetch(1, site, 0);}
extern "C" void gizmo_particles_prefetch_to_host(const char *site) {particles_prefetch(0, site);}
extern "C" void gizmo_particles_prefetch_owned_to_device(int n_owned, const char *site) {particles_prefetch(1, site, 1, n_owned);}
extern "C" void gizmo_particles_prefetch_P_to_host(const char *site) {particles_prefetch(0, site, 0);}
extern "C" int gizmo_treehost_begin(long long n_active)
{
    treehost_armed_ = 0;
#if defined(GIZMO_GPU_TREEHOST_SUPPORTED) && defined(KOKKOS_ENABLE_CUDA)
    static int enabled = -1;
    if(enabled < 0) {
        const char *env = getenv("GIZMO_GPU_TREEHOST");
        enabled = (env && env[0]) ? (atoi(env) != 0) : 1;
        if(ThisTask == 0 && particles_prefetch_enabled()) {printf("Particle storage: tree build to density pass on the device (GIZMO_GPU_TREEHOST) is %s.\n", enabled ? "on" : "off"); fflush(stdout);}
    }
    if(enabled && particles_prefetch_enabled() && NumPart > 0 && All.TotN_gas > 0 && n_active * 8 >= (long long) NumPart) {treehost_armed_ = 1;}
#else
    (void) n_active;
#endif
    return treehost_armed_;
}
extern "C" int  gizmo_treehost_armed(void) {return treehost_armed_;}
extern "C" void gizmo_treehost_disarm(void) {treehost_armed_ = 0;}
extern "C" void gizmo_treehost_open(void)
{
    if(!treehost_armed_ || prefetch_side_ != 1) {return;}
    treehost_span_ = 1; treehost_cells_ = 0;
}
extern "C" int  gizmo_treehost_span_active(void) {return treehost_span_;}
extern "C" void gizmo_treehost_cells_to_device(const char *site)
{
    if(!treehost_span_ || treehost_cells_) {return;}
    particles_prefetch(1, site, 1, -1, 0);
    treehost_cells_ = 1;
}
extern "C" int gizmo_treehost_consume(const char *site)
{
    if(!treehost_span_) {return 0;}
    if(!treehost_cells_) {particles_prefetch(1, site, 1, -1, 0);}
    treehost_span_ = 0; treehost_cells_ = 0;
    return 1;
}
extern "C" void gizmo_treehost_end(const char *site)
{
    if(!treehost_span_) {return;}
    particles_prefetch(0, site, treehost_cells_ ? 1 : 0);   /* ends the span */
}

static void *treehost_scratch_[4] = {NULL, NULL, NULL, NULL};
static size_t treehost_scratch_bytes_[4] = {0, 0, 0, 0};
extern "C" void *gizmo_treehost_device_scratch(int slot, size_t bytes)
{
    if(slot < 0 || slot >= 4) {return NULL;}
    if(bytes > treehost_scratch_bytes_[slot]) {
        if(treehost_scratch_[slot]) {gizmo_gpu_free_device(treehost_scratch_[slot]);}
        treehost_scratch_bytes_[slot] = bytes + bytes / 4;
        treehost_scratch_[slot] = gizmo_gpu_alloc_device(treehost_scratch_bytes_[slot], "treehost_scratch");
        if(!treehost_scratch_[slot]) {treehost_scratch_bytes_[slot] = 0;}
    }
    return treehost_scratch_[slot];
}
/* The index list on the device (scratch slot 0), or NULL when the scratch cannot be had. */
static int *treehost_upload_indices(const int *idx, int n)
{
    int *d_idx = (int *) gizmo_treehost_device_scratch(0, (size_t) n * sizeof(int));
    if(!d_idx) {return NULL;}
    Kokkos::View<const int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h(idx, (size_t) n);
    Kokkos::View<int *, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> d(d_idx, (size_t) n);
    Kokkos::deep_copy(d, h);
    return d_idx;
}

extern "C" void gizmo_particles_copy_oldacc_on_device(int n)
{
    if(n <= 0) {return;}
    struct particle_data *Pd = P;
    Kokkos::parallel_for("treehost_oldacc", n, KOKKOS_LAMBDA(int i) {Pd[i].OldAcc = Pd[i].OldAcc_LatestWalk;});
    Kokkos::fence("treehost_oldacc");
}

extern "C" void gizmo_particles_zero_gravaccel_on_device(const int *idx, int n)
{
    if(n <= 0) {return;}
    int *d_idx = treehost_upload_indices(idx, n);
    if(!d_idx) {for(int k = 0; k < n; k++) {P[idx[k]].GravAccel = {};} return;}   /* no scratch: on the host (correct, only slower) */
    struct particle_data *Pd = P;
    Kokkos::parallel_for("treehost_zero_gravaccel", n, KOKKOS_LAMBDA(int k) {
        const int i = d_idx[k];
        Pd[i].GravAccel[0] = 0; Pd[i].GravAccel[1] = 0; Pd[i].GravAccel[2] = 0;
    });
    Kokkos::fence("treehost_zero_gravaccel");
}

extern "C" void gizmo_particles_count_active_gas_on_device(const int *idx, int n, long long *n_gas_with_mass, long long *n_gas)
{
    *n_gas_with_mass = 0; *n_gas = 0;
    if(n <= 0) {return;}
    int *d_idx = treehost_upload_indices(idx, n);
    if(!d_idx) {   /* no scratch: on the host (correct, only slower) */
        for(int k = 0; k < n; k++) {const int i = idx[k]; if(P[i].Type != 0) {continue;} (*n_gas)++; if(P[i].Mass > 0) {(*n_gas_with_mass)++;}}
        return;
    }
    const struct particle_data *Pd = P;
    /* Both counts in one sum: gas in the high 32 bits, gas with mass in the low ones (each count is below 2^31). */
    long long packed = 0;
    Kokkos::parallel_reduce("treehost_count_active_gas", n, KOKKOS_LAMBDA(int k, long long &c) {
        const int i = d_idx[k];
        if(Pd[i].Type != 0) {return;}
        c += (1LL << 32) + ((Pd[i].Mass > 0) ? 1 : 0);
    }, packed);
    Kokkos::fence("treehost_count_active_gas");
    *n_gas = packed >> 32;
    *n_gas_with_mass = packed & 0xffffffffLL;
}

extern "C" void gizmo_particles_prefetch_stats(double *seconds, double *bytes_to_device, double *bytes_to_host, long long *calls)
{
    if(seconds) {*seconds = prefetch_seconds_;}
    if(bytes_to_device) {*bytes_to_device = prefetch_bytes_dev_;}
    if(bytes_to_host) {*bytes_to_host = prefetch_bytes_host_;}
    if(calls) {*calls = prefetch_calls_;}
}

extern "C" void gpu_particles_arena_mark_clean_after_scatter(const char *site)
{
    (void)site;
}

extern "C" void gpu_particles_arena_refresh_from_host(int min_capacity,
                                                     struct particle_data *P_host,
                                                     struct gas_cell_data *CellP_host,
                                                     const char *site)
{
    (void)min_capacity; (void)P_host; (void)CellP_host; (void)site;
}

/* ---- compact staging buffers (see the contract in gpu_particles_arena.h) ---- */

extern "C" int particle_staging_acquire(struct ParticleStagingBatch *batch, int capacity)
{
    if(!batch) {return 0;}
    /* Free anything the batch is already holding, so re-acquiring cannot strand the
       previous buffers. Requires the caller to have zero-initialised it once. */
    particle_staging_release(batch);
    if(capacity <= 0) {capacity = 1;}
    /* new[] rather than malloc for the host side: particle_data is over-aligned (32
       bytes, measured, against a max_align_t of 8) and only the C++ allocator honours
       that; malloc'd storage faults on the aligned vector moves the compiler emits. */
    try {
        batch->host_P    = new struct particle_data[(size_t)capacity];
        batch->host_Cell = new struct gas_cell_data[(size_t)capacity];
        batch->index     = new int[(size_t)capacity];
    }
    catch(const std::exception &) { particle_staging_release(batch); return 0; }
    batch->dev_P    = (struct particle_data *) gizmo_gpu_alloc_device(
                          (size_t)capacity * sizeof(struct particle_data), "particle_staging_P");
    batch->dev_Cell = (struct gas_cell_data *) gizmo_gpu_alloc_device(
                          (size_t)capacity * sizeof(struct gas_cell_data), "particle_staging_Cell");
    if(!batch->host_P || !batch->host_Cell || !batch->index || !batch->dev_P || !batch->dev_Cell)
        { particle_staging_release(batch); return 0; }
    batch->capacity = capacity;
    return 1;
}

extern "C" void particle_staging_release(struct ParticleStagingBatch *batch)
{
    if(!batch) {return;}
    delete[] batch->host_P;    batch->host_P    = NULL;
    delete[] batch->host_Cell; batch->host_Cell = NULL;
    delete[] batch->index;     batch->index     = NULL;
    if(batch->dev_P)    {Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(batch->dev_P);    batch->dev_P    = NULL;}
    if(batch->dev_Cell) {Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(batch->dev_Cell); batch->dev_Cell = NULL;}
    batch->capacity = batch->count = batch->gas_count = 0;
}

using UmHostP = Kokkos::View<struct particle_data*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using UmHostC = Kokkos::View<struct gas_cell_data*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using UmDevP  = Kokkos::View<struct particle_data*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using UmDevC  = Kokkos::View<struct gas_cell_data*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

extern "C" int particle_staging_gather(struct ParticleStagingBatch *batch, const int *idx, int n,
                                       struct particle_data *pp, struct gas_cell_data *cell)
{
    if(!batch) {return 0;}
    batch->count = batch->gas_count = 0;
    if(!idx || !pp || n <= 0) {return 0;}
    if(n > batch->capacity) {
        /* Stage nothing. Staging a prefix and letting the caller keep driving its own
           loops would feed the kernel slots that were never written, and scatter back
           through index entries that were never written -- an out-of-bounds write into
           the particle arrays, which can land before the controlled stop drains. */
        char msg[192];
        snprintf(msg, sizeof(msg),
                 "particle staging: asked to stage %d elements into %d slots",
                 n, batch->capacity);
        gizmo_request_controlled_stop(7718, msg, __FILE__, __LINE__, __FUNCTION__);
        return 0;
    }

    /* Order the slots so the gas comes first. The physics touches a cell only for gas,
       and CellP is not allocated beyond the gas particles, so this is what makes the
       cell staging both correct and a contiguous copy. Reordering is free of
       consequence here: the drift and cooling bodies act on one particle each, with no
       pair coupling and no random draws. */
    int n_gas = 0, tail = n, k;
    for(k = 0; k < n; k++)
    {
        if(pp[idx[k]].Type == 0) {batch->index[n_gas++] = idx[k];}
        else                     {batch->index[--tail]  = idx[k];}
    }
    /* CellP is null when the run has no gas at all, which is a legal configuration
       (a pure N-body run allocates no gas cell storage). Nothing here needs it in that
       case: the partition found no gas slots, and the drift body reaches a cell only
       under its Type==0 guard. Gas with no CellP to read is a different thing -- an
       inconsistency the caller cannot recover from -- so that one is refused. */
    if(n_gas > 0 && !cell) {
        gizmo_request_controlled_stop(7719, "particle staging: gas particles staged but CellP is null",
                                      __FILE__, __LINE__, __FUNCTION__);
        return 0;
    }

    batch->count     = n;
    batch->gas_count = n_gas;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int j = 0; j < n; j++)
    {
        const int i = batch->index[j];
        batch->host_P[j] = pp[i];
        if(j < n_gas) {batch->host_Cell[j] = cell[i];}
    }

    Kokkos::deep_copy(UmDevP(batch->dev_P, (size_t)n), UmHostP(batch->host_P, (size_t)n));
    if(n_gas > 0) {Kokkos::deep_copy(UmDevC(batch->dev_Cell, (size_t)n_gas), UmHostC(batch->host_Cell, (size_t)n_gas));}
    return 1;
}

extern "C" void particle_staging_scatter(struct ParticleStagingBatch *batch,
                                        struct particle_data *pp, struct gas_cell_data *cell)
{
    if(!batch || !pp || batch->count <= 0) {return;}
    const int n = batch->count, n_gas = batch->gas_count;
    if(n_gas > 0 && !cell) {return;}   /* gather refuses this state; never reached */

    Kokkos::deep_copy(UmHostP(batch->host_P, (size_t)n), UmDevP(batch->dev_P, (size_t)n));
    if(n_gas > 0) {Kokkos::deep_copy(UmHostC(batch->host_Cell, (size_t)n_gas), UmDevC(batch->dev_Cell, (size_t)n_gas));}

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int j = 0; j < n; j++)
    {
        const int i = batch->index[j];
        pp[i] = batch->host_P[j];
        if(j < n_gas) {cell[i] = batch->host_Cell[j];}
    }
}

/* See the header for why the storage is per-caller and why the refresh is
   unconditional. */
extern "C" int drift_kick_table_mirror_refresh(double **storage, struct DriftKickTableView *view)
{
    if(!storage || !view) {return 1;}

    *view = drift_kick_table_view(NULL, NULL, 0.0, 0.0, All.Timebase_interval,
                                  All.ComovingIntegrationOn ? 1 : 0);
    if(!view->comoving) {return 0;}   /* tables never built; the elapsed-time branch needs none of them */

    if(!*storage) {
        *storage = (double *) gizmo_gpu_alloc_shared(2 * DRIFT_TABLE_LENGTH * sizeof(double), NULL);
        if(!*storage) {
            gizmo_request_controlled_stop(929701, "drift/gravkick table mirror allocation failed",
                                          __FILE__, __LINE__, __FUNCTION__);
            return 1;
        }
    }
    for(int i = 0; i < DRIFT_TABLE_LENGTH; i++) {
        (*storage)[i]                      = DriftTable[i];
        (*storage)[DRIFT_TABLE_LENGTH + i] = GravKickTable[i];
    }
    *view = drift_kick_table_view(*storage, *storage + DRIFT_TABLE_LENGTH,
                                  DriftTable_logTimeBegin, DriftTable_logTimeMax,
                                  All.Timebase_interval, 1);
    return 0;
}

extern "C" void gpu_particles_arena_release(void)
{
    /* P/CellP storage is owned by allocate.cc and persists to process exit;
     * the arena does not own it under UVM-canonical, so nothing to free. */
    arena_P         = NULL;
    arena_CellP     = NULL;
    arena_capacity_ = 0;
    arena_valid_    = 0;
}

/* ---------------------------------------------------------------------------------------
 * Placement policy for the bulk managed particle arrays (P and CellP).
 *
 * On AMD GPUs these arrays live in HIP managed memory and are demand-paged: a page moves
 * to whichever side touched it last. Every timestep both sides sweep them -- the host in
 * domain exchange, Peano-Hilbert ordering, ghost writeback and merge/split, the device in
 * the tree data path, the neighbour walks and the physics kernels -- so the traffic
 * pattern decides whether the pages settle anywhere at all.
 *
 * Which way that falls depends on how much reuse each page gets, and measurement on
 * Frontier shows the two regimes cleanly. On a SMALL footprint the device's repeated walks
 * give each page enough reuse that it settles device-resident, and biasing the pages toward
 * the host merely turns those reads into permanent Infinity-Fabric traffic: a 1e7-particle
 * run loses 2-5%. On a LARGE footprint no page gets that reuse, every cycle re-migrates the
 * array wholesale, and the migration never converges: a 4e7-particle run gains ~10% from a
 * stable host placement, and a 291.8-million-particle restart on 16 nodes gains a factor of
 * 4.2 on the first density pass -- without it that job spent its entire 30-minute wall on
 * startup and reached one sync-point, against 78 with it. Adding nodes does not help,
 * because the page count is set by the problem size and not by the rank count.
 *
 * So the policy is chosen from the per-rank footprint rather than applied unconditionally,
 * and it is applied only to P and CellP. It must NOT be extended to the tree's device
 * mirror, which is read tens of thousands of times per build and is unambiguously
 * device-owned; the same advice applied there costs ~43 s in the gravity walk alone.
 *
 * What the advice does is set a PREFERRED location and map the other side in as an
 * accessor. It is a placement bias, not a prohibition on migrating, and it does not change
 * what any code may read or write. Correctness never depends on it: if the runtime declines
 * the hint the run is slower, and nothing else changes.
 *
 * NVIDIA GPUs split by memory model. A PCIe-attached GPU (A100, H100 PCIe) demand-pages managed
 * memory between host and device as above, but measured there the device side should own it: on
 * a 12M-particle cosmological zoom on Lonestar6 A100 nodes (6 ranks, ~7 GB of particle storage
 * per rank) device-preferred placement ran 5% faster than no hint, while host-preferred gained in
 * the tree build and the gradient and force loops and gave all of it back in the density walk,
 * the kicks and the sinks, finishing level with no hint. So a PCIe-attached GPU prefers the
 * device at any footprint. A Grace-Hopper node reads managed memory coherently from both sides;
 * there the default is no hint, since a host preference would hold the arrays in the Grace
 * memory, which the GPU reaches at a fraction of its own memory's bandwidth. An explicit
 * override applies on both.
 * -------------------------------------------------------------------------------------- */

/*! Footprint above which host-preferred placement is chosen, as total per-rank bytes of P
 *  plus CellP. Empirical, from the three Frontier workloads above, at 64 to 128 ranks: the
 *  4e7-particle case measures ~2.3 GB per rank and wins, the 3e8-particle case ~9 GB per rank
 *  and wins outright, and the 1e7-particle case -- several times smaller again, and with a
 *  much smaller gas fraction -- loses. That leaves a wide window rather than a boundary, so
 *  the exact value is not delicate; it is a first cut to be revisited as production-size
 *  measurements accumulate, and a run can be forced either way to measure it.
 *
 *  Note that this is ALLOCATED CAPACITY, not the count of live particles, so it carries
 *  PartAllocFactor with it -- as do the three figures above, which were read from the
 *  allocations themselves, so the cutoff and the calibration are in the same units. Live
 *  particles are the better predictor of how many pages actually move, and the two part
 *  company on a run with an unusually generous allocation factor; a run near the cutoff
 *  should be measured both ways rather than reasoned about. */
#define PARTICLE_STORAGE_HOST_PREFERRED_MIN_BYTES ((size_t) 1024 * 1024 * 1024)

/*! ⚠ THE HOST-PREFERRED PLACEMENT IS A STOPGAP, NOT A DESIGN. It is here because the largest
 *  runs would otherwise not complete, and it is accepted on those terms alone. Keeping the bulk
 *  particle storage on the host is the opposite of what this port is for: every routine moved
 *  onto the device has to reach across the fabric for it, so the hint buys a large win today by
 *  making a larger one harder to reach. It is a debt to be repaid by putting enough of the
 *  consumers on the device that the storage can follow them, not a setting to tune around.
 *
 *  ⛔ THEREFORE: a single device flip measured against this placement is NOT a verdict on that
 *  flip. Priced one at a time against host-resident storage, every one of them loses, because
 *  each pays the fabric crossing alone while the saving only appears once enough of them move
 *  together. Any such measurement must be paired with an arm that moves the placement too --
 *  which is what the DEVICE setting below exists for.
 *
 *  Where this rank's particle storage should prefer to live. `particle_arena_bytes` is the total
 *  for P plus CellP, so the two arrays always decide together and a gas-free run is judged on P
 *  alone. GPU_PARTICLE_STORAGE_PLACEMENT overrides the decision for validation and tuning;
 *  leaving it unset is the production path. */
#define PARTICLE_STORAGE_PLACEMENT_NONE   0   /* no hint: pages settle wherever they are used */
#define PARTICLE_STORAGE_PLACEMENT_HOST   1   /* prefer host, map the device in as an accessor */
#define PARTICLE_STORAGE_PLACEMENT_DEVICE 2   /* prefer device, map the host in as an accessor */

#if defined(KOKKOS_ENABLE_CUDA)
/*! Whether the host reads and writes CUDA managed memory in place (Grace-Hopper) rather than through page
 *  migration (a PCIe-attached GPU). Storage outside CUDA managed memory is not paged at all and counts as
 *  coherent; an attribute that cannot be read counts as not coherent. */
static int cuda_managed_memory_is_host_coherent(void)
{
    static int coherent = -1;
    if(coherent < 0) {
        int attr = 0, dev = 0;
        if(!std::is_same<GIZMO_KOKKOS_SHARED_SPACE, Kokkos::CudaUVMSpace>::value) {coherent = 1;}
        else if(cudaGetDevice(&dev) == cudaSuccess &&
                cudaDeviceGetAttribute(&attr, cudaDevAttrDirectManagedMemAccessFromHost, dev) == cudaSuccess) {coherent = attr ? 1 : 0;}
        else {coherent = 0;}
    }
    return coherent;
}
#endif

static int particle_storage_placement(size_t particle_arena_bytes)
{
#if defined(GPU_PARTICLE_STORAGE_PLACEMENT)
    (void) particle_arena_bytes;
    /* Anything outside the three named values keeps the original meaning of a nonzero setting,
     * so a Config carrying an older value still selects what it always selected. */
    return (GPU_PARTICLE_STORAGE_PLACEMENT == PARTICLE_STORAGE_PLACEMENT_NONE)
             ? PARTICLE_STORAGE_PLACEMENT_NONE
         : (GPU_PARTICLE_STORAGE_PLACEMENT == PARTICLE_STORAGE_PLACEMENT_DEVICE)
             ? PARTICLE_STORAGE_PLACEMENT_DEVICE
             : PARTICLE_STORAGE_PLACEMENT_HOST;
#else
#if defined(KOKKOS_ENABLE_CUDA)
    (void) particle_arena_bytes;
    return cuda_managed_memory_is_host_coherent() ? PARTICLE_STORAGE_PLACEMENT_NONE : PARTICLE_STORAGE_PLACEMENT_DEVICE;
#else
    return (particle_arena_bytes >= PARTICLE_STORAGE_HOST_PREFERRED_MIN_BYTES)
             ? PARTICLE_STORAGE_PLACEMENT_HOST : PARTICLE_STORAGE_PLACEMENT_NONE;
#endif
#endif
}

/*! Ask the driver for one placement on one managed buffer: a preferred location on one side and
 *  in-place access from the other, so whichever side is preferred, the other reaches the pages
 *  without moving them. The two requests are independent and either can be refused alone, so each
 *  outcome is returned as text for the caller's report. Returns 1 if both were accepted, 0 if
 *  either was refused, and -1 where nothing was asked of the driver (no GPU, no current device,
 *  or storage that is not managed memory), so the caller reports neither a hint nor a refusal. */
static int managed_memory_advise(void *p, size_t nbytes, int placement, const char **pref_result, const char **access_result)
{
    *pref_result = *access_result = "not applicable";
    const int on_device = (placement == PARTICLE_STORAGE_PLACEMENT_DEVICE);
#if defined(KOKKOS_ENABLE_HIP)
    int dev = 0;
    if(hipGetDevice(&dev) != hipSuccess) {return -1;}   /* nothing was asked of the driver */
    hipError_t rc_pref = hipMemAdvise(p, nbytes, hipMemAdviseSetPreferredLocation, on_device ? dev : hipCpuDeviceId);
    hipError_t rc_acc  = hipMemAdvise(p, nbytes, hipMemAdviseSetAccessedBy, on_device ? hipCpuDeviceId : dev);
    *pref_result = hipGetErrorString(rc_pref); *access_result = hipGetErrorString(rc_acc);
    return (rc_pref == hipSuccess && rc_acc == hipSuccess);
#elif defined(KOKKOS_ENABLE_CUDA)
    if(!std::is_same<GIZMO_KOKKOS_SHARED_SPACE, Kokkos::CudaUVMSpace>::value) {return -1;}
    int dev = 0;
    if(cudaGetDevice(&dev) != cudaSuccess) {return -1;}   /* nothing was asked of the driver */
    cudaError_t rc_pref = cudaMemAdvise(p, nbytes, cudaMemAdviseSetPreferredLocation, on_device ? dev : cudaCpuDeviceId);
    cudaError_t rc_acc  = cudaMemAdvise(p, nbytes, cudaMemAdviseSetAccessedBy, on_device ? cudaCpuDeviceId : dev);
    *pref_result = cudaGetErrorString(rc_pref); *access_result = cudaGetErrorString(rc_acc);
    return (rc_pref == cudaSuccess && rc_acc == cudaSuccess);
#else
    (void) p; (void) nbytes; (void) on_device;
    return -1;
#endif
}


/*! Apply the placement policy to one freshly allocated buffer, before anything touches it, so
 *  that the first write already lands where the policy wants it. `particle_arena_bytes` is
 *  zero for buffers that are not one of the bulk particle record arrays, which is how
 *  everything else served by this allocator is left alone. */
static void particle_storage_apply_placement(void *p, size_t nbytes, size_t particle_arena_bytes)
{
    if(!p || nbytes == 0 || particle_arena_bytes == 0) {return;}
    const int placement = particle_storage_placement(particle_arena_bytes);
    if(placement == PARTICLE_STORAGE_PLACEMENT_NONE) {return;}
    const char *where = (placement == PARTICLE_STORAGE_PLACEMENT_DEVICE) ? "device" : "host";
    const char *pref_result, *access_result;
    const int accepted = managed_memory_advise(p, nbytes, placement, &pref_result, &access_result);
    if(accepted < 0) {return;}
    if(!accepted)
    {
        /* The two calls are independent, so one of them can be refused on its own and leave
         * the other standing. Name which, rather than claim the default placement is back:
         * a later measurement made against a half-applied hint is otherwise read as a
         * measurement of the default. Said once; the run continues either way. */
        static int reported = 0;
        if(!reported && ThisTask == 0)
        {
            reported = 1;
            printf("Particle storage: the %s-preferred placement hint was not fully applied "
                   "(preferred location: %s; other-side access: %s). Whatever part of it was accepted "
                   "stands, and the run continues at whatever speed that gives.\n",
                   where, pref_result, access_result);
            fflush(stdout);
        }
        return;
    }
    /* Deliberately not latched: the capacity can change while the run is going, and each
     * change reallocates and re-applies. One line per application is what shows that it did,
     * and capacity changes are rare enough to be worth a line of their own anyway. */
    if(ThisTask == 0)
    {
        printf("Particle storage: %s-preferred placement applied to %g MByte "
               "(particle arrays total %g MByte on this rank).\n",
               where,
               (double) nbytes / (1024.0 * 1024.0),
               (double) particle_arena_bytes / (1024.0 * 1024.0));
        fflush(stdout);
    }
}

extern "C" void *gpu_particles_uvm_alloc(size_t nbytes, const char *label, size_t particle_arena_bytes)
{
    if(nbytes == 0) {return NULL;}
    /* kokkos_malloc THROWS on host-OOM; catch -> NULL so the caller's NULL-check
       (allocate.cc alloc_fail_local) fires instead of a hard terminate. The label
       names the buffer in the Kokkos allocation stream and any future OOM message. */
    void *p = NULL;
    try { p = Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(label ? label : "particle_soa_unlabeled", nbytes); }
    catch(const std::exception &) { return NULL; }
    if(p) {
        particle_storage_apply_placement(p, nbytes, particle_arena_bytes);
#if defined(KOKKOS_ENABLE_CUDA)
        /* The zeroing below, and a capacity change's copy into this buffer, are host writes: populate the pages on
           the host in bulk rather than one first-touch fault at a time. */
        if(particle_arena_bytes > 0 && particles_prefetch_enabled()) {
            cudaMemPrefetchAsync(p, nbytes, cudaCpuDeviceId, 0);
            if(cudaStreamSynchronize(0) != cudaSuccess) {(void) cudaGetLastError();}
        }
#endif
        memset(p, 0, nbytes);
    }
    return p;
}

/* Whether MPI must receive into the particle storage (P, CellP) through host staging rather than directly.
 * Where CUDA managed memory is not coherent with the host (a PCIe-attached GPU) its pages migrate between host
 * and device, and an MPI receive written straight into them can be lost: measured on Lonestar6 A100 nodes,
 * imported and exchanged particles arrived still holding what their slots held before.  Where the host reads
 * managed memory in place (GH200) nothing migrates, and the receive is direct.  HIP builds receive directly:
 * their large particle arrays are placed host-preferred (particle_storage_apply_placement) and no loss has been
 * seen there; host builds hold the storage in ordinary host memory.  Decided here, where the Kokkos memory space
 * and the device are visible, for the host-compiled exchange code that cannot see them; an attribute that cannot
 * be read counts as not coherent. */
extern "C" int gizmo_particle_storage_needs_staged_mpi_receive(void)
{
#if defined(KOKKOS_ENABLE_CUDA)
    return cuda_managed_memory_is_host_coherent() ? 0 : 1;
#else
    return 0;
#endif
}

/* Non-throwing allocation for the GPU transients, one entry point per memory space. Kokkos throws
   when it cannot serve a request, and an exception leaving a dispatcher takes the rank down where it
   stands, before the phase boundary that drains a controlled stop -- so one rank dies and the others
   wait on it. Returning NULL instead lets the caller name what it could not get, ask for the stop and
   return, and the run finishes the way every other failure does. Nothing is caught on success, so a
   run that never exhausts memory pays nothing. The label is forwarded exactly as given, including
   absent: the memory ledger buckets allocations by label prefix, so inventing one here would move a
   call site into a different bucket. */
extern "C" void *gizmo_gpu_alloc_shared(size_t nbytes, const char *label)
{
    try { return label ? Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(label, nbytes)
                       : Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(nbytes); }
    catch(const std::exception &) { return NULL; }
}

/*! Place a shared-space buffer that one side reads far more than the other.
 *
 *  The particle records get their placement inside their own allocator, keyed on how big the
 *  arena is; this is for the buffers that are not particle records and were therefore, until now,
 *  left with no hint at all -- which is not a neutral state. An unhinted managed buffer settles
 *  where it is FIRST TOUCHED and is then reached from the other side page by page, so a structure
 *  the host builds and the device reads ends up host-resident for the whole of every device read
 *  of it, at whatever the fabric costs.
 *
 *  `placement` is one of the PARTICLE_STORAGE_PLACEMENT_* values, so a caller can carry its own
 *  tri-state knob and this stays the one place that knows how to speak to the driver. */
extern "C" void gizmo_gpu_place_shared(void *p, size_t nbytes, int placement, const char *what)
{
    if(!p || nbytes == 0 || placement == PARTICLE_STORAGE_PLACEMENT_NONE) {return;}
    const int on_device = (placement == PARTICLE_STORAGE_PLACEMENT_DEVICE);
    const char *pref_result, *access_result;
    const int accepted = managed_memory_advise(p, nbytes, placement, &pref_result, &access_result);
    if(accepted < 0) {return;}
    /* Reported once per buffer kind rather than per allocation: the tree reallocates whenever it
     * grows, and a line each time would bury the one fact worth having, which is whether the hint
     * was accepted at all. A refusal is named because a later measurement taken against a
     * half-applied hint otherwise reads as a measurement of the default. */
    static int reported_ok = 0, reported_bad = 0;
    if(!accepted) {
        if(!reported_bad && ThisTask == 0) {
            reported_bad = 1;
            printf("%s placement: the %s-preferred hint was not fully applied (preferred location: %s; "
                   "other-side access: %s). Whatever part of it was accepted stands.\n",
                   what ? what : "shared buffer", on_device ? "device" : "host",
                   pref_result, access_result);
            fflush(stdout);
        }
        return;
    }
    if(!reported_ok && ThisTask == 0) {
        reported_ok = 1;
        printf("%s placement: %s-preferred applied (first buffer %g MByte).\n",
               what ? what : "shared buffer", on_device ? "device" : "host",
               (double) nbytes / (1024.0 * 1024.0));
        fflush(stdout);
    }
}

extern "C" void *gizmo_gpu_alloc_device(size_t nbytes, const char *label)
{
    try { return label ? Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>(label, nbytes)
                       : Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>(nbytes); }
    catch(const std::exception &) { return NULL; }
}

extern "C" void gizmo_gpu_free_device(void *ptr)
{
    if(ptr) {Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(ptr);}
}

/* ---- grow-only per-site device buffers (contract in gpu_particles_arena.h) ---- */
struct gizmo_gpu_pool_slot {void *p; size_t bytes; int busy;};
static struct gizmo_gpu_pool_slot gpu_pool_[GIZMO_POOL_NSITES];

static int gpu_pool_enabled(void)
{
    static int on = -1;
    if(on < 0) {
        on = 1;
        const char *env = getenv("GIZMO_GPU_POOL_VIEWS");
        if(env && env[0]) {on = (atoi(env) != 0);}
        if(ThisTask == 0) {printf("[GPU] per-call device buffers kept between calls: %s\n", on ? "on" : "off"); fflush(stdout);}
    }
    return on;
}

/* Outside the stream-ordered pool on CUDA: a buffer held for the whole run would count against the pool's
   release threshold (cooling.cc) and shrink what it keeps cached for the transients that still come and go.
   Measured at 256^3 24x4: with the hydro mirror's run-long buffer in the pool, the neighbour-build row rose by
   ~35 ms/step while its scratch was a per-call allocation, and not once the scratch came from here. These
   buffers are not in the Kokkos memory ledger. */
static void *gpu_pool_alloc_raw(size_t bytes, const char *label)
{
#if defined(KOKKOS_ENABLE_CUDA)
    (void) label;
    void *p = NULL;
    if(cudaMalloc(&p, bytes) != cudaSuccess) {(void) cudaGetLastError(); return NULL;}
    return p;
#else
    return gizmo_gpu_alloc_device(bytes, label);
#endif
}

static void gpu_pool_free_raw(void *p)
{
    if(!p) {return;}
#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::fence("gizmo_gpu_pool_free");   /* no kernel may still read it */
    if(cudaFree(p) != cudaSuccess) {(void) cudaGetLastError();}
#else
    Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(p);
#endif
}

extern "C" void *gizmo_gpu_pool_take(int site, size_t bytes, const char *label)
{
    if(bytes == 0) {bytes = 1;}
    if(site < 0 || site >= GIZMO_POOL_NSITES || !gpu_pool_enabled() || gpu_pool_[site].busy) {return gizmo_gpu_alloc_device(bytes, label);}
    struct gizmo_gpu_pool_slot *s = &gpu_pool_[site];
    if(s->bytes < bytes) {
        gpu_pool_free_raw(s->p);
        s->p = NULL; s->bytes = 0;
        size_t want = bytes + bytes / 16;   /* headroom: a count that creeps up should not regrow every call */
        if(!(s->p = gpu_pool_alloc_raw(want, label))) {want = bytes; s->p = gpu_pool_alloc_raw(want, label);}
        if(!s->p) {return gizmo_gpu_alloc_device(bytes, label);}   /* let the ordinary allocator decide (NULL on exhaustion) */
        s->bytes = want;
    }
    s->busy = 1;
    return s->p;
}

extern "C" void gizmo_gpu_pool_give(int site, void *p)
{
    if(!p) {return;}
    if(site >= 0 && site < GIZMO_POOL_NSITES && gpu_pool_[site].busy && gpu_pool_[site].p == p) {gpu_pool_[site].busy = 0; return;}
    Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(p);
}

extern "C" void gizmo_gpu_pool_release(void)
{
    for(int k = 0; k < GIZMO_POOL_NSITES; k++) {gpu_pool_free_raw(gpu_pool_[k].p); gpu_pool_[k].p = NULL; gpu_pool_[k].bytes = 0; gpu_pool_[k].busy = 0;}
}

extern "C" void gizmo_gpu_copy_to_device(void *dst_device, const void *src_host, size_t nbytes)
{
    if(nbytes == 0) {return;}
    Kokkos::deep_copy(Kokkos::View<char*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>((char *) dst_device, nbytes),
                      Kokkos::View<const char*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>((const char *) src_host, nbytes));
}

extern "C" void gizmo_gpu_copy_to_host(void *dst_host, const void *src_device, size_t nbytes)
{
    if(nbytes == 0) {return;}
    Kokkos::fence("gizmo_gpu_copy_to_host");   /* device-wide: the source may have been written by a library on its own stream */
    Kokkos::deep_copy(Kokkos::View<char*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>((char *) dst_host, nbytes),
                      Kokkos::View<const char*, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>((const char *) src_device, nbytes));
}

extern "C" void *gizmo_gpu_alloc_host(size_t nbytes, const char *label)
{
    try { return label ? Kokkos::kokkos_malloc<Kokkos::HostSpace>(label, nbytes)
                       : Kokkos::kokkos_malloc<Kokkos::HostSpace>(nbytes); }
    catch(const std::exception &) { return NULL; }
}

/* Page-locked host memory is a node resource: it cannot be paged out, so what one rank pins is
   gone for every other process on the node until it is released. Callers hold it for a bounded
   span and check the result like every allocation here. */
extern "C" void *gizmo_gpu_alloc_pinned_host(size_t nbytes, const char *label)
{
    try { return label ? Kokkos::kokkos_malloc<GIZMO_KOKKOS_PINNED_HOST_SPACE>(label, nbytes)
                       : Kokkos::kokkos_malloc<GIZMO_KOKKOS_PINNED_HOST_SPACE>(nbytes); }
    catch(const std::exception &) { return NULL; }
}

extern "C" void gizmo_gpu_free_pinned_host(void *ptr)
{
    if(ptr) {Kokkos::kokkos_free<GIZMO_KOKKOS_PINNED_HOST_SPACE>(ptr);}
}

extern "C" void gpu_particles_uvm_free(void *ptr)
{
    /* Paired with gpu_particles_uvm_alloc so a capacity change can allocate the replacement
       buffer before releasing the old one. Lives here, not in allocate.cc, for the same reason
       the alloc does: that TU must not include <Kokkos_Core.hpp>. */
    if(ptr) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(ptr);}
}

extern "C" struct particle_data *gpu_particles_arena_P(void)     {return arena_valid_ ? arena_P     : NULL;}
extern "C" struct gas_cell_data *gpu_particles_arena_CellP(void) {return arena_valid_ ? arena_CellP : NULL;}
extern "C" int gpu_particles_arena_capacity(void)                {return arena_capacity_;}
extern "C" int gpu_particles_arena_valid(void)                   {return arena_valid_;}
