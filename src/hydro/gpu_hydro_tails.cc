/* hydro/gpu_hydro_tails.cc -- the gradients writeback, the gradients finalization loop and the hydro preloop as
 * device kernels, for the span from the big-step gradients pass to the hydro pass in which the rank's own particles
 * stay on the device. Contract, admitted configurations and the knob: hydro/gpu_hydro_tails.h.
 *
 * This translation unit is compiled without contraction to fused multiply-add (target-specific flag in the
 * Makefile): the bodies are the host loops' own (hydro/gradients_tail.h, hydro/gpu_hydro_tails.h) and give the
 * same bits only under the host's arithmetic. Keep every device instantiation of those bodies in this file. */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"  /* MUST precede allvars.h (device-pass All redirect) */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../system/gpu_particles_arena.h"
#include "../declarations/gpu_dispatch_templates.h"   /* gizmo_gpu_kernel_launch */
#include "../mesh/kernel.h"                           /* MUST precede gradients_loop.h (no include guards) */
#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/ghost_writeback.h"                  /* ghost_get_num_local */
#include "gradients_loop.h"
#include "gradients_tail.h"

#if GIZMO_HYDRO_TAILS_DEVICE && defined(KOKKOS_ENABLE_CUDA)
#define HYDRO_TAILS_ON_DEVICE 1
#else
#define HYDRO_TAILS_ON_DEVICE 0
#endif

/* GIZMO_GPU_GRADTAIL: 0 keeps the host loops in an admitted build (same-binary comparison); default on. */
static int hydro_tails_enabled(void)
{
    static int enabled = -1;
    if(enabled < 0) {
        enabled = HYDRO_TAILS_ON_DEVICE;
        const char *env = getenv("GIZMO_GPU_GRADTAIL");
        if(enabled && env && env[0]) {enabled = (atoi(env) != 0);}
        if(ThisTask == 0 && HYDRO_TAILS_ON_DEVICE) {
            printf("Gradients writeback, gradients finalization and hydro preloop on the device while the particles stay there: %s.\n", enabled ? "on" : "off");
            fflush(stdout);
        }
    }
    return enabled;
}

static int finalize_pending_ = 0;   /* the writeback of this gradients call ran on the device */
static int preloop_pending_ = 0;    /* ... and the hydro preloop that follows it may run there too */

#if HYDRO_TAILS_ON_DEVICE
/* The gradient scratch the device writeback fills, indexed by particle like the host's: d_passer_[i] holds this
 * call's values only where d_stamp_[i] == stamp_ (any other active gas cell reads zeros, as after the host's
 * memset), so nothing has to be cleared between calls. Grow-only device buffers, kept for the run. */
static struct temporary_data_topass *d_passer_ = NULL;
static int *d_stamp_ = NULL;
static size_t scratch_cap_ = 0;
static int stamp_ = 0;
static int *d_list_ = NULL;
static size_t list_cap_ = 0;

static int hydro_tails_scratch(int n_gas)
{
    if(n_gas <= 0) {n_gas = 1;}
    if((size_t) n_gas <= scratch_cap_) {return 1;}
    if(d_passer_) {gizmo_gpu_free_device(d_passer_);}
    if(d_stamp_) {gizmo_gpu_free_device(d_stamp_);}
    scratch_cap_ = (size_t) n_gas + (size_t) n_gas / 4;
    d_passer_ = (struct temporary_data_topass *) gizmo_gpu_alloc_device(scratch_cap_ * sizeof(struct temporary_data_topass), "gradients_tail_passer");
    d_stamp_ = (int *) gizmo_gpu_alloc_device(scratch_cap_ * sizeof(int), "gradients_tail_stamp");
    if(!d_passer_ || !d_stamp_) {
        if(d_passer_) {gizmo_gpu_free_device(d_passer_);}
        if(d_stamp_) {gizmo_gpu_free_device(d_stamp_);}
        d_passer_ = NULL; d_stamp_ = NULL; scratch_cap_ = 0;
        return 0;
    }
    Kokkos::View<int *, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> st(d_stamp_, scratch_cap_);
    Kokkos::deep_copy(st, 0);
    stamp_ = 0;
    return 1;
}

/* The active list on the device. It does not change between the gradients pass and the hydro pass; the buffer is
 * sized when the span begins (so the kernels after it cannot fail for memory) and filled for each use. */
static int hydro_tails_list_capacity(int n)
{
    if((size_t) n <= list_cap_) {return 1;}
    if(d_list_) {gizmo_gpu_free_device(d_list_);}
    list_cap_ = (size_t) n + (size_t) n / 4;
    d_list_ = (int *) gizmo_gpu_alloc_device(list_cap_ * sizeof(int), "hydro_tails_active_list");
    if(!d_list_) {list_cap_ = 0; return 0;}
    return 1;
}
static const int *hydro_tails_active_list(int n)
{
    if(!hydro_tails_list_capacity(n)) {return NULL;}
    Kokkos::View<const int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h(ActiveParticleList.data(), (size_t) n);
    Kokkos::View<int *, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> d(d_list_, (size_t) n);
    Kokkos::deep_copy(d, h);
    return d_list_;
}

/* GIZMO_GPU_GRADTAIL_VERIFY=N (diagnostic): for the first N spans on the device, the host loops also run, on host
 * copies of the active records taken before the writeback, and the records the kernels wrote are compared with
 * them byte for byte. Costs a copy of every active record, twice, per verified span. */
struct HydroTailsVerify {
    int on = 0, spans = 0;
    long long unmapped = 0;
    std::vector<struct particle_data> P0;
    std::vector<struct gas_cell_data> C0;
    std::vector<struct temporary_data_topass> pas;
    std::vector<int> slot;
};
static HydroTailsVerify verify_;
static int hydro_tails_verify_wanted(void)
{
    static int left = -1;
    if(left < 0) {const char *env = getenv("GIZMO_GPU_GRADTAIL_VERIFY"); left = (env && env[0]) ? atoi(env) : 0; if(left < 0) {left = 0;}}
    if(left == 0) {return 0;}
    left--;
    return 1;
}
/* The records of the active list as they are now, copied to the host (whole records, as words). */
static void hydro_tails_gather_actives(const int *dl, int n, std::vector<struct particle_data> &hp, std::vector<struct gas_cell_data> &hc)
{
    typedef unsigned int W;
    static_assert(sizeof(struct particle_data) % sizeof(W) == 0 && sizeof(struct gas_cell_data) % sizeof(W) == 0, "records copied as words");
    const int wp = (int) (sizeof(struct particle_data) / sizeof(W)), wc = (int) (sizeof(struct gas_cell_data) / sizeof(W));
    hp.resize((size_t) n); hc.resize((size_t) n);
    W *d_p = (W *) gizmo_gpu_alloc_device((size_t) n * sizeof(struct particle_data), "gradients_tail_verify_P");
    W *d_c = (W *) gizmo_gpu_alloc_device((size_t) n * sizeof(struct gas_cell_data), "gradients_tail_verify_C");
    if(!d_p || !d_c) {   /* verify only: read them on the host instead */
        if(d_p) {gizmo_gpu_free_device(d_p);}
        if(d_c) {gizmo_gpu_free_device(d_c);}
        for(int k = 0; k < n; k++) {hp[(size_t) k] = P[ActiveParticleList[k]]; hc[(size_t) k] = CellP[ActiveParticleList[k]];}
        return;
    }
    const W *Pw = (const W *) P;
    const W *Cw = (const W *) CellP;
    gizmo_gpu_kernel_launch("gradients_tail_verify_gather", n, KOKKOS_LAMBDA(int k) {
        const int i = dl[k];
        for(int w = 0; w < wp; w++) {d_p[(size_t) k * wp + w] = Pw[(size_t) i * wp + w];}
        for(int w = 0; w < wc; w++) {d_c[(size_t) k * wc + w] = Cw[(size_t) i * wc + w];}
    });
    Kokkos::View<W *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hpv((W *) hp.data(), (size_t) n * wp), hcv((W *) hc.data(), (size_t) n * wc);
    Kokkos::View<const W *, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dpv(d_p, (size_t) n * wp), dcv(d_c, (size_t) n * wc);
    Kokkos::deep_copy(hpv, dpv);
    Kokkos::deep_copy(hcv, dcv);
    gizmo_gpu_free_device(d_p);
    gizmo_gpu_free_device(d_c);
}
#endif

extern "C" int gizmo_gradients_tail_begin(int corridor_list, int n_active)
{
    finalize_pending_ = preloop_pending_ = 0;
    if(!hydro_tails_enabled()) {return 0;}
#if HYDRO_TAILS_ON_DEVICE
    /* The corridor's pass on a big step, which the bulk prefetch has just put on the device. */
    if(!corridor_list || n_active <= 0 || (long long) n_active * 8 < (long long) NumPart) {return 0;}
    /* No scratch: the host loops (correct, only slower). */
    if(!hydro_tails_scratch(N_gas) || !hydro_tails_list_capacity((int) ActiveParticleList.size())) {return 0;}
    if(!gizmo_particles_keep_owned_on_device("gradients_tail")) {return 0;}
    GIZMO_GPU_ENSURE_ALL_FRESH();
    stamp_++;
    finalize_pending_ = preloop_pending_ = 1;
    if(stamp_ == 1 && ThisTask == 0) {printf("Gradients tail: first span on the device (%d actives on task 0).\n", n_active); fflush(stdout);}
    verify_.on = hydro_tails_verify_wanted();
    return 1;
#else
    (void) corridor_list; (void) n_active;
    return 0;
#endif
}

bool GradientsSpec::device_writeback_begin(const neighbor_loop_args& args, int num_active)
{
    Aux *aux = nlr_aux<GradientsSpec>(args);
    if(aux == nullptr || aux->grad_iter != 0) {return false;}
    return gizmo_gradients_tail_begin(args.external_csr != nullptr, num_active) != 0;
}

void GradientsSpec::device_writeback(const neighbor_loop_args& args, const int *d_active_idx, int first, int n,
                                     const AccumData *accums)
{
#if HYDRO_TAILS_ON_DEVICE
    if(n <= 0) {return;}
    if(verify_.on) {   /* the host writeback of this chunk, on the copies */
        const int na = (int) ActiveParticleList.size();
        if(first == 0) {
            hydro_tails_gather_actives(hydro_tails_active_list(na), na, verify_.P0, verify_.C0);
            verify_.pas.assign((size_t) na, temporary_data_topass{});
            verify_.slot.assign((size_t) (NumPart > 0 ? NumPart : 1), -1);
            for(int k = 0; k < na; k++) {verify_.slot[(size_t) ActiveParticleList[k]] = k;}
            verify_.unmapped = 0;
        }
        std::vector<AccumData> acc((size_t) n);
        Kokkos::View<AccumData *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> ha(acc.data(), (size_t) n);
        Kokkos::View<const AccumData *, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> da(accums, (size_t) n);
        Kokkos::deep_copy(ha, da);
        for(int kk = 0; kk < n; kk++) {
            const int i = args.active_list[first + kk];
            const int k = (i >= 0 && i < NumPart) ? verify_.slot[(size_t) i] : -1;
            if(k < 0) {verify_.unmapped++; continue;}
            gradients_writeback_cell(verify_.P0.data(), verify_.C0.data(), verify_.pas[(size_t) k], k, &acc[(size_t) kk]);
        }
    }
    struct particle_data *Pp = P;
    struct gas_cell_data *Cp = CellP;
    struct temporary_data_topass *dp = d_passer_;
    int *ds = d_stamp_;
    const int stamp = stamp_;
    gizmo_gpu_kernel_launch("gradients_writeback_device", n, KOKKOS_LAMBDA(int kk) {
        const int i = d_active_idx[first + kk];
        struct temporary_data_topass pas;
        for(size_t b = 0; b < sizeof(pas); b++) {((char *) &pas)[b] = 0;}   /* the host zeroes it before the pass */
        gradients_writeback_cell(Pp, Cp, pas, i, &accums[kk]);
        dp[i] = pas;
        ds[i] = stamp;
    });
#else
    (void) args; (void) d_active_idx; (void) first; (void) n; (void) accums;
#endif
}

void GradientsSpec::device_writeback_end(const neighbor_loop_args& /*args*/)
{
    /* The ghost refresh before the hydro pass writes the ghost slots on the host, and the hydro pass's ghost
       hooks read them there. */
    if(ghost_get_num_ghosts() > 0) {gizmo_particles_prefetch_ghosts_to_host(ghost_get_num_local(), "gradients_tail");}
}

extern "C" int gizmo_gradients_finalize_on_device(void)
{
#if HYDRO_TAILS_ON_DEVICE
    if(!finalize_pending_) {return 0;}
    finalize_pending_ = 0;
    const int n = (int) ActiveParticleList.size();
    if(n <= 0) {return 1;}
    const int *dl = hydro_tails_active_list(n);   /* sized at gizmo_gradients_tail_begin: cannot fail */
    struct particle_data *Pp = P;
    struct gas_cell_data *Cp = CellP;
    const struct temporary_data_topass *dp = d_passer_;
    const int *ds = d_stamp_;
    const int stamp = stamp_;
    gizmo_gpu_kernel_launch("gradients_finalize_device", n, KOKKOS_LAMBDA(int idx) {
        const int i = dl[idx];
        if(Pp[i].Type != 0) {return;}
        struct temporary_data_topass pas;
        if(ds[i] == stamp) {pas = dp[i];} else {for(size_t b = 0; b < sizeof(pas); b++) {((char *) &pas)[b] = 0;}}
        gradients_finalize_cell(Pp, Cp, pas, i);
    });
    if(verify_.on) {   /* the host finalization on the copies, then compare what the kernels wrote */
        verify_.on = 0;
        verify_.spans++;
        std::vector<struct particle_data> P1;
        std::vector<struct gas_cell_data> C1;
        hydro_tails_gather_actives(dl, n, P1, C1);
        long long n_gas = 0, n_cell_diff = 0, n_p_diff = 0, n_grad_diff = 0, n_vort_diff = 0;
        for(int k = 0; k < n; k++) {
            if(verify_.P0[(size_t) k].Type != 0) {continue;}
            n_gas++;
            gradients_finalize_cell(verify_.P0.data(), verify_.C0.data(), verify_.pas[(size_t) k], k);
            const struct gas_cell_data &a = verify_.C0[(size_t) k], &b = C1[(size_t) k];
            if(memcmp(&a, &b, sizeof(a)) != 0) {n_cell_diff++;}
            if(memcmp(&verify_.P0[(size_t) k], &P1[(size_t) k], sizeof(struct particle_data)) != 0) {n_p_diff++;}
            if(memcmp(&a.Gradients, &b.Gradients, sizeof(a.Gradients)) != 0) {n_grad_diff++;}
#if defined(TURB_DRIVING) || defined(OUTPUT_VORTICITY)
            if(memcmp(&a.Vorticity, &b.Vorticity, sizeof(a.Vorticity)) != 0) {n_vort_diff++;}
#endif
        }
        if(ThisTask == 0 || n_cell_diff || n_p_diff || verify_.unmapped) {
            printf("Gradients tail verify (task %d, span %d): %lld active gas cells; records differing from the host loops: "
                   "cell %lld (gradients %lld, vorticity %lld), particle %lld; rows not in the active list %lld\n",
                   ThisTask, verify_.spans, n_gas, n_cell_diff, n_grad_diff, n_vort_diff, n_p_diff, verify_.unmapped);
            fflush(stdout);
        }
        verify_.P0.clear(); verify_.P0.shrink_to_fit(); verify_.C0.clear(); verify_.C0.shrink_to_fit();
        verify_.pas.clear(); verify_.pas.shrink_to_fit(); verify_.slot.clear(); verify_.slot.shrink_to_fit();
    }
    return 1;
#else
    return 0;
#endif
}

extern "C" int gizmo_hydro_preloop_on_device(void)
{
#if HYDRO_TAILS_ON_DEVICE
    const int pending = preloop_pending_;
    preloop_pending_ = 0;
    if(!pending || !gizmo_particles_owned_kept_on_device()) {return 0;}
    const int n = (int) ActiveParticleList.size();
    if(n <= 0) {return 1;}
    const int *dl = hydro_tails_active_list(n);
    if(!dl) {return 0;}   /* the host loop (correct, only slower) */
    GIZMO_GPU_ENSURE_ALL_FRESH();
    struct particle_data *Pp = P;
    struct gas_cell_data *Cp = CellP;
    gizmo_gpu_kernel_launch("hydro_preloop_device", n, KOKKOS_LAMBDA(int idx) {
        const int i = dl[idx];
        if(Pp[i].Type == 0) {hydro_preloop_cell(Pp, Cp, i);}
    });
    return 1;
#else
    return 0;
#endif
}
