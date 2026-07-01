/* Device-resident fine-tree receiver-walk sidecar — device memory owner.
 * See mesh/gpu_fine_sidecar.h + OPEN_topleaf_router_design.md §45.
 *
 * This TU owns ONLY the device arrays + their deep_copy/readback + the stored
 * freshness key.  All host staging (positions from P[], reach from the host SSOT
 * gx_policy_scaled_h, band, j_to_pool) happens in the host owner ghost_exchange.cc
 * and arrives here as plain host buffers, so the device reach == the host receiver
 * reach by construction and no reach formula is duplicated on the device side.
 *
 * PASSIVE / ORACLE-ONLY at S2a: nothing consumes the device arrays yet (the
 * bounded device walk is S2b).  Temporary validation scaffolding (teardown §39). */

#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>

#include "gpu_fine_sidecar.h"

#include "../declarations/macros.h"   /* GIZMO_KOKKOS_DEVICE_SPACE */

namespace {

using UV = Kokkos::MemoryTraits<Kokkos::Unmanaged>;

struct FineSidecar {
    double *d_sx = nullptr, *d_sy = nullptr, *d_sz = nullptr, *d_sh = nullptr;
    int    *d_stype = nullptr;
    int    *d_j_to_pool = nullptr;
    double *d_band = nullptr;
    /* Allocated capacities (in elements) so we can grow-or-reuse without freeing
     * every call — mirrors the gpu_spatial_index grow pattern. */
    int  cap_pool = 0;
    int  cap_numpart = 0;
    long cap_band = 0;
    /* Lengths actually uploaded (<= capacities) — the readback guard. */
    int  up_num_pool = 0;
    int  up_numpart = 0;
    long up_band_len = 0;
    int  valid = 0;
    struct gx_fine_sidecar_key_t key;
};

FineSidecar g_sc;

/* Grow a single DEVICE_SPACE array to at least `need` elements (free+realloc on
 * growth; reuse otherwise).  Returns pointer or nullptr on failure.  Each helper
 * owns exactly one (ptr, cap) pair — never share a cap across arrays. */
double *grow_dev_double_l(double *cur, long *cap, long need)
{
    if(need <= 0) need = 1;
    if(cur && *cap >= need) return cur;
    if(cur) Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(cur);
    double *p = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>((size_t)need * sizeof(double));
    *cap = p ? need : 0;
    return p;
}
int *grow_dev_int(int *cur, int *cap, long need)
{
    if(need <= 0) need = 1;
    if(cur && *cap >= need) return cur;
    if(cur) Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(cur);
    int *p = (int *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>((size_t)need * sizeof(int));
    *cap = (int)(p ? need : 0);
    return p;
}

void free_pool_arrays(FineSidecar *s)
{
    if(s->d_sx)    Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(s->d_sx);
    if(s->d_sy)    Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(s->d_sy);
    if(s->d_sz)    Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(s->d_sz);
    if(s->d_sh)    Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(s->d_sh);
    if(s->d_stype) Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(s->d_stype);
    s->d_sx = s->d_sy = s->d_sz = s->d_sh = nullptr;
    s->d_stype = nullptr;
    s->cap_pool = 0;
}

/* Ensure the five same-length pool arrays (4 double + 1 int) all fit `need`; on
 * growth free+realloc the whole group so a shared capacity can never leave one
 * array undersized.  Returns 0 on success, <0 on allocation failure. */
int ensure_pool_arrays(FineSidecar *s, long need)
{
    if(need <= 0) need = 1;
    if(s->d_sx && s->cap_pool >= need) return 0;   /* reuse */
    free_pool_arrays(s);
    s->d_sx    = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>((size_t)need * sizeof(double));
    s->d_sy    = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>((size_t)need * sizeof(double));
    s->d_sz    = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>((size_t)need * sizeof(double));
    s->d_sh    = (double *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>((size_t)need * sizeof(double));
    s->d_stype = (int *)    Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>((size_t)need * sizeof(int));
    if(!s->d_sx || !s->d_sy || !s->d_sz || !s->d_sh || !s->d_stype) {
        free_pool_arrays(s);
        return -1;
    }
    s->cap_pool = (int)need;
    return 0;
}

void copy_h2d_double(double *dev, const double *host, long n)
{
    if(n <= 0) return;
    Kokkos::View<const double*, Kokkos::HostSpace, UV>         hv(host, (size_t)n);
    Kokkos::View<double*, GIZMO_KOKKOS_DEVICE_SPACE, UV>       dv(dev,  (size_t)n);
    Kokkos::deep_copy(dv, hv);
}
void copy_h2d_int(int *dev, const int *host, long n)
{
    if(n <= 0) return;
    Kokkos::View<const int*, Kokkos::HostSpace, UV>       hv(host, (size_t)n);
    Kokkos::View<int*, GIZMO_KOKKOS_DEVICE_SPACE, UV>     dv(dev,  (size_t)n);
    Kokkos::deep_copy(dv, hv);
}
void copy_d2h_double(double *host, const double *dev, long n)
{
    if(n <= 0) return;
    Kokkos::View<const double*, GIZMO_KOKKOS_DEVICE_SPACE, UV> dv(dev,  (size_t)n);
    Kokkos::View<double*, Kokkos::HostSpace, UV>              hv(host, (size_t)n);
    Kokkos::deep_copy(hv, dv);
}
void copy_d2h_int(int *host, const int *dev, long n)
{
    if(n <= 0) return;
    Kokkos::View<const int*, GIZMO_KOKKOS_DEVICE_SPACE, UV> dv(dev,  (size_t)n);
    Kokkos::View<int*, Kokkos::HostSpace, UV>              hv(host, (size_t)n);
    Kokkos::deep_copy(hv, dv);
}

} /* namespace */

extern "C" int gpu_fine_sidecar_upload(
    const double *sx, const double *sy, const double *sz,
    const double *sh, const int *stype, int num_pool,
    const int *j_to_pool, int numpart,
    const double *band, long band_len,
    const struct gx_fine_sidecar_key_t *key)
{
    if(num_pool < 0 || numpart < 0 || band_len < 0 || key == nullptr) return -1;
    if(num_pool > 0 && (!sx || !sy || !sz || !sh || !stype)) return -1;
    if(numpart  > 0 && !j_to_pool) return -1;
    if(band_len > 0 && !band)      return -1;

    g_sc.valid = 0;   /* invalidate until the whole upload succeeds */

    if(ensure_pool_arrays(&g_sc, num_pool) != 0) return -2;
    g_sc.d_j_to_pool = grow_dev_int     (g_sc.d_j_to_pool, &g_sc.cap_numpart, numpart);
    g_sc.d_band      = grow_dev_double_l(g_sc.d_band,      &g_sc.cap_band,    band_len);
    if(!g_sc.d_j_to_pool || !g_sc.d_band) return -2;

    copy_h2d_double(g_sc.d_sx,    sx,    num_pool);
    copy_h2d_double(g_sc.d_sy,    sy,    num_pool);
    copy_h2d_double(g_sc.d_sz,    sz,    num_pool);
    copy_h2d_double(g_sc.d_sh,    sh,    num_pool);
    copy_h2d_int   (g_sc.d_stype, stype, num_pool);
    copy_h2d_int   (g_sc.d_j_to_pool, j_to_pool, numpart);
    copy_h2d_double(g_sc.d_band,  band,  band_len);
    Kokkos::fence();

    g_sc.up_num_pool  = num_pool;
    g_sc.up_numpart   = numpart;
    g_sc.up_band_len  = band_len;
    g_sc.key   = *key;
    g_sc.valid = 1;
    return 0;
}

extern "C" void gpu_fine_sidecar_free(void)
{
    free_pool_arrays(&g_sc);   /* d_sx/d_sy/d_sz/d_sh/d_stype + cap_pool */
    if(g_sc.d_j_to_pool) Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(g_sc.d_j_to_pool);
    if(g_sc.d_band)      Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(g_sc.d_band);
    g_sc.d_j_to_pool = nullptr;
    g_sc.d_band = nullptr;
    g_sc.cap_numpart = 0; g_sc.cap_band = 0;
    g_sc.up_num_pool = g_sc.up_numpart = 0; g_sc.up_band_len = 0;
    g_sc.valid = 0;
}

extern "C" int gpu_fine_sidecar_is_valid(const struct gx_fine_sidecar_key_t *want)
{
    if(!g_sc.valid || want == nullptr) return 0;
    const struct gx_fine_sidecar_key_t *k = &g_sc.key;
    if(k->numpart          != want->numpart)          return 0;
    if(k->maxpart          != want->maxpart)          return 0;
    if(k->numnodestree     != want->numnodestree)     return 0;
    if(k->fb_maxpart       != want->fb_maxpart)       return 0;
    if(k->fb_nnodes        != want->fb_nnodes)        return 0;
    if(k->num_pool         != want->num_pool)         return 0;
    if(k->eligible_mask    != want->eligible_mask)    return 0;
    if(k->radius_policy    != want->radius_policy)    return 0;
    if(k->j_scale          != want->j_scale)          return 0;
    if(k->safety           != want->safety)           return 0;
    if(k->treebuild_gen    != want->treebuild_gen)    return 0;
    if(k->hmax_refresh_gen != want->hmax_refresh_gen) return 0;
    if(k->ti               != want->ti)               return 0;
    if(k->pool_ti          != want->pool_ti)          return 0;
    /* SoA-drift certification: must match AND not be the uncertified sentinel. */
    if(want->soa_drift_ti == GX_FINE_SIDECAR_SOA_DRIFT_UNCERTIFIED) return 0;
    if(k->soa_drift_ti     != want->soa_drift_ti)     return 0;
    return 1;
}

extern "C" int gpu_fine_sidecar_readback(
    double *sx, double *sy, double *sz, double *sh,
    int *stype, int num_pool,
    int *j_to_pool, int numpart,
    double *band, long band_len)
{
    if(!g_sc.valid) return -1;
    if(num_pool != g_sc.up_num_pool) return -1;
    if(numpart  != g_sc.up_numpart)  return -1;
    if(band_len != g_sc.up_band_len) return -1;
    copy_d2h_double(sx,    g_sc.d_sx,    num_pool);
    copy_d2h_double(sy,    g_sc.d_sy,    num_pool);
    copy_d2h_double(sz,    g_sc.d_sz,    num_pool);
    copy_d2h_double(sh,    g_sc.d_sh,    num_pool);
    copy_d2h_int   (stype, g_sc.d_stype, num_pool);
    copy_d2h_int   (j_to_pool, g_sc.d_j_to_pool, numpart);
    copy_d2h_double(band,  g_sc.d_band,  band_len);
    Kokkos::fence();
    return 0;
}
