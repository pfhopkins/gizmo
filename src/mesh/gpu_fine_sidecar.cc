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

#include <mpi.h>   /* required BEFORE allvars.h: declarations/typedefs.h uses MPI_Datatype/MPI_LONG_LONG */
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>

#include "gpu_fine_sidecar.h"

#include "../declarations/macros.h"   /* GIZMO_KOKKOS_DEVICE_SPACE */
/* GPU-TU convention: gpu_all_mirror.h precedes allvars.h.  allvars.h provides
 * Vec3<MyFloat>/MyFloat for the gravity SoA type consumed by the device walk; the
 * shared geometry predicates come from ghost_exchange_functions.h (device pass, since
 * <Kokkos_Core.hpp> above already defined KOKKOS_INLINE_FUNCTION). */
#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../gravity/gpu_gravity_tree.h"      /* gpu_gravity_tree_soa()/capacity() + SoA struct */
#include "ghost_exchange_functions.h"         /* gx_pair_accept + gx_node_sphere_overlap_center_len */

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
char *grow_dev_char(char *cur, long *cap, long need)
{
    if(need <= 0) need = 1;
    if(cur && *cap >= need) return cur;
    if(cur) Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(cur);
    char *p = (char *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>((size_t)need);
    *cap = p ? need : 0;
    return p;
}
long *grow_dev_long(long *cur, long *cap, long need)
{
    if(need <= 0) need = 1;
    if(cur && *cap >= need) return cur;
    if(cur) Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(cur);
    long *p = (long *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_DEVICE_SPACE>((size_t)need * sizeof(long));
    *cap = p ? need : 0;
    return p;
}
void copy_d2h_char(char *host, const char *dev, long n)
{
    if(n <= 0) return;
    Kokkos::View<const char*, GIZMO_KOKKOS_DEVICE_SPACE, UV> dv(dev,  (size_t)n);
    Kokkos::View<char*, Kokkos::HostSpace, UV>              hv(host, (size_t)n);
    Kokkos::deep_copy(hv, dv);
}
void copy_d2h_long(long *host, const long *dev, long n)
{
    if(n <= 0) return;
    Kokkos::View<const long*, GIZMO_KOKKOS_DEVICE_SPACE, UV> dv(dev,  (size_t)n);
    Kokkos::View<long*, Kokkos::HostSpace, UV>              hv(host, (size_t)n);
    Kokkos::deep_copy(hv, dv);
}

/* Runaway-walk guard: a TOPLEVEL-bounded local walk cannot exceed the node count;
 * a step count past this means a cycle (index bug) — counted as bad_index. */
constexpr long GX_DEVWALK_STEP_GUARD = 20000000L;

/* Device scratch for the bounded fine-tree walk (grow-or-reuse; freed by
 * gpu_fine_sidecar_free).  Query batch + start-node CSR + matched output + the
 * three pseudo/foreign/bad-index counters. */
struct FineWalkScratch {
    double *d_qpos    = nullptr;  long cap_qpos    = 0;
    double *d_qh      = nullptr;  long cap_qh      = 0;
    int    *d_soff    = nullptr;  int  cap_soff    = 0;
    int    *d_starts  = nullptr;  int  cap_starts  = 0;
    char   *d_matched = nullptr;  long cap_matched = 0;
    long   *d_cnt     = nullptr;  long cap_cnt     = 0;
};
FineWalkScratch g_ws;

void free_walk_scratch(FineWalkScratch *w)
{
    if(w->d_qpos)    Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(w->d_qpos);
    if(w->d_qh)      Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(w->d_qh);
    if(w->d_soff)    Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(w->d_soff);
    if(w->d_starts)  Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(w->d_starts);
    if(w->d_matched) Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(w->d_matched);
    if(w->d_cnt)     Kokkos::kokkos_free<GIZMO_KOKKOS_DEVICE_SPACE>(w->d_cnt);
    w->d_qpos = w->d_qh = nullptr;
    w->d_soff = w->d_starts = nullptr;
    w->d_matched = nullptr;
    w->d_cnt = nullptr;
    w->cap_qpos = w->cap_qh = w->cap_matched = w->cap_cnt = 0;
    w->cap_soff = w->cap_starts = 0;
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
    free_walk_scratch(&g_ws);
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

extern "C" int gpu_fine_sidecar_walk(
    const double *q_pos, const double *q_h, int nq,
    const int *starts_off, const int *starts, long starts_len,
    int search_mode, unsigned int supply_mask,
    const int periodic_flags[3], const double box_sizes[3],
    unsigned int topflag_mask, int fineband_ntypes, int num_ptypes,
    int maxpart, int maxnodes, int maxforeign, int num_local,
    int numpart, int nnodes,
    char *matched_out,
    long *pseudo_hits, long *foreign_node_visits, long *bad_index_hits)
{
    if(pseudo_hits)         *pseudo_hits = 0;
    if(foreign_node_visits) *foreign_node_visits = 0;
    if(bad_index_hits)      *bad_index_hits = 0;
    if(!g_sc.valid) return -1;
    const int num_pool = g_sc.up_num_pool;
    if(num_pool <= 0) return 0;          /* no supply => nothing to compare this batch */
    if(nq <= 0)       return 0;
    if(!matched_out || !q_pos || !q_h || !starts_off || (starts_len > 0 && !starts)) return -1;

    struct gpu_gravity_tree_soa_t *soa = gpu_gravity_tree_soa();
    if(!soa || !soa->center || !soa->len || !soa->sibling || !soa->nextnode ||
       !soa->bitflags || !soa->nextnode_aux) return -2;
    const int soa_cap     = gpu_gravity_tree_capacity();
    const int nn_aux_size = soa->nextnode_aux_size;

    /* grow-or-reuse device scratch (the caller bounds nq so nq*num_pool <= cap). */
    g_ws.d_qpos    = grow_dev_double_l(g_ws.d_qpos,    &g_ws.cap_qpos,    (long)nq * 3);
    g_ws.d_qh      = grow_dev_double_l(g_ws.d_qh,      &g_ws.cap_qh,      (long)nq);
    g_ws.d_soff    = grow_dev_int     (g_ws.d_soff,    &g_ws.cap_soff,    (long)nq + 1);
    g_ws.d_starts  = grow_dev_int     (g_ws.d_starts,  &g_ws.cap_starts,  starts_len > 0 ? starts_len : 1);
    g_ws.d_matched = grow_dev_char    (g_ws.d_matched, &g_ws.cap_matched, (long)nq * num_pool);
    g_ws.d_cnt     = grow_dev_long    (g_ws.d_cnt,     &g_ws.cap_cnt,     3);
    if(!g_ws.d_qpos || !g_ws.d_qh || !g_ws.d_soff || !g_ws.d_starts || !g_ws.d_matched || !g_ws.d_cnt)
        return -2;

    copy_h2d_double(g_ws.d_qpos,   q_pos,      (long)nq * 3);
    copy_h2d_double(g_ws.d_qh,     q_h,        (long)nq);
    copy_h2d_int   (g_ws.d_soff,   starts_off, (long)nq + 1);
    copy_h2d_int   (g_ws.d_starts, starts,     starts_len);
    {
        Kokkos::View<char*, GIZMO_KOKKOS_DEVICE_SPACE, UV> mv(g_ws.d_matched, (size_t)nq * num_pool);
        Kokkos::deep_copy(mv, (char)0);
        Kokkos::View<long*, GIZMO_KOKKOS_DEVICE_SPACE, UV> cv(g_ws.d_cnt, 3);
        Kokkos::deep_copy(cv, (long)0);
    }
    Kokkos::fence();

    /* capture-by-value locals (raw device/UVM pointers + scalars). */
    const double *d_sx = g_sc.d_sx, *d_sy = g_sc.d_sy, *d_sz = g_sc.d_sz, *d_sh = g_sc.d_sh;
    const int    *d_stype = g_sc.d_stype, *d_j2p = g_sc.d_j_to_pool;
    const double *d_band = g_sc.d_band;
    const double *qpos = g_ws.d_qpos, *qh = g_ws.d_qh;
    const int    *soff = g_ws.d_soff, *sarr = g_ws.d_starts;
    char *matched = g_ws.d_matched;
    long *cnt = g_ws.d_cnt;
    const Vec3<MyFloat> *center = soa->center;
    const MyFloat       *lenn   = soa->len;
    const int *sib = soa->sibling, *nxt = soa->nextnode, *nxa = soa->nextnode_aux;
    const unsigned int  *bits = soa->bitflags;
    const int    pf0 = periodic_flags[0], pf1 = periodic_flags[1], pf2 = periodic_flags[2];
    const double bs0 = box_sizes[0], bs1 = box_sizes[1], bs2 = box_sizes[2];
    const int    pseudo_start = maxpart + maxnodes + maxforeign;
    const int    oneway = (search_mode == NGB_SEARCH_ONEWAY);
    const int    np_ = num_pool;

    /* One thread per query; per-query stackless DFS over the start-node list.  The
     * device twin of host gx_walk_fine_tree: particle leaf -> pool via d_j2p +
     * shared gx_pair_accept; internal node -> TOPLEVEL stop / supply-mask-reduced
     * fine-band opener via shared gx_node_sphere_overlap_center_len; pseudo/foreign
     * from a local start -> stop + count (unsupported this step). */
    Kokkos::parallel_for("gx_fine_devwalk", nq, KOKKOS_LAMBDA(const int qi) {
        const double pos_q[3] = { qpos[(long)qi*3+0], qpos[(long)qi*3+1], qpos[(long)qi*3+2] };
        const double h_q = qh[qi];
        const int    periodic_flags_l[3] = { pf0, pf1, pf2 };
        const double box_sizes_l[3]      = { bs0, bs1, bs2 };
        char *mrow = matched + (long)qi * np_;
        const int s0 = soff[qi], s1 = soff[qi+1];
        for(int si = s0; si < s1; si++) {
            int no = sarr[si];
            const int start_node = no;
            long steps = 0;
            while(no >= 0) {
                if(++steps > GX_DEVWALK_STEP_GUARD) { Kokkos::atomic_add(&cnt[2], (long)1); break; }
                if(no < maxpart) {
                    if(no < num_local && no < numpart) {
                        const int pool_pos = d_j2p[no];
                        if(pool_pos >= 0 && pool_pos < np_) {
                            const int pt = d_stype[pool_pos];
                            if(pt >= 0 && pt < num_ptypes && (supply_mask & (1u << (unsigned)pt)) != 0u) {
                                if(gx_pair_accept(pos_q, h_q,
                                                  d_sx[pool_pos], d_sy[pool_pos], d_sz[pool_pos], d_sh[pool_pos],
                                                  search_mode, periodic_flags_l, box_sizes_l))
                                    mrow[pool_pos] = 1;
                            }
                        }
                    }
                    if(no < nn_aux_size) no = nxa[no];
                    else { Kokkos::atomic_add(&cnt[2], (long)1); break; }
                } else if(no < maxpart + maxnodes) {
                    const int idx = no - maxpart;
                    if(idx < 0 || idx >= soa_cap) { Kokkos::atomic_add(&cnt[2], (long)1); break; }
                    if(no != start_node && (bits[idx] & topflag_mask) != 0u) { no = sib[idx]; continue; }
                    double R_eff = h_q;
                    if(!oneway && no < maxpart + nnodes) {
                        const double *bb = d_band + (long)(no - maxpart) * fineband_ntypes;
                        double be = 0;
                        for(int t = 0; t < fineband_ntypes; t++) {
                            if((supply_mask & (1u << (unsigned)t)) == 0u) continue;
                            if(bb[t] > be) be = bb[t];
                        }
                        if(be > R_eff) R_eff = be;
                    }
                    const int do_open = gx_node_sphere_overlap_center_len(
                        pos_q, (double)center[idx][0], (double)center[idx][1], (double)center[idx][2],
                        (double)lenn[idx], R_eff, periodic_flags_l, box_sizes_l);
                    no = do_open ? nxt[idx] : sib[idx];
                } else if(no < pseudo_start) {
                    Kokkos::atomic_add(&cnt[1], (long)1);   /* foreign node from a local start — unsupported */
                    break;
                } else {
                    Kokkos::atomic_add(&cnt[0], (long)1);   /* pseudo node — unsupported */
                    break;
                }
            }
        }
    });
    Kokkos::fence();

    copy_d2h_char(matched_out, g_ws.d_matched, (long)nq * np_);
    long cnt_host[3] = {0, 0, 0};
    copy_d2h_long(cnt_host, g_ws.d_cnt, 3);
    Kokkos::fence();
    if(pseudo_hits)         *pseudo_hits = cnt_host[0];
    if(foreign_node_visits) *foreign_node_visits = cnt_host[1];
    if(bad_index_hits)      *bad_index_hits = cnt_host[2];
    return 0;
}
