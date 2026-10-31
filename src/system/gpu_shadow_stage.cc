/* gpu_shadow_stage.cc — compact staging of particle fields between device-resident storage and host shadows
 * (contract in gpu_shadow_stage.h).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <Kokkos_Core.hpp>
#include <vector>

#include "../declarations/gpu_all_mirror.h"   /* must precede allvars.h (see gpu_particles_arena.cc) */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "gpu_shadow_stage.h"

/* ---- bookkeeping ---- */
static double stage_seconds_ = 0, stage_host_seconds_ = 0, stage_bytes_host_ = 0, stage_bytes_dev_ = 0;
static long long stage_calls_ = 0;

void gizmo_shadow_stage_stats(double *seconds, double *bytes_to_host, double *bytes_to_device, long long *calls)
{
    *seconds = stage_seconds_; *bytes_to_host = stage_bytes_host_; *bytes_to_device = stage_bytes_dev_; *calls = stage_calls_;
}

void gizmo_shadow_stage_report(void)
{
    const char *env = getenv("GIZMO_UVM_PREFETCH_REPORT");
    if(ThisTask != 0 || !(env && env[0] && atoi(env) != 0)) {return;}
    printf("Shadow stage copies (rank 0): %lld calls, %.2f GB to host, %.2f GB to device, %.3f s (%.3f s of it in the host pack/unpack loops)\n",
           stage_calls_, stage_bytes_host_ / 1e9, stage_bytes_dev_ / 1e9, stage_seconds_, stage_host_seconds_);
    fflush(stdout);
}

int gizmo_stage_verify_level(void)
{
    static int level = -1;
    if(level < 0) {
        const char *env = getenv("GIZMO_STAGE_VERIFY");
        level = (env && env[0]) ? atoi(env) : 0;
        if(level < 0) {level = 0;}
    }
    return level;
}

/* ---- host shadow records: grow-only, per slot ---- */
static char  *shadow_buf_[4][2] = {{NULL, NULL}, {NULL, NULL}, {NULL, NULL}, {NULL, NULL}};
static size_t shadow_bytes_[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};

static char *shadow_grow(int slot, int which, size_t bytes)
{
    if(bytes == 0) {bytes = 64;}
    if(bytes > shadow_bytes_[slot][which]) {
        free(shadow_buf_[slot][which]);
        const size_t want = bytes + bytes / 8;
        void *p = NULL;
        if(posix_memalign(&p, 64, want) != 0) {p = NULL;}
        shadow_buf_[slot][which] = (char *) p;
        shadow_bytes_[slot][which] = p ? want : 0;
    }
    return shadow_buf_[slot][which];
}

int gizmo_shadow_records(int slot, int n, int want_cells, struct particle_data **Psh, struct gas_cell_data **Csh)
{
    *Psh = NULL; *Csh = NULL;
    if(slot < 0 || slot >= 4 || n < 0) {return 1;}
    char *p = shadow_grow(slot, 0, (size_t) n * sizeof(struct particle_data));
    char *c = want_cells ? shadow_grow(slot, 1, (size_t) n * sizeof(struct gas_cell_data)) : NULL;
    if(!p || (want_cells && !c)) {return 1;}
    *Psh = (struct particle_data *) p;
    *Csh = (struct gas_cell_data *) c;
    return 0;
}

#if defined(KOKKOS_ENABLE_CUDA)
/* ---- staging buffers: one device and one page-locked host buffer, grow-only, never assumed zero ---- */
static char  *stage_dev_ = NULL, *stage_pin_ = NULL;
static size_t stage_dev_bytes_ = 0, stage_pin_bytes_ = 0;

static char *stage_device(size_t bytes)
{
    if(bytes > stage_dev_bytes_) {
        if(stage_dev_) {gizmo_gpu_free_device(stage_dev_);}
        const size_t want = bytes + bytes / 4;
        stage_dev_ = (char *) gizmo_gpu_alloc_device(want, "shadow_stage_device");
        stage_dev_bytes_ = stage_dev_ ? want : 0;
    }
    return stage_dev_;
}
static char *stage_pinned(size_t bytes)
{
    if(bytes > stage_pin_bytes_) {
        if(stage_pin_) {gizmo_gpu_free_pinned_host(stage_pin_);}
        const size_t want = bytes + bytes / 4;
        stage_pin_ = (char *) gizmo_gpu_alloc_pinned_host(want, "shadow_stage_pinned");
        stage_pin_bytes_ = stage_pin_ ? want : 0;
    }
    return stage_pin_;
}

static int shadow_field_set(const ShadowField *f, int nf, ShadowFieldSet &s)
{
    if(nf < 0 || nf > SHADOW_MAX_FIELDS) {return 1;}
    s.n = nf;
    int off = 0;
    for(int k = 0; k < nf; k++) {
        s.cell[k] = f[k].cell; s.obj_off[k] = f[k].offset; s.bytes[k] = f[k].bytes; s.rec_off[k] = off;
        const int a = f[k].offset | f[k].bytes;   /* rec_off is always a multiple of 8 */
        s.unit[k] = (a % 8 == 0) ? 8 : (a % 4 == 0) ? 4 : (a % 2 == 0) ? 2 : 1;
        off += (f[k].bytes + 7) & ~7;
    }
    s.rec_bytes = off > 0 ? off : 8;
    return 0;
}

KOKKOS_INLINE_FUNCTION void shadow_copy_unit(char *dst, const char *src, int bytes, int unit)
{
    if(unit == 8) {for(int b = 0; b < bytes; b += 8) {*(uint64_t *) (dst + b) = *(const uint64_t *) (src + b);}}
    else if(unit == 4) {for(int b = 0; b < bytes; b += 4) {*(uint32_t *) (dst + b) = *(const uint32_t *) (src + b);}}
    else if(unit == 2) {for(int b = 0; b < bytes; b += 2) {*(uint16_t *) (dst + b) = *(const uint16_t *) (src + b);}}
    else {for(int b = 0; b < bytes; b++) {dst[b] = src[b];}}
}

#endif

void gizmo_shadow_fill_sentinel(void *rec, size_t bytes)
{
    const uint64_t snan = 0x7FF4000000000000ULL;   /* signalling NaN as a double */
    for(size_t b = 0; b + 8 <= bytes; b += 8) {memcpy((char *) rec + b, &snan, 8);}
}

#if defined(KOKKOS_ENABLE_CUDA)
/* A second page-locked buffer, so the rows read in and the rows written out of one stage never share one. */
static char  *stage_pin_out_ = NULL;
static size_t stage_pin_out_bytes_ = 0;
static char *stage_pinned_out(size_t bytes)
{
    if(bytes > stage_pin_out_bytes_) {
        if(stage_pin_out_) {gizmo_gpu_free_pinned_host(stage_pin_out_);}
        const size_t want = bytes + bytes / 4;
        stage_pin_out_ = (char *) gizmo_gpu_alloc_pinned_host(want, "shadow_stage_pinned_out");
        stage_pin_out_bytes_ = stage_pin_out_ ? want : 0;
    }
    return stage_pin_out_;
}
static void stage_upload_indices(int *d_idx, const int *idx, int n)
{
    Kokkos::View<const int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hv(idx, (size_t) n);
    Kokkos::View<int *, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dv(d_idx, (size_t) n);
    Kokkos::deep_copy(dv, hv);
}
#endif

int gizmo_shadow_stage_in(struct ShadowStage *st, const int *idx, int n, const ShadowField *read, int nread, const char *site)
{
    (void) site;
    st->idx = idx; st->n = n; st->ngas = N_gas; st->h_in = NULL; st->h_out = NULL; st->out.n = 0;
#if defined(KOKKOS_ENABLE_CUDA)
    if(shadow_field_set(read, nread, st->in)) {return 1;}
    if(n <= 0) {return 0;}
    const ShadowFieldSet fs = st->in;
    const size_t idx_bytes = ((size_t) n * sizeof(int) + 255) & ~(size_t) 255;
    const size_t rec_total = (size_t) n * (size_t) fs.rec_bytes;
    char *d = stage_device(idx_bytes + rec_total);
    char *h = stage_pinned(rec_total);
    if(!d || !h) {return 1;}
    const double t0 = my_second();
    int *d_idx = (int *) d;
    char *d_rec = d + idx_bytes;
    stage_upload_indices(d_idx, idx, n);
    const char *Pb = (const char *) P;
    const char *Cb = (const char *) CellP;
    const int ngas = N_gas;
    const size_t sp = sizeof(struct particle_data), sc = sizeof(struct gas_cell_data);
    const int rb = fs.rec_bytes;
    Kokkos::parallel_for("shadow_gather_pack", n, KOKKOS_LAMBDA(int k) {
        const int i = d_idx[k];
        char *rec = d_rec + (size_t) k * rb;
        for(int s = 0; s < fs.n; s++) {
            if(fs.cell[s] && (!Cb || i >= ngas)) {continue;}
            const char *src = (fs.cell[s] ? Cb + (size_t) i * sc : Pb + (size_t) i * sp) + fs.obj_off[s];
            shadow_copy_unit(rec + fs.rec_off[s], src, fs.bytes[s], fs.unit[s]);
        }
    });
    {
        Kokkos::View<char *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hv(h, rec_total);
        Kokkos::View<const char *, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dv(d_rec, rec_total);
        Kokkos::deep_copy(hv, dv);   /* fences the pack first */
    }
    st->h_in = h;
    stage_seconds_ += timediff(t0, my_second());
    stage_bytes_host_ += (double) rec_total;
    stage_calls_++;
    return 0;
#else
    (void) read; (void) nread;
    return 1;
#endif
}

int gizmo_shadow_stage_out_begin(struct ShadowStage *st, const ShadowField *written, int nwritten)
{
#if defined(KOKKOS_ENABLE_CUDA)
    if(shadow_field_set(written, nwritten, st->out)) {return 1;}
    if(st->n <= 0) {return 0;}
    st->h_out = stage_pinned_out((size_t) st->n * (size_t) st->out.rec_bytes);
    return st->h_out ? 0 : 1;
#else
    (void) st; (void) written; (void) nwritten;
    return 1;
#endif
}

int gizmo_shadow_stage_out_end(struct ShadowStage *st, const char *site)
{
    (void) site;
    const int n = st->n;
    if(n <= 0) {return 0;}
#if defined(KOKKOS_ENABLE_CUDA)
    const ShadowFieldSet fs = st->out;
    const size_t idx_bytes = ((size_t) n * sizeof(int) + 255) & ~(size_t) 255;
    const size_t rec_total = (size_t) n * (size_t) fs.rec_bytes;
    char *d = stage_device(idx_bytes + rec_total);
    if(!d || !st->h_out) {return 1;}
    const double t0 = my_second();
    int *d_idx = (int *) d;
    char *d_rec = d + idx_bytes;
    stage_upload_indices(d_idx, st->idx, n);
    {
        Kokkos::View<const char *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hr(st->h_out, rec_total);
        Kokkos::View<char *, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dr(d_rec, rec_total);
        Kokkos::deep_copy(dr, hr);
    }
    char *Pb = (char *) P;
    char *Cb = (char *) CellP;
    const int ngas = N_gas;
    const size_t sp = sizeof(struct particle_data), sc = sizeof(struct gas_cell_data);
    const int rb = fs.rec_bytes;
    Kokkos::parallel_for("shadow_scatter_unpack", n, KOKKOS_LAMBDA(int k) {
        const int i = d_idx[k];
        const char *rec = d_rec + (size_t) k * rb;
        for(int s = 0; s < fs.n; s++) {
            if(fs.cell[s] && (!Cb || i >= ngas)) {continue;}
            char *dst = (fs.cell[s] ? Cb + (size_t) i * sc : Pb + (size_t) i * sp) + fs.obj_off[s];
            shadow_copy_unit(dst, rec + fs.rec_off[s], fs.bytes[s], fs.unit[s]);
        }
    });
    Kokkos::fence("shadow_scatter_unpack");
    stage_seconds_ += timediff(t0, my_second());
    stage_bytes_dev_ += (double) rec_total;
    stage_calls_++;
    return 0;
#else
    return 1;
#endif
}

int gizmo_shadow_stage_reserve(int n, const ShadowField *read, int nread, const ShadowField *written, int nwritten, int slot, int want_cells)
{
    if(n <= 0) {return 0;}
    struct particle_data *Psh = NULL; struct gas_cell_data *Csh = NULL;
    if(gizmo_shadow_records(slot, n, want_cells, &Psh, &Csh)) {return 1;}
    memset((void *) Psh, 0, (size_t) n * sizeof(struct particle_data));
    if(Csh) {memset((void *) Csh, 0, (size_t) n * sizeof(struct gas_cell_data));}
#if defined(KOKKOS_ENABLE_CUDA)
    ShadowFieldSet in, out;
    if(shadow_field_set(read, nread, in) || shadow_field_set(written, nwritten, out)) {return 1;}
    const size_t idx_bytes = ((size_t) n * sizeof(int) + 255) & ~(size_t) 255;
    const size_t rec_max = (size_t) n * (size_t) (in.rec_bytes > out.rec_bytes ? in.rec_bytes : out.rec_bytes);
    char *h_in = stage_pinned((size_t) n * (size_t) in.rec_bytes);
    char *h_out = stage_pinned_out((size_t) n * (size_t) out.rec_bytes);
    if(!stage_device(idx_bytes + rec_max) || !h_in || !h_out) {return 1;}
    memset(h_in, 0, (size_t) n * (size_t) in.rec_bytes);
    memset(h_out, 0, (size_t) n * (size_t) out.rec_bytes);
#else
    (void) read; (void) nread; (void) written; (void) nwritten;
#endif
    return 0;
}

int gizmo_shadow_gather(const int *idx, int n, const ShadowField *fields, int nfields,
                        struct particle_data *Psh, struct gas_cell_data *Csh, int sentinel, const char *site)
{
    struct ShadowStage st;
    if(gizmo_shadow_stage_in(&st, idx, n, fields, nfields, site)) {return 1;}
    const double th0 = my_second();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n >= 4096)
#endif
    for(int k = 0; k < n; k++) {gizmo_shadow_unpack_row(&st, k, Psh, Csh, sentinel);}
    stage_host_seconds_ += timediff(th0, my_second());
    return 0;
}

int gizmo_shadow_scatter(const int *idx, int n, const ShadowField *fields, int nfields,
                         const struct particle_data *Psh, const struct gas_cell_data *Csh, const char *site)
{
    struct ShadowStage st;
    st.idx = idx; st.n = n; st.ngas = N_gas; st.h_in = NULL; st.in.n = 0;
    if(gizmo_shadow_stage_out_begin(&st, fields, nfields)) {return 1;}
    const double th0 = my_second();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n >= 4096)
#endif
    for(int k = 0; k < n; k++) {gizmo_shadow_pack_row(&st, k, Psh, Csh);}
    stage_host_seconds_ += timediff(th0, my_second());
    return gizmo_shadow_stage_out_end(&st, site);
}

int gizmo_shadow_verify_full_copy(const int *idx, int n, int with_cells, struct particle_data **Pfull, struct gas_cell_data **Cfull,
                                  struct particle_data **Pbefore, struct gas_cell_data **Cbefore)
{
    if(gizmo_shadow_records(2, n, with_cells, Pfull, Cfull) || gizmo_shadow_records(3, n, with_cells, Pbefore, Cbefore)) {return 1;}
    ShadowField whole[2] = {ShadowField{0, 0, (int) sizeof(struct particle_data)}, ShadowField{1, 0, (int) sizeof(struct gas_cell_data)}};
    if(gizmo_shadow_gather(idx, n, whole, with_cells ? 2 : 1, *Pfull, *Cfull, 0, "stage_verify")) {return 1;}
    memcpy(*Pbefore, *Pfull, (size_t) n * sizeof(struct particle_data));
    if(with_cells) {memcpy(*Cbefore, *Cfull, (size_t) n * sizeof(struct gas_cell_data));}
    return 0;
}

long long gizmo_shadow_verify_compare(const char *phase, int call, const int *idx, int n, const ShadowField *written, int nwritten,
                                      const struct particle_data *Psh, const struct gas_cell_data *Csh,
                                      const struct particle_data *Pfull, const struct gas_cell_data *Cfull,
                                      const struct particle_data *Pbefore, const struct gas_cell_data *Cbefore)
{
    const size_t sp = sizeof(struct particle_data), sc = sizeof(struct gas_cell_data);
    std::vector<unsigned char> wmask_p(sp, 0), wmask_c(sc, 0);
    for(int s = 0; s < nwritten; s++) {
        unsigned char *m = written[s].cell ? wmask_c.data() : wmask_p.data();
        for(int b = 0; b < written[s].bytes; b++) {m[written[s].offset + b] = 1;}
    }
    std::vector<long long> field_diff((size_t) (nwritten > 0 ? nwritten : 1), 0);
    long long rows_written_diff = 0, rows_unlisted_write = 0;
    int first_unlisted_off = -1, first_unlisted_cell = -1;
    for(int k = 0; k < n; k++) {
        const int has_cell = Csh && Cfull && Cbefore && idx[k] < N_gas;
        int bad = 0;
        for(int s = 0; s < nwritten; s++) {
            if(written[s].cell && !has_cell) {continue;}
            const char *a = (written[s].cell ? (const char *) &Csh[k] : (const char *) &Psh[k]) + written[s].offset;
            const char *b = (written[s].cell ? (const char *) &Cfull[k] : (const char *) &Pfull[k]) + written[s].offset;
            if(memcmp(a, b, (size_t) written[s].bytes) != 0) {field_diff[(size_t) s]++; bad = 1;}
        }
        rows_written_diff += bad;
        int ubad = 0;
        const char *pa = (const char *) &Pfull[k], *pb = (const char *) &Pbefore[k];
        for(size_t b = 0; b < sp; b++) {if(!wmask_p[b] && pa[b] != pb[b]) {ubad = 1; if(first_unlisted_off < 0) {first_unlisted_off = (int) b; first_unlisted_cell = 0;} break;}}
        if(has_cell) {
            const char *ca = (const char *) &Cfull[k], *cb = (const char *) &Cbefore[k];
            for(size_t b = 0; b < sc; b++) {if(!wmask_c[b] && ca[b] != cb[b]) {ubad = 1; if(first_unlisted_off < 0) {first_unlisted_off = (int) b; first_unlisted_cell = 1;} break;}}
        }
        rows_unlisted_write += ubad;
    }
    if(ThisTask == 0 || rows_written_diff || rows_unlisted_write) {
        char fields_txt[512]; fields_txt[0] = 0;
        size_t used = 0;
        for(int s = 0; s < nwritten && used + 32 < sizeof(fields_txt); s++) {
            if(field_diff[(size_t) s]) {used += (size_t) snprintf(fields_txt + used, sizeof(fields_txt) - used, " %c@%d:%lld", written[s].cell ? 'C' : 'P', written[s].offset, field_diff[(size_t) s]);}
        }
        printf("Stage verify %s (task %d, call %d): %d rows; %lld rows differ in written fields%s; %lld rows with an unlisted write%s",
               phase, ThisTask, call, n, rows_written_diff, used ? fields_txt : "", rows_unlisted_write,
               first_unlisted_off >= 0 ? "" : "\n");
        if(first_unlisted_off >= 0) {printf(" (first at %s byte %d)\n", first_unlisted_cell ? "cell" : "particle", first_unlisted_off);}
        fflush(stdout);
    }
    return rows_written_diff + rows_unlisted_write;
}
