/* gpu_shadow_stage.h — compact staging of particle fields between device-resident storage and host shadows.
 *
 * On a PCIe-attached GPU (managed memory that is not host-coherent) a host phase that sits between two device
 * passes would otherwise move the whole particle storage to the host and back.  A shadow stage moves only the
 * fields the phase reads and writes:
 *   gather:  a device kernel packs the listed fields of rows idx[0,n) of P / CellP into one compact buffer, which
 *            is copied to page-locked host memory and unpacked into full-size host records Psh[k] / Csh[k]
 *            (shadows indexed by row k, not by particle);
 *   phase:   the unchanged host code runs on (Psh, Csh, k) instead of (P, CellP, idx[k]);
 *   scatter: the listed written fields of the shadows are packed on the host, copied to the device and written
 *            into P / CellP[idx[k]] by a device kernel.
 * Everything that crosses is a byte copy, so the phase computes exactly what it computes on the storage itself.
 *
 * Field lists name (record, byte offset, bytes) triples built with SHADOW_P / SHADOW_C.  A phase's lists must be
 * exact: every field it reads must be gathered, and every field it writes must be scattered and must be written
 * for EVERY row (a field written for only some rows is also gathered, so the others carry their stored value).
 * GIZMO_STAGE_VERIFY=N checks both for the first N stages of each phase that supports it: the shadows' unlisted
 * bytes are filled with a signalling-NaN pattern (an unlisted read turns into a NaN somewhere in the output), the
 * phase is also run on full copies of the rows, and the written fields are byte-compared with it, as are the
 * copies' unlisted bytes with their state before the phase (an unlisted write).
 *
 * Buffers are grow-only and kept until exit; nothing in them is assumed zero.  No-op stubs and a NULL/failure
 * return on builds without CUDA: callers keep their host path for those.  Include after allvars.h (the row copies
 * below need the complete record types). */

#ifndef GIZMO_GPU_SHADOW_STAGE_H
#define GIZMO_GPU_SHADOW_STAGE_H

#include <stddef.h>
#include <string.h>

struct particle_data;
struct gas_cell_data;

struct ShadowField {int cell; int offset; int bytes;};   /* cell 0: particle_data, 1: gas_cell_data */
#define SHADOW_P(field) ShadowField{0, (int) offsetof(struct particle_data, field), (int) sizeof(((struct particle_data *) 0)->field)}
#define SHADOW_C(field) ShadowField{1, (int) offsetof(struct gas_cell_data, field), (int) sizeof(((struct gas_cell_data *) 0)->field)}
#define SHADOW_MAX_FIELDS 32

/* A field list as the copy loops take it: each field's place in the packed record and its copy unit (bytes). */
struct ShadowFieldSet {
    int n, rec_bytes;
    int cell[SHADOW_MAX_FIELDS], obj_off[SHADOW_MAX_FIELDS], rec_off[SHADOW_MAX_FIELDS], bytes[SHADOW_MAX_FIELDS], unit[SHADOW_MAX_FIELDS];
};

/* Split form of a stage, for a phase that unpacks and packs each row inside its own loop (each shadow row is then
 * brought into cache once):
 *     struct ShadowStage st;
 *     gizmo_shadow_stage_in(&st, idx, n, read, nread, site);      device pack, copy to page-locked host memory
 *     gizmo_shadow_stage_out_begin(&st, written, nwritten);       the page-locked buffer the written rows go into
 *     for each row k, on any thread:
 *         gizmo_shadow_unpack_row(&st, k, Psh, Csh, sentinel);  phase(Psh, Csh, k);  gizmo_shadow_pack_row(&st, k, Psh, Csh);
 *     gizmo_shadow_stage_out_end(&st, site);                      copy to the device, device unpack, fenced
 * Each call returns 0 on success and 1 when its buffers cannot be had (nothing written to the storage). */
struct ShadowStage {
    const int *idx; int n, ngas;
    struct ShadowFieldSet in, out;
    const char *h_in; char *h_out;
};
int gizmo_shadow_stage_in(struct ShadowStage *st, const int *idx, int n, const ShadowField *read, int nread, const char *site);
int gizmo_shadow_stage_out_begin(struct ShadowStage *st, const ShadowField *written, int nwritten);
int gizmo_shadow_stage_out_end(struct ShadowStage *st, const char *site);
void gizmo_shadow_fill_sentinel(void *rec, size_t bytes);
/* Allocate (and touch) now, at startup, what a stage of n rows with these lists needs, including full-size shadow
 * records in slot `slot`: the buffers are grow-only, and pinning them on the first big step costs that step up to
 * seconds when every rank on the node pins at once. */
int gizmo_shadow_stage_reserve(int n, const ShadowField *read, int nread, const ShadowField *written, int nwritten, int slot, int want_cells);
static inline void gizmo_shadow_copy_host(char *dst, const char *src, int bytes, int unit)
{
    if(unit == 8) {for(int b = 0; b < bytes; b += 8) {memcpy(dst + b, src + b, 8);}}
    else if(unit == 4) {for(int b = 0; b < bytes; b += 4) {memcpy(dst + b, src + b, 4);}}
    else {memcpy(dst, src, (size_t) bytes);}
}
static inline void gizmo_shadow_unpack_row(const struct ShadowStage *st, int k, struct particle_data *Psh, struct gas_cell_data *Csh, int sentinel)
{
    char *ps = Psh ? (char *) Psh + (size_t) k * sizeof(struct particle_data) : (char *) 0;
    char *cs = Csh ? (char *) Csh + (size_t) k * sizeof(struct gas_cell_data) : (char *) 0;
    if(sentinel) {if(ps) {gizmo_shadow_fill_sentinel(ps, sizeof(struct particle_data));} if(cs) {gizmo_shadow_fill_sentinel(cs, sizeof(struct gas_cell_data));}}
    const char *rec = st->h_in + (size_t) k * (size_t) st->in.rec_bytes;
    const int has_cell = (cs != 0) && (st->idx[k] < st->ngas);
    for(int s = 0; s < st->in.n; s++) {
        if(st->in.cell[s] ? !has_cell : !ps) {continue;}
        gizmo_shadow_copy_host((st->in.cell[s] ? cs : ps) + st->in.obj_off[s], rec + st->in.rec_off[s], st->in.bytes[s], st->in.unit[s]);
    }
}
static inline void gizmo_shadow_pack_row(const struct ShadowStage *st, int k, const struct particle_data *Psh, const struct gas_cell_data *Csh)
{
    const char *ps = (const char *) Psh + (size_t) k * sizeof(struct particle_data);
    const char *cs = Csh ? (const char *) Csh + (size_t) k * sizeof(struct gas_cell_data) : (const char *) 0;
    char *rec = st->h_out + (size_t) k * (size_t) st->out.rec_bytes;
    const int has_cell = (cs != 0) && (st->idx[k] < st->ngas);
    for(int s = 0; s < st->out.n; s++) {
        if(st->out.cell[s] && !has_cell) {continue;}
        gizmo_shadow_copy_host(rec + st->out.rec_off[s], (st->out.cell[s] ? cs : ps) + st->out.obj_off[s], st->out.bytes[s], st->out.unit[s]);
    }
}

/* Full-size host shadow records for n rows, grow-only, slot 0..3 independent (contents unspecified).  NULL when the
 * memory cannot be had.  want_cells = 0 returns no cell records (*Csh = NULL). */
int gizmo_shadow_records(int slot, int n, int want_cells, struct particle_data **Psh, struct gas_cell_data **Csh);

/* Gather the listed fields of P/CellP[idx[k]] (device-resident storage; idx a host array) into Psh[k] / Csh[k].
 * Cell fields of a row with idx[k] >= N_gas are left alone.  sentinel != 0 first fills every byte of each shadow
 * record with the signalling-NaN pattern (GIZMO_STAGE_VERIFY).  Psh or Csh may be NULL when no field of that record
 * is listed.  Returns 0 on success, 1 when the staging buffers cannot be had (nothing was written). */
int gizmo_shadow_gather(const int *idx, int n, const ShadowField *fields, int nfields,
                        struct particle_data *Psh, struct gas_cell_data *Csh, int sentinel, const char *site);
/* Write the listed fields of Psh[k] / Csh[k] into P/CellP[idx[k]] on the device (cell fields only for idx[k] <
 * N_gas).  Fenced before return.  Returns 0 on success, 1 when the staging buffers cannot be had (nothing was
 * written: the caller writes on the host). */
int gizmo_shadow_scatter(const int *idx, int n, const ShadowField *fields, int nfields,
                         const struct particle_data *Psh, const struct gas_cell_data *Csh, const char *site);

/* GIZMO_STAGE_VERIFY: how many stages of each verifiable phase to verify (0 off). */
int gizmo_stage_verify_level(void);
/* Verify helpers.  _full_copy: the whole records of rows idx[0,n) into Pfull / Cfull (slot 2) and an untouched
 * second copy into Pbefore / Cbefore (slot 3).  _compare: rows [0,n) of the shadows against the full copies over the
 * written fields, and the full copies against their state before over every other byte (cells only for idx[k] <
 * N_gas); prints one line on any rank that differs (and on rank 0 always) and returns the number of differing rows. */
int gizmo_shadow_verify_full_copy(const int *idx, int n, int with_cells, struct particle_data **Pfull, struct gas_cell_data **Cfull,
                                  struct particle_data **Pbefore, struct gas_cell_data **Cbefore);
long long gizmo_shadow_verify_compare(const char *phase, int call, const int *idx, int n, const ShadowField *written, int nwritten,
                                      const struct particle_data *Psh, const struct gas_cell_data *Csh,
                                      const struct particle_data *Pfull, const struct gas_cell_data *Cfull,
                                      const struct particle_data *Pbefore, const struct gas_cell_data *Cbefore);

/* Bytes and seconds of the staged copies so far, and a one-table report on rank 0 (GIZMO_UVM_PREFETCH_REPORT). */
void gizmo_shadow_stage_stats(double *seconds, double *bytes_to_host, double *bytes_to_device, long long *calls);
void gizmo_shadow_stage_report(void);

#endif /* GIZMO_GPU_SHADOW_STAGE_H */
