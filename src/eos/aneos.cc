#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aneos.h"

#ifdef EOS_ANEOS

/* ============================================================
   ANEOS: SESAME-format tabulated equation of state

   Tables are 2D grids in (log10 rho, log10 T) with columns:
     rho [g/cm^3], T [K], P [dyn/cm^2], u [erg/g], S [erg/g/K],
     optionally cs [cm/s] and phase (int).

   Interpolation is bilinear in log-log space for P, u, S, cs.
   Phase uses nearest-neighbor lookup (no interpolation on integers).

   Temperature inversion (rho, u) -> T uses Newton-Raphson with
   bisection fallback.
   ============================================================ */


#include "../core/proto.h"                  /* gizmo_gpu_alloc_shared */
#include "../system/gpu_particles_arena.h"  /* gpu_particles_uvm_free */

/* The descriptor array and every array it points at are allocated in memory the
   host and the device can both read, once, here. They are read-only after
   startup and are never copied again: the lookups in aneos.h run inside a
   per-particle Newton iteration, so staging this table per call would cost more
   to move than the solve it serves. Nothing frees them before the process ends,
   which is the policy every other physics table in the code already follows. */
struct aneos_table *ANEOS_Tables = NULL;
int ANEOS_Num_Tables_Loaded = 0;

/* Allocate the descriptor array on first use. begrun calls aneos_read_table
   once per material, well after Kokkos is initialised in main, so there is no
   window in which a table could be read before this has run. */
static int aneos_tables_ready(void)
{
    if(ANEOS_Tables) {return 1;}
    ANEOS_Tables = (struct aneos_table *) gizmo_gpu_alloc_shared(
        ANEOS_MAX_MATERIALS * sizeof(struct aneos_table), "aneos_tables");
    if(!ANEOS_Tables) {
        fprintf(stderr, "ANEOS: could not allocate the table descriptors\n");
        return 0;
    }
    memset(ANEOS_Tables, 0, ANEOS_MAX_MATERIALS * sizeof(struct aneos_table));
    return 1;
}

/* Table interiors come from the same place, for the same reason. calloc's
   zeroing is reproduced explicitly because the shared allocator does not
   promise it. Returns NULL the way calloc does, so the callers below are
   unchanged. */
static void *aneos_shared_calloc(size_t n, size_t sz, const char *label)
{
    void *p = gizmo_gpu_alloc_shared(n * sz, label);
    if(p) {memset(p, 0, n * sz);}
    return p;
}

/* ---- Table I/O ---- */

/* Read a SESAME-format ASCII table. Expected format:
   Line 1: mat_id  nrho  nT  ncols
   Lines 2..(1+nrho): density values [g/cm^3] (one per line or space-separated)
   Next nT values: temperature values [K]
   Then nrho*nT rows, each with ncols columns:
     rho  T  P  u  S  [cs]  [phase]
   ncols=5: (rho, T, P, u, S)
   ncols=6: (rho, T, P, u, S, cs)
   ncols=7: (rho, T, P, u, S, cs, phase)

   Grid must be uniform in log10 space. */
int aneos_read_table(const char *filename, int mat_index)
{
    if(mat_index < 0 || mat_index >= ANEOS_MAX_MATERIALS) {
        fprintf(stderr, "ANEOS: mat_index %d out of range [0, %d)\n", mat_index, ANEOS_MAX_MATERIALS);
        return 1;
    }

    if(!aneos_tables_ready()) {return 1;}

    FILE *fp = fopen(filename, "r");
    if(!fp) {
        fprintf(stderr, "ANEOS: could not open table file \"%s\"\n", filename);
        return 1;
    }

    struct aneos_table *tbl = &ANEOS_Tables[mat_index];
    memset(tbl, 0, sizeof(*tbl));

    /* Read header */
    int ncols;
    if(fscanf(fp, "%d %d %d %d", &tbl->mat_id, &tbl->nrho, &tbl->nT, &ncols) != 4) {
        fprintf(stderr, "ANEOS: error reading header from \"%s\"\n", filename);
        fclose(fp); return 1;
    }

    int nrho = tbl->nrho, nT = tbl->nT, ntot = nrho * nT;
    if(nrho < 2 || nT < 2) {
        fprintf(stderr, "ANEOS: table too small: nrho=%d nT=%d\n", nrho, nT);
        fclose(fp); return 1;
    }
    if(ncols < 5 || ncols > 7) {
        fprintf(stderr, "ANEOS: unexpected ncols=%d (expected 5, 6, or 7)\n", ncols);
        fclose(fp); return 1;
    }

    tbl->has_csound = (ncols >= 6);
    tbl->has_phase  = (ncols >= 7);

    /* Allocate grid arrays */
    tbl->logrho = (double *)aneos_shared_calloc(nrho, sizeof(double), "aneos_logrho");
    tbl->logT   = (double *)aneos_shared_calloc(nT, sizeof(double), "aneos_logT");

    /* Allocate 2D arrays */
    tbl->log_pressure = (double *)aneos_shared_calloc(ntot, sizeof(double), "aneos_log_pressure");
    tbl->log_energy   = (double *)aneos_shared_calloc(ntot, sizeof(double), "aneos_log_energy");
    tbl->log_entropy  = (double *)aneos_shared_calloc(ntot, sizeof(double), "aneos_log_entropy");
    if(tbl->has_csound)
        tbl->log_csound = (double *)aneos_shared_calloc(ntot, sizeof(double), "aneos_log_csound");
    if(tbl->has_phase)
        tbl->phase = (int *)aneos_shared_calloc(ntot, sizeof(int), "aneos_phase");

    /* Read density grid */
    for(int i = 0; i < nrho; i++) {
        double val;
        if(fscanf(fp, "%lf", &val) != 1) {
            fprintf(stderr, "ANEOS: error reading density grid at index %d\n", i);
            fclose(fp); return 1;
        }
        tbl->logrho[i] = log10(val);
    }

    /* Read temperature grid */
    for(int j = 0; j < nT; j++) {
        double val;
        if(fscanf(fp, "%lf", &val) != 1) {
            fprintf(stderr, "ANEOS: error reading temperature grid at index %d\n", j);
            fclose(fp); return 1;
        }
        tbl->logT[j] = log10(val);
    }

    /* Read 2D data: nrho*nT rows, each with ncols columns
       Row order: outer loop over rho (slow), inner loop over T (fast)
       Columns: rho T P u S [cs] [phase] */
    for(int i = 0; i < nrho; i++) {
        for(int j = 0; j < nT; j++) {
            int idx = i * nT + j;
            double rho_val, T_val, P_val, u_val, S_val, cs_val = 0;
            int phase_val = 0;

            if(ncols == 5) {
                if(fscanf(fp, "%lf %lf %lf %lf %lf", &rho_val, &T_val, &P_val, &u_val, &S_val) != 5) {
                    fprintf(stderr, "ANEOS: error reading data row irho=%d iT=%d\n", i, j);
                    fclose(fp); return 1;
                }
            } else if(ncols == 6) {
                if(fscanf(fp, "%lf %lf %lf %lf %lf %lf", &rho_val, &T_val, &P_val, &u_val, &S_val, &cs_val) != 6) {
                    fprintf(stderr, "ANEOS: error reading data row irho=%d iT=%d\n", i, j);
                    fclose(fp); return 1;
                }
            } else {
                if(fscanf(fp, "%lf %lf %lf %lf %lf %lf %d", &rho_val, &T_val, &P_val, &u_val, &S_val, &cs_val, &phase_val) != 7) {
                    fprintf(stderr, "ANEOS: error reading data row irho=%d iT=%d\n", i, j);
                    fclose(fp); return 1;
                }
            }

            /* Store in log10 space. Handle zero/negative values by flooring. */
            tbl->log_pressure[idx] = (P_val > 0) ? log10(P_val) : -30.0;
            tbl->log_energy[idx]   = (u_val > 0) ? log10(u_val) : -30.0;
            tbl->log_entropy[idx]  = (S_val > 0) ? log10(S_val) : -30.0;
            if(tbl->has_csound)
                tbl->log_csound[idx] = (cs_val > 0) ? log10(cs_val) : -30.0;
            if(tbl->has_phase)
                tbl->phase[idx] = phase_val;
        }
    }
    fclose(fp);

    /* Compute grid metadata */
    tbl->logrho_min = tbl->logrho[0];
    tbl->logrho_max = tbl->logrho[nrho - 1];
    tbl->logT_min   = tbl->logT[0];
    tbl->logT_max   = tbl->logT[nT - 1];
    tbl->d_logrho   = (tbl->logrho_max - tbl->logrho_min) / (nrho - 1);
    tbl->d_logT     = (tbl->logT_max - tbl->logT_min) / (nT - 1);
    tbl->inv_d_logrho = 1.0 / tbl->d_logrho;
    tbl->inv_d_logT   = 1.0 / tbl->d_logT;

    tbl->loaded = 1;

    printf("ANEOS: loaded material %d from \"%s\" (%d x %d grid, %d columns)\n",
           tbl->mat_id, filename, nrho, nT, ncols);
    printf("  log10(rho) range: [%.3f, %.3f],  log10(T) range: [%.3f, %.3f]\n",
           tbl->logrho_min, tbl->logrho_max, tbl->logT_min, tbl->logT_max);

    if(mat_index >= ANEOS_Num_Tables_Loaded)
        ANEOS_Num_Tables_Loaded = mat_index + 1;

    return 0;
}


int aneos_cleanup(void)
{
    if(!ANEOS_Tables) {return 0;}
    for(int k = 0; k < ANEOS_MAX_MATERIALS; k++) {
        struct aneos_table *tbl = &ANEOS_Tables[k];
        if(!tbl->loaded) continue;
        /* Freed through the allocator that served them: these came from shared
           space, and handing them to free() is undefined. The descriptor array
           itself is left in place, as the other physics tables are -- it is one
           small allocation and nothing reads it once the slots are clear. */
        gpu_particles_uvm_free(tbl->logrho);       tbl->logrho = NULL;
        gpu_particles_uvm_free(tbl->logT);         tbl->logT = NULL;
        gpu_particles_uvm_free(tbl->log_pressure); tbl->log_pressure = NULL;
        gpu_particles_uvm_free(tbl->log_energy);   tbl->log_energy = NULL;
        gpu_particles_uvm_free(tbl->log_entropy);  tbl->log_entropy = NULL;
        gpu_particles_uvm_free(tbl->log_csound);   tbl->log_csound = NULL;
        gpu_particles_uvm_free(tbl->phase);        tbl->phase = NULL;
        tbl->loaded = 0;
    }
    ANEOS_Num_Tables_Loaded = 0;
    return 0;
}

#endif /* EOS_ANEOS */
