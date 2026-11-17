#include <limits.h>
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

   Interpolation is bilinear in (log10 rho, log10 T) on the table's own, arbitrarily
   spaced axes: log10 of P, u and cs, and entropy as a value (it may be negative).
   Uses nearest-neighbor lookup (no interpolation on integers).

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

   The axes must be positive and strictly increasing; their spacing is arbitrary.
   Each row repeats its (rho, T), which must match the axes to ANEOS_AXIS_MATCH_RTOL
   (a table writer may print the rows with fewer digits than the axes). A table
   failing any check is refused rather than read as something it is not. */
#define ANEOS_AXIS_MATCH_RTOL 1.0e-8

/* Free every array a table slot holds (through the allocator that served them:
   these came from shared space, and handing them to free() is undefined) and
   mark the slot empty. */
static void aneos_free_table_arrays(struct aneos_table *tbl)
{
    double **dbl[] = {&tbl->logrho, &tbl->logT, &tbl->log_pressure, &tbl->log_energy, &tbl->entropy, &tbl->log_csound};
    if(tbl->fold_logu) {gpu_particles_uvm_free(tbl->fold_logu); tbl->fold_logu = NULL;}
    for(int k = 0; k < (int)(sizeof(dbl) / sizeof(dbl[0])); k++) {if(*dbl[k]) {gpu_particles_uvm_free(*dbl[k]); *dbl[k] = NULL;}}
    int **ints[] = {&tbl->phase, &tbl->map_rho, &tbl->map_T, &tbl->lowest_valid_T};
    for(int k = 0; k < (int)(sizeof(ints) / sizeof(ints[0])); k++) {if(*ints[k]) {gpu_particles_uvm_free(*ints[k]); *ints[k] = NULL;}}
    tbl->loaded = 0;
}

/* refuse the table: report why, release what was allocated, close the file, and return the error code */
static int aneos_refuse(FILE *fp, struct aneos_table *tbl, const char *filename, const char *why, int index)
{
    fprintf(stderr, "ANEOS: refusing table \"%s\": %s (index %d)\n", filename, why, index);
    aneos_free_table_arrays(tbl);
    if(fp) {fclose(fp);}
    return 1;
}

/* Build the coarse search map of one axis: nmap equal bins, map[b] = the lowest
   interval (largest k <= n-2 with axis[k] <= left edge of bin b). */
static void aneos_build_axis_map(const double *axis, int n, int nmap, int *map, double *inv_binwidth)
{
    double width = (axis[n - 1] - axis[0]) / nmap;
    *inv_binwidth = 1.0 / width;
    int k = 0;
    for(int b = 0; b <= nmap; b++) {
        double edge = (b == nmap) ? axis[n - 1] : axis[0] + b * width;
        while(k < n - 2 && axis[k + 1] <= edge) {k++;}
        map[b] = k;
    }
}

/* largest relative departure of the axis spacing from its mean (0 for a uniform axis) */
static double aneos_axis_nonuniformity(const double *axis, int n)
{
    double mean = (axis[n - 1] - axis[0]) / (n - 1), worst = 0;
    for(int k = 0; k < n - 1; k++) {double d = fabs((axis[k + 1] - axis[k]) / mean - 1.0); if(d > worst) {worst = d;}}
    return worst;
}

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
    aneos_free_table_arrays(tbl); /* a slot read again releases what it held */
    memset(tbl, 0, sizeof(*tbl));

    /* Read header */
    int ncols;
    if(fscanf(fp, "%d %d %d %d", &tbl->mat_id, &tbl->nrho, &tbl->nT, &ncols) != 4) {return aneos_refuse(fp, tbl, filename, "header unreadable", 0);}

    int nrho = tbl->nrho, nT = tbl->nT;
    if(nrho < 2 || nT < 2) {return aneos_refuse(fp, tbl, filename, "fewer than two points on an axis", (nrho < 2) ? nrho : nT);}
    /* lookups index the 2D arrays with int arithmetic (irho * nT + iT); the search maps hold 4n+1 entries */
    if(nrho > INT_MAX / 4 - 1 || nT > INT_MAX / 4 - 1) {return aneos_refuse(fp, tbl, filename, "axis too long for the search map", (nrho > nT) ? nrho : nT);}
    if((long long)nrho * (long long)nT > (long long)INT_MAX) {return aneos_refuse(fp, tbl, filename, "nrho*nT exceeds the int index range", nrho);}
    if(ncols < 5 || ncols > 7) {return aneos_refuse(fp, tbl, filename, "ncols must be 5, 6 or 7", ncols);}
    int ntot = nrho * nT;

    tbl->has_csound = (ncols >= 6);
    tbl->has_phase  = (ncols >= 7);
    tbl->nmap_rho = 4 * nrho;
    tbl->nmap_T   = 4 * nT;

    tbl->logrho = (double *)aneos_shared_calloc(nrho, sizeof(double), "aneos_logrho");
    tbl->logT   = (double *)aneos_shared_calloc(nT, sizeof(double), "aneos_logT");
    tbl->map_rho = (int *)aneos_shared_calloc(tbl->nmap_rho + 1, sizeof(int), "aneos_map_rho");
    tbl->map_T   = (int *)aneos_shared_calloc(tbl->nmap_T + 1, sizeof(int), "aneos_map_T");
    tbl->log_pressure = (double *)aneos_shared_calloc(ntot, sizeof(double), "aneos_log_pressure");
    tbl->log_energy   = (double *)aneos_shared_calloc(ntot, sizeof(double), "aneos_log_energy");
    tbl->entropy      = (double *)aneos_shared_calloc(ntot, sizeof(double), "aneos_entropy");
    if(tbl->has_csound) {tbl->log_csound = (double *)aneos_shared_calloc(ntot, sizeof(double), "aneos_log_csound");}
    if(tbl->has_phase)  {tbl->phase = (int *)aneos_shared_calloc(ntot, sizeof(int), "aneos_phase");}
    if(!tbl->logrho || !tbl->logT || !tbl->map_rho || !tbl->map_T || !tbl->log_pressure || !tbl->log_energy || !tbl->entropy
       || (tbl->has_csound && !tbl->log_csound) || (tbl->has_phase && !tbl->phase)) {return aneos_refuse(fp, tbl, filename, "could not allocate the table", ntot);}

    /* Read the axes: positive, finite, strictly increasing. They are held as the values read
       until the rows have been checked against them, then converted to log10 in place. */
    for(int i = 0; i < nrho; i++) {
        double val;
        if(fscanf(fp, "%lf", &val) != 1) {return aneos_refuse(fp, tbl, filename, "density axis unreadable", i);}
        if(!(isfinite(val) && val > 0)) {return aneos_refuse(fp, tbl, filename, "density axis value not positive and finite", i);}
        if(i > 0 && !(val > tbl->logrho[i - 1])) {return aneos_refuse(fp, tbl, filename, "density axis not strictly increasing", i);}
        tbl->logrho[i] = val;
    }
    for(int j = 0; j < nT; j++) {
        double val;
        if(fscanf(fp, "%lf", &val) != 1) {return aneos_refuse(fp, tbl, filename, "temperature axis unreadable", j);}
        if(!(isfinite(val) && val > 0)) {return aneos_refuse(fp, tbl, filename, "temperature axis value not positive and finite", j);}
        if(j > 0 && !(val > tbl->logT[j - 1])) {return aneos_refuse(fp, tbl, filename, "temperature axis not strictly increasing", j);}
        tbl->logT[j] = val;
    }

    /* Read 2D data: nrho*nT rows, each with ncols columns
       Row order: outer loop over rho (slow), inner loop over T (fast)
       Columns: rho T P u S [cs] [phase] */
    for(int i = 0; i < nrho; i++) {
        for(int j = 0; j < nT; j++) {
            int idx = i * nT + j;
            double rho_val, T_val, P_val, u_val, S_val, cs_val = 0;
            int phase_val = 0, nread;

            if(ncols == 5) {
                nread = fscanf(fp, "%lf %lf %lf %lf %lf", &rho_val, &T_val, &P_val, &u_val, &S_val);
            } else if(ncols == 6) {
                nread = fscanf(fp, "%lf %lf %lf %lf %lf %lf", &rho_val, &T_val, &P_val, &u_val, &S_val, &cs_val);
            } else {
                nread = fscanf(fp, "%lf %lf %lf %lf %lf %lf %d", &rho_val, &T_val, &P_val, &u_val, &S_val, &cs_val, &phase_val);
            }
            if(nread != ncols) {return aneos_refuse(fp, tbl, filename, "data row unreadable", idx);}
            if(!(fabs(rho_val / tbl->logrho[i] - 1.0) <= ANEOS_AXIS_MATCH_RTOL)) {return aneos_refuse(fp, tbl, filename, "data row density does not match the density axis", idx);}
            if(!(fabs(T_val / tbl->logT[j] - 1.0) <= ANEOS_AXIS_MATCH_RTOL)) {return aneos_refuse(fp, tbl, filename, "data row temperature does not match the temperature axis", idx);}
            if(!(isfinite(P_val) && isfinite(u_val) && isfinite(S_val) && isfinite(cs_val))) {return aneos_refuse(fp, tbl, filename, "non-finite value in data row", idx);}

            /* P, u and cs are interpolated in log10; zero/negative values are floored */
            tbl->log_pressure[idx] = (P_val > 0) ? log10(P_val) : -30.0;
            tbl->log_energy[idx]   = (u_val > 0) ? log10(u_val) : -30.0;
            tbl->entropy[idx]      = S_val;
            if(tbl->has_csound)
                tbl->log_csound[idx] = (cs_val > 0) ? log10(cs_val) : -30.0;
            if(tbl->has_phase)
                tbl->phase[idx] = phase_val;
        }
    }
    /* Optional trailing block: per density, the index of the lowest valid temperature row (a table whose cold corner
       holds no valid states marks it; below that row a lookup returns the state at the row) */
    char block_name[64];
    if(fscanf(fp, " %63s", block_name) == 1) {
        if(strcmp(block_name, "lowest_valid_T_index") != 0) {return aneos_refuse(fp, tbl, filename, "unrecognised block after the data rows", 0);}
        tbl->lowest_valid_T = (int *)aneos_shared_calloc(nrho, sizeof(int), "aneos_lowest_valid_T");
        if(!tbl->lowest_valid_T) {return aneos_refuse(fp, tbl, filename, "could not allocate the lowest-valid-row block", nrho);}
        for(int i = 0; i < nrho; i++) {
            int k;
            if(fscanf(fp, "%d", &k) != 1) {return aneos_refuse(fp, tbl, filename, "lowest-valid-row block unreadable", i);}
            if(k < 0 || k > nT - 2) {return aneos_refuse(fp, tbl, filename, "lowest valid row outside [0, nT-2]", i);}
            tbl->lowest_valid_T[i] = k;
        }
    }
    fclose(fp);
    for(int i = 0; i < nrho; i++) {tbl->logrho[i] = log10(tbl->logrho[i]);}
    for(int j = 0; j < nT; j++) {tbl->logT[j] = log10(tbl->logT[j]);}
    for(int i = 1; i < nrho; i++) {if(!(tbl->logrho[i] > tbl->logrho[i - 1])) {return aneos_refuse(NULL, tbl, filename, "density axis points too close to resolve in log10", i);}}
    for(int j = 1; j < nT; j++) {if(!(tbl->logT[j] > tbl->logT[j - 1])) {return aneos_refuse(NULL, tbl, filename, "temperature axis points too close to resolve in log10", j);}}

    /* Grid bounds and search maps */
    tbl->logrho_min = tbl->logrho[0];
    tbl->logrho_max = tbl->logrho[nrho - 1];
    tbl->logT_min   = tbl->logT[0];
    tbl->logT_max   = tbl->logT[nT - 1];
    aneos_build_axis_map(tbl->logrho, nrho, tbl->nmap_rho, tbl->map_rho, &tbl->inv_binwidth_rho);
    aneos_build_axis_map(tbl->logT, nT, tbl->nmap_T, tbl->map_T, &tbl->inv_binwidth_T);

    /* Where an energy can name several temperatures. At a density inside [i, i+1] each row's log10(u) mixes the two columns' values
       linearly, so a cell of that column falls only where a cell of one of the two columns falls, and its end values lie between the
       four values there. An energy reached at two temperatures (or touched at an extremum) is crossed by a falling cell; so the range of
       every falling cell's end values (a fall larger than the solve's tolerance) bounds the energies a solve must check the column for. */
    tbl->fold_logu = (double *)aneos_shared_calloc(2 * (size_t)nrho, sizeof(double), "aneos_fold_logu");
    if(!tbl->fold_logu) {return aneos_refuse(NULL, tbl, filename, "could not allocate the fold ranges", nrho);}
    const double fall = log10(1.0 + ANEOS_INVERSION_TOLERANCE);
    for(int i = 0; i < nrho; i++) {
        double lo = 1e300, hi = -1e300;
        if(i < nrho - 1) {
            int k = tbl->lowest_valid_T ? tbl->lowest_valid_T[i] : 0, k1 = tbl->lowest_valid_T ? tbl->lowest_valid_T[i + 1] : 0;
            for(int j = (k1 > k) ? k1 : k; j < nT - 1; j++) {
                int falls = 0;
                for(int c = i; c <= i + 1; c++) {if(tbl->log_energy[c * nT + j + 1] < tbl->log_energy[c * nT + j] - fall) {falls = 1;}}
                if(!falls) continue;
                for(int c = i; c <= i + 1; c++) {for(int jj = j; jj <= j + 1; jj++) {double v = tbl->log_energy[c * nT + jj]; if(v < lo) {lo = v;} if(v > hi) {hi = v;}}}
            }
        }
        tbl->fold_logu[2 * i] = lo; tbl->fold_logu[2 * i + 1] = hi;
    }

    tbl->loaded = 1;

    if(mat_index >= ANEOS_Num_Tables_Loaded)
        ANEOS_Num_Tables_Loaded = mat_index + 1;

    return 0;
}


/* Bytes one loaded table holds (the same on every rank: each reads every table into memory the host and the device share). */
size_t aneos_table_bytes(int mat_index)
{
    if(!ANEOS_Tables || mat_index < 0 || mat_index >= ANEOS_MAX_MATERIALS || !ANEOS_Tables[mat_index].loaded) return 0;
    const struct aneos_table *tbl = &ANEOS_Tables[mat_index];
    size_t ntot = (size_t)tbl->nrho * (size_t)tbl->nT;
    size_t bytes = ((size_t)tbl->nrho + (size_t)tbl->nT) * sizeof(double) + ((size_t)tbl->nmap_rho + (size_t)tbl->nmap_T + 2) * sizeof(int)
                 + 3 * ntot * sizeof(double);
    if(tbl->has_csound) bytes += ntot * sizeof(double);
    if(tbl->has_phase) bytes += ntot * sizeof(int);
    if(tbl->lowest_valid_T) bytes += (size_t)tbl->nrho * sizeof(int);
    if(tbl->fold_logu) bytes += 2 * (size_t)tbl->nrho * sizeof(double);
    return bytes;
}

/* What a loaded table holds, for the startup log (begrun prints it once, on the first rank). */
void aneos_report_table(int mat_index, const char *filename)
{
    if(!ANEOS_Tables || mat_index < 0 || mat_index >= ANEOS_MAX_MATERIALS || !ANEOS_Tables[mat_index].loaded) return;
    const struct aneos_table *tbl = &ANEOS_Tables[mat_index];
    double nonuni_rho = aneos_axis_nonuniformity(tbl->logrho, tbl->nrho), nonuni_T = aneos_axis_nonuniformity(tbl->logT, tbl->nT);
    printf("ANEOS: table %d = material %d from \"%s\" (%d x %d grid, %d columns%s, %.1f MB)\n", mat_index, tbl->mat_id, filename, tbl->nrho, tbl->nT,
           5 + tbl->has_csound + tbl->has_phase, tbl->lowest_valid_T ? ", per-density lowest valid temperature" : "", aneos_table_bytes(mat_index) / 1048576.0);
    printf("  log10(rho) range: [%.3f, %.3f] %s,  log10(T) range: [%.3f, %.3f] %s\n",
           tbl->logrho_min, tbl->logrho_max, (nonuni_rho < 1.0e-6) ? "uniform" : "irregular",
           tbl->logT_min, tbl->logT_max, (nonuni_T < 1.0e-6) ? "uniform" : "irregular");
    int n_fold = 0; for(int i = 0; i < tbl->nrho - 1; i++) {n_fold += (tbl->fold_logu[2 * i] <= tbl->fold_logu[2 * i + 1]);}
    if(n_fold) {printf("  %d of %d density intervals have u falling with T above the valid floor: a solve whose energy lies in that range checks the whole column\n", n_fold, tbl->nrho - 1);}
}

int aneos_cleanup(void)
{
    if(!ANEOS_Tables) {return 0;}
    for(int k = 0; k < ANEOS_MAX_MATERIALS; k++) {
        struct aneos_table *tbl = &ANEOS_Tables[k];
        if(!tbl->loaded) continue;
        aneos_free_table_arrays(tbl);
    }
    ANEOS_Num_Tables_Loaded = 0;
    return 0;
}

#endif /* EOS_ANEOS */
