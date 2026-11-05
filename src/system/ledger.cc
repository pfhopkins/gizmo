/* Mass/metal/energy ledger (OUTPUT_MASS_METAL_ENERGY_LEDGER): modules add to their rows during their own work, per thread,
   into exact sums (integer limbs: the result does not depend on the order of the additions, so not on threads, schedule or
   rank count). Rows are folded and reduced only when a snapshot is written, one MPI_Reduce, and written to ledger.txt. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../declarations/allvars.h"
#include "../core/proto.h"

#ifdef OUTPUT_MASS_METAL_ENERGY_LEDGER

#define LEDGER_NLIMB 72          /* 32-bit limbs from 2^LEDGER_EXP0 up: every finite double fits */
#define LEDGER_EXP0  (-1152)
#define LEDGER_ROW_NAME(id, name, unit, state) name,
#define LEDGER_ROW_STATE(id, name, unit, state) state,
static const char *ledger_row_names[LEDGER_NROWS] = {LEDGER_ROWS(LEDGER_ROW_NAME)};
static const int ledger_row_is_state[LEDGER_NROWS] = {LEDGER_ROWS(LEDGER_ROW_STATE)};

struct ledger_sum {int64_t limb[LEDGER_NLIMB]; int64_t n_add;};
static struct ledger_sum *ledger_thread = NULL;      /* [thread][row]: since the last fold */
static struct ledger_sum ledger_total[LEDGER_NROWS]; /* this rank, cumulative: carried in the restart files */
static int ledger_nthreads = 0;

static void ledger_normalize(struct ledger_sum *s)  /* carries: limbs 0..N-2 in [0, 2^32), the sign in the top limb */
{
    for(int i = 0; i < LEDGER_NLIMB - 1; i++) {int64_t c = s->limb[i] >> 32; s->limb[i] -= c * ((int64_t) 1 << 32); s->limb[i+1] += c;}
    s->n_add = 0;
}

static void ledger_add_exact(struct ledger_sum *s, double x)
{
    if(x == 0 || !isfinite(x)) {return;}
    int e; double m = frexp(fabs(x), &e);
    int64_t M = (int64_t) ldexp(m, 53); /* |x| = M 2^(e-53), exactly */
    int pos = e - 53 - LEDGER_EXP0, k = pos / 32, sh = pos % 32, sg = (x < 0) ? -1 : 1;
    unsigned __int128 v = (unsigned __int128) M << sh;
    for(int j = 0; j < 3 && k + j < LEDGER_NLIMB; j++) {s->limb[k+j] += sg * (int64_t) ((v >> (32*j)) & 0xffffffffu);}
    if(++s->n_add >= ((int64_t) 1 << 30)) {ledger_normalize(s);} /* each limb gains < 2^32 per addition */
}

static void ledger_merge(struct ledger_sum *into, struct ledger_sum *from)
{
    ledger_normalize(into); ledger_normalize(from);
    for(int i = 0; i < LEDGER_NLIMB; i++) {into->limb[i] += from->limb[i]; from->limb[i] = 0;}
    ledger_normalize(into);
}

static double ledger_value(const struct ledger_sum *s) /* deterministic: a function of the (normalized) limbs only */
{
    struct ledger_sum c = *s; int neg = (c.limb[LEDGER_NLIMB-1] < 0); /* the top limb holds the sign: sum the magnitude, or -1 x 2^1120 overflows */
    if(neg) {for(int i = 0; i < LEDGER_NLIMB; i++) {c.limb[i] = -c.limb[i];} ledger_normalize(&c);}
    double v = 0;
    for(int i = LEDGER_NLIMB - 1; i >= 0; i--) {v += ldexp((double) c.limb[i], 32*i + LEDGER_EXP0);}
    return neg ? -v : v;
}

void ledger_init(void)
{
    if(ledger_thread) {return;}
#ifdef _OPENMP
    ledger_nthreads = omp_get_max_threads();
#else
    ledger_nthreads = 1;
#endif
    ledger_thread = (struct ledger_sum *) calloc((size_t) ledger_nthreads * LEDGER_NROWS, sizeof(struct ledger_sum));
    if(RestartFlag != 1) {memset(ledger_total, 0, sizeof(ledger_total));}
}

void ledger_add(int row, double x) /* from inside a module's (threaded) loop; no communication */
{
    int t = 0;
#ifdef _OPENMP
    t = omp_get_thread_num();
#endif
    if(!ledger_thread || t >= ledger_nthreads) {endrun(78201);}
    ledger_add_exact(&ledger_thread[(size_t) t * LEDGER_NROWS + row], x);
}

static void ledger_state_rows(void) /* the state rows: sums over this rank's particles now */
{
    for(int r = 0; r < LEDGER_NROWS; r++) {if(ledger_row_is_state[r]) {memset(&ledger_total[r], 0, sizeof(ledger_total[r]));}}
#if defined(SGCHEM) && defined(COSMIC_RAY_FLUID)
    for(int i = 0; i < N_gas; i++) {if(P[i].Type == 0) {for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {ledger_add_exact(&ledger_total[LEDGER_E_CR], CellP[i].CosmicRayEnergy[k]);}}}
#endif
#ifdef GALSF
    for(int i = 0; i < NumPart; i++) {
        if(P[i].Type == 0) {ledger_add_exact(&ledger_total[LEDGER_M_GAS], P[i].Mass); ledger_add_exact(&ledger_total[LEDGER_MZ_GAS], P[i].Mass * P[i].Metallicity[0]);}
        if(P[i].Type == 4) {ledger_add_exact(&ledger_total[LEDGER_M_STARS], P[i].Mass); ledger_add_exact(&ledger_total[LEDGER_MZ_STARS], P[i].Mass * P[i].Metallicity[0]);}
    }
#endif
}

void ledger_output(void) /* at every snapshot: fold the threads, one MPI_Reduce, rank 0 writes a row */
{
    if(!ledger_thread) {return;}
    for(int t = 0; t < ledger_nthreads; t++) {for(int r = 0; r < LEDGER_NROWS; r++) {ledger_merge(&ledger_total[r], &ledger_thread[(size_t) t * LEDGER_NROWS + r]);}}
    ledger_state_rows();
    static int64_t glob[LEDGER_NROWS][LEDGER_NLIMB];
    int64_t loc[LEDGER_NROWS][LEDGER_NLIMB];
    for(int r = 0; r < LEDGER_NROWS; r++) {memcpy(loc[r], ledger_total[r].limb, sizeof(loc[r]));}
    MPI_Reduce(loc, glob, LEDGER_NROWS * LEDGER_NLIMB, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask != 0) {return;}
#define LEDGER_ROW_UNIT(id, name, unit, state) (double) (unit),
    const double ledger_row_unit[LEDGER_NROWS] = {LEDGER_ROWS(LEDGER_ROW_UNIT)}; /* booked in code units */
    char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE];
    snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%sledger.txt", All.OutputDir);
    static FILE *fd = NULL;
    if(!fd) {
        if(!(fd = fopen(buf, (RestartFlag == 0) ? "w" : "a"))) {printf("ledger: cannot open '%s'\n", buf); endrun(78202);}
        fseek(fd, 0, SEEK_END);
        if(ftell(fd) == 0) {fprintf(fd, "# time"); for(int r = 0; r < LEDGER_NROWS; r++) {fprintf(fd, " %s", ledger_row_names[r]);} fprintf(fd, "  (cumulative; M_gas M_stars MZ_gas MZ_stars: at output)\n");} /* a new file, also on a start from a snapshot */
    }
    fprintf(fd, "%.17g", All.Time);
    for(int r = 0; r < LEDGER_NROWS; r++) {
        struct ledger_sum s; memcpy(s.limb, glob[r], sizeof(s.limb)); ledger_normalize(&s);
        fprintf(fd, " %.17g", ledger_value(&s) * ledger_row_unit[r]);
    }
    fprintf(fd, "\n"); fflush(fd);
}

void ledger_restart_io(void (*io)(void *, size_t, int), int modus) /* modus 0 write, 1 read; the layout is checked */
{
    int layout[2] = {LEDGER_NROWS, LEDGER_NLIMB}, stored[2] = {LEDGER_NROWS, LEDGER_NLIMB};
    if(modus == 0) {for(int t = 0; t < ledger_nthreads; t++) {for(int r = 0; r < LEDGER_NROWS; r++) {ledger_merge(&ledger_total[r], &ledger_thread[(size_t) t * LEDGER_NROWS + r]);}}}
    io(modus ? (void *) stored : (void *) layout, sizeof(layout), modus);
    if(stored[0] != LEDGER_NROWS || stored[1] != LEDGER_NLIMB) {printf("ledger: the restart file holds %d rows x %d limbs, this build %d x %d\n", stored[0], stored[1], LEDGER_NROWS, LEDGER_NLIMB); endrun(78203);}
    io(ledger_total, sizeof(ledger_total), modus);
}

#ifdef OUTPUT_MASS_BALANCE_STEP_CHECKPOINTS
/* mass_balance_steps.txt: at a checkpoint inside the step, baryon mass (gas + stars) and gas, star and ledger-corrected metal
   mass; a line only when one of M_bary, M_Z_gas, M_Z_ledger changed by more than 1e-14 since the previous checkpoint, so a leak
   is found between the two checkpoints that bracket it. M_Z_ledger = gas + star metals - (metals delivered to gas - metals taken
   from stars), constant when the books close. One MPI_Allreduce per call; code units. */
void ledger_step_checkpoint(const char *tag)
{
    static double last[3] = {-1., -1., -1.};
    static long long calls = 0;
    double loc[5] = {0, 0, 0, 0, 0}, glob[5]; /* M_gas, M_stars, MZ_gas, MZ_stars, net yield (cumulative rows of this rank) */
    for(int i = 0; i < NumPart; i++) {
        if(P[i].Mass <= 0) {continue;}
        if(P[i].Type == 0) {loc[0] += P[i].Mass; loc[2] += P[i].Mass * P[i].Metallicity[0];}
        else if(P[i].Type == 4) {loc[1] += P[i].Mass; loc[3] += P[i].Mass * P[i].Metallicity[0];}
    }
#if defined(GALSF_SINGLE_STARS_EVOLUTION_TABLES) && defined(GALSF_SINGLE_STARS_FROM_IMF) && defined(GALSF_FB_MECHANICAL)
    if(ledger_thread) {
        for(int t = 0; t < ledger_nthreads; t++) {for(int r = 0; r < LEDGER_NROWS; r++) {ledger_merge(&ledger_total[r], &ledger_thread[(size_t) t * LEDGER_NROWS + r]);}}
        loc[4] = ledger_value(&ledger_total[LEDGER_MZ_FB_TO_GAS]) - ledger_value(&ledger_total[LEDGER_MZ_STARS_LOST]);
    }
#endif
    MPI_Allreduce(loc, glob, 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(ThisTask != 0) {return;}
    double now[3] = {glob[0] + glob[1], glob[2], glob[2] + glob[3] - glob[4]}, d[3] = {0, 0, 0};
    int log_it = (calls == 0);
    for(int k = 0; k < 3; k++) {
        if(last[k] >= 0) {d[k] = now[k] - last[k]; if(last[k] > 0 && fabs(d[k]) / last[k] > 1.0e-14) {log_it = 1;}}
        last[k] = now[k];
    }
    if(log_it) {
        char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE];
        snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%smass_balance_steps.txt", All.OutputDir);
        FILE *f = fopen(buf, "a");
        if(f) {
            if(ftell(f) == 0) {fprintf(f, "# call tag time M_bary dM_bary M_Z_gas dM_Z_gas M_Z_stars M_Z_netyield_cumulative M_Z_ledger dM_Z_ledger  (code units; lines only where a change exceeds 1e-14)\n");}
            fprintf(f, "%lld %s %.10e %.16e %+.16e %.16e %+.16e %.16e %+.16e %.16e %+.16e\n", calls, tag, All.Time, now[0], d[0], now[1], d[1], glob[3], glob[4], now[2], d[2]);
            fclose(f);
        }
    }
    calls++;
}
#endif /* OUTPUT_MASS_BALANCE_STEP_CHECKPOINTS */

#endif /* OUTPUT_MASS_METAL_ENERGY_LEDGER */
