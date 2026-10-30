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
static const char *ledger_row_names[LEDGER_NROWS] = {"E_chem_heating_erg", "E_chem_cooling_erg"};

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
    double v = 0;
    for(int i = LEDGER_NLIMB - 1; i >= 0; i--) {v += ldexp((double) s->limb[i], 32*i + LEDGER_EXP0);}
    return v;
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

void ledger_output(void) /* at every snapshot: fold the threads, one MPI_Reduce, rank 0 writes a row */
{
    if(!ledger_thread) {return;}
    for(int t = 0; t < ledger_nthreads; t++) {for(int r = 0; r < LEDGER_NROWS; r++) {ledger_merge(&ledger_total[r], &ledger_thread[(size_t) t * LEDGER_NROWS + r]);}}
    static int64_t glob[LEDGER_NROWS][LEDGER_NLIMB];
    int64_t loc[LEDGER_NROWS][LEDGER_NLIMB];
    for(int r = 0; r < LEDGER_NROWS; r++) {memcpy(loc[r], ledger_total[r].limb, sizeof(loc[r]));}
    MPI_Reduce(loc, glob, LEDGER_NROWS * LEDGER_NLIMB, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask != 0) {return;}
    char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE];
    snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%sledger.txt", All.OutputDir);
    static FILE *fd = NULL;
    if(!fd) {
        int fresh = (RestartFlag == 0);
        if(!(fd = fopen(buf, fresh ? "w" : "a"))) {printf("ledger: cannot open '%s'\n", buf); endrun(78202);}
        if(fresh) {fprintf(fd, "# time"); for(int r = 0; r < LEDGER_NROWS; r++) {fprintf(fd, " %s", ledger_row_names[r]);} fprintf(fd, "  (cumulative)\n");}
    }
    fprintf(fd, "%.17g", All.Time);
    for(int r = 0; r < LEDGER_NROWS; r++) {
        struct ledger_sum s; memcpy(s.limb, glob[r], sizeof(s.limb)); ledger_normalize(&s);
        fprintf(fd, " %.17g", ledger_value(&s) * UNIT_ENERGY_IN_CGS /* energy rows, booked in code units */);
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

#endif /* OUTPUT_MASS_METAL_ENERGY_LEDGER */
