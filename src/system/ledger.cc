/* Mass/metal/energy ledger (OUTPUT_MASS_METAL_ENERGY_LEDGER): modules add to their rows during their own work, per thread,
   into exact sums (integer limbs: the result does not depend on the order of the additions, so not on threads, schedule or
   rank count). Rows are folded and reduced only when a snapshot is written, one MPI_Reduce, and written to ledger.txt. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <mpi.h>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/ghost_writeback.h"

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
#ifdef LEDGER_SLOTS
static struct ledger_sum *ledger_slot_thread = NULL;                     /* [thread][row][slot]: since the last fold */
static struct ledger_sum ledger_slot_total[LEDGER_SLOT_ROWS][NUM_METAL_SPECIES]; /* this rank, cumulative: carried in the restart files */
#endif

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
#ifdef LEDGER_SLOTS
    ledger_slot_thread = (struct ledger_sum *) calloc((size_t) ledger_nthreads * LEDGER_SLOT_ROWS * NUM_METAL_SPECIES, sizeof(struct ledger_sum));
    if(RestartFlag != 1) {memset(ledger_slot_total, 0, sizeof(ledger_slot_total));}
#endif
}

#ifdef LEDGER_SLOTS
void ledger_add_slot(int row, int k, double x) /* as ledger_add, for metal slot k */
{
    int t = 0;
#ifdef _OPENMP
    t = omp_get_thread_num();
#endif
    if(!ledger_slot_thread || t >= ledger_nthreads) {endrun(78204);}
    ledger_add_exact(&ledger_slot_thread[((size_t) t * LEDGER_SLOT_ROWS + row) * NUM_METAL_SPECIES + k], x);
}
static void ledger_slot_fold(void)
{
    for(int t = 0; t < ledger_nthreads; t++) {for(int r = 0; r < LEDGER_SLOT_ROWS; r++) {for(int k = 0; k < NUM_METAL_SPECIES; k++) {
        ledger_merge(&ledger_slot_total[r][k], &ledger_slot_thread[((size_t) t * LEDGER_SLOT_ROWS + r) * NUM_METAL_SPECIES + k]);}}}
}
#endif

void ledger_add(int row, double x) /* from inside a module's (threaded) loop; no communication */
{
    int t = 0;
#ifdef _OPENMP
    t = omp_get_thread_num();
#endif
    if(!ledger_thread || t >= ledger_nthreads) {endrun(78201);}
    ledger_add_exact(&ledger_thread[(size_t) t * LEDGER_NROWS + row], x);
#ifdef OUTPUT_ENERGY_STEP_TRACE
    estep_ledger_hook(row, x); /* the rows that are energy-trace bookings */
#endif
}

#ifdef COSMIC_RAY_FLUID
/* the kick of CR bin k of cell i, before the new energy eCR_new is stored: fluxes, streaming, and what the cap and the zero floor change */
void ledger_cr_kick(int i, int k, double dt, double eCR_new)
{
    double loss = 0, d = CellP[i].DtCosmicRayEnergy[k] * dt;
#if !defined(CRFLUID_EVOLVE_SCATTERINGWAVES) && !defined(CRFLUID_EVOLVE_SPECTRUM)
    loss = -CellP[i].DtCosmicRayEnergy_StreamingLoss[k] * dt;
#endif
    ledger_add(LEDGER_E_CR_STREAMING, loss); ledger_add(LEDGER_E_CR_TRANSPORT, d + loss);
    ledger_add(LEDGER_E_CR_CLIP, eCR_new - (CellP[i].CosmicRayEnergy[k] + d));
}
void ledger_cr_exchange(double dCR_div) {ledger_add(LEDGER_E_CR_EXCHANGE, dCR_div);}
void ledger_cr_shock(double dE) {ledger_add(LEDGER_E_CR_SHOCK, dE);}
#endif
#ifdef SGCHEM
void ledger_kick_floor(double dE) {ledger_add(LEDGER_E_KICK_FLOOR, dE);}
#endif

#ifdef LEDGER_OUTFLOW_BOUNDARY
/* a gas cell or star that left through an outflow face, once: where the dead particle is removed, or before the ledger is read */
void ledger_book_outflow(int i)
{
    const double m = P[i].OutflowMass; P[i].OutflowMass = 0;
    if(!(m > 0) || (P[i].Type != 0 && P[i].Type != 4)) {return;}
    ledger_add(LEDGER_M_OUTFLOW, m); ledger_add(LEDGER_MZ_OUTFLOW, m * P[i].Metallicity[0]);
#ifdef LEDGER_SLOTS
    for(int k = 0; k < NUM_METAL_SPECIES; k++) {ledger_add_slot(LEDGER_SLOT_OUTFLOW, k, m * P[i].Metallicity[k]);}
#endif
}
static void ledger_book_outflow_pending(void) /* particles not yet removed; imported ghost copies sit above the local ones */
{
    for(int i = 0, n_local = ghost_get_num_local(); i < n_local; i++) {if(P[i].Mass <= 0) {ledger_book_outflow(i);}}
}
#endif

static void ledger_state_rows(void) /* the state rows: sums over this rank's particles now */
{
    for(int r = 0; r < LEDGER_NROWS; r++) {if(ledger_row_is_state[r]) {memset(&ledger_total[r], 0, sizeof(ledger_total[r]));}}
#ifdef COSMIC_RAY_FLUID
    for(int i = 0; i < N_gas; i++) {if(P[i].Type == 0) {for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {ledger_add_exact(&ledger_total[LEDGER_E_CR], CellP[i].CosmicRayEnergy[k]);}}}
#endif
#ifdef TEST_MHD_ENERGY_LEDGER_OUTPUT
    for(int i = 0; i < N_gas; i++) {
        if(P[i].Type != 0 || P[i].Mass <= 0 || CellP[i].Density <= 0) {continue;}
        ledger_add_exact(&ledger_total[LEDGER_E_TH], P[i].Mass * CellP[i].InternalEnergy);
        ledger_add_exact(&ledger_total[LEDGER_E_KIN], 0.5 * P[i].Mass * P[i].Vel.norm_sq() * All.cf_a2inv);
        double eb = 0.5 * CellP[i].B.norm_sq() * CellP[i].Density / (P[i].Mass * All.cf_atime); /* kicked B, as u and v */
        ledger_add_exact(&ledger_total[LEDGER_E_B], eb);
        if(CellP[i].LedgerEB >= 0) {ledger_add_exact(&ledger_total[LEDGER_E_B_VOLUME_CHANGE_PENDING], eb - CellP[i].LedgerEB);} /* drifted volume since the last B kick */
#ifdef DIVBCLEANING_CONSERVATIVE_PSI_OVER_CH
        ledger_add_exact(&ledger_total[LEDGER_E_PSI], 0.5 * CellP[i].PsiZ * CellP[i].PsiZ / All.cf_atime);
#endif
    }
#endif
#ifdef GALSF_ISMDUSTCHEM_HU_MODEL /* carbonaceous dust: element 2; silicate: all other dust (O, Mg, Si, Fe) */
    for(int i = 0; i < N_gas; i++) {
        if(P[i].Type != 0 || P[i].Mass <= 0) {continue;}
        ledger_add_exact(&ledger_total[LEDGER_M_DUST_C], P[i].Mass * CellP[i].ISMDustChem_Dust_Metal[2]);
        ledger_add_exact(&ledger_total[LEDGER_M_DUST_SIL], P[i].Mass * (CellP[i].ISMDustChem_Dust_Metal[0] - CellP[i].ISMDustChem_Dust_Metal[2]));
    }
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
#ifdef LEDGER_OUTFLOW_BOUNDARY
    ledger_book_outflow_pending();
#endif
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
#ifdef LEDGER_SLOTS
    int slot_layout[2] = {LEDGER_SLOT_ROWS, NUM_METAL_SPECIES}, slot_stored[2] = {LEDGER_SLOT_ROWS, NUM_METAL_SPECIES};
    if(modus == 0) {ledger_slot_fold();}
    io(modus ? (void *) slot_stored : (void *) slot_layout, sizeof(slot_layout), modus);
    if(slot_stored[0] != LEDGER_SLOT_ROWS || slot_stored[1] != NUM_METAL_SPECIES) {printf("ledger: the restart file holds %d x %d slot rows, this build %d x %d\n", slot_stored[0], slot_stored[1], LEDGER_SLOT_ROWS, NUM_METAL_SPECIES); endrun(78205);}
    io(ledger_slot_total, sizeof(ledger_slot_total), modus);
#endif
}

#ifdef OUTPUT_MASS_BALANCE_STEP_CHECKPOINTS
/* mass_balance_steps.txt: at a checkpoint inside the step, baryon mass (gas + stars) and gas, star and ledger-corrected metal
   mass; a line only when one of M_bary, M_Z_gas, M_Z_ledger changed by more than 1e-14 since the previous checkpoint, so a leak
   is found between the two checkpoints that bracket it. M_Z_ledger = gas + star metals + diffusion exchange not yet applied -
   (metals delivered to gas - metals taken from stars), constant when the books close. The sums are long double (their terms
   cancel to a much smaller total, which a double sum resolves only to the spacing of doubles near the terms). With LEDGER_SLOTS the
   same closure per metal slot, in the same reduction, to mass_balance_elements.txt (stella_dev 8227ec120, 9335b5ced). One
   MPI_Allreduce per call; code units. */
static long long ledger_step_ncalls_all = 0; /* checkpoints so far, the same on all tasks (each is collective) */
long long ledger_step_ncalls(void) {return ledger_step_ncalls_all;}
void ledger_step_checkpoint(const char *tag)
{
    ledger_step_ncalls_all++;
    static long double last[3] = {-1., -1., -1.};
    static long long calls = 0;
    enum {N_SCALAR = 8}; /* M_gas, M_stars, MZ_gas, MZ_stars, net yield (cumulative rows of this rank), diffusion exchange in flight, mass and metals gone through an outflow face */
#ifdef LEDGER_SLOTS
    enum {EL_GAS, EL_STARS, EL_NET, EL_OUT, EL_N}; /* per slot: gas, stars, booked net yield, gone through an outflow face */
    const int n_red = N_SCALAR + EL_N * NUM_METAL_SPECIES;
#else
    const int n_red = N_SCALAR;
#endif
    std::vector<long double> loc(n_red, 0.0L), glob(n_red, 0.0L);
    for(int i = 0, n_local = ghost_get_num_local(); i < n_local; i++) { /* imported ghost copies sit above the local cells */
        if(P[i].Mass <= 0) {continue;}
        if(P[i].Type == 0) {loc[0] += P[i].Mass; loc[2] += (long double) P[i].Mass * P[i].Metallicity[0];}
        else if(P[i].Type == 4) {loc[1] += P[i].Mass; loc[3] += (long double) P[i].Mass * P[i].Metallicity[0];}
#ifdef LEDGER_SLOTS
        if(P[i].Type == 0 || P[i].Type == 4) {
            long double *el = &loc[N_SCALAR + ((P[i].Type == 0) ? EL_GAS : EL_STARS) * NUM_METAL_SPECIES];
            for(int k = 0; k < NUM_METAL_SPECIES; k++) {el[k] += (long double) P[i].Mass * P[i].Metallicity[k];}
        }
#endif
    }
#ifdef LEDGER_OUTFLOW_BOUNDARY
    if(ledger_thread) {
        ledger_book_outflow_pending();
        for(int t = 0; t < ledger_nthreads; t++) {for(int r = LEDGER_M_OUTFLOW; r <= LEDGER_MZ_OUTFLOW; r++) {ledger_merge(&ledger_total[r], &ledger_thread[(size_t) t * LEDGER_NROWS + r]);}}
        loc[6] = (long double) ledger_value(&ledger_total[LEDGER_M_OUTFLOW]); loc[7] = (long double) ledger_value(&ledger_total[LEDGER_MZ_OUTFLOW]);
    }
#endif
#if defined(GALSF_SINGLE_STARS_EVOLUTION_TABLES) && defined(GALSF_SINGLE_STARS_FROM_IMF) && defined(GALSF_FB_MECHANICAL)
    if(ledger_thread) {
        for(int t = 0; t < ledger_nthreads; t++) {for(int r = 0; r < LEDGER_NROWS; r++) {ledger_merge(&ledger_total[r], &ledger_thread[(size_t) t * LEDGER_NROWS + r]);}}
        loc[4] = (long double) ledger_value(&ledger_total[LEDGER_MZ_FB_TO_GAS]) - (long double) ledger_value(&ledger_total[LEDGER_MZ_STARS_LOST]);
    }
#endif
#ifdef LEDGER_SLOTS
    if(ledger_slot_thread) {
        ledger_slot_fold();
        for(int k = 0; k < NUM_METAL_SPECIES; k++) {loc[N_SCALAR + EL_NET * NUM_METAL_SPECIES + k] = (long double) ledger_value(&ledger_slot_total[LEDGER_SLOT_TO_GAS][k]) - (long double) ledger_value(&ledger_slot_total[LEDGER_SLOT_FROM_STARS][k]);
            loc[N_SCALAR + EL_OUT * NUM_METAL_SPECIES + k] = (long double) ledger_value(&ledger_slot_total[LEDGER_SLOT_OUTFLOW][k]);}
    }
#endif
    MPI_Allreduce(loc.data(), glob.data(), n_red, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(ThisTask != 0) {calls++; return;}
    long double now[3] = {glob[0] + glob[1] + glob[6], glob[2], glob[2] + glob[3] + glob[5] + glob[7] - glob[4]}, d[3] = {0, 0, 0};
    int log_it = (calls == 0);
    for(int k = 0; k < 3; k++) {
        if(last[k] >= 0) {d[k] = now[k] - last[k]; if(last[k] > 0 && fabsl(d[k]) / last[k] > 1.0e-14) {log_it = 1;}}
        last[k] = now[k];
    }
    if(log_it) {
        char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE];
        snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%smass_balance_steps.txt", All.OutputDir);
        FILE *f = fopen(buf, "a");
        if(f) {
            if(ftell(f) == 0) {fprintf(f, "# call tag time M_bary dM_bary M_Z_gas dM_Z_gas M_Z_stars M_Z_netyield_cumulative M_Z_ledger dM_Z_ledger M_Z_inflight (code units; lines only where a change exceeds 1e-14)\n");}
            fprintf(f, "%lld %s %.10e %.16e %+.16e %.16e %+.16e %.16e %+.16e %.16e %+.16e %+.16e\n", calls, tag, All.Time, (double) now[0], (double) d[0], (double) now[1], (double) d[1], (double) glob[3], (double) glob[4], (double) now[2], (double) d[2], (double) glob[5]);
            fclose(f);
        }
    }
#ifdef LEDGER_SLOTS
    { /* per slot L = gas + stars - net yield, constant when the books close; dL relative to S = |gas| + |stars| of that slot */
        static std::vector<long double> last_L;
        const long double *el = &glob[N_SCALAR];
        std::vector<long double> L(NUM_METAL_SPECIES);
        std::vector<double> rel(NUM_METAL_SPECIES, 0.0);
        int worst = -1;
        for(int k = 0; k < NUM_METAL_SPECIES; k++) {
            long double g = el[EL_GAS * NUM_METAL_SPECIES + k], st = el[EL_STARS * NUM_METAL_SPECIES + k], out = el[EL_OUT * NUM_METAL_SPECIES + k], S = fabsl(g) + fabsl(st) + fabsl(out);
#ifdef GALSF_SINGLE_STARS_METAL_ORIGIN_TRACERS
            if(k >= SINGLE_STAR_METAL_ORIGIN_OFFSET && k < SINGLE_STAR_METAL_ORIGIN_OFFSET + 4) {continue;} /* labels, not conserved on merges */
#endif
            L[k] = g + st + out - el[EL_NET * NUM_METAL_SPECIES + k];
            if(!last_L.empty()) {
                long double dL = L[k] - last_L[k];
                rel[k] = (S > 0) ? (double) (dL / S) : ((dL != 0) ? HUGE_VAL : 0.);
                if(worst < 0 || fabs(rel[k]) > fabs(rel[worst])) {worst = k;}
            }
        }
        if(last_L.empty() || (worst >= 0 && fabs(rel[worst]) > 1.0e-14)) {
            char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE];
            snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%smass_balance_elements.txt", All.OutputDir);
            FILE *f = fopen(buf, "a");
            if(f) {
                if(ftell(f) == 0) {
                    fprintf(f, "# per metal slot L = gas + stars + gone through an outflow face - (deposited into gas by feedback - taken from stars), constant when the books close; a line at the first call and\n");
                    fprintf(f, "# where some slot's |dL|/S > 1e-14 since the previous checkpoint (S = |gas| + |stars| + |outflow| of the slot). Slot 0: all metals; 1..10 He C N O Ne Mg Si S Ca Fe in the FIRE layout.\n");
                    fprintf(f, "# call tag time worst_slot its_dL/S dL/S per slot:"); for(int k = 0; k < NUM_METAL_SPECIES; k++) {fprintf(f, " s%d", k);} fprintf(f, "\n");
                }
                fprintf(f, "%lld %s %.10e %d %+.6e", calls, tag, All.Time, worst, (worst >= 0) ? rel[worst] : 0.);
                for(int k = 0; k < NUM_METAL_SPECIES; k++) {fprintf(f, " %+.3e", rel[k]);}
                fprintf(f, "\n"); fclose(f);
            }
        }
        last_L = L;
    }
#endif
    calls++;
}

#ifdef OUTPUT_MASS_BALANCE_BREAKDOWN
/* diff_unpack_breakdown.txt, per hydro step: sum over the cells of the diffusive exchange applied (0 for a conservative solver), of
   what the floor at zero added on top (0 unless the floor fired) and of the absolute exchange. One MPI_Allreduce per call; code units. */
void ledger_diffusion_breakdown(double dMZ_pair, double dMZ_clamp, double dMZ_gross)
{
    double loc[3] = {dMZ_pair, dMZ_clamp, dMZ_gross}, glob[3];
    MPI_Allreduce(loc, glob, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(ThisTask != 0) {return;}
    char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE];
    snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%sdiff_unpack_breakdown.txt", All.OutputDir);
    FILE *f = fopen(buf, "a");
    if(f) {
        if(ftell(f) == 0) {fprintf(f, "# time dM_Z_pair_flux dM_Z_clamp dM_Z_gross (code units)\n");}
        fprintf(f, "%.10e %+.16e %+.16e %.16e\n", All.Time, glob[0], glob[1], glob[2]);
        fclose(f);
    }
}

/* MCBAL lines on stdout at each checkpoint: masses in Msun, gas mass in every metal slot, range and count of cells whose
   sum(Met[1..]) leaves one by more than 1e-6 (only the full table layout closes to one). Four reductions per call. */
void ledger_element_balance(const char *tag)
{
    double loc[2 + NUM_METAL_SPECIES] = {0}, glob[2 + NUM_METAL_SPECIES], smin = 1e30, smax = -1e30, gmin, gmax; long nviol = 0, gviol;
    for(int i = 0, n_local = ghost_get_num_local(); i < n_local; i++) { /* imported ghost copies sit above the local cells */
        if(P[i].Mass <= 0) {continue;}
        if(P[i].Type == 4) {loc[1] += P[i].Mass;}
        if(P[i].Type != 0) {continue;}
        loc[0] += P[i].Mass; double sum = 0;
        for(int k = 0; k < NUM_METAL_SPECIES; k++) {loc[2 + k] += P[i].Mass * P[i].Metallicity[k]; if(k > 0) {sum += P[i].Metallicity[k];}}
        smin = DMIN(smin, sum); smax = DMAX(smax, sum); if(fabs(sum - 1.) > 1.e-6) {nviol++;}
    }
    MPI_Reduce(loc, glob, 2 + NUM_METAL_SPECIES, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&smin, &gmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD); MPI_Reduce(&smax, &gmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nviol, &gviol, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask != 0) {return;}
    printf("MCBAL %s t=%.6e M_tot=%.10e M_gas=%.10e M_star=%.10e sumMet: min=%.9e max=%.9e #viol=%ld\n", tag, All.Time, (glob[0] + glob[1]) * UNIT_MASS_IN_SOLAR, glob[0] * UNIT_MASS_IN_SOLAR, glob[1] * UNIT_MASS_IN_SOLAR, gmin, gmax, gviol);
    printf("MCBAL %s per-element:", tag); for(int k = 0; k < NUM_METAL_SPECIES; k++) {printf(" %.6e", glob[2 + k] * UNIT_MASS_IN_SOLAR);} printf("\n"); fflush(stdout);
}
#endif
#endif /* OUTPUT_MASS_BALANCE_STEP_CHECKPOINTS */

#endif /* OUTPUT_MASS_METAL_ENERGY_LEDGER */
