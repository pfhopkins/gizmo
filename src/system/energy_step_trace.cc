/* Energy step trace (OUTPUT_ENERGY_STEP_TRACE; stella_dev core/energy_step_trace.cc with the fix/trace-imf-2026-11-17 changes):
 * per-stage energy closure, the energy twin of the mass trace (OUTPUT_MASS_BALANCE_STEP_CHECKPOINTS). Writes
 * <OutputDir>/energy_step_trace.txt, checked by gizmo.conservation_trace.
 *
 * E = kinetic (all types) + thermal (gas) + magnetic + CR (+ raw Rad_E_gamma) [+ potential at full syncs].
 * Stage brackets (ESTEP_BEGIN/END) are rank-local O(N) sums, no communication; every booked source or sink is added on the rank
 * where it happens (ESTEP_BOOK, or a ledger row through estep_ledger_hook). One MPI_Reduce at the end of every Nth step
 * (N = flag value) and of every full-sync step closes the window:
 *   STAGE line: an exact stage whose measured dE minus its booked terms exceeds the round-off bar.
 *   SYNC  line: at full syncs, E_tot incl. potential, cumulative stage changes, bookings and the integration residual
 *               dE_tot - sum(stage dE) - integration bookings. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"

#ifdef OUTPUT_ENERGY_STEP_TRACE

#define ESTEP_EVERY ((OUTPUT_ENERGY_STEP_TRACE + 0) > 0 ? (OUTPUT_ENERGY_STEP_TRACE + 0) : 1) /* flag value N: close a window every N steps */
#define ESTEP_BAR_EXACT 1.0e-12 /* exact stages: relative to E_abs + |booked| */

enum {EC_KIN, EC_TH, EC_MAG, EC_CR, EC_RAD, EC_N};
static const char *estep_comp_name[EC_N] = {"kin", "th", "mag", "cr", "rad"};

static const char *estep_stage_name[] = {"merge_split", "fb_mech", "fb_radp", "photoion", "cooling", "sf",
    "int_grav", "int_hydro", "int_kick1", "int_kick2", "int_drift", "int_nbody"};
/* 1 = exact (booked terms must close it to round-off), 0 = measured, no energy booking yet (a source in tot_res),
 * 2 = part of the time integration, measured for attribution only (contains nested stages; in no closure) */
static const int estep_stage_exact[] = {0, 1, 0, 0,
#ifdef SGCHEM
    1,
#else
    0,
#endif
    0, 2, 2, 2, 2, 2, 2};

static const char *estep_book_name[] = {"fb_sn", "fb_wind", "fb_remnant_kin", "cool_solve", "cool_pi", "cool_floor", "cool_cr_loss",
    "kick_floor", "cr_clip", "info_mhd_num", "info_cr_shock"};
/* stage each booking belongs to; -1 = time integration (kicks/drift), -2 = information only (in no closure) */
static const int estep_book_stage[] = {ESTEP_FB_MECH, ESTEP_FB_MECH, ESTEP_FB_MECH, ESTEP_COOLING, ESTEP_COOLING, ESTEP_COOLING,
    ESTEP_COOLING, -1, -1, -2, -2};

static_assert(sizeof(estep_stage_name) / sizeof(estep_stage_name[0]) == ESTEP_NSTAGE, "stage names");
static_assert(sizeof(estep_stage_exact) / sizeof(estep_stage_exact[0]) == ESTEP_NSTAGE, "stage kinds");
static_assert(sizeof(estep_book_name) / sizeof(estep_book_name[0]) == EBK_N, "booking names");
static_assert(sizeof(estep_book_stage) / sizeof(estep_book_stage[0]) == EBK_N, "booking stages");

double EstepBooked[EBK_N];                    /* rank-local bookings in this window [code energy] */
static double estep_pre[ESTEP_NSTAGE][EC_N];  /* rank-local E at the open bracket */
static double estep_dE[ESTEP_NSTAGE][EC_N];   /* rank-local measured stage change in this window */
static int estep_open[ESTEP_NSTAGE];
static double estep_wall; /* seconds this rank spent in the trace */

/* magnetic energy of a gas cell from the conserved B*V: 1/2 |B|^2 V, V = m/rho (as energy_statistics) */
double estep_cell_mag_energy(int i)
{
#ifdef MAGNETIC
    if(P[i].Type != 0 || P[i].Mass <= 0 || CellP[i].Density <= 0) {return 0;}
    Vec3<double> b = CellP[i].B * (CellP[i].Density / P[i].Mass) * All.cf_a2inv;
    return 0.5 * b.norm_sq() * P[i].Mass / (CellP[i].Density * All.cf_a3inv);
#else
    (void)i; return 0;
#endif
}

/* ledger rows that are trace bookings: the row's own sign convention turned into the energy the stage adds */
void estep_ledger_hook(int row, double x)
{
    switch(row) {
#ifdef SGCHEM
        case LEDGER_CHEM_HEATING: ESTEP_BOOK(EBK_COOL_SOLVE, x); break;   /* the solve only, |dE| */
        case LEDGER_CHEM_COOLING: ESTEP_BOOK(EBK_COOL_SOLVE, -x); break;
        case LEDGER_E_KICK_FLOOR: ESTEP_BOOK(EBK_KICK_FLOOR, x); break;   /* what the kick limiter and floor add */
#ifdef MAGNETIC
        case LEDGER_E_MHD_NUMERICAL: ESTEP_BOOK(EBK_INFO_MHD_NUM, x); break;
#endif
#endif
#if defined(GALSF_FB_PHOTOIONIZATION_PER_DIRECTION) && defined(SGCHEM)
        case LEDGER_PI_HEATING: ESTEP_BOOK(EBK_COOL_PI, x); break;       /* the raise to 1e4 K in the chemistry call */
#endif
#if defined(SGCHEM) && defined(COSMIC_RAY_FLUID)
        case LEDGER_E_CR_LOSSES: ESTEP_BOOK(EBK_COOL_CR_LOSS, -x); break; /* CR energy the losses remove */
#endif
#ifdef COSMIC_RAY_FLUID
        case LEDGER_E_CR_CLIP: ESTEP_BOOK(EBK_CR_CLIP, x); break;
        case LEDGER_E_CR_SHOCK: ESTEP_BOOK(EBK_INFO_CR_SHOCK, x); break; /* thermal -> CR at shocks: a transfer */
#endif
        default: break;
    }
}

/* Neumaier-compensated local sums of the components (and of 1/2 m Phi when pot != NULL) */
static void estep_local_sums(double e[EC_N], double *pot)
{
    double s[EC_N + 1], c[EC_N + 1];
    for(int k = 0; k <= EC_N; k++) {s[k] = c[k] = 0;}
    auto add = [&](int k, double x) {double t = s[k] + x; c[k] += (fabs(s[k]) >= fabs(x)) ? (s[k] - t) + x : (x - t) + s[k]; s[k] = t;};
    for(int i = 0; i < NumPart; i++) {
        if(P[i].Mass <= 0) {continue;}
        add(EC_KIN, 0.5 * P[i].Mass * P[i].Vel.norm_sq() * All.cf_a2inv);
#ifdef EVALPOTENTIAL
        if(pot) {add(EC_N, 0.5 * P[i].Mass * P[i].Potential);}
#endif
        if(P[i].Type != 0) {continue;}
        add(EC_TH, P[i].Mass * CellP[i].InternalEnergy);
        add(EC_MAG, estep_cell_mag_energy(i));
#ifdef COSMIC_RAY_FLUID
        for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {add(EC_CR, CellP[i].CosmicRayEnergy[k]);}
#endif
#ifdef RADTRANSFER
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {add(EC_RAD, CellP[i].Rad_E_gamma[k]);}
#endif
    }
    for(int k = 0; k < EC_N; k++) {e[k] = s[k] + c[k];}
    if(pot) {*pot = s[EC_N] + c[EC_N];}
}

void estep_begin(int stage)
{
    double t0 = MPI_Wtime();
    estep_local_sums(estep_pre[stage], NULL);
    estep_open[stage] = 1;
    estep_wall += MPI_Wtime() - t0;
}

void estep_end(int stage)
{
    if(!estep_open[stage]) {return;}
    double t0 = MPI_Wtime(), e[EC_N];
    estep_local_sums(e, NULL);
    for(int k = 0; k < EC_N; k++) {estep_dE[stage][k] += e[k] - estep_pre[stage][k];}
    estep_open[stage] = 0;
    estep_wall += MPI_Wtime() - t0;
}

void energy_step_trace_report(void)
{
    static int first = 1;
    static long long n_calls = 0;
    int is_sync = (All.HighestActiveTimeBin == All.HighestOccupiedTimeBin);
    n_calls++;
    if(!first && !is_sync && (n_calls % ESTEP_EVERY) != 0) {return;} /* rank-uniform: step count and timebins are global */

    double t0 = MPI_Wtime();
    const int NS = ESTEP_NSTAGE * EC_N, NV = NS + EBK_N + EC_N + 2;
    double loc[NV + 2], glob[NV + 2];
    for(int s = 0; s < ESTEP_NSTAGE; s++) {for(int k = 0; k < EC_N; k++) {loc[s * EC_N + k] = estep_dE[s][k]; estep_dE[s][k] = 0;}}
    for(int b = 0; b < EBK_N; b++) {loc[NS + b] = EstepBooked[b]; EstepBooked[b] = 0;}
    double e[EC_N], pot = 0;
    estep_local_sums(e, &pot);
    double eabs = 0;
    for(int k = 0; k < EC_N; k++) {loc[NS + EBK_N + k] = e[k]; eabs += fabs(e[k]);}
    loc[NV - 2] = eabs; loc[NV - 1] = pot;
    loc[NV + 1] = 0; /* Layzer-Irvine weights: sum 3(gamma-1) m u + 3(gamma_CR-1) E_CR, the expansion loss rate over H */
    if(All.ComovingIntegrationOn && is_sync) {
        for(int i = 0; i < NumPart; i++) {
            if(P[i].Type != 0 || P[i].Mass <= 0) {continue;}
            loc[NV + 1] += 3. * (CellP[i].gamma_eos_value() - 1.) * P[i].Mass * CellP[i].InternalEnergy;
#ifdef COSMIC_RAY_FLUID
            for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {loc[NV + 1] += 3. * (GAMMA_COSMICRAY(k) - 1.) * CellP[i].CosmicRayEnergy[k];}
#endif
        }
    }
    loc[NV] = estep_wall + (MPI_Wtime() - t0); /* rank sum of trace time up to this reduce */
    MPI_Reduce(loc, glob, NV + 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    estep_wall += MPI_Wtime() - t0;
    if(ThisTask != 0) {first = 0; return;}

    int pot_ok = 0;
#if defined(EVALPOTENTIAL) && !defined(ADAPTIVE_TREEFORCE_UPDATE) /* skipped tree walks keep an old Potential */
    pot_ok = 1;
#endif
    static FILE *fd = NULL;
    static double cum_stage[ESTEP_NSTAGE], cum_book[EBK_N], E0 = 0, E0_abs = 0;
    static int have_E0 = 0;
    if(!fd) {
        char path[DEFAULT_PATH_BUFFERSIZE_TOUSE];
        snprintf(path, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s/energy_step_trace.txt", All.OutputDir);
        fd = fopen(path, "a");
        if(!fd) {endrun(88731);}
        if(ftell(fd) == 0) {
            fprintf(fd, "# Energy step trace [code energy units; erg = x %.10e]. key=value lines.\n", UNIT_ENERGY_IN_CGS);
            fprintf(fd, "# E = kin(all types, peculiar) + th + mag + cr + rad(raw Rad_E_gamma); pot = 1/2 sum m Phi / a at full syncs.\n");
            fprintf(fd, "# Comoving: LI = int (2 E_kin + E_pot + sum 3(gamma-1) m u + E_mag + 3(gamma_CR-1) E_cr + E_rad) dln a (Layzer-Irvine), trapezoid over syncs; 0 if not comoving.\n");
            fprintf(fd, "# STAGE: exact stage whose window residual = dE(kin+th+cr+rad) - booked exceeds bar = %.1e x (E_abs + |booked|).\n", ESTEP_BAR_EXACT);
            fprintf(fd, "#        d_mag is shown but not closed per stage: tree walks drift Density, and E_B = 1/2|BV|^2 m/rho moves with it.\n");
            fprintf(fd, "# SYNC: full sync. int_res = (E_tot - E_tot0) + LI - sum(stage dE) - integration bookings (kick_floor, cr_clip).\n");
            fprintf(fd, "#       b_info_*: information only, in no closure. tot_res = (E_tot - E_tot0) - all bookings - measured stages.\n");
            fprintf(fd, "#       E_scale: |E| + |E_pot| at the first sync, fixed for the run (the checker adds the booked SN energy from the SN on).\n");
            fprintf(fd, "# cooling stage: the solve (ledger), PI raise, input and output floors, CR losses booked from their formulas at the site;\n");
            fprintf(fd, "#        unbooked, so a STAGE finding: the NaN revert, the hydro du around the solve without COOLING_OPERATOR_SPLIT.\n");
            fprintf(fd, "# exact stages:");
            for(int s = 0; s < ESTEP_NSTAGE; s++) {if(estep_stage_exact[s] == 1) {fprintf(fd, " %s", estep_stage_name[s]);}}
            fprintf(fd, "\n# pot_valid=%d every=%d\n", pot_ok, ESTEP_EVERY);
        }
        if(RestartFlag == 1) {fprintf(fd, "RESTART t=%.16e\n", All.Time);}
    }

    double Eabs = glob[NV - 2];
    for(int s = 0; s < ESTEP_NSTAGE; s++) {
        double dE = 0, dE_nomag = 0, book = 0, babs = 0;
        for(int k = 0; k < EC_N; k++) {dE += glob[s * EC_N + k]; if(k != EC_MAG) {dE_nomag += glob[s * EC_N + k];}}
        for(int b = 0; b < EBK_N; b++) {if(estep_book_stage[b] == s) {book += glob[NS + b]; babs += fabs(glob[NS + b]);}}
        double res = dE_nomag - book, bar = ESTEP_BAR_EXACT * (Eabs + babs);
        if(estep_stage_exact[s] == 1 && fabs(res) > bar) {
            fprintf(fd, "STAGE step=%lld t=%.16e stage=%s res=%+.16e bar=%.6e booked=%+.16e", (long long)All.NumCurrentTiStep, All.Time, estep_stage_name[s], res, bar, book);
            for(int k = 0; k < EC_N; k++) {fprintf(fd, " d_%s=%+.16e", estep_comp_name[k], glob[s * EC_N + k]);}
            for(int b = 0; b < EBK_N; b++) {if(estep_book_stage[b] == s && glob[NS + b] != 0) {fprintf(fd, " b_%s=%+.16e", estep_book_name[b], glob[NS + b]);}}
            fprintf(fd, " E_abs=%.16e\n", Eabs);
        }
        if(!first) {cum_stage[s] += dE;}
    }
    double Etot = 0; for(int k = 0; k < EC_N; k++) {Etot += glob[NS + EBK_N + k];}
    double Epot = pot_ok ? glob[NV - 1] / All.cf_atime : 0; /* physical: 1/2 sum m Phi / a */
    Etot += Epot;
    if(!first) {for(int b = 0; b < EBK_N; b++) {cum_book[b] += glob[NS + b];}}
    if(is_sync) {
        static double li = 0, li_lna = 0, li_T = 0;
        double lna = All.ComovingIntegrationOn ? log(All.Time) : 0.;
        double T = 2. * glob[NS + EBK_N + EC_KIN] + Epot + glob[NV + 1] + glob[NS + EBK_N + EC_MAG] + glob[NS + EBK_N + EC_RAD];
        if(!have_E0) {E0 = Etot; E0_abs = Eabs + fabs(Epot); have_E0 = 1; for(int s = 0; s < ESTEP_NSTAGE; s++) {cum_stage[s] = 0;} for(int b = 0; b < EBK_N; b++) {cum_book[b] = 0;} li = 0;}
        else if(All.ComovingIntegrationOn) {li += 0.5 * (T + li_T) * (lna - li_lna);}
        li_lna = lna; li_T = T;
        double sum_stage = 0, int_book = 0, sum_book = 0, sources = 0;
        for(int s = 0; s < ESTEP_NSTAGE; s++) {if(estep_stage_exact[s] == 2) {continue;} sum_stage += cum_stage[s]; if(estep_stage_exact[s] == 0) {sources += cum_stage[s];}}
        for(int b = 0; b < EBK_N; b++) {if(estep_book_stage[b] == -2) {continue;} sum_book += cum_book[b]; if(estep_book_stage[b] == -1) {int_book += cum_book[b];}}
        double dEtot = Etot - E0 + li; /* comoving: the expansion work, so dEtot is again the source/sink budget */
        fprintf(fd, "SYNC step=%lld t=%.16e", (long long)All.NumCurrentTiStep, All.Time);
        for(int k = 0; k < EC_N; k++) {fprintf(fd, " E_%s=%.16e", estep_comp_name[k], glob[NS + EBK_N + k]);}
        if(pot_ok) {fprintf(fd, " E_pot=%.16e E_tot=%.16e LI=%+.16e dE_tot=%+.16e int_res=%+.16e tot_res=%+.16e", Epot, Etot, li, dEtot, dEtot - sum_stage - int_book, dEtot - sum_book - sources);}
        else {fprintf(fd, " E_pot=nan E_tot=nan dE_tot=nan int_res=nan tot_res=nan");}
        fprintf(fd, " E_scale=%.16e trace_cpu_s=%.4e", E0_abs, glob[NV]); /* fixed: |E| + |E_pot| at the first sync */
        for(int s = 0; s < ESTEP_NSTAGE; s++) {fprintf(fd, " s_%s=%+.16e", estep_stage_name[s], cum_stage[s]);}
        for(int b = 0; b < EBK_N; b++) {fprintf(fd, " b_%s=%+.16e", estep_book_name[b], cum_book[b]);}
        fprintf(fd, "\n");
    }
    fflush(fd);
    first = 0;
}
#endif /* OUTPUT_ENERGY_STEP_TRACE */
