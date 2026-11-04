/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "single_star_record.h"
#include "stellar_evolution_tables.h"

/* Single-star source model for the shared mechanical feedback loop: IMF-sampled stars (single_star_record.h) take their
 * supernova times, ejecta, energies, yields and remnants from the stellar-evolution tables (stellar_evolution_tables.h).
 * The coupling is the mechanical feedback loop's mode 0 (supernovae).
 *   - death (age > PMS time + table lifetime): the star's mass at the death trigger is frozen in the record
 *     (M_at_SN_trigger), and every pack of the event recomputes ejecta, energy and yields from it;
 *   - supernova classes (table remnant type): CCSN, ECSN, PPISN and PISN 1e51 erg; failed SN and direct BH no energy;
 *     ejecta = frozen mass - table remnant, composition from the table's SN element ejecta scaled to that mass;
 *   - after the loop the star keeps exactly the table remnant mass; the difference to what the loop removed (rounding
 *     of the weight sums) is booked in the event log. An event that found no neighbour weight stays pending and fires
 *     again next step (counted).
 * Reference: stella_dev resolvedism_fb.cc, resolvedism_fb_thermal.cc. */

#if defined(GALSF_SINGLE_STARS_EVOLUTION_TABLES) && defined(GALSF_SINGLE_STARS_FROM_IMF) && defined(GALSF_FB_MECHANICAL)

#define SINGLE_STAR_NO_DEPOSIT_MAX 1000  /* steps an event may wait for neighbour weight */

/* table element of each FIRE metal slot 1..10 (slot 0 = sum of all table metals C..Zn; H implicit) */
static const int single_star_fire_slot_elem[11] = {-1, ELEM_He, ELEM_C, ELEM_N, ELEM_O, ELEM_Ne, ELEM_Mg, ELEM_Si, ELEM_S, ELEM_Ca, ELEM_Fe};

enum single_star_event_kind {SSE_NONE = 0, SSE_SN = 1};

struct single_star_event {
    int kind, remnant_type;
    double M_trigger, M_rem, Mej, E, v_kms; /* [Msun], [Msun], [Msun], [erg], [km/s] */
    double y[STBL_NELEM];                   /* element ejecta [Msun], sum = Mej */
};

static inline void single_star_logMZ(int s, double *logM, double *logZ)
{
    *logM = log10(SingleStars.M_drawn[s]);
    *logZ = log10(DMAX(SingleStars.Z_birth[s], 1e-10));
}

static inline double single_star_age_yr(int i) {return evaluate_stellar_age_Gyr(i) * 1.e9;}
static inline double single_star_death_age_yr(double logM, double logZ) {return stellar_t_PMS(logM, logZ) + stellar_lifetime(logM, logZ);}

/* scale the element ejecta y[] to the total mass Mej */
static void single_star_scale(double *y, double Mej)
{
    double sum = 0; for(int k = 0; k < STBL_NELEM; k++) {sum += y[k];}
    double f = (sum > 0) ? Mej / sum : 0;
    for(int k = 0; k < STBL_NELEM; k++) {y[k] *= f;}
}

/* the event of star i this step (the kind set by single_star_eventrates), all from the record and the tables */
static void single_star_event_of(int i, int kind, struct single_star_event *ev)
{
    int s = single_star_slot(P[i].StarIdx);
    double logM, logZ; single_star_logMZ(s, &logM, &logZ);
    memset(ev, 0, sizeof(*ev)); ev->kind = kind;
    ev->remnant_type = stellar_remnant_type(logM, logZ);
    ev->M_trigger = SingleStars.M_at_SN_trigger[s] * UNIT_MASS_IN_SOLAR;
    ev->M_rem = DMIN(stellar_remnant_mass(logM, logZ), ev->M_trigger); /* a star never gains mass at death */
    ev->Mej = ev->M_trigger - ev->M_rem;
    ev->E = 1.e51;
    if(ev->remnant_type == REM_PISN) {ev->M_rem = 0; ev->Mej = ev->M_trigger;}
    if(ev->remnant_type == REM_FSN || ev->remnant_type == REM_DBH) {ev->E = 0;}
    for(int k = 0; k < STBL_NELEM; k++) {ev->y[k] = stellar_elem_ej_SN(logM, logZ, k);}
    single_star_scale(ev->y, ev->Mej);
}

static inline int single_star_kind_this_step(int i)
{
    return (P[i].SNe_ThisTimeStep > 0) ? SSE_SN : SSE_NONE;
}

/* single-star branch of mechanical_fb_calculate_eventrates: sets SNe_ThisTimeStep at the star's death */
double single_star_eventrates(int i, double dt)
{
    int s = single_star_slot(P[i].StarIdx);
    P[i].SNe_ThisTimeStep = 0;
    double logM, logZ; single_star_logMZ(s, &logM, &logZ);
    if(SingleStars.event_bits[s] & SINGLE_STAR_EVENT_SN_DONE) {return 0;}
    if(stellar_remnant_type(logM, logZ) == REM_WD) {return 0;} /* white-dwarf progenitors: no event in this source model yet */
    if(SingleStars.M_at_SN_trigger[s] > 0) {P[i].SNe_ThisTimeStep = 1; return 1;} /* a death that did not couple yet */
    if(single_star_age_yr(i) <= single_star_death_age_yr(logM, logZ)) {return 0;}
    SingleStars.M_at_SN_trigger[s] = P[i].Mass; /* death: freeze the mass for every pack of the event */
    SingleStars.no_deposit_count[s] = 0;
    P[i].SNe_ThisTimeStep = 1;
    return 1;
}

/* single-star branch of particle2in_addFB_fromstars: the event's ejecta mass, ejecta speed and yields */
void single_star_particle2in(struct addFB_evaluate_data_in_ *in, int i, int fb_loop_iteration)
{
    in->Msne = 0; in->SNe_v_ejecta = 0;
    int kind = single_star_kind_this_step(i);
    if(kind == SSE_NONE || fb_loop_iteration != 0) {return;}
    struct single_star_event ev; single_star_event_of(i, kind, &ev);
    if(ev.Mej <= 0) {return;}
    in->Msne = ev.Mej / UNIT_MASS_IN_SOLAR;
    in->SNe_v_ejecta = (ev.E > 0) ? sqrt(2. * (ev.E / UNIT_ENERGY_IN_CGS) / in->Msne) : ev.v_kms / UNIT_VEL_IN_KMS;
#ifdef METALS
    double z = 0; for(int k = ELEM_C; k < STBL_NELEM; k++) {z += ev.y[k];}
    for(int k = 0; k < NUM_METAL_SPECIES; k++) {in->yields[k] = 0;} /* other slots (tracers) are only diluted */
    in->yields[0] = z / ev.Mej;
#if (NUM_LIVE_SPECIES_FOR_COOLTABLES >= 10)
    for(int k = 1; k <= 10; k++) {in->yields[k] = ev.y[single_star_fire_slot_elem[k]] / ev.Mej;}
#endif
#endif
}

/* event log rows, gathered and written by rank 0 in the step of the event:
   time ID kind remnant_type M_drawn M_trigger M_ejecta M_remnant E[erg] M_Z_ejecta mass_residual no_deposit_steps */
#define SSFB_LOG_COLS 12
static std::vector<double> ssfb_rows;
static FILE *FdSingleStarEvents = NULL;
static std::vector<double> ssfb_mass_before; /* star masses before the loop, by position in the active list */

/* collective, from the feedback loop's active steps only (one integer reduction); gathers only if a row exists */
static void single_star_events_flush(void)
{
    int n = (int) ssfb_rows.size(), n_any = 0, tot = 0;
    MPI_Allreduce(&n, &n_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(n_any == 0) {return;}
    std::vector<int> counts(NTask), displs(NTask);
    MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(ThisTask == 0) {for(int t = 0; t < NTask; t++) {displs[t] = tot; tot += counts[t];}}
    std::vector<double> all(DMAX(tot, 1));
    MPI_Gatherv(ssfb_rows.data(), n, MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    ssfb_rows.clear();
    if(ThisTask != 0 || tot == 0) {return;}
    if(!FdSingleStarEvents) {
        char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE];
        snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s%s", All.OutputDir, "single_star_events.txt");
        FdSingleStarEvents = fopen(buf, (RestartFlag == 0) ? "w" : "a");
        if(FdSingleStarEvents && RestartFlag == 0) {fprintf(FdSingleStarEvents, "# time ID kind(1=SN) remnant_type M_drawn[Msun] M_trigger[Msun] M_ejecta[Msun] M_remnant[Msun] E[erg] M_Z_ejecta[Msun] mass_residual[Msun] no_deposit_steps\n");}
    }
    if(!FdSingleStarEvents) {return;}
    for(int r = 0; r < tot / SSFB_LOG_COLS; r++) {
        const double *x = &all[SSFB_LOG_COLS * r];
        fprintf(FdSingleStarEvents, "%.16g %llu %d %d %.17g %.17g %.17g %.17g %.6e %.17g %.6e %d\n", x[0], (unsigned long long)x[1], (int)x[2], (int)x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], (int)x[11]);
    }
    fflush(FdSingleStarEvents);
}

/* before the feedback loop: the masses of the sources */
void single_star_feedback_prepare(const int *list, int n)
{
    ssfb_mass_before.resize(n > 0 ? n : 1);
    for(int a = 0; a < n; a++) {ssfb_mass_before[a] = P[list[a]].Mass;}
}

/* after the feedback loop (all ranks): a death that coupled leaves exactly the table remnant; an event that found no
   neighbour weight stays pending */
void single_star_feedback_finalize(const int *list, int n)
{
    for(int a = 0; a < n; a++) {
        int i = list[a];
        if(P[i].StarIdx <= 0) {continue;}
        int kind = single_star_kind_this_step(i);
        if(kind == SSE_NONE) {continue;}
        int s = single_star_slot(P[i].StarIdx);
        struct single_star_event ev; single_star_event_of(i, kind, &ev);
        double removed = (ssfb_mass_before[a] - P[i].Mass) * UNIT_MASS_IN_SOLAR;
        if(ev.Mej > 0 && removed <= 0) { /* nothing coupled: try again next step */
            if(++SingleStars.no_deposit_count[s] > SINGLE_STAR_NO_DEPOSIT_MAX) {
                gizmo_request_controlled_stop(7831, "single-star feedback event found no gas to couple to for too many steps", __FILE__, __LINE__, __FUNCTION__);
            }
            continue;
        }
        double z = 0; for(int k = ELEM_C; k < STBL_NELEM; k++) {z += ev.y[k];}
        double residual = P[i].Mass * UNIT_MASS_IN_SOLAR - ev.M_rem;
        double row[SSFB_LOG_COLS] = {All.Time, (double)P[i].ID, (double)kind, (double)ev.remnant_type, SingleStars.M_drawn[s],
                                     ev.M_trigger, ev.Mej, ev.M_rem, (ev.Mej > 0) ? ev.E : 0., z, residual, (double)SingleStars.no_deposit_count[s]};
        SingleStars.no_deposit_count[s] = 0;
        ssfb_rows.insert(ssfb_rows.end(), row, row + SSFB_LOG_COLS);
        P[i].Mass = ev.M_rem / UNIT_MASS_IN_SOLAR; /* exactly the table remnant; 0 (PISN, Type Ia) removes the particle */
        SingleStars.event_bits[s] |= SINGLE_STAR_EVENT_SN_DONE;
        SingleStars.M_at_SN_trigger[s] = 0;
        P[i].SNe_ThisTimeStep = 0;
    }    single_star_events_flush();
}

#endif
