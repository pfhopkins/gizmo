/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../declarations/gpu_rng.h"
#include "single_star_record.h"
#include "stellar_evolution_tables.h"

/* Single-star source model for the shared mechanical feedback loop: IMF-sampled stars (single_star_record.h) take their
 * event times, ejecta, energies, yields and remnants from the stellar-evolution tables (stellar_evolution_tables.h).
 * The coupling is the mechanical feedback loop's: mode 0 = supernovae (pure thermal by default), mode 1 = mass return
 * (winds and AGB deaths, momentum-conserving).
 *   - winds (drawn mass >= 8 Msun, after the pre-main sequence): every star step returns the table's cumulative wind
 *     ejecta between the age of the last delivery and now, at the table wind speed, with that interval's composition;
 *     the last delivery age advances only when the mass coupled;
 *   - death (age > PMS time + table lifetime): first any wind mass still owed up to the end of life, then the death
 *     itself; the star's mass at the death trigger is frozen in the record (M_at_SN_trigger), and every pack of the event
 *     recomputes ejecta, energy and yields from it;
 *   - supernova classes (table remnant type): CCSN, ECSN, PPISN and PISN 1e51 erg; failed SN and direct BH no energy;
 *     ejecta = frozen mass - table remnant, composition from the table's SN element ejecta scaled to that mass;
 *   - white-dwarf remnants (AGB death): ejecta = frozen mass - table remnant at 30 km/s, AGB element ejecta scaled;
 *   - with GALSF_SINGLE_STARS_TYPE_IA a white dwarf explodes with the per-step hazard of the reference implementation
 *     (t^-1 delay-time distribution normalised to the dead-progenitor fraction, keyed random number): the whole mass at
 *     1e51 erg with the table's Type Ia yields, no remnant;
 *   - after the loop the star keeps exactly the table remnant mass; the difference to what the loop removed (rounding
 *     of the weight sums) is booked in the event log. An event that found no neighbour weight stays pending and fires
 *     again next step (counted).
 * Reference: stella_dev resolvedism_fb.cc, resolvedism_fb_thermal.cc, resolvedism_fb_momentum.cc. */

#if defined(GALSF_SINGLE_STARS_EVOLUTION_TABLES) && defined(GALSF_SINGLE_STARS_FROM_IMF) && defined(GALSF_FB_MECHANICAL)

#define SINGLE_STAR_NO_DEPOSIT_MAX 1000  /* steps an event may wait for neighbour weight */
#define SINGLE_STAR_WIND_MIN_MASS 8.0    /* drawn mass [Msun] above which winds are returned during the star's life */
#define SINGLE_STAR_WIND_FLUSH_MIN 1.e-6 /* wind mass [Msun] still owed at death that is returned before the death event */
#define SINGLE_STAR_AGB_SPEED_KMS 30.0   /* planetary-nebula ejection speed */
#ifdef GALSF_SINGLE_STARS_TYPE_IA
#define IA_T_MIN_GYR 0.04     /* minimum delay [Gyr] */
#define IA_F_WD 0.76          /* mass fraction of the IMF in white-dwarf progenitors */
#define IA_DTD_NORM 2.93e-4   /* rate = IA_DTD_NORM / t per Msun formed */
static constexpr uint64_t SINGLE_STAR_IA_RNG_SALT = gizmo_loop_rng_salt("single_star_type_ia");
#endif

/* table element of each FIRE metal slot 1..10 (slot 0 = sum of all table metals C..Zn; H implicit) */
static const int single_star_fire_slot_elem[11] = {-1, ELEM_He, ELEM_C, ELEM_N, ELEM_O, ELEM_Ne, ELEM_Mg, ELEM_Si, ELEM_S, ELEM_Ca, ELEM_Fe};

enum single_star_event_kind {SSE_NONE = 0, SSE_SN = 1, SSE_AGB = 2, SSE_WIND = 3, SSE_IA = 4};

struct single_star_event {
    int kind, remnant_type;
    double M_trigger, M_rem, Mej, E, v_kms; /* [Msun], [Msun], [Msun], [erg], [km/s] */
    double log_age_to;                      /* winds: table age (log10 yr) the delivery reaches */
    double y[STBL_NELEM];                   /* element ejecta [Msun], sum = Mej */
};

static inline void single_star_logMZ(int s, double *logM, double *logZ)
{
    *logM = log10(SingleStars.M_drawn[s]);
    *logZ = log10(DMAX(SingleStars.Z_birth[s], 1e-10));
}

static inline double single_star_age_yr(int i) {return evaluate_stellar_age_Gyr(i) * 1.e9;}
static inline double single_star_death_age_yr(double logM, double logZ) {return stellar_t_PMS(logM, logZ) + stellar_lifetime(logM, logZ);}

/* table age since the zero-age main sequence (log10 yr) at stellar age age_yr; < 0 means still pre-main-sequence */
static inline double single_star_table_log_age(double age_yr, double logM, double logZ)
{
    double t = age_yr - stellar_t_PMS(logM, logZ);
    return (t > 0) ? log10(DMAX(t, 100.)) : -1;
}

/* scale the element ejecta y[] to the total mass Mej */
static void single_star_scale(double *y, double Mej)
{
    double sum = 0; for(int k = 0; k < STBL_NELEM; k++) {sum += y[k];}
    double f = (sum > 0) ? Mej / sum : 0;
    for(int k = 0; k < STBL_NELEM; k++) {y[k] *= f;}
}

/* the wind mass owed between the last delivery and log age la_to, with its composition */
static void single_star_wind_owed(int s, double logM, double logZ, double la_to, struct single_star_event *ev)
{
    double la_from = SingleStars.last_wind_log_age[s], dM = 0;
    for(int k = 0; k < STBL_NELEM; k++) {
        ev->y[k] = stellar_elem_ej_wind_cumulative(logM, logZ, la_to, k) - ((la_from > 0) ? stellar_elem_ej_wind_cumulative(logM, logZ, la_from, k) : 0);
        dM += ev->y[k];
    }
    ev->Mej = DMAX(dM, 0);
    if(ev->Mej > 0) {
        int neg = 0; for(int k = 0; k < STBL_NELEM; k++) {if(ev->y[k] < 0) {neg = 1;}}
        if(neg) {for(int k = 0; k < STBL_NELEM; k++) {ev->y[k] = stellar_surface_abundance(logM, logZ, la_to, k);}} /* flat interval: surface composition */
        single_star_scale(ev->y, ev->Mej);
    }
    ev->v_kms = DMAX(stellar_v_wind(logM, logZ, la_to), 10.);
    ev->log_age_to = la_to;
}

/* add the winds not returned during the life (below 8 Msun, or not coupled), with their composition, to the death ejecta */
static void single_star_add_owed_winds(int s, double logM, double logZ, double *y)
{
    double la_end = log10(DMAX(stellar_lifetime(logM, logZ), 100.)), la_from = SingleStars.last_wind_log_age[s];
    for(int k = 0; k < STBL_NELEM; k++) {
        double owed = stellar_elem_ej_wind_cumulative(logM, logZ, la_end, k) - ((la_from > 0) ? stellar_elem_ej_wind_cumulative(logM, logZ, la_from, k) : 0);
        y[k] += DMAX(owed, 0);
    }
}

/* the event of star i this step (the kind set by single_star_eventrates), all from the record and the tables */
static void single_star_event_of(int i, int kind, struct single_star_event *ev)
{
    int s = single_star_slot(P[i].StarIdx);
    double logM, logZ; single_star_logMZ(s, &logM, &logZ);
    memset(ev, 0, sizeof(*ev)); ev->kind = kind;
    ev->remnant_type = stellar_remnant_type(logM, logZ);
    if(kind == SSE_WIND) {
        double age = single_star_age_yr(i), age_end = single_star_death_age_yr(logM, logZ);
        single_star_wind_owed(s, logM, logZ, single_star_table_log_age(DMIN(age, age_end), logM, logZ), ev);
        return;
    }
    ev->M_trigger = SingleStars.M_at_SN_trigger[s] * UNIT_MASS_IN_SOLAR;
    if(kind == SSE_IA) {
        ev->M_rem = 0; ev->E = 1.e51; ev->Mej = ev->M_trigger;
        for(int k = 0; k < STBL_NELEM; k++) {ev->y[k] = stellar_type_ia_yield(k);}
        single_star_scale(ev->y, ev->Mej);
        return;
    }
    ev->M_rem = DMIN(stellar_remnant_mass(logM, logZ), ev->M_trigger); /* a star never gains mass at death */
    ev->Mej = ev->M_trigger - ev->M_rem;
    if(kind == SSE_AGB) {
        ev->v_kms = SINGLE_STAR_AGB_SPEED_KMS;
        for(int k = 0; k < STBL_NELEM; k++) {ev->y[k] = stellar_elem_ej_AGB(logM, logZ, k);}
        single_star_add_owed_winds(s, logM, logZ, ev->y);
    } else {
        ev->E = 1.e51;
        if(ev->remnant_type == REM_PISN) {ev->M_rem = 0; ev->Mej = ev->M_trigger;}
        if(ev->remnant_type == REM_FSN || ev->remnant_type == REM_DBH) {ev->E = 0;}
        for(int k = 0; k < STBL_NELEM; k++) {ev->y[k] = stellar_elem_ej_SN(logM, logZ, k);}
        single_star_add_owed_winds(s, logM, logZ, ev->y);
    }
    single_star_scale(ev->y, ev->Mej);
}

static inline int single_star_kind_this_step(int i)
{
    int s = single_star_slot(P[i].StarIdx);
    if(P[i].SNe_ThisTimeStep > 0) {return (SingleStars.event_bits[s] & SINGLE_STAR_EVENT_AGB_DONE) ? SSE_IA : SSE_SN;}
    if(P[i].MassReturn_ThisTimeStep > 0) {return (SingleStars.M_at_SN_trigger[s] > 0) ? SSE_AGB : SSE_WIND;}
    return SSE_NONE;
}

/* largest timestep of single star p [code]: MaxStellarTimestep, 1/50 of the time to death within 100 of those, and never
   past the death (as the reference implementation, floor 10 yr) */
double single_star_max_timestep(int p)
{
    int s = single_star_slot(P[p].StarIdx);
    double logM, logZ; single_star_logMZ(s, &logM, &logZ);
    double to_death = single_star_death_age_yr(logM, logZ) - single_star_age_yr(p), dt_yr = All.MaxStellarTimestep;
    if(to_death > 0 && to_death < 100. * dt_yr) {dt_yr = DMIN(dt_yr, to_death / 50.);}
    if(to_death > 0 && to_death < 2. * dt_yr) {dt_yr = DMIN(dt_yr, 0.9 * to_death);}
    if(dt_yr < 10.) {dt_yr = 10.;}
    return dt_yr / UNIT_TIME_IN_YR;
}

/* single-star branch of mechanical_fb_calculate_eventrates: sets SNe_ThisTimeStep (supernova, Type Ia) or
   MassReturn_ThisTimeStep (wind, AGB death; the fraction of the star's mass) */
double single_star_eventrates(int i, double dt)
{
    int s = single_star_slot(P[i].StarIdx);
    P[i].SNe_ThisTimeStep = 0; P[i].MassReturn_ThisTimeStep = 0;
    uint8_t bits = SingleStars.event_bits[s];
    double logM, logZ; single_star_logMZ(s, &logM, &logZ);
    if(bits & SINGLE_STAR_EVENT_SN_DONE) {return 0;}
    if(bits & SINGLE_STAR_EVENT_AGB_DONE) { /* white dwarf */
#ifdef GALSF_SINGLE_STARS_TYPE_IA
        if(SingleStars.M_at_SN_trigger[s] > 0) {P[i].SNe_ThisTimeStep = 1; return 1;} /* pending */
        double t_Gyr = evaluate_stellar_age_Gyr(i), dt_Gyr = dt * UNIT_TIME_IN_GYR;
        if(t_Gyr < IA_T_MIN_GYR || dt_Gyr <= 0) {return 0;}
        double p = (IA_DTD_NORM / t_Gyr) * SingleStars.M_drawn[s] * dt_Gyr * IA_F_WD / DMAX(stellar_ia_fdead(t_Gyr * 1.e9, logZ), 1.e-6);
#ifdef TEST_SINGLE_STARS_TYPE_IA_RATE_MULTIPLIER
        p *= (TEST_SINGLE_STARS_TYPE_IA_RATE_MULTIPLIER); /* test only */
#endif
        if(particle_random_number(i, SINGLE_STAR_IA_RNG_SALT, (uint64_t)All.Ti_Current, 0) < p) {
            SingleStars.M_at_SN_trigger[s] = P[i].Mass; P[i].SNe_ThisTimeStep = 1; return 1;
        }
#endif
        return 0;
    }
    int is_wd = (stellar_remnant_type(logM, logZ) == REM_WD);
    if(SingleStars.M_at_SN_trigger[s] > 0) { /* a death that did not couple yet */
        if(is_wd) {P[i].MassReturn_ThisTimeStep = 1;} else {P[i].SNe_ThisTimeStep = 1;}
        return 1;
    }
    double age = single_star_age_yr(i), age_end = single_star_death_age_yr(logM, logZ);
    if(SingleStars.M_drawn[s] >= SINGLE_STAR_WIND_MIN_MASS) {
        double la = single_star_table_log_age(DMIN(age, age_end), logM, logZ);
        if(la > 0) {
            struct single_star_event ev; single_star_wind_owed(s, logM, logZ, la, &ev);
            if(age <= age_end && ev.Mej > 0) {P[i].MassReturn_ThisTimeStep = ev.Mej / UNIT_MASS_IN_SOLAR / P[i].Mass; return 0;}
            if(age > age_end && ev.Mej > SINGLE_STAR_WIND_FLUSH_MIN && SingleStars.no_deposit_count[s] < 2) {
                P[i].MassReturn_ThisTimeStep = ev.Mej / UNIT_MASS_IN_SOLAR / P[i].Mass; return 0; /* winds owed up to the end of life go first */
            }
        }
    }
    if(age <= age_end) {return 0;}
    SingleStars.M_at_SN_trigger[s] = P[i].Mass; /* death: freeze the mass for every pack of the event */
    SingleStars.no_deposit_count[s] = 0;
    if(is_wd) {P[i].MassReturn_ThisTimeStep = 1;} else {P[i].SNe_ThisTimeStep = 1;}
    return 1;
}

/* single-star branch of particle2in_addFB_fromstars: the event's ejecta mass, ejecta speed and yields */
void single_star_particle2in(struct addFB_evaluate_data_in_ *in, int i, int fb_loop_iteration)
{
    in->Msne = 0; in->SNe_v_ejecta = 0;
    int kind = single_star_kind_this_step(i);
    if(kind == SSE_NONE || fb_loop_iteration != ((kind == SSE_SN || kind == SSE_IA) ? 0 : 1)) {return;}
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
        if(FdSingleStarEvents && RestartFlag == 0) {fprintf(FdSingleStarEvents, "# time ID kind(1=SN,2=AGB,3=wind,4=Ia) remnant_type M_drawn[Msun] M_trigger[Msun] M_ejecta[Msun] M_remnant[Msun] E[erg] M_Z_ejecta[Msun] mass_residual[Msun] no_deposit_steps\n");}
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

/* after the feedback loop (all ranks): a death that coupled leaves exactly the table remnant; a wind that coupled
   advances the last delivery age; an event that found no neighbour weight stays pending */
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
        double residual = (kind == SSE_WIND) ? removed - ev.Mej : P[i].Mass * UNIT_MASS_IN_SOLAR - ev.M_rem;
        double row[SSFB_LOG_COLS] = {All.Time, (double)P[i].ID, (double)kind, (double)ev.remnant_type, SingleStars.M_drawn[s],
                                     (kind == SSE_WIND) ? ssfb_mass_before[a] * UNIT_MASS_IN_SOLAR : ev.M_trigger, ev.Mej, (kind == SSE_WIND) ? P[i].Mass * UNIT_MASS_IN_SOLAR : ev.M_rem,
                                     (ev.Mej > 0) ? ev.E : 0., z, residual, (double)SingleStars.no_deposit_count[s]};
        SingleStars.no_deposit_count[s] = 0;
        if(kind == SSE_WIND) {
            SingleStars.last_wind_log_age[s] = ev.log_age_to;
            continue; /* winds are frequent: not logged per step */
        }
        ssfb_rows.insert(ssfb_rows.end(), row, row + SSFB_LOG_COLS);
        P[i].Mass = ev.M_rem / UNIT_MASS_IN_SOLAR; /* exactly the table remnant; 0 (PISN, Type Ia) removes the particle */
        SingleStars.event_bits[s] |= (kind == SSE_AGB) ? SINGLE_STAR_EVENT_AGB_DONE : SINGLE_STAR_EVENT_SN_DONE;
        SingleStars.M_at_SN_trigger[s] = 0;
        P[i].SNe_ThisTimeStep = 0; P[i].MassReturn_ThisTimeStep = 0;
    }
#ifdef ISRF_AND_CR_RATE_SCALE_WITH_RECENT_SFR
    for(int a = 0; a < n; a++) {recent_sfr_young_mass_add(P[list[a]].StellarAge, P[list[a]].Mass - ssfb_mass_before[a]);} /* winds and deaths */
#endif
    single_star_events_flush();
}

#endif
