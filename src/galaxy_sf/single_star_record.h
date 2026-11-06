/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* Compact record of an IMF-sampled single star (GALSF_SINGLE_STARS_FROM_IMF): the few quantities only such stars carry,
   in rank-local struct-of-arrays slots, reached through P[i].StarIdx (slot + 1; 0 = no record). It follows its particle
   through domain exchange and restart files and is written to snapshots. */
#ifndef SINGLE_STAR_RECORD_H
#define SINGLE_STAR_RECORD_H
#include <stdint.h>
#ifdef SINGLE_STAR_RECORDS
#ifdef KETJU_REGULARIZATION
#include "../gravity/ketju/ketju_member_state.h"
#ifdef KETJU_PN_REMNANT_TAG
#define SINGLE_STAR_RECORD_KETJU_FIELDS(X) X(struct ketju_member_state, ketju, "chain member state") X(int8_t, remnant_tag, "remnant type + 1; 0 = live star")
#else
#define SINGLE_STAR_RECORD_KETJU_FIELDS(X) X(struct ketju_member_state, ketju, "chain member state")
#endif
#else
#define SINGLE_STAR_RECORD_KETJU_FIELDS(X)
#endif

#ifdef GALSF_SINGLE_STARS_EVENT_LOGS
#define SINGLE_STAR_RECORD_LOG_FIELDS(X) X(double, u_ambient, "mass-weighted gas specific energy around the star [code]")
#else
#define SINGLE_STAR_RECORD_LOG_FIELDS(X)
#endif

/* the fields: X(type, name, comment); a field is added together with its first user */
#define SINGLE_STAR_RECORD_FIELDS(X) \
    X(double, M_drawn, "drawn IMF mass [Msun]: the evolution mass for the tables") \
    X(double, Z_birth, "total metallicity at birth (mass fraction)") \
    SINGLE_STAR_RECORD_TABLE_FIELDS(X) \
    SINGLE_STAR_RECORD_FEEDBACK_FIELDS(X) \
    SINGLE_STAR_RECORD_KETJU_FIELDS(X)
#ifdef GALSF_SINGLE_STARS_EVOLUTION_TABLES
#define SINGLE_STAR_RECORD_TABLE_FIELDS(X) \
    X(float, rho_birth, "physical gas density of the parent cell at birth [code]; 0 = unknown")
#else
#define SINGLE_STAR_RECORD_TABLE_FIELDS(X)
#endif

#if defined(GALSF_SINGLE_STARS_EVOLUTION_TABLES) && defined(GALSF_FB_MECHANICAL)
#define SINGLE_STAR_RECORD_FEEDBACK_FIELDS(X) \
    X(double, M_at_SN_trigger, "particle mass when its supernova was triggered [code]; 0 = none pending") \
    X(uint16_t, no_deposit_count, "steps a triggered event found no neighbour weight") \
    X(double, last_wind_log_age, "table age (log10 yr) up to which winds have been returned; 0 = none yet") \
    X(uint8_t, event_bits, "single_star_event_bits") \
    SINGLE_STAR_RECORD_LOG_FIELDS(X)
enum single_star_event_bits {SINGLE_STAR_EVENT_SN_DONE = 1, SINGLE_STAR_EVENT_AGB_DONE = 2};
#else
#define SINGLE_STAR_RECORD_FEEDBACK_FIELDS(X)
#endif

/* one record, as it travels between ranks */
struct single_star_record {
#define SSR_MEMBER(type, name, comment) type name;
    SINGLE_STAR_RECORD_FIELDS(SSR_MEMBER)
#undef SSR_MEMBER
};

/* the slots: one array per field, plus the used flag */
struct single_star_records {
#define SSR_ARRAY(type, name, comment) type *name;
    SINGLE_STAR_RECORD_FIELDS(SSR_ARRAY)
#undef SSR_ARRAY
    uint8_t *used;
    int capacity, n_used;
};
extern struct single_star_records SingleStars;

#ifdef GALSF_SINGLE_STARS_EVENT_LOGS
/* one row of SNinfo.txt (file 0), AGBinfo.txt (1) or WINDinfo.txt (2); wind: mstar = M_init, mej = dM, E = dp [g cm/s] */
struct single_star_log_row {double t, x[3], u, rho, mstar, mej, zej, C, O, Si, Fe, age, life, E; unsigned long long id; int file, remnant, timebin, status;};
void single_star_event_log_push(const struct single_star_log_row *r);
#endif

int  single_star_record_new(const struct single_star_record *r); /* returns slot + 1 */
void single_star_record_free(int star_idx);
void single_star_record_get(int star_idx, struct single_star_record *r);
void single_star_record_resize(int capacity);
void single_star_record_reset(int capacity);   /* restart read: empty arrays of this capacity, then read them */
void single_star_record_rebuild_free_list(void); /* restart read: after the arrays are read */
static inline int single_star_slot(int star_idx) {return star_idx - 1;}

#endif
#endif
