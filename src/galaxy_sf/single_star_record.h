/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* Compact record of an IMF-sampled single star (GALSF_SINGLE_STARS_FROM_IMF): the few quantities only such stars carry,
   in rank-local struct-of-arrays slots, reached through P[i].StarIdx (slot + 1; 0 = no record). It follows its particle
   through domain exchange and restart files and is written to snapshots. */
#ifndef SINGLE_STAR_RECORD_H
#define SINGLE_STAR_RECORD_H
#include <stdint.h>
#include <vector>
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
#ifdef STARS_INTEGRATED_BY_RUMI
#include "../gravity/rumi/rumi_member_state.h"
#define SINGLE_STAR_RECORD_RUMI_FIELDS(X) X(struct rumi_member_state, rumi, "RUMI++ step state")
#else
#define SINGLE_STAR_RECORD_RUMI_FIELDS(X)
#endif

#ifdef GALSF_SINGLE_STARS_EVENT_LOGS
#define SINGLE_STAR_RECORD_LOG_FIELDS(X) X(double, u_ambient, "mass-weighted gas specific energy around the star [code]")
#else
#define SINGLE_STAR_RECORD_LOG_FIELDS(X)
#endif

/* the fields: X(type, name, comment); a field is added together with its first user */
#ifdef GALSF_SINGLE_STARS_POPIII
#define SINGLE_STAR_RECORD_POPIII_FIELDS(X) X(int8_t, popIII, "1: formed through the Pop III channel (pristine gas, Pop III IMF, birth Z below the critical Z)")
#else
#define SINGLE_STAR_RECORD_POPIII_FIELDS(X)
#endif
#define SINGLE_STAR_RECORD_FIELDS(X) \
    X(double, M_drawn, "drawn IMF mass [Msun]: the evolution mass for the tables") \
    X(double, Z_birth, "total metallicity at birth (mass fraction)") \
    SINGLE_STAR_RECORD_POPIII_FIELDS(X) \
    SINGLE_STAR_RECORD_TABLE_FIELDS(X) \
    SINGLE_STAR_RECORD_FEEDBACK_FIELDS(X) \
    SINGLE_STAR_RECORD_KETJU_FIELDS(X) \
    SINGLE_STAR_RECORD_RUMI_FIELDS(X)
#ifdef GALSF_SINGLE_STARS_EVOLUTION_TABLES
#if defined(GALSF_SINGLE_STARS_LIGHT_REFRESHES_TREE) || defined(GALSF_SINGLE_STARS_LIGHT_UPDATES_NODE_COPIES)
struct single_star_tree_lum {float L[N_RT_FREQ_BINS];};
#define SINGLE_STAR_RECORD_TREE_FIELDS(X) X(struct single_star_tree_lum, tree_lum, "band luminosities as the gravity-tree nodes last saw them [code]")
#else
#define SINGLE_STAR_RECORD_TREE_FIELDS(X)
#endif
#ifdef SINGLE_STAR_ION_PHOTON_ENERGY_FROM_TABLES
#define SINGLE_STAR_RECORD_ION_FIELDS(X) X(double, L_ion_counted, "ionising light [erg/s] in the rank sum of the mean photon energy") X(double, Q_ion_counted, "ionising photons [1/s], likewise")
#else
#define SINGLE_STAR_RECORD_ION_FIELDS(X)
#endif
#ifdef SINGLE_STAR_RADPRESSURE_LIVE_DUST
#define SINGLE_STAR_RECORD_DUST_FIELDS(X) X(double, dust_ambient, "mass-weighted dust mass fraction of the gas around the star")
#else
#define SINGLE_STAR_RECORD_DUST_FIELDS(X)
#endif
#define SINGLE_STAR_RECORD_TABLE_FIELDS(X) \
    X(float, rho_birth, "physical gas density of the parent cell at birth [code]; 0 = unknown") \
    SINGLE_STAR_RECORD_ION_FIELDS(X) \
    SINGLE_STAR_RECORD_DUST_FIELDS(X) \
    SINGLE_STAR_RECORD_TREE_FIELDS(X)
#else
#define SINGLE_STAR_RECORD_TABLE_FIELDS(X)
#endif

#if defined(GALSF_SINGLE_STARS_EVOLUTION_TABLES) && defined(GALSF_FB_MECHANICAL)
#define SINGLE_STAR_RECORD_FEEDBACK_FIELDS(X) \
    X(double, M_at_SN_trigger, "particle mass when its supernova was triggered [code]; 0 = none pending") \
    X(uint16_t, no_deposit_count, "steps a triggered event found no neighbour weight") \
    X(double, last_wind_log_age, "table age (log10 yr) up to which winds have been returned; 0 = none yet") \
    X(double, M_birth, "mass the star was born with, after gas collection [Msun]: the winds returned so far are M_birth less its mass; 0 = M_drawn") \
    X(double, wind_mass, "wind mass of this step's delivery [Msun], held while the delivery couples") \
    X(double, wind_speed, "wind speed of this step's delivery [km/s]") \
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
struct single_star_log_row {double t, x[3], u, rho, mstar, mej, zej, C, O, Si, Fe, age, life, E, nH, T, h_pc, nngb, nH_birth, nH_stale; unsigned long long id; int file, remnant, timebin, status;};
void single_star_event_log_push(const struct single_star_log_row *r);
#endif

int  single_star_record_new(const struct single_star_record *r); /* returns slot + 1 */
void single_star_record_free(int star_idx);
void single_star_record_get(int star_idx, struct single_star_record *r);
void single_star_record_resize(int capacity);
void single_star_record_reset(int capacity);   /* restart read: empty arrays of this capacity, then read them */
void single_star_record_rebuild_free_list(void); /* restart read: after the arrays are read */
static inline int single_star_slot(int star_idx) {return star_idx - 1;}
#if defined(GALSF_SINGLE_STARS_EVOLUTION_TABLES) && defined(GALSF_FB_MECHANICAL)
static inline double single_star_birth_mass(int s) {return (SingleStars.M_birth[s] > 0) ? SingleStars.M_birth[s] : SingleStars.M_drawn[s];} /* [Msun] */
double single_star_wind_cumulative(double logM, double logZ, double log_age); /* [Msun], single_star_feedback.cc */
#endif

/* records travelling with their stars through a raw particle exchange (SUBFIND redistribution; domain_exchange has its own copy of this):
   detach each outgoing star's record in send order, swap them with the same partner as the particles, attach to the received stars in order */
struct single_star_xfer {std::vector<std::vector<struct single_star_record> > send, recv;};
void single_star_xfer_init(struct single_star_xfer &x, int ntask);
void single_star_xfer_detach(struct single_star_xfer &x, const struct particle_data &p, int target); /* frees the local slot; p keeps StarIdx > 0 as the marker */
void single_star_xfer_swap(struct single_star_xfer &x, int target, const struct particle_data *recv_first, int count_recv); /* after the particles arrived */
void single_star_xfer_attach(struct single_star_xfer &x, int target, struct particle_data *recv_first, int count_recv);

#endif
#endif
