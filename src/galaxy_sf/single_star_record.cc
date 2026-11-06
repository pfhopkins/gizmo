/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
#include <stdlib.h>
#include <string.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "single_star_record.h"

/* compact per-star record of IMF-sampled single stars; see single_star_record.h */
#ifdef GALSF_SINGLE_STARS_FROM_IMF
#if defined(SUBFIND) || defined(FOF)
#error "GALSF_SINGLE_STARS_FROM_IMF: the single-star records do not yet follow particles through the FoF/SUBFIND redistribution"
#endif
/* until the single-star source model drives them, the stellar-population feedback and radiation would treat a sampled star as a population */
#if (defined(GALSF_FB_MECHANICAL) && !defined(GALSF_SINGLE_STARS_EVOLUTION_TABLES)) || defined(GALSF_FB_THERMAL) || defined(GALSF_FB_FIRE_STELLAREVOLUTION)
#error "GALSF_SINGLE_STARS_FROM_IMF: single stars drive only the mechanical feedback loop, and only with GALSF_SINGLE_STARS_EVOLUTION_TABLES; other feedback would give sampled stars stellar-population rates"
#endif
#if defined(GALSF_FB_FIRE_RT_LOCALRP) || defined(GALSF_FB_FIRE_RT_HIIHEATING) || defined(GALSF_FB_FIRE_RT_LONGRANGE) || defined(RADTRANSFER) || defined(RT_USE_GRAVTREE)
#error "GALSF_SINGLE_STARS_FROM_IMF: no single-star branch in the stellar radiation sources yet; sampled stars would get stellar-population luminosities"
#endif
#if defined(COSMIC_RAY_FLUID) && defined(GALSF_SINGLE_STARS_EVOLUTION_TABLES) && defined(GALSF_FB_MECHANICAL)
#error "GALSF_SINGLE_STARS_FROM_IMF: single-star supernovae inject no cosmic rays yet (the injection of the feedback loop is FIRE-only and the pure-thermal coupling heats with all the energy)"
#endif
#if defined(GALSF_SFR_IMF_SAMPLING) || defined(GALSF_SFR_IMF_VARIATION) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(GALSF_MERGER_STARCLUSTER_PARTICLES)
#error "GALSF_SINGLE_STARS_FROM_IMF: another stellar-mass model is switched on (GALSF_SFR_IMF_SAMPLING, GALSF_SFR_IMF_VARIATION, SINGLE_STAR_SINK_DYNAMICS or GALSF_MERGER_STARCLUSTER_PARTICLES)"
#endif

struct single_star_records SingleStars;
static int *single_star_free_slots = NULL, single_star_n_free = 0; /* stack of unused slots */

/* bytes per slot: every field, the used flag and the free-list entry */
static size_t ssr_slot_bytes(void)
{
    size_t b = sizeof(uint8_t) + sizeof(int);
#define SSR_SIZE(type, name, comment) b += sizeof(type);
    SINGLE_STAR_RECORD_FIELDS(SSR_SIZE)
#undef SSR_SIZE
    return b;
}

static void *ssr_grow(void *p, size_t elem, int n_old, int n_new)
{
    void *q = realloc(p, (size_t)n_new * elem);
    if(!q) {endrun(7821);}
    memset((char *)q + (size_t)n_old * elem, 0, (size_t)(n_new - n_old) * elem);
    return q;
}

void single_star_record_resize(int capacity)
{
    int n_old = SingleStars.capacity;
    if(capacity <= n_old) {return;}
#define SSR_GROW(type, name, comment) SingleStars.name = (type *) ssr_grow(SingleStars.name, sizeof(type), n_old, capacity);
    SINGLE_STAR_RECORD_FIELDS(SSR_GROW)
#undef SSR_GROW
    SingleStars.used = (uint8_t *) ssr_grow(SingleStars.used, sizeof(uint8_t), n_old, capacity);
    single_star_free_slots = (int *) ssr_grow(single_star_free_slots, sizeof(int), n_old, capacity);
    for(int s = capacity - 1; s >= n_old; s--) {if(!SingleStars.used[s]) {single_star_free_slots[single_star_n_free++] = s;}}
    SingleStars.capacity = capacity;
    gizmo_mem_account_add(GIZMO_MEM_SINGLE_STARS, (long long)(capacity - n_old) * (long long)ssr_slot_bytes());
}

void single_star_record_reset(int capacity)
{
#define SSR_FREE(type, name, comment) free(SingleStars.name);
    SINGLE_STAR_RECORD_FIELDS(SSR_FREE)
#undef SSR_FREE
    free(SingleStars.used); free(single_star_free_slots);
    gizmo_mem_account_add(GIZMO_MEM_SINGLE_STARS, -(long long)SingleStars.capacity * (long long)ssr_slot_bytes());
    memset(&SingleStars, 0, sizeof(SingleStars)); single_star_free_slots = NULL; single_star_n_free = 0;
    single_star_record_resize(capacity);
}

void single_star_record_rebuild_free_list(void)
{
    single_star_n_free = 0; SingleStars.n_used = 0;
    for(int s = SingleStars.capacity - 1; s >= 0; s--) {if(SingleStars.used[s]) {SingleStars.n_used++;} else {single_star_free_slots[single_star_n_free++] = s;}}
}

int single_star_record_new(const struct single_star_record *r)
{
    if(single_star_n_free == 0) {single_star_record_resize((SingleStars.capacity > 0) ? 2 * SingleStars.capacity : 64);}
    int s = single_star_free_slots[--single_star_n_free];
    SingleStars.used[s] = 1; SingleStars.n_used++;
#define SSR_SET(type, name, comment) SingleStars.name[s] = r->name;
    SINGLE_STAR_RECORD_FIELDS(SSR_SET)
#undef SSR_SET
    return s + 1;
}

void single_star_record_free(int star_idx)
{
    int s = single_star_slot(star_idx);
    if(star_idx <= 0 || s >= SingleStars.capacity || !SingleStars.used[s]) {return;}
    SingleStars.used[s] = 0; SingleStars.n_used--;
    single_star_free_slots[single_star_n_free++] = s;
}

void single_star_record_get(int star_idx, struct single_star_record *r)
{
    int s = single_star_slot(star_idx);
    memset(r, 0, sizeof(*r));
    if(star_idx <= 0 || s >= SingleStars.capacity || !SingleStars.used[s]) {return;}
#define SSR_GET(type, name, comment) r->name = SingleStars.name[s];
    SINGLE_STAR_RECORD_FIELDS(SSR_GET)
#undef SSR_GET
}

#endif
