#ifndef HBT_KEY_H
#define HBT_KEY_H
/* Stable particle key for SUBFIND-HBT: (ID, child number) is constant over a particle's life (the generation is a
   counter and is not part of it). A split or spawn gives the newborn child = c + 2^g, where c < 2^g is the parent's
   child number and g its generation before the increment (mesh/merge_split.cc, galaxy_sf/sfr_eff.cc), so clearing
   the newborn's highest child bit gives the parent's key. Not covered: SINK_WIND_SPAWN cells (sinks/sink_swallow_and_kick.cc
   gives them a shared ID and child = c + generation), generation rollover after 31 splits;
   FOF_UNIQUE_KEY_CHECK detects the resulting duplicates.
   Device-safe: plain functions over flat arrays, no STL, no recursion, no MPI; loops have independent iterations
   (Kokkos::parallel_for / parallel_reduce). */
#include <stdint.h>

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

typedef struct {uint64_t id; uint64_t child;} hbt_key_t;

KOKKOS_INLINE_FUNCTION hbt_key_t hbt_key_make(uint64_t id, uint64_t child) {hbt_key_t k; k.id = id; k.child = child; return k;}

KOKKOS_INLINE_FUNCTION int hbt_key_cmp(hbt_key_t a, hbt_key_t b)
{
    if(a.id != b.id) {return (a.id < b.id) ? -1 : 1;}
    if(a.child != b.child) {return (a.child < b.child) ? -1 : 1;}
    return 0;
}

KOKKOS_INLINE_FUNCTION int hbt_key_equal(hbt_key_t a, hbt_key_t b) {return (a.id == b.id) && (a.child == b.child);}

/* highest set bit of x, -1 for x = 0 */
KOKKOS_INLINE_FUNCTION int hbt_top_bit(uint64_t x) {int b = -1; while(x) {x >>= 1; b++;} return b;}

/* parent key of a split or spawned particle; returns 0 for an original particle (child 0) */
KOKKOS_INLINE_FUNCTION int hbt_key_parent(hbt_key_t k, hbt_key_t *parent)
{
    if(k.child == 0) {return 0;}
    *parent = hbt_key_make(k.id, k.child ^ (((uint64_t)1) << hbt_top_bit(k.child)));
    return 1;
}

/* GADGET-4 IDS_64BIT compatible packing ID*2^32 + child (top bit stays free); returns 0 if it does not fit */
KOKKOS_INLINE_FUNCTION int hbt_key_pack64(hbt_key_t k, uint64_t *packed)
{
    if(k.id >= (((uint64_t)1) << 31) || k.child >= (((uint64_t)1) << 32)) {return 0;}
    *packed = (k.id << 32) | k.child;
    return 1;
}

KOKKOS_INLINE_FUNCTION hbt_key_t hbt_key_unpack64(uint64_t p) {return hbt_key_make(p >> 32, p & 0xffffffffULL);}

/* index of k in a table sorted by hbt_key_cmp, -1 if absent */
KOKKOS_INLINE_FUNCTION long hbt_key_find(const hbt_key_t *table, long n, hbt_key_t k)
{
    long lo = 0, hi = n - 1;
    while(lo <= hi)
    {
        long mid = lo + (hi - lo) / 2; int c = hbt_key_cmp(table[mid], k);
        if(c == 0) {return mid;}
        if(c < 0) {lo = mid + 1;} else {hi = mid - 1;}
    }
    return -1;
}

/* index of k or of its nearest ancestor in a sorted table, -1 if none (at most 64 parent steps) */
KOKKOS_INLINE_FUNCTION long hbt_key_find_lineage(const hbt_key_t *table, long n, hbt_key_t k)
{
    for(int step = 0; step <= 64; step++)
    {
        long j = hbt_key_find(table, n, k);
        if(j >= 0) {return j;}
        hbt_key_t p;
        if(!hbt_key_parent(k, &p)) {return -1;}
        k = p;
    }
    return -1;
}

/* number of keys equal to their predecessor in a sorted array (parallel_reduce over i) */
static inline long hbt_count_adjacent_duplicates(const hbt_key_t *sorted, long n)
{
    long ndup = 0;
    for(long i = 1; i < n; i++) {ndup += hbt_key_equal(sorted[i - 1], sorted[i]);}
    return ndup;
}

/* match[i] = table index of query[i] or of its nearest ancestor, -1 if none (parallel_for over i) */
static inline void hbt_resolve_lineage(const hbt_key_t *table, long ntable, const hbt_key_t *query, long nquery, long *match)
{
    for(long i = 0; i < nquery; i++) {match[i] = hbt_key_find_lineage(table, ntable, query[i]);}
}

#endif
