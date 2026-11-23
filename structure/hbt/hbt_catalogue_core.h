#ifndef HBT_CATALOGUE_CORE_H
#define HBT_CATALOGUE_CORE_H
/* Group-catalogue bookkeeping of the keyed catalogue: per-type offsets of groups into per-type member lists (as
   GADGET-4's GroupOffsetType into its group-ordered snapshot). Flat arrays, no STL, no MPI; loops map onto
   Kokkos::parallel_scan (offsets, per type) and parallel_reduce (totals, counts). */

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#define HBT_NTYPES 6

/* offsettype[g*6+t] = base[t] + sum_{h<g} lentype[h*6+t] */
static inline void hbt_group_type_offsets(const int *lentype, long ngroups, const long long *base, long long *offsettype)
{
    for(int t = 0; t < HBT_NTYPES; t++)
    {
        long long run = base ? base[t] : 0;
        for(long g = 0; g < ngroups; g++) {offsettype[g * HBT_NTYPES + t] = run; run += lentype[g * HBT_NTYPES + t];}
    }
}

/* tot[t] = sum_g lentype[g*6+t] */
static inline void hbt_group_type_totals(const int *lentype, long ngroups, long long *tot)
{
    for(int t = 0; t < HBT_NTYPES; t++) {tot[t] = 0;}
    for(long g = 0; g < ngroups; g++) {for(int t = 0; t < HBT_NTYPES; t++) {tot[t] += lentype[g * HBT_NTYPES + t];}}
}

/* tot[t] = number of members with type t */
static inline void hbt_count_members_by_type(const int *type, long n, long long *tot)
{
    for(int t = 0; t < HBT_NTYPES; t++) {tot[t] = 0;}
    for(long i = 0; i < n; i++) {if(type[i] >= 0 && type[i] < HBT_NTYPES) {tot[type[i]]++;}}
}

#endif
