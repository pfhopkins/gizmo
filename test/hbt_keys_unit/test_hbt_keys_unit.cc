/* Unit test (T1) for structure/hbt/hbt_key.h and hbt_key_mpi.cc: keys made with GIZMO's split/spawn rules
   (galaxy_sf/single_star_identity.h, used by mesh/merge_split.cc and galaxy_sf/sfr_eff.cc) are unique, the top-bit rule recovers every parent,
   lineage lookup passes a parent's label to all descendants, and the duplicate counter finds injected collisions
   on any number of MPI ranks. Exit 0 only if every check passes. */
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include "structure/hbt/hbt_key.h"
#include "structure/hbt/hbt_key_mpi.h"

struct sim_particle {hbt_key_t key; int gen; long parent; int is_star;};

static uint64_t rng_state = 12345;
static uint64_t rng_next() {rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17; return rng_state;}

/* GIZMO's rules (a star takes the parent's incremented generation, sfr_eff.cc Stella and non-Stella paths);
   star_gen_zero emulates the pre-STL-148 Stella rule (star born with generation 0); returns the newborn's index */
static long spawn(std::vector<sim_particle> &p, long i, int as_star, int star_gen_zero = 0)
{
    sim_particle c; c.key = hbt_key_make(p[i].key.id, p[i].key.child + (((uint64_t)1) << p[i].gen));
    p[i].gen++; if(p[i].gen > 30) {p[i].gen = 0;}
    c.gen = (as_star && star_gen_zero) ? 0 : p[i].gen; c.parent = i; c.is_star = as_star;
    p.push_back(c); return (long)p.size() - 1;
}

static void sort_keys(std::vector<hbt_key_t> &k) {std::sort(k.begin(), k.end(), [](const hbt_key_t &a, const hbt_key_t &b) {return hbt_key_cmp(a, b) < 0;});}

static int nfail = 0;
static void check(int ok, const char *what, long long a, long long b)
{
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {printf("%s %s (%lld vs %lld)\n", ok ? "PASS" : "FAIL", what, a, b);}
    if(!ok) {nfail++;}
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, ntask; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ntask);

    /* (a) genealogy without rollover: 20000 originals, 200000 split/spawn events; stars never spawn (as in Stella) */
    const long n0 = 20000, nev = 200000;
    std::vector<sim_particle> p;
    for(long i = 0; i < n0; i++) {sim_particle s; s.key = hbt_key_make(1000 + 7 * i, 0); s.gen = 0; s.parent = -1; s.is_star = 0; p.push_back(s);}
    long done = 0;
    while(done < nev)
    {
        long i = (long)(rng_next() % p.size());
        if(p[i].is_star || p[i].gen >= 30) {continue;}
        spawn(p, i, (rng_next() % 4) == 0); done++;
    }
    std::vector<hbt_key_t> all(p.size());
    for(size_t i = 0; i < p.size(); i++) {all[i] = p[i].key;}
    std::vector<hbt_key_t> srt(all); sort_keys(srt);
    check(hbt_count_adjacent_duplicates(srt.data(), (long)srt.size()) == 0, "(a) keys unique", hbt_count_adjacent_duplicates(srt.data(), (long)srt.size()), 0);
    long nparent_ok = 0, nchild = 0, npack_ok = 0, nroot_ok = 0;
    std::vector<hbt_key_t> roots(all.begin(), all.begin() + n0); sort_keys(roots);
    for(size_t i = 0; i < p.size(); i++)
    {
        hbt_key_t par; int has = hbt_key_parent(p[i].key, &par);
        if(p[i].parent >= 0) {nchild++; if(has && hbt_key_equal(par, p[p[i].parent].key)) {nparent_ok++;}}
        long r = (long)i; while(p[r].parent >= 0) {r = p[r].parent;}
        long j = hbt_key_find_lineage(roots.data(), (long)roots.size(), p[i].key);
        if(j >= 0 && hbt_key_equal(roots[j], p[r].key)) {nroot_ok++;}
        uint64_t pk; if(hbt_key_pack64(p[i].key, &pk) && hbt_key_equal(hbt_key_unpack64(pk), p[i].key)) {npack_ok++;}
    }
    check(nparent_ok == nchild, "(a) top-bit rule recovers every parent", nparent_ok, nchild);
    check(nroot_ok == (long)p.size(), "(a) lineage lookup reaches the original ancestor", nroot_ok, (long)p.size());
    check(npack_ok == (long)p.size(), "(a) pack64/unpack64 round trip", npack_ok, (long)p.size());
    uint64_t pk;
    hbt_key_t edge = hbt_key_make((((uint64_t)1) << 31) - 1, (((uint64_t)1) << 32) - 1);
    check(hbt_key_pack64(edge, &pk) && hbt_key_equal(hbt_key_unpack64(pk), edge) && pk < (((uint64_t)1) << 63), "(a) pack64 at the limits keeps the top bit free", 0, 0);
    check(!hbt_key_pack64(hbt_key_make(((uint64_t)1) << 31, 0), &pk) && !hbt_key_pack64(hbt_key_make(1, ((uint64_t)1) << 32), &pk), "(a) pack64 refuses ID >= 2^31 and child >= 2^32", 0, 0);

    /* (b) synthetic split of 10% of the particles: labels pass through lineage unchanged */
    std::vector<int> label_old(p.size());
    for(size_t i = 0; i < p.size(); i++) {label_old[i] = (int)(rng_next() % 1000);}
    std::vector<hbt_key_t> table(all); std::vector<long> order(all.size());
    for(size_t i = 0; i < order.size(); i++) {order[i] = (long)i;}
    std::sort(order.begin(), order.end(), [&](long a, long b) {return hbt_key_cmp(all[a], all[b]) < 0;});
    for(size_t i = 0; i < order.size(); i++) {table[i] = all[order[i]];}
    const long nold = (long)p.size(); long nsplit = 0;
    for(long i = 0; i < nold; i++) {if(!p[i].is_star && p[i].gen < 30 && (rng_next() % 10) == 0) {long c = spawn(p, i, 0); if((rng_next() % 3) == 0 && p[c].gen < 30) {spawn(p, c, 0);} nsplit++;}}
    std::vector<hbt_key_t> q(p.size()); for(size_t i = 0; i < p.size(); i++) {q[i] = p[i].key;}
    std::vector<long> match(q.size());
    hbt_resolve_lineage(table.data(), (long)table.size(), q.data(), (long)q.size(), match.data());
    long nlabel_ok = 0;
    for(size_t i = 0; i < q.size(); i++)
    {
        long r = (long)i; while(r >= nold) {r = p[r].parent;}
        if(match[i] >= 0 && label_old[order[match[i]]] == label_old[r]) {nlabel_ok++;}
    }
    std::vector<hbt_key_t> srt2(q); sort_keys(srt2);
    check(hbt_count_adjacent_duplicates(srt2.data(), (long)srt2.size()) == 0, "(b) keys unique after 10% synthetic split", hbt_count_adjacent_duplicates(srt2.data(), (long)srt2.size()), 0);
    check(nlabel_ok == (long)q.size() && nsplit > nold / 20, "(b) every particle inherits its pre-split label", nlabel_ok, (long)q.size());

    /* (c) negative controls: the duplicate check must fire */
    {
        std::vector<sim_particle> r; sim_particle s; s.key = hbt_key_make(5, 0); s.gen = 0; s.parent = -1; s.is_star = 0; r.push_back(s);
        for(int k = 0; k < 32; k++) {spawn(r, 0, 0);}   /* 32 splits of one cell: generation rolls over after 31 */
        std::vector<hbt_key_t> k(r.size()); for(size_t i = 0; i < r.size(); i++) {k[i] = r[i].key;} sort_keys(k);
        check(hbt_count_adjacent_duplicates(k.data(), (long)k.size()) > 0, "(c) rollover after 31 splits yields a duplicate that is detected", hbt_count_adjacent_duplicates(k.data(), (long)k.size()), 1);
    }
    {
        std::vector<sim_particle> r; sim_particle s; s.key = hbt_key_make(9, 0); s.gen = 0; s.parent = -1; s.is_star = 0; r.push_back(s);
        long a = spawn(r, 0, 0); long st = spawn(r, 0, 1, 1); spawn(r, a, 0); spawn(r, st, 0);   /* old rule: a generation-0 star that spawned */
        std::vector<hbt_key_t> k(r.size()); for(size_t i = 0; i < r.size(); i++) {k[i] = r[i].key;} sort_keys(k);
        check(hbt_count_adjacent_duplicates(k.data(), (long)k.size()) > 0, "(c) old rule: a spawn from a generation-0 star collides and is detected", hbt_count_adjacent_duplicates(k.data(), (long)k.size()), 1);
    }
    {
        /* STL-148 rule: stars spawn and their children spawn again, many times; keys stay unique and parents are recovered */
        std::vector<sim_particle> r; sim_particle s; s.key = hbt_key_make(9, 0); s.gen = 0; s.parent = -1; s.is_star = 0; r.push_back(s);
        long a = spawn(r, 0, 0); long st = spawn(r, 0, 1); spawn(r, a, 0); spawn(r, st, 0);
        for(int e = 0; e < 20000; e++) {long i = (long)(rng_next() % r.size()); if(r[i].gen < 30) {spawn(r, i, (rng_next() % 2) == 0);}}
        std::vector<hbt_key_t> k(r.size()); long npar = 0, nch = 0;
        for(size_t i = 0; i < r.size(); i++) {k[i] = r[i].key; hbt_key_t q; if(r[i].parent >= 0) {nch++; if(hbt_key_parent(r[i].key, &q) && hbt_key_equal(q, r[r[i].parent].key)) {npar++;}}}
        sort_keys(k);
        check(hbt_count_adjacent_duplicates(k.data(), (long)k.size()) == 0, "(c) STL-148 rule: spawns from stars keep keys unique", hbt_count_adjacent_duplicates(k.data(), (long)k.size()), 0);
        check(npar == nch, "(c) STL-148 rule: parent recovered for every spawn from a star", npar, nch);
    }

    /* (d) distributed count: keys of (a) split round-robin over ranks, then injected duplicates on other ranks */
    std::vector<hbt_key_t> mine;
    for(size_t i = 0; i < all.size(); i++) {if((long)(i % ntask) == rank) {mine.push_back(all[i]);}}
    long long nd = hbt_count_duplicate_keys(mine.data(), (long)mine.size(), MPI_COMM_WORLD);
    check(nd == 0, "(d) MPI count on unique keys", nd, 0);
    const int ninj = 7;
    for(int k = 0; k < ninj; k++) {if(((k + 1) % ntask) == rank) {mine.push_back(all[(size_t)k * 97]);}}  /* original sits on rank (k*97)%ntask */
    if(rank == ntask - 1) {mine.push_back(all[5]); mine.push_back(all[5]);}                                  /* one key three times */
    nd = hbt_count_duplicate_keys(mine.data(), (long)mine.size(), MPI_COMM_WORLD);
    check(nd == ninj + 2, "(d) MPI count finds injected cross-rank duplicates", nd, ninj + 2);
    nd = hbt_count_duplicate_keys(NULL, 0, MPI_COMM_WORLD);
    check(nd == 0, "(d) MPI count with no keys", nd, 0);

    if(rank == 0) {printf("ranks=%d particles=%zu failures=%d\n", ntask, p.size(), nfail);}
    MPI_Finalize();
    return nfail ? 1 : 0;
}
