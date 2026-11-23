// Unit test: identities of particles created from a parent (src/galaxy_sf/single_star_identity.h) on a struct with the Stella identity
// fields (ID_wraps, ID_lineage). A star's random-number key is (ID, child number, generation, type) plus single_star_identity_extra,
// so two particles with the same tuple draw the same streams. Bars: no repeated tuple among all stars ever spawned and all live
// cells in (a) 130 spawns from one cell and its split child with the heavier child merged back (stella_dev driver), (b) a cell
// that splits and then both cells spawn past a roll-over (review round 2, S2), (c) 4000 random spawns, splits and merges; and
// the first 31 stars of a cell keep extra = 0 (streams without a roll-over unchanged).
#include "test_harness.h"
#include "src/galaxy_sf/single_star_identity.h"
#include <cstdint>
#include <cstdio>
#include <set>
#include <tuple>
#include <vector>

struct Cell {
    uint64_t ID, ID_child_number;
    int ID_generation, ID_wraps;
    uint64_t ID_lineage;
    int Type;
};
typedef std::tuple<uint64_t, uint64_t, int, int, uint64_t> Ident;
static Ident ident(const Cell &c) {return Ident(c.ID, c.ID_child_number, c.ID_generation, c.Type, single_star_identity_extra(c));}

static Cell spawn(Cell &p) {Cell s = p; single_star_identity_from_parent(p, s); s.Type = 4; return s;}
static Cell split(Cell &p) {Cell s = p; single_star_identity_from_parent(p, s); return s;}

static int repeats(const std::vector<Cell> &all)
{
    std::set<Ident> seen; int n = 0;
    for(const Cell &c : all) {if(!seen.insert(ident(c)).second) {n++;}}
    return n;
}

TEST_CASE("130 spawns from a cell and its split child, heavier child merged back")
{
    Cell R = {7, 0, 0, 0, 0, 0}, X = {};
    std::vector<Cell> stars; int have_X = 0;
    for(int k = 0; k < 130; k++) {
        if(k == 20) {X = split(R); have_X = 1;}
        if(k == 100 && have_X) {single_star_identity_merge(X, R, 1); have_X = 0;} /* X (heavier) merged into R: R keeps X's identity */
        Cell &p = (have_X && (k % 2)) ? X : R;
        stars.push_back(spawn(p));
    }
    std::vector<Cell> all = stars; all.push_back(R);
    int n = repeats(all);
    printf("  (a) repeated identities: %d of %zu\n", n, all.size());
    CHECK(n == 0);
    int pre = 0; for(int k = 0; k < 20; k++) {pre += (single_star_identity_extra(stars[k]) == 0);}
    CHECK(pre == 20);
}

TEST_CASE("a split cell and its parent both spawn past a roll-over")
{
    Cell R = {11, 0, 0, 0, 0, 0};
    Cell X = split(R);
    std::vector<Cell> all;
    for(int k = 0; k < 32; k++) {all.push_back(spawn(R));}
    for(int k = 0; k < 32; k++) {all.push_back(spawn(X));}
    all.push_back(R); all.push_back(X);
    int n = repeats(all);
    printf("  (b) repeated identities: %d of %zu\n", n, all.size());
    CHECK(n == 0);
}

TEST_CASE("the first 31 stars of a cell keep their streams")
{
    Cell R = {13, 0, 0, 0, 0, 0}; int zero = 0;
    for(int k = 0; k < 31; k++) {Cell s = spawn(R); zero += (single_star_identity_extra(s) == 0);}
    CHECK(zero == 31);
}

TEST_CASE("4000 random spawns, splits and merges")
{
    std::vector<Cell> cells = {{17, 0, 0, 0, 0, 0}}, made;
    uint64_t r = 12345;
    auto rnd = [&](uint64_t n) {r = r * 6364136223846793005ULL + 1442695040888963407ULL; return (r >> 33) % n;};
    for(int k = 0; k < 4000; k++) {
        size_t a = rnd(cells.size());
        uint64_t u = rnd(100);
        if(u < 10) {cells.push_back(split(cells[a]));}
        else if(u < 14 && cells.size() > 1) {
            size_t b = rnd(cells.size()); if(b == a) {continue;}
            single_star_identity_merge(cells[a], cells[b], (int)rnd(2)); cells.erase(cells.begin() + a);
        }
        else {made.push_back(spawn(cells[a]));}
    }
    std::vector<Cell> all = made; all.insert(all.end(), cells.begin(), cells.end());
    int n = repeats(all);
    printf("  (c) repeated identities: %d of %zu (%zu cells)\n", n, all.size(), cells.size());
    CHECK(n == 0);
}

TEST_MAIN()
