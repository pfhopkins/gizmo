#ifndef SOURCE_REACH_MAP_H
#define SOURCE_REACH_MAP_H
/* Source-reach map: exact routing of SYMMETRIC queries to remote top leaves, built only when it pays.
 *
 * A symmetric pair (q, j) is a neighbour pair when r < max(h_q, s*h_j), i.e. when r < h_q OR r < s*h_j.  The node bound
 * routes a query to a remote top leaf T when the query is within max(h_q, band_T) of T's box, band_T being the largest
 * reach of any member of T.  That pairs one member's position with another member's reach: a few far-reaching members
 * of a large outer leaf then attract every query in the run, although no member can pair with them.
 *
 * The map keeps the two halves apart.  r < h_q is tested exactly as before, with the query's own radius against T's
 * box.  For r < s*h_j, the owner of T lists, at each tree build, every remote top leaf U that a member of T may reach
 * before the tree is rebuilt, together with a box of those members' positions and their largest reach per type; the
 * list for U goes to U's owner.  A query then looks up the top leaf U whose cube holds it and asks U's list whether
 * any member of T can reach it.  Members whose sphere, grown and moved by the most the tree's lifetime allows, does not
 * reach U's cube cannot pair with any query in U while those bounds hold; the others are inside the listed box, widened
 * by their motion since the build, with at most the listed reach, grown since the build.  Whenever a bound the
 * argument rests on is exceeded -- more motion or growth than the build allowed for, a larger reach scale than the build
 * used, a member whose radius changed other than by drift (an O(1) test against its radius at the build), or a member
 * placed in the standing tree -- the node bound
 * applies instead, so the map never routes fewer pairs than the node bound would.
 *
 * Measured-cost gate.  The map is built only when it is wanted.  Each symmetric corridor discovery measures its own
 * imbalance (slowest minus fastest rank), summed over the current tree's life; the map turns on when that exceeds the
 * map's measured build time (before its first build, the measured tree build time).  A mapped tree that did not save
 * more than the map's build cost turns it off, and it is retried only above the imbalance that failed to pay.  While it
 * is not wanted a tree build builds nothing and frees everything the map owns.  When wanted it is built after the
 * top-leaf moment exchange of a tree build or moment refresh (one Alltoall + Alltoallv), or at the start of a symmetric
 * discovery when no time has passed since the last tree build.  Every decision is rank-uniform and routing stays exact
 * whatever the gate decides.  Each rank keeps the lists for its own leaves U.  Host only; read-only during the walks. */

/* Collective.  After the top-leaf moment exchange of a tree build or a moment refresh; usable = 0 on a build that
 * covers only part of the particles (the map then stays off until the next full build). */
void source_reach_map_build(int usable);
/* At the start of a tree build: until rebuilt, symmetric exports use the node bound. */
void source_reach_map_invalidate(void);
/* At the start of a symmetric discovery: build the map now if the gate wants it, it is not built, and no time has passed
 * since the last tree build.  Collective; the condition is rank-uniform. */
void source_reach_map_build_if_due(void);
/* The gate's measurements ride the final receiver vote of a corridor discovery (MPI_MIN over SOURCE_REACH_GATE_NVOTE long longs, times in microseconds):
 * fill before it with this discovery's own time, apply after it.  Rank-uniform decisions. */
enum {SOURCE_REACH_GATE_NVOTE = 6, SOURCE_REACH_GATE_STOPPED = 4};
void source_reach_map_gate_fill(long long *vote, double t_discovery, int stopped, long long n_queries);
/* Cold-call probe: the receiver-walk time after which a symmetric discovery is stopped, the map built and the discovery redone;
 * -1 = do not probe (a map exists, none could be built now, or one already proved not worth it).  Rank-local value, rank-uniform sign. */
double source_reach_map_cold_budget(void);
void source_reach_map_gate_apply(const long long *vote_all, int symmetric);
/* Bring the map to the current time; outside parallel regions, before a walk that uses it. */
void source_reach_map_prepare(void);

/* The query's own top leaf for the map: the leaf of this rank whose cube at the build holds pos, or -1 (no verdict). */
int  source_reach_map_query_leaf(const double pos[3]);
/* Verdict on exporting query (pos, h_q) to remote top leaf `leaf` (node `leaf_node`), for a loop with neighbour types
 * `type_mask` and j-side reach scale j_reach_scale: 1 export, 0 no member of the leaf can pair with it, -1 no verdict
 * (the caller keeps the node bound).  Thread-safe. */
int  source_reach_map_verdict(int query_leaf, int leaf, int leaf_node, const double pos[3], double h_q,
                              unsigned int type_mask, double j_reach_scale);

/* Post-density update (force_update_hmax): particle i now has reach `reach` and grows at `growth_rate` (0 if its radius does
 * not drift with the divergence).  1 if the map still covers it -- its radius at the build, grown at that rate since, bounds
 * the new one -- or nothing relies on the map; 0 if its leaf must be marked stale.  O(1), thread-safe. */
int  source_reach_map_member_covered(int i, double reach, double growth_rate);
/* The node bound applies to `leaf` until the next build (a member the map no longer covers, a particle placed in the
 * standing tree, or another rank's flag).  Returns 1 if the leaf was not already stale.  Thread-safe. */
int  source_reach_map_mark_stale(int leaf);
/* A particle moved between slots of the standing tree (rearrange_particle_sequence through swap_treewalk_pointers), or left it. */
void source_reach_map_swap_slots(int i, int j);
void source_reach_map_clear_slot(int i);
int  source_reach_map_is_stale(int leaf);

#endif /* SOURCE_REACH_MAP_H */
