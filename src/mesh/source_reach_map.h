#ifndef SOURCE_REACH_MAP_H
#define SOURCE_REACH_MAP_H
/* Source-reach map: exact routing of SYMMETRIC queries to remote top leaves.
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
 * used, a member whose radius changed other than by drift, or a member placed in the standing tree -- the node bound
 * applies instead, so the map never routes fewer pairs than the node bound would.
 *
 * Built after the top-leaf moment exchange of every tree build and moment refresh (one Alltoall + Alltoallv, build
 * cadence only).  Each rank keeps the lists for its own leaves U and a copy of those it sent for its own leaves T.
 * Host only; read-only during the walks. */

/* Collective.  After the top-leaf moment exchange of a tree build or a moment refresh; usable = 0 on a build that
 * covers only part of the particles (the map then stays off until the next full build). */
void source_reach_map_build(int usable);
/* At the start of a tree build: until rebuilt, symmetric exports use the node bound. */
void source_reach_map_invalidate(void);
/* Bring the map to the current time; outside parallel regions, before a walk that uses it. */
void source_reach_map_prepare(void);

/* The query's own top leaf for the map: the leaf of this rank whose cube at the build holds pos, or -1 (no verdict). */
int  source_reach_map_query_leaf(const double pos[3]);
/* Verdict on exporting query (pos, h_q) to remote top leaf `leaf` (node `leaf_node`), for a loop with neighbour types
 * `type_mask` and j-side reach scale j_reach_scale: 1 export, 0 no member of the leaf can pair with it, -1 no verdict
 * (the caller keeps the node bound).  Thread-safe. */
int  source_reach_map_verdict(int query_leaf, int leaf, int leaf_node, const double pos[3], double h_q,
                              unsigned int type_mask, double j_reach_scale);

/* Post-density update (force_update_hmax): member i of this rank's top leaf `leaf` now has reach `reach`.  If the map
 * no longer covers it, the leaf is marked stale and 1 is returned, so the caller ships the leaf to the other ranks.
 * Thread-safe. */
int  source_reach_map_note_member(int i, int leaf, double reach);
/* A leaf another rank marked stale, or one holding a particle placed in the standing tree: the node bound applies to
 * it until the next build. */
void source_reach_map_mark_stale(int leaf);
int  source_reach_map_is_stale(int leaf);

#endif /* SOURCE_REACH_MAP_H */
