# tree_node_momentum

Node bulk velocity against the particles, pure N-body, no KETJU. 128 hard binaries among 256 field stars in a Plummer sphere (timebins 51 and 55), built with `TEST_TREE_NODE_MOMENTUM_AUDIT` (mode 2 adds a moment refresh every third step).

A build reads every velocity as it stands, so momentum still in `P.dp` is already in the nodes; the next tree update must not add it again. Bar: largest node error below 1e-10 km/s (roundoff, the velocities are about 10 km/s).

Before (7ebc666df): 4.2e-4 (1 rank, build), 1.6e-4 (2 ranks, build), 4.2e-4 (1 rank, refresh). After: 1.9e-15 (1 rank), 8.7e-16 (2), 6.6e-16 (4), 3.7e-16 (refresh, 1 rank).
A refresh alone does not need the clear: `force_update_tree` has flushed all pending momentum by then.
