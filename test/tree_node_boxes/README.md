# tree_node_boxes

A particle whose position is set rather than drifted must stay inside every node box above it. Node boxes keep their centre and grow only with the motion bound of their members, so a sink moved onto the potential minimum of its neighbours (`SINK_REPOSITION_ON_POTMIN`) can sit outside the boxes above it until the next tree build, where an opening test or neighbour search can miss it.

Setup: a 1e-3 Msun sink 1.2e-2 pc off the centre of a cold uniform sphere (r = 0.02 pc, 4224 cells of 7.8e-5 Msun), STARFORGE defaults. A 0.1 Msun star 3.5e-3 pc nearer the sink is the potential minimum the sink moves onto (gas is excluded from that search). Built with `TREE_INTEGRITY_AUDITS`; the geometry audit totals in `cpu.txt` count the audits that found a particle outside a box above it.

Bar: the sink moves toward the star, and no audit finds a particle outside a box.
