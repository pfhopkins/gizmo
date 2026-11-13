# tree_node_boxes

A particle whose position is set rather than drifted must stay inside every node box above it. Node boxes keep their centre and grow only with the motion bound of their members, so a cell spawned a step away from its parent, or a sink re-centred on the mass it accreted, can sit outside the boxes above it until the next tree build. An opening test or neighbour search can then miss it.

Setup: a 0.5 Msun sink at the centre of a cold uniform sphere (r = 0.02 pc, 4224 cells of 7.8e-5 Msun), STARFORGE defaults with jets along z. Accretion moves the sink (SINK_FOLLOW_ACCRETED_COM), and jet cells are spawned into the standing tree (MAINTAIN_TREE_IN_REARRANGE). The sink softening (its accretion radius) is 1e-3 pc so that gas reaches it within the 8 kyr run. Built with `TEST_TREE_NODE_BOX_AUDIT`, which counts after every tree update without a rebuild the particles outside a box above them, each box taken at the length the walk uses (stored length plus motion-bound growth to now).

Bar: the sink accretes and spawns jet cells, and the count of particles outside a box is zero.
