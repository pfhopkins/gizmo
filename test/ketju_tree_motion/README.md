# ketju_tree_motion

The tree's nodes after KETJU changes velocities outside the kick. 256 hard circular binaries in a Plummer sphere, FX0 parameters
(k2b of the KETJU test stack, `make_ketju_tree_motion_ics.py`), built with `TEST_KETJU_TREE_MOTION_AUDIT` (stella/ketju/ketju_tree_audit.cc).

Checks, 1 rank: chain members under every ancestor's `vmax` and inside its widened box; each node's bulk velocity equals its
members, also across mergers (pairs set to collide). 2 ranks: the same bounds on all copies of the top-level nodes; imported single-star leaves
within `LEAF_MEAN_MAX` of a real star (the node velocity is not asserted there: about 6e-2 remains in a few nodes, not yet explained).
