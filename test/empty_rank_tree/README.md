# empty_rank_tree

Three stars on two ranks (problem 0 of the fewbody set). A later domain step puts all of them on one rank; the other rank's tree build then finds no Father mirror (`father_mirror_ensure_(0)` returns NULL once the pool has been freed with the tree), `gpu_moment_refresh` returns 1 with no message, and the other rank aborts in `let_finalize_unredirected_foreign_topleaves` on NaN top-leaf moments.

Bar: the run reaches TimeMax. Before: aborts at t = 0.0053. A rank empty from the first build fails elsewhere (topology emit, rc 3), not covered here.
