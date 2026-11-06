# timeline_restart_tree

As `test/timeline_restart`, with a self-gravitating gas sphere on 2 ranks, so the tree in the restart files is in use from the first
step. A run stopped at step 8 restarts with TimeMax doubled and must reach the end where a plain restart at the same step ends.
DomainBuild_ActiveFraction and TreeRebuild_ActiveFraction above one: no rebuild at the restart, so the first drifts run on the reloaded tree.
