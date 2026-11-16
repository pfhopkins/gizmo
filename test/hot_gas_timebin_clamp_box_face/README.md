# hot_gas_timebin_clamp_box_face

`RESOLVEDISM_HOTSYNC_CLAMP` in a periodic box: the sphere around the hot gas (T > 1e6 K) is the hottest cell plus the mean
nearest-image offset of the hot cells, with radius the largest nearest-image hot-cell distance plus 3 largest hot kernel
radii. With plain coordinates a remnant on a box face gets its centre half a box away and a radius that takes in the box.

Problem: the setup of `hot_gas_timebin_clamp` in a periodic 85.8 x 14.3 x 14.3 pc box (`BOX_LONG_X=6`, 60 x 10 x 10 cells of
1 Msun); the 10 Msun star sits 0.17 pc inside the +x face and dies 200 yr in, so its hot gas straddles the face. 2 ranks, 7.4 kyr.

Check, on the snapshots just after each coarsest sync (bars written before the runs): hot gas on both sides of the face;
fewer than 5 % of the cells beyond 1.25 sphere radii (nearest image) are on the hot gas's step (at most 2 x its longest step);
at least 98 % of the gas inside the sphere is on it (at most 2 x its shortest step, the bar of `hot_gas_timebin_clamp`).
