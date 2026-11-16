# stella_hotsync_periodic

`RESOLVEDISM_HOTSYNC_CLAMP` holds every cell near hot gas (T > 1e6 K) on the hot gas's fine step, so that pair fluxes
across a remnant's edge are applied by both sides at the same cadence. The cells it holds are those inside a sphere:
the hot cells' centre, plus the largest hot-cell distance from it, plus three largest hot kernel radii. In a periodic
box both the centre and the distances must be taken with the nearest image, or a remnant on a box face gets a sphere
centred half a box away.

## Setup

- Stella production model (`Config_stella_default.sh`) with `GALSF_RESOLVEDISM_ISOLATED_FB_TEST` and
  `RESOLVEDISM_HOTSYNC_CLAMP`; periodic box 80 x 10 x 10 pc (`BOX_LONG_X=8`), gravity not periodic.
- 1 Msun cells, n_H = 50 cm^-3, 1e4 K, 0.1 Zsun; the cell on the last x-plane nearest the +x face centre becomes a
  12 Msun star, forced to explode at 2.1e-6 (~2 kyr), so the remnant straddles the periodic x-boundary.

## What is tested

- The run reaches `TimeMax`.
- In every snapshot with hot gas, the clamp sphere is rebuilt from the snapshot with nearest-image distances; fewer
  than 5% of the cells beyond 1.25 times its radius may sit on the hot gas's step (they keep their own long step).
