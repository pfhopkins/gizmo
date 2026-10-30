# Stella subgrid photoionisation across MPI ranks

One star with a fixed ionising rate Q = 1e49 photons/s at the centre of a uniform box, run with the Stella production
model (`GALSF_RESOLVEDISM_PHOTOION`, SGCHEM network 5) on 4 ranks.

## Setup

- 40 pc periodic box, n_H = 10 cm^-3, 10^4 K, Z = 0.1 Z_sun, 28^3 cells of 1 Msun (IC built by the test)
- `GALSF_RESOLVEDISM_ISOLATED_FB_TEST`: the cell with ID 1 becomes a 15 Msun star; `TestFixedQion` pins Q
- Stromgren radius ~15 pc, so the star's search sphere reaches tree nodes on every other rank
- `BufferSize 1` (MB), run to 24 kyr (~30 front solves, about a minute on 4 cores)

## What is tested

- The run reaches `TimeMax`. Each export entry of the front solve used to carry the star's full per-direction
  histograms (12 kB), so a 1 MB buffer held 42 entries and one star needed more: the run stopped with "buffer appears too
  small to hold a single particle". The histograms now return to the star's rank after the sweep, sparse.
- Photon budget: in every solve the recombination rate of the marked cells over Q (the code's ledger line) is within 3%
  of 1. Losing the remote part of the histograms gives 2.7-3.9.
- Shape: in the uniform medium the largest per-direction front is at most 1.2 times the smallest.
