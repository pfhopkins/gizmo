# Forged in FIRE Nuclear-Scale AGN Disk Test

Short run of a nuclear-zoom AGN accretion disk simulation with full radiation-MHD, dust physics, and star formation (SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM). Based on a downsampled snapshot from a forged-in-FIRE m_x=0.03 simulation.

## Setup

- ~960k gas particles + 1 BH (PartType3) + ~770 star particles
- Non-cosmological, Box_Size=1 (code units: AU, Msun, km/s)
- Full radiation transport (5 bands, RT_SPEEDOFLIGHT_REDUCTION=1)
- Magnetic fields, dust temperature tracking
- Restart flag 2 (read snapshot as IC)
- Zoom refinement on: the cells here are well above their target mass, so the run roughly doubles its element count (962637 -> 1929215 gas) and is still refining at the end. The per-rank particle capacity is raised to follow it; the IC's own sizing holds about a quarter of what is wanted.
- Runs 20 code time units (92800 -> 92820), 823 steps at the RT light-crossing timestep, ~25 min at 8 ranks x 2 threads

## What is tested

- That refinement can outgrow the capacity the IC was read at, without splits being refused
- Gas and star mass evolution compared to reference
- Radial profiles: density, gas/radiation/dust temperature, B-field, energy densities
- Gas density-temperature phase diagram
- Gas density histogram

## Files

- `forgedinfire_ics.hdf5`: Downsampled IC (~260 MB)
- `forgedinfire_exact.hdf5`: Reference final snapshot (set after initial run)
