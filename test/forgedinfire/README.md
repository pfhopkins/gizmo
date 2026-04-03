# Forged in FIRE Nuclear-Scale AGN Disk Test

Short run of a nuclear-zoom AGN accretion disk simulation with full radiation-MHD, dust physics, and star formation (SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM). Based on a downsampled snapshot from a forged-in-FIRE m_x=0.03 simulation.

## Setup

- ~960k gas particles + 1 BH (PartType3) + ~770 star particles
- Non-cosmological, Box_Size=1 (code units: AU, Msun, km/s)
- Full radiation transport (5 bands, RT_SPEEDOFLIGHT_REDUCTION=1)
- Magnetic fields, dust temperature tracking
- Restart flag 2 (read snapshot as IC)

## What is tested

- Gas and star mass evolution compared to reference
- Radial profiles: density, gas/radiation/dust temperature, B-field, energy densities
- Gas density-temperature phase diagram
- Gas density histogram

## Files

- `forgedinfire_ics.hdf5`: Downsampled IC (~260 MB)
- `forgedinfire_exact.hdf5`: Reference final snapshot (set after initial run)
