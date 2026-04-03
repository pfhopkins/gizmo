# FIRE Cosmological Galaxy Formation Test

Short cosmological FIRE simulation of a downsampled m11i-like dwarf galaxy at z~2.9. Tests the full FIRE physics pipeline including cooling, star formation, stellar feedback, and BH physics.

## Setup

- Downsampled (10%) m11i snapshot at z~2.9, particle masses increased 10x to conserve total mass
- FIRE_PHYSICS_DEFAULTS=3 with FIRE_BHS, periodic box with PMGRID=512
- Cosmological run from a=0.2564103 to a=0.257
- Uses restart flag 2 (read snapshot as IC for a new run)

## What is tested

- Dark matter mass conservation (exact)
- Gas, star, and BH total masses compared to reference (within 10%)
- Gas density vs temperature scatter plot
- Gas density histogram

## Files

- `fire_ics.hdf5`: Downsampled IC (snapshot_046 from m11i)
- `fire_reference.hdf5`: Reference final snapshot (set after initial run)
