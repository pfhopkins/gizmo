# Hall MHD Wave Test

1D whistler/ion-cyclotron wave propagation test with non-ideal MHD (Hall effect, Ohmic resistivity). Based on the test problem in Berlok & Pfrommer (2023), arXiv:2309.15907.

## Setup

- 160 particles in a 1D periodic box (BoxSize=1)
- Uniform density (rho=1), internal energy (u=1.5), gamma=5/3
- Background magnetic field with small perturbation
- Fixed non-ideal MHD coefficients: eta_hall=0.01, eta_ohmic=0.0002, eta_ad=0

## What is tested

The wave propagates for one period (TimeMax=250) and should return to the initial conditions. The test compares initial and final snapshots for Density, Velocities, InternalEnergy, and MagneticField.

## Non-ideal MHD coefficient handling

This test requires fixed (constant) non-ideal MHD coefficients rather than the self-consistent calculation in `eos/eos.cc`. The test script temporarily patches the source to inject fixed values before building, then restores the original file. No permanent modifications are made to the source tree.
