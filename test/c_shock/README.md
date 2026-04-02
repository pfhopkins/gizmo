# C-type Shock Test

1D C-type shock test with non-ideal MHD (ambipolar diffusion dominated). This tests the propagation and structure of a continuous (C-type) shock driven by ion-neutral drift in a weakly-ionized medium.

## Setup

- 3000 particles in a 1D periodic box (BoxSize=6e7)
- Uniform density (rho=1), enforced adiabat with coefficient 0.01, gamma=5/3
- Initial Bx=By=1/sqrt(2), counter-propagating velocity perturbation
- Non-ideal MHD coefficients: eta_ad = 1e5 * vA^2, eta_ohmic = 1e-5 * eta_ad, eta_hall = 0

## What is tested

The shock propagates and develops the characteristic smooth C-shock profile (as opposed to a sharp J-shock) due to ambipolar diffusion. The test compares the final snapshot against a reference numerical solution for Density, InternalEnergy, Velocities, and MagneticField.

## Non-ideal MHD coefficient handling

Same approach as hall_wave: the test temporarily patches eos/eos.cc to inject the problem-specific coefficients before building, then restores the original file.
