# Shu 1977 test

This is the Shu 1977 singular isothermal sphere setup with initial density 2x the critical density. The physically-correct behavior is for the central region to undergo runaway collapse into exactly one sink (by symmetry). The test only runs for a short time, just checking that exactly 1 sink does form.

Compile-time flags used for this setup:
```
SINGLE_STAR_STARFORGE_DEFAULTS
EOS_ENFORCE_ADIABAT=4e4
EOS_GAMMA=1.001
```
