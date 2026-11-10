# Deliberate differences from the SGChem Fortran reference

- CVODE (BDF, dense, difference-quotient Jacobian) replaces DVODE; a failed substep restores the state, halves dt and re-initialises instead of aborting.
- The outer substep loop, safety factors, tolerances (RTOL 1e-4, per-species ATOL, energy ATOL 0) and the TNOCHEM hot bypass follow evolve_abundances.F.
- All per-cell COMMON state lives in an explicit ChemCell struct; solver workspaces are thread_local, so OpenMP threads share nothing mutable.
- validate_input acts on a local copy of y; flagged input returns zero ydot and a recoverable code instead of changing the integrator state.
- Fortran ABORT/stop paths return error codes (100 for dt < 1e-8 s, 1 for an exhausted substep budget, 2 for a bad dt) that the caller handles.
- calc_photo follows the per-network branches of calc_photo.F; network 15 fills only phrates 48-51 beyond the shared set.
- Physical constants are the float32-rounded values the Fortran uses (default REAL literals); other literals are kept as in the .F source.
- lambda and lambda_chem are recomputed with one extra RHS evaluation at the accepted state (CHEMCXX_LEGACY_LAMBDA=1 restores the Fortran "last RHS evaluation wins").
- Photorates, G_dust, chi_isrf and fshield_CO are computed once per cell call; H2/HD self-shielding and the Doppler b are recomputed at every RHS evaluation.
- Table splines for cl52, cl59, cl60, cl61 are built directly from coolinmo.F:1028-1071; the dust temperature Newton iteration is bounded.
- Network 18 (non-equilibrium metal ions) and network 20 (TIGRESS-NCR thermochemistry) exist only in C++; network 5 additionally handles ionising photons on H2.
- Diagnostic and test-only Fortran branches (THERMAL_INFO_DUMP, COLLECT_EQB_DATA, FF_COLLAPSE_TEST, KOYAMA) are not carried over.
