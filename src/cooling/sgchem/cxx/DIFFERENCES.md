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

## Network 17 and oxygen

Network 17 tracks no oxygen species and has no O–H2 reactions, as in the Fortran: atomic O = abundo − CO only feeds the NL97 CO formation and O I cooling.

Ploeckinger et al. (arXiv:2506.15773, arXiv:2512.13447) find that O + H2 → OH + H and the OH/H2O chain lower the H2 fraction by up to 1 dex. Their result is for gas held at a fixed temperature in chemical equilibrium. We tested it in one zone, with net 17 plus an oxygen block (OH, H2O, O2 time-dependent; O+, OH+, H2O+, H3O+ in equilibrium; rates from net 3):

- At 0.01–0.3 Zsun in thermal equilibrium the HI→H2 transition moves by ≤0.01–0.04 dex. The transition sits at 15–180 K, where O + H2 → OH + H (barrier ~3150 K) does not run.
- The barrier-free ion chain (H+ + O → O+, then O+/OH+/H2O+ + H2 → … → H3O+) moves it by +0.15 dex, only at solar Z under a Milky Way cosmic-ray rate (3e-17 s^-1, N_H = 1e21 cm^-2). It scales with x(H+) and vanishes under weak fields.
- The large effect needs warm gas (1000–3000 K) kept in chemical equilibrium for ≳100 Myr. After 10 Myr it is ≤0.02 dex for Z ≤ 0.1 and n ≤ 100 cm^-3.

If it is ever needed, the ion chain can be added as four algebraic species.
