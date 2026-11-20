# ketju_pm

KETJU region members under PMGRID. A 1 + 1 Msun, 200 AU pair in an 800 AU region and a 20 Msun star 20000 AU away, in a periodic
2e5 AU box with PMGRID = 128 (r_s = 1953 AU): the perturber's pull lives entirely in the mesh, about 13 cells away. A quarter outer
orbit against an open-box DOP853 three-body integration.

Asserted: pair CoM relative to the perturber within 0.05 R_out, pair semi-major axis within 1e-2.
Comoving variant (a = 0.5, 2 ranks, 2 regions): two binaries and their mirror single stars; CoM mirror symmetry within 1e-3 R_out,
traces close.

Energy: the step trace's own bars (mass 1e-12, exact stages, integration residual 1e-2 of E_scale). The trace's E_pot leaves out
the particles' own mesh self-potential, now the exact CIC value (test/pm_self_potential). 2026-11-20: worst residual 2.0e-2 of
E_scale at one sync (pair at pericentre), others <= 6.1e-3: FAILS the 1e-2 bar (continuum self term: 4.9e-2 at 39 syncs).

History: PMGRID = 64 with R_out = 5000 AU passed the orbit bars but left a 2.6e-2 integration residual from the coarse mesh (also with
KETJU off); PMGRID = 256 there put r_s (244 AU) at the pair separation (|da/a| 2.4e-2). Redesigned 2026-11-19 (Uli).
