# pm_self_potential

One 20 Msun particle moving through a periodic 2e5 AU box with PMGRID = 32. Alone in the box its whole PM potential is its own
mesh self-potential, so the energy step trace's E_pot (which leaves that out, PM_EXACT_SELF_POTENTIAL_DIAG) must be zero at every
sync: bar |E_pot| <= 1e-5 of 1/2 m G m/(sqrt(pi) r_s) (single-precision FFT). Mass and energy step traces must close.
The diagnostic changes no force: forces and orbits are bitwise those of a build without it.
