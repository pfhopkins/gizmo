# KETJU coupling: deliberate deviations from Mike Grudic's scheme

Reference: origin/starforge_dev+ketju @ a59f31a08. Decided by Uli on 2026-11-19 (audit items K1, K3). Issue draft for Mike:
reports/issue_drafts/ketju_pm_kick_and_expansion.md (not posted).

## 1. Region members get the PM long-range kick

- **Mike:** core/kicks.cc:205 skips `apply_long_range_kick` for every `P[i].KetjuIntegrated` particle.
- **Ours:** members get the PM kick like every particle (core/kicks.cc, `apply_long_range_kick`; commit ff3f058f6).
- **Why:**
  - The tree applies only the short-range part S F_N.
  - The negative half-kicks subtract the full softened pair force F_N (tree_pair_gravity_factors, no S).
  - MSTAR integrates F_N.
  - With the mesh's (1 − S) F_N the chain force is counted once: S F_N + (1 − S) F_N − F_N + F_N = F_N.
  - Without it, members lose the whole long-range field: in a zoom, the halo beyond about one PM split scale.
- **Guard:** with PMGRID the run aborts if KetjuRegionRadius > r_s/2 on any PM level (commit 7abc4b43f). At r_s/2 the mesh share
  1 − S is 1.1e-2.
- **Evidence:**
  - test/ketju_pm, first design (R_out 5000 AU, PMGRID 64): pair CoM 0.94 R_out off the three-body reference without the kick,
    0.019 with it (bar 0.05).
  - Redesigned test (R_out 20000 AU, box 2e5 AU, PMGRID 128): see test/ketju_pm/README.md.

## 2. Tight-binary expansion doubles a at fixed e, and is reachable

- **Mike:** galaxy_sf/ketju_coupling.cc:1181-1284 scales only the CoM-relative velocities by 1/sqrt(2) (line 1239), at fixed r.
  - E' = E − K/2, so the binary becomes more bound: a circular pair goes to 2/3 of its separation.
  - KetjuExpandBinariesFactor (checked at 1184, 1214) is never registered as a parameter, so the code cannot run.
- **Ours** (commits 5c5c6e67b, eea32c810):
  - r → 2r and v → v/sqrt(2) about the pair CoM, so E → E/2, a → 2a, L → sqrt(2) L and e is unchanged;
  - positions and velocities are broadcast to the compute tasks;
  - the injected energy is booked in the KETJU ledger and in the energy step trace (ketju_expand);
  - the parameter is registered, default 0 (off).
- **Evidence** (test/ketju_expand_binary):
  - parameter not registered: 0 expansions;
  - old transform: a/a0 = 0.36 and e = 0.88 after 3 events (expected 8 and 0.6);
  - new transform: a/a0 = 2^k (3.99996 for k = 2) and e equal to e0 within the third star's tide;
  - the first event's injected energy equals G m1 m2 / (4 a0).
