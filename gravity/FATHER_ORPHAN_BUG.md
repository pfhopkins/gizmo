# Particles silently dropped from the gravity tree

Two defects in one chain. The first puts a local particle under a top node owned by another task;
the second turns that into either a silent wrong force or a segfault. Both are fixed here, and
neither is related to `RANDOMIZE_GRAVTREE` (found while testing it, reproduces with it off).

## 1. Root cause: the position -> key mapping was computed in four places

Domain decomposition decides **which task owns** a particle from its Peano-Hilbert key. The treebuild
decides **which top node it is inserted under** from the same key. These must agree bit-for-bit.

The expression `((P[i].Pos[k] - DomainCorner[k]) / DomainLen) + 1.0` was written out at four sites --
three in `domain.cc` (`domain_determineTopTree`, twice; the `PersistentKey` fill) and one in
`forcetree.cc` (`force_treebuild_single`). Under `-O3 -ffast-math` (every SYSTYPE in the Makefile)
the compiler is free to contract and reassociate it, and it did so **differently per translation
unit**:

| object | generated code |
|---|---|
| `domain.o` | `vfmadd132sd` -- `domain_double_to_int` is defined here, so it inlined and the whole expression fused to `fma(Pos-Corner, 1/DomainLen, 1.0)` |
| `forcetree.o` | `vsubsd` / `vdivsd` / `vaddsd` -- only the declaration is visible, so it emitted a call and computed `(Pos-Corner)/DomainLen + 1.0` |

These round differently. `domain_double_to_int` then extracts the top `BITS_PER_DIMENSION` mantissa
bits, so a 1-ULP difference is usually discarded -- but when it propagates a carry up into the bits
that select the top-node leaf, the two sides disagree about which leaf, and therefore which task,
the particle belongs to.

### Which optimization, exactly

Bisected with two translation units that reproduce the asymmetry (conversion visible in one, only
declared in the other) over 4e6 positions spanning the domain, same flags applied to both, counting
keys that differ. Production compiler (GCC 13.3, `-march=znver4`):

| flags | keys differing |
|---|---|
| `-O3` | 0 |
| `-O3 -ffp-contract=fast` / `-ffp-contract=off` | 0 |
| `-O3 -freciprocal-math` | 0 |
| `-O3 -fassociative-math -fno-signed-zeros -fno-trapping-math` | 0 |
| `-O3 -ffinite-math-only` | 0 |
| `-O3 -freciprocal-math -fassociative-math -fno-signed-zeros -fno-trapping-math` | 0 |
| **`-O3 -funsafe-math-optimizations`** | **206** |
| `-O3 -ffast-math` | 206 |
| `-O3 -funsafe-math-optimizations -fno-reciprocal-math` | 206 |
| `-O3 -funsafe-math-optimizations -fno-associative-math` | 211 |
| `-O3 -funsafe-math-optimizations -fno-reciprocal-math -fno-associative-math` | 211 |
| `-O3 -ffast-math -ffp-contract=off` | 278 |
| **`-O3 -ffast-math -fno-unsafe-math-optimizations`** | **0** |

`-funsafe-math-optimizations` is necessary and sufficient, and it is **not reducible to its
documented sub-options**: those combined give 0, while unsafe-math with them explicitly disabled
still gives 211. GCC gates the relevant transformations on the internal
`flag_unsafe_math_optimizations`, which passes test directly. So there is no finer-grained flag that
keeps unsafe-math and avoids this; the only flag-level mitigation is `-fno-unsafe-math-optimizations`
(which may be appended to `-ffast-math`, retaining the rest of the umbrella).

Note that FMA contraction is *not* the cause, despite being the visible difference in the
disassembly: `-ffp-contract=off` makes the disagreement **worse** (278), so contraction was partly
masking it. Enabling contraction alone, without unsafe-math, produces no disagreement at all.

Measured on `test/hernquist`, 48 ranks, 32768 particles, 8194 tree builds per run:

| stage | per run | fraction of previous |
|---|---|---|
| particle-key evaluations | 2.7e8 | -- |
| raw key differs between the two TUs | 115197 | 4.3e-4 |
| top-node **leaf** differs | 77 | 6.7e-4 |
| owning **task** differs -- **mass lost from the global tree here** | 3-10 | ~4-10% |
| additionally lands in `suns[0]`, so `Father[]` breaks too | 1-4 | ~1/3 |

Each filter is why a pervasive discrepancy (7850 of 8194 builds contain at least one) surfaced as a
handful of events per run and a rare crash.

Note where the mass is actually lost. `force_treeupdate_pseudos` **overwrites** a foreign top node's
moments with the owner's values (`Nodes[no].N_part = DomainMoment[i].N_part`), so a local particle
under a foreign top node has its contribution discarded whether or not it was detached from
`suns[0]`. The `suns[0]` clobbering is what additionally corrupts `Father[]` and produces the
segfault, but every foreign-keyed particle is missing from the global moments -- 3-10 per run, not
the 1-4 that are also orphaned. The integrity check below measures the wider number directly. Diagnostics that confirmed it and ruled out everything else:
`dt = 0` between decomposition and treebuild (nothing drifted); `DomainCorner`/`DomainLen` unchanged;
`DomainTask[]` and top-tree checksums, base pointers and array sizes all unchanged; and 0 foreign
particles *at the end of the decomposition* versus 3-10 at the treebuild.

**Fix:** `domain_peano_key()` in `domain.cc` is now the only place a position becomes a key, and all
four sites call it.

### Portability

Measured on both compilers used for production runs, same probe, 4e6 positions:

| | GCC 13.3 | AOCC 17 (clang) |
|---|---|---|
| original, duplicated expression | 206 differing | 224 differing |
| original, `-flto` | 0 | 0 |
| fixed, shared `noinline` definition | 0 | 0 |
| fixed, shared definition without `noinline` | 0 | 0 |
| fixed, either, with `-flto` | 0 | 0 |

So the bug is **not** GCC-specific -- AOCC diverges too -- and the fix holds on both, with and
without LTO. Both accept `__attribute__((noinline))` without an unknown-attribute warning, and both
support it, as does Intel's compiler; it is a GNU extension honoured by every compiler GIZMO
targets.

Two corrections to earlier claims made here. `-flto` does **not** reintroduce the divergence: with
whole-program visibility both compilers unify the computation, so LTO alone reduces the original to 0
differing keys. And `noinline` is **not** demonstrably required -- the shared definition without it
also gives 0 everywhere tested. It is retained as insurance, because the probe's two call sites are
structurally identical whereas the real callers sit in different loops, which is the case where
per-call-site optimisation could in principle diverge again. It costs one call per particle per
treebuild.

This is deliberately a structural fix rather than a flag change, so it holds regardless of build
flags, compiler, or whether LTO is enabled. Dropping `-funsafe-math-optimizations` would also close
this particular hole, but it would leave the underlying fragility -- four copies of one formula whose
agreement is assumed -- in place for the next optimiser to find.

### And the flag is worth keeping

Measured on the STARFORGE benchmark problem (`gizmo_benchmark`: M2e3_R3 Res126, MHD + cooling +
jets + radiation + winds + SNe), 48 ranks x 2 threads, exclusive genoa node, fixed 10-minute wall
budget per arm, arms alternated with the order reversed in the second round. Both arms were
bit-reproducible across rounds, so run-to-run variance is negligible:

| build | sync-points | sim time reached | vs plain `-O3` |
|---|---|---|---|
| `-O3` | 85, 85 | 0.0127625 | -- |
| `-O3 -ffast-math -fno-unsafe-math-optimizations` | 92, 92 | 0.0138135 | +8.2% |
| `-O3 -ffast-math` | 102, 102, 102 | 0.0153149 | +20.0% |

Every arm reproduced bit-identically, including `-ffast-math` across two different nodes (a same-node
control was run alongside the third arm precisely to check that), so run-to-run and node-to-node
variance are both negligible.

`-ffast-math` is **20.0% faster** (102/85 = 1.200; 0.0153149/0.0127625 = 1.2000 -- same timestep
cadence, simply more steps). Per sync-point, by component:

| component | share of runtime | speedup |
|---|---|---|
| total | -- | 1.21x |
| hydro/fluids | 60-65% | 1.12x |
| cooling+chem | 19-22% | 1.37x |
| domain | 8-9% | 1.35x |
| misc | 4-5% | 1.53x |
| treewalk | 0.8% | 1.23x |

The gain is concentrated in cooling/chemistry -- table lookups and transcendental rate arithmetic --
not in the tree walk, which is under 1% of runtime in this configuration.

### Cost of the flag-level mitigation

`-fno-unsafe-math-optimizations` is the only flag setting that removes the divergence (see the
bisection above), and it retains **41% of the speedup** -- 7 of the 17 sync-points of gain, i.e. +8.2%
over baseline against the full +20.0%. Choosing it over plain `-ffast-math` therefore costs about
**10% throughput** (102/92 = 1.109; per-step 5.370 vs 4.774 s, 1.125x).

Since the structural fix already makes the mapping immune to the flag -- on both GCC and AOCC, with
and without LTO -- paying 10% for redundant protection is not worthwhile, so **`-ffast-math` stays**.
The number does matter for anyone running a GIZMO without the fix: for them
`-ffast-math -fno-unsafe-math-optimizations` buys correctness at 10% rather than the 17% lost by
dropping the flag entirely.

Worth recording that instruction counts predicted this badly. `-fno-unsafe-math-optimizations`
produces almost exactly the plain `-O3` division count (`vdivsd` 3014 vs 3060, against 1903 for
`-ffast-math`), which suggested it would perform like plain `-O3`; it actually landed 41% of the way
to `-ffast-math`. The retained gain evidently comes from umbrella components such as
`-ffinite-math-only` and `-fno-math-errno`, which strip guard branches and error paths from hot loops
without changing the arithmetic.

The 10-minute budget includes initialisation (IC read, cooling tables, setup): GIZMO accounts 486.69 s
of step loop for `fast` and 488.68 s for `nofast`, so ~112 s of the 600 s is startup in both. Because
that cost is the same to within 2 s, it cancels rather than biasing the result: per-step from the
accounted loop alone is 5.749/4.771 = 1.205, against a throughput ratio of 1.200. The throughput
figure is therefore marginally conservative, and for production runs, where startup amortises to
nothing, the relevant number is the per-step 20.5%.

**Verified:** 3 runs x 8194 builds, 0 foreign and 0 orphans, versus 3-10 and 1-4 before. The
cross-TU divergence itself is *still measurable* in those runs (60-76 leaf differences between the
two original expressions), which is the point: the discrepancy is harmless once ownership and
placement can no longer disagree about it.

## 2. Consequence: `Father[]` had no defined value for a particle the treebuild missed

When a local particle is inserted under a foreign top node, `force_insert_pseudo_particles()`
overwrites `u.suns[0]` of that node with the pseudo-particle, detaching the subtree the particle sits
in. `force_update_node_recursive` walks `u.suns[]`, never reaches it, and so never writes its
`Father[]` entry -- the only place `Father[]` is assigned.

`Father` comes from `mymalloc`, a LIFO stack allocator over one arena, so it lands at the same offset
every build and an unwritten slot holds **the previous build's value for that particle** -- usually a
valid node index. That is why the symptom was intermittent rather than constant.

Consumers then followed a plausible but wrong father:

- `force_refresh_node_moments()` accumulated the particle's mass, momentum, softening and luminosity
  into an **arbitrary** node. Those are the multipole moments used for every subsequent force
  evaluation, so forces were wrong globally and Newton's third law was violated. It runs whenever
  `TreeMomentsStaleFlag` is set -- on sink/star formation, so constantly in STARFORGE runs.
- the `GravCost` accumulation in `gravity_tree()` skewed the load-balance weight, and segfaulted when
  the stale index fell outside the current node array. **This was the only visible symptom and the
  rarest one.**

Note that a particle orphaned this way is missing from its own node's moments regardless of
`Father[]`, and `force_exchange_pseudodata()` only exports moments for top nodes a task **owns**, so
its mass appeared in no rank's tree at all.

**Fix (defence in depth; the real fix is section 1):** seed `Father[i] = -1` per treebuild so "never
reached" is distinguishable from a stale index. Both consumers above already treated a negative entry
as "no parent". The seed alone would be a regression, because five sites dereference `Father[i]`
before checking it and `Nodes`/`Extnodes` are offset pointers, so index `-1` lands `All.MaxPart + 1`
elements before the base -- a silent corrupt read, and a wild *write* for `Extnodes`. Those five are
guarded here: the three kernel-radius guesses in `init.cc` and the link-length guess in
`subfind_density.cc` fall back to the root node, and `force_add_element_to_tree` returns before
touching `Extnodes`. The `forcetree_update.cc` sites already use `while(no >= 0)`.

Non-orphan behaviour is bit-identical: every guard fires only when `Father < 0`.

## Standing check

Nothing in the code tested the tree against a global invariant, which is why this survived for years:
every symptom was either silent or an intermittent crash with no attribution. `force_treebuild()` now
ends with one:

`Nodes[All.MaxPart].N_part` must equal `All.TotNumPart`. `N_part` is exchanged with the
pseudo-particle data and re-accumulated across foreign domains, so after the build the root node
counts every particle globally. A particle missing from the tree shows up as a deficit immediately.

Details that make it usable:

- **Integer, not mass.** A mass-based check cannot resolve one lost particle in ~1e6 -- node masses are
  `MyFloat` and the deficit falls below the accumulated summation error. The count is exact.
- **Zero-mass particles are allowed for.** Leaves count unconditionally but internal nodes only
  propagate `N_part` when `mass > 0`, so an all-zero-mass subtree is legitimately dropped (swallowed
  sinks await cleanup at `Mass = 0`). The global zero-mass count is subtracted before flagging.
- **Whole-tree builds only** (`mp == NULL`); FOF/SUBFIND build over subsets where the invariant does
  not hold.
- **Warns, does not abort.** A false positive must not be able to kill a long production run, and a
  warning is enough to stop the failure being silent -- which was the actual problem.
- **Negative deficits are flagged too**, catching double-counting rather than only loss.

Cost is one `MPI_Allreduce` of a single long plus an `O(N)` local count per treebuild: negligible
against the build.

Validated by running it against a build with the key fix deliberately reverted -- a check that never
fires is worthless:

| build | integrity warnings |
|---|---|
| key fix reverted | **8** (`root node holds 32767, expected 32768, deficit 1, 0 zero-mass allowed`) |
| key fix present | **0** |

Both runs completed to `TimeMax` with no crash, so the check detects the defect it was designed for
and does not false-positive.

## Reproducer

`test/hernquist` at 48 ranks, `TimeMax=118` (~10 min). Without the fix, ~3-10 foreign-keyed particles
and 1-4 orphans per run, and a ~20-25% chance of a SIGSEGV in the `gravtree.cc` GravCost loop. The
rate depends on particle distribution, rank count and top-node depth -- it fell to 0 at 8 ranks --
so this is a floor, not a bound. The original failures were on clustered cosmological zoom-ins at
high rank counts, the opposite corner from this test.
