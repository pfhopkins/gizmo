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
| owning **task** differs (particle now foreign) | 3-10 | ~4-10% |
| lands in `suns[0]` and is orphaned | 1-4 | ~1/3 |

Each filter is why a pervasive discrepancy (7850 of 8194 builds contain at least one) surfaced as a
once-per-run orphan and a rare crash. Diagnostics that confirmed it and ruled out everything else:
`dt = 0` between decomposition and treebuild (nothing drifted); `DomainCorner`/`DomainLen` unchanged;
`DomainTask[]` and top-tree checksums, base pointers and array sizes all unchanged; and 0 foreign
particles *at the end of the decomposition* versus 3-10 at the treebuild.

**Fix:** `domain_peano_key()` in `domain.cc` is now the only place a position becomes a key, and all
four sites call it. `__attribute__((noinline))` is required rather than cosmetic -- an inlinable
definition can be optimised differently at each call site, which `-flto` (offered as `OPT_EXTRA`)
would reintroduce.

This is deliberately a structural fix rather than a flag change, so it holds regardless of build
flags, compiler, or whether LTO is enabled. Dropping `-funsafe-math-optimizations` would also close
this particular hole, but it would leave the underlying fragility -- four copies of one formula whose
agreement is assumed -- in place for the next optimiser to find.

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

## Reproducer

`test/hernquist` at 48 ranks, `TimeMax=118` (~10 min). Without the fix, ~3-10 foreign-keyed particles
and 1-4 orphans per run, and a ~20-25% chance of a SIGSEGV in the `gravtree.cc` GravCost loop. The
rate depends on particle distribution, rank count and top-node depth -- it fell to 0 at 8 ranks --
so this is a floor, not a bound. The original failures were on clustered cosmological zoom-ins at
high rank counts, the opposite corner from this test.
