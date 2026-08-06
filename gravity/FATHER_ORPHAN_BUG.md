# Tree particles can be orphaned by pseudo-particle insertion

**Status:** mechanism confirmed by instrumentation. The damage-bounding half is implemented (see
*Fix*); **the orphaning itself is not fixed** and needs a decision on which invariant is intended.
Pre-existing and **independent of `RANDOMIZE_GRAVTREE`** — found while testing that feature,
reproduces with it off.

**Symptom as first seen:** intermittent SIGSEGV in `gravity_tree()`, `gravtree.cc` GravCost loop.
**Actual scope:** wrong tree multipole moments, hence wrong forces, plus corrupt `GravCost`. The
crash was the rarest consequence and the only visible one.

## Mechanism

In `force_treebuild_single()` (`gravity/forcetree.cc`) the order of operations is:

1. insert all local particles into the tree by Peano key
2. `force_insert_pseudo_particles()`
3. `force_update_node_recursive(All.MaxPart, -1, -1)`

Step 2 does:

```c
for(i = 0; i < NTopleaves; i++)
{
    index = DomainNodeIndex[i];
    if(DomainTask[i] != ThisTask)
        Nodes[index].u.suns[0] = All.MaxPart + MaxNodes + i;   /* overwrite */
}
```

If a **local** particle's current position keys into a top-node region owned by **another** task,
step 1 inserts it under that foreign top node. Step 2 then overwrites `u.suns[0]` of that node with
the pseudo-particle index, detaching whatever subtree occupied octant 0. Step 3 walks `u.suns[]`, so
it never reaches the detached particle.

### Why it is intermittent rather than constant

`Father` is `myfree`d and re-`mymalloc`d on every treebuild. `mymalloc` is a strict LIFO stack
allocator carving from one `malloc`ed arena (`system/mymalloc.cc`: `Table[Nblocks] = Base +
(TotBytes - FreeBytes)`, and `myfree` aborts if the freed block is not the last one). The
allocation sequence per step is deterministic, so `Father` lands at the same arena offset every
build and an unwritten slot holds **the previous build's value for that particle** — usually a
valid node index. So the usual outcome is a plausible-but-wrong father, not a crash. The crash
only appears when the stale index happens to fall outside the current node array (e.g. after the
node count shrinks).

This has a direct consequence for the fix: seeding `Father` to `-1` is **not** safe on its own,
because it replaces a plausible stale index with a wild one at every consumer that dereferences
`Father[i]` without checking the initial value (see *Fix*).

`Father[]` is written **only** in `force_update_node_recursive`:

```c
if(no < All.MaxPart) {Father[no] = father;}    /* only set it for single particles */
```

so a particle the recursion never reaches never gets a `Father` entry.

### Why position and ownership can disagree

Domain decomposition assigns ownership by Peano key and migrates particles accordingly, so the
common case is consistent. The mismatch needs the particle's key at *treebuild* time to differ from
the key its ownership was computed from — e.g. drift between decomposition and rebuild, or a
rebuild without an intervening decomposition. Only octant 0 of the foreign top node is clobbered,
so a mismatch does not always orphan anything, which is the second reason the rate is low.

## Consequences

Two consumers read `Father[]`, and **both already treat a negative entry as "no parent"** — but
nothing ever established that convention, so they follow garbage instead:

- `gravtree.cc`, GravCost accumulation: `int no = Father[i]; while(no >= 0) {... Nodes[no] ...}`
- `forcetree.cc`, `force_refresh_node_moments()`: `no = Father[i]; if(no < 0) {continue;}`

Garbage that is **out of range** → SIGSEGV (the originally observed crash).
Garbage that is **in range** → silent corruption, which is the more serious case:

- `force_refresh_node_moments()` accumulates the particle's mass, momentum, softening and
  luminosity into an **arbitrary** node. Those are the multipole moments used for every subsequent
  force evaluation, so forces are wrong for all particles, and Newton's third law is violated
  (spurious net momentum). This function is called from `gravtree.cc:84` and `potential.cc:55`
  whenever `TreeMomentsStaleFlag` is set — i.e. on sink/star formation, so constantly in
  STARFORGE-type runs.
- `GravCost` is the load-balance weight, so the domain decomposition is skewed by the bad entry.

Independently of `Father[]`, the orphaned particle is missing from its correct node's moments, so
the forces are wrong by its mass regardless.

### Why a local fix cannot be correct

`force_exchange_pseudodata()` packs moments only for the top-node leaves this task owns
(`DomainStartList[ThisTask*MULTIPLEDOMAINS+m] .. DomainEndList[...]`). So ownership matching
position is a *requirement* of the pseudo-particle scheme, not an incidental assumption.

If a local particle sits under a top node owned by another task, we never export that node's
moments (we do not own it) and the owner exports moments computed from its own particles, which do
not include ours. The particle's mass is therefore **absent from every other rank's tree** while
present in our own — which is exactly the asymmetry that shows up as a Newton's-third-law
violation.

The consequence for the fix: any approach that leaves the particle local while its position keys
foreign — reattaching the detached subtree, giving the pseudo-particle a non-colliding slot,
attaching the particle to the root — leaves this global error untouched. Attaching to the root is
the tempting one and is geometrically harmless for *our* walk, since the opening criterion uses
node centre and length, which stay correct; but our root moments are not what other ranks import.
All of these bound damage. Only making ownership match position fixes it.

## Measured rate

`test/hernquist`, 48 ranks, 32768 particles, full run to t=118, with `Father[]` filled each build
with a detectable sentinel (`-999999999`; negative so the `while(no >= 0)` consumers skip it
safely) and the check counting rather than aborting:

| quantity | value |
|---|---|
| tree builds in the run | 8194 |
| orphaned particles, total | 4 |
| max orphans in any one build | 1 |
| never-written (sentinel) | 4 / 4 |
| position in a foreign top-node region | 4 / 4 |
| worst-build orphan fraction | 3.05e-05 (= 1/32768) |

So roughly one orphan per 2000 tree builds, never more than one at a time, and **every** one
matched the predicted mechanism on both independent checks.

### Rank scaling

Repeating at two rank counts, six runs total, identical particle count (32768) and tree-build count
(8194 per run):

| ranks | runs | tree builds | orphans | segfaults | reached end |
|---|---|---|---|---|---|
| 48 | 4 | 32776 | 4, 2, 1, 1 = **8** | 0 | 4/4 |
| 8 | 2 | 16388 | 0, 0 = **0** | 0 | 2/2 |

The 48-rank counts are Poisson (mean 2.0, variance 2.0), i.e. independent rare events rather than
one pathological particle or region recurring.

Against the null that both rank counts share the 48-rank rate, observing 0 where 4 were expected
gives p = e^-4 ~ 0.018, so equal rates are rejected at the 5% level. The same data bound the 8-rank
rate at < 1.5/run (95% CL), so the reduction is real but its magnitude is not well determined --
these two runs cannot separate "somewhat lower" from "zero".

This matters for the diagnosis because it is independent of the per-event checks. Those establish
that each orphan *is* in a foreign top-node region; the rate falling with fewer top-node boundaries
is a direct consequence of the causal story and is hard to explain if the foreign flag were merely
a correlate.

It also means **the measured rate must not be read as an upper bound for production**: rank count
is a first-order knob, and the original segfaults were seen at high rank counts on clustered
cosmological zoom-ins -- the opposite corner from this test.

Consistency: 4 orphans/run against the ~20-25% per-run segfault rate observed without
instrumentation implies ~5% of orphans have a stale father index that lands outside the current
node array — the right order for "previous build's value, occasionally stale after the node count
shrinks".

Impact at this rate is small: one particle of mass M/32768 missing from the tree moments for a
single step perturbs neighbour forces at the 1e-5 level, which is why `hernquist` energy
conservation (|dE|/KE0 = 2e-4 to 4e-3, tolerance 4e-2) never flagged it. **This rate should not be
assumed to generalize** -- the mechanism requires position/ownership disagreement, so it should
scale with drift between domain decompositions and with the number of top-node boundaries. A
clustered zoom-in, or a run calling `force_refresh_node_moments()` frequently, could be well above
this.

## Evidence

Earlier instrumentation on the same test, which aborted on the first bad entry rather than counting
(so it establishes the mechanism per event, whereas the runs above establish the rate):

- `SENTINEL=1` in 7/7 trips — the slot was never written, confirming the particle was never
  reached by the recursion (not a stale-but-valid value).
- `FOREIGN=1` in 3/3 ownership checks, with `OWNER`/`THISTASK` = 24/44, 17/0, 22/7 — the particle's
  position keys into a region owned by a different task.
- `OUTSIDE_ROOT=0` always — refutes the alternative that particles had drifted outside the tree
  root box.
- Trip rate rose from ~20% to 87.5% once `Father[]` was pre-filled with a sentinel, which
  demonstrates that garbage frequently lands **in range** and corrupts silently rather than
  crashing.

## Fix

Two parts. Only the first is implemented; the real fix is ordering/insertion.

1. **Hygiene (implemented).** Seed `Father[i] = -1` for `i < All.MaxPart` immediately before
   `force_update_node_recursive`, per treebuild, **and guard every consumer that dereferences
   `Father[i]` without checking the initial value.** The seed alone is a regression: it converts a
   plausible stale index into index `-1`, and since `Nodes`/`Extnodes` are offset pointers
   (`Nodes[no]` maps to `Nodes_base[no - All.MaxPart]`), `-1` lands `All.MaxPart + 1` elements
   before the base — inside the arena, so a silent corrupt read, and a wild *write* for `Extnodes`.

   Five sites guard only the ancestor step (`p = Nodes[no].u.d.father; if(p < 0) break;`) and not
   the initial value:

   | site | fix |
   |---|---|
   | `core/init.cc` kernel-radius guess (×3) | fall back to the root node |
   | `subfind_density.cc` link-length guess | fall back to the root node |
   | `forcetree.cc` `force_add_element_to_tree` | return before touching `Extnodes` |

   The five sites in `forcetree_update.cc` and the two original consumers already use
   `while(no >= 0)` and need no change. Non-orphan behaviour is bit-identical, since the guards
   only fire when `Father < 0`. Also bound-check the chain in the GravCost loop so genuine
   `u.d.father` corruption reports itself instead of dying anonymously.

   This does **not** fix the orphaning — the particle is still absent from the tree moments and
   forces are still wrong by its mass. It bounds the damage.

2. **Root fix (not attempted).** Given the exchange constraint above, the only correct fix is to
   restore ownership == position. Two variants:

   - **Assert/enforce in `domain_Decomposition`.** Correct and cheap *if* the disagreement is a
     decomposition-side bug (e.g. a boundary case in the key assignment). But if some legitimate
     path rebuilds the tree without an intervening decomposition, a bare assert fires in normal
     operation and the fix degrades into "force a full decomposition before every treebuild",
     which is far more expensive than a treebuild.
   - **Lazy migration at treebuild.** Detect foreign-keyed local particles and exchange just those,
     instead of redoing the whole decomposition. Affordable in the structural case, but a
     substantially bigger change.

   Note that simply reordering — inserting pseudo-particles *before* local particles — does not
   work. The insertion loop follows a non-empty slot unconditionally
   (`nn = Nodes[th].u.suns[subnode]; if(nn >= 0) {parent = th; th = nn;}`), and a pseudo-particle
   index `All.MaxPart + MaxNodes + i` satisfies the `th >= All.MaxPart` internal-node test, so the
   next iteration indexes `Nodes[]` past its end. Reordering converts a detached subtree into an
   out-of-bounds access unless the insertion loop is taught about pseudo-particles. It also would
   not prevent foreign-keyed particles, since only `suns[0]` is ever overwritten.

   Which variant applies depends on whether time advances between the decomposition and the
   treebuild that orphans a particle. Both compute the key from identical expressions, so with
   unchanged positions the keys must agree; a mismatch requires the positions to have moved.

## Reproducer

`test/hernquist` at 32-48 ranks, `TimeMax=118`. Detection is probabilistic: measured trips at a
median of t=62 of 118, so a short run only catches 8-17%. With `Father[]` sentinel-filled the
detection rate is ~87%. Cost ~7-10 min at 48 ranks — viable as a nightly, not as CI.
