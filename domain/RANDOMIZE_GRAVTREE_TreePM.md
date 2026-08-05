# RANDOMIZE_GRAVTREE and periodic TreePM

**Status:** diagnosed, reproduced, fix **implemented and verified** (see Results below). Not
committed. Two items remain unproven: imbalance-at-scale and `PM_PLACEHIGHRESREGION`.
**Scope:** why `RANDOMIZE_GRAVTREE` breaks periodic TreePM (esp. cosmological zoom-ins), and how to fix it to match AREPO's periodic method.

---

## TL;DR

`RANDOMIZE_GRAVTREE` decorrelates tree-force errors between rebuilds by moving the
tree geometry relative to the particles each timestep (Grudić+ 2020,
[arXiv:2010.11254](https://arxiv.org/abs/2010.11254)). It was ported from AREPO in
commit `d8703f1e` (2020-04-06) and validated for **tree-only** gravity (open
boundaries, non-periodic box, Ewald-periodic).

The current implementation uses **one** mechanism for all cases: shift the domain
center by a random vector and double the root node (`len *= 2`) in
`domain_findExtent`. That is AREPO's **non-periodic** method. AREPO uses a
**different** method for periodic gravity, and applying the non-periodic method to a
periodic box is what breaks TreePM zoom-ins.

The fix is to add the periodic branch AREPO actually uses: a random **coordinate
translation modulo the box**, with the box size left unchanged.

---

## Background: two AREPO methods, not one

From the AREPO public code release (Weinberger, Springel & Pakmor 2020, ApJS 248, 32,
[arXiv:1909.04667](https://arxiv.org/abs/1909.04667)), §3.1:

> "To reduce this effect, Arepo randomizes the placement of the tree domain center for
> each tree construction for simulations with **nonperiodic** boundary conditions for
> gravity … For simulations with **periodic** boundary conditions, Arepo instead
> **shifts the whole box by a new random vector in every domain decomposition**. This
> shift is done for **all of the coordinate variables used in the code** and is
> **transparent to input and output**."

- **Non-periodic:** move (and enlarge) the tree root node. ← what GIZMO does today.
- **Periodic:** translate all coordinates by a random vector mod box; box size fixed.

GADGET-4 (Springel+ 2021, [arXiv:2010.03567](https://arxiv.org/abs/2010.03567)) uses
the same randomized-displacement scheme.

---

## Why the non-periodic method breaks periodic TreePM

In GADGET/GIZMO the gravity octree and the domain decomposition share **one**
coordinate→integer-key mapping, defined by three globals set in `domain_findExtent`:
`DomainCorner`, `DomainLen`, `DomainFac`. The Peano-Hilbert key of every particle is
`domain_double_to_int((Pos - DomainCorner)/DomainLen)`, used to build the top-tree,
assign particles to MPI tasks (`DomainTask`), and insert particles into the force
tree. Move or resize that box and *everything* downstream moves with it.

Two facts make the periodic-TreePM case special:

1. **The short-range force is min-imaged at exactly `All.BoxSize`.** The tree walk
   computes particle↔node distances with `NGB_PERIODIC_BOX_LONG_*`
   (`forcetree.cc`), i.e. periodic wrapping at the physical box, and it *stops*
   walking a branch beyond `Rcut` (`r2 > rcut2`, gated by `#ifdef PMGRID`). The tree
   box must therefore stay commensurate with the periodic box.

2. **TreePM domain decomposition imports only within `Rcut`.** Unlike pure-tree
   gravity — where every task holds the *entire* top-tree as pseudo-particles, so the
   box may be shifted/enlarged freely (this is why the Ewald-tree-only tests passed) —
   TreePM exploits the `Rcut` cutoff to import only the neighborhood of each domain.
   That import assumes the SFC domain grid is aligned with the periodic box so the
   wrap-at-`BoxSize` maps cleanly onto the grid.

`len *= 2` with an arbitrary center offset violates both: the periodic boundary now
falls at an arbitrary non-grid-aligned location in the interior of a `2·BoxSize`
domain grid.

### The dominant symptom: load imbalance

`DomainLen` spans a fixed `BITS_PER_DIMENSION` Peano budget. Doubling it throws away
**one bit of SFC resolution per dimension — a factor of 8 in top-tree cells** —
because all particles now live in the central octant and the first refinement level is
wasted. For a **zoom-in**, the high-res clump is already a tiny fraction of the box;
halving the resolution and randomly offsetting it means the top-tree — bounded by
`MaxTopNodes`/refinement depth — can no longer cut finely enough *inside* the dense
region to spread its gravity cost across ranks. A few top-nodes hold most of the work
→ severe imbalance. Pure-tree runs never depended on that bit, so the regression went
unnoticed.

---

## Reproduction

Red test: [`test/zeldovich/test_zeldovich_randomize.py`](../test/zeldovich/test_zeldovich_randomize.py),
run on the periodic `PMGRID=256` zeldovich setup with `RANDOMIZE_GRAVTREE` added.
Diagnostic added to `domain_findExtent` (`domain.cc`, gated to `RANDOMIZE_GRAVTREE`)
prints `DomainLen`/`DomainCorner`/`BoxSize` each decomposition.

Result on current code (8 ranks, ~7 min):

```
RANDOMIZE_GRAVTREE: DomainLen=128128 DomainCorner=(-21197,-46917,-58022) BoxSize=64000
DomainLen/BoxSize = 2.002   →  FAIL (invariant: expected ~1.0)
load imbalance over run: median=0.002 max=0.050 (n=587)
```

Two honest observations:

- The **cause** reproduces deterministically: `DomainLen/BoxSize = 2.002`, the box is
  doubled, `DomainCorner` varies each decomposition (shift is live). This is a
  scale-independent sentinel — it flips green the instant the periodic path stops
  doubling.
- The **symptom** (imbalance) does *not* reproduce on zeldovich: the pancake is
  nearly uniform, so 8 ranks balance fine (5% max) despite the lost bit. The
  imbalance pathology is specific to **high-dynamic-range zoom-ins** and would need a
  clumpy IC at many ranks to exhibit — too heavy for CI. Hence the test asserts the
  invariant, not an imbalance threshold, and merely records the imbalance as a
  diagnostic.

---

## Fix

Add AREPO's periodic branch; keep the current doubling only for the non-periodic path.
A pure periodic translation preserves periodicity, keeps `DomainLen == BoxSize` (no
lost SFC bit, no imbalance), and — because it touches *all* coordinates before the
keys are computed — keeps the tree, `DomainTask`, and both PM grids mutually
consistent automatically. Forces are translation-invariant, so physics is unchanged.

### 1. Restrict the existing block to non-periodic (`domain.cc`, `domain_findExtent`)

```c
#if defined(RANDOMIZE_GRAVTREE) && !defined(BOX_PERIODIC)   // add !BOX_PERIODIC
  double dx[3];
  if(ThisTask == 0) { for(j=0;j<3;j++) {dx[j] = len * (get_random_number((MyIDType)(All.NumCurrentTiStep)+j) - 0.5);} }
  MPI_Bcast(dx, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  for(j=0;j<3;j++){ DomainCenter[j]+=dx[j]; DomainCorner[j]=DomainCenter[j]-len; }
  len *= 2;
#endif
```

### 2. Carry the shift in `All` (allvars.h) — makes restart free

```c
#if defined(RANDOMIZE_GRAVTREE) && defined(BOX_PERIODIC)
  double RandomShift[3];   /*!< current random translation of the coordinate frame */
#endif
```

Restart dumps the whole `All` struct, so a resumed run stays in the same frame with no
extra code.

### 3. The periodic shift (`domain.cc`), called from `domain_Decomposition`

```c
#if defined(RANDOMIZE_GRAVTREE) && defined(BOX_PERIODIC)
void domain_apply_random_shift(void)
{
    double newshift[3]={0,0,0}, box[3]={boxSize_X, boxSize_Y, boxSize_Z};
    if(ThisTask==0) {for(int j=0;j<3;j++){newshift[j]=box[j]*get_random_number((MyIDType)(All.NumCurrentTiStep)*3+j);}}
    MPI_Bcast(newshift, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);   // every rank agrees on the frame
    double delta[3]; for(int j=0;j<3;j++){delta[j]=newshift[j]-All.RandomShift[j];}
#pragma omp parallel for schedule(dynamic,256)
    for(int i=0;i<NumPart;i++){for(int j=0;j<3;j++){P[i].Pos[j]+=delta[j];}}
    for(int j=0;j<3;j++){All.RandomShift[j]=newshift[j];}
    do_box_wrapping();   // fold back into [0,box)
}
#endif
```

Call it right after the pre-decomposition drift in `domain_Decomposition` (~`domain.cc:238`),
so the top-tree, force tree, and PM meshes that follow all see the same shifted frame:

```c
    for(i=0;i<NumPart;i++){if(P[i].Ti_current!=All.Ti_Current){drift_particle(i, All.Ti_Current);}}
#if defined(RANDOMIZE_GRAVTREE) && defined(BOX_PERIODIC)
    domain_apply_random_shift();
#endif
```

### 4. Undo at particle output — one real site

Positions must be written in the physical frame. Only **one** `IO_POS` site actually
writes coordinate values — `fill_write_buffer` in `io.cc`. The other seven `case IO_POS`
occurrences are metadata switches (datatype, byte count, dataset name, …) and touch no
coordinates. So the output tax is a single subtract, right before the existing periodic
wrap:

```c
    fp_pos[k] = (MyOutputPosFloat) P[pindex].Pos[k];
#if defined(RANDOMIZE_GRAVTREE) && defined(BOX_PERIODIC)
    fp_pos[k] -= (MyOutputPosFloat) All.RandomShift[k];   // back to physical frame
#endif
#ifdef BOX_PERIODIC
    ... existing wrap into [0,box) ...
```

Velocities and accelerations are translation-invariant and need no change.

### 5. Keep the high-res PM region locked to the frame (zoom-ins)

`pmforce_is_particle_high_res()` reads `All.Xmintot[1]/Xmaxtot[1]` on **every** tree
step, but that region is only refreshed lazily at PM steps (`pm_init_regionsize`, via the
particle-outside-region retry in `long_range_force`). An inter-frame delta can be ~a full
box, so a frame change on a non-PM step would put every high-res particle "outside" the
stale region → misclassification → the short-range `Asmth[1]` no longer complements the
held long-range `Asmth[1]` → corrupted high-res forces. `domain_apply_random_shift` therefore
translates `Xmintot[1]/Xmaxtot[1]/Corner[1]/UpperCorner[1]` by the same `delta`, preserving
each particle's membership across the shift. (Grid 0 in a periodic box uses `All.BoxSize`,
not cached bounds, so it needs nothing.)

### 6. Group catalogs — guarded, not yet un-shifted

FOF/SUBFIND write group positions (`Group.CM/Pos`, `SubGroup.Pos/CM`) via `my_fwrite`,
**not** `fill_write_buffer`, so the single output un-shift does not cover them. Rather than
ship untested un-shifts into halo-finder output, the frame is (a) never re-randomized during
group-finding — `domain_apply_random_shift` is gated to the main-integration decomposition
(`UseAllTimeBins == 0`; FOF/SUBFIND pass `1`), and (b) `fof_save_groups` / `subfind_save_final`
`endrun` if called while `RandomShift != 0`. On-the-fly FOF for sink/BH seeding never reaches
those save routines and runs fine (translation-invariant). Postprocessing group-finding on a
snapshot never applies the shift (`RandomShift == 0`), so it works. **Follow-up:** to enable
on-the-fly catalogs, subtract `All.RandomShift` (and wrap) at the ~5 group position-output
sites in `fof.cc`/`subfind*.cc`, then drop the guards.

### 7. Guardrails — where the periodic shift refuses to run

- **`BOX_SHEARING`**: an x-translation couples into `Vel` (see `do_box_wrapping`,
  `predict.cc`). Compile-time `#error` in `allvars.h`.
- **Reflecting/outflow boundaries** (`BOX_REFLECT_*`, `BOX_OUTFLOW_*`) with `BOX_PERIODIC`:
  those dimensions are not translation-invariant. Compile-time `#error`.
- **Non-cubic (`BOX_LONG_*`)** is fine — the shift is per-dimension (`boxSize_X/Y/Z`).
- **Fixed external/analytic potentials** on box coordinates, or any other output path that
  writes absolute positions (e.g. `OUTPUT_LINEOFSIGHT`), are *not* covered by the single
  particle un-shift and would need the same `- RandomShift` correction. Documented, not
  guarded. A self-consistent zoom-in (high-res region derived from the particle distribution)
  is the intended use and is fully covered.

---

## Why not just migrate to integer coordinates like AREPO?

AREPO stores positions as **integers**, which makes this shift trivial and exact:
periodic wrap is free (unsigned overflow at 2ⁿ), the shift is bit-for-bit reversible
(no float round-off), and there is a single internal↔physical boundary so "undo at
I/O" is folded into the one int→float conversion (no 8-site tax). GIZMO stores
positions as `double` and derives integer keys transiently
(`domain_double_to_int`), so it pays the float tax.

That is **not** a reason to retrofit integer coordinates: it would rewrite the
position representation across hydro, gravity, feedback, sinks, I/O, and analysis to
save eight one-line subtractions and a round-off margin (~9 orders below zoom
resolution) that is unmeasurable here. AREPO adopted integers for **uniform precision
across large dynamic range**, not for this feature; the clean shift is a side benefit.
GIZMO should pay the bounded float tax and move on.

---

## Testing plan

- **Sentinel (done, red):** `DomainLen/BoxSize < 1.1` under `RANDOMIZE_GRAVTREE +
  BOX_PERIODIC + PMGRID`. Deterministic, scale-free, catches any regression to
  doubling. Flips green when the fix lands.
- **Regression guard for the working path:** keep a non-periodic `RANDOMIZE_GRAVTREE`
  case green (the method that already works).
- **Symptom (optional, heavy):** a clumpy/zoom periodic IC at higher rank count to
  exhibit the actual imbalance and confirm the fix removes it — not for CI.
- **Physics payoff:** the momentum-conservation benefit is an isolated-system
  (non-periodic) effect and a periodic box has ~zero net momentum by symmetry, so it
  is not the right signal for the *periodic* fix. The invariant + imbalance are.
  (This expectation turned out to be too pessimistic — see Results.)

---

## Results

### Verified

| check | result |
|---|---|
| Compile matrix, 7 configs | 7/7. Includes the flag-OFF build (proves no unguarded references) and both `#error` guards firing for the right reason. Configs added to `test/compile_suite`. |
| Root-node invariant | `DomainLen/BoxSize` **2.002 → 1.001** (residual is the pre-existing `len *= 1.001` margin). `DomainCorner` now stable instead of jumping ±40000/step. Red test flipped green. |
| Zel'dovich vs analytic solution | passes at **unchanged** L1 < 0.3 tolerance (`randomize_gravtree` variant of `test_zeldovich.py`). |
| Output un-shift | pancake position fixed to within 0.073% of the box across 96 snapshots. Missing un-shift would scatter it over ~U[0, BoxSize). |
| Non-periodic path preserved | still compiles and is now correctly the *only* user of the box-doubling. |

### Momentum drift (`|Δv_com| / v_rms(0)`, see `test/momentum_drift_common.py`)

| test | pair | flag OFF | flag ON | ratio |
|---|---|---|---|---|
| plummer | non-periodic | 3.24e-03 | 1.41e-04 | 22.9x |
| hernquist | non-periodic | 7.65e-03 | 2.27e-04 | 33.7x |
| evrard | non-periodic | 7.16e-02 | 1.80e-02 | 4.0x |
| plummer | periodic TreePM | 9.58e-04 | 1.36e-03 | 0.71x |
| hernquist | periodic TreePM | 2.89e-02 | 1.13e-03 | 25.7x |

The periodic path works; plummer's 0.71x is a *low flag-off draw*, not a regression. The
randomized values are tight across both ICs (1.13e-3, 1.36e-3) while flag-off spans 30x
(9.58e-4 to 2.89e-2) — correlated error magnitude is configuration-dependent, whereas
randomization replaces it with a bounded random walk.

Mechanism observed directly: fitting drift ~ t^p, wherever flag-off is secular the
randomized run becomes a random walk — plummer non-periodic 0.98 → 0.62, plummer periodic
1.37 → 0.86, hernquist periodic 1.60 → 0.59. (Evrard's slopes are not meaningful: it is a
violent collapse, not a system at rest, so real dynamics dominate its COM signal; only its
4x amplitude win counts.)

Contrary to the pessimistic expectation in the Testing plan above, the periodic path *does*
show a large momentum-conservation benefit — the periodic gravity path carries substantial
correlated force error of its own (see below).

### Side finding: pre-existing periodic-gravity r_h runaway in hernquist

Surfaced by this work, unrelated to the patch (proved by preprocessing `domain.cc` with the
affected config: zero occurrences of any new symbol). Every flag-OFF periodic hernquist
variant secularly inflates the half-mass radius to ~10% by `TimeMax=118`, straddling the
0.10 tolerance: ewald 0.1076, pmgrid256 0.1019, pmgrid 0.0984/0.1005 on repeats. Momentum
drift and r_h both reproduce to ~1-2%, so the pathology is deterministic; only the *test* is
marginal, because `TimeMax` lands where the runaway crosses the tolerance. Energy is
conserved throughout in all variants (0.0002-0.0043 vs tol 0.04), so it is a structural
rearrangement, not an integration failure.

Four candidate causes tested and eliminated:

| hypothesis | test | verdict |
|---|---|---|
| PM under-resolution | PMGRID 64 → 256 (cell 7.81 → 1.95) | no effect (momentum 2.92e-2 → 2.85e-2) |
| tree/PM split location | Rcut 43.9 → 11.0 | no effect |
| periodic image tides | a periodic translation is an exact symmetry of the image lattice, so randomization could not affect image tides — yet it removes the runaway | logically excluded |
| halo fills its box | BoxSize 500 → 2000, halo/image gap 17.5 → 1517.5 (86x volume) | momentum drift unchanged to 1% (2.933e-2 → 2.916e-2); r_h only 1.31x better |

The effect is identical with PM absent (ewald), PMGRID=64 and PMGRID=256, so it is
periodicity itself, not the mesh. **The mechanism is not identified**; narrowing it further
needs direct force-error measurement against a brute-force reference, not more parameter
sweeps. `RANDOMIZE_GRAVTREE` removes it entirely (randomize_pmgrid r_h drift 0.0118, better
than the non-periodic baseline's 0.0152).

Handled by a surgical `pytest.xfail` in `test/hernquist/test_hernquist.py` that fires only
for those three flag-off variants and only when the assertion actually exceeds tolerance, so
the straddling `pmgrid` still passes cleanly when it is under, and the energy/momentum
assertions continue to gate for every variant. Do **not** loosen the 0.10 tolerance or
enlarge the box to make it green: BoxSize=2000 yields a passing 0.0821 while leaving the
underlying force error untouched, i.e. it masks the defect.

### Still unproven

- **Imbalance at scale** — the original zoom-in symptom. Zeldovich is too uniform to exhibit
  it (load imbalance median 0.002, max 0.05-0.07 with the flag on). Needs a clumpy/zoom IC at
  high rank count.
- **`PM_PLACEHIGHRESREGION`** — the `All.Xmintot[1]` shift is compile-verified only, never
  exercised at runtime. `randomize_pmgrid` covers the periodic path but without a nested
  high-res region.

---

## References

- Weinberger, Springel & Pakmor 2020, ApJS 248, 32 — [arXiv:1909.04667](https://arxiv.org/abs/1909.04667) §3.1 (periodic vs non-periodic randomization)
- Springel, Pakmor, Zier & Reinecke 2021 — [arXiv:2010.03567](https://arxiv.org/abs/2010.03567) (GADGET-4 randomized displacements)
- Grudić et al. 2021 (STARFORGE methods) — [arXiv:2010.11254](https://arxiv.org/abs/2010.11254) (cited by the `RANDOMIZE_GRAVTREE` flag)
- GIZMO commit `d8703f1e` (2020-04-06) — original port of the non-periodic method
