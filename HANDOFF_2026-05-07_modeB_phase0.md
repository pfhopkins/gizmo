# HANDOFF — Mode B Phase 0 decision table (2026-05-07)

**Status: DRAFT — rows 1/2/3/5/6 fill in when Vista 697487/8/9 land. Row 4 final.**

## Pin

> **tiny-N cannot pay for global state.**
> Mode B sidesteps SIDX/halo for Nactive < threshold; Mode A keeps owning bulk-N.

## Six-row decision table

| row | required answer | status | source |
|---|---|---|---|
| 1 | Nactive histogram + threshold | **N=64 below → Mode B; bimodal at N=2-4 (median 1.77ms / p90 1.76s)** | Vista 697718 fire_m11i R1T72 |
| 2 | tiny-N phase costs | **`sidx_id=alltypes` cache: 102 calls, 57.3s sidx_dec; `step` cache: 1111 calls, 362μs total. >150,000× per-call asymmetry.** | sidx_id breakdown |
| 3 | ghost/SIDX touched? (must be 0 under Mode B) | baseline: ghost/NGL ratio = 0.514; 34 tiny-N calls with sidx_dec >100us | PHASE0 R1T72 |
| 4 | gravity tree reusable? | **YES, with three constraints** | codex 2026-05-07 |
| 5 | rank-scaling tiny-N | **3.7× anti-scaling R1T72 → R2T36; tiny-N (N=1) is 420× slower; step-cache sidx_dec 650,000× slower** | PHASE0 R2T36 (697734) |
| 6 | Mode B first target | **`sink_env1` (single-Nbin canary, 60% of NGL cost). Density tiny-N is broader-leverage second target.** | per-caller table |

---

## Row 1 — Nactive histogram + cost curve (R1T72 fire_m11i, job 697718)

```text
N range          count   tot_med  tot_p90    sidx_dec_med  sidx_dec_p90  refresh_med  gpu_med
     1              97   1.29ms   1.62ms     0us           1us           303us        537us
     2-4           290   1.77ms   1.760s     0us           1.562s        365us        1.55ms
     5-16          265   1.49ms   2.49ms     0us           1us           176us        656us
    17-64          181   1.30ms   3.44ms     0us           1us           165us        959us
    65-256         141   3.11ms   89.80ms    0us           1us           178us        1.40ms
   257-1024        126   4.51ms   24.50ms    0us           1us           613us        4.08ms
  1025-4096         10   11.69ms  17.29ms    0us           1us           1.57ms       10.46ms
  4097-16384        74   20.30ms  65.64ms    1us           1us           17.22ms      4.24ms
  16385-65536        9   94.90ms  1.224s     1us           1us           59.68ms      8.94ms
  65537-262144       5   98.06ms  1.211s     1us           1us           63.41ms      16.45ms
 262145-1048576      3   1.218s   1.224s     1us           1us           1.202s       16.30ms
 1048577+           10   930.91ms 1.300s     1us           1us           866.94ms     66.63ms
```

The N=2-4 row is bimodal: median 1.77ms, p90 1.760s. The p90 calls are sink_env1 (single Nbin=4, see Row 6).

**Threshold**: N=64 stays as the Mode A/B boundary. Below 64, median total is 1.3-1.8ms — Mode B target if the global-state burden is removed. Above 64, costs scale with N (gpu_med rises from 1.4ms → 66ms across the bins).

## Row 2 — Phase-cost decomposition (R1T72)

The dominant cost is sidx_dec but ONLY on the alltypes cache; refresh is the secondary cost; gpu walk itself is negligible at tiny-N:

| sidx_id | calls | sidx_dec_med | sidx_dec_p90 | sidx_dec_sum |
|---|---|---|---|---|
| alltypes | 102 | 0us | **1.761s** | **57.3s** |
| step | 1111 | 0us | 1us | 362us |

Per-call asymmetry: alltypes takes >150,000× longer than step on the rare paths that hit it. **The cliff codex predicted is real and is the alltypes cache rebuild, not Nactive.**

Refresh (compact_h refresh) dominates at large N: median 866ms at N>1M, scaling roughly with num_total. That's a Mode A target for commit C, not Mode B.

GPU walk itself at tiny-N is sub-ms — the work is real but cheap. The expensive part is everything around it.

## Per-caller summary (R1T72)

| caller | calls | N range | tot_sum | leverage |
|---|---|---|---|---|
| **sink_env1** | **34** | **all N=4** | **62.3s** | **highest — single Nbin, deterministic, all alltypes hits** |
| density | 591 | 1 - 4.58M | 24.0s | broad; tiny-N tail (N=1) is 1.3ms median |
| sink_feed | 34 | all N=4 | 6.4s | step cache, 190ms/call (refresh-dominated) |
| sink_swk | 34 | all N=4 | 6.3s | step cache, 190ms/call (refresh-dominated) |
| symlist | 452 | 0 - 4.57M | 2.8s | density's broad sibling |
| hii_fb | 25 | 1 - 612 | 0.4s | event-fired every call here (probabilistic gate; lucky) |
| mech_fb | 19 | 1 - 6641 | 0.4s | event-fired every call here |
| radfb_g | 24 | 1 - 543 | 0.3s | event-fired every call here |

**sink_env1 alone is 60% of all NGL cost on this run.**

## Row 3 — Ghost/SIDX touched? (Mode B target = 0)

This row's answer is "no, under Mode B" by design. The PHASE0 sweep gives the **pre-Mode-B baseline** (current path *does* touch them on every NGL call). After Day 1 Mode B density spike + acceptance test, the same field is asserted = 0 by `phase0_summarize.py` row-3 block.

Pre-Mode-B baseline (from sweep): __populate__ ghost/NGL ratio, sidx_dec>100us at N<100 count.

## Row 4 — Gravity-tree reusable for Mode B local search: **YES, with corrections**

Use the existing host-resident gravity tree (`Nodes[]`/`Nextnode[]`/`sibling`) for local pruning. Walker is a thin host range-walk on top of these structures. Cost is O(depth × candidates) per query, not a global rebuild.

**Three constraints baked into the design (codex 2026-05-07):**

1. **Scope honestly.** Only the local walker is ~100 lines. The full Mode B path (active-export protocol, MPI request/response, same-run oracle, CSR merge into the existing density consumer, per-call diagnostics) is *not* 100 lines. Plan for the full surface.
2. **Node drift ≠ particle drift.** `force_drift_node()` keeps node centers/extents fresh enough for pruning. But returned candidate j must be evaluated against `P[j].Pos` / `P[j].KernelRadius` at the same time semantics as today's NGL path — match the lazy-drift policy explicitly or drift before predicate, don't rely on node state alone.
3. **Local-particles authority.** Remote handler answers queries from *its own local particles*, never from LET/foreign tree state. Use the tree for pruning; returned candidates must be real local `j` indices in `[0, ghost_get_num_local())` passing type/mass/search predicates.

**Implementation status (commit `479c47a1`):** working. Brute-force walk (the oracle) + tree walk (the perf path) + `mode_b_local_walk_with_oracle()` wrapper that runs both paths under `GIZMO_MODE_B_ORACLE=1` and asserts equality. ONEWAY mode prunes via sphere-vs-AABB. SYMMETRIC mode currently always opens (no per-node max-h tracking yet) — oracle catches any mismatch.

**Fallback** (minimal per-domain persistent local host BVH): only if gravity-tree freshness or LET entanglement becomes a real blocker — not a precaution.

## Row 5 — Rank-scaling tiny-N (R1T72 vs R2T36, jobs 697718 / 697734)

Same fire_m11i problem, same code, same wall-time budget. R2T36 is dramatically slower:

| metric | R1T72 (697718) | R2T36 (697734) | ratio |
|---|---|---|---|
| Total NGL cost | ~103s | ~382s | **3.7× worse** |
| N=1 median total | 1.29ms | 542ms | **420× worse** |
| N=1 p90 total | 1.62ms | 1.287s | **794× worse** |
| step-cache sidx_dec sum | 362μs | 237.9s | **650,000× worse** |
| ghost/NGL ratio | 0.514 | 1.07 | 2× more ghost per NGL |
| symlist tot_sum | 2.8s | 153.0s | 55× worse |
| density tot_sum | 24.0s | 144.6s | 6× worse |
| sink_env1 tot_sum | 62.3s | 41.1s | (per-call similar; fewer calls reached) |

The anti-scaling source is unambiguous: **the global-state caches** (step + alltypes SIDX) **and ghost exchange** explode under multi-rank cross-rank synchronization. The step cache that was nearly free at single-rank costs ~443ms/call across 537 calls at R2T36.

**Implication for Mode B Day-1 protocol design**: peer protocol MUST avoid collectives. Codex's spec already mandates this; the data confirms it's not theoretical caution. Any collective-bearing path inherits the 100s-of-seconds anti-scaling tax.

R2T72-2node (697745, queued) will tell us whether the explosion is per-rank-shared-GPU or scales with node count.

## Row 6 — Mode B first target (revised after R1T72 data)

**Primary: `sink_env1`.** 34 calls all at Nactive=4 → 62.3s total NGL cost, **60% of the entire run**. Each call pays ~1.85s in `alltypes` SIDX rebuild to find ~4 neighbors. Single-Nbin (no threshold dispatch needed yet). Deterministic in this run (no probabilistic gate). Clean win-or-fail validation: if Mode B drops sink_env1 from 1.85s to milliseconds the row-3 assertions also pass.

**Secondary: `density` tiny-N tail (N=1).** 97 calls at median 1.29ms. Lower per-call leverage but exercises the threshold dispatch logic (N=1 < 64 → Mode B; N>>64 → Mode A) and the broad-N CSR merge integration. Lands after sink_env1 passes acceptance.

**Tertiary: `sink_feed` / `sink_swk`.** 34 calls each at N=4, 190ms median. Both hit `step` cache (not alltypes), so leverage per call is lower than sink_env1 — refresh dominates. Mode B should still help (no global state), but win is ~6s each, not 60s.

**Defer**: `symlist`, `hii_fb`, `mech_fb`, `radfb_g`. Probabilistic gates (Phil caveat) — they all fired in this run but won't always; lower per-call cost in this sample.

**Day 1 oracle strategy**: `GIZMO_MODE_B_ORACLE=1` runs `mode_b_local_brute_walk` alongside the tree walk every call, sorts both, asserts equality. Auto-disable after first 100 calls if no mismatch (set TODO).

### Why sink_env1 was demoted to "secondary canary" in the legacy preview, and why it's now primary

Legacy preview was rank-0-only on the downscoped `test/fire/` problem (different IC, different physics scale). Real fire_m11i shows sink_env1 paying **1.85s/call** (vs ~50ms/call in legacy) — 35× larger absolute cost, 60% of all NGL work. The order is dictated by the production-target data, not the downscoped preview.

---

## Acceptance gate (Day 2)

- Nactive ~ 1 → **milliseconds** (not 0.1–1s).
- Nactive ~ 100 → **comfortably <0.1s**.
- Same-run oracle: BIT-EXACT agreement on neighbor sets.
- Diagnostics show 0/0/0 on global-work assertions (`phase0_summarize.py` row-3 block).

If acceptance misses, debug Mode B itself before extending.

## Open issues NOT addressed by Phase 0 (tracked separately)

- **Call-19 multi-rank divergence** — proven NOT a walker bug (codex same-state oracle). Upstream multi-rank state divergence. Deferred until after Mode B passes + commit C lands. State-hash harness ready (`mesh/state_hash.{h,cc}`, commit `2f525862`) for first-divergent-step detection when we resume.
- **Vista anti-scaling** — separate red-alert track. Phase 0 row 5 captures the magnitude.
- **Implementation overhead** (build_sfc_tiles 1.5s = ~15× theoretical) — deferred. Mode B doesn't go through that path on tiny-N.

## Commits this session

- `6d329baa` drift-mark hardening + forensic instrumentation (negative-result note)
- `2d1aafa5` PHASE0_NGL + PHASE0_GHOST instrumentation
- `4ea99249` parser + Vista sweep sbatch (BUGGY — superseded)
- `7a3ce2f4` Mode B local-walker skeleton + legacy DIAG_NGL parser
- `2f525862` state-hash diagnostic harness (unwired)
- `3a1c7681` per-rank-count sbatch (fixes ibrun -n bug) + sidx_id field
- `479c47a1` working Mode B local-walker body + brute-force oracle
- `bec8a39d` phase0_summarize sidx_id + probabilistic-gate detection

## Next concrete steps after this lands

1. **Wire the walker** into density iter behind `GIZMO_MODE_B_DENSITY=1` threshold dispatch. Physics-path edit — needs Phil approval before landing.
2. Run Day 1 Mode B density spike with oracle on every call.
3. Run acceptance test (Day 2 above).
4. If passes: extend to symlist/gradient/hydro tiny-N.
