# HANDOFF — Mode B Phase 0 decision table (2026-05-07)

**Status: DRAFT — rows 1/2/3/5/6 fill in when Vista 697487/8/9 land. Row 4 final.**

## Pin

> **tiny-N cannot pay for global state.**
> Mode B sidesteps SIDX/halo for Nactive < threshold; Mode A keeps owning bulk-N.

## Six-row decision table

| row | required answer | status | source |
|---|---|---|---|
| 1 | density Nactive histogram + threshold | __TBD (Vista 697487 np=2)__ | `tools/phase0_summarize.py` |
| 2 | density tiny-N phase costs (sidx_dec / refresh / ghost / GPU) | __TBD (Vista 697487)__ | PHASE0_NGL + PHASE0_GHOST |
| 3 | ghost/SIDX touched? (must be 0 under Mode B) | acceptance assertion only — pre-Mode-B baseline ≠ 0 | PHASE0_GHOST count |
| 4 | gravity tree reusable? | **YES, with three constraints** | codex 2026-05-07 |
| 5 | rank-scaling tiny-N (-np 2/4/8) | __TBD (Vista 697488/9)__ | PHASE0_NGL ratios |
| 6 | Mode B first target | __density tiny-N tail; sink_env1 secondary canary__ (legacy preview) | per-caller table |

---

## Row 1 — Nactive histogram + threshold

[populate from `phase0_summarize.py phase0_*_jid697487*/np2/run.log`]

**Legacy preview (rank 0 only, run.log 2026-05-06):**

```
N range           count
      1              89
      2-4            51
      5-16           56
     17-64           92
     65-256          68
    257-1024         53
   1025-4096         26
   4097-16384         6
  16385-65536         9
  65537-262144       15
```

**Tentative threshold**: N=64 — below, Mode B; at/above, Mode A. Refines after Vista data shows the cost-vs-N curve from PHASE0 lines.

## Row 2 — Density tiny-N cost breakdown

[populate from PHASE0_NGL caller=density rows in 697487/np2 log]

**Legacy rank-0 preview** (median totals at N=1: 4ms; refresh dominates; sidx_dec≈0 at the very tail; sidx_dec=20ms in N=65–1024 band — driven by `sidx_id` cache-id swap, not Nactive itself; codex caveat 1).

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

## Row 5 — Rank-scaling tiny-N

[populate from PHASE0_NGL across np=2 (697487) / np=4 (697488) / np=8 (697489) at matched Nactive bins]

Looking for: super-linear cost growth with rank count at tiny-N, which would indicate Mode B's peer protocol must avoid collectives Day 1 (codex spec).

## Row 6 — Mode B first target

**Primary**: `density` tiny-N tail (N < 64). Many calls per step, deterministic (no probabilistic gate), broad range so the threshold dispatch logic gets exercised early.

**Secondary canary**: `sink_env1`. Legacy preview shows 32 calls at N=1–3 with `sidx_dec_sum=1.638s` total — paying ~50ms/call for ~1 neighbor is the textbook tiny-N global-state penalty. Clean validation case for Mode B's "no hidden global work" assertion.

**Probabilistic-gate callers** (`sink_swk`, `sink_feed`, `radfb_g`, `mech_fb`, `hii_fb`) — defer until later. Per Phil caution: a sample where the event didn't fire shows misleadingly low cost; can't dismiss as "no work forever."

**Day 1 oracle strategy**: `GIZMO_MODE_B_ORACLE=1` runs `mode_b_local_brute_walk` alongside the tree walk every call, sorts both, asserts equality. Auto-disable after first 100 calls if no mismatch (set TODO).

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
