/* State-hash diagnostic harness — for detecting where two multi-rank
 * runs first diverge in physical/particle state.
 *
 * Motivation (codex 2026-05-07): the call-19 NGL divergence has been proven
 * to NOT be a walker bug (same-state oracle: bad_rows=0). The bug is upstream
 * multi-rank state divergence. The most useful next diagnostic is a per-step
 * state hash so that comparing two B-vs-B logs identifies the first step
 * where the hashes diverge — i.e. ground zero.
 *
 * Env-gated on GIZMO_STATE_HASH=1. Off by default. When off, all functions
 * compile to a static-bool check + return — no work.
 *
 * Output line shape (parser-friendly):
 *   STATE_HASH rank=%d step=%lld label=%s NumPart=%d NumGas=%d Ti=%lld
 *              h_pos=0x%016llx h_h=0x%016llx h_active=0x%016llx h_total=0x%016llx
 *
 * Hash function: FNV-1a-like 64-bit xor-fold over bit-representations of
 * doubles. Bit-exact: no rounding, no FP-floor masking. Two runs whose
 * state matches will produce identical hashes; any divergence is real.
 */

#ifndef STATE_HASH_H
#define STATE_HASH_H

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 1 if state hashing is enabled this run. Cached static read. */
int state_hash_enabled(void);

/* Emit a STATE_HASH line for the current rank's state, labeled by an
 * arbitrary string (e.g. "post_drift", "post_decomp", "pre_call19").
 * Hashes: P[].Pos, P[].KernelRadius, ActiveParticleList contents.
 * Step counter is monotonic; pass -1 to use an internal auto-increment.
 *
 * No-op when state_hash_enabled() is 0. Cheap when off (one bool read).
 */
void state_hash_record(const char *label, long long step);

#ifdef __cplusplus
}
#endif

#endif /* STATE_HASH_H */
