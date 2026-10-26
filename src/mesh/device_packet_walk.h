/* mesh/device_packet_walk.h -- the traversal mechanics a team of threads uses to walk a tree once
 * for a PACKET of queries.
 *
 * Several queries near one another reach mostly the same nodes, so a team can walk the tree once on
 * their behalf: one traversal, each member still deciding every node for itself.  This header holds
 * the part of that walk which does not depend on what is being decided -- the work item, the
 * per-walker stack and the team frontier that hold the continuations, the round boundary at which
 * walkers publish work to one another, and the counted ways a packet gives itself up.  A client
 * (the gravity walk, the neighbour walk) owns everything that does depend on it: which index
 * classes a walk may follow, the node and leaf decisions, the member inputs and sums, and how the
 * members evaluate what the traversal recorded.
 *
 * WORK ITEM.  A walker carries (no, exit, mask): the next index, the first index NOT in the item,
 * and the members still taking part.  When some members open a node and others do not, the packet
 * descends for the openers and the others re-join at the node's sibling through the continuation
 * (sibling, exit, mask).  Every stored item is that FULL triple; narrowing a mask in place would
 * lose the enclosing exit or the re-joining members.
 *
 * WHERE CONTINUATIONS GO.  Each walker keeps a bounded LIFO of its own in team scratch; the team
 * frontier, also in team scratch, holds what a walker offers to the others and what overflows a full
 * local stack.  A walker can therefore always progress on its own item, and the frontier affects
 * only balance, never termination.  When the local stack and the frontier are both full the packet
 * gives up -- counted, never a spin -- and the client replays its members another way.
 *
 * ONE WALKER is the depth-first traversal: the frontier is a plain stack that publishes on the
 * push, the local stack keeps the newest continuation and spills the OLDEST, so each member's
 * records arrive in the order its own single-query walk would have produced them.  SEVERAL WALKERS
 * is the cooperative traversal: every opened subtree is offered through the frontier, and the order
 * is free.
 *
 * Everything a team holds lives in its level-0 scratch, laid out by the client with
 * DevicePacketScratchLayout so that the legality check (team_size_max on the scratch request) sees
 * the whole request.
 *
 * Include after <Kokkos_Core.hpp>.
 */
#ifndef GIZMO_DEVICE_PACKET_WALK_H
#define GIZMO_DEVICE_PACKET_WALK_H

#include <stddef.h>

#define DEVICE_PACKET_MASK_BITS 64
typedef unsigned long long device_packet_mask_word_t;

/* a work item's indices; its mask words live beside it in a parallel array */
struct device_packet_item_t { int no, exit; };

/* mask words for a packet of q members */
KOKKOS_INLINE_FUNCTION int device_packet_mask_words(int q) {return (q + DEVICE_PACKET_MASK_BITS - 1) / DEVICE_PACKET_MASK_BITS;}

KOKKOS_INLINE_FUNCTION void device_packet_mask_clear(device_packet_mask_word_t *m, int words) {for(int w = 0; w < words; w++) {m[w] = 0ULL;}}
KOKKOS_INLINE_FUNCTION void device_packet_mask_copy(device_packet_mask_word_t *dst, const device_packet_mask_word_t *src, int words) {for(int w = 0; w < words; w++) {dst[w] = src[w];}}
KOKKOS_INLINE_FUNCTION int  device_packet_mask_test(const device_packet_mask_word_t *m, int b) {return (int)((m[b / DEVICE_PACKET_MASK_BITS] >> (b % DEVICE_PACKET_MASK_BITS)) & 1ULL);}
KOKKOS_INLINE_FUNCTION void device_packet_mask_set(device_packet_mask_word_t *m, int b) {m[b / DEVICE_PACKET_MASK_BITS] |= (1ULL << (b % DEVICE_PACKET_MASK_BITS));}
KOKKOS_INLINE_FUNCTION int  device_packet_mask_any(const device_packet_mask_word_t *m, int words) {for(int w = 0; w < words; w++) {if(m[w]) {return 1;}} return 0;}
KOKKOS_INLINE_FUNCTION int  device_packet_mask_equal(const device_packet_mask_word_t *a, const device_packet_mask_word_t *b, int words) {for(int w = 0; w < words; w++) {if(a[w] != b[w]) {return 0;}} return 1;}

/* The team's counters, one int slot each, in a block of DEVICE_PACKET_COUNTER_SLOTS.  The traversal
 * owns the slots below DEVICE_PACKET_CTR_CLIENT_BASE; a client numbers its own counters from that
 * base and checks its last one against the block with a static_assert. */
enum {
    DEVICE_PACKET_CTR_RECORDS = 0,       /* records in the chunk; reserved by atomic increment, clamped to the chunk once per round */
    /* The frontier's three counters.  Slots [0, FR_PUB) are the published stack and do not move
     * for the length of a round; new pushes grow above it and are read by nobody until the round
     * boundary compacts them in.  That is what makes a failed reservation harmless: an overflowing
     * pusher leaves its index above the cap and simply does not write, where a pusher that tried to
     * hand the index back could race a second overflowing pusher and wind the counter back below a
     * slot already holding a live continuation, which the next push would then overwrite. */
    DEVICE_PACKET_CTR_FR_PUB,            /* published items, poppable this round */
    DEVICE_PACKET_CTR_FR_TAKEN,          /* published items claimed this round */
    DEVICE_PACKET_CTR_FR_NEW,            /* reservations made above the published block this round */
    DEVICE_PACKET_CTR_FAILED,            /* the packet gave up */
    DEVICE_PACKET_CTR_FAIL_REASON,       /* why, in the client's numbering; read only when FAILED is set */
    DEVICE_PACKET_CTR_DONE,              /* the traversal is finished: a team-level verdict taken at the round boundary, never by one walker */
    DEVICE_PACKET_CTR_WORKING,           /* walkers still holding an item or a continuation at the end of the round; reset at the boundary */
    DEVICE_PACKET_CTR_STEPPED,           /* nodes stepped this round, by the whole team */
    DEVICE_PACKET_CTR_GAVE_UP,           /* the round boundary's verdict that the packet gave up: written by one thread at the
                                            boundary, read by all after it; walkers never write it (they write FAILED, which a
                                            fast walker may already be setting in the next round while a slow one reads) */
    DEVICE_PACKET_CTR_CLIENT_BASE        /* the first slot a client may number its own counters from */
};

/* The block is a fixed size rather than sized to a client's count: the scratch request feeds
 * team_size_max, so a block that changed size with a client's counters could change the launch
 * shape.  It is the LAST region a client lays out, so a counter numbered past it would run off the
 * end of the team's scratch with nothing to say so -- hence the client's static_assert. */
#define DEVICE_PACKET_COUNTER_SLOTS 16
static_assert(DEVICE_PACKET_CTR_CLIENT_BASE <= DEVICE_PACKET_COUNTER_SLOTS, "the traversal's counters must fit the counter block");

/* Lays a team's scratch out region by region.  Each region starts at the next multiple of its
 * alignment; `bytes` is the whole request.  A region asked for with zero bytes takes no space,
 * which matters: the request is what team_size_max is asked about, so an unused region is a real
 * cost in occupancy, not just in bytes. */
struct DevicePacketScratchLayout {
    size_t bytes = 0;
    size_t take(size_t region_bytes, size_t alignment)
    {
        bytes = ((bytes + alignment - 1) / alignment) * alignment;
        const size_t here = bytes;
        bytes += region_bytes;
        return here;
    }
};

/* True when a call of `n_queries` this rank must walk leaves no more than `ratio` lanes of the device
 * per query: the point from which independent per-lane walks keep the device busy and sharing a
 * traversal stops paying for its bookkeeping.  The count is this RANK's own; a global count divided
 * by the rank count is not this quantity.  Each client states its ratio where it measured it. */
KOKKOS_INLINE_FUNCTION int device_packet_lanes_saturated(int n_queries, int ratio, int lane_count)
{
    return ((long long) ratio * (long long) n_queries >= (long long) lane_count) ? 1 : 0;
}

/* The traversal state a team shares, and the operations on it.  `LocalStack` is compile-time so a
 * walker's ring index stays a mask rather than a division. */
template <int LocalStack>
struct DevicePacketTraversal {
    static_assert(LocalStack > 0 && (LocalStack & (LocalStack - 1)) == 0, "the local stack depth must be a power of two");
    using TeamMember = Kokkos::TeamPolicy<>::member_type;

    /* How many of the team's threads traverse, and how far each one goes before returning to the
     * round boundary.  steps_per_round bounds a walker's run so that what it published becomes
     * visible to an idle walker: too large and the others wait, too small and the round boundary is
     * the cost. */
    int n_walkers, steps_per_round;
    int frontier_cap, chunk_cap;
    int mask_words;

    /* Give the packet up, and say why.  Several threads can reach a give-up at once, so the flag is
     * CLAIMED rather than assigned: whoever claims it writes the reason, and the reason kept is the
     * one that actually stopped a traversal first -- a later thread writing over it would report a
     * symptom.
     *
     * Always claimed, never a cheaper unsynchronised path for a single walker: a client reaches this
     * from its evaluation as well as from the traversal, and evaluation runs on every member thread
     * whatever n_walkers is.  Keying the cheap path on n_walkers would be sound only while the two
     * happen to move together, and the failure it would produce -- FAILED set with the reason still
     * zero -- is invisible in the output and corrupts the very count a give-up is diagnosed from.
     * It costs an atomic on a path that runs at most once per packet. */
    KOKKOS_INLINE_FUNCTION void fail(int *ctr, int reason) const
    {
        if(Kokkos::atomic_compare_exchange(&ctr[DEVICE_PACKET_CTR_FAILED], 0, 1) == 0) {
            ctr[DEVICE_PACKET_CTR_FAIL_REASON] = reason;
        }
    }

    /* Add to a team counter several walkers may be adding to at once.  Plain arithmetic with one
     * walker, so the single-walker traversal keeps its plain instruction sequence. */
    KOKKOS_INLINE_FUNCTION void ctr_add(int *slot, int by) const
    {
        if(n_walkers == 1) {*slot += by;} else {Kokkos::atomic_fetch_add(slot, by);}
    }

    /* Offer a continuation to the other walkers, or keep it if there is no room.  Returns 1 when
     * the item is on the frontier and the caller is rid of it.
     *
     * With one walker the frontier is a plain stack that publishes on the push, which is the
     * depth-first overflow store the single-walker traversal relies on.  With several, a push
     * reserves above the published block and a reservation past the cap is simply not taken: the
     * counter is left where it is and clamped once, at the round boundary, by the one thread doing
     * the compaction.  Handing a failed index back would be the race -- two overflowing pushers
     * each decrementing can leave the counter below a slot that already holds a live continuation. */
    KOKKOS_INLINE_FUNCTION int frontier_push(int *ctr, device_packet_item_t *frontier, device_packet_mask_word_t *fmasks,
                                             const device_packet_item_t &item, const device_packet_mask_word_t *item_mask) const
    {
        const int W = mask_words;
        int slot;
        if(n_walkers == 1) {
            if(ctr[DEVICE_PACKET_CTR_FR_PUB] == frontier_cap) {return 0;}
            slot = ctr[DEVICE_PACKET_CTR_FR_PUB]++;
        } else {
            const int n = Kokkos::atomic_fetch_add(&ctr[DEVICE_PACKET_CTR_FR_NEW], 1);
            slot = ctr[DEVICE_PACKET_CTR_FR_PUB] + n;
            if(slot >= frontier_cap) {return 0;}
        }
        frontier[slot] = item;
        device_packet_mask_copy(fmasks + (size_t) slot * W, item_mask, W);
        return 1;
    }

    /* Take a published item, if this walker gets one.  A claim past the published block is not
     * handed back either: the compaction clamps it against what was actually published. */
    KOKKOS_INLINE_FUNCTION int frontier_pop(int *ctr, const device_packet_item_t *frontier, const device_packet_mask_word_t *fmasks,
                                            int &no, int &exit, device_packet_mask_word_t *mask) const
    {
        const int W = mask_words;
        const int pub = ctr[DEVICE_PACKET_CTR_FR_PUB];
        if(pub <= 0) {return 0;}
        int slot;
        if(n_walkers == 1) {
            slot = --ctr[DEVICE_PACKET_CTR_FR_PUB];
        } else {
            const int k = Kokkos::atomic_fetch_add(&ctr[DEVICE_PACKET_CTR_FR_TAKEN], 1);
            if(k >= pub) {return 0;}
            slot = pub - 1 - k;   /* newest first, over a block that does not move during the round */
        }
        no = frontier[slot].no; exit = frontier[slot].exit;
        device_packet_mask_copy(mask, fmasks + (size_t) slot * W, W);
        return 1;
    }

    /* Reserve a record slot.  Past the chunk nothing is written and nothing is handed back: the
     * walker yields with its item intact, the client's flush follows, and the node is reached again
     * and decided identically.  The clamp at the round boundary is what keeps the count honest. */
    KOKKOS_INLINE_FUNCTION int record_reserve(int *ctr) const
    {
        const int r = (n_walkers == 1) ? ctr[DEVICE_PACKET_CTR_RECORDS]++
                                       : Kokkos::atomic_fetch_add(&ctr[DEVICE_PACKET_CTR_RECORDS], 1);
        return (r < chunk_cap) ? r : -1;
    }

    /* The walker's item is complete: resume its newest continuation, else one the frontier has
     * published.  Returns 1 with the item live, or 0 when the walker has nothing -- which is not the
     * end of the traversal, since another walker may be about to publish, so the walker simply comes
     * back idle and the round boundary decides. */
    KOKKOS_INLINE_FUNCTION int next_item(int &no, int &exit, device_packet_mask_word_t *mask, int &item_live,
                                         int &local_head, int &local_count,
                                         const device_packet_item_t *local, const device_packet_mask_word_t *lmasks,
                                         int *ctr, const device_packet_item_t *frontier, const device_packet_mask_word_t *fmasks) const
    {
        const int W = mask_words;
        item_live = 0;
        if(local_count > 0) {
            const int slot = (local_head + local_count - 1) % LocalStack;
            no = local[slot].no; exit = local[slot].exit; device_packet_mask_copy(mask, lmasks + (size_t) slot * W, W);
            local_count--; item_live = 1; return 1;
        }
        if(frontier_pop(ctr, frontier, fmasks, no, exit, mask)) {item_live = 1; return 1;}
        return 0;
    }

    /* Some members of the item open the node whose first child is `child` and whose sibling is
     * `sibling`: the packet descends for them, and the item's other members re-join at the sibling
     * through the continuation (sibling, exit, mask).  Returns 1 with (no, exit, mask) set to the
     * descent, or 0 when the packet had to give up for want of room, with reason `no_room_reason`.
     *
     * WHERE that continuation goes is the difference between one walker and many, and it is the
     * whole of the cooperative traversal.
     *
     * One walker keeps it, because the only thing the frontier can do for it is hold an overflow,
     * and keeping the newest locally and spilling the OLDEST is what preserves depth-first order.
     * It also descends in place when nobody was excluded, since there is nothing to hand anyone.
     *
     * Several walkers OFFER it, on every open and even when nobody was excluded: a subtree nobody
     * can reach is a subtree the other walkers sit idle through, and the sibling continuation is
     * precisely the independent piece of work to give away.  Only when the frontier is full does the
     * walker keep it, and only when its own stack is full too does the packet give up.
     *
     * An empty continuation (the node's sibling IS the item's exit) is not worth a slot in either
     * regime. */
    KOKKOS_INLINE_FUNCTION int descend(int child, int sibling, int &no, int &exit,
                                       device_packet_mask_word_t *mask, const device_packet_mask_word_t *open_mask,
                                       int &local_head, int &local_count,
                                       device_packet_item_t *local, device_packet_mask_word_t *lmasks,
                                       int *ctr, device_packet_item_t *frontier, device_packet_mask_word_t *fmasks,
                                       int no_room_reason) const
    {
        const int W = mask_words;
        const int share = (n_walkers > 1);
        const int have_continuation = (sibling != exit);
        const int all_descend = device_packet_mask_equal(open_mask, mask, W);
        if(!share && all_descend) {no = child; return 1;}   /* same item, deeper */
        if(have_continuation) {
            device_packet_item_t cont; cont.no = sibling; cont.exit = exit;
            int placed = 0;
            if(share) {placed = frontier_push(ctr, frontier, fmasks, cont, mask);}
            if(!placed) {
                if(local_count == LocalStack) {
                    /* the oldest of this walker's own continuations makes room; with one walker that
                       is the spill that keeps the local stack depth-first */
                    device_packet_item_t oldest = local[local_head];
                    if(!frontier_push(ctr, frontier, fmasks, oldest, lmasks + (size_t) local_head * W)) {
                        fail(ctr, no_room_reason); return 0;
                    }
                    local_head = (local_head + 1) % LocalStack; local_count--;
                }
                const int slot = (local_head + local_count) % LocalStack;
                local[slot] = cont; device_packet_mask_copy(lmasks + (size_t) slot * W, mask, W);
                local_count++;
            }
        }
        no = child; exit = sibling; device_packet_mask_copy(mask, open_mask, W);
        return 1;
    }

    /* The round boundary, entered by the whole team once its walkers have returned.  The team meets;
     * if the packet gave up this round, or gives up here for want of progress, it returns -1 and the
     * caller leaves its round loop.  Otherwise it returns whether the record chunk is full, for the
     * client's flush, with the frontier compacted, the done verdict set, and the team met again.
     *
     * It is the only place the frontier changes shape.  What survives is the published block the
     * pops did not reach, and on top of it this round's new pushes moved down to meet it -- so the
     * stack stays newest-on-top and holds no gap.  When nothing was popped the two blocks are already
     * adjacent and nothing moves at all. */
    KOKKOS_INLINE_FUNCTION int round_boundary(const TeamMember &team, int *ctr,
                                              device_packet_item_t *frontier, device_packet_mask_word_t *fmasks,
                                              int no_progress_reason) const
    {
        const int W = mask_words;
        const int t = team.team_rank();
        team.team_barrier();
        if(ctr[DEVICE_PACKET_CTR_FAILED]) {return -1;}
        /* read before the compaction writes anything, so no thread is reading a counter another is
           settling */
        const int chunk_full = (ctr[DEVICE_PACKET_CTR_RECORDS] >= chunk_cap);
        const int fr_pub    = ctr[DEVICE_PACKET_CTR_FR_PUB];
        const int fr_taken  = (ctr[DEVICE_PACKET_CTR_FR_TAKEN] < fr_pub) ? ctr[DEVICE_PACKET_CTR_FR_TAKEN] : fr_pub;
        const int survivors = fr_pub - fr_taken;
        const int room      = frontier_cap - fr_pub;
        const int fr_new    = (ctr[DEVICE_PACKET_CTR_FR_NEW] < room) ? ctr[DEVICE_PACKET_CTR_FR_NEW] : room;
        if(fr_taken > 0 && fr_new > 0) {
            /* ONE thread, ASCENDING, and both halves of that are load-bearing.
             *
             * The source and destination blocks OVERLAP whenever more items were pushed than popped,
             * which is the ordinary case: the slot this step writes, survivors + j, is the source of
             * item j - fr_taken, an EARLIER index.  Ascending order is therefore safe by construction
             * -- that earlier item has already been copied out before anything writes over it --
             * while a parallel copy has no order at all, so one lane can destroy a continuation
             * before the lane that owns it has read it.  The subtree in the destroyed slot is then
             * lost and the overwriting one is walked twice, with no failure raised, because both are
             * perfectly well-formed items.  Measured that way on the gravity walk: 79% of evrard's
             * particles wrong at the first snapshot, and per-step interaction counts moving in BOTH
             * directions.
             *
             * It costs one thread a run over at most the frontier once per round, against T walkers
             * each taking up to steps_per_round node steps; if a round boundary ever shows up in a
             * measurement, the fix is a ring buffer that never moves anything, not a parallel copy. */
            if(t == 0) {
                for(int j = 0; j < fr_new; j++) {
                    frontier[survivors + j] = frontier[fr_pub + j];
                    device_packet_mask_copy(fmasks + (size_t) (survivors + j) * W, fmasks + (size_t) (fr_pub + j) * W, W);
                }
            }
        }
        team.team_barrier();

        if(t == 0) {
            ctr[DEVICE_PACKET_CTR_FR_PUB] = survivors + fr_new;
            ctr[DEVICE_PACKET_CTR_FR_TAKEN] = 0; ctr[DEVICE_PACKET_CTR_FR_NEW] = 0;
            if(ctr[DEVICE_PACKET_CTR_RECORDS] > chunk_cap) {ctr[DEVICE_PACKET_CTR_RECORDS] = chunk_cap;}
            ctr[DEVICE_PACKET_CTR_DONE] = (ctr[DEVICE_PACKET_CTR_WORKING] == 0 && ctr[DEVICE_PACKET_CTR_FR_PUB] == 0);
            ctr[DEVICE_PACKET_CTR_WORKING] = 0;   /* counted afresh by the walkers next round */
            /* A round in which nobody advanced and nobody was waiting on the chunk cannot happen: a
             * walker holding an item steps at least once, and a team holding nothing is done.  It is
             * tested rather than asserted because the alternative to noticing is a kernel that never
             * returns, and the packet has a counted way out. */
            if(ctr[DEVICE_PACKET_CTR_STEPPED] == 0 && !ctr[DEVICE_PACKET_CTR_DONE] && !chunk_full) {
                fail(ctr, no_progress_reason);
            }
            ctr[DEVICE_PACKET_CTR_STEPPED] = 0;
            ctr[DEVICE_PACKET_CTR_GAVE_UP] = ctr[DEVICE_PACKET_CTR_FAILED] ? 1 : 0;
        }
        team.team_barrier();
        if(ctr[DEVICE_PACKET_CTR_GAVE_UP]) {return -1;}
        return chunk_full;
    }
};

#endif /* GIZMO_DEVICE_PACKET_WALK_H */
