/* RUMI++ state of a star, one field of the compact star record (galaxy_sf/single_star_record.h) */
#ifndef RUMI_MEMBER_STATE_H
#define RUMI_MEMBER_STATE_H
struct rumi_member_state {
    double target_pos[3];    /* RUMI++ end-of-step position (the chord drift reaches it to roundoff) */
    double true_vel[3];      /* RUMI++ end-of-step velocity, set after the drift */
    double direct_acc[3];    /* exact star-star acceleration of a not-yet-member (G-free), taken out of its opening kick when it joins */
    long long ti_start, ti_end; /* the RUMI++ step the star is on */
};
#endif
