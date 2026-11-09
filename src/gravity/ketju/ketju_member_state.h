/* KETJU state of a chain member, one field of the compact star record (galaxy_sf/single_star_record.h) */
#ifndef KETJU_MEMBER_STATE_H
#define KETJU_MEMBER_STATE_H
struct ketju_member_state {
    double final_vel[3];     /* MSTAR end-of-step velocity, set after the drift */
    double true_vel[3];      /* the same before the second negative half-kick: what a snapshot shows */
    double chain_vel[3];     /* chain centre-of-mass velocity: what another rank's imported copy moves with */
    double target_pos[3];    /* exact end-of-step position, snapped to after the drift */
    double chain_sphere[4];  /* chain bounding sphere (centre, radius; comoving), for the LET sender */
    double com_accel[3];     /* field external to the chain, for find_timesteps */
    double com_dt_tidal;     /* |T_external|^(-1/2); 0 = not set */
};
#endif
