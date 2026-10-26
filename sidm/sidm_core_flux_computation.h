/* here is where we call the core of the SIDM calculation for DM particle-particle interactions */
#ifdef DM_SIDM
{
    /* check if target+neighbor are an SIDM candidate, and against self-interaction */
    double Pj_dtime = get_particle_timestep_in_physical(j);
    if( ((1 << local.Type) & (DM_SIDM)) && ((1 << P[j].Type) & (DM_SIDM)) && (local.ID != P[j].ID) && (local.dtime <= Pj_dtime))
    {
        if((local.dtime==Pj_dtime) && (local.ID > P[j].ID)) continue; // ensures interaction will only be calculated once for each pair //
        double h_si = 0.5*(kernel.h_i + kernel.h_j), m_si = 0.5*(local.Mass + P[j].Mass);
#ifdef GRAIN_COLLISIONS
        double prob = prob_of_grain_interaction(local.Grain_CrossSection_PerUnitMass , local.Mass, kernel.r, h_si, kernel.dv, local.dtime, j);
#else
        double prob = prob_of_interaction(m_si, kernel.r, h_si, kernel.dv, local.dtime);
#endif
        if(prob > 0.2) {out.dtime_sidm = DMIN(out.dtime_sidm , local.dtime*(0.2/prob));} // timestep condition not being met as desired, warn code to lower timestep next turn //
        /* pair key symmetric under exchange and distinct between pairs: identifiers repeat (a same-ID pair XORs to zero), so a mix of each end's position is folded in, ordered so either side computes the same key */
        static constexpr uint64_t AGS_FORCE_RNG_SALT = gizmo_loop_rng_salt("ags_force");
        uint64_t mix_i = gizmo_position_mix(local.Pos[0], local.Pos[1], local.Pos[2]), mix_j = gizmo_position_mix(P[j].Pos[0], P[j].Pos[1], P[j].Pos[2]);
        uint64_t mix_lo = (mix_i < mix_j) ? mix_i : mix_j, mix_hi = (mix_i < mix_j) ? mix_j : mix_i;
        uint64_t rng_key = ((uint64_t)local.ID ^ (uint64_t)P[j].ID) + mix_lo * 0x9E3779B97F4A7C15ULL;
        rng_key ^= (mix_hi + 0xBF58476D1CE4E5B9ULL + (rng_key << 6) + (rng_key >> 2));
        rng_key ^= rng_key >> 31;
        uint64_t rng_ti = ((uint64_t)All.Ti_Current) << 32, rng_ctr_threshold = rng_ti ^ AGS_FORCE_RNG_SALT ^ UINT64_C(0), rng_ctr_direction = rng_ti ^ AGS_FORCE_RNG_SALT ^ UINT64_C(1);
        if (gizmo_gpu_rand_double(rng_key, rng_ctr_threshold) < prob)
        {
#ifdef WAKEUP
            if(!(TimeBinActive[P[j].TimeBin])) {if(WAKEUP*local.dtime < Pj_dtime) {
                #pragma omp atomic write
                P[j].wakeup=-1;
                #pragma omp atomic write
                NeedToWakeupParticles_local = 1;
            }}
#endif
            Vec3<double> kick; calculate_interact_kick(kernel.dv, kick, m_si, rng_key, rng_ctr_direction);
            int k; for(k=0;k<3;k++) {
                double dv_sidm = (local.Mass/m_si)*kick[k];
                out.sidm_kick[k] -= (P[j].Mass/m_si)*kick[k];
                #pragma omp atomic
                P[j].Vel[k] += dv_sidm; // this variable is modified here so need to do this carefully here to ensure we don't multiply-write at the same time
                #pragma omp atomic
                P[j].dp[k] += dv_sidm * P[j].Mass;
            }
            force_note_motion_change(j);
            out.si_count++;
            #pragma omp atomic
            P[j].NInteractions++;
        }
    } // if((1 << ptype) & (DM_SIDM))
}
#endif
