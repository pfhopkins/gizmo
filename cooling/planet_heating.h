/* planet_heating.h — radiogenic decay and accretional background heating for solid bodies.
 *
 * Activated by PLANET_HEATING. Auto-implies EOS_TILLOTSON (see precompiler_logic.h).
 * Compatible with COOLING (gas cells are unaffected if COOLING handles them separately).
 *
 * du/dt = Q_rad * exp(-t / tau_rad) + Q_acc
 *
 * Q_rad [code u / code t]: initial specific radiogenic heating rate (e.g., Al-26 + Fe-60).
 * tau_rad [code t]:        e-folding decay time (set large to approximate constant heat).
 * Q_acc  [code u / code t]: background accretional rate; 0 to disable.
 *
 * All three stored in code units in All, converted from user-supplied CGS in begrun.cc.
 * Applied to all active Type-0 particles once per timestep (operator-split with hydro).
 * Resolved impact heating is already in the hydro solver; Q_acc covers unresolved terms.
 *
 * Relies on includer having pulled in declarations/allvars.h and core/proto.h.
 * Don't include those headers here — they are not double-include safe.
 */
#ifndef PLANET_HEATING_H
#define PLANET_HEATING_H

#ifdef PLANET_HEATING

#include <math.h>

static inline void planet_heating_parent_routine(void)
{
    const double Q_rad0  = All.PlanetHeating_RadQ0;   /* code u / code t */
    const double tau_rad = All.PlanetHeating_RadTau;   /* code t; 0 disables */
    const double Q_acc   = All.PlanetHeating_AccQ0;    /* code u / code t */
    const double t_now   = All.Time;

    double Q_rad_now = 0.0;
    if(tau_rad > 0.0) { Q_rad_now = Q_rad0 * exp(-t_now / tau_rad); }
    if(Q_rad_now + Q_acc <= 0.0) { return; }

    /* Iterate strictly over the active list (already TimeBinActive-filtered).
     * Never loop NumPart in a per-step routine — see Phil's full-loop rule.
     * Loop body is embarrassingly parallel; index-based iteration so OMP can
     * partition cleanly over the std::vector. */
    const int n_act = (int)ActiveParticleList.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int idx = 0; idx < n_act; ++idx)
    {
        int i = ActiveParticleList[idx];
        if(P[i].Type != 0 || P[i].Mass <= 0) continue;
        double dt = get_particle_timestep_in_physical(i, P);
        if(dt <= 0) continue;

        double du = (Q_rad_now + Q_acc) * dt;
        CellP[i].InternalEnergy     += du;
        CellP[i].InternalEnergyPred += du;
        if(CellP[i].InternalEnergy < All.MinEgySpec)
            { CellP[i].InternalEnergy = All.MinEgySpec; }
        if(CellP[i].InternalEnergyPred < All.MinEgySpec)
            { CellP[i].InternalEnergyPred = All.MinEgySpec; }
    }
}

#endif /* PLANET_HEATING */
#endif /* PLANET_HEATING_H */
