#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

#if (SINGLE_STAR_TIMESTEPPING > 0)
// wraps around angle to the interval [0, 2pi)
double wrap_angle(double angle){
    if (angle > 2*M_PI)	return fmod(angle, 2*M_PI);
    else if (angle < 0) return 2*M_PI + fmod(angle, 2*M_PI);
    else return angle;
}

// Solve Kepler's equation to convert mean anomaly into eccentric anomaly
double eccentric_anomaly(double mean_anomaly, double ecc){
    double x0 = mean_anomaly;
    double err = 1e100;
    int iterations = 0;
    double twopi = 2*M_PI;
    while(fabs(err/twopi) > 1e-14 && iterations < 20){ // do Newton iterations
        err = (x0 - ecc*sin(x0) - mean_anomaly)/(1 - ecc*cos(x0));
        x0 -= err;
        x0 = wrap_angle(x0);
        iterations += 1;
    }
    return x0;
}


/*
Advances the binary by timestep dt
mode 0 - Just fill out the particle's kick and drift for the timestep, without doing the update
mode 1 - Actually update the binary separation and relative velocity. This should be done on the full-step drift.
*/

void kepler_timestep(int i, double dt, Vec3<double>& kick_dv, Vec3<double>& drift_dx, int mode){
    double dr = P[i].comp_dx.norm();
    double dv = P[i].comp_dv.norm();

    Vec3<double> dx_normalized = P[i].comp_dx / dr;
    Vec3<double> dx_new, dv_new;
    double norm, true_anomaly, mean_anomaly, ecc_anomaly, cos_true_anomaly,sin_true_anomaly;
    double x = 0, y =0, vx =0, vy = 0; // Coordinates in the frame aligned with the binary
    double Mtot = P[i].Mass + P[i].comp_Mass;

    double specific_energy = .5*dv*dv - All.G * Mtot / dr;
    double semimajor_axis = -All.G * Mtot / (2*specific_energy);

    Vec3<double> h = cross(P[i].comp_dx, P[i].comp_dv); // specific angular momentum vector

    double h2 = h.norm_sq();
    double ecc = sqrt(1 + 2 * specific_energy * h2 / (All.G*All.G*Mtot*Mtot));
    Vec3<double> n_x = cross(P[i].comp_dv, h) - All.G * Mtot * dx_normalized; // LRL vector: dv x h - GM dx/r

    norm = n_x.norm();
    n_x /= norm; // direction should be so that x points from periapsis to apoapsis

    Vec3<double> n_y = cross(n_x, h) / sqrt(h2); // cross product of n_x with angular momentum to get a vector along the minor axis

    // Transform to coordinates in the plane of the ellipse
    x = dot(P[i].comp_dx, n_x);
    y = dot(P[i].comp_dx, n_y);
    //printf("Kepler transform stuff x %g y %g nx %g %g %g ny %g %g %g h %g %g %g\n", x,y,n_x[0],n_x[1],n_x[2],n_y[0],n_y[1],n_y[2],h[0],h[1],h[2]);

    true_anomaly = wrap_angle(atan2(y,x));
    ecc_anomaly = wrap_angle(atan2(sqrt(1 - ecc*ecc) * sin(true_anomaly), ecc + cos(true_anomaly)));
    mean_anomaly = wrap_angle(ecc_anomaly - ecc * sin(ecc_anomaly));
    //printf("Kepler x %g y %g dr orig %g dv orig %g ecc_anomaly %g mean_anomaly %g true anomaly %g change in mean anomaly %g ID %d \n", x, y, dr, dv, ecc_anomaly, mean_anomaly, true_anomaly, (dt/P[i].Min_Sink_OrbitalTime * 2 * M_PI),P[i].ID);
    //Changes mean anomaly as time passes
    mean_anomaly -= dt/P[i].Min_Sink_OrbitalTime * 2 * M_PI;
    mean_anomaly = wrap_angle(mean_anomaly);
    //Get eccentric anomaly for new position
    ecc_anomaly = eccentric_anomaly(mean_anomaly, ecc);
    //Get sine and cosine of new true anomaly (we don't actually need the new value)
    sin_true_anomaly = sqrt(1.0-ecc*ecc)*sin(ecc_anomaly)/( 1 - ecc*cos(ecc_anomaly) );
    cos_true_anomaly = ( cos(ecc_anomaly) - ecc )/( 1 - ecc*cos(ecc_anomaly) );
    dr = semimajor_axis * (1-ecc*ecc)/(1+ecc*cos_true_anomaly);
    x = dr * cos_true_anomaly;
    y = dr * sin_true_anomaly;


    dv = sqrt(All.G * Mtot * (2/dr - 1/semimajor_axis)); // We conserve energy exactly

    double v_phi = -sqrt(h2) / dr; // conserving angular momentum
    double v_r = sqrt(DMAX(0, dv*dv - v_phi*v_phi));
    if(ecc_anomaly < M_PI) v_r = -v_r; // if radius is decreasing, make sure v_r is negative

    //relative velocities in the frame aligned with the ellipse:
    vx = v_phi * (-y/dr) + v_r * x/dr;
    vy = v_phi * (x/dr) + v_r * y/dr;

    // transform back to global coordinates
    double two_body_factor=-P[i].comp_Mass/Mtot;
    //printf("Kepler comp_dx %g %g %g  comp_dv %g %g %g ID %d \n", P[i].comp_dx[0],P[i].comp_dx[1],P[i].comp_dx[2],P[i].comp_dv[0],P[i].comp_dv[1], P[i].comp_dv[2], P[i].ID);
    dx_new = x * n_x + y * n_y;
    dv_new = vx * n_x + vy * n_y;
    drift_dx = (dx_new - P[i].comp_dx) * two_body_factor;
    kick_dv = (dv_new - P[i].comp_dv) * two_body_factor;
    if(mode==1){ // if we want to do the actual self-consistent binary update
        P[i].comp_dx = dx_new;
        P[i].comp_dv = dv_new;
    }
}

// Quantity needed for gravitational acceleration and jerk; mass / r^3 in Newtonian gravity
double gravfac(double r, double mass){
    if(r < SinkParticle_GravityKernelRadius) {
	double u = r / SinkParticle_GravityKernelRadius;
	double h_inv = 1. / SinkParticle_GravityKernelRadius;
	return mass * kernel_gravity(u, h_inv, h_inv*h_inv*h_inv, 1);
    } else return mass / (r*r*r);
}

// quantity needed for the jerk, 3* mass/r^5 in Newtonian gravity
double gravfac2(double r, double mass)
{
    double hinv = 1. / SinkParticle_GravityKernelRadius;
    return mass * kernel_gravity(r*hinv, hinv, hinv*hinv*hinv, 2);
}

// Computes the gravitational acceleration of a body at separation dx from a mass, accounting for softening
void grav_accel(double mass, Vec3<double>& dx, Vec3<double>& accel){
    double r = dx.norm();
    double fac = gravfac(r, mass); // mass / r^3 for Newtonian gravity
    accel = -dx * fac;
}

// Computes the gravitational acceleration and time derivative of acceleration (the jerk) of a body at separation dx and relative velocity dv from a mass, accounting for softening
void grav_accel_jerk(double mass, Vec3<double>& dx, Vec3<double>& dv, Vec3<double>& accel, Vec3<double>& jerk){
    double r = dx.norm();
    double dv_dot_dx = dot(dv, dx);
    double fac = gravfac(r, mass); // mass / r^3 for Newtonian gravity
    double fac2 = gravfac2(r, mass);
    accel = All.G * (-dx * fac);
    jerk = All.G * (-dv * fac + dv_dot_dx * fac2 * dx);
}

// Perform a 4th order Hermite timestep for the softened Kepler problem, evolving the orbital separation dx and relative velocity dv
void hermite_step(double mass, Vec3<double>& dx, Vec3<double>& dv, double dt){
    Vec3<double> old_accel, old_jerk, old_dx, old_dv, accel, jerk;
    double dt2 = dt*dt, dt3 = dt2 * dt;
    grav_accel_jerk(mass, dx, dv, old_accel, old_jerk);

    // Predictor step
    old_dx = dx;
    old_dv = dv;
    dx += dv * dt + 0.5*dt2 * old_accel + (dt3/6) * old_jerk;
    dv += old_accel * dt + 0.5*dt2 * old_jerk;

    grav_accel_jerk(mass, dx, dv, accel, jerk);

    dv = old_dv + 0.5*(accel + old_accel) * dt + (dt2/12) * (old_jerk - jerk);
    dx = old_dx + 0.5*(dv + old_dv) * dt + (dt2/12) * (old_accel - accel);
}

/*
Advances the binary by timestep dt
mode 0 - Just fill out the particle's kick and drift for the timestep, without doing the update
mode 1 - Actually update the binary separation and relative velocity. This should be done on the full-step drift.
*/
void odeint_super_timestep(int i, double dt_super, Vec3<double>& kick_dv, Vec3<double>& drift_dx, int mode)
{
    double t = 0, total_mass = P[i].comp_Mass + P[i].Mass, dt;
    Vec3<double> dx_old = P[i].comp_dx, dv_old = P[i].comp_dv;
    Vec3<double> dx = -P[i].comp_dx, dv = -P[i].comp_dv; // note sign change from comp_dx to the effective 1-body problem

    while(t < dt_super){
	// Determine timestep adaptively; tuned here to give 1% energy error over 10^5 orbits for a 0.9 eccentricty binary
	double vSqr = dv.norm_sq();
	double rSqr = dx.norm_sq();
    double r_effective = KERNEL_FAC_FROM_FORCESOFT_TO_PLUMMER * SinkParticle_GravityKernelRadius;
    rSqr += r_effective*r_effective;
	dt = DMIN(0.1/(sqrt(vSqr/rSqr) + sqrt((All.G * total_mass)/sqrt(rSqr*rSqr*rSqr))), dt_super-t); // harmonic mean of approach time and freefall time
    // could swap in any integration scheme you want here; default to Hermite
	hermite_step(total_mass, dx, dv, dt);
	t += dt;
    }

    double two_body_factor=-P[i].comp_Mass/total_mass;

    drift_dx = (-dx - P[i].comp_dx) * two_body_factor;
    kick_dv = (-dv - P[i].comp_dv) * two_body_factor;
    if(mode==1){ // if we want to do the actual self-consistent binary update
        P[i].comp_dx = -dx;
        P[i].comp_dv = -dv;
    }

}

#endif
