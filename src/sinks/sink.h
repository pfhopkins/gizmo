/*! \file sink.h
 *  \brief routine declarations for gas accretion onto sink particles, and sink particle mergers
 */
/*
* This file is largely written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
* see notes in sink.c for details on code history.
*/

#ifdef SINK_PARTICLES // top-level flag [needs to be here to prevent compiler breaking when this is not active] //


#define SINK_MINPOTVALUE_INIT 1.0e30
extern int N_active_loc_Sink;    /*!< number of active sink particles on the LOCAL processor */

extern struct sink_temp_particle_data       // sinkdata_topass
{
    MyIDType index;
    MyFloat Sink_SurroudingGasInternalEnergy;
    MyFloat Mgas_in_Kernel;
    MyFloat Mstar_in_Kernel;
    MyFloat Malt_in_Kernel;
    Vec3<MyFloat> Jgas_in_Kernel;
    Vec3<MyFloat> Jstar_in_Kernel;
    Vec3<MyFloat> Jalt_in_Kernel; // mass/angular momentum for GAS/STAR/TOTAL components computed always now
    MyDouble accreted_Mass;
    MyDouble accreted_Sink_Mass;
    MyDouble accreted_Sink_Mass_reservoir;
#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
    MyDouble Sink_AccretionDeficit;
#endif
#ifdef GRAIN_FLUID
    MyFloat accreted_dust_Mass;
#endif    
#ifdef RT_REINJECT_ACCRETED_PHOTONS
    MyFloat accreted_photon_energy;
#endif
#ifdef SINK_ALPHADISK_ACCRETION
    MyFloat mdot_reservoir;             /*!< gives mdot of mass going into alpha disk */
#endif
#if defined(SINK_OUTPUT_MOREINFO)
    MyFloat Sfr_in_Kernel;
#endif
#if defined(SINK_GRAVACCRETION) && (SINK_GRAVACCRETION == 0)
    MyFloat MgasBulge_in_Kernel;
    MyFloat MstarBulge_in_Kernel;
#endif
#ifdef SINK_CALC_LOCAL_ANGLEWEIGHTS
    MyFloat Sink_angle_weighted_kernel_sum;
#endif
#ifdef SINK_REPOSITION_ON_POTMIN
    MyFloat DF_rms_vel; Vec3<MyFloat> DF_mean_vel; MyFloat DF_mmax_particles;
#endif
#if (SINK_GRAVACCRETION >= 5) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_TIMESTEPPING)
    Vec3<MyFloat> Sink_SurroundingGasVel;
#endif
#ifdef JET_DIRECTION_FROM_KERNEL_AND_SINK
    Vec3<MyFloat> Sink_SurroundingGasCOM;
#endif
#if (SINK_GRAVACCRETION == 8)
    MyFloat hubber_mdot_vr_estimator, hubber_mdot_disk_estimator, hubber_mdot_bondi_limiter;
#endif
#if defined(SINK_FOLLOW_ACCRETED_MOMENTUM)
    Vec3<MyDouble> accreted_momentum;        /*!< accreted linear momentum */
#endif
#if defined(SINK_RETURN_BFLUX)
    Vec3<MyDouble> accreted_B;
#endif    
#if defined(SINK_FOLLOW_ACCRETED_COM)
    Vec3<MyDouble> accreted_centerofmass;    /*!< accreted center-of-mass */
#endif    
#if defined(SINK_FOLLOW_ACCRETED_ANGMOM)
    Vec3<MyDouble> accreted_J;               /*!< accreted angular momentum */
#endif
#if defined(SINK_GRAVCAPTURE_GAS)
    MyFloat mass_to_swallow_edd;        /*!< gives the mass we want to swallow that contributes to eddington */
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
    Vec3<MyFloat> angmom_prepass_sum_for_passback; /*!< Normalization term for angular momentum feedback kicks, see denominator of Eq 22 of Hubber 2013 */
    MyFloat angmom_norm_topass_in_swallowloop;  /*!< corresponding scalar normalization calculated from the vector above */
#endif
#if defined(SINK_RETURN_BFLUX)
    MyFloat kernel_norm_topass_in_swallowloop;
#endif    
}
*SinkTempInfo;


/* sink_utils.c */
void sink_start(void);
void sink_end(void);
void sink_properties_loop(void);
double sink_eddington_mdot(double sink_mass);
double sink_lum_bol(double mdot, double mass, long pindex);
double evaluate_sink_radiative_efficiency(double mdot, double mass, long pindex);
double evaluate_sink_cosmicray_efficiency(double mdot, double mass, long pindex);

/* sinks.c */
void sink_final_operations(void);
int sink_isactive(int i);

/* sink_environment.c */
void sink_environment_loop(void);
#ifdef SINK_GRAVACCRETION
void sink_environment_second_loop(void);
#endif

/* sink_swallow_and_kick.c */
void sink_swallow_and_kick_loop(void);
double target_mass_for_wind_spawning(int i);

/* sink_feed.c */
void sink_feed_loop(void);

int sink_check_boundedness(int j, double vrel, double vesc, double dr_code, double sink_radius);
double sink_vesc(int j, double mass, double r_code, double sink_softening);
void set_sink_mdot(int i, int n, double dt);
void set_sink_new_mass(int i, int n, double dt);
void set_sink_drag(int i, int n, double dt);
void set_sink_long_range_rp(int i, int n);



#endif // top-level flag
