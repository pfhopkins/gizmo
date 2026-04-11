/* density_functions.h — GPU-callable density kernel functions (AthenaK pattern).
 *
 * Contains: struct kernel_density, struct density_evaluate_data_in_,
 *   struct density_evaluate_data_out_, density_accumulate_neighbor(),
 *   density_evaluate_extra_physics_gas().
 *
 * These are the per-neighbor-pair accumulation functions extracted from
 * density_evaluate() in density.cc. The h-iteration, tree walk, MPI dispatch,
 * and post-processing remain in density.cc.
 *
 * For CPU builds: density.cc includes this header with KOKKOS_INLINE_FUNCTION
 * redefined to empty, providing externally-visible symbols.
 *
 * For GPU builds: the GPU TU includes this header with KOKKOS_INLINE_FUNCTION
 * as __device__ __host__ inline, making these functions callable from kernels.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef DENSITY_FUNCTIONS_H
#define DENSITY_FUNCTIONS_H

/* Requires: allvars.h, proto.h, kernel.h included before this header. */

/* working struct for kernel quantities computed per neighbor pair */
struct kernel_density
{
    Vec3<double> dp; Vec3<double> dv; double r, wk, dwk, hinv, hinv3, hinv4, mj_wk, mj_dwk_r;
};


/* input struct: data sent from the 'searching' element */
struct density_evaluate_data_in_
{
    Vec3<MyDouble> Pos;
#if defined(SPHAV_CD10_VISCOSITY_SWITCH)
    Vec3<MyFloat> Accel;
#endif
    Vec3<MyFloat> Vel;
    MyFloat KernelRadius;
#ifdef GALSF_SUBGRID_WINDS
    MyFloat DelayTime;
#endif
    int NodeList[NODELISTLENGTH];
    int Type;
};


/* output struct: data sent back to the 'searching' element */
struct density_evaluate_data_out_
{
    MyDouble Ngb;
    MyDouble Rho;
    MyDouble DrkernNgb;
    MyDouble Particle_DivVel;
    SymmetricTensor2<MyDouble> NV_T;
    Vec3<MyDouble> NV_T_face_weights;
#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && ((HYDRO_FIX_MESH_MOTION==5)||(HYDRO_FIX_MESH_MOTION==6))
    Vec3<MyDouble> ParticleVel;
#endif
#ifdef HYDRO_SPH
    MyDouble DrkernHydroSumFactor;
#endif
#ifdef RT_SOURCE_INJECTION
    MyDouble KernelSum_Around_RT_Source;
#endif
#ifdef HYDRO_PRESSURE_SPH
    MyDouble EgyRho;
#endif
#if defined(SPHAV_CD10_VISCOSITY_SWITCH)
    MyFloat NV_D[3][3];
    MyFloat NV_A[3][3];
#endif
#ifdef DO_DENSITY_AROUND_NONGAS_PARTICLES
    Vec3<MyFloat> GradRho;
#endif
#if defined(SINK_PARTICLES)
    int Sink_TimeBinGasNeighbor;
#if defined(BH_ACCRETE_NEARESTFIRST) || defined(SINGLE_STAR_TIMESTEPPING)
    MyDouble Sink_dr_to_NearestGasNeighbor;
#endif
#endif
#if defined(TURB_DRIVING) || defined(GRAIN_FLUID)
    Vec3<MyDouble> GasVel;
#endif
#if defined(GRAIN_FLUID)
    MyDouble Gas_InternalEnergy;
#if defined(GRAIN_LORENTZFORCE)
    Vec3<MyDouble> Gas_B;
#endif
#endif
#ifdef HYDRO_PARTITION_UNITY_IMPROVE_FD
    Vec3<MyDouble> GradH_numer;
    MyDouble GradH_denom;
#endif
};


/* extra physics accumulation for non-self neighbor pairs (GRAIN_FLUID, SINK_PARTICLES, SPHAV_CD10, TURB_DRIVING, etc.) */
KOKKOS_INLINE_FUNCTION
void density_evaluate_extra_physics_gas(struct density_evaluate_data_in_ *local, struct density_evaluate_data_out_ *out,
                                        struct kernel_density *kernel, int j,
                                        struct particle_data *P, struct gas_cell_data *CellP)
{
    kernel->mj_dwk_r = P[j].Mass * kernel->dwk / kernel->r;

    if(local->Type != 0)
    {
#if defined(GRAIN_FLUID)
        if((1 << local->Type) & (GRAIN_PTYPES))
        {
            out->Gas_InternalEnergy += kernel->mj_wk * CellP[j].InternalEnergyPred;
            out->GasVel += kernel->mj_wk * (local->Vel - kernel->dv);
#if defined(GRAIN_LORENTZFORCE)
            out->Gas_B += kernel->wk * CellP[j].BPred;
#endif
        }
#endif

#if defined(SINK_PARTICLES)
        if(local->Type == 5)
        {
            /* NOTE: the shared-memory writes to P[j].SwallowID, P[j].SwallowTime, P[j].Sink_Ngb_Flag
               that exist in the CPU version (density.cc) are NOT included here because they require
               atomic writes and are not suitable for the GPU per-pair kernel. Those writes are handled
               separately in the CPU dispatch path (density.cc density_evaluate) or via a dedicated
               GPU pass. The read-only accumulations below are safe. */
            short int TimeBin_j = P[j].TimeBin; if(TimeBin_j < 0) {TimeBin_j = -TimeBin_j - 1;}
            if(out->Sink_TimeBinGasNeighbor > TimeBin_j) {out->Sink_TimeBinGasNeighbor = TimeBin_j;}
#if defined(SINGLE_STAR_TIMESTEPPING)
            double dr_eff_wtd = P[j].Get_Particle_Size();
            dr_eff_wtd=sqrt(dr_eff_wtd*dr_eff_wtd + (kernel->r)*(kernel->r));
            if((dr_eff_wtd < out->Sink_dr_to_NearestGasNeighbor) && (P[j].Mass > 0)) {out->Sink_dr_to_NearestGasNeighbor = dr_eff_wtd;}
#endif
        }
#endif
    } else { /* local->Type == 0 */

#if defined(TURB_DRIVING)
        out->GasVel += kernel->mj_wk * (local->Vel - kernel->dv);
#endif

#if defined(SPHAV_CD10_VISCOSITY_SWITCH)
        double wk = kernel->wk;
        out->NV_A[0][0] += (local->Accel[0] - All.cf_a2inv*P[j].GravAccel[0] - CellP[j].HydroAccel[0]) * kernel->dp[0] * wk;
        out->NV_A[0][1] += (local->Accel[0] - All.cf_a2inv*P[j].GravAccel[0] - CellP[j].HydroAccel[0]) * kernel->dp[1] * wk;
        out->NV_A[0][2] += (local->Accel[0] - All.cf_a2inv*P[j].GravAccel[0] - CellP[j].HydroAccel[0]) * kernel->dp[2] * wk;
        out->NV_A[1][0] += (local->Accel[1] - All.cf_a2inv*P[j].GravAccel[1] - CellP[j].HydroAccel[1]) * kernel->dp[0] * wk;
        out->NV_A[1][1] += (local->Accel[1] - All.cf_a2inv*P[j].GravAccel[1] - CellP[j].HydroAccel[1]) * kernel->dp[1] * wk;
        out->NV_A[1][2] += (local->Accel[1] - All.cf_a2inv*P[j].GravAccel[1] - CellP[j].HydroAccel[1]) * kernel->dp[2] * wk;
        out->NV_A[2][0] += (local->Accel[2] - All.cf_a2inv*P[j].GravAccel[2] - CellP[j].HydroAccel[2]) * kernel->dp[0] * wk;
        out->NV_A[2][1] += (local->Accel[2] - All.cf_a2inv*P[j].GravAccel[2] - CellP[j].HydroAccel[2]) * kernel->dp[1] * wk;
        out->NV_A[2][2] += (local->Accel[2] - All.cf_a2inv*P[j].GravAccel[2] - CellP[j].HydroAccel[2]) * kernel->dp[2] * wk;

        out->NV_D[0][0] += kernel->dv[0] * kernel->dp[0] * wk;
        out->NV_D[0][1] += kernel->dv[0] * kernel->dp[1] * wk;
        out->NV_D[0][2] += kernel->dv[0] * kernel->dp[2] * wk;
        out->NV_D[1][0] += kernel->dv[1] * kernel->dp[0] * wk;
        out->NV_D[1][1] += kernel->dv[1] * kernel->dp[1] * wk;
        out->NV_D[1][2] += kernel->dv[1] * kernel->dp[2] * wk;
        out->NV_D[2][0] += kernel->dv[2] * kernel->dp[0] * wk;
        out->NV_D[2][1] += kernel->dv[2] * kernel->dp[1] * wk;
        out->NV_D[2][2] += kernel->dv[2] * kernel->dp[2] * wk;
#endif

    } // Type == 0 check

#ifdef DO_DENSITY_AROUND_NONGAS_PARTICLES
    out->GradRho += kernel->mj_dwk_r * kernel->dp;
#endif
}


/* Per-neighbor-pair accumulation: processes one neighbor j for the searching particle described by 'local'.
   This is the inner body of the density_evaluate neighbor loop. On the CPU path, density_evaluate() calls
   this in a loop over tree-walk results. On the GPU path, the Kokkos kernel calls this in a loop over the
   CSR neighbor list. */
KOKKOS_INLINE_FUNCTION
void density_accumulate_neighbor(struct density_evaluate_data_in_ *local, struct density_evaluate_data_out_ *out,
                                 struct kernel_density *kernel, int j, double h2,
                                 struct particle_data *P, struct gas_cell_data *CellP)
{
#ifdef GALSF_SUBGRID_WINDS
    if(CellP[j].DelayTime > 0) {if(local->DelayTime <= 0) {return;}}
#endif
    if(P[j].Mass <= 0) return;
    kernel->dp = local->Pos - P[j].Pos;
    nearest_xyz(kernel->dp);
    double r2 = kernel->dp.norm_sq();
    if(r2 >= h2) return; /* only consider particles inside local.KernelRadius */

    kernel->r = sqrt(r2);
    double u = kernel->r * kernel->hinv;
    kernel_main(u, kernel->hinv3, kernel->hinv4, &kernel->wk, &kernel->dwk, 0);
    double mass_j = P[j].Mass;
    kernel->mj_wk = (mass_j * kernel->wk);

    out->Ngb += kernel->wk;
    out->Rho += kernel->mj_wk;
#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && ((HYDRO_FIX_MESH_MOTION==5)||(HYDRO_FIX_MESH_MOTION==6))
    if(local->Type == 0 && kernel->r==0) {out->ParticleVel += kernel->mj_wk * CellP[j].VelPred;}
#endif
#if defined(RT_SOURCE_INJECTION)
#if defined(RT_SINK_ANGLEWEIGHT_PHOTON_INJECTION)
    if(All.TimeStep == 0)
#endif
    if((1 << local->Type) & (RT_SOURCES)) {out->KernelSum_Around_RT_Source += 1.-u*u;}
#endif
    out->DrkernNgb += -(NUMDIMS * kernel->hinv * kernel->wk + u * kernel->dwk);
#ifdef HYDRO_SPH
    double mass_eff = mass_j;
#ifdef HYDRO_PRESSURE_SPH
    mass_eff *= CellP[j].InternalEnergyPred;
    out->EgyRho += kernel->wk * mass_eff;
#endif
    out->DrkernHydroSumFactor += -mass_eff * (NUMDIMS * kernel->hinv * kernel->wk + u * kernel->dwk);
#endif

    /* for everything below, do NOT include the particle self-contribution */
    if(kernel->r > 0)
    {
        if(local->Type == 0)
        {
            double wk = kernel->wk;
            out->NV_T += wk * outer_product(kernel->dp);
            out->NV_T_face_weights += wk * kernel->dp;
#ifdef HYDRO_PARTITION_UNITY_IMPROVE_FD
            out->GradH_numer += (kernel->dwk / kernel->r) * kernel->dp;
            out->GradH_denom += u * kernel->dwk;
#endif
        }
        kernel->dv = local->Vel - CellP[j].VelPred;
        NGB_SHEARBOX_BOUNDARY_VELCORR_(local->Pos,P[j].Pos,kernel->dv,1);
#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && ((HYDRO_FIX_MESH_MOTION==5)||(HYDRO_FIX_MESH_MOTION==6))
        out->ParticleVel += kernel->mj_wk * (local->Vel - kernel->dv);
#endif
        out->Particle_DivVel -= kernel->dwk * dot(kernel->dp, kernel->dv) / kernel->r;

        density_evaluate_extra_physics_gas(local, out, kernel, j, P, CellP);
    } // kernel->r > 0
}

#endif /* DENSITY_FUNCTIONS_H */
