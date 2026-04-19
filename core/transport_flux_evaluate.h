/* transport_flux_evaluate.h — core function for the transport subcycle flux exchange.
 *
 * This is #include'd into transport_subcycle.cc inside the code_block_xchange framework.
 * It follows the same structure as hydro_evaluate.h but only computes RT transport fluxes.
 * The face geometry (MFM face areas) is recomputed from scratch using the same kernel
 * and face area routines as the hydro pass.
 */

int transport_flux_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)
{
    int startnode, numngb, listindex = 0, j, k, n;
    struct INPUT_STRUCT_NAME local;
    struct OUTPUT_STRUCT_NAME out;
    memset(&out, 0, sizeof(struct OUTPUT_STRUCT_NAME));

    if(mode == 0) { particle2in_transport(&local, target, loop_iteration); }
    else { local = DATAGET_NAME[target]; }
    if(local.Mass <= 0 || local.Density <= 0) return 0;

    struct kernel_hydra kernel;
    memset(&kernel, 0, sizeof(struct kernel_hydra));
    kernel.h_i = local.KernelRadius;
    kernel.sound_i = local.SoundSpeed;

    double hinv_i, hinv3_i, hinv4_i, V_i, Particle_Size_i, dt_hydrostep_i, dt_hydrostep_j, dt_hydrostep;
    double face_vel_i=0, face_vel_j=0, Face_Area_Norm=0;
    Vec3<double> Face_Area_Vec;

    hinv_i = 1.0/kernel.h_i;
    hinv3_i = hinv_i*hinv_i*hinv_i;
    hinv4_i = hinv3_i*hinv_i;
    V_i = local.Mass / local.Density;
    Particle_Size_i = pow(V_i, 1./NUMDIMS) * All.cf_atime;

    if(mode == 0) { dt_hydrostep_i = get_particle_timestep_in_physical(target); }
    else { dt_hydrostep_i = All.MaxSizeTimestep; /* conservative fallback for imported particles */ }

    /* condition number criterion for face area computation */
    double cnumcrit2 = ((double)CONDITION_NUMBER_DANGER)*((double)CONDITION_NUMBER_DANGER) - local.ConditionNumber*local.ConditionNumber;

    /* RT-specific setup */
#if defined(RT_SOLVER_EXPLICIT)
    double tau_c_i[N_RT_FREQ_BINS];
    for(k=0; k<N_RT_FREQ_BINS; k++) tau_c_i[k] = Particle_Size_i * local.Rad_Kappa[k] * local.Density * All.cf_a3inv;
#endif

    if(mode == 0) { startnode = All.MaxPart; }
    else { startnode = DATAGET_NAME[target].NodeList[0]; startnode = Nodes[startnode].u.d.nextnode; }

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb = ngb_treefind_pairs_threads(local.Pos, kernel.h_i, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist);
            if(numngb < 0) return -2;

            for(n = 0; n < numngb; n++)
            {
                j = ngblist[n];
                if(P[j].Mass <= 0 || CellP[j].Density <= 0) continue;

                /* compute kernel geometry */
                kernel.dp = P[target].Pos - P[j].Pos;
                if(mode == 1) { for(k=0;k<3;k++) kernel.dp[k] = local.Pos[k] - P[j].Pos[k]; }
                NEAREST_XYZ(kernel.dp[0], kernel.dp[1], kernel.dp[2], -1);
                double r2 = kernel.dp.norm_sq();
                kernel.h_j = P[j].KernelRadius;
                if(r2 <= 0 || (r2 > kernel.h_i*kernel.h_i && r2 > kernel.h_j*kernel.h_j)) continue;
                kernel.r = sqrt(r2);
                double rinv = 1.0/kernel.r;

                double hinv_j = 1.0/kernel.h_j;
                double u_i = kernel.r * hinv_i, u_j = kernel.r * hinv_j;
                kernel.wk_i=0; kernel.wk_j=0; kernel.dwk_i=0; kernel.dwk_j=0;
                if(u_i < 1) kernel_main(u_i, hinv3_i, hinv4_i, &kernel.wk_i, &kernel.dwk_i, -1);
                if(u_j < 1) kernel_main(u_j, hinv_j*hinv_j*hinv_j, hinv_j*hinv_j*hinv_j*hinv_j, &kernel.wk_j, &kernel.dwk_j, -1);

                double V_j = P[j].Mass / CellP[j].Density;
                double Particle_Size_j = P[j].Get_Particle_Size() * All.cf_atime;

                /* compute MFM face area — standard face area computation */
                double rinv_soft = rinv;
                {
                    double Vi_inv_corr_unused, Vj_inv_corr_unused;
                    compute_finitevol_faces(local, CellP[j], kernel, rinv, r2, V_i, V_j,
                                            Particle_Size_i, Particle_Size_j, cnumcrit2,
                                            Face_Area_Vec, Face_Area_Norm,
                                            Vi_inv_corr_unused, Vj_inv_corr_unused);
                }
                if(Face_Area_Norm <= 0) continue;

                /* timestep for flux limiting */
                dt_hydrostep_j = get_particle_timestep_in_physical(j);
                dt_hydrostep = DMAX(dt_hydrostep_i, dt_hydrostep_j) * All.Transport_Subcycle_dt_fraction; /* use sub-step dt for stability limiting — each sub-step must be stable independently */

                /* face velocities for HLL */
                face_vel_i = dot(local.Vel, kernel.dp) * rinv / All.cf_atime;
                face_vel_j = dot(CellP[j].VelPred, kernel.dp) * rinv / All.cf_atime;

                /* signal velocity — must match hydro_evaluate.h for consistent flux computation */
                kernel.sound_j = CellP[j].effective_soundspeed();
                kernel.vsig = kernel.sound_i + kernel.sound_j; /* sum of sound speeds, same as hydro pass line 190 */

                Vec3<MyDouble> VelPred_j = CellP[j].VelPred;
                Vec3<MyDouble> ParticleVel_j = {};
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
                ParticleVel_j = P[j].Vel;
#endif

                {
                    const double *tau_c_i_ptr = nullptr;
#if defined(RT_SOLVER_EXPLICIT) && (N_RT_FREQ_BINS > 0)
                    tau_c_i_ptr = tau_c_i;
#endif
                    rt_direct_ray_transport_compute_pair(local, P[j], CellP[j], VelPred_j, ParticleVel_j,
                                                         Face_Area_Vec, Face_Area_Norm, V_i, V_j, Particle_Size_j,
                                                         tau_c_i_ptr, dt_hydrostep, out);
                    rt_diffusion_explicit_compute_pair(local, j, P, CellP, VelPred_j, ParticleVel_j,
                                                       kernel, rinv, Face_Area_Vec, Face_Area_Norm,
                                                       V_i, V_j, Particle_Size_i, Particle_Size_j,
                                                       face_vel_i, face_vel_j, tau_c_i_ptr, dt_hydrostep, out);
                }

            } /* for(n = 0; n < numngb; n++) */
        } /* while(startnode >= 0) inner */
        if(mode == 1) {
            listindex++;
            if(listindex < NODELISTLENGTH) {
                startnode = DATAGET_NAME[target].NodeList[listindex];
                if(startnode >= 0) startnode = Nodes[startnode].u.d.nextnode;
            }
        }
    } /* while(startnode >= 0) outer */

    if(mode == 0) {out2particle_transport(&out, target, 0, loop_iteration);} else {DATARESULT_NAME[target] = out;}
    return 0;
}
