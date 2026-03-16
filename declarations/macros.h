/*------- PRECOMPILER MACROS DEFINED BELOW -------*/


#define ASSIGN_ADD(x,y,mode) (mode == 0 ? (x=y) : (x+=y))


#ifdef USE_TIMESTEP_DILATION_FOR_ZOOMS
#define TIMESTEP_DILATION_FACTOR(i,mode) (return_timestep_dilation_factor(i,mode))
#else
#define TIMESTEP_DILATION_FACTOR(i,mode) (1)
#endif
#define UNIT_INTEGERTIME_IN_PHYSICAL(i) ((All.Timebase_interval/All.cf_hubble_a) * TIMESTEP_DILATION_FACTOR(i,0))
#define GET_INTEGERTIME_FROM_TIMEBIN(bin) ((bin ? (((integertime) 1) << bin) : 0))
#define GET_PHYSICAL_TIMESTEP_FROM_TIMEBIN(bin, i) ((GET_INTEGERTIME_FROM_TIMEBIN(bin) * UNIT_INTEGERTIME_IN_PHYSICAL(i)))
#ifndef WAKEUP
#define GET_PARTICLE_INTEGERTIME(i) ((GET_INTEGERTIME_FROM_TIMEBIN(P[i].TimeBin)))
#else
#define GET_PARTICLE_INTEGERTIME(i) ((P[i].dt_step))
#endif
#define GET_PARTICLE_TIMESTEP_IN_PHYSICAL(i) ((GET_PARTICLE_INTEGERTIME(i) * UNIT_INTEGERTIME_IN_PHYSICAL(i)))

#ifdef GALSF_LIMIT_FBTIMESTEPS_FROM_BELOW
#define GET_PARTICLE_FEEDBACK_TIMESTEP_IN_PHYSICAL(i) DMAX(GET_PARTICLE_TIMESTEP_IN_PHYSICAL(i), All.Dt_Min_Between_FBCalc_Gyr/UNIT_TIME_IN_GYR)
#else
#define GET_PARTICLE_FEEDBACK_TIMESTEP_IN_PHYSICAL(i) GET_PARTICLE_TIMESTEP_IN_PHYSICAL(i)
#endif

#ifdef DILATION_FOR_STELLAR_KINEMATICS_ONLY
#undef GET_PARTICLE_FEEDBACK_TIMESTEP_IN_PHYSICAL
#define GET_PARTICLE_FEEDBACK_TIMESTEP_IN_PHYSICAL(i) (GET_PARTICLE_INTEGERTIME(i) * (All.Timebase_interval/All.cf_hubble_a)) /* this timestep does -not- involve dilation */
#endif



#ifndef DISABLE_MEMORY_MANAGER

#define  mymalloc(x, y)            mymalloc_fullinfo(x, y, __FUNCTION__, __FILE__, __LINE__)
#define  mymalloc_movable(x, y, z) mymalloc_movable_fullinfo(x, y, z, __FUNCTION__, __FILE__, __LINE__)
#define  myrealloc(x, y)           myrealloc_fullinfo(x, y, __FUNCTION__, __FILE__, __LINE__)
#define  myrealloc_movable(x, y)   myrealloc_movable_fullinfo(x, y, __FUNCTION__, __FILE__, __LINE__)
#define  myfree(x)                 myfree_fullinfo(x, __FUNCTION__, __FILE__, __LINE__)
#define  myfree_movable(x)         myfree_movable_fullinfo(x, __FUNCTION__, __FILE__, __LINE__)
#define  report_memory_usage(x, y) report_detailed_memory_usage_of_largest_task(x, y, __FUNCTION__, __FILE__, __LINE__)

// compiler specific data alignment hints: use only with memory manager as malloc'd memory is not sufficiently aligned
// (experimenting right now with removing this, as many compilers internal AVX optimizations appear to be doing marginally better, and can resolve crashes on some compilers)
#if defined(__xlC__) // XLC compiler
#define ALIGN(n) __attribute__((__aligned__(n)))
#elif defined(__GNUC__) // GNU compiler
#define ALIGN(n) __attribute__((__aligned__(n)))
#elif defined(__INTEL_COMPILER) // Intel Compiler
#define ALIGN(n) __declspec(align(n))
#endif

#else

#define  mymalloc(x, y)            malloc(y)
#define  mymalloc_movable(x, y, z) malloc(z)
#define  myrealloc(x, y)           realloc(x, y)
#define  myrealloc_movable(x, y)   realloc(x, y)
#define  myfree(x)                 free(x)
#define  myfree_movable(x)         free(x)
#define  report_memory_usage(x, y) printf("Memory manager disabled.\n")

#endif

#ifndef ALIGN // Unknown Compiler or using default malloc
#define ALIGN(n)
#endif





/****************************************************************************************************************************/
/* Here we define the box-wrapping macros NEAREST_XYZ and NGB_PERIODIC_BOX_LONG_X,NGB_PERIODIC_BOX_LONG_Y,NGB_PERIODIC_BOX_LONG_Z.
 *   The inputs to these functions are (dx_position, dy_position, dz_position, sign), where
 *     'sign' = -1 if dx_position = x_test_point - x_reference (reference = particle from which we are doing a calculation),
 *     'sign' = +1 if dx_position = x_reference - x_test_point
 *
 *   For non-periodic cases these functions are trivial (do nothing, or just take absolute values).
 *
 *   For standard periodic cases it will wrap in each dimension, allowing for a different box length in X/Y/Z.
 *      here the "sign" term is irrelevant. Also NGB_PERIODIC_BOX_LONG_X, NGB_PERIODIC_BOX_LONG_Y, NGB_PERIODIC_BOX_LONG_Z will each
 *      compile to only use the x,y, or z information, but all four inputs are required for the sake of completeness
 *      and consistency.
 *
 *   The reason for the added complexity is for shearing boxes. In this case, the Y(phi)-coordinate for particles being
 *      wrapped in the X(r)-direction must be modified by a time-dependent term. It also matters for the sign of that
 *      term "which side" of the box we are wrapping across (i.e. does the 'virtual particle' -- the test point which is
 *      not the particle for which we are currently calculating forces, etc -- lie on the '-x' side or the '+x' side)
 *      (note after all that: if very careful, sign -cancels- within the respective convention, for the type of wrapping below)
 */
/****************************************************************************************************************************/

#if defined(BOX_PERIODIC) && !(defined(BOX_REFLECT_X) || defined(BOX_OUTFLOW_X)) // x-axis is periodic
#define TMP_WRAP_X_S(x,y,z,sign) (x=((x)>boxHalf_X)?((x)-boxSize_X):(((x)<-boxHalf_X)?((x)+boxSize_X):(x))) /* normal (signed) periodic wrap */
#define NGB_PERIODIC_BOX_LONG_X(x,y,z,sign) (xtmp=fabs(x),(xtmp>boxHalf_X)?(boxSize_X-xtmp):xtmp) /* absolute value of normal periodic wrap */
#else // x-axis is non-periodic
#define TMP_WRAP_X_S(x,y,z,sign) /* this is an empty macro: nothing will happen to the variables input here */
#define NGB_PERIODIC_BOX_LONG_X(x,y,z,sign) (fabs(x)) /* simple absolute value */
#endif

#if defined(BOX_PERIODIC) && !(defined(BOX_REFLECT_Z) || defined(BOX_OUTFLOW_Z)) // z-axis is periodic
#define TMP_WRAP_Z_S(x,y,z,sign) (z=((z)>boxHalf_Z)?((z)-boxSize_Z):(((z)<-boxHalf_Z)?((z)+boxSize_Z):(z))) /* normal (signed) periodic wrap */
#define NGB_PERIODIC_BOX_LONG_Z(x,y,z,sign) (xtmp=fabs(z),(xtmp>boxHalf_Z)?(boxSize_Z-xtmp):xtmp) /* absolute value of normal periodic wrap */
#else // z-axis is non-periodic
#define TMP_WRAP_Z_S(x,y,z,sign) /* this is an empty macro: nothing will happen to the variables input here */
#define NGB_PERIODIC_BOX_LONG_Z(x,y,z,sign) (fabs(z)) /* simple absolute value */
#endif

#if defined(BOX_PERIODIC) && !(defined(BOX_REFLECT_Y) || defined(BOX_OUTFLOW_Y)) // y-axis is periodic
#if (BOX_SHEARING > 1) // Shearing Periodic Box:: in this case, we have a shearing box with the '1' coordinate being phi, so there is a periodic extra wrap

#define TMP_WRAP_Y_S(x,y,z,sign) (\
y += Shearing_Box_Pos_Offset * (((x)>boxHalf_X)?(1):(((x)<-boxHalf_X)?(-1):(0))),\
y = ((y)>boxSize_Y)?((y)-boxSize_Y):(((y)<-boxSize_Y)?((y)+boxSize_Y):(y)),\
y=((y)>boxHalf_Y)?((y)-boxSize_Y):(((y)<-boxHalf_Y)?((y)+boxSize_Y):(y))) /* shear-periodic wrap in y, accounting for the position offset needed for azimuthal wrap off the radial axis */

#define NGB_PERIODIC_BOX_LONG_Y(x,y,z,sign) (\
xtmp = y + Shearing_Box_Pos_Offset * (((x)>boxHalf_X)?(1):(((x)<-boxHalf_X)?(-1):(0))),\
xtmp = fabs(((xtmp)>boxSize_Y)?((xtmp)-boxSize_Y):(((xtmp)<-boxSize_Y)?((xtmp)+boxSize_Y):(xtmp))),\
(xtmp>boxHalf_Y)?(boxSize_Y-xtmp):xtmp) /* shear periodic wrap in y, accounting for the position offset needed for azimuthal wrap off the radial axis: absolute value here */

#else // 'normal' periodic y-axis, nothing special
#define TMP_WRAP_Y_S(x,y,z,sign) (y=((y)>boxHalf_Y)?((y)-boxSize_Y):(((y)<-boxHalf_Y)?((y)+boxSize_Y):(y))) /* normal (signed) periodic wrap */
#define NGB_PERIODIC_BOX_LONG_Y(x,y,z,sign) (xtmp=fabs(y),(xtmp>boxHalf_Y)?(boxSize_Y-xtmp):xtmp) /* absolute value of normal periodic wrap */
#endif
#else // y-axis is non-periodic
#define TMP_WRAP_Y_S(x,y,z,sign) /* this is an empty macro: nothing will happen to the variables input here */
#define NGB_PERIODIC_BOX_LONG_Y(x,y,z,sign) (fabs(y)) /* simple absolute value */
#endif

#define NEAREST_XYZ(x,y,z,sign) {\
TMP_WRAP_Y_S(x,y,z,sign);\
TMP_WRAP_X_S(x,y,z,sign);\
TMP_WRAP_Z_S(x,y,z,sign);} /* note the ORDER MATTERS here for shearing boxes: Y-wrap must precede x/z wrap to allow correct re-assignment. collect the box-wrapping terms into one function here */





/* this function, like the NEAREST and NGB_PERIODIC functions above, does -velocity wrapping- for periodic boundary
 conditions. this is currently only relevant for shearing boxes, where the box ends in the '0' axis direction have
 systematically different (shear-periodic instead of periodic) velocities associated, so the box needs to be able to
 know how to wrap them. this takes the vector of positions of particle "i" pos_i (the particle "seeing" particle j),
 particle j position pos_j, the velocity difference vector dv_ij=v_i-v_j. last  dv_sign_flipped = 1 if dv_ij=v_i-v_j,
 but dv_sign_flipped=-1 if dv_ij=v_j-v_i (flipped from normal order) */
#ifdef BOX_SHEARING
#define NGB_SHEARBOX_BOUNDARY_VELCORR_(pos_i,pos_j,dv_ij,dv_sign_flipped) (dv_ij[BOX_SHEARING_PHI_COORDINATE] += dv_sign_flipped*Shearing_Box_Vel_Offset * ((pos_i[0]-pos_j[0]>boxHalf_X)?(1):((pos_i[0]-pos_j[0]<-boxHalf_X)?(-1):(0))))
#else
#define NGB_SHEARBOX_BOUNDARY_VELCORR_(pos_i,pos_j,dv_ij,dv_sign_flipped)
#endif

/* equivalent for shear-periodic magnetic fields */
#define NGB_SHEARBOX_BOUNDARY_BCORR_(pos_i,pos_j,db_ij,db_sign_flipped)




/*****************************************************************/
/*  Utility functions used for printing status, warning, endruns */
/*****************************************************************/

#define terminate(x) {char termbuf[MAX_PATH_BUFFERSIZE_TOUSE]; snprintf(termbuf, MAX_PATH_BUFFERSIZE_TOUSE, "TERMINATE issued on task=%d, function '%s()', file '%s', line %d: '%s'\n", ThisTask, __FUNCTION__, __FILE__, __LINE__, x); fflush(stdout); printf("%s", termbuf); fflush(stdout); MPI_Abort(MPI_COMM_WORLD, 1); exit(0);}
#define endrun(x) {if(x==0) {MPI_Finalize(); exit(0);} else {char termbuf[MAX_PATH_BUFFERSIZE_TOUSE]; snprintf(termbuf, MAX_PATH_BUFFERSIZE_TOUSE, "ENDRUN issued on task=%d, function '%s()', file '%s', line %d: error level %d\n", ThisTask, __FUNCTION__, __FILE__, __LINE__, x); fflush(stdout); printf("%s", termbuf); fflush(stdout); MPI_Abort(MPI_COMM_WORLD, x); exit(0);}}
#define PRINT_WARNING(...) {char termbuf1[MAX_PATH_BUFFERSIZE_TOUSE], termbuf2[MAX_PATH_BUFFERSIZE_TOUSE]; snprintf(termbuf1, MAX_PATH_BUFFERSIZE_TOUSE, "WARNING issued on task=%d, function %s(), file %s, line %d", ThisTask, __FUNCTION__, __FILE__, __LINE__); snprintf(termbuf2, MAX_PATH_BUFFERSIZE_TOUSE, __VA_ARGS__); fflush(stdout); printf("%s: %s\n", termbuf1, termbuf2); fflush(stdout);}
#ifndef OUTPUT_ADDITIONAL_RUNINFO
#define PRINT_STATUS(...) {if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin) {if(ThisTask==0) {fflush(stdout); printf( __VA_ARGS__ ); printf("\n"); fflush(stdout);}}}
#else
#define PRINT_STATUS(...) {if(ThisTask==0) {fflush(stdout); printf( __VA_ARGS__ ); printf("\n"); fflush(stdout);}}
#endif


/* macro name concatenation for our modular precompiler method of writing new loops easily */

#define MACRO_NAME_CONCATENATE(A, B) MACRO_NAME_CONCATENATE_(A, B)
#define MACRO_NAME_CONCATENATE_(A, B) A##B
