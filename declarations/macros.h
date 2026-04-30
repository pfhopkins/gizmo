/*------- PRECOMPILER MACROS DEFINED BELOW -------*/


#define ASSIGN_ADD(x,y,mode) (mode == 0 ? (x=y) : (x+=y))


#define GET_INTEGERTIME_FROM_TIMEBIN(bin) ((bin ? (((integertime) 1) << bin) : 0))
/* Timestep macros replaced by functions in timestep.cc (all take optional pp=P param) — see proto.h */



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

/* ---- GPU portability layer ----
   GIZMO_GPU_COMPILER: true when compiled by any GPU device compiler (nvcc, hipcc).
   Primary definition is in global_data_all_struct.h (earliest universal header).
   Redundant guard here ensures it is defined even if macros.h is included first.
   GIZMO_GPU_FUNCTION: marks functions __device__ __host__ on GPU, nothing on CPU. */
#if (defined(__CUDACC__) || defined(__HIPCC__)) && !defined(GIZMO_GPU_COMPILER)
#define GIZMO_GPU_COMPILER
#endif
#ifdef GIZMO_GPU_COMPILER
#define GIZMO_GPU_FUNCTION __device__ __host__
#define GIZMO_GPU_DEVICE   __device__           /* for variables (not functions) that must be device-accessible */
#else
#define GIZMO_GPU_FUNCTION
#define GIZMO_GPU_DEVICE
#endif
/* KOKKOS_FUNCTION / KOKKOS_INLINE_FUNCTION: defined by Kokkos headers when
   Kokkos is active.  On non-GPU builds, define to nothing/inline
   so code annotated with these compiles without Kokkos. */
#ifndef KOKKOS_FUNCTION
#define KOKKOS_FUNCTION
#endif
#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif
/* Kokkos memory space abstraction: Kokkos::SharedSpace is backend-agnostic
   (maps to CudaUVMSpace on NVIDIA, HIPManagedSpace on AMD).  Define a
   convenience alias so allocation code doesn't hardcode a backend. */
#define GIZMO_KOKKOS_SHARED_SPACE Kokkos::SharedSpace

/* DMAX/DMIN/IMAX/IMIN as macros: the static inline function versions in proto.h
   lack __device__ annotations and nvcc silently stubs them to return 0 on device.
   These macro versions expand directly into the source and work on both host and
   device with no linkage or type-matching issues. They are defined here (macros.h)
   so they are visible to all TUs before proto.h is included. proto.h still has the
   inline function definitions for backward compatibility with any code that takes
   their address, but the macros take precedence at call sites. */
#define DMAX(a,b) ((a) > (b) ? (a) : (b))
#define DMIN(a,b) ((a) < (b) ? (a) : (b))
#define IMAX(a,b) ((a) > (b) ? (a) : (b))
#define IMIN(a,b) ((a) < (b) ? (a) : (b))

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
/* GPU-device endrun/PRINT_WARNING.  Device code cannot call MPI_Abort, but it
   must not continue after a fatal physics/error path.  Print the source
   location, then trap the kernel so the host post-kernel error check sees the
   failure. */
#if defined(__CUDA_ARCH__)
#define GIZMO_GPU_DEVICE_TRAP() asm volatile("trap;")
#else
#define GIZMO_GPU_DEVICE_TRAP() __builtin_trap()
#endif
#define endrun(x) do { \
    printf("ENDRUN: file '%s', line %d, error %d\n", __FILE__, __LINE__, (x)); \
    GIZMO_GPU_DEVICE_TRAP(); \
    } while(0)
#define PRINT_WARNING(...) do { printf(__VA_ARGS__); printf("\n"); } while(0)
#else
#define endrun(x) {if(x==0) {MPI_Finalize(); exit(0);} else {char termbuf[MAX_PATH_BUFFERSIZE_TOUSE]; snprintf(termbuf, MAX_PATH_BUFFERSIZE_TOUSE, "ENDRUN issued on task=%d, function '%s()', file '%s', line %d: error level %d\n", ThisTask, __FUNCTION__, __FILE__, __LINE__, x); fflush(stdout); printf("%s", termbuf); fflush(stdout); MPI_Abort(MPI_COMM_WORLD, x); exit(0);}}
#define PRINT_WARNING(...) {char termbuf1[MAX_PATH_BUFFERSIZE_TOUSE], termbuf2[MAX_PATH_BUFFERSIZE_TOUSE]; snprintf(termbuf1, MAX_PATH_BUFFERSIZE_TOUSE, "WARNING issued on task=%d, function %s(), file %s, line %d", ThisTask, __FUNCTION__, __FILE__, __LINE__); snprintf(termbuf2, MAX_PATH_BUFFERSIZE_TOUSE, __VA_ARGS__); fflush(stdout); printf("%s: %s\n", termbuf1, termbuf2); fflush(stdout);}
#endif
#ifndef OUTPUT_ADDITIONAL_RUNINFO
#define PRINT_STATUS(...) {if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin) {if(ThisTask==0) {fflush(stdout); printf( __VA_ARGS__ ); printf("\n"); fflush(stdout);}}}
#else
#define PRINT_STATUS(...) {if(ThisTask==0) {fflush(stdout); printf( __VA_ARGS__ ); printf("\n"); fflush(stdout);}}
#endif


/* macro name concatenation for our modular precompiler method of writing new loops easily */

#define MACRO_NAME_CONCATENATE(A, B) MACRO_NAME_CONCATENATE_(A, B)
#define MACRO_NAME_CONCATENATE_(A, B) A##B
