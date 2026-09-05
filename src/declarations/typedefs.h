/* contains global type definitions */
#pragma once

typedef  long long integertime;
#ifndef GIZMO_GPU_COMPILER
static MPI_Datatype MPI_TYPE_TIME = MPI_LONG_LONG;
#endif
#define  TIMEBINS        60
#define  TIMEBASE        (((integertime) 1)<<TIMEBINS)  /* The simulated timespan is mapped onto the integer interval [0,TIMESPAN], where TIMESPAN needs to be a power of 2. Note that (1<<28) corresponds to 2^29 */

/* Position->key contract. Several places map a position to a Peano/Morton key, and every one of
 * them must produce the same integer for the same position: the domain exchange decides which rank
 * owns a particle from its key, and the tree build decides where to insert it from the key it
 * computes itself. If those disagree the particle is owned by one rank and inserted by none, so it
 * silently exerts no gravity for that step.
 *
 * Sub, divide and add are each correctly rounded, so the expression has no freedom -- as long as
 * unsafe floating-point transformations stay off. -ffast-math does not keep them off: it permits
 * both the reciprocal substitution and fused multiply-add, and the compiler applies them where it
 * can vectorise and not where it cannot, so the SAME expression in a loop and in a scalar call
 * stop agreeing. Measured on this code's own flags: 1453 differing keys per 6 million coordinates,
 * always in the last bit. Writing the expression once in a shared inline does NOT fix it -- the
 * inline is expanded into both contexts and then transformed differently in each.
 *
 * So the contract is enforced here rather than defended at each site. */
#ifdef __FAST_MATH__
#error "-ffast-math breaks the position->key contract: it licenses reciprocal and fused-multiply-add rewrites, so the domain exchange and the tree build can compute different keys for the same particle and silently detach it from every rank's tree. Remove it from this systype's OPTIMIZE in the Makefile."
#endif

#define  BITS_PER_DIMENSION 42    /* for Peano-Hilbert order. Note: Maximum is 10 to fit in 32-bit integer, 21 for 64-bit integer, 42 for 128-bit integer */
#define  PEANOCELLS (((peanokey)1)<<(3*BITS_PER_DIMENSION))
#if(BITS_PER_DIMENSION <= 21)
typedef unsigned long long peanokey;
typedef unsigned int peano1D;
#else
typedef __int128 peanokey;
typedef unsigned long long peano1D;
#endif


typedef unsigned long long MyIDType;
typedef double   MyFloat;
typedef double  MyDouble;

#ifdef GIZMO_MIXED_PRECISION_GRAVITY
typedef float  MyGravFloat;
#else
typedef double MyGravFloat;
#endif

#ifdef OUTPUT_IN_DOUBLEPRECISION
typedef double MyOutputFloat;
#else
typedef float MyOutputFloat;
#endif
#ifdef INPUT_IN_DOUBLEPRECISION
typedef double MyInputFloat;
#else
typedef float MyInputFloat;
#endif


typedef double MyOutputPosFloat;
#ifdef INPUT_POSITIONS_IN_DOUBLE
typedef double MyInputPosFloat;
#else
typedef MyInputFloat MyInputPosFloat;
#endif

struct unbind_data
{
    int index;
};


#define DEFAULT_PATH_BUFFERSIZE_TOUSE 512
#define MAX_PATH_BUFFERSIZE_TOUSE 2048
