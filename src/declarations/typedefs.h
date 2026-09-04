/* contains global type definitions */
#pragma once

typedef  long long integertime;
#ifndef GIZMO_GPU_COMPILER
static MPI_Datatype MPI_TYPE_TIME = MPI_LONG_LONG;
#endif
#define  TIMEBINS        60
#define  TIMEBASE        (((integertime) 1)<<TIMEBINS)  /* The simulated timespan is mapped onto the integer interval [0,TIMESPAN], where TIMESPAN needs to be a power of 2. Note that (1<<28) corresponds to 2^29 */

/* Position->key discretization contract: every site that maps a position to a Peano/Morton key
 * (domain.cc x4, system/peano.cc, gravity/gpu_morton.cc, gravity/gpu_topology_build.cc) computes
 * the IDENTICAL expression ((Pos - DomainCorner)/DomainLen) + 1.0 into the identical mantissa
 * extraction. Sub/div/add are each correctly rounded and the expression has no contraction or
 * reassociation freedom, so every copy is bit-identical under any conforming optimization -- the
 * duplication is safe EXACTLY as long as unsafe FP transforms stay off. -freciprocal-math (inside
 * -ffast-math) licenses div -> x*(1/L), which shifts keys by one ulp-window per ~2e4 evaluations;
 * when the exchange and the tree build disagree, the particle is DETACHED from every rank's tree
 * and silently exerts no gravity that step (measured: ~15 dropped-star steps per fewbody run,
 * burst energy errors up to 50%). Hence: */
#ifdef __FAST_MATH__
#error "-ffast-math breaks the position->key contract (reciprocal-math re-splits domain exchange vs tree build; particles get silently detached at np>=2). Route every key computation through one noinline helper (starforge_dev 0614e00a) before re-enabling."
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
