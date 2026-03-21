/* contains global type definitions */

typedef  long long integertime;
static MPI_Datatype MPI_TYPE_TIME = MPI_LONG_LONG;
#define  TIMEBINS        60
#define  TIMEBASE        (((integertime) 1)<<TIMEBINS)  /* The simulated timespan is mapped onto the integer interval [0,TIMESPAN], where TIMESPAN needs to be a power of 2. Note that (1<<28) corresponds to 2^29 */

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
