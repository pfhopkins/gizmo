/* minimal stand-in for GIZMO's declarations, enough to compile galaxy_sf/stellar_evolution_tables.cc on its own */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hdf5.h>
#include <string>
#include <vector>
#define DMIN(a,b) ((a) < (b) ? (a) : (b))
#define DMAX(a,b) ((a) > (b) ? (a) : (b))
struct stub_all {char StellarTablesFile[256]; double IMFSampleMinMass;}; /* the reference implementation reads IMFSampleMinMass */
extern struct stub_all All;
extern int ThisTask, NTask;
extern MPI_Comm GizmoNodeComm;
extern int GizmoNodeRankOfTask;
void gizmo_node_comm_init(void);
void gizmo_request_controlled_stop(int code, const char *reason, const char *file, int line, const char *func);
static inline void *stub_mymalloc(size_t n) {return malloc(n);}
#define mymalloc(name, n) stub_mymalloc(n)
#define myfree(p) free(p)
static inline void endrun(int c) {printf("endrun(%d)\n", c); MPI_Abort(MPI_COMM_WORLD, c);}
#define GIZMO_MEM_SINGLE_STARS 0
static inline void gizmo_mem_account_add(int family, long long bytes) {(void)family; (void)bytes;}
