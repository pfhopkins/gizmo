#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>

#include "../declarations/allvars.h"
#include "../core/proto.h"




/*!
 * This file was originally part of the GADGET3 code by Volker Springel. Minor modifications for updated variable
 * and memory structures in GIZMO by Phil Hopkins.
 */


static struct peano_hilbert_data
{
  peanokey key;
  int index;
}
 *mp;

static int *Id;


/*! Coincident particles -- bitwise-identical positions -- are never safe. Beyond having no
 *  relative geometry for a spatial index to order them by, they break every routine that treats a
 *  separation of zero as "the same particle" or divides by it, which is most of the pair physics.
 *  So this is an invariant, checked on every ordering, not a one-off validation of the input.
 *
 *  Identical positions imply identical Peano keys, so the array sorted just above already groups
 *  them and the scan is one pass over it: no extra sort, allocation, or communication. Reporting
 *  is capped, and the scan stops early once the cap is reached, since the run ends regardless.
 */
#define COINCIDENT_REPORT_MAX 8
#define COINCIDENT_SCAN_CAP 4096

static void report_coincident_pair(int iu, int iv, long nseen)
{
    if(nseen < COINCIDENT_REPORT_MAX)
    {
        printf("Coincident particle positions: IDs %llu and %llu (types %d and %d) both at (%.17g, %.17g, %.17g)\n",
               (unsigned long long) P[iu].ID, (unsigned long long) P[iv].ID, P[iu].Type, P[iv].Type,
               P[iu].Pos[0], P[iu].Pos[1], P[iu].Pos[2]);
        fflush(stdout);
    }
}

/*! scan one key-sorted block, from the sort array (keys already in hand, nothing recomputed) */
static long scan_block_for_coincident(struct peano_hilbert_data *m, int n, long nseen)
{
    long ndup = 0; int a, b;
    for(a = 0; a < n; a = b)
    {
        for(b = a + 1; b < n && m[b].key == m[a].key; b++) {}   /* run of identical keys */
        if(b - a < 2) {continue;}
        for(int u = a; u < b; u++) for(int v = u + 1; v < b; v++)
        {
            int iu = m[u].index, iv = m[v].index;
            if(P[iu].Pos[0] == P[iv].Pos[0] && P[iu].Pos[1] == P[iv].Pos[1] && P[iu].Pos[2] == P[iv].Pos[2])
            {
                report_coincident_pair(iu, iv, nseen + ndup); ndup++;
                if(ndup >= COINCIDENT_SCAN_CAP) {return ndup;}
            }
        }
    }
    return ndup;
}

/*! Peano key of a particle's current position, built exactly as domain.cc builds Key[]. */
static peanokey particle_peano_key(int i)
{
    peano1D xb = domain_double_to_int(((P[i].Pos[0] - DomainCorner[0]) / DomainLen) + 1.0);
    peano1D yb = domain_double_to_int(((P[i].Pos[1] - DomainCorner[1]) / DomainLen) + 1.0);
    peano1D zb = domain_double_to_int(((P[i].Pos[2] - DomainCorner[2]) / DomainLen) + 1.0);
    return peano_hilbert_key(xb, yb, zb, BITS_PER_DIMENSION);
}

/*! A gas particle coincident with a collisionless one is missed by the per-block scans above,
 *  because gas and collisionless particles are ordered separately. Catching it means merging the
 *  two sorted blocks, which costs one key evaluation per particle, so it runs only on the first
 *  ordering -- i.e. as validation of the input. That is not a hole in the runtime invariant: a
 *  runtime cross-species coincidence cannot be created in the first place, since star formation
 *  converts a particle rather than duplicating it and every spawn path enforces a minimum
 *  separation (merge_split.cc, and the sink analog).
 */
#define COINCIDENT_RUN_MAX 64

static long scan_across_blocks_for_coincident(long nseen)
{
    long ndup = 0; int a = 0, b = N_gas, members[COINCIDENT_RUN_MAX];
    peanokey ka = (a < N_gas) ? particle_peano_key(a) : 0;
    peanokey kb = (b < NumPart) ? particle_peano_key(b) : 0;
    while(a < N_gas || b < NumPart)
    {
        peanokey k;
        if(a >= N_gas) {k = kb;} else if(b >= NumPart) {k = ka;} else {k = (ka <= kb) ? ka : kb;}
        int nm = 0, gas_in_run = 0;
        while(a < N_gas && ka == k)
        {
            if(nm < COINCIDENT_RUN_MAX) {members[nm++] = a; gas_in_run++;}
            a++; ka = (a < N_gas) ? particle_peano_key(a) : 0;
        }
        while(b < NumPart && kb == k)
        {
            if(nm < COINCIDENT_RUN_MAX) {members[nm++] = b;}
            b++; kb = (b < NumPart) ? particle_peano_key(b) : 0;
        }
        if(gas_in_run == 0 || nm == gas_in_run) {continue;}   /* same-block pairs already covered */
        for(int u = 0; u < gas_in_run; u++) for(int v = gas_in_run; v < nm; v++)
        {
            int iu = members[u], iv = members[v];
            if(P[iu].Pos[0] == P[iv].Pos[0] && P[iu].Pos[1] == P[iv].Pos[1] && P[iu].Pos[2] == P[iv].Pos[2])
            {
                report_coincident_pair(iu, iv, nseen + ndup); ndup++;
                if(ndup >= COINCIDENT_SCAN_CAP) {return ndup;}
            }
        }
    }
    return ndup;
}

void peano_hilbert_order(void)
{
  int i; PRINT_STATUS("Begin Peano-Hilbert order...");
  static int first_ordering = 1;   /* the cross-block pass is input validation: first ordering only */
  long ndup_local = 0;

  if(N_gas)
    {
      mp = (struct peano_hilbert_data *) mymalloc("mp", sizeof(struct peano_hilbert_data) * N_gas);
      Id = (int *) mymalloc("Id", sizeof(int) * N_gas);

      for(i = 0; i < N_gas; i++)
	{
	  mp[i].index = i;
	  mp[i].key = Key[i];
	}

      mysort_peano(mp, N_gas, sizeof(struct peano_hilbert_data), peano_compare_key);

      /* before reorder_gas(), while mp[].index still refers to current P[] slots */
      ndup_local += scan_block_for_coincident(mp, N_gas, ndup_local);

      for(i = 0; i < N_gas; i++)
	Id[mp[i].index] = i;

      reorder_gas();

      myfree(Id);
      myfree(mp);
    }


  if(NumPart - N_gas > 0)
    {
      mp =
	(struct peano_hilbert_data *) mymalloc("mp", sizeof(struct peano_hilbert_data) * (NumPart - N_gas));
      mp -= (N_gas);

      Id = (int *) mymalloc("Id", sizeof(int) * (NumPart - N_gas));
      Id -= (N_gas);

      for(i = N_gas; i < NumPart; i++)
	{
	  mp[i].index = i;
	  mp[i].key = Key[i];
	}

      mysort_peano(mp + N_gas, NumPart - N_gas, sizeof(struct peano_hilbert_data), peano_compare_key);

      ndup_local += scan_block_for_coincident(mp + N_gas, NumPart - N_gas, ndup_local);

      for(i = N_gas; i < NumPart; i++)
	Id[mp[i].index] = i;

      reorder_particles();

      Id += N_gas;
      myfree(Id);
      mp += N_gas;
      myfree(mp);
    }

  if(first_ordering && N_gas > 0 && NumPart > N_gas) {ndup_local += scan_across_blocks_for_coincident(ndup_local);}

  /* every rank reduces the same total and so stops together: no peer is left in a collective */
  long ndup_total = 0;
  MPI_Allreduce(&ndup_local, &ndup_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if(ndup_total > 0)
    {
      if(ThisTask == 0)
        {
          printf("\nProblem: found %ld coincident particle pair%s -- positions identical to the bit (first few listed above).\n"
                 "Coincident particles are never valid: they have no relative geometry for a neighbor search or an\n"
                 "opening criterion to act on, and any pair term that divides by their separation is undefined.\n",
                 ndup_total, (ndup_total == 1) ? "" : "s");
          if(first_ordering)
            {printf("These came in with the initial conditions. Separate or merge them and rerun.\n");}
          else
            {printf("These appeared during the run, after step %d: this is a code bug, not a bad input.\n", All.NumCurrentTiStep);}
          fflush(stdout);
        }
      endrun(223);
    }
  first_ordering = 0;

    PRINT_STATUS(" ..Peano-Hilbert done");
}


int peano_compare_key(const void *a, const void *b)
{
  if(((struct peano_hilbert_data *) a)->key < (((struct peano_hilbert_data *) b)->key)) {return -1;}
  if(((struct peano_hilbert_data *) a)->key > (((struct peano_hilbert_data *) b)->key)) {return +1;}
  return 0;
}

void reorder_gas(void)
{
  int i;
  struct particle_data Psave, Psource;
  struct gas_cell_data GasPsave, GasPsource;
  int idsource, idsave, dest;
#ifdef CHIMES 
  struct gasVariables gasVarsSave, gasVarsSource;
#endif 

  for(i = 0; i < N_gas; i++)
    {
      if(Id[i] != i)
	{
	  Psource = P[i];
	  GasPsource = CellP[i];
#ifdef CHIMES 
	  gasVarsSource = ChimesGasVars[i];
#endif
	  idsource = Id[i];
	  dest = Id[i];

	  do
	    {
	      Psave = P[dest];
	      GasPsave = CellP[dest];
#ifdef CHIMES 
	      gasVarsSave = ChimesGasVars[dest];
#endif 
	      idsave = Id[dest];

	      P[dest] = Psource;
	      CellP[dest] = GasPsource;
#ifdef CHIMES 
	      ChimesGasVars[dest] = gasVarsSource;
#endif 
	      Id[dest] = idsource;

	      if(dest == i)
		break;

	      Psource = Psave;
	      GasPsource = GasPsave;
#ifdef CHIMES 
	      gasVarsSource = gasVarsSave; 
#endif 
	      idsource = idsave;
	      dest = idsource;
	    }
	  while(1);
	}
    }
}


void reorder_particles(void)
{
  int i;
  struct particle_data Psave, Psource;
  int idsource, idsave, dest;

  for(i = N_gas; i < NumPart; i++)
    {
      if(Id[i] != i)
	{
	  Psource = P[i];
	  idsource = Id[i];

	  dest = Id[i];

	  do
	    {
	      Psave = P[dest];
	      idsave = Id[dest];
	      P[dest] = Psource;
	      Id[dest] = idsource;
	      if(dest == i)
		break;

	      Psource = Psave;
	      idsource = idsave;
	      dest = idsource;
	    }
	  while(1);
	}
    }
}





/*  The following rewrite of the original function
 *  peano_hilbert_key_old() has been written by Martin Reinecke. 
 *  It is about a factor 2.3 - 2.5 faster than Volker's old routine!
 */
const unsigned char rottable3[48][8] = {
  {36, 28, 25, 27, 10, 10, 25, 27},
  {29, 11, 24, 24, 37, 11, 26, 26},
  {8, 8, 25, 27, 30, 38, 25, 27},
  {9, 39, 24, 24, 9, 31, 26, 26},
  {40, 24, 44, 32, 40, 6, 44, 6},
  {25, 7, 33, 7, 41, 41, 45, 45},
  {4, 42, 4, 46, 26, 42, 34, 46},
  {43, 43, 47, 47, 5, 27, 5, 35},
  {33, 35, 36, 28, 33, 35, 2, 2},
  {32, 32, 29, 3, 34, 34, 37, 3},
  {33, 35, 0, 0, 33, 35, 30, 38},
  {32, 32, 1, 39, 34, 34, 1, 31},
  {24, 42, 32, 46, 14, 42, 14, 46},
  {43, 43, 47, 47, 25, 15, 33, 15},
  {40, 12, 44, 12, 40, 26, 44, 34},
  {13, 27, 13, 35, 41, 41, 45, 45},
  {28, 41, 28, 22, 38, 43, 38, 22},
  {42, 40, 23, 23, 29, 39, 29, 39},
  {41, 36, 20, 36, 43, 30, 20, 30},
  {37, 31, 37, 31, 42, 40, 21, 21},
  {28, 18, 28, 45, 38, 18, 38, 47},
  {19, 19, 46, 44, 29, 39, 29, 39},
  {16, 36, 45, 36, 16, 30, 47, 30},
  {37, 31, 37, 31, 17, 17, 46, 44},
  {12, 4, 1, 3, 34, 34, 1, 3},
  {5, 35, 0, 0, 13, 35, 2, 2},
  {32, 32, 1, 3, 6, 14, 1, 3},
  {33, 15, 0, 0, 33, 7, 2, 2},
  {16, 0, 20, 8, 16, 30, 20, 30},
  {1, 31, 9, 31, 17, 17, 21, 21},
  {28, 18, 28, 22, 2, 18, 10, 22},
  {19, 19, 23, 23, 29, 3, 29, 11},
  {9, 11, 12, 4, 9, 11, 26, 26},
  {8, 8, 5, 27, 10, 10, 13, 27},
  {9, 11, 24, 24, 9, 11, 6, 14},
  {8, 8, 25, 15, 10, 10, 25, 7},
  {0, 18, 8, 22, 38, 18, 38, 22},
  {19, 19, 23, 23, 1, 39, 9, 39},
  {16, 36, 20, 36, 16, 2, 20, 10},
  {37, 3, 37, 11, 17, 17, 21, 21},
  {4, 17, 4, 46, 14, 19, 14, 46},
  {18, 16, 47, 47, 5, 15, 5, 15},
  {17, 12, 44, 12, 19, 6, 44, 6},
  {13, 7, 13, 7, 18, 16, 45, 45},
  {4, 42, 4, 21, 14, 42, 14, 23},
  {43, 43, 22, 20, 5, 15, 5, 15},
  {40, 12, 21, 12, 40, 6, 23, 6},
  {13, 7, 13, 7, 41, 41, 22, 20}
};

const unsigned char subpix3[48][8] = {
  {0, 7, 1, 6, 3, 4, 2, 5},
  {7, 4, 6, 5, 0, 3, 1, 2},
  {4, 3, 5, 2, 7, 0, 6, 1},
  {3, 0, 2, 1, 4, 7, 5, 6},
  {1, 0, 6, 7, 2, 3, 5, 4},
  {0, 3, 7, 4, 1, 2, 6, 5},
  {3, 2, 4, 5, 0, 1, 7, 6},
  {2, 1, 5, 6, 3, 0, 4, 7},
  {6, 1, 7, 0, 5, 2, 4, 3},
  {1, 2, 0, 3, 6, 5, 7, 4},
  {2, 5, 3, 4, 1, 6, 0, 7},
  {5, 6, 4, 7, 2, 1, 3, 0},
  {7, 6, 0, 1, 4, 5, 3, 2},
  {6, 5, 1, 2, 7, 4, 0, 3},
  {5, 4, 2, 3, 6, 7, 1, 0},
  {4, 7, 3, 0, 5, 6, 2, 1},
  {6, 7, 5, 4, 1, 0, 2, 3},
  {7, 0, 4, 3, 6, 1, 5, 2},
  {0, 1, 3, 2, 7, 6, 4, 5},
  {1, 6, 2, 5, 0, 7, 3, 4},
  {2, 3, 1, 0, 5, 4, 6, 7},
  {3, 4, 0, 7, 2, 5, 1, 6},
  {4, 5, 7, 6, 3, 2, 0, 1},
  {5, 2, 6, 1, 4, 3, 7, 0},
  {7, 0, 6, 1, 4, 3, 5, 2},
  {0, 3, 1, 2, 7, 4, 6, 5},
  {3, 4, 2, 5, 0, 7, 1, 6},
  {4, 7, 5, 6, 3, 0, 2, 1},
  {6, 7, 1, 0, 5, 4, 2, 3},
  {7, 4, 0, 3, 6, 5, 1, 2},
  {4, 5, 3, 2, 7, 6, 0, 1},
  {5, 6, 2, 1, 4, 7, 3, 0},
  {1, 6, 0, 7, 2, 5, 3, 4},
  {6, 5, 7, 4, 1, 2, 0, 3},
  {5, 2, 4, 3, 6, 1, 7, 0},
  {2, 1, 3, 0, 5, 6, 4, 7},
  {0, 1, 7, 6, 3, 2, 4, 5},
  {1, 2, 6, 5, 0, 3, 7, 4},
  {2, 3, 5, 4, 1, 0, 6, 7},
  {3, 0, 4, 7, 2, 1, 5, 6},
  {1, 0, 2, 3, 6, 7, 5, 4},
  {0, 7, 3, 4, 1, 6, 2, 5},
  {7, 6, 4, 5, 0, 1, 3, 2},
  {6, 1, 5, 2, 7, 0, 4, 3},
  {5, 4, 6, 7, 2, 3, 1, 0},
  {4, 3, 7, 0, 5, 2, 6, 1},
  {3, 2, 0, 1, 4, 5, 7, 6},
  {2, 5, 1, 6, 3, 4, 0, 7}
};

/*! This function computes a Peano-Hilbert key for an integer triplet (x,y,z),
  *  with x,y,z in the range between 0 and 2^bits-1.
  */
peanokey peano_hilbert_key(peano1D x, peano1D y, peano1D z, int bits)
{
  peano1D mask;
  unsigned char rotation = 0;
  peanokey key = 0;

  for(mask = ((peano1D)1) << (bits - 1); mask > 0; mask >>= 1)
    {
      unsigned char pix = ((x & mask) ? 4 : 0) | ((y & mask) ? 2 : 0) | ((z & mask) ? 1 : 0);

      key <<= 3;
      key |= subpix3[rotation][pix];
      rotation = rottable3[rotation][pix];
    }

  return key;
}



peanokey morton_key(peano1D x, peano1D y, peano1D z, int bits)
{
  peano1D mask;
  peanokey morton = 0;

  for(mask = ((peano1D)1) << (bits - 1); mask > 0; mask >>= 1)
    {
      morton <<= 3;
      morton += ((z & mask) ? 4 : 0) + ((y & mask) ? 2 : 0) + ((x & mask) ? 1 : 0);
    }

  return morton;
}


peanokey peano_and_morton_key(peano1D x, peano1D y, peano1D z, int bits, peanokey * morton_key)
{
  peano1D mask;
  unsigned char rotation = 0;
  peanokey key = 0;
  peanokey morton = 0;


  for(mask = ((peano1D)1) << (bits - 1); mask > 0; mask >>= 1)
    {
      unsigned char pix = ((x & mask) ? 4 : 0) | ((y & mask) ? 2 : 0) | ((z & mask) ? 1 : 0);

      key <<= 3;
      key |= subpix3[rotation][pix];
      rotation = rottable3[rotation][pix];

      morton <<= 3;
      morton += ((z & mask) ? 4 : 0) + ((y & mask) ? 2 : 0) + ((x & mask) ? 1 : 0);
    }

  *morton_key = morton;

  return key;
}




static int quadrants[24][2][2][2] = {
  /* rotx=0, roty=0-3 */
  {{{0, 7}, {1, 6}}, {{3, 4}, {2, 5}}},
  {{{7, 4}, {6, 5}}, {{0, 3}, {1, 2}}},
  {{{4, 3}, {5, 2}}, {{7, 0}, {6, 1}}},
  {{{3, 0}, {2, 1}}, {{4, 7}, {5, 6}}},
  /* rotx=1, roty=0-3 */
  {{{1, 0}, {6, 7}}, {{2, 3}, {5, 4}}},
  {{{0, 3}, {7, 4}}, {{1, 2}, {6, 5}}},
  {{{3, 2}, {4, 5}}, {{0, 1}, {7, 6}}},
  {{{2, 1}, {5, 6}}, {{3, 0}, {4, 7}}},
  /* rotx=2, roty=0-3 */
  {{{6, 1}, {7, 0}}, {{5, 2}, {4, 3}}},
  {{{1, 2}, {0, 3}}, {{6, 5}, {7, 4}}},
  {{{2, 5}, {3, 4}}, {{1, 6}, {0, 7}}},
  {{{5, 6}, {4, 7}}, {{2, 1}, {3, 0}}},
  /* rotx=3, roty=0-3 */
  {{{7, 6}, {0, 1}}, {{4, 5}, {3, 2}}},
  {{{6, 5}, {1, 2}}, {{7, 4}, {0, 3}}},
  {{{5, 4}, {2, 3}}, {{6, 7}, {1, 0}}},
  {{{4, 7}, {3, 0}}, {{5, 6}, {2, 1}}},
  /* rotx=4, roty=0-3 */
  {{{6, 7}, {5, 4}}, {{1, 0}, {2, 3}}},
  {{{7, 0}, {4, 3}}, {{6, 1}, {5, 2}}},
  {{{0, 1}, {3, 2}}, {{7, 6}, {4, 5}}},
  {{{1, 6}, {2, 5}}, {{0, 7}, {3, 4}}},
  /* rotx=5, roty=0-3 */
  {{{2, 3}, {1, 0}}, {{5, 4}, {6, 7}}},
  {{{3, 4}, {0, 7}}, {{2, 5}, {1, 6}}},
  {{{4, 5}, {7, 6}}, {{3, 2}, {0, 1}}},
  {{{5, 2}, {6, 1}}, {{4, 3}, {7, 0}}}
};


static int rotxmap_table[24] = { 4, 5, 6, 7, 8, 9, 10, 11,
  12, 13, 14, 15, 0, 1, 2, 3, 17, 18, 19, 16, 23, 20, 21, 22
};

static int rotymap_table[24] = { 1, 2, 3, 0, 16, 17, 18, 19,
  11, 8, 9, 10, 22, 23, 20, 21, 14, 15, 12, 13, 4, 5, 6, 7
};

static int rotx_table[8] = { 3, 0, 0, 2, 2, 0, 0, 1 };
static int roty_table[8] = { 0, 1, 1, 2, 2, 3, 3, 0 };

static int sense_table[8] = { -1, -1, -1, +1, +1, -1, -1, -1 };


static void parallel_sort_phdata_peano(struct peano_hilbert_data *arr, size_t n)
{
  auto cmp = [](const peano_hilbert_data &a, const peano_hilbert_data &b) { return a.key < b.key; };
  const size_t CUTOFF = 10000;
  if(n <= CUTOFF) { std::sort(arr, arr + n, cmp); return; }
  size_t mid = n / 2;
#ifdef _OPENMP
  #pragma omp task shared(arr) if(n > CUTOFF)
#endif
  parallel_sort_phdata_peano(arr, mid);
#ifdef _OPENMP
  #pragma omp task shared(arr) if(n > CUTOFF)
#endif
  parallel_sort_phdata_peano(arr + mid, n - mid);
#ifdef _OPENMP
  #pragma omp taskwait
#endif
  std::inplace_merge(arr, arr + mid, arr + n, cmp);
}

void mysort_peano(void *b, size_t n, size_t s, int (*cmp_fn) (const void *, const void *))
{
  struct peano_hilbert_data *arr = (struct peano_hilbert_data *)b;
#ifdef _OPENMP
  #pragma omp parallel
  {
    #pragma omp single
    parallel_sort_phdata_peano(arr, n);
  }
#else
  parallel_sort_phdata_peano(arr, n);
#endif
}
