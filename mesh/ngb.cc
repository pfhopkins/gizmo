#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../system/vector.h"


/*!
 * This file contains routines for neighbour finding.  We use the gravity-tree and a range-searching technique to find neighbours.
 */
/*!
 * This file was originally part of the GADGET3 code developed by Volker Springel. The code has been heavily modified
 * by Phil Hopkins (phopkins@caltech.edu) for GIZMO (adding/consolidating some of the search routines as needed for different fluids).
 * the modules now are more modular and primarily run on a generic structure which was built entirely for GIZMO, as opposed to the
 * GADGET3-style neighbor finding, to allow for greater flexibility, more stable memory use, and efficient multi-threading.
 * More importantly general routines for neighbor-searching and walking trees have been completely rewritten into a new
 * set of modular files and structured portions of routines designed to be called by a high-level evaluation routine, so that all code can use
 * consistent definitions and these only need to be modified in a single place.
 */


/*! This function constructs the neighbour tree. To this end, we actually need to construct the gravitational tree, 
 *  because we use it now for the neighbour search.
 */
void ngb_treebuild(void)
{
    if(ThisTask == 0) {printf("Begin Ngb-tree construction.\n");}
    CPU_Step[CPU_MISC] += measure_time();
    force_treebuild(NumPart, NULL);
    TreeReconstructFlag = 0; /* a whole-tree build satisfies the condemnation init() raised; the neighbour loops that follow in init walk this tree */
    CPU_Step[CPU_TREEBUILD] += measure_time();
    if(ThisTask == 0) {printf("Ngb-Tree contruction finished \n");}
}


/*! This routine finds all neighbours `j' that can interact with the particle `i' in the communication buffer.
 *  Note that an interaction can take place if: \f$ r_{ij} < h_i \f$  OR if  \f$ r_{ij} < h_j \f$.
 *
 *  In the range-search this is taken into account, i.e. it is guaranteed that all particles are found that fulfill this condition,
 *  including the (more difficult) second part of it. For this purpose, each node knows the maximum h occuring among the particles it represents.
 *  NB that guarantee holds only while node hmax values are CURRENT: between a force_add_element_to_tree()
 *  insertion (which bumps only the immediate parent) and the next force_update_hmax()/rebuild/refresh,
 *  ancestor nodes can under-report hmax and prune a search that should have descended.
 */
#ifdef NGB_PAIR_CACHE
/* Cache of the local (mode 0) pair searches: the gradient loop's first pass records, for each element it searches from,
   the neighbour list and the pseudo-particles (export targets) met, in walk order; the second gradient pass and the hydro
   loop replay them instead of walking the tree again. Nothing between those loops moves an element, changes a kernel
   radius or mass, or touches the tree, and every neighbour a replayed walk would visit was already drifted by the
   recording walk, so a replay returns exactly what the walk would. A replay is used only for the same element, centre
   and radius; anything else walks. Lifetime: hydro_gradient_calc() begin -> end of hydro_force(). */
static struct {
    int state = 0; /* 0 off, 1 record, 2 replay */
    std::vector<long long> offset; /* per element index: start in data, or -1 */
    std::vector<int> nngb, npseudo;
    std::vector<double> key; /* 4 per element: radius and centre of the recorded search */
    std::vector<int> data; /* per entry: neighbours, then pseudo-particles */
} NgbPairCache;
static thread_local int ngb_pair_cache_tl_recording = 0;
static thread_local std::vector<int> ngb_pair_cache_tl_pseudo;

void ngb_pair_cache_begin_record(void)
{
    NgbPairCache.offset.assign(NumPart, -1); NgbPairCache.nngb.assign(NumPart, 0); NgbPairCache.npseudo.assign(NumPart, 0);
    NgbPairCache.key.assign(4*(size_t)NumPart, 0.); NgbPairCache.data.clear(); NgbPairCache.state = 1;
}
void ngb_pair_cache_begin_replay(void) {if(NgbPairCache.state) {NgbPairCache.state = 2;}}
void ngb_pair_cache_free(void)
{
    NgbPairCache.state = 0;
    std::vector<long long>().swap(NgbPairCache.offset); std::vector<int>().swap(NgbPairCache.nngb); std::vector<int>().swap(NgbPairCache.npseudo);
    std::vector<double>().swap(NgbPairCache.key); std::vector<int>().swap(NgbPairCache.data);
}

#define NGB_PSEUDO_HOOK(no_) {if(ngb_pair_cache_tl_recording) {ngb_pair_cache_tl_pseudo.push_back(no_);}}
#endif

static int ngb_treefind_pairs_threads_walk(MyDouble searchcenter[3], MyFloat rkern, int target, int *startnode,
                               int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist)
{
#include "../system/ngb_codeblock_before_condition.h"
    if(P[p].Type > 0) continue; // skip particles with non-gas types
    if(P[p].Mass <= 0) continue; // skip zero-mass particles
#define NGB_ONLY_OPEN_NODES_CONTAINING_GAS // only want gas
#define SEARCHBOTHWAYS 1 // need neighbors that can -mutually- see one another, not just single-directional searching here
#include "../system/ngb_codeblock_after_condition_threaded.h"
#undef SEARCHBOTHWAYS // must be undefined after code block inserted, or compiler will crash
#undef NGB_ONLY_OPEN_NODES_CONTAINING_GAS
}
#ifdef NGB_PSEUDO_HOOK
#undef NGB_PSEUDO_HOOK
#endif

int ngb_treefind_pairs_threads(MyDouble searchcenter[3], MyFloat rkern, int target, int *startnode,
                               int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist)
{
#ifdef NGB_PAIR_CACHE
    int use_cache = (NgbPairCache.state != 0) && (mode == 0) && (target >= 0) && (target < (int)NgbPairCache.offset.size()) && (*startnode == All.MaxPart);
    if(use_cache && NgbPairCache.state == 2)
    {
        long long off = NgbPairCache.offset[target]; const double *key = &NgbPairCache.key[4*(size_t)target];
        if(off >= 0 && key[0] == (double)rkern && key[1] == (double)searchcenter[0] && key[2] == (double)searchcenter[1] && key[3] == (double)searchcenter[2])
        {
            int nngb = NgbPairCache.nngb[target], npseudo = NgbPairCache.npseudo[target];
            const int *list = &NgbPairCache.data[off];
            int maxPart = All.MaxPart, maxNodes = MaxNodes, task; long bunchSize = All.BunchSize;
            for(int k = 0; k < npseudo; k++) /* exports exactly as the walk makes them (ngb_codeblock_after_condition_threaded.h) */
            {
                int no = list[nngb + k];
                if(exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != target)
                {
                    exportflag[task] = target;
                    exportnodecount[task] = NODELISTLENGTH;
                }
                if(exportnodecount[task] == NODELISTLENGTH)
                {
                    int exitFlag = 0, nexp;
#ifdef _OPENMP
#pragma omp critical(_nexportngb_)
#endif
                    {
                        if(Nexport >= bunchSize) {BufferFullFlag = 1; exitFlag = 1;} else {nexp = Nexport; Nexport++;}
                    }
                    if(exitFlag) {return -1;}
                    exportnodecount[task] = 0;
                    exportindex[task] = nexp;
                    DataIndexTable[nexp].Task = task;
                    DataIndexTable[nexp].Index = target;
                    DataIndexTable[nexp].IndexGet = nexp;
                }
                DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] = DomainNodeIndex[no - (maxPart + maxNodes)];
                if(exportnodecount[task] < NODELISTLENGTH) {DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;}
            }
            if(nngb > 0) {memcpy(ngblist, list, nngb * sizeof(int));}
            *startnode = -1;
            return nngb;
        }
    }
    if(use_cache && NgbPairCache.state == 1) {ngb_pair_cache_tl_recording = 1; ngb_pair_cache_tl_pseudo.clear();}
    int numngb = ngb_treefind_pairs_threads_walk(searchcenter, rkern, target, startnode, mode, exportflag, exportnodecount, exportindex, ngblist);
    if(ngb_pair_cache_tl_recording)
    {
        ngb_pair_cache_tl_recording = 0;
        if(numngb >= 0 && *startnode < 0) /* completed walk (a full export buffer returns -1 and the element is searched again later) */
        {
            int npseudo = (int)ngb_pair_cache_tl_pseudo.size();
#ifdef _OPENMP
#pragma omp critical(_ngbpaircache_)
#endif
            {
                long long off = (long long)NgbPairCache.data.size();
                NgbPairCache.data.insert(NgbPairCache.data.end(), ngblist, ngblist + numngb);
                NgbPairCache.data.insert(NgbPairCache.data.end(), ngb_pair_cache_tl_pseudo.begin(), ngb_pair_cache_tl_pseudo.end());
                NgbPairCache.offset[target] = off; NgbPairCache.nngb[target] = numngb; NgbPairCache.npseudo[target] = npseudo;
                double *key = &NgbPairCache.key[4*(size_t)target];
                key[0] = (double)rkern; key[1] = (double)searchcenter[0]; key[2] = (double)searchcenter[1]; key[3] = (double)searchcenter[2];
            }
        }
    }
    return numngb;
#else
    return ngb_treefind_pairs_threads_walk(searchcenter, rkern, target, startnode, mode, exportflag, exportnodecount, exportindex, ngblist);
#endif
}


/*! This function returns neighbours with distance <= rkern and returns them in Ngblist. Actually, particles in a box of half side length rkern are
 *  returned, i.e. the reduction to a sphere still needs to be done in the calling routine.
 */
int ngb_treefind_variable_threads(MyDouble searchcenter[3], MyFloat rkern, int target, int *startnode,
				  int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist)
{
#include "../system/ngb_codeblock_before_condition.h"
    if(P[p].Type > 0) continue; // skip particles with non-gas types
    if(P[p].Mass <= 0) continue; // skip zero-mass particles
#define NGB_ONLY_OPEN_NODES_CONTAINING_GAS // only want gas
#define SEARCHBOTHWAYS 0 // only need neighbors inside of search radius, not particles 'looking at' primary
#include "../system/ngb_codeblock_after_condition_threaded.h"
#undef SEARCHBOTHWAYS
#undef NGB_ONLY_OPEN_NODES_CONTAINING_GAS
}

/* this is the same as above, but the simpler un-threaded version, useful for historical reasons and because some sub-routines use 
    this model without threading because there is no real performance gain. note the slightly different construction of the subroutine below.
    TARGET_BITMASK should be set as a bitmask, i.e. SUM(2^n), where n are all the particle types desired for neighbor finding,
    so e.g. if you want particle types 0 and 4, set TARGET_BITMASK = 17 = 1 + 16 = 2^0 + 2^4
*/
int ngb_treefind_variable_targeted(MyDouble searchcenter[3], MyFloat rkern, int target, int *startnode, int mode, int *nexport, int *nsend_local, int TARGET_BITMASK)
{
    long nexport_save = *nexport; /* this line must be here in the un-threaded versions */
#include "../system/ngb_codeblock_before_condition.h" // call the same variable/initialization block
    if(!((1 << P[p].Type) & (TARGET_BITMASK))) continue; // skip anything not of the desired type
    if(P[p].Mass <= 0) continue; // skip zero-mass particles
#define SEARCHBOTHWAYS 0 // only need neighbors inside of search radius, not particles 'looking at' primary
#include "../system/ngb_codeblock_after_condition_unthreaded.h" // call the main loop block as above, but this time the -unthreaded- version
#undef SEARCHBOTHWAYS
}
/* identical to above but includes 'both ways' search for interacting neighbors */
int ngb_treefind_pairs_targeted(MyDouble searchcenter[3], MyFloat rkern, int target, int *startnode, int mode, int *nexport, int *nsend_local, int TARGET_BITMASK)
{
    long nexport_save = *nexport; /* this line must be here in the un-threaded versions */
#include "../system/ngb_codeblock_before_condition.h" // call the same variable/initialization block
    if(!((1 << P[p].Type) & (TARGET_BITMASK))) continue; // skip anything not of the desired type
    if(P[p].Mass <= 0) continue; // skip zero-mass particles
#define SEARCHBOTHWAYS 1 // only need neighbors inside of search radius, not particles 'looking at' primary
#include "../system/ngb_codeblock_after_condition_unthreaded.h" // call the main loop block as above, but this time the -unthreaded- version
#undef SEARCHBOTHWAYS
}


/*  slightly modified version of treefind that searches for one or more types of particles: 
        TARGET_BITMASK should be set as a bitmask, i.e. SUM(2^n), where n are all the particle types desired for neighbor finding,
        so e.g. if you want particle types 0 and 4, set TARGET_BITMASK = 17 = 1 + 16 = 2^0 + 2^4
 */
int ngb_treefind_variable_threads_targeted(MyDouble searchcenter[3], MyFloat rkern, int target, int *startnode,
                                           int mode, int *exportflag, int *exportnodecount, int *exportindex,
                                           int *ngblist, int TARGET_BITMASK)
{
#include "../system/ngb_codeblock_before_condition.h"
    if(!((1 << P[p].Type) & (TARGET_BITMASK))) continue; // skip anything not of the desired type
    if(P[p].Mass <= 0) continue; // skip zero-mass particles
#define SEARCHBOTHWAYS 0 // only need neighbors inside of search radius, not particles 'looking at' primary
#include "../system/ngb_codeblock_after_condition_threaded.h"
#undef SEARCHBOTHWAYS
}
/* identical to above but includes 'both ways' search for interacting neighbors */
int ngb_treefind_pairs_threads_targeted(MyDouble searchcenter[3], MyFloat rkern, int target, int *startnode,
                                           int mode, int *exportflag, int *exportnodecount, int *exportindex,
                                           int *ngblist, int TARGET_BITMASK)
{
#include "../system/ngb_codeblock_before_condition.h"
    if(!((1 << P[p].Type) & (TARGET_BITMASK))) continue; // skip anything not of the desired type
    if(P[p].Mass <= 0) continue; // skip zero-mass particles
#define SEARCHBOTHWAYS 1 // also want particles 'looking at' primary
#include "../system/ngb_codeblock_after_condition_threaded.h"
#undef SEARCHBOTHWAYS
}







/* 
    custom code for FOF finder -- needs to be able to deal with complications like pure node-linkages and hard-codes a
    local requirement, so we can't use our simple routines above. this is a customized version of the "ngb_treefind_variable" routine above. 
    as a result, updates to the core neighbor search routine will not alter this subroutine 
 */
int ngb_treefind_fof_primary(MyDouble searchcenter[3], MyFloat rkern, int target, int *startnode, int mode, int *nexport, int *nsend_local, int MyFOF_PRIMARY_LINK_TYPES)
{
    int numngb, no, p, task, nexport_save;
    struct NODE *current;
    MyDouble dx, dy, dz, dist, r2;
    // cache some global vars locally for improved compiler alias analysis
    int maxPart = All.MaxPart;
    int maxNodes = MaxNodes;
    long bunchSize = All.BunchSize;
    MyDouble xtmp; xtmp=0;

    nexport_save = *nexport;
    
    numngb = 0;
    no = *startnode;
    
    while(no >= 0)
    {
        if(no < maxPart)		/* single particle */
        {
            p = no;
            no = Nextnode[no];
            
            if(!((1 << P[p].Type) & (MyFOF_PRIMARY_LINK_TYPES)))
                continue;
            
            if(mode == 0)
                continue;
            
            dist = rkern;
            dx = NGB_PERIODIC_BOX_LONG_X(P[p].Pos[0] - searchcenter[0], P[p].Pos[1] - searchcenter[1], P[p].Pos[2] - searchcenter[2],-1);
            if(dx > dist) continue;
            dy = NGB_PERIODIC_BOX_LONG_Y(P[p].Pos[0] - searchcenter[0], P[p].Pos[1] - searchcenter[1], P[p].Pos[2] - searchcenter[2],-1);
            if(dy > dist) continue;
            dz = NGB_PERIODIC_BOX_LONG_Z(P[p].Pos[0] - searchcenter[0], P[p].Pos[1] - searchcenter[1], P[p].Pos[2] - searchcenter[2],-1);
            if(dz > dist) continue;
            if(dx * dx + dy * dy + dz * dz > dist * dist) continue;

            Ngblist[numngb++] = p;
        }
        else
        {
            if(no >= maxPart + maxNodes)	/* pseudo particle */
            {
                if(mode == 1)
                    endrun(123125);
                
                if(mode == 0)
                {
                    if(Exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != target)
                    {
                        Exportflag[task] = target;
                        Exportnodecount[task] = NODELISTLENGTH;
                    }
                    
                    if(Exportnodecount[task] == NODELISTLENGTH)
                    {
                        if(*nexport >= bunchSize)
                        {
                            *nexport = nexport_save;
                            if(nexport_save == 0) {endrun(13005);}	/* in this case, the buffer is too small to process even a single particle */
                            for(task = 0; task < NTask; task++) {nsend_local[task] = 0;}
                            for(no = 0; no < nexport_save; no++) {nsend_local[DataIndexTable[no].Task]++;}
                            return -1; /* buffer has filled -- important that only this and other buffer-full conditions return the negative condition for the routine */
                        }
                        Exportnodecount[task] = 0;
                        Exportindex[task] = *nexport;
                        DataIndexTable[*nexport].Task = task;
                        DataIndexTable[*nexport].Index = target;
                        DataIndexTable[*nexport].IndexGet = *nexport;
                        *nexport = *nexport + 1;
                        nsend_local[task]++;
                    }
                    
                    DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]++] =
                    DomainNodeIndex[no - (maxPart + maxNodes)];
                    
                    if(Exportnodecount[task] < NODELISTLENGTH)
                        DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;
                }
                
                if(mode == -1)
                {
                    *nexport = 1;
                }
                
                no = Nextnode[no - maxNodes];
                continue;
                
            }
            
            current = &Nodes[no];
            
            if(mode == 1)
            {
                if(current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL))	/* we reached a top-level node again, which means that we are done with the branch */
                {
                    *startnode = -1;
                    return numngb;
                }
            }
            
            if(mode == 0)
            {
                if(!(current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)))	/* we have a node with only local particles, can skip branch */
                {
                    no = current->u.d.sibling;
                    continue;
                }
            }
            
            no = current->u.d.sibling;	/* in case the node can be discarded */
            
            dist = rkern + 0.5 * current->len;
            dx = NGB_PERIODIC_BOX_LONG_X(current->center[0]-searchcenter[0],current->center[1]-searchcenter[1],current->center[2]-searchcenter[2],-1);
            if(dx > dist) continue;
            dy = NGB_PERIODIC_BOX_LONG_Y(current->center[0]-searchcenter[0],current->center[1]-searchcenter[1],current->center[2]-searchcenter[2],-1);
            if(dy > dist) continue;
            dz = NGB_PERIODIC_BOX_LONG_Z(current->center[0]-searchcenter[0],current->center[1]-searchcenter[1],current->center[2]-searchcenter[2],-1);
            if(dz > dist) continue;
            /* now test against the minimal sphere enclosing everything */
            dist += CUBE_EDGEFACTOR_1 * current->len;
            if((r2 = (dx * dx + dy * dy + dz * dz)) > dist * dist) continue;
            
            if((current->u.d.bitflags & ((1 << BITFLAG_TOPLEVEL) + (1 << BITFLAG_DEPENDS_ON_LOCAL_ELEMENT))) == 0)	/* only use fully local nodes */
            {
                /* test whether the node is contained within the sphere */
                dist = rkern - CUBE_EDGEFACTOR_2 * current->len;
                if(dist > 0)
                    if(r2 < dist * dist)
                    {
                        if(current->u.d.bitflags & (1 << BITFLAG_INSIDE_LINKINGLENGTH))	/* already flagged */
                        {
                            /* sufficient to return only one particle inside this cell */
                            
                            p = current->u.d.nextnode;
                            while(p >= 0)
                            {
                                if(p < maxPart)
                                {
                                    if(((1 << P[p].Type) & (MyFOF_PRIMARY_LINK_TYPES)))
                                    {
                                        dx = NGB_PERIODIC_BOX_LONG_X(P[p].Pos[0] - searchcenter[0], P[p].Pos[1] - searchcenter[1], P[p].Pos[2] - searchcenter[2],-1);
                                        dy = NGB_PERIODIC_BOX_LONG_Y(P[p].Pos[0] - searchcenter[0], P[p].Pos[1] - searchcenter[1], P[p].Pos[2] - searchcenter[2],-1);
                                        dz = NGB_PERIODIC_BOX_LONG_Z(P[p].Pos[0] - searchcenter[0], P[p].Pos[1] - searchcenter[1], P[p].Pos[2] - searchcenter[2],-1);
                                        if(dx * dx + dy * dy + dz * dz > rkern * rkern) break;

                                        Ngblist[numngb++] = p;
                                        break;
                                    }
                                    p = Nextnode[p];
                                }
                                else if(p >= maxPart + maxNodes)
                                    p = Nextnode[p - maxNodes];
                                else
                                    p = Nodes[p].u.d.nextnode;
                            }
                            continue;
                        }
                        else
                        {
                            /* flag it now */
                            current->u.d.bitflags |= (1 << BITFLAG_INSIDE_LINKINGLENGTH);
                        }
                    }
            }
            
            no = current->u.d.nextnode;	/* ok, we need to open the node */
        }
    }
    
    *startnode = -1;
    return numngb;
}





