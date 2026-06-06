#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include "../../declarations/allvars.h"
#include "../../core/proto.h"
#include "../../mesh/kernel.h"
/*!
* This file was originally part of the GADGET3 code developed by Volker Springel.
* It has been updated significantly by PFH for basic compatibility with GIZMO,
* as well as code cleanups, and accommodating new GIZMO functionality for various
* other operations. See notes in subfind.c and GIZMO User Guide for details.
*/


#ifdef SUBFIND

#include "../fof.h"
#include "../group_search.h"
#include "subfind.h"


/*! Structure for communication during the density computation. Holds data that is sent to other processors.
 */



void subfind_contamination(void)
{
  int contam_mask;
#ifdef FOF_DENSITY_SPLIT_TYPES
  contam_mask = 63 & ~(FOF_DENSITY_SPLIT_TYPES);
#else
  contam_mask = 63 & ~(FOF_PRIMARY_LINK_TYPES);
#endif

  std::vector<int> groups;
  std::vector<double> centers, radii;
  groups.reserve(Ngroups);
  for(int i = 0; i < Ngroups; i++)
    {
      Group[i].ContaminationLen = 0;
      Group[i].ContaminationMass = 0;
      if(Group[i].SubHaloProps_vsDelta[0].R200 <= 0) continue;
      groups.push_back(i);
      radii.push_back(Group[i].SubHaloProps_vsDelta[0].R200);
      for(int k = 0; k < 3; k++) centers.push_back(Group[i].Pos[k]);
    }

  group_search_import_pool_t pool;
  group_search_import_particles_around_points(contam_mask, centers, radii, pool);
  if(groups.empty()) return;

  int query_start = pool.particles.size();
  pool.particles.resize(query_start + groups.size());

  std::vector<int> sources(groups.size());
  for(size_t a = 0; a < groups.size(); a++)
    {
      int gr = groups[a];
      particle_data q;
      memset(&q, 0, sizeof(particle_data));
      q.Type = 6;
      q.Mass = 1;
      q.KernelRadius = radii[a];
      for(int k = 0; k < 3; k++) q.Pos[k] = Group[gr].Pos[k];
      pool.particles[query_start + a] = q;
      sources[a] = query_start + a;
    }

  neighbor_list_t nl;
  group_search_build_cross_type_nl(pool.particles, sources.data(), groups.size(), radii.data(), contam_mask, &nl);

  for(size_t a = 0; a < groups.size(); a++)
    {
      int gr = groups[a];
      for(int n = nl.offsets[a]; n < nl.offsets[a + 1]; n++)
        {
          int j = nl.neighbors[n];
          Group[gr].ContaminationLen++;
          Group[gr].ContaminationMass += pool.particles[j].Mass;
        }
    }

  free_neighbor_list(&nl);
  return;
}


/*! This function represents the core of the density computation. The
 *  target particle may either be local, or reside in the communication
 *  buffer.
 */




#endif
