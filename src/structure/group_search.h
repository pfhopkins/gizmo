#ifndef GROUP_SEARCH_H
#define GROUP_SEARCH_H

#include <vector>

#include "../declarations/allvars.h"
#include "../mesh/neighbor_list.h"

struct group_search_import_pool_t
{
  std::vector<particle_data> particles;
  std::vector<gas_cell_data> cells;
  std::vector<int> ghost_home_task;
  std::vector<int> ghost_home_index;
  std::vector<MyIDType> ghost_label_id;
  std::vector<MyIDType> ghost_label_task;
};

void group_search_import_particles(int source_type_mask, int import_type_mask, double search_radius,
                                   const MyIDType *label_id, const MyIDType *label_task,
                                   group_search_import_pool_t &pool, int group_filter = -1,
                                   bool import_cells = false);

void group_search_import_particles_around_points(int import_type_mask, const std::vector<double> &centers,
                                                 const std::vector<double> &radii,
                                                 group_search_import_pool_t &pool,
                                                 bool import_cells = false);

void group_search_build_cross_type_nl(std::vector<particle_data> &particles, int *sources, int nsources,
                                      double *radii, int j_type_mask, neighbor_list_t *nl);

double group_search_distance2_particles(const particle_data &a, const particle_data &b);

#endif
