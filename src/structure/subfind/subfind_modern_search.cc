#include <algorithm>
#include <math.h>
#include <vector>

#include "../../declarations/allvars.h"
#include "../../core/proto.h"
#include "../../mesh/kernel.h"
#include "../../mesh/neighbor_list.h"
#include "../fof.h"
#include "../group_search.h"
#include "subfind.h"

#ifdef SUBFIND

struct subfind_modern_ngb_t
{
  double r2;
  int index;
};

static int subfind_modern_ngb_r2_less(const subfind_modern_ngb_t &a, const subfind_modern_ngb_t &b)
{
  if(a.r2 < b.r2) return 1;
  if(a.r2 > b.r2) return 0;
  return a.index < b.index;
}

static int subfind_modern_type_mask_for_density_target(int j_in)
{
#ifdef FOF_DENSITY_SPLIT_TYPES
  if(ThisTask == 0)
    printf("FOF_DENSITY_SPLIT_TYPES is not yet supported by the modern SUBFIND neighbor-list path.\n");
  endrun(990505);
  return 0;
#else
  (void)j_in;
  return FOF_PRIMARY_LINK_TYPES;
#endif
}

static double subfind_modern_max_radius(const std::vector<int> &indices)
{
  double max_radius = 0;
  for(size_t a = 0; a < indices.size(); a++)
    if(P[indices[a]].DM_KernelRadius > max_radius)
      max_radius = P[indices[a]].DM_KernelRadius;
  return max_radius;
}

static void subfind_modern_select_neighbors(const group_search_import_pool_t &pool, const neighbor_list_t &nl,
                                            int list_pos, int source_index, int type_mask, int group_filter,
                                            int max_count, std::vector<subfind_modern_ngb_t> &selected)
{
  selected.clear();
  if(source_index < 0 || source_index >= (int)pool.particles.size())
    {
      printf("modern SUBFIND neighbor selection received invalid source index=%d pool_size=%d list_pos=%d on task=%d\n",
             source_index, (int)pool.particles.size(), list_pos, ThisTask);
      fflush(stdout);
      endrun(990508);
      return;	/* invalid source index: return with empty selection (avoids out-of-bounds pool access); drains downstream */
    }
  double h = pool.particles[source_index].DM_KernelRadius;
  double h2 = h * h;
  for(int n = nl.offsets[list_pos]; n < nl.offsets[list_pos + 1]; n++)
    {
      int j = nl.neighbors[n];
      if(j < 0 || j >= (int)pool.particles.size())
        {
          printf("modern SUBFIND neighbor list returned invalid neighbor index=%d pool_size=%d row=%d offset=%d/%d task=%d\n",
                 j, (int)pool.particles.size(), list_pos, n, nl.offsets[list_pos + 1], ThisTask);
          fflush(stdout);
          endrun(990509);
          continue;	/* invalid neighbor index: skip it (avoids out-of-bounds pool access); drains downstream */
        }
      const particle_data &pj = pool.particles[j];
      if(pj.Mass <= 0) continue;
      if(!((1 << pj.Type) & type_mask)) continue;
      if(group_filter >= 0 && pj.GrNr != group_filter) continue;
      double r2 = group_search_distance2_particles(pool.particles[source_index], pj);
      if(r2 >= h2) continue;
      subfind_modern_ngb_t entry = {r2, j};
      selected.push_back(entry);
    }

  std::sort(selected.begin(), selected.end(), subfind_modern_ngb_r2_less);
  if(max_count > 0 && (int)selected.size() > max_count) selected.resize(max_count);
}

void subfind_density_modern(int j_in)
{
  int target_mask = subfind_modern_type_mask_for_density_target(j_in);
  std::vector<int> targets;
  targets.reserve(NumPart);
  for(int i = 0; i < NumPart; i++)
    if(P[i].Mass > 0 && ((1 << P[i].Type) & target_mask))
      targets.push_back(i);

  std::vector<MyFloat> left(NumPart, 0), right(NumPart, 0);
  std::vector<unsigned char> todo(NumPart, 0);
  for(size_t a = 0; a < targets.size(); a++) todo[targets[a]] = 1;

  long long ntot = 0;
  int iter = 0;
  do
    {
      std::vector<int> active;
      active.reserve(targets.size());
      for(size_t a = 0; a < targets.size(); a++)
        if(todo[targets[a]])
          {
            int i = targets[a];
            P[i].DM_NumNgb = 0;
            P[i].u.DM_Density = 0;
            P[i].v.DM_VelDisp = 0;
            active.push_back(i);
          }

      double max_radius = subfind_modern_max_radius(active);
      group_search_import_pool_t pool;
      group_search_import_particles(target_mask, FOF_PRIMARY_LINK_TYPES, max_radius, NULL, NULL, pool);

      std::vector<double> radii(active.size());
      for(size_t a = 0; a < active.size(); a++) radii[a] = P[active[a]].DM_KernelRadius;
      neighbor_list_t nl = {NULL, NULL, 0, 0};
      if(!active.empty())
        group_search_build_cross_type_nl(pool.particles, active.data(), active.size(), radii.data(), FOF_PRIMARY_LINK_TYPES, &nl);

      std::vector<subfind_modern_ngb_t> selected;
      for(size_t a = 0; a < active.size(); a++)
        {
          int i = active[a];
          subfind_modern_select_neighbors(pool, nl, a, i, FOF_PRIMARY_LINK_TYPES, -1, All.DesLinkNgb, selected);
          P[i].DM_NumNgb = selected.size();
          if(P[i].DM_NumNgb >= All.DesLinkNgb)
            P[i].DM_KernelRadius = sqrt(selected[All.DesLinkNgb - 1].r2);

          double rho = 0, vx = 0, vy = 0, vz = 0, v2 = 0;
          double h = P[i].DM_KernelRadius;
          double hinv, hinv3, hinv4;
          kernel_hinv(h, &hinv, &hinv3, &hinv4);
          for(size_t n = 0; n < selected.size(); n++)
            {
              const particle_data &pj = pool.particles[selected[n].index];
              double r = sqrt(selected[n].r2);
              double wk, dwk;
              kernel_main(r * hinv, hinv3, hinv4, &wk, &dwk, -1);
              rho += pj.Mass * wk;
              vx += pj.Vel[0];
              vy += pj.Vel[1];
              vz += pj.Vel[2];
              v2 += pj.Vel[0] * pj.Vel[0] + pj.Vel[1] * pj.Vel[1] + pj.Vel[2] * pj.Vel[2];
            }
          P[i].u.DM_Density = rho;
          P[i].v.DM_VelDisp = v2;
          if(P[i].DM_NumNgb > 0)
            {
              vx /= P[i].DM_NumNgb;
              vy /= P[i].DM_NumNgb;
              vz /= P[i].DM_NumNgb;
              P[i].v.DM_VelDisp /= P[i].DM_NumNgb;
              double sig2 = P[i].v.DM_VelDisp - vx * vx - vy * vy - vz * vz;
              P[i].v.DM_VelDisp = (All.ComovingIntegrationOn ? 1.0 / All.Time : 1.0) * sqrt(DMAX(sig2, 0));
            }
        }
      if(!active.empty()) free_neighbor_list(&nl);

      int nleft = 0;
      for(size_t a = 0; a < targets.size(); a++)
        {
          int i = targets[a];
          if(!todo[i]) continue;
          if(P[i].DM_NumNgb != All.DesLinkNgb &&
             ((right[i] - left[i]) > 1.0e-4 * left[i] || left[i] == 0 || right[i] == 0))
            {
              nleft++;
              if(P[i].DM_NumNgb < All.DesLinkNgb)
                left[i] = DMAX(P[i].DM_KernelRadius, left[i]);
              else if(right[i] != 0)
                right[i] = DMIN(P[i].DM_KernelRadius, right[i]);
              else
                right[i] = P[i].DM_KernelRadius;

              if(right[i] > 0 && left[i] > 0)
                P[i].DM_KernelRadius = pow(0.5 * (pow(left[i], 3) + pow(right[i], 3)), 1.0 / 3);
              else if(right[i] == 0 && left[i] > 0)
                P[i].DM_KernelRadius *= 1.26;
              else if(right[i] > 0 && left[i] == 0)
                P[i].DM_KernelRadius /= 1.26;
              else
                endrun(8187);
            }
          else
            todo[i] = 0;
        }

      sumup_large_ints(1, &nleft, &ntot);
      if(ntot > 0 && ++iter > MAXITER)
        {
          printf("modern SUBFIND density failed to converge\n");
          fflush(stdout);
          endrun(990506);
          break;	/* MAXITER: ntot is symmetric (from sumup_large_ints), all ranks break together; avoids an infinite loop */
        }
    }
  while(ntot > 0);
}

void subfind_find_linkngb_modern(void)
{
  if(ThisTask == 0) printf("Start modern find_linkngb (%d particles on task=%d)\n", NumPartGroup, ThisTask);

  std::vector<MyFloat> left(NumPartGroup, 0), right(NumPartGroup, 0);
  std::vector<unsigned char> todo(NumPartGroup, 1);
  long long ntot = 0;
  int iter = 0;
  do
    {
      std::vector<int> active;
      active.reserve(NumPartGroup);
      for(int i = 0; i < NumPartGroup; i++)
        if(todo[i])
          {
            P[i].DM_NumNgb = 0;
            active.push_back(i);
          }

      double max_radius = subfind_modern_max_radius(active);
      group_search_import_pool_t pool;
      group_search_import_particles(63, FOF_PRIMARY_LINK_TYPES, max_radius, NULL, NULL, pool, GrNr);
      std::vector<double> radii(active.size());
      for(size_t a = 0; a < active.size(); a++) radii[a] = P[active[a]].DM_KernelRadius;
      neighbor_list_t nl = {NULL, NULL, 0, 0};
      if(!active.empty())
        group_search_build_cross_type_nl(pool.particles, active.data(), active.size(), radii.data(), FOF_PRIMARY_LINK_TYPES, &nl);

      std::vector<subfind_modern_ngb_t> selected;
      for(size_t a = 0; a < active.size(); a++)
        {
          int i = active[a];
          subfind_modern_select_neighbors(pool, nl, a, i, FOF_PRIMARY_LINK_TYPES, GrNr, All.DesLinkNgb, selected);
          P[i].DM_NumNgb = selected.size();
          if(P[i].DM_NumNgb >= All.DesLinkNgb)
            P[i].DM_KernelRadius = sqrt(selected[All.DesLinkNgb - 1].r2);
        }
      if(!active.empty()) free_neighbor_list(&nl);

      int nleft = 0;
      for(int i = 0; i < NumPartGroup; i++)
        if(todo[i])
          {
            if(P[i].DM_NumNgb != All.DesLinkNgb &&
               ((right[i] - left[i]) > 1.0e-3 * left[i] || left[i] == 0 || right[i] == 0))
              {
                nleft++;
                if(P[i].DM_NumNgb < All.DesLinkNgb)
                  left[i] = DMAX(P[i].DM_KernelRadius, left[i]);
                else if(right[i] != 0)
                  right[i] = DMIN(P[i].DM_KernelRadius, right[i]);
                else
                  right[i] = P[i].DM_KernelRadius;

                if(right[i] > 0 && left[i] > 0)
                  P[i].DM_KernelRadius = pow(0.5 * (pow(left[i], 3) + pow(right[i], 3)), 1.0 / 3);
                else if(right[i] == 0 && left[i] > 0)
                  P[i].DM_KernelRadius *= 1.26;
                else if(right[i] > 0 && left[i] == 0)
                  P[i].DM_KernelRadius /= 1.26;
                else
                  endrun(8189);
              }
            else
              todo[i] = 0;
          }

      sumup_large_ints(1, &nleft, &ntot);
      if(ntot > 0 && ++iter > MAXITER)
        {
          printf("modern SUBFIND find_linkngb failed to converge\n");
          fflush(stdout);
          endrun(990507);
          break;	/* MAXITER: ntot is symmetric (from sumup_large_ints), all ranks break together; avoids an infinite loop */
        }
    }
  while(ntot > 0);
}

void subfind_find_nearesttwo_modern(void)
{
  if(ThisTask == 0) printf("Start modern finding nearest two (%d particles on task=%d)\n", NumPartGroup, ThisTask);
  for(int i = 0; i < NumPartGroup; i++) NgbLoc[i].count = 0;

  std::vector<int> active;
  active.reserve(NumPartGroup);
  for(int i = 0; i < NumPartGroup; i++) active.push_back(i);
  double max_radius = subfind_modern_max_radius(active);

  group_search_import_pool_t pool;
  group_search_import_particles(63, FOF_PRIMARY_LINK_TYPES, max_radius, NULL, NULL, pool, GrNr);
  std::vector<double> radii(active.size());
  for(size_t a = 0; a < active.size(); a++) radii[a] = P[active[a]].DM_KernelRadius;

  neighbor_list_t nl = {NULL, NULL, 0, 0};
  if(!active.empty())
    group_search_build_cross_type_nl(pool.particles, active.data(), active.size(), radii.data(), FOF_PRIMARY_LINK_TYPES, &nl);

  for(size_t a = 0; a < active.size(); a++)
    {
      int i = active[a];
      double best_dist[2] = {MAX_REAL_NUMBER, MAX_REAL_NUMBER};
      long long best_index[2] = {-1, -1};
      int count = 0;
      double h2 = P[i].DM_KernelRadius * P[i].DM_KernelRadius;
      for(int n = nl.offsets[a]; n < nl.offsets[a + 1]; n++)
        {
          int j = nl.neighbors[n];
          const particle_data &pj = pool.particles[j];
          if(pj.Mass <= 0) continue;
          if(!((1 << pj.Type) & FOF_PRIMARY_LINK_TYPES)) continue;
          if(pj.GrNr != GrNr) continue;
          if(pj.ID == P[i].ID) continue;
          if(pj.u.DM_Density <= P[i].u.DM_Density) continue;
          double r2 = group_search_distance2_particles(pool.particles[i], pj);
          if(r2 >= h2) continue;

          int slot = -1;
          if(count < 2)
            slot = count++;
          else
            {
              slot = (best_dist[0] > best_dist[1]) ? 0 : 1;
              if(r2 >= best_dist[slot]) continue;
            }
          best_dist[slot] = r2;
          if(j < NumPart)
            best_index[slot] = (((long long)ThisTask) << 32) + j;
          else
            {
              int g = j - NumPart;
              best_index[slot] = (((long long)pool.ghost_home_task[g]) << 32) + pool.ghost_home_index[g];
            }
        }

      NgbLoc[i].count = count;
      for(int k = 0; k < count; k++)
        {
          R2Loc[i].dist[k] = best_dist[k];
          NgbLoc[i].index[k] = best_index[k];
        }
    }
  if(!active.empty()) free_neighbor_list(&nl);
}

#endif /* SUBFIND */
