/* radfb_local_gpu.h — declaration for radiation_pressure_winds GPU dispatch. */
#pragma once

#ifdef GALSF_FB_FIRE_RT_LOCALRP
void radiation_pressure_winds_gpu(struct particle_data *P_host,
                                   struct gas_cell_data *CellP_host,
                                   int num_total,
                                   const int *i_active_host, int num_active);
#endif
