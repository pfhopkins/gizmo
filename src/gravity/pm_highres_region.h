#ifndef PM_HIGHRES_REGION_H
#define PM_HIGHRES_REGION_H

/* High-resolution-region membership test for PM_PLACEHIGHRESREGION zoom runs — the single home
 * for pmforce_is_particle_high_res, shared by the PM solver (pm_nonperiodic.cc) and the gravity
 * tree walks (forcetree.cc CPU walk + gpu_gravtree.cc device walk). A high-res particle uses the
 * finer short-range PM cutoff (All.Rcut[1]/All.Asmth[1]); the device walk needs this on-device,
 * so the body is a KOKKOS_INLINE_FUNCTION. Pure in the standard case (function of type + the
 * compile-time region mask); the SPECIAL_GAS_TREATMENT variant reads All.Xmintot/Xmaxtot, which
 * are device-accessible under the All-mirror.
 *
 * Include AFTER declarations/allvars.h (All.* and Vec3 come from the including TU). */

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#if defined(PMGRID) && defined(PM_PLACEHIGHRESREGION)
KOKKOS_INLINE_FUNCTION int pmforce_is_particle_high_res(int type, Vec3<double>& Pos)
{
#ifndef SPECIAL_GAS_TREATMENT_IN_HIGHRESREGION
  /* standard treatment */
  (void)Pos;
  return (1 << type) & (PM_PLACEHIGHRESREGION);
#else

  if((1 << type) & (PM_PLACEHIGHRESREGION))
    return 1;

  /* special treatment */
  int j, flag = 1;
  for(j = 0; j < 3; j++)
    if(Pos[j] < All.Xmintot[1][j] || Pos[j] > All.Xmaxtot[1][j])
      flag = 0;

  return flag;
#endif
}
#endif /* PMGRID && PM_PLACEHIGHRESREGION */

#endif /* PM_HIGHRES_REGION_H */
