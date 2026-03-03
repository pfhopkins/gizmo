#include "../../GIZMO_config.h"
c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////     E V O L V E _ A B U N D A N C E S     \\\\\\\\\\
c
c=======================================================================
c
c     Written by S. Glover (AMNH, 2004-2005, AIP 2006-2007)
c
      subroutine evolve_abundances(time, dl, yn, divv, energy, 
     $                             non_eq_abundances
     $                            , column_est)
#ifdef CHEMCOOL
      implicit NONE
#include "cool.h"
#include "non_eq.h"
c
      REAL    time, dl, yn, divv, energy
      REAL    non_eq_abundances(num_non_eq_species)
      REAL    radius, column_est
      REAL    abe, abh2, yntot, ekn, temp, gamma
      REAL    t_start, t_end, dt_est, dt, t_in, t_out
      REAL    tdust_old
      integer new_step
c
      integer nstep
      parameter (nstep = NSTEP)
c
      REAL safety(nspec)
      integer used_for_stepsize_control(nspec)
#if CHEMISTRYNETWORK == 5
      DATA safety /1d-1, 1d-1, 1d-1, DTCOOL_SCALE_FACTOR/
      DATA used_for_stepsize_control /1, 1, 1, 1/
#endif
c
      integer I, J
c
c DVODE control variables -- NB atol declared in non_eq.h
c
      REAL    rtol
      integer itask, iopt, itol, mf, istate, iflag_dvi
c
c Other DVODE variables
c
      integer lrw, liw
      parameter (lrw   = 2000)
      parameter (liw   = 2000)
      REAL    rpar(nrpar), rwork(lrw)
      integer ipar(nipar), iwork(liw)
      REAL    y(nspec), ydot(nspec), y_old(nspec)
c
      external jac, rate_eq
c
c Sanity check for our input time
c
#ifdef TEST
      time = t_start
#endif
      if (time .lt. 0d0) then
        return
      endif        
c
c Initialize abundance vector, treating the internal energy as 
c an additional abundance. 
c
c N.B. We assume that abundances are stored consecutively, with
c no gaps, and that itmp == nspec (i.e. that the energy is the last
c value stored)
c
      do I = 1, nspec-1
        y(I) = non_eq_abundances(I)
      enddo
      y(itmp)  = energy
#ifdef CHEM_EQ_H2
        y(0) = 1e-10
#endif

c
c Setup initial parameters for DVODE integrator. These don't vary over 
c the course of the timestep.
c
c Optional input supplied -- see iwork, rwork init. below:
c
      iopt = 1
c
c Evolve ODEs from t -> tout, allowing overshooting & interpolation
c [See DVODE documentation for more details].
c
      itask = 1
c
c
c Use same relative tolerance for each species, different absolute tolerances
c (since magnitudes may vary by many orders of magnitude). 
c
      itol = 2
c
c Set relative tolerance -- for simplicity we keep this fixed at 0.1% for now  
c
      rtol = RTOL
c
c ODE integration method: 22 => BDF with an internally generated Jacobian
c
      mf = 22
c
c Start DVODE in start-up mode (which forces it to initialize all of its
c internal arrays). Subsequent calls within the loop below use istate = 2
c
      istate = 1
c
c Optional input -- increase maximum number of steps that DVODE can take 
c internally without returning an to 20000, as 500 doesn't appear to be 
c enough for portions of our parameter space
c
      do I = 5, 10      
        rwork(I) = 0d0
        iwork(I) = 0
      enddo
c      
      iwork(6) = 200000
c
c
c Parameters to pass to rate_eq
c
      rpar(1) = yn
      rpar(2) = dl
      rpar(3) = divv
c
c Not using the Raytracing, so can put the values needed for
c iphoto = 4 option into this array... note that do_chemcool_step()
c puts everything into cgs units, before it gets passed through to here.
c It then gets passed on from rate_eq* to calc_photo, where it finally gets
c unpacked/used. Note that dl contains either the radius or the smoothing
c by this point...
c
      rpar(5) = column_est
c
      ipar(1) = 0
c
#ifndef TEST
      t_start  = 0d0
      t_end    = time
#endif
      new_step = 1
c
      do I = 1, nstep
c        print*, "step", I, new_step, tdust, y(itmp)
c
c If we're starting a new step, then figure out a reasonable timestep to
c pass to DVODE -- we need to do this here to avoid DVODE producing -ve
c abundances or -ve temperatures, which can occur if it starts with too
c large a timestep.
c
c If we're not starting a new step, then dt has already been set. 
c
        if (new_step .eq. 1) then
c
c First, save a copy of the current abundances
c
          do J = 1, nspec
            y_old(J) = y(J)
          enddo
          tdust_old = tdust
c
c Now compute ydot. On the first step, we must use rate_eq directly,
c but on subsequent steps, we can just call dvindy [if we're using
c the dvode integrator]
c
c          if (I .eq. 1) then
            call rate_eq(nspec, t_start, y, ydot, rpar, ipar)
c          else
c            call dvindy(t_start, 1, rwork(21), nspec, ydot, iflag_dvi)
c          endif
c
c Init. dt_est to an unrealistic value
c
          dt_est = 1d18

          do J = 1, nspec
            if (used_for_stepsize_control(J) .eq. 1) then
              if (ydot(J)   .ne. 0d0 .and.
     $            safety(J) .ne. 0d0 .and.
     $            y(J)      .gt. atol(J)) then
                dt_est = min(dt_est, abs(safety(J) * y(J) / ydot(J)))
              endif
            endif
          enddo
c
          ipar(2) = 0 
c
c Don't evolve past the end of the global timestep
c
          dt = min(dt_est, t_end - t_start)
        endif
c
        t_in  = t_start
        t_out = t_start + dt
        if (dt .lt. 0d0) then
          print*, 'Negative timestep!'
          ABORT(1)
        endif
        if (dt .eq. 0d0) then 
          if (t_end .eq. t_start) then
            return
          else
            print*, 'Zero timestep!'
            ABORT(2)
          endif
        endif
c
        call dvode(rate_eq, nspec, y, t_in, t_out, itol, rtol, atol, 
     $             itask, istate, iopt, rwork, lrw, iwork, liw, jac, 
     $             mf, rpar, ipar)
c
c Ensure that dvode has produced sane output values -- if not, we
c need to shrink the timestep. Note that if ipar(1) is non-zero, 
c or if istate != 2 (indicating an integrator error), we're going
c to shrink the timestep anyway, so we can skip the check.
c
        if (ipar(1) .eq. 0 .and. istate .eq. 2) then
          call check_abundances(y, ipar)
        endif
c
        if ((ipar(1) .ne. 0) .or. (istate .ne. 2)) then 
#ifdef DEBUG_EVOLVE
         if (istate .ne. 2) then
           print*, "Error: istate = ", istate
         endif
#endif
          do J = 1, nspec
            y(J) = y_old(J)
          enddo
          tdust    = tdust_old
          dt       = dt / 2d0
          if (dt .lt. 1d-8) then
            print*, "Excessively small timestep! ", dt, dt_est, 
     $              t_start, t_end
            print*, "ID: ", id_current
            print*, yn, dl, divv
            print*, y
	    print*, ydot
#ifdef TREE_RAD
            print*, "Tree rad totals: ", column_density_projection
#endif
#ifdef TREE_RAD_H2
            print*, "Tree rad h2: ", column_density_projection_h2
            if (nchem_network .ne. 1 .and. nchem_network .ne. 4) then
              print*, "Tree rad co: ", column_density_projection_co
            endif
#endif
#ifndef NOSTOP_WHEN_BELOW_MINTIMESTEP
            ABORT(100)
#endif
          endif
          new_step = 0
          ipar(1)  = 0
          istate   = 1
        else
          t_start = t_start + dt
          new_step = 1
        endif
c
        if (t_start .ge. t_end) then
          goto 100
        endif
      enddo
c
      write(*,'(A)') 'EVOLVE_ABUNDANCES: nstep exceeded,' 
      print*, id_current
      print*, dt
      print*, rpar
      print*, y
      print*, ydot
      ABORT(0)
      return

100   continue

      do I = 1, nspec-1
        non_eq_abundances(I) = y(I)
      enddo
      energy = y(itmp)
c
#endif /* CHEMCOOL */
      return      
      end
c=======================================================================
c
c
c    \\\\\\\\\\        E N D     S U B R O U T I N E      //////////
c    //////////     E V O L V E _ A B U N D A N C E S     \\\\\\\\\\
c
c=======================================================================
c
c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////      C H E C K _ A B U N D A N C E S    \\\\\\\\\\
c
c=======================================================================
c
      subroutine check_abundances(y, ipar)
#ifdef CHEMCOOL
      implicit NONE
#include "cool.h"
#include "non_eq.h"
      REAL y(nspec), max_abundance(nspec)
      REAL current_carbon, current_oxygen
      REAL diff
      integer ipar(nipar)
      integer J
c
      if (nchem_network .eq. 5) then
        max_abundance(ih2)   = 0.5d0
        max_abundance(ihp)   = 1.0d0
        max_abundance(ico)   = min(abundc, abundo)
      endif
c
c Check for NaNs
c
      do J = 1, nspec
        if (y(J) .ne. y(J)) then
#ifdef DEBUG_EVOLVE
          print*, "NaN for ID ", id_current, y
#endif
          ipar(1) = 1
          return
        endif
      enddo
c
c Check for negative abundances. Small negative values (which have
c absolute values less than the corresponding absolute tolerance)
c are accepted, but are clipped to zero. For large negative values,
c we repeat the step with a smaller dt.
c
      do J = 1, nspec
        if (y(J) .lt. 0d0) then
          if (y(J) .lt. -atol(J)) then
#ifdef DEBUG_EVOLVE
            print*, "Negative abundance for ID ", id_current, y
#endif
            ipar(1) = 1
            return
          else
            y(J) = 0d0
          endif
        endif
      enddo
c
c Don't allow ion abundances to exceed corresponding total metal
c abundances. Again, small errors are silently corrected, large
c errors lead to us repeating with a smaller timestep
c
      do J = 1, nspec
        if (J .ne. itmp .and. y(J) .gt. max_abundance(J)) then
          diff = y(J) - max_abundance(J)
          if (diff .lt. RTOL * y(J)) then
            y(J) = max_abundance(J)
          else
#ifdef DEBUG_EVOLVE
            print*, "Excess abundance for ID ", id_current, y
#endif
            ipar(1) = 1
            return
          endif
        endif
      enddo
c
#endif /* CHEMCOOL */
      return
      end
c=======================================================================
c
c
c    \\\\\\\\\\        E N D   S U B R O U T I N E      //////////
c    //////////      C H E C K _ A B U N D A N C E S    \\\\\\\\\\
c
c=======================================================================
c=======================================================================
c
c
c    \\\\\\\\\\      B E G I N   S U B R O U T I N E      //////////
c    //////////       I N I T _ T O L E R A N C E S       \\\\\\\\\\
c
c=======================================================================
c
      subroutine init_tolerances
#ifdef CHEMCOOL
      implicit NONE
#include "non_eq.h"
#include "cool.h"
c
c Set-up absolute tolerances
c

c
      if (nchem_network .eq. 5) then
        atol(ihp) = ATOL_HP
        atol(ih2) = ATOL_H2
        atol(ico) = ATOL_CO
      endif
c
c
      atol(itmp)  = ATOL_TMP
c
#endif /* CHEMCOOL */
      return
      end
c=======================================================================
c
c
c    \\\\\\\\\\        E N D     S U B R O U T I N E      //////////
c    //////////       I N I T _ T O L E R A N C E S       \\\\\\\\\\
c
c=======================================================================


#ifdef DUST_ONE_FLUID
c======================================================================= 
c======================================================================= 
      subroutine get_yield_thSput_C(Y, T)
      implicit NONE
      REAL Y, T
      REAL x1, x2, x3, x4, x5
      REAL a0, a1, a2, a3, a4, a5
c     Nozawa et al. 2006
      if (T < 3e4) then
         Y = 1e-30
         return
      else if (T > 1e10) then
         T = 1e10
      endif

      x1 = log10(T)  
      x2 = x1*x1  
      x3 = x1*x2  
      x4 = x1*x3  
      x5 = x1*x4
      a0 = -2.34333937e+02   
      a1 =  1.38485732e+02   
      a2 = -3.39021615e+01   
      a3 =  4.17705353e+00   
      a4 = -2.58281473e-01   
      a5 =  6.38827523e-03

c     unit = [micron yr^-1 cm^3] 
      Y = 10**(a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5)
      return
      end

c======================================================================= 
c======================================================================= 

      subroutine get_yield_thSput_Si(Y, T)
      implicit NONE
      REAL Y, T
      REAL x1, x2, x3, x4, x5
      REAL a0, a1, a2, a3, a4, a5
c     Nozawa et al. 2006
      if (T < 3e4) then
         Y = 1e-30
         return
      else if (T > 1e10) then
         T = 1e10
      endif

      x1 = log10(T)  
      x2 = x1*x1  
      x3 = x1*x2  
      x4 = x1*x3  
      x5 = x1*x4
      a0 = -2.34790500e+02   
      a1 =  1.33208637e+02   
      a2 = -3.13027448e+01
      a3 =  3.71345730e+00   
      a4 = -2.21823668e-01   
      a5 =  5.31746427e-03

c     unit = [micron yr^-1 cm^3] 
      Y = 10**(a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5)
      return
      end

c======================================================================= 
c======================================================================= 

      subroutine get_yield_nthSput_C(Y, vel)
      implicit NONE
      REAL Y, vel
      REAL x1, x2, x3, x4, x5
      REAL a0, a1, a2, a3, a4, a5
c     Nozawa et al. 2006
      if (vel < 20.) then
         Y = 1e-30
         return
      else if (vel > 3e5) then
         vel = 3e5
      endif

      x1 = log10(vel)
      x2 = x1*x1  
      x3 = x1*x2  
      x4 = x1*x3  
      x5 = x1*x4
      a0 = -4.87021030e+01
      a1 =  5.57114303e+01   
      a2 = -2.88275125e+01   
      a3 =  7.44225761e+00   
      a4 = -9.65469006e-01   
      a5 =  5.00882872e-02

c     unit = [micron yr^-1 cm^3] 
      Y = 10**(a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5)
      return
      end

c======================================================================= 
c======================================================================= 

      subroutine get_yield_nthSput_Si(Y, vel)
      implicit NONE
      REAL Y, vel
      REAL x1, x2, x3, x4, x5
      REAL a0, a1, a2, a3, a4, a5
c     Nozawa et al. 2006
      if (vel < 20.) then
         Y = 1e-30
         return
      else if (vel > 3e5) then
         vel = 3e5
      endif

      x1 = log10(vel)
      x2 = x1*x1  
      x3 = x1*x2  
      x4 = x1*x3  
      x5 = x1*x4
      a0 = -3.24171434e+01   
      a1 =  2.86540492e+01   
      a2 = -1.15865054e+01   
      a3 =  2.23338917e+00
      a4 = -2.08259379e-01   
      a5 =  7.36059545e-03

c     unit = [micron yr^-1 cm^3] 
      Y = 10**(a0 + a1*x1 + a2*x2 + a3*x3 + a4*x4 + a5*x5)
      return
      end

c======================================================================= 
c======================================================================= 

      subroutine get_rate_and_timescale(nH, temp, vrel, acc_hydro, )
     $     m_c(NSIZE), m_si(NSIZE), v_rel(NSIZE),
     $     m_c_dot(NSIZE), m_si_dot(NSIZE), acc_rel(NSIZE)
      implicit NONE
      REAL    nH, temp, vrel(3), acc_hydro, grain_size
      REAL    t_inv_M_C, t_inv_M_Si
      REAL    t_mdust, t_mdust_inv
      REAL    m_c(NSIZE), m_si(NSIZE)
      REAL    v_rel(NSIZE)
      REAL    m_c_dot(NSIZE), m_si_dot(NSIZE)
      REAL    acc_rel(NSIZE)
      REAL    grain_size(NSIZE)
      integer   I

      call get_tdust_inv(nH, temp, vrel, acc_hydro, 
     $                   t_inv_M_C, t_inv_M_Si)

      
      if (abs(t_inv_M_C) > abs(t_inv_M_Si)) then
         t_mdust = 1.0 / abs(t_inv_M_C)
      else
         t_mdust = 1.0 / abs(t_inv_M_Si)
      endif

      
      do I = 1, NSIZE
         m_c_dot(I) = m_c(I) * t_inv_M_C / grain_size(I)
         m_si_dot(I) = m_si(I) * t_inv_M_Si / grain_size(I)
         
      enddo
         
      return
      end
c======================================================================= 
c======================================================================= 

      subroutine get_tdust_inv(nH, temp, vrel, acc_hydro, 
     $                         t_inv_M_C, t_inv_M_Si)
      implicit NONE
      REAL    nH, temp, vrel(3), acc_hydro, grain_size
      REAL    t_inv_M_C, t_inv_M_Si

      REAL    fac_sput
      REAL    Y_th_C, Y_th_Si, Y_nth_C, Y_nth_Si
      REAL    t_thSput_C_inv, t_thSput_Si_inv
      REAL    t_nthSput_C_inv, t_nthSput_Si_inv
      REAL    t_accr_C_inv, t_accr_Si_inv
      REAL    fac_accr
      REAL    s_C, s_Si, f_C, f_Si
      REAL    t_sput

      fac_sput = 3.0 * nH / grain_size

      call get_yield_thSput_C(Y_th_C, temp)
      call get_yield_thSput_Si(Y_th_Si, temp)
      call get_yield_nthSput_C(Y_nth_C, vrel)
      call get_yield_nthSput_Si(Y_nth_Si, vrel)
 
      t_thSput_C_inv = fac_sput * Y_th_C
      t_thSput_Si_inv = fac_sput * Y_th_Si
      t_nthSput_C_inv = fac_sput * Y_nth_C
      t_nthSput_Si_inv = fac_sput * Y_nth_Si

      t_accr_C_inv = 0.0
      t_accr_Si_inv = 0.0
#ifdef DUST_GROWTH_IN_ISM
c     sticking probability
      s_C = 1.0
      s_Si = 1.0

      m_C_tot = z_c * m_gas
      m_Si_tot = z_s * m_gas

C     Condensation mass fraction
      f_C  = m_c_pred / m_C_tot
      f_Si = m_s_pred / m_Si_tot
      
      if(f_C > 1.0) then
         f_C = 1.0
      endif
      if(f_Si > 1.0) then
         f_Si = 1.0
      endif
      
      fac_accr = nH / grain_size * sqrt(temp)
  
c     units = [yr^-1]
      t_accr_C_inv  =
     $     2.15e-12 * fac_accr * (z_c / 2.07e-3) * s_C * (1.0 - f_C)
      t_accr_Si_inv = 
     $     2.01e-12 * fac_accr * (z_s / 6.83e-4) * s_Si * (1.0 - f_Si)

#endif

      t_inv_M_C = (t_accr_C_inv - t_thSput_C_inv - t_nthSput_C_inv)
      t_inv_M_Si = (t_accr_Si_inv - t_thSput_Si_inv - t_nthSput_Si_inv)


      

      return
      end
#endif
