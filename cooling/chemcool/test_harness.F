      program test
      implicit none
c#include "param.h"
#include "cool.h"
#include "non_eq.h"
#include "chemcool_consts.h"
c#include "cons.h"
      REAL total_time, timestep, dt, dust_temp
      REAL tstart, tend
      REAL fshield_H2, fshield_CO, AV_mean, chi_mean
      REAL dl, divv, density, internal_energy, yn, cs_amb
      REAL abh2, abhp, abco
      REAL ch_mh, ch_muHe, ch_muC, ch_muO, ch_muSi, ch_abar
      REAL non_eq_abundances(num_non_eq_species)
      REAL temperature
      integer I, nstep
      logical last_step
      REAL, parameter :: gamma=1.6667

      ch_mH   = 1.672621637e-24 !!1.66053e-24
      ch_muHe = 4.002602
      ch_muC  = 12.011
      ch_muO  = 15.9994
      ch_muSi = 28.0855

c Read in adjustable parameters, starting density, temp. and abundances

      open(1, file='test.inp', status='old')
      read(1,*) total_time, timestep
      read(1,*) abundc, abundo
      read(1,*) dust_to_gas_ratio, dust_temp
      read(1,*) G0, cosmic_ray_ion_rate
      read(1,*) NH_ext
      read(1,*) fshield_H2, fshield_CO, AV_mean, chi_mean
      read(1,*) dl, divv
      read(1,*) density, cs_amb
      read(1,*) abh2, abhp, abco
      close(1, status='keep')

c Print the values we read in
      print*,'----------------------'
      print*, 'Parameters in test.inp'
      print*, total_time, timestep
      print*, abundc, abundo
      print*, dust_to_gas_ratio, dust_temp
      print*, G0, cosmic_ray_ion_rate
      print*, NH_ext
      print*, fshield_H2, fshield_CO, AV_mean, chi_mean
      print*, dl, divv
      print*, density, cs_amb
      print*, abh2, abhp, abco
      print*,'----------------------'

      tdust=dust_temp
c      internal_energy=cs_amb**2e0/(gamma*(gamma-1.))
      internal_energy=cs_amb**2e0 !!*3.e0/2.d0
      internal_energy=internal_energy*density
      internal_energy=internal_energy/(gamma*(gamma-1.))
c      print*,'internal energy=',internal_energy

c These are the same defaults as in the Config file, and are things
c we probably won't want to change while testing
      deff = 1e0
      abundsi = 0.0
      abundD  = 2.6e-5
      abundM  = 1e-7
      abundN  = 0e0
      f_rsc   = 1e0
      phi_pah = 0.5e0
      AV_conversion_factor = 5.348e-22
      redshift = 0e0
      h2_form_ex  = 0e0
      h2_form_kin = 0e0
c      xray_scaling = 1e0

      iphoto   = 0
      iflag_mn = 1
      iflag_ad = 1
      iflag_atom = 2
      iflag_3bh2a = 1
      iflag_3bh2b = 1
      iflag_h3pra = 1
      isrf_option = 1
      no_chem     = 0

c Set AV_ext based on specificed NH_ext, conversion factor and dust:gas ratio
      AV_ext = AV_conversion_factor * NH_ext * dust_to_gas_ratio

c Mean atomic weight
      ch_abar = 1.0 + abhe*ch_muHe + abundc*ch_muC
     $        + abundo*ch_muO + abundsi*ch_muSi

      dt = timestep
      yn = density / ch_mH /ch_abar

      non_eq_abundances(ih2) = abh2
      non_eq_abundances(ihp) = abhp
#if CHEMISTRYNETWORK == 5
      non_eq_abundances(ico) = abco
#endif

      temperature=internal_energy/yn/boltz*2./3.!*(gamma*(gamma-1.))
      nstep = int(total_time / timestep) !!+ 1
      tstart = 0e0

      call coolinmo
      call cheminmo
      call init_tolerances

c Open logfile
      open(2, file='test.log', status='replace')

c Write initial setup to logfile
#if CHEMISTRYNETWORK == 5
        write(2,*) tstart, non_eq_abundances(ih2), 
     $             non_eq_abundances(ihp), 
     $             non_eq_abundances(ico),
     $             internal_energy, temperature
#else
        write(2,*) tstart, non_eq_abundances(ih2), 
     $             non_eq_abundances(ihp), 0.0,
     $             internal_energy,temperature
#endif


      do I = 1, nstep
        dt   = timestep
        tend = tstart + dt
        if (tend .gt. total_time) then
          dt = total_time - tstart
          last_step = .true.
        else
          last_step = .false.
        endif

        print*,'before evolve_abundances'
        print*,'abun (H2, Hp)=',non_eq_abundances
        print*,'eint=',internal_energy,' temp=',temperature
        print*,dt, dl, yn, divv, internal_energy, 
     $                         redshift, non_eq_abundances,
     $                         fshield_H2, fshield_CO, AV_mean, 
     $                         chi_mean, dust_temp



        call evolve_abundances(dt, dl, yn, divv, internal_energy, 
     $                         redshift, non_eq_abundances,
     $                         fshield_H2, fshield_CO, AV_mean, 
     $                         chi_mean, dust_temp)


        temperature=internal_energy/yn/boltz*2./3.!*(gamma*(gamma-1.))
        print*,'Step=',I
        print*,'abun (H2, Hp)=',non_eq_abundances
        print*,'eint=',internal_energy,' temp=',temperature
        print*,dt, dl, yn, divv, internal_energy, 
     $                         redshift, non_eq_abundances,
     $                         fshield_H2, fshield_CO, AV_mean, 
     $                         chi_mean, dust_temp



c Write output to logfile
#if CHEMISTRYNETWORK == 5
        write(2,*) tend, non_eq_abundances(ih2), 
     $             non_eq_abundances(ihp),
c     $             non_eq_abundances(iha),
     $             non_eq_abundances(ico),
c     $             non_eq_abundances(icp),
     $             internal_energy, temperature
#else
        write(2,*) tend, non_eq_abundances(ih2), 
     $             non_eq_abundances(ihp), 0.0,
     $             internal_energy,temperature
#endif

        if (last_step) then
          goto 100
        endif

        tstart = tend
      enddo

 100  continue
 
      end
