#ifdef SGCHEM
      integer isize, isize2
      integer netab_x
      parameter (netab_x = 21)
c     energy table: netab_x blocks in x = abh2/(1+abhe-abh2), dx = 1/(netab_x-1)
      parameter (isize = 50000 * netab_x)
      parameter (isize2 = 50000)
      REAL temp_table(isize), gamma_table(isize)
      REAL eh2_table(isize2)

      common /gamma_data/temp_table, gamma_table,
     $        eh2_table
#endif
