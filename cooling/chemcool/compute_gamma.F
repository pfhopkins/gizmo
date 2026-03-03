#include "../../GIZMO_config.h"
c=======================================================================
c
c    \\\\\\\\\\        B E G I N  S U B R O U T I N E     //////////
c    //////////          C O M P U T E _ G A M M A        \\\\\\\\\\
c
c=======================================================================

      subroutine compute_gamma(abh2, ekn, gamma)
#ifdef CHEMCOOL
      implicit NONE
#include "cool.h"
#include "highn_gamma.h"
      REAL abh2, ekn, lekn, gamma
      REAL scale, scale_h2, gl, gh
      integer index, index1, index2
 
      lekn = dlog10(ekn)

      if (lekn .le. 1.5) then
        index = 1
        scale = 0d0
      elseif (lekn .ge. 4.5) then
        index = 25
        scale = 0d0
      else
        index = int((lekn - 1.5) / 0.125) + 1
        scale = (lekn - 1.5 - 0.125*(index-1)) / 0.125
      endif
c
      if (abh2 .eq. 0d0) then
        gamma = 5d0 / 3d0
      elseif (abh2 .lt. 0.1d0) then
        gl = 5d0 / 3d0
        if (scale .gt. 0d0) then        
          gh = hng(index) + (hng(index+1) - hng(index)) * scale
        else
          gh = hng(index)
        endif
        gamma = gl + (abh2 / 0.1d0) * (gh - gl)
      elseif (abh2 .eq. 0.5d0) then
        index = 100 + index
        if (scale .gt. 0d0) then        
          gamma = hng(index) + (hng(index+1) - hng(index)) * scale
        else
          gamma = hng(index)
        endif
      else
        scale_h2 = (abh2 / 0.1d0) - int(abh2 / 0.1d0)
        index1   = index + 25 * (int(abh2 / 0.1d0) - 1)
        index2   = index + 25 * int(abh2 / 0.1d0)
        if (scale .gt. 0d0) then        
          gl = hng(index1) + (hng(index1+1) - hng(index1)) * scale
          gh = hng(index2) + (hng(index2+1) - hng(index2)) * scale
        else
          gl = hng(index1)
          gh = hng(index2)
        endif
        gamma = gl + (gh - gl) * scale_h2
      endif

#endif /* CHEMCOOL */
      return
      end
c=======================================================================
c
c    \\\\\\\\\\          E N D    S U B R O U T I N E     //////////
c    //////////          C O M P U T E _ G A M M A        \\\\\\\\\\
c
c=======================================================================
