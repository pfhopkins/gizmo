/*! \file mg_gradient_correction.h
 *  \brief header for the modified-gradient (MG) exact div(B)=0 correction method
 *
 *  Implements Tu, Wang, Gao & Tang (2026), arXiv:2603.04077.
 */
#ifndef MG_GRADIENT_CORRECTION_H
#define MG_GRADIENT_CORRECTION_H

#ifdef MHD_MODIFIED_GRADIENT
void mg_gradient_correction_calc(void);
#endif

#endif /* MG_GRADIENT_CORRECTION_H */
