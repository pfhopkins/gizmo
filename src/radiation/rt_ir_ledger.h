/* RT_DIAG_IR_DOUBLE_COUNT: radiation band and cooling-step energy ledger (RT_INFRARED builds). Inert diagnostic.
 *
 * Host state only: the tallies are per-rank statics reduced with MPI and written by task 0 to <OutputDir>/rt_ir_diag.txt
 * and rt_cool_diag.txt (rt_utilities.cc). The accumulation sites sit in device-callable code (rt_functions.h, cooling.cc,
 * rt_source_injection_loop.h) behind RT_DIAG_IR_LEDGER, which precompiler_logic.h defines for the host pass only, so a
 * device pass compiles them out and those channels read zero for work done on the device. */
#pragma once

void rt_diag_ir_double_count_report(void);
/* per-band channels of rt_ir_diag.txt */
enum {RT_DIAG_ABSORBED, RT_DIAG_DONATED_IN, RT_DIAG_IR_GAS_SHARE, RT_DIAG_COOLING_TO_BAND, RT_DIAG_INJECTED, RT_DIAG_KICK_SOURCE, RT_DIAG_NCHAN, RT_DIAG_COOL_GAS, RT_DIAG_COOL_OFFERED};
void rt_diag_add(int channel, int k, double de);
void rt_diag_ir_add_donated(double de);
/* scalar tallies written to rt_cool_diag.txt; names in rt_diag_sc_names (rt_utilities.cc), same order */
enum {RT_DIAG_SC_DEU, RT_DIAG_SC_Q, RT_DIAG_SC_DTIE, RT_DIAG_SC_DTIE_CODE, RT_DIAG_SC_THICK, RT_DIAG_SC_MOL, RT_DIAG_SC_MOL_HEAT,
      RT_DIAG_SC_DUST, RT_DIAG_SC_METAL, RT_DIAG_SC_HHE, RT_DIAG_SC_COMPTON, RT_DIAG_SC_CR, RT_DIAG_SC_PE, RT_DIAG_SC_ION_HEAT,
      RT_DIAG_SC_HEAT, RT_DIAG_SC_LAMBDA, RT_DIAG_SC_ROUTED_GAS, RT_DIAG_SC_UNROUTED, RT_DIAG_SC_OFFER_NUV, RT_DIAG_SC_OFFER_IR,
      RT_DIAG_SC_RET_NUV, RT_DIAG_SC_RET_IR, RT_DIAG_SC_CLIP_MAX_10UM, RT_DIAG_SC_CLIP_MAX_DEU, RT_DIAG_SC_CLIP_MAX_ZERO,
      RT_DIAG_SC_CLIP_MIN_099E, RT_DIAG_SC_CLIP_MIN_DEU, RT_DIAG_SC_CLIP_MIN_ZERO, RT_DIAG_SC_DEU_WORK, RT_DIAG_SC_DEU_RADABS,
      RT_DIAG_SC_N_TOUSE_POS, RT_DIAG_SC_N_COOL, RT_DIAG_SC_DC_ABS_IR, RT_DIAG_SC_DC_ABS_NONIR, RT_DIAG_SC_DC_EMIT, RT_DIAG_SC_DC_COUPLE,
      RT_DIAG_SC_DK_REEMIT, RT_DIAG_SC_DK_EMIT_IMPLIED, RT_DIAG_SC_DK_COUPLE_IMPLIED, RT_DIAG_SC_M_COOL, RT_DIAG_SC_MT_GAS,
      RT_DIAG_SC_MT_DUST_COOL, RT_DIAG_SC_MT_RAD, RT_DIAG_SC_M_KICK, RT_DIAG_SC_MT_DUST_KICK, RT_DIAG_SC_MT_GAS_KICK, RT_DIAG_SC_KICK_DU,
      RT_DIAG_SC_KICK_DERAD, RT_DIAG_SC_DTIE_DISCARD, RT_DIAG_SC_DEPOSIT_K1, RT_DIAG_SC_DEPOSIT_K2, RT_DIAG_NSC};
void rt_diag_sc(int idx, double v);
extern int rt_diag_kick_mode; /* do_the_kick's mode: 0 = opening half-kick, 1 = closing */
/* terms of the most recent CoolingRate call for a cell, erg cm^3 s^-1 (lambda-type: positive = gas cooling); after DoCooling
   this is the converged-u call. dc_*: the dust balance of rt_ir_lambdadust at its final Tdust, in band energy units */
struct rt_diag_cool_terms {double mol, dust, metal, hhe, compton, pelec, h_ion, h_cr, heat, lambda, q_pre, q_post, dtie, fcorr,
    dc_abs_ir, dc_abs_nonir, dc_emit, dc_couple;};
extern thread_local struct rt_diag_cool_terms rt_diag_cr;
