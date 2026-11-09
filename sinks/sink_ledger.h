/*! \file sink_ledger.h
 *  \brief record layouts for the sink event ledger (see sink_ledger.cc). Each layout is one field list,
 *   expanded into both the C struct and the numpy dtype written to the .npy header, so they cannot disagree.
 *   All fields are 8 bytes (F = double, U = unsigned 64-bit integer), so structs have no padding.
 */
#ifndef SINK_LEDGER_H
#define SINK_LEDGER_H
#include <stddef.h>

#if defined(SINK_PARTICLES) && (defined(OUTPUT_SINK_ACCRETION_HIST) || defined(OUTPUT_SINK_FORMATION_PROPS))
#define SINK_LEDGER_ACTIVE
#endif
#ifndef SINK_LEDGER_FLUSH_INTERVAL_SEC
#define SINK_LEDGER_FLUSH_INTERVAL_SEC 60. /* wall-clock seconds between flushes; restarts and snapshots also flush */
#endif

/* field naming: *_mass_dyn = dynamical mass (P.Mass), *_mass = Sink_Mass, *_reservoir = Sink_Mass_Reservoir */

/* one row per gas element captured (fully or partially); sink state is as of the start of this accretion pass;
   dx,dv = gas - sink, dv from the gas element's velocity before any kick in this accretion pass (half-kicked if inactive) */
#define SINK_LEDGER_ACCRETION_FIELDS(F,U) \
    F(time) U(sink_id) U(sink_child) F(sink_mass_dyn) F(sink_mass) F(sink_mass_reservoir) \
    F(sink_x) F(sink_y) F(sink_z) F(sink_vx) F(sink_vy) F(sink_vz) \
    U(gas_id) U(gas_child) U(gas_gen) F(gas_mass) F(accreted_mass) \
    F(dx) F(dy) F(dz) F(dvx) F(dvy) F(dvz) F(gas_u) F(gas_rho) F(gas_Bx) F(gas_By) F(gas_Bz)

/* one row per sink formed; cell_mass is the converted cell's mass, sink_mass/sink_mass_reservoir its initial split */
#define SINK_LEDGER_FORMATION_FIELDS(F,U) \
    F(time) U(sink_id) U(sink_child) F(cell_mass) F(sink_mass) F(sink_mass_reservoir) \
    F(x) F(y) F(z) F(vx) F(vy) F(vz) F(Bx) F(By) F(Bz) F(u) F(rho) F(cs_eff) F(cell_size) F(NH) F(dv2_turb) F(min_dist_to_sink)

/* one row per sink-sink merger; sink 1 absorbs sink 2 */
#define SINK_LEDGER_MERGER_FIELDS(F,U) \
    F(time) U(id1) U(child1) F(mass_dyn1) F(mass1) F(mass_reservoir1) F(x1) F(y1) F(z1) F(vx1) F(vy1) F(vz1) \
    U(id2) U(child2) F(mass_dyn2) F(mass2) F(mass_reservoir2) F(x2) F(y2) F(z2) F(vx2) F(vy2) F(vz2)

#define SINK_LEDGER_STRUCT_F(n) double n;
#define SINK_LEDGER_STRUCT_U(n) unsigned long long n;
struct sink_ledger_accretion {SINK_LEDGER_ACCRETION_FIELDS(SINK_LEDGER_STRUCT_F, SINK_LEDGER_STRUCT_U)};
struct sink_ledger_formation {SINK_LEDGER_FORMATION_FIELDS(SINK_LEDGER_STRUCT_F, SINK_LEDGER_STRUCT_U)};
struct sink_ledger_merger {SINK_LEDGER_MERGER_FIELDS(SINK_LEDGER_STRUCT_F, SINK_LEDGER_STRUCT_U)};
/* restart truncation reads each record's time from its first 8 bytes */
static_assert(offsetof(struct sink_ledger_accretion, time) == 0, "time must be the first ledger field");
static_assert(offsetof(struct sink_ledger_formation, time) == 0, "time must be the first ledger field");
static_assert(offsetof(struct sink_ledger_merger, time) == 0, "time must be the first ledger field");

enum {SINK_LEDGER_ACCRETION = 0, SINK_LEDGER_FORMATION = 1, SINK_LEDGER_MERGER = 2, SINK_LEDGER_NTYPES = 3};
void sink_ledger_open(void);
void sink_ledger_append(int type, const void *record, size_t nbytes);
void sink_ledger_flush(void);

#endif
