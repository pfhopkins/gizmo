/* Evaluates every stellar-evolution table accessor on a fixed grid of (M, Z, age) and writes the doubles to a file.
   Built against galaxy_sf/stellar_evolution_tables.cc (or, for the reference, stella_dev's resolvedism_stellar_tables.cc)
   with allvars_stub.h; usage: lookup_grid TABLE OUT (the wind table is read from TABLE's directory; every rank writes OUT.<rank>).
   GIZMO's build also writes OUT.nodes.<rank>: per (Z, M) grid node lifetime, remnant mass, t_PMS and the SN and
   AGB element ejecta, evaluated exactly at the node. */
#include "allvars_stub.h"
#include TABLES_HEADER
struct stub_all All; int ThisTask, NTask; MPI_Comm GizmoNodeComm = MPI_COMM_NULL; int GizmoNodeRankOfTask;
void gizmo_node_comm_init(void) {if(GizmoNodeComm == MPI_COMM_NULL) {MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &GizmoNodeComm); MPI_Comm_rank(GizmoNodeComm, &GizmoNodeRankOfTask);}}
#ifdef GALSF_SINGLE_STARS_EVOLUTION_TABLES
int stellar_tables_verify(const std::vector<std::string> &) {return 0;} /* the hash check is GIZMO's start-up test */
#endif
void gizmo_request_controlled_stop(int code, const char *reason, const char *file, int line, const char *func) {printf("STOP %d %s\n", code, reason); MPI_Abort(MPI_COMM_WORLD, code);}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv); MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask); MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    snprintf(All.StellarTablesFile, 256, "%s", argv[1]); All.IMFSampleMinMass = 0.08;
    LOAD_TABLES();
    char fn[600]; snprintf(fn, sizeof(fn), "%s.%d", argv[2], ThisTask);
    FILE *fd = fopen(fn, "wb"); long n = 0;
#define W(x) do {double v_ = (x); fwrite(&v_, sizeof(double), 1, fd); n++;} while(0)
    /* masses: grid nodes, midpoints and points near both ends; metallicities: inside, on and outside the grid */
    const double logZs[] = {-6.5, -4.0, -3.3, -2.7, -2.154, -1.85, -1.5, -1.2};
    for(int iz = 0; iz < (int)(sizeof(logZs) / sizeof(double)); iz++) {
        for(int im = 0; im < 64; im++) {
            double logM = log10(0.07) + im * (log10(400.) - log10(0.07)) / 63.0, logZ = logZs[iz];
            W(stellar_lifetime(logM, logZ)); W(stellar_remnant_mass(logM, logZ)); W(stellar_remnant_type(logM, logZ));
            W(stellar_t_PMS(logM, logZ));
            double lt = stellar_lifetime(logM, logZ);
            const double fage[] = {1e-4, 0.01, 0.3, 0.7, 0.95, 0.999, 1.2};
            for(int ia = 0; ia < 7; ia++) {
                double age = DMAX(fage[ia] * lt, 100.), la = log10(age);
                W(stellar_v_wind(logM, logZ, la));
                for(int k = 0; k < STBL_NELEM; k++) {W(stellar_surface_abundance(logM, logZ, la, k)); W(stellar_elem_ej_wind_cumulative(logM, logZ, la, k));}
            }
            for(int k = 0; k < STBL_NELEM; k++) {W(stellar_elem_ej_SN(logM, logZ, k)); W(stellar_elem_ej_AGB(logM, logZ, k));}
        }
    }
    for(int k = 0; k < STBL_NELEM; k++) {W(stellar_type_ia_yield(k));}
    fclose(fd);
#ifdef GALSF_SINGLE_STARS_EVOLUTION_TABLES
    snprintf(fn, sizeof(fn), "%s.nodes.%d", argv[2], ThisTask);
    fd = fopen(fn, "wb");
    for(int iz = 0; iz < STBL_NZ; iz++) {
        for(int im = 0; im < STBL_NM; im++) {
            double logM = StellarTbl.log_M[im], logZ = StellarTbl.log_Z[iz];
            W(stellar_lifetime(logM, logZ)); W(stellar_remnant_mass(logM, logZ)); W(stellar_t_PMS(logM, logZ));
            for(int k = 0; k < STBL_NELEM; k++) {W(stellar_elem_ej_SN(logM, logZ, k));}
            for(int k = 0; k < STBL_NELEM; k++) {W(stellar_elem_ej_AGB(logM, logZ, k));}
        }
    }
    fclose(fd);
#endif
#if defined(GALSF_SINGLE_STARS_TYPE_IA) || defined(GALSF_RESOLVEDISM_TYPE_IA)
    /* dead white-dwarf-progenitor fraction (Type Ia hazard normalisation) at 40 ages, 10^6.3 to 10^10.5 yr, past both ends of its table */
    snprintf(fn, sizeof(fn), "%s.ia.%d", argv[2], ThisTask);
    fd = fopen(fn, "wb");
    for(int iz = 0; iz < (int)(sizeof(logZs) / sizeof(double)); iz++) {
        for(int it = 0; it < 40; it++) {W(stellar_ia_fdead(pow(10., 6.3 + it * (10.5 - 6.3) / 39.), logZs[iz]));}
    }
    fclose(fd);
#endif
    /* band luminosities of the gravity-tree sources, on the same (M, Z, age) grid */
    snprintf(fn, sizeof(fn), "%s.bands.%d", argv[2], ThisTask);
    fd = fopen(fn, "wb");
    for(int iz = 0; iz < (int)(sizeof(logZs) / sizeof(double)); iz++) {
        for(int im = 0; im < 64; im++) {
            double logM = log10(0.07) + im * (log10(400.) - log10(0.07)) / 63.0, logZ = logZs[iz], lt = stellar_lifetime(logM, logZ);
            const double fage[] = {1e-4, 0.01, 0.3, 0.7, 0.95, 0.999, 1.2};
            for(int ia = 0; ia < 7; ia++) {double la = log10(DMAX(fage[ia] * lt, 100.)); W(stellar_log_L_LW(logM, logZ, la)); W(stellar_log_L_FUV_total(logM, logZ, la)); W(stellar_log_L_FUV_M1(logM, logZ, la)); W(stellar_log_L_NUV(logM, logZ, la)); W(stellar_log_L_OPT_NIR(logM, logZ, la));}
        }
    }
    fclose(fd);
    if(ThisTask == 0) {printf("lookup_grid: %ld values\n", n);}
    MPI_Finalize();
    return 0;
}
