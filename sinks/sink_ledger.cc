/*! \file sink_ledger.cc
 *  \brief Restart-safe log of every sink formation, gas accretion and sink merger event.
 *
 *  Output: OutputDir/sink_events/{accretion,formation,merger}.npy, one file per event type, written by rank 0,
 *  plus README.txt (columns, units, notes) and restarts.log. Each .npy file is a standard NumPy array file
 *  holding a 1-D structured array, so np.load("accretion.npy") reads it directly (mmap_mode="r" for large
 *  files). The record layouts are defined in sink_ledger.h; every field is 8 bytes (double or uint64).
 *
 *  Writing: event sites call sink_ledger_append(), which only copies the record into a per-rank buffer.
 *  sink_ledger_flush() gathers all buffers to rank 0, which appends them to the files and then updates the
 *  record count in each file's header. It is collective, so it is only called where every rank already
 *  synchronizes: on a wall-clock cadence piggybacked on the run loop's per-step broadcast (core/run.cc),
 *  before every restart write, and before every snapshot. Records are in time order between flushes but not
 *  within one. If a run dies mid-flush, the header undercounts until the next flush recounts from the file size;
 *  readers meanwhile miss the newest records.
 *
 *  Restarts: a restart (flag 1 from restart files, flag 2 from a snapshot) resumes from All.Time, and every
 *  event with a later time belongs to a trajectory that is about to be recomputed. sink_ledger_open() removes
 *  those records, so the files never hold duplicates. Restart files and snapshots are written after a flush,
 *  so all events up to All.Time are already on disk.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdint.h>
#include <string>
#include <vector>
#include "../declarations/allvars.h"
#include "../core/proto.h"

#ifdef SINK_LEDGER_ACTIVE

/* .npy version 1.0: 6-byte magic, 2 version bytes, uint16 little-endian header length, then a Python dict
   literal padded with spaces and ending in '\n'. A fixed 4096-byte header (a multiple of 64, as the format
   requires) lets the record count be rewritten in place as the file grows. */
#define SINK_LEDGER_HEADER_BYTES 4096
#define SINK_LEDGER_NPY_PREAMBLE_BYTES 10

static const char *ledger_name[SINK_LEDGER_NTYPES] = {"accretion", "formation", "merger"};

static const size_t ledger_record_bytes[SINK_LEDGER_NTYPES] = {
    sizeof(struct sink_ledger_accretion), sizeof(struct sink_ledger_formation), sizeof(struct sink_ledger_merger)};

/* numpy dtype descriptions, "('name', '@f8'), ..." with @ standing for the byte-order mark, expanded from the
   same field lists as the structs */
#define SINK_LEDGER_DESCR_F(n) "('" #n "', '@f8'), "
#define SINK_LEDGER_DESCR_U(n) "('" #n "', '@u8'), "
static const char *ledger_descr[SINK_LEDGER_NTYPES] = {
    SINK_LEDGER_ACCRETION_FIELDS(SINK_LEDGER_DESCR_F, SINK_LEDGER_DESCR_U),
    SINK_LEDGER_FORMATION_FIELDS(SINK_LEDGER_DESCR_F, SINK_LEDGER_DESCR_U),
    SINK_LEDGER_MERGER_FIELDS(SINK_LEDGER_DESCR_F, SINK_LEDGER_DESCR_U)};

static const char *ledger_notes[SINK_LEDGER_NTYPES] = {
    "One record per gas element captured (fully or partially). Sink identity = (sink_id, sink_child).\n"
    "  Sink state is as of the start of this accretion pass. dx,dv = gas - sink; dv uses the gas element's velocity\n"
    "  before any kick in this pass (half-kicked if inactive). accreted_mass = mass moved into sink + reservoir.",
    "One record per sink formed. cell_mass = mass of the converted cell; sink_mass and sink_mass_reservoir are\n"
    "  its initial split. Position, velocity and gas properties are those of the converted cell.",
    "One record per sink-sink merger; sink 1 absorbs sink 2. vx2..vz2 are half a step behind if sink 2 was inactive."};

static std::string ledger_buffer[SINK_LEDGER_NTYPES]; /* this rank's records not yet flushed, as raw bytes */


static int ledger_type_enabled(int type)
{
#ifdef OUTPUT_SINK_ACCRETION_HIST
    if(type == SINK_LEDGER_ACCRETION || type == SINK_LEDGER_MERGER) {return 1;}
#endif
#ifdef OUTPUT_SINK_FORMATION_PROPS
    if(type == SINK_LEDGER_FORMATION) {return 1;}
#endif
    return 0;
}

/* analysis and conversion modes (RestartFlag > 2) do not evolve the system: no ledger */
static int ledger_run_enabled(void) {return RestartFlag <= 2;}

static void ledger_path(int type, char *path)
{
    snprintf(path, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%ssink_events/%s.npy", All.OutputDir, ledger_name[type]);
}

static void ledger_fail(const char *what, const char *path)
{
    printf("sink ledger: %s '%s'\n", what, path);
    endrun(1);
}

/* the dtype description in this machine's byte order, e.g. "[('time', '<f8'), ('sink_id', '<u8'), ...]" */
static std::string ledger_dtype(int type)
{
    const uint16_t one = 1;
    const char order = (*(const char *)&one) ? '<' : '>';
    std::string d(ledger_descr[type]);
    for(char &c : d) {if(c == '@') {c = order;}} /* '@' cannot occur in a field name */
    d.resize(d.size() - 2); /* trailing ", " */
    return "[" + d + "]";
}

/* write the fixed-size .npy header for n_records records at the start of an open file */
static void ledger_write_header(FILE *fd, int type, long n_records)
{
    char header[SINK_LEDGER_HEADER_BYTES];
    memset(header, ' ', sizeof(header));
    memcpy(header, "\x93NUMPY\x01\x00", 8);
    const uint16_t dict_bytes = SINK_LEDGER_HEADER_BYTES - SINK_LEDGER_NPY_PREAMBLE_BYTES;
    header[8] = (char)(dict_bytes & 0xff); /* stored little-endian regardless of the data's byte order */
    header[9] = (char)(dict_bytes >> 8);
    std::string dict = "{'descr': " + ledger_dtype(type) + ", 'fortran_order': False, 'shape': (" + std::to_string(n_records) + ",), }";
    if(dict.size() >= dict_bytes) {ledger_fail("header too long for", ledger_name[type]);}
    memcpy(header + SINK_LEDGER_NPY_PREAMBLE_BYTES, dict.data(), dict.size());
    header[sizeof(header) - 1] = '\n';
    fseeko(fd, 0, SEEK_SET);
    if(fwrite(header, 1, sizeof(header), fd) != sizeof(header)) {ledger_fail("cannot write header for", ledger_name[type]);}
}

static void ledger_create(int type)
{
    char path[DEFAULT_PATH_BUFFERSIZE_TOUSE];
    ledger_path(type, path);
    FILE *fd = fopen(path, "wb");
    if(!fd) {ledger_fail("cannot create", path);}
    ledger_write_header(fd, type, 0);
    fclose(fd);
}

/* whole records in an open ledger file, from its size (a partial record left by a crash is not counted) */
static long ledger_count_records(FILE *fd, int type)
{
    fseeko(fd, 0, SEEK_END);
    off_t file_bytes = ftello(fd);
    return (file_bytes < SINK_LEDGER_HEADER_BYTES) ? 0 : (long)((file_bytes - SINK_LEDGER_HEADER_BYTES) / ledger_record_bytes[type]);
}

/* Remove records with time > t_keep, and any partial record left by a crash during a write. Returns the
   number of whole records removed. Records after the restart point were all written by later flushes, so
   they form a contiguous tail: scan back from the end until a record with time <= t_keep. */
static long ledger_truncate(int type, double t_keep)
{
    char path[DEFAULT_PATH_BUFFERSIZE_TOUSE];
    ledger_path(type, path);
    FILE *fd = fopen(path, "r+b");
    if(!fd) {ledger_create(type); return 0;} /* e.g. a flag 2 restart into a new output directory */

    /* refuse to cut a file this build cannot parse: not .npy, or a different record layout */
    char header[SINK_LEDGER_HEADER_BYTES + 1] = {0};
    size_t got = fread(header, 1, SINK_LEDGER_HEADER_BYTES, fd);
    if(got == 0) {fclose(fd); ledger_create(type); return 0;} /* crashed while creating it */
    if(got != SINK_LEDGER_HEADER_BYTES || memcmp(header, "\x93NUMPY", 6) || !strstr(header + SINK_LEDGER_NPY_PREAMBLE_BYTES, ledger_dtype(type).c_str()))
        {ledger_fail("record layout differs from this build in", path);}

    const size_t record_bytes = ledger_record_bytes[type];
    const long n_records = ledger_count_records(fd, type), chunk_records = 65536;
    std::vector<char> chunk(record_bytes * chunk_records);
    long n_keep = n_records;
    while(n_keep > 0)
    {
        long n_read = (n_keep < chunk_records) ? n_keep : chunk_records, first = n_keep - n_read; /* records [first, n_keep) */
        fseeko(fd, SINK_LEDGER_HEADER_BYTES + (off_t)first * record_bytes, SEEK_SET);
        if(fread(chunk.data(), record_bytes, n_read, fd) != (size_t)n_read) {ledger_fail("read error in", path);}
        int found_kept_record = 0;
        for(long r = n_read - 1; r >= 0; r--)
        {
            double time; /* every record starts with its time */
            memcpy(&time, chunk.data() + r * record_bytes, sizeof(double));
            if(time <= t_keep) {n_keep = first + r + 1; found_kept_record = 1; break;}
        }
        if(found_kept_record) {break;}
        n_keep = first;
    }
    ledger_write_header(fd, type, n_keep);
    fclose(fd);
    if(truncate(path, SINK_LEDGER_HEADER_BYTES + (off_t)n_keep * record_bytes)) {ledger_fail("cannot truncate", path);}
    return n_records - n_keep;
}

/* sink_events/README.txt: what each file holds, its columns, and the code units */
static void ledger_write_readme(void)
{
    char path[DEFAULT_PATH_BUFFERSIZE_TOUSE];
    snprintf(path, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%ssink_events/README.txt", All.OutputDir);
    FILE *fd = fopen(path, "w");
    if(!fd) {ledger_fail("cannot create", path);}
    fprintf(fd, "Sink event ledger written by GIZMO. Each <type>.npy holds a NumPy structured array:\n"
                "    events = numpy.load('accretion.npy')          # mmap_mode='r' for large files\n"
                "    events['time'], events['sink_id'], ...\n"
                "Records are in time order between flushes (about every %g s of wall time) but not within one: sort by\n"
                "time if order matters. On a restart, records later than the restart time are removed (see restarts.log),\n"
                "so the files never hold duplicates. Sinks also gain mass continuously from their reservoir and lose it through\n"
                "feedback channels, none of which is logged here: summing accreted_mass does not give a sink's mass history;\n"
                "use snapshots for that.\n\n"
                "Mass columns: *_mass_dyn = dynamical (particle) mass, *_mass = sink mass, *_reservoir = alpha-disk\n"
                "reservoir (0 when not used).\n\n"
                "Code units: UnitLength_in_cm=%.10g UnitMass_in_g=%.10g UnitVelocity_in_cm_per_s=%.10g UnitB_in_gauss=%.10g\n"
                "(time in UnitLength/UnitVelocity; u in UnitVelocity^2; rho in UnitMass/UnitLength^3; NH in UnitMass/UnitLength^2).\n"
                "formation rho, cell_size, NH and dv2_turb are physical; all other quantities are code (comoving) units.\n",
                (double)SINK_LEDGER_FLUSH_INTERVAL_SEC, UNIT_LENGTH_IN_CGS, UNIT_MASS_IN_CGS, UNIT_VEL_IN_CGS, UNIT_B_IN_GAUSS);
    for(int type = 0; type < SINK_LEDGER_NTYPES; type++)
    {
        if(!ledger_type_enabled(type)) {continue;}
        fprintf(fd, "\n%s.npy\n  %s\n  dtype: %s\n", ledger_name[type], ledger_notes[type], ledger_dtype(type).c_str());
    }
    fclose(fd);
}

/* called once at startup by all ranks (open_outputfiles): create the files on a fresh start, or remove the
   events after the restart point on a restart, and note the restart in sink_events/restarts.log */
void sink_ledger_open(void)
{
    if(ThisTask == 0 && ledger_run_enabled())
    {
        char path[DEFAULT_PATH_BUFFERSIZE_TOUSE];
        snprintf(path, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%ssink_events", All.OutputDir);
        mkdir(path, 02755);
        ledger_write_readme();

        std::string dropped_summary;
        for(int type = 0; type < SINK_LEDGER_NTYPES; type++)
        {
            if(!ledger_type_enabled(type)) {continue;}
            if(RestartFlag == 0) {ledger_create(type); continue;} /* a fresh start overwrites any previous ledger */
            long n_dropped = ledger_truncate(type, All.Time);
            dropped_summary += std::string(" ") + ledger_name[type] + "_dropped=" + std::to_string(n_dropped);
            if(n_dropped) {printf("sink ledger: removed %ld %s records later than the restart time %.17g\n", n_dropped, ledger_name[type], All.Time);}
        }

        snprintf(path, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%ssink_events/restarts.log", All.OutputDir);
        if(RestartFlag == 0) {unlink(path);}
        else
        {
            FILE *fd = fopen(path, "a");
            if(fd) {fprintf(fd, "restart flag=%d time=%.17g%s\n", RestartFlag, All.Time, dropped_summary.c_str()); fclose(fd);}
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

/* record one event on this rank; safe to call from inside OpenMP-threaded loops */
void sink_ledger_append(int type, const void *record, size_t nbytes)
{
    if(!ledger_run_enabled() || !ledger_type_enabled(type)) {return;}
    #pragma omp critical(sink_ledger_append)
    ledger_buffer[type].append((const char *)record, nbytes);
}

/* Collective: every rank must call this at the same point. Gathers each event type's buffered records to
   rank 0, which appends them to that type's file and then updates the record count in its header. Both
   early-outs depend only on compile-time flags and RestartFlag, so every rank takes the same path. */
void sink_ledger_flush(void)
{
    if(!ledger_run_enabled()) {return;}
    for(int type = 0; type < SINK_LEDGER_NTYPES; type++)
    {
        if(!ledger_type_enabled(type)) {continue;}
        std::string &mine = ledger_buffer[type];

        int my_bytes = (int)mine.size();
        std::vector<int> bytes_per_rank(ThisTask == 0 ? NTask : 0), offsets(ThisTask == 0 ? NTask : 0);
        MPI_Gather(&my_bytes, 1, MPI_INT, bytes_per_rank.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        long total_bytes = 0;
        if(ThisTask == 0) {for(int r = 0; r < NTask; r++) {offsets[r] = (int)total_bytes; total_bytes += bytes_per_rank[r];}}
        if(total_bytes > 2147483647L) {printf("sink ledger: a flush of %ld bytes exceeds the MPI int count limit\n", total_bytes); endrun(1);}
        std::vector<char> all_records(total_bytes > 0 ? total_bytes : 1);
        MPI_Gatherv(mine.data(), my_bytes, MPI_CHAR, all_records.data(), bytes_per_rank.data(), offsets.data(), MPI_CHAR, 0, MPI_COMM_WORLD);
        mine.clear();

        if(ThisTask == 0 && total_bytes > 0)
        {
            char path[DEFAULT_PATH_BUFFERSIZE_TOUSE];
            ledger_path(type, path);
            FILE *fd = fopen(path, "r+b");
            if(!fd) {ledger_fail("cannot append to", path);}
            fseeko(fd, 0, SEEK_END);
            if(fwrite(all_records.data(), 1, total_bytes, fd) != (size_t)total_bytes) {ledger_fail("short write (disk full?) to", path);}
            fflush(fd); /* records reach the file before the header counts them */
            ledger_write_header(fd, type, ledger_count_records(fd, type));
            fclose(fd);
        }
    }
}

#else /* ledger compiled out: keep the call sites unconditional */
void sink_ledger_open(void) {}
void sink_ledger_append(int type, const void *record, size_t nbytes) {}
void sink_ledger_flush(void) {}
#endif
