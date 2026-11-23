/* Keyed GADGET-4-format group catalogue: members sorted by (group, type, key), per-type offsets, written by task 0. */
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <vector>
#include "../../declarations/allvars.h"
#include "../../core/proto.h"

#if defined(FOF) && defined(FOF_WRITE_KEYED_CATALOGUE)
#include "hbt_catalogue.h"
#include "hbt_catalogue_mpi.h"

static int hbt_member_cmp(const void *a, const void *b)
{
    const hbt_member_t *x = (const hbt_member_t *)a, *y = (const hbt_member_t *)b;
    if(x->GrNr != y->GrNr) {return (x->GrNr < y->GrNr) ? -1 : 1;}
    if(x->Type != y->Type) {return (x->Type < y->Type) ? -1 : 1;}
    if(x->ID != y->ID) {return (x->ID < y->ID) ? -1 : 1;}
    if(x->Child != y->Child) {return (x->Child < y->Child) ? -1 : 1;}
    return 0;
}

/* concatenate every task's bytes on task 0, in task order */
static void hbt_gather_bytes(const void *local, long long nbytes, std::vector<char> &out)
{
    std::vector<long long> nb(NTask);
    MPI_Gather(&nbytes, 1, MPI_LONG_LONG, nb.data(), 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    std::vector<int> cnt(NTask), dsp(NTask);
    int bad = 0;
    if(ThisTask == 0)
    {
        long long tot = 0;
        for(int t = 0; t < NTask; t++) {if(tot > INT_MAX || nb[t] > INT_MAX) {bad = 1;} cnt[t] = (int)nb[t]; dsp[t] = (int)tot; tot += nb[t];}
        if(tot > INT_MAX) {bad = 1;}
        if(!bad) {out.resize(tot > 0 ? tot : 1);}
    }
    MPI_Bcast(&bad, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(bad) {if(ThisTask == 0) {printf("KEYED CATALOGUE: more than 2^31 bytes in one dataset; not supported yet\n");} endrun(1187);}
    MPI_Gatherv((void *)local, (int)nbytes, MPI_BYTE, ThisTask == 0 ? out.data() : NULL, cnt.data(), dsp.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
}

static void hbt_h5_write(hid_t loc, const char *name, hid_t type, int rank, hsize_t n, hsize_t ncol, const void *data)
{
    hsize_t dims[2] = {n, ncol};
    hid_t space = H5Screate_simple(rank, dims, NULL);
    hid_t ds = H5Dcreate(loc, name, type, space, H5P_DEFAULT);
    if(n > 0) {H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);}
    H5Dclose(ds); H5Sclose(space);
}

static void hbt_h5_attr(hid_t loc, const char *name, hid_t type, const void *v)
{
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t at = H5Acreate(loc, name, type, space, H5P_DEFAULT);
    H5Awrite(at, type, v); H5Aclose(at); H5Sclose(space);
}

struct hbt_group_row {long long OffsetType[HBT_NTYPES]; int Len, LenType[HBT_NTYPES]; float Mass, MassType[HBT_NTYPES], CM[3], Vel[3], Sfr;};

void hbt_write_keyed_catalogue(int num, group_properties *Grp, int ngroups, hbt_member_t *mem, long nmem)
{
    double t0 = my_second();
    /* members in global (group, type, key) order: per type, the group-ordered member list */
    parallel_sort(mem, nmem, sizeof(hbt_member_t), hbt_member_cmp);

    std::vector<int> lentype((size_t)ngroups * HBT_NTYPES + 1);
    for(int g = 0; g < ngroups; g++) {for(int t = 0; t < HBT_NTYPES; t++) {lentype[(size_t)g * HBT_NTYPES + t] = Grp[g].LenType[t];}}
    std::vector<long long> off((size_t)ngroups * HBT_NTYPES + 1), tot(HBT_NTYPES);
    hbt_group_type_offsets_global(lentype.data(), ngroups, off.data(), tot.data(), MPI_COMM_WORLD);

    std::vector<hbt_group_row> rows(ngroups > 0 ? ngroups : 1);
    const double vfac = sqrt(All.cf_a3inv); /* snapshot velocity convention (file_io/io.cc IO_VEL) */
    for(int g = 0; g < ngroups; g++)
    {
        hbt_group_row &r = rows[g];
        r.Len = Grp[g].Len; r.Mass = (float)Grp[g].Mass;
#ifdef GALSF
        r.Sfr = (float)Grp[g].Sfr;
#else
        r.Sfr = 0;
#endif
        for(int t = 0; t < HBT_NTYPES; t++) {r.LenType[t] = Grp[g].LenType[t]; r.MassType[t] = (float)Grp[g].MassType[t]; r.OffsetType[t] = off[(size_t)g * HBT_NTYPES + t];}
        for(int k = 0; k < 3; k++) {r.CM[k] = (float)Grp[g].CM[k]; r.Vel[k] = (float)(Grp[g].Vel[k] * vfac);}
    }
    std::vector<char> grows;
    hbt_gather_bytes(rows.data(), (long long)ngroups * (long long)sizeof(hbt_group_row), grows);

    long long nloc = ngroups, ngtot = 0, nmloc = nmem, nmtot = 0;
    MPI_Allreduce(&nloc, &ngtot, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&nmloc, &nmtot, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if(nmtot != tot[0] + tot[1] + tot[2] + tot[3] + tot[4] + tot[5]) {if(ThisTask == 0) {printf("KEYED CATALOGUE: %lld members but group lengths sum to %lld\n", nmtot, tot[0] + tot[1] + tot[2] + tot[3] + tot[4] + tot[5]);} endrun(1188);}

    hid_t file = 0, gids = 0;
    char buf[DEFAULT_PATH_BUFFERSIZE_TOUSE], fname[DEFAULT_PATH_BUFFERSIZE_TOUSE];
    if(ThisTask == 0)
    {
        snprintf(fname, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s/groups_%03d/fof_subhalo_tab_%03d.hdf5", All.OutputDir, num, num);
        file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        hid_t hd = H5Gcreate(file, "/Header", 0);
        long long zero = 0, one = 1; double redshift = 1. / All.Time - 1.;
        hbt_h5_attr(hd, "Ngroups_ThisFile", H5T_NATIVE_LLONG, &ngtot); hbt_h5_attr(hd, "Ngroups_Total", H5T_NATIVE_LLONG, &ngtot);
        hbt_h5_attr(hd, "Nsubhalos_ThisFile", H5T_NATIVE_LLONG, &zero); hbt_h5_attr(hd, "Nsubhalos_Total", H5T_NATIVE_LLONG, &zero);
        hbt_h5_attr(hd, "Nids_ThisFile", H5T_NATIVE_LLONG, &nmtot); hbt_h5_attr(hd, "Nids_Total", H5T_NATIVE_LLONG, &nmtot);
        hbt_h5_attr(hd, "NumFiles", H5T_NATIVE_LLONG, &one);
        hbt_h5_attr(hd, "Time", H5T_NATIVE_DOUBLE, &All.Time); hbt_h5_attr(hd, "Redshift", H5T_NATIVE_DOUBLE, &redshift);
        hbt_h5_attr(hd, "BoxSize", H5T_NATIVE_DOUBLE, &All.BoxSize); hbt_h5_attr(hd, "HubbleParam", H5T_NATIVE_DOUBLE, &All.HubbleParam);
        H5Gclose(hd);

        hbt_group_row *R = (hbt_group_row *)grows.data(); const hsize_t n = (hsize_t)ngtot;
        std::vector<int> i1(n + 1), i6(6 * n + 1), first(n + 1, -1), nsub(n + 1, 0); std::vector<long long> l6(6 * n + 1);
        std::vector<float> f1(n + 1), f6(6 * n + 1), f3(3 * n + 1);
        hid_t gg = H5Gcreate(file, "/Group", 0);
        for(hsize_t g = 0; g < n; g++) {i1[g] = R[g].Len;} hbt_h5_write(gg, "GroupLen", H5T_NATIVE_INT, 1, n, 1, i1.data());
        for(hsize_t g = 0; g < n; g++) {for(int t = 0; t < 6; t++) {i6[6 * g + t] = R[g].LenType[t]; l6[6 * g + t] = R[g].OffsetType[t]; f6[6 * g + t] = R[g].MassType[t];}}
        hbt_h5_write(gg, "GroupLenType", H5T_NATIVE_INT, 2, n, 6, i6.data());
        hbt_h5_write(gg, "GroupOffsetType", H5T_NATIVE_LLONG, 2, n, 6, l6.data());
        hbt_h5_write(gg, "GroupMassType", H5T_NATIVE_FLOAT, 2, n, 6, f6.data());
        for(hsize_t g = 0; g < n; g++) {f1[g] = R[g].Mass;} hbt_h5_write(gg, "GroupMass", H5T_NATIVE_FLOAT, 1, n, 1, f1.data());
        for(hsize_t g = 0; g < n; g++) {f1[g] = R[g].Sfr;} hbt_h5_write(gg, "GroupSFR", H5T_NATIVE_FLOAT, 1, n, 1, f1.data());
        for(hsize_t g = 0; g < n; g++) {for(int k = 0; k < 3; k++) {f3[3 * g + k] = R[g].CM[k];}} hbt_h5_write(gg, "GroupCM", H5T_NATIVE_FLOAT, 2, n, 3, f3.data());
        for(hsize_t g = 0; g < n; g++) {for(int k = 0; k < 3; k++) {f3[3 * g + k] = R[g].Vel[k];}} hbt_h5_write(gg, "GroupVel", H5T_NATIVE_FLOAT, 2, n, 3, f3.data());
        hbt_h5_write(gg, "GroupFirstSub", H5T_NATIVE_INT, 1, n, 1, first.data());
        hbt_h5_write(gg, "GroupNsubs", H5T_NATIVE_INT, 1, n, 1, nsub.data());
        H5Gclose(gg);
        gids = H5Gcreate(file, "/IDs", 0);
    }
    /* per-type member keys, in global order */
    for(int t = 0; t < HBT_NTYPES; t++)
    {
        std::vector<unsigned long long> id, ch;
        for(long i = 0; i < nmem; i++) {if(mem[i].Type == t) {id.push_back((unsigned long long)mem[i].ID); ch.push_back((unsigned long long)mem[i].Child);}}
        std::vector<char> gid, gch;
        hbt_gather_bytes(id.data(), (long long)id.size() * 8, gid);
        hbt_gather_bytes(ch.data(), (long long)ch.size() * 8, gch);
        if(ThisTask == 0 && tot[t] > 0)
        {
            snprintf(buf, DEFAULT_PATH_BUFFERSIZE_TOUSE, "/IDs/PartType%d", t);
            hid_t gt = H5Gcreate(file, buf, 0);
            hbt_h5_write(gt, "ParticleIDs", H5T_NATIVE_ULLONG, 1, (hsize_t)tot[t], 1, gid.data());
            hbt_h5_write(gt, "ParticleChildIDsNumber", H5T_NATIVE_ULLONG, 1, (hsize_t)tot[t], 1, gch.data());
            H5Gclose(gt);
        }
    }
    if(ThisTask == 0) {H5Gclose(gids); H5Fclose(file); printf("KEYED CATALOGUE: %s: %lld groups, %lld member keys, %g s\n", fname, ngtot, nmtot, timediff(t0, my_second()));}
}
#endif
