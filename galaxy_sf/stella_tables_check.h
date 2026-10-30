/* Copyright (c) 2026 Ulrich Steinwandel
 * Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* Startup check of the installed Stella tables against the hashes this source version expects.
 * Hash definition as scripts/stella_tables_hash.py: HDF5 = sha256 over the datasets sorted by path (path, zero byte,
 * native-order data); other files = sha256 of the bytes. Tables come from scripts/get_stella_tables.sh; never fetched here. */
#ifndef STELLA_TABLES_CHECK_H
#define STELLA_TABLES_CHECK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <string>
#include <vector>
#include <algorithm>
#include <hdf5.h>

#define STELLA_TABLES_REF "a4830ab18252c8786539e95598fc789f3b6d5142" /* stella-tables commit; keep in step with scripts/get_stella_tables.sh */

static const char *const StellaTablesExpected[][2] = { /* basename, content hash */
    {"stellar_tables_unified.hdf5",  "14a23087b3d22c09eefa0d6097ae059f92598fb55cf2f86687419dcced5efada"},
    {"stellar_wind_cumulative.hdf5", "5ad360969afac76fc0bec10e0afddf2a210fec1b47787ce9374215c7d426ae50"},
    {"energy-table-x.txt",           "e1b03445ee95b020e611f73d3d84ca395378c607517e3367ceb18d95a712b96d"},
    {"eh2-table.txt",                "d423c682d6e9036edc9100e9e2ee990084ee191d500289dc485ce0a486f4450b"},
    {"H2-cooling-ratios.dat",        "13cf3c756a6b9e4e993ba9fb5fa93883e1dedd788d6415149497ef1fee893a0d"},
    {"wss_z_collis.txt",             "eaa290e0895be4846147e99880f18e5b981d9293cde632e4b3c4be71f10f8372"},
    {"sgchem_net18_tables.bin",      "e3c41f2c5a676557d06f44854b14a1ee64b0d98142e1459779b1b8c8631f928d"}, /* SGChem network 18 only */
};

struct StellaSha256 {
    uint32_t h[8]; uint64_t len; unsigned char buf[64]; size_t nbuf;
    StellaSha256() : len(0), nbuf(0) {
        static const uint32_t h0[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        memcpy(h, h0, sizeof(h));
    }
    static uint32_t ror(uint32_t x, int n) {return (x >> n) | (x << (32 - n));}
    void block(const unsigned char *p) {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64], a[8];
        for(int i = 0; i < 16; i++) {w[i] = ((uint32_t)p[4*i] << 24) | ((uint32_t)p[4*i+1] << 16) | ((uint32_t)p[4*i+2] << 8) | p[4*i+3];}
        for(int i = 16; i < 64; i++) {
            uint32_t s0 = ror(w[i-15], 7) ^ ror(w[i-15], 18) ^ (w[i-15] >> 3), s1 = ror(w[i-2], 17) ^ ror(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        memcpy(a, h, sizeof(a));
        for(int i = 0; i < 64; i++) {
            uint32_t t1 = a[7] + (ror(a[4], 6) ^ ror(a[4], 11) ^ ror(a[4], 25)) + ((a[4] & a[5]) ^ (~a[4] & a[6])) + k[i] + w[i];
            uint32_t t2 = (ror(a[0], 2) ^ ror(a[0], 13) ^ ror(a[0], 22)) + ((a[0] & a[1]) ^ (a[0] & a[2]) ^ (a[1] & a[2]));
            for(int j = 7; j > 0; j--) {a[j] = a[j-1];}
            a[4] += t1; a[0] = t1 + t2;
        }
        for(int i = 0; i < 8; i++) {h[i] += a[i];}
    }
    void update(const void *data, size_t n) {
        const unsigned char *p = (const unsigned char *)data; len += n;
        while(n > 0) {
            size_t m = std::min(n, (size_t)64 - nbuf); memcpy(buf + nbuf, p, m); nbuf += m; p += m; n -= m;
            if(nbuf == 64) {block(buf); nbuf = 0;}
        }
    }
    std::string hex() {
        uint64_t bits = len * 8; unsigned char pad = 0x80, z = 0, l[8];
        update(&pad, 1); while(nbuf != 56) {update(&z, 1);}
        for(int i = 0; i < 8; i++) {l[i] = (unsigned char)(bits >> (56 - 8*i));}
        update(l, 8);
        char s[65]; for(int i = 0; i < 8; i++) {snprintf(s + 8*i, 9, "%08x", h[i]);}
        return std::string(s);
    }
};

#if H5_VERSION_GE(1, 12, 0)
typedef H5O_info2_t stella_h5o_info_t;
#else
typedef H5O_info_t stella_h5o_info_t;
#endif
static herr_t stella_tables_collect_dataset(hid_t, const char *name, const stella_h5o_info_t *info, void *out)
{
    if(info->type == H5O_TYPE_DATASET) {((std::vector<std::string> *)out)->push_back(name);}
    return 0;
}

/* content hash of one table file; "" if it cannot be read */
static std::string stella_tables_hash(const char *path)
{
    StellaSha256 sh;
    size_t n = strlen(path);
    if(n > 5 && strcmp(path + n - 5, ".hdf5") == 0) {
        hid_t f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
        if(f < 0) {return "";}
        std::vector<std::string> names;
#if H5_VERSION_GE(1, 12, 0)
        H5Ovisit3(f, H5_INDEX_NAME, H5_ITER_INC, stella_tables_collect_dataset, &names, H5O_INFO_BASIC);
#else
        H5Ovisit1(f, H5_INDEX_NAME, H5_ITER_INC, stella_tables_collect_dataset, &names);
#endif
        std::sort(names.begin(), names.end());
        std::vector<char> data;
        for(const std::string &name : names) {
            hid_t d = H5Dopen2(f, name.c_str(), H5P_DEFAULT), t = H5Dget_type(d), nt = H5Tget_native_type(t, H5T_DIR_ASCEND), sp = H5Dget_space(d);
            data.resize((size_t)H5Sget_simple_extent_npoints(sp) * H5Tget_size(nt));
            herr_t rc = data.empty() ? 0 : H5Dread(d, nt, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
            H5Sclose(sp); H5Tclose(nt); H5Tclose(t); H5Dclose(d);
            if(rc < 0) {H5Fclose(f); return "";}
            sh.update(name.c_str(), name.size() + 1); sh.update(data.data(), data.size());
        }
        H5Fclose(f);
    } else {
        FILE *fd = fopen(path, "rb");
        if(!fd) {return "";}
        std::vector<char> b(1 << 20); size_t m;
        while((m = fread(b.data(), 1, b.size(), fd)) > 0) {sh.update(b.data(), m);}
        fclose(fd);
    }
    return sh.hex();
}

/* 0 if every file matches its expected hash; prints each mismatch. STELLA_TABLES_ALLOW_MISMATCH=1 turns mismatches into warnings. */
static int stella_tables_verify(const std::vector<std::string> &paths)
{
    int bad = 0;
    for(const std::string &p : paths) {
        const char *base = strrchr(p.c_str(), '/'); base = base ? base + 1 : p.c_str();
        const char *want = NULL;
        for(size_t k = 0; k < sizeof(StellaTablesExpected) / sizeof(StellaTablesExpected[0]); k++) {if(strcmp(base, StellaTablesExpected[k][0]) == 0) {want = StellaTablesExpected[k][1];}}
        if(!want) {continue;}
        std::string got = stella_tables_hash(p.c_str());
        if(got == want) {printf("STELLA TABLES: %s ok\n", p.c_str()); continue;}
        printf("STELLA TABLES: %s %s (expected %s, stella-tables %s); install with scripts/get_stella_tables.sh\n",
               p.c_str(), got.empty() ? "cannot be read" : ("has hash " + got).c_str(), want, STELLA_TABLES_REF);
        bad++;
    }
    const char *allow = getenv("STELLA_TABLES_ALLOW_MISMATCH");
    if(bad && allow && atoi(allow) == 1) {printf("STELLA TABLES: STELLA_TABLES_ALLOW_MISMATCH=1, continuing with %d mismatched table(s)\n", bad); bad = 0;}
    fflush(stdout);
    return bad;
}

#endif /* STELLA_TABLES_CHECK_H */
