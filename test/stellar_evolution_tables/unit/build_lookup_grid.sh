#!/bin/bash
# build_lookup_grid.sh TABLES_CC TABLES_H OUT [-Ddefines...]: compile lookup_grid.cc against one table source with the stub
# declarations (copied into a scratch tree so the source's "../declarations/allvars.h" resolves to the stub)
set -e
CC_SRC=$1; H_SRC=$2; OUT=$3; shift 3
HERE="$(cd "$(dirname "$0")" && pwd)"; T=$(mktemp -d)
mkdir -p $T/galaxy_sf $T/declarations $T/core
cp $CC_SRC $T/galaxy_sf/tables.cc; cp $H_SRC $T/galaxy_sf/$(basename $H_SRC)
cp $HERE/allvars_stub.h $T/declarations/allvars.h; : > $T/core/proto.h
mpicxx -std=c++17 -O2 -DH5_USE_16_API ${HDF5_HOME:+-I$HDF5_HOME/include} -I$T -I$HERE "$@" \
    -DTABLES_HEADER="\"galaxy_sf/$(basename $H_SRC)\"" $HERE/lookup_grid.cc $T/galaxy_sf/tables.cc -o $OUT ${HDF5_HOME:+-L$HDF5_HOME/lib} -lhdf5 -lm
rm -rf $T
