# Generic commands to build GIZMO for a given test

rm GIZMO
rm test/*/GIZMO
cp test/$TEST_NAME/Config.sh . # retrieve the config file
make clean && make -j8
# Stella/SGCHEM tests read the installed tables through the in-tree default paths (no network if already installed)
if grep -qE '^(SGCHEM|GALSF_RESOLVEDISM_STELLAR_TABLES)\b' Config.sh; then ./scripts/get_stella_tables.sh --into-tree . ; fi
