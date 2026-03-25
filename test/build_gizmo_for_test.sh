# Generic commands to build GIZMO for a given test

rm GIZMO
rm test/*/GIZMO
cp test/$TEST_NAME/Config.sh . # retrieve the config file
make clean && make -j8
