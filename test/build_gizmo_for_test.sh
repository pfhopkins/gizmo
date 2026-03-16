# Generic commands to build GIZMO for a given test

rm GIZMO
cp test/$TEST_NAME/Config.sh . # retrieve the config file
make
