# Generic commands to build GIZMO for a given test

cp test/$TEST_NAME/Config.sh .
echo 'SYSTYPE="Ubuntu"' > Makefile.systype
make

mpirun ./GIZMO