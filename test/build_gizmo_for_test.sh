# Generic commands to build GIZMO for a given test

cp test/$TEST_NAME/Config.sh . # retrieve the config file
echo 'SYSTYPE="PopOS"' > Makefile.systype # update makefile.systype
make
