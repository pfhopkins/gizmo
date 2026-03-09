# Generic commands to build GIZMO for a given test

cp test/$TEST_NAME/Config.sh . # retrieve the config file
echo 'SYSTYPE="MacBookCellar"' > Makefile.systype # update makefile.systype
make
