# script to build GIZMO for the test; should be run from the gizmo source code root directory

TEST_NAME="gmc_cooling"
cp test/$TEST_NAME/Config.sh .
echo 'SYSTYPE="Ubuntu"' > Makefile.systype