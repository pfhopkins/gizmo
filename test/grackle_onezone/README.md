# grackle_onezone (optional: needs Grackle >= 3)

Checks the Grackle 3 interface (`COOL_GRACKLE` with `COOL_GRACKLE_APIVERSION=3`) against Grackle itself.

**This test is skipped unless Grackle is available**, so it does not affect machines without it. To enable it:

1. Build Grackle (3.x; tested with 3.4.1) with its CMake build and install it:
   ```bash
   git clone --recursive https://github.com/grackle-project/grackle   # --recursive also fetches the data tables
   cd grackle
   cmake -B build -DCMAKE_INSTALL_PREFIX=<prefix> -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_BUILD_TYPE=Release
   cmake --build build && cmake --install build
   ```
   (`-DCMAKE_INSTALL_LIBDIR=lib`: Grackle 3.4.1's `grackle.pc` assumes the library is in `<prefix>/lib`, while CMake may install
   it in `lib64`.)
2. Point GIZMO and the test at it:
   ```bash
   export GRACKLE_PATH=<prefix>                       # or add <prefix>/lib/pkgconfig to PKG_CONFIG_PATH
   export GRACKLE_DATA_DIR=<grackle repository>/input # contains CloudyData_UVB=HM2012_shielded.h5
   pytest test/grackle_onezone
   ```

What it does: a uniform, static, periodic gas box (512 cells) is N identical one-zone problems. For three Grackle
configurations (`COOL_GRACKLE_CHEMISTRY=0` with `METALS`, `=2`, and `=3` with `METALS`) GIZMO is built and run on a hot box
(nH = 10 cm^-3, 10^6 K) and a dense box (nH = 10^4 cm^-3, 1000 K), and every snapshot is compared with a standalone one-zone
Grackle integration (`onezone_reference.cc`, compiled against the same `grackle.pc`) that uses the same Grackle settings as
GIZMO's `InitGrackle`. Pass: the box stays uniform, `InternalEnergy` and `Temperature` agree within 2%, and the electron
abundance and each species above 10^-3 by mass within 5%, at every snapshot. A failure to compile the reference driver points
at the Grackle installation; a failure of the comparison at the GIZMO interface.

The runs use `MaxSizeTimestep 1e-6`: GIZMO calls Grackle once per hydro step, and with much longer steps the answer during fast
cooling transitions depends on the step (e.g. ~5% in u for the H2 network at 2e-5).
