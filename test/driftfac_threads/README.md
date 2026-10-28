# driftfac_threads

Links `driftfac_threads_driver.cc` against the build in place of `core/main.o` and calls the comoving drift and
gravity-kick factors (`get_drift_factor`, `get_gravkick_factor` in `core/driftfac.cc`) 2e7 times from up to 8 OpenMP
threads, cycling through four intervals with each interval used twice in a row, so calls both replace and hit the
cached value. Every result must equal the serial value for its interval. Takes about 10 s including the build.

## Known sensitivity

Fails when the last-interval cache is a plain function-scope static shared by all threads: a thread can match the times
another thread just stored and return the previous interval's factor (about 3e5 wrong results per 2e7 calls at -O3).
