# frozen_test_stars

`TEST_FROZEN_STARS_ZERO_VELOCITY_AND_ACCELERATION` (test only): in the kick every star particle (type 4) gets zero
acceleration and velocity and nothing else is done to it, so star sources can be held fixed while the gravity tree (needed
by the tree radiation and columns) runs. Gas is left free.

Problem: two 1e3 Msun stars 1 pc apart, given 3 km/s in opposite directions, tree gravity, 0.5 time units
(pc / km/s). Without the flag they fall together (the tree gravity would move them by about 0.5 pc).

Check: both stars end within 1e-6 pc of their start with zero velocity.
