#!/usr/bin/env python3

from snowlaps.snowlaps import SnowlapsEmulator

# See documentation for the order of input parameters.
# (0) solar zenith angle (unitless)
# (1) optical radius of snow grain (um)
# (2) algal concentration (cells/mL)
# (3) liquid water content (fraction, unitless)
# (4) black carbon concentration (ppb)
# (5) mineral dust concentration (ppb)

example_set_of_paramters = [38, 500, 110000, 0.015, 800, 78000]

my_emulator = SnowlapsEmulator()

# the emulator will output hemispherical albedo between 0.295 and 2.405um.

emulator_results = my_emulator.run(parameters=example_set_of_paramters)
