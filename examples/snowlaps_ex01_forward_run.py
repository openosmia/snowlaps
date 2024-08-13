#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@authors: Adrien Wehrlé (University of Zurich), Lou-Anne Chevrollier (Aarhus University)

"""

from snowlaps.snowlaps import SnowlapsEmulator

example_set_of_paramters = [38, 500, 110000, 0.015, 800, 78000]

my_emulator = SnowlapsEmulator()
emulator_results = my_emulator.run(parameters=example_set_of_paramters)
