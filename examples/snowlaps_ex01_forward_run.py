#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@authors: Adrien Wehrlé (University of Zurich), Lou-Anne Chevrollier (Aarhus University)

"""

from snowlaps import SnowLaps

input_path = "../../data/inputs/ex01_snowlaps_inputs.csv"

my_emulator = SnowLaps()
emulator_results = my_emulator.run(parameters=input_path)
