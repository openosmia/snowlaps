#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@authors: Adrien Wehrlé (University of Zurich), Lou-Anne Chevrollier (Aarhus University)

"""

from snowlaps.snowlaps import SnowlapsEmulator
import numpy as np

my_emulator = SnowlapsEmulator()

emulator_results = my_emulator.run([38, 500, 110000, 0.015, 800, 78000])

emulator_results2 = my_emulator.run(
    [[38, 500, 110000, 0.015, 800, 78000], [38, 500, 110000, 0.015, 800, 78000]]
)

assert all(emulator_results2[0, :] == emulator_results)
assert isinstance(emulator_results, np.ndarray)
assert isinstance(emulator_results2, np.ndarray)
