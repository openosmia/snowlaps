#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@authors: Adrien Wehrlé (University of Zurich), Lou-Anne Chevrollier (Aarhus University)

"""

from snowlaps.snowlaps import SnowlapsEmulator
import numpy as np


class TestSnowlapsEmulator:
    my_emulator = SnowlapsEmulator()

    def test_run(self):
        emulator_results_single = self.my_emulator.run(
            [38, 500, 110000, 0.015, 800, 78000]
        )

        emulator_results_duo = self.my_emulator.run(
            [[38, 500, 110000, 0.015, 800, 78000], [38, 500, 110000, 0.015, 800, 78000]]
        )

        assert all(emulator_results_duo[0, :] == emulator_results_single)
        assert isinstance(emulator_results_single, np.ndarray)
        assert isinstance(emulator_results_duo, np.ndarray)