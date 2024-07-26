#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@authors: Adrien Wehrlé (University of Zurich), Lou-Anne Chevrollier (Aarhus University)

"""

from snowlaps import SnowLaps

spectra_path = "../../data/spectra/Chevrollier_et_al_2024a_TC.csv"

my_emulator = SnowLaps(albedo_spectra_path=spectra_path)
optimization_results = my_emulator.optimize()
