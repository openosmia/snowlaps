#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@authors: Adrien Wehrlé (University of Zurich), Lou-Anne Chevrollier (Aarhus University)

"""

from snowlaps.snowlaps import SnowlapsEmulator

spectra_path = "./data/spectra/Chevrollier_et_al_2024_TC_spectra.csv"
metadata_path = "./data/spectra/Chevrollier_et_al_2024_TC_metadata.csv"

# creating a SnowlapsEmulator instance (loading the emulator and the scaler in the
# background)
my_emulator = SnowlapsEmulator()

# see documentation of SnowlapsEmulator.optimize for a description of the different
# outputs
(
    full_batch_optimization_results,
    best_optimization_results,
    best_emulator_spectra,
) = my_emulator.optimize(
    albedo_spectra_path=spectra_path,
    spectra_metadata_path=metadata_path,
    save_results=False,
)
