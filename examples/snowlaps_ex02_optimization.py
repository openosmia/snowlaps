#!/usr/bin/env python3
"""

@authors: Adrien Wehrlé (University of Zurich), Lou-Anne Chevrollier (Aarhus University)

"""

import pandas as pd

from snowlaps.snowlaps import SnowlapsEmulator

spectra_path = "./data/spectra/Chevrollier_et_al_2024_TC_spectra.csv"
metadata_path = "./data/spectra/Chevrollier_et_al_2024_TC_metadata.csv"

# creating a SnowlapsEmulator instance (loading the emulator and the scaler in the
# background)
my_emulator = SnowlapsEmulator()

# Can be used wherever a "file-like" object is accepted:
albedo_spectra = pd.read_csv(spectra_path, index_col=0)
albedo_metadata = pd.read_csv(metadata_path, index_col=0)

# see documentation of SnowlapsEmulator.optimize for a description of the different
# outputs
(
    full_batch_optimization_results,
    best_optimization_results,
    best_emulator_spectra,
) = my_emulator.optimize(
    albedo_spectra_path=albedo_spectra,
    spectra_metadata_path=albedo_metadata,
    save_results=False,
)
