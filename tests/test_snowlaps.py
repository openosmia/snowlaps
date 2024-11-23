#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@authors: Adrien Wehrlé (University of Zurich), Lou-Anne Chevrollier (Aarhus University)

"""

from snowlaps.snowlaps import SnowlapsEmulator
import numpy as np
import keras


#%%
class TestSnowlapsEmulator:
    
    my_emulator = SnowlapsEmulator()
    
    emulator_path = "./data/emulator/mlp_snw_alg_3.h5"

    test_parameters_single = [38, 500, 110000, 0.015, 800, 78000]
    
    test_features_training = np.load("./tests/test_data/training_features_reduced.npy").tolist()
    
    test_targets_training =  np.load("./tests/test_data/training_targets_reduced.npy")


    def test_load_emulator(self) -> None:

        emulator, emulator_wavelengths = self.my_emulator.load_emulator(self.emulator_path)

        assert isinstance(emulator_wavelengths, np.ndarray)
        assert all(emulator_wavelengths == np.arange(295, 2415, 10))
        assert isinstance(emulator, keras.src.models.sequential.Sequential)

    def test_run(self) -> None:
        emulator_results_single = self.my_emulator.run(self.test_parameters_single)

        emulator_results_duo = self.my_emulator.run(
            [self.test_parameters_single, self.test_parameters_single]
        )
        
        emulator_results_training_data = self.my_emulator.run(
            self.test_features_training
        ).astype(np.float16)
        
        residuals = np.abs(emulator_results_training_data - self.test_targets_training)

        assert all(residuals.flatten() < 5e-3)
        assert all(np.nanmean(residuals, axis=1) < 5.8e-4)
        assert all(emulator_results_duo[0, :] == emulator_results_single)
        assert isinstance(emulator_results_single, np.ndarray)
        assert isinstance(emulator_results_duo, np.ndarray)
