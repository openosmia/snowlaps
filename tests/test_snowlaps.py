#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@authors: Adrien Wehrlé (University of Zurich), Lou-Anne Chevrollier (Aarhus University)

"""

from snowlaps.snowlaps import SnowlapsEmulator
import numpy as np
import keras
import joblib
import sklearn
import pandas as pd

#%%
class TestSnowlapsEmulator:
    
    my_emulator = SnowlapsEmulator()
    
    emulator_path = "./data/emulator/mlp_snw_alg_3.h5"
    
    scaler_path = "./data/scaler/minmax_scaler.save"
    
    test_parameters_single = [38, 500, 110000, 0.015, 800, 78000]
    
    test_features_training = np.load("./tests/test_data/training_features_reduced.npy").tolist()
    
    test_targets_training =  np.load("./tests/test_data/training_targets_reduced.npy")
    
    test_spectra_Chevrollier2024_path = './tests/test_data/Chevrollier_et_al_2024_TC_spectra_reduced.csv' 
    
    test_metadata_Chevrollier2024_path = './tests/test_data/Chevrollier_et_al_2024_TC_metadata_reduced.csv' 

    test_optimization_results = pd.read_csv(
        "./tests/test_data/results_inversion_no_anisotropy_correction_Chevrollier2024_TC_reduced.csv", 
        index_col=0
        )

    def test_load_emulator(self) -> None:

        emulator, emulator_wavelengths = self.my_emulator.load_emulator(self.emulator_path)

        assert isinstance(emulator_wavelengths, np.ndarray)
        assert all(emulator_wavelengths == np.arange(295, 2415, 10))
        assert isinstance(emulator, keras.src.models.sequential.Sequential)
        
    def test_load_scaler(self) -> None:

        scaler = joblib.load(self.scaler_path)

        assert isinstance(scaler, sklearn.preprocessing._data.MinMaxScaler)
        
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
        
    
    def test_optimize(self) -> None:
        emulator_optimization_results = self.my_emulator.optimize(
            albedo_spectra_path = self.test_spectra_Chevrollier2024_path, 
            spectra_metadata_path = self.test_metadata_Chevrollier2024_path,
            save_results = False)
        
        surface_properties_retrieved = emulator_optimization_results[1]
        
        residuals_algae = (
            self.test_optimization_results['algae'].values 
            - surface_properties_retrieved['algae'].values
            )
        
        residuals_lwc = (
            self.test_optimization_results['lwc'].values 
            - surface_properties_retrieved['lwc'].values
            )
        
        residuals_bc = (
            self.test_optimization_results['bc'].values 
            - surface_properties_retrieved['bc'].values
            )
        
        residuals_dust = (
            self.test_optimization_results['dust'].values 
            - surface_properties_retrieved['dust'].values
            )
        
        residuals_reff = (
            self.test_optimization_results['grain_size'].values 
            - surface_properties_retrieved['grain_size'].values
            )
        
        assert all(np.abs(residuals_algae) < 2000)
        assert all(np.abs(residuals_bc) < 150)
        assert all(np.abs(residuals_dust) < 1e5)
        assert all(np.abs(residuals_lwc) < 0.01)
        assert all(np.abs(residuals_reff) < 50)

