#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@authors: Adrien Wehrlé (University of Zurich), Lou-Anne Chevrollier (Aarhus University)

"""

import numpy as np
import pandas as pd
from typing import Union
import joblib
import tensorflow as tf
import time
import pysolar
import pytz
import tzwhere


class SnowlapsEmulator:
    def __init__(
        self,
        emulator_path: str = "./data/emulator/mlp_snw_alg_3.h5",
        scaler_path: str = "./data/scaler/minmax_scaler.save",
    ) -> None:
        """
        :param albedo_spectra_path: full path to file containing albedo spectra.
        :type albedo_spectra_path: str

        :param emulator_path: full path to emulator.
        :type emulator_path: str

        :param scaler_path: full path to scaler.
        :type scaler_path: str
        """

        self.emulator_path = emulator_path
        self.scaler_path = scaler_path

        self.emulator = self.load_emulator(self.emulator_path)
        self.scaler = self.load_scaler(self.scaler_path)

        return None

    def load_emulator(self, emulator_path):
        emulator = tf.keras.models.load_model(emulator_path)
        return emulator

    def load_scaler(self, scaler_path):
        scaler = joblib.load(scaler_path)
        return scaler

    def read_data(self, data_path: str) -> pd.DataFrame:
        """
        :param data_path: full path to file to read.
        :type data_path: str

        :return: DataFrame containing data to read.
        :rtype: pd.DataFrame
        """
        data = pd.read_csv(data_path)

        return data

    def run(self, parameters):
        # if only one list of parameters, parameters needs to be repeated to be 2D
        if not all(isinstance(elem, list) for elem in parameters):
            transformed_parameters = self.scaler.transform([parameters, parameters])
            emulator_results = np.vstack(self.emulator(transformed_parameters))[0, :]
        else:
            transformed_parameters = self.scaler.transform(parameters)
            emulator_results = np.vstack(self.emulator(transformed_parameters))

        return emulator_results

    def compute_SZA(self, longitude, latitude, date, time):
        tz = tzwhere.tzwhere()
        time_zone = tz.tzNameAt(longitude, latitude)
        time_zone_tzobj = pytz.timezone(time_zone)
        date_pyobj = pd.Timestamp(date + " " + time, tz=time_zone_tzobj).to_pydatetime()

        sza = 90 - pysolar.solar.get_altitude(latitude, longitude, date_pyobj)

        return sza

    def optimize(
        self,
        albedo_spectra_path,
        nb_optimization_steps=1000,
        nb_optimization_repeats=20,
        optimizer=tf.keras.optimizers.Adagrad(learning_rate=1.0),
        optimization_init=None,
        gradient_mask=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        sza_list=None,
        save_results=True,
    ) -> None:
        def get_minimum_MAE_index(sub_df):
            index_minimum = np.where(
                sub_df["mean_MAE"] == np.nanmin(sub_df["mean_MAE"])
            )[0][0]

            global_index = int(sub_df.iloc[index_minimum]["single_indices"])

            return global_index

        albedo_spectra = self.read_data(albedo_spectra_path)

        nb_spectra = len(albedo_spectra)

        constant_gradients = tf.constant(
            np.tile(
                gradient_mask,
                (nb_spectra * nb_optimization_repeats, 1),
            ),
            dtype="float64",
        )

        # if no optimization initializations provided, set random ones
        if optimization_init is None:
            variable_inits = np.random.uniform(
                size=(nb_spectra * nb_optimization_repeats, 6)
            )
            optimization_results = tf.Variable(variable_inits)

        # ------------ optimization start
        start_time = time.time()
        start_local_time = time.ctime(start_time)

        for i in range(nb_optimization_steps):
            with tf.GradientTape(watch_accessed_variables=False) as t:
                # watch the input (remember them)
                t.watch(optimization_results)

                # compute residuals on all spextra
                residuals = (
                    self.emulator(optimization_results)[:, 6 : 6 + 100]
                    - albedo_spectra[:, :100]
                )

                # cost is computed as integrated squared error (coded by hand to vectorize)
                COST = tf.math.reduce_sum(residuals**2, axis=1) / 2

                # applying constant gradient to computed gradients
                grads = t.gradient(COST, optimization_results) * constant_gradients

                # apply gradients on all spectra
                optimizer.apply_gradients(zip([grads], [optimization_results]))

                # re-assign negative values to their absolute value
                optimization_results.assign(tf.abs(optimization_results))

        # ------------ optimization end
        end_time = time.time()
        end_local_time = time.ctime(end_time)
        processing_time = (end_time - start_time) / 60
        print("--- Processing time: %s minutes ---" % processing_time)
        print("--- Start time: %s ---" % start_local_time)
        print("--- End time: %s ---" % end_local_time)

        results = optimization_results.numpy()

        results_transformed = self.scaler.inverse_transform(results)

        # create an id for each batch of random initializations
        repeat_ids = np.tile(np.arange(nb_spectra), (nb_optimization_repeats))

        # concatenate result variables and mean absolute error
        fullbatch_optimization_results = pd.DataFrame(
            {
                "mean_MAE": np.nanmean(np.abs(residuals), axis=1),
                "sza": results_transformed[:, 0],
                "grain_size": results_transformed[:, 1],
                "algae": results_transformed[:, 2],
                "lwc": results_transformed[:, 3],
                "bc": results_transformed[:, 4],
                "dust": results_transformed[:, 5],
            },
            index=np.tile(albedo_spectra.columns, 20),
        )

        # add batch indices
        fullbatch_optimization_results["indices"] = repeat_ids

        # add single indices
        fullbatch_optimization_results["single_indices"] = np.arange(
            len(fullbatch_optimization_results)
        )

        # extract index with minimum MAE for each batch of random initializations
        indexes_minimum_batch_MAE = fullbatch_optimization_results.groupby(
            "indices"
        ).apply(get_minimum_MAE_index)

        # extract results on results with minimum MAE
        best_optimization_results = fullbatch_optimization_results.iloc[
            indexes_minimum_batch_MAE
        ]

        # re-evaluate modeled spectra to compare with observations
        best_emulator_spectra = self.emulator(best_optimization_results)

        if save_results:
            time_tag = time.strftime("%Y%m%d_%H%M%S")
            fullbatch_optimization_results.to_csv(
                f"../../data/results/snowlaps_fullbatch_optimizations_{time_tag}.csv"
            )
            best_optimization_results.to_csv(
                f"../../data/results/snowlaps_best_optimizations_{time_tag}.csv"
            )
            best_emulator_spectra.to_csv(
                f"../../data/results/snowlaps_best_emulator_spectra_{time_tag}.csv"
            )

        return (
            fullbatch_optimization_results,
            best_optimization_results,
            best_emulator_spectra,
        )
