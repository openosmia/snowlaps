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
from timezonefinder import TimezoneFinder
from datetime import datetime
import sklearn.preprocessing


class SnowlapsEmulator:
    def __init__(
        self,
        emulator_path: str = "./data/emulator/mlp_snw_alg_3.h5",
        scaler_path: str = "./data/scaler/minmax_scaler.save",
    ) -> None:
        """
        :param emulator_path: full path to emulator.
        :type emulator_path: str

        :param scaler_path: full path to scaler.
        :type scaler_path: str
        """

        self.emulator_path = emulator_path
        self.scaler_path = scaler_path

        self.emulator, self.emulator_wavelengths = self.load_emulator(
            self.emulator_path
        )
        self.scaler = self.load_scaler(self.scaler_path)

        return None

    def load_emulator(self, emulator_path: str) -> tuple:
        """
        :param emulator_path: full path to file containing the emulator.
        :type emulator_path: str

        :return: a 2-element tuple. emulator is the emulator object,
                 emulator_wavelengths is the list of emulator wavelengths.
        :rtype: tuple (tf.keras.models, np.ndarray)
        """

        emulator = tf.keras.models.load_model(emulator_path)

        if emulator_path == "./data/emulator/mlp_snw_alg_3.h5":
            emulator_wavelengths = np.arange(295, 2415, 10)
        return (emulator, emulator_wavelengths)

    def load_scaler(self, scaler_path: str) -> sklearn.preprocessing:
        """
        :param scaler_path: full path to file containing the scaler.
        :type scaler_path: str

        :return: the scaler object.
        :rtype: sklearn.preprocessing
        """

        scaler = joblib.load(scaler_path)
        return scaler

    def read_data(self, data_path: str) -> pd.DataFrame:
        """
        :param data_path: full path to file to read.
        :type data_path: str

        :return: DataFrame containing data to read.
        :rtype: pd.DataFrame
        """
        data = pd.read_csv(data_path, index_col=0)

        return data

    def run(self, parameters: list) -> np.ndarray:
        """
        :param parameters: list of emulator parameters where each list has parameters
                           in the following order: [Solar Zenith Angle (SZA), optical
                           radius of the snow grains, red algal concentration, liquid
                           water content, black carbon concentration, dust
                           concentration].
        :type data_path: list

        :return: a 2D matrix containing emulator results with rows corresponding
                 to the number of runs passed as inputs, and columns to the emulator
                 wavelengths.
        :rtype: np.ndarray
        """
        # if only one list of parameters, parameters needs to be repeated to be 2D
        if not all(isinstance(elem, list) for elem in parameters):
            transformed_parameters = self.scaler.transform([parameters, parameters])
            emulator_results = np.vstack(self.emulator(transformed_parameters))[0, :]
        else:
            transformed_parameters = self.scaler.transform(parameters)
            emulator_results = np.vstack(self.emulator(transformed_parameters))

        return emulator_results

    def compute_SZA(
        self, longitude: float, latitude: float, date: str, time: str
    ) -> float:
        """
        Compute the Solar Zenith Angle (SZA) at a given location and time.
        SZA is one of the emulator inputs.

        :param longitude: Longitude at which the spectrum was acquired.
        :type longitude: float

        :param latitude: Latitude at which the spectrum was acquired.
        :type latitude: float

        :param date: Date at which the spectrum was acquired (YYYY-MM-DD)
        :type date: str

        :param time: Time at which the spectrum was acquired (HH:MM or HH:MM:SS)
        :type time: str

        :return: the Solar Zenith Angle (SZA).
        :rtype: float
        """

        time_zone = self.tzf.timezone_at(lng=longitude, lat=latitude)
        time_zone_tzobj = pytz.timezone(time_zone)
        date_pyobj = datetime.strptime(date + " " + time, "%Y-%m-%d %H:%M").replace(
            tzinfo=time_zone_tzobj
        )
        solar_altitude = pysolar.solar.get_altitude(latitude, longitude, date_pyobj)

        sza = np.round(90 - solar_altitude, 1)

        return sza

    def optimize(
        self,
        albedo_spectra_path: str,
        nb_optimization_steps: int = 1000,
        nb_optimization_repeats: int = 20,
        optimizer: tf.keras.optimizers = tf.keras.optimizers.Adagrad(learning_rate=1.0),
        optimization_init: Union[list, None] = None,
        gradient_mask: list = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        sza_list: Union[list, None] = None,
        spectra_metadata_path: Union[str, None] = None,
        save_results: bool = True,
    ) -> tuple:
        """
        Find the emulator variables and associated emulator albedo spectra that best
        reproduce albedo spectra passed as inputs.

        :param albedo_spectra_path: Full path to file containing albedo spectra with
                                    wavelengths as indexes, and measurement tag/name as
                                    columns.
                                    See i.e. ./data/spectra/Chevrollier_et_al_2024_TC_spectra.csv.
        :type albedo_spectra_path: str

        :param nb_optimization_steps: Number of times the fit between emulator and albedo
                                      spectra passed as inputs is being updated. The larger
                                      this number the better each individual fit, but also the
                                      slower the full optimization. Default to 1000.
        :type nb_optimization_steps: int, optional

        :param nb_optimization_repeats: Number of times the optimization will be repeated for
                                        each albedo spectrum.The larger this number the better
                                        the best fit for a given spectrum, but also the slower
                                        the optimization. Default to 20.
        :type nb_optimization_repeats: int, optional

        :param optimizer: Keras optimizer to use for the gradient descent algorithm. Default to
                          Keras Adagrad optimizer with a learning rate of 1.0.
        :type optimizer: keras.src.optimizers, optional

        :param optimization_init: List of initial values for each input variable of the emulator.
                                  Default to None and random initialisations.
        :type optimization_init: list, optional

        :param gradient_mask: Boolean mask to enable (1) or freeze (0) the optimization of each
                              input variable of the emulator. Default to SZA frozen and all other
                              variables enabled, as SZA is either given or computed.
        :type gradient_mask: list, optional

        :param sza_list: List of Solar Zenith Angle (SZA) values for each albedo spectrum passed as
                         input. Default to None, and hence SZA computed based on albedo spectra
                         metadata (see spectra_metadata_path).
        :type sza_list: list, optional

        :param spectra_metadata_path: Full path to file containing the metadata for each albedo
                                      spectrum passed as input, with columns ["tag", "longitude",
                                      "latitude", "date", "time"].
                                      See i.e. ./data/spectra/Chevrollier_et_al_2024_TC_metadata.csv.
        :type spectra_metadata_path: str, optional

        :param save_results: To save optimization results in ./data/optimization_results or not.
                             Default to True.
        :type spectra_metadata_path: bool, optional


        :return: three-element tuple. fullbatch_optimization_results contains the mean MAE and
                                      associated optimized emulator variables of each optimization
                                      repeat of each albedo spectrum passed as input, hence the
                                      number of lines corresponds to the number of input albedo
                                      spectra times the number of optimization_repeats.
                                      best_optimization_results contains the lowest mean MAE and
                                      associated best optimized emulator variables from the batch
                                      of optimization repeats of each albedo spectrum passed as
                                      input, hence number of lines equals the number of input albedo
                                      spectra.
                                      best_emulator_spectra contains the emulator albedo spectra
                                      resulting from the best optimization results, with the same
                                      shape and formatting as the file albedo_spectra_path.
        :rtype: tuple (pandas.core.frame.DataFrame, pandas.core.frame.DataFrame, numpy.ndarray)
        """

        def get_minimum_MAE_index(sub_df: pd.Series) -> int:
            """Get the global index of the minimum MAE of each batch

            :return: the global index.
            :rtype: int

            """
            index_minimum = np.where(
                sub_df["mean_MAE"] == np.nanmin(sub_df["mean_MAE"])
            )[0][0]

            global_index = int(sub_df.iloc[index_minimum]["single_indices"])

            return global_index

        albedo_spectra = self.read_data(albedo_spectra_path)

        if sza_list is None and spectra_metadata_path is not None:
            self.tzf = TimezoneFinder()
            self.spectra_metadata = self.read_data(spectra_metadata_path)
            sza_list = np.array(
                [
                    self.compute_SZA(
                        spectrum_metadata.longitude,
                        spectrum_metadata.latitude,
                        spectrum_metadata.date,
                        spectrum_metadata.time,
                    )
                    for idx, spectrum_metadata in self.spectra_metadata.iterrows()
                ]
            )
            sza_list[self.spectra_metadata.diffuse] = 50

        else:
            raise ValueError(
                "Either a SZA list or acquisition dates, times and locations should be given"
            )

        sza_spectra_transformed = [
            self.scaler.transform(np.tile(np.repeat(sza, 6), (2, 1)))[0, 0]
            for sza in sza_list
        ]

        nb_spectra = albedo_spectra.shape[1]

        constant_gradients = tf.constant(
            np.tile(
                gradient_mask,
                (nb_spectra * nb_optimization_repeats, 1),
            ),
            dtype="float64",
        )

        # if no optimization initializations provided, set random ones
        if optimization_init is None:
            variables_init_without_sza = np.random.uniform(
                size=(nb_spectra * nb_optimization_repeats, 5)
            )

        # sza initalizations (constant known SZA)
        sza_inits = tf.Variable(
            np.tile(
                sza_spectra_transformed[:nb_spectra], (1, nb_optimization_repeats)
            ).T,
            dtype=tf.float32,
        )

        # spectrum to fit
        albedo_spectra_arr = albedo_spectra.to_numpy()
        albedo_spectra_batched = tf.Variable(
            np.tile(albedo_spectra_arr[5::10, :], (1, nb_optimization_repeats)).T,
            dtype=tf.float32,
        )

        variables_init = np.concatenate((sza_inits, variables_init_without_sza), axis=1)
        optimization_results = tf.Variable(variables_init)

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
                    - albedo_spectra_batched[:, :100]
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

        best_optimization_parameters = self.scaler.transform(
            best_optimization_results.iloc[:, 1:-2].values
        )

        # re-evaluate modeled spectra to compare with observations
        best_emulator_spectra_tf = self.emulator(best_optimization_parameters)
        best_emulator_spectra = pd.DataFrame(
            best_emulator_spectra_tf.numpy().T,
            columns=albedo_spectra.columns,
            index=self.emulator_wavelengths,
        )

        if save_results:
            time_tag = time.strftime("%Y%m%d_%H%M%S")
            fullbatch_optimization_results.to_csv(
                f"./data/optimization_results/snowlaps_fullbatch_optimizations_{time_tag}.csv"
            )
            best_optimization_results.to_csv(
                f"./data/optimization_results/snowlaps_best_optimizations_{time_tag}.csv"
            )
            best_emulator_spectra.to_csv(
                f"./data/optimization_results/snowlaps_best_emulator_spectra_{time_tag}.csv"
            )

        return (
            fullbatch_optimization_results,
            best_optimization_results,
            best_emulator_spectra,
        )
