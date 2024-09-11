#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, Lou-Anne Chevrollier

"""

import numpy as np
import pandas as pd
from snowlaps.snowlaps import SnowlapsEmulator
import plotly.express as px
import streamlit as st
from io import StringIO

st.set_page_config(
    page_title="snowlaps",
    page_icon="❄️",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/cryobiogeo-projects/snowlaps-emulator",
        "Report a bug": "https://github.com/cryobiogeo-projects/snowlaps-emulator/issues",
        "About": f"# biosnicar frontent version X",
    },
)

tab1, tab2 = st.tabs(["Snowlaps forward", "Snowlaps inversion"])

with tab1:
    st.title(f"snowlaps emulator")
    st.markdown(
        f"""
    snow ❄️ ice 🧊 and life :space_invader: albedo model (vX)

    [GitHub](https://github.com/cryobiogeo-projects/snowlaps-emulator)
    [Documentation](https://github.com/cryobiogeo-projects/snowlaps-emulator)

    *Note that impurities are assumed to exist in the upper 2 cm of the snow or ice only, and that the grain shape in the granular layer set up is spherical.
    To access other configurations, download and run the full model as Python code instead.*
    """
    )

    st.markdown("""---""")

    st.sidebar.header("Solar geometry")
    SZA = st.sidebar.number_input("Solar Zenith Angle (SZA; degrees)", 0.0, 90.0, 42.0)
    optical_radius = st.sidebar.number_input(
        "Snow optical radius (um)", 0.0, 1000.0, 500.0
    )

    st.sidebar.header("Light Absorbing Particles (LAPs)")
    algae_concentration = st.sidebar.number_input(
        "Algae concentration (cells/mL)", 0.0, 1000000.0, 110000.0
    )
    black_carbon_concentration = st.sidebar.number_input(
        "Black carbon concentration (ppb)", 0.0, 10000.0, 800.0
    )
    mineral_dust_concentration = st.sidebar.number_input(
        "Mineral dust concentration (um)", 0.0, 780000.0, 78000.0
    )

    st.sidebar.header("Water")
    liquid_water_content = st.sidebar.number_input(
        "Liquid water content (%)", 0.0, 0.1, 0.015
    )

    def run_snowlaps(
        SZA,
        optical_radius,
        algae_concentration,
        liquid_water_content,
        black_carbon_concentration,
        mineral_dust_concentration,
    ) -> dict:
        """Runs biosnicar

        Args:
            layer (str): _description_
            thickness (float): _description_
            radius (int): _description_
            density (int): _description_
            black_carbon (int): _description_
            glacier_algae (int): _description_
            snow_algae (int): _description_
            solar_zenith_angle (int): _description_

        Returns:
            dict: Dict with result for display.
        """

        my_emulator = SnowlapsEmulator()

        emulator_results = my_emulator.run(
            parameters=[
                SZA,
                optical_radius,
                algae_concentration,
                liquid_water_content,
                black_carbon_concentration,
                mineral_dust_concentration,
            ]
        )

        return {
            "albedo": pd.DataFrame(
                emulator_results, index=my_emulator.emulator_wavelengths
            ),
        }

    def plot_albedo(albedo: pd.Series):
        fig = px.line(
            result["albedo"],
            range_y=[0, 1],
            # range_x=[0.205, 2.5],
            labels={"index": "wavelengths (microns)", "value": "Albedo"},
        )
        fig.update_layout(showlegend=False)
        return fig

    result = run_snowlaps(
        SZA,
        optical_radius,
        algae_concentration,
        liquid_water_content,
        black_carbon_concentration,
        mineral_dust_concentration,
    )

    # display results
    # st.metric("Broadband Albedo", result["broadband"])
    st.plotly_chart(plot_albedo(result["albedo"]))
    # st.download_button("download data", data=result["albedo_csv"], file_name="albedo.csv")

    with st.expander("Show raw data"):
        st.dataframe(result["albedo"])


with tab2:
    my_emulator = SnowlapsEmulator()

    uploaded_data = st.file_uploader(
        "Choose a CSV file containing albedo spectra to be inverted for snow properties"
    )

    if uploaded_data is not None:
        # Can be used wherever a "file-like" object is accepted:
        albedo_spectra = pd.read_csv(uploaded_data, index_col=0)
        st.write(albedo_spectra)

    uploaded_metadata = st.file_uploader(
        "Choose a CSV file containing the metadata of the albedo spectra"
    )

    if uploaded_metadata is not None:
        # Can be used wherever a "file-like" object is accepted:
        albedo_metadata = pd.read_csv(uploaded_metadata, index_col=0)
        st.write(albedo_metadata)

    def plot_albedo(spectra):
        fig = px.line(
            spectra,
            range_y=[0, 1],
            labels={"index": "wavelengths (microns)", "value": "Albedo"},
        )
        fig.update_layout(showlegend=False)
        return fig

    if uploaded_data is not None and uploaded_metadata is not None:
        with st.spinner("Please wait..."):
            (
                full_batch_optimization_results,
                best_optimization_results,
                best_emulator_spectra,
            ) = my_emulator.optimize(
                albedo_spectra_path=albedo_spectra,
                spectra_metadata_path=albedo_metadata,
                save_results=False,
            )

        st.plotly_chart(plot_albedo(best_emulator_spectra))

        st.success("Done!")

        with st.expander("Show snowlaps inversion results"):
            st.dataframe(best_optimization_results)
