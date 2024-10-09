import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from snowlaps.snowlaps import SnowlapsEmulator
from importlib import reload
import matplotlib.pyplot as plt
import numpy as np
import copy
st.set_page_config(
    page_title="Snowlaps Inversion",
    page_icon="❄️",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/cryobiogeo-projects/snowlaps-emulator",
        "Report a bug": "https://github.com/cryobiogeo-projects/snowlaps-emulator/issues",
        "About": f"# biosnicar frontent version X",
    },
)

st.markdown("# Please drop your files below")


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




def run_snowlaps(
    SZA,
    optical_radius,
    algae_concentration,
    liquid_water_content,
    black_carbon_concentration,
    mineral_dust_concentration,
) -> pd.DataFrame:
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
        Dataframe: Dataframe with result for display.
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

    return pd.DataFrame(emulator_results, index=my_emulator.emulator_wavelengths)


def plot_inversion(emulator, measure):
    emulator.rename(columns={emulator.columns[0]: 'emulator_pred'}, inplace=True)
    df_m = pd.DataFrame({"measures": measure})
    df = pd.concat([df_m, emulator], axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["measures"],
                    mode='lines',
                    name='measures',
                    line = dict(color='royalblue', width=4)))
    fig.add_trace(go.Scatter(x=df.index, y=df["emulator_pred"],
                    mode='lines+markers',
                    name='emulator',
                    line = dict(color='gray', width=5)))
    fig.update_traces(marker=dict(size=5))
    fig.update_xaxes(range=[350, 2500])
    return fig



@st.cache_data
def run_model():
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

        st.plotly_chart(plot_inversion(best_emulator_spectra, albedo_spectra[spectra]))
        st.success("Done!")

        with st.expander("Show snowlaps inversion results"):
            st.dataframe(best_optimization_results)
    return best_optimization_results


placeholder_title_1 = st.sidebar.empty()
placeholder_num_1 = st.sidebar.empty()
placeholder_num_2 = st.sidebar.empty()
placeholder_title_2 = st.sidebar.empty()
placeholder_num_3 = st.sidebar.empty()
placeholder_num_4 = st.sidebar.empty()
placeholder_num_5 = st.sidebar.empty()
placeholder_title_3 = st.sidebar.empty()
placeholder_num_6 = st.sidebar.empty()
placeholder_button = st.sidebar.empty()


if st.button("Run inversion"):
    best_optimization_results = run_model()
    st.session_state.best_optimization_results = best_optimization_results
spectra = st.selectbox("Choose Spectra", albedo_spectra.columns)


with st.sidebar:
    placeholder_title_1.header("Solar geometry")


    SZA = placeholder_num_1.number_input(
        "Solar Zenith Angle (SZA; degrees)", 0.0, 90.0,
        value=st.session_state.best_optimization_results.iloc[0]["sza"]
    )
    optical_radius = placeholder_num_2.number_input(
        "Snow optical radius (um)", 0.0, 1000.0,
        value=st.session_state.best_optimization_results.iloc[0]["grain_size"]
    )

    placeholder_title_2.header("Light Absorbing Particles (LAPs)")


    algae_concentration = placeholder_num_3.number_input(
        "Algae concentration (cells/mL)", 0.0, 1000000.0,
        value=st.session_state.best_optimization_results.iloc[0]["algae"]
    )
    black_carbon_concentration = placeholder_num_4.number_input(
        "Black carbon concentration (ppb)", 0.0, 10000.0,
        value=st.session_state.best_optimization_results.iloc[0]["bc"]
    )
    mineral_dust_concentration = placeholder_num_5.number_input(
        "Mineral dust concentration (um)", 0.0, 780000.0,
        value=st.session_state.best_optimization_results.iloc[0]["dust"]
    )

    placeholder_title_3.header("Water")


    liquid_water_content = placeholder_num_6.number_input(
        "Liquid water content (%)", 0.0, 0.1,
        value=st.session_state.best_optimization_results.iloc[0]["lwc"]
    )


emulator_results = run_snowlaps(SZA,
                                optical_radius,
                                algae_concentration,
                                black_carbon_concentration,
                                mineral_dust_concentration,
                                liquid_water_content)

st.plotly_chart(plot_inversion(emulator_results, albedo_spectra[spectra]))
