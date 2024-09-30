import streamlit as st
import pandas as pd
import plotly.express as px
from snowlaps.snowlaps import SnowlapsEmulator

st.set_page_config(
    page_title="Snowlaps forward",
    page_icon="❄️",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/cryobiogeo-projects/snowlaps-emulator",
        "Report a bug": "https://github.com/cryobiogeo-projects/snowlaps-emulator/issues",
        "About": f"# biosnicar frontent version X",
    },
)


st.markdown("# Snowlaps forward")
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


default_values = {"SZA": 42.0, "SOR": 500.0, "AC": 110000.0, "BCC": 800.0, "MDC": 78000.0, "LWC": 0.015}


if placeholder_button.button('Reset'):
    st.session_state.SZA = default_values["SZA"]
    st.session_state.SOR = default_values["SOR"]
    st.session_state.AC = default_values["AC"]
    st.session_state.BCC = default_values["BCC"]
    st.session_state.MDC = default_values["MDC"]
    st.session_state.LWC = default_values["LWC"]


with st.sidebar:
    placeholder_title_1.header("Solar geometry")


    SZA = placeholder_num_1.number_input(
        "Solar Zenith Angle (SZA; degrees)", 0.0, 90.0, value=default_values["SZA"], key="SZA"
    )
    optical_radius = placeholder_num_2.number_input(
        "Snow optical radius (um)", 0.0, 1000.0, value=default_values["SOR"], key="SOR"
    )

    placeholder_title_2.header("Light Absorbing Particles (LAPs)")


    algae_concentration = placeholder_num_3.number_input(
        "Algae concentration (cells/mL)", 0.0, 1000000.0, value=default_values["AC"], key="AC"
    )
    black_carbon_concentration = placeholder_num_4.number_input(
        "Black carbon concentration (ppb)", 0.0, 10000.0, value=default_values["BCC"], key="BCC"
    )
    mineral_dust_concentration = placeholder_num_5.number_input(
        "Mineral dust concentration (um)", 0.0, 780000.0, value=default_values["MDC"], key="MDC"
    )

    placeholder_title_3.header("Water")


    liquid_water_content = placeholder_num_6.number_input(
        "Liquid water content (%)", 0.0, 0.1, value=default_values["LWC"], key="LWC"
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
