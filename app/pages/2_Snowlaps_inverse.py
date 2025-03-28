import pandas as pd
import plotly.graph_objs as go
import streamlit as st

from snowlaps.snowlaps import SnowlapsEmulator

st.markdown(
    """


### Try snowlaps to retrieve snow properties from observations! :zap:


#### 1 - Load your files and run the inversion :boom:

First, you must give snowlaps your spectral observations as a csv file
with the spectral observations in the range 350-2500nm (typically measured with
a field spectrometer or a hyperspectral satellite). The unique IDs of your
spectra are given as columns, and the wavelengths are given as index. Then,
the metadata associated with the spectra must be prescribed, with the spectral
IDs as index and at minimum 1) the latitude, longitude and time for each of the
measurements or 2) the SZA corresponding to each measurement.

Once you are done, you can click on 'run inversion' and snowlaps will look for
the surface properties that best explain your measurements.

:bulb: Check [the example files](https://github.com/openosmia/snowlaps-emulator/tree/main/data/spectra)
to see the formatting required.

:hourglass_flowing_sand: We are working on making the back-end code more
flexible with the input file formatting. Feel free to [share your ideas](https://github.com/openosmia/snowlaps-emulator/discussions)
with us!



"""
)


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
    albedo_metadata = pd.read_csv(uploaded_metadata, index_col=0)
    st.session_state.albedo_spectra = albedo_spectra
    st.write(albedo_metadata)


def run_snowlaps(parameters) -> pd.DataFrame:
    emulator_results = my_emulator.run(parameters=parameters)

    return pd.DataFrame(emulator_results, index=my_emulator.emulator_wavelengths)


@st.cache_data(show_spinner=False)
def run_model():
    (
        full_batch_optimization_results,
        best_optimization_results,
        best_emulator_spectra,
    ) = my_emulator.optimize(
        albedo_spectra_path=albedo_spectra,
        spectra_metadata_path=albedo_metadata,
        save_results=False,
    )
    st.session_state.best_optimization_results = best_optimization_results
    st.session_state.best_emulator_spectra = best_emulator_spectra
    return best_optimization_results


def plot_inversion(spectrum):
    emulator = st.session_state.best_emulator_spectra[spectrum]
    df_m = pd.DataFrame({"measures": st.session_state.albedo_spectra[spectrum]})
    df = pd.concat([df_m, emulator], axis=1)
    fig = go.Figure(
        layout=go.Layout(
            xaxis=dict(title="Wavelengths (nm)"), yaxis=dict(title="Albedo")
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["measures"],
            mode="lines",
            name=spectrum,
            line=dict(color="royalblue", width=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[spectrum],
            mode="lines+markers",
            name="snowlaps-emulator",
            line=dict(color="white", width=3),
        )
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_xaxes(range=[350, 2400])
    fig.update_yaxes(range=[0, 1])
    return fig


def plot_forward(spectrum, parameters):
    df_m = pd.DataFrame({"measures": st.session_state.albedo_spectra[spectrum]})
    fig1 = go.Figure(
        layout=go.Layout(
            xaxis=dict(title="Wavelengths (nm)"), yaxis=dict(title="Albedo")
        )
    )
    emulator_results = run_snowlaps(parameters)
    df = pd.concat([df_m, emulator_results], axis=1)
    df = df.rename(columns={df.columns[1]: "emulator"})
    fig1.add_trace(
        go.Scatter(
            x=df.index,
            y=df["measures"],
            mode="lines",
            name=spectrum,
            line=dict(color="royalblue", width=4),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=df.index,
            y=df["emulator"],
            mode="lines+markers",
            name="snowlaps-emulator",
            line=dict(color="white", width=2),
        )
    )
    fig1.update_xaxes(range=[350, 2400])
    fig1.update_yaxes(range=[0, 1])
    return fig1


placeholder_title_solar_geometry = st.sidebar.empty()
placeholder_SZA = st.sidebar.empty()

placeholder_title_snow_structure = st.sidebar.empty()
placeholder_optical_radius = st.sidebar.empty()
placeholder_lwc = st.sidebar.empty()


placeholder_title_LAPs = st.sidebar.empty()
placeholder_algae = st.sidebar.empty()
placeholder_black_carbon = st.sidebar.empty()
placeholder_dust = st.sidebar.empty()


placeholder_button = st.sidebar.empty()


def change_input():
    st.session_state.spectrum = spectrum


if (
    st.button("Run inversion")
    and "inv" not in st.session_state
    and uploaded_data is not None
    and uploaded_metadata is not None
):
    with st.spinner("Please wait..."):
        best_optimization_results = run_model()
        st.session_state.inv = True


st.markdown(
    """



#### 2 - Visually inspect the inversion :female-detective: and download the results

:point_down: The first graph below displays your measurement with the best fit
found by snowlaps, for any selected measurement.



"""
)


if "inv" in st.session_state:
    spectrum = st.selectbox(
        "Choose spectrum",
        st.session_state.best_optimization_results.index,
        on_change=change_input,
    )
    st.plotly_chart(plot_inversion(spectrum))
    with st.expander("Show snowlaps inversion results"):
        st.dataframe(st.session_state.best_optimization_results.loc[spectrum])
    with st.sidebar:
        placeholder_title_solar_geometry.header("Solar geometry")

        SZA = placeholder_SZA.number_input(
            "Solar Zenith Angle (SZA; degrees)",
            0.0,
            90.0,
            value=st.session_state.best_optimization_results.loc[spectrum]["sza"],
        )

        placeholder_title_snow_structure.header("Snow structure")

        optical_radius = placeholder_optical_radius.number_input(
            "Snow optical radius (µm)",
            0.0,
            1000.0,
            value=st.session_state.best_optimization_results.loc[spectrum][
                "grain_size"
            ],
        )

        liquid_water_content = placeholder_lwc.number_input(
            "Liquid water content (%)",
            0.0,
            0.1,
            value=st.session_state.best_optimization_results.loc[spectrum]["lwc"],
        )

        placeholder_title_LAPs.header("Light Absorbing Particles (LAPs)")

        algae_concentration = placeholder_algae.number_input(
            "Algae concentration (cells/mL)",
            0.0,
            1000000.0,
            value=st.session_state.best_optimization_results.loc[spectrum]["algae"],
        )
        black_carbon_concentration = placeholder_black_carbon.number_input(
            "Black carbon concentration (ppb)",
            0.0,
            10000.0,
            value=st.session_state.best_optimization_results.loc[spectrum]["bc"],
        )
        mineral_dust_concentration = placeholder_dust.number_input(
            "Mineral dust concentration (ppb)",
            0.0,
            780000.0,
            value=st.session_state.best_optimization_results.loc[spectrum]["dust"],
        )

    parameters = [
        SZA,
        optical_radius,
        algae_concentration,
        liquid_water_content,
        black_carbon_concentration,
        mineral_dust_concentration,
    ]

    st.markdown(
        """



    :point_left: A side bar appeared! The second gaph lets you play with the
    surface parameters retrieved by snowlaps and check how it changes the albedo.
    You can for example check what the albedo would look like if there was no algal
    bloom, or a much bigger grain size? :snowman:


    """
    )

    st.plotly_chart(plot_forward(spectrum, parameters))
