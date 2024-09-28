import streamlit as st
import pandas as pd
import plotly.express as px
from snowlaps.snowlaps import SnowlapsEmulator
from importlib import reload

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


def plot_albedo(spectra):
    fig = px.line(
        spectra,
        range_y=[0, 1],
        labels={"index": "wavelengths (microns)", "value": "Albedo"},
    )
    fig.update_layout(showlegend=False)
    return fig


spectra = st.sidebar.selectbox("Choose Spectra", albedo_spectra.columns)

if st.button("Click Me"):
    if uploaded_data is not None and uploaded_metadata is not None:
        with st.spinner("Please wait..."):
            (
                full_batch_optimization_results,
                best_optimization_results,
                best_emulator_spectra,
            ) = my_emulator.optimize(
                albedo_spectra_path=albedo_spectra.loc[:, spectra],
                spectra_metadata_path=albedo_metadata.loc[spectra, :],
                save_results=False,
            )

        st.plotly_chart(plot_albedo(best_emulator_spectra))

        st.success("Done!")

        with st.expander("Show snowlaps inversion results"):
            st.dataframe(best_optimization_results)
