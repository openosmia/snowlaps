import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from snowlaps.snowlaps import SnowlapsEmulator
from importlib import reload
import matplotlib.pyplot as plt

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
    spectra = st.sidebar.selectbox("Choose Spectra", albedo_spectra.columns)


uploaded_metadata = st.file_uploader(
    "Choose a CSV file containing the metadata of the albedo spectra"
)

if uploaded_metadata is not None:
    # Can be used wherever a "file-like" object is accepted:
    albedo_metadata = pd.read_csv(uploaded_metadata, index_col=0)
    st.write(albedo_metadata)



def plot_albedo(emulator, measure):
    fig1 = px.line(measure)
    fig1.update_traces(line=dict(color="Blue", width=3))
    fig2 = px.line(emulator)
    fig2.update_traces(line=dict(color="Gray", width=2.5, dash="dash"))
    layout = go.Layout(xaxis=dict(title='wavelengths (microns)'),
                   yaxis=dict(title='Albedo'))
    fig3 = go.Figure(data= fig1.data + fig2.data, layout=layout)
    fig3.update_xaxes(range=[350, 2500])
    series_names = ["measures", "emulator"]
    for idx, name in enumerate(series_names):
        fig3.data[idx].name = name
        fig3.data[idx].hovertemplate = name
    return fig3


if st.button("Run inversion"):
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
        st.plotly_chart(plot_albedo(best_emulator_spectra, albedo_spectra[spectra]))
        st.success("Done!")

        with st.expander("Show snowlaps inversion results"):
            st.dataframe(best_optimization_results)
