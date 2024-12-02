import streamlit as st

st.write("# Welcome to Snowlaps! 👋 ❄️ ")

st.markdown(
    """
    Snowlaps is an open-source python package to study the albedo reducing
    effect of red snow algae, mineral dust and black carbon on melting snow 
    surfaces. The package is built on a deep-learning emulator of the radiative
    transfer model [biosnicar](https://biosnicar.vercel.app/) and can be used
    in forward and inverse mode.
    
    In forward mode, the albedo of a snow surface is predicted from the solar
    zenith angle, snow grain size, liquid water content, and abundance of 
    algae, mineral dust and black carbon. In inverse mode, the surface 
    properties are retrieved from prescribed spectral measurements. Snowlaps
    also directly calculates the albedo-reduction caused by each type of 
    surface particles. 
    
    More details and performance evaluation of the model were presented in a
    [recent scientific publication](https://doi.org/10.5194/egusphere-2024-2583).
    
    **👈 Select a mode from the sidebar** to start working with Snowlaps!
    
"""
)


