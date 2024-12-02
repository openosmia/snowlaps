#!/usr/bin/env python3

"""

NOTE: this script requires an installation of the main branch of the biosnicar 
model (https://github.com/jmcook1186/biosnicar-py/tree/master)

"""

# paths to the biosnicar model must be specified
path_to_biosnicar = ""
import sys

sys.path.append(path_to_biosnicar)
import matplotlib.pyplot as plt
import numpy as np
from biosnicar.adding_doubling_solver import adding_doubling_solver
from biosnicar.column_OPs import get_layer_OPs, mix_in_impurities
from biosnicar.setup_snicar import setup_snicar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from snowlaps.snowlaps import SnowlapsEmulator

##############################################################################
# example reproducing the Figure 1b from Chevrollier et al. 2024
##############################################################################

# See documentation for the order of input parameters.
# (0) solar zenith angle (unitless)
# (1) optical radius of snow grain (um)
# (2) algal concentration (cells/mL)
# (3) liquid water content (fraction, unitless)
# (4) black carbon concentration (ppb)
# (5) mineral dust concentration (ppb)

params_higher_residuals_Chevrollier_2024 = [35, 350, 0, 0, 0, 0]

emulator = SnowlapsEmulator()

predicted_spectrum_emulator = emulator.run(params_higher_residuals_Chevrollier_2024)


# set up snicar with emulator config
emulator_config_file = "./data/inputs/snicar_config_for_emulator.yaml"

(
    ice,
    illumination,
    rt_config,
    model_config,
    plot_config,
    impurities,
) = setup_snicar(emulator_config_file)

# run snicar with given parameters

zen = params_higher_residuals_Chevrollier_2024[0]
reff = params_higher_residuals_Chevrollier_2024[1]
algae = params_higher_residuals_Chevrollier_2024[2]
lwc = params_higher_residuals_Chevrollier_2024[3]
bc = params_higher_residuals_Chevrollier_2024[4]
dust = params_higher_residuals_Chevrollier_2024[5]

ice.rds = [reff] * len(ice.dz)
ice.lwc = [lwc] * len(ice.dz)
illumination.solzen = zen
illumination.calculate_irradiance()
impurities[0].conc = [
    bc,
    0,
]
impurities[1].conc = [
    algae,
    0,
]
impurities[2].conc = [
    dust,
    0,
]

ssa_snw, g_snw, mac_snw = get_layer_OPs(ice, model_config)
tau, ssa, g, L_snw = mix_in_impurities(
    ssa_snw,
    g_snw,
    mac_snw,
    ice,
    impurities,
    model_config,
)

outputs = adding_doubling_solver(tau, ssa, g, L_snw, ice, illumination, model_config)

predicted_spectrum_biosnicar = np.float16(outputs.albedo[9:221])

# make figure comparing biosnicar to the emulator

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
wvs = np.arange(0.295, 2.415, 0.01)
color_plot = "gray"
label_size = 15
ticks_label_size = 12

ax.plot(
    wvs,
    predicted_spectrum_biosnicar,
    label="Biosnicar",
    color=color_plot,
    lw=2,
    alpha=1,
)
ax.plot(
    wvs,
    predicted_spectrum_emulator,
    ls="--",
    label="Emulator",
    color="black",
    lw=2,
    alpha=1,
)
ax.set_xlabel(r"Wavelength (μm)", fontsize=label_size, labelpad=7)
ax.tick_params(labelsize=ticks_label_size)
ax.set_ylim([0, 1])
ax.set_xlim([0.25, 2.45])
ax.grid(alpha=0.3)
ax.set_ylabel("Albedo", fontsize=label_size)


divider = make_axes_locatable(ax)
axtop = divider.append_axes("top", size="40%", pad=0.1, sharex=ax)
axtop.plot(
    wvs,
    np.abs(predicted_spectrum_biosnicar - predicted_spectrum_emulator),
    color="black",
)
axtop.tick_params(axis="y", labelsize=ticks_label_size)
axtop.tick_params(
    axis="x",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=False,
)
axtop.set_ylim(-4e-4, 5.05e-3)
axtop.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
axtop.yaxis.offsetText.set_fontsize(ticks_label_size)
axtop.grid(alpha=0.3)
axtop.set_ylabel("Abs. \n error", fontsize=18)

fig.tight_layout()
