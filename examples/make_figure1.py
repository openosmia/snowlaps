#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@authors: Adrien Wehrlé (University of Zurich), Lou-Anne Chevrollier (Aarhus University)

"""

#%%

import sys
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
path_to_snowlaps = '/Users/au660413/Documents/AU_PhD/snowlaps-emulator'
path_to_biosnicar = '/Users/au660413/Desktop/github/biosnicar-py'

os.chdir(path_to_snowlaps)
sys.path.append(path_to_biosnicar)

from snowlaps.snowlaps import SnowlapsEmulator
from biosnicar.setup_snicar import setup_snicar
from biosnicar.column_OPs import get_layer_OPs, mix_in_impurities
from biosnicar.adding_doubling_solver import adding_doubling_solver

params_higher_residuals = [35, 350, 0, 0, 0, 0]

emulator = SnowlapsEmulator()

predicted_spectrum_emulator = emulator.run(params_higher_residuals)


# set up snicar with emulator config
emulator_config_file = './data/inputs/snicar_config_for_emulator.yaml'

(
        ice,
        illumination,
        rt_config,
        model_config,
        plot_config,
        impurities,
    ) = setup_snicar(emulator_config_file)


# reshape params for snicar
zen = params_higher_residuals[0]
reff = params_higher_residuals[1]
algae = params_higher_residuals[2]
lwc = params_higher_residuals[3]
bc = params_higher_residuals[4]
dust = params_higher_residuals[5]

ice.rds = [reff] * len(ice.dz)
ice.lwc = [lwc] * len(ice.dz)
illumination.solzen = zen
# remember to recalculate irradiance values when any of the dependency values change
# i.e. irradiance is derived from solzen, so call the recalc func after updating solzen
illumination.calculate_irradiance()
impurities[0].conc = [
    bc,
    0,
]  # bc 
impurities[1].conc = [
    algae,
    0,
]  # alg 
impurities[2].conc = [
    dust,
    0,
]  # dust

ssa_snw, g_snw, mac_snw = get_layer_OPs(ice, model_config)
tau, ssa, g, L_snw = mix_in_impurities(
    ssa_snw,
    g_snw,
    mac_snw,
    ice,
    impurities,
    model_config,
)
# now call the solver of your choice (here AD solver)
outputs = adding_doubling_solver(
    tau, ssa, g, L_snw, ice, illumination, model_config
)
# spectral albedo appended to output array
predicted_spectrum_biosnicar = np.float16(outputs.albedo[9:221])

#%matplotlib qt
fig, ax = plt.subplots(1, 1, figsize=(5,5))
wvs = np.arange(0.295, 2.415, 0.01)
color_plot="gray"
label_size=15
ticks_label_size=12

ax.plot(wvs, predicted_spectrum_biosnicar, 
             label="Biosnicar",  color=color_plot, lw=2, alpha=1)
ax.plot(wvs, predicted_spectrum_emulator, ls='--',
             label="Emulator", color="black", lw=2, alpha=1)
ax.set_xlabel(r'Wavelength (μm)', fontsize=label_size, labelpad=7)
ax.tick_params(labelsize=ticks_label_size)
ax.set_ylim([0, 1])
ax.set_xlim([0.25, 2.45])
ax.grid(alpha=0.3)
ax.set_ylabel('Albedo', fontsize=label_size)


divider = make_axes_locatable(ax)
axtop = divider.append_axes("top", size="40%", pad=0.1, sharex = ax)
axtop.plot(wvs, np.abs(predicted_spectrum_biosnicar - predicted_spectrum_emulator), color='black')
axtop.tick_params(axis='y', labelsize=ticks_label_size)
axtop.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
axtop.set_ylim(-4e-4, 5.05e-3) 
axtop.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
axtop.yaxis.offsetText.set_fontsize(ticks_label_size)
axtop.grid(alpha=0.3)
axtop.set_ylabel('Abs. \n error', fontsize=18)

fig.tight_layout()
