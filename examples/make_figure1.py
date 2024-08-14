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
params_lower_residuals = [35, 2200, 19000, 0.09, 1220, 1100000]

#%%
emulator = SnowlapsEmulator()

predicted_spectra_emulator = emulator.run([params_higher_residuals,
                                           params_lower_residuals])

#%%
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

#%%
# reshape params for snicar
zenList = [params_higher_residuals[0], 
           params_lower_residuals[0]]
reffList = [params_higher_residuals[1], 
           params_lower_residuals[1]]
algList = [params_higher_residuals[2], 
           params_lower_residuals[2]]
lwcList = [params_higher_residuals[3], 
           params_lower_residuals[3]]
bcList = [params_higher_residuals[4], 
           params_lower_residuals[4]]
dustList = [params_higher_residuals[5], 
           params_lower_residuals[5]]

predicted_spectra_biosnicar = []
 
# iterate over all your values
counter = 0
for algae, dust, bc, reff, zen, lwc in zip(algList,
                                           dustList,
                                           bcList,
                                           reffList,
                                           zenList,
                                           lwcList):
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
        predicted_spectra_biosnicar.append(np.float16(outputs.albedo[9:221]))
        counter += 1

#%%
#%matplotlib qt
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
wvs = np.arange(0.295, 2.415, 0.01)
color_plot="gray"
label_size=15
ticks_label_size=12

# 1. Albedo with max residuals
ax[0].plot(wvs, predicted_spectra_biosnicar[0], 
             label="Biosnicar",  color=color_plot, lw=2, alpha=1)
ax[0].plot(wvs, predicted_spectra_emulator[0], ls='--',
             label="Emulator", color="black", lw=2, alpha=1)
ax[0].set_xlabel(r'Wavelength (μm)', fontsize=label_size, labelpad=7)
ax[0].tick_params(labelsize=ticks_label_size)
ax[0].set_ylim([0, 1])
ax[0].set_xlim([0.25, 2.45])
ax[0].set_yticklabels([])
ax[0].yaxis.tick_right()
ax[0].grid(alpha=0.3)

divider = make_axes_locatable(ax[0])
axtop = divider.append_axes("top", size="40%", pad=0.1, sharex = ax[0])
axtop.plot(wvs, np.abs(predicted_spectra_biosnicar[0] - predicted_spectra_emulator[0]), color='black')
axtop.tick_params(axis='y', labelsize=ticks_label_size)
axtop.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
axtop.set_ylim(-4e-4, 5.05e-3) 
axtop.set_yticks([0, 5e-3])
axtop.yaxis.tick_right()
axtop.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
axtop.yaxis.offsetText.set_fontsize(ticks_label_size)
axtop.grid(alpha=0.3)


# 4. Albedo with min residuals
ax[1].plot(wvs, predicted_spectra_biosnicar[1], label="Biosnicar", 
              color=color_plot, lw=2, alpha=1)
ax[1].plot(wvs, predicted_spectra_emulator[1], ls='--', label="Emulator", 
             color="black", lw=2,  alpha=1)
ax[1].set_xlabel(r'Wavelength (μm)', fontsize=label_size, labelpad=7)
ax[1].yaxis.set_label_position("right")
ax[1].set_ylabel('Albedo', fontsize=label_size, rotation=-90, labelpad=22)
ax[1].tick_params(labelsize=ticks_label_size)
ax[1].yaxis.tick_right()
ax[1].set_ylim([0, 1])
ax[1].set_xlim([0.25, 2.45])
ax[1].grid(alpha=0.3)
ax[1].legend(loc="upper right", fontsize=15)

divider = make_axes_locatable(ax[1])
axtop = divider.append_axes("top", size="40%", pad=0.1, sharex = ax[1])
axtop.plot(wvs, np.abs(predicted_spectra_emulator[1] - predicted_spectra_biosnicar[1]), color='black', label='AE')
axtop.tick_params(axis='y', labelsize=ticks_label_size)
axtop.yaxis.tick_right()
axtop.yaxis.set_label_position("right")
axtop.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
axtop.set_ylabel('Abs. \n error', fontsize=18, rotation=-90, labelpad=40)
axtop.set_ylim(-2.75e-5,  2e-4) 
axtop.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
axtop.yaxis.offsetText.set_fontsize(ticks_label_size)
axtop.grid(alpha=0.3)

fig.tight_layout()

