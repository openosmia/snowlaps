#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@authors: Adrien Wehrlé (University of Zurich), Lou-Anne Chevrollier (Aarhus University)

"""

import pandas as pd
import numpy as np
from snowlaps.snowlaps import SnowlapsEmulator
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.image as image

path_to_snowlaps = ''

spectra = pd.read_csv(f"{path_to_snowlaps}/data/spectra/Chevrollier_et_al_2024_TC_spectra.csv",
                      index_col=0)

metadata = pd.read_csv(f"{path_to_snowlaps}/data/spectra/Chevrollier_et_al_2024_TC_metadata.csv")


results_inversion = pd.read_csv(f'{path_to_snowlaps}/data/optimization_results/results_inversion_no_anisotropy_correction_Chevrollier2024_TC.csv', 
                                index_col=0)


X_features = [[results_inversion.sza.values[i],
               results_inversion.grain_size.values[i],
               results_inversion.algae.values[i],
               results_inversion.lwc.values[i],
               results_inversion.bc.values[i],
               results_inversion.dust.values[i],
               
    ] for i in range(len(results_inversion.index))]

emulator = SnowlapsEmulator()

predicted_spectra = emulator.run(X_features)
wavelengths_emulator = np.arange(0.295, 2.410, 0.01)

# mask spectra around absorption features
spectra.loc[1350:1450] = np.nan
spectra.loc[1775:2100] = np.nan

def make_figure_spectra_inversion(ax, spectra, letterfig, results_inversion, 
                                  predicted_spectra, spec_name,
                                  set_y_label,
                                  set_y_ticks,
                                  set_x_label,
                                  set_x_ticks):
    
    spec_idx = int(np.where(spectra.columns == spec_name)[0][0])
    ax.plot(spectra.index/1000, 
        np.array(spectra.iloc[:,spec_idx]), 
              color='gray', 
              alpha=0.8,
              lw=2
              )
    mae = results_inversion.maes.iloc[spec_idx]
    alg = results_inversion.algae.iloc[spec_idx]
    dust = results_inversion.dust.iloc[spec_idx]
    bc = results_inversion.bc.iloc[spec_idx]
    lwc = int(results_inversion.lwc.iloc[spec_idx]*100)
    ax.plot(wavelengths_emulator, 
                predicted_spectra[spec_idx], 
                '--',
                color='black',
                lw=2
                )
    
    ax.set_ylim(0, 1)

    ax.annotate((f'MAE {mae:.3f}' 
               +f'\n {alg:.0f} cells' +r' mL$^{-1}$' 
               +f'\n {dust/1000:.0f} ppm dust'
               +f'\n {bc:.0f} ppb bc' 
               + f'\n {lwc:.0f} % LWC'
              ), 
            xy=(0.95,0.92), ha='right',
            xycoords='axes fraction',
            size=9,  va='top', 
            backgroundcolor='none',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='square, pad=0.6',
                      lw=0.25))
    
    ax.annotate(letterfig, 
            xy=(0.105,0.105), ha='right',
            xycoords='axes fraction',
            size=9,  va='top', 
            backgroundcolor='none',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='square, pad=0.6',
                      lw=0.25))
    
    
    ax.set_xlim(0.350, 1.8)
    ax.xaxis.tick_top()
    
    if set_y_label:
        ax.set_ylabel('Albedo', fontsize=14)
    else:
        ax.set_ylabel('', fontsize=12)
    if set_x_label:
        ax.set_xlabel('Wavelengths (µm)' , fontsize=14,
                      labelpad=7)
        ax.xaxis.set_label_position('top') 
    else:
        ax.set_xlabel('', fontsize=12)
    if not set_y_ticks:
        ax.tick_params(
            axis='both',         
            which='both',         
            labelleft=False)
    if not set_x_ticks:
        ax.tick_params(
            axis='both',         
            which='both',      
            labeltop=False)
        

plt.figure(figsize=(11,6))
gs = gridspec.GridSpec(2, 3, width_ratios = [1, 1, 1], 
                       height_ratios = [1, 1]) 
gs.update(wspace=0.075, hspace=0.075)

ax1 = plt.subplot(gs[0,0])
make_figure_spectra_inversion(ax1, spectra, 
                              "(a)",
                              results_inversion, 
                              predicted_spectra, 
                              '072823_SNOW18', 
                              set_y_label=True,
                              set_y_ticks=True,
                              set_x_label=True,
                              set_x_ticks=True)
ax2 = plt.subplot(gs[0,1])
make_figure_spectra_inversion(ax2, spectra, 
                              "(b)",
                              results_inversion, 
                              predicted_spectra,
                              '073123_SNOW7',
                              set_y_label=False,
                              set_y_ticks=False,
                              set_x_label=True,
                              set_x_ticks=True)
ax3 = plt.subplot(gs[0,2])
make_figure_spectra_inversion(ax3, spectra, 
                              "(c)",
                              results_inversion, 
                              predicted_spectra, 
                              '071423_SNOW13',
                              set_y_label=False,
                              set_y_ticks=False,
                              set_x_label=True,
                              set_x_ticks=True)

ax5 = plt.subplot(gs[1,0])
file = f"{path_to_snowlaps}/data/surface_pictures/IMG_20230728_155114_MIDDALENVEGEN-2-18.jpg"
img = image.imread(file)
ax5.imshow(img[750:2500, 860:2900], aspect='auto')
ax5.set_xticks([])
ax5.set_yticks([])

ax6 = plt.subplot(gs[1,1])
file = f"{path_to_snowlaps}/data/surface_pictures/IMG_20230731_152434_STYGGELVANE-07.jpg"
img = image.imread(file)
ax6.imshow(img[600:2600, 620:2800], aspect='auto') 
ax6.set_xticks([])
ax6.set_yticks([])

ax7 = plt.subplot(gs[1,2])
file = f"{path_to_snowlaps}/data/surface_pictures/IMG_20230714_141116_FARC-BLÅISENVEGEN-13.jpg"
img = image.imread(file)
ax7.imshow(np.flip(img[750:2550, 1320:3320], axis=(1)), aspect='auto') #[750:2550, 700:2700]
ax7.set_xticks([])
ax7.set_yticks([])

plt.tight_layout()

# plt.savefig(f"{path_to_snowlaps}/examples/figures_Chevrollier2024_TC/figa1_Chevrollier2024_TC", dpi=300,
#         bbox_inches='tight', pad_inches=0.1,
#         facecolor='auto', edgecolor='auto'
#     )