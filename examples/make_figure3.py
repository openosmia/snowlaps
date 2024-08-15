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
import xarray as xr 


path_to_snowlaps = '/Users/au660413/Documents/AU_PhD/snowlaps-emulator'

spectra = pd.read_csv(f"{path_to_snowlaps}/data/spectra/Chevrollier_et_al_2024_TC_spectra.csv",
                      index_col=0)

metadata = pd.read_csv(f"{path_to_snowlaps}/data/spectra/Chevrollier_et_al_2024_TC_metadata.csv")


results_inversion = pd.read_csv(f'{path_to_snowlaps}/data/results/results_inversion_no_anisotropy_correction_Chevrollier2024_TC.csv', 
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

data_aws = pd.read_csv(f'{path_to_snowlaps}/data/irradiance/Finseflux_level1_30min.csv')
data_aws.index = pd.to_datetime(data_aws.iloc[:, 0])
data_aws[data_aws["shortwave_incoming_W/m2"] < 0] = 0

# replace days without data with similar irradiance days
for date, row in data_aws.iterrows():
    if (date.day == 9) and (date.month == 7):
        data_aws["shortwave_incoming_W/m2"].loc[date] = data_aws[
            "shortwave_incoming_W/m2"
        ].loc[date.replace(day=14)]
    if (date.day == 11) and (date.month == 7):
        data_aws["shortwave_incoming_W/m2"].loc[date] = data_aws[
            "shortwave_incoming_W/m2"
        ].loc[date.replace(day=20)]

date_mask = (
    (data_aws.index > "2023-7-8")
    & (data_aws.index <= "2023-8-7")
    & (data_aws["shortwave_incoming_W/m2"] >= 0)
)
data_aws_filtered = data_aws.loc[date_mask]

mean_daily_rad = np.nanmean(
    data_aws_filtered["shortwave_incoming_W/m2"].resample("1D").mean()
)


def calculate_bba(model_run, sza): 
    if sza == 50.0:
        irradiance_data = xr.open_dataset(
            f"{path_to_snowlaps}/data/irradiance/swnb_480bnd_mls_cld.nc"
            ).flx_dwn_sfc.values[9:221]
    else: 
        sza = int(sza)
        irradiance_data = xr.open_dataset(
                f"{path_to_snowlaps}/data/irradiance/swnb_480bnd_mls_clr_SZA{sza}.nc"
                ).flx_dwn_sfc.values[9:221]
    bba = (np.sum(model_run 
                    * irradiance_data) 
                    / np.sum(irradiance_data
                            ))
    return bba


bba_diffs_algae = []
bba_diffs_dust = []
bba_diffs_bc = []
bba_wout_algae = []
bba_wout_dust = []
bba_wout_bc = []
all_laps_bbas = []

for i, spec in enumerate(spectra.columns): 
    all_lap_spectrum = predicted_spectra[i]
    sza = metadata.sza.iloc[i]
    all_lap_bba = calculate_bba(all_lap_spectrum, sza)

    without_laps_spectrum = emulator.run([X_features[i][0], 
                                                X_features[i][1],
                                  0, 
                                  X_features[i][3],
                                  0,
                                  0])
    
    without_laps_bba = calculate_bba(without_laps_spectrum, sza) 
    
    without_alg_spectrum =  emulator.run([X_features[i][0], 
                                                X_features[i][1],
                                  0, 
                                  X_features[i][3],
                                  X_features[i][4],
                                  X_features[i][5]])
    
    without_alg_bba =  calculate_bba(without_alg_spectrum, sza)

    without_dust_spectrum = emulator.run([X_features[i][0], 
                                                X_features[i][1],
                                  X_features[i][2], 
                                  X_features[i][3],
                                  X_features[i][4],
                                  0])
    without_dust_bba = calculate_bba(without_dust_spectrum, sza)
    
    without_bc_spectrum = emulator.run([X_features[i][0], 
                                                X_features[i][1],
                                  X_features[i][2], 
                                  X_features[i][3],
                                  0,
                                  X_features[i][5]])
    
    without_bc_bba = calculate_bba(without_bc_spectrum, sza)
    
    bba_diffs_algae.append(without_alg_bba - all_lap_bba)
    bba_diffs_dust.append(without_dust_bba - all_lap_bba)
    bba_diffs_bc.append(without_bc_bba - all_lap_bba)
    all_laps_bbas.append(all_lap_bba)
    bba_wout_algae.append(without_alg_bba)
    bba_wout_bc.append(without_bc_bba)
    bba_wout_dust.append(without_dust_bba)

 

results_inversion['bba_wout_algae'] = bba_wout_algae
results_inversion['bba_wout_dust'] = bba_wout_dust
results_inversion['bba_wout_bc'] = bba_wout_bc

results_inversion['bba_drop_algae'] = bba_diffs_algae
results_inversion['bba_drop_dust'] = bba_diffs_dust
results_inversion['bba_drop_bc'] = bba_diffs_bc
results_inversion['bba_drop_laps'] = (np.array(bba_diffs_bc) 
                    + np.array(bba_diffs_dust)
                    + np.array(bba_diffs_algae))
results_inversion['bba_drop_algae_pct_total'] = results_inversion['bba_drop_algae'].values / results_inversion['bba_drop_laps'].values
results_inversion['bba_drop_bc_pct_total'] = results_inversion['bba_drop_bc'].values / results_inversion['bba_drop_laps'].values
results_inversion['bba_drop_dust_pct_total'] = results_inversion['bba_drop_dust'].values / results_inversion['bba_drop_laps'].values

results_inversion['rf_algae'] = [bba_drop * mean_daily_rad 
                                for bba_drop
                                in results_inversion.bba_drop_algae.values]
results_inversion['rf_dust'] = [bba_drop * mean_daily_rad 
                                for bba_drop 
                                in results_inversion.bba_drop_dust.values]
results_inversion['rf_bc'] = [bba_drop * mean_daily_rad 
                                for bba_drop
                                in results_inversion.bba_drop_bc.values]
#%%

plt.figure(figsize=(8,4.75))
gs = gridspec.GridSpec(2, 1, width_ratios = [1], height_ratios = [1, 1]) # rows, col


ax2 = plt.subplot(gs[0, 0])
plt.text(0.93, 0.895, "(a)", fontsize=12, transform=plt.gcf().transFigure)
ax2.bar(range(len(bba_diffs_bc)),
        results_inversion.sort_values('bba_drop_algae_pct_total').rf_algae.values,
        label = 'Algal blooms', color='darkred',
        alpha=0.7)
ax2.bar(range(len(bba_diffs_bc)),
        results_inversion.sort_values('bba_drop_algae_pct_total').rf_bc.values,
        bottom = results_inversion.sort_values('bba_drop_algae_pct_total').rf_algae.values,
        label = 'Dark particles', color='black',
        alpha=0.7)
ax2.bar(range(len(bba_diffs_bc)),
        results_inversion.sort_values('bba_drop_algae_pct_total').rf_dust.values,
        bottom = results_inversion.sort_values('bba_drop_algae_pct_total').rf_algae.values
        + results_inversion.sort_values('bba_drop_algae_pct_total').rf_bc.values,
        label = 'Mineral dusts', color='rosybrown',
        alpha=0.7)
ax2.set_ylabel('Mean daily RF'+'\n (W m$^{-2}$)', fontsize=14)
ax2.tick_params(labelsize=14,
                labelbottom=False)
ax2.legend(fontsize=12, ncols=3, frameon=False, handletextpad=0.5)
ax2.set_ylim(0, 45)
ax2.set_xlim(-1,180)

ax1 = plt.subplot(gs[1, 0])
plt.text(0.93, 0.46, "(b)", fontsize=12, transform=plt.gcf().transFigure)
ax1.bar(range(len(bba_diffs_bc)),
        results_inversion.sort_values('bba_drop_algae_pct_total')['bba_drop_algae_pct_total'].values,
        label = 'Algal blooms', color='darkred',
        alpha=0.7)
ax1.bar(range(len(bba_diffs_bc)),
        results_inversion.sort_values('bba_drop_algae_pct_total')['bba_drop_bc_pct_total'].values,
        bottom = results_inversion.sort_values('bba_drop_algae_pct_total')['bba_drop_algae_pct_total'].values,
        label = 'Dark particles', color='black',
        alpha=0.7)
ax1.bar(range(len(bba_diffs_bc)),
        results_inversion.sort_values('bba_drop_algae_pct_total')['bba_drop_dust_pct_total'].values,
        bottom = results_inversion.sort_values('bba_drop_algae_pct_total')['bba_drop_algae_pct_total'].values
        + results_inversion.sort_values('bba_drop_algae_pct_total')['bba_drop_bc_pct_total'].values,
        label = 'Mineral dusts', color='rosybrown',
        alpha=0.7)
ax1.set_ylabel('Fraction of LAP-driven \n darkening', fontsize=13)
ax1.tick_params(labelsize=14)
ax1.set_xlabel('Number of spectral measurements', fontsize=13)
ax1.set_ylim(0,1)
ax1.set_xlim(-1,180)


plt.tight_layout()

plt.savefig(f"{path_to_snowlaps}/examples/figures/fig03_Chevrollier2024_TC.png", dpi=300,
        bbox_inches='tight', pad_inches=0.1,
        facecolor='auto', edgecolor='auto'
    )
    