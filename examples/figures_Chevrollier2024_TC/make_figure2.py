#!/usr/bin/env python3

import matplotlib.gridspec as gridspec
import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from snowlaps.snowlaps import SnowlapsEmulator

path_to_snowlaps = ""

spectra = pd.read_csv(
    f"{path_to_snowlaps}/data/spectra/Chevrollier_et_al_2024_TC_spectra.csv",
    index_col=0,
)

metadata = pd.read_csv(
    f"{path_to_snowlaps}/data/spectra/Chevrollier_et_al_2024_TC_metadata.csv"
)


results_inversion = pd.read_csv(
    f"{path_to_snowlaps}/data/optimization_results/results_inversion_no_anisotropy_correction_Chevrollier2024_TC.csv",
    index_col=0,
)


X_features = [
    [
        results_inversion.sza.values[i],
        results_inversion.grain_size.values[i],
        results_inversion.algae.values[i],
        results_inversion.lwc.values[i],
        results_inversion.bc.values[i],
        results_inversion.dust.values[i],
    ]
    for i in range(len(results_inversion.index))
]

emulator = SnowlapsEmulator()

predicted_spectra = emulator.run(X_features)
wavelengths_emulator = np.arange(0.295, 2.410, 0.01)

# mask spectra around absorption features
spectra.loc[1350:1450] = np.nan
spectra.loc[1775:2100] = np.nan


def make_figure_spectra_inversion(
    ax,
    spectra,
    letterfig,
    results_inversion,
    predicted_spectra,
    spec_name,
    set_y_label,
    set_y_ticks,
    set_x_label,
    set_x_ticks,
):
    spec_idx = int(np.where(spectra.columns == spec_name)[0][0])
    ax.plot(
        spectra.index / 1000,
        np.array(spectra.iloc[:, spec_idx]),
        color="gray",
        alpha=0.8,
        lw=2,
    )
    mae = results_inversion.maes.iloc[spec_idx]
    alg = results_inversion.algae.iloc[spec_idx]
    dust = results_inversion.dust.iloc[spec_idx]
    bc = results_inversion.bc.iloc[spec_idx]
    lwc = int(results_inversion.lwc.iloc[spec_idx] * 100)
    ax.plot(
        wavelengths_emulator, predicted_spectra[spec_idx], "--", color="black", lw=2
    )

    ax.set_ylim(0, 1)

    ax.annotate(
        (
            f"MAE {mae:.3f}"
            + f"\n {alg:.0f} cells"
            + r" mL$^{-1}$"
            + f"\n {dust/1000:.0f} ppm dust"
            + f"\n {bc:.0f} ppb bc"
            + f"\n {lwc:.0f} % LWC"
        ),
        xy=(0.95, 0.92),
        ha="right",
        xycoords="axes fraction",
        size=9,
        va="top",
        backgroundcolor="none",
        bbox=dict(
            facecolor="none", edgecolor="none", boxstyle="square, pad=0.6", lw=0.25
        ),
    )

    ax.annotate(
        letterfig,
        xy=(0.125, 0.105),
        ha="right",
        xycoords="axes fraction",
        size=9,
        va="top",
        backgroundcolor="none",
        bbox=dict(
            facecolor="none", edgecolor="none", boxstyle="square, pad=0.6", lw=0.25
        ),
    )

    ax.set_xlim(0.350, 1.8)
    ax.xaxis.tick_top()

    if set_y_label:
        ax.set_ylabel("Albedo", fontsize=14)
    else:
        ax.set_ylabel("", fontsize=12)
    if set_x_label:
        ax.set_xlabel("Wavelengths (µm)", fontsize=14, labelpad=7)
        ax.xaxis.set_label_position("top")
    else:
        ax.set_xlabel("", fontsize=12)
    if not set_y_ticks:
        ax.tick_params(axis="both", which="both", labelleft=False)
    if not set_x_ticks:
        ax.tick_params(axis="both", which="both", labeltop=False)


plt.figure(figsize=(12, 5.5))
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1])
gs.update(wspace=0.075, hspace=0.075)


ax1 = plt.subplot(gs[0, 0])
make_figure_spectra_inversion(
    ax1,
    spectra,
    "(a)",
    results_inversion,
    predicted_spectra,
    "072923_SNOW2",
    set_y_label=True,
    set_y_ticks=True,
    set_x_label=True,
    set_x_ticks=True,
)
ax2 = plt.subplot(gs[0, 1])
make_figure_spectra_inversion(
    ax2,
    spectra,
    "(b)",
    results_inversion,
    predicted_spectra,
    "071423_SNOW9",
    set_y_label=False,
    set_y_ticks=False,
    set_x_label=True,
    set_x_ticks=True,
)
ax3 = plt.subplot(gs[0, 2])
make_figure_spectra_inversion(
    ax3,
    spectra,
    "(c)",
    results_inversion,
    predicted_spectra,
    "080623_SNOW1",
    set_y_label=False,
    set_y_ticks=False,
    set_x_label=True,
    set_x_ticks=True,
)
ax4 = plt.subplot(gs[0, 3])
make_figure_spectra_inversion(
    ax4,
    spectra,
    "(d)",
    results_inversion,
    predicted_spectra,
    "080123_SNOW23",
    set_y_label=False,
    set_y_ticks=False,
    set_x_label=True,
    set_x_ticks=True,
)
ax5 = plt.subplot(gs[1, 0])
file = (
    f"{path_to_snowlaps}/data/surface_pictures/IMG_20230729_135154_KLEMSBU-02_gimp.jpg"
)
img = image.imread(file)
ax5.imshow(img[1200:2600, 910:2500], aspect="auto")  # [300:2800, 630:3300]
ax5.set_xticks([])
ax5.set_yticks([])

ax6 = plt.subplot(gs[1, 1])
file = f"{path_to_snowlaps}/data/surface_pictures/IMG_20230714_135017_FARC-BLÅISENVEGEN-09_gimp.jpg"
img = image.imread(file)
ax6.imshow(
    np.flip(img, axis=1)[1200:3000, 330:2600], aspect="auto"
)  # [830+275:3550-275, 154+275:2850-275]
ax6.set_xticks([])
ax6.set_yticks([])

ax7 = plt.subplot(gs[1, 2])
file = f"{path_to_snowlaps}/data/surface_pictures/IMG_20230806_132724_BUKKASKINN-01_gimp.jpg"
img = image.imread(file)
ax7.imshow(
    img[1350:3250, 420:2350], aspect="auto"
)  # [910+250:3700-275, 50+250:2850-275]
ax7.set_xticks([])
ax7.set_yticks([])

ax8 = plt.subplot(gs[1, 3])
file = f"{path_to_snowlaps}/data/surface_pictures/IMG_20230801_153759_STYGGELVANE-2-23_gimp.jpg"
img = image.imread(file)
ax8.imshow(
    np.flip(img, axis=1)[1500 - 200 : 3400 + 50, 500 - 150 : 2400 + 120], aspect="auto"
)  # [950+275:3770-275, 90+275:2920-275]
ax8.set_xticks([])
ax8.set_yticks([])


# plt.savefig(f"{path_to_snowlaps}/examples/figures_Chevrollier2024_TC/fig02_Chevrollier2024_TC.png", dpi=300,
#         bbox_inches='tight', pad_inches=0.1,
#         facecolor='auto', edgecolor='auto'
#     )
