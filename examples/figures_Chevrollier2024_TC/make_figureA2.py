#!/usr/bin/env python3
"""
@authors: Adrien Wehrlé (University of Zurich), Lou-Anne Chevrollier (Aarhus University)

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path_to_snowlaps = ""

results_inversion_no_correction = pd.read_csv(
    f"{path_to_snowlaps}/data/optimization_results/results_inversion_no_anisotropy_correction_Chevrollier2024_TC.csv",
    index_col=0,
)
results_30deg = pd.read_csv(
    f"{path_to_snowlaps}/data/optimization_results/results_inversion_30deg_anisotropy_correction_Chevrollier2024_TC.csv",
    index_col=0,
)
results_60deg = pd.read_csv(
    f"{path_to_snowlaps}/data/optimization_results/results_inversion_60deg_anisotropy_correction_Chevrollier2024_TC.csv",
    index_col=0,
)


plt.figure(figsize=(11, 4))
plt.subplot(131)
plt.text(0.005, 13.7, "(a)", fontsize=12)
plt.scatter(
    results_inversion_no_correction.algae / 1e4,
    results_30deg.algae / 1e4,
    color="gray",
    label=r"30 $\degree$ incident angle",
    alpha=0.4,
)
plt.scatter(
    results_inversion_no_correction.algae / 1e4,
    results_60deg.algae / 1e4,
    color="darkred",
    label=r"60 $\degree$ incident angle",
    alpha=0.3,
)
plt.plot(np.arange(0, 1.5e5 / 1e4), np.arange(0, 1.5e5 / 1e4), "--", color="black")
plt.title("Algal concentration \n" + r" (10$^{-4}$ cells mL$^{-1}$)", fontsize=14)
plt.ylabel("With anisotropy correction", fontsize=14)
plt.xlabel("No anisotropy correction", fontsize=14)
plt.legend(handletextpad=0.01, frameon=False, fontsize=12)

plt.subplot(132)
plt.text(0.008, 1.21, "(b)", fontsize=12)
plt.scatter(
    results_inversion_no_correction.dust / 1e6,
    results_30deg.dust / 1e6,
    color="gray",
    alpha=0.4,
)
plt.scatter(
    results_inversion_no_correction.dust / 1e6,
    results_60deg.dust / 1e6,
    color="darkred",
    alpha=0.3,
)
plt.plot(np.arange(0, 1.4, 0.1), np.arange(0, 1.4, 0.1), "--", color="black")
plt.title("Dust concentration \n" + r" (10$^{-6}$ ppb)", fontsize=14)
plt.xlabel("No anisotropy correction", fontsize=14)

plt.subplot(133)
plt.text(15, 2320, "(c)", fontsize=12)
plt.scatter(
    results_inversion_no_correction.bc, results_30deg.bc, color="gray", alpha=0.4
)
plt.scatter(
    results_inversion_no_correction.bc, results_60deg.bc, color="darkred", alpha=0.3
)
plt.plot(np.arange(0, 2300, 100), np.arange(0, 2300, 100), "--", color="black")
plt.title("BC concentration \n" + " (ppb)", fontsize=14)
plt.xlabel("No anisotropy correction", fontsize=14)

plt.tight_layout()

# plt.savefig(f"{path_to_snowlaps}/examples/figures_Chevrollier2024_TC/figa2_Chevrollier2024_TC.png", dpi=300,
#         bbox_inches='tight', pad_inches=0.1,
#         facecolor='auto', edgecolor='auto'
#     )
