""" UPDRS analysis cDBS vs. burstDBS"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from ..utils import find_folders as find_folders
from ..utils import io as io


GROUP_RESULTS_PATH = find_folders.get_patterned_dbs_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_patterned_dbs_project_path(folder="GroupFigures")

STIMULATION_DICT = {
    "burst": "B",
    "continuous": "A",
}

UPDRS_SUBSCORES = {
    "total": "UPDRS_III",
    "tremor": "subscore_tremor_total",
    "rigidity": "subscore_rigidity_total",
    "bradykinesia": "subscore_bradykinesia_total",
    "gait": "Gait",
}


def load_updrsiii_scores(
    sub: str,
    stimulation: str,
    subscore: str,
    hemisphere=None,
):
    """
    Input:
        - sub: "089"
        - stimulation: "burst" or "continuous"
        - subscore: "total", "tremor", "rigidity", "bradykinesia", "gait",
        - hemisphere: "right", "left" (only if needed, otherwise None)

    """

    score_sheet = io.load_metadata_excel(sub=sub, sheet_name="updrs")

    # only keep stimulation burst or continuous
    stim_sheet = score_sheet[
        score_sheet["Stimulation"].str.contains(
            f"StimOn{STIMULATION_DICT[stimulation]}|StimOff{STIMULATION_DICT[stimulation]}",
            na=False,
        )
    ]

    # check if hemisphere input exists
    if hemisphere in ["right", "left"]:

        # check if subscore of only one hemisphere is correct
        # subscore must be in ["tremor", "rigidity", "bradykinesia"]

        subscore_column = f"subscore_{subscore}_{hemisphere}"

    elif hemisphere is None:

        subscore_column = UPDRS_SUBSCORES[subscore]

    else:
        # Error
        print("Input hemisphere must be empty or 'left' or 'right'.")

    # get the desired scores for each sessions

    return subscore_column, stim_sheet


def lineplot_absolute_updrsiii(
    sub_list: list,
    subscore: str,
    hemisphere=None,
):
    """
    Plot the absolute UPDRS-III scores for a list of subjects

    Input:
        - sub: list of subjects, e.g. ["089", "090"]
        - subscore: "total", "tremor", "rigidity", "bradykinesia", "gait",
        - hemisphere: "right", "left" (only if needed, otherwise None)
    """

    # plot
    colors = plt.cm.Set2(
        np.linspace(0, 1, len(sub_list))
    )  # Use a colormap for distinct colors

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, sub in enumerate(sub_list):

        # load data
        subscore_column, stim_sheet_burst = load_updrsiii_scores(
            sub=sub, stimulation="burst", subscore=subscore, hemisphere=hemisphere
        )

        subscore_column, stim_sheet_continuous = load_updrsiii_scores(
            sub=sub, stimulation="continuous", subscore=subscore, hemisphere=hemisphere
        )

        y_burst = stim_sheet_burst[subscore_column].values
        y_continuous = stim_sheet_continuous[subscore_column].values

        x_burst = range(
            len(stim_sheet_burst["Stimulation"].values)
        )  # Create numerical x-axis values for the categorical labels
        x_continuous = range(
            len(stim_sheet_continuous["Stimulation"].values)
        )  # Create numerical x-axis values for the categorical labels

        # Plot the continuous line
        ax.plot(
            x_burst,
            y_burst,
            linestyle=":",
            marker="o",
            color=colors[idx],
        )

        # Plot the dotted line
        ax.plot(
            x_continuous,
            y_continuous,
            label=f"sub-{idx+1}",
            linestyle="-",
            marker="o",
            color=colors[idx],
        )

    # Add labels and title
    ax.set_xlabel("Time Points")
    ax.set_ylabel(f"{subscore_column}")
    ax.set_title("UPDRS-III scores during and after stimulation")

    x_labels = ["StimON", "StimOFF-0", "StimOFF-30", "StimOFF-60"]

    # Set x-axis and y-axis limits and labels
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)  # Apply categorical labels to the x-axis
    ax.set_ylim(0, 80)  # Set y-axis limits

    # Add a legend
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Subjects",
    )

    # Save the figure
    io.save_fig_png_and_svg(
        path=GROUP_FIGURES_PATH,
        filename=f"lineplot_updrsiii_{subscore}{hemisphere}",
        figure=fig,
    )
    # Display the plot
    plt.show()


def lineplot_normalized_to_StimOnA_updrsiii(
    sub_list: list,
    subscore: str,
    hemisphere=None,
):
    """
    Plot the relative UPDRS-III scores for a list of subjects
    StimOnA (continuous DBS) is 100 % and all other scores are relative to this score

    Input:
        - sub: list of subjects, e.g. ["089", "090"]
        - subscore: "total", "tremor", "rigidity", "bradykinesia", "gait",
        - hemisphere: "right", "left" (only if needed, otherwise None)
    """

    # plot
    colors = plt.cm.Set2(
        np.linspace(0, 1, len(sub_list))
    )  # Use a colormap for distinct colors

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, sub in enumerate(sub_list):

        # load data
        subscore_column, stim_sheet_burst = load_updrsiii_scores(
            sub=sub, stimulation="burst", subscore=subscore, hemisphere=hemisphere
        )

        subscore_column, stim_sheet_continuous = load_updrsiii_scores(
            sub=sub, stimulation="continuous", subscore=subscore, hemisphere=hemisphere
        )

        y_burst = stim_sheet_burst[subscore_column].values
        y_continuous = stim_sheet_continuous[subscore_column].values

        # Normalize to StimOnA
        StimOnA_baseline = stim_sheet_continuous.loc[
            stim_sheet_continuous["Stimulation"] == "StimOnA"
        ]
        StimOnA_baseline = StimOnA_baseline[subscore_column].values[0]
        y_burst = y_burst / StimOnA_baseline * 100
        y_continuous = y_continuous / StimOnA_baseline * 100

        x_burst = range(
            len(stim_sheet_burst["Stimulation"].values)
        )  # Create numerical x-axis values for the categorical labels
        x_continuous = range(
            len(stim_sheet_continuous["Stimulation"].values)
        )  # Create numerical x-axis values for the categorical labels

        # Plot the continuous line
        ax.plot(
            x_burst,
            y_burst,
            linestyle=":",
            marker="o",
            color=colors[idx],
        )

        # Plot the dotted line
        ax.plot(
            x_continuous,
            y_continuous,
            label=f"sub-{idx+1}",
            linestyle="-",
            marker="o",
            color=colors[idx],
        )

    # Add labels and title
    ax.set_xlabel("Time Points")
    ax.set_ylabel(f"{subscore_column}")
    ax.set_title("UPDRS-III scores during and after stimulation")

    x_labels = ["StimON", "StimOFF-0", "StimOFF-30", "StimOFF-60"]

    # Set x-axis and y-axis limits and labels
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)  # Apply categorical labels to the x-axis
    # ax.set_ylim(0, 80)  # Set y-axis limits

    # Add a legend
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Subjects",
    )

    # Save the figure
    io.save_fig_png_and_svg(
        path=GROUP_FIGURES_PATH,
        filename=f"lineplot_updrsiii_normalized_to_StimOnA_{subscore}{hemisphere}",
        figure=fig,
    )
    # Display the plot
    plt.show()
