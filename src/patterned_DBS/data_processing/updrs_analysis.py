""" UPDRS analysis cDBS vs. burstDBS"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

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
    # total: (0, 80)
    # tremor, bradykinesia: (0, 30)
    # rigidity: (0, 15)
    # gait (0, 6)

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
    ax.set_ylabel(f"{subscore_column} [% of continuous DBS ON]")
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


def barplot_absolute_updrsiii(
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
    # Define colors for continuous and burst
    continuous_color = "#4e79a7"  # Blue for continuous stimulation
    burst_color = "#f28e2b"  # Orange for burst stimulation
    bar_width = 0.20
    x_labels = ["StimON", "StimOFF-0", "StimOFF-30", "StimOFF-60"]

    # Plotting each subject's histograms
    fig, axes = plt.subplots(
        len(sub_list), 1, figsize=(10, 6 * len(sub_list)), sharex=True
    )

    # If only one subject, ensure axes is a list
    if len(sub_list) == 1:
        axes = [axes]

    for idx, sub in enumerate(sub_list):

        ax = axes[idx]  # Current subplot for the subject

        # load data
        subscore_column, stim_sheet_burst = load_updrsiii_scores(
            sub=sub, stimulation="burst", subscore=subscore, hemisphere=hemisphere
        )

        subscore_column, stim_sheet_continuous = load_updrsiii_scores(
            sub=sub, stimulation="continuous", subscore=subscore, hemisphere=hemisphere
        )

        y_burst = stim_sheet_burst[subscore_column].values
        y_continuous = stim_sheet_continuous[subscore_column].values

        x_burst = np.arange(
            len(stim_sheet_burst["Stimulation"].values)
        )  # Create numerical x-axis values for the categorical labels
        x_continuous = np.arange(
            len(stim_sheet_continuous["Stimulation"].values)
        )  # Create numerical x-axis values for the categorical labels

        # Plot continuous bars
        ax.bar(
            x_burst - bar_width / 2,
            y_burst,
            bar_width,
            label="Burst DBS",
            color=burst_color,
        )

        # Plot burst bars
        ax.bar(
            x_continuous + bar_width / 2,
            y_continuous,
            bar_width,
            label="Continous DBS",
            color=continuous_color,
        )

        # Set labels and title for each subject
        ax.set_ylabel(f"{subscore} Score")
        ax.set_title(f"sub-{sub} - Continuous vs. Burst Stimulation")
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)

        # Show legend for the first subject only to avoid repetition
        if idx == 0:
            ax.legend(loc="upper right")

    # Set common x-axis label
    fig.text(0.5, 0.04, "Time Points", ha="center")
    fig.suptitle(
        "Comparison of UPDRS-III Scores: Continuous vs. Burst Stimulation per Subject"
    )

    # Save the figure
    io.save_fig_png_and_svg(
        path=GROUP_FIGURES_PATH,
        filename=f"barplot_updrsiii_{subscore}",
        figure=fig,
    )
    # Display the plot
    plt.show()


############## GROUP ANALYSIS ############


def group_updrs_scores(sub_list: list, stimulation: str):
    """
    Load the UPDRS scores for a list of subjects and stimulation type
    """

    # Concatenate the dataframes
    for sub in sub_list:
        updrs_scores = load_updrsiii_scores(
            sub=sub, stimulation=stimulation, subscore="total"
        )
        if sub == sub_list[0]:
            all_updrs_scores = updrs_scores[1]
        else:
            all_updrs_scores = pd.concat([all_updrs_scores, updrs_scores[1]])

    return all_updrs_scores


def extract_data_for_bar_and_line_plot(subscore: str):
    """

    Extract the data for the bar and line plots for the group analysis

    Input:
        - subscore: "total", "tremor", "rigidity", "bradykinesia", "

    Extract the scores for all subjects for
        - continuous DBS ON
        - burst DBS ON
        - StimOFF after cDBS (last StimOFF score, run 1, 2, or 3)

    Output:
        - structured_data: pd.DataFrame with columns: subject, stimulation, score
        - StimOFF_scores: pd.DataFrame with columns: subject, StimOFFA_run
    """
    sub_list = ["084", "080", "075", "086", "087", "088", "089"]

    structured_data = {
        "subject": ["sub-1", "sub-2", "sub-3", "sub-4", "sub-5", "sub-6", "sub-7"] * 3,
        "stimulation": ["continuous"] * 7 + ["burst"] * 7 + ["StimOFF"] * 7,
        "score": [],
    }

    StimOFF_scores = {
        "subject": sub_list,
        "StimOFFA_run": [],
    }

    # first extract continuous score data per subject
    for sub in sub_list:
        subscore_column, stim_sheet = load_updrsiii_scores(
            sub=sub, stimulation="continuous", subscore=subscore
        )

        stim_sheet = stim_sheet.loc[stim_sheet["Stimulation"] == "StimOnA"]

        structured_data["score"].append(stim_sheet[subscore_column].values[0])

    # then extract burst score data per subject
    for sub in sub_list:
        subscore_column, stim_sheet = load_updrsiii_scores(
            sub=sub, stimulation="burst", subscore=subscore
        )

        stim_sheet = stim_sheet.loc[stim_sheet["Stimulation"] == "StimOnB"]

        structured_data["score"].append(stim_sheet[subscore_column].values[0])

    # then extract StimOFF score data per subject: last Stim OFF score per cDSB OFF
    for sub in sub_list:
        subscore_column, stim_sheet = load_updrsiii_scores(
            sub=sub, stimulation="continuous", subscore=subscore
        )

        # check if StimOFFA_run-3 exists
        if "StimOffA_run-3" in stim_sheet["Stimulation"].values:
            stim_sheet = stim_sheet.loc[stim_sheet["Stimulation"] == "StimOffA_run-3"]
            StimOFF_scores["StimOFFA_run"].append("run-3")

        elif "StimOffA_run-2" in stim_sheet["Stimulation"].values:
            stim_sheet = stim_sheet.loc[stim_sheet["Stimulation"] == "StimOffA_run-2"]
            StimOFF_scores["StimOFFA_run"].append("run-2")

        elif "StimOffA_run-1" in stim_sheet["Stimulation"].values:
            stim_sheet = stim_sheet.loc[stim_sheet["Stimulation"] == "StimOffA_run-1"]
            StimOFF_scores["StimOFFA_run"].append("run-1")

        else:
            print(
                "StimOffA_run-3, StimOffA_run-2, StimOffA_run-1 not found in the data"
            )
            StimOFF_scores["StimOFFA_run"].extend("not found")

        structured_data["score"].append(stim_sheet[subscore_column].values[0])

    return pd.DataFrame(structured_data), pd.DataFrame(StimOFF_scores)


def plot_bar_line_group_plot(
    subscore: str,
):
    """
    Plot the bar and line plot for the group UPDRS analysis

    """
    # has to be this order, because I use sub-1, sub-2, etc. in the plot
    sub_list = ["084", "080", "075", "086", "087", "088", "089"]

    # load the structured data
    structured_data, StimOFF_scores = extract_data_for_bar_and_line_plot(
        subscore=subscore
    )

    colors = plt.cm.Set2(
        np.linspace(0, 1, len(sub_list))
    )  # Use a colormap for distinct colors

    # Create the barplot
    # plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=structured_data,
        x="stimulation",
        y="score",
        ci=None,
        color="lightgray",
        alpha=0.7,
        ax=ax,
    )

    # Add individual data points with lines connecting the same subjects
    for idx, sub in enumerate(structured_data["subject"].unique()):
        subj_data = structured_data[structured_data["subject"] == sub]
        ax.plot(
            subj_data["stimulation"],
            subj_data["score"],
            marker="o",
            label=sub,
            alpha=0.7,
            color=colors[idx],
        )

    # Add some visual touches
    ax.set_title("UPDRS-III scores", fontsize=14)
    ax.set_xlabel("Stimulation", fontsize=12)
    ax.set_ylabel(subscore, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(title="Subjects", bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()

    # save figure
    io.save_fig_png_and_svg(
        path=GROUP_FIGURES_PATH,
        filename=f"line_and_barplot_updrsiii_group_{subscore}",
        figure=fig,
    )


####### STATISTICAL ANALYSIS ########
