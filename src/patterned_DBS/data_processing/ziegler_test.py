""" Ziegler test scores analysis """

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import scipy.stats as stats

from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests


from ..utils import find_folders as find_folders
from ..utils import io as io


GROUP_RESULTS_PATH = find_folders.get_patterned_dbs_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_patterned_dbs_project_path(folder="GroupFigures")

STIMULATION_DICT = {
    "burst": "B",
    "continuous": "A",
}


def load_ziegler(
    sub: str,
    stimulation: str,
    sheet: str,
):
    """
    Input:
        - sub: "084" only works for case subject
        - stimulation: "burst" or "continuous"
        - task: "ziegler_score", "ziegler_times"

    """

    score_sheet = io.load_gait_excel(sub=sub, sheet_name=sheet)

    # only keep stimulation burst or continuous
    stim_sheet = score_sheet[
        score_sheet["stimulation"].str.contains(
            f"StimOn{STIMULATION_DICT[stimulation]}|StimOff{STIMULATION_DICT[stimulation]}",
            na=False,
        )
    ]

    # get the desired scores for each sessions

    return stim_sheet, score_sheet


def plot_ziegler_result(result: str):
    """

    Input:
        - result: "score" or "times"
    """

    sub_path = find_folders.get_patterned_dbs_project_path(folder="figures", sub="084")

    loaded_data = load_ziegler(
        sub="084", stimulation="continuous", sheet=f"ziegler_{result}"
    )
    loaded_data = loaded_data[1]

    # Filter the DataFrame to keep only rows where interval == "total_time"
    df_filtered = loaded_data[loaded_data["interval"] == "total_time"]

    # Define the conditions for the three x-axis categories
    conditions = [
        (df_filtered["stimulation"] == "StimOnA") & (df_filtered["run"] == 1),
        (df_filtered["stimulation"] == "StimOffA") & (df_filtered["run"] == 1),
        (df_filtered["stimulation"] == "StimOffA") & (df_filtered["run"] == 2),
    ]

    # Create labels for the x-axis
    x_labels = ["Stim On", "Stim Off - 30 min", "Stim Off - 100 min"]

    # Assign a new categorical variable based on conditions
    df_filtered["x_category"] = None
    for i, condition in enumerate(conditions):
        df_filtered.loc[condition, "x_category"] = x_labels[i]

    # Drop any NaN values (if any rows don't match the conditions)
    df_filtered = df_filtered.dropna(subset=["x_category"])

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # colors = plt.cm.Set2(np.linspace(0, 1, len(conditions)))

    sns.barplot(
        data=df_filtered,
        x="x_category",
        y="time",
        hue="task_level",
        palette="muted",
        errorbar="se",  # Shows standard error
        ax=ax,
        order=x_labels,  # Ensure correct order of x-axis
    )

    # Styling
    ax.set_xlabel("Stimulation Condition")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Task Performance Time Across Conditions")
    ax.legend(title="Task Level")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Save the figure (adjust filename and format as needed)
    # fig.savefig("task_performance_plot.png", dpi=300, bbox_inches="tight")

    # Show the plot

    fig.tight_layout()

    # Save figure
    io.save_fig_png_and_svg(
        path=sub_path,
        filename=f"barplot_ziegler_{result}_sub_084",
        figure=fig,
    )
