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


def load_gait_data(
    sub: str,
    stimulation: str,
    sheet: str,
):
    """
    Input:
        - sub: "084" only works for case subject
        - stimulation: "burst" or "continuous"
        - sheet: "ziegler_score", "ziegler_times", "walk_10m"

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


def format_time(seconds):
    """Convert seconds to MM:SS format."""
    total_seconds = round(seconds)  # Round to nearest whole number
    mm = total_seconds // 60  # Get minutes
    ss = total_seconds % 60  # Get seconds
    return f"{mm}:{ss:02d}"  # Ensure two-digit seconds format
    # mm = int(seconds // 60)
    # ss = int(seconds % 60)
    # return f"{mm}:{ss:02d}"


def plot_ziegler_result(result: str):
    """

    Input:
        - result: "score" or "times"
    """

    sub_path = find_folders.get_patterned_dbs_project_path(folder="figures", sub="084")

    loaded_data = load_gait_data(
        sub="084", stimulation="continuous", sheet=f"ziegler_{result}"
    )
    loaded_data = loaded_data[1]

    # Filter the DataFrame to keep only rows where interval == "total_time"
    df_filtered = loaded_data[loaded_data["interval"] == "total_time"]

    # text_data = df_filtered["time"]

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

    barplot = sns.barplot(
        data=df_filtered,
        x="x_category",
        y="time",
        hue="task_level",
        palette="muted",
        errorbar="se",  # Shows standard error
        ax=ax,
        order=x_labels,  # Ensure correct order of x-axis
    )

    # Convert time back to MM:SS for annotation
    df_filtered["time_display"] = df_filtered["time"].apply(format_time)

    # # Add text annotations on top of each bar
    # for bar, (x_category, task_level, time_display) in zip(
    #     barplot.patches,
    #     df_filtered[["x_category", "task_level", "time_display"]].values,
    # ):
    #     height = bar.get_height()  # Get bar height
    #     ax.text(
    #         bar.get_x() + bar.get_width() / 2,  # X position (centered)
    #         height + 1,  # Slightly above the bar
    #         time_display,  # MM:SS format
    #         ha="center",  # Center-align text
    #         va="bottom",
    #         fontsize=12,
    #         fontweight="bold",
    #         color="black",
    #     )

    # Add text annotations on top of each bar
    for bar in ax.patches:
        height = bar.get_height()  # Get the height of each bar (which is in seconds)
        time_display = format_time(height)  # Convert seconds to MM:SS

        ax.text(
            bar.get_x() + bar.get_width() / 2,  # Centered X position
            height + 2,  # Y position slightly above the bar
            time_display,  # MM:SS format
            ha="center",  # Center-align text
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="black",
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

    return ax, df_filtered


# def plot_10m_walk_result():
#     """ """

#     sub_path = find_folders.get_patterned_dbs_project_path(folder="figures", sub="084")

#     loaded_data = load_gait_data(sub="084", stimulation="continuous", sheet=f"walk_10m")
#     loaded_data = loaded_data[1]

#     text_data = loaded_data["time"]

#     # Define the conditions for the three x-axis categories
#     conditions = [
#         (loaded_data["stimulation"] == "StimOnB") & (loaded_data["run"] == 1),
#         (loaded_data["stimulation"] == "StimOnA") & (loaded_data["run"] == 1),
#         (loaded_data["stimulation"] == "StimOffA") & (loaded_data["run"] == 1),
#         (loaded_data["stimulation"] == "StimOffA") & (loaded_data["run"] == 2),
#     ]

#     # Create labels for the x-axis
#     x_labels = [
#         "burst DBS Stim ON",
#         "cDBS Stim On",
#         "cDBS Stim Off - 30 min",
#         "cDBS Stim Off - 100 min",
#     ]

#     # Assign a new categorical variable based on conditions
#     loaded_data["x_category"] = None
#     for i, condition in enumerate(conditions):
#         loaded_data.loc[condition, "x_category"] = x_labels[i]

#     # Drop any NaN values (if any rows don't match the conditions)
#     loaded_data = loaded_data.dropna(subset=["x_category"])

#     # Create figure and axis
#     fig, ax = plt.subplots(figsize=(8, 6))

#     # colors = plt.cm.Set2(np.linspace(0, 1, len(conditions)))

#     barplot = sns.barplot(
#         data=loaded_data,
#         x="x_category",
#         y="time",
#         ax=ax,
#         order=x_labels,  # Ensure correct order of x-axis
#         palette="muted",
#     )

#     # Add text annotations on top of each bar
#     for i, bar in enumerate(barplot.patches):
#         height = bar.get_height()  # Get bar height
#         ax.text(
#             bar.get_x() + bar.get_width() / 2,  # X position (centered)
#             height + 0.05,  # Y position (a little above the bar)
#             text_data.iloc[i],  # MM:SS format
#             ha="center",  # Center-align text
#             va="bottom",
#             fontsize=12,
#             fontweight="bold",
#             color="black",
#         )

#     # Styling
#     ax.set_xlabel("Stimulation Condition")
#     ax.set_ylabel("Time (seconds)")
#     ax.set_title("10 m Walk Across Stimulation Conditions")

#     # ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
#     ax.tick_params(axis="x", rotation=30)
#     ax.grid(axis="y", linestyle="--", alpha=0.6)

#     # Show the plot

#     fig.tight_layout()

#     # Save figure
#     io.save_fig_png_and_svg(
#         path=sub_path,
#         filename=f"barplot_10m_walk_sub_084",
#         figure=fig,
#     )


# Convert "MM:SS:milliseconds" to minutes
def time_to_minutes(time_str):
    """Convert MM:SS:milliseconds format to minutes (float)."""
    mm, ss, ms = time_str.split(":")  # Split into minutes, seconds, milliseconds
    return int(mm) + int(ss) / 60 + int(ms) / 60000  # Convert to total minutes


def plot_10m_walk():
    """ """

    sub_path = find_folders.get_patterned_dbs_project_path(folder="figures", sub="084")

    # Load data (assuming already loaded as DataFrame)
    loaded_data = load_gait_data(sub="084", stimulation="continuous", sheet="walk_10m")[
        1
    ]

    text_data = loaded_data["time"]

    # Apply conversion to time column
    loaded_data["time"] = loaded_data["time"].apply(time_to_minutes)

    # Define the conditions for x-axis categories
    conditions = [
        ((loaded_data["stimulation"] == "StimOnB") & (loaded_data["run"] == 1)),  # x1
        ((loaded_data["stimulation"] == "StimOnA") & (loaded_data["run"] == 1)),  # x2
        ((loaded_data["stimulation"] == "StimOffA") & (loaded_data["run"] == 1)),  # x3
        ((loaded_data["stimulation"] == "StimOffA") & (loaded_data["run"] == 2)),  # x4
    ]

    x_labels = ["Stim On B", "Stim On A", "Stim Off - 30 min", "Stim Off - 100 min"]

    # Assign categories
    loaded_data["x_category"] = pd.NA
    for i, condition in enumerate(conditions):
        loaded_data.loc[condition, "x_category"] = x_labels[i]

    # Drop rows without a category
    loaded_data = loaded_data.dropna(subset=["x_category"])

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot bar chart
    barplot = sns.barplot(
        data=loaded_data,
        x="x_category",
        y="time",
        ax=ax,
        order=x_labels,
        palette="muted",
    )

    # Add text annotations on top of each bar
    for i, bar in enumerate(barplot.patches):
        height = bar.get_height()  # Get bar height
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # X position (centered)
            height + 0.05,  # Y position (a little above the bar)
            text_data.iloc[i],  # MM:SS format
            ha="center",  # Center-align text
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="black",
        )

    # Styling
    ax.set_xlabel("Stimulation Condition")
    ax.set_ylabel("Time (minutes)")
    ax.set_title("10m Walk Across Stimulation Conditions")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Save figure (optional)
    io.save_fig_png_and_svg(path=sub_path, filename="barplot_10m_walk", figure=fig)
