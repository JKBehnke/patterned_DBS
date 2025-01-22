""" GAIT analysis cDBS vs. burstDBS"""

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


def load_gait_task_results(
    sub: str,
    stimulation: str,
    task: str,
):
    """
    Input:
        - sub: "089"
        - stimulation: "burst" or "continuous"
        - task: "10m_walk", "timed_up_and_go"

    """

    score_sheet = io.load_metadata_excel(sub=sub, sheet_name="gait_tasks")

    # only keep stimulation burst or continuous
    stim_sheet = score_sheet[
        score_sheet["stimulation"].str.contains(
            f"StimOn{STIMULATION_DICT[stimulation]}|StimOff{STIMULATION_DICT[stimulation]}",
            na=False,
        )
    ]

    # only keep the desired subscore
    subscore_data = stim_sheet.loc[stim_sheet["task"] == task]

    # get the desired scores for each sessions

    return subscore_data, stim_sheet, score_sheet


def group_gait_task_results(sub_list: list, stimulation: str, task: str):
    """
    Load the gait tasks for a list of subjects and stimulation type
    """

    # Concatenate the dataframes
    for sub in sub_list:
        subscore_data, stim_sheet, score_sheet = load_gait_task_results(
            sub=sub, stimulation=stimulation, task=task
        )
        if sub == sub_list[0]:
            all_scores = subscore_data
            all_scores["subject"] = sub
        else:
            subscore_data["subject"] = sub
            all_scores = pd.concat([all_scores, subscore_data])

    return all_scores


# def convert_time_to_float(time_str):
#     """
#     Convert a time string in the format 'HH:MM:SS' or 'HH:MM:SS min' to float (minutes).

#     Parameters:
#         time_str (str): Time in the format "HH:MM:SS" or "HH:MM:SS min".

#     Returns:
#         float: Time converted to minutes.
#     """
#     # Remove 'min' if it exists
#     time_str = time_str.replace(" min", "")

#     # Split the time string into hours, minutes, and seconds
#     try:
#         h, m, s = map(int, time_str.split(":"))
#         total_minutes = h * 60 + m + s / 60  # Convert to float minutes
#         return round(total_minutes, 2)  # Rounding to 2 decimal places
#     except ValueError:
#         print(f"Error: Incorrect time format: {time_str}")
#         return None  # Handle potential incorrect formats


def convert_time_to_seconds(time_str):
    """
    Convert a time string in the format 'Min:Sec:Milliseconds' or 'Min:Sec:Milliseconds min' to float (seconds).

    Parameters:
        time_str (str): Time in the format "Min:Sec:Milliseconds" or "Min:Sec:Milliseconds min".

    Returns:
        float: Time converted to seconds.
    """
    # Remove 'min' if it exists
    time_str = time_str.replace("min", "").strip()

    # Split the time string into minutes, seconds, and milliseconds
    try:
        m, s, ms = map(int, time_str.split(":"))
        total_seconds = (m * 60) + s + (ms / 1000)  # Convert to float seconds
        return round(total_seconds, 3)  # Round to 3 decimal places for milliseconds
    except ValueError:
        print(f"Error: Incorrect time format: {time_str}")
        return None  # Handle potential incorrect formats


def extract_data_for_bar_and_line_plot(task: str, subscore_column: str):
    """

    Extract the data for the bar and line plots for the group analysis

    Input:
        - task: "10m_walk", "timed_up_and_go"
        - subscore_column: "time" or "steps"

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
        subscore_data, stim_sheet, score_sheet = load_gait_task_results(
            sub=sub, stimulation="continuous", task=task
        )

        StimOnA_data = subscore_data.loc[subscore_data["stimulation"] == "StimOnA"]
        if subscore_column == "time":
            score = convert_time_to_seconds(StimOnA_data[subscore_column].values[0])

        elif subscore_column == "steps":
            score = StimOnA_data[subscore_column].values[0]

        structured_data["score"].append(score)

    # then extract burst score data per subject
    for sub in sub_list:
        subscore_data, stim_sheet, score_sheet = load_gait_task_results(
            sub=sub, stimulation="burst", task=task
        )

        StimOnB_data = subscore_data.loc[subscore_data["stimulation"] == "StimOnB"]
        if subscore_column == "time":
            score = convert_time_to_seconds(StimOnB_data[subscore_column].values[0])

        elif subscore_column == "steps":
            score = StimOnB_data[subscore_column].values[0]

        structured_data["score"].append(score)

    # then extract StimOFF score data per subject: last Stim OFF score per cDSB OFF
    for sub in sub_list:
        subscore_data, stim_sheet, score_sheet = load_gait_task_results(
            sub=sub, stimulation="continuous", task=task
        )

        # check if StimOFFA_run-3 exists
        if 3 in subscore_data["run"].values:
            subscore_data = subscore_data.loc[subscore_data["run"] == 3]
            StimOFF_scores["StimOFFA_run"].append("run-3")

        elif 2 in subscore_data["run"].values:
            subscore_data = subscore_data.loc[subscore_data["run"] == 2]
            StimOFF_scores["StimOFFA_run"].append("run-2")

        elif 1 in subscore_data["run"].values:
            subscore_data = subscore_data.loc[subscore_data["run"] == 1]
            StimOFF_scores["StimOFFA_run"].append("run-1")

        else:
            print(
                "StimOffA_run-3, StimOffA_run-2, StimOffA_run-1 not found in the data"
            )
            StimOFF_scores["StimOFFA_run"].extend("not found")

        if subscore_column == "time":
            score = convert_time_to_seconds(subscore_data[subscore_column].values[0])

        elif subscore_column == "steps":
            score = subscore_data[subscore_column].values[0]

        structured_data["score"].append(score)

    return pd.DataFrame(structured_data), pd.DataFrame(StimOFF_scores)
