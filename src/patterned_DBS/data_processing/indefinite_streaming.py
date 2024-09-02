""" LFP processing of indefinite streaming data """

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

PICK_CHANNEL_AROUND_STIM = {
    "0": "02",
    "1": "02",
    "2": "13",
    "3": "13",
}  # add more stimulation contacts if needed, this is only for ring stimulation so far


def structure_indef_str(sub: str, medication: str, stimulation: str, run: str):
    """
    Main function to process indefinite streaming data
    Input:
        - sub: str, e.g. "084"
        - medication: str, e.g. "on"
        - stimulation: str, e.g. "burst"
        - run: str, e.g. "1"

    1. Load the data from: onedrive burst dbs > data > sub-xxx > Perceive_data
    2. Get the info from the data
    3. Structure the data
    4. Only keep 3 minutes of the data, if the duration is shorter, keep the whole recording

    """

    # load the data
    data = io.load_perceive_file(
        sub=sub,
        modality="indefinite_streaming",
        task="rest",
        medication=medication,
        stimulation=f"StimOff{STIMULATION_DICT[stimulation]}",
        run=run,
    )

    data = data["data"]

    # get info from data
    fs = data.info["sfreq"]
    ch_names = data.info["ch_names"]  # e.g. 'LFP_Stn_L_03

    n_channels = len(ch_names)
    n_samples = int(data.n_times)

    duration = float(n_samples / fs)

    # get the data
    raw_data = data.get_data()  # shape (n_channels, n_times)

    # structure the data info
    data_info = {
        "sub": [sub],
        "medication": [medication],
        "stimulation": [stimulation],
        "run": [run],
        "fs": [fs],
        "n_channels": [n_channels],
        "n_samples_original": [n_samples],
        "duration_original": [duration],
    }
    data_info = pd.DataFrame(data_info)

    # structure the data
    raw_data_df = pd.DataFrame(raw_data.T, columns=ch_names)

    # only keep 3-minutes of the data if the duration is long enough
    samples_to_keep = 250 * 180  # 3 minutes

    if duration < 180:
        print(
            f"sub-{sub}, med-{medication}, stim-{stimulation}, run-{run}:\nRecording duration is below 180 seconds: {duration} seconds"
        )
        samples_to_keep = n_samples

    raw_data_df = raw_data_df.iloc[:samples_to_keep]
    data_info["n_samples_kept"] = samples_to_keep
    data_info["duration_kept"] = samples_to_keep / fs

    return {"data_info": data_info, "raw_data": raw_data_df}


def pick_channel(sub: str, medication: str, stimulation: str, run: str):
    """
    Find the stimulation contacts from the metadata and pick the corresponding channels around them

    1. Load the metadata excel file
    2. Get the stimulation contacts for Left and Right hemispheres
    3. Pick the channels around the stimulation contacts (e.g. 13 or 02)
    4. Save the picked channels

    Input:
        - sub: str, e.g. "084"
        - medication: str, e.g. "on"
        - stimulation: str, e.g. "burst"
        - run: str, e.g. "1"
    """

    picked_channels_df = pd.DataFrame()

    metadata = io.load_metadata_excel(
        sub=sub,
        sheet_name="stimulation_parameters",
    )

    # get the stimulation contacts
    left_stim_contact = metadata.loc[metadata["hemisphere"] == "left"]
    left_stim_contact = left_stim_contact["active_contacts"].values[0]  # e.g. 1 or 2

    right_stim_contact = metadata.loc[metadata["hemisphere"] == "right"]
    right_stim_contact = right_stim_contact["active_contacts"].values[0]

    # get the data
    data = structure_indef_str(
        sub=sub, medication=medication, stimulation=stimulation, run=run
    )
    raw_data = data["raw_data"]
    data_info = data["data_info"]

    # make sure that stim contacts are in the PICK_CHANNEL_AROUND_STIM dict
    if str(left_stim_contact) not in PICK_CHANNEL_AROUND_STIM.keys():
        raise ValueError(
            f"Stimulation contact {left_stim_contact} not in PICK_CHANNEL_AROUND_STIM dict"
        )

    # pick the channels
    left_chan = PICK_CHANNEL_AROUND_STIM[str(left_stim_contact)]
    right_chan = PICK_CHANNEL_AROUND_STIM[str(right_stim_contact)]

    picked_channels_df[f"Left_{left_chan}"] = raw_data[f"LFP_Stn_L_{left_chan}"]
    picked_channels_df[f"Right_{right_chan}"] = raw_data[f"LFP_Stn_R_{right_chan}"]

    return {"raw_data": picked_channels_df, "data_info": data_info}


def plot_raw_time_series(
    sub: str, medication: str, stimulation: str, run: str, channels: str
):
    """
    Plot the raw time series of the picked channels
    Input:
        - sub: str, e.g. "084"
        - medication: str, e.g. "on"
        - stimulation: str, e.g. "burst"
        - run: str, e.g. "1"
        - channels: str, "all" or "around_stim"
    """

    sub_path = os.path.join(GROUP_FIGURES_PATH, f"sub-{sub}")

    if channels == "all":
        data = structure_indef_str(
            sub=sub, medication=medication, stimulation=stimulation, run=run
        )
    elif channels == "around_stim":
        data = pick_channel(
            sub=sub, medication=medication, stimulation=stimulation, run=run
        )
    data_info = data["data_info"]
    raw_data = data["raw_data"]

    # add a column to raw_data with the time
    raw_data["time"] = np.arange(
        0, data_info["duration_kept"].values[0], 1 / data_info["fs"].values[0]
    )

    # plot the raw time series
    fig = raw_data.plot(subplots=True, figsize=(20, 10), x="time")
    plt.suptitle(
        f"sub-{sub}, med-{medication}, stim-OFF-{stimulation}, run-{run}\nDuration: {data_info['duration_kept'].values[0]} seconds"
    )

    plt.xlabel("Time [s]")

    plt.tight_layout()
