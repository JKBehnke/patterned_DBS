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
    "1A_1B": "02",
    "1_2": "03",
}  # add more stimulation contacts if needed, this is only for ring stimulation so far


def structure_indef_str(sub: str, medication: str, stimulation: str, run: str):
    """
    Main function to process indefinite streaming data
    Input:
        - sub: str, e.g. "084"
        - medication: str, e.g. "on"
        - stimulation: str, e.g. "burst", "continuous"
        - run: str, e.g. "1"

    1. Load the data from: onedrive burst dbs > data > sub-xxx > Perceive_data
    2. Get the info from the data
    3. Structure the data
    4. Only keep 3 minutes of the data, if the duration is shorter, keep the whole recording

    """

    # load the data
    loaded_data = io.load_perceive_file(
        sub=sub,
        modality="indefinite_streaming",
        task="rest",
        medication=medication,
        stimulation=f"StimOff{STIMULATION_DICT[stimulation]}",
        run=run,
    )
    # if the data is None, return None
    if loaded_data is None:
        return None

    data = loaded_data["mne_data"]

    # get info from data
    fs = data.info["sfreq"]
    ch_names = data.info["ch_names"]  # e.g. 'LFP_Stn_L_03

    n_channels = len(ch_names)
    n_samples = int(data.n_times)

    duration = float(n_samples / fs)

    # get the data
    raw_data = data.get_data()  # shape (n_channels, n_times)
    ecg_cleaned_data = loaded_data["ecg_cleaned_data"]  # shape (n_channels, n_times)

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

    # check if ecg_cleaned_data is None, if so, create a DataFrame with zeros
    if ecg_cleaned_data is None:
        ecg_cleaned_data_df = None

    else:
        ecg_cleaned_data_df = pd.DataFrame(ecg_cleaned_data.T, columns=ch_names)
        ecg_cleaned_data_df = ecg_cleaned_data_df.iloc[:samples_to_keep]

    data_info["n_samples_kept"] = samples_to_keep
    data_info["duration_kept"] = samples_to_keep / fs

    return {
        "data_info": data_info,
        "raw_data": raw_data_df,
        "ecg_cleaned_data": ecg_cleaned_data_df,
    }


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
    picked_channels_ecg_clean_df = pd.DataFrame()

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

    if data is None:
        return None

    raw_data = data["raw_data"]
    data_info = data["data_info"]
    ecg_cleaned_data = data["ecg_cleaned_data"]

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

    if ecg_cleaned_data is None:
        picked_channels_ecg_clean_df = None

    else:
        picked_channels_ecg_clean_df[f"Left_{left_chan}"] = ecg_cleaned_data[
            f"LFP_Stn_L_{left_chan}"
        ]
        picked_channels_ecg_clean_df[f"Right_{right_chan}"] = ecg_cleaned_data[
            f"LFP_Stn_R_{right_chan}"
        ]

    return {
        "raw_data": picked_channels_df,
        "data_info": data_info,
        "ecg_cleaned_data": picked_channels_ecg_clean_df,
    }


def plot_time_series(
    sub: str,
    medication: str,
    stimulation: str,
    run: str,
    channels: str,
    raw_or_ecg_cleaned: str,
):
    """
    Plot the raw time series of the picked channels
    Input:
        - sub: str, e.g. "084"
        - medication: str, e.g. "on"
        - stimulation: str, e.g. "burst"
        - run: str, e.g. "1"
        - channels: str, "all" or "around_stim"
        - raw_or_ecg_cleaned: str, "raw" or "ecg_cleaned"
    """

    sub_path = os.path.join(GROUP_FIGURES_PATH, f"sub-{sub}")

    if channels == "all":
        data = structure_indef_str(
            sub=sub, medication=medication, stimulation=stimulation, run=run
        )
        figsize = (30, 40)  # 15, 24
    elif channels == "around_stim":
        data = pick_channel(
            sub=sub, medication=medication, stimulation=stimulation, run=run
        )
        figsize = (30, 40)  # 20, 10

    if data is None:
        return None

    data_info = data["data_info"]

    if raw_or_ecg_cleaned == "raw":
        data = data["raw_data"]

    elif raw_or_ecg_cleaned == "ecg_cleaned":
        data = data["ecg_cleaned_data"]

    if data is None:
        print("No data available")

    else:
        n_chans = data.shape[1]

        # add a column to raw_data with the time
        data["time"] = np.arange(
            0, data_info["duration_kept"].values[0], 1 / data_info["fs"].values[0]
        )

        # plot the raw time series
        # fig = data.plot(subplots=True, figsize=(20, 10), x="time", lw=0.5)

        fig, axes = plt.subplots(n_chans, 1, figsize=figsize)
        plt.setp(axes.flat, xlabel="Time [s]", ylabel="")

        plt.suptitle(
            f"sub-{sub}, med-{medication}, stim-OFF-{stimulation}, run-{run}\nDuration: {data_info['duration_kept'].values[0]} seconds"
        )
        # plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)

        for i, column in enumerate(data.columns):
            if column == "time":
                continue

            axes[i].plot(data["time"], data[column], lw=0.3)  # 0.5
            axes[i].set_ylabel(column)

        # save the figure
        io.save_fig_png_and_svg(
            path=sub_path,
            filename=f"sub-{sub}_med-{medication}_stim-OFF-{stimulation}_run-{run}_{channels}-channels_{raw_or_ecg_cleaned}_time_series_IS",
            figure=fig,
        )


def figure_layout_time_frequency():
    """ """
    ########################### Figure Layout ###########################
    # set layout for figures: using the object-oriented interface

    cols = ["0-min", "30-min", "60-min"]
    rows = ["c-DBS", "burst-DBS"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.setp(axes.flat, xlabel="Time [sec]", ylabel="Frequency [Hz]")

    pad = 5  # in points

    for ax, col in zip(axes[0], cols):
        ax.annotate(
            col,
            xy=(0.5, 1),
            xytext=(0, pad),
            xycoords="axes fraction",
            textcoords="offset points",
            size="large",
            ha="center",
            va="baseline",
        )

    for ax, row in zip(axes[:, 0], rows):
        ax.annotate(
            row,
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - pad, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
        )

    fig.tight_layout()
    # tight_layout doesn't take these labels into account. We'll need
    # to make some room. These numbers are are manually tweaked.
    # You could automatically calculate them, but it's a pain.
    fig.subplots_adjust(left=0.15, top=0.95)
    # fig.suptitle(f"sub-075 3MFU pilot")

    return fig, axes


# def time_frequency(sub: str, medication: str, stimulation: str, run: str):


def plot_time_frequency(sub: str, medication: str, raw_or_ecg_cleaned: str):
    """
    Plot the time-frequency plots of the picked channels
    Input:
        - sub: str, e.g. "084"
        - medication: str, e.g. "on"
    """

    sub_path = os.path.join(GROUP_FIGURES_PATH, f"sub-{sub}")

    for hem in ["Left", "Right"]:

        fig, axes = figure_layout_time_frequency()

        for row, stim in enumerate(["continuous", "burst"]):

            for col, run in enumerate(["1", "2", "3"]):
                # check if file exists, if not continue
                try:
                    data = pick_channel(
                        sub=sub, medication=medication, stimulation=stim, run=run
                    )
                except FileNotFoundError:
                    continue

                data = pick_channel(
                    sub=sub, medication=medication, stimulation=stim, run=run
                )

                if data is None:
                    continue

                data_info = data["data_info"]

                if raw_or_ecg_cleaned == "raw":
                    time_series = data["raw_data"]

                elif raw_or_ecg_cleaned == "ecg_cleaned":
                    time_series = data["ecg_cleaned_data"]

                if time_series is None:
                    print("No data available")
                    continue

                # get the data
                fs = data_info["fs"].values[0]

                # get the column that contains the correct hemisphere
                channel = time_series.columns[time_series.columns.str.contains(hem)][
                    0
                ]  # channel e.g. "Left_02"

                # each column contains the data from run 1, 2 or 3
                hem_time_series = time_series[channel].values

                # compute the spectrogram
                axes[row, col].specgram(
                    x=hem_time_series,
                    Fs=fs,
                    noverlap=0,
                    cmap="viridis",
                    vmin=-25,
                    vmax=10,
                )

        # save the figure
        io.save_fig_png_and_svg(
            path=sub_path,
            filename=f"sub-{sub}_med-{medication}_stim-OFF_time_frequency_IS_{hem}_{raw_or_ecg_cleaned}",
            figure=fig,
        )
        plt.close(fig)

    return fig
