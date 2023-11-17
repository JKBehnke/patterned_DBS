""" Patterned DBS Pilot"""


import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
import scipy
from scipy import signal
from scipy.signal import butter, filtfilt, freqz, hann, spectrogram
from scipy.integrate import simps


# internal Imports
from ..utils import find_folders as find_folders
from ..utils import io as io
from ..utils import lfp_preprocessing as lfp_preprocessing

HEMISPHERES = ["Right", "Left"]
SAMPLING_FREQ = 250

FREQUENCY_BANDS = {
    "beta": [13, 36],
    "low_beta": [13, 21],
    "high_beta": [21, 36],
}

GROUP_RESULTS_PATH = find_folders.get_patterned_dbs_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_patterned_dbs_project_path(folder="GroupFigures")

sub_075_pilot_streaming_dict = {
    "0": ["pre", "burstDBS", "1min", "Left"],
    "1": ["pre", "burstDBS", "1min", "Right"],
    "2": ["post", "burstDBS", "1min", "Left"],
    "3": ["post", "burstDBS", "1min", "Right"],
    "4": ["pre", "cDBS", "1min", "Left"],
    "5": ["pre", "cDBS", "1min", "Right"],
    "6": ["post", "cDBS", "1min", "Left"],
    "7": ["post", "cDBS", "1min", "Right"],
    "8": ["pre", "burstDBS", "5min", "Left"],
    "9": ["pre", "burstDBS", "5min", "Right"],
    "10": ["post", "burstDBS", "5min", "Left"],
    "11": ["post", "burstDBS", "5min", "Right"],
    "12": ["pre", "cDBS", "5min", "Left"],
    "13": ["pre", "cDBS", "5min", "Right"],
    "16": ["post", "cDBS", "5min", "Left"],
    "17": ["post", "cDBS", "5min", "Right"],
    "18": ["pre", "burstDBS", "30min", "Left"],
    "19": ["pre", "burstDBS", "30min", "Right"],
    "22": ["post", "burstDBS", "30min", "Left"],  # 8.45 min
    "23": ["post", "burstDBS", "30min", "Right"],  # 8.45 min
}


def write_json_streaming_info(sub: str, incl_session: list, run: str):
    """ """
    streaming_info = pd.DataFrame()
    raw_objects = {}

    load_json = io.load_source_json_patterned_dbs(
        sub=sub, incl_session=incl_session, run=run
    )

    # number of BrainSense Streamings
    # n_streamings = len(load_json["BrainSenseTimeDomain"])

    # get info of each recording
    for streaming in list(sub_075_pilot_streaming_dict.keys()):
        time_domain_data = load_json["BrainSenseTimeDomain"][int(streaming)][
            "TimeDomainData"
        ]
        channel = load_json["BrainSenseTimeDomain"][int(streaming)]["Channel"]

        pre_or_post = sub_075_pilot_streaming_dict[streaming][0]
        burstDBS_or_cDBS = sub_075_pilot_streaming_dict[streaming][1]
        DBS_duration = sub_075_pilot_streaming_dict[streaming][2]
        hemisphere = sub_075_pilot_streaming_dict[streaming][3]

        # transform to mne
        units = ["µVolt"]
        scale = np.array([1e-6 if u == "µVolt" else 1 for u in units])

        info = mne.create_info(ch_names=[channel], sfreq=250, ch_types="dbs")
        raw = mne.io.RawArray(time_domain_data * np.expand_dims(scale, axis=1), info)

        # save raw
        raw_objects[streaming] = raw

        # get more info
        time_domain_dataframe = raw.to_data_frame()
        rec_duration = raw.tmax

        # save into dataframe
        streaming_data = {
            "streaming_index": [streaming],
            "original_time_domain_data": [time_domain_data],
            "channel": [channel],
            "time_domain_dataframe": [time_domain_dataframe],
            "rec_duration": [rec_duration],
            "pre_or_post": [pre_or_post],
            "burstDBS_or_cDBS": [burstDBS_or_cDBS],
            "DBS_duration": [DBS_duration],
            "hemisphere": [hemisphere],
        }

        streaming_data_single = pd.DataFrame(streaming_data)
        streaming_info = pd.concat(
            [streaming_info, streaming_data_single], ignore_index=True
        )

    # save data as pickle
    io.save_result_dataframe_as_pickle(
        data=streaming_info, filename="streaming_info_patterned_pilot_sub-075"
    )
    io.save_result_dataframe_as_pickle(
        data=raw_objects, filename="raw_objects_patterned_pilot_sub-075"
    )

    return {
        "streaming_info": streaming_info,
        "raw": raw_objects,
    }


def figure_layout_time_frequency():
    """ """
    ########################### Figure Layout ###########################
    # set layout for figures: using the object-oriented interface

    cols = ["pre-DBS", "post-DBS"]
    rows = ["c-DBS", "burst-DBS"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
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


def plot_time_frequency(dbs_duration: str):
    """
    Input:
        - dbs_duration: str, e.g. "1min", "5min", "30min"


    """
    ########################### Load data ###########################

    cDBS_or_burst_DBS = ["cDBS", "burstDBS"]

    streaming_data = io.load_pickle_files(
        filename="streaming_info_patterned_pilot_sub-075"
    )

    # select for dbs_duration
    streaming_data = streaming_data[streaming_data["DBS_duration"] == dbs_duration]

    for hem in HEMISPHERES:
        # figure layout
        fig, axes = figure_layout_time_frequency()

        # select for hemisphere
        hem_data = streaming_data[streaming_data["hemisphere"] == hem]

        if dbs_duration == "30min":
            # select for cDBS and burstDBS and pre and post DBS

            DBS_data = hem_data[hem_data["burstDBS_or_cDBS"] == "burstDBS"]
            pre_DBS = DBS_data[DBS_data["pre_or_post"] == "pre"]
            post_DBS = DBS_data[DBS_data["pre_or_post"] == "post"]

            # get data
            pre_DBS_data = pre_DBS["original_time_domain_data"].values[0]
            post_DBS_data = post_DBS["original_time_domain_data"].values[0]

            # band-pass filter
            pre_DBS_filtered = lfp_preprocessing.band_pass_filter_percept(
                fs=SAMPLING_FREQ, signal=pre_DBS_data
            )
            post_DBS_filtered = lfp_preprocessing.band_pass_filter_percept(
                fs=SAMPLING_FREQ, signal=post_DBS_data
            )

            # plot TF
            noverlap = 0

            axes[1, 0].specgram(
                x=pre_DBS_filtered,
                Fs=SAMPLING_FREQ,
                noverlap=noverlap,
                cmap="viridis",
                vmin=-25,
                vmax=10,
            )
            axes[1, 1].specgram(
                x=post_DBS_filtered,
                Fs=SAMPLING_FREQ,
                noverlap=noverlap,
                cmap="viridis",
                vmin=-25,
                vmax=10,
            )

            axes[1, 0].grid(False)
            axes[1, 1].grid(False)

            fig.suptitle(
                f"sub-075 3MFU pilot {hem} hemisphere {dbs_duration} DBS duration"
            )
            io.save_fig_png_and_svg(
                path=GROUP_FIGURES_PATH,
                filename=f"time_frequency_plot_sub-075_{hem}_3MFU_pilot_{dbs_duration}",
                figure=fig,
            )

        else:
            for dbs, dbs_type in enumerate(cDBS_or_burst_DBS):
                # select for cDBS and burstDBS and pre and post DBS

                DBS_data = hem_data[hem_data["burstDBS_or_cDBS"] == dbs_type]
                pre_DBS = DBS_data[DBS_data["pre_or_post"] == "pre"]
                post_DBS = DBS_data[DBS_data["pre_or_post"] == "post"]

                # get data
                pre_DBS_data = pre_DBS["original_time_domain_data"].values[0]
                post_DBS_data = post_DBS["original_time_domain_data"].values[0]

                # band-pass filter
                pre_DBS_filtered = lfp_preprocessing.band_pass_filter_percept(
                    fs=SAMPLING_FREQ, signal=pre_DBS_data
                )
                post_DBS_filtered = lfp_preprocessing.band_pass_filter_percept(
                    fs=SAMPLING_FREQ, signal=post_DBS_data
                )

                # plot TF
                noverlap = 0

                axes[dbs, 0].specgram(
                    x=pre_DBS_filtered,
                    Fs=SAMPLING_FREQ,
                    noverlap=noverlap,
                    cmap="viridis",
                    vmin=-25,
                    vmax=10,
                )
                axes[dbs, 1].specgram(
                    x=post_DBS_filtered,
                    Fs=SAMPLING_FREQ,
                    noverlap=noverlap,
                    cmap="viridis",
                    vmin=-25,
                    vmax=10,
                )

                axes[dbs, 0].grid(False)
                axes[dbs, 1].grid(False)

            fig.suptitle(
                f"sub-075 3MFU pilot {hem} hemisphere {dbs_duration} DBS duration"
            )
            io.save_fig_png_and_svg(
                path=GROUP_FIGURES_PATH,
                filename=f"time_frequency_plot_sub-075_{hem}_3MFU_pilot_{dbs_duration}",
                figure=fig,
            )


def fourier_transform(time_domain_data: np.array):
    """
    - 1 second window length = 250 samples at sampling frequency 250 Hz
    - 50% overlap e.g. 2min pre-DBS baseline -> 239 x 0.5 seconds = 120 seconds
    """

    window_length = int(SAMPLING_FREQ)  # 1 second window length
    overlap = (
        window_length // 2
    )  # 50% overlap e.g. 2min pre-DBS baseline -> 239 x 0.5 seconds = 120 seconds

    # Calculate the short-time Fourier transform (STFT) using Hann window
    window = hann(window_length, sym=False)

    frequencies, times, Zxx = scipy.signal.spectrogram(
        time_domain_data,
        fs=SAMPLING_FREQ,
        window=window,
        noverlap=overlap,
        scaling="density",
        mode="psd",
        axis=0,
    )

    # Frequencies: 0-125 Hz (1 Hz resolution), Nyquist fs/2
    # times: len=161, 0, 0.75, 1.5 .... 120.75
    # Zxx: 126 arrays, each len=239

    # average PSD across duration of the recording
    average_Zxx = np.mean(Zxx, axis=1)
    std_Zxx = np.std(Zxx, axis=1)
    sem_Zxx = std_Zxx / np.sqrt(Zxx.shape[1])

    return frequencies, times, Zxx, average_Zxx, std_Zxx, sem_Zxx


def normalize_to_psd(to_sum: str, power_spectrum: np.array):
    """ """

    if to_sum == "40_to_90":
        # sum frequencies 40-90 Hz
        sum_40_90 = np.sum(power_spectrum[40:90])
        # normalize
        normalized_power_spectrum = power_spectrum / sum_40_90

    elif to_sum == "5_to_95":
        # sum frequencies 5-95 Hz
        sum_5_95 = np.sum(power_spectrum[5:95])
        # normalize
        normalized_power_spectrum = power_spectrum / sum_5_95

    return normalized_power_spectrum


def find_peaks(power_spectrum, frequencies):
    """ """

    peak_result_all = pd.DataFrame()

    # find all peaks in the power spectrum
    # peak_freq, peak_power = scipy.signal.find_peaks(
    #     x=power_spectrum, height=0.3 * max(power_spectrum), distance=1
    # )

    peak_freq, peak_power = scipy.signal.find_peaks(
        x=power_spectrum, height=0.05, distance=1
    )

    for freq in FREQUENCY_BANDS.keys():
        # check if peaks are found in each frequency range
        if (
            len(
                np.where(
                    (peak_freq >= FREQUENCY_BANDS[freq][0])
                    & (peak_freq <= FREQUENCY_BANDS[freq][1])
                )[0]
            )
            == 0
        ):
            print(f"No peak found in {freq} range")
            continue

        # get the peak frequencies and their indices in each frequency range
        mask = (peak_freq >= FREQUENCY_BANDS[freq][0]) & (
            peak_freq <= FREQUENCY_BANDS[freq][1]
        )

        # Use boolean indexing to extract values and indices
        peak_freq_within_range = peak_freq[mask]
        peak_power_within_range = peak_power["peak_heights"][mask]
        # peak_indices_within_range = np.where(mask)[0]

        # find the peak with maximal power + its index in the peak_power_array in that frequency range
        max_peak_power = np.max(peak_power_within_range)
        idx_max_peak_power = np.where(peak_power_within_range == max_peak_power)[0][0]

        # now get the frequency of the peak with maximal power
        max_peak_freq = peak_freq_within_range[idx_max_peak_power]

        # from this frequency take an array ± 3 Hz
        freq_range_around_max_peak = np.arange(max_peak_freq - 3, max_peak_freq + 4, 1)

        # get the power values within that freq range
        power_in_freq_range_around_max_peak = power_spectrum[
            freq_range_around_max_peak[0] : freq_range_around_max_peak[6] + 1
        ]

        # calculate the power average in that freq range
        power_average_in_freq_range_around_max_peak = np.mean(
            power_in_freq_range_around_max_peak
        )

        # calculate power area under the curve
        power_area_under_curve = simps(
            power_in_freq_range_around_max_peak, freq_range_around_max_peak
        )

        # save relevant data
        # save data
        peak_dict = {
            "power_spectrum": [power_spectrum],
            "frequencies": [frequencies],
            "freq_band": [freq],
            "max_peak_freq": [max_peak_freq],
            "max_peak_power": [max_peak_power],
            "freq_range_around_max_peak": [freq_range_around_max_peak],
            "power_in_freq_range_around_max_peak": [
                power_in_freq_range_around_max_peak
            ],
            "power_average_in_freq_range_around_max_peak": [
                power_average_in_freq_range_around_max_peak
            ],
            "power_area_under_curve": [power_area_under_curve],
        }

        peak_result_single = pd.DataFrame(peak_dict)
        peak_result_all = pd.concat(
            [peak_result_all, peak_result_single], ignore_index=True
        )

    return peak_result_all


def calculate_beta_baseline(
    DBS_duration: str,
    burstDBS_or_cDBS: str,
    filtered: str,
    hemisphere: str,
    pre_or_post: str,
    freq_average_or_peak: str,
):
    """
    Input:
        - DBS_duration: str, e.g. "1min", "5min", "30min"
        - cDBS_or_burst_DBS: str, e.g. "cDBS", "burstDBS"
        - filtered: str, e.g. "band-pass_5_95" or "unfiltered"
        - hemisphere: str, e.g. "Left", "Right"
        - freq_average_or_peak: str, e.g. "average", "peak"

    1) select for DBS_duration, cDBS or burstDBS, pre DBS
    2) get time domain data, only take the last 120 seconds of the original time domain data (30000 samples = 2 minutes)
    3) band-pass filter 5-95 Hz
    4) calculate PSD


    """

    DBS_beta_baseline = pd.DataFrame()

    if pre_or_post == "pre":
        ####################### pre-DBS baseline (2 min pre DBS) #######################
        streaming_data = io.load_pickle_files(
            filename="streaming_info_patterned_pilot_sub-075"
        )

        # select for DBS_duration
        streaming_data = streaming_data[streaming_data["DBS_duration"] == DBS_duration]

        # select for cDBS and burstDBS and pre DBS
        streaming_data = streaming_data[
            streaming_data["burstDBS_or_cDBS"] == burstDBS_or_cDBS
        ]
        streaming_data = streaming_data[streaming_data["pre_or_post"] == "pre"]

        hem_data = streaming_data[streaming_data["hemisphere"] == hemisphere]

        # get data
        time_domain_data = hem_data.original_time_domain_data.values[0]

        # only take the last 120 seconds of the original time domain data (30000 samples = 2 minutes)
        time_domain_data = np.array(time_domain_data[-30000:])

    elif pre_or_post == "post":
        ####################### post-DBS baseline (2-3 min post Stim OFF #######################
        # load the post_DBS baseline data = last 1 minute of the post_DBS time series (after DBS OFF starting 2min until 3min post turned OFF)
        time_domain_data = get_post_dbs_time_series(
            DBS_duration=DBS_duration,
            burstDBS_or_cDBS=burstDBS_or_cDBS,
            hemisphere=hemisphere,
        )
        time_domain_data = time_domain_data[
            1
        ]  # len 15000 samples = 1 minute (2-3 minutes after DBS OFF)

    if filtered == "band-pass_5_95":
        # band-pass filter 5-95 Hz
        time_domain_data = lfp_preprocessing.band_pass_filter_percept(
            fs=SAMPLING_FREQ, signal=time_domain_data
        )

    # calculate PSD
    frequencies, times, Zxx, average_Zxx, std_Zxx, sem_Zxx = fourier_transform(
        time_domain_data
    )

    # normalize PSD
    normalized_to_5_95 = normalize_to_psd(to_sum="5_to_95", power_spectrum=average_Zxx)
    normalized_to_40_90 = normalize_to_psd(
        to_sum="40_to_90", power_spectrum=average_Zxx
    )

    if freq_average_or_peak == "average":
        # calculate average of beta range 13-35 Hz
        for freq in FREQUENCY_BANDS.keys():
            f_average_rel_to_5_95 = np.mean(
                normalized_to_5_95[FREQUENCY_BANDS[freq][0] : FREQUENCY_BANDS[freq][1]]
            )
            f_average_rel_to_40_90 = np.mean(
                normalized_to_40_90[FREQUENCY_BANDS[freq][0] : FREQUENCY_BANDS[freq][1]]
            )
            f_average_raw = np.mean(
                average_Zxx[FREQUENCY_BANDS[freq][0] : FREQUENCY_BANDS[freq][1]]
            )

            # save data
            DBS_beta_baseline_dict = {
                "hemisphere": [hemisphere],
                "DBS_duration": [DBS_duration],
                "burstDBS_or_cDBS": [burstDBS_or_cDBS],
                "filtered": [filtered],
                "freq_band": [freq],
                "frequencies": [frequencies],
                "times": [times],
                "Zxx": [Zxx],
                "average_Zxx": [average_Zxx],
                "std_Zxx": [std_Zxx],
                "sem_Zxx": [sem_Zxx],
                "normalized_to_5_95": [normalized_to_5_95],
                "normalized_to_40_90": [normalized_to_40_90],
                "beta_baseline": [pre_or_post],
                "f_average_rel_to_5_95": [f_average_rel_to_5_95],
                "f_average_rel_to_40_90": [f_average_rel_to_40_90],
                "f_average_raw": [f_average_raw],
            }

            DBS_beta_baseline_single = pd.DataFrame(DBS_beta_baseline_dict)
            DBS_beta_baseline = pd.concat(
                [DBS_beta_baseline, DBS_beta_baseline_single], ignore_index=True
            )

    elif freq_average_or_peak == "peak":
        peak_average_Zxx_df = find_peaks(
            power_spectrum=average_Zxx, frequencies=frequencies
        ).copy()
        peak_average_Zxx_df["power_type"] = "average_Zxx"

        peak_normalized_to_5_95_df = find_peaks(
            power_spectrum=normalized_to_5_95, frequencies=frequencies
        ).copy()
        peak_normalized_to_5_95_df["power_type"] = "normalized_to_5_95"

        peak_normalized_to_40_90_df = find_peaks(
            power_spectrum=normalized_to_40_90, frequencies=frequencies
        ).copy()
        peak_normalized_to_40_90_df["power_type"] = "normalized_to_40_90"

        DBS_beta_baseline = pd.concat(
            [
                DBS_beta_baseline,
                peak_average_Zxx_df,
                peak_normalized_to_5_95_df,
                peak_normalized_to_40_90_df,
            ],
            ignore_index=True,
        )

        # add more relevant columns
        DBS_beta_baseline["hemisphere"] = hemisphere
        DBS_beta_baseline["DBS_duration"] = DBS_duration
        DBS_beta_baseline["burstDBS_or_cDBS"] = burstDBS_or_cDBS
        DBS_beta_baseline["filtered"] = filtered
        DBS_beta_baseline["beta_baseline"] = pre_or_post
        # DBS_beta_baseline["times"] = times
        # DBS_beta_baseline["Zxx"] = Zxx
        # DBS_beta_baseline["std_Zxx"] = std_Zxx
        # DBS_beta_baseline["sem_Zxx"] = sem_Zxx

    return DBS_beta_baseline


def concatenate_beta_baseline_data(pre_or_post: str):
    """
    Input:
        - pre_or_post: str, e.g. "pre", "post"

    """

    DBS_duration = ["1min", "5min"]
    burstDBS_or_cDBS = ["cDBS", "burstDBS"]
    filtered = ["band-pass_5_95", "unfiltered"]

    beta_baseline_average = (
        pd.DataFrame()
    )  # baseline is the power average within a frequency range
    beta_baseline_peak = (
        pd.DataFrame()
    )  # baseline is the power area under the curve around a peak frequency

    for hem in HEMISPHERES:
        for duration in DBS_duration:
            for dbs_type in burstDBS_or_cDBS:
                for filt in filtered:
                    # baseline is the power average within a frequency range
                    DBS_baseline_average = calculate_beta_baseline(
                        DBS_duration=duration,
                        burstDBS_or_cDBS=dbs_type,
                        filtered=filt,
                        hemisphere=hem,
                        pre_or_post=pre_or_post,
                        freq_average_or_peak="average",
                    )

                    beta_baseline_average = pd.concat(
                        [beta_baseline_average, DBS_baseline_average], ignore_index=True
                    )

                    # baseline is the power area under the curve around a peak frequency
                    DBS_baseline_peak = calculate_beta_baseline(
                        DBS_duration=duration,
                        burstDBS_or_cDBS=dbs_type,
                        filtered=filt,
                        hemisphere=hem,
                        pre_or_post=pre_or_post,
                        freq_average_or_peak="peak",
                    )

                    beta_baseline_peak = pd.concat(
                        [beta_baseline_peak, DBS_baseline_peak], ignore_index=True
                    )

        # 30 min DBS duration only available for burst DBS so far...
        for filt in filtered:
            # baseline is the power average within a frequency range
            burstDBS_30min_average = calculate_beta_baseline(
                DBS_duration="30min",
                burstDBS_or_cDBS="burstDBS",
                filtered=filt,
                hemisphere=hem,
                pre_or_post=pre_or_post,
                freq_average_or_peak="average",
            )

            beta_baseline_average = pd.concat(
                [beta_baseline_average, burstDBS_30min_average], ignore_index=True
            )

            # baseline is the power area under the curve around a peak frequency
            burstDBS_30min_peak = calculate_beta_baseline(
                DBS_duration="30min",
                burstDBS_or_cDBS="burstDBS",
                filtered=filt,
                hemisphere=hem,
                pre_or_post=pre_or_post,
                freq_average_or_peak="peak",
            )

            beta_baseline_peak = pd.concat(
                [beta_baseline_peak, burstDBS_30min_peak], ignore_index=True
            )

    # save data as pickle
    io.save_result_dataframe_as_pickle(
        data=beta_baseline_average,
        filename=f"beta_average_baseline_{pre_or_post}_DBS_patterned_pilot_sub-075",
    )

    io.save_result_dataframe_as_pickle(
        data=beta_baseline_peak,
        filename=f"beta_peak_baseline_{pre_or_post}_DBS_patterned_pilot_sub-075",
    )

    return {
        "beta_baseline_average": beta_baseline_average,
        "beta_baseline_peak": beta_baseline_peak,
    }


def power_spectrum_baseline(
    DBS_duration: str,
    burstDBS_or_cDBS: str,
    hemisphere: str,
    normalized: str,
    freq_band: str,
    filtered: str,
    pre_or_post: str,
):
    """
    Plot the Power Spectrum of the baseline (pre or post DBS)
    """

    normalize_dict = {
        "normalized_to_5_95": "normalized_to_5_95",
        "normalized_to_40_90": "normalized_to_40_90",
        "not_normalized": "average_Zxx",
    }

    ############## load the beta baseline of the given recording ##############
    beta_average_baseline = io.load_pickle_files(
        filename=f"beta_average_baseline_{pre_or_post}_DBS_patterned_pilot_sub-075"
    )
    beta_average_baseline = beta_average_baseline[
        beta_average_baseline["DBS_duration"] == DBS_duration
    ]
    beta_average_baseline = beta_average_baseline[
        beta_average_baseline["burstDBS_or_cDBS"] == burstDBS_or_cDBS
    ]
    beta_average_baseline = beta_average_baseline[
        beta_average_baseline["hemisphere"] == hemisphere
    ]
    beta_average_baseline = beta_average_baseline[
        beta_average_baseline["filtered"] == filtered
    ]
    beta_average_baseline = beta_average_baseline[
        beta_average_baseline["freq_band"] == freq_band
    ]

    power_spectrum = beta_average_baseline[normalize_dict[normalized]].values[0]
    frequencies = beta_average_baseline["frequencies"].values[0]

    # load the Peak data
    beta_peak_baseline = io.load_pickle_files(
        filename=f"beta_peak_baseline_{pre_or_post}_DBS_patterned_pilot_sub-075"
    )
    beta_peak_baseline = beta_peak_baseline[
        beta_peak_baseline["power_type"] == normalize_dict[normalized]
    ]
    peak_freq_range = beta_peak_baseline["freq_range_around_max_peak"].values[0]

    # fig = plt.figure(figsize=(10, 5), layout="tight")
    plt.subplot(1, 1, 1)
    plt.title(
        f"Power Spectrum {filtered} sub-075 {hemisphere}: \n {pre_or_post} {burstDBS_or_cDBS}, {DBS_duration}, normalized: {normalized}",
        fontdict={"size": 20},
    )
    plt.plot(frequencies, power_spectrum, linewidth=3)

    plt.xlabel("Frequency [Hz]", fontdict={"size": 20})
    plt.ylabel("PSD", fontdict={"size": 20})
    # plt.ylim(1, 100)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.axvline(x=peak_freq_range[0], color="black", linewidth=3)
    plt.axvline(x=peak_freq_range[6], color="black", linewidth=3)

    # Plot the legend only for the first row "postop"
    # plt.legend(loc="upper right", edgecolor="black", fontsize=40)

    plt.show()


def load_value_beta_baseline(
    DBS_duration: str,
    burstDBS_or_cDBS: str,
    hemisphere: str,
    normalized: str,
    freq_band: str,
    filtered: str,
    freq_average_or_peak: str,
    pre_or_post: str,
):
    """
    Input:
        - DBS_duration: str, e.g. "1min", "5min", "30min"
        - burstDBS_or_cDBS: str, e.g. "cDBS", "burstDBS"
        - hemisphere: str, e.g. "Left", "Right"
        - normalized: str, e.g. "normalized_to_5_95", "normalized_to_40_90", "not_normalized"
        - freq_band: str, e.g. "beta", "low_beta", "high_beta"
        - filtered: str, e.g. "band-pass_5_95", "unfiltered"
        - pre_or_post: str, e.g. "pre", "post"
        - freq_average_or_peak: str, "average", "peak" -> peak will give the area under the curve ± 3 Hz around the peak frequency

    """

    normalize_dict_average = {
        "normalized_to_5_95": "f_average_rel_to_5_95",
        "normalized_to_40_90": "f_average_rel_to_40_90",
        "not_normalized": "f_average_raw",
    }

    normalize_dict_peak = {
        "normalized_to_5_95": "normalized_to_5_95",
        "normalized_to_40_90": "normalized_to_40_90",
        "not_normalized": "average_Zxx",
    }

    ############## load the beta baseline of the given recording ##############
    beta_baseline = io.load_pickle_files(
        filename=f"beta_{freq_average_or_peak}_baseline_{pre_or_post}_DBS_patterned_pilot_sub-075"
    )
    beta_baseline = beta_baseline[beta_baseline["DBS_duration"] == DBS_duration]
    beta_baseline = beta_baseline[beta_baseline["burstDBS_or_cDBS"] == burstDBS_or_cDBS]
    beta_baseline = beta_baseline[beta_baseline["hemisphere"] == hemisphere]
    beta_baseline = beta_baseline[beta_baseline["filtered"] == filtered]
    beta_baseline = beta_baseline[beta_baseline["freq_band"] == freq_band]

    if freq_average_or_peak == "average":
        return beta_baseline[normalize_dict_average[normalized]].values[0]

    elif freq_average_or_peak == "peak":
        beta_baseline = beta_baseline[
            beta_baseline["power_type"] == normalize_dict_peak[normalized]
        ]
        return beta_baseline["power_area_under_curve"].values[0]


def plot_post_dbs_time_series(
    DBS_duration: str, burstDBS_or_cDBS: str, hemisphere: str
):
    """
    Input:
        - dbs_duration: str, e.g. "1min", "5min", "30min"
        - burst_or_cDBS: str, e.g. "cDBS", "burstDBS"
        - hemisphere: str, e.g. "Left", "Right"

    plot the raw time series, manually write the time in seconds when stimulation is turned off = start for collecting beta post-DBS


    """

    # load the raw object
    all_raw_objects = io.load_pickle_files(
        filename="raw_objects_patterned_pilot_sub-075"
    )

    # find correct key by input values
    for key, value in sub_075_pilot_streaming_dict.items():
        if (
            value[0] == "post"
            and value[1] == burstDBS_or_cDBS
            and value[2] == DBS_duration
            and value[3] == hemisphere
        ):
            raw = all_raw_objects[key]

    raw.plot()

    return raw


def get_post_dbs_time_series(DBS_duration: str, burstDBS_or_cDBS: str, hemisphere: str):
    """
    Input:
        - DBS_duration: str, e.g. "1min", "5min", "30min"
        - burstDBS_or_cDBS: str, e.g. "cDBS", "burstDBS"
        - hemisphere: str, e.g. "Left", "Right"


    """

    ############## load the excel file to get the index when stimulation is turned OFF ##############
    dbs_turned_off = io.load_excel_files(filename="streaming_dbs_turned_OFF")

    # select
    dbs_turned_off = dbs_turned_off[dbs_turned_off["DBS_duration"] == DBS_duration]
    dbs_turned_off = dbs_turned_off[
        dbs_turned_off["burstDBS_or_cDBS"] == burstDBS_or_cDBS
    ]
    dbs_turned_off = dbs_turned_off[dbs_turned_off["hemisphere"] == hemisphere]

    # get index when stimulation is turned off
    dbs_OFF_sec = dbs_turned_off["DBS_OFF_sec"].values[0]

    ############## load the time domain data ##############
    streaming_data = io.load_pickle_files(
        filename="streaming_info_patterned_pilot_sub-075"
    )

    # select
    streaming_data = streaming_data[streaming_data["pre_or_post"] == "post"]
    streaming_data = streaming_data[streaming_data["DBS_duration"] == DBS_duration]
    streaming_data = streaming_data[
        streaming_data["burstDBS_or_cDBS"] == burstDBS_or_cDBS
    ]
    streaming_data = streaming_data[streaming_data["hemisphere"] == hemisphere]

    # get the time domain data
    streaming_data = streaming_data["original_time_domain_data"].values[0]

    # now get the index of the time domain matching the dbs_OFF_sec
    dbs_OFF_index = int(
        dbs_OFF_sec * SAMPLING_FREQ
    )  # this is start of the beta post-DBS readout

    # cut the time domain data at the dbs_OFF_index and take the following 3 min (45000 samples)
    streaming_data = streaming_data[dbs_OFF_index : dbs_OFF_index + 45000]

    # only get the last 1 min of the selected time domain data
    post_dbs_baseline_data = streaming_data[-15000:]

    return np.array(streaming_data), np.array(post_dbs_baseline_data)


def calculate_rel_power_post_dbs(
    DBS_duration: str,
    burstDBS_or_cDBS: str,
    hemisphere: str,
    normalized: str,
    freq_band: str,
    freq_average_or_peak: str,
    filtered: str,
):
    """

    Input:
        - dbs_duration: str, e.g. "1min", "5min", "30min"
        - burst_or_cDBS: str, e.g. "cDBS", "burstDBS"
        - hemisphere: str, e.g. "Left", "Right"
        - normalized: str, e.g. "normalized_to_5_95", "normalized_to_40_90", "not_normalized"
        - freq_band: str, e.g. "beta", "low_beta", "high_beta"
        - filtered: str, e.g. "band-pass_5_95", "unfiltered"
        - rel_pre_or_post_DBS: str, e.g. "pre", "post"

    1) load the post DBS time series
    2) load the beta baseline of the given recording: band-pass-filtered 5-95 Hz
    """

    post_DBS_dataframe = pd.DataFrame()

    ############## load the post DBS time series ##############

    normalized_dict = {
        "normalized_to_5_95": "5_to_95",
        "normalized_to_40_90": "40_to_90",
    }
    # length = 3 minutes (45000 samples)
    streaming_data, post_dbs_baseline_data = get_post_dbs_time_series(
        DBS_duration=DBS_duration,
        burstDBS_or_cDBS=burstDBS_or_cDBS,
        hemisphere=hemisphere,
    )

    ############## load the beta baseline of the given recording ##############
    # baseline pre DBS 2 min before Stim ON
    beta_baseline_pre = load_value_beta_baseline(
        DBS_duration=DBS_duration,
        burstDBS_or_cDBS=burstDBS_or_cDBS,
        hemisphere=hemisphere,
        normalized=normalized,
        freq_band=freq_band,
        filtered=filtered,
        freq_average_or_peak=freq_average_or_peak,
        pre_or_post="pre",
    )

    # baseline post DBS 2-3 min after stim OFF
    beta_baseline_post = load_value_beta_baseline(
        DBS_duration=DBS_duration,
        burstDBS_or_cDBS=burstDBS_or_cDBS,
        hemisphere=hemisphere,
        normalized=normalized,
        freq_band=freq_band,
        filtered=filtered,
        freq_average_or_peak=freq_average_or_peak,
        pre_or_post="post",
    )

    ############## calculate PSD ##############
    if filtered == "band-pass_5_95":
        # band-pass filter 5-95 Hz
        time_domain_data = lfp_preprocessing.band_pass_filter_percept(
            fs=SAMPLING_FREQ, signal=streaming_data
        )

    elif filtered == "unfiltered":
        time_domain_data = streaming_data

    # calculate PSD from the postDBS time series (3 minutes = 45000 samples)
    frequencies, times, Zxx, average_Zxx, std_Zxx, sem_Zxx = fourier_transform(
        time_domain_data
    )
    # Zxx = 126 arrays, each len=359 -> 359*0.5s = 179.5s = 2.99 min

    # from Zxx get the power spectra for each 0.5 seconds
    power_spectra_half_sec = np.transpose(Zxx)  # 359 arrays, each len=126

    # save data
    post_DBS_dict = {
        "hemisphere": [hemisphere],
        "DBS_duration": [DBS_duration],
        "burstDBS_or_cDBS": [burstDBS_or_cDBS],
        "filtered": [filtered],
        "freq_band": [freq_band],
        "frequencies": [frequencies],
        "times": [times],
        "Zxx": [Zxx],
        "average_Zxx": [average_Zxx],
        "std_Zxx": [std_Zxx],
        "sem_Zxx": [sem_Zxx],
        "power_spectra_half_sec": [power_spectra_half_sec],
    }

    post_DBS_single_dataframe = pd.DataFrame(post_DBS_dict)
    post_DBS_dataframe = pd.concat(
        [post_DBS_dataframe, post_DBS_single_dataframe], ignore_index=True
    )

    # normalize PSD for each 0.5 seconds

    half_sec_spectra_df = pd.DataFrame()

    for half_sec_psd in np.arange(0, len(power_spectra_half_sec), 1):
        single_spectrum = power_spectra_half_sec[half_sec_psd]
        seconds_post_DBS_OFF = times[half_sec_psd]

        # normalize PSD
        if normalized != "not_normalized":
            normalized_psd = normalize_to_psd(
                to_sum=normalized_dict[normalized], power_spectrum=single_spectrum
            )

        elif normalized == "not_normalized":
            normalized_psd = single_spectrum

        if freq_average_or_peak == "average":
            # calculate average of frequency range of interest
            half_sec_data = np.mean(
                normalized_psd[
                    FREQUENCY_BANDS[freq_band][0] : FREQUENCY_BANDS[freq_band][1]
                ]
            )

        elif freq_average_or_peak == "peak":
            half_sec_data = find_peaks(
                power_spectrum=normalized_psd, frequencies=frequencies
            )
            half_sec_data = half_sec_data[half_sec_data["freq_band"] == freq_band]
            half_sec_data = half_sec_data["power_area_under_curve"].values[0]

        half_sec_spectra_dict = {
            "half_sec_psd": [half_sec_psd],
            "seconds_post_DBS_OFF": [seconds_post_DBS_OFF],
            "single_spectrum": [normalized_psd],
            "freq_band": [freq_band],
            f"half_sec_{freq_average_or_peak}": [half_sec_data],
            f"rel_freq_{freq_average_or_peak}_to_pre_DBS": [
                half_sec_data / beta_baseline_pre
            ],
            "beta_baseline_pre_DBS": [beta_baseline_pre],
            f"rel_freq_{freq_average_or_peak}_to_post_DBS": [
                half_sec_data / beta_baseline_post
            ],
            "beta_baseline_post_DBS": [beta_baseline_post],
        }

        half_sec_spectra_single = pd.DataFrame(half_sec_spectra_dict)
        half_sec_spectra_df = pd.concat(
            [half_sec_spectra_df, half_sec_spectra_single], ignore_index=True
        )

    return post_DBS_dataframe, half_sec_spectra_df
