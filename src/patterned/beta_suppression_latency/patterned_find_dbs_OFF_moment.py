""" Patterned DBS find DBS OFF moment"""


import os
import pickle
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
import scipy
from scipy.signal import find_peaks


# internal Imports
from ..utils import find_folders as find_folders
from ..utils import lfp_preprocessing as lfp_preprocessing
from ..utils import io as io


def find_LFP_sync_artefact(
    lfp_data: np.ndarray,
    sf_LFP,
    use_kernel: str = "2",  # or 1
    consider_first_seconds_LFP=None,
):
    """
    Function that finds artefacts caused by
    augmenting-reducing stimulation from 0 to 1mA without ramp.
    For correct functioning, the LFP data should
    start in stim-off, and typically short pulses
    are given (without ramping).
    The function uses a kernel which mimics the stimulation-
    artefact. This kernel is multiplied with time-series
    snippets of the same length. If the time-serie is
    similar to the kernel, the dot-product is high, and this
    indicates a stim-artefact.

    Input:
        - lfp_data: single channel as np.ndarray (the function
            automatically inverts the signal if first a positive
            peak is found, this indicates an inverted signal)
        - sf_LFP (int): sampling frequency of intracranial recording
        - use_kernel: decides whether kernel 1 or 2 is used,
            kernel 1 is straight-forward and finds a steep decrease,
            kernel 2 mimics the steep decrease and slow recovery of the signal.
            In our tests, kernel 2 was the best in 52.7% of the cases.
        - consider_first_seconds_LFP: if given, only artefacts in the first
            (and last) n-seconds are considered

    Returns:
        - stim_idx: a list with all stim-artefact starts.
    """

    # import settings
    json_path = os.path.join(os.getcwd(), "config")
    json_filename = "config.json"  # dont forget json extension
    with open(os.path.join(json_path, json_filename), "r") as f:
        loaded_dict = json.load(f)

    signal_inverted = False  # defaults false

    # checks correct input for use_kernel variable
    assert use_kernel in ["1", "2"], "use_kernel incorrect"

    # kernel 1 only searches for the steep decrease
    # kernel 2 is more custom and takes into account the steep decrease and slow recover
    kernels = {
        "1": np.array([1, -1]),
        "2": np.array([1, 0, -1] + list(np.linspace(-1, 0, 20))),
    }
    ker = kernels[use_kernel]

    # get dot-products between kernel and time-serie snippets
    res = []  # store results of dot-products
    for i in np.arange(0, len(lfp_data) - len(ker)):
        res.append(ker @ lfp_data[i : i + len(ker)])  # calculate dot-product of vectors
        # the dot-product result is high when the timeseries snippet
        # is very similar to the kernel
    res = np.array(res)  # convert list to array

    # # normalise dot product results
    res = res / max(res)

    # calculate a ratio between std dev and maximum during
    # the first seconds to check whether an stim-artef was present
    ratio_max_sd = np.max(res[: sf_LFP * 30] / np.std(res[: sf_LFP * 5]))

    # use peak of kernel dot products
    pos_idx = find_peaks(x=res, height=0.3 * max(res), distance=sf_LFP)[0]
    neg_idx = find_peaks(x=-res, height=-0.3 * min(res), distance=sf_LFP)[0]

    # check whether signal is inverted
    if neg_idx[0] < pos_idx[0]:
        # the first peak should be POSITIVE (this is for the dot-product results)
        # actual signal is first peak negative
        # if NEG peak before POS then signal is inverted
        print("signal is inverted")
        signal_inverted = True
        # print(pos_idx[0], neg_idx[0])
        # re-check inverted for difficult cases with small pos-lfp peak before negative stim-artefact
        if (
            pos_idx[0] - neg_idx[0]
        ) < 50:  # if first positive and negative are very close
            width_pos = 0
            r_i = pos_idx[0]
            while res[r_i] > (max(res) * 0.3):
                r_i += 1
                width_pos += 1
            width_neg = 0
            r_i = neg_idx[0]
            while res[r_i] < (min(res) * 0.3):
                r_i += 1
                width_neg += 1
            # undo invertion if negative dot-product (pos lfp peak) is very narrow
            if width_pos > (2 * width_neg):
                signal_inverted = False
                print("invertion undone")

    # return either POS or NEG peak-indices based on normal or inverted signal
    if not signal_inverted:
        stim_idx = pos_idx  # this is for 'normal' signal

    elif signal_inverted:
        stim_idx = neg_idx

    # check warn if NO STIM artefacts are suspected
    if len(stim_idx) > 20 and ratio_max_sd < 8:
        print(
            "WARNING: probably the LFP signal did NOT"
            " contain any artefacts. Many incorrect timings"
            " could be returned"
        )

    if consider_first_seconds_LFP:
        border_start = sf_LFP * consider_first_seconds_LFP
        border_end = len(lfp_data) - (sf_LFP * consider_first_seconds_LFP)
        sel = np.logical_or(
            np.array(stim_idx) < border_start, np.array(stim_idx) > border_end
        )
        stim_idx = list(compress(stim_idx, sel))

    # filter out inconsistencies in peak heights (assuming sync-stim-artefacts are stable)
    abs_heights = [max(abs(lfp_data[i - 5 : i + 5])) for i in stim_idx]
    diff_median = np.array([abs(p - np.median(abs_heights)) for p in abs_heights])
    sel_idx = diff_median < (np.median(abs_heights) * 0.66)
    stim_idx = list(compress(stim_idx, sel_idx))
    # check polarity of peak
    if not signal_inverted:
        sel_idx = np.array([min(lfp_data[i - 5 : i + 5]) for i in stim_idx]) < (
            np.median(abs_heights) * -0.5
        )
    elif signal_inverted:
        sel_idx = np.array([max(lfp_data[i - 5 : i + 5]) for i in stim_idx]) > (
            np.median(abs_heights) * 0.5
        )
    stim_idx = list(compress(stim_idx, sel_idx))

    return stim_idx
