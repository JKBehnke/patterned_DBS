""" Load result files from results folder"""

import os
import pandas as pd
import pickle
import pyxdf
import mne

from ..utils import find_folders as find_folders

# import py_perceive
from PerceiveImport.classes import main_class

GROUP_RESULTS_PATH = find_folders.get_patterned_dbs_project_path(folder="GroupResults")
GROUP_FIGURES_PATH = find_folders.get_patterned_dbs_project_path(folder="GroupFigures")

FILENAME_DICT = {
    "streaming": "-BrainSense",
    "indefinite_streaming": "-IS",
    "rest": "Rest",
    "updrs": "UPDRS",
    "on": "MedOn",
    "off": "MedOff",
}


########### save results ############
def save_result_dataframe_as_pickle(data: pd.DataFrame, filename: str):
    """
    Input:
        - data: must be a pd.DataFrame()
        - filename: str, e.g."externalized_preprocessed_data"

    picklefile will be written in the group_results_path:

    """

    group_data_path = os.path.join(GROUP_RESULTS_PATH, f"{filename}.pickle")
    with open(group_data_path, "wb") as file:
        pickle.dump(data, file)

    print(f"{filename}.pickle", f"\nwritten in: {GROUP_RESULTS_PATH}")


def save_fig_png_and_svg(path: str, filename: str, figure=None):
    """
    Input:
        - path: str
        - filename: str
        - figure: must be a plt figure

    """

    figure.savefig(
        os.path.join(path, f"{filename}.svg"),
        bbox_inches="tight",
        format="svg",
    )

    figure.savefig(
        os.path.join(path, f"{filename}.png"),
        bbox_inches="tight",
    )

    print(f"Figures {filename}.svg and {filename}.png", f"\nwere written in: {path}.")


########### load results ############


def load_source_json_patterned_dbs(sub: str, incl_session: list, run: str):
    """ """

    # load with pyPerceive
    py_perceive_data = main_class.PerceiveData(
        sub=sub,
        incl_modalities=["streaming"],
        incl_session=incl_session,
        incl_condition=["m0s1"],
        incl_task=["rest"],
        import_json=True,
        # warn_for_metaNaNs = True,
        # allow_NaNs_in_metadata = True,
    )

    # get JSON
    json_data = {
        "1": py_perceive_data.streaming.fu3m.m0s1.rest.run1.json,
        "2": py_perceive_data.streaming.fu3m.m0s1.rest.run2.json,
    }

    return json_data[run]


def load_pickle_files(filename: str):
    """
    Input:
        - filename: str, e.g.
            "streaming_info_patterned_pilot_sub-075"
            "raw_objects_patterned_pilot_sub-075"
            "beta_baseline_patterned_pilot_sub-075"

    Returns:
        - data
    """

    group_data_path = os.path.join(GROUP_RESULTS_PATH, f"{filename}.pickle")
    with open(group_data_path, "rb") as file:
        data = pickle.load(file)

    return data


def load_excel_files(filename: str):
    """ """

    dbs_turned_off_sheet = ["streaming_dbs_turned_OFF"]

    # find the path to the results folder
    path = find_folders.get_patterned_dbs_project_path(folder="data")

    # create filename
    f_name = f"{filename}.xlsx"

    if filename in dbs_turned_off_sheet:
        sheet_name = "dbs_OFF"

    filepath = os.path.join(path, f_name)

    # load the file
    data = pd.read_excel(filepath, keep_default_na=True, sheet_name=sheet_name)
    print("Excel file loaded: ", f_name, "\nloaded from: ", path)

    return data


def load_xdf_files(
    sub: str,
    stimulation: str,
    medication: str,
    task: str,
    run: str,
):
    """
    xdf file with filename structure: sub-084_ses-burst_on_med-off_task-updrs_run-001_eeg.xdf
    Input:
        - sub: str e.g. "084
        - stimulation: str: ["StimOnB", "StimOffB", "StimOnA", "StimOffA"] (A=continuous, B=burst)
        - medication: str ["off", "on"]
        - task: str ["updrs", "rest"]
        - run: str ["1", "2", "3] (in Stim OFF there are 3 runs: 0, 30 and 60 min after Stim OFF)

    """

    # find the path to the burst DBS onedrive folder
    path = find_folders.get_onedrive_path_burst_dbs(folder="sub_lsl_data", sub=sub)

    path = os.path.join(path, f"ses-{stimulation}_med-{medication}")
    path = os.path.join(path, "eeg")
    path = os.path.join(
        path,
        f"sub-{sub}_ses-{stimulation}_med-{medication}_task-Default_acq-{task}_run-00{run}_eeg.xdf",
    )

    data, header = pyxdf.load_xdf(path)

    return data, header, path


def load_perceive_file(
    sub: str, modality: str, task: str, medication: str, stimulation: str, run: str
):
    """
    Input:
    - sub: str, e.g. "084"
    - modality: str, e.g. "streaming", "indefinite_streaming"
    - task: str, e.g. "updrs", "rest"
    - medication: str, e.g. "on", "off"
    - stimulation: str, e.g. ["StimOnB", "StimOffB", "StimOnA", "StimOffA"] (A=continuous, B=burst)
    - run: str, e.g. "1", "2", "3" (run-1 is the first run, in Stim OFF there are 3 runs: 0, 30 and 60 min after Stim OFF)

    """

    # get a list with files from the path
    path = find_folders.get_onedrive_path_burst_dbs(folder="sub_perceive_data", sub=sub)

    # check for error:
    if modality == "indefinite_streaming":
        task = "rest"

        if "On" in stimulation:
            raise ValueError("Stimulation On is not available for indefinite streaming")

    # get the correct file
    file_path = None
    for filename in os.listdir(path):
        if (
            FILENAME_DICT[modality] in filename
            and FILENAME_DICT[task] in filename
            and FILENAME_DICT[medication] in filename
            and stimulation in filename
            and f"run-{run}" in filename
        ):
            file_path = os.path.join(path, filename)

    if file_path is None:
        raise FileNotFoundError(
            f"File not found in {path}. \nAvailable files: \n{os.listdir(path)}"
        )

    # load the file
    data = mne.io.read_raw_fieldtrip(file_path, info={}, data_name="data")

    return {"data": data, "file_path": file_path}
