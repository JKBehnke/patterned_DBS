""" Structuring the xdf file """

import pandas as pd

from ..utils import find_folders
from ..utils import io


def get_xdf_structure(sub: str, stimulation: str, medication: str, task: str, run: str):
    """
    Load the xdf file, structure the data and return the structured data

    1. Load the xdf file

    """

    # load the xdf file
    xdf_data, header, xdf_path = io.load_xdf_files(
        sub=sub, stimulation=stimulation, medication=medication, task=task, run=run
    )

    # there should be 2 streams, check if not
    if len(xdf_data) != 2:
        print(
            "There are not 2 streams in the xdf file. Instead there are: ",
            len(xdf_data),
        )
        return None

    # get the details from each stream
    for stream_nr in [0, 1]:
        stream = xdf_data[stream_nr]

        timestamps = stream["time_stamps"]
        samples = stream["time_series"]

        # extract the details
        stream_info = stream["info"]
        stream_name = stream_info["name"][0]
        stream_type = stream_info["type"][0]
        stream_id = stream_info["stream_id"]
        channel_count = int(stream_info["channel_count"][0])
        nominal_srate = float(stream_info["nominal_srate"][0])
        duration = len(timestamps) / nominal_srate

        # channels
        channel_list = []
        for ch in range(channel_count):
            channel_name = stream_info["desc"][0]["channels"][0]["channel"][ch][
                "label"
            ][0]
            channel_list.append(channel_name)

        # samples as dataframe
        samples_df = pd.DataFrame(samples, columns=channel_list)
        samples_df["timestamps"] = timestamps

        # save the stream details
        if "Ultraleap" in stream_name:

            ultraleap_metadata = {
                "stream_name": [stream_name],
                "stream_type": [stream_type],
                "stream_id": [stream_id],
                "channel_count": [channel_count],
                "nominal_srate": [nominal_srate],
                "duration": [duration],
            }
            ultraleap_metadata_df = pd.DataFrame(ultraleap_metadata)

            ultraleap_channels = channel_list
            ultraleap_timestamps = timestamps
            ultraleap_samples = samples
            ultraleap_data = samples_df

        elif "SAGA" in stream_name:

            SAGA_metadata = {
                "stream_name": [stream_name],
                "stream_type": [stream_type],
                "stream_id": [stream_id],
                "channel_count": [channel_count],
                "nominal_srate": [nominal_srate],
                "duration": [duration],
            }
            SAGA_metadata_df = pd.DataFrame(SAGA_metadata)

            SAGA_channels = channel_list
            SAGA_timestamps = timestamps
            SAGA_samples = samples
            SAGA_data = samples_df

        else:
            print("Stream name not recognized")
            return None

    return {
        "ultraleap_metadata": ultraleap_metadata_df,
        "ultraleap_channels": ultraleap_channels,
        "ultraleap_timestamps": ultraleap_timestamps,
        "ultraleap_samples": ultraleap_samples,
        "ultraleap_data": ultraleap_data,
        "SAGA_metadata": SAGA_metadata_df,
        "SAGA_channels": SAGA_channels,
        "SAGA_timestamps": SAGA_timestamps,
        "SAGA_samples": SAGA_samples,
        "SAGA_data": SAGA_data,
        "header": header,
        "xdf_path": xdf_path,
        "xdf_data": xdf_data,
    }
