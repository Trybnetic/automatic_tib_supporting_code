"""
This script creates the 1min aggregated data from the raw data files
"""
import os

import pandas as pd 
from tqdm import tqdm


def calibrate(acc, scale, offset):
    columns = ["Y", "X", "Z"]
    index = acc.index.copy()
    acc = (scale * acc[columns].values) + offset
    
    acc = pd.DataFrame(acc.astype(float), 
                       columns=columns, 
                       index=index)
    
    return acc[["X", "Y", "Z"]]


base_path = "/path/to/raw/data"

datasets = [
    ("Accelerometer dataset", os.path.join(base_path, "Tromso_Study_dataset/Tromso_Study_Accelerometer_Dataset/"), "1min_data/accelerometer_dataset/"),
    ("TiB dataset", os.path.join(base_path, "TiB_dataset/Tromso_Study_Time_in_Bed_Dataset/"), "1min_data/tib_dataset/")    
]

for dataset_name, in_path, out_path in datasets:
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    calibration = pd.read_csv(in_path + "calibration_error.txt", sep=";")
    for _, row in tqdm(calibration.iterrows(), desc=dataset_name, total=len(calibration)):
        # Load relevant data from calibration file
        file = row["file_path"]
        scale = row[["scale1", "scale2", "scale3"]].values
        offset = row[["offset1", "offset2", "offset3"]].values

        # Load data
        data = pd.read_csv(os.path.join(in_path, file), compression="gzip", index_col=0)
        data.index = pd.to_datetime(data.index)

        # Autocalibrate data
        data[["X", "Y", "Z"]] = calibrate(data, scale, offset)

        # Resample to 1min data
        data = data.resample("1min").agg({"X": "mean", "Y": "mean", "Z": "mean", "is_in_bed": pd.Series.mode})

        # Save 1min data
        file = file.split("/")[-1].replace(".gz", "")     
        data.to_csv(os.path.join(out_path, file))