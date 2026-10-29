# Collating all the 150 group data folders and turning them into a single cleanly labelled table formatted like the UCI datset (g, with gravity positive), 
# so that preprocessing_group.py can treat the recordings in the same way.

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FOLDER = PROJECT_ROOT / "data" / "raw"
CONFIG_FILE = PROJECT_ROOT / "data" / "group_configuration.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "group" / "group_combined.csv"

ACCELEROMETER_FILE = "AccelerometerUncalibrated.csv"
GYROSCOPE_FILE = "Gyroscope.csv"

# What to multiply the acceleration by, to get UCI's units (g, gravity positive). Worked out from the flat table tests.
UNIT_CONVERSION = {
    "ios": -1.0,                # Already in g, but the sign is flipped
    "android": 1 / 9.80665,     # m/s^2 -> g (1 g = 9.80665 m/s^2)
}

# Seconds removed from the start and end of each recording due to positional setup.
TRIM_SECONDS = {
    "walking": 4.0,
    "walking_upstairs": 2.5,
    "walking_downstairs": 2.5,
}

VALID_POSITIONS = ["pocket", "waist", "hand"]

PAIRING_TOLERANCE_NANOSECONDS = 15_000_000   # 15 milliseconds

LABEL_COLUMNS = ["recording_id", "participant", "activity", "position", "platform"]
SENSOR_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------


def load_sensor(path, prefix, multiply_by=1.0): 
    # Load one sensor file and clean it.
    data = pd.read_csv(path)
 
    # Only keep the columns we need, picked by name (the files store them as z, y, x).
    data = data[["time", "seconds_elapsed", "x", "y", "z"]]
    data = data.rename(columns={"x": f"{prefix}_x", "y": f"{prefix}_y", "z": f"{prefix}_z"})
 
    # Convert the units / sign
    for axis in ["x", "y", "z"]:
        data[f"{prefix}_{axis}"] = data[f"{prefix}_{axis}"] * multiply_by
 
    # Remove missing values and duplicate readings, and put the readings in time order.
    data = data.dropna()
    data = data.drop_duplicates(subset="time")
    data = data.sort_values("time").reset_index(drop=True)
    return data
 
 
def pair_sensors(acc, gyro):
    # Pair each accelerometer reading with the nearest gyroscope reading (within 15 ms).
    paired = pd.merge_asof(
        acc,
        gyro[["time", "gyro_x", "gyro_y", "gyro_z"]],
        on="time",
        direction="nearest",
        tolerance=PAIRING_TOLERANCE_NANOSECONDS,
    )
    return paired.dropna()   # Readings with no gyro reading close enough are dropped.
 
 
def trim_ends(recording, seconds):

    start = recording["seconds_elapsed"].min() + seconds
    end = recording["seconds_elapsed"].max() - seconds
    keep = (recording["seconds_elapsed"] >= start) & (recording["seconds_elapsed"] <= end)
    return recording[keep]
 

def clean_one_recording(row):
    # Clean one recording (one row of the config file) and return it as a table.
    folder = RAW_FOLDER / row["folder"]
 
    # Load both the sensors data, converting units.
    acc = load_sensor(folder / ACCELEROMETER_FILE, "acc", multiply_by=UNIT_CONVERSION[row["platform"]])
    gyro = load_sensor(folder / GYROSCOPE_FILE, "gyro")
 
    # Pair the accelerometer and gyroscope readings and drop any bad readings
    recording = pair_sensors(acc, gyro)
 
    # Trim start and end of recordings
    recording = trim_ends(recording, TRIM_SECONDS[row["activity"]]).copy()
 
    # Add the labels to every reading
    for column in LABEL_COLUMNS:
        recording[column] = row[column]
 
    print(f"  {row['recording_id']}: {len(recording):,} readings kept")
    return recording[LABEL_COLUMNS + ["seconds_elapsed"] + SENSOR_COLUMNS]
 
 
# --------------------------------------------------------------------------
# The main function
# --------------------------------------------------------------------------