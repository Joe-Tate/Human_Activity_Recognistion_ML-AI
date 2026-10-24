# Loads UCI's windows (Inertial Signals), attaches activity and volunteer labels and keeps only the 3 walking activities

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UCI_FOLDER = PROJECT_ROOT / "data" / "processed" / "uci"

SIGNAL_NAMES = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]

ACTIVITIES = {
    "WALKING": "walking",
    "WALKING_UPSTAIRS": "walking_upstairs",
    "WALKING_DOWNSTAIRS": "walking_downstairs",
}


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------


def check_file_exists(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}\n"
        )
    
    
def read_activity_names():
    # Reads activity_labels.txt, which says what each activity number means and returns a dictionary. 
    path = UCI_FOLDER / "activity_labels.txt"
    check_file_exists(path)

    activity_names = {}
    table = pd.read_csv(path, sep=r"\s+", header=None)
    for number, name in zip(table[0], table[1]):
        activity_names[number] = name
    return activity_names


def read_one_column(path):
    check_file_exists(path)
    return pd.read_csv(path, header=None)[0].tolist()


def read_one_signal(split, signal_name):
    # Reads a signal file and returns a 2D array (windows, 128 readings).
    
    path = UCI_FOLDER / split / "Inertial Signals" / f"{signal_name}_{split}.txt"
    check_file_exists(path)
    return pd.read_csv(path, sep=r"\s+", header=None).to_numpy()


# --------------------------------------------------------------------------
# Main function
# --------------------------------------------------------------------------

def load_uci_inertial(split="train"):
    # Load the UCI windows and their labels for one split (train or test) and return the windows witht their labels.
    if split not in ("train", "test"):
        raise ValueError('split must be "train" or "test"')

    # Reads the activity and volunteer number of every window.
    activity_numbers = read_one_column(UCI_FOLDER / split / f"y_{split}.txt")
    volunteer_numbers = read_one_column(UCI_FOLDER / split / f"subject_{split}.txt")

    # Turn the activity numbers into names.
    activity_names = read_activity_names()
    activities = [activity_names[number] for number in activity_numbers]

    # Read the 9 signal files into a single 3D array (windows, 128 readings, 9 signals).
    signals = []
    for signal_name in SIGNAL_NAMES:
        signals.append(read_one_signal(split, signal_name))
    windows = np.stack(signals, axis=2).astype(np.float32)

    # Check each file has a row per window, so they line up.
    if not (len(windows) == len(activities) == len(volunteer_numbers)):
        raise ValueError(
            f"The files have different numbers of rows: signals {len(windows)}, "
            f"activities {len(activities)}, volunteers {len(volunteer_numbers)}"
        )

    # Build the labels table (matching the columns of our group data)
    labels = pd.DataFrame({
        "activity": activities,
        "participant": [f"UCI_{number:02d}" for number in volunteer_numbers],
        "position": "waist",       # All UCI data was collected with the phone worn on the waist.
        "platform": "android",     # All UCI data was collected on a Samsung Galaxy S II.
    })
    labels["recording_id"] = labels["participant"]   # UCI doesnt use recording IDs, so the volunteer is used instead

    # Only keep only the activities we collected and renames them to our naming standard.
    keep = labels["activity"].isin(ACTIVITIES.keys()).to_numpy()
    windows = windows[keep]
    labels = labels[keep].reset_index(drop=True)
    labels["activity"] = labels["activity"].map(ACTIVITIES)

    return windows, labels

# Runs and displays the split stats
if __name__ == "__main__":
    for split in ("train", "test"):
        windows, labels = load_uci_inertial(split)
        print(f"\n{split.upper()}")
        print(f"  Windows: {windows.shape}  (windows, readings, signals)")
        print(f"  Volunteers: {labels['participant'].nunique()}")
        print("  Windows per activity:")
        for activity, count in labels["activity"].value_counts().items():
            print(f"    {activity:<20} {count}")