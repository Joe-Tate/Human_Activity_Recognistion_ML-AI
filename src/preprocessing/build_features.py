# Aggregates and calls the functions defined in 'features.py' to extract the features and put them in a file 
# so this script doesnt need to be re ran each time the model is.


import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing.load_uci_dataset import load_uci_inertial
from src.preprocessing.features import extract_features


UCI_FOLDER = PROJECT_ROOT / "data" / "processed" / "uci"
UCI_WINDOWS_FILE = UCI_FOLDER / "uci_windows.npz"
UCI_FEATURES_FILE = UCI_FOLDER / "uci_features.csv"

GROUP_FOLDER = PROJECT_ROOT / "data" / "processed" / "group"
GROUP_WINDOWS_FILE = GROUP_FOLDER / "group_windows.npz"
GROUP_FEATURES_FILE = GROUP_FOLDER / "group_features.csv"

# The label columns every window has (the same for both datasets)
LABEL_COLUMNS = ["split", "activity", "participant", "position", "platform", "recording_id"]


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def save_windows(path, windows, labels):
    # Save windows and their labels into an .npz file (numpys format for saving several arrays together). 
    # The deep learning models will load this because it uses the windows not the aggregated features.
    label_arrays = {}
    for column in labels.columns:
        label_arrays[column] = labels[column].to_numpy(dtype=str)
    np.savez_compressed(path, windows=windows, **label_arrays)


def make_feature_table(windows, labels):
    # Extract the features and put them next to their labels.
    print(f"  Extracting features from {len(windows):,} windows...")
    features = extract_features(windows)
    return pd.concat([labels.reset_index(drop=True), features], axis=1)


# --------------------------------------------------------------------------
# UCI dataset
# --------------------------------------------------------------------------

def build_uci():
    print("UCI dataset")

    # Load the train and test windows, labelling which split each window is in
    train_windows, train_labels = load_uci_inertial("train")
    test_windows, test_labels = load_uci_inertial("test")
    train_labels.insert(0, "split", "train")
    test_labels.insert(0, "split", "test")

    # Combines train and test together in one table (the "split" column keeps them apart)
    windows = np.concatenate([train_windows, test_windows])
    labels = pd.concat([train_labels, test_labels], ignore_index=True)

    # Save the windows for the deep learning models
    save_windows(UCI_WINDOWS_FILE, windows, labels)
    print(f"  Saved {UCI_WINDOWS_FILE.relative_to(PROJECT_ROOT)}")

    # Extract the features and save them for the machine learning models
    table = make_feature_table(windows, labels)
    table.to_csv(UCI_FEATURES_FILE, index=False)
    print(f"  Saved {UCI_FEATURES_FILE.relative_to(PROJECT_ROOT)}  "
          f"({len(table):,} rows: {(table['split'] == 'train').sum():,} train, "
          f"{(table['split'] == 'test').sum():,} test)")


# --------------------------------------------------------------------------
# Group dataset
# --------------------------------------------------------------------------


def build_group():
    print("\nGroup dataset")

    # Check the prerequisite group windows have been made by preprocess_group.py
    if not GROUP_WINDOWS_FILE.exists():
        print(f"  Skipped: {GROUP_WINDOWS_FILE.relative_to(PROJECT_ROOT)} not found.")
        print("  Run build_group_dataset.py and preprocess_group.py first, then run this again.")
        return

    # Load the windows and their labels
    data = np.load(GROUP_WINDOWS_FILE)
    windows = data["windows"]
    labels = pd.DataFrame()
    for column in LABEL_COLUMNS:
        labels[column] = data[column]

    # Extract the features and save them for the initial ML models
    table = make_feature_table(windows, labels)
    table.to_csv(GROUP_FEATURES_FILE, index=False)
    print(f"  Saved {GROUP_FEATURES_FILE.relative_to(PROJECT_ROOT)}  ({len(table):,} rows)")


if __name__ == "__main__":
    build_uci()
    build_group()