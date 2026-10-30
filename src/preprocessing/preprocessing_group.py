# This script turns the cleaned recordings into summary windows in exactly the same format as UCI’s Inertial Signals 
# allowing the results to be directly comparable. Also everything afterwards (features, models) treats them the same as UCI.


from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, medfilt, sosfiltfilt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "group" / "group_combined.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "group" / "group_windows.npz"

SAMPLING_RATE = 50
WINDOW_SIZE = 128     # readings per window (2.56 seconds)
WINDOW_STEP = 64      # move 64 readings each time, so windows overlap by 50% to match UCI

# Butterworth low-pass filter used to keep slow changes and remove fast ones. [https://www.sciencedirect.com/topics/engineering/butterworth-filter] changes
# faster than 20 are removed.
NOISE_FILTER = butter(3, 20, btype="low", fs=SAMPLING_RATE, output="sos")
GRAVITY_FILTER = butter(3, 0.3, btype="low", fs=SAMPLING_RATE, output="sos")

LABEL_COLUMNS = ["split", "activity", "participant", "position", "platform", "recording_id"]


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
 
def remove_noise(signal):
    # Uses a median filter, then a 20 Hz low-pass filter to remove noise from a signal.
    smoothed = medfilt(signal, kernel_size=3) # replaces a spikes with the median of 3 readings 
    return sosfiltfilt(NOISE_FILTER, smoothed)
 
 
def remove_gravity(total_acc):
    # Split an acceleration signal into gravity and movement then only return the movement. Gravity changes slowly (only when the phone tilts), 
    # therefore a 0.3 Hz low-pass filter keeps only gravity as done in the UCI experiment 
    # [https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf].
    padding = min(len(total_acc) - 1, 150)          # Adding extra padding reduces distortion at the ends
    gravity = sosfiltfilt(GRAVITY_FILTER, total_acc, padlen=padding)
    return total_acc - gravity
 

def cut_into_windows(signals):
    # Cut a (readings, 9) array into windows of 128 readings, moving 64 readings each time for the 50% overlap.
    windows = []
    for start in range(0, len(signals) - WINDOW_SIZE + 1, WINDOW_STEP):
        windows.append(signals[start:start + WINDOW_SIZE])
    return windows
 
 
def process_one_recording(recording):
    # Filter one recording and cut it into windows.
    acc_x = remove_noise(recording["acc_x"].to_numpy())
    acc_y = remove_noise(recording["acc_y"].to_numpy())
    acc_z = remove_noise(recording["acc_z"].to_numpy())
    gyro_x = remove_noise(recording["gyro_x"].to_numpy())
    gyro_y = remove_noise(recording["gyro_y"].to_numpy())
    gyro_z = remove_noise(recording["gyro_z"].to_numpy())
 
    body_acc_x = remove_gravity(acc_x)
    body_acc_y = remove_gravity(acc_y)
    body_acc_z = remove_gravity(acc_z)
 
    # Put the 9 signals side by side, in UCI's order: (readings, 9)
    nine_signals = np.column_stack([
        acc_x, acc_y, acc_z,
        body_acc_x, body_acc_y, body_acc_z,
        gyro_x, gyro_y, gyro_z,
    ])
 
    return cut_into_windows(nine_signals)
 
 
# --------------------------------------------------------------------------
# The main function
# --------------------------------------------------------------------------

 
def preprocess_group():
    print(f"Reading {INPUT_FILE.relative_to(PROJECT_ROOT)}")
    data = pd.read_csv(INPUT_FILE)
 
    all_windows = []
    all_labels = []
 
    # Goes through each recording one at a time
    for recording_id, recording in data.groupby("recording_id"):
        recording = recording.sort_values("seconds_elapsed") # Puts them in time order
 
        if len(recording) < WINDOW_SIZE:
            continue    # Skip recordings that are too short
 
        windows = process_one_recording(recording)
        labels = recording.iloc[0][LABEL_COLUMNS].to_dict()   # Gets the recording’s labels
        all_windows.extend(windows)
        all_labels.extend([labels] * len(windows))            # Add the windows and their labels to the lists
 
    windows = np.array(all_windows, dtype=np.float32)
    labels = pd.DataFrame(all_labels)
 
    # Save the windows and their labels together (the same format as uci_windows.npz)
    label_arrays = {}
    for column in LABEL_COLUMNS:
        label_arrays[column] = labels[column].to_numpy(dtype=str)
    np.savez_compressed(OUTPUT_FILE, windows=windows, **label_arrays)
 
    # Summary of processed data
    print(f"\nWindows: {windows.shape}  (windows, readings, signals)")
    total_acc_size = np.sqrt((windows[:, :, 0:3] ** 2).sum(axis=2)).mean()
    print(f"Average total acceleration: {total_acc_size:.3f} g   (expect about 1: gravity is still in it)")
    print(f"Average body acceleration:  {windows[:, :, 3:6].mean():.4f} g  (expect about 0: gravity removed)")
    print("\nWindows per activity:")
    print(labels["activity"].value_counts().to_string())
    print("\nWindows per participant and activity:")
    print(labels.groupby(["participant", "activity"]).size().unstack(fill_value=0).to_string())
    print(f"\nSaved {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
 
 
if __name__ == "__main__":
    preprocess_group()
 