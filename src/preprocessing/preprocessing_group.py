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

# Butterworth low-pass filter used to keep slow changes and remove fast ones. [ https://www.sciencedirect.com/topics/engineering/butterworth-filter ]
# Changes faster than 20 are removed.
NOISE_FILTER = butter(3, 20, btype="low", fs=SAMPLING_RATE, output="sos")
GRAVITY_FILTER = butter(3, 0.3, btype="low", fs=SAMPLING_RATE, output="sos")

LABEL_COLUMNS = ["activity", "participant", "position", "platform", "recording_id"]
