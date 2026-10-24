# Turns each window (2.56 seconds of data) into 84 summary features (mean, std, step rhythm, etc). 
# This is necessary as the ML models cannot interpret each window due to it having 128 readings x 9 signals.

import numpy as np
import pandas as pd

SAMPLING_RATE = 50

SIGNAL_NAMES = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]


def add_magnitudes(window):
    # Combines the 9 signals (total_acc(x,y,z), body_acc(x,y,z) and body_gyro(x,y,z)) to create 3 more signals, the magnitude of 3 signals: 
    
    # total_acc
    # body_acc
    # body_gyro) 
    
    # Using Pythagoras's theorum in 3D = sqrt(x² + y² + z²). Removing the effect the direction the phone is facing adds.
    
    # This is necessary as the models will be trained on 2 of the participants and tested on the 3rd.

    total_acc_mag = np.sqrt(window[:, 0] ** 2 + window[:, 1] ** 2 + window[:, 2] ** 2)
    body_acc_mag = np.sqrt(window[:, 3] ** 2 + window[:, 4] ** 2 + window[:, 5] ** 2)
    body_gyro_mag = np.sqrt(window[:, 6] ** 2 + window[:, 7] ** 2 + window[:, 8] ** 2)

    signals = {name: window[:, i] for i, name in enumerate(SIGNAL_NAMES)}
    signals["total_acc_mag"] = total_acc_mag
    signals["body_acc_mag"] = body_acc_mag
    signals["body_gyro_mag"] = body_gyro_mag
    return signals


def rhythm_features(values):
    # Frequency features for one signal, using a Fourier transform (FFT). FFT converts the signal from a second-by-second timeline 
    # into a map of the steady rhythms. This is done by splitting a signal into rhythms: how strong it is at 0.4 Hz, 0.8 Hz ... up to 25 Hz.
    
    # This is necessary as the rhythm is one of the strongest indicators at identifying which activity it is, walking upstairs is often slower and heavier
    # compared to walking downstairs which is often faster, with sharper landings.
    
    movement = values - values.mean()
    strength = np.abs(np.fft.rfft(movement)) ** 2
    rhythms_hz = np.fft.rfftfreq(len(values), d=1 / SAMPLING_RATE)

    strongest = strength[1:].argmax() + 1
    dominant_freq = rhythms_hz[strongest]
    spectral_energy = strength.sum() / len(values)
    return dominant_freq, spectral_energy


def features_for_one_window(window):
    # Calculate the 84 features for one window (128 readings x 9 signals).
    features = {}
    for name, values in add_magnitudes(window).items():
        features[f"{name}_mean"] = values.mean()
        features[f"{name}_std"] = values.std()
        features[f"{name}_min"] = values.min()
        features[f"{name}_max"] = values.max()
        features[f"{name}_energy"] = (values ** 2).mean()
        dominant_freq, spectral_energy = rhythm_features(values)
        features[f"{name}_dominant_freq"] = dominant_freq
        features[f"{name}_spectral_energy"] = spectral_energy
    return features


def extract_features(windows):
    # Calculate the features for every window. Returns a table with one row per window.
    rows = []
    for window in windows:
        rows.append(features_for_one_window(window))
    return pd.DataFrame(rows)