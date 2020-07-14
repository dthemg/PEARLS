import os
import json

from scipy.io import wavfile
import numpy as np

import pearls


def save_data(data, filename):
    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", filename + ".txt")
    print(f"Saving {save_path}...")
    np.savetxt(save_path, data, delimiter=",")


def save_settings(settings, filename):
    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", filename + ".json")
    print(f"Saving {save_path}")
    with open(save_path, "w") as f:
        json.dump(settings, f)


if __name__ == "__main__":
    """Load data"""
    DATA_PATH = os.path.join("data", "bach_air.wav")
    sample_rate, data = wavfile.read(DATA_PATH, mmap=False)

    """Test code"""
    N = 100
    S = 10000
    signal = data[:, 0]


    """ Made up data """
    freq = 225
    sample_rate = 44100
    time = np.arange(10000) / sample_rate
    signal = np.sin(time * 2 * np.pi * freq)
    #signal[5000:] = np.sin(time[5000:]/sample_rate * 2 * np.pi * 200)

    forgetting_factor = 0.995
    smoothness_factor = 1e4
    max_num_harmonics = 1
    sampling_frequency = sample_rate
    minimum_pitch = 50
    maximum_pitch = 500
    initial_frequency_resolution = 100

    # Calculate pitch estimates
    rls_filter_hist, pitch_hist = pearls.PEARLS(
        signal,
        forgetting_factor,
        smoothness_factor,
        max_num_harmonics,
        sampling_frequency,
        minimum_pitch,
        maximum_pitch,
        initial_frequency_resolution,
    )

    # Save results
    save_data(rls_filter_hist, "rls_history")
    save_data(pitch_hist, "pitch_est_history")
    save_data(signal, "input_signal")

    settings = {
        "max_num_harmonics": max_num_harmonics,
        "minimum_pitch": minimum_pitch,
        "maximum_pitch": maximum_pitch,
        "initial_frequency_resolution": initial_frequency_resolution,
        "forgetting_factor": forgetting_factor,
        "sampling_frequency": sample_rate,
        "smoothness_factor": smoothness_factor,
    }
    save_settings(settings, "input_settings")
