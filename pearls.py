# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'

import numpy as np


# https://dl.acm.org/doi/pdf/10.1109/TASLP.2016.2634118

# # Matlab code translation...


# Create window length from forgetting factor
def get_window_length(forgetting_factor):
    return np.log(0.01) // np.log(forgetting_factor)


# Create new candidate A matrix
def get_new_candidates(
    time_vector, vector_length, frequency_matrix, sampling_frequency
):
    # Flatten MatLab-style
    candidates_exponent = (
        time_vector.reshape(vector_length, 1)
        * frequency_matrix.flatten("F")
        * 2
        * np.pi
        / sampling_frequency
    )

    candidates_exponent_no_phase = candidates_exponent
    candidates = np.exp(1j * candidates_exponent)
    return candidates_exponent, candidates_exponent_no_phase, candidates


def proximial_gradient_update(
    coeffs,
    cov_matrix_est,
    cov_vector,
    num_pitch_candidates,
    max_num_harmonics,
    penalty_factor_1,
    penalty_factor_2,
    max_gradient_iterations,
    step_size,
):
    first_harmonic_amplitudes = np.ones((num_pitch_candidates, 1))
    for _ in range(max_gradient_iterations):
        # Calculate gradient
        gradient = -cov_vector + cov_matrix_est @ coeffs
        new_coeffs = coeffs - step_size * gradient
        new_coeffs = soft_threshold(new_coeffs, penalty_factor_1 * step_size)

        # Update each pitch candidate
        for pitch_idx in range(num_pitch_candidates):
            harmonic_idxs = np.arange(
                pitch_idx * max_num_harmonics, (pitch_idx + 1) * max_num_harmonics
            )
            harmonic_coeffs = new_coeffs[harmonic_idxs]
            pitch_penalty_factor_2 = penalty_factor_2 * max(
                1, min(1000, 1 / (abs(first_harmonic_amplitudes[pitch_idx]) + 1e-5))
            )
            factor = max(
                np.linalg.norm(harmonic_coeffs, 2)
                - pitch_penalty_factor_2 * (step_size ** 2),
                0,
            )
            coeffs[harmonic_idxs] = (
                harmonic_coeffs
                * factor
                / (factor + pitch_penalty_factor_2 * (step_size ** 2))
            )

        first_harmonic_amplitudes = coeffs[
            0 : num_pitch_candidates * max_num_harmonics : max_num_harmonics
        ]

    return coeffs


def soft_threshold(vector, penalty_factor):

    thresh_vector = max((np.abs(vector) - penalty_factor).max(), 0)
    return (thresh_vector / (thresh_vector + penalty_factor)) * vector


def PEARLS(
    signal,
    forgetting_factor,
    smoothness_factor,
    max_num_harmonics,
    sampling_frequency,
    minimum_pitch,
    maximum_pitch,
    init_freq_resolution,
):
    var = 1

    signal_length = len(signal)

    ##### ADDITIONAL CONSTANTS #####
    # Number of samples for dictionary update
    num_samples_pitch = np.floor(45 * 1e-3 * sampling_frequency)

    # Length of dictionary
    history_len = 20

    # Penalty parameters
    penalty_factor_1 = 4
    penalty_factor_2 = 80

    # Gradient step-size
    step_size = 1e-4
    max_gradient_iterations = 20

    ##### INITIALIZE CANDIDATES #####
    # Initialize frequency candidates
    pitch_candidates = np.arange(minimum_pitch, maximum_pitch + 1, init_freq_resolution)
    num_pitch_candidates = len(pitch_candidates)
    num_filter_coeffs = num_pitch_candidates * max_num_harmonics

    # Define the frequency matrix
    frequency_matrix = (
        np.arange(1, max_num_harmonics + 1).reshape(max_num_harmonics, 1)
        * pitch_candidates
    )
    # Define time indexes
    time = np.arange(signal_length)

    # Define history indicies
    time_history = np.arange(history_len)

    # Define 45 ms candidates
    (
        candidates_exponent,
        candidates_exponent_no_phase,
        candidates,
    ) = get_new_candidates(
        time_history, history_len, frequency_matrix, sampling_frequency
    )

    ##### DEFINE PENALTY WINDOW #####
    # Define the window length
    window_length = get_window_length(forgetting_factor)

    ##### INITIALIZE VARIABLES #####
    # Get first candidate vector
    candidate = candidates[0, :][np.newaxis].T

    # Initial estimate of covariance matrix (R(t))
    cov_matrix_est = candidate * candidate.conj().T

    # Initial value of candidate value vector (r(t))
    cov_vector = signal[0] * candidate

    # Initialize filter weights
    coeffs_estimate = np.zeros((num_filter_coeffs, 1))  # Better variable name?
    rls_filter = np.zeros((num_filter_coeffs, 1))
    rls_filter_history = np.zeros((num_filter_coeffs, signal_length))

    # Pitch history
    pitch_history = np.zeros((num_pitch_candidates, signal_length))

    ##### PERFORM ALGORITHM #####

    for iter_idx, signal_value in enumerate(signal):
        print("SAMPLE NUMBER:", iter_idx)
        # Store current estimates
        pitch_history[:, iter_idx] = pitch_candidates

        ##### SAMPLE SELECTION #####
        history_idx = iter_idx % history_len

        # Vector of time frequency candidates
        candidate = candidates[iter_idx, :][
            np.newaxis
        ].T  # Feel like this should be after...

        # Renew candidate matrix if history is filled
        if history_idx == 0:
            prev_candidates = candidates
            upper_time_idx = min(signal_length, iter_idx + history_len)
            time_history = time[iter_idx:upper_time_idx]
            # If end of signal
            if upper_time_idx - history_idx < history_len:
                time_history = np.append(
                    time_history, np.zeros(history_len - (upper_time_idx - smaple_idx))
                )

            (
                candidates_exponent,
                candidates_exponent_no_phase,
                candidates,
            ) = get_new_candidates(
                time_history, history_len, frequency_matrix, sampling_frequency
            )

        sample = signal[iter_idx]

        # Update covariance estimate
        cov_matrix_est = (
            forgetting_factor * cov_matrix_est + candidate * candidate.conj().T
        )
        cov_vector = forgetting_factor * cov_vector + candidate * sample

        # SKIP UPDATING PENALTY PARAMETERS...
        # update_penalties( ... )
        # SKIP DO ACTIVE UPDATE
        # update_actives(...)

        # VERIFIED UP TO THIS POINT

        # DO THE GOOD STUFF :D

        # Update filter estimates
        coeffs_estimate = proximial_gradient_update(
            coeffs_estimate,
            cov_matrix_est,
            cov_vector,
            num_pitch_candidates,
            max_num_harmonics,
            penalty_factor_1,
            penalty_factor_2,
            max_gradient_iterations,
            step_size,
        )

    # var = candidate, candidates

    # To return something...

    filter_history = []
    candidate_frequency_history = []
    return filter_history, candidate_frequency_history, var
