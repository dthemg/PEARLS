# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'

import numpy as np
from tqdm import tqdm

# https://dl.acm.org/doi/pdf/10.1109/TASLP.2016.2634118


"""
CURRENT STATUS:
TO DO:
* Penalty parameter updates
* Re-structure code better
* Sketch on visualization
* Active block updates
* Do some basic optimization

IN PROGRESS:
* Dictionary learning scheme

DONE:
* Initialization
* Proximal gradient descent
* RLS update
"""

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


# Do gradient descent for coefficients
def proximial_gradient_update(
    coeffs,
    cov_matrix,
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
        gradient = -cov_vector + cov_matrix @ coeffs
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


# Thresholding function used in gradient descent
def soft_threshold(vector, penalty_factor):
    thresh_vector = max((np.abs(vector) - penalty_factor).max(), 0)
    return (thresh_vector / (thresh_vector + penalty_factor)) * vector


# Update RLS filters
def rls_update(
    rls_filter, cov_matrix, cov_vector, max_num_harmonics, smoothness_factor
):
    num_pitch_candidates = int(rls_filter.size / max_num_harmonics)
    penalty_matrix = smoothness_factor * np.eye(max_num_harmonics)

    all_candidates_idxs = range(num_pitch_candidates * max_num_harmonics)

    for pitch_idx in range(num_pitch_candidates):
        harmonic_idxs = np.in1d(
            all_candidates_idxs,
            np.arange(
                pitch_idx * max_num_harmonics, (pitch_idx + 1) * max_num_harmonics
            ),
        )
        harmonic_rows_cov = cov_matrix[harmonic_idxs, :]
        harmonic_others = harmonic_rows_cov[:, ~harmonic_idxs]
        harmonic_pitch = harmonic_rows_cov[:, harmonic_idxs]

        harmonic_vector_cov = cov_vector[harmonic_idxs]
        harmonic_vector_cov = harmonic_vector_cov - np.dot(
            harmonic_others, rls_filter[~harmonic_idxs]
        )

        harmonic_matrix_tilde = harmonic_pitch + penalty_matrix
        harmonic_vector_tilde = harmonic_vector_cov + np.dot(
            smoothness_factor, rls_filter[harmonic_idxs]
        )

        # Which function should be used? lstsq finds minimal norm solution...
        rls_filter[harmonic_idxs], _, _, _ = np.linalg.lstsq(
            harmonic_matrix_tilde, harmonic_vector_tilde, rcond=None
        )

    return rls_filter


def dictionary_update(
    rls_filter,
    reference_signal,
    pitch_limit,
    candidates,
    candidates_exponent,
    candidates_exponent_no_phase,
    pitch_candidates,
    time,
    sampling_frequency,
    max_num_harmonics,
    num_pitch_candidates,
    start_index_time,
    stop_index_time,
    batch_start_idx,
    batch_stop_idx,
    prev_candidates,
    prev_batch_start_idx,
):
    """Update the pitch frequency grid"""
    rls_filter_matrix = rls_filter.reshape(
        max_num_harmonics, num_pitch_candidates, order="F"
    )
    # Verified up to here.


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
    complex_dtype = "complex_"
    signal_length = len(signal)
    update_dictionary_interval = 40

    ##### ADDITIONAL CONSTANTS #####
    # Number of samples for dictionary update
    num_samples_pitch = 40  # np.floor(45 * 1e-3 * sampling_frequency)

    # Length of dictionary
    batch_len = 50

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

    # Define batch indicies
    time_batch = np.arange(batch_len)

    # Define 45 ms candidates
    (
        candidates_exponent,
        candidates_exponent_no_phase,
        candidates,
    ) = get_new_candidates(time_batch, batch_len, frequency_matrix, sampling_frequency)
    prev_candidates = candidates

    ##### DEFINE PENALTY WINDOW #####
    # Define the window length
    window_length = get_window_length(forgetting_factor)

    ##### INITIALIZE VARIABLES #####
    # Get first candidate vector
    candidate = candidates[0, :][np.newaxis].T

    # Initial estimate of covariance matrix (R(t))
    cov_matrix = candidate * candidate.conj().T

    # Initial value of candidate value vector (r(t))
    cov_vector = signal[0] * candidate

    # Initialize filter weights
    coeffs_estimate = np.zeros((num_filter_coeffs, 1), dtype=complex_dtype)
    rls_filter = np.zeros((num_filter_coeffs, 1), dtype=complex_dtype)
    rls_filter_batch = np.zeros((num_filter_coeffs, signal_length), dtype=complex_dtype)

    # Pitch history
    pitch_history = np.zeros((num_pitch_candidates, signal_length))

    ##### PERFORM ALGORITHM #####

    for iter_idx, signal_value in enumerate(signal):
        # print("SAMPLE NUMBER:", iter_idx)

        # Store current estimates
        pitch_history[:, iter_idx] = pitch_candidates

        ##### SAMPLE SELECTION #####
        batch_idx = iter_idx % batch_len

        # Renew candidate matrix if batch is filled
        if batch_idx == 0:

            prev_candidates = candidates
            upper_time_idx = min(signal_length, iter_idx + batch_len)
            time_batch = time[iter_idx:upper_time_idx]
            # If end of signal
            if upper_time_idx - batch_idx < batch_len:
                time_batch = np.append(
                    time_batch, np.zeros(batch_len - (upper_time_idx - smaple_idx))
                )

            (
                candidates_exponent,
                candidates_exponent_no_phase,
                candidates,
            ) = get_new_candidates(
                time_batch, batch_len, frequency_matrix, sampling_frequency
            )

        # Vector of time frequency candidates
        candidate = (
            candidates[batch_idx, :][np.newaxis].conj().T
        )  # Feel like this should be after...

        sample = signal[iter_idx]

        # Update covariance estimate
        cov_matrix = forgetting_factor * cov_matrix + candidate * candidate.conj().T
        cov_vector = forgetting_factor * cov_vector + candidate * sample

        # SKIP UPDATING PENALTY PARAMETERS...
        # update_penalties( ... )
        # SKIP DO ACTIVE UPDATE
        # update_actives(...)

        ##### UPDATE COEFFICIENTS ######
        coeffs_estimate = proximial_gradient_update(
            coeffs_estimate,
            cov_matrix,
            cov_vector,
            num_pitch_candidates,
            max_num_harmonics,
            penalty_factor_1,
            penalty_factor_2,
            max_gradient_iterations,
            step_size,
        )

        ##### UPDATE RLS FILTER #####
        # 100 -> 10
        start_update_rls_idx = 9  # Should be 100 - 1
        if iter_idx > start_update_rls_idx:
            rls_filter = rls_update(
                rls_filter, cov_matrix, cov_vector, max_num_harmonics, smoothness_factor
            )

        ##### DICTIONARY LEARNING #####
        start_dictionary_learning_idx = 10
        horizon = 10
        if (
            iter_idx >= start_dictionary_learning_idx - 1
            and iter_idx % update_dictionary_interval == 0
        ):
            print("Dictionary learning!")
            print(f"batch idx: {batch_idx}, iter_idx: {iter_idx}")

            # Find start and stop indicies for this batch
            start_idx = max(iter_idx - num_samples_pitch, 1)  # good
            stop_idx = min(iter_idx + horizon, signal_length)  # good

            pitch_limit = init_freq_resolution / 2  # good

            reference_signal = signal[start_idx : iter_idx + 1]  # good

            # If necessary find start index of previous batch... but we shouldn't do this
            batch_start_idx = max(0, batch_idx - num_samples_pitch + 1)  # good
            batch_stop_idx = min(batch_len, batch_idx + horizon) - 1  # good

            if batch_idx - num_samples_pitch < 0:
                prev_batch_start_idx = batch_len - (num_samples_pitch - iter_idx)
                prev_cands = prev_candidates
            else:
                prev_batch_start_idx = None
                prev_cands = None

            # Compute dictionary update... just awful...
            a = 4
            (
                candidates,
                candidates_exponent,
                candidates_exponent_no_phase,
                prev_candidates,
                pitch_candidates,
                time,
                sampling_frequency,
                max_num_harmonics,
                num_pitch_candidates,
                start_idx,
                stop_idx,
                batch_idx,
            ) = dictionary_update(
                rls_filter,
                reference_signal,
                pitch_limit,
                candidates,
                candidates_exponent,
                candidates_exponent_no_phase,
                pitch_candidates,
                time,
                sampling_frequency,
                max_num_harmonics,
                num_pitch_candidates,
                # Sort out below parameters...
                start_idx,
                stop_idx,
                batch_start_idx,
                batch_stop_idx,
                prev_cands,
                prev_batch_start_idx,
            )

    # To return something...

    filter_batch = []
    candidate_frequency_batch = []
    return filter_batch, candidate_frequency_batch, var
