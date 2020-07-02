# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'

import numpy as np
from tqdm import tqdm

from penalty_update import update_penalty_factors
from dictionary_update import dictionary_update
from gradient_descent import proximial_gradient_update
from rls_update import rls_update

# https://dl.acm.org/doi/pdf/10.1109/TASLP.2016.2634118


"""
CURRENT STATUS:
TO DO:
* Re-structure code better
* Sketch on visualization
* Active block updates
* Do some basic optimization

IN PROGRESS:

DONE:
* Penalty parameter updates
* Dictionary learning scheme
* Initialization
* Proximal gradient descent
* RLS update
"""


# Create window length from forgetting factor
def get_window_length(forgetting_factor):
    return np.log(0.01) // np.log(forgetting_factor)


# Create new batch
def get_new_batch(time_vector, vector_length, frequency_matrix, sampling_frequency):
    batch_exponent = (
        time_vector.reshape(vector_length, 1)
        * frequency_matrix.flatten("F")
        * 2
        * np.pi
        / sampling_frequency
    )

    batch_exponent_no_phase = batch_exponent.copy()
    batch = np.exp(1j * batch_exponent.copy())
    return batch_exponent, batch_exponent_no_phase, batch


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
    float_dtype = "float64"
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
    pitch_candidates = np.arange(
        minimum_pitch, maximum_pitch + 1, init_freq_resolution, dtype=float_dtype
    )
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

    # Define 45 ms batch
    (batch_exponent, batch_exponent_no_phase, batch,) = get_new_batch(
        time_batch, batch_len, frequency_matrix, sampling_frequency
    )
    prev_batch = batch

    ##### DEFINE PENALTY WINDOW #####
    # Define the window length
    window_length = 50  # get_window_length(forgetting_factor)

    ##### INITIALIZE VARIABLES #####
    # Get first candidate vector
    batch_vector = batch[0, :][np.newaxis].T

    # Initial estimate of covariance matrix (R(t))
    cov_matrix = batch_vector * batch_vector.conj().T

    # Initial value of candidate value vector (r(t))
    cov_vector = signal[0] * batch_vector

    # Initialize filter weights
    coeffs_estimate = np.zeros((num_filter_coeffs, 1), dtype=complex_dtype)
    rls_filter = np.zeros((num_filter_coeffs, 1), dtype=complex_dtype)
    rls_filter_history = np.zeros(
        (num_filter_coeffs, signal_length), dtype=complex_dtype
    )

    # Pitch history
    pitch_history = np.zeros((num_pitch_candidates, signal_length))

    ##### PERFORM ALGORITHM #####

    for iter_idx, signal_value in enumerate(signal):
        # print("SAMPLE NUMBER:", iter_idx)

        # Store current estimates
        pitch_history[:, iter_idx] = pitch_candidates

        ##### SAMPLE SELECTION #####
        batch_idx = iter_idx % batch_len

        # Renew matrix if batch is filled
        if batch_idx == 0:

            prev_batch = batch
            upper_time_idx = min(signal_length, iter_idx + batch_len)
            time_batch = time[iter_idx:upper_time_idx]
            # If end of signal
            if upper_time_idx - batch_idx < batch_len:
                time_batch = np.append(
                    time_batch, np.zeros(batch_len - (upper_time_idx - smaple_idx))
                )

            (batch_exponent, batch_exponent_no_phase, batch,) = get_new_batch(
                time_batch, batch_len, frequency_matrix, sampling_frequency
            )

        # Vector of time frequency
        batch_vector = (
            batch[batch_idx, :][np.newaxis].conj().T
        )  # Feel like this should be after...

        sample = signal[iter_idx]

        # Update covariance estimate
        cov_matrix = (
            forgetting_factor * cov_matrix + batch_vector * batch_vector.conj().T
        )
        cov_vector = forgetting_factor * cov_vector + batch_vector * sample

        # Update penalty parameters
        if iter_idx >= window_length and (iter_idx + 1) % 40 == 0:
            penalty_factor_1, penalty_factor_2 = update_penalty_factors(
                batch,
                prev_batch,
                window_length,
                signal,
                iter_idx,
                batch_idx,
                forgetting_factor,
            )

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
            rls_filter_history[:, iter_idx] = rls_filter.ravel()

        ##### DICTIONARY LEARNING #####
        start_dictionary_learning_idx = 10
        horizon = 10
        if (
            iter_idx >= start_dictionary_learning_idx - 1
            and iter_idx % update_dictionary_interval == 0
        ):
            print("Dictionary learning")
            print(f"batch idx: {batch_idx}, iter_idx: {iter_idx}")

            # Find start and stop indicies for this batch
            start_idx_time = max(iter_idx - num_samples_pitch + 1, 1)
            stop_idx_time = min(iter_idx + horizon, signal_length)

            pitch_limit = init_freq_resolution / 2

            reference_signal = signal[start_idx_time : iter_idx + 1]

            # If necessary find start index of previous batch
            batch_start_idx = max(0, batch_idx - num_samples_pitch + 1)
            batch_stop_idx = min(batch_len - 1, batch_idx + horizon)

            if batch_idx - num_samples_pitch < 0:
                prev_batch_start_idx = batch_len - (num_samples_pitch - batch_idx) + 1
                temp_prev_batch = prev_batch

            else:
                prev_batch_start_idx = None
                temp_prev_batch = None

            # Compute dictionary update
            (
                batch,
                batch_exponent,
                batch_exponent_no_phase,
                prev_batch,
                pitch_candidates,
                rls_filter,
            ) = dictionary_update(
                rls_filter,
                reference_signal,
                pitch_limit,
                batch,
                batch_exponent,
                batch_exponent_no_phase,
                pitch_candidates,
                time,
                sampling_frequency,
                max_num_harmonics,
                num_pitch_candidates,
                start_idx_time,
                stop_idx_time,
                batch_idx,
                batch_start_idx,
                batch_stop_idx,
                temp_prev_batch,
                prev_batch_start_idx,
            )

    return rls_filter_history, pitch_history
