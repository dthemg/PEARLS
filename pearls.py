# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'

import numpy as np
from tqdm import tqdm
from collections import namedtuple

from penalty_update import update_penalty_factors
from dictionary_update import dictionary_update
from gradient_descent import proximial_gradient_update
from rls_update import rls_update

# https://dl.acm.org/doi/pdf/10.1109/TASLP.2016.2634118
# http://www.maths.lu.se/fileadmin/maths/personal_staff/Andreas_Jakobsson/openPEARLS.pdf


"""
CURRENT STATUS:
TO DO:
* Sketch on visualization
* Do some basic optimization

IN PROGRESS:
* Active block updates
* Bugfixes

DONE:
* Penalty parameter updates
* Dictionary learning scheme
* Initialization
* Proximal gradient descent
* RLS update
"""


# Create window length from forgetting factor
def get_window_length(forgetting_factor):
    return int(np.log(0.01) / np.log(forgetting_factor))

# Conjugate transpose
def ct(arr):
    return arr.conj().T

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

def as_column(arr):
    return arr.reshape(len(arr), 1)


class Pearls:
    def __init__(
        self,
        signal: np.array,
        lambda_: float,
        xi: float,
        H: int,
        fs: float,
        f_int: tuple,
        f_spacing: float,
        A_int: int,
        A_size: int,
        K_msecs: float,
        p1: float,
        p2: float,
        ss: float,
        mgi: int
    ):
        """
        signal:     input signal (1 channel)
        lambda:     forgetting factor
        xi:         smoothing factor
        H:          maximum number of harmonics
        fs:         sampling frequency
        f_int:      (min, max) frequency search interval
        f_spacing:  initial spacing between frequencies
        A_int:      interval for frequency dictionary update
        A_size:     size of frequency dictionary in samples
        K_msecs:    number of milliseconds to produce a pitch
        p1:         penalty factor 1
        p2:         penalty factor 2
        ss:         step size for gradient descent
        mgi:        max gradient iterations for gradient descent
        """
        self.s = signal
        self.complex_dtype = "complex_"
        self.float_dtype = "float64"
        self.L = len(signal)

        self.lambda_ = lambda_
        self.xi = xi
        self.H = H
        self.fs = fs
        self.f_int = f_int
        self.f_spacing = f_spacing
        self.A_int = A_int
        self.A_size = A_size
        self.K = int(np.floor(K_msecs * 1e-3 * fs))
        self.p1 = p1
        self.p2 = p2
        self.ss = ss
        self.mgi = mgi
        self.t = np.arange(self.L) / self.fs
        self.w_len = get_window_length(self.lambda_)

    def initialize_algorithm(self):
        # Initialize frequency candidates & frequency matrix
        self.p = np.arange(self.f_int[0], self.f_int[1] + 0.01, self.f_spacing, dtype=self.float_dtype)
        self.n_p = len(self.p)
        self.n_coef = self.n_p * self.H
        self.f_mat = as_column(np.arange(1, self.H + 1)) * self.p

        # Initialize A as shape (number of pitches * harmonics, samples to generate a pitch)
        freqs = self.f_mat.ravel()
        self.s_idx = self.K
        freqs = self.f_mat.ravel() * as_column(self.t[:self.s_idx])
        t_pitch = self.t[:self.s_idx]
        self.A = np.exp(2 * np.pi * 1j * freqs * t_pitch)
        
        # Initialize covariance matrix
        self.R = np.zeros((n_coef, n_coef))
        self.r = np.zeros(n_coef)





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
    update_dictionary_interval = 100

    # Number of samples for dictionary update
    num_samples_pitch = int(np.floor(45 * 1e-3 * sampling_frequency))  # 45

    # Length of dictionary
    batch_len = 2000

    # Penalty parameters
    penalty_factor_1 = 4
    penalty_factor_2 = 80

    # Gradient step-size
    step_size = 1e-4
    max_gradient_iterations = 20

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

    # Define 45 ms batch
    (batch_exponent, batch_exponent_no_phase, batch,) = get_new_batch(
        np.arange(batch_len), batch_len, frequency_matrix, sampling_frequency
    )
    prev_batch = batch

    # Define the window length for penalty update
    window_length = get_window_length(forgetting_factor)

    # Initialize coariances
    batch_vector = batch[0, :][np.newaxis].T
    cov_matrix = batch_vector * batch_vector.conj().T
    cov_vector = signal[0] * batch_vector

    # Initialize filter weights
    coeffs_estimate = np.zeros((num_filter_coeffs, 1), dtype=complex_dtype)
    rls_filter = np.zeros((num_filter_coeffs, 1), dtype=complex_dtype)
    rls_filter_history = np.zeros(
        (num_filter_coeffs, signal_length), dtype=complex_dtype
    )

    # Pitch history
    pitch_history = np.zeros((num_pitch_candidates, signal_length))

    # Main algorithm
    for iter_idx, signal_value in enumerate(tqdm(signal)):
        # Store current estimates
        pitch_history[:, iter_idx] = pitch_candidates

        # Renew matrix if batch is filled
        batch_idx = iter_idx % batch_len
        if batch_idx == 0:
            prev_batch = batch.copy()
            upper_time_idx = min(signal_length, iter_idx + batch_len)
            time_batch = time[iter_idx:upper_time_idx]
            # If end of signal
            if upper_time_idx - batch_idx < batch_len:
                time_batch = np.append(
                    time_batch, np.zeros(batch_len - (upper_time_idx - batch_idx))
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
        if iter_idx >= window_length and iter_idx % 400 == 0:
            penalty_factor_1, penalty_factor_2 = update_penalty_factors(
                batch,
                prev_batch,
                window_length,
                signal,
                iter_idx,
                batch_idx,
                forgetting_factor,
            )

        # Update RLS coefficients
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

        # Update RLS Filter
        start_update_rls_idx = 99
        if iter_idx > start_update_rls_idx:
            rls_filter = rls_update(
                rls_filter, cov_matrix, cov_vector, max_num_harmonics, smoothness_factor
            )
            rls_filter_history[:, iter_idx] = rls_filter.ravel()

        # Update frequency candidates
        start_dictionary_learning_idx = 2000
        horizon = 600
        if (
            iter_idx >= start_dictionary_learning_idx - 1
            and (iter_idx % update_dictionary_interval) == 0
        ):
            # Find start and stop indicies for this batch
            start_idx_time = max(iter_idx - num_samples_pitch, 0)
            stop_idx_time = min(iter_idx + horizon, signal_length)

            pitch_limit = init_freq_resolution / 2

            reference_signal = signal[start_idx_time:iter_idx]

            # If necessary find start index of previous batch
            batch_start_idx = max(batch_idx - num_samples_pitch, 0)
            batch_stop_idx = min(batch_idx + horizon, batch_len)

            if batch_idx - num_samples_pitch < 0:
                prev_batch_start_idx = batch_len + batch_idx - num_samples_pitch
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
