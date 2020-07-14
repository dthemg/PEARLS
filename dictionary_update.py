import numpy as np

MIN_PITCH_RATIO = 0.05
MIN_HARMONIC_RATIO = 0.2
DEFAULT_NUM_SEARCH_POINTS = 2 ** 20
GRID_TOLERANCE = 1  # Hz


def dictionary_update(
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
    start_index_time,
    stop_index_time,
    batch_idx,
    batch_start_idx,
    batch_stop_idx,
    prev_batch,
    prev_batch_start_idx,
):
    """Update the pitch frequency grid"""

    rls_filter_matrix = rls_filter.reshape(
        max_num_harmonics, num_pitch_candidates, order="F"
    )
    rows_to_change = np.arange(batch_start_idx, batch_stop_idx, dtype=int)[
        :, np.newaxis
    ]
    pitch_norms = np.linalg.norm(rls_filter_matrix, axis=0)
    batch_time = time[start_index_time:stop_index_time]

    if prev_batch is None:
        batch_update_idx = 0
    else:
        batch_update_idx = len(batch) - prev_batch_start_idx

    # Get pitch peaks
    peak_locations = _find_peak_locations(pitch_norms)
    peak_idxs = [i for i, is_peak in enumerate(peak_locations) if is_peak]

    # Do dictionary learning
    for peak_idx in peak_idxs:
        harmonic_amplitudes = abs(rls_filter_matrix[:, peak_idx])
        largest_harmonic = max(harmonic_amplitudes)

        significant_harmonics = (
            harmonic_amplitudes > MIN_HARMONIC_RATIO * largest_harmonic
        )
        highest_harmonic = (
            max([i for i, v in enumerate(significant_harmonics) if v]) + 1
        )

        pitch = pitch_candidates[peak_idx]
        pitch_update_range = np.array([pitch - pitch_limit, pitch + pitch_limit])

        updated_pitch = _interval_pitch_search(
            reference_signal, highest_harmonic, pitch_update_range, sampling_frequency,
        )

        # Update batch and pitch candidates
        pitch_candidates[peak_idx] = updated_pitch
        columns_to_change = np.arange(
            peak_idx * max_num_harmonics, (peak_idx + 1) * max_num_harmonics, dtype=int
        )

        new_batch_exponent_no_phase = np.outer(
            2 * np.pi * batch_time * updated_pitch / sampling_frequency,
            np.arange(1, max_num_harmonics + 1),
        )

        # Estimate new phase
        new_batch_exponent = _phase_update(
            reference_signal, new_batch_exponent_no_phase, max_num_harmonics
        )
        new_batch = np.exp(1j * new_batch_exponent)
        idx_update = batch_update_idx + len(rows_to_change)

        # Assign new values
        batch_exponent_no_phase[
            rows_to_change, columns_to_change
        ] = new_batch_exponent_no_phase[batch_update_idx:idx_update, :]
        batch_exponent[rows_to_change, columns_to_change] = new_batch_exponent[
            batch_update_idx:idx_update, :
        ]
        batch[rows_to_change, columns_to_change] = new_batch[
            batch_update_idx:idx_update, :
        ]

        if prev_batch is not None:
            prev_batch[prev_batch_start_idx:, columns_to_change] = new_batch[
                prev_batch_start_idx, :
            ]

    return (
        batch,
        batch_exponent,
        batch_exponent_no_phase,
        prev_batch,
        pitch_candidates,
        rls_filter,
    )


# Does not capture peaks at either end of spectrum
def _find_peak_locations(arr):
    is_peak = np.r_[False, arr[1:] > arr[:-1]] & np.r_[arr[:-1] > arr[1:], False]
    return is_peak & (arr > MIN_PITCH_RATIO * arr[1:-1].max())


def _interval_pitch_search(
    signal,
    highest_harmonic,
    search_range,
    sampling_frequency,
    num_search_points=DEFAULT_NUM_SEARCH_POINTS,
):

    frequency_grid = (
        np.arange(0, num_search_points) / num_search_points * sampling_frequency
    )
    signal_length = len(signal)

    # Interval edges
    a = np.argmax(frequency_grid > search_range[0])
    b = np.argmax(frequency_grid > search_range[1]) - 1
    center_freq_lower, center_freq_upper = frequency_grid[a], frequency_grid[b]

    m = (a + b) // 2

    # Zoom in until we are sufficiently close to a frequency
    while center_freq_upper - center_freq_lower > GRID_TOLERANCE:
        center_freq_lower = (frequency_grid[a] + frequency_grid[m]) / 2
        center_freq_upper = (frequency_grid[m] + frequency_grid[b]) / 2

        match_lower = frequency_match(
            signal,
            signal_length,
            sampling_frequency,
            center_freq_lower,
            highest_harmonic,
        )
        match_upper = frequency_match(
            signal,
            signal_length,
            sampling_frequency,
            center_freq_upper,
            highest_harmonic,
        )

        if match_lower > match_upper:
            b = m
        else:
            a = m

        m = (a + b) // 2

    return frequency_grid[(a + b) // 2]


def frequency_match(signal, signal_length, fs, f, highest_harmonic):
    arr_exp = -2j * np.pi * f * np.arange(signal_length) / fs
    match = 0
    for harmonic in np.arange(1, 1 + highest_harmonic):
        match += np.power(np.abs(np.dot(np.exp(arr_exp * harmonic), signal)), 2)
    return match


# Estimate phase
def _phase_update(signal, exponent, num_harmonics):
    signal_len = len(signal)

    for harmonic in range(num_harmonics):
        t_batch = np.exp(1j * exponent[:signal_len, harmonic])
        t_res = np.divide(signal, t_batch)
        phase_est = np.angle(np.mean(t_res))
        exponent[:, harmonic] = exponent[:, harmonic] + phase_est

    return exponent
