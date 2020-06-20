import numpy as np

MIN_PITCH_RATIO = 0.05
MIN_HARMONIC_RATIO = 0.2
DEFAULT_NUM_SEARCH_POINTS = 2 ** 20


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
    batch_start_idx,
    batch_stop_idx,
    prev_batch,
    prev_batch_start_idx,
):
    """Update the pitch frequency grid"""
    rls_filter_matrix = rls_filter.reshape(
        max_num_harmonics, num_pitch_candidates, order="F"
    )
    # Verified up to here.
    pitch_norms = np.linalg.norm(rls_filter_matrix, axis=0)

    if prev_batch is None:
        batch_for_est = batch[batch_start_idx:batch_stop_idx, :]
    else:
        np.concatenate(
            (prev_batch[prev_batch_start_idx:, :], batch[:batch_stop_idx, :]), axis=0,
        )
        batch_for_est = batch[batch_start_idx:batch_stop_idx, :]

    # Get pitch peaks
    peak_locations = _find_peak_locations(pitch_norms)
    peak_idxs = [i for i, is_peak in enumerate(peak_locations) if is_peak]

    # If no peaks are found, skip dictionary learning (Probably unnecessary...)
    if (~peak_locations).all():
        return (
            batch,
            batch_exponent,
            batch_exponent_no_phase,
            prev_batch,
            pitch_candidates,
            rls_filter,
        )

    # Do dictionary learning
    for peak_idx in peak_idxs:
        harmonic_amplitudes = abs(rls_filter_matrix[:, peak_idx])
        largest_harmonic = max(harmonic_amplitudes)

        significant_harmonics = (
            harmonic_amplitudes > MIN_HARMONIC_RATIO * largest_harmonic
        )
        highest_harmonic = max([i for i, v in enumerate(significant_harmonics) if v])

        pitch = pitch_candidates[peak_idx]
        pitch_update_range = np.array([pitch - pitch_limit, pitch + pitch_limit])

        updated_pitch = _interval_pitch_search(
            reference_signal, highest_harmonic, pitch_update_range, sampling_frequency,
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
    m = np.floor(a + b, 2)
    _lambda = m - 1
    tol = 3
    counter = 0

    while b - a > tol:
        f_lambda = f(signal, signal_length, frequency_grid, _lambda, highest_harmonic)


def f(signal, signal_length, num_search_points, k, highest_harmonic):
    val = 0
    for harmonic in arange(1, highest_harmonic + 1):
        pass
    return val


def _phase_update():
    pass
