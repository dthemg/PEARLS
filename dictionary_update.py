import numpy as np

MIN_PITCH_RATIO = 0.05
MIN_HARMONIC_RATIO = 0.2

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
            (prev_batch[prev_batch_start_idx:, :], batch[:batch_stop_idx, :]),
            axis=0,
        )
        batch_for_est = batch[batch_start_idx:batch_stop_idx, :]

    # Get pitch peaks
    peak_locations = _find_peak_locations(pitch_norms)
    peak_idxs = [i for i, peak in enumerate(is_peak) if peak]

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
        a = 4
        harmonic_amplitudes = abs(rls_filter_matrix[:, peak_idx])
        largest_harmonic = max(harmonic_amplitudes)
        significant_harmonics = harmonic_amplitudes > 0.2 * largest_harmonic
        
        

        # Seems to work up to here


# Does not capture peaks at either end of spectrum
def _find_peak_locations(arr):
    is_peak = np.r_[False, arr[1:] > arr[:-1]] & np.r_[arr[:-1] > arr[1:], False]
    return is_peak & (arr > MIN_PITCH_RATIO * arr[1:-1].max())

def _phase_update():
    pass

def _interval_pitch_search():
    pass