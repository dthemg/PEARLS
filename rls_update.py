import numpy as np

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