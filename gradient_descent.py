import numpy as np


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
        new_coeffs = _soft_threshold(new_coeffs, penalty_factor_1 * step_size)

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
def _soft_threshold(vector, penalty_factor):
    thresh_vector = max((np.abs(vector) - penalty_factor).max(), 0)
    return (thresh_vector / (thresh_vector + penalty_factor)) * vector
