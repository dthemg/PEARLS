import numpy as np


def update_penalty_factors(
    batch, prev_batch, window_length, signal, iter_idx, batch_idx, forgetting_factor
):
    signal_indices = np.arange(iter_idx - (window_length - 1), iter_idx + 1)

    if window_length > batch_idx + 1:
        diff = window_length - batch_idx - 1
        signal_section = signal[signal_indices]
        forgetting_vec = np.power(forgetting_factor, np.flip(np.arange(window_length)))
        prev_batch_section = prev_batch[-(diff):, :]
        batch_section = batch[: batch_idx + 1, :]
        max_norm = np.max(
            np.abs(
                np.dot(
                    prev_batch_section.conj().T,
                    signal_section[:diff] * forgetting_vec[:diff],
                )
                + np.dot(
                    batch_section.conj().T,
                    signal_section[diff:] * forgetting_vec[diff:],
                )
            )
        )
    else:
        batch_section = np.arange(batch_idx - (window_length - 1), batch_idx)
        max_norm = np.max(
            np.abs(
                np.dot(
                    batch[batch_section, :].conj().T,
                    signal[batch_section]
                    * np.power(
                        forgetting_factor, np.flip(np.arange(window_length - 1))
                    ),
                )
            )
        )

    penalty_factor_1 = 0.1 * max_norm
    penalty_factor_2 = 1 * max_norm
    return penalty_factor_1, penalty_factor_2
