import numpy as np
import pandas as pd
from tqdm import tqdm

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


def get_window_length(lambda_: float):
    """Get penalty window length
    lambda_:     forgetting factor
    """
    return int(np.log(0.01) / np.log(lambda_))


def ct(arr):
    """Conjugate transpose
    """
    return arr.conj().T


def c(arr):
    """Conjugate"""
    return arr.conj()


def as_column(arr):
    """Convert array to column vector"""
    return arr.reshape(len(arr), 1)


def as_row(arr):
    """Convert array to row vector"""
    return arr.reshape(1, len(arr))


class Pearls:
    def __init__(
        self,
        signal: np.array,
        lambda_: float,
        xi: float,
        H: int,
        fs: float,
        A_int: int,
        A_size: int,
        K_msecs: float,
        p1: float,
        p2: float,
        ss: float,
        mgi: int,
    ):
        """
        signal:     input signal (1 channel)
        lambda:     forgetting factor
        xi:         smoothing factor
        H:          maximum number of harmonics
        fs:         sampling frequency
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
        self.A_int = A_int
        self.A_size = A_size
        self.K = int(np.floor(K_msecs * 1e-3 * fs))
        self.p1 = p1
        self.p2 = p2
        self.ss = ss
        self.mgi = mgi
        self.w_len = get_window_length(self.lambda_)
        self.t_idx = self.K - 1
        self.t = np.arange(self.L + self.K) / self.fs

    def initialize_variables(self, f_int: tuple, f_spacing: float):
        """
        f_int:      (min, max) frequency search interval
        f_spacing:  initial spacing between frequencies
        """
        # Initialize frequency matrix
        ps = np.arange(f_int[0], f_int[1] + 0.001, f_spacing, dtype=self.float_dtype)
        n_p = len(ps)
        self.f_mat = as_column(np.arange(1, self.H + 1)) * ps
        self.f_active = [True] * n_p

        # Initialize starting index at num values for pitch
        self.s_idx = 0

        # Initialize pitch-time matrix/vector
        a_no_t = as_column(np.exp(2 * np.pi * 1j * self.f_mat.ravel()))
        self.A = a_no_t * self.t[: self.K]
        self.a = a_no_t * self.t[self.t_idx]

        # Initialize covariance matrix/vector
        self.R = self.a @ ct(self.a)
        self.r = self.s[self.s_idx] * np.conj(self.a)

        # Initialize RLS filter coefficients
        n_coef = n_p * self.H
        self.rls = np.zeros(n_coef, dtype=self.complex_dtype)

        # Initialize result history
        self.rls_hist = np.zeros((n_coef, self.L), dtype=self.complex_dtype)
        self.freq_hist = np.zeros((n_p, self.L), dtype=self.complex_dtype)

    def update_a(self, fs_updated: bool):
        """Update a vector and A matrix
        f_updated:  If updates has been done to the frequency matrix
        """
        a_no_t = as_column(np.exp(2 * np.pi * 1j * self.f_mat.ravel()))
        self.a = a_no_t * self.t[self.t_idx]

        if fs_updated:
            self.A = a_no_t * self.t[self.t_idx - K : self.t_idx]
        else:
            self.A = np.roll(self.A, -1, axis=1)
            self.A[:, -1] = self.a.ravel()

    def update_covariance(self, s_val: float):
        """Update covariance r vector and R matrix
        s_val:      signal value
        """
        self.R = self.lambda_ * self.R + self.a @ ct(self.a)
        self.r = self.lambda_ * self.r + s_val * c(self.a)

    def run_algorithm(self):
        """Run PEARLS algorithm through signal"""

        # If frequency matrix has been updated
        fs_upd = False

        for idx in range(self.L):
            self.s_idx = idx
            self.t_idx = idx + self.K

            self.update_a(fs_upd)
            self.update_covariance(self.s[idx])
