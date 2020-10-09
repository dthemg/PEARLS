from pearls import Pearls
import numpy as np


def pure_sinusoid(f, fs, N, A=100):
    t = np.arange(N) / fs
    return A * np.exp(t * 2 * np.pi * 1j * f)


if __name__ == "__main__":
    fs = 44100
    signal = pure_sinusoid(400, fs, 1000)
    P = Pearls(
        signal=signal,
        lambda_=0.997,
        xi=1e5,
        H=2,
        fs=fs,
        A_int=1000,
        A_size=1000000,  # Might not need anymore?
        K_msecs=2,
        p1=4,
        p2=80,
        ss=1e-4,
        mgi=10,
    )

    P.initialize_algorithm(
        f_int=(200, 600),
        f_spacing=200
    )
