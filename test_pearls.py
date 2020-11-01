import numpy as np
import matplotlib.pyplot as plt
from pearls import Pearls


def harmonic_signal(f, fs, N, H, A):
	t = np.arange(N) / fs
	sig = np.zeros(N, dtype=complex)
	for i in range(H):
		sig += A * np.sin(t * 2 * np.pi * f * (i + 1)) + 0j
	return sig


def plot_results(signal: np.ndarray, results: dict, P: Pearls):
	fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)

	t = np.arange(len(signal)) / P.fs

	ax3.plot(t, np.real(signal))

	w_hat_hist = results["w_hat_hist"]

	def get_weights(arr):
		return np.linalg.norm(arr.reshape(P.P, P.H), axis=1)

	p_weights = np.apply_along_axis(get_weights, 0, w_hat_hist)

	ax1.plot(t, p_weights.T)
	ax2.plot(t, abs(results["freq_hist"].T))

	ax4.plot(t, results["p1_hist"])
	ax4.plot(t, results["p2_hist"])

	plt.show()


if __name__ == "__main__":
	fs = 44100
	signal = harmonic_signal(f=600, fs=44100, N=1000, H=3, A=1000)
	P = Pearls(
		signal=signal,
		lambda_=0.98,
		xi=1e5,
		H=3,
		fs=fs,
		K_msecs=10,
		p1=200,
		p2=1000,
		ss=1e-4,
		mgi=10,
		mu=0.1,
	)

	P.initialize_variables(f_int=(200, 600), f_spacing=200)
	results = P.run_algorithm()
	breakpoint()
	plot_results(signal, results, P)
