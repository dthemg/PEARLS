import numpy as np
import matplotlib.pyplot as plt
from pearls import Pearls
from utils import r


def harmonic_signal(f, fs, N, H, A):
	t = np.arange(N) / fs
	sig = np.zeros(N, dtype=complex)
	for i in range(H):
		sig += A * np.sin(t * 2 * np.pi * f * (i + 1)) + 0j
	return sig


def plot_results(
	signal: np.ndarray, results: dict, P: Pearls, true_freq: float, true_H: int
):
	fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

	t = np.arange(len(signal)) / P.fs

	ax3.plot(t, np.real(signal))

	w_hat_hist = results["w_hat_hist"]
	freq_hist = results["freq_hist"]

	def get_weights(arr):
		return np.linalg.norm(arr.reshape(P.P, P.H), axis=1)

	p_weights = np.apply_along_axis(get_weights, 0, w_hat_hist)

	# Weight history
	ax1.plot(t, p_weights.T)
	ax2.plot(t, freq_hist.T)

	# Penalty parameters hist
	ax4.plot(t, results["p1_hist"])
	ax4.plot(t, results["p2_hist"])

	# Final prediction
	w_hat_final = w_hat_hist[:, -1].reshape(P.P, P.H)
	freq_final = r(freq_hist[:, -1])
	pred_signal = np.zeros(P.L, dtype="complex")

	hs = np.arange(P.H)
	for i, row in enumerate(w_hat_final):
		freq = freq_hist[i]
		for h in range(P.H):
			pred_signal += row[h] * np.exp(1j * 2 * np.pi * t * freq * (h + 1))

	ax5.plot(t, np.real(pred_signal))

	# Fourier transform of signal
	# breakpoint()
	s_fft = np.fft.fft(signal)[: len(signal) // 2 + 1]
	f_ax = np.linspace(0, P.fs / 2, num=(len(s_fft)))
	ax6.plot(f_ax, np.abs(s_fft))
	for i in range(true_H):
		ax6.axvline(x=true_freq * (i + 1), color="r")
	ax6.set_xlim([0, 3000])

	plt.show()


if __name__ == "__main__":
	fs = 44100
	true_H = 1
	true_freq = 400
	signal = harmonic_signal(f=true_freq, fs=44100, N=2000, H=true_H, A=10)
	signal += harmonic_signal(f=150, fs=44100, N=2000, H=true_H, A=50)
	P = Pearls(
		signal=signal,
		lambda_=0.97,
		xi=1e5,
		H=true_H,
		fs=fs,
		K_msecs=100,
		p1=0,
		p2=0,
		ss=1e-4,
		mgi=10,
		mu=0.1,
	)

	P.initialize_variables(f_int=(100, 500), f_spacing=50)
	results = P.run_algorithm()
	plot_results(signal, results, P, true_freq, true_H)
