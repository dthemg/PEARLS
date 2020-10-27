import numpy as np
import matplotlib.pyplot as plt
from pearls import Pearls


def pure_sinusoid(f, fs, N, A=100):
	t = np.arange(N) / fs
	return A * np.exp(t * 2 * np.pi * 1j * f)


def plot_results(results):
	fig, (ax1, ax2) = plt.subplots(2, 1)

	ax2.plot(abs(results["freq_hist"].T))
	ax1.plot(abs(results["w_hat_hist"].T))

	plt.show()


if __name__ == "__main__":
	fs = 44100
	signal = pure_sinusoid(400, fs, 1000)
	P = Pearls(
		signal=signal,
		lambda_=0.97,
		xi=1e5,
		H=1,
		fs=fs,
		K_msecs=0.1,
		p1=4,
		p2=80,
		ss=1e-5,
		mgi=10,
	)

	P.initialize_variables(f_int=(200, 600), f_spacing=200)
	results = P.run_algorithm()

	plot_results(results)
