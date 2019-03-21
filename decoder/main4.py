#!/bin/python3

import numpy as np
from commpy import QAMModem, awgn
from scipy import stats
import matplotlib.pyplot as plt

from decoder import LDPC

M = 16 # Use QAM-16

modem = QAMModem(M)

snr_dB = 10
print(f"SNR = {snr_dB} dB")

H = np.loadtxt('../H.txt', usecols=range(512), dtype=int)
q = 16
p = 4
n = H.shape[1]
R = (n - H.shape[0]) / n
l = np.count_nonzero(H[0, :])

print("H shape is", H.shape)
print(f"q = {q}, R = {R}, n = {n}, l = {l}")

ldpc = LDPC(q, H)

N = 32 * n # Length of transmission

def to_q(vs, N, p):
    def f(v):
        return sum([(v[i] << i) for i in range(p)])

    return np.array(list(map(f, [v for v in vs.reshape((N, p))])))


def count_ber(c, n):
        non_zero_sum = 0

        for i in c:
                bin_str = bin(i)[2:]
                non_zero_sum += np.count_nonzero(np.array(list(bin_str)).astype(dtype=int))

        return float(non_zero_sum) / float(n * 4)

from tqdm import tqdm

snr_list = np.linspace(0, 20, num=9)
tx = np.zeros(N * p, dtype=int)
mod_tx = modem.modulate(tx)
fig = plt.figure()
ts = range(0, l)

for t in tqdm(ts):
    error_rates = []
    bit_err_rates = []
    for snr_db in tqdm(snr_list):
        rx = awgn(mod_tx, snr_db)
        demod_rx = modem.demodulate(rx, 'hard')
        q_rx = to_q(demod_rx, N, p)

        for j in range(N // n):
                c, F = ldpc.single_threshold_majority(q_rx[j:(j + n)], t=t)
                error_rates.append(count_ber(c, n))

        bit_err_rates.append(np.array(error_rates).mean())
    plt.plot(snr_list, bit_err_rates, '.-', label=f"t = {t}")

bit_err_rates = []
for snr_db in tqdm(snr_list):
    rx = awgn(mod_tx, snr_db)
    demod_rx = modem.demodulate(rx, 'hard')
    bit_err_rates.append(np.count_nonzero(demod_rx) / len(demod_rx))

plt.plot(snr_list, bit_err_rates, '.-', label=f"No LDPC")

plt.yscale("log")
plt.grid()
plt.legend()
plt.show()

