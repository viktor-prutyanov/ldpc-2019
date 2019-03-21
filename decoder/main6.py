#!/bin/python3

import numpy as np
from commpy import QAMModem, awgn
from scipy import stats
import matplotlib.pyplot as plt

from decoder import LDPC

M = 16 # Use QAM-16

modem = QAMModem(M)

H = np.loadtxt('H2.txt', usecols=range(512), dtype=int)
q = 16
p = 4
n = H.shape[1]
R = (n - H.shape[0]) / n
l = np.count_nonzero(H[:, 0])

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

def get_mean_power(signal):
    return np.linalg.norm(signal)**2 / len(signal)

tx = np.zeros(N * p, dtype=int)
scramble_seq = np.random.randint(2, size=(N * p), dtype=int)
# tx = np.bitwise_xor(tx, scramble_seq)
mod_tx = modem.modulate(tx)
Ps = get_mean_power(mod_tx)
fig = plt.figure(figsize=(10,6))

sigmas = np.logspace(-1, 0, num=100) # -1 0.5 20

print("Single threshold case processing...")

ts = range(0, l)

for t in tqdm(ts):
    snr_list = []
    bit_err_rates = []
    for sigma in tqdm(sigmas):
        error_rates = []
        noise = np.random.normal(0, sigma, N) + 1.j * np.random.normal(0, sigma, N)
        Pn = get_mean_power(noise)
        snr_db = 10 * np.log10(Ps / Pn)
        snr_list.append(snr_db)
        rx = mod_tx + noise
        demod_rx = modem.demodulate(rx, 'hard')
        # demod_rx = np.bitwise_xor(demod_rx, scramble_seq)
        q_rx = to_q(demod_rx, N, p)

        for j in range(N // n):
            c, F = ldpc.single_threshold_majority(q_rx[j:(j + n)], t=t)
            error_rates.append(count_ber(c, n))

        bit_err_rates.append(np.array(error_rates).mean())
    plt.plot(snr_list, bit_err_rates, '.-', label=f"t = {t}")

'''
print("Multiple threshold case processing...")

tss = [[0, 1], [0, 2], [0, 1, 2]]

for ts in tqdm(tss):
    snr_list = []
    bit_err_rates = []
    mFs = []
    for sigma in tqdm(sigmas):
        error_rates = []
        Fs = []
        noise = np.random.normal(0, sigma, N) + 1.j * np.random.normal(0, sigma, N)
        Pn = get_mean_power(noise)
        snr_db = 10 * np.log10(Ps / Pn)
        snr_list.append(snr_db)
        rx = mod_tx + noise
        demod_rx = modem.demodulate(rx, 'hard')
#        demod_rx = np.bitwise_xor(demod_rx, scramble_seq)
        q_rx = to_q(demod_rx, N, p)

        for j in range(N // n):
            c, F = ldpc.multiple_threshold_majority(q_rx[j:(j + n)], ts)
            error_rates.append(count_ber(c, n))
            Fs.append(F)

        bit_err_rates.append(np.array(error_rates).mean())
        mFs.append(np.array(Fs).any())
    plt.plot(snr_list, bit_err_rates, '.-', label=f"ts = {ts}")
    print(mFs)
'''
print("No LDPC case processing...")

snr_list = []
bit_err_rates = []
for sigma in tqdm(sigmas):
    noise = np.random.normal(0, sigma, N) + 1.j * np.random.normal(0, sigma, N)
    Pn = get_mean_power(noise)
    snr_db = 10 * np.log10(Ps / Pn)
    snr_list.append(snr_db)
    rx = mod_tx + noise
    demod_rx = modem.demodulate(rx, 'hard')
    # demod_rx = np.bitwise_xor(demod_rx, scramble_seq)
    bit_err_rates.append(np.count_nonzero(demod_rx) / len(demod_rx))

plt.plot(snr_list, bit_err_rates, '.-', label=f"No LDPC")

plt.xlabel("SNR dB")
plt.ylabel("BRE")
plt.yscale("log")
plt.grid()
plt.legend()
plt.show()

