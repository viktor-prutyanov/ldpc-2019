#!/bin/python3

import numpy as np
from commpy import QAMModem, awgn
from scipy import stats
import matplotlib.pyplot as plt


from decoder import LDPC


def signaltonoise(a, axis=0, ddof=0):
	a = np.asanyarray(a)
	m = a.mean(axis)
	sd = a.std(axis=axis, ddof=ddof)
	return np.where(sd == 0, 0, m/sd)


M = 16 # Use QAM-16

modem = QAMModem(M)

snr_dB = 10
print(f"SNR = {snr_dB} dB")

H = np.loadtxt('../H.txt', usecols=range(512), dtype=int)
q = 16
p = 4
n = H.shape[1]
R = (n - H.shape[0]) / n

print("H shape is", H.shape)
print(f"q = {q}, R = {R}, n = {n}")

ldpc = LDPC(q, H)

N = 32 * n # Length of transmission

def to_q(vs, N, p):
    def f(v):
        return sum([(v[i] << i) for i in range(p)])

    return np.array(list(map(f, [v for v in vs.reshape((N, p))])))

tx = np.zeros(N * p, dtype=int)
print(tx)
mod_tx = modem.modulate(tx)
print(mod_tx)

rx = awgn(mod_tx, snr_dB, 0.75)
#rx = mod_tx + (np.random.normal(0, 1, N) + np.random.normal(0, 1, N) * 1.j) / 2

print ("rx shape: ", rx.shape)
print(rx)
demod_rx = modem.demodulate(rx, 'hard')
print(demod_rx)
print (demod_rx.shape)
q_rx = to_q(demod_rx, N, p)

print ("q_rx shape: ", q_rx.shape)
print (q_rx)


error_rates = []

def count_ber(c, n):
	non_zero_sum = 0

	for i in c:
		bin_str = bin(i)[2:]
		non_zero_sum += np.count_nonzero(np.array(list(bin_str)).astype(dtype=int))

	return float(non_zero_sum) / float(n * 4)

'''
ts=[0, 1, 2, 3]
for i in range(32):
	c, F = ldpc.single_threshold_majority(q_rx[i:i+n], t=3)
	print(f"Digit error rate = {np.count_nonzero(c) / n}, F = {F}")
	print ("BER: ", count_ber(c, n))
	error_rates.append(np.count_nonzero(c) / n)

	c1, F = ldpc.multiple_threshold_majority(q_rx[i:i+n], ts)
	print(f"[Myltiple] Digit error rate = {np.count_nonzero(c1) / n}, F = {F}")
	print ("[Multiple] BER = ", count_ber(c1, n))

'''

from tqdm import tqdm

snr_list = [1, 2, 4, 6, 8, 10, 12, 14, 16]
#snr_list = [1, 2, 4, 6, 8]
bit_err_rates = []

for snr_db in tqdm(snr_list):
	
	tx = np.zeros(N * p, dtype=int)
	mod_tx = modem.modulate(tx)

	rx = awgn(mod_tx, snr_db, 0.75)
	#rx = mod_tx + (np.random.normal(0, 1, N) + np.random.normal(0, 1, N) * 1.j) / 2

	demod_rx = modem.demodulate(rx, 'hard')
	q_rx = to_q(demod_rx, N, p)

	for j in range(32):
		c, F = ldpc.single_threshold_majority(q_rx[j:j+n], t=3)
		#print(f"Digit error rate = {np.count_nonzero(c) / n}, F = {F}")
		#print ("BER: ", count_ber(c, n))
		error_rates.append(count_ber(c, n))

	bit_err_rates.append(np.array(error_rates).mean())

print (snr_list)
print (bit_err_rates)

fig = plt.figure()
plt.plot(snr_list, bit_err_rates)
plt.yscale("log")
plt.grid()
plt.show()



