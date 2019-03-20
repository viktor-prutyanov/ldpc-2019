#!/bin/python3

import numpy as np
from commpy import QAMModem, awgn

from decoder import LDPC

M = 16 # Use QAM-16

modem = QAMModem(M)

#snr_dB = 0
#print(f"SNR = {snr_dB} dB")

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
#rx = awgn(mod_tx, snr_dB)
rx = mod_tx + (np.random.normal(0, 1, N) + np.random.normal(0, 1, N) * 1.j) / 2
print(rx)
demod_rx = modem.demodulate(rx, 'hard')
print(demod_rx)
q_rx = to_q(demod_rx, N, p)

for i in range(32):
    c, F = ldpc.single_threshold_majority(q_rx[i:i+n], t=3)
    print(f"Digit error rate = {np.count_nonzero(c) / n}, F = {F}")
