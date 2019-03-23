#!/bin/python3

import argparse
import numpy as np
from commpy import QAMModem
from scipy import stats
import matplotlib.pyplot as plt
import sys

from decoder import LDPC

parser = argparse.ArgumentParser()
parser.add_argument("--num", dest='num', help="number of code words", type=int)
parser.add_argument("--num-points", dest='num_points',
        help="number of experimental points", type=int)
parser.add_argument("--num-ts", dest='num_ts',
        help="number of thresholds", type=int)
parser.add_argument('--ts', dest='ts', nargs='+', help='list of thresholds',
        type=int)
args = parser.parse_args()

H = np.loadtxt('H.txt', usecols=range(512), dtype=int)
q = 16
p = 4 # Bits per element of GF(q)
n = H.shape[1]
R = (n - H.shape[0]) / n
l = np.count_nonzero(H[:, 0]) # Columns weigth for regular LDPC
#print(f"Loaded parity-check matrix of shape {H.shape};"
#        f" q = {q}, R = {R}, n = {n}, l = {l}")

M = 16
modem = QAMModem(M)

ldpc = LDPC(q, H)

N = args.num * n # Length (in elements of GF(16)) of transmission

def bits_to_GF(bits, N, p):
    '''
        Converts (N * p) bits into N elements of GF(2^p)
    '''
    def vec_to_GF(bv):
        return sum([(bv[i] << i) for i in range(p)])

    return np.array(list(map(vec_to_GF, [bv for bv in bits.reshape((N, p))])))

def get_mean_power(signal):
    return np.linalg.norm(signal)**2 / len(signal)

def count_ber(c, n):
    acc = 0

    for i in c:
        bin_str = bin(i)[2:]
        acc += np.count_nonzero(np.array(list(bin_str)).astype(dtype=int))

    return float(acc) / float(n * 4)

tx = np.zeros(N * p, dtype=int)
#scramble_seq = np.random.randint(2, size=(N * p), dtype=int)
#tx = np.bitwise_xor(tx, scramble_seq)
mod_tx = modem.modulate(tx)
mod_tx_power = get_mean_power(mod_tx)

sigmas = np.logspace(-0.8, -0.1, num=args.num_points)
num_ts = args.num_ts
if num_ts != 0:
    ts = args.ts

snrs = [] # Signal-to-noise ratios in dB
bers = [] # Bit error rates

noises = np.loadtxt('noises.txt', dtype=np.cdouble)

for i, sigma in enumerate(sigmas):
    #noise = np.random.normal(0, sigma, N) + 1.j * np.random.normal(0, sigma, N)
    noise = noises[i, :]
    noise_power = get_mean_power(noise)
    snr_db = 10 * np.log10(mod_tx_power / noise_power)
    snrs.append(snr_db)

    rx = mod_tx + noise # AWGN channel

    demod_rx = modem.demodulate(rx, 'hard')
    #demod_rx = np.bitwise_xor(demod_rx, scramble_seq)

    if num_ts == 0:
        bers.append(np.count_nonzero(demod_rx) / len(demod_rx)) # Uncoded
    elif num_ts == 1:
        rx = bits_to_GF(demod_rx, N, p)
        word_bers = []

        for j in range(N // n):
            c, F = ldpc.single_threshold_majority(rx[j:(j + n)], t=ts[0])
            word_bers.append(count_ber(c, n))

        bers.append(np.array(word_bers).mean())
    else:
        rx = bits_to_GF(demod_rx, N, p)
        word_bers = []

        for j in range(N // n):
            c, F = ldpc.multiple_threshold_majority(rx[j:(j + n)], ts)
            word_bers.append(count_ber(c, n))

        bers.append(np.array(word_bers).mean())

for ber, snr in zip(bers, snrs):
    print(ber, snr)
