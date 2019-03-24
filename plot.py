#!/bin/python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig = plt.figure(figsize=(20, 12))

ts = ['uncoded', '0', '2', '3', '4', '024', '0234', '01234']

for t in ts:
    bers = []
    snrs = []
    with open(f'data-{t}.txt', 'r') as f:
        ls = f.readlines()
        for l in ls:
            ber, snr = map(float, l.rsplit('\n')[0].split(' '))
            bers.append(ber)
            snrs.append(snr)

    idxs = np.argsort(snrs).tolist()
    snrs = [snrs[idx] for idx in idxs]
    bers = [bers[idx] for idx in idxs]

    if t == 'uncoded':
        label = 'uncoded'
    elif len(str(t)) > 1:
        label = f"t = {list(map(int, list(t)))}"
    else:
        label = f"t = {t}"

    plt.plot(snrs, bers, '.-', label=label)

plt.xlabel("SNR dB", fontsize=33)
plt.ylabel("BER", fontsize=33)
plt.yscale("log")
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=28)
plt.tick_params(axis='both', which='minor', labelsize=28)
plt.legend(fontsize=28)
plt.show()

