#!/bin/python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig = plt.figure(figsize=(20, 12))

ts = ['uncoded', '0', '1', '3', '024', '01234']
#ts = ['uncoded']

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

    #ber_smooth = pd.Series(bers).rolling(window=3).mean().iloc[2:].values
    #plt.plot(snrs[2:], ber_smooth, '.-', label=label)
    plt.plot(snrs, bers, '.-', label=label)

plt.xlabel("SNR dB")
plt.ylabel("BER")
plt.yscale("log")
plt.grid()
plt.legend()
plt.show()

