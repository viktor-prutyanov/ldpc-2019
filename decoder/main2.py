#!/bin/python3

import numpy as np

from decoder import LDPC

if __name__ == "__main__":
    H = np.loadtxt('../H.txt', usecols=range(512), dtype=int)
    q = 16
    n = H.shape[1]
    R = (n - H.shape[0]) / n

    print("H shape is", H.shape)
    print(f"q = {q}, R = {R}, n = {n}")

    ldpc = LDPC(q, H)

    print("")
    print("Single threshold:\n")

    for i in range(16):
        v = np.zeros(n)

        e = np.zeros(n)
        e[np.random.randint(low=1, high=n)] = np.random.randint(low=1, high=q)

        r = (v + e) % q

        c, F = ldpc.single_threshold_majority(r_seq=r, t=3)

    #    print(f"v = {v}, e = {e}, r = {r}, c = {c}, F = {F}")
        print(f"F = {F}")

    print("")
    print("Multiple threshold:\n")

    ts = [0, 1, 2]

    for i in range(16):
        v = np.zeros(n)

        e = np.zeros(n)
        e[np.random.randint(low=1, high=n)] = np.random.randint(low=1, high=q)

        r = (v + e) % q

        c, F = ldpc.multiple_threshold_majority(r, ts=ts)

    #    print(f"v = {v}, e = {e}, r = {r}, c = {c}, F = {F}")
        print(f"F = {F}")
