#!/bin/python3

import numpy as np

from gen_tests import get_H
from gen_tests import gen_codewords
from decoder import LDPC


if __name__ == "__main__":
        print("LDPC")

        n0 = 5
        b = 2
        n = n0 * b
        q = 3
        l = 4

        H = get_H(q, n0, l, b)
        print("H shape: ", H.shape)
        print(f"q = {q}, R = {l / n0}")

        codebook = gen_codewords(H, q, n)

        v = codebook[np.random.randint(low=1, high=len(codebook))]
        print("Initial codeword:\t", v)

        e = np.zeros_like(v)
        e[np.random.randint(low=1, high=(n+1))] = np.random.randint(low=1, high=q)
        print("Error word:\t\t", e)

        r = (v + e) % q
        print("Noised codeword:\t", r)

        ldpc = LDPC(q, H)
        c, F = ldpc.single_threshold_majority(r_seq=r, t=3)

        print("Decoded codeword:\t", c)
        print("Validation status:\t", F)
        print("Initial - Decoded:\t", (v - c))

        print("")

        # multiple
        ts = [0, 1, 2]

        r = (v + e) % q
        c1, F1 = ldpc.multiple_threshold_majority(r, ts=ts)

        print("Decoded codeword:\t", c1)
        print("Validation status:\t", F1)
        print("Initial - Decoded:\t", (v - c) % q)
