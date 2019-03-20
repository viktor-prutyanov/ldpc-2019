#!/bin/python3

import numpy as np
from itertools import product
from time import time
from tqdm import tqdm

def get_H0(q, n0):
    return np.random.randint(low=1, high=q, size=n0)

def get_Hb(q, b, n0):
    H0 = get_H0(q, n0)
    return np.kron(np.eye(b), H0)

def phi(H_b, n0, b, q):
    permutation = np.random.permutation(range(n0 * b))
    H_b1 = H_b[:, permutation].copy()
    for j in range(n0 * b):
        H_b1[:, j] *= np.random.randint(low=1, high=q)
        H_b1[:, j] %= q
    return H_b1

def get_H(q, n0, l, b):
    H_b = get_Hb(q, b, n0)
    Hs = [phi(H_b, n0, b, q) for i in range(l)]
    H = np.vstack(tuple(Hs))
    return H

n0 = 5
b = 3
n = n0 * b
q = 3
l = 4
k = l * b
print(f"n = {n}, k = {k}, R = {k / n}")

H = get_H(q, n0, l, b)
print(f"H of size", H.shape)

cs = []

for c in tqdm(product(range(q), repeat=n), total=q**n):
    s = H @ np.array(c).T % q
    if not np.any(s):
        cs.append(c)

print(f"|M| = {len(cs)}")
