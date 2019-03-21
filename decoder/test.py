#!/bin/python3

import numpy as np
from decoder import LDPC
from tqdm import tqdm

H = np.loadtxt('H2.txt', usecols=range(512), dtype=int)
q = 16
p = 4
n = H.shape[1]
R = (n - H.shape[0]) / n
l = np.count_nonzero(H[:, 0])

print("H shape is", H.shape)
print(f"q = {q}, R = {R}, n = {n}, l = {l}")

ldpc = LDPC(q, H)

errs = []

for i in tqdm(range(n)):
    for j in range(q):
        v = np.zeros(n)
        v[i] = j
        c, F = ldpc.single_threshold_majority(v, t=2)
        if np.count_nonzero(c) != 0 or F == False:
            errs.append((i, j, np.count_nonzero(c) == 0, F))

print(errs)

'''

v = np.zeros(n)
v[7] = 1
c, F = ldpc.single_threshold_majority(v, t=2)
print(v)
print(c)
print(F)
'''
