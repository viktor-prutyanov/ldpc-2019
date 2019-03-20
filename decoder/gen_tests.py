import numpy as np
import scipy as sp
from itertools import product

def get_H0(q, g, n0):
    r = np.random.randint(q, size=n0)
    f = np.full((n0,), g)
    return np.power(f, r) % q

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

def gen_codewords(H, q, n):
    cs = np.array(list(product(range(q), repeat=n))).T
    ss = H @ cs % q
    js = []
    codewords = []

    for j in range(ss.shape[1]):
        if (ss[:, j].max() == 0):
            codewords.append(cs[:, j])
            js.append(j)

    #print (len(js))
    #print (codewords)
    return codewords