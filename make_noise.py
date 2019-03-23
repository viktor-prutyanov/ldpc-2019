#!/bin/python3

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--num", dest='num', help="number of code words", type=int)
parser.add_argument("--num-points", dest='num_points',
        help="number of experimental points", type=int)
args = parser.parse_args()

n = 512
N = args.num * n
sigmas = np.logspace(-0.8, -0.1, num=args.num_points)

noises = np.empty((args.num_points, N), dtype=np.cdouble)

for i, sigma in enumerate(sigmas):
    noises[i, :] = np.random.normal(0, sigma, N) + 1.j * np.random.normal(0, sigma, N)

np.savetxt("noises.txt", noises)
