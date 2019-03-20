import numpy as np

#import decoder
#import gen_tests

from gen_tests import get_H
from gen_tests import gen_codewords
from decoder import LDPC


if __name__ == "__main__":
	print ("LDPC")
	
	n0 = 5
	b = 2
	n = n0 * b
	q = 3
	l = 4

	H = get_H(q, n0, l, b)
	print ("H shape: ", H.shape)

	codebook = gen_codewords(H, q, n)

	v = codebook[2]
	print ("Initial codeword:\t", v)

	e = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
	print ("Error word:\t\t", e)

	r = (v + e) % q
	print ("Noised codeword:\t", r)

	ldpc = LDPC(q, H)
	c, F = ldpc.single_threshold_majority(r_seq=r, t=3)

	print ("Decoded codeword:\t", c)
	print ("Validation status:\t", F)
	print ("Initial - Decoded:\t", (v - c) % q)


	# multiple
	
	ts = [0, 1, 2]

	r = (v + e) % q
	c1, F1 = ldpc.multiple_threshold_majority(r, ts=ts)

	print ("Decoded codeword:\t", c1)
	print ("Validation status:\t", F1)
	print ("Initial - Decoded:\t", (v - c) % q)