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
	l = 2

	H = get_H(q, n0, l, b)
	print ("H shape: ", H.shape)

	codebook = gen_codewords(H, q, n)
	print ("Zero codeword: ", codebook[0])

	e = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
	r = codebook[0] + e
	print (r)

	ldpc = LDPC(q, H, l, n)

	ldpc.single_threshold_majority(r_seq=r, t=1)	
