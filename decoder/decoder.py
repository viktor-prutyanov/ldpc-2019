import numpy as np
import scipy as sp 

class LDPC:

    def __init__(self, q, H, l, n):
        self.q = q
        self.H = H
        self.l = l
        self.n = n

    def single_threshold_majority(self, r_seq, t):
        # Initialization
        r = r_seq
        r = (np.array(r)).reshape(-1, 1)
        S = self.H @ r
        b = True
    
        # Algorithm
        N = r.shape[0]
        
        while b:
            b = False
        
            for i in range (N):
                messages = np.zeros(self.l)

                idx = np.nonzero(H[:,i])[0]

                for j in idx:
                    
                    dif = self.q - (H[j, i] * r[i] % self.q)
                    av = S[j] + dif
                    av = av % self.q

                    for k in range(self.q):
                        if av == k * H[j,i] % self.q:
                            messages[j] = k
                            break

                unique, counts = np.unique(messages, return_counts = True)
                A_dict = dict(zip(counts, unique))
                key_max = max(A_dict.keys(), key=(lambda k: A_dict[k]))
                v = key_max
                a = A_dict[key_max]

                z = self.l - np.count_nonzero(messages)

                if (a - z > t):
                    r[i] = v
                    S = self.H @ r
                    b = True
        F = False
        c = r
    
        if (np.count_nonzero(S) == 0):
            F = True
    
        return (F, c)


if __name__ == "__main__":
    H = np.array([[1, 0, 4, 0], [0, 2, 4, 0], [0, 0, 2, 5]])

    idx = np.nonzero(H[:,2])[0]

    print(H)
    for j in idx[0]:
        print(j)