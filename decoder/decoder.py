import numpy as np
import scipy as sp 

class LDPC:

    def __init__(self, q, H):
        self.q = q
        self.H = H
        self.n = H.shape[1]

    def single_threshold_majority(self, r_seq, t):
        # Initialization
        r = r_seq
        r = (np.array(r)).reshape(-1, 1)
        S = self.H @ r % self.q
        S = S
        b = True

        # print("S:", S)
    
        # Algorithm
        N = r.shape[0]
        
        if (np.count_nonzero(S) == 0):
            b = False

        while b:
            b = False
        
            for i in range(N):

                idx = np.nonzero(self.H[:,i])[0]
                l = len(idx)
                messages = np.zeros(l)
                for t, j in enumerate(idx):
                    
                    # dif = self.q - (self.H[j, i] * r[i] % self.q)
                    # print("dif", dif)
                    # av = S[j,0] + dif
                    # av = av % self.q

                    for k in range(self.q):
                        ms = (self.q - S[j,0]) % self.q
                        if ms == k * self.H[j,i] % self.q:
                            messages[t] = k
                            break

                a = messages[messages.nonzero()]
                unique, counts = np.unique(a, return_counts = True)

                if len(a) == 0:
                    continue

                idx = np.argmax(counts)
                a = counts[idx]
                v = int(unique[idx])

                z = l - np.count_nonzero(messages)

                if (a - z > t):
                    r[i] += v
                    r = r % self.q
                    S = self.H @ r % self.q
                    b = True
                
                if (np.count_nonzero(S) == 0):
                    b=False
                    break

        F = False
        c = r
    
        if (np.count_nonzero(S) == 0):
            F = True
    
        return c.T[0], F

    
    def multiple_threshold_majority(self, r, ts):
        ts[::-1].sort()
        for t in ts:
            r, _  = self.single_threshold_majority(r, t)

        S = self.H @ r % self.q
        F = False

        if (np.count_nonzero(S) == 0):
            F = True
        
        return r, F
