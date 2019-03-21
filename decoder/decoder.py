import numpy as np
import scipy as sp

class LDPC:

    def __init__(self, q, H):
        self.q = q
        self.H = H
        self.n = H.shape[1]
        self.l = np.count_nonzero(H[:, 0])

    def single_threshold_majority(self, r_seq, t):
        # Initialization
        r = r_seq
        r = (np.array(r)).reshape(-1, 1)
        S = self.H @ r % self.q
        # print(S)
        b = True

        if (np.count_nonzero(S) == 0):
            b = False

        #print("S weight =", np.count_nonzero(S))

        while b:
            b = False

            for i in range(self.n):
                idx = np.nonzero(self.H[:, i])[0]
                l = len(idx)
                messages = np.zeros(l)
                for t, j in enumerate(idx):
                    
                    if np.count_nonzero(r[self.H[j,:].nonzero()]) == 1:
                        messages[t] = (self.q - r[i]) % self.q
                        continue

                    for k in range(0, self.q):
                        ms = (self.q - S[j,0]) % self.q
                        
                        if ms == k * self.H[j,i] % self.q:
                            messages[t] = k
                            break
                # print("messages =", messages)

                a = messages[messages.nonzero()]
                unique, counts = np.unique(a, return_counts = True)

                if len(a) == 0:
                    continue

                idx = np.argmax(counts)
                a = counts[idx]
                v = int(unique[idx])

                z = self.l - np.count_nonzero(messages)

                if (a - z > t):
                    r[i] += v
                    r = r % self.q
                    S = self.H @ r % self.q
                    b = True

                if (np.count_nonzero(S) == 0):
                    b = False
                    break

        F = False
        c = r

        if (np.count_nonzero(S) == 0):
            F = True

        return c.flatten(), F

    def multiple_threshold_majority(self, r, ts):
        ts[::-1].sort()
        for t in ts:
            r, _  = self.single_threshold_majority(r, t)

        S = self.H @ r % self.q
        F = False

        if (np.count_nonzero(S) == 0):
            F = True

        return r, F
