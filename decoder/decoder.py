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
                        if self.q - S[j,0] == k * self.H[j,i] % self.q:
                            messages[t] = k
                            break

                a = messages[messages.nonzero()]
                unique, counts = np.unique(a, return_counts = True)
                A_dict = dict(zip(counts, unique))

                # print("Messages:", messages)
                # print("a:", a)
                
                if len(a) == 0:
                    continue

                # print("unique:", unique)
                # print("counts:", counts)
                # print("A_dict:", A_dict)

                key_max = max(A_dict.keys(), key=(lambda k: A_dict[k]))
                a = key_max
                v = int(A_dict[key_max])

                # print("a: ", a)
                # print("v: ", v)

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
    
        return (F, c.T[0])


if __name__ == "__main__":
    H = np.array([[1, 0, 4, 0], [0, 2, 4, 0], [0, 0, 2, 5]])

    idx = np.nonzero(H[:,2])[0]

    print(H)
    for j in idx[0]:
        print(j)