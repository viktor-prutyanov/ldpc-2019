import numpy as np
from pyfinite import ffield
from pyfinite import genericmatrix

class LDPC:

    def __init__(self, q, H):
        self.q = q
        self.H = H
        self.n = H.shape[1]

        self.F = ffield.FField(int(np.log2(self.q)))

        ADD = self.F.Add
        MUL = self.F.Multiply
        SUB = self.F.Subtract
        DIV = self.F.Divide

        h = genericmatrix.GenericMatrix(size=H.shape,
                zeroElement=0, identityElement=1,
                add=ADD, mul=MUL, sub=SUB, div=DIV)

        for i in range(H.shape[0]):
            h.SetRow(i, H[i,:])

        self.h = h

    def single_threshold_majority(self, r, t):
        # Initialization
        #print("Init")
        S = self.h.LeftMulColumnVec(r.astype(int).tolist())
        b = True

        S = np.array(S)
        if (np.count_nonzero(S) == 0):
            #print(f"|S| = 0, early exit")
            b = False

        while b:
            b = False

            #print(f"\t|S| = {np.count_nonzero(np.array(S))}")
            for i in range(self.n):
                #print("\t\t", i)
                idxs = np.nonzero(self.H[:, i])[0]
                l = len(idxs)
                messages = np.zeros(l)

                for k, j in enumerate(idxs):
                    ms = self.F.Subtract(0, S[j])
                    inv = self.F.Inverse(self.H[j, i])
                    messages[k] = self.F.Multiply(ms, inv)

                A = messages[messages.nonzero()]
                unique, counts = np.unique(A, return_counts=True)

                if len(A) == 0:
                    #print("\t\tcontinue", [S[j] for j in idxs], i, idxs)
                    continue

                idx = np.argmax(counts)
                a = counts[idx]
                v = int(unique[idx])

                z = l - np.count_nonzero(messages)

                #print("\t\t", "a =", a, "z =", z, "a-z =", a - z, "t =", t, ((a - z) > t))
                if (a - z) > t:
                    r[i] = self.F.Add(int(r[i]), v)
                    S = self.h.LeftMulColumnVec(r.astype(int).tolist())
                    #print(f"\t\t|S| = {np.count_nonzero(np.array(S))}")
                    b = True

                S = np.array(S)
                if (np.count_nonzero(S) == 0):
                    #print(f"\t\t|S| = 0, late exit")
                    b = False
                    break
            #print("\tnext, b =", b)

        F = False
        c = r

        S = np.array(S)
        if (np.count_nonzero(S) == 0):
            #print(f"|S| = 0, late exit")
            F = True

        return c, F

    def multiple_threshold_majority(self, r, ts):
        #print("Init")
        ts.sort()
        for t in reversed(ts):
            #print("\t t=", t)
            r, _  = self.single_threshold_majority(r, t)

        S = self.h.LeftMulColumnVec(r.astype(int).tolist())
        F = False

        S = np.array(S)
        if (np.count_nonzero(S) == 0):
            F = True

        return r, F
