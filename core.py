from config import Config
import numpy as np
from cmath import e
from utils import show_status

# parameters
class LRError(Exception):
    pass

class DataSpace(object):
    def __init__(self):
        self.B = 0.5                     # rand b #TODO single?
        self.C = 0.5                     # rand 0-1 b_j
        self.W = np.random.rand(Config.J)       # rand 0-1 w_i
        self.W_ = np.random.rand(Config.J, Config.K)   # rand 0-1 w_jk

    def tofile(self, path):
        """
        put model parameters to file
        """
        show_status(".. put parameters to file: %s" % path)

        with open(path, 'w') as f:
            tem = []
            # input config K, J
            tem.append(' '.join([str(i) 
                for i in (Config.K, Config.J)]))
            # input config THETA SIGMA
            tem.append(' '.join([str(i) 
                for i in (Config.THETA, Config.SIGMA)]))

            # input parameters
            # input a line of B, C
            tem.append(' '.join([str(i) 
                for i in (self.B, self.C)]))
            # input a line of Ws
            tem.append(' '.join([str(i) for i in self.W]))
            # input K lines of W_s
            for j in range(Config.J):
                tem.append(' '.join([str(i) for i in self.W_[j]]))
            f.write('\n'.join(tem))

    def fromfile(self, path):
        show_status(".. load parameters from file : %s" % path)
        def split_line_trans_type(line, _type):
            return [_type(i) for i in line.split()]

        with open(path) as f:
            for no, line in enumerate(f.readlines()):
                if no == 0:
                    Config.K, Config.J = split_line_trans_type(line, int)
                elif no == 1:
                    Config.THETA, Config.SIGMA = split_line_trans_type(line, float)
                elif no == 2:
                    self.B, self.C = split_line_trans_type(line, float)
                elif no == 3:
                    Ws = split_line_trans_type(line, float)
                    for i in range(Config.J):
                        self.W[i] = Ws[i]
                else:
                    j = no - 3
                    w_j = split_line_trans_type(line, float)
                    for k in range(Config.K):
                        self.W_[j][k] = w_j[k]


class LambdaRank(object):
    def __init__(self):
        self.dataspace = DataSpace()

    def init(self, X1, X2):
        """
        init pair (X1, X2)
        """
        self.X1, self.X2 = X1, X2
        self.S1, self.S2 = self.s(self.X1), self.s(self.X2)
        self._diff_L = self.diff_L(None, None, self.S1, self.S2)

    def diff_L_b(self):
        """
        \partial L
        -----------
        \partial b
        """
        res = \
            self._diff_L * \
                (
                    sum([self.dataspace.W[j] * 
                        self.diff_f(None, self.J(self.X1, j)) for j in range(Config.J)])
                    - 
                    sum([self.dataspace.W[j] * 
                        self.diff_f(None, self.J(self.X2, j)) for j in range(Config.J)]))
        return res

    def diff_L_w_(self, j):
        """
        \partial L
        -----------
        \partial w_j
        """
        return self._diff_L *\
                (self.f_(j, self.X1) - self.f_(j, self.X2))


    def diff_L_w__(self, j, k):
        """
        \partial L
        -----------
        \partial W_jk
        """
        return self._diff_L * self.dataspace.W[j] * (
                    self.diff_f(j, self.X1) * self.X1[k]
                    -
                    self.diff_f(j, self.X2) * self.X2[k]
                )
    # ---------------updater ----------
    def update_w_(self, j):
        """
        update w_k
        """
        self.dataspace.W[j] = self.dataspace.W[j] - \
                Config.THETA * self.diff_L_w_(j)

    def update_w__(self, j, k):
        self.dataspace.W[j][k] = self.dataspace.W[j][k] - \
                Config.THETA * self.diff_L_w__(j, k)

    def update_b(self):
        self.dataspace.B = self.dataspace.B - \
                Config.THETA * self.diff_L_b()

    # ----------------API---------------
    def study_line(self, X1, X2):
        """
        study from a line of record from trainset
        """
        self.init(X1, X2)
        for j in range(Config.J):
            self.update_w_(j)
            self.update_b_(j)

        for j in range(Config.J):
            for k in range(Config.K):
                self.update_w__(j, k)

    def predict(self, X):
        return self.s(X)


    # ----------------private detail functions -----
    def J(self, X, j):
        """
        \sum_k{w_{jk}x_{(k)} + b_j}
        """
        return sum([self.dataspace.W_[j][k] * X[k] 
                for k in range(Config.K)]) + self.dataspace.B_[j]

    def f(x):
        """
        >>> f(2)
        0.8807970779778823
        """
        return 1/(1+e**(-x))

    def f_(self, j, X):
        """
        f_j(X)
        """
        return self.f(self.J(X, j))

    def s(self, X):
        """
        \sum_j{w_j f_j(J(X)) + b}
        """
        return sum([self.dataspace.W[j] * self.f_(j, X) 
                    for j in Config.J]) + self.dataspace.B

    def diff_L(self, X1, X2, S1=None, S2=None):
        """
        cal L'
        """
        S1 = self.s(X1) if not S1 else S1
        S2 = self.s(X2) if not S2 else S2
        return - Config.SIGMA / (
                1 + e ^ (Config.SIGMA * (S1 - S2))
            )

    def diff_f(self, j, X, JX=None):
        """
        cal 
        \partial{f(D)}
        --------------
        \partial{D}
        """
        _D = self.J(X, j) if not JX else JX
        return e^(-_D) / (1+e^(-_D))^2

if __name__ == '__main__':
    import doctest
    doctest.testmod()
