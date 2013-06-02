from config import K, J, THETA
import numpy as np
from cmath import e

# parameters

class Data(object):
    W = np.random.rand(J)       # rand 0-1 w_i
    W_ = np.random.rand(J, K)   # rand 0-1 w_jk
    B_ = np.random.rand(J)      # rand 0-1 b_j

    B = 1                       # rand b #TODO single?

def f(x):
    return 1/(1+e**(-x))

def f_(j, X):
    """
    f_j(X)
    """
    return f(sum([Data.W_[j][k] * X[k] for k in range(K)]) + Data.B_[j])

def s_(self, X):
    """
    cal score for featrue: X
    S(x)
    """
    return f( sum(
            [Data.W[j] * f_(j, X) for j in range(J)]) 
        + Data.B)

def diff_L(X1, X2):
    """
    cal L'
    """
    return - (e**(-(s_(X1) - s_(X2))
                  /(1 + e**(s_(X1) - s_(X2)))))

def J(X, j):
    """
    \sum_k{w_{jk}x_{(k)} + b_j}
    """
    return sum([Data.W_[k] * X[k] for k in range(K)]) + Data.B_[j]

def D(X):
    """
    \sum_j{w_j f_j(J(X)) + b}
    """
    return sum([Data.W[j] * f_(j, X) for j in J]) + Data.B

def diff_f(X):
    """
    cal 
    \partial{f(D)}
    --------------
    \partial{D}
    """
    _D = D(X)
    return e^(-_D) / (1+e^(-_D))^2


class LambdaRank(object):
    def init(self, X1, X2):
        """
        init pair (X1, X2)
        """
        self.X1, self.X2 = X1, X2

    def diff_L_b(self):
        """
        \partial L
        -----------
        \partial b
        """
        res = \
            diff_L(self.X1, self.X2) * \
                (diff_f(self.X1) - diff_f(self.X2))
        return res

    def diff_L_w_(self, j):
        """
        \partial L
        -----------
        \partial w_j
        """
        tem = diff_f(self.X1) * f_(j, self.X1) - diff_L(self.X2) * f_(j, self.X2)
        return diff_L(self.X1, self.X2) * tem

    def diff_L_b_(self, j):
        """
        \partial L
        -------------
        \partial bj
        """
        return diff_L(self.X1, self.X2) * \
            (diff_f(self.X1) - diff_f(self.X2))

    def diff_L_w__(self, j, k):
        """
        \partial L
        -----------
        \partial W_jk
        """
        tem = diff_f(self.X1) * self.X1[k] \
                - diff_f(self.X2) * self.X2[k]
        return diff_L(self.X1, self.X2) * tem


    # ---------------updater ----------
    def update_w_(self, j):
        """
        update w_k
        """
        Data.W[j] = Data.W[j] - THETA * self.diff_L_w_(j)

    def update_w__(self, j, k):
        Data.W[j][k] = Data.W[j][k] - THETA * self.diff_L_w__(j, k)

    def update_b(self, j):
        Data.B = Data.B - THETA * self.diff_L_b()

    def update_b_(self, j):
        Data.B_[j] = Data.B_[j] - THETA * self.diff_L_b_(j)
