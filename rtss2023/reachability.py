import numpy as np
from utils.formal.zonotope import Zonotope
import time

class Reachability:
    def __init__(self, A, B, P: Zonotope, U: Zonotope, target_low, target_up, max_step=20):
        self.A = A
        self.B = B
        self.P = P
        self.U = U
        self.t_lo = target_low
        self.t_up = target_up
        self.max_step = max_step

        # A^i
        self.A_i = [np.eye(A.shape[0])]
        for i in range(max_step):
            self.A_i.append(A @ self.A_i[-1])
        # A^i BU
        self.A_i_B_U = [val @ B @ U for val in self.A_i]
        # A^i P
        self.A_i_P = [self.A_i[val] @ self.P for val in range(max_step)]
        # l for support function
        self.l = np.eye(A.shape[0])