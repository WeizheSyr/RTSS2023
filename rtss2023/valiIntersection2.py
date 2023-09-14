import numpy as np
from utils.formal.zonotope import Zonotope
import cvxpy as cp
from utils.Baseline import Platoon
import time

np.set_printoptions(suppress=True)


def checkinBox(low, up, point):
    result = 1
    for i in range(low.shape[0]):
        if point[i] <= low[i]:
            result = 0
            return result
        if point[i] >= up[i]:
            result = 0
            return result
    return result


A = Platoon.model.sysd.A
B = Platoon.model.sysd.B

U = Zonotope.from_box(np.ones(4) * -10, np.ones(4) * 10)
A_i = [np.eye(A.shape[0])]
for i in range(20):
    A_i.append(A @ A_i[-1])
A_i_B_U = [val @ B @ U for val in A_i]
E = []
for i in range(20):
    if i == 0:
        E.append(A_i_B_U[0])
    else:
        E.append(E[-1] + A_i_B_U[i])


temp = E[4]
# temp = E[4].order_reduction(7)
# temp.g = temp.g * 2.8

# E_bound
l = np.eye(A.shape[0])
E_up = []
E_low = []
for j in range(7):
    E_low.append(temp.support(l[j], -1))
    E_up.append(temp.support(l[j]))
print("E_low")
print(E_low)
print("E_up")
print(E_up)

D_low = [-0.6327720639435364, -0.602399556735256, -0.6596764214473849, -1.3641168110318211, -1.2465597978222978, -1.4271499822702531, -1.3953298454509189]
D_up = [-0.03429068321966455, 0.03286102079409425, -0.05438234810840625, -0.3470178501868151, -0.2749886047735577, -0.46296348863231485, -0.40030780345735284]


