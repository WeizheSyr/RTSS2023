import numpy as np
from utils.formal.zonotope import Zonotope
import cvxpy as cp
from utils.Baseline import Platoon
import time

# np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

def checkin(low, up, point):
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
        # E.append(A_i_B_U[0])

# l = np.eye(A.shape[0])
# E_up = []
# E_low = []
# for j in range(7):
#     E_low.append(E[4].support(l[j], -1))
#     E_up.append(E[4].support(l[j]))
# print("E_low")
# print(E_low)
# print("E_up")
# print(E_up)

# temp = E[4]
temp = E[4].order_reduction(7)
temp.g = temp.g * 2.8

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

D_low = np.array(D_low)
D_up = np.array(D_up)

# D_low[1] = D_low[1] * 1.5
# D_up[1] = D_up[1] * 1.5
# D_low[2] = D_low[2] * 2
# D_up[2] = D_up[2] * 2

steps = (D_up - D_low) / 2

ord = temp.g.shape[1]
dim = 7
# temp = E[9]

beta = cp.Variable([ord], name="beta")

constraints = [
    (beta[k] <= 1) for k in range(ord)
]
constraints += [
    (beta[k] >= -1) for k in range(ord)
]
constraints += [
    ((temp.c + temp.g @ beta)[k] <= D_up[k]) for k in range(dim)
]
constraints += [
    ((temp.c + temp.g @ beta)[k] >= D_low[k]) for k in range(dim)
]
problem = cp.Problem(cp.Minimize(0), constraints)
result = problem.solve()
print("beta.value")
print(beta.value)
print("result point")
result = temp.g @ beta.value
print(result)

E_inv = np.linalg.pinv(temp.g)
# print(E_inv)
# print("up")
# print(E_inv @ D_up)
# print("low")
# print(E_inv @ D_low)

# intersection on dim
# new box
select = []
new_low = []
new_up = []
for i in range(dim):
    # contain
    if E_low[i] <= D_low[i] and E_up[i] >= D_up[i]:
        new_up.append(D_up[i])
        new_low.append(D_low[i])
        # select.append(0)
    elif E_low[i] >= D_low[i] and E_up[i] <= D_up[i]:
        new_up.append(E_up[i])
        new_low.append(E_low[i])
        # select.append(0)
    elif E_low[i] <= D_low[i] and E_up[i] <= D_up[i]:
        new_up.append(E_up[i])
        new_low.append(D_low[i])
        # select.append()
    elif E_low[i] >= D_low[i] and E_up[i] >= D_up[i]:
        new_up.append(D_up[i])
        new_low.append(E_low[i])
    elif E_up[i] <= D_low[i] or E_low[i] >= D_up[i]:
        print("no intersection")
        exit(1)
new_up = np.array(new_up)
new_low = np.array(new_low)
print("new_up")
print(new_up)
print("new_low")
print(new_low)

print("up")
print(E_inv @ new_up)
print("low")
print(E_inv @ new_low)
print(E_inv @ result)

def check_pass(a, b):
    up = np.ones(7)
    up = up - b
    low = -np.ones(7) - b

    for i in range(7):
        t = a[i] - b[i]
        if t > 0:
            up[i] = up[i] / t
            low[i] = low[i] / t
        elif t < 0:
            temp = up[i]
            up[i] = low[i] / t
            low[i] = temp / t

    f_up = 0
    f_low = 0
    signal = 1
    for i in range(7):
        if i == 0:
            f_up = up[i]
            f_low = low[i]
        else:
            if f_up >= up[i] and f_low <= low[i]:
                f_up = up[i]
                f_low = low[i]
            elif f_up <= up[i] and f_low <= low[i]:
                f_low = low[i]
            elif f_up >= up[i] and f_low >= low[i]:
                f_up = up[i]
            elif f_low >= up[i] or f_up <= low[i]:
                signal = -1
                break
    if f_low > 1 or f_up < -0:
        signal = -1
    return f_low, f_up, signal

# start from center
center = (new_up + new_low) / 2
print("center", center)
re1 = E_inv @ center
print("re1", re1)

# start from new_up
# re1 = E_inv @ new_up
# print("re1", re1)


# face
def to_face(a):
    a = abs(a)
    r = 0
    for i in range(7):
        if i == 0:
            r = 1 / a[i]
        else:
            t = 1 / a[i]
            if t <= r:
                r = t
    return r

face_point = re1 * to_face(re1)
dis1 = temp.g @ face_point
print("dis1", dis1)


def next_step(dis1, step, new_up, new_low):
    step1 = np.zeros(7)
    for i in range(7):
        if dis1[i] < step[i]:
            if dis1[i] < new_low[i]:
                step1[i] = new_low[i]
            else:
                step1[i] = dis1[i]
        elif dis1[i] > step[i]:
            if dis1[i] < new_up[i]:
                step1[i] = dis1[i]
            else:
                step1[i] = new_up[i]
    return step1


# next step
step1 = next_step(dis1, center, new_up, new_low)
print("step1", step1)
re2 = E_inv @ step1
print("re2", re2)

# check pass
f_low, f_up, signal = check_pass(re1, re2)
print(f_low, f_up, signal)

face_point2 = re2 * to_face(re2)
dis2 = temp.g @ face_point2
print("dis2", dis2)

step2 = next_step(dis2, step1, new_up, new_low)
print("step2", step2)
re3 = E_inv @ step2
print("re3", re3)

f_low, f_up, signal = check_pass(re2, re3)
print(f_low, f_up, signal)

face_point3 = re3 * to_face(re3)
dis3 = temp.g @ face_point3
print("dis3", dis3)

step3 = next_step(dis3, step2, new_up, new_low)
print("step3", step3)
re4 = E_inv @ step3
print("re4", re4)

f_low, f_up, signal = check_pass(re4, re1)
print(f_low, f_up, signal)
# print(temp.g @ (f_up * re4 + (1-f_up) * re1))


face_point4 = re4 * to_face(re4)
dis4 = temp.g @ face_point4
print("dis4", dis4)

step4 = next_step(dis4, step3, new_up, new_low)
print("step4", step4)
re5 = E_inv @ step4
print("re5", re5)

f_low, f_up, signal = check_pass(re5, re4)
print(f_low, f_up, signal)

re = re5
step = step4
start = time.time()
for i in range(10):
    face_point = re * to_face(re)
    dis = temp.g @ face_point
    # print("dis", dis)

    old_step = step
    step = next_step(dis, step, new_up, new_low)
    # print("step", step)
    old_re = re
    re = E_inv @ step
    # print("re", re)
end = time.time()
print(end - start)