import numpy as np
from rtss2023.Authenticate import Authenticate
from newRecoverability import Reachability
from utils.formal.zonotope import Zonotope
from copy import deepcopy

from newSys import Sys
from utils.Baseline import Platoon
from utils.detector.windowBased import window
from utils.detector.cusum1 import cusum
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(suppress=True)


class Sys:
    def __init__(self, detector, detector1, cusum, exp, attack=None, attack_duration=100):
        self.i = 0
        self.index_list = []
        self.reference_list = []
        self.u = []
        self.x = []         # x real
        self.x_tilda = []   # x tilda
        self.x_hat = []     # x hat
        self.A = exp.model.sysd.A   # A
        self.B = exp.model.sysd.B   # B

        # authentication
        self.auth = Authenticate(exp.model, 4)  # 4 for platoon
        self.auth_input = []                # authentication input queue
        self.auth_feed = []                 # authentication feedback queue
        self.auth_step = 0                   # timestep 7 for platoon
        self.authT = []                         # authentication timestep in system

        # error calculator
        # self.theta = np.zeros([1, 7, 2])        # timestep, dimension of state, low/up
        self.theta = []

        # residual calculator
        self.residuals = []
        self.detect_results = []
        self.detector = detector
        self.detector1 = detector1
        self.cusum = cusum
        self.pOrN = [0] * self.detector.m

        # recoverability
        self.pz = Zonotope.from_box(np.ones(7) * -0.001, np.ones(7) * 0.001)    # noise
        self.p_low = np.ones(7) * -0.001
        self.p_up = np.ones(7) * 0.001
        self.uz = Zonotope.from_box(np.ones(4) * -3, np.ones(4) * 3)            # control box
        # self.target_low = np.array([0.4, 0.4, 0.4, -0.4, -0.4, -0.4, -0.4])
        # self.target_up = np.array([1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2])

        # self.target_low = np.array([0.4, 0.4, 0.4, -1, -1, -1, -1])
        # self.target_up = np.array([1.2, 1.2, 1.2, 1, 1, 1, 1])
        self.target_low = np.array([0.8, 0.8, 0.8, -1, -1, -1, -1])
        self.target_up = np.array([1.8, 1.8, 1.8, 1, 1, 1, 1])

        self.safe_low = np.array([0, 0, 0, -3, -3, -3, -3])
        self.safe_up = np.array([2, 2, 2, 3, 3, 3, 3])
        self.klevel = 3
        self.klevels = []
        self.reach = Reachability(self.A, self.B, self.uz, self.pz, self.p_low, self.p_up, self.target_low, self.target_up, self.safe_low, self.safe_up)
        self.originalK = []

        # detector
        self.taus = []
        self.alarm1st = 0
        self.alarm1st1 = 0
        self.alarm1st2 = 0

        while True:
            first = 0
            exp.model.update_current_ref(exp.ref[self.i])
            exp.model.evolve()

            self.i += 1
            self.index_list.append(exp.model.cur_index)
            self.reference_list.append(exp.ref[self.i])
            t1 = deepcopy(exp.model.cur_x)
            self.x.append(t1)
            t2 = deepcopy(exp.model.sensor_x)
            self.x_tilda.append(t2)
            t3 = deepcopy(exp.model.predict)
            self.x_hat.append(t3)

            # under attack
            if exp.model.cur_index >= exp.attack_start_index and exp.model.cur_index <= exp.attack_start_index + attack_duration - 1:
                # attack here
                attack_step = exp.model.cur_index - exp.attack_start_index
                # exp.model.cur_feedback[0] = exp.model.cur_feedback[0] + attack[attack_step]
                # 0.01
                exp.model.cur_feedback[0] = exp.model.cur_feedback[0] + 0.0015
                exp.model.post_x[0] = exp.model.post_x[0] + 0.0015
                # exp.model.cur_feedback[0] = exp.model.feedbacks[-1][0]
                # exp.model.post_x[0] = exp.model.feedbacks[-1][0]
                print("attack")
                # sensor measurement with attack
                self.x_tilda[-1] = deepcopy(exp.model.post_x)

            # residual calculator
            # print('x_tilda', self.x_tilda[self.i - 1][0])
            residual = self.x_hat[self.i - 1] - self.x_tilda[self.i - 1]
            self.residuals.append(residual)
            self.detect_results.append(self.detector.detect(residual))
            alarm = self.detector.alarmOrN()
            temp = deepcopy(self.detector.tao)
            self.taus.append(temp)
            # print('sum residuals', sum(self.detector.queue[0]))
            if alarm:
                if self.alarm1st == 0:
                    self.alarm1st = self.i - 1
                print("alarm at", exp.model.cur_index)
            if self.i >= 300:
                return

            # fixed detector
            self.detector1.detect(residual)
            alarm1 = self.detector1.alarmOrN()
            if alarm1:
                if self.alarm1st1 == 0:
                    self.alarm1st1 = self.i - 1
                print("fixed alarm at", exp.model.cur_index)
            # cusum
            self.cusum.detect(residual)
            alarm2 = self.cusum.alarmOrN()
            if alarm2:
                if self.alarm1st2 == 0:
                    self.alarm1st2 = self.i - 1
                print("cusum at", exp.model.cur_index)

            # after attack
            if exp.model.cur_index == exp.attack_start_index + attack_duration:
                exp.model.reset()
                self.attack_rise_alarm = False
                break

tao = np.ones(7) * 0.02
detector = window(tao, 7, 20)
tao1 = np.ones(7) * 0.024
detector1 = window(tao1, 7, 20)
cusum = cusum(tao1, 7, 10, noise=0.001)
exp = Platoon
attack = [np.array([0.01])] * 30
attack_duration = 400

sys = Sys(detector=detector, detector1=detector1, cusum=cusum ,exp=exp, attack=attack, attack_duration=attack_duration)

max_index = sys.i
x_arr = [x[0] for x in sys.x[70:]]
# x_arr = [x[0] for x in sys.taus]
# np.save("data/1/x", sys.x)

plt.figure()
plt.plot(x_arr, c='blue', linestyle=':', label='x')
# plt.plot((sys.x[sys.alarm1st][0], sys.alarm1st), 'o', color='black')
# plt.plot((sys.x[sys.alarm1st1][0], sys.alarm1st1), 'v', color='yellow')
plt.show()
