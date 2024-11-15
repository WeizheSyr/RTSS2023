import numpy as np
from Authenticate import Authenticate
from reachability1 import Reachability1
from utils.formal.zonotope import Zonotope
from copy import deepcopy
import time

np.set_printoptions(suppress=True)


class TimeConsuming:
    # @with_goto
    def __init__(self, detector, exp, attack=None, attack_duration=50):
        self.i = 0
        self.index_list = []
        self.reference_list = []
        self.real_cur_y = 0
        self.u = []
        self.y = []
        self.y_tilda = []
        self.y_hat = []
        self.A = exp.model.sysd.A
        self.B = exp.model.sysd.B
        self.y_tilda1 = []
        self.y1 = []

        # authentication
        self.auth = Authenticate(exp.model, 4)    # 4 for platoon
        self.authQueueInput = []            # authentication input queue
        self.authQueueFeed = []             # authentication feedback queue
        self.authTimestep = 0               # timestep 14 for platoon
        self.authT = []                     # authentication timestep in system

        # real state calculate
        self.theta = np.zeros([1, 7, 2])  # timestep, dimension, low or up
        self.est1 = []
        self.est2 = []

        # window-based detector
        self.residuals = []
        self.detectResults = []
        self.detector = detector
        self.alarm_list = []

        # recovery-ability
        self.pz = Zonotope.from_box(np.ones(7) * -0.002, np.ones(7) * 0.002)    # process noise
        self.uz = Zonotope.from_box(np.ones(4) * -5, np.ones(4) * 5)            # setting in Baseline.py
        self.targetz = Zonotope.from_box(np.array([0, 0, 0, 0, 0, 0, 0]), np.array([1, 1, 1, 1, 1, 1, 1]))
        self.target_low = np.array([0, 0, 0, -1, -1, -1, -1])
        self.target_up = np.array([1, 1, 1, 1, 1, 1, 1])
        self.klevel = 3                                                       # keep k level recover-ability
        self.klevels = []                                                        # k-level recover-ability
        self.reach = Reachability1(self.A, self.B, self.pz, self.uz, self.targetz, self.target_low, self.target_up)

        self.taos = []

        # time-consuming
        self.timeAll = 0
        self.timeAuth = []
        self.timeQuick = []
        self.timeReach = []
        self.timeAdjust = []

        justAuth = 0

        startTimeAll = time.time()
        while True:
            first = 0
            exp.model.update_current_ref(exp.ref[self.i])
            exp.model.evolve()
            self.i += 1
            self.index_list.append(exp.model.cur_index)
            self.reference_list.append(exp.ref[self.i])
            self.real_cur_y = exp.model.cur_y
            # self.u.append(exp.cur_u)
            self.y.append(exp.model.cur_y)
            self.y_tilda.append(exp.model.cur_y)
            self.y_hat.append(exp.model.predict)

            # under attack
            if exp.model.cur_index >= exp.attack_start_index and exp.model.cur_index <= exp.attack_start_index + attack_duration - 1:
                # attack here
                attack_step = exp.model.cur_index - exp.attack_start_index
                exp.model.cur_y[0] = exp.model.cur_y[0] + attack[attack_step]
                # sensor measurement with attack
                self.y_tilda[-1] = exp.model.cur_y
                self.y_tilda1.append(self.y_tilda[-1])
                # sensor measurement without attack
                self.y1.append(self.y[-1])

            # window-based detector
            residual = self.y_hat[self.i - 1] - self.y_tilda[self.i - 1]
            self.residuals.append(residual)
            self.detectResults.append(self.detector.detect(residual))
            alarm = self.detector.alarmOrN()
            temp = deepcopy(self.detector.tao)
            self.taos.append(temp)
            if alarm:
                # print("alarm at", exp.model.cur_index)
                return
            if self.i >= 400:
                # time
                endTimeAll = time.time()
                self.timeAll = endTimeAll - startTimeAll
                return

            # authentication
            if len(self.authQueueInput) == self.auth.timestep:
                self.authQueueInput.pop()
                self.authQueueFeed.pop()
            self.authQueueInput.insert(0, exp.model.inputs[exp.model.cur_index - 1])
            self.authQueueFeed.insert(0, exp.model.feedbacks[exp.model.cur_index - 1])

            self.authTimestep += 1
            if self.authTimestep == self.auth.timestep:
                justAuth = 0
                self.authTimestep = 0
                authQueueInput1 = self.authQueueInput[::-1]
                authQueueFeed1 = self.authQueueFeed[::-1]
                self.auth.getInputs(authQueueInput1)
                self.auth.getFeedbacks(authQueueFeed1)

                # time
                startTimeAuth = time.time()
                self.auth.getAuth()

                # time
                endTimeAuth = time.time()
                self.timeAuth.append(endTimeAuth - startTimeAuth)

                self.auth.getAllBound()
                self.authT.append(exp.model.cur_index - self.auth.timestep)

                # from auth bound to theta
                t = self.boundToTheta(self.auth.x, self.auth.bound, self.y_hat[exp.model.cur_index - self.auth.timestep])
                if len(self.authT) == 1:
                    self.theta[0] = t
                else:
                    t = t.reshape(1, 7, 2)
                    # print('rewrite ', self.i - 6, t)
                    self.theta[self.i - 7] = t

                # update real state calculate
                for k in range(5):
                    # bound from detector
                    startTimeQuick = time.time()

                    theta1 = self.boundByDetector(self.i - 7 + k + 1)
                    t = theta1.reshape(1, 7, 2)     # only use detector estimation

                    endTimeQuick = time.time()
                    self.timeQuick.append(endTimeQuick - startTimeQuick)

                    # first time authentication
                    if len(self.theta) <= 7:
                        self.theta = np.append(self.theta, t, axis=0)
                        first = 1
                        self.klevels.append(0)
                    # Rewrite previous theta
                    else:
                        self.theta[self.i - 7 + k + 1] = t
                        # print("recalculate, ", self.i - 7 + k + 1, self.theta[self.i - 7 + k + 1][0])
            else:
                justAuth = justAuth + 1

            if len(self.authT) != 0 and self.authT[-1] != self.i:
                # bound from detector
                startTimeQuick = time.time()

                theta1 = self.boundByDetector(self.i - 1)
                t = theta1                                      # only use detector estimation
                t = t.reshape(1, 7, 2)
                self.theta = np.append(self.theta, t, axis=0)
                if first == 1:
                    self.klevels.append(0)

                endTimeQuick = time.time()
                self.timeQuick.append(endTimeQuick - startTimeQuick)

                # print('theta ', self.theta[-1][0])
                # print('i ', self.i)
                # print('state', exp.model.feedbacks[self.i - 1][0])
                # print('hat', self.y_hat[self.i - 1][0])

                # reachability anaylze
                startTimeReach = time.time()

                x_hatz = self.y_hat[-1]
                thetaz = Zonotope.from_box(self.theta[-1, :, 0], self.theta[-1, :, 1])
                kresult, start_step, end_step = self.reach.k_level(x_hatz, thetaz)
                self.klevels.append(kresult)

                endTimeReach = time.time()
                self.timeReach.append(endTimeReach - startTimeReach)
                # print('recovery-ability: ', self.klevels[-1])

                while(True):
                    if abs(self.klevels[-1] - self.klevel) >= 2:
                        # adjust threshold
                        # print('adjust threshold')

                        startTimeAdjust = time.time()

                        delta_theta = self.reach.adjust_new(kresult, start_step, end_step, self.klevel)
                        # print("delta_theta", delta_theta)
                        self.detector.adjust(delta_theta)

                        endTimeAdjust = time.time()
                        self.timeAdjust.append(endTimeAdjust - startTimeAdjust)

                        # temp = deepcopy(self.detector.tao)
                        self.taos[-1] = deepcopy(self.detector.tao)
                    else:
                        break

                    self.detectResults[-1] = self.detector.detectagain1(residual)
                    alarm = self.detector.alarmOrN()
                    if alarm:
                        # print("alarm at", exp.model.cur_index)
                        return

                    for k in range(5 + justAuth):
                        # bound from detector
                        startTimeQuick = time.time()
                        theta1 = self.boundByDetector(self.i - 7 - justAuth + k + 1)
                        t = theta1.reshape(1, 7, 2)  # only use detector estimation
                        self.theta[self.i - 7 - justAuth + k + 1] = t
                        endTimeQuick = time.time()
                        self.timeQuick.append(endTimeQuick - startTimeQuick)

                    # bound from detector
                    startTimeQuick = time.time()
                    theta1 = self.boundByDetector(self.i - 1)
                    endTimeQuick = time.time()
                    self.timeQuick.append(endTimeQuick - startTimeQuick)
                    t = theta1  # only use detector estimation
                    t = t.reshape(1, 7, 2)
                    self.theta[-1] = t
                    # print('theta ', self.theta[-1])
                    # print('i ', self.i)
                    # print('state', exp.model.feedbacks[self.i - 1])
                    # print('hat', self.y_hat[self.i - 1])

                    startTimeReach = time.time()

                    x_hatz = self.y_hat[-1]
                    thetaz = Zonotope.from_box(self.theta[-1, :, 0], self.theta[-1, :, 1])
                    kresult, start_step, end_step = self.reach.k_level(x_hatz, thetaz)
                    self.klevels[-1] = kresult

                    endTimeReach = time.time()
                    self.timeReach.append(endTimeReach - startTimeReach)

                    # print('recovery-ability: ', self.klevels[-1])

            # after attack
            if exp.model.cur_index == exp.attack_start_index + attack_duration:
                exp.model.reset()
                self.attack_rise_alarm = False
                break

    def boundToTheta(self, x_prime, bound, x_hat):
        theta = np.zeros([len(bound), 2])
        # each dimension
        for i in range(len(bound)):
            low = x_prime[i] + bound[i][0]
            up = x_prime[i] + bound[i][1]
            theta[i][0] = low - x_hat[i]
            theta[i][1] = up - x_hat[i]
        return theta

    # t: timestep according to residuals
    def boundByDetector(self, t):
        pOrN = [0] * self.detector.m
        for i in range(self.detector.m):
            pOrN[i] = self.residuals[t][i] < 0 and -1 or 1
        l = len(self.y)
        rsum = np.zeros(self.detector.m)
        # for i in range(t):
        if len(self.residuals) >= self.detector.w:
            for i in range(self.detector.w):
                rsum += abs(self.residuals[t - i])
        else:
            for i in range(len(self.residuals)):
                rsum += abs(self.residuals[t - i])

        temp = np.array(self.detector.tao) - rsum + abs(self.y_hat[t] - self.y_tilda[t])
        temp = self.A @ temp

        theta1 = np.zeros([self.detector.m, 2])

        for i in range(len(pOrN)):
            if pOrN[i] > 0:
                theta1[i][0] = (-self.y_hat[t] - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[t - 1] + self.A @ self.theta[t - 1, :, 0] - 0.002 + self.y_tilda[t])[i]
                theta1[i][1] = (self.detector.tao - rsum + self.A @ (self.y_hat[t - 1] - self.y_tilda[t - 1]) + self.A @ self.theta[t - 1, :, 1] + 0.002)[i]
            else:
                theta1[i][0] = (-self.detector.tao + rsum - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[t - 1] + self.A @ self.theta[t - 1, :, 0] - 0.002)[i]
                theta1[i][1] = (-self.y_hat[t] - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[t - 1] + self.A @ self.theta[t - 1, :, 1] + 0.002 + self.y_tilda[t])[i]
        return theta1