import numpy as np
from Air_authenticate import Authenticate
from Air_reachability1 import Reachability
from utils.formal.zonotope import Zonotope
from copy import deepcopy
np.set_printoptions(suppress=True)
import time


class SystemALLDim:
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
        self.auth = Authenticate(exp.model, 1)    # 4 for platoon
        self.authQueueInput = []            # authentication input queue
        self.authQueueFeed = []             # authentication feedback queue
        self.authTimestep = 0               # timestep 14 for platoon
        self.authT = []                     # authentication timestep in system

        # real state calculate
        self.theta = np.zeros([1, 5, 2])  # timestep, dimension, low or up
        self.est1 = []
        self.est2 = []

        # window-based detector
        self.residuals = []
        self.detectResults = []
        self.detector = detector
        self.alarm_list = []
        self.pOrN = None

        # recovery-ability
        self.pz = Zonotope.from_box(np.ones(5) * -0.01, np.ones(5) * 0.01)    # process noise
        self.uz = Zonotope.from_box(np.array([-30]), np.array([30]))
        self.target_low = np.array([10, -4, -0.5, 12, -30])
        self.target_up = np.array([20, 4, 0.5, 16, 30])
        self.klevel = 3                                                      # keep k level recover-ability
        self.klevels = []                                                        # k-level recover-ability
        self.reach = Reachability(self.A, self.B, self.pz, self.uz, self.target_low, self.target_up)
        self.originalK = []

        self.taos = []

        self.ReachTime = 0
        self.numReach = 0
        self.AuthTime = 0
        self.numAuth = 0
        self.stateTime = 0
        self.numState = 0
        self.numAdjust = 0
        self.avgAdj = 0
        self.adjTime = 0

        justAuth = 0

        while True:
            first = 0
            exp.model.update_current_ref(exp.ref[self.i])
            exp.model.evolve()
            self.i += 1
            self.index_list.append(exp.model.cur_index)
            self.reference_list.append(exp.ref[self.i])
            self.real_cur_y = exp.model.cur_x
            self.y.append(exp.model.cur_x)
            self.y_tilda.append(exp.model.cur_x)
            self.y_hat.append(exp.model.predict)

            # under attack
            if exp.model.cur_index >= exp.attack_start_index and exp.model.cur_index <= exp.attack_start_index + attack_duration - 1:
                # attack here
                attack_step = exp.model.cur_index - exp.attack_start_index
                exp.model.cur_x[0] = exp.model.cur_x[0] + attack[attack_step]
                exp.model.cur_y = exp.model.C @ exp.model.cur_x
                exp.model.states[-1] = deepcopy(exp.model.cur_x)
                exp.model.outputs[-1] = deepcopy(exp.model.cur_y)
                # sensor measurement with attack
                self.y_tilda[-1] = exp.model.cur_x
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
                print("alarm at", exp.model.cur_index)
                return
            if self.i >= 80:
                return

            # authentication
            if len(self.authQueueInput) == self.auth.timestep:
                self.authQueueInput.pop()
                self.authQueueFeed.pop()
            self.authQueueInput.insert(0, exp.model.inputs[exp.model.cur_index - 1])
            self.authQueueFeed.insert(0, exp.model.states[exp.model.cur_index - 1])

            self.authTimestep += 1
            if self.authTimestep == self.auth.timestep:
                justAuth = 0
                self.authTimestep = 0
                authQueueInput1 = self.authQueueInput[::-1]
                authQueueFeed1 = self.authQueueFeed[::-1]
                self.auth.getInputs(authQueueInput1)
                self.auth.getFeedbacks(authQueueFeed1)

                # startAuth = time.time()
                self.auth.getAuth()
                # print('auth.x', self.auth.x)
                # print('states', exp.model.states[exp.model.cur_index - self.auth.timestep])
                # print('x_hat', self.y_hat[exp.model.cur_index - self.auth.timestep])
                self.auth.getAllBound()
                # endAuth = time.time()
                # self.AuthTime = self.AuthTime + endAuth - startAuth
                # self.numAuth = self.numAuth + 1
                # print("auth time:", self.AuthTime / self.numAuth)


                self.authT.append(exp.model.cur_index - self.auth.timestep)
                # print('timestep ', self.authT[-1])
                # print('auth.bound', self.auth.bound)

                # from auth bound to theta
                t = self.boundToTheta(self.auth.x, self.auth.bound, self.y_hat[exp.model.cur_index - self.auth.timestep])
                if len(self.authT) == 1:
                    self.theta[0] = t
                else:
                    t = t.reshape(1, 5, 2)
                    # print('rewrite ', self.i - 4, t)
                    self.theta[self.i - 5] = t

                # update real state calculate
                for k in range(3):
                    theta1 = self.boundByDetector(self.i - 5 + k + 1)
                    t = theta1.reshape(1, 5, 2)     # only use detector estimation

                    # first time authentication
                    if len(self.theta) <= 5:
                        self.theta = np.append(self.theta, t, axis=0)
                        first = 1
                        self.klevels.append(0)
                    # Rewrite previous theta
                    else:
                        self.theta[self.i - 5 + k + 1] = t
                        # print("recalculate, ", self.i - 5 + k + 1, self.theta[self.i - 5 + k + 1][0])

            else:
                justAuth = justAuth + 1

            if len(self.authT) != 0 and self.authT[-1] != self.i:
                # bound from detector


                # startState = time.time()
                theta1 = self.boundByDetector(self.i - 1)
                # endState = time.time()
                # self.stateTime = self.stateTime + endState - startState
                # self.numState += 1
                # print(self.stateTime / self.numState)



                t = theta1                                      # only use detector estimation
                t = t.reshape(1, 5, 2)
                self.theta = np.append(self.theta, t, axis=0)
                if first == 1:
                    self.klevels.append(0)
                # print('theta ', self.theta[-1])
                # print('i ', self.i)
                # print('state', exp.model.states[self.i - 1])
                # print('hat', self.y_hat[self.i - 1])

                # check correctness
                # if self.i >= 9:
                #     for i in range(7):
                #         if self.theta[-1][i][0] >= self.theta[-1][i][1]:
                #             print("violate")

                # reachability anaylze
                x_hatz = self.y_hat[-1]
                thetaz = Zonotope.from_box(self.theta[-1, :, 0], self.theta[-1, :, 1])
                # start_reach = time.time()
                kresult, start_step, end_step = self.reach.k_level(x_hatz, thetaz)
                # end_reach = time.time()
                # self.ReachTime += end_reach - start_reach
                # self.numReach += 1
                # print("avgReach: ", self.ReachTime / self.numReach)

                self.klevels.append(kresult)
                self.originalK.append(kresult)
                # print('recovery-ability: ', self.klevels[-1])



                while(True):
                    if self.klevels[-1] - self.klevel < 0 or self.klevels[-1] - self.klevel > 3:
                        # print("adjust threshold")
                        if self.klevels[-1] - self.klevel < 0:
                            inOrDe = 0
                        else:
                            inOrDe = 1

                        self.numAdjust += 1
                        startAdj = time.time()
                        delta_tau = self.reach.adjustTauNew(self.pOrN, start_step, end_step, inOrDe, self.detector)
                        endAdj = time.time()
                        self.adjTime += endAdj - startAdj
                        self.avgAdj = self.adjTime / self.numAdjust

                        if not np.any(delta_tau):
                            print("not any delta_tau")
                            exit()
                        # print("delta tao", delta_tau)
                        self.detector.adjust(delta_tau, inOrDe)
                        self.taos[-1] = deepcopy(self.detector.tao)
                        # print("self.taos", self.taos[-1])
                    else:
                        break

                    self.detectResults[-1] = self.detector.detectagain1(residual)
                    alarm = self.detector.alarmOrN()
                    if alarm:
                        print("alarm at", exp.model.cur_index)
                        # self.detector.continueWork()
                        return

                    for k in range(3 + justAuth):
                        # bound from detector
                        theta1 = self.boundByDetector(self.i - 5 - justAuth + k + 1)
                        t = theta1.reshape(1, 5, 2)  # only use detector estimation
                        self.theta[self.i - 5 - justAuth + k + 1] = t

                    # bound from detector
                    theta1 = self.boundByDetector(self.i - 1)
                    t = theta1  # only use detector estimation
                    t = t.reshape(1, 5, 2)
                    self.theta[-1] = t
                    # if first == 1:
                    #     self.klevels.append(0)
                    # print('theta ', self.theta[-1])
                    # print('i ', self.i)
                    # print('state', exp.model.states[self.i - 1])
                    # print('hat', self.y_hat[self.i - 1])

                    x_hatz = self.y_hat[-1]
                    thetaz = Zonotope.from_box(self.theta[-1, :, 0], self.theta[-1, :, 1])
                    kresult, start_step, end_step = self.reach.k_level(x_hatz, thetaz)
                    self.klevels[-1] = kresult
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
        self.pOrN = pOrN
        l = len(self.y)
        rsum = np.zeros(self.detector.m)
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
            A_theta_lo = 0
            A_theta_up = 0
            for j in range(self.A.shape[0]):
                if self.A[i][j] >=0:
                    A_theta_lo += self.A[i][j] * self.theta[t - 1, j, 0]
                    A_theta_up += self.A[i][j] * self.theta[t - 1, j, 1]
                else:
                    A_theta_lo += self.A[i][j] * self.theta[t - 1, j, 1]
                    A_theta_up += self.A[i][j] * self.theta[t - 1, j, 0]
            if pOrN[i] > 0:
                theta1[i][0] = (-self.y_hat[t] - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[t - 1] - 0.002 + self.y_tilda[t])[i] + A_theta_lo
                theta1[i][1] = (self.detector.tao - rsum + self.A @ (self.y_hat[t - 1] - self.y_tilda[t - 1]) + 0.002)[i] + A_theta_up
            else:
                theta1[i][0] = (-self.detector.tao + rsum - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[t - 1] - 0.002)[i] + A_theta_lo
                theta1[i][1] = (-self.y_hat[t] - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[t - 1] + 0.002 + self.y_tilda[t])[i] + A_theta_up
        return theta1
