import numpy as np
from Authenticate import Authenticate
from reachability1 import Reachability1
from reachability import Reachability
from utils.formal.zonotope import Zonotope
from copy import deepcopy
# from goto import with_goto

np.set_printoptions(suppress=True)


class ThreeDetector:
    # @with_goto
    def __init__(self, detector, fixed_detector, noauth_detector, exp, attack=None, attack_duration=50):
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
        self.pOrN = None

        # recovery-ability
        self.pz = Zonotope.from_box(np.ones(7) * -0.002, np.ones(7) * 0.002)    # process noise
        # self.uz = Zonotope.from_box(exp.control_lo, exp.control_up)             # setting in Baseline.py
        self.uz = Zonotope.from_box(np.ones(4) * -5, np.ones(4) * 5)
        # self.targetz = Zonotope.from_box(np.ones(7) * 0, np.ones(7) * 1)        # target set in zonotope
        # self.targetz = Zonotope.from_box(np.array([0, 0, 0, -1, -1, -1, -1]), np.array([1, 1, 1, 1, 1, 1, 1]))
        # self.targetz = Zonotope.from_box(np.array([0, 0, 0, 0, 0, 0, 0]), np.array([1, 1, 1, 1, 1, 1, 1]))
        self.target_low = np.array([0.4, 0.4, 0.4, -0.4, -0.4, -0.4, -0.4])
        self.target_up = np.array([1.2, 1.2, 1.2, 0.4, 0.4, 0.4, 0.4])
        self.klevel = 3                                                       # keep k level recover-ability
        self.klevels = []                                                        # k-level recover-ability
        self.reach = Reachability(self.A, self.B, self.pz, self.uz, self.target_low, self.target_up)
        # self.reach = Reachability1(self.A, self.B, self.pz, self.uz, self.targetz, self.target_low, self.target_up)
        self.originalK = []

        self.taos = []

        self.alertat = 0

        # fixed detector
        self.fixed_detector = fixed_detector
        self.fixed_residuals = []
        self.fixed_detectResults = []
        self.fixed_alarm_list = []
        self.fixed_theta = np.zeros([1, 7, 2])  # timestep, dimension, low or up

        # fixed detector's reachability
        self.fixed_reach = Reachability(self.A, self.B, self.pz, self.uz, self.target_low, self.target_up)
        # self.fixed_reach = Reachability1(self.A, self.B, self.pz, self.uz, self.targetz, self.target_low, self.target_up)
        self.fixed_taos = []
        self.fixed_klevels = []

        # no auth detector
        self.stop_flag = 0
        self.noauth_detector = noauth_detector
        self.noauth_residuals = []
        self.noauth_detectResults = []
        self.noauth_alarm_list = []
        self.noauth_theta = np.zeros([1, 7, 2])
        self.noauth_reach = Reachability(self.A, self.B, self.pz, self.uz, self.target_low, self.target_up)
        # self.noauth_reach = Reachability1(self.A, self.B, self.pz, self.uz, self.targetz, self.target_low, self.target_up)
        self.noauth_taos = []
        self.noauth_klevels = []

        justAuth = 0

        while True:
            first = 0
            exp.model.update_current_ref(exp.ref[self.i])
            exp.model.evolve()
            self.i += 1
            self.index_list.append(exp.model.cur_index)
            self.reference_list.append(exp.ref[self.i])
            self.real_cur_y = exp.model.cur_y
            self.y.append(exp.model.cur_y)
            self.y_tilda.append(exp.model.cur_y)
            self.y_hat.append(exp.model.predict)
            y1 = deepcopy(exp.model.cur_y)
            self.y1.append(y1)

            # under attack
            if exp.model.cur_index >= exp.attack_start_index and exp.model.cur_index <= exp.attack_start_index + attack_duration - 1:
                # print("under attack")
                # attack here
                attack_step = exp.model.cur_index - exp.attack_start_index
                print("attack step", attack_step)
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
                print("***********************************")
                print("alarm at", exp.model.cur_index)
                if self.alertat == 0:
                    self.alertat = exp.model.cur_index
                # return
            if self.i >= 75:
                return

            # fixed window-based detector
            self.fixed_residuals.append(residual)
            self.fixed_detectResults.append(self.fixed_detector.detect(residual))
            fixed_alarm = self.fixed_detector.alarmOrN()
            fixed_temp = deepcopy(self.fixed_detector.tao)
            self.fixed_taos.append(fixed_temp)
            if fixed_alarm:
                print("fixed detector alarm at", exp.model.cur_index)
                return

            # no auth detector
            if self.stop_flag == 0:
                self.noauth_residuals.append(residual)
                self.noauth_detectResults.append(self.noauth_detector.detect(residual))
                noauth_alarm = self.noauth_detector.alarmOrN()
                noauth_temp = deepcopy(self.noauth_detector.tao)
                self.noauth_taos.append(noauth_temp)
                if noauth_alarm:
                    print("no auth detector alarm at", exp.model.cur_index)
                    self.stop_flag = 1

            # authentication
            if len(self.authQueueInput) == self.auth.timestep:
                self.authQueueInput.pop()
                self.authQueueFeed.pop()
            self.authQueueInput.insert(0, exp.model.inputs[exp.model.cur_index - 1])
            self.authQueueFeed.insert(0, exp.model.feedbacks[exp.model.cur_index - 1])

            self.authTimestep += 1
            if self.authTimestep == self.auth.timestep and self.alertat == 0 :
                justAuth = 0
                self.authTimestep = 0
                authQueueInput1 = self.authQueueInput[::-1]
                authQueueFeed1 = self.authQueueFeed[::-1]
                self.auth.getInputs(authQueueInput1)
                self.auth.getFeedbacks(authQueueFeed1)
                self.auth.getAuth()
                # print('auth.x', self.auth.x)
                # print('states', exp.model.feedbacks[exp.model.cur_index - self.auth.timestep])
                # print('x_hat', self.y_hat[exp.model.cur_index - self.auth.timestep])
                self.auth.getAllBound()
                self.authT.append(exp.model.cur_index - self.auth.timestep)
                # print('timestep ', self.authT[-1])
                # print('auth.bound', self.auth.bound)

                # from auth bound to theta
                t = self.boundToTheta(self.auth.x, self.auth.bound, self.y_hat[exp.model.cur_index - self.auth.timestep])
                if len(self.authT) == 1:
                    self.theta[0] = t
                    # fixed detector
                    self.fixed_theta[0] = t
                    # no auth detector
                    if self.stop_flag == 0:
                        self.noauth_theta[0] = t
                else:
                    t = t.reshape(1, 7, 2)
                    # print('rewrite ', self.i - 6, t)
                    self.theta[self.i - 7] = t
                    # fixed detector
                    self.fixed_theta[self.i - 7] = t

                # update real state calculate
                for k in range(5):
                    # bound from detector
                    theta1 = self.boundByDetector(self.i - 7 + k + 1)
                    t = theta1.reshape(1, 7, 2)     # only use detector estimation
                    # fixed detector
                    fixed_theta1 = self.boundByDetector1(self.i - 7 + k + 1, self.fixed_detector)
                    fixed_t = fixed_theta1.reshape(1, 7, 2)
                    # no auth detector
                    if self.stop_flag == 0:
                        noauth_theta1 = self.boundByDetector2(self.i - 7 + k + 1, self.noauth_detector)
                        noauth_t = noauth_theta1.reshape(1, 7, 2)

                    # first time authentication
                    if len(self.theta) <= 7:
                        self.theta = np.append(self.theta, t, axis=0)
                        # fixed detector
                        self.fixed_theta = np.append(self.fixed_theta, fixed_t, axis=0)
                        # no auth detector
                        if self.stop_flag == 0:
                            self.noauth_theta = np.append(self.noauth_theta, noauth_t, axis=0)
                        first = 1
                        self.klevels.append(0)
                        self.originalK.append(0)
                        # fixed detector
                        self.fixed_klevels.append(0)
                        # no auth detector
                        if self.stop_flag == 0:
                            self.noauth_klevels.append(0)
                    # Rewrite previous theta
                    else:
                        self.theta[self.i - 7 + k + 1] = t
                        # print("recalculate, ", self.i - 7 + k + 1, self.theta[self.i - 7 + k + 1][0])
                        # fixed detector
                        self.fixed_theta[self.i - 7 + k + 1] = fixed_t

            else:
                justAuth = justAuth + 1

            if len(self.authT) != 0 and self.authT[-1] != self.i:
                # bound from detector
                theta1 = self.boundByDetector(self.i - 1)
                t = theta1                                      # only use detector estimation
                t = t.reshape(1, 7, 2)
                self.theta = np.append(self.theta, t, axis=0)
                if first == 1:
                    self.klevels.append(0)
                    self.originalK.append(0)
                # print('theta ', self.theta[-1][0])
                # print('i ', self.i)
                # print('state', exp.model.feedbacks[self.i - 1][0])
                # print('hat', self.y_hat[self.i - 1][0])

                # fixed detector
                fixed_theta1 = self.boundByDetector1(self.i - 1, self.fixed_detector)
                fixed_t = fixed_theta1
                fixed_t = fixed_t.reshape(1, 7, 2)
                self.fixed_theta = np.append(self.fixed_theta, fixed_t, axis=0)
                if first == 1:
                    self.fixed_klevels.append(0)

                # no auth detector
                if self.stop_flag == 0:
                    noauth_theta1 = self.boundByDetector2(self.i - 1, self.noauth_detector)
                    noauth_t = noauth_theta1
                    noauth_t = noauth_t.reshape(1, 7, 2)
                    self.noauth_theta = np.append(self.noauth_theta, noauth_t, axis=0)
                    if first == 1:
                        self.fixed_klevels.append(0)

                # reachability anaylze
                x_hatz = self.y_hat[-1]
                thetaz = Zonotope.from_box(self.theta[-1, :, 0], self.theta[-1, :, 1])
                kresult, start_step, end_step = self.reach.k_level(x_hatz, thetaz)
                self.klevels.append(kresult)
                temp = deepcopy(kresult)
                self.originalK.append(temp)
                # print('recovery-ability: ', self.klevels[-1])

                # fixed detector
                fixed_thetaz = Zonotope.from_box(self.fixed_theta[-1, :, 0], self.fixed_theta[-1, :, 1])
                fixed_kresult, fixed_start_step, fixed_end_step = self.fixed_reach.k_level(x_hatz, fixed_thetaz)
                self.fixed_klevels.append(fixed_kresult)

                # no auth detector
                if self.stop_flag == 0:
                    noauth_thetaz = Zonotope.from_box(self.noauth_theta[-1, :, 0], self.noauth_theta[-1, :, 1])
                    noauth_kresult, noauth_start_step, noauth_end_step = self.noauth_reach.k_level(x_hatz, noauth_thetaz)
                    self.noauth_klevels.append(noauth_kresult)
                    # bound too large
                    if noauth_theta1[0][1] - noauth_theta1[0][0] >= 1:
                        self.stop_flag = 1

                while(True):
                    print("self.i", self.i)
                    if self.alertat != 0:
                        break
                    if self.klevels[-1] - self.klevel < 0 or self.klevels[-1] - self.klevel > 3:
                        print("adjust threshold")
                        if self.klevels[-1] - self.klevel < 0:
                            inOrDe = 0
                        else:
                            inOrDe = 1
                        delta_tau = self.reach.adjustTauNew(self.pOrN, start_step, end_step, inOrDe, self.detector)
                        # delta_tau = self.reach.adjustTau(self.pOrN, start_step, end_step)
                        if not np.any(delta_tau):
                            print("not any delta_tau")
                            exit()
                        print("delta tao", delta_tau)
                        self.detector.adjust(delta_tau, inOrDe)
                        self.taos[-1] = deepcopy(self.detector.tao)
                        print("self.taos", self.taos[-1])
                    else:
                        break
                    # if self.alertat != 0:
                    #     break
                    # if abs(self.klevels[-1] - self.klevel) >= 2:
                    #     # adjust threshold
                    #     # print('adjust threshold')
                    #     delta_theta = self.reach.adjust_new(kresult, start_step, end_step, self.klevel)
                    #     # print("delta_theta", delta_theta)
                    #     self.detector.adjust(delta_theta)
                    #     # temp = deepcopy(self.detector.tao)
                    #     self.taos[-1] = deepcopy(self.detector.tao)
                    # else:
                    #     break

                    self.detectResults[-1] = self.detector.detectagain1(residual)
                    alarm = self.detector.alarmOrN()
                    if alarm:
                        print("alarm at", exp.model.cur_index)
                        if self.alertat == 0:
                            self.alertat = exp.model.cur_index
                        # return

                    for k in range(5 + justAuth):
                        # bound from detector
                        theta1 = self.boundByDetector(self.i - 7 - justAuth + k + 1)
                        t = theta1.reshape(1, 7, 2)  # only use detector estimation
                        self.theta[self.i - 7 - justAuth + k + 1] = t

                    # bound from detector
                    theta1 = self.boundByDetector(self.i - 1)
                    t = theta1  # only use detector estimation
                    t = t.reshape(1, 7, 2)
                    self.theta[-1] = t
                    # print('theta ', self.theta[-1])
                    # print('i ', self.i)
                    # print('state', exp.model.feedbacks[self.i - 1])
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

        # for i in range(len(pOrN)):
        #     if pOrN[i] > 0:
        #         theta1[i][0] = (-self.y_hat[t] - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[t - 1] + self.A @ self.theta[t - 1, :, 0] - 0.002 + self.y_tilda[t])[i]
        #         theta1[i][1] = (self.detector.tao - rsum + self.A @ (self.y_hat[t - 1] - self.y_tilda[t - 1]) + self.A @ self.theta[t - 1, :, 1] + 0.002)[i]
        #     else:
        #         theta1[i][0] = (-self.detector.tao + rsum - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[t - 1] + self.A @ self.theta[t - 1, :, 0] - 0.002)[i]
        #         theta1[i][1] = (-self.y_hat[t] - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[t - 1] + self.A @ self.theta[t - 1, :, 1] + 0.002 + self.y_tilda[t])[i]
        # return theta1

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

    def boundByDetector1(self, t, detector):
        pOrN = [0] * detector.m
        for i in range(detector.m):
            pOrN[i] = self.residuals[t][i] < 0 and -1 or 1
        l = len(self.y)
        rsum = np.zeros(detector.m)
        # for i in range(t):
        if len(self.residuals) >= detector.w:
            for i in range(detector.w):
                rsum += abs(self.residuals[t - i])
        else:
            for i in range(len(self.residuals)):
                rsum += abs(self.residuals[t - i])

        temp = np.array(detector.tao) - rsum + abs(self.y_hat[t] - self.y_tilda[t])
        temp = self.A @ temp

        theta1 = np.zeros([detector.m, 2])

        # for i in range(len(pOrN)):
        #     if pOrN[i] > 0:
        #         theta1[i][0] = (-self.y_hat[t] - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[
        #             t - 1] + self.A @ self.fixed_theta[t - 1, :, 0] - 0.002 + self.y_tilda[t])[i]
        #         theta1[i][1] = (detector.tao - rsum + self.A @ (
        #                     self.y_hat[t - 1] - self.y_tilda[t - 1]) + self.A @ self.fixed_theta[t - 1, :, 1] + 0.002)[i]
        #     else:
        #         theta1[i][0] = (-detector.tao + rsum - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[
        #             t - 1] + self.A @ self.fixed_theta[t - 1, :, 0] - 0.002)[i]
        #         theta1[i][1] = (-self.y_hat[t] - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[
        #             t - 1] + self.A @ self.fixed_theta[t - 1, :, 1] + 0.002 + self.y_tilda[t])[i]
        # return theta1
        for i in range(len(pOrN)):
            A_theta_lo = 0
            A_theta_up = 0
            for j in range(self.A.shape[0]):
                if self.A[i][j] >=0:
                    A_theta_lo += self.A[i][j] * self.fixed_theta[t - 1, j, 0]
                    A_theta_up += self.A[i][j] * self.fixed_theta[t - 1, j, 1]
                else:
                    A_theta_lo += self.A[i][j] * self.fixed_theta[t - 1, j, 1]
                    A_theta_up += self.A[i][j] * self.fixed_theta[t - 1, j, 0]
            if pOrN[i] > 0:
                theta1[i][0] = (-self.y_hat[t] - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[t - 1] - 0.002 + self.y_tilda[t])[i] + A_theta_lo
                theta1[i][1] = (detector.tao - rsum + self.A @ (self.y_hat[t - 1] - self.y_tilda[t - 1]) + 0.002)[i] + A_theta_up
            else:
                theta1[i][0] = (-detector.tao + rsum - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[t - 1] - 0.002)[i] + A_theta_lo
                theta1[i][1] = (-self.y_hat[t] - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[t - 1] + 0.002 + self.y_tilda[t])[i] + A_theta_up
        return theta1

    def boundByDetector2(self, t, detector):
        pOrN = [0] * detector.m
        for i in range(detector.m):
            pOrN[i] = self.residuals[t][i] < 0 and -1 or 1
        l = len(self.y)
        rsum = np.zeros(detector.m)
        # for i in range(t):
        if len(self.residuals) >= detector.w:
            for i in range(detector.w):
                rsum += abs(self.residuals[t - i])
        else:
            for i in range(len(self.residuals)):
                rsum += abs(self.residuals[t - i])

        temp = np.array(detector.tao) - rsum + abs(self.y_hat[t] - self.y_tilda[t])
        temp = self.A @ temp

        theta1 = np.zeros([detector.m, 2])

        # for i in range(len(pOrN)):
        #     if pOrN[i] > 0:
        #         theta1[i][0] = (-self.y_hat[t] - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[
        #             t - 1] + self.A @ self.noauth_theta[t - 1, :, 0] - 0.002 + self.y_tilda[t])[i]
        #         theta1[i][1] = (detector.tao - rsum + self.A @ (
        #                 self.y_hat[t - 1] - self.y_tilda[t - 1]) + self.A @ self.noauth_theta[t - 1, :, 1] + 0.002)[
        #             i]
        #     else:
        #         theta1[i][0] = (-detector.tao + rsum - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[
        #             t - 1] + self.A @ self.noauth_theta[t - 1, :, 0] - 0.002)[i]
        #         theta1[i][1] = (-self.y_hat[t] - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[
        #             t - 1] + self.A @ self.noauth_theta[t - 1, :, 1] + 0.002 + self.y_tilda[t])[i]
        # return theta1

        for i in range(len(pOrN)):
            A_theta_lo = 0
            A_theta_up = 0
            for j in range(self.A.shape[0]):
                if self.A[i][j] >=0:
                    A_theta_lo += self.A[i][j] * self.noauth_theta[t - 1, j, 0]
                    A_theta_up += self.A[i][j] * self.noauth_theta[t - 1, j, 1]
                else:
                    A_theta_lo += self.A[i][j] * self.noauth_theta[t - 1, j, 1]
                    A_theta_up += self.A[i][j] * self.noauth_theta[t - 1, j, 0]
            if pOrN[i] > 0:
                theta1[i][0] = (-self.y_hat[t] - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[t - 1] - 0.002 + self.y_tilda[t])[i] + A_theta_lo
                theta1[i][1] = (detector.tao - rsum + self.A @ (self.y_hat[t - 1] - self.y_tilda[t - 1]) + 0.002)[i] + A_theta_up
            else:
                theta1[i][0] = (-detector.tao + rsum - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[t - 1] - 0.002)[i] + A_theta_lo
                theta1[i][1] = (-self.y_hat[t] - self.A @ self.y_tilda[t - 1] + self.A @ self.y_hat[t - 1] + 0.002 + self.y_tilda[t])[i] + A_theta_up
        return theta1