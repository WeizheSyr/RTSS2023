import numpy as np
from rtss2023.Authenticate import Authenticate
from newRecoverability import Reachability
from utils.formal.zonotope import Zonotope
from copy import deepcopy

np.set_printoptions(suppress=True)


class Sys:
    def __init__(self, detector, detector1, cusum, exp, attack=None, attack_duration=50):
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
        self.uz = Zonotope.from_box(np.ones(4) * -5, np.ones(4) * 5)            # control box
        self.target_low = np.array([0.4, 0.4, 0.4, -0.4, -0.4, -0.4, -0.4])
        self.target_up = np.array([1.2, 1.2, 1.2, 0.4, 0.4, 0.4, 0.4])

        self.target_low = np.array([0.4, 0.4, 0.4, -1, -1, -1, -1])
        self.target_up = np.array([1.2, 1.2, 1.2, 1, 1, 1, 1])

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
            self.x.append(exp.model.cur_x)
            self.x_tilda.append(exp.model.cur_y)
            self.x_hat.append(exp.model.predict)

            # under attack
            if exp.model.cur_index >= exp.attack_start_index and exp.model.cur_index <= exp.attack_start_index + attack_duration - 1:
                # attack here
                attack_step = exp.model.cur_index - exp.attack_start_index
                # exp.model.cur_feedback[0] = exp.model.cur_feedback[0] + attack[attack_step]
                exp.model.cur_feedback[0] = 0.983
                # print("attack", attack[attack_step])
                # sensor measurement with attack
                self.x_tilda[-1] = exp.model.cur_feedback

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
            if self.i >= 70:
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


            # authentication
            if len(self.auth_input) == self.auth.timestep:
                self.auth_input.pop()
                self.auth_feed.pop()
            self.auth_input.insert(0, exp.model.inputs[exp.model.cur_index - 1])
            self.auth_feed.insert(0, exp.model.feedbacks[exp.model.cur_index - 1])

            self.auth_step += 1
            # if self.auth_step == self.auth.timestep:
            if self.auth_step >= 3 and len(self.auth_input) == self.auth.timestep:
                print("auth at ", self.i)
                self.auth_step = 0
                authQueueInput1 = self.auth_input[::-1]
                authQueueFeed1 = self.auth_feed[::-1]
                self.auth.getInputs(authQueueInput1)
                self.auth.getFeedbacks(authQueueFeed1)
                t = self.auth.getAuth()
                self.authT.append(exp.model.cur_index - self.auth.timestep)
                if t:
                    self.auth.getAllBound()
                    print('auth x ', self.authT[-1])
                else:
                    self.auth.x = self.x[self.authT[-1]]
                    for i in range(self.A.shape[0]):
                        self.auth.bound[i, 0] = - 0.001 * 2
                        self.auth.bound[i, 1] = 0.001 * 2

                # from auth bound to theta
                t = self.boundToTheta(self.auth.x, self.auth.bound, self.x_hat[self.authT[-1]])
                if len(self.authT) == 1:
                    self.theta.append(t)
                else:
                    self.theta[self.authT[-1]] = t

                # update real state calculate
                if self.i < 50:
                    for k in range(self.auth.timestep - 2):
                        theta1 = self.boundByDetector1(self.authT[-1] + 1 + k)
                        # first time authentication
                        if len(self.authT) == 1:
                            self.theta.append(theta1)
                        # Rewrite previous theta
                        else:
                            self.theta[self.authT[-1] + 1 + k] = theta1

            # error estimator
            if len(self.authT) != 0:
                theta1 = self.boundByDetector1(self.i - 1)
                self.theta.append(theta1)

            # recoverability calculator
            if self.i >= 30:
                thetaz = np.array(self.theta)
                recover = self.reach.recoverable(self.x_hat[-1], thetaz[-1, :, ])
                print("recoverable time window", recover)
                if recover[1] == 0:
                    delta_tau = self.reach.threshold_decrease()
                    self.detector.tao = self.detector.tao - delta_tau
                    print(self.detector.tao)
                elif recover[2] != 0:
                    delta_tau = self.reach.threshold_increase()
                    self.detector.tao = self.detector.tao + delta_tau
                    print(self.detector.tao)


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

    def boundByDetector(self, t):
        for i in range(self.detector.m):
            self.pOrN[i] = self.residuals[t][i] < 0 and -1 or 1
        rsum = np.zeros(self.detector.m)
        # for i in range(t):
        if len(self.residuals) >= self.detector.w:
            for i in range(self.detector.w):
                rsum += abs(self.residuals[t - i])
        else:
            for i in range(len(self.residuals)):
                rsum += abs(self.residuals[t - i])

        theta1 = np.zeros([self.detector.m, 2])

        for i in range(len(self.pOrN)):
            A_theta_lo = 0
            A_theta_up = 0
            for j in range(self.A.shape[0]):
                if self.A[i][j] >=0:
                    A_theta_lo += self.A[i][j] * self.theta[t - 1][j][0]
                    A_theta_up += self.A[i][j] * self.theta[t - 1][j][1]
                else:
                    A_theta_lo += self.A[i][j] * self.theta[t - 1][j][1]
                    A_theta_up += self.A[i][j] * self.theta[t - 1][j][0]
            if self.pOrN[i] > 0:
                theta1[i][0] = (-self.x_hat[t] - self.A @ self.x_tilda[t - 1] + self.A @ self.x_hat[t - 1] - 0.001 + self.x_tilda[t])[i] + A_theta_lo
                theta1[i][1] = (self.detector.tao - rsum + self.A @ (self.x_hat[t - 1] - self.x_tilda[t - 1]) + 0.001)[i] + A_theta_up
            else:
                theta1[i][0] = (-self.detector.tao + rsum - self.A @ self.x_tilda[t - 1] + self.A @ self.x_hat[t - 1] - 0.001)[i] + A_theta_lo
                theta1[i][1] = (-self.x_hat[t] - self.A @ self.x_tilda[t - 1] + self.A @ self.x_hat[t - 1] + 0.001 + self.x_tilda[t])[i] + A_theta_up
        return theta1

    def boundByDetector1(self, t):
        for i in range(self.detector.m):
            self.pOrN[i] = self.residuals[t][i] < 0 and -1 or 1
        rsum = np.zeros(self.detector.m)
        # for i in range(t):
        if len(self.residuals) >= self.detector.w:
            for i in range(self.detector.w):
                rsum += abs(self.residuals[t - i])
        else:
            for i in range(len(self.residuals)):
                rsum += abs(self.residuals[t - i])

        theta1 = np.zeros([self.detector.m, 2])
        for i in range(len(self.pOrN)):
            A_theta_lo = 0
            A_theta_up = 0
            for j in range(self.A.shape[0]):
                if self.A[i][j] >=0:
                    A_theta_lo += self.A[i][j] * self.theta[t - 1][j][0]
                    A_theta_up += self.A[i][j] * self.theta[t - 1][j][1]
                else:
                    A_theta_lo += self.A[i][j] * self.theta[t - 1][j][1]
                    A_theta_up += self.A[i][j] * self.theta[t - 1][j][0]
            theta1[i][0] = (-self.detector.tao +rsum + self.A @ (self.x_hat[t - 1] - self.x_tilda[t - 1]) - 0.001)[i] + A_theta_lo
            theta1[i][1] = (self.detector.tao - rsum + self.A @ (self.x_hat[t - 1] - self.x_tilda[t - 1]) + 0.001)[i] + A_theta_up
        return theta1
