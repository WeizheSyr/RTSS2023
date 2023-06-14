import numpy as np
from Authenticate import Authenticate

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
        self.alarm_list = []
        # timestep, dimension, low or up
        self.theta = np.zeros(0, detector.m, 2)

        # self.theta1 = [[] for i in range(detector.m)]
        # self.theta2 = [[] for i in range(detector.m)]
        # self.theta1 = np.array(self.theta1)
        # self.theta2 = np.array(self.theta2)

        self.est1 = []
        self.est2 = []
        self.A = exp.model.sysd.A
        self.B = exp.model.sysd.B
        self.y_tilda1 = []
        self.y1 = []

        # window-based detector
        self.residuals = []
        self.detectResults = []
        self.detector = detector

        # authentication
        self.auth = Authenticate(exp.model, 4)    # 4 for platoon
        self.authQueueInput = []            # authentication input queue
        self.authQueueFeed = []             # authentication feedback queue
        self.authTimestep = 0               # timestep 14 for platoon

        while True:
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

            # without attack
            if exp.model.cur_index < exp.attack_start_index:
                # authentication
                if exp.model.cur_index >= 600:
                    if len(self.authQueueInput) == self.auth.timestep:
                        self.authQueueInput.pop()
                        self.authQueueFeed.pop()
                    self.authQueueInput.insert(0, exp.model.inputs[exp.model.cur_index - 1])
                    # print('exp.model.inputs[-1]', exp.model.inputs[exp.model.cur_index - 1])
                    self.authQueueFeed.insert(0, exp.model.feedbacks[exp.model.cur_index - 1])
                    # print('exp.model.cur_y', exp.model.feedbacks[exp.model.cur_index - 1])

                    self.authTimestep += 1
                    if self.authTimestep == self.auth.timestep:
                        authQueueInput1 = self.authQueueInput[::-1]
                        authQueueFeed1 = self.authQueueFeed[::-1]
                        self.auth.getInputs(authQueueInput1)
                        self.auth.getFeedbacks(authQueueFeed1)
                        self.auth.getAuth()
                        print('auth.x', self.auth.x)
                        print('states', exp.model.feedbacks[exp.model.cur_index - self.auth.timestep])
                        self.auth.getAllBound()
                        print('auth.bound[0]', self.auth.bound[0])

                # window-based detector
                residual = self.y_hat[self.i - 1] - self.y_tilda[self.i - 1]
                self.residuals.append(residual)
                self.detectResults.append(self.detector.detect(residual))
                alarm = self.detector.alarmOrN()
                if alarm:
                    print("alarm at", exp.model.cur_index)
                    exit(1)

            # under attack
            if exp.model.cur_index >= exp.attack_start_index and exp.model.cur_index <= exp.attack_start_index + attack_duration - 1:
                # attack here
                attack_step = exp.model.cur_index - exp.attack_start_index
                # print(attack[attack_step])
                exp.model.cur_y[0] = exp.model.cur_y[0] + attack[attack_step]
                # print(exp.model.cur_y)
                # sensor measurement with attack
                self.y_tilda[-1] = exp.model.cur_y
                self.y_tilda1.append(self.y_tilda[-1])
                # sensor measurement without attack
                self.y1.append(self.y[-1])

                # authentication
                if len(self.authQueueInput) == self.auth.timestep:
                    self.authQueueInput.pop()
                    self.authQueueFeed.pop()
                self.authQueueInput.insert(0, exp.model.inputs[-1])
                self.authQueueFeed.insert(0, exp.model.feedbacks[-1])
                self.authTimestep += 1
                if self.authTimestep == self.auth.timestep:
                    self.auth.getInputs(self.authQueueInput)
                    self.auth.getFeedbacks(self.authQueueFeed)
                    self.auth.getAuth()
                    print('auth.x', self.auth.x)
                    self.auth.getAllBound()
                    print('auth.bound[0]', self.auth.bound[0])

                # window-based detector
                residual = self.y_hat[self.i - 1] - self.y_tilda[self.i - 1]
                self.residuals.append(residual)
                self.detectResults.append(self.detector.detect(residual))
                alarm = self.detector.alarmOrN()
                if alarm:
                    print("alarm at", exp.model.cur_index)
                    exit(1)

            # real state calculate
            # from auth bound to theta
            #
            # bound from system dynamic
            # bound from detector
            #   * pOrN, y_hat, y_tilda, rsum, w
            # combine bound


                # # real state calculate
                # pOrN = [0] * detector.m
                # for i in range(detector.m):
                #     pOrN[i] = residual[i] < 0 and -1 or 1
                # # rsum = 0
                # l = len(self.y)
                # # for t in range(detector.w):
                # #     rsum = rsum + abs(self.y_hat[l - 1 - t][0] - self.y_tilda[l - 1 - t][0])
                # # print(rsum)
                # rsum = self.detector.rsum
                # # tao - sum_{i=t-w}^{t-1} |\hat{x}_i - \widetilde{x}_i|
                # temp = detector.tao - rsum + abs(self.y_hat[l - 2] - self.y_tilda[l - 2])
                # temp = self.A @ temp
                #
                # if len(self.theta1) == 0:
                #
                # for i in range(len(pOrN)):
                #     if pOrN[i] > 0:
                #         # np.append(self.theta1[i], (self.A @ self.theta1[:, -1])[i])
                #         self.theta1[i].append((self.A @ self.theta1[:, -1])[i])
                #         self.theta2[i].append((temp + [0.002] * detector.m + self.A @ self.theta2[:, -1])[i])
                #     else:
                #         self.theta1[i].append((-temp - [0.002] * detector.m + self.A @ self.theta1[:, -1])[i])
                #         self.theta2[i].append((self.A @ self.theta2[:, -1])[i])
                #
                # print('bound')
                # print('attack steps ', attack_step)
                # print(self.theta1)
                # print(self.theta2)
                #
                # self.est1.append(self.y_hat[-1][0] + self.theta1[-1])
                # self.est2.append(self.y_hat[-1][0] + self.theta2[-1])

            # after attack
            if exp.model.cur_index == exp.attack_start_index + attack_duration:
                exp.model.reset()
                self.attack_rise_alarm = False
                break

    def boundToTheta(self, x_prime, bound, x_hat):
        theta = np.zeros(len(bound), 2)
        # each dimension
        for i in range(len(bound)):
            low = x_prime[i] + bound[0]
            up = x_prime[i] + bound[1]
            theta[i][0] = low - x_hat[i]
            theta[i][1] = up - x_hat[i]
        return theta

    # t: timestep according to residuals
    def boundByDetector(self, y_hat, y_tilda, w, t):
        pOrN = [0] * self.detector.m
        for i in range(self.detector.m):
            pOrN[i] = self.residuals[t][i] < 0 and -1 or 1
        l = len(self.y)
        rsum = np.zeros(self.detector.m)
        for i in range(t):
            rsum += abs(self.residuals[t - i])

        temp = np.array(self.detector.tao) - rsum + abs(self.y_hat[t] - self.y_tilda[t])
        temp = self.A @ temp

        theta1 = np.zeros(self.detector.m, 2)
        for i in range(len(pOrN)):
            if pOrN[i] > 0:
                theta1[i][0] = (self.A @ self.theta[t - 1, :, 0])[i]
                theta1[i][1] = (temp + [0.002] * self.detector.m + self.A @ self.theta[t - 1, :, 1])[i]
            else:
                theta1[i][0] = (-temp - [0.002] * self.detector.m + self.A @ self.theta[t - 1, :, 0])[i]
                theta1[i][1] = (self.A @ self.theta[t - 1, :, 1])[i]
        return theta1

    def boundByDynamic(self, t, u):
        theta2 = np.zeros(self.detector.m, 2)
        theta2[:][0] = self.A @ (self.y_hat[t - 1] + self.theta[t - 1, :, 0]) + self.B @ u - 0.002
        theta2[:][1] = self.A @ (self.y_hat[t - 1] + self.theta[t - 1, :, 1]) + self.B @ u + 0.002
        return theta2

    def combineBound(self, theta1, theta2):
        theta = np.zeros(self.detector.m, 2)
        for i in range(len(theta)):
            if theta1[i][0] > theta2[i][0]:
                theta[i][0] = theta1[i][0]
            else:
                theta[i][0] = theta2[i][0]

            if theta1[i][1] > theta2[i][1]:
                theta[i][1] = theta2[i][1]
            else:
                theta[i][1] = theta1[i][1]
        return theta
