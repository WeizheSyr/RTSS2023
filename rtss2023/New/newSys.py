import numpy as np
from rtss2023.Authenticate import Authenticate
from rtss2023.reachability import Reachability
from utils.formal.zonotope import Zonotope
from copy import deepcopy

np.set_printoptions(suppress=True)


class Sys:
    def __init__(self, detector, exp, attack=None, attack_duration=50):
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
        self.auth_step = 0                   # timestep 14 for platoon
        self.authT = []                         # authentication timestep in system

        # error calculator
        self.theta = np.zeros([1, 7, 2])        # timestep, dimension of state, low/up

        # residual calculator
        self.residuals = []
        self.detect_results = []
        self.detector = detector
        self.pOrN = None

        # recoverability
        self.pz = Zonotope.from_box(np.ones(7) * -0.002, np.ones(7) * 0.002)    # noise
        self.uz = Zonotope.from_box(np.ones(4) * -5, np.ones(4) * 5)            # control box
        self.target_low = np.array([0.4, 0.4, 0.4, -0.4, -0.4, -0.4, -0.4])
        self.target_up = np.array([1.2, 1.2, 1.2, 0.4, 0.4, 0.4, 0.4])
        self.klevel = 3
        self.klevels = []
        self.reach = Reachability(self.A, self.B, self.pz, self.uz, self.target_low, self.target_up)
        self.originalK = []

        # detector
        self.taus = []

        justAuth = 0

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
                exp.model.cur_feedback[0] = exp.model.cur_feedback[0] + attack[attack_step]
                # sensor measurement with attack
                self.x_tilda[-1] = exp.model.cur_feedback

            # residual calculator
            residual = self.x_hat[self.i - 1] - self.x_tilda[self.i - 1]
            self.residuals.append(residual)
            self.detect_results.append(self.detector.detect(residual))
            alarm = self.detector.alarmOrN()
            temp = deepcopy(self.detector.tao)
            self.taus.append(temp)
            if alarm:
                print("alarm at", exp.model.cur_index)
                return
            if self.i >= 200:
            # if self.i >= 40:
                return

            # authentication
            if len(self.auth_input) == self.auth.timestep:
                self.auth_input.pop()
                self.auth_feed.pop()
            self.auth_input.insert(0, exp.model.inputs[exp.model.cur_index - 1])
            self.auth_feed.insert(0, exp.model.feedbacks[exp.model.cur_index - 1])

            self.auth_step += 1
            if self.auth_step == self.auth.timestep:
                justAuth = 0
                self.auth_step = 0
                authQueueInput1 = self.auth_input[::-1]
                authQueueFeed1 = self.auth_feed[::-1]
                self.auth.getInputs(authQueueInput1)
                self.auth.getFeedbacks(authQueueFeed1)
                self.auth.getAuth()
                print('auth.x', self.auth.x)
                print('states', self.x[exp.model.cur_index - self.auth.timestep])
                print('x_hat', self.x_hat[exp.model.cur_index - self.auth.timestep])
                self.auth.getAllBound()
                self.authT.append(exp.model.cur_index - self.auth.timestep)
                print('auth timestep ', self.authT[-1])

                # from auth bound to theta
                t = self.boundToTheta(self.auth.x, self.auth.bound,
                                      self.x_hat[exp.model.cur_index - self.auth.timestep])
                if len(self.authT) == 1:
                    self.theta[0] = t
                else:
                    t = t.reshape(1, 7, 2)
                    print('rewrite ', self.i - 6, t)
                    self.theta[self.i - 7] = t

                # update real state calculate
                for k in range(5):
                    theta1 = self.boundByDetector(self.i - 7 + k + 1)
                    t = theta1.reshape(1, 7, 2)  # only use detector estimation

                    # first time authentication
                    if len(self.theta) <= 7:
                        self.theta = np.append(self.theta, t, axis=0)
                        first = 1
                        self.klevels.append(0)
                    # Rewrite previous theta
                    else:
                        self.theta[self.i - 7 + k + 1] = t
                        print("recalculate, ", self.i - 7 + k + 1, self.theta[self.i - 7 + k + 1][0])

            else:
                justAuth = justAuth + 1