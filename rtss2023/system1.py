import numpy as np
from Authenticate import Authenticate

class System1:
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
        self.theta1 = [-0.0001]
        self.theta2 = [0.0001]
        self.est1 = []
        self.est2 = []
        self.A = exp.model.sysd.A
        self.B = exp.model.sysd.B
        self.y_tilda1 = []
        self.y1 = []

        self.queue = []         # detector queue

        # authentication
        self.auth = Authenticate(exp.model, 4)    # 4 for platoon
        self.authQueueInput = []            # authentication input queue
        self.authQueueFeed = []             # authentication feedback queue


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

            if exp.model.cur_index < exp.attack_start_index:
                # authentication
                if exp.model.cur_index >= 11:
                    if len(self.authQueueInput) == self.auth.timestep:
                        self.authQueueInput.pop()
                        self.authQueueFeed.pop()
                    self.authQueueInput.insert(0, exp.model.inputs[exp.model.cur_index])
                    # print('exp.model.inputs[-1]', exp.model.inputs[exp.model.cur_index])
                    # self.authQueueFeed.insert(0, exp.model.feedbacks[-1])
                    self.authQueueFeed.insert(0, exp.model.cur_y)
                    print('exp.model.feedbacks[exp.model.cur_index]', exp.model.cur_y)

                    if len(self.authQueueInput) == self.auth.timestep:
                        self.auth.getInputs(self.authQueueInput)
                        self.auth.getFeedbacks(self.authQueueFeed)
                        self.auth.getAuth()
                        print('auth.x', self.auth.x)
                        print('states', exp.model.cur_y)
                        # self.auth.getAllBound()
                        # print('auth.bound[0]', self.auth.bound[0])

                # window-based detector
                residual = (self.y_hat[self.i - 1] - self.y_tilda[self.i - 1])[0]
                if len(self.queue) == detector.w:
                    self.queue.pop()
                self.queue.insert(0, abs(residual))
                print(sum(self.queue))
                if sum(self.queue) > detector.tao:
                    alarm = True
                    print("alarm at", exp.model.cur_index)
                    exit(1)
                else:
                    alarm = False

            # under attack
            if exp.model.cur_index >= exp.attack_start_index and exp.model.cur_index <= exp.attack_start_index + attack_duration - 1:
                # attack here
                attack_step = exp.model.cur_index - exp.attack_start_index
                print(attack[attack_step])
                exp.model.cur_y = exp.model.cur_y[0] + attack[attack_step]
                # print(exp.model.cur_y)
                self.y_tilda[-1] = exp.model.cur_y
                self.y_tilda1.append(self.y_tilda[-1])
                self.y1.append(self.y[-1])

                # authentication
                if len(self.authQueueInput) == self.auth.timestep:
                    self.authQueueInput.pop()
                    self.authQueueFeed.pop()
                self.authQueueInput.insert(0, exp.model.inputs[-1])
                self.authQueueFeed.insert(0, exp.model.feedbacks[-1])
                if len(self.authQueueInput) == self.auth.timestep:
                    self.auth.getInputs(self.authQueueInput)
                    self.auth.getFeedbacks(self.authQueueFeed)
                    self.auth.getAuth()
                    print('auth.x', self.auth.x)
                    self.auth.getAllBound()
                    print('auth.bound[0]', self.auth.bound[0])

                # window-based detector
                residual = (self.y_hat[self.i - 1] - self.y_tilda[self.i - 1])[0]
                if len(self.queue) == detector.w:
                    self.queue.pop()
                self.queue.insert(0, abs(residual))
                print(sum(self.queue))
                if sum(self.queue) > detector.tao:
                    alarm = True
                    print("alarm at", exp.model.cur_index)
                    exit(1)
                else:
                    alarm = False
                print(alarm)
                self.alarm_list.append(alarm)
                if alarm:
                    # self.reachability()
                    break

                pOrN = residual < 0 and -1 or 1
                rsum = 0
                l = len(self.y)
                for t in range(detector.w):
                    rsum = rsum + abs(self.y_hat[l - 1 - t][0] - self.y_tilda[l - 1 - t][0])
                print(rsum)
                temp = detector.tao - rsum + abs(self.y_hat[l - 2][0] - self.y_tilda[l - 2][0])
                if pOrN > 0:
                    self.theta1.append((self.A @ [self.theta1[-1], 0])[0] - 0.0001)
                    self.theta2.append((self.A @ [temp, 0] + self.A @ [self.theta2[-1], 0])[0] + 0.0001)
                else:
                    self.theta1.append((-self.A @ [temp, 0] + self.A @ [self.theta1[-1], 0])[0] - 0.0001)
                    self.theta2.append((self.A @ [self.theta2[-1], 0])[0] + 0.0001)
                self.est1.append(self.y_hat[-1][0] + self.theta1[-1])
                self.est2.append(self.y_hat[-1][0] + self.theta2[-1])


            # after attack
            if exp.model.cur_index == exp.attack_start_index + attack_duration:
                exp.model.reset()
                self.attack_rise_alarm = False
                break
