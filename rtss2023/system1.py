import numpy as np

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
        self.theta1 = [-0.001]
        self.theta2 = [0.001]
        self.est1 = []
        self.est2 = []
        self.A = exp.model.sysd.A
        self.B = exp.model.sysd.B

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

            residual = (self.y_hat[self.i - 1] - self.y_tilda[self.i - 1])[0]
            # alarm = detector.detect(residual)
            # self.alarm_list.append(alarm)

            # under attack
            if exp.model.cur_index >= exp.attack_start_index and exp.model.cur_index <= exp.attack_start_index + attack_duration - 1:
                # attack here
                attack_step = exp.model.cur_index - exp.attack_start_index
                # exp.model.cur_y += exp.model.cur_y + attack[attack_step]
                self.y_tilda[-1] = exp.model.cur_y

                pOrN = residual < 0 and -1 or 1
                rsum = 0
                l = len(self.y)
                for t in range(detector.w):
                    rsum = rsum + abs(self.y_hat[l - 1 - t][0] - self.y_tilda[l - 1 - t][0])
                # print(rsum)
                temp = detector.tao - rsum + abs(self.y_hat[l - 2][0] - self.y_tilda[l - 2][0])
                if pOrN > 0:
                    self.theta1.append((self.A @ [self.theta1[-1], 0])[0] - 0.001)
                    self.theta2.append((self.A @ [temp, 0] + self.A @ [self.theta2[-1], 0])[0] + 0.001)
                else:
                    self.theta1.append((-self.A @ [temp, 0] + self.A @ [self.theta1[-1], 0])[0] - 0.001)
                    self.theta2.append((self.A @ [self.theta2[-1], 0])[0] + 0.001)
                self.est1.append(self.y_hat[-1][0] + self.theta1[-1])
                self.est2.append(self.y_hat[-1][0] + self.theta2[-1])


            # after attack
            if exp.model.cur_index == exp.attack_start_index + attack_duration:
                exp.model.reset()
                self.attack_rise_alarm = False
                break