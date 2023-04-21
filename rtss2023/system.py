import numpy as np


class System:
    def __init__(self, detector, exp, attack=None, attack_duration=10):
        self.data_file = ('res/1_1data_collect_all_points' + exp.name + '.csv')
        exp_name = f"{exp.name}"
        # logger.info(f"{exp_name:=^40}")
        self.detector = detector

        # x_update = None
        self.index_list = []
        self.reference_list = []
        self.x_update_list = []
        self.x_real_list = []
        self.y_list = []
        self.control_list = []
        self.alarm_list = []
        self.residual_list = []
        self.delta_y_list = []
        self.g = []
        self.real_y = []
        self.i = 0
        self.attack_rise_alarm = False
        self.rise_alarm_index = 0
        self.predict_list = []
        self.real_cur_y = 0
        self.upbound = []
        self.lowbound = []
        self.theta = []
        self.v1 = [0]
        self.v2 = [0]

        # exp.attack_start_index = 170
        while True:
            exp.model.update_current_ref(exp.ref[self.i])
            exp.model.evolve()
            if self.i == 0:
                exp.model.cur_x = np.random.uniform(high=exp.model.cur_x, size=np.size(exp.model.cur_x))
                self.i += 1
            else:
                self.i += 1
                self.real_cur_y = exp.model.cur_y[0]

                self.index_list.append(exp.model.cur_index)
                self.reference_list.append(exp.ref[self.i])
                self.y_list.append(exp.model.cur_y)
                self.predict_list.append(exp.model.predict[exp.model.cur_index])

                if exp.model.cur_index >= exp.attack_start_index and exp.model.cur_index <= exp.attack_start_index + attack_duration - 1:
                    step = exp.model.cur_index - exp.attack_start_index
                    print(len(self.predict_list))
                    print(exp.model.cur_index)
                    pn = (self.predict_list[exp.model.cur_index - 2] - self.y_list[exp.model.cur_index - 2])[0] < 0 and -1 or 1

                    rsum = 0
                    for t in range(detector.w):
                        rsum = rsum + abs(self.predict_list[exp.model.cur_index - t - 2] - self.y_list[exp.model.cur_index - t - 2])[0]

                    temp = detector.tao - rsum + abs(self.predict_list[exp.model.cur_index - 2] - self.y_list[exp.model.cur_index - 2])[0]
                    print("temp")
                    print(temp)
                    print(pn)

                    if exp.model.cur_index == exp.attack_start_index:
                        if pn:
                            self.v2.append((exp.model.sysd.A @ [-0.01, 0])[0] - 0.01)
                            self.v1.append(temp + (exp.model.sysd.A @ [0.01, 0])[0] + 0.01)
                        else:
                            self.v2.append(-temp + (exp.model.sysd.A @ [-0.01, 0])[0] - 0.01)
                            self.v1.append((exp.model.sysd.A @ [0.01, 0])[0] + 0.01)
                    if pn:
                        self.v2.append(temp + (exp.model.sysd.A @ [self.v1[step - 1], 0])[0] + 0.01)
                        self.v1.append((exp.model.sysd.A @ [self.v2[step - 1], 0])[0] - 0.01)
                    else:
                        self.v1.append((-temp + exp.model.sysd.A @ [self.v2[step - 1], 0] - 0.01))
                        self.v2.append((exp.model.sysd.A @ [self.v1[step - 1], 0])[0] + 0.01)

                    exp.model.cur_y[0] = exp.model.cur_y[0] + attack[exp.model.cur_index - exp.attack_start_index]
                    print(exp.model.cur_y)

                if exp.model.cur_index == exp.attack_start_index + attack_duration:
                    exp.model.reset()
                    self.attack_rise_alarm = False
                    break
                # elif exp.attack_start_index <= exp.model.cur_index < exp.attack_start_index + attack_duration and alarm: #stop when alarmed
                # # elif exp.attack_start_index <= exp.model.cur_index < exp.attack_start_index + attack_duration:
                #     self.rise_alarm_index = exp.model.cur_index - exp.attack_start_index
                #     exp.model.reset()
                #     self.attack_rise_alarm = True
                #     break

        # while True:
        #     # assert exp.model.cur_index == i
        #     exp.model.update_current_ref(exp.ref[self.i])
        #     exp.model.evolve()
        #     if self.i == 0:
        #         exp.model.cur_x = np.random.uniform(high=exp.model.cur_x, size=np.size(exp.model.cur_x))
        #         # x_update = exp.model.cur_x
        #         self.i += 1
        #     else:
        #         # exp.model.cur_y = exp.attack.launch(exp.model.cur_y, i, exp.model.states)
        #         self.i += 1
        #         self.real_cur_y = exp.model.cur_y[0]
        #
        #         if attack is not None and exp.model.cur_index >= exp.attack_start_index and exp.model.cur_index <= exp.attack_start_index + attack_duration - 1:
        #             # exp.model.cur_y = exp.model.cur_y + attack[exp.model.cur_index - exp.attack_start_index]
        #
        #             step = exp.model.cur_index - exp.attack_start_index
        #             # theta
        #             rsum = 0
        #             for t in range(detector.w):
        #                 rsum = rsum + abs(self.predict_list[step - t - 1][0] - self.y_list[step - t - 1][0])
        #                 if t == detector.w - 1:
        #                     break
        #             pn = (self.predict_list[step] - self.y_list[step])[1] < 0 and -1 or 1
        #
        #             temp = detector.tao - rsum + abs(self.predict_list[step] - self.y_list[step])[1]
        #             print("temp")
        #             print(temp)
        #             print(pn)
        #             if exp.model.cur_index == exp.attack_start_index:
        #                 if pn:
        #                     self.v2.append((exp.model.sysd.A @ [-0.01, 0])[0] - 0.01)
        #                     self.v1.append(temp + (exp.model.sysd.A @ [0.01, 0])[0] + 0.01)
        #                 else:
        #                     self.v2.append(-temp + (exp.model.sysd.A @ [-0.01, 0])[0] - 0.01)
        #                     self.v1.append((exp.model.sysd.A @ [0.01, 0])[0] + 0.01)
        #             if pn:
        #                 self.v2.append(temp + (exp.model.sysd.A @ [self.v1[step - 1], 0])[0] + 0.01)
        #                 self.v1.append((exp.model.sysd.A @ [self.v2[step - 1], 0])[0] - 0.01)
        #             else:
        #                 self.v1.append((-temp + exp.model.sysd.A @ [self.v2[step - 1], 0] - 0.01))
        #                 self.v2.append((exp.model.sysd.A @ [self.v1[step - 1], 0])[0] + 0.01)
        #             # print(self.v1)
        #             # print(self.v2)
        #
        #             exp.model.cur_y[0] = exp.model.cur_y[0] + attack[exp.model.cur_index - exp.attack_start_index]
        #             print(exp.model.cur_y)
        #
        #         # logger.debug(f"i = {exp.model.cur_index}, state={exp.model.cur_x}, update={x_update},y={exp.model.cur_y}, residual={residual}, alarm={alarm}")
        #         if exp.model.cur_index >= 0:
        #             # delta_y = exp.model.cur_y - self.real_cur_y
        #             # residual = exp.model.predict[exp.model.cur_index][0] - exp.model.cur_y[0]
        #             # alarm = self.detector.detect(residual)
        #             # print(residual)
        #             # # print(alarm)
        #             # if alarm:
        #             #     print(alarm)
        #             #     break
        #
        #             self.index_list.append(exp.model.cur_index)
        #             self.reference_list.append(exp.ref[self.i])
        #             self.x_real_list.append(exp.model.cur_x)
        #             self.real_y.append(self.real_cur_y)
        #             self.y_list.append(exp.model.cur_y)
        #             self.control_list.append(exp.model.cur_u)
        #             self.predict_list.append(exp.model.predict[exp.model.cur_index])
        #
        #         # if exp.model.cur_index >= query_start_index and exp.model.cur_index - query_start_index >= N_step - 1:
        #         # print(f"model reset at {exp.model.cur_index}")
        #
        #         if exp.model.cur_index == exp.attack_start_index + attack_duration:
        #             exp.model.reset()
        #             self.attack_rise_alarm = False
        #             break
        #         elif exp.attack_start_index <= exp.model.cur_index < exp.attack_start_index + attack_duration and alarm: #stop when alarmed
        #         # elif exp.attack_start_index <= exp.model.cur_index < exp.attack_start_index + attack_duration:
        #             self.rise_alarm_index = exp.model.cur_index - exp.attack_start_index
        #             exp.model.reset()
        #             self.attack_rise_alarm = True
        #             break