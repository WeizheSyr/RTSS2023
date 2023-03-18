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
        self.v = []


        # exp.attack_start_index = 170

        while True:
            # assert exp.model.cur_index == i
            exp.model.update_current_ref(exp.ref[self.i])
            exp.model.evolve()
            # print(exp.model.cur_index)
            if self.i == 0:
                exp.model.cur_x = np.random.uniform(high=exp.model.cur_x, size=np.size(exp.model.cur_x))
                # x_update = exp.model.cur_x
                self.i += 1
            else:
                # exp.model.cur_y = exp.attack.launch(exp.model.cur_y, i, exp.model.states)
                self.i += 1
                self.real_cur_y = exp.model.cur_y[0]
                # print(cur_y[0])

                if attack is not None and exp.model.cur_index >= exp.attack_start_index and exp.model.cur_index <= exp.attack_start_index + attack_duration - 1:
                    # exp.model.cur_y = exp.model.cur_y + attack[exp.model.cur_index - exp.attack_start_index]

                    step = 50 + exp.model.cur_index - exp.attack_start_index
                    # theta
                    rsum = 0
                    for t in range(detector.w):
                        rsum = rsum + abs(self.predict_list[step - t - 1][0] - self.y_list[step - t - 1][0])
                        if t == detector.w - 1:
                            break
                    print(detector.tao - rsum + exp.model.sysd.A @ [0.01, 0.01] - 0.01)
                    self.v.append(detector.tao - rsum + exp.model.sysd.A @ [0.01, 0.01] - 0.01)

                    # k = exp.model.cur_index - exp.attack_start_index
                    # if k == 0:
                    #     self.theta.append(exp.model.sysd.A @ (self.v[0], self.v[0]) + 0.01)
                    # elif k == 1:
                    #     self.theta.append(abs(self.predict_list[k-1] - self.y_list[k-1]))
                    # else:
                    #     a = exp.model.sysd.A @ (self.theta[k - 1] - self.theta[k - 2])
                    #     self.theta.append(abs(self.theta[k-1] + abs(self.predict_list[k-1] - self.y_list[k-1]) + a))
                    # print('theta')
                    # print(self.theta[k])
                    # print(attack[exp.model.cur_index - exp.attack_start_index])


                    exp.model.cur_y[0] = exp.model.cur_y[0] + attack[exp.model.cur_index - exp.attack_start_index]
                    # print(self.real_cur_y)
                    # print(exp.model.cur_y[0])

                    #
                    # if k == 0:
                    #     rsum = 0
                    #     for t in range(detector.w):
                    #         rsum = rsum + abs(self.predict_list[step - t - 1][0] - self.y_list[step - t - 1][0])
                    #         if t == detector.w - 1:
                    #             break
                    #     self.lowbound.append(detector.tao - rsum - (exp.model.sysd.A @ self.y_list[step - 1])[0] + (exp.model.sysd.A @ (self.predict_list[step - 1] + [0.01, 0.01]))[0] + 0.01 + exp.model.cur_y[0])
                    #     self.upbound.append((exp.model.sysd.A @ [1,1] * detector.tao)[0] - )
                    # b = 0
                    # c = 0
                    # if k == 0:
                    #     b = 0.01
                    #     c = 0.01
                    #     self.lowbound.append(detector.tao - b + (exp.model.sysd.A @ [0.01, 0.01])[0] + exp.model.predict[exp.model.cur_index][0] + 0.01)
                    #     self.upbound.append((exp.model.sysd.A @ [3, 3])[0] - c + exp.model.cur_y[0] - 0.01 * ((exp.model.sysd.A + np.eye(2)) @ [1,1])[0] + (exp.model.sysd.A @ exp.model.sysd.A @ [0.01, 0.01])[0])
                    # if k == 1:
                    #     b = abs(self.predict_list[k - 1] - self.y_list[k - 1])
                    #     c = 0.01
                    #     self.lowbound.append(detector.tao - b[0] + (exp.model.sysd.A @ self.theta[k - 1])[0] + exp.model.predict[exp.model.cur_index][0] + 0.01)
                    #     self.upbound.append((exp.model.sysd.A @ [3, 3])[0] - c + exp.model.cur_y[0] - 0.01 * ((exp.model.sysd.A + np.eye(2)) @ [1, 1])[0] + (exp.model.sysd.A @ exp.model.sysd.A @ [0.01, 0.01])[0])
                    # if k >= 2:
                    #     k = exp.model.cur_index - exp.attack_start_index
                    #     for t in range(k - 1):
                    #         b = b + abs(self.predict_list[k - t] - self.y_list[k - t])
                    #     if k == 2:
                    #         c = abs(self.predict_list[k - t] - self.y_list[k - t])
                    #     if k > 2:
                    #         for t in range(k - 2):
                    #             c = c + abs(self.predict_list[k - t] - self.y_list[k - t])
                    #     print('ABC')
                    #     print(abs(self.predict_list[k - t] - self.y_list[k - t]))
                    #     self.lowbound.append(detector.tao - b[0] + (exp.model.sysd.A @ self.theta[k - 1])[0] + exp.model.predict[exp.model.cur_index][0] + 0.01)
                    #     self.upbound.append((exp.model.sysd.A @ [3, 3])[0] - (exp.model.sysd.A @ c)[0] + exp.model.cur_y[0] - 0.01 * ((exp.model.sysd.A + np.eye(2)) @ [1,1])[0] + (exp.model.sysd.A @ exp.model.sysd.A @ self.theta[k - 2])[0])
                    # print('lowbound')
                    # print(self.lowbound[k])
                    # print('upbound')
                    # print(self.upbound[k])
                    # print(f'exp.model.cur_y:{exp.model.cur_y}')
                # exp.model.cur_y, end_query = self.query.launch(exp.model.cur_y, exp.model.cur_index, query_type)


                # delta_y = exp.model.cur_y - self.real_cur_y
                # residual = exp.model.predict[exp.model.cur_index][0] - exp.model.cur_y[0]



                # x_update, P_update, residual = kf.one_step(x_update, kf_P, exp.model.cur_u, exp.model.cur_y)
                # exp.model.cur_feedback = x_update
                # kf_P = P_update
                # if detector.name =='CUSUM':
                #     # detector = CUSUM(threshold=0.6, drift=0.1)
                #     alarm = False
                #     for i in range(residual.size):
                #         alarm = alarm or detector.detect(residual[i])
                # else:
                #     alarm = detector.detect(residual)
                # print(residual)


                # alarm = self.detector.detect(residual)
                # print(residual)
                # # print(alarm)
                # if alarm:
                #     print(alarm)
                #     break

                # logger.debug(f"i = {exp.model.cur_index}, state={exp.model.cur_x}, update={x_update},y={exp.model.cur_y}, residual={residual}, alarm={alarm}")
                if exp.model.cur_index >= 25:
                    delta_y = exp.model.cur_y - self.real_cur_y
                    residual = exp.model.predict[exp.model.cur_index][0] - exp.model.cur_y[0]
                    alarm = self.detector.detect(residual)
                    print(residual)
                    # print(alarm)
                    if alarm:
                        print(alarm)
                        break


                    self.index_list.append(exp.model.cur_index)
                    self.reference_list.append(exp.ref[self.i])
                    self.x_real_list.append(exp.model.cur_x)
                    # self.x_update_list.append(x_update)
                    self.real_y.append(self.real_cur_y)
                    self.y_list.append(exp.model.cur_y)
                    self.control_list.append(exp.model.cur_u)
                    self.alarm_list.append(alarm)
                    self.residual_list.append(residual)
                    # self.g.append(detector.g)
                    self.delta_y_list.append(delta_y)
                    self.predict_list.append(exp.model.predict[exp.model.cur_index])

                # if exp.model.cur_index >= query_start_index and exp.model.cur_index - query_start_index >= N_step - 1:
                # print(f"model reset at {exp.model.cur_index}")

                if exp.model.cur_index == exp.attack_start_index + attack_duration:
                    exp.model.reset()
                    self.attack_rise_alarm = False
                    break
                elif exp.attack_start_index <= exp.model.cur_index < exp.attack_start_index + attack_duration and alarm: #stop when alarmed
                # elif exp.attack_start_index <= exp.model.cur_index < exp.attack_start_index + attack_duration:
                    self.rise_alarm_index = exp.model.cur_index - exp.attack_start_index
                    exp.model.reset()
                    self.attack_rise_alarm = True

                    break