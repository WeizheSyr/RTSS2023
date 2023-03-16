import numpy as np


class System:
    def __init__(self, detector, exp, attack=None, attack_duration=10):
        self.data_file = ('res/1_1data_collect_all_points' + exp.name + '.csv')
        exp_name = f"{exp.name}"
        # logger.info(f"{exp_name:=^40}")
        detector = detector

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
                cur_y = exp.model.cur_y

                if attack is not None and exp.model.cur_index >= exp.attack_start_index and exp.model.cur_index <= exp.attack_start_index + attack_duration:
                    exp.model.cur_y = exp.model.cur_y + attack[exp.model.cur_index - exp.attack_start_index]
                    # print(f'exp.model.cur_y:{exp.model.cur_y}')
                # exp.model.cur_y, end_query = self.query.launch(exp.model.cur_y, exp.model.cur_index, query_type)
                delta_y = exp.model.cur_y - cur_y
                residual = exp.model.predict[exp.model.cur_index] - exp.model.cur_y
                # x_update, P_update, residual = kf.one_step(x_update, kf_P, exp.model.cur_u, exp.model.cur_y)
                # exp.model.cur_feedback = x_update
                # kf_P = P_update
                if detector.name =='CUSUM':
                    # detector = CUSUM(threshold=0.6, drift=0.1)
                    alarm = False
                    for i in range(residual.size):
                        alarm = alarm or detector.detect(residual[i])
                else:
                    alarm = detector.detect(residual)


                # logger.debug(f"i = {exp.model.cur_index}, state={exp.model.cur_x}, update={x_update},y={exp.model.cur_y}, residual={residual}, alarm={alarm}")
                if exp.model.cur_index >= exp.attack_start_index - 50:
                    self.index_list.append(exp.model.cur_index)
                    self.reference_list.append(exp.ref[self.i])
                    self.x_real_list.append(exp.model.cur_x)
                    # self.x_update_list.append(x_update)
                    self.real_y.append(cur_y)
                    self.y_list.append(exp.model.cur_y)
                    self.control_list.append(exp.model.cur_u)
                    self.alarm_list.append(alarm)
                    self.residual_list.append(detector.gp)
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