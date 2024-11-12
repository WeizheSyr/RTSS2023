import numpy as np



class Testbed:
    def __init__(self):
        self.mass_fl = 0.4  # 左前悬的质量
        self.mass_fr = 0.4  # 右前悬的质量
        self.mass_rl = 0.4  # 左后悬的质量
        self.mass_rr = 0.4
        self.mass_front = self.mass_fl + self.mass_fr
        self.mass_rear = self.mass_rl + self.mass_rr
        self.mass = self.mass_front + self.mass_rear
        # self.wheelbase_ = 0.17  # 左右轮的距离
        self.wheelbase_ = 0.15  # 左右轮的距离
        self.Cf_ = -1.60 * 0.35 * 0.165 * 57.29578  # 前轮测偏刚度，左右轮之和
        self.Cr_ = -1.60 * 0.65 * 0.165 * 57.29578  # 后轮测偏刚度，左右轮之和
        self.lf_ = self.wheelbase_ * (1.0 - self.mass_front / self.mass)  # 汽车前轮到中心点的距离
        self.lr_ = self.wheelbase_ * (1.0 - self.mass_rear / self.mass)  # 汽车后轮到中心点的距离
        self.iz_ = self.lf_ * self.lf_ * self.mass_front + self.lr_ * self.lr_ * self.mass_rear  # 汽车的转动惯量
        # self.wheelfront_ = 0.23  # 前后轮的距离
        self.wheelfront_ = 0.14  # 前后轮的距离
        a = b = self.wheelfront_ / 2
        self.velocity = 0.35
        self.A = np.array(([0, 1, 0, 0],
                          [0, (self.Cf_ + self.Cr_) / self.mass / self.velocity, -(self.Cf_ + self.Cr_) / self.mass,
                           (a * self.Cf_ - b * self.Cr_) / (self.mass * self.velocity)],
                          [0, 0, 0, 1],
                          [0, (a * self.Cf_ - b * self.Cr_) / (self.iz_ * self.velocity),
                           -(a * self.Cf_ - b * self.Cr_) / self.iz_,
                           (a ** 2 * self.Cf_ + b ** 2 * self.Cr_) / (self.iz_ * self.velocity)]))
        self.B = np.array(([0], [-self.Cf_ / self.mass], [0], [-a * self.Cf_ / self.iz_]))
        self.C = np.eye(4)
        self.D = np.array(([0], [0], [0], [0]))

    

if __name__ == '__main__':
    # from utils.controllers.LQR import LQR
    sys_model = Testbed()
    print(sys_model.A)
    print(sys_model.B)

    cf_ = 155494.663
    cr_ = 155494.663
    wheelbase_ = 2.852
    steer_ratio_ = 16
    steer_single_direction_max_degree_ = 470.0
    mass_fl = 520
    mass_fr = 520
    mass_rl = 520
    mass_rr = 520
    mass_front = mass_fl + mass_fr
    mass_rear = mass_rl + mass_rr
    mass_ = mass_front + mass_rear
    lf_ = wheelbase_ * (1.0 - mass_front / mass_)
    lr_ = wheelbase_ * (1.0 - mass_rear / mass_)
    iz_ = lf_ * lf_ * mass_front + lr_ * lr_ * mass_rear
    vx = 5

    A = np.zeros((4, 4), dtype=np.float32)
    A[0, 1] = 1.0
    A[1, 1] = -(cf_ + cr_) / mass_ / vx
    A[1, 2] = (cf_ + cr_) / mass_
    A[1, 3] = (lr_ * cr_ - lf_ * cf_) / mass_ / vx
    A[2, 3] = 1.0
    A[3, 1] = (lr_ * cr_ - lf_ * cf_) / iz_ / vx
    A[3, 2] = (lf_ * cf_ - lr_ * cr_) / iz_
    A[3, 3] = -1.0 * (lf_ * lf_ * cf_ + lr_ * lr_ * cr_) / iz_ / vx

    B = np.zeros((4, 1), dtype=np.float32)
    B[1, 0] = cf_ / mass_
    B[3, 0] = lf_ * cf_ / iz_

    print(A)
    print(B)

    exit()
    Q = np.eye(4)
    R = np.eye(1) * 10
    lqr = LQR(sys_model.A, sys_model.B, Q, R)
    # feedback = np.array(([0.5, 0, 0, 0]))
    # feedback = np.array(([-0.5, 0, 0, 0]))
    # feedback = np.array(([0, 0, -0.5, 0]))
    feedback = np.array(([-0.4, 0.0, 0.0, 0.0]))
    angle = lqr.update(feedback)
    print(angle)