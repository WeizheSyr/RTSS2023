import numpy as np

# window based detector

# class window:
#     def __init__(self, w=10, tao=0.1):
#         self.w = w
#         self.tao = tao
#         self.r = 0
#         self.queue = []
#         self.step = 0
#
#     def test(self):
#         print('hello')
#
#     def detect(self, residual):  # s is residual
#         if len(self.queue) == self.w:
#             self.queue.pop()
#         self.queue.insert(0, abs(residual))
#
#         self.r = sum(self.queue)
#         print("detector")
#         print(self.r)
#         if self.r > self.tao:
#             return True
#         else:
#             return False

class window:
    def __init__(self, m=7, w=14):
        self.m = m              # dimension of states
        self.w = w              # observation window
        self.tao = [0] * m      # threshold tao
        self.queue = [] * m     # residual queue
        # self.queue = [[0] * m for i in range(w)]    # residual queue
        self.results = [0] * m  # detection results in this timestep
        self.rsum = [0] * m     # sum of residual for each dimension

    # residual: residual data for this dimension
    # dim: which dimension this function detect
    def detectOneDim(self, residual, dim):
        if len(self.queue) == self.w:
            self.queue[dim].pop()
        self.queue[dim].insert(0, abs(residual))

        self.rsum[dim] = sum(self.queue[dim])
        if sum(self.queue[dim]) > self.tao[dim]:
            return True    # alert
        else:
            return False    # no alert

    # residuals: residual data for all dimension in this timestep
    def detect(self, residuals):
        for i in range(self.m):
            self.results[i] = self.detectOneDim(residuals[i], i)
        return self.results
