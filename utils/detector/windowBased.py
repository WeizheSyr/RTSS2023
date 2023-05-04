import numpy as np

# window based detector

class window:
    def __init__(self, w=10, tao=0.1):
        self.w = w
        self.tao = tao
        self.r = 0
        self.queue = []
        self.step = 0

    def test(self):
        print('hello')

    def detect(self, residual):  # s is residual
        if len(self.queue) == self.w:
            self.queue.pop()
        self.queue.insert(0, abs(residual))

        self.r = sum(self.queue)
        if self.r > self.tao:
            return True
        else:
            return False