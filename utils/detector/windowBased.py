import numpy as np

# window based detector

class WindowBased:
    def __init__(self, w=50, tao=3):
        self.w = w
        self.tao = tao
        self.r = 0
        self.queue = []
        self.step = 0

    def detect(self, s): # s is residual
        if len(self.queue) == self.w:
            self.queue.pop()
        self.queue.insert(0, abs(s))

        self.r = sum(self.queue)
        if self.r > self.tao:
            return False
        else:
            return True