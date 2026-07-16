import numpy as np
from learning.optimizers.optimizer import Optimizer

class SGD(Optimizer):
    name = "SGD"
    def __init__(self, learning_rate=0.001):
        super().__init__(learning_rate)

    def step(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad