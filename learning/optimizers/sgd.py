import numpy as np
from learning.optimizers.optimizer import Optimizer

class SGD(Optimizer):
    name = "SGD"

    def step(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad