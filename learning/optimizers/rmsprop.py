import numpy as np
from learning.optimizers.optimizer import Optimizer

class RMSProp(Optimizer):
    name = "RMSProp"

    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)

        self.beta = beta
        self.epsilon = epsilon
        self.cache = {}

    def step(self, params, grads):
        for i, (param, grad) in enumerate(zip(params, grads)):
            if i not in self.cache:
                self.cache[i] = np.zeros_like(param)

            self.cache[i] = (
                self.beta * self.cache[i]
                + (1 - self.beta) * grad**2
            )

            param -= (
                self.learning_rate
                * grad
                / (np.sqrt(self.cache[i]) + self.epsilon)
            )