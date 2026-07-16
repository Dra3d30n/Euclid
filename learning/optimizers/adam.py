import numpy as np
from learning.optimizers.optimizer import Optimizer


class Adam(Optimizer):
    name = "Adam"

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):
        super().__init__(learning_rate)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        # Initialize moment estimates
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]

        self.t += 1

        for i, (param, grad) in enumerate(zip(params, grads)):
            # Update biased first moment
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update biased second moment
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameter
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)