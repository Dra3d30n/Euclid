import numpy as np
from losses.loss import Loss

class Huber(Loss):
    name = "Huber"

    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, prediction, target):
        error = prediction - target
        abs_error = np.abs(error)

        quadratic = 0.5 * error**2
        linear = self.delta * (abs_error - 0.5 * self.delta)

        return np.mean(np.where(abs_error <= self.delta, quadratic, linear))

    def backward(self, prediction, target):
        error = prediction - target
        abs_error = np.abs(error)

        grad = np.where(
            abs_error <= self.delta,
            error,
            self.delta * np.sign(error)
        )

        return grad / prediction.size