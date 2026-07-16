import numpy as np
from neural_network.losses.loss import Loss

class MSE(Loss):
    name = "Mean Squared Error"

    def forward(self, prediction, target):
        return np.mean((prediction - target) ** 2)

    def backward(self, prediction, target):
        return 2 * (prediction - target) / prediction.size