import numpy as np
from neural_network.losses.loss import Loss

class CrossEntropy(Loss):
    name = "Binary Cross Entropy"

    def forward(self, prediction, target):
        prediction = np.clip(prediction, 1e-12, 1 - 1e-12)

        return -np.mean(
            target * np.log(prediction)
            + (1 - target) * np.log(1 - prediction)
        )

    def backward(self, prediction, target):
        prediction = np.clip(prediction, 1e-12, 1 - 1e-12)

        return (
            (prediction - target)
            / (prediction * (1 - prediction) * prediction.size)
        )