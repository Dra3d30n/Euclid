import numpy as np
from neural_network.layers.layer import Layer
from neural_network.activations.linear import Linear

class Dense(Layer):
    name = "Dense"

    def __init__(self, input_size, output_size, activation=None):
        super().__init__(activation or Linear())

        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.uniform(
            -1 / np.sqrt(input_size),
            1 / np.sqrt(input_size),
            (input_size, output_size)
        )
        self.biases = np.zeros((1, output_size))

        self.d_weights = None
        self.d_biases = None

    def forward(self, x):
        self.input = x

        z = np.dot(x, self.weights) + self.biases
        self.output = z

        return self.activation.forward(z)

    def backward(self, grad):
        grad = grad * self.activation.backward(self.output)

        self.d_weights = np.dot(self.input.T, grad)
        self.d_biases = np.sum(grad, axis=0, keepdims=True)

        return np.dot(grad, self.weights.T)

    def parameters(self):
        return [self.weights, self.biases]

    def gradients(self):
        return [self.d_weights, self.d_biases]