from neural_network.activations.linear import Linear

class Layer:
    name = "Layer"

    def __init__(self, activation=None):
        self.activation = activation or Linear()
        self.input = None
        self.output = None

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def parameters(self):
        """
        Return trainable parameters of this layer.
        Override in layers with weights/biases.
        """
        return []

    def gradients(self):
        """
        Return gradients corresponding to parameters().
        Override in layers with trainable values.
        """
        return []