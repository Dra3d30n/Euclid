class Network:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def gradients(self):
        raise NotImplementedError

    def train(self):
        self.training = True

    def eval(self):
        self.training = False