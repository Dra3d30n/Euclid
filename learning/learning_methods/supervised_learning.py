from learning_methods.base_learning import Learning

class SupervisedLearning(Learning):
    name = "Supervised"

    def __init__(self, loss, optimizer):
        super().__init__()

        self.loss = loss
        self.optimizer = optimizer

    def train(self, x, y, epochs):
        for _ in range(epochs):
            prediction = self.network.forward(x)

            loss = self.loss.forward(prediction, y)
            grad = self.loss.backward()

            self.network.backward(grad)

            self.optimizer.step(
                self.network.parameters(),
                self.network.gradients()
            )

    def evaluate(self, x, y):
        prediction = self.network.forward(x)
        return self.loss.forward(prediction, y)