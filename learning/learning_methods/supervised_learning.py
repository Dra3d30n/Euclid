from learning.learning_methods.learning import Learning
import settings

class SupervisedLearning(Learning):
    name = "Supervised"

    def __init__(self, loss, optimizer):
        super().__init__()

        self.loss = loss
        self.optimizer = optimizer

    def train(self, x, y, epochs):
        for epoch in range(epochs):
            prediction = self.network.forward(x)

            loss = self.loss.forward(prediction, y)

            if settings.log_training_progress and epoch%settings.training_progress_iterations_per_print==0:
                print(
                    "Epoch " + str(epoch) +
                    " out of " + str(epochs) +
                    ", Loss: " + str(loss)
                )

            grad = self.loss.backward(prediction, y)

            self.network.backward(grad)

            self.optimizer.step(
                self.network.parameters(),
                self.network.gradients()
            )

    def evaluate(self, x, y):
        prediction = self.network.forward(x)
        return self.loss.forward(prediction, y)