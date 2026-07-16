class Learning:
    name = "Learning"

    def __init__(self):
        self.network = None

    def attach_network(self,network):
        self.network=network
    def train(self, data):
        raise NotImplementedError

    def evaluate(self, data):
        raise NotImplementedError