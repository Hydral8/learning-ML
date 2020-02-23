import pandas as pd
import numpy as np

rng = np.random.default_rng()

train = pd.DataFrame({"A": [2, 1, 5, 12], "B": [3, 1, 2, 3], "OUTPUT": [
    10, 4, 14, 30]}, index=[1, 2, 3, 4])

# print(train)


class neuralNet:

    def __init__(self):
        self.weights = rng.uniform(-1, 1, (2))

    def train(self, inputs, outputs, iterations):
        for i in range(iterations):
            output = self.output(inputs)
            error = outputs.T - output
            adjustment = 0.01 * np.dot(inputs.T, error)
            self.weights = self.weights + adjustment

    def output(self, inputs):
        return np.dot(inputs, self.weights.T)


class neuron:
    def __init__(self):
        self.w = rng.uniform(0, 1)

    def getOutput(self, input):
        return self.w * input


inputs = np.array([[2, 3], [1, 1], [5, 2], [12, 3]])
outputs = np.array([10, 4, 14, 30])

neural_network = neuralNet()
print(neural_network.weights)

neural_network.train(inputs, outputs, 10000)
print(neural_network.weights)
