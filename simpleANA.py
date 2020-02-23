# successful at analyzing patterns where only 1 value influences output
# still working on getting the network to recognize when 2 values (0, 1st index have to both be 1 to result in 1, and otherwise 0)
# basically an AND circuit
# will likely need a multi-layer neural network
import numpy as np

rng = np.random.default_rng()
inputs = np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1]])

outputs = np.array([1, 1, 0])


def normalize(weightedSum):
    return 1 / (1 + np.e ** (-weightedSum))


vector_normalize = np.vectorize(normalize)


class neural_network:

    def __init__(self):
        self.weights = [0, 0, 1]
        print(self.weights)

    def train(self, inputs, outputs, iterations):
        for iteration in range(iterations):
            for i in range(len(inputs)):
                output = self.guess(inputs[i])
                error = outputs[i] - output
                # print(f"expected: {outputs[i]} error: {error}")
                # print(output)
                adjustment = error * inputs[i]
                # print(f"adjustment: {adjustment}")
                self.weights = self.weights + adjustment

    def guess(self, inputVal):
        weightedSum = inputVal.dot(self.weights)
        return normalize(weightedSum)


neural_net = neural_network()
neural_net.train(inputs, outputs, 10000)
print(neural_net.weights)

inputVal = np.array([0, 0, 1])
print(neural_net.guess(inputVal))
