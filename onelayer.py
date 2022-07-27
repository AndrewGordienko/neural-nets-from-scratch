import numpy as np

epochs = 40
lr = 0.01
input_size = 3
output_size = 1
weights = 2 * np.random.random((output_size, input_size))[0] - 1

streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1],
                         [1, 0, 1]])
walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])

class Network:
    def forward(self, input):
        return input.dot(weights)

network = Network()

for epoch in range(epochs):
    error = 0
    
    for i in range(len(walk_vs_stop)):
        input = streetlights[i]
        ground_truth = walk_vs_stop[i]

        value = network.forward(input)

        error += (ground_truth - value) ** 2
        delta = value - ground_truth

        weights -= lr * (input* delta)

    print(f"error {error}")
