import numpy as np

epochs = 40
lr = 0.01
input_size = 3
hidden_size = 4
output_size = 1

weights_0_1 = 2 * np.random.random((input_size, hidden_size)) - 1
weights_1_2 = 2 * np.random.random((hidden_size, output_size)) - 1

streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1],
                         [1, 0, 1]])
walk_vs_stop = np.array([[0, 1, 0, 1, 1, 0]])

class Network:
    def forward(self, input):
        self.input = input
        self.layer1 = self.relu(self.input.dot(weights_0_1))
        self.layer2 = self.relu(self.layer1.dot(weights_1_2))
        
        return self.layer2

    def relu(self, input):
        return (input > 0) * input

    def relu2deriv(self, input):
        return input > 0

network = Network()

for epoch in range(epochs):
    error = 0
    
    for i in range(len(walk_vs_stop)):
        input = np.array([streetlights[i]])
        ground_truth = walk_vs_stop[0][i]

        value = network.forward(input)

        error += np.sum((ground_truth - value)**2)

        layer_2_delta = value - ground_truth

        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * network.relu2deriv(network.layer1)

        weights_1_2 -= lr * network.layer1.T.dot(layer_2_delta)
        weights_0_1 -= lr * input.T.dot(layer_1_delta)

    print(f"error is {error}")

        
    
    

