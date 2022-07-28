import numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
images, labels = (x_train[0:1000].reshape(1000, 28*28)/255, y_train[0:1000])

lr = 0.005
input_size = 784
hidden_size = 40
output_size = 10
epochs = 40

weights_0_1 = 0.2 * np.random.random((input_size, hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, output_size)) - 0.1

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
    counter = 0

    for i in range(len(images)):
        image = images[i:i+1]
        ground_truth = np.zeros((1, output_size))
        ground_truth[0][labels[i]] = 1
        
        value = network.forward(image)

        error += np.sum((ground_truth - value)**2)

        layer_2_delta = value - ground_truth
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * network.relu2deriv(network.layer1)

        weights_1_2 -= lr * network.layer1.T.dot(layer_2_delta)
        weights_0_1 -= lr * network.input.T.dot(layer_1_delta)

        if np.argmax(value) == np.argmax(ground_truth):
            counter += 1

    print(f"error is {error} counter is {counter}")

    