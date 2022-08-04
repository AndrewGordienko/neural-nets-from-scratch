from network_generation import Agent
from network_training import Network
from keras.datasets import mnist
import numpy as np

mutations = 5

batch_size = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()
images, labels = (x_train[0:batch_size].reshape(batch_size, 28*28)/255, y_train[0:batch_size])
x_test, y_test = (x_test[0:batch_size].reshape(batch_size, 28*28)/255, y_test[0:batch_size])

agent = Agent(784, 10)
for i in range(mutations): agent.mutation()
network = Network(agent.network)


epochs = 40
for epoch in range(epochs):
    error = 0

    for i in range(len(images)):
        image = images[i:i+1]
        ground_truth = np.zeros((1, 10))
        ground_truth[0][labels[i]] = 1

        value = network.forward(image)

        error += np.sum((ground_truth - value)**2)

        network.backprop(value, ground_truth)

    print(f"error is {error}")
