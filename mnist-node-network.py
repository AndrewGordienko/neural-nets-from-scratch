import random
from copy import deepcopy
import numpy as np
from keras.datasets import mnist

batch_size = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()
images, labels = (x_train[0:batch_size].reshape(batch_size, 28*28)/255, y_train[0:batch_size])
x_test, y_test = (x_test[0:batch_size].reshape(batch_size, 28*28)/255, y_test[0:batch_size])

random.seed(0)

class Neuron():
    def __init__(self):
        self.number = None
        self.layer_number = None
        self.value = None
        self.connections_out = []

class Connection():
    def __init__(self):
        self.output_node_number = None
        self.output_node_layer = None
        self.weight = random.uniform(-0.1, 0.1)

class Network():
    def __init__(self):
        self.input_size = 784
        self.hidden_size = 40
        self.output_size = 10
        self.network_topology = [[], [], []]
        self.amount = [self.input_size, self.hidden_size, self.output_size]
        neuron_total = 0
        self.lr = 0.005

        for i in range(len(self.network_topology)):
            for j in range(self.amount[i]):
                neuron = Neuron()
                neuron.number = neuron_total
                neuron.layer_number = i
                self.network_topology[i].append(deepcopy(neuron))
                neuron_total += 1
        
        for i in range(len(self.network_topology)-1):
            for j in range(len(self.network_topology[i])):
                for q in range(len(self.network_topology[i+1])):
                    connection = Connection()
                    connection.output_node_number = self.network_topology[i+1][q].number
                    connection.output_node_layer = i+1

                    self.network_topology[i][j].connections_out.append(deepcopy(connection))

    def forward(self, state):
        for i in range(len(self.network_topology)):
            for j in range(len(self.network_topology[i])):
                self.network_topology[i][j].value = 0

        for i in range(len(self.network_topology[0])):
            self.network_topology[0][i].value = state[0][i]
        
        for i in range(len(self.network_topology)-1):
            if i != 0 and i != len(self.network_topology)-1:
                for j in range(len(self.network_topology[i])):
                        self.network_topology[i][j].value = self.ReLU(self.network_topology[i][j].value)

            for j in range(len(self.network_topology[i])):
                for k in range(len(self.network_topology[i][j].connections_out)):
                    output_node_layer = self.network_topology[i][j].connections_out[k].output_node_layer
                    output_node_number = self.network_topology[i][j].connections_out[k].output_node_number
                    weight = self.network_topology[i][j].connections_out[k].weight

                    for p in range(len(self.network_topology[output_node_layer])):
                        if self.network_topology[output_node_layer][p].number == output_node_number:
                            self.network_topology[output_node_layer][p].value += self.network_topology[i][j].value * weight

        last_values = []
        maximum_index = len(self.network_topology)-1
        for i in range(len(self.network_topology[maximum_index])):
            last_values.append(np.tanh(self.network_topology[maximum_index][i].value))

        return last_values
    
    def backprop(self, value, ground_truth):
        layer_2_delta = value - ground_truth
        layer_1_delta = []

        last_layer = len(self.network_topology) - 1
        second_last_layer = len(self.network_topology) - 2
        first_layer = len(self.network_topology) - 3
        
        for i in range(len(self.network_topology[second_last_layer])):
            for j in range(len(self.network_topology[second_last_layer][i].connections_out)):
                weight = self.network_topology[second_last_layer][i].connections_out[j].weight
                deriv = self.relu2deriv(self.network_topology[second_last_layer][i].value)

                node_number = self.network_topology[second_last_layer][i].connections_out[j].output_node_number
                for p in range(len(self.network_topology[last_layer])):
                        if self.network_topology[last_layer][p].number == node_number:
                            index = p

                layer_1_delta.append(layer_2_delta[0][index] * weight * deriv)

        for i in range(len(self.network_topology[second_last_layer])):
            for j in range(len(self.network_topology[second_last_layer][i].connections_out)):
                node_number = self.network_topology[second_last_layer][i].connections_out[j].output_node_number
                for p in range(len(self.network_topology[last_layer])):
                        if self.network_topology[last_layer][p].number == node_number:
                            index = p
                self.network_topology[second_last_layer][i].connections_out[j].weight -= self.lr * self.network_topology[second_last_layer][i].value * layer_2_delta[0][index]
        
        for i in range(len(self.network_topology[first_layer])):
            for j in range(len(self.network_topology[first_layer][i].connections_out)):
                node_number = self.network_topology[first_layer][i].connections_out[j].output_node_number
                for p in range(len(self.network_topology[second_last_layer])):
                        if self.network_topology[second_last_layer][p].number == node_number:
                            index = p
                self.network_topology[first_layer][i].connections_out[j].weight -= self.lr * self.network_topology[first_layer][i].value * layer_1_delta[index]

    def ReLU(self, x):
        return x * (x > 0)
    
    def relu2deriv(self, input):
        return int(input > 0)

    def printing_stats(self):
        for i in range(len(self.network_topology)):
            print("--")
            print("layer {}".format(i))
            for j in range(len(self.network_topology[i])):
                print("node {} layer {} number {}".format(self.network_topology[i][j], self.network_topology[i][j].layer_number, self.network_topology[i][j].number))
                for k in range(len(self.network_topology[i][j].connections_out)):
                    print("connected to node on layer {} number {} weight {}".format(self.network_topology[i][j].connections_out[k].output_node_layer, self.network_topology[i][j].connections_out[k].output_node_number, self.network_topology[i][j].connections_out[k].weight))

network = Network()

#print("original network:")
#network.printing_stats()

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

#print("v2")
#network.printing_stats()
