import numpy as np
from copy import deepcopy
import random 

class Node():
    def __init__(self):
        self.number = None
        self.layer_number = None
        self.value = None
        self.delta = None

        self.connections_out = []

class Connection():
    def __init__(self):
        self.output_node_number = None
        self.output_node_layer = None
        self.weight = random.uniform(-1, 1)
        self.enabled = True

class Network():
    def __init__(self, dimensions, lr):
        self.dimensions = dimensions
        self.node_number = 0
        self.lr = lr
        
        self.connections = []
        self.network_topology = []
        self.forward_run = []

        for layer in range(len(self.dimensions)):
            self.network_topology.append([])
            self.forward_run.append([])

        for layer in range(len(self.network_topology)):
            for spot in range(self.dimensions[layer]):
                node = Node()
                node.number = self.node_number
                node.layer_number = layer
                self.network_topology[layer].append(deepcopy(node))
                self.forward_run[layer].append(0)
                self.node_number += 1
        
        for layer in range(len(self.network_topology)-1):
            for node in range(len(self.network_topology[layer])):
                for following_node in range(len(self.network_topology[layer+1])):
                    connection = Connection()
                    connection.output_node_number = self.network_topology[layer+1][following_node].number
                    connection.output_node_layer = layer+1

                    self.network_topology[layer][node].connections_out.append(deepcopy(connection))
    
    def forward(self, state, run):
        for i in range(len(self.network_topology)):
            for j in range(len(self.network_topology[i])):
                self.network_topology[i][j].value = 0

        for i in range(len(self.network_topology[0])):
            self.network_topology[0][i].value = state[0][i]
        
        for i in range(len(self.network_topology)-1):
            for j in range(len(self.network_topology[i])):
                for k in range(len(self.network_topology[i][j].connections_out)):
                    if self.network_topology[i][j].connections_out[k].enabled == True:
                        output_node_layer = self.network_topology[i][j].connections_out[k].output_node_layer
                        output_node_number = self.network_topology[i][j].connections_out[k].output_node_number
                        weight = self.network_topology[i][j].connections_out[k].weight

                        for p in range(len(self.network_topology[output_node_layer])):
                            if self.network_topology[output_node_layer][p].number == output_node_number:
                                self.network_topology[output_node_layer][p].value += self.network_topology[i][j].value * weight

            if i != 0 and i != len(self.network_topology)-1:
                for j in range(len(self.network_topology[i])):
                        self.network_topology[i][j].value = self.ReLU(self.network_topology[i][j].value)

                        if run == True:
                            self.forward_run[i][j] = self.ReLU(self.network_topology[i][j].value)

        last_values = []
        maximum_index = len(self.network_topology)-1
        for i in range(len(self.network_topology[maximum_index])):
            last_values.append(np.tanh(self.network_topology[maximum_index][i].value))

        return last_values

    def backprop(self, mse):
        for layer in range(len(self.network_topology)):
            for node in range(len(self.network_topology[layer])):
                self.network_topology[layer][node].delta = 0

        last_layer = len(self.network_topology)-1
        for node in range(len(self.network_topology[last_layer])):
            self.network_topology[last_layer][node].delta = mse * self.forward_run[last_layer][node]

        for layer in range(len(self.network_topology)-2, -1, -1):
            for node in range(len(self.network_topology[layer])):
                for connection in range(len(self.network_topology[layer][node].connections_out)):
                    if self.network_topology[layer][node].connections_out[connection].enabled:
                        node_number = self.network_topology[layer][node].connections_out[connection].output_node_number
                        node_layer = self.network_topology[layer][node].connections_out[connection].output_node_layer
                        weight = self.network_topology[layer][node].connections_out[connection].weight

                        for element in range(len(self.network_topology[node_layer])):
                            if self.network_topology[node_layer][element].number == node_number:
                                delta = self.network_topology[node_layer][element].delta
                        
                        self.network_topology[layer][node].delta += weight * delta

            self.network_topology[layer][node].delta *= self.relu2deriv(self.forward_run[layer][node]) 

        for layer in range(len(self.network_topology)-2, -1, -1):
            for node in range(len(self.network_topology[layer])):
                for connection in range(len(self.network_topology[layer][node].connections_out)):
                    node_number = self.network_topology[layer][node].connections_out[connection].output_node_number
                    node_layer = self.network_topology[layer][node].connections_out[connection].output_node_layer

                    for element in range(len(self.network_topology[node_layer])):
                        if self.network_topology[node_layer][element].number == node_number:
                            delta = self.network_topology[node_layer][element].delta
                    
                    change = self.forward_run[layer][node] * delta
                
                    self.network_topology[layer][node].connections_out[connection].weight -= self.lr * change

    def ReLU(self, x):
        return x * (x > 0)
    
    def relu2deriv(self, input):
        return int(input > 0)
    
    def printing_stats(self):
        # this was just for me to print out the entire network and all the connections to see that everything works
        for i in range(len(self.network_topology)):
            print("--")
            print("layer {}".format(i))
            for j in range(len(self.network_topology[i])):
                print("node {} layer {} number {}".format(self.network_topology[i][j], self.network_topology[i][j].layer_number, self.network_topology[i][j].number))

                for k in range(len(self.network_topology[i][j].connections_out)):
                    print("layer {} number {} innovation {} enabled {}".format(self.network_topology[i][j].connections_out[k].output_node_layer, self.network_topology[i][j].connections_out[k].output_node_number))
