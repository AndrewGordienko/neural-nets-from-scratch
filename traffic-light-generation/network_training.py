import numpy as np
from copy import deepcopy
import random 

class Network:
    def __init__(self, topology):
        self.network_topology = topology
        self.lr = 0.01
    
    def forward(self, state):
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

        last_values = []
        maximum_index = len(self.network_topology)-1
        for i in range(len(self.network_topology[maximum_index])):
            last_values.append(np.tanh(self.network_topology[maximum_index][i].value))

        return last_values

    def backprop(self, value, ground_truth):
        for layer in range(len(self.network_topology)):
            for node in range(len(self.network_topology[layer])):
                self.network_topology[layer][node].delta = 0
        
        last_delta = value - ground_truth

        last_layer = len(self.network_topology)-1
        for node in range(len(self.network_topology[last_layer])):
            self.network_topology[last_layer][node].delta = last_delta[node]
        
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

            self.network_topology[layer][node].delta *= self.relu2deriv(self.network_topology[layer][node].value)  

        for layer in range(len(self.network_topology)-2, -1, -1):
            for node in range(len(self.network_topology[layer])):
                for connection in range(len(self.network_topology[layer][node].connections_out)):
                    node_number = self.network_topology[layer][node].connections_out[connection].output_node_number
                    node_layer = self.network_topology[layer][node].connections_out[connection].output_node_layer

                    for element in range(len(self.network_topology[node_layer])):
                        if self.network_topology[node_layer][element].number == node_number:
                            delta = self.network_topology[node_layer][element].delta
                    
                    change = self.network_topology[layer][node].value * delta
                
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
                    print("layer {} number {} innovation {} enabled {}".format(self.network_topology[i][j].connections_out[k].output_node_layer, self.network_topology[i][j].connections_out[k].output_node_number, self.network_topology[i][j].connections_out[k].innovation, self.network_topology[i][j].connections_out[k].enabled))
