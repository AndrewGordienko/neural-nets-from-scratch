import numpy as np
from copy import deepcopy
import random 

class Network:
    def __init__(self, agent):
        self.network_topology = agent.network
        self.connections = agent.connections
        self.connections_weight = agent.connections_weight
        self.node_count = agent.node_count
        self.all_nodes = agent.all_nodes
        self.agent = agent

        self.lr = 0.01
    
    def setting(self):
        self.network_topology = self.agent.network
        self.connections = self.agent.connections
        self.connections_weight = self.agent.connections_weight
        self.node_count = self.agent.node_count
        self.all_nodes = self.agent.all_nodes
    
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
    
    def choose_action(self, state):
        last_values = self.forward(state)
        action = np.argmax(last_values)
        return action
    
    def ReLU(self, x):
        return x * (x > 0)
    
    def relu2deriv(self, input):
        return int(input > 0)
