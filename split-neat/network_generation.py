from copy import deepcopy
import random
import numpy as np
import math

all_connections = []

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
        self.enabled = True
        self.innovation = None

class Agent():
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        self.network = [[], []]
        self.connections = []
        self.connections_weight = []
        self.node_count = 0
        self.all_nodes = []

        for i in range(self.observation_space):
            node = Neuron()
            node.number = i
            node.layer_number = 0
            self.network[0].append(deepcopy(node))
            self.node_count += 1
        for i in range(self.action_space):
            node = Neuron()
            node.number = self.observation_space + i
            node.layer_number = 1
            self.network[1].append(deepcopy(node))
            self.node_count += 1
        
        for i in range(len(self.network[0])):
            for j in range(len(self.network[1])):
                connection = Connection()
                connection.output_node_number = self.observation_space + j
                connection.output_node_layer = 1
                connection.innovation = self.finding_innovation(self.network[0][i].number, self.observation_space + j)
                self.connections.append([self.network[0][i].number, self.observation_space + j])
                self.connections_weight.append(connection.weight)
                self.network[0][i].connections_out.append(deepcopy(connection))

    def finding_innovation(self, starting_node_number, ending_node_number):
        connection_innovation = [starting_node_number, ending_node_number]

        if connection_innovation not in all_connections:
            all_connections.append(connection_innovation)
        innovation_number = all_connections.index(connection_innovation)

        return innovation_number

    def mutate_node(self): # adding node between two connected
        random_layer = random.randint(0, len(self.network)-2)
        random_node_index = random.randint(0, len(self.network[random_layer])-1)
        random_node = self.network[random_layer][random_node_index]
        random_connection_index = random.randint(0, len(random_node.connections_out)-1)
        random_connection = random_node.connections_out[random_connection_index]
        
        random_node.connections_out[random_connection_index].enabled = False
        
        output_node_number = random_connection.output_node_number
        output_node_layer = random_connection.output_node_layer
        
        if abs(random_layer - output_node_layer) == 1:
            # print("added layer")
            self.network.insert(random_layer + 1, [])
            node = Neuron()
            node.number = self.node_count
            self.node_count += 1
            node.layer_number = random_layer + 1
            self.network[random_layer + 1].append(deepcopy(node))
            new_node_index = len(self.network[random_layer + 1]) - 1
        
            connection = Connection()
            connection.output_node_number = node.number
            connection.output_node_layer = node.layer_number - 1
            connection.innovation = self.finding_innovation(random_node.number, node.number)
            self.connections.append([random_node.number, node.number])
            self.connections_weight.append(connection.weight)
            self.network[random_layer][random_node_index].connections_out.append(deepcopy(connection))

            connection = Connection()
            connection.output_node_number = output_node_number
            connection.output_node_layer = output_node_layer
            connection.innovation = self.finding_innovation(node.number, output_node_number)
            self.connections.append([node.number, output_node_number])
            self.connections_weight.append(connection.weight)
            self.network[node.layer_number][new_node_index].connections_out.append(deepcopy(connection))

            for i in range(node.layer_number+1, len(self.network)):
                for j in range(len(self.network[i])):
                    self.network[i][j].layer_number += 1
        
            for i in range(len(self.network)):
                for j in range(len(self.network[i])):
                    for k in range(len(self.network[i][j].connections_out)):
                            self.network[i][j].connections_out[k].output_node_layer += 1
            
        else:
            node = Neuron()
            node.number = self.node_count
            self.node_count += 1
            node.layer_number = random_layer + 1
            self.network[random_layer + 1].append(deepcopy(node))
            new_node_index = len(self.network[random_layer + 1]) - 1
            
            connection = Connection()
            connection.output_node_number = node.number
            connection.output_node_layer = node.layer_number
            connection.innovation = self.finding_innovation(random_node.number, node.number)
            self.connections.append([random_node.number, node.number])
            self.connections_weight.append(connection.weight)
            self.network[random_layer][random_node_index].connections_out.append(deepcopy(connection))

            connection = Connection()
            connection.output_node_number = output_node_number
            connection.output_node_layer = output_node_layer
            connection.innovation = self.finding_innovation(node.number, output_node_number)
            self.connections.append([node.number, output_node_number])
            self.connections_weight.append(connection.weight)
            self.network[node.layer_number][new_node_index].connections_out.append(deepcopy(connection))
            
    def mutate_link(self): # connect two unconnected nodes possible infinite loop exists here
        if len(self.network) > 2:
            connected = True
            bypass = 0

            while connected == True and bypass == 0:                    
                first_random_layer = random.randint(0, len(self.network)-2)
                first_random_node_index = random.randint(0, len(self.network[first_random_layer])-1)
                first_random_node = self.network[first_random_layer][first_random_node_index]

                counter = 0
                second_random_layer = random.randint(0, len(self.network)-2)
                while first_random_layer == second_random_layer:
                    second_random_layer = random.randint(0, len(self.network)-2)
                    counter += 1

                    if counter == len(self.all_nodes):
                        bypass = 1
                        break
                second_random_node_index = random.randint(0, len(self.network[second_random_layer])-1)
                second_random_node = self.network[second_random_layer][second_random_node_index]

                connected = False
                for i in range(len(first_random_node.connections_out)):
                    if first_random_node.connections_out[i].output_node_number == second_random_node.number:
                        connected = True
                for i in range(len(second_random_node.connections_out)):
                    if second_random_node.connections_out[i].output_node_number == first_random_node.number:
                        connected = True
                
                if connected == False and bypass == 0:
                    connection = Connection()
                    connection.output_node_number = second_random_node.number
                    connection.output_node_layer = second_random_node.layer_number
                    connection.innovation = self.finding_innovation(first_random_node.number, second_random_node.number)
                    self.connections.append([first_random_node.number, second_random_node.number])
                    self.connections_weight.append(connection.weight)
                    self.network[first_random_layer][first_random_node_index].connections_out.append(deepcopy(connection))
                             
    def mutate_enable_disable(self):
        random_layer = random.randint(0, len(self.network)-2)
        random_node_index = random.randint(0, len(self.network[random_layer])-1)
        random_node = self.network[random_layer][random_node_index]
        random_connection_index = random.randint(0, len(random_node.connections_out)-1)
        
        if random_node.connections_out[random_connection_index].enabled == False:
            self.network[random_layer][random_node_index].connections_out[random_connection_index].enabled = True
        else:
            self.network[random_layer][random_node_index].connections_out[random_connection_index].enabled = False

    def mutate_weight_shift(self):
        random_layer = random.randint(0, len(self.network)-2)
        random_node_index = random.randint(0, len(self.network[random_layer])-1)
        random_node = self.network[random_layer][random_node_index]
        random_connection_index = random.randint(0, len(random_node.connections_out)-1)
        number = random.uniform(0, 2)

        self.network[random_layer][random_node_index].connections_out[random_connection_index].weight *= number
        
        combination = [random_node.number, self.network[random_layer][random_node_index].connections_out[random_connection_index].output_node_number]
        number_index = self.connections.index(combination)
        self.connections_weight[number_index] *= number

    def mutate_weight_random(self):
        random_layer = random.randint(0, len(self.network)-2)
        random_node_index = random.randint(0, len(self.network[random_layer])-1)
        random_node = self.network[random_layer][random_node_index]
        random_connection_index = random.randint(0, len(random_node.connections_out)-1)
        number = random.uniform(-2, 2)

        self.network[random_layer][random_node_index].connections_out[random_connection_index].weight = random.uniform(-2, 2)

        combination = [random_node.number, self.network[random_layer][random_node_index].connections_out[random_connection_index].output_node_number]
        number_index = self.connections.index(combination)
        self.connections_weight[number_index] *= number

    def mutation(self):
        choice = random.randint(0, 10)
        if choice == 0:
            #print("mutate link")
            self.mutate_link()
        if choice == 1:
            #print("mutate node")
            self.mutate_node()
        if choice == 2 or choice == 5 or choice == 8:
            #print("enable disable")
            self.mutate_enable_disable()
        if choice == 3 or choice == 6 or choice == 9:
            #print("weight shift")
            self.mutate_weight_shift()
        if choice == 4 or choice == 7 or choice == 10:
            #print("weight random")
            self.mutate_weight_random()
