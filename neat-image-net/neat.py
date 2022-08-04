import numpy as np
import random
from network_generation import Agent
from network_training import Network

class Neat():
    def speciation(self, species1, species2):
        N = max(len(species1.connections), len(species2.connections))
        E = abs(species1.node_count - species2.node_count)

        D = 0
        both = species1.connections + species2.connections
        for i in range(len(both)):
            if both.count(both[i]) == 1:
                D += 1
        
        W = 0
        shorter_species = species1.connections
        if species1.connections <= species2.connections:
            shorter_species = species1.connections

        for i in range(len(shorter_species)):
            connection_identified = shorter_species[i]
            if connection_identified in species1.connections:
                if connection_identified in species2.connections:
                    index_species_one = species1.connections.index(connection_identified)
                    index_species_two = species2.connections.index(connection_identified)
                    
                    W += abs(species1.connections_weight[index_species_one] - species2.connections_weight[index_species_two])

        number = E/N + D/N + 0.5*W
        #print("value is {}".format(number))
        return number

    def selecting_score(self, all_parents):
        fitness_scores = []
        for i in range(len(all_parents)):
            fitness_scores.append(all_parents[i].fitness)

        probabilities = []
        added_probs = []
        total_score = 0

        for i in range(len(fitness_scores)):
            total_score += fitness_scores[i]
        factor = 1/total_score

        for i in range(len(fitness_scores)):
            probabilities.append(round(fitness_scores[i] * factor, 2))
        
        added_probs.append(probabilities[0])
        for i in range(1, len(probabilities)):
            added_probs.append(added_probs[i-1] + probabilities[i])
        added_probs = [0] + added_probs

        roll = round(random.random(), 2)

        for i in range(1, len(added_probs)):
            if added_probs[i-1] <= roll <= added_probs[i]:
                return i-1
            if roll > added_probs[len(added_probs)-1]:
                return len(added_probs)-2

    
    def making_children(self, species1, species2, observation_space, action_space):
        parents = [species1, species2]
        index = np.argmax([int(species1.fitness), int(species2.fitness)])
        fit_parent = parents[index]
        less_fit_parent = parents[abs(1-index)]

        child = Network(Agent(observation_space, action_space))
        child.network = fit_parent
        child.node_count = fit_parent.node_count
        
        both = species1.connections + species2.connections
        connections_both = []
        parent_chosen = []
        copy_indexs = []
        both_weights = species1.connections_weight + species2.connections_weight

        for i in range(len(both)):
            if both.count(both[i]) == 2:
                if both[i] not in connections_both:
                    connections_both.append(both[i])

        for i in range(len(connections_both)):
            parent_chosen.append(random.randint(0, 1))

        for i in range(len(connections_both)):
            indices = [index for index, element in enumerate(both) if element == connections_both[i]]
            copy_indexs.append(indices)

        for i in range(len(child.network_topology)):
            for j in range(len(child.network_topology[i])):
                for k in range(len(child.network_topology[i][j].connections_out)):
                    if child.network_topology[i][j].connections_out[k].enabled == True:
                        for p in range(len(connections_both)):
                            if connections_both[p][0] == child.network_topology[i][j].number and connections_both[p][1] == child.network_topology[i][j].connections_out[k].output_node_number:
                                child.network_topology[i][j].connections_out[k].weight = both_weights[copy_indexs[p][parent_chosen[p]]]

        child.connections = []
        child.connections_weight = []
        child.all_nodes = []

        for i in range(len(child.network_topology)):
            for j in range(len(child.network_topology[i])):
                child.all_nodes.append(child.network_topology[i][j])

                for k in range(len(child.network_topology[i][j].connections_out)):
                    child.connections.append([child.network_topology[i][j].number, child.network_topology[i][j].connections_out[k].output_node_number])
                    child.connections_weight.append(child.network_topology[i][j].connections_out[k].weight)     

        return child
