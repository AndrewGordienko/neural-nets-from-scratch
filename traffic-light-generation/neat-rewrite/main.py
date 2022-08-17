import numpy as np
import gym
import math
import random
from copy import deepcopy

from network_generation import Agent
from network_training import Network
from neat_functionality import Neat

env = gym.make('CartPole-v1')
observation_space = env.observation_space.shape[0] 
action_space = env.action_space.n

network_amount = 20
networks = []
threshold_species = 2
best_score = float("-inf")
best_agent = None
epochs = 10
mutations = 5
neat = Neat()

for i in range(network_amount):
    agent = Agent(observation_space, action_space)
    for i in range(mutations): agent.mutation()
    network = Network()
    network.setting(agent)
    networks.append(deepcopy(network))

best_agents = []
for i in range(int(network_amount * 0.1)):
    agent = Agent(observation_space, action_space)
    agent.fitness = float("-inf")
    network = Network()
    network.setting(agent)
    best_agents.append(deepcopy(network))

for epoch in range(epochs):
    species = []
    average_general_fitness = 0

    local_best_agents = []
    for i in range(int(network_amount * 0.1)):
        agent = Agent(observation_space, action_space)
        agent.fitness = float("-inf")
        network = Network()
        network.setting(agent)
        local_best_agents.append(deepcopy(network))

    for i in range(len(networks)):
        agent = networks[i]
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        score = 0

        while True:
            env.render()
            action = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            state_ = np.reshape(state_, [1, observation_space])
            state = state_
            score += reward

            if done:
                print("score {}".format(score))
                agent.fitness = score 
                average_general_fitness += score

                best_agents.sort(key=lambda x: x.fitness)
                if best_agents[0].fitness < agent.fitness:
                    best_fitnesss = []
                    best_agents[0] = deepcopy(agent)
                best_agents.sort(key=lambda x: x.fitness)

                local_best_agents.sort(key=lambda x: x.fitness)
                if local_best_agents[0].fitness < agent.fitness:
                    local_best_agents[0] = deepcopy(agent)
                local_best_agents.sort(key=lambda x: x.fitness)

                if score > best_score:
                    best_score = score
                    best_agent = agent

                break
    
    average_general_fitness /= len(networks)
    networks.append(best_agent)
    print("epoch {} best score {} average fitness {}".format(epoch, best_score, average_general_fitness))

    # speciation
    species.append([networks[0]])
    for i in range(1, len(networks)):
        added = False
        for j in range(len(species)):
            #print(neat.speciation(species[j][0], networks[i]))
            if neat.speciation(species[j][0], networks[i]) >= threshold_species:
                species[j].append(networks[i])
                added = True
                break
        if added == False:
            species.append([networks[i]])

    # cutting worse half
    for i in range(len(species)):
        species[i].sort(key=lambda x: x.fitness, reverse=True)
    for i in range(len(species)):
        cutting = len(species[i])//2
        new_species = species[i][0:len(species[i]) - cutting]
        species[i] = new_species

    # how many kids
    species_average_fitness = []
    for i in range(len(species)):
        isolated_average = 0
        for j in range(len(species[i])):
            isolated_average += species[i][j].fitness
        isolated_average /= len(species[i])
        species_average_fitness.append(isolated_average)

        amount = math.ceil(isolated_average/average_general_fitness * len(species[i]))

        # making new networks
        new_networks = []
        for j in range(amount):
            if amount == 1 or len(species[i]) == 1: # if only one network in species keep it
                new_networks.append(species[i][0])
                break
            else:
                generated = []
                generated.append(int(neat.selecting_score(species[i])))

                second_index = neat.selecting_score(species[i])
                while second_index == generated[0]:
                    second_index = neat.selecting_score(species[i])
                
                generated.append(int(second_index))

                first_parent = species[i][generated[0]]
                second_parent = species[i][generated[1]]

                child = neat.making_children(first_parent, second_parent, observation_space, action_space)
                new_networks.append(deepcopy(child))

    new_networks += deepcopy(local_best_agents)
    new_networks += deepcopy(best_agents)

    for i in range(len(new_networks)):
        new_networks[i].agent.mutation() 
        new_networks[i].setting(new_networks[i].agent)

    new_networks.append(deepcopy(best_agent))
    new_networks += deepcopy(local_best_agents)
    new_networks += deepcopy(best_agents)
    
    if len(new_networks) < network_amount:
        for i in range(abs(network_amount - len(new_networks))):
            agent = Agent(observation_space, action_space)
            for i in range(mutations): agent.mutation()
            network = Network()
            network.setting(agent)
            new_networks.append(deepcopy(network))


    networks = deepcopy(new_networks)
    threshold_species += 0.1 * (5 - len(species))
