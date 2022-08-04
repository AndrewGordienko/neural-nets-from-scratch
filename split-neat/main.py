import gym
from copy import deepcopy
import random
import numpy as np
import math

from network_generation import Agent
from neat import Neat
from network_training import Network

env = gym.make('CartPole-v1') 
observation_space = env.observation_space.shape[0] 
action_space = env.action_space.n

network_amount = 10
networks = []
threshold_species = 5.5
species = []
average_general_fitness = 0
best_score = float("-inf")
best_agent = None
epochs = 10
mutations = 5

neat_functions = Neat()

for i in range(network_amount):
    agent = Agent(observation_space, action_space)
    for i in range(mutations): agent.mutation()
    networks.append(deepcopy(Network(agent)))

for e in range(epochs): # some amount of generations
    species = []
    average_general_fitness = 0

    for i in range(len(networks)): # play through each network
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
                agent.fitness = score # this is important later for making kids
                average_general_fitness += score

                if score > best_score:
                    best_score = score
                    best_agent = agent

                break

    average_general_fitness /= network_amount
    networks.append(best_agent) # always keep the best agent
    print("epoch {} best score {} average fitness {}".format(e, best_score, average_general_fitness))
    
    species.append([networks[0]])

    for i in range(1, len(networks)):
        added = False
        for j in range(len(species)):
            if neat_functions.speciation(species[j][0], networks[i]) >= threshold_species: # create the species based on if its higher than the treshold
                species[j].append(networks[i])
                added = True
                break

        if added == False:
            species.append([networks[i]])
    
    # the code here is all about cutting the worse 50%
    for i in range(len(species)):
        species[i].sort(key=lambda x: x.fitness, reverse=True)

    for i in range(len(species)):
        cutting = len(species[i])//2
        new_species = species[i][0:len(species[i]) - cutting]
        species[i] = new_species

    print(len(species))
    # how many kids
    species_average_fitness = []
    new_networks = []

    for i in range(len(species)):
        isolated_average = 0
        for j in range(len(species[i])):
            isolated_average += species[i][j].fitness
        isolated_average /= len(species[i])
        species_average_fitness.append(isolated_average)

        amount = math.ceil(isolated_average/average_general_fitness * len(species[i])) # there is a calculation how many kids each species should have based on how it performs relatively 

        for j in range(amount):
            if amount == 1 or len(species[i]) == 1: # if only one network in species keep it
                new_networks.append(species[i][0])
                break
            else:
                generated = random.sample(species[i], 2) # else randomly make a new kid based off two parents that exist in the species
                first_parent = generated[0]
                second_parent = generated[1]

                child = neat_functions.making_children(first_parent, second_parent) # make child and add it
                new_networks.append(deepcopy(child))

    for i in range(len(new_networks)):
        new_networks[i].agent.mutation() # mutate everyone
        new_networks[i].setting()

    new_networks.append(deepcopy(best_agent))
    networks = new_networks # put the new networks in the networks list and run it again
