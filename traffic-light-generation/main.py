import numpy as np
from network_generation import Agent
from network_training import Network

observation_space = 3
action_space = 1

mutations = 50
epochs = 20

agent = Agent(observation_space, action_space)
streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1],
                         [1, 0, 1]])
walk_vs_stop = np.array([[0, 1, 0, 1, 1, 0]])

for i in range(mutations):
    agent.mutation()

network = Network(agent.network)
network.printing_stats()

for epoch in range(epochs):
    error = 0

    for i in range(len(walk_vs_stop[0])):
        input = np.array([streetlights[i]])
        ground_truth = walk_vs_stop[0][i]

        value = network.forward(input)

        error += np.sum((ground_truth - value)**2)

        network.backprop(value, ground_truth)
    
    print(f"error is {error}")
