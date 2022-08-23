import gym
import random
import numpy as np
from network_generation import Network

env = gym.make('CartPole-v1').unwrapped
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

LEARNING_RATE = 0.0001
MEM_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.001

class ReplayBuffer:
    def __init__(self):
        self.mem_count = 0
        
        self.states = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool)
    
    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % MEM_SIZE
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1
    
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones

class Agent():
    def __init__(self):
        self.memory = ReplayBuffer()
        self.exploration_rate = EXPLORATION_MAX
        self.network = Network([observation_space, 10, action_space], LEARNING_RATE)

    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return env.action_space.sample()
        #self.action_network.printing_stats()

        q_values = self.network.forward(state, False)
        return np.argmax(q_values).item()
    
    def learn(self):
        if self.memory.mem_count < BATCH_SIZE:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample()

        for i in range(BATCH_SIZE):
            q_value = self.network.forward([states[i]], True)
            next_q_value = self.network.forward([states_[i]], False)

            predicted_value_of_now = q_value[actions[i]]
            predicted_value_of_future = np.max(next_q_value)

            q_target = rewards[i] + GAMMA * predicted_value_of_future * dones[i]

            loss = np.sum((predicted_value_of_now - q_target)**2)

            self.network.backprop(loss)

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
            



