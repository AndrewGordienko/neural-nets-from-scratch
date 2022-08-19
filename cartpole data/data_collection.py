import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import gym
import random
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

EPISODES = 500
LEARNING_RATE = 0.001
MEM_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.001

FC1_DIMS = 1024
FC2_DIMS = 512
DEVICE = torch.device("cpu")

best_reward = 0
average_reward = 0
episode_number = []
average_reward_number = []

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = env.observation_space.shape

        self.action_space = action_space

        self.fc1 = nn.Linear(*self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)

        self.advantage = nn.Linear(FC2_DIMS, self.action_space)
        self.value = nn.Linear(FC2_DIMS, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.to(DEVICE)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        advantage = self.advantage(x)
        value = self.value(x)

        return (value + advantage) - advantage.mean()

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

class Agent:
    def __init__(self):
        self.memory = ReplayBuffer()
        self.exploration_rate = EXPLORATION_MAX
        self.learn_step_counter = 0
        self.net_copy_interval = 10
        self.q_values = None

        self.action_network = Network()
        self.target_network = Network()
    
    def choose_action(self, observation):
        if random.random() < self.exploration_rate:
            action = env.action_space.sample()
            
            self.q_values = []
            for i in range(action_space):
                self.q_values.append(0)
            self.q_values[action] = 1

            return env.action_space.sample()

        state = torch.tensor(observation).float().detach()
        state = state.to(DEVICE)
        state = state.unsqueeze(0)
        
        q_values = self.action_network(state)
        cleaned_q_values = q_values[0][0]
        cleaned_q_values = cleaned_q_values/sum(cleaned_q_values)
        self.q_values = cleaned_q_values.tolist()
        return np.argmax(self.q_values)
    
    def learn(self):
        if self.memory.mem_count < BATCH_SIZE:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states , dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        states_ = torch.tensor(states_, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        q_values = self.action_network(states)[batch_indices, actions] 
        next_q_values = self.target_network(states_)
        actions_ = self.action_network(states_).max(dim=1)[1]
        actions_ = next_q_values[batch_indices, actions_]

        q_target = rewards + GAMMA * actions_ * dones
        td = q_target - q_values

        self.action_network.optimizer.zero_grad()
        loss = ((td ** 2.0)).mean()
        loss.backward()
        self.action_network.optimizer.step()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

        if self.learn_step_counter % self.net_copy_interval == 0:
            self.target_network.load_state_dict(self.action_network.state_dict())

        self.learn_step_counter += 1
    
    def returning_epsilon(self):
        return self.exploration_rate
    
    def transfer(self):
        self.target_network.load_state_dict(self.action_network.state_dict())
    
agent = Agent()
all_states = []
fp = open(r'data.txt', 'w')

for i in range(1, EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, observation_space])
    score = 0
    state_pairs = []

    while True:
        env.render()
        action = agent.choose_action(state)
        state_, reward, done, info = env.step(action)
        state_ = np.reshape(state_, [1, observation_space])
        agent.memory.add(state, action, reward, state_, done)

        if done == False:
            info = agent.q_values
            saving_state = state.tolist()[0]
            saving_state += info

            state_pairs.append(saving_state)

        agent.learn()
        state = state_
        score += reward

        if done:
            if score > best_reward:
                best_reward = score
            average_reward += score 
            print("Episode {} Average Reward {} Best Reward {} Last Reward {} Epsilon {}".format(i, average_reward/i, best_reward, score, agent.returning_epsilon()))
            break
            
        episode_number.append(i)
        average_reward_number.append(average_reward/i)
    
    if score == 500:
        for item in state_pairs:
            fp.write(str(item) + '\n')
        num_lines = sum(1 for line in open('data.txt'))
        print(f"data entries {num_lines}")

        if num_lines >= 10000:
            break
        
num_lines = sum(1 for line in open('data.txt'))
print(f"data entries {num_lines}")
plt.plot(episode_number, average_reward_number)
plt.show()