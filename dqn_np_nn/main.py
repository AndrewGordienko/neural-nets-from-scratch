import gym
import numpy as np
import matplotlib.pyplot as plt

from ddqn import Agent

agent = Agent()
env = gym.make('CartPole-v1').unwrapped
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

EPISODES = 500
best_reward = 0
average_reward = 0
episode_number = []
average_reward_number = []

for i in range(1, EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, observation_space])
    score = 0

    while True:
        env.render()
        action = agent.choose_action(state)
        state_, reward, done, info = env.step(action)
        state_ = np.reshape(state_, [1, observation_space])

        agent.memory.add(state, action, reward, state_, done)
        agent.learn()
        
        state = state_
        score += reward

        if done:
            if score > best_reward:
                best_reward = score
            average_reward += score 
            print("Episode {} Average Reward {} Best Reward {} Last Reward {} Epsilon {}".format(i, average_reward/i, best_reward, score, agent.exploration_rate))
            break
            
        episode_number.append(i)
        average_reward_number.append(average_reward/i)

plt.plot(episode_number, average_reward_number)
plt.show()

