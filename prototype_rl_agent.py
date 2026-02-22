"""
Reinforcement Learning (RL) Agent Prototype for Jarvis AI Platform

This script demonstrates a simple RL agent using OpenAI Gym and Stable Baselines3.
"""

import gym
from stable_baselines3 import DQN

# --- Environment ---
env = gym.make("CartPole-v1")

# --- Agent ---
model = DQN("MlpPolicy", env, verbose=1)

# --- Training ---
model.learn(total_timesteps=10000)

# --- Evaluation ---
obs = env.reset()
total_reward = 0
for _ in range(200):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        break
print(f"RL agent test episode reward: {total_reward}")
