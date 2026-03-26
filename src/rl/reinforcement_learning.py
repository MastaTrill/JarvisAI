"""
Advanced Reinforcement Learning Module for Aetheron AI Platform
Includes multiple RL algorithms, environments, and training frameworks
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import deque
import json
import pickle
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RLConfig:
    """Configuration for reinforcement learning"""
    algorithm: str = "dqn"
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 32
    target_update_frequency: int = 100
    max_episodes: int = 1000
    max_steps_per_episode: int = 200
    
class ReplayBuffer:
    """Experience replay buffer for RL algorithms"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Store a transition"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a batch of transitions"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class SimpleNeuralNetwork:
    """Simple neural network for RL agents"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 learning_rate: float = 0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        """Forward pass"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def backward(self, X, y, output):
        """Backward pass"""
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def predict(self, X):
        """Make predictions"""
        return self.forward(X)
    
    def copy_weights(self):
        """Return a copy of current weights"""
        return {
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy()
        }
    
    def set_weights(self, weights):
        """Set weights from dictionary"""
        self.W1 = weights['W1'].copy()
        self.b1 = weights['b1'].copy()
        self.W2 = weights['W2'].copy()
        self.b2 = weights['b2'].copy()

class DQNAgent:
    """Deep Q-Network agent"""
    
    def __init__(self, state_size: int, action_size: int, config: RLConfig):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Neural networks
        self.q_network = SimpleNeuralNetwork(state_size, 64, action_size, config.learning_rate)
        self.target_network = SimpleNeuralNetwork(state_size, 64, action_size, config.learning_rate)
        
        # Initialize target network with same weights
        self.target_network.set_weights(self.q_network.copy_weights())
        
        # Replay buffer
        self.memory = ReplayBuffer(config.memory_size)
        
        # Exploration parameters
        self.epsilon = config.epsilon_start
        
        # Training stats
        self.step_count = 0
        self.episode_rewards = []
        self.losses = []
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        q_values = self.q_network.predict(state.reshape(1, -1))
        return np.argmax(q_values[0])
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        """Train the agent using experience replay"""
        if len(self.memory) < self.config.batch_size:
            return
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)
        
        # Get current Q values
        current_q_values = self.q_network.predict(states)
        
        # Get next Q values from target network
        next_q_values = self.target_network.predict(next_states)
        max_next_q_values = np.max(next_q_values, axis=1)
        
        # Compute target Q values
        target_q_values = current_q_values.copy()
        for i in range(self.config.batch_size):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                target_q_values[i, actions[i]] = rewards[i] + self.config.discount_factor * max_next_q_values[i]
        
        # Train the network
        output = self.q_network.forward(states)
        self.q_network.backward(states, target_q_values, output)
        
        # Calculate loss for monitoring
        loss = np.mean((output - target_q_values) ** 2)
        self.losses.append(loss)
        
        # Update exploration rate
        if self.epsilon > self.config.epsilon_end:
            self.epsilon *= self.config.epsilon_decay
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.config.target_update_frequency == 0:
            self.target_network.set_weights(self.q_network.copy_weights())
    
    def save(self, filepath: str):
        """Save agent state"""
        state = {
            'q_network_weights': self.q_network.copy_weights(),
            'target_network_weights': self.target_network.copy_weights(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.q_network.set_weights(state['q_network_weights'])
        self.target_network.set_weights(state['target_network_weights'])
        self.epsilon = state['epsilon']
        self.step_count = state['step_count']
        self.episode_rewards = state['episode_rewards']
        self.losses = state['losses']
        
        logger.info(f"Agent loaded from {filepath}")

class PolicyGradientAgent:
    """REINFORCE policy gradient agent"""
    
    def __init__(self, state_size: int, action_size: int, config: RLConfig):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Policy network
        self.policy_network = SimpleNeuralNetwork(state_size, 64, action_size, config.learning_rate)
        
        # Episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        # Training stats
        self.episode_total_rewards = []
        self.losses = []
    
    def softmax(self, x):
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def act(self, state: np.ndarray) -> int:
        """Choose action based on policy"""
        logits = self.policy_network.predict(state.reshape(1, -1))
        probabilities = self.softmax(logits)[0]
        action = np.random.choice(self.action_size, p=probabilities)
        return action
    
    def remember(self, state: np.ndarray, action: int, reward: float):
        """Store episode step"""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def train_episode(self):
        """Train on completed episode"""
        if not self.episode_states:
            return
        
        # Calculate discounted rewards
        discounted_rewards = self._calculate_discounted_rewards()
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        
        # Convert to arrays
        states = np.array(self.episode_states)
        actions = np.array(self.episode_actions)
        
        # Forward pass
        logits = self.policy_network.predict(states)
        probabilities = self.softmax(logits)
        
        # Calculate loss gradients
        loss_gradients = probabilities.copy()
        for i in range(len(actions)):
            loss_gradients[i, actions[i]] -= 1
            loss_gradients[i] *= discounted_rewards[i]
        
        # Backward pass
        self.policy_network.backward(states, probabilities - loss_gradients, probabilities)
        
        # Store episode reward
        total_reward = sum(self.episode_rewards)
        self.episode_total_rewards.append(total_reward)
        
        # Clear episode memory
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
    
    def _calculate_discounted_rewards(self) -> np.ndarray:
        """Calculate discounted cumulative rewards"""
        discounted_rewards = np.zeros_like(self.episode_rewards, dtype=float)
        cumulative_reward = 0
        
        for i in reversed(range(len(self.episode_rewards))):
            cumulative_reward = self.episode_rewards[i] + self.config.discount_factor * cumulative_reward
            discounted_rewards[i] = cumulative_reward
        
        return discounted_rewards

class GridWorldEnvironment:
    """Simple grid world environment for testing"""
    
    def __init__(self, size: int = 5):
        self.size = size
        self.state_size = size * size
        self.action_size = 4  # up, down, left, right
        
        # Initialize environment
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        self.steps = 0
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action in environment"""
        self.steps += 1
        
        # Move agent
        if action == 0:  # up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # down
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # right
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        
        # Calculate reward
        if self.agent_pos == self.goal_pos:
            reward = 10.0
            done = True
        else:
            reward = -0.1  # Small negative reward for each step
            done = self.steps >= 100  # Max steps
        
        return self._get_state(), reward, done, {}
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        state = np.zeros(self.state_size)
        agent_index = self.agent_pos[0] * self.size + self.agent_pos[1]
        state[agent_index] = 1.0
        return state

class RLTrainer:
    """Reinforcement learning trainer"""
    
    def __init__(self, agent, environment, config: RLConfig):
        self.agent = agent
        self.environment = environment
        self.config = config
        
        # Training metrics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'average_rewards': [],
            'losses': []
        }
    
    def train(self) -> Dict[str, Any]:
        """Train the agent"""
        logger.info(f"Starting RL training with {self.config.algorithm}")
        
        for episode in range(self.config.max_episodes):
            state = self.environment.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(self.config.max_steps_per_episode):
                # Choose action
                action = self.agent.act(state, training=True)
                
                # Take step
                next_state, reward, done, _ = self.environment.step(action)
                episode_reward += reward
                episode_length += 1
                
                # Store experience
                if hasattr(self.agent, 'remember'):
                    self.agent.remember(state, action, reward, next_state, done)
                    
                    # Train DQN agent
                    if hasattr(self.agent, 'train'):
                        self.agent.train()
                elif hasattr(self.agent, 'remember'):
                    # Store for policy gradient
                    self.agent.remember(state, action, reward)
                
                state = next_state
                
                if done:
                    break
            
            # Train policy gradient agent at end of episode
            if hasattr(self.agent, 'train_episode'):
                self.agent.train_episode()
            
            # Record metrics
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            
            # Calculate moving average
            if len(self.training_history['episode_rewards']) >= 100:
                avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                self.training_history['average_rewards'].append(avg_reward)
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-100:]) if episode >= 100 else episode_reward
                logger.info(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {getattr(self.agent, 'epsilon', 'N/A')}")
        
        logger.info("RL training completed")
        return self.training_history
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate trained agent"""
        rewards = []
        
        for episode in range(num_episodes):
            state = self.environment.reset()
            episode_reward = 0
            
            for step in range(self.config.max_steps_per_episode):
                action = self.agent.act(state, training=False)
                state, reward, done, _ = self.environment.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }

class RLExperimentTracker:
    """Track and manage RL experiments"""
    
    def __init__(self, experiment_dir: str = "experiments/rl"):
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)
        
        self.experiments = []
    
    def create_experiment(self, name: str, config: RLConfig, 
                         description: str = "") -> str:
        """Create new experiment"""
        experiment_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_path = os.path.join(self.experiment_dir, experiment_id)
        os.makedirs(experiment_path, exist_ok=True)
        
        # Save config
        config_path = os.path.join(experiment_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        
        # Save metadata
        metadata = {
            'id': experiment_id,
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'status': 'created'
        }
        
        metadata_path = os.path.join(experiment_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.experiments.append(metadata)
        logger.info(f"Created RL experiment: {experiment_id}")
        
        return experiment_id
    
    def save_results(self, experiment_id: str, results: Dict[str, Any]):
        """Save experiment results"""
        experiment_path = os.path.join(self.experiment_dir, experiment_id)
        results_path = os.path.join(experiment_path, "results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                serializable_results[key] = [v.tolist() for v in value]
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved results for experiment: {experiment_id}")

def create_rl_system(algorithm: str = "dqn", environment: str = "gridworld") -> Dict[str, Any]:
    """Create and configure RL system"""
    
    # Create configuration
    config = RLConfig(algorithm=algorithm)
    
    # Create environment
    if environment == "gridworld":
        env = GridWorldEnvironment(size=5)
    else:
        raise ValueError(f"Unknown environment: {environment}")
    
    # Create agent
    if algorithm == "dqn":
        agent = DQNAgent(env.state_size, env.action_size, config)
    elif algorithm == "policy_gradient":
        agent = PolicyGradientAgent(env.state_size, env.action_size, config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Create trainer
    trainer = RLTrainer(agent, env, config)
    
    # Create experiment tracker
    tracker = RLExperimentTracker()
    
    return {
        'agent': agent,
        'environment': env,
        'trainer': trainer,
        'tracker': tracker,
        'config': config
    }

# Example usage and testing
if __name__ == "__main__":
    # Test DQN
    print("Testing DQN Agent...")
    rl_system = create_rl_system("dqn", "gridworld")
    
    # Create experiment
    experiment_id = rl_system['tracker'].create_experiment(
        "dqn_gridworld_test",
        rl_system['config'],
        "Testing DQN agent on GridWorld environment"
    )
    
    # Train agent
    training_history = rl_system['trainer'].train()
    
    # Evaluate agent
    evaluation_results = rl_system['trainer'].evaluate()
    
    # Save results
    results = {
        'training_history': training_history,
        'evaluation_results': evaluation_results
    }
    rl_system['tracker'].save_results(experiment_id, results)
    
    print(f"DQN Training completed. Final average reward: {evaluation_results['mean_reward']:.2f}")
    
    # Test Policy Gradient
    print("\nTesting Policy Gradient Agent...")
    rl_system_pg = create_rl_system("policy_gradient", "gridworld")
    
    # Create experiment
    experiment_id_pg = rl_system_pg['tracker'].create_experiment(
        "pg_gridworld_test",
        rl_system_pg['config'],
        "Testing Policy Gradient agent on GridWorld environment"
    )
    
    # Train agent
    training_history_pg = rl_system_pg['trainer'].train()
    
    # Evaluate agent
    evaluation_results_pg = rl_system_pg['trainer'].evaluate()
    
    # Save results
    results_pg = {
        'training_history': training_history_pg,
        'evaluation_results': evaluation_results_pg
    }
    rl_system_pg['tracker'].save_results(experiment_id_pg, results_pg)
    
    print(f"Policy Gradient Training completed. Final average reward: {evaluation_results_pg['mean_reward']:.2f}")
    
    print("\n✅ Reinforcement Learning module tests completed successfully!")
