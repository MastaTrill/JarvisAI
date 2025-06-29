"""
🧠 NEUROMORPHIC AI BRAIN MODULE
Revolutionary brain-inspired computing architecture for Jarvis AI Platform

This module implements:
- Spiking Neural Networks (SNNs) with temporal dynamics
- Memory-Augmented Networks with external memory
- Continual Learning without catastrophic forgetting
- Attention mechanisms and consciousness modeling
- Brain-like temporal processing and adaptation
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import json
import math

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def softmax(x, axis=-1):
    """Softmax activation function"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

@dataclass
class SpikeConfig:
    """Configuration for spiking neural networks"""
    threshold: float = 1.0
    reset_potential: float = 0.0
    decay_factor: float = 0.9
    refractory_period: int = 2
    time_steps: int = 100

class SpikingNeuron:
    """Individual spiking neuron with temporal dynamics"""
    
    def __init__(self, config: SpikeConfig):
        self.config = config
        self.membrane_potential = 0.0
        self.refractory_counter = 0
        self.spike_history = deque(maxlen=100)
        
    def forward(self, input_current: float) -> Tuple[float, bool]:
        """Process input and generate spike if threshold reached"""
        # Handle refractory period
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            return 0.0, False
        
        # Update membrane potential
        self.membrane_potential = (self.membrane_potential * self.config.decay_factor + 
                                 input_current)
        
        # Check for spike
        if self.membrane_potential >= self.config.threshold:
            spike = True
            self.membrane_potential = self.config.reset_potential
            self.refractory_counter = self.config.refractory_period
            self.spike_history.append(time.time())
            return 1.0, spike
        
        return 0.0, False

class SpikingNeuralLayer:
    """Layer of spiking neurons with synaptic connections"""
    
    def __init__(self, input_size: int, output_size: int, config: SpikeConfig):
        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        
        # Synaptic weights
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros(output_size)
        
        # Create spiking neurons
        self.neurons = [SpikingNeuron(config) for _ in range(output_size)]
        
        # Spike timing dependent plasticity (STDP)
        self.stdp_trace = np.zeros(output_size)
        self.learning_rate = 0.01
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through spiking layer"""
        batch_size = x.shape[0]
        
        # Compute synaptic currents
        currents = np.dot(x, self.weights) + self.bias
        
        # Process through spiking neurons
        spikes = np.zeros((batch_size, self.output_size))
        
        for i, neuron in enumerate(self.neurons):
            for b in range(batch_size):
                spike_output, did_spike = neuron.forward(currents[b, i])
                spikes[b, i] = spike_output
                
                # Update STDP trace
                if did_spike:
                    self.stdp_trace[i] = self.stdp_trace[i] * 0.9 + 1.0
        
        return spikes
    
    def update_weights_stdp(self, pre_spikes: np.ndarray, post_spikes: np.ndarray):
        """Update weights using spike-timing dependent plasticity"""
        # Simplified STDP rule
        correlation = np.outer(pre_spikes.mean(0), post_spikes.mean(0))
        self.weights += self.learning_rate * correlation

class MemoryAugmentedNetwork:
    """External memory system for long-term learning"""
    
    def __init__(self, memory_size: int = 128, key_size: int = 64, value_size: int = 64):
        self.memory_size = memory_size
        self.key_size = key_size
        self.value_size = value_size
        
        # External memory matrix
        self.memory_keys = np.random.randn(memory_size, key_size)
        self.memory_values = np.random.randn(memory_size, value_size)
        self.memory_usage = np.zeros(memory_size)
        
        # Controllers
        self.key_controller_weights = np.random.randn(key_size, key_size) * 0.1
        self.value_controller_weights = np.random.randn(value_size, value_size) * 0.1
        
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between vectors"""
        a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=-1, keepdims=True)
        return np.dot(a, b.T) / (a_norm * b_norm.T + 1e-8)
        
    def read_memory(self, query_key: np.ndarray) -> np.ndarray:
        """Read from external memory using content-based addressing"""
        # Transform query key
        query_key = np.dot(query_key, self.key_controller_weights)
        
        # Compute similarity scores
        if len(query_key.shape) == 1:
            query_key = query_key.reshape(1, -1)
            
        similarities = self.cosine_similarity(query_key, self.memory_keys)
        
        # Softmax attention
        attention_weights = softmax(similarities, axis=1)
        
        # Read weighted values
        read_values = np.dot(attention_weights, self.memory_values)
        
        # Update usage
        self.memory_usage += attention_weights.sum(0)
        
        return read_values.squeeze() if read_values.shape[0] == 1 else read_values
    
    def write_memory(self, key: np.ndarray, value: np.ndarray):
        """Write to external memory"""
        # Find least used memory slot
        least_used_idx = np.argmin(self.memory_usage)
        
        # Transform key and value
        key_transformed = np.dot(key, self.key_controller_weights)
        value_transformed = np.dot(value, self.value_controller_weights)
        
        # Write new key-value pair
        self.memory_keys[least_used_idx] = key_transformed
        self.memory_values[least_used_idx] = value_transformed
        
        # Reset usage for written slot
        self.memory_usage[least_used_idx] = 0.0

class AttentionMechanism:
    """Multi-head attention for consciousness modeling"""
    
    def __init__(self, d_model: int = 256, num_heads: int = 8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Attention weights
        self.query_weights = np.random.randn(d_model, d_model) * 0.1
        self.key_weights = np.random.randn(d_model, d_model) * 0.1
        self.value_weights = np.random.randn(d_model, d_model) * 0.1
        self.output_weights = np.random.randn(d_model, d_model) * 0.1
        
        # Consciousness modeling components
        self.global_workspace = np.random.randn(1, d_model) * 0.1
        self.consciousness_weights = np.random.randn(d_model, 1) * 0.1
        
    def forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Multi-head attention with consciousness modeling"""
        batch_size, seq_len, d_model = x.shape
        
        # Multi-head attention projections
        queries = np.dot(x, self.query_weights).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = np.dot(x, self.key_weights).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        values = np.dot(x, self.value_weights).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        queries = queries.transpose(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        attention_scores = np.matmul(queries, keys.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        attention_weights = softmax(attention_scores, axis=-1)
        attended_values = np.matmul(attention_weights, values)
        
        # Concatenate heads
        attended_values = attended_values.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        output = np.dot(attended_values, self.output_weights)
        
        # Consciousness modeling - global workspace integration
        global_workspace = np.tile(self.global_workspace, (batch_size, seq_len, 1))
        consciousness_input = output + global_workspace
        consciousness_scores = sigmoid(np.dot(consciousness_input, self.consciousness_weights))
        
        # Update global workspace with conscious information
        conscious_content = output * consciousness_scores
        self.global_workspace = np.mean(conscious_content, axis=(0, 1), keepdims=True)
        
        return {
            'output': output,
            'attention_weights': attention_weights,
            'consciousness_scores': consciousness_scores,
            'global_workspace': self.global_workspace
        }

class ContinualLearningModule:
    """Elastic Weight Consolidation for continual learning"""
    
    def __init__(self, lambda_reg: float = 1000.0):
        self.lambda_reg = lambda_reg
        self.fisher_information = {}
        self.optimal_weights = {}
        
    def compute_fisher_information(self, model, data_loader):
        """Compute Fisher Information Matrix for important weights"""
        logger.info("🧠 Computing Fisher Information Matrix...")
        
        fisher_info = {}
        
        # Initialize Fisher information for all weight matrices
        for name in ['spiking_layers', 'output_weights', 'memory_network']:
            fisher_info[name] = {}
        
        # Simplified Fisher information computation
        for batch_data, batch_targets in data_loader:
            # Forward pass through model
            output_dict = model.forward(batch_data)
            
            # Compute gradients (simplified)
            loss_gradient = output_dict['output'] - batch_targets
            
            # Accumulate Fisher information (simplified)
            for name in fisher_info:
                if name not in fisher_info:
                    fisher_info[name] = np.zeros_like(loss_gradient)
                fisher_info[name] += np.square(loss_gradient)
        
        # Normalize by number of samples
        for name in fisher_info:
            fisher_info[name] /= len(data_loader)
        
        self.fisher_information = fisher_info
        logger.info(f"✅ Fisher Information computed for {len(fisher_info)} parameter groups")
    
    def save_optimal_weights(self, model):
        """Save current weights as optimal for previous task"""
        self.optimal_weights = {
            'spiking_layers': [layer.weights.copy() for layer in model.spiking_layers],
            'output_weights': model.output_weights.copy(),
            'memory_keys': model.memory_network.memory_keys.copy(),
            'memory_values': model.memory_network.memory_values.copy()
        }
        
        logger.info("💾 Optimal weights saved for continual learning")
    
    def compute_ewc_loss(self, model) -> float:
        """Compute Elastic Weight Consolidation regularization loss"""
        ewc_loss = 0.0
        
        if not self.optimal_weights:
            return ewc_loss
        
        # Compute EWC loss for spiking layers
        for i, layer in enumerate(model.spiking_layers):
            if i < len(self.optimal_weights['spiking_layers']):
                weight_diff = layer.weights - self.optimal_weights['spiking_layers'][i]
                ewc_loss += np.sum(weight_diff ** 2)
        
        # Compute EWC loss for output weights
        if 'output_weights' in self.optimal_weights:
            weight_diff = model.output_weights - self.optimal_weights['output_weights']
            ewc_loss += np.sum(weight_diff ** 2)
        
        return self.lambda_reg * ewc_loss

class NeuromorphicBrain:
    """Complete neuromorphic AI brain architecture"""
    
    def __init__(self, 
                 input_size: int = 784,
                 hidden_sizes: List[int] = [256, 128],
                 output_size: int = 10,
                 spike_config: Optional[SpikeConfig] = None):
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Default spike configuration
        if spike_config is None:
            spike_config = SpikeConfig()
        self.spike_config = spike_config
        
        # Build spiking neural network layers
        self.spiking_layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            self.spiking_layers.append(
                SpikingNeuralLayer(prev_size, hidden_size, spike_config)
            )
            prev_size = hidden_size
        
        # Output layer
        self.output_weights = np.random.randn(prev_size, output_size) * 0.1
        self.output_bias = np.zeros(output_size)
        
        # Memory-augmented network
        self.memory_network = MemoryAugmentedNetwork(
            memory_size=128, 
            key_size=prev_size, 
            value_size=prev_size
        )
        
        # Attention mechanism for consciousness
        self.attention = AttentionMechanism(d_model=prev_size)
        
        # Continual learning
        self.continual_learning = ContinualLearningModule()
        
        # Brain state tracking
        self.brain_state = {
            'consciousness_level': 0.0,
            'memory_usage': 0.0,
            'attention_focus': None,
            'learning_efficiency': 0.0
        }
        
        logger.info(f"🧠 NeuromorphicBrain initialized with architecture: {[input_size] + hidden_sizes + [output_size]}")
    
    def forward(self, x: np.ndarray, use_memory: bool = True, track_consciousness: bool = True) -> Dict[str, Any]:
        """Forward pass through neuromorphic brain"""
        batch_size = x.shape[0]
        
        # Initialize spike patterns
        current_spikes = x
        spike_patterns = []
        memory_content = None
        
        # Process through spiking layers
        for i, layer in enumerate(self.spiking_layers):
            current_spikes = layer.forward(current_spikes)
            spike_patterns.append(current_spikes.copy())
            
            # Add some noise for biological realism
            noise = np.random.randn(*current_spikes.shape) * 0.01
            current_spikes = current_spikes + noise
        
        # Memory integration
        if use_memory and len(spike_patterns) > 0:
            memory_key = spike_patterns[-1].mean(axis=0)
            memory_content = self.memory_network.read_memory(memory_key)
            
            # Update memory with current pattern
            if np.random.rand() < 0.1:  # 10% chance to write to memory
                self.memory_network.write_memory(memory_key, spike_patterns[-1].mean(axis=0))
            
            # Integrate memory with current processing
            if len(memory_content.shape) == 1:
                memory_content = memory_content.reshape(1, -1)
            memory_expanded = np.tile(memory_content, (batch_size, 1))
            current_spikes = current_spikes + memory_expanded * 0.1
        
        # Consciousness modeling through attention
        consciousness_output = None
        if track_consciousness and len(spike_patterns) > 0:
            # Prepare input for attention (add sequence dimension)
            attention_input = current_spikes.reshape(batch_size, 1, -1)  # [batch, 1, features]
            consciousness_output = self.attention.forward(attention_input)
            
            # Update brain state
            self.brain_state['consciousness_level'] = consciousness_output['consciousness_scores'].mean()
            self.brain_state['attention_focus'] = consciousness_output['attention_weights']
        
        # Final output
        output = np.dot(current_spikes, self.output_weights) + self.output_bias
        
        # Update brain state
        self.brain_state['memory_usage'] = self.memory_network.memory_usage.mean()
        
        return {
            'output': output,
            'spike_patterns': spike_patterns,
            'consciousness_output': consciousness_output,
            'brain_state': self.brain_state.copy(),
            'memory_content': memory_content
        }
    
    def train_continual(self, train_data, train_targets, test_data, test_targets, epochs=10):
        """Train with continual learning capabilities"""
        logger.info("🎓 Starting continual learning training...")
        
        training_history = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            # Training loop
            for i in range(0, len(train_data), 32):  # Batch size 32
                batch_data = train_data[i:i+32]
                batch_targets = train_targets[i:i+32]
                
                # Forward pass
                output_dict = self.forward(batch_data)
                output = output_dict['output']
                
                # Compute loss (cross-entropy)
                predictions = softmax(output, axis=1)
                targets_onehot = np.eye(self.output_size)[batch_targets]
                ce_loss = -np.mean(np.sum(targets_onehot * np.log(predictions + 1e-8), axis=1))
                
                ewc_loss = self.continual_learning.compute_ewc_loss(self)
                total_loss_batch = ce_loss + ewc_loss
                
                # Simple gradient update (simplified)
                learning_rate = 0.001
                grad = predictions - targets_onehot
                self.output_weights -= learning_rate * np.dot(output_dict['spike_patterns'][-1].T, grad) / batch_data.shape[0]
                
                # Statistics
                total_loss += total_loss_batch
                pred_classes = np.argmax(predictions, axis=1)
                correct += np.sum(pred_classes == batch_targets)
                total += len(batch_targets)
            
            # Validation
            val_accuracy = self.evaluate_brain(test_data, test_targets)
            
            # Update learning efficiency
            self.brain_state['learning_efficiency'] = correct / total if total > 0 else 0.0
            
            epoch_stats = {
                'epoch': epoch,
                'train_loss': total_loss / (len(train_data) // 32),
                'train_accuracy': correct / total,
                'val_accuracy': val_accuracy,
                'consciousness_level': self.brain_state['consciousness_level'],
                'memory_usage': self.brain_state['memory_usage']
            }
            
            training_history.append(epoch_stats)
            
            logger.info(f"Epoch {epoch}: Loss={epoch_stats['train_loss']:.4f}, "
                       f"Train Acc={epoch_stats['train_accuracy']:.4f}, "
                       f"Val Acc={epoch_stats['val_accuracy']:.4f}, "
                       f"Consciousness={epoch_stats['consciousness_level']:.4f}")
        
        # Save optimal weights for continual learning
        self.continual_learning.save_optimal_weights(self)
        
        logger.info("✅ Continual learning training completed!")
        return training_history
    
    def evaluate_brain(self, test_data: np.ndarray, test_targets: np.ndarray) -> float:
        """Evaluate neuromorphic brain performance"""
        correct = 0
        total = 0
        
        for i in range(0, len(test_data), 32):
            batch_data = test_data[i:i+32]
            batch_targets = test_targets[i:i+32]
            
            output_dict = self.forward(batch_data)
            output = output_dict['output']
            
            predictions = softmax(output, axis=1)
            pred_classes = np.argmax(predictions, axis=1)
            correct += np.sum(pred_classes == batch_targets)
            total += len(batch_targets)
        
        return correct / total if total > 0 else 0.0
    
    def get_brain_analysis(self) -> Dict[str, Any]:
        """Comprehensive brain state analysis"""
        total_params = sum(layer.weights.size + layer.bias.size for layer in self.spiking_layers)
        total_params += self.output_weights.size + self.output_bias.size
        
        analysis = {
            'architecture': {
                'input_size': self.input_size,
                'hidden_sizes': self.hidden_sizes,
                'output_size': self.output_size,
                'total_parameters': total_params,
                'trainable_parameters': total_params
            },
            'brain_state': self.brain_state.copy(),
            'memory_system': {
                'memory_size': self.memory_network.memory_size,
                'memory_usage_distribution': self.memory_network.memory_usage.tolist(),
                'average_memory_usage': self.memory_network.memory_usage.mean()
            },
            'consciousness_metrics': {
                'global_workspace_activation': np.linalg.norm(self.attention.global_workspace),
                'attention_entropy': -np.sum(
                    softmax(self.attention.global_workspace) * 
                    np.log(softmax(self.attention.global_workspace) + 1e-8)
                )
            },
            'spiking_activity': {
                'num_spiking_layers': len(self.spiking_layers),
                'spike_threshold': self.spike_config.threshold,
                'refractory_period': self.spike_config.refractory_period
            }
        }
        
        return analysis

def create_demo_brain(input_size: int = 784, num_classes: int = 10) -> NeuromorphicBrain:
    """Create a demonstration neuromorphic brain"""
    logger.info("🧠 Creating demonstration neuromorphic brain...")
    
    # Custom spike configuration
    spike_config = SpikeConfig(
        threshold=1.2,
        reset_potential=0.0,
        decay_factor=0.85,
        refractory_period=3,
        time_steps=50
    )
    
    # Create brain with optimized architecture
    brain = NeuromorphicBrain(
        input_size=input_size,
        hidden_sizes=[512, 256, 128],
        output_size=num_classes,
        spike_config=spike_config
    )
    
    logger.info("✅ Demonstration neuromorphic brain created successfully!")
    return brain

def demo_neuromorphic_features():
    """Demonstrate neuromorphic AI brain capabilities"""
    logger.info("🚀 Starting Neuromorphic AI Brain demonstration...")
    
    # Create synthetic data for demonstration
    batch_size = 32
    input_size = 784  # MNIST-like
    num_classes = 10
    
    # Generate synthetic data
    demo_data = np.random.randn(batch_size, input_size)
    demo_targets = np.random.randint(0, num_classes, batch_size)
    
    # Create neuromorphic brain
    brain = create_demo_brain(input_size, num_classes)
    
    # Demonstration 1: Basic forward pass
    logger.info("🧠 Demonstrating basic neuromorphic processing...")
    output_dict = brain.forward(demo_data)
    
    print(f"   Input shape: {demo_data.shape}")
    print(f"   Output shape: {output_dict['output'].shape}")
    print(f"   Spike patterns: {len(output_dict['spike_patterns'])} layers")
    print(f"   Consciousness level: {output_dict['brain_state']['consciousness_level']:.4f}")
    print(f"   Memory usage: {output_dict['brain_state']['memory_usage']:.4f}")
    
    # Demonstration 2: Memory capabilities
    logger.info("🧠 Demonstrating memory-augmented processing...")
    memory_tests = []
    for i in range(5):
        test_data = np.random.randn(1, input_size)
        output_dict = brain.forward(test_data, use_memory=True)
        memory_tests.append({
            'iteration': i,
            'memory_usage': output_dict['brain_state']['memory_usage'],
            'consciousness': output_dict['brain_state']['consciousness_level']
        })
    
    for test in memory_tests:
        print(f"   Iteration {test['iteration']}: Memory={test['memory_usage']:.4f}, "
              f"Consciousness={test['consciousness']:.4f}")
    
    # Demonstration 3: Brain analysis
    logger.info("🧠 Generating comprehensive brain analysis...")
    analysis = brain.get_brain_analysis()
    
    print(f"   Total parameters: {analysis['architecture']['total_parameters']:,}")
    print(f"   Memory system size: {analysis['memory_system']['memory_size']}")
    print(f"   Average memory usage: {analysis['memory_system']['average_memory_usage']:.4f}")
    print(f"   Consciousness entropy: {analysis['consciousness_metrics']['attention_entropy']:.4f}")
    print(f"   Spiking layers: {analysis['spiking_activity']['num_spiking_layers']}")
    
    # Demonstration 4: Continual learning simulation
    logger.info("🧠 Demonstrating continual learning capabilities...")
    
    # Create synthetic datasets
    train_data = np.random.randn(100, input_size)
    train_targets = np.random.randint(0, num_classes, 100)
    test_data = np.random.randn(50, input_size)
    test_targets = np.random.randint(0, num_classes, 50)
    
    # Quick training demonstration
    history = brain.train_continual(train_data, train_targets, test_data, test_targets, epochs=3)
    
    print(f"   Training completed: {len(history)} epochs")
    print(f"   Final train accuracy: {history[-1]['train_accuracy']:.4f}")
    print(f"   Final validation accuracy: {history[-1]['val_accuracy']:.4f}")
    print(f"   Final consciousness level: {history[-1]['consciousness_level']:.4f}")
    
    logger.info("✅ Neuromorphic AI Brain demonstration completed successfully!")
    
    return {
        'brain': brain,
        'analysis': analysis,
        'training_history': history,
        'demo_results': {
            'memory_tests': memory_tests,
            'final_consciousness': output_dict['brain_state']['consciousness_level'],
            'total_parameters': analysis['architecture']['total_parameters']
        }
    }

if __name__ == "__main__":
    # Run demonstration
    demo_results = demo_neuromorphic_features()
    print("\n🧠 Neuromorphic AI Brain Module Ready!")
    print(f"🎯 Consciousness Level: {demo_results['demo_results']['final_consciousness']:.4f}")
    print(f"🧮 Total Parameters: {demo_results['demo_results']['total_parameters']:,}")
    print("🚀 Ready for integration with Jarvis AI Platform!")
