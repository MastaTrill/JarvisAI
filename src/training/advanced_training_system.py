#!/usr/bin/env python3
"""
Advanced Training System for Jarvis AI Platform.
Includes hyperparameter tuning, experiment tracking, and advanced optimization.
"""

import numpy as np
import pandas as pd
import json
import yaml
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
import logging
import time
import hashlib
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# Import our advanced modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.advanced_neural_network import AdvancedNeuralNetwork
from data.advanced_data_pipeline import AdvancedDataPipeline

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a training experiment."""
    experiment_name: str
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    data_config: Dict[str, Any]
    optimization_config: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class ExperimentResult:
    """Results from a training experiment."""
    experiment_id: str
    config: ExperimentConfig
    metrics: Dict[str, Any]
    model_path: str
    duration: float
    timestamp: str
    status: str
    logs: List[str]


class HyperparameterOptimizer(ABC):
    """Abstract base class for hyperparameter optimization."""
    
    @abstractmethod
    def suggest_parameters(self, trial_number: int) -> Dict[str, Any]:
        """Suggest parameters for the given trial."""
        pass
    
    @abstractmethod
    def update_results(self, parameters: Dict[str, Any], score: float):
        """Update optimizer with trial results."""
        pass


class GridSearchOptimizer(HyperparameterOptimizer):
    """Grid search hyperparameter optimization."""
    
    def __init__(self, parameter_grid: Dict[str, List[Any]]):
        self.parameter_grid = parameter_grid
        self.parameter_combinations = self._generate_combinations()
        self.current_trial = 0
        self.results = []
    
    def _generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        import itertools
        
        keys = list(self.parameter_grid.keys())
        values = list(self.parameter_grid.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def suggest_parameters(self, trial_number: int) -> Dict[str, Any]:
        """Suggest parameters for the given trial."""
        if trial_number < len(self.parameter_combinations):
            return self.parameter_combinations[trial_number]
        return {}
    
    def update_results(self, parameters: Dict[str, Any], score: float):
        """Update optimizer with trial results."""
        self.results.append({
            'parameters': parameters,
            'score': score
        })
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """Get the best parameters found so far."""
        if not self.results:
            return {}
        
        best_result = max(self.results, key=lambda x: x['score'])
        return best_result['parameters']


class RandomSearchOptimizer(HyperparameterOptimizer):
    """Random search hyperparameter optimization."""
    
    def __init__(self, parameter_space: Dict[str, Any], max_trials: int = 50):
        self.parameter_space = parameter_space
        self.max_trials = max_trials
        self.results = []
    
    def suggest_parameters(self, trial_number: int) -> Dict[str, Any]:
        """Suggest random parameters for the given trial."""
        if trial_number >= self.max_trials:
            return {}
        
        suggested = {}
        for param, space in self.parameter_space.items():
            if isinstance(space, dict):
                if space['type'] == 'uniform':
                    suggested[param] = np.random.uniform(space['low'], space['high'])
                elif space['type'] == 'int':
                    suggested[param] = np.random.randint(space['low'], space['high'])
                elif space['type'] == 'choice':
                    suggested[param] = np.random.choice(space['choices'])
                elif space['type'] == 'log_uniform':
                    suggested[param] = np.exp(np.random.uniform(
                        np.log(space['low']), np.log(space['high'])
                    ))
            elif isinstance(space, list):
                suggested[param] = np.random.choice(space)
        
        return suggested
    
    def update_results(self, parameters: Dict[str, Any], score: float):
        """Update optimizer with trial results."""
        self.results.append({
            'parameters': parameters,
            'score': score
        })
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """Get the best parameters found so far."""
        if not self.results:
            return {}
        
        best_result = max(self.results, key=lambda x: x['score'])
        return best_result['parameters']


class ExperimentTracker:
    """Experiment tracking and management system."""
    
    def __init__(self, storage_path: str = "experiments"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database for experiments
        self.db_path = self.storage_path / "experiments.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize the experiments database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    experiment_name TEXT,
                    config TEXT,
                    metrics TEXT,
                    model_path TEXT,
                    duration REAL,
                    timestamp TEXT,
                    status TEXT,
                    logs TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hyperparameter_trials (
                    trial_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    parameters TEXT,
                    score REAL,
                    trial_number INTEGER,
                    timestamp TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            """)
    
    def start_experiment(self, config: ExperimentConfig) -> str:
        """Start a new experiment and return its ID."""
        experiment_id = self._generate_experiment_id(config.experiment_name)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO experiments 
                (experiment_id, experiment_name, config, timestamp, status)
                VALUES (?, ?, ?, ?, ?)
            """, (
                experiment_id,
                config.experiment_name,
                json.dumps(asdict(config)),
                datetime.now().isoformat(),
                'running'
            ))
        
        logger.info(f"🚀 Started experiment: {experiment_id}")
        return experiment_id
    
    def finish_experiment(self, experiment_id: str, result: ExperimentResult):
        """Finish an experiment and save results."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE experiments 
                SET metrics = ?, model_path = ?, duration = ?, status = ?, logs = ?
                WHERE experiment_id = ?
            """, (
                json.dumps(result.metrics),
                result.model_path,
                result.duration,
                result.status,
                json.dumps(result.logs),
                experiment_id
            ))
        
        logger.info(f"✅ Finished experiment: {experiment_id}")
    
    def log_hyperparameter_trial(self, experiment_id: str, trial_number: int, 
                                parameters: Dict[str, Any], score: float):
        """Log a hyperparameter optimization trial."""
        trial_id = f"{experiment_id}_trial_{trial_number}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO hyperparameter_trials 
                (trial_id, experiment_id, parameters, score, trial_number, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                trial_id,
                experiment_id,
                json.dumps(parameters),
                score,
                trial_number,
                datetime.now().isoformat()
            ))
    
    def get_experiment_results(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get results for a specific experiment."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM experiments WHERE experiment_id = ?
            """, (experiment_id,))
            
            row = cursor.fetchone()
            if row:
                return ExperimentResult(
                    experiment_id=row[0],
                    config=ExperimentConfig(**json.loads(row[2])),
                    metrics=json.loads(row[3]) if row[3] else {},
                    model_path=row[4] or "",
                    duration=row[5] or 0.0,
                    timestamp=row[6],
                    status=row[7],
                    logs=json.loads(row[8]) if row[8] else []
                )
        return None
    
    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """Get summary of all experiments."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT experiment_id, experiment_name, timestamp, status, 
                       metrics, duration FROM experiments
                ORDER BY timestamp DESC
            """)
            
            experiments = []
            for row in cursor.fetchall():
                experiments.append({
                    'experiment_id': row[0],
                    'experiment_name': row[1],
                    'timestamp': row[2],
                    'status': row[3],
                    'metrics': json.loads(row[4]) if row[4] else {},
                    'duration': row[5]
                })
            
            return experiments
    
    def get_best_experiment(self, metric: str = 'val_accuracy') -> Optional[Dict[str, Any]]:
        """Get the best experiment based on a specific metric."""
        experiments = self.get_all_experiments()
        
        valid_experiments = [
            exp for exp in experiments 
            if exp['status'] == 'completed' and metric in exp['metrics']
        ]
        
        if not valid_experiments:
            return None
        
        best_experiment = max(valid_experiments, key=lambda x: x['metrics'][metric])
        return best_experiment
    
    def _generate_experiment_id(self, experiment_name: str) -> str:
        """Generate a unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_suffix = hashlib.md5(f"{experiment_name}_{timestamp}".encode()).hexdigest()[:8]
        return f"{experiment_name}_{timestamp}_{hash_suffix}"


class AdvancedOptimizer:
    """Advanced optimization algorithms for neural networks."""
    
    @staticmethod
    def adam(weights: np.ndarray, gradients: np.ndarray, 
             m: np.ndarray, v: np.ndarray, t: int,
             learning_rate: float = 0.001, beta1: float = 0.9, 
             beta2: float = 0.999, epsilon: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Adam optimizer implementation."""
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * gradients
        
        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (gradients ** 2)
        
        # Compute bias-corrected first moment estimate
        m_corrected = m / (1 - beta1 ** t)
        
        # Compute bias-corrected second raw moment estimate
        v_corrected = v / (1 - beta2 ** t)
        
        # Update weights
        weights_updated = weights - learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)
        
        return weights_updated, m, v
    
    @staticmethod
    def adamw(weights: np.ndarray, gradients: np.ndarray,
              m: np.ndarray, v: np.ndarray, t: int,
              learning_rate: float = 0.001, beta1: float = 0.9,
              beta2: float = 0.999, epsilon: float = 1e-8,
              weight_decay: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """AdamW optimizer implementation with weight decay."""
        # Apply weight decay
        weights = weights * (1 - learning_rate * weight_decay)
        
        # Apply Adam
        weights_updated, m, v = AdvancedOptimizer.adam(
            weights, gradients, m, v, t, learning_rate, beta1, beta2, epsilon
        )
        
        return weights_updated, m, v
    
    @staticmethod
    def rmsprop(weights: np.ndarray, gradients: np.ndarray,
                v: np.ndarray, learning_rate: float = 0.001,
                decay: float = 0.9, epsilon: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        """RMSprop optimizer implementation."""
        # Update moving average of squared gradients
        v = decay * v + (1 - decay) * (gradients ** 2)
        
        # Update weights
        weights_updated = weights - learning_rate * gradients / (np.sqrt(v) + epsilon)
        
        return weights_updated, v


class LearningRateScheduler:
    """Learning rate scheduling strategies."""
    
    @staticmethod
    def exponential_decay(initial_lr: float, decay_rate: float, step: int) -> float:
        """Exponential decay schedule."""
        return initial_lr * (decay_rate ** step)
    
    @staticmethod
    def cosine_annealing(initial_lr: float, min_lr: float, step: int, max_steps: int) -> float:
        """Cosine annealing schedule."""
        if step >= max_steps:
            return min_lr
        
        return min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * step / max_steps)) / 2
    
    @staticmethod
    def step_decay(initial_lr: float, drop_rate: float, epochs_drop: int, epoch: int) -> float:
        """Step decay schedule."""
        return initial_lr * (drop_rate ** np.floor(epoch / epochs_drop))
    
    @staticmethod
    def warmup_cosine(initial_lr: float, warmup_steps: int, total_steps: int, step: int) -> float:
        """Warmup followed by cosine annealing."""
        if step < warmup_steps:
            # Linear warmup
            return initial_lr * (step / warmup_steps)
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return initial_lr * 0.5 * (1 + np.cos(np.pi * progress))


class AdvancedTrainingSystem:
    """
    Comprehensive training system with hyperparameter optimization,
    experiment tracking, and advanced optimization techniques.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        self.data_pipeline = AdvancedDataPipeline()
        self.experiment_tracker = ExperimentTracker()
        self.current_experiment_id = None
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load training system configuration."""
        try:
            config_file = Path(config_path)
            if config_file.suffix.lower() == '.yaml':
                with open(config_file, 'r') as f:
                    self.config = yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                with open(config_file, 'r') as f:
                    self.config = json.load(f)
            
            logger.info(f"✅ Loaded training config from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    
    def prepare_data(self, data_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training and validation data."""
        # Use data pipeline to load and process data
        pipeline_name = data_config.get('pipeline', 'default')
        
        # Configure the data pipeline
        self.data_pipeline.config = {'pipelines': {pipeline_name: data_config}}
        
        # Run the pipeline
        df = self.data_pipeline.run_pipeline(pipeline_name)
        
        if df is None or df.empty:
            raise ValueError("Failed to load data through pipeline")
        
        # Split features and target
        target_column = data_config.get('target_column', 'target')
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values
        
        # Train-validation split
        split_ratio = data_config.get('train_split', 0.8)
        split_index = int(len(X) * split_ratio)
        
        # Shuffle data
        if data_config.get('shuffle', True):
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
        
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]
        
        logger.info(f"📊 Data prepared: Train {X_train.shape}, Val {X_val.shape}")
        return X_train, X_val, y_train, y_val
    
    def create_model(self, model_config: Dict[str, Any]) -> AdvancedNeuralNetwork:
        """Create model from configuration."""
        return AdvancedNeuralNetwork(
            input_size=model_config['input_size'],
            hidden_sizes=model_config['hidden_sizes'],
            output_size=model_config['output_size'],
            activation=model_config.get('activation', 'relu'),
            output_activation=model_config.get('output_activation', 'linear'),
            dropout_rate=model_config.get('dropout_rate', 0.0),
            l1_reg=model_config.get('l1_reg', 0.0),
            l2_reg=model_config.get('l2_reg', 0.0),
            optimizer=model_config.get('optimizer', 'adam'),
            learning_rate=model_config.get('learning_rate', 0.001),
            random_seed=model_config.get('random_seed', None)
        )
    
    def train_single_experiment(self, experiment_config: ExperimentConfig) -> ExperimentResult:
        """Train a single experiment."""
        start_time = time.time()
        logs = []
        
        try:
            # Prepare data
            X_train, X_val, y_train, y_val = self.prepare_data(experiment_config.data_config)
            logs.append(f"Data prepared: {X_train.shape[0]} train, {X_val.shape[0]} val samples")
            
            # Create model
            model = self.create_model(experiment_config.model_config)
            logs.append(f"Model created with architecture: {model.hidden_sizes}")
            
            # Training configuration
            training_config = experiment_config.training_config
            epochs = training_config.get('epochs', 100)
            batch_size = training_config.get('batch_size', 32)
            
            # Use built-in training method from AdvancedNeuralNetwork
            validation_data = (X_val, y_val) if len(X_val) > 0 else None
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                verbose=True
            )
            
            # Save model
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            experiment_id = self.current_experiment_id or "unknown"
            model_path = model_dir / f"{experiment_id}_model.pkl"
            model.save(str(model_path))
            
            # Get final metrics from history
            final_train_loss = history['train_loss'][-1] if history['train_loss'] else 0.0
            final_train_mse = history['train_mse'][-1] if history['train_mse'] else 0.0
            
            final_val_loss = history.get('val_loss', [0.0])[-1] if history.get('val_loss') else 0.0
            final_val_mse = history.get('val_mse', [0.0])[-1] if history.get('val_mse') else 0.0
            
            # Calculate accuracy if applicable
            train_predictions = model.predict(X_train)
            val_predictions = model.predict(X_val) if len(X_val) > 0 else None
            
            # For regression tasks, use R² score as accuracy metric
            train_ss_res = np.sum((y_train.flatten() - train_predictions.flatten()) ** 2)
            train_ss_tot = np.sum((y_train.flatten() - np.mean(y_train)) ** 2)
            train_r2 = 1 - (train_ss_res / train_ss_tot) if train_ss_tot != 0 else 0.0
            
            val_r2 = 0.0
            if val_predictions is not None and len(y_val) > 0:
                val_ss_res = np.sum((y_val.flatten() - val_predictions.flatten()) ** 2)
                val_ss_tot = np.sum((y_val.flatten() - np.mean(y_val)) ** 2)
                val_r2 = 1 - (val_ss_res / val_ss_tot) if val_ss_tot != 0 else 0.0
            
            # Prepare metrics
            metrics = {
                'train_loss': float(final_train_loss),
                'val_loss': float(final_val_loss),
                'train_mse': float(final_train_mse),
                'val_mse': float(final_val_mse),
                'train_r2': float(train_r2),
                'val_r2': float(val_r2),
                'train_accuracy': float(train_r2),  # Use R² as accuracy for regression
                'val_accuracy': float(val_r2),
                'total_epochs': epochs,
                'train_loss_history': [float(x) for x in history['train_loss']],
                'val_loss_history': [float(x) for x in history.get('val_loss', [])],
                'train_mse_history': [float(x) for x in history['train_mse']],
                'val_mse_history': [float(x) for x in history.get('val_mse', [])]
            }
            
            duration = time.time() - start_time
            
            result = ExperimentResult(
                experiment_id=experiment_id,
                config=experiment_config,
                metrics=metrics,
                model_path=str(model_path),
                duration=duration,
                timestamp=datetime.now().isoformat(),
                status='completed',
                logs=logs
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Training failed: {str(e)}"
            logs.append(error_msg)
            logger.error(error_msg)
            
            experiment_id = self.current_experiment_id or "unknown"
            result = ExperimentResult(
                experiment_id=experiment_id,
                config=experiment_config,
                metrics={},
                model_path="",
                duration=duration,
                timestamp=datetime.now().isoformat(),
                status='failed',
                logs=logs
            )
            
            return result
    
    def run_hyperparameter_optimization(self, base_config: ExperimentConfig, 
                                       optimizer_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        optimizer_type = optimizer_config.get('type', 'random')
        max_trials = optimizer_config.get('max_trials', 20)
        metric = optimizer_config.get('metric', 'val_accuracy')
        
        # Initialize optimizer
        if optimizer_type == 'grid':
            optimizer = GridSearchOptimizer(optimizer_config['parameter_grid'])
        elif optimizer_type == 'random':
            optimizer = RandomSearchOptimizer(optimizer_config['parameter_space'], max_trials)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        logger.info(f"🔍 Starting hyperparameter optimization with {optimizer_type} search")
        
        best_score = float('-inf')
        best_experiment_id = None
        trial_results = []
        
        for trial in range(max_trials):
            # Get suggested parameters
            suggested_params = optimizer.suggest_parameters(trial)
            if not suggested_params:
                break
            
            logger.info(f"🔬 Trial {trial + 1}/{max_trials}: {suggested_params}")
            
            # Update configuration with suggested parameters
            trial_config = ExperimentConfig(
                experiment_name=f"{base_config.experiment_name}_trial_{trial}",
                model_config={**base_config.model_config, **suggested_params.get('model', {})},
                training_config={**base_config.training_config, **suggested_params.get('training', {})},
                data_config=base_config.data_config,
                optimization_config=base_config.optimization_config,
                metadata={**base_config.metadata, 'trial_number': trial, 'hyperparameter_search': True}
            )
            
            # Start experiment
            self.current_experiment_id = self.experiment_tracker.start_experiment(trial_config)
            
            # Train model
            result = self.train_single_experiment(trial_config)
            
            # Finish experiment
            self.experiment_tracker.finish_experiment(self.current_experiment_id, result)
            
            # Get score for this trial
            if metric in result.metrics:
                score = result.metrics[metric]
                optimizer.update_results(suggested_params, score)
                
                # Log trial result
                self.experiment_tracker.log_hyperparameter_trial(
                    self.current_experiment_id, trial, suggested_params, score
                )
                
                trial_results.append({
                    'trial': trial,
                    'parameters': suggested_params,
                    'score': score,
                    'experiment_id': self.current_experiment_id
                })
                
                # Track best result
                if score > best_score:
                    best_score = score
                    best_experiment_id = self.current_experiment_id
                
                logger.info(f"✅ Trial {trial + 1} completed: {metric} = {score:.4f}")
            else:
                logger.warning(f"⚠️ Trial {trial + 1} failed or missing metric {metric}")
        
        # Get best parameters
        best_parameters = optimizer.get_best_parameters()
        
        optimization_results = {
            'best_score': best_score,
            'best_parameters': best_parameters,
            'best_experiment_id': best_experiment_id,
            'total_trials': len(trial_results),
            'trial_results': trial_results,
            'optimization_type': optimizer_type,
            'target_metric': metric
        }
        
        logger.info(f"🏆 Hyperparameter optimization completed!")
        logger.info(f"🎯 Best {metric}: {best_score:.4f}")
        logger.info(f"⚙️ Best parameters: {best_parameters}")
        
        return optimization_results
    
    def run_experiment(self, experiment_name: str, config: Optional[Dict[str, Any]] = None) -> str:
        """Run a single experiment."""
        if config is None:
            config = self.config.get('experiments', {}).get(experiment_name, {})
        
        if not config:
            raise ValueError(f"No configuration found for experiment '{experiment_name}'")
        
        # Create experiment configuration
        exp_config = ExperimentConfig(
            experiment_name=experiment_name,
            model_config=config.get('model', {}),
            training_config=config.get('training', {}),
            data_config=config.get('data', {}),
            optimization_config=config.get('optimization', {}),
            metadata=config.get('metadata', {})
        )
        
        # Start experiment
        self.current_experiment_id = self.experiment_tracker.start_experiment(exp_config)
        
        # Check if hyperparameter optimization is requested
        if config.get('hyperparameter_optimization'):
            results = self.run_hyperparameter_optimization(
                exp_config, config['hyperparameter_optimization']
            )
            
            # Create summary result
            summary_result = ExperimentResult(
                experiment_id=self.current_experiment_id,
                config=exp_config,
                metrics={'hyperparameter_optimization_results': results},
                model_path="",
                duration=0.0,
                timestamp=datetime.now().isoformat(),
                status='completed_hyperparameter_search',
                logs=[f"Hyperparameter optimization completed with {results['total_trials']} trials"]
            )
            
            self.experiment_tracker.finish_experiment(self.current_experiment_id, summary_result)
        else:
            # Run single training experiment
            result = self.train_single_experiment(exp_config)
            self.experiment_tracker.finish_experiment(self.current_experiment_id, result)
        
        return self.current_experiment_id
    
    def create_experiment(self, experiment_name: str, config: Dict[str, Any], description: str = "") -> str:
        """Create and start a new experiment."""
        exp_config = ExperimentConfig(
            experiment_name=experiment_name,
            model_config=config.get('model', {}),
            training_config=config.get('training', {}),
            data_config=config.get('data', {}),
            optimization_config=config.get('optimization', {}),
            metadata={**config.get('metadata', {}), 'description': description}
        )
        
        # Start experiment
        experiment_id = self.experiment_tracker.start_experiment(exp_config)
        self.current_experiment_id = experiment_id
        
        logger.info(f"✅ Created experiment '{experiment_name}' with ID: {experiment_id}")
        return experiment_id
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        experiments = self.experiment_tracker.get_all_experiments()
        best_experiment = self.experiment_tracker.get_best_experiment()
        
        summary = {
            'total_experiments': len(experiments),
            'completed_experiments': len([e for e in experiments if e['status'] == 'completed']),
            'failed_experiments': len([e for e in experiments if e['status'] == 'failed']),
            'best_experiment': best_experiment,
            'recent_experiments': experiments[:5]  # Last 5 experiments
        }
        
        return summary
