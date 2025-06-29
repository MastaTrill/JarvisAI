"""
Advanced Data Augmentation for Machine Learning.

This module provides sophisticated data augmentation techniques:
- Synthetic data generation
- SMOTE for imbalanced datasets
- Noise injection strategies
- Feature space augmentation
- Time series augmentation
- Text augmentation techniques
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from abc import ABC, abstractmethod
import random
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseAugmenter(ABC):
    """Base class for all data augmenters."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize base augmenter.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
    
    @abstractmethod
    def augment(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment the dataset.
        
        Args:
            X: Feature matrix
            y: Target vector
            **kwargs: Additional parameters
            
        Returns:
            Augmented features and targets
        """
        pass


class NoiseAugmenter(BaseAugmenter):
    """Add various types of noise to features."""
    
    def __init__(
        self,
        noise_type: str = "gaussian",
        noise_level: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize noise augmenter.
        
        Args:
            noise_type: Type of noise ('gaussian', 'uniform', 'salt_pepper')
            noise_level: Intensity of noise
            random_state: Random state
        """
        super().__init__(random_state)
        self.noise_type = noise_type
        self.noise_level = noise_level
    
    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augmentation_factor: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add noise to features.
        
        Args:
            X: Original features
            y: Original targets
            augmentation_factor: Factor to multiply dataset size
            
        Returns:
            Augmented dataset
        """
        try:
            logger.info(f"🔊 Applying {self.noise_type} noise augmentation")
            
            n_samples = int(len(X) * augmentation_factor)
            
            # Select random samples to augment
            indices = np.random.choice(len(X), size=n_samples, replace=True)
            X_selected = X[indices]
            y_selected = y[indices]
            
            # Apply noise
            if self.noise_type == "gaussian":
                noise = np.random.normal(0, self.noise_level, X_selected.shape)
            elif self.noise_type == "uniform":
                noise = np.random.uniform(-self.noise_level, self.noise_level, X_selected.shape)
            elif self.noise_type == "salt_pepper":
                noise = np.zeros_like(X_selected)
                salt_pepper_mask = np.random.random(X_selected.shape) < self.noise_level
                noise[salt_pepper_mask] = np.random.choice([-1, 1], size=np.sum(salt_pepper_mask))
            else:
                raise ValueError(f"Unknown noise type: {self.noise_type}")
            
            X_augmented = X_selected + noise
            
            # Combine with original data
            X_combined = np.vstack([X, X_augmented])
            y_combined = np.hstack([y, y_selected])
            
            logger.info(f"✅ Generated {len(X_augmented)} augmented samples")
            
            return X_combined, y_combined
            
        except Exception as e:
            logger.error(f"❌ Noise augmentation failed: {e}")
            raise


class SMOTEAugmenter(BaseAugmenter):
    """SMOTE (Synthetic Minority Oversampling Technique) for imbalanced datasets."""
    
    def __init__(
        self,
        k_neighbors: int = 5,
        random_state: int = 42
    ):
        """
        Initialize SMOTE augmenter.
        
        Args:
            k_neighbors: Number of nearest neighbors
            random_state: Random state
        """
        super().__init__(random_state)
        self.k_neighbors = k_neighbors
    
    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampling_strategy: str = "auto"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to balance dataset.
        
        Args:
            X: Features
            y: Targets (class labels)
            sampling_strategy: Sampling strategy ('auto', 'minority', or dict)
            
        Returns:
            Balanced dataset
        """
        try:
            logger.info("🎯 Applying SMOTE for class balancing")
            
            # Get class distribution
            unique_classes, class_counts = np.unique(y, return_counts=True)
            logger.info(f"Original class distribution: {dict(zip(unique_classes, class_counts))}")
            
            if len(unique_classes) < 2:
                logger.warning("⚠️ Only one class found, skipping SMOTE")
                return X, y
            
            # Determine target counts
            if sampling_strategy == "auto":
                target_count = max(class_counts)
                target_counts = {cls: target_count for cls in unique_classes}
            elif sampling_strategy == "minority":
                max_count = max(class_counts)
                target_counts = {}
                for cls, count in zip(unique_classes, class_counts):
                    if count < max_count:
                        target_counts[cls] = max_count
            else:
                target_counts = sampling_strategy
            
            X_synthetic_list = []
            y_synthetic_list = []
            
            for target_class in target_counts:
                current_count = np.sum(y == target_class)
                target_count = target_counts[target_class]
                
                if target_count <= current_count:
                    continue
                
                samples_needed = target_count - current_count
                logger.info(f"Generating {samples_needed} synthetic samples for class {target_class}")
                
                # Get minority class samples
                minority_samples = X[y == target_class]
                
                if len(minority_samples) < self.k_neighbors:
                    k = len(minority_samples) - 1
                    if k <= 0:
                        continue
                else:
                    k = self.k_neighbors
                
                # Generate synthetic samples
                for _ in range(samples_needed):
                    # Choose random minority sample
                    random_idx = np.random.randint(0, len(minority_samples))
                    sample = minority_samples[random_idx]
                    
                    # Find k nearest neighbors
                    distances = np.sum((minority_samples - sample) ** 2, axis=1)
                    neighbor_indices = np.argsort(distances)[1:k+1]  # Exclude self
                    
                    if len(neighbor_indices) == 0:
                        continue
                    
                    # Choose random neighbor
                    neighbor_idx = np.random.choice(neighbor_indices)
                    neighbor = minority_samples[neighbor_idx]
                    
                    # Generate synthetic sample
                    gap = np.random.random()
                    synthetic_sample = sample + gap * (neighbor - sample)
                    
                    X_synthetic_list.append(synthetic_sample)
                    y_synthetic_list.append(target_class)
            
            if X_synthetic_list:
                X_synthetic = np.array(X_synthetic_list)
                y_synthetic = np.array(y_synthetic_list)
                
                X_balanced = np.vstack([X, X_synthetic])
                y_balanced = np.hstack([y, y_synthetic])
                
                # Log new distribution
                unique_classes_new, class_counts_new = np.unique(y_balanced, return_counts=True)
                logger.info(f"New class distribution: {dict(zip(unique_classes_new, class_counts_new))}")
                
                return X_balanced, y_balanced
            else:
                logger.info("No synthetic samples generated")
                return X, y
            
        except Exception as e:
            logger.error(f"❌ SMOTE augmentation failed: {e}")
            raise


class FeatureMixupAugmenter(BaseAugmenter):
    """Feature-level mixup augmentation."""
    
    def __init__(
        self,
        alpha: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize mixup augmenter.
        
        Args:
            alpha: Beta distribution parameter
            random_state: Random state
        """
        super().__init__(random_state)
        self.alpha = alpha
    
    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augmentation_factor: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup augmentation.
        
        Args:
            X: Features
            y: Targets
            augmentation_factor: Fraction of data to augment
            
        Returns:
            Augmented dataset
        """
        try:
            logger.info("🎭 Applying Mixup augmentation")
            
            n_augmented = int(len(X) * augmentation_factor)
            
            X_mixup_list = []
            y_mixup_list = []
            
            for _ in range(n_augmented):
                # Sample two random indices
                idx1, idx2 = np.random.choice(len(X), size=2, replace=False)
                
                # Sample mixing ratio from Beta distribution
                if self.alpha > 0:
                    lam = np.random.beta(self.alpha, self.alpha)
                else:
                    lam = 1
                
                # Mix features
                x_mixed = lam * X[idx1] + (1 - lam) * X[idx2]
                
                # For classification: mix labels, for regression: mix targets
                if y.dtype in [np.int32, np.int64] and len(np.unique(y)) < 20:
                    # Classification: create soft labels
                    n_classes = len(np.unique(y))
                    y_mixed = np.zeros(n_classes)
                    y_mixed[y[idx1]] = lam
                    y_mixed[y[idx2]] = 1 - lam
                else:
                    # Regression: mix targets
                    y_mixed = lam * y[idx1] + (1 - lam) * y[idx2]
                
                X_mixup_list.append(x_mixed)
                y_mixup_list.append(y_mixed)
            
            X_mixup = np.array(X_mixup_list)
            y_mixup = np.array(y_mixup_list)
            
            # Combine with original data
            X_combined = np.vstack([X, X_mixup])
            if y_mixup.ndim > 1:
                # Soft labels: convert original to one-hot if needed
                n_classes = y_mixup.shape[1]
                y_onehot = np.eye(n_classes)[y]
                y_combined = np.vstack([y_onehot, y_mixup])
            else:
                y_combined = np.hstack([y, y_mixup])
            
            logger.info(f"✅ Generated {len(X_mixup)} mixup samples")
            
            return X_combined, y_combined
            
        except Exception as e:
            logger.error(f"❌ Mixup augmentation failed: {e}")
            raise


class CutmixAugmenter(BaseAugmenter):
    """Cutmix augmentation for feature matrices."""
    
    def __init__(
        self,
        alpha: float = 1.0,
        random_state: int = 42
    ):
        """
        Initialize cutmix augmenter.
        
        Args:
            alpha: Beta distribution parameter
            random_state: Random state
        """
        super().__init__(random_state)
        self.alpha = alpha
    
    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augmentation_factor: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply cutmix augmentation.
        
        Args:
            X: Features (assumes 2D: samples x features)
            y: Targets
            augmentation_factor: Fraction of data to augment
            
        Returns:
            Augmented dataset
        """
        try:
            logger.info("✂️ Applying Cutmix augmentation")
            
            n_augmented = int(len(X) * augmentation_factor)
            
            X_cutmix_list = []
            y_cutmix_list = []
            
            for _ in range(n_augmented):
                # Sample two random indices
                idx1, idx2 = np.random.choice(len(X), size=2, replace=False)
                
                # Sample mixing ratio
                lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 0.5
                
                # Determine cut region
                n_features = X.shape[1]
                cut_size = int(n_features * (1 - lam))
                cut_start = np.random.randint(0, n_features - cut_size + 1)
                cut_end = cut_start + cut_size
                
                # Create mixed sample
                x_cutmix = X[idx1].copy()
                x_cutmix[cut_start:cut_end] = X[idx2][cut_start:cut_end]
                
                # Adjust lambda based on actual cut ratio
                actual_lam = 1 - (cut_size / n_features)
                
                # Mix labels
                if y.dtype in [np.int32, np.int64] and len(np.unique(y)) < 20:
                    # Classification: create soft labels
                    n_classes = len(np.unique(y))
                    y_cutmix = np.zeros(n_classes)
                    y_cutmix[y[idx1]] = actual_lam
                    y_cutmix[y[idx2]] = 1 - actual_lam
                else:
                    # Regression: mix targets
                    y_cutmix = actual_lam * y[idx1] + (1 - actual_lam) * y[idx2]
                
                X_cutmix_list.append(x_cutmix)
                y_cutmix_list.append(y_cutmix)
            
            X_cutmix = np.array(X_cutmix_list)
            y_cutmix = np.array(y_cutmix_list)
            
            # Combine with original data
            X_combined = np.vstack([X, X_cutmix])
            if y_cutmix.ndim > 1:
                # Soft labels: convert original to one-hot if needed
                n_classes = y_cutmix.shape[1]
                y_onehot = np.eye(n_classes)[y]
                y_combined = np.vstack([y_onehot, y_cutmix])
            else:
                y_combined = np.hstack([y, y_cutmix])
            
            logger.info(f"✅ Generated {len(X_cutmix)} cutmix samples")
            
            return X_combined, y_combined
            
        except Exception as e:
            logger.error(f"❌ Cutmix augmentation failed: {e}")
            raise


class TimeSeriesAugmenter(BaseAugmenter):
    """Augmentation techniques for time series data."""
    
    def __init__(self, random_state: int = 42):
        """Initialize time series augmenter."""
        super().__init__(random_state)
    
    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        methods: List[str] = ["jittering", "scaling", "rotation"],
        augmentation_factor: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply time series augmentation.
        
        Args:
            X: Time series data (samples x time_steps x features)
            y: Targets
            methods: List of augmentation methods
            augmentation_factor: Factor to multiply dataset size
            
        Returns:
            Augmented time series dataset
        """
        try:
            logger.info("📈 Applying time series augmentation")
            
            X_augmented_list = []
            y_augmented_list = []
            
            n_samples = int(len(X) * augmentation_factor)
            
            for _ in range(n_samples):
                # Choose random sample
                idx = np.random.randint(0, len(X))
                x_sample = X[idx].copy()
                y_sample = y[idx]
                
                # Apply random augmentation method
                method = np.random.choice(methods)
                
                if method == "jittering":
                    # Add noise to time series
                    noise_std = 0.01 * np.std(x_sample)
                    noise = np.random.normal(0, noise_std, x_sample.shape)
                    x_augmented = x_sample + noise
                
                elif method == "scaling":
                    # Scale time series
                    scale_factor = np.random.uniform(0.8, 1.2)
                    x_augmented = x_sample * scale_factor
                
                elif method == "rotation":
                    # Time warping (simple rotation)
                    if x_sample.ndim == 2:  # multivariate time series
                        # Apply rotation to feature space
                        angle = np.random.uniform(-0.1, 0.1)
                        cos_a, sin_a = np.cos(angle), np.sin(angle)
                        if x_sample.shape[1] >= 2:
                            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                            x_augmented = x_sample.copy()
                            x_augmented[:, :2] = x_sample[:, :2] @ rotation_matrix.T
                        else:
                            x_augmented = x_sample
                    else:
                        x_augmented = x_sample
                
                elif method == "time_shift":
                    # Shift time series
                    shift = np.random.randint(-len(x_sample)//10, len(x_sample)//10)
                    x_augmented = np.roll(x_sample, shift, axis=0)
                
                else:
                    x_augmented = x_sample
                
                X_augmented_list.append(x_augmented)
                y_augmented_list.append(y_sample)
            
            X_augmented = np.array(X_augmented_list)
            y_augmented = np.array(y_augmented_list)
            
            # Combine with original data
            X_combined = np.vstack([X, X_augmented])
            y_combined = np.hstack([y, y_augmented])
            
            logger.info(f"✅ Generated {len(X_augmented)} time series augmented samples")
            
            return X_combined, y_combined
            
        except Exception as e:
            logger.error(f"❌ Time series augmentation failed: {e}")
            raise


class DataAugmentationPipeline:
    """Pipeline for combining multiple augmentation techniques."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize augmentation pipeline.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.augmenters = []
    
    def add_augmenter(
        self,
        augmenter: BaseAugmenter,
        probability: float = 1.0,
        **kwargs
    ):
        """
        Add an augmenter to the pipeline.
        
        Args:
            augmenter: Augmenter instance
            probability: Probability of applying this augmenter
            **kwargs: Additional parameters for the augmenter
        """
        self.augmenters.append({
            "augmenter": augmenter,
            "probability": probability,
            "kwargs": kwargs
        })
        
        logger.info(f"➕ Added {type(augmenter).__name__} to pipeline")
    
    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        apply_all: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation pipeline.
        
        Args:
            X: Features
            y: Targets
            apply_all: Whether to apply all augmenters or sample randomly
            
        Returns:
            Augmented dataset
        """
        try:
            logger.info("🔄 Applying augmentation pipeline")
            
            X_current, y_current = X.copy(), y.copy()
            
            for augmenter_config in self.augmenters:
                augmenter = augmenter_config["augmenter"]
                probability = augmenter_config["probability"]
                kwargs = augmenter_config["kwargs"]
                
                # Decide whether to apply this augmenter
                if apply_all or np.random.random() < probability:
                    logger.info(f"  🔧 Applying {type(augmenter).__name__}")
                    
                    X_aug, y_aug = augmenter.augment(X_current, y_current, **kwargs)
                    X_current, y_current = X_aug, y_aug
                else:
                    logger.info(f"  ⏭️ Skipping {type(augmenter).__name__}")
            
            logger.info(f"✅ Pipeline complete: {len(X)} → {len(X_current)} samples")
            
            return X_current, y_current
            
        except Exception as e:
            logger.error(f"❌ Augmentation pipeline failed: {e}")
            raise
    
    def get_augmentation_summary(self) -> Dict[str, Any]:
        """
        Get summary of augmentation pipeline.
        
        Returns:
            Summary dictionary
        """
        summary = {
            "total_augmenters": len(self.augmenters),
            "augmenters": []
        }
        
        for config in self.augmenters:
            augmenter_info = {
                "name": type(config["augmenter"]).__name__,
                "probability": config["probability"],
                "parameters": config["kwargs"]
            }
            summary["augmenters"].append(augmenter_info)
        
        return summary


def create_augmentation_config(
    task_type: str = "classification",
    data_type: str = "tabular",
    imbalanced: bool = False
) -> DataAugmentationPipeline:
    """
    Create a pre-configured augmentation pipeline.
    
    Args:
        task_type: Type of ML task ('classification' or 'regression')
        data_type: Type of data ('tabular', 'timeseries')
        imbalanced: Whether dataset is imbalanced
        
    Returns:
        Configured augmentation pipeline
    """
    pipeline = DataAugmentationPipeline()
    
    if data_type == "tabular":
        # Add noise augmentation
        pipeline.add_augmenter(
            NoiseAugmenter(noise_type="gaussian", noise_level=0.05),
            probability=0.7,
            augmentation_factor=0.3
        )
        
        # Add mixup for regularization
        pipeline.add_augmenter(
            FeatureMixupAugmenter(alpha=0.2),
            probability=0.5,
            augmentation_factor=0.2
        )
        
        # Add SMOTE for imbalanced classification
        if task_type == "classification" and imbalanced:
            pipeline.add_augmenter(
                SMOTEAugmenter(k_neighbors=5),
                probability=1.0,
                sampling_strategy="auto"
            )
    
    elif data_type == "timeseries":
        # Add time series specific augmentations
        pipeline.add_augmenter(
            TimeSeriesAugmenter(),
            probability=0.8,
            methods=["jittering", "scaling", "time_shift"],
            augmentation_factor=0.5
        )
    
    logger.info(f"✅ Created {data_type} augmentation pipeline for {task_type}")
    
    return pipeline
