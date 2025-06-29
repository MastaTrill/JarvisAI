"""
Simple Model Validation without scikit-learn dependencies.

This module provides basic model validation functionality using only numpy:
- Simple cross-validation
- Basic performance metrics
- Robustness testing
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)


class SimpleValidator:
    """Simple model validation framework using only numpy."""
    
    def __init__(
        self,
        task_type: str = "classification",  # 'classification' or 'regression'
        random_state: int = 42
    ):
        """
        Initialize simple validator.
        
        Args:
            task_type: Type of ML task ('classification' or 'regression')
            random_state: Random state for reproducibility
        """
        self.task_type = task_type
        self.random_state = random_state
        self.validation_results = {}
        
        np.random.seed(random_state)
        
        logger.info(f"🔍 Initialized SimpleValidator for {task_type} task")
    
    def simple_cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        test_size: float = 0.2
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform simple cross-validation.
        
        Args:
            model: Model to validate
            X: Feature matrix
            y: Target vector
            n_splits: Number of CV folds
            test_size: Test size fraction
            
        Returns:
            Dictionary with CV results
        """
        try:
            logger.info(f"🔄 Starting simple cross-validation with {n_splits} folds")
            
            n_samples = len(X)
            indices = np.random.permutation(n_samples)
            fold_size = n_samples // n_splits
            
            cv_results = {"accuracy": [], "mse": []}
            
            for fold in range(n_splits):
                logger.info(f"  📋 Processing fold {fold + 1}/{n_splits}")
                
                # Split data
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < n_splits - 1 else n_samples
                
                test_indices = indices[start_idx:end_idx]
                train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
                
                X_train, X_test = X[train_indices], X[test_indices]
                y_train, y_test = y[train_indices], y[test_indices]
                
                # Train model
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train, epochs=20, verbose=False)
                
                # Make predictions
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test)
                else:
                    y_pred = model.forward(X_test)
                    if isinstance(y_pred, tuple):
                        y_pred = y_pred[0]
                
                # Compute metrics
                if self.task_type == "classification":
                    # Convert to class predictions
                    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                        y_pred_classes = np.argmax(y_pred, axis=1)
                    else:
                        y_pred_classes = (y_pred > 0.5).astype(int)
                    
                    accuracy = np.mean(y_test == y_pred_classes)
                    cv_results["accuracy"].append(accuracy)
                    
                else:  # Regression
                    mse = np.mean((y_test - y_pred) ** 2)
                    cv_results["mse"].append(mse)
            
            # Compute summary statistics
            summary_results = {}
            for metric, values in cv_results.items():
                if values:  # Only include metrics that have values
                    summary_results[metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "values": values
                    }
            
            self.validation_results["cross_validation"] = summary_results
            
            logger.info("✅ Cross-validation completed successfully")
            return summary_results
            
        except Exception as e:
            logger.error(f"❌ Cross-validation failed: {e}")
            raise
    
    def simple_bias_analysis(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Simple bias analysis.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: True labels
            sensitive_features: Sensitive feature values
            
        Returns:
            Bias analysis results
        """
        try:
            logger.info("🔍 Performing simple bias analysis")
            
            # Make predictions
            if hasattr(model, 'predict'):
                y_pred = model.predict(X)
            else:
                y_pred = model.forward(X)
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]
            
            # Convert to class predictions if needed
            if self.task_type == "classification":
                if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                    y_pred = np.argmax(y_pred, axis=1)
                else:
                    y_pred = (y_pred > 0.5).astype(int)
            
            bias_results = {}
            unique_groups = np.unique(sensitive_features)
            
            # Compute metrics per group
            group_metrics = {}
            for group in unique_groups:
                group_mask = sensitive_features == group
                y_true_group = y[group_mask]
                y_pred_group = y_pred[group_mask]
                
                if len(y_true_group) > 0:
                    if self.task_type == "classification":
                        accuracy = np.mean(y_true_group == y_pred_group)
                        group_metrics[group] = {"accuracy": accuracy}
                    else:
                        mse = np.mean((y_true_group - y_pred_group) ** 2)
                        group_metrics[group] = {"mse": mse}
            
            bias_results["group_metrics"] = group_metrics
            
            # Calculate fairness violations
            if self.task_type == "classification":
                accuracies = [metrics.get("accuracy", 0) for metrics in group_metrics.values()]
                bias_results["accuracy_disparity"] = max(accuracies) - min(accuracies) if accuracies else 0.0
            
            self.validation_results["bias_analysis"] = bias_results
            
            logger.info("✅ Simple bias analysis completed")
            return bias_results
            
        except Exception as e:
            logger.error(f"❌ Bias analysis failed: {e}")
            raise
    
    def simple_robustness_test(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        noise_levels: List[float] = [0.01, 0.05, 0.1]
    ) -> Dict[str, Any]:
        """
        Simple robustness testing.
        
        Args:
            model: Trained model
            X: Clean feature matrix
            y: True labels
            noise_levels: List of noise levels to test
            
        Returns:
            Robustness test results
        """
        try:
            logger.info("🛡️ Performing simple robustness testing")
            
            robustness_results = {}
            
            # Get baseline performance
            if hasattr(model, 'predict'):
                y_pred_clean = model.predict(X)
            else:
                y_pred_clean = model.forward(X)
                if isinstance(y_pred_clean, tuple):
                    y_pred_clean = y_pred_clean[0]
            
            if self.task_type == "classification":
                if y_pred_clean.ndim > 1 and y_pred_clean.shape[1] > 1:
                    y_pred_clean = np.argmax(y_pred_clean, axis=1)
                else:
                    y_pred_clean = (y_pred_clean > 0.5).astype(int)
                baseline_score = np.mean(y == y_pred_clean)
                metric_name = "accuracy"
            else:
                baseline_score = np.mean((y - y_pred_clean) ** 2)
                metric_name = "mse"
            
            robustness_results["baseline"] = {metric_name: baseline_score}
            
            # Test different noise levels
            for noise_level in noise_levels:
                logger.info(f"  🔧 Testing gaussian noise at level {noise_level}")
                
                # Apply noise
                noise = np.random.normal(0, noise_level, X.shape)
                X_perturbed = X + noise
                
                # Make predictions on perturbed data
                if hasattr(model, 'predict'):
                    y_pred_perturbed = model.predict(X_perturbed)
                else:
                    y_pred_perturbed = model.forward(X_perturbed)
                    if isinstance(y_pred_perturbed, tuple):
                        y_pred_perturbed = y_pred_perturbed[0]
                
                if self.task_type == "classification":
                    if y_pred_perturbed.ndim > 1 and y_pred_perturbed.shape[1] > 1:
                        y_pred_perturbed = np.argmax(y_pred_perturbed, axis=1)
                    else:
                        y_pred_perturbed = (y_pred_perturbed > 0.5).astype(int)
                    perturbed_score = np.mean(y == y_pred_perturbed)
                    degradation = baseline_score - perturbed_score
                else:
                    perturbed_score = np.mean((y - y_pred_perturbed) ** 2)
                    degradation = perturbed_score - baseline_score
                
                robustness_results[f"noise_{noise_level}"] = {
                    metric_name: perturbed_score,
                    "degradation": degradation
                }
            
            self.validation_results["robustness"] = robustness_results
            
            logger.info("✅ Simple robustness testing completed")
            return robustness_results
            
        except Exception as e:
            logger.error(f"❌ Robustness testing failed: {e}")
            raise
    
    def generate_simple_report(self) -> Dict[str, Any]:
        """
        Generate simple validation report.
        
        Returns:
            Summary of validation results
        """
        try:
            logger.info("📝 Generating simple validation report")
            
            report_summary = {}
            
            if "cross_validation" in self.validation_results:
                cv_results = self.validation_results["cross_validation"]
                if cv_results:
                    best_metric = max(cv_results.keys(), 
                                    key=lambda m: cv_results[m]["mean"] if m == "accuracy" 
                                    else -cv_results[m]["mean"])
                    report_summary["best_cv_metric"] = best_metric
                    report_summary["best_cv_score"] = cv_results[best_metric]["mean"]
            
            if "bias_analysis" in self.validation_results:
                bias_results = self.validation_results["bias_analysis"]
                if "accuracy_disparity" in bias_results:
                    report_summary["accuracy_disparity"] = bias_results["accuracy_disparity"]
            
            if "robustness" in self.validation_results:
                rob_results = self.validation_results["robustness"]
                max_degradation = 0
                for key, result in rob_results.items():
                    if key != "baseline" and "degradation" in result:
                        max_degradation = max(max_degradation, result["degradation"])
                report_summary["max_robustness_degradation"] = max_degradation
            
            logger.info("✅ Simple validation report generated")
            return report_summary
            
        except Exception as e:
            logger.error(f"❌ Failed to generate validation report: {e}")
            return {}
