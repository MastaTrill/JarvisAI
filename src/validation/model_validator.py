"""
Advanced Model Validation and Testing Framework.

This module provides comprehensive model validation with:
- Cross-validation strategies
- Model performance metrics
- Bias and fairness testing
- Robustness testing
- Statistical significance testing
- Model interpretability analysis
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, explained_variance_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelValidator:
    """Advanced model validation framework."""
    
    def __init__(
        self,
        task_type: str = "classification",  # 'classification' or 'regression'
        random_state: int = 42
    ):
        """
        Initialize model validator.
        
        Args:
            task_type: Type of ML task ('classification' or 'regression')
            random_state: Random state for reproducibility
        """
        self.task_type = task_type
        self.random_state = random_state
        self.validation_results = {}
        
        logger.info(f"🔍 Initialized ModelValidator for {task_type} task")
    
    def cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv_strategy: str = "kfold",
        n_splits: int = 5,
        scoring_metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform cross-validation with multiple metrics.
        
        Args:
            model: Model to validate
            X: Feature matrix
            y: Target vector
            cv_strategy: Cross-validation strategy ('kfold', 'stratified', 'timeseries')
            n_splits: Number of CV folds
            scoring_metrics: List of metrics to compute
            
        Returns:
            Dictionary with CV results
        """
        try:
            logger.info(f"🔄 Starting {cv_strategy} cross-validation with {n_splits} folds")
            
            # Set up cross-validation strategy
            if cv_strategy == "kfold":
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            elif cv_strategy == "stratified":
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            elif cv_strategy == "timeseries":
                cv = TimeSeriesSplit(n_splits=n_splits)
            else:
                raise ValueError(f"Unknown CV strategy: {cv_strategy}")
            
            # Set default metrics based on task type
            if scoring_metrics is None:
                if self.task_type == "classification":
                    scoring_metrics = ["accuracy", "precision", "recall", "f1", "auc"]
                else:
                    scoring_metrics = ["mse", "mae", "r2", "explained_variance"]
            
            # Initialize results storage
            cv_results = {metric: [] for metric in scoring_metrics}
            fold_predictions = []
            fold_true_labels = []
            
            # Perform cross-validation
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                logger.info(f"  📋 Processing fold {fold + 1}/{n_splits}")
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train)
                
                # Make predictions
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_val)
                else:
                    y_pred = model.forward(X_val)
                    if isinstance(y_pred, tuple):
                        y_pred = y_pred[0]
                
                # Store predictions for ensemble analysis
                fold_predictions.append(y_pred)
                fold_true_labels.append(y_val)
                
                # Compute metrics
                fold_metrics = self._compute_metrics(y_val, y_pred, scoring_metrics, model)
                
                for metric, value in fold_metrics.items():
                    cv_results[metric].append(value)
            
            # Compute summary statistics
            summary_results = {}
            for metric, values in cv_results.items():
                summary_results[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "values": values
                }
            
            self.validation_results["cross_validation"] = summary_results
            self.validation_results["fold_predictions"] = fold_predictions
            self.validation_results["fold_true_labels"] = fold_true_labels
            
            logger.info("✅ Cross-validation completed successfully")
            
            return summary_results
            
        except Exception as e:
            logger.error(f"❌ Cross-validation failed: {e}")
            raise
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: List[str],
        model: Any = None
    ) -> Dict[str, float]:
        """Compute specified metrics."""
        results = {}
        
        try:
            if self.task_type == "classification":
                # Convert predictions to class labels if needed
                if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    y_pred_probs = y_pred
                else:
                    y_pred_classes = (y_pred > 0.5).astype(int) if y_pred.ndim == 1 else y_pred
                    y_pred_probs = y_pred
                
                for metric in metrics:
                    if metric == "accuracy":
                        results[metric] = accuracy_score(y_true, y_pred_classes)
                    elif metric == "precision":
                        results[metric] = precision_score(y_true, y_pred_classes, average="macro", zero_division=0)
                    elif metric == "recall":
                        results[metric] = recall_score(y_true, y_pred_classes, average="macro", zero_division=0)
                    elif metric == "f1":
                        results[metric] = f1_score(y_true, y_pred_classes, average="macro", zero_division=0)
                    elif metric == "auc":
                        try:
                            if len(np.unique(y_true)) == 2:  # Binary classification
                                if y_pred_probs.ndim > 1 and y_pred_probs.shape[1] > 1:
                                    results[metric] = roc_auc_score(y_true, y_pred_probs[:, 1])
                                else:
                                    results[metric] = roc_auc_score(y_true, y_pred_probs)
                            else:  # Multiclass
                                results[metric] = roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average="macro")
                        except:
                            results[metric] = 0.0
            
            else:  # Regression
                for metric in metrics:
                    if metric == "mse":
                        results[metric] = mean_squared_error(y_true, y_pred)
                    elif metric == "mae":
                        results[metric] = mean_absolute_error(y_true, y_pred)
                    elif metric == "r2":
                        results[metric] = r2_score(y_true, y_pred)
                    elif metric == "explained_variance":
                        results[metric] = explained_variance_score(y_true, y_pred)
        
        except Exception as e:
            logger.warning(f"⚠️ Error computing some metrics: {e}")
            for metric in metrics:
                if metric not in results:
                    results[metric] = 0.0
        
        return results
    
    def bias_fairness_analysis(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_features: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze model bias and fairness across sensitive features.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: True labels
            sensitive_features: Sensitive feature values
            feature_names: Names of sensitive features
            
        Returns:
            Bias analysis results
        """
        try:
            logger.info("🔍 Performing bias and fairness analysis")
            
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
                    group_metrics[group] = self._compute_metrics(
                        y_true_group, y_pred_group, 
                        ["accuracy", "precision", "recall"] if self.task_type == "classification" else ["mse", "mae"]
                    )
            
            bias_results["group_metrics"] = group_metrics
            
            # Compute fairness metrics
            if self.task_type == "classification":
                # Demographic parity
                group_positive_rates = {}
                for group in unique_groups:
                    group_mask = sensitive_features == group
                    positive_rate = np.mean(y_pred[group_mask])
                    group_positive_rates[group] = positive_rate
                
                bias_results["demographic_parity"] = group_positive_rates
                
                # Equalized odds
                group_tpr = {}
                group_fpr = {}
                for group in unique_groups:
                    group_mask = sensitive_features == group
                    y_true_group = y[group_mask]
                    y_pred_group = y_pred[group_mask]
                    
                    if len(y_true_group) > 0:
                        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
                        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                        
                        group_tpr[group] = tpr
                        group_fpr[group] = fpr
                
                bias_results["true_positive_rates"] = group_tpr
                bias_results["false_positive_rates"] = group_fpr
            
            # Calculate bias metrics
            fairness_violations = self._calculate_fairness_violations(bias_results)
            bias_results["fairness_violations"] = fairness_violations
            
            self.validation_results["bias_analysis"] = bias_results
            
            logger.info("✅ Bias and fairness analysis completed")
            return bias_results
            
        except Exception as e:
            logger.error(f"❌ Bias analysis failed: {e}")
            raise
    
    def _calculate_fairness_violations(self, bias_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate fairness violation metrics."""
        violations = {}
        
        try:
            if "group_metrics" in bias_results:
                # Calculate max difference in accuracy across groups
                accuracies = [metrics.get("accuracy", 0) for metrics in bias_results["group_metrics"].values()]
                violations["accuracy_disparity"] = max(accuracies) - min(accuracies) if accuracies else 0.0
            
            if "demographic_parity" in bias_results:
                # Calculate demographic parity violation
                rates = list(bias_results["demographic_parity"].values())
                violations["demographic_parity_violation"] = max(rates) - min(rates) if rates else 0.0
            
            if "true_positive_rates" in bias_results:
                # Calculate equalized odds violation
                tprs = list(bias_results["true_positive_rates"].values())
                violations["equalized_odds_violation"] = max(tprs) - min(tprs) if tprs else 0.0
                
        except Exception as e:
            logger.warning(f"⚠️ Error calculating fairness violations: {e}")
        
        return violations
    
    def robustness_testing(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2],
        perturbation_types: List[str] = ["gaussian", "uniform"]
    ) -> Dict[str, Any]:
        """
        Test model robustness to input perturbations.
        
        Args:
            model: Trained model
            X: Clean feature matrix
            y: True labels
            noise_levels: List of noise levels to test
            perturbation_types: Types of perturbations to apply
            
        Returns:
            Robustness test results
        """
        try:
            logger.info("🛡️ Performing robustness testing")
            
            robustness_results = {}
            
            # Get baseline performance
            if hasattr(model, 'predict'):
                y_pred_clean = model.predict(X)
            else:
                y_pred_clean = model.forward(X)
                if isinstance(y_pred_clean, tuple):
                    y_pred_clean = y_pred_clean[0]
            
            baseline_metrics = self._compute_metrics(
                y, y_pred_clean,
                ["accuracy"] if self.task_type == "classification" else ["mse"]
            )
            robustness_results["baseline"] = baseline_metrics
            
            # Test different perturbation types and levels
            for perturbation_type in perturbation_types:
                robustness_results[perturbation_type] = {}
                
                for noise_level in noise_levels:
                    logger.info(f"  🔧 Testing {perturbation_type} noise at level {noise_level}")
                    
                    # Apply perturbation
                    if perturbation_type == "gaussian":
                        noise = np.random.normal(0, noise_level, X.shape)
                    elif perturbation_type == "uniform":
                        noise = np.random.uniform(-noise_level, noise_level, X.shape)
                    else:
                        continue
                    
                    X_perturbed = X + noise
                    
                    # Make predictions on perturbed data
                    if hasattr(model, 'predict'):
                        y_pred_perturbed = model.predict(X_perturbed)
                    else:
                        y_pred_perturbed = model.forward(X_perturbed)
                        if isinstance(y_pred_perturbed, tuple):
                            y_pred_perturbed = y_pred_perturbed[0]
                    
                    # Compute metrics
                    perturbed_metrics = self._compute_metrics(
                        y, y_pred_perturbed,
                        ["accuracy"] if self.task_type == "classification" else ["mse"]
                    )
                    
                    # Calculate performance degradation
                    degradation = {}
                    for metric in baseline_metrics:
                        baseline_val = baseline_metrics[metric]
                        perturbed_val = perturbed_metrics[metric]
                        
                        if metric in ["accuracy", "r2"]:  # Higher is better
                            degradation[f"{metric}_degradation"] = baseline_val - perturbed_val
                        else:  # Lower is better (loss metrics)
                            degradation[f"{metric}_degradation"] = perturbed_val - baseline_val
                    
                    robustness_results[perturbation_type][noise_level] = {
                        "metrics": perturbed_metrics,
                        "degradation": degradation
                    }
            
            self.validation_results["robustness"] = robustness_results
            
            logger.info("✅ Robustness testing completed")
            return robustness_results
            
        except Exception as e:
            logger.error(f"❌ Robustness testing failed: {e}")
            raise
    
    def statistical_significance_test(
        self,
        model1_results: List[float],
        model2_results: List[float],
        test_type: str = "paired_ttest",
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform statistical significance test between model results.
        
        Args:
            model1_results: Results from first model
            model2_results: Results from second model
            test_type: Type of test ('paired_ttest', 'wilcoxon', 'mannwhitney')
            alpha: Significance level
            
        Returns:
            Statistical test results
        """
        try:
            logger.info(f"📊 Performing {test_type} statistical significance test")
            
            test_results = {}
            
            if test_type == "paired_ttest":
                statistic, p_value = stats.ttest_rel(model1_results, model2_results)
            elif test_type == "wilcoxon":
                statistic, p_value = stats.wilcoxon(model1_results, model2_results)
            elif test_type == "mannwhitney":
                statistic, p_value = stats.mannwhitneyu(model1_results, model2_results)
            else:
                raise ValueError(f"Unknown test type: {test_type}")
            
            test_results = {
                "test_type": test_type,
                "statistic": statistic,
                "p_value": p_value,
                "is_significant": p_value < alpha,
                "alpha": alpha,
                "model1_mean": np.mean(model1_results),
                "model2_mean": np.mean(model2_results),
                "difference": np.mean(model1_results) - np.mean(model2_results)
            }
            
            logger.info(f"✅ Statistical test completed (p-value: {p_value:.6f})")
            
            return test_results
            
        except Exception as e:
            logger.error(f"❌ Statistical significance test failed: {e}")
            raise
    
    def generate_validation_report(
        self,
        output_dir: str = "validation_reports"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            output_dir: Directory to save report files
            
        Returns:
            Summary of validation results
        """
        try:
            logger.info("📝 Generating validation report")
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            report_summary = {}
            
            # Generate plots and summaries for each validation type
            if "cross_validation" in self.validation_results:
                cv_summary = self._generate_cv_report(output_path)
                report_summary["cross_validation"] = cv_summary
            
            if "bias_analysis" in self.validation_results:
                bias_summary = self._generate_bias_report(output_path)
                report_summary["bias_analysis"] = bias_summary
            
            if "robustness" in self.validation_results:
                robustness_summary = self._generate_robustness_report(output_path)
                report_summary["robustness"] = robustness_summary
            
            # Save summary to JSON
            import json
            with open(output_path / "validation_summary.json", "w") as f:
                json.dump(report_summary, f, indent=2, default=str)
            
            logger.info(f"✅ Validation report generated in {output_path}")
            
            return report_summary
            
        except Exception as e:
            logger.error(f"❌ Failed to generate validation report: {e}")
            raise
    
    def _generate_cv_report(self, output_path: Path) -> Dict[str, Any]:
        """Generate cross-validation report."""
        cv_results = self.validation_results["cross_validation"]
        
        # Create CV results plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot metric distributions
        metrics = list(cv_results.keys())
        means = [cv_results[metric]["mean"] for metric in metrics]
        stds = [cv_results[metric]["std"] for metric in metrics]
        
        axes[0].bar(metrics, means, yerr=stds, capsize=5, alpha=0.7)
        axes[0].set_title("Cross-Validation Results")
        axes[0].set_ylabel("Score")
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot metric stability (coefficient of variation)
        cv_coefficients = [cv_results[metric]["std"] / cv_results[metric]["mean"] 
                          for metric in metrics if cv_results[metric]["mean"] != 0]
        valid_metrics = [metric for metric in metrics if cv_results[metric]["mean"] != 0]
        
        if cv_coefficients:
            axes[1].bar(valid_metrics, cv_coefficients, alpha=0.7, color='orange')
            axes[1].set_title("Metric Stability (Coefficient of Variation)")
            axes[1].set_ylabel("CV = std/mean")
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / "cross_validation_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Summary statistics
        summary = {
            "best_metric": max(metrics, key=lambda m: cv_results[m]["mean"]),
            "most_stable_metric": min(valid_metrics, key=lambda m: cv_results[m]["std"] / cv_results[m]["mean"]) if valid_metrics else None,
            "avg_performance": np.mean([cv_results[m]["mean"] for m in metrics]),
            "results": cv_results
        }
        
        return summary
    
    def _generate_bias_report(self, output_path: Path) -> Dict[str, Any]:
        """Generate bias analysis report."""
        bias_results = self.validation_results["bias_analysis"]
        
        # Create bias visualization
        if "group_metrics" in bias_results:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            groups = list(bias_results["group_metrics"].keys())
            metrics = list(bias_results["group_metrics"][groups[0]].keys()) if groups else []
            
            x = np.arange(len(groups))
            width = 0.8 / len(metrics) if metrics else 0.8
            
            for i, metric in enumerate(metrics):
                values = [bias_results["group_metrics"][group].get(metric, 0) for group in groups]
                ax.bar(x + i * width, values, width, label=metric, alpha=0.7)
            
            ax.set_xlabel("Groups")
            ax.set_ylabel("Score")
            ax.set_title("Performance by Group")
            ax.set_xticks(x + width * (len(metrics) - 1) / 2)
            ax.set_xticklabels(groups)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(output_path / "bias_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        summary = {
            "fairness_violations": bias_results.get("fairness_violations", {}),
            "max_accuracy_disparity": bias_results.get("fairness_violations", {}).get("accuracy_disparity", 0),
            "group_count": len(bias_results.get("group_metrics", {}))
        }
        
        return summary
    
    def _generate_robustness_report(self, output_path: Path) -> Dict[str, Any]:
        """Generate robustness report."""
        robustness_results = self.validation_results["robustness"]
        
        # Create robustness plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        for perturbation_type in robustness_results:
            if perturbation_type == "baseline":
                continue
                
            noise_levels = list(robustness_results[perturbation_type].keys())
            degradations = []
            
            for level in noise_levels:
                degradation_dict = robustness_results[perturbation_type][level]["degradation"]
                # Use first degradation metric
                if degradation_dict:
                    degradations.append(list(degradation_dict.values())[0])
                else:
                    degradations.append(0)
            
            ax.plot(noise_levels, degradations, marker='o', label=perturbation_type)
        
        ax.set_xlabel("Noise Level")
        ax.set_ylabel("Performance Degradation")
        ax.set_title("Model Robustness to Input Perturbations")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "robustness_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate robustness scores
        max_degradation = 0
        for perturbation_type in robustness_results:
            if perturbation_type == "baseline":
                continue
            for level in robustness_results[perturbation_type]:
                degradation_dict = robustness_results[perturbation_type][level]["degradation"]
                if degradation_dict:
                    max_degradation = max(max_degradation, max(degradation_dict.values()))
        
        summary = {
            "max_performance_degradation": max_degradation,
            "robustness_score": max(0, 1 - max_degradation),  # Simple robustness score
            "perturbation_types_tested": [t for t in robustness_results.keys() if t != "baseline"]
        }
        
        return summary
