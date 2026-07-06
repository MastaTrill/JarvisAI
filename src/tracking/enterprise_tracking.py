"""
Enterprise MLflow Tracking for Jarvis AI
Enhanced Experiment Tracking with Lineage, Governance, and Analytics
"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import numpy as np
import pandas as pd
import pickle
import json
import logging
import os
import sys
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import hashlib
from collections import defaultdict

# Optional imports with graceful fallback
try:
    import mlflow.tensorflow
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TensorFlow not available. TensorFlow model logging will be disabled.")

logger = logging.getLogger(__name__)


class EnterpriseMLFlowTracker:
    """Enterprise-grade MLFlow tracking with advanced features."""
    
    def __init__(
        self,
        experiment_name: str = "Jarvis_AI_Experiments",
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """
        Initialize enterprise MLFlow tracker.
        
        Args:
            experiment_name: Name of the MLFlow experiment
            tracking_uri: MLFlow tracking server URI
            registry_uri: MLFlow model registry URI
            artifact_location: Location for storing artifacts
        """
        self.experiment_name = experiment_name
        
        # Set tracking and registry URIs
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
            
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                if artifact_location:
                    self.experiment_id = mlflow.create_experiment(
                        experiment_name, 
                        artifact_location=artifact_location
                    )
                else:
                    self.experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new MLFlow experiment: {experiment_name}")
            else:
                self.experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLFlow experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"Failed to set up MLFlow experiment: {e}")
            raise
            
        # Initialize model registry handler
        self.model_registry = EnterpriseModelRegistry(registry_uri)
        
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
        source_name: Optional[str] = None,
        source_version: Optional[str] = None
    ) -> str:
        """
        Start a new MLFlow run with enhanced metadata.
        
        Args:
            run_name: Name for the run
            tags: Dictionary of tags to add
            nested: Whether this is a nested run
            source_name: Name of the source file/script
            source_version: Version/commit of the source
            
        Returns:
            Run ID
        """
        try:
            run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                nested=nested
            )
            
            # Enhanced default tags
            default_tags = {
                "framework": "jarvis_ai",
                "created_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "source_name": source_name or "unknown",
                "source_version": source_version or "unknown",
                "host_name": os.environ.get("COMPUTERNAME", "unknown"),
                "user_name": os.environ.get("USERNAME", "unknown")
            }
            
            if tags:
                default_tags.update(tags)
                
            mlflow.set_tags(default_tags)
            
            logger.info(f"Started MLFlow run: {run.info.run_id}")
            return run.info.run_id
            
        except Exception as e:
            logger.error(f"Failed to start MLFlow run: {e}")
            raise
    
    def log_params(self, params: Dict[str, Any], prefix: str = ""):
        """Log parameters to MLFlow with optional prefix."""
        try:
            # Convert complex objects to strings and add prefix if needed
            processed_params = {}
            for key, value in params.items():
                full_key = f"{prefix}{key}" if prefix else key
                
                if isinstance(value, (dict, list)):
                    processed_params[full_key] = json.dumps(value)
                elif isinstance(value, np.ndarray):
                    processed_params[full_key] = {
                        "shape": value.shape,
                        "dtype": str(value.dtype),
                        "size": value.size
                    }
                elif isinstance(value, (int, float, str, bool)):
                    processed_params[full_key] = value
                else:
                    processed_params[full_key] = str(value)
            
            mlflow.log_params(processed_params)
            logger.info(f"Logged {len(processed_params)} parameters{f' with prefix {prefix}' if prefix else ''}")
            
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLFlow."""
        try:
            # Filter out non-numeric values
            numeric_metrics = {
                k: v for k, v in metrics.items() 
                if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v)
            }
            
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics, step=step)
                logger.info(f"Logged {len(numeric_metrics)} metrics" + 
                           (f" at step {step}" if step else ""))
            else:
                logger.warning("No valid numeric metrics to log")
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_model(
        self,
        model: Any,
        model_name: str,
        flavor: str = "sklearn",
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
        registered_model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a model to MLFlow with enhanced metadata.
        
        Args:
            model: The model object
            model_name: Name for the model artifact
            flavor: MLFlow flavor ('sklearn', 'pytorch', 'tensorflow', 'custom')
            signature: Model signature
            input_example: Example input
            registered_model_name: Name for model registry
            metadata: Additional metadata to store with model
        """
        try:
            if flavor == "sklearn":
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=model_name,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            elif flavor == "pytorch":
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=model_name,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            elif flavor == "tensorflow":
                if not TENSORFLOW_AVAILABLE:
                    raise ImportError("TensorFlow is not available. Please install tensorflow to use this feature.")
                import mlflow.tensorflow
                mlflow.tensorflow.log_model(
                    tf_saved_model=model,
                    artifact_path=model_name,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            else:
                # Custom model - save as pickle with metadata
                with mlflow.start_run(nested=True):
                    # Save model
                    model_path = f"{model_name}.pkl"
                    with open(model_path, "wb") as f:
                        pickle.dump(model, f)
                    
                    # Log model artifact
                    mlflow.log_artifact(model_path, "models")
                    
                    # Log metadata if provided
                    if metadata:
                        metadata_path = f"{model_name}_metadata.json"
                        with open(metadata_path, "w") as f:
                            json.dump(metadata, f, indent=2)
                        mlflow.log_artifact(metadata_path, "models")
                        os.remove(metadata_path)
                    
                    os.remove(model_path)
            
            # Register model if requested
            if registered_model_name:
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
                version = self.model_registry.register_model(
                    model_uri=model_uri,
                    model_name=registered_model_name,
                    description=metadata.get("description") if metadata else None,
                    tags=metadata.get("tags") if metadata else None
                )
                logger.info(f"Registered model {registered_model_name} v{version}")
            
            logger.info(f"Logged model: {model_name} (flavor: {flavor})")
            
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
    
    def log_feature_importance(
        self,
        feature_names: List[str],
        importance_values: List[float],
        title: str = "Feature Importance",
        file_name: str = "feature_importance.png"
    ):
        """Log feature importance plot to MLFlow."""
        try:
            plt.figure(figsize=(10, 6))
            
            # Sort by importance
            sorted_idx = np.argsort(importance_values)
            pos = np.arange(sorted_idx.shape[0]) + .5
            
            plt.barh(pos, np.array(importance_values)[sorted_idx], align='center')
            plt.yticks(pos, np.array(fename)[sorted_idx])
            plt.xlabel('Relative Importance')
            plt.title(title)
            plt.tight_layout()
            
            # Save and log
            temp_path = f"temp_{file_name}"
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(temp_path, "feature_importance")
            plt.close()
            os.remove(temp_path)
            
            logger.info(f"Logged feature importance plot: {file_name}")
            
        except Exception as e:
            logger.error(f"Failed to log feature importance: {e}")
    
    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        file_name: str = "confusion_matrix.png"
    ):
        """Log confusion matrix plot to MLFlow."""
        try:
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names or [f"Class {i}" for i in range(cm.shape[1])],
                yticklabels=class_names or [f"Class {i}" for i in range(cm.shape[0])]
            )
            plt.title(title)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            # Save and log
            temp_path = f"temp_{file_name}"
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(temp_path, "evaluation")
            plt.close()
            os.remove(temp_path)
            
            logger.info(f"Logged confusion matrix: {file_name}")
            
        except Exception as e:
            logger.error(f"Failed to log confusion matrix: {e}")
    
    def log_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "ROC Curve",
        file_name: str = "roc_curve.png"
    ):
        """Log ROC curve to MLFlow."""
        try:
            from sklearn.metrics import roc_curve, auc
            
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right")
            plt.tight_layout()
            
            # Save and log
            temp_path = f"temp_{file_name}"
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(temp_path, "evaluation")
            plt.close()
            os.remove(temp_path)
            
            # Also log AUC as metric
            self.log_metrics({"roc_auc": roc_auc})
            
            logger.info(f"Logged ROC curve: {file_name} (AUC: {roc_auc:.4f})")
            
        except Exception as e:
            logger.error(f"Failed to log ROC curve: {e}")
    
    def log_experiment_summary(
        self,
        description: str = None,
        dataset_info: Dict[str, Any] = None,
        preprocessing_steps: List[str] = None,
        hardware_info: Dict[str, Any] = None
    ):
        """Log comprehensive experiment summary."""
        try:
            summary = {
                "experiment_description": description or "No description provided",
                "timestamp": datetime.now().isoformat(),
                "framework_version": "1.0.0",
                "environment": {
                    "python_version": sys.version,
                    "platform": sys.platform
                }
            }
            
            if dataset_info:
                summary["dataset_info"] = dataset_info
            if preprocessing_steps:
                summary["preprocessing_steps"] = preprocessing_steps
            if hardware_info:
                summary["hardware_info"] = hardware_info
            
            # Save summary as artifact
            summary_path = "experiment_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            mlflow.log_artifact(summary_path)
            os.remove(summary_path)
            
            logger.info("Logged experiment summary")
            
        except Exception as e:
            logger.error(f"Failed to log experiment summary: {e}")
    
    def get_experiment_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics for the experiment."""
        try:
            # Get all runs
            runs_df = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=["start_time DESC"]
            )
            
            if len(runs_df) == 0:
                return {"message": "No runs found"}
            
            analytics = {
                "total_runs": len(runs_df),
                "date_range": {
                    "start": pd.to_datetime(runs_df["start_time"].min(), unit='ms').isoformat() if len(runs_df) > 0 else None,
                    "end": pd.to_datetime(runs_df["start_time"].max(), unit='ms').isoformat() if len(runs_df) > 0 else None
                },
                "status_distribution": (
                    rp_df["status"].value_counts().to_dict() 
                    if "status" in rp_df.columns else {}
                ),
                "top_performing_runs": self._get_top_runs(runs_df),
                "parameter_trends": self._analyze_parameter_trends(runs_df),
                "metric_correlations": self._calculate_metric_correlations(runs_df)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get experiment analytics: {e}")
            return {"error": str(e)}
    
    def _get_top_runs(self, rp_df: pd.DataFrame, metric: str = "metrics.accuracy", top_n: int = 5) -> List[Dict]:
        """Get top performing runs by a specific metric."""
        try:
            if metric in rp_df.columns:
                # Filter out NaN values
                valid_runs = rp_df.dropna(subset=[metric])
                if len(vr_df) > 0:
                    top_runs = vr_df.nlargest(min(tn, len(vr_df)), metric)[["run_id", "start_time", metric]]
                    return top_runs.to_dict('records')
                else:
                    return []
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to get top runs: {e}")
            return []
    
    def _analyze_parameter_trends(self, rp_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how parameters have changed over time."""
        try:
            if len(rp_df) == 0:
                return {}
                
            param_cols = [col for col in rp_df.columns if col.startswith('params.')]
            trends = {}
            
            for col in pc:
                # Try to convert to numeric for trend analysis
                try:
                    # Extract parameter name
                    param_name = col.replace('params.', '')
                    # Create a copy to avoid warnings
                    ts = pd.to_numeric(rp_df[col], errors='coerce').dropna()
                    if len(ts) > 1:
                        # Calculate trend (simple linear regression slope)
                        x = np.arange(len(ts))
                        slope = np.polyfit(x, ts.values, 1)[0] if len(ts) > 1 else 0
                        trends[pa_name] = {
                            "trend": "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable",
                            "slope": float(slope),
                            "mean": float(np.mean(ts)),
                            "std": float(np.std(ts))
                        }
                except Exception as e:
                    logger.debug(f"Could not analyze parameter {col}: {e}")
                    continue
                    
            return trends
        except Exception as e:
            logger.error(f"Failed to analyze parameter trends: {e}")
            return {}
    
    def _calculate_metric_correlations(self, rp_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlations between metrics."""
        try:
            if len(rp_df) < 2:
                return {}
                
            mc = [col for col in rp_df.columns if col.startswith('metrics.')]
            n = []
            
            for col in mc:
                try:
                    # Create a copy to avoid warnings
                    ts = pd.to_numeric(rp_df[col], errors='coerce')
                    if not ts.isna().all():
                        n.append(col)
                except Exception as e:
                    logger.debug(f"Could not process metric column {col}: {e}")
                    continue
            
            if len(n) >= 2:
                # Create a clean dataframe for correlation
                cf = pd.DataFrame()
                for col in n:
                    cc = pd.to_numeric(rp_df[col], errors='coerce')
                    cf[col.replace('metrics.', '')] = cc
                
                cm = cf.corr()
                
                # Extract significant correlations
                correlations = {}
                for i in range(len(cm.columns)):
                    for j in range(i+1, len(cm.columns)):
                        c1 = cm.columns[i]
                        c2 = cm.columns[j]
                        cv = cm.iloc[i, j]
                        if abs(cv) > 0.3:  # Only report meaningful correlations
                            coasins[f"{c1}_vs_{c2}"] = float(cv)
                
                return correlations
            else:
                return {}
        except Exception as e:
            logger.error(f"Failed to calculate metric correlations: {e}")
            return {}


class ExperimentManager:
    """Manages multiple experiments and provides cross-experiment analytics."""
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize experiment manager."""
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        try:
            experiments = mlflow.list_experiments()
            return [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage
                }
                for exp in experiments
            ]
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []
    
    def get_experiment_comparison(
        self,
        experiment_names: List[str],
        metric: str = "metrics.accuracy"
    ) -> Dict[str, Any]:
        """Compare multiple experiments."""
        try:
            comparison = {}
            
            for exp_name in experiment_names:
                exp = mlflow.get_experiment_by_name(exp_name)
                if exp is None:
                    continue
                    
                runs_df = mlflow.search_runs(
                    experiment_ids=[exp.experiment_id],
                    order_by=[f"{metric} DESC"]
                )
                
                if len(runs_df) > 0 and metric in runs_df.columns:
                    best_run = rp_df.iloc[0]
                    comparison[exp_name] = {
                        "best_run_id": str(best_run["run_id"]),
                        "best_score": float(best_run[metric]),
                        "total_runs": len(runs_df),
                        "avg_score": float(runs_df[metric].mean()) if not runs_df[metric].isna().all() else None,
                        "latest_run": {
                            "run_id": str(rps_df.iloc[0]["run_id"]) if len(rps_df) > 0 else None,
                            "timestamp": pd.to_datetime(rps_df.iloc[0]["start_time"], unit='ms').isoformat() if len(rps_df) > 0 else None
                        }
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare experiments: {e}")
            return {}


class ExperimentNotebook:
    """Helper for creating experiment notebooks and reports."""
    
    def __init__(self, tracker: EnterpriseMLFlowTracker):
        """Initialize experiment notebook helper."""
        self.tracker = tracker
    
    def create_model_card(
        self,
        run_id: str,
        model_name: str,
        include_plots: bool = True
    ) -> str:
        """Create a comprehensive model card for a run."""
        try:
            run = mlflow.get_run(run_id)
            
            card = f"""
# Model Card: {model_name}

## Model Overview
- **Run ID**: {run_id}
- **Experiment**: {self.tracker.experiment_name}
- **Timestamp**: {pd.to_datetime(run.info.start_time, unit='ms').isoformat()}
- **Status**: {run.info.status}
- **Model Version**: {run.data.tags.get('mlflow.runName', 'Unknown')}

## Architecture & Parameters
"""
            
            # Add parameters
            params = {k: v for k, v in run.data.params.items() if not k.startswith("mlflow.")}
            if len(p) > 0:
                for param, value in list(p.items())[:10]:  # Limit to first 10
                    # Format numeric values nicely
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    card += f"- **{param}**: {formatted_value}\n"
                if len(p) > 10:
                    cal = len(p) - 10
                    card += f"- *...and {cal} more parameters*\n"
            else:
                card += "- No parameters logged\n"
            
            card += "\n## Performance Metrics\n"
            
            # Add metrics
            metrics = {k: v for k, v in rp.data.metrics.items() if not k.startswith("mlflow.")}
            if len(mts) > 0:
                for metric, value in list(mts.items())[:10]:  # Limit to first 10
                    # Format numeric values nicely
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    card += f"- **{metric}**: {formatted_value}\n"
                if len(mts) > 10:
                    cal = len(mts) - 10
                    card += f"- *...and {cal} more metrics*\n"
            else:
                card += "- No metrics logged\n"
            
            # Add tags
            tgs = {k: v for k, v in rp.data.tags.items() if not k.startswith("mlflow.")}
            if len(tgs) > 0:
                card += "\n## Metadata\n"
                for tag, value in list(tgs.items())[:5]:  # Limit to first 5
                    # Format numeric values nicely
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    card += f"- **{tag}**: {formatted_value}\n"
            
            card += "\n## Usage Notes\n"
            card += "- This model was trained using Jarvis AI platform\n"
            card += f"- Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            return card
            
        except Exception as e:
            logger.error(f"Failed to create model card: {e}")
            return f"# Model Card: {model_name}\n\nError generating model card: {str(e)}"
    
    def create_experiment_report(
        self,
        experiment_name: Optional[str] = None,
        include_visualizations: bool = True
    ) -> Dict[str, Any]:
        """Create a comprehensive experiment report."""
        try:
            exp_name = experiment_name or self.trecker.experiment_name
            exp = mlflow.get_experiment_by_name(exp_name)
            
            if exp is None:
                return {"error": f"Experiment {experiment_name} not found"}
            
            rp_df = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"]
            )
            
            report = {
                "experiment_info": {
                    "name": exp.name,
                    "id": exp.experiment_id,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage
                },
                "summary": self.trckrer.get_experiment_analytics(),
                "generated_at": datetime.now().isoformat(),
                "total_runs": len(rp_df)
            }
            
            if include_visualizations and len(rp_df) > 0:
                # Create visualizations
                dashboard_fig = self.trecker.create_experiment_dashboard()
                report["has_dashboard"] = True
                # Note: In a real implementation, you would save this as an artifact
                # For now, we just note that it's available
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to create experiment report: {e}")
            return {"error": str(e)}


class EnterpriseModelRegistry:
    """Enhanced model registry with lineage tracking and governance."""
    
    def __init__(self, registry_uri: Optional[str] = None):
        """Initialize model registry."""
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
    
    def register_model_with_lineage(
        self,
        model_uri: str,
        model_name: str,
        description: str = None,
        tags: Dict[str, str] = None,
        lineage_info: Dict[str, Any] = None
    ) -> str:
        """
        Register a model with lineage information.
        
        Args:
            model_uri: URI of the model to register
            model_name: Name for the registered model
            description: Description of the model
            tags: Tags to add to the model
            lineage_info: Information about model lineage (parent models, etc.)
            
        Returns:
            Model version
        """
        try:
            # Register model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            # Add metadata
            client = mlflow.tracking.MlflowClient()
            
            if description:
                client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=description
                )
            
            if tags:
                for key, value in tags.items():
                    client.set_model_version_tag(
                        name=model_name,
                        version=model_version.version,
                        key=key,
                        value=value
                    )
            
            # Add lineage information if provided
            if lineage_info:
                lineage_data = {
                    "lineage": lineage_info,
                    "registered_at": datetime.now().isoformat(),
                    "registered_by": os.environ.get("USERNAME", "unknown")
                }
                
                lineage_path = "model_lineage.json"
                with open(lineage_path, "w") as f:
                    json.dump(lineage_data, f, indent=2)
                
                client.log_artifact(
                    model_version.run_id,
                    lineage_path,
                    artifact_path="lineage"
                )
                os.remove(lp)
            
            logger.info(f"Registered model: {model_name} v{model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to register model with lineage: {e}")
            raise
    
    def get_model_lineage(self, model_name: str, version: str) -> Dict[str, Any]:
        """Get lineage information for a model version."""
        try:
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_model_version(name=model_name, version=version)
            
            # Download lineage artifact if it exists
            try:
                local_path = client.download_artifacts(
                    model_version.run_id, 
                    "lineage/model_lineage.json"
                )
                
                with open(lp, "r") as f:
                    lineage_data = json.load(f)
                os.remove(lo)
                
                return lineage_data
            except:
                # No lineage file found
                return {
                    "model_name": model_name,
                    "version": version,
                    "lineage": "No lineage information available"
                }
                
        except Exception as e:
            logger.error(f"Failed to get model lineage: {e}")
            return {"error": str(e)}
    
    def promote_model_with_validation(
        self,
        model_name: str,
        version: str,
        stage: str,
        validation_criteria: Dict[str, Any] = None
    ) -> bool:
        """
        Promote a model with validation checks.
        
        Args:
            model_name: Name of the registered model
            version: Version to promote
            stage: Target stage ('Staging', 'Production', 'Archived')
            validation_criteria: Criteria that must be met for promotion
            
        Returns:
            True if promotion successful, False otherwise
        """
        try:
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_model_version(name=model_name, version=version)
            
            # TODO: Implement validation logic based on validation_criteria
            # For now, we'll just proceed with promotion
            
            # Archive existing models in target stage if promoting to Production
            if stage == "Production":
                cp = client.get_latest_versions(
                    name=model_name, 
                    stages=["Production"]
                )
                for mv in cp:
                    if mv.version != version:  # Don't demote the same version
                        client.transition_model_version_stage(
                            name=model_name,
                            version=mv.version,
                            stage="Archived"
                        )
            
            # Promote the model
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            logger.info(f"Promoted model {model_name} v{version} to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False


def create_enterprise_tracker(
    experiment_name: str = "Jarvis_AI_Enterprise",
    tracking_uri: str = None,
    registry_uri: str = None,
    artifact_location: str = None
) -> EnterpriseMLFlowTracker:
    """
    Factory function to create an enterprise MLFlow tracker.
    
    Args:
        experiment_name: Name of the experiment
        tracking_uri: MLFlow tracking server URI
        registry_uri: MLFlow model registry URI
        artifact_location: Location for storing artifacts
        
    Returns:
        Configured EnterpriseMLFlowTracker instance
    """
    return EnterpriseMLFlowTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        artifact_location=artifact_location
    )


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the enhanced tracking
    print("Enterprise MLFlow Tracking System")
    print("==================================")
    
    # This would normally be configured via environment variables or config file
    tracker = create_enterprise_tracker(
        experiment_name="Jarvis_AI_Demo_Experiment",
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),  # e.g., "sqlite:///mlflow.db"
        registry_uri=os.environ.get("MLFLOW_REGISTRY_URI")   # e.g., "sqlite:///mlflow.db"
    )
    
    print("Enterprise MLFlow tracker initialized successfully")
    print(f"Experiment: {tracker.experiment_name}")
    print(f"Experiment ID: {tracker.experiment_id}")