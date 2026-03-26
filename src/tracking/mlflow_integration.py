"""
MLFlow Integration for Advanced Experiment Tracking.

This module provides MLFlow-based experiment tracking with enhanced features:
- Model versioning and registration
- Artifact logging (datasets, models, plots)
- Hyperparameter tracking
- Metrics visualization
- Model deployment utilities
"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import numpy as np
import pandas as pd
import pickle
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logger = logging.getLogger(__name__)


class MLFlowTracker:
    """Advanced MLFlow-based experiment tracking."""
    
    def __init__(
        self,
        experiment_name: str = "Jarvis_AI_Experiments",
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None
    ):
        """
        Initialize MLFlow tracker.
        
        Args:
            experiment_name: Name of the MLFlow experiment
            tracking_uri: MLFlow tracking server URI
            registry_uri: MLFlow model registry URI
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
                self.experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"✅ Created new MLFlow experiment: {experiment_name}")
            else:
                self.experiment_id = experiment.experiment_id
                logger.info(f"✅ Using existing MLFlow experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"❌ Failed to set up MLFlow experiment: {e}")
            raise
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ) -> str:
        """
        Start a new MLFlow run.
        
        Args:
            run_name: Name for the run
            tags: Dictionary of tags to add
            nested: Whether this is a nested run
            
        Returns:
            Run ID
        """
        try:
            run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                nested=nested
            )
            
            # Add default tags
            default_tags = {
                "framework": "jarvis_ai",
                "created_at": datetime.now().isoformat(),
                "version": "1.0.0"
            }
            
            if tags:
                default_tags.update(tags)
                
            mlflow.set_tags(default_tags)
            
            logger.info(f"🚀 Started MLFlow run: {run.info.run_id}")
            return run.info.run_id
            
        except Exception as e:
            logger.error(f"❌ Failed to start MLFlow run: {e}")
            raise
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLFlow."""
        try:
            # Convert complex objects to strings
            processed_params = {}
            for key, value in params.items():
                if isinstance(value, (dict, list)):
                    processed_params[key] = json.dumps(value)
                elif isinstance(value, (np.ndarray,)):
                    processed_params[key] = str(value.shape)
                else:
                    processed_params[key] = value
            
            mlflow.log_params(processed_params)
            logger.info(f"📊 Logged {len(processed_params)} parameters")
            
        except Exception as e:
            logger.error(f"❌ Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLFlow."""
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.info(f"📈 Logged {len(metrics)} metrics" + 
                       (f" at step {step}" if step else ""))
            
        except Exception as e:
            logger.error(f"❌ Failed to log metrics: {e}")
    
    def log_model(
        self,
        model: Any,
        model_name: str,
        flavor: str = "sklearn",
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
        registered_model_name: Optional[str] = None
    ):
        """
        Log a model to MLFlow.
        
        Args:
            model: The model object
            model_name: Name for the model artifact
            flavor: MLFlow flavor ('sklearn', 'pytorch', 'custom')
            signature: Model signature
            input_example: Example input
            registered_model_name: Name for model registry
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
            else:
                # Custom model - save as pickle
                with mlflow.start_run(nested=True):
                    mlflow.log_artifact(model, model_name)
            
            logger.info(f"🤖 Logged model: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to log model: {e}")
    
    def log_dataset(
        self,
        dataset: Union[pd.DataFrame, np.ndarray],
        name: str,
        format: str = "csv"
    ):
        """
        Log a dataset to MLFlow.
        
        Args:
            dataset: Dataset to log
            name: Name for the dataset
            format: Format to save ('csv', 'json', 'pickle')
        """
        try:
            temp_path = Path(f"temp_{name}.{format}")
            
            if isinstance(dataset, pd.DataFrame):
                if format == "csv":
                    dataset.to_csv(temp_path, index=False)
                elif format == "json":
                    dataset.to_json(temp_path, orient="records")
                elif format == "pickle":
                    dataset.to_pickle(temp_path)
            elif isinstance(dataset, np.ndarray):
                if format == "csv":
                    np.savetxt(temp_path, dataset, delimiter=",")
                elif format == "pickle":
                    with open(temp_path, "wb") as f:
                        pickle.dump(dataset, f)
            
            mlflow.log_artifact(str(temp_path), "datasets")
            temp_path.unlink()  # Clean up temp file
            
            logger.info(f"📁 Logged dataset: {name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to log dataset: {e}")
    
    def log_plot(
        self,
        figure: plt.Figure,
        name: str,
        close_figure: bool = True
    ):
        """
        Log a matplotlib plot to MLFlow.
        
        Args:
            figure: Matplotlib figure
            name: Name for the plot
            close_figure: Whether to close the figure after logging
        """
        try:
            temp_path = f"temp_{name}.png"
            figure.savefig(temp_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(temp_path, "plots")
            
            # Clean up
            Path(temp_path).unlink()
            if close_figure:
                plt.close(figure)
            
            logger.info(f"📊 Logged plot: {name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to log plot: {e}")
    
    def log_training_progress(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log training progress for an epoch.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        try:
            metrics = {"train_loss": train_loss}
            
            if val_loss is not None:
                metrics["val_loss"] = val_loss
            
            if train_metrics:
                for key, value in train_metrics.items():
                    metrics[f"train_{key}"] = value
            
            if val_metrics:
                for key, value in val_metrics.items():
                    metrics[f"val_{key}"] = value
            
            mlflow.log_metrics(metrics, step=epoch)
            
        except Exception as e:
            logger.error(f"❌ Failed to log training progress: {e}")
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLFlow run.
        
        Args:
            status: Run status ('FINISHED', 'FAILED', 'KILLED')
        """
        try:
            mlflow.end_run(status=status)
            logger.info(f"✅ Ended MLFlow run with status: {status}")
            
        except Exception as e:
            logger.error(f"❌ Failed to end MLFlow run: {e}")
    
    def get_experiment_runs(
        self,
        max_results: int = 100,
        order_by: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get runs from the current experiment.
        
        Args:
            max_results: Maximum number of runs to return
            order_by: List of columns to order by
            
        Returns:
            DataFrame with run information
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                max_results=max_results,
                order_by=order_by
            )
            
            logger.info(f"📊 Retrieved {len(runs)} runs from experiment")
            return runs
            
        except Exception as e:
            logger.error(f"❌ Failed to get experiment runs: {e}")
            return pd.DataFrame()
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare
            
        Returns:
            DataFrame with comparison data
        """
        try:
            runs_data = []
            
            for run_id in run_ids:
                run = mlflow.get_run(run_id)
                run_data = {
                    "run_id": run_id,
                    "run_name": run.data.tags.get("mlflow.runName", ""),
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time
                }
                
                # Add parameters
                for key, value in run.data.params.items():
                    run_data[f"param_{key}"] = value
                
                # Add metrics
                for key, value in run.data.metrics.items():
                    if metrics is None or key in metrics:
                        run_data[f"metric_{key}"] = value
                
                runs_data.append(run_data)
            
            comparison_df = pd.DataFrame(runs_data)
            logger.info(f"📊 Compared {len(run_ids)} runs")
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"❌ Failed to compare runs: {e}")
            return pd.DataFrame()
    
    def create_experiment_dashboard(self) -> plt.Figure:
        """
        Create a dashboard with experiment overview.
        
        Returns:
            Matplotlib figure with dashboard
        """
        try:
            runs_df = self.get_experiment_runs()
            
            if runs_df.empty:
                logger.warning("⚠️ No runs found for dashboard")
                return plt.figure()
            
            # Create dashboard
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f"Experiment Dashboard: {self.experiment_name}", fontsize=16)
            
            # Plot 1: Run status distribution
            if 'status' in runs_df.columns:
                status_counts = runs_df['status'].value_counts()
                axes[0, 0].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
                axes[0, 0].set_title("Run Status Distribution")
            
            # Plot 2: Runs over time
            if 'start_time' in runs_df.columns:
                runs_df['start_date'] = pd.to_datetime(runs_df['start_time'], unit='ms').dt.date
                daily_runs = runs_df.groupby('start_date').size()
                axes[0, 1].plot(daily_runs.index, daily_runs.values, marker='o')
                axes[0, 1].set_title("Runs Over Time")
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Metric distribution (if available)
            metric_cols = [col for col in runs_df.columns if col.startswith('metrics.')]
            if metric_cols:
                metric_col = metric_cols[0]  # Use first metric
                axes[1, 0].hist(runs_df[metric_col].dropna(), bins=20, alpha=0.7)
                axes[1, 0].set_title(f"Distribution: {metric_col}")
            
            # Plot 4: Parameter correlation (if available)
            param_cols = [col for col in runs_df.columns if col.startswith('params.')]
            numeric_params = []
            for col in param_cols:
                try:
                    runs_df[col] = pd.to_numeric(runs_df[col])
                    numeric_params.append(col)
                except:
                    continue
            
            if len(numeric_params) >= 2:
                corr_matrix = runs_df[numeric_params].corr()
                sns.heatmap(corr_matrix, annot=True, ax=axes[1, 1], cmap='coolwarm')
                axes[1, 1].set_title("Parameter Correlation")
            
            plt.tight_layout()
            logger.info("📊 Created experiment dashboard")
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Failed to create dashboard: {e}")
            return plt.figure()


class ModelRegistry:
    """Model registry utilities for MLFlow."""
    
    def __init__(self, registry_uri: Optional[str] = None):
        """
        Initialize model registry.
        
        Args:
            registry_uri: MLFlow model registry URI
        """
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
    
    def register_model(
        self,
        model_uri: str,
        model_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register a model in the model registry.
        
        Args:
            model_uri: URI of the model to register
            model_name: Name for the registered model
            description: Description of the model
            tags: Tags to add to the model
            
        Returns:
            Model version
        """
        try:
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            # Add description and tags if provided
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
            
            logger.info(f"🏷️ Registered model: {model_name} v{model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"❌ Failed to register model: {e}")
            raise
    
    def promote_model(
        self,
        model_name: str,
        version: str,
        stage: str = "Production"
    ):
        """
        Promote a model to a specific stage.
        
        Args:
            model_name: Name of the registered model
            version: Version to promote
            stage: Stage to promote to ('Staging', 'Production', 'Archived')
        """
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            logger.info(f"🚀 Promoted model {model_name} v{version} to {stage}")
            
        except Exception as e:
            logger.error(f"❌ Failed to promote model: {e}")
