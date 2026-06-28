"""
JarvisAI Model Registry Integration
MLflow-based model versioning, registration, and A/B testing
"""

import mlflow
import mlflow.pyfunc
import os
from typing import Any, Dict, Optional
from datetime import datetime

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def register_model(model_path: str, model_name: str, description: str = None, tags: Dict[str, str] = None) -> str:
    """Register a model in MLflow Model Registry"""
    result = mlflow.register_model(model_uri=model_path, name=model_name)
    if description or tags:
        client = mlflow.tracking.MlflowClient()
        client.update_registered_model(
            name=model_name,
            description=description or ""
        )
        if tags:
            client.set_registered_model_tag(model_name, "created_by", "jarvisai")
            for k, v in tags.items():
                client.set_registered_model_tag(model_name, k, v)
    return result.version


def get_latest_model_version(model_name: str) -> Optional[str]:
    """Get the latest version of a registered model"""
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production", "Staging", "None"])
    if versions:
        return versions[0].version
    return None


def load_model(model_name: str, stage: str = "Production") -> Any:
    """Load a model from the registry by name and stage"""
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.pyfunc.load_model(model_uri)


def list_models() -> Dict[str, Any]:
    """List all registered models"""
    client = mlflow.tracking.MlflowClient()
    models = client.search_registered_models()
    return {m.name: [v.version for v in m.latest_versions] for m in models}


def transition_model_version(model_name: str, version: str, stage: str):
    """Transition a model version to a new stage (e.g., Staging, Production)"""
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True
    )


def ab_test_models(model_name: str, version_a: str, version_b: str, test_data, metric_fn) -> Dict[str, float]:
    """A/B test two model versions and compare metrics"""
    model_a = mlflow.pyfunc.load_model(f"models:/{model_name}/{version_a}")
    model_b = mlflow.pyfunc.load_model(f"models:/{model_name}/{version_b}")
    
    preds_a = model_a.predict(test_data)
    preds_b = model_b.predict(test_data)
    
    metric_a = metric_fn(test_data, preds_a)
    metric_b = metric_fn(test_data, preds_b)
    
    return {"version_a": metric_a, "version_b": metric_b}


if __name__ == "__main__":
    print("📦 JarvisAI Model Registry Test")
    print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print("Registered models:")
    print(list_models())
