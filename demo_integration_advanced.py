"""
Advanced Feature Integration Demo.

This script demonstrates the integration of all new advanced features:
- MLFlow experiment tracking
- Model validation framework
- Data augmentation pipeline
- End-to-end ML workflow
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_demo_dataset():
    """Create a comprehensive demo dataset for testing."""
    logger.info("🔄 Creating comprehensive demo dataset...")
    
    np.random.seed(42)
    
    # Create a more complex dataset with different patterns
    n_samples = 1000
    n_features = 20
    
    # Generate base features
    X = np.random.randn(n_samples, n_features)
    
    # Add some structure and patterns
    # Feature interactions
    X[:, 1] = X[:, 0] * 0.5 + np.random.randn(n_samples) * 0.3
    X[:, 2] = X[:, 0] ** 2 + np.random.randn(n_samples) * 0.2
    
    # Polynomial features
    X[:, 3] = X[:, 0] * X[:, 1] + np.random.randn(n_samples) * 0.1
    
    # Seasonal pattern (for time series demo)
    t = np.linspace(0, 4*np.pi, n_samples)
    X[:, 4] = np.sin(t) + 0.1 * np.random.randn(n_samples)
    X[:, 5] = np.cos(t) + 0.1 * np.random.randn(n_samples)
    
    # Create targets with different complexity
    # Classification target (imbalanced)
    linear_combination = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2] + 
                         0.2 * X[:, 3] + 0.1 * X[:, 4])
    y_class = (linear_combination > np.percentile(linear_combination, 70)).astype(int)
    
    # Regression target
    y_reg = (2 * X[:, 0] + 1.5 * X[:, 1] + X[:, 2] + 
             0.5 * X[:, 3] + 0.3 * X[:, 4] + np.random.randn(n_samples) * 0.5)
    
    # Add some sensitive features for bias analysis
    sensitive_feature = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    # Create bias in the target based on sensitive feature
    bias_mask = sensitive_feature == 1
    y_class[bias_mask] = np.random.choice([0, 1], size=np.sum(bias_mask), p=[0.4, 0.6])
    
    logger.info(f"✅ Created dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"📊 Class distribution: {np.bincount(y_class)}")
    logger.info(f"🎯 Sensitive feature distribution: {np.bincount(sensitive_feature)}")
    
    return X, y_class, y_reg, sensitive_feature

def demonstrate_mlflow_tracking():
    """Demonstrate MLFlow experiment tracking."""
    logger.info("\n" + "="*60)
    logger.info("📊 MLFLOW EXPERIMENT TRACKING DEMONSTRATION")
    logger.info("="*60)
    
    try:
        from src.tracking.mlflow_integration import MLFlowTracker
        
        # Initialize MLFlow tracker
        tracker = MLFlowTracker(
            experiment_name="Advanced_Jarvis_Demo",
            tracking_uri="file:./mlruns"
        )
        
        # Start a run
        run_id = tracker.start_run(
            run_name=f"demo_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={
                "demo_type": "advanced_features",
                "model_type": "neural_network",
                "dataset": "synthetic"
            }
        )
        
        # Log parameters
        params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "dropout_rate": 0.3,
            "optimizer": "adam",
            "activation": "relu"
        }
        tracker.log_params(params)
        
        # Simulate training metrics
        for epoch in range(10):
            train_loss = 1.0 - (epoch * 0.05) + np.random.normal(0, 0.01)
            val_loss = 1.1 - (epoch * 0.04) + np.random.normal(0, 0.02)
            
            tracker.log_training_progress(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics={"accuracy": 0.5 + (epoch * 0.03)},
                val_metrics={"accuracy": 0.4 + (epoch * 0.025)}
            )
        
        # Create and log a dummy dataset
        demo_data = pd.DataFrame(np.random.randn(100, 5), 
                                columns=[f"feature_{i}" for i in range(5)])
        tracker.log_dataset(demo_data, "demo_dataset", format="csv")
        
        # Get experiment overview
        runs_df = tracker.get_experiment_runs(max_results=10)
        logger.info(f"📈 Retrieved {len(runs_df)} runs from experiment")
        
        # Create dashboard
        dashboard_fig = tracker.create_experiment_dashboard()
        tracker.log_plot(dashboard_fig, "experiment_dashboard")
        
        # End run
        tracker.end_run("FINISHED")
        
        logger.info("✅ MLFlow tracking demonstration completed")
        return tracker
        
    except ImportError as e:
        logger.error(f"❌ MLFlow not available: {e}")
        logger.info("💡 To enable MLFlow tracking, install: pip install mlflow")
        return None
    except Exception as e:
        logger.error(f"❌ MLFlow demonstration failed: {e}")
        return None

def demonstrate_model_validation():
    """Demonstrate advanced model validation."""
    logger.info("\n" + "="*60)
    logger.info("🔍 MODEL VALIDATION DEMONSTRATION")
    logger.info("="*60)
    
    try:
        from src.validation.model_validator import ModelValidator
        from src.models.advanced_neural_network import AdvancedNeuralNetwork
        
        # Create demo data
        X, y_class, y_reg, sensitive_feature = create_demo_dataset()
        
        # Create and train a simple model for validation
        model = AdvancedNeuralNetwork(
            input_size=X.shape[1],
            hidden_sizes=[32, 16],
            output_size=2,  # Binary classification
            activation='relu',
            dropout_rate=0.2,
            learning_rate=0.01
        )
        
        # Quick training for demo
        logger.info("🏃 Quick model training for validation demo...")
        model.fit(X, y_class, epochs=50, batch_size=32, verbose=False)
        
        # Initialize validator
        validator = ModelValidator(task_type="classification")
        
        # Perform cross-validation
        logger.info("🔄 Performing cross-validation...")
        cv_results = validator.cross_validate(
            model=model,
            X=X,
            y=y_class,
            cv_strategy="stratified",
            n_splits=5,
            scoring_metrics=["accuracy", "precision", "recall", "f1"]
        )
        
        logger.info("📊 Cross-validation results:")
        for metric, results in cv_results.items():
            logger.info(f"  {metric}: {results['mean']:.4f} ± {results['std']:.4f}")
        
        # Bias and fairness analysis
        logger.info("⚖️ Performing bias analysis...")
        bias_results = validator.bias_fairness_analysis(
            model=model,
            X=X,
            y=y_class,
            sensitive_features=sensitive_feature,
            feature_names=["sensitive_attr"]
        )
        
        logger.info("📊 Bias analysis results:")
        for violation, value in bias_results.get("fairness_violations", {}).items():
            logger.info(f"  {violation}: {value:.4f}")
        
        # Robustness testing
        logger.info("🛡️ Performing robustness testing...")
        robustness_results = validator.robustness_testing(
            model=model,
            X=X[:100],  # Use subset for speed
            y=y_class[:100],
            noise_levels=[0.01, 0.05, 0.1],
            perturbation_types=["gaussian", "uniform"]
        )
        
        logger.info("📊 Robustness test results:")
        for perturbation_type in robustness_results:
            if perturbation_type == "baseline":
                continue
            logger.info(f"  {perturbation_type} perturbation:")
            for level, results in robustness_results[perturbation_type].items():
                degradation = list(results["degradation"].values())[0]
                logger.info(f"    Level {level}: {degradation:.4f} degradation")
        
        # Generate validation report
        logger.info("📝 Generating validation report...")
        report_summary = validator.generate_validation_report("validation_reports/demo")
        
        logger.info("✅ Model validation demonstration completed")
        return validator
        
    except Exception as e:
        logger.error(f"❌ Model validation demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def demonstrate_data_augmentation():
    """Demonstrate data augmentation pipeline."""
    logger.info("\n" + "="*60)
    logger.info("🎭 DATA AUGMENTATION DEMONSTRATION")
    logger.info("="*60)
    
    try:
        from src.augmentation.data_augmenter import (
            DataAugmentationPipeline, NoiseAugmenter, SMOTEAugmenter,
            FeatureMixupAugmenter, create_augmentation_config
        )
        
        # Create demo data
        X, y_class, y_reg, sensitive_feature = create_demo_dataset()
        
        # Demonstrate individual augmenters
        logger.info("🔊 Testing Noise Augmentation...")
        noise_augmenter = NoiseAugmenter(noise_type="gaussian", noise_level=0.1)
        X_noise, y_noise = noise_augmenter.augment(X[:100], y_class[:100], augmentation_factor=0.5)
        logger.info(f"  Original: {X[:100].shape}, Augmented: {X_noise.shape}")
        
        logger.info("🎯 Testing SMOTE Augmentation...")
        smote_augmenter = SMOTEAugmenter(k_neighbors=3)
        X_smote, y_smote = smote_augmenter.augment(X[:200], y_class[:200])
        logger.info(f"  Original: {X[:200].shape}, Balanced: {X_smote.shape}")
        logger.info(f"  Original class dist: {np.bincount(y_class[:200])}")
        logger.info(f"  Balanced class dist: {np.bincount(y_smote)}")
        
        logger.info("🎭 Testing Mixup Augmentation...")
        mixup_augmenter = FeatureMixupAugmenter(alpha=0.2)
        X_mixup, y_mixup = mixup_augmenter.augment(X[:100], y_class[:100], augmentation_factor=0.3)
        logger.info(f"  Original: {X[:100].shape}, Mixup: {X_mixup.shape}")
        
        # Demonstrate augmentation pipeline
        logger.info("🔗 Testing Augmentation Pipeline...")
        pipeline = create_augmentation_config(
            task_type="classification",
            data_type="tabular",
            imbalanced=True
        )
        
        X_pipeline, y_pipeline = pipeline.augment(X[:300], y_class[:300])
        logger.info(f"  Pipeline result: {X[:300].shape} → {X_pipeline.shape}")
        
        # Get pipeline summary
        summary = pipeline.get_augmentation_summary()
        logger.info("📋 Pipeline configuration:")
        for augmenter_info in summary["augmenters"]:
            logger.info(f"  - {augmenter_info['name']} (p={augmenter_info['probability']})")
        
        logger.info("✅ Data augmentation demonstration completed")
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ Data augmentation demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def demonstrate_end_to_end_workflow():
    """Demonstrate complete end-to-end ML workflow."""
    logger.info("\n" + "="*60)
    logger.info("🚀 END-TO-END ML WORKFLOW DEMONSTRATION")
    logger.info("="*60)
    
    try:
        # Import all required modules
        from src.data.advanced_data_pipeline import AdvancedDataPipeline
        from src.augmentation.data_augmenter import create_augmentation_config
        from src.models.advanced_neural_network import AdvancedNeuralNetwork
        from src.training.advanced_training_system import AdvancedTrainingSystem
        from src.validation.model_validator import ModelValidator
        
        # 1. Data Pipeline
        logger.info("📊 Step 1: Data Pipeline")
        X, y_class, y_reg, sensitive_feature = create_demo_dataset()
        
        # Save demo data
        demo_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        demo_df['target'] = y_class
        demo_df['sensitive'] = sensitive_feature
        
        data_path = Path("data/processed/end_to_end_demo.csv")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        demo_df.to_csv(data_path, index=False)
        
        # Initialize data pipeline
        pipeline = AdvancedDataPipeline()
        source_config = {
            'type': 'csv',
            'filepath': str(data_path),
            'encoding': 'utf-8'
        }
        
        if pipeline.connect_to_source(source_config):
            df = pipeline.fetch_data()
            
            # Data validation
            quality_report = pipeline.validate_data(df)
            logger.info(f"  Data quality score: {quality_report['overall_score']:.1f}/100")
            
            # Feature engineering
            df_enhanced = pipeline.engineer_features(
                df,
                polynomial_degree=2,
                create_interactions=True,
                feature_selection=True
            )
            logger.info(f"  Features: {df.shape[1]} → {df_enhanced.shape[1]}")
        
        # 2. Data Augmentation
        logger.info("🎭 Step 2: Data Augmentation")
        augmentation_pipeline = create_augmentation_config(
            task_type="classification",
            data_type="tabular",
            imbalanced=True
        )
        
        X_aug, y_aug = augmentation_pipeline.augment(X, y_class)
        logger.info(f"  Augmented dataset: {X.shape} → {X_aug.shape}")
        
        # 3. Model Training
        logger.info("🧠 Step 3: Model Training")
        model = AdvancedNeuralNetwork(
            input_size=X_aug.shape[1],
            hidden_sizes=[64, 32, 16],
            output_size=2,
            activation='relu',
            dropout_rate=0.3,
            l1_reg=0.01,
            l2_reg=0.01,
            optimizer='adam',
            learning_rate=0.001
        )
        
        # Train model
        history = model.fit(
            X_aug, y_aug,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=True
        )
        logger.info("  ✅ Model training completed")
        
        # 4. Model Validation
        logger.info("🔍 Step 4: Model Validation")
        validator = ModelValidator(task_type="classification")
        
        # Cross-validation
        cv_results = validator.cross_validate(
            model=model,
            X=X,  # Use original data for fair evaluation
            y=y_class,
            cv_strategy="stratified",
            n_splits=3,  # Reduced for demo speed
            scoring_metrics=["accuracy", "f1"]
        )
        
        best_metric = max(cv_results.keys(), key=lambda m: cv_results[m]["mean"])
        logger.info(f"  Best metric: {best_metric} = {cv_results[best_metric]['mean']:.4f}")
        
        # 5. Experiment Tracking (if available)
        logger.info("📊 Step 5: Experiment Tracking")
        try:
            from src.tracking.mlflow_integration import MLFlowTracker
            
            tracker = MLFlowTracker(experiment_name="End_to_End_Demo")
            run_id = tracker.start_run(run_name="complete_workflow")
            
            # Log everything
            tracker.log_params({
                "dataset_size": len(X),
                "augmented_size": len(X_aug),
                "model_architecture": str(model.hidden_sizes),
                "cv_strategy": "stratified",
                "best_cv_score": cv_results[best_metric]["mean"]
            })
            
            tracker.log_metrics({
                f"cv_{metric}": results["mean"] 
                for metric, results in cv_results.items()
            })
            
            tracker.end_run("FINISHED")
            logger.info("  ✅ Experiment logged to MLFlow")
            
        except ImportError:
            logger.info("  ⚠️ MLFlow not available, skipping experiment tracking")
        
        # 6. Generate Reports
        logger.info("📝 Step 6: Generate Reports")
        report_dir = Path("reports/end_to_end_demo")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save(str(report_dir / "trained_model.pkl"))
        
        # Save validation results
        validation_summary = validator.generate_validation_report(str(report_dir / "validation"))
        
        # Create workflow summary
        workflow_summary = {
            "timestamp": datetime.now().isoformat(),
            "dataset": {
                "original_size": X.shape,
                "augmented_size": X_aug.shape,
                "class_distribution": np.bincount(y_class).tolist()
            },
            "model": {
                "type": "AdvancedNeuralNetwork",
                "architecture": model.hidden_sizes,
                "parameters": sum(w.size for w in model.weights) + sum(b.size for b in model.biases)
            },
            "validation": {
                "cv_results": {k: v["mean"] for k, v in cv_results.items()},
                "best_metric": best_metric,
                "best_score": cv_results[best_metric]["mean"]
            }
        }
        
        import json
        with open(report_dir / "workflow_summary.json", "w") as f:
            json.dump(workflow_summary, f, indent=2)
        
        logger.info(f"  📁 Reports saved to: {report_dir}")
        logger.info("✅ End-to-end workflow demonstration completed successfully!")
        
        return workflow_summary
        
    except Exception as e:
        logger.error(f"❌ End-to-end workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all advanced feature demonstrations."""
    logger.info("🚀 JARVIS AI - ADVANCED FEATURES INTEGRATION DEMO")
    logger.info("=" * 80)
    logger.info("This demonstration showcases the integration of all new advanced features:")
    logger.info("✨ MLFlow experiment tracking")
    logger.info("🔍 Comprehensive model validation")
    logger.info("🎭 Advanced data augmentation")
    logger.info("🔗 End-to-end ML workflow")
    logger.info("=" * 80)
    
    results = {}
    
    # Run individual demonstrations
    try:
        results["mlflow"] = demonstrate_mlflow_tracking()
        results["validation"] = demonstrate_model_validation()
        results["augmentation"] = demonstrate_data_augmentation()
        results["end_to_end"] = demonstrate_end_to_end_workflow()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("🎉 DEMONSTRATION COMPLETE!")
        logger.info("="*60)
        
        successful_demos = [k for k, v in results.items() if v is not None]
        failed_demos = [k for k, v in results.items() if v is None]
        
        logger.info(f"✅ Successful demonstrations: {len(successful_demos)}")
        for demo in successful_demos:
            logger.info(f"   • {demo}")
        
        if failed_demos:
            logger.info(f"❌ Failed demonstrations: {len(failed_demos)}")
            for demo in failed_demos:
                logger.info(f"   • {demo}")
        
        logger.info("\n🔧 The Jarvis AI platform now includes:")
        logger.info("   • Advanced neural network architectures")
        logger.info("   • Comprehensive data processing pipelines")
        logger.info("   • Professional experiment tracking")
        logger.info("   • Robust model validation framework")
        logger.info("   • Sophisticated data augmentation")
        logger.info("   • End-to-end ML workflow automation")
        logger.info("   • Bias and fairness analysis")
        logger.info("   • Robustness testing capabilities")
        logger.info("   • Hyperparameter optimization")
        logger.info("   • Data versioning and lineage tracking")
        
        logger.info("\n🚀 Ready for production-level AI/ML workflows!")
        
    except Exception as e:
        logger.error(f"❌ Demo execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
