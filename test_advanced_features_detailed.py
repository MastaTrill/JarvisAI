#!/usr/bin/env python3
"""
Test script for the new advanced features.
"""

import pytest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_advanced_data_pipeline():
    """Test the advanced data pipeline."""
    try:
        from src.data.advanced_data_pipeline import AdvancedDataPipeline, CSVConnector, DataValidator
        
        # Create test data
        test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Add some missing values and duplicates
        test_data.loc[5:10, 'feature1'] = np.nan
        test_data = pd.concat([test_data, test_data.iloc[:5]], ignore_index=True)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Test CSV connector
            connector = CSVConnector()
            assert connector.connect({'filepath': temp_file})
            
            fetched_data = connector.fetch_data()
            assert not fetched_data.empty
            assert len(fetched_data.columns) == 3
            
            # Test data validation
            validator = DataValidator()
            report = validator.validate_data_quality(fetched_data)
            
            assert 'quality_score' in report
            assert 'missing_values' in report
            assert 'duplicates' in report
            assert report['duplicates'] > 0  # We added duplicates
            
            # Test suggestions
            suggestions = validator.suggest_fixes(report)
            assert isinstance(suggestions, list)
            
            print("✅ Advanced data pipeline tests passed")
            return True
            
        finally:
            # Clean up
            Path(temp_file).unlink()
            
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Advanced data pipeline test failed: {e}")
        return False

def test_hyperparameter_optimization():
    """Test hyperparameter optimization."""
    try:
        from src.training.advanced_training_system import GridSearchOptimizer, RandomSearchOptimizer
        
        # Test Grid Search
        parameter_grid = {
            'learning_rate': [0.001, 0.01],
            'batch_size': [16, 32]
        }
        
        grid_optimizer = GridSearchOptimizer(parameter_grid)
        assert len(grid_optimizer.parameter_combinations) == 4
        
        # Test parameter suggestion
        params1 = grid_optimizer.suggest_parameters(0)
        assert 'learning_rate' in params1
        assert 'batch_size' in params1
        
        # Test result updating
        grid_optimizer.update_results(params1, 0.85)
        assert len(grid_optimizer.results) == 1
        
        best_params = grid_optimizer.get_best_parameters()
        assert best_params == params1
        
        # Test Random Search
        parameter_space = {
            'learning_rate': {'type': 'log_uniform', 'low': 0.001, 'high': 0.1},
            'batch_size': {'type': 'choice', 'choices': [16, 32, 64]}
        }
        
        random_optimizer = RandomSearchOptimizer(parameter_space, max_trials=5)
        
        # Test parameter suggestion
        params2 = random_optimizer.suggest_parameters(0)
        assert 'learning_rate' in params2
        assert 'batch_size' in params2
        assert params2['batch_size'] in [16, 32, 64]
        
        print("✅ Hyperparameter optimization tests passed")
        return True
        
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Hyperparameter optimization test failed: {e}")
        return False

def test_experiment_tracking():
    """Test experiment tracking system."""
    try:
        from src.training.advanced_training_system import ExperimentTracker, ExperimentConfig
        
        # Create temporary directory for experiments
        temp_dir = tempfile.mkdtemp()
        
        try:
            tracker = ExperimentTracker(temp_dir)
            
            # Test experiment creation
            config = ExperimentConfig(
                experiment_name="test_experiment",
                model_config={},
                training_config={},
                data_config={},
                optimization_config={},
                metadata={}
            )
            
            experiment_id = tracker.start_experiment(config)
            assert experiment_id is not None
            assert "test_experiment" in experiment_id
            
            # Test experiment retrieval
            experiments = tracker.get_all_experiments()
            assert len(experiments) == 1
            assert experiments[0]['experiment_name'] == "test_experiment"
            
            print("✅ Experiment tracking tests passed")
            return True
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir)
            
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Experiment tracking test failed: {e}")
        return False

def test_data_versioning():
    """Test data versioning system."""
    try:
        from src.data.advanced_data_pipeline import DataVersioning
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            versioning = DataVersioning(temp_dir)
            
            # Create test data
            test_data = pd.DataFrame({
                'x': [1, 2, 3, 4, 5],
                'y': [2, 4, 6, 8, 10]
            })
            
            # Test saving version
            version_id = versioning.save_version(test_data, "test_data", "Test dataset")
            assert version_id is not None
            assert "test_data" in version_id
            
            # Test loading version
            loaded_data = versioning.load_version(version_id)
            assert loaded_data is not None
            assert loaded_data.equals(test_data)
            
            # Test transformation tracking
            versioning.track_transformation(
                version_id, "processed_version", "data_cleaning", {"method": "test"}
            )
            
            # Test lineage graph
            lineage = versioning.get_lineage_graph()
            assert "versions" in lineage
            assert "transformations" in lineage
            assert len(lineage["versions"]) == 1
            assert len(lineage["transformations"]) == 1
            
            print("✅ Data versioning tests passed")
            return True
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir)
            
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Data versioning test failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering capabilities."""
    try:
        from src.data.advanced_data_pipeline import FeatureEngineer
        
        # Create test data
        test_data = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [2, 4, 6, 8, 10],
            'category': ['A', 'B', 'A', 'C', 'B'],
            'date': pd.date_range('2023-01-01', periods=5)
        })
        
        engineer = FeatureEngineer()
        
        # Test polynomial features
        poly_data = engineer.create_polynomial_features(test_data, ['numeric1', 'numeric2'], degree=2)
        assert 'numeric1_pow2' in poly_data.columns
        assert 'numeric2_pow2' in poly_data.columns
        
        # Test interaction features
        interact_data = engineer.create_interaction_features(test_data, ['numeric1', 'numeric2'])
        assert 'numeric1_x_numeric2' in interact_data.columns
        assert 'numeric1_div_numeric2' in interact_data.columns
        
        # Test categorical features
        cat_data = engineer.create_categorical_features(test_data, ['category'])
        assert any('category_' in col for col in cat_data.columns)
        
        # Test temporal features
        temporal_data = engineer.create_temporal_features(test_data, ['date'])
        assert 'date_year' in temporal_data.columns
        assert 'date_month' in temporal_data.columns
        
        print("✅ Feature engineering tests passed")
        return True
        
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        return False

def run_all_tests():
    """Run all advanced feature tests."""
    print("🧪 RUNNING ADVANCED FEATURES TESTS")
    print("="*50)
    
    tests = [
        ("Data Pipeline", test_advanced_data_pipeline),
        ("Hyperparameter Optimization", test_hyperparameter_optimization),
        ("Experiment Tracking", test_experiment_tracking),
        ("Data Versioning", test_data_versioning),
        ("Feature Engineering", test_feature_engineering)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Testing {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
    
    print("\n" + "="*50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All advanced features are working correctly!")
        return True
    else:
        print("⚠️ Some tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    run_all_tests()
