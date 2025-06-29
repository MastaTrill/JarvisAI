#!/usr/bin/env python3
"""
🧪 ADVANCED FEATURES VALIDATION TEST
===================================

Quick validation test for the new advanced Computer Vision and Time Series modules.
This script ensures the modules can be imported and basic functionality works.

Author: Aetheron Platform Team
Date: June 2025
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_computer_vision_module():
    """Test the advanced computer vision module"""
    try:
        print("🔍 Testing Advanced Computer Vision module...")
        
        from src.cv.advanced_computer_vision import AdvancedComputerVision
        
        # Initialize the system
        cv_system = AdvancedComputerVision()
        print("✅ Computer Vision module imported and initialized successfully")
        
        # Create sample image
        sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test object detection
        objects = cv_system.detect_objects(sample_image)
        print(f"✅ Object detection: detected {len(objects)} objects")
        
        # Test image classification
        classification = cv_system.classify_image(sample_image)
        print(f"✅ Image classification: {classification['predicted_class']}")
        
        # Test face detection
        faces = cv_system.detect_faces(sample_image)
        print(f"✅ Face detection: detected {len(faces)} faces")
        
        # Test OCR
        ocr_result = cv_system.extract_text_ocr(sample_image)
        print(f"✅ OCR: confidence {ocr_result['confidence']:.2f}")
        
        # Test style transfer
        style_result = cv_system.apply_style_transfer(sample_image, "neural")
        print(f"✅ Style transfer: {style_result['style_applied']}")
        
        # Test quality analysis
        quality = cv_system.analyze_image_quality(sample_image)
        print(f"✅ Quality analysis: score {quality['overall_score']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Computer Vision test failed: {e}")
        return False

def test_time_series_module():
    """Test the advanced time series module"""
    try:
        print("\n📈 Testing Advanced Time Series module...")
        
        from src.timeseries.advanced_time_series import AdvancedTimeSeries
        
        # Initialize the system
        ts_system = AdvancedTimeSeries()
        print("✅ Time Series module imported and initialized successfully")
        
        # Create sample time series data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        values = np.random.randn(100).cumsum() + 100
        ts_data = pd.DataFrame({'timestamp': dates, 'value': values})
        
        # Test ARIMA forecasting
        arima_forecast = ts_system.forecast_arima(ts_data, forecast_steps=10)
        print(f"✅ ARIMA forecasting: {len(arima_forecast['forecast'])} predictions")
        
        # Test exponential smoothing
        es_forecast = ts_system.forecast_exponential_smoothing(ts_data, forecast_steps=10)
        print(f"✅ Exponential smoothing: method {es_forecast['method']}")
        
        # Test LSTM forecasting
        lstm_forecast = ts_system.forecast_lstm(ts_data, forecast_steps=10)
        print(f"✅ LSTM forecasting: {lstm_forecast['model_architecture']}")
        
        # Test anomaly detection
        anomalies = ts_system.detect_anomalies(ts_data)
        print(f"✅ Anomaly detection: {anomalies['num_anomalies']} anomalies found")
        
        # Test trend analysis
        trend_analysis = ts_system.analyze_trend_seasonality(ts_data)
        print(f"✅ Trend analysis: {trend_analysis['trend_direction']} trend")
        
        # Test pattern recognition
        patterns = ts_system.recognize_patterns(ts_data)
        print(f"✅ Pattern recognition: {len(patterns['patterns'])} patterns found")
        
        return True
        
    except Exception as e:
        print(f"❌ Time Series test failed: {e}")
        return False

def test_api_integration():
    """Test API integration readiness"""
    try:
        print("\n🌐 Testing API integration readiness...")
        
        # Test if modules can be imported in API context
        from src.cv.advanced_computer_vision import AdvancedComputerVision
        from src.timeseries.advanced_time_series import AdvancedTimeSeries
        
        cv_system = AdvancedComputerVision()
        ts_system = AdvancedTimeSeries()
        
        print("✅ Both modules can be initialized for API integration")
        
        # Test basic functionality that would be used in API
        sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        sample_ts = pd.DataFrame({
            'timestamp': pd.date_range(start="2024-01-01", periods=50, freq="D"),
            'value': np.random.randn(50).cumsum()
        })
        
        # Quick CV test
        objects = cv_system.detect_objects(sample_image)
        print(f"✅ CV API ready: detected {len(objects)} objects")
        
        # Quick TS test
        forecast = ts_system.forecast_arima(sample_ts, forecast_steps=5)
        print(f"✅ TS API ready: generated {len(forecast['forecast'])} predictions")
        
        return True
        
    except Exception as e:
        print(f"❌ API integration test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🚀 AETHERON ADVANCED FEATURES VALIDATION")
    print("="*50)
    
    results = []
    
    # Test Computer Vision
    cv_result = test_computer_vision_module()
    results.append(("Computer Vision", cv_result))
    
    # Test Time Series
    ts_result = test_time_series_module()
    results.append(("Time Series", ts_result))
    
    # Test API Integration
    api_result = test_api_integration()
    results.append(("API Integration", api_result))
    
    # Summary
    print("\n" + "="*50)
    print("📋 VALIDATION SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL ADVANCED FEATURES VALIDATED SUCCESSFULLY!")
        print("🚀 Ready for production deployment")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
