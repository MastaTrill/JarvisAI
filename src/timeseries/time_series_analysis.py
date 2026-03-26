"""
Advanced Time Series Analysis Module for Aetheron AI Platform
Includes forecasting, anomaly detection, trend analysis, and more
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
import json
import os
from datetime import datetime, timedelta
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TimeSeriesConfig:
    """Configuration for time series analysis"""
    task_type: str = "forecasting"  # forecasting, anomaly_detection, classification
    window_size: int = 30
    prediction_horizon: int = 7
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    seasonal_period: int = 24  # For hourly data
    trend_method: str = "linear"  # linear, polynomial, exponential

class TimeSeriesPreprocessor:
    """Time series data preprocessing utilities"""
    
    def __init__(self, normalize: bool = True, handle_missing: str = "interpolate"):
        self.normalize = normalize
        self.handle_missing = handle_missing
        self.scaler_mean = None
        self.scaler_std = None
    
    def normalize_series(self, series: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize time series data"""
        if fit:
            self.scaler_mean = np.mean(series)
            self.scaler_std = np.std(series)
        
        if self.scaler_std == 0:
            return series - self.scaler_mean
        
        return (series - self.scaler_mean) / self.scaler_std
    
    def denormalize_series(self, series: np.ndarray) -> np.ndarray:
        """Denormalize time series data"""
        if self.scaler_std == 0:
            return series + self.scaler_mean
        
        return series * self.scaler_std + self.scaler_mean
    
    def handle_missing_values(self, series: np.ndarray) -> np.ndarray:
        """Handle missing values in time series"""
        if not np.any(np.isnan(series)):
            return series
        
        series_clean = series.copy()
        
        if self.handle_missing == "interpolate":
            # Linear interpolation
            valid_indices = ~np.isnan(series_clean)
            if np.any(valid_indices):
                series_clean[~valid_indices] = np.interp(
                    np.where(~valid_indices)[0],
                    np.where(valid_indices)[0],
                    series_clean[valid_indices]
                )
        elif self.handle_missing == "forward_fill":
            # Forward fill
            last_valid = None
            for i in range(len(series_clean)):
                if not np.isnan(series_clean[i]):
                    last_valid = series_clean[i]
                elif last_valid is not None:
                    series_clean[i] = last_valid
        elif self.handle_missing == "mean":
            # Fill with mean
            mean_value = np.nanmean(series_clean)
            series_clean[np.isnan(series_clean)] = mean_value
        
        return series_clean
    
    def create_sequences(self, series: np.ndarray, window_size: int, 
                        prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for supervised learning"""
        X, y = [], []
        
        for i in range(len(series) - window_size - prediction_horizon + 1):
            X.append(series[i:i + window_size])
            y.append(series[i + window_size:i + window_size + prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def detect_outliers(self, series: np.ndarray, method: str = "iqr", 
                       threshold: float = 1.5) -> np.ndarray:
        """Detect outliers in time series"""
        if method == "iqr":
            Q1 = np.percentile(series, 25)
            Q3 = np.percentile(series, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (series < lower_bound) | (series > upper_bound)
        elif method == "zscore":
            z_scores = np.abs((series - np.mean(series)) / np.std(series))
            outliers = z_scores > threshold
        else:
            outliers = np.zeros(len(series), dtype=bool)
        
        return outliers

class TrendAnalyzer:
    """Trend analysis for time series"""
    
    def __init__(self, method: str = "linear"):
        self.method = method
        self.trend_coefficients = None
    
    def fit_trend(self, series: np.ndarray) -> Dict[str, Any]:
        """Fit trend to time series"""
        x = np.arange(len(series))
        
        if self.method == "linear":
            # Linear trend: y = ax + b
            coeffs = np.polyfit(x, series, 1)
            trend = np.polyval(coeffs, x)
            self.trend_coefficients = coeffs
        elif self.method == "polynomial":
            # Polynomial trend (degree 2)
            coeffs = np.polyfit(x, series, 2)
            trend = np.polyval(coeffs, x)
            self.trend_coefficients = coeffs
        elif self.method == "exponential":
            # Exponential trend (simplified)
            log_series = np.log(np.maximum(series, 1e-8))
            coeffs = np.polyfit(x, log_series, 1)
            trend = np.exp(np.polyval(coeffs, x))
            self.trend_coefficients = coeffs
        else:
            trend = np.full_like(series, np.mean(series))
            self.trend_coefficients = [np.mean(series)]
        
        detrended = series - trend
        
        return {
            'trend': trend,
            'detrended': detrended,
            'coefficients': self.trend_coefficients.tolist(),
            'trend_strength': np.std(trend) / np.std(series) if np.std(series) > 0 else 0
        }
    
    def predict_trend(self, steps: int) -> np.ndarray:
        """Predict future trend"""
        if self.trend_coefficients is None:
            return np.zeros(steps)
        
        start_x = 0  # This should be set to the length of training data
        future_x = np.arange(start_x, start_x + steps)
        
        if self.method == "linear" or self.method == "polynomial":
            future_trend = np.polyval(self.trend_coefficients, future_x)
        elif self.method == "exponential":
            future_trend = np.exp(np.polyval(self.trend_coefficients, future_x))
        else:
            future_trend = np.full(steps, self.trend_coefficients[0])
        
        return future_trend

class SeasonalityDetector:
    """Seasonality detection and decomposition"""
    
    def __init__(self, seasonal_period: int = 24):
        self.seasonal_period = seasonal_period
        self.seasonal_pattern = None
    
    def detect_seasonality(self, series: np.ndarray) -> Dict[str, Any]:
        """Detect seasonal patterns"""
        if len(series) < 2 * self.seasonal_period:
            return {
                'seasonal': np.zeros_like(series),
                'deseasoned': series,
                'seasonal_strength': 0,
                'has_seasonality': False
            }
        
        # Simple seasonal decomposition
        seasonal_averages = []
        for i in range(self.seasonal_period):
            # Get all values at this position in the cycle
            positions = np.arange(i, len(series), self.seasonal_period)
            if len(positions) > 0:
                seasonal_averages.append(np.mean(series[positions]))
            else:
                seasonal_averages.append(0)
        
        # Normalize seasonal pattern
        seasonal_pattern = np.array(seasonal_averages)
        seasonal_pattern -= np.mean(seasonal_pattern)
        self.seasonal_pattern = seasonal_pattern
        
        # Create full seasonal component
        seasonal_full = np.tile(seasonal_pattern, len(series) // self.seasonal_period + 1)[:len(series)]
        
        # Remove seasonal component
        deseasoned = series - seasonal_full
        
        # Calculate seasonality strength
        seasonal_strength = np.std(seasonal_full) / np.std(series) if np.std(series) > 0 else 0
        
        return {
            'seasonal': seasonal_full,
            'deseasoned': deseasoned,
            'seasonal_pattern': seasonal_pattern,
            'seasonal_strength': seasonal_strength,
            'has_seasonality': seasonal_strength > 0.1
        }
    
    def predict_seasonal(self, steps: int) -> np.ndarray:
        """Predict future seasonal component"""
        if self.seasonal_pattern is None:
            return np.zeros(steps)
        
        # Repeat seasonal pattern for future steps
        future_seasonal = np.tile(self.seasonal_pattern, steps // self.seasonal_period + 1)[:steps]
        return future_seasonal

class SimpleRNNForecaster:
    """Simple RNN for time series forecasting"""
    
    def __init__(self, input_size: int, hidden_size: int = 32, output_size: int = 1,
                 learning_rate: float = 0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.1
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.Why = np.random.randn(hidden_size, output_size) * 0.1
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))
        
        # Training history
        self.training_history = {'loss': []}
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Forward pass through RNN"""
        seq_length = X.shape[0]
        hidden_states = []
        
        h = np.zeros((1, self.hidden_size))
        
        for t in range(seq_length):
            x_t = X[t].reshape(1, -1)
            h = self.tanh(np.dot(x_t, self.Wxh) + np.dot(h, self.Whh) + self.bh)
            hidden_states.append(h.copy())
        
        # Output from last hidden state
        output = np.dot(h, self.Why) + self.by
        
        return output, hidden_states
    
    def predict(self, X: np.ndarray) -> float:
        """Make prediction for single sequence"""
        output, _ = self.forward(X)
        return output[0, 0]
    
    def train_step(self, X: np.ndarray, y: float) -> float:
        """Single training step"""
        # Forward pass
        output, hidden_states = self.forward(X)
        
        # Calculate loss (MSE)
        prediction = output[0, 0]
        loss = (prediction - y) ** 2
        
        # Simple weight update (approximation)
        error = prediction - y
        
        # Update output weights
        self.Why -= self.learning_rate * np.outer(hidden_states[-1], error)
        self.by -= self.learning_rate * error
        
        return loss

class AnomalyDetector:
    """Time series anomaly detection"""
    
    def __init__(self, method: str = "statistical", window_size: int = 30):
        self.method = method
        self.window_size = window_size
        self.baseline_stats = None
    
    def fit(self, series: np.ndarray):
        """Fit anomaly detector to baseline data"""
        if self.method == "statistical":
            self.baseline_stats = {
                'mean': np.mean(series),
                'std': np.std(series),
                'median': np.median(series),
                'iqr': np.percentile(series, 75) - np.percentile(series, 25)
            }
        elif self.method == "isolation_forest":
            # Simplified isolation forest concept
            self.baseline_stats = {
                'percentiles': np.percentile(series, np.arange(0, 101, 5)),
                'mean': np.mean(series),
                'std': np.std(series)
            }
    
    def detect_anomalies(self, series: np.ndarray, threshold: float = 2.0) -> np.ndarray:
        """Detect anomalies in time series"""
        if self.baseline_stats is None:
            self.fit(series)
        
        anomalies = np.zeros(len(series), dtype=bool)
        
        if self.method == "statistical":
            # Z-score based detection
            z_scores = np.abs((series - self.baseline_stats['mean']) / 
                            (self.baseline_stats['std'] + 1e-8))
            anomalies = z_scores > threshold
        
        elif self.method == "moving_window":
            # Moving window statistics
            for i in range(len(series)):
                start_idx = max(0, i - self.window_size)
                window = series[start_idx:i+1]
                
                if len(window) >= 3:
                    window_mean = np.mean(window)
                    window_std = np.std(window)
                    
                    if window_std > 0:
                        z_score = abs(series[i] - window_mean) / window_std
                        anomalies[i] = z_score > threshold
        
        return anomalies

class TimeSeriesForecaster:
    """Complete time series forecasting system"""
    
    def __init__(self, config: TimeSeriesConfig):
        self.config = config
        
        # Components
        self.preprocessor = TimeSeriesPreprocessor()
        self.trend_analyzer = TrendAnalyzer(config.trend_method)
        self.seasonality_detector = SeasonalityDetector(config.seasonal_period)
        self.rnn_forecaster = SimpleRNNForecaster(config.window_size)
        self.anomaly_detector = AnomalyDetector()
        
        # State
        self.is_fitted = False
        self.decomposition_results = None
    
    def fit(self, series: np.ndarray) -> Dict[str, Any]:
        """Fit forecasting model to time series"""
        logger.info("Fitting time series forecasting model...")
        
        # Preprocess data
        clean_series = self.preprocessor.handle_missing_values(series)
        normalized_series = self.preprocessor.normalize_series(clean_series, fit=True)
        
        # Decompose time series
        trend_results = self.trend_analyzer.fit_trend(normalized_series)
        seasonal_results = self.seasonality_detector.detect_seasonality(trend_results['detrended'])
        
        # Get residuals (what's left after trend and seasonality removal)
        residuals = seasonal_results['deseasoned']
        
        # Prepare training data for RNN
        X, y = self.preprocessor.create_sequences(
            residuals, self.config.window_size, self.config.prediction_horizon
        )
        
        # Train RNN on residuals
        if len(X) > 0:
            for epoch in range(min(self.config.epochs, 50)):  # Limit epochs for demo
                epoch_loss = 0
                for i in range(len(X)):
                    if len(y[i]) > 0:  # Make sure we have target values
                        loss = self.rnn_forecaster.train_step(X[i], y[i][0])
                        epoch_loss += loss
                
                avg_loss = epoch_loss / len(X) if len(X) > 0 else 0
                self.rnn_forecaster.training_history['loss'].append(avg_loss)
                
                if epoch % 10 == 0:
                    logger.info(f"RNN training epoch {epoch}, loss: {avg_loss:.4f}")
        
        # Fit anomaly detector
        self.anomaly_detector.fit(normalized_series)
        
        # Store decomposition results
        self.decomposition_results = {
            'trend': trend_results,
            'seasonal': seasonal_results,
            'residuals': residuals
        }
        
        self.is_fitted = True
        
        return self.decomposition_results
    
    def forecast(self, steps: int) -> Dict[str, Any]:
        """Generate forecasts"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Predict trend component
        trend_forecast = self.trend_analyzer.predict_trend(steps)
        
        # Predict seasonal component
        seasonal_forecast = self.seasonality_detector.predict_seasonal(steps)
        
        # For residuals, use last known values (simple approach)
        residuals = self.decomposition_results['residuals']
        if len(residuals) >= self.config.window_size:
            last_window = residuals[-self.config.window_size:]
            residual_forecast = []
            
            for _ in range(steps):
                prediction = self.rnn_forecaster.predict(last_window)
                residual_forecast.append(prediction)
                
                # Update window for next prediction
                last_window = np.roll(last_window, -1)
                last_window[-1] = prediction
        else:
            residual_forecast = np.zeros(steps)
        
        # Combine components
        combined_forecast = trend_forecast + seasonal_forecast + residual_forecast
        
        # Denormalize
        final_forecast = self.preprocessor.denormalize_series(combined_forecast)
        
        return {
            'forecast': final_forecast,
            'trend_component': self.preprocessor.denormalize_series(trend_forecast),
            'seasonal_component': seasonal_forecast,  # This is already centered around 0
            'residual_component': residual_forecast,
            'confidence_intervals': self._calculate_confidence_intervals(final_forecast)
        }
    
    def _calculate_confidence_intervals(self, forecast: np.ndarray, 
                                      confidence: float = 0.95) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals for forecasts"""
        # Simple approach using historical residuals
        if self.decomposition_results is not None:
            residual_std = np.std(self.decomposition_results['residuals'])
        else:
            residual_std = 0.1
        
        # Z-score for 95% confidence
        z_score = 1.96 if confidence == 0.95 else 2.576
        
        margin = z_score * residual_std
        
        return {
            'lower_bound': forecast - margin,
            'upper_bound': forecast + margin
        }

class TimeSeriesExperimentTracker:
    """Track time series experiments"""
    
    def __init__(self, experiment_dir: str = "experiments/timeseries"):
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)
        
        self.experiments = []
    
    def create_experiment(self, name: str, config: TimeSeriesConfig, 
                         description: str = "") -> str:
        """Create new time series experiment"""
        experiment_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_path = os.path.join(self.experiment_dir, experiment_id)
        os.makedirs(experiment_path, exist_ok=True)
        
        # Save config
        config_path = os.path.join(experiment_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        
        # Save metadata
        metadata = {
            'id': experiment_id,
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'task_type': config.task_type,
            'status': 'created'
        }
        
        metadata_path = os.path.join(experiment_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.experiments.append(metadata)
        logger.info(f"Created time series experiment: {experiment_id}")
        
        return experiment_id
    
    def save_results(self, experiment_id: str, results: Dict[str, Any]):
        """Save experiment results"""
        experiment_path = os.path.join(self.experiment_dir, experiment_id)
        results_path = os.path.join(experiment_path, "results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_results[key][k] = v.tolist()
                    else:
                        serializable_results[key][k] = v
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved results for experiment: {experiment_id}")

def create_sample_timeseries(length: int = 365, noise_level: float = 0.1) -> np.ndarray:
    """Create sample time series with trend, seasonality, and noise"""
    x = np.arange(length)
    
    # Trend component
    trend = 0.02 * x + 10
    
    # Seasonal component (annual cycle)
    seasonal = 5 * np.sin(2 * np.pi * x / 365.25)
    
    # Weekly pattern
    weekly = 2 * np.sin(2 * np.pi * x / 7)
    
    # Daily pattern (if hourly data)
    if length > 365:
        daily = 1 * np.sin(2 * np.pi * x / 24)
    else:
        daily = 0
    
    # Noise
    noise = np.random.normal(0, noise_level, length)
    
    # Combine components
    series = trend + seasonal + weekly + daily + noise
    
    # Add some outliers
    outlier_indices = np.random.choice(length, size=int(0.02 * length), replace=False)
    series[outlier_indices] += np.random.normal(0, 5, len(outlier_indices))
    
    return series

def create_timeseries_system(task_type: str = "forecasting") -> Dict[str, Any]:
    """Create and configure time series system"""
    
    # Create configuration
    config = TimeSeriesConfig(task_type=task_type)
    
    # Create components
    forecaster = TimeSeriesForecaster(config)
    tracker = TimeSeriesExperimentTracker()
    
    return {
        'config': config,
        'forecaster': forecaster,
        'tracker': tracker
    }

# Example usage and testing
if __name__ == "__main__":
    print("Testing Time Series Analysis System...")
    
    # Create time series system
    ts_system = create_timeseries_system("forecasting")
    
    # Create sample data
    sample_series = create_sample_timeseries(length=500, noise_level=0.2)
    
    # Create experiment
    experiment_id = ts_system['tracker'].create_experiment(
        "forecasting_test",
        ts_system['config'],
        "Testing time series forecasting on synthetic data"
    )
    
    # Split data into train/test
    train_size = int(0.8 * len(sample_series))
    train_series = sample_series[:train_size]
    test_series = sample_series[train_size:]
    
    print(f"Training on {len(train_series)} points, testing on {len(test_series)} points")
    
    # Fit forecaster
    print("Fitting forecasting model...")
    decomposition = ts_system['forecaster'].fit(train_series)
    
    print(f"Trend strength: {decomposition['trend']['trend_strength']:.3f}")
    print(f"Seasonal strength: {decomposition['seasonal']['seasonal_strength']:.3f}")
    print(f"Has seasonality: {decomposition['seasonal']['has_seasonality']}")
    
    # Generate forecasts
    print("Generating forecasts...")
    forecast_steps = min(len(test_series), 30)  # Forecast up to 30 steps
    forecast_results = ts_system['forecaster'].forecast(forecast_steps)
    
    # Calculate forecast accuracy
    actual_values = test_series[:forecast_steps]
    predicted_values = forecast_results['forecast'][:forecast_steps]
    
    mae = np.mean(np.abs(actual_values - predicted_values))
    mse = np.mean((actual_values - predicted_values) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"Forecast accuracy (MAE): {mae:.3f}")
    print(f"Forecast accuracy (RMSE): {rmse:.3f}")
    
    # Test anomaly detection
    print("Testing anomaly detection...")
    anomaly_detector = AnomalyDetector(method="statistical")
    anomalies = anomaly_detector.detect_anomalies(train_series, threshold=2.5)
    
    print(f"Detected {np.sum(anomalies)} anomalies out of {len(train_series)} points")
    
    # Test outlier detection
    print("Testing outlier detection...")
    preprocessor = TimeSeriesPreprocessor()
    outliers = preprocessor.detect_outliers(train_series, method="iqr", threshold=1.5)
    
    print(f"Detected {np.sum(outliers)} outliers using IQR method")
    
    # Save results
    results = {
        'forecast_accuracy': {
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        },
        'decomposition': {
            'trend_strength': decomposition['trend']['trend_strength'],
            'seasonal_strength': decomposition['seasonal']['seasonal_strength'],
            'has_seasonality': decomposition['seasonal']['has_seasonality']
        },
        'anomaly_detection': {
            'num_anomalies': int(np.sum(anomalies)),
            'anomaly_rate': float(np.sum(anomalies) / len(anomalies))
        },
        'outlier_detection': {
            'num_outliers': int(np.sum(outliers)),
            'outlier_rate': float(np.sum(outliers) / len(outliers))
        },
        'forecast_results': forecast_results
    }
    
    ts_system['tracker'].save_results(experiment_id, results)
    
    print("\n✅ Time Series Analysis module tests completed successfully!")
