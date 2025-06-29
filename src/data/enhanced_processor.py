"""
Enhanced Data Pipeline for Jarvis AI.

This module provides advanced data processing capabilities:
- Multiple data source connectors (CSV, JSON, Database, APIs)
- Data validation and quality checks
- Feature engineering pipeline
- Data versioning and lineage tracking
"""

import pandas as pd
import numpy as np
import logging
import json
import sqlite3
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from abc import ABC, abstractmethod
import hashlib
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)


class DataValidator:
    """Data quality validation and checks."""
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values in the dataset."""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        return {
            'total_missing': missing_counts.sum(),
            'missing_by_column': missing_counts.to_dict(),
            'missing_percentages': missing_percentages.to_dict(),
            'columns_with_missing': missing_counts[missing_counts > 0].index.tolist()
        }
    
    @staticmethod
    def check_data_types(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types and suggest improvements."""
        type_info = {}
        for col in df.columns:
            dtype = df[col].dtype
            unique_count = df[col].nunique()
            
            type_info[col] = {
                'current_type': str(dtype),
                'unique_values': unique_count,
                'is_numeric': pd.api.types.is_numeric_dtype(df[col]),
                'is_categorical': unique_count < 50 and dtype == 'object'
            }
        
        return type_info
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, method: str = 'iqr') -> Dict[str, Any]:
        """Detect outliers in numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers[col] = {
                    'count': outlier_mask.sum(),
                    'percentage': (outlier_mask.sum() / len(df)) * 100,
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }
        
        return outliers
    
    @staticmethod
    def generate_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        return {
            'dataset_info': {
                'shape': df.shape,
                'memory_usage': df.memory_usage(deep=True).sum(),
                'columns': df.columns.tolist()
            },
            'missing_values': DataValidator.check_missing_values(df),
            'data_types': DataValidator.check_data_types(df),
            'outliers': DataValidator.detect_outliers(df),
            'timestamp': datetime.now().isoformat()
        }


class DataConnector(ABC):
    """Abstract base class for data connectors."""
    
    @abstractmethod
    def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """Load data from the specified source."""
        pass
    
    @abstractmethod
    def save_data(self, df: pd.DataFrame, destination: str, **kwargs) -> bool:
        """Save data to the specified destination."""
        pass


class CSVConnector(DataConnector):
    """Connector for CSV files."""
    
    def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(source, **kwargs)
            logger.info(f"Successfully loaded CSV data from {source}. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV from {source}: {e}")
            raise
    
    def save_data(self, df: pd.DataFrame, destination: str, **kwargs) -> bool:
        """Save data to CSV file."""
        try:
            df.to_csv(destination, index=False, **kwargs)
            logger.info(f"Successfully saved CSV data to {destination}")
            return True
        except Exception as e:
            logger.error(f"Failed to save CSV to {destination}: {e}")
            return False


class JSONConnector(DataConnector):
    """Connector for JSON files."""
    
    def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """Load data from JSON file."""
        try:
            df = pd.read_json(source, **kwargs)
            logger.info(f"Successfully loaded JSON data from {source}. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load JSON from {source}: {e}")
            raise
    
    def save_data(self, df: pd.DataFrame, destination: str, **kwargs) -> bool:
        """Save data to JSON file."""
        try:
            df.to_json(destination, **kwargs)
            logger.info(f"Successfully saved JSON data to {destination}")
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON to {destination}: {e}")
            return False


class DatabaseConnector(DataConnector):
    """Connector for SQL databases."""
    
    def __init__(self, connection_string: str):
        """Initialize database connector."""
        self.connection_string = connection_string
    
    def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """Load data from database using SQL query."""
        try:
            conn = sqlite3.connect(self.connection_string)
            df = pd.read_sql_query(source, conn, **kwargs)
            conn.close()
            logger.info(f"Successfully loaded data from database. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
            raise
    
    def save_data(self, df: pd.DataFrame, destination: str, **kwargs) -> bool:
        """Save data to database table."""
        try:
            conn = sqlite3.connect(self.connection_string)
            df.to_sql(destination, conn, if_exists='replace', index=False, **kwargs)
            conn.close()
            logger.info(f"Successfully saved data to database table {destination}")
            return True
        except Exception as e:
            logger.error(f"Failed to save data to database: {e}")
            return False


class APIConnector(DataConnector):
    """Connector for REST APIs."""
    
    def __init__(self, base_url: str, headers: Optional[Dict] = None):
        """Initialize API connector."""
        self.base_url = base_url
        self.headers = headers or {}
    
    def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """Load data from API endpoint."""
        try:
            url = f"{self.base_url}/{source}" if not source.startswith('http') else source
            response = requests.get(url, headers=self.headers, **kwargs)
            response.raise_for_status()
            
            data = response.json()
            df = pd.json_normalize(data)
            logger.info(f"Successfully loaded data from API. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data from API: {e}")
            raise
    
    def save_data(self, df: pd.DataFrame, destination: str, **kwargs) -> bool:
        """Post data to API endpoint."""
        try:
            url = f"{self.base_url}/{destination}" if not destination.startswith('http') else destination
            data = df.to_dict('records')
            response = requests.post(url, json=data, headers=self.headers, **kwargs)
            response.raise_for_status()
            logger.info(f"Successfully posted data to API endpoint {destination}")
            return True
        except Exception as e:
            logger.error(f"Failed to post data to API: {e}")
            return False


class FeatureEngineer:
    """Feature engineering pipeline."""
    
    @staticmethod
    def create_polynomial_features(df: pd.DataFrame, columns: List[str], degree: int = 2) -> pd.DataFrame:
        """Create polynomial features."""
        result_df = df.copy()
        
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                for d in range(2, degree + 1):
                    result_df[f"{col}_poly_{d}"] = df[col] ** d
        
        logger.info(f"Created polynomial features up to degree {degree}")
        return result_df
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Create interaction features between specified columns."""
        result_df = df.copy()
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                if col1 in df.columns and col2 in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                        result_df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
        
        logger.info(f"Created interaction features for {len(columns)} columns")
        return result_df
    
    @staticmethod
    def create_binned_features(df: pd.DataFrame, column: str, bins: int = 5, labels: Optional[List] = None) -> pd.DataFrame:
        """Create binned categorical features from numeric columns."""
        result_df = df.copy()
        
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            result_df[f"{column}_binned"] = pd.cut(df[column], bins=bins, labels=labels)
        
        logger.info(f"Created binned features for column {column}")
        return result_df
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame, column: str, windows: List[int]) -> pd.DataFrame:
        """Create rolling statistical features."""
        result_df = df.copy()
        
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            for window in windows:
                result_df[f"{column}_rolling_mean_{window}"] = df[column].rolling(window).mean()
                result_df[f"{column}_rolling_std_{window}"] = df[column].rolling(window).std()
        
        logger.info(f"Created rolling features for column {column}")
        return result_df


class DataLineage:
    """Track data lineage and versioning."""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.lineage_file = Path(f"data/lineage/{project_name}_lineage.json")
        self.lineage_file.parent.mkdir(parents=True, exist_ok=True)
        self.lineage = self._load_lineage()
    
    def _load_lineage(self) -> Dict:
        """Load existing lineage data."""
        if self.lineage_file.exists():
            with open(self.lineage_file, 'r') as f:
                return json.load(f)
        return {"versions": [], "operations": []}
    
    def _save_lineage(self):
        """Save lineage data."""
        with open(self.lineage_file, 'w') as f:
            json.dump(self.lineage, f, indent=2, default=str)
    
    def _calculate_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of dataframe for versioning."""
        return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()
    
    def log_operation(self, operation: str, input_hash: str, output_df: pd.DataFrame, 
                     parameters: Optional[Dict] = None):
        """Log a data processing operation."""
        output_hash = self._calculate_hash(output_df)
        
        operation_record = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "output_shape": output_df.shape,
            "parameters": parameters or {}
        }
        
        self.lineage["operations"].append(operation_record)
        self._save_lineage()
        
        return output_hash
    
    def create_version(self, df: pd.DataFrame, version_name: str, description: str = ""):
        """Create a new data version."""
        data_hash = self._calculate_hash(df)
        
        version_record = {
            "version_name": version_name,
            "hash": data_hash,
            "timestamp": datetime.now().isoformat(),
            "shape": df.shape,
            "description": description,
            "columns": df.columns.tolist()
        }
        
        self.lineage["versions"].append(version_record)
        self._save_lineage()
        
        # Save the actual data
        version_file = Path(f"data/versions/{self.project_name}_{version_name}_{data_hash[:8]}.pkl")
        version_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(version_file, 'wb') as f:
            pickle.dump(df, f)
        
        logger.info(f"Created data version '{version_name}' with hash {data_hash[:8]}")
        return data_hash


class EnhancedDataProcessor:
    """Enhanced data processor with advanced capabilities."""
    
    def __init__(self, project_name: str = "jarvis"):
        self.project_name = project_name
        self.connectors = {
            'csv': CSVConnector(),
            'json': JSONConnector(),
            'sqlite': DatabaseConnector(':memory:'),  # Default in-memory DB
            'api': None  # Will be set when needed
        }
        self.validator = DataValidator()
        self.feature_engineer = FeatureEngineer()
        self.lineage = DataLineage(project_name)
        self.preprocessor = None
        self.is_fitted = False
    
    def load_data(self, source: str, connector_type: str = 'csv', **kwargs) -> pd.DataFrame:
        """Load data using specified connector."""
        if connector_type not in self.connectors:
            raise ValueError(f"Unknown connector type: {connector_type}")
        
        if connector_type == 'api' and self.connectors['api'] is None:
            raise ValueError("API connector not configured. Use set_api_connector() first.")
        
        connector = self.connectors[connector_type]
        df = connector.load_data(source, **kwargs)
        
        # Create initial version
        self.lineage.create_version(df, "raw_data", "Initial data load")
        
        return df
    
    def set_api_connector(self, base_url: str, headers: Optional[Dict] = None):
        """Configure API connector."""
        self.connectors['api'] = APIConnector(base_url, headers)
    
    def set_database_connector(self, connection_string: str):
        """Configure database connector."""
        self.connectors['sqlite'] = DatabaseConnector(connection_string)
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality."""
        return self.validator.generate_quality_report(df)
    
    def engineer_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply feature engineering based on configuration."""
        result_df = df.copy()
        input_hash = self.lineage._calculate_hash(df)
        
        if 'polynomial' in config:
            poly_config = config['polynomial']
            result_df = self.feature_engineer.create_polynomial_features(
                result_df, poly_config.get('columns', []), poly_config.get('degree', 2)
            )
        
        if 'interactions' in config:
            result_df = self.feature_engineer.create_interaction_features(
                result_df, config['interactions']
            )
        
        if 'binning' in config:
            for bin_config in config['binning']:
                result_df = self.feature_engineer.create_binned_features(
                    result_df, bin_config['column'], bin_config.get('bins', 5)
                )
        
        if 'rolling' in config:
            for roll_config in config['rolling']:
                result_df = self.feature_engineer.create_rolling_features(
                    result_df, roll_config['column'], roll_config.get('windows', [3, 5])
                )
        
        # Log the feature engineering operation
        self.lineage.log_operation("feature_engineering", input_hash, result_df, config)
        
        return result_df
    
    def save_data(self, df: pd.DataFrame, destination: str, connector_type: str = 'csv', **kwargs) -> bool:
        """Save data using specified connector."""
        if connector_type not in self.connectors:
            raise ValueError(f"Unknown connector type: {connector_type}")
        
        connector = self.connectors[connector_type]
        return connector.save_data(df, destination, **kwargs)
    
    def get_lineage_summary(self) -> Dict[str, Any]:
        """Get summary of data lineage."""
        return {
            "project": self.project_name,
            "total_versions": len(self.lineage.lineage.get("versions", [])),
            "total_operations": len(self.lineage.lineage.get("operations", [])),
            "latest_version": self.lineage.lineage.get("versions", [])[-1] if self.lineage.lineage.get("versions") else None
        }


# Export classes
__all__ = [
    'EnhancedDataProcessor', 'DataValidator', 'FeatureEngineer', 
    'DataLineage', 'CSVConnector', 'JSONConnector', 'DatabaseConnector', 'APIConnector'
]
