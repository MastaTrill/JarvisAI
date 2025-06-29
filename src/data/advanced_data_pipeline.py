#!/usr/bin/env python3
"""
Advanced Data Pipeline for Jarvis AI Platform.
Provides real data connectors, validation, feature engineering, and versioning.
"""

import numpy as np
import pandas as pd
import json
import yaml
import pickle
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import logging
from abc import ABC, abstractmethod
import requests
import urllib.parse

logger = logging.getLogger(__name__)


class DataConnector(ABC):
    """Abstract base class for data connectors."""
    
    @abstractmethod
    def connect(self, config: Dict[str, Any]) -> bool:
        """Establish connection to data source."""
        pass
    
    @abstractmethod
    def fetch_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """Fetch data from the source."""
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate the connection is working."""
        pass


class CSVConnector(DataConnector):
    """Connector for CSV files."""
    
    def __init__(self):
        self.filepath = None
        self.encoding = 'utf-8'
        self.separator = ','
    
    def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to CSV file."""
        try:
            self.filepath = Path(config['filepath'])
            self.encoding = config.get('encoding', 'utf-8')
            self.separator = config.get('separator', ',')
            
            if not self.filepath.exists():
                logger.error(f"CSV file not found: {self.filepath}")
                return False
            
            logger.info(f"✅ Connected to CSV: {self.filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to CSV: {e}")
            return False
    
    def fetch_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """Fetch data from CSV file."""
        try:
            df = pd.read_csv(
                self.filepath,
                encoding=self.encoding,
                sep=self.separator
            )
            
            # Apply query if provided (simple column selection)
            if query:
                columns = [col.strip() for col in query.split(',')]
                available_cols = [col for col in columns if col in df.columns]
                if available_cols:
                    df = df[available_cols]
            
            logger.info(f"📊 Fetched {len(df)} rows, {len(df.columns)} columns from CSV")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch CSV data: {e}")
            return pd.DataFrame()
    
    def validate_connection(self) -> bool:
        """Validate CSV file is accessible."""
        return self.filepath and self.filepath.exists()


class JSONConnector(DataConnector):
    """Connector for JSON files and APIs."""
    
    def __init__(self):
        self.source_type = None  # 'file' or 'api'
        self.filepath = None
        self.api_url = None
        self.headers = {}
        self.timeout = 30
    
    def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to JSON source."""
        try:
            self.source_type = config.get('type', 'file')
            
            if self.source_type == 'file':
                self.filepath = Path(config['filepath'])
                if not self.filepath.exists():
                    logger.error(f"JSON file not found: {self.filepath}")
                    return False
            elif self.source_type == 'api':
                self.api_url = config['url']
                self.headers = config.get('headers', {})
                self.timeout = config.get('timeout', 30)
            
            logger.info(f"✅ Connected to JSON {self.source_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to JSON: {e}")
            return False
    
    def fetch_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """Fetch data from JSON source."""
        try:
            if self.source_type == 'file':
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif self.source_type == 'api':
                response = requests.get(self.api_url, headers=self.headers, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Try to find the main data array
                data_key = query or 'data'
                if data_key in data and isinstance(data[data_key], list):
                    df = pd.DataFrame(data[data_key])
                else:
                    df = pd.DataFrame([data])
            else:
                df = pd.DataFrame()
            
            logger.info(f"📊 Fetched {len(df)} rows, {len(df.columns)} columns from JSON")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch JSON data: {e}")
            return pd.DataFrame()
    
    def validate_connection(self) -> bool:
        """Validate JSON source is accessible."""
        try:
            if self.source_type == 'file':
                return self.filepath and self.filepath.exists()
            elif self.source_type == 'api':
                response = requests.head(self.api_url, headers=self.headers, timeout=5)
                return response.status_code == 200
            return False
        except:
            return False


class DatabaseConnector(DataConnector):
    """Connector for SQL databases."""
    
    def __init__(self):
        self.connection = None
        self.db_type = None
        self.connection_string = None
    
    def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to database."""
        try:
            self.db_type = config.get('type', 'sqlite')
            
            if self.db_type == 'sqlite':
                db_path = config['database']
                self.connection = sqlite3.connect(db_path)
                logger.info(f"✅ Connected to SQLite: {db_path}")
                return True
            else:
                logger.error(f"Database type {self.db_type} not supported yet")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def fetch_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """Fetch data from database."""
        try:
            if not self.connection:
                logger.error("No database connection")
                return pd.DataFrame()
            
            if not query:
                # Default query - get first table
                cursor = self.connection.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                if tables:
                    query = f"SELECT * FROM {tables[0][0]} LIMIT 1000"
                else:
                    return pd.DataFrame()
            
            df = pd.read_sql_query(query, self.connection)
            logger.info(f"📊 Fetched {len(df)} rows, {len(df.columns)} columns from database")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch database data: {e}")
            return pd.DataFrame()
    
    def validate_connection(self) -> bool:
        """Validate database connection."""
        try:
            if self.connection:
                cursor = self.connection.cursor()
                cursor.execute("SELECT 1")
                return True
            return False
        except:
            return False


class DataValidator:
    """Data validation and quality checks."""
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data quality checks."""
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'data_types': {},
            'duplicates': 0,
            'outliers': {},
            'quality_score': 0.0,
            'issues': []
        }
        
        if df.empty:
            report['quality_score'] = 0.0
            report['issues'].append('Dataset is empty')
            return report
        
        # Missing values analysis
        missing_counts = df.isnull().sum()
        for col in df.columns:
            missing_pct = (missing_counts[col] / len(df)) * 100
            report['missing_values'][col] = {
                'count': int(missing_counts[col]),
                'percentage': round(missing_pct, 2)
            }
            if missing_pct > 50:
                report['issues'].append(f'Column {col} has {missing_pct:.1f}% missing values')
        
        # Data types
        for col in df.columns:
            report['data_types'][col] = str(df[col].dtype)
        
        # Duplicates
        report['duplicates'] = int(df.duplicated().sum())
        if report['duplicates'] > 0:
            report['issues'].append(f'Found {report["duplicates"]} duplicate rows')
        
        # Outliers detection (for numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(df)) * 100
            
            report['outliers'][col] = {
                'count': outlier_count,
                'percentage': round(outlier_pct, 2),
                'bounds': [float(lower_bound), float(upper_bound)]
            }
            
            if outlier_pct > 10:
                report['issues'].append(f'Column {col} has {outlier_pct:.1f}% outliers')
        
        # Calculate quality score
        missing_penalty = min(50, np.mean([v['percentage'] for v in report['missing_values'].values()]))
        duplicate_penalty = min(20, (report['duplicates'] / len(df)) * 100)
        outlier_penalty = min(20, np.mean([v['percentage'] for v in report['outliers'].values()]))
        
        report['quality_score'] = max(0, 100 - missing_penalty - duplicate_penalty - outlier_penalty)
        
        return report
    
    @staticmethod
    def suggest_fixes(validation_report: Dict[str, Any]) -> List[str]:
        """Suggest fixes for data quality issues."""
        suggestions = []
        
        # Missing values suggestions
        for col, info in validation_report['missing_values'].items():
            if info['percentage'] > 30:
                suggestions.append(f'Consider dropping column {col} (high missing values)')
            elif info['percentage'] > 5:
                suggestions.append(f'Impute missing values in column {col}')
        
        # Duplicates
        if validation_report['duplicates'] > 0:
            suggestions.append('Remove duplicate rows')
        
        # Outliers
        for col, info in validation_report['outliers'].items():
            if info['percentage'] > 5:
                suggestions.append(f'Handle outliers in column {col}')
        
        return suggestions


class FeatureEngineer:
    """Automated feature engineering."""
    
    @staticmethod
    def create_temporal_features(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        """Create temporal features from date columns."""
        df_enhanced = df.copy()
        
        for col in date_columns:
            if col in df.columns:
                try:
                    # Convert to datetime if not already
                    df_enhanced[col] = pd.to_datetime(df_enhanced[col])
                    
                    # Extract temporal features
                    df_enhanced[f'{col}_year'] = df_enhanced[col].dt.year
                    df_enhanced[f'{col}_month'] = df_enhanced[col].dt.month
                    df_enhanced[f'{col}_day'] = df_enhanced[col].dt.day
                    df_enhanced[f'{col}_dayofweek'] = df_enhanced[col].dt.dayofweek
                    df_enhanced[f'{col}_quarter'] = df_enhanced[col].dt.quarter
                    df_enhanced[f'{col}_is_weekend'] = df_enhanced[col].dt.dayofweek.isin([5, 6]).astype(int)
                    
                    logger.info(f"✅ Created temporal features for {col}")
                except Exception as e:
                    logger.warning(f"Failed to create temporal features for {col}: {e}")
        
        return df_enhanced
    
    @staticmethod
    def create_polynomial_features(df: pd.DataFrame, numeric_columns: List[str], degree: int = 2) -> pd.DataFrame:
        """Create polynomial features."""
        df_enhanced = df.copy()
        
        for col in numeric_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                for d in range(2, degree + 1):
                    df_enhanced[f'{col}_pow{d}'] = df_enhanced[col] ** d
                
                logger.info(f"✅ Created polynomial features for {col}")
        
        return df_enhanced
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
        """Create interaction features between numeric columns."""
        df_enhanced = df.copy()
        
        numeric_cols = [col for col in numeric_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # Multiplication interaction
                df_enhanced[f'{col1}_x_{col2}'] = df_enhanced[col1] * df_enhanced[col2]
                
                # Division interaction (avoid division by zero)
                df_enhanced[f'{col1}_div_{col2}'] = df_enhanced[col1] / (df_enhanced[col2] + 1e-8)
        
        logger.info(f"✅ Created interaction features")
        return df_enhanced
    
    @staticmethod
    def create_categorical_features(df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """Create features from categorical columns."""
        df_enhanced = df.copy()
        
        for col in categorical_columns:
            if col in df.columns:
                # One-hot encoding for low cardinality
                unique_count = df[col].nunique()
                if unique_count <= 10:
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df_enhanced = pd.concat([df_enhanced, dummies], axis=1)
                else:
                    # Label encoding for high cardinality
                    df_enhanced[f'{col}_encoded'] = pd.factorize(df[col])[0]
        
        return df_enhanced


class DataVersioning:
    """Data versioning and lineage tracking."""
    
    def __init__(self, storage_path: str = "data/versions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.lineage_file = self.storage_path / "lineage.json"
        self.lineage = self._load_lineage()
    
    def _load_lineage(self) -> Dict[str, Any]:
        """Load existing lineage information."""
        if self.lineage_file.exists():
            with open(self.lineage_file, 'r') as f:
                return json.load(f)
        return {"versions": {}, "transformations": []}
    
    def _save_lineage(self):
        """Save lineage information."""
        with open(self.lineage_file, 'w') as f:
            json.dump(self.lineage, f, indent=2, default=str)
    
    def _calculate_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of DataFrame for versioning."""
        return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
    
    def save_version(self, df: pd.DataFrame, name: str, description: str = "") -> str:
        """Save a new version of the data."""
        try:
            data_hash = self._calculate_hash(df)
            timestamp = datetime.now().isoformat()
            version_id = f"{name}_{data_hash[:8]}"
            
            # Save data
            file_path = self.storage_path / f"{version_id}.pkl"
            df.to_pickle(file_path)
            
            # Update lineage
            self.lineage["versions"][version_id] = {
                "name": name,
                "description": description,
                "timestamp": timestamp,
                "hash": data_hash,
                "file_path": str(file_path),
                "shape": df.shape,
                "columns": list(df.columns)
            }
            
            self._save_lineage()
            logger.info(f"✅ Saved data version: {version_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to save data version: {e}")
            return ""
    
    def load_version(self, version_id: str) -> Optional[pd.DataFrame]:
        """Load a specific version of the data."""
        try:
            if version_id in self.lineage["versions"]:
                file_path = self.lineage["versions"][version_id]["file_path"]
                df = pd.read_pickle(file_path)
                logger.info(f"✅ Loaded data version: {version_id}")
                return df
            else:
                logger.error(f"Version {version_id} not found")
                return None
        except Exception as e:
            logger.error(f"Failed to load data version: {e}")
            return None
    
    def track_transformation(self, input_version: str, output_version: str, 
                           transformation: str, parameters: Dict[str, Any]):
        """Track a data transformation."""
        transformation_record = {
            "timestamp": datetime.now().isoformat(),
            "input_version": input_version,
            "output_version": output_version,
            "transformation": transformation,
            "parameters": parameters
        }
        
        self.lineage["transformations"].append(transformation_record)
        self._save_lineage()
        logger.info(f"✅ Tracked transformation: {transformation}")
    
    def get_lineage_graph(self) -> Dict[str, Any]:
        """Get the complete data lineage graph."""
        return self.lineage


class AdvancedDataPipeline:
    """
    Comprehensive data pipeline with connectors, validation, feature engineering, and versioning.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        self.connectors = {
            'csv': CSVConnector(),
            'json': JSONConnector(),
            'database': DatabaseConnector()
        }
        self.validator = DataValidator()
        self.feature_engineer = FeatureEngineer()
        self.versioning = DataVersioning()
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load pipeline configuration."""
        try:
            config_file = Path(config_path)
            if config_file.suffix.lower() == '.yaml':
                with open(config_file, 'r') as f:
                    self.config = yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                with open(config_file, 'r') as f:
                    self.config = json.load(f)
            
            logger.info(f"✅ Loaded pipeline config from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    
    def connect_to_source(self, source_config: Dict[str, Any]) -> bool:
        """Connect to a data source."""
        source_type = source_config.get('type', 'csv')
        
        if source_type in self.connectors:
            return self.connectors[source_type].connect(source_config)
        else:
            logger.error(f"Unknown source type: {source_type}")
            return False
    
    def fetch_data(self, source_type: str, query: Optional[str] = None) -> pd.DataFrame:
        """Fetch data from connected source."""
        if source_type in self.connectors:
            return self.connectors[source_type].fetch_data(query)
        else:
            logger.error(f"No connector for source type: {source_type}")
            return pd.DataFrame()
    
    def process_data(self, df: pd.DataFrame, processing_config: Dict[str, Any]) -> pd.DataFrame:
        """Process data according to configuration."""
        processed_df = df.copy()
        
        # Data validation
        if processing_config.get('validate', True):
            validation_report = self.validator.validate_data_quality(processed_df)
            logger.info(f"📊 Data quality score: {validation_report['quality_score']:.1f}/100")
            
            if validation_report['issues']:
                logger.warning("Data quality issues found:")
                for issue in validation_report['issues']:
                    logger.warning(f"  - {issue}")
        
        # Feature engineering
        fe_config = processing_config.get('feature_engineering', {})
        
        if fe_config.get('temporal_features'):
            date_columns = fe_config.get('date_columns', [])
            processed_df = self.feature_engineer.create_temporal_features(processed_df, date_columns)
        
        if fe_config.get('polynomial_features'):
            numeric_columns = fe_config.get('numeric_columns', [])
            degree = fe_config.get('polynomial_degree', 2)
            processed_df = self.feature_engineer.create_polynomial_features(
                processed_df, numeric_columns, degree
            )
        
        if fe_config.get('interaction_features'):
            numeric_columns = fe_config.get('numeric_columns', [])
            processed_df = self.feature_engineer.create_interaction_features(
                processed_df, numeric_columns
            )
        
        if fe_config.get('categorical_features'):
            categorical_columns = fe_config.get('categorical_columns', [])
            processed_df = self.feature_engineer.create_categorical_features(
                processed_df, categorical_columns
            )
        
        # Data cleaning
        cleaning_config = processing_config.get('cleaning', {})
        
        if cleaning_config.get('remove_duplicates', False):
            before_count = len(processed_df)
            processed_df = processed_df.drop_duplicates()
            removed = before_count - len(processed_df)
            if removed > 0:
                logger.info(f"🧹 Removed {removed} duplicate rows")
        
        if cleaning_config.get('handle_missing'):
            strategy = cleaning_config.get('missing_strategy', 'drop')
            if strategy == 'drop':
                processed_df = processed_df.dropna()
            elif strategy == 'fill_mean':
                processed_df = processed_df.fillna(processed_df.mean())
            elif strategy == 'fill_median':
                processed_df = processed_df.fillna(processed_df.median())
            elif strategy == 'fill_mode':
                processed_df = processed_df.fillna(processed_df.mode().iloc[0])
        
        return processed_df
    
    def run_pipeline(self, pipeline_name: str = "default") -> Optional[pd.DataFrame]:
        """Run the complete data pipeline."""
        try:
            pipeline_config = self.config.get('pipelines', {}).get(pipeline_name, {})
            
            if not pipeline_config:
                logger.error(f"Pipeline configuration '{pipeline_name}' not found")
                return None
            
            # Connect to data source
            source_config = pipeline_config.get('source', {})
            if not self.connect_to_source(source_config):
                logger.error("Failed to connect to data source")
                return None
            
            # Fetch data
            source_type = source_config.get('type', 'csv')
            query = source_config.get('query')
            df = self.fetch_data(source_type, query)
            
            if df.empty:
                logger.error("No data fetched from source")
                return None
            
            # Save original version
            original_version = self.versioning.save_version(
                df, f"{pipeline_name}_original", "Original data from source"
            )
            
            # Process data
            processing_config = pipeline_config.get('processing', {})
            processed_df = self.process_data(df, processing_config)
            
            # Save processed version
            processed_version = self.versioning.save_version(
                processed_df, f"{pipeline_name}_processed", "Processed data"
            )
            
            # Track transformation
            self.versioning.track_transformation(
                original_version, processed_version, "data_processing", processing_config
            )
            
            logger.info(f"✅ Pipeline '{pipeline_name}' completed successfully")
            logger.info(f"📊 Final dataset: {processed_df.shape[0]} rows, {processed_df.shape[1]} columns")
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return None
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about available pipelines and connections."""
        info = {
            "available_pipelines": list(self.config.get('pipelines', {}).keys()),
            "available_connectors": list(self.connectors.keys()),
            "data_versions": len(self.versioning.lineage.get("versions", {})),
            "transformations": len(self.versioning.lineage.get("transformations", []))
        }
        
        # Check connector statuses
        connector_status = {}
        for name, connector in self.connectors.items():
            connector_status[name] = connector.validate_connection()
        
        info["connector_status"] = connector_status
        return info
