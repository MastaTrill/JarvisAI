"""
Jarvis AI - Data Integrations Package
Real-world data source connections for live AI/ML applications
"""

from .data_connectors import (
    DataSourceType,
    ConnectionConfig,
    RESTAPIConfig,
    DatabaseConfig,
    MessageQueueConfig,
    CloudStorageConfig,
    StreamingConfig,
    DataConnector,
    RESTAPIConnector,
    WebSocketConnector,
    PostgreSQLConnector,
    S3Connector,
    DataIntegrationManager,
    create_connector,
)

__all__ = [
    "DataSourceType",
    "ConnectionConfig",
    "RESTAPIConfig",
    "DatabaseConfig",
    "MessageQueueConfig",
    "CloudStorageConfig",
    "StreamingConfig",
    "DataConnector",
    "RESTAPIConnector",
    "WebSocketConnector",
    "PostgreSQLConnector",
    "S3Connector",
    "DataIntegrationManager",
    "create_connector",
]
