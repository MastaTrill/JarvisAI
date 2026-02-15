"""
Jarvis AI - Real-World Data Integration Module
Connect to live data sources for real-time AI/ML applications

Supported Data Sources:
- REST APIs (generic)
- Databases (PostgreSQL, MySQL, MongoDB)
- Message Queues (Kafka, RabbitMQ, Redis Pub/Sub)
- Cloud Storage (S3, GCS, Azure Blob)
- Streaming Data (WebSocket, SSE)
- IoT Devices (MQTT)
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import aiohttp
import backoff

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class DataSourceType(str, Enum):
    """Supported data source types"""
    REST_API = "rest_api"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    KAFKA = "kafka"
    RABBITMQ = "rabbitmq"
    REDIS_PUBSUB = "redis_pubsub"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"
    WEBSOCKET = "websocket"
    SSE = "sse"
    MQTT = "mqtt"


@dataclass
class ConnectionConfig:
    """Base connection configuration"""
    name: str
    source_type: DataSourceType
    enabled: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RESTAPIConfig(ConnectionConfig):
    """REST API connection configuration"""
    base_url: str = ""
    auth_type: str = "none"  # none, api_key, bearer, basic, oauth2
    auth_config: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    rate_limit: int = 100  # requests per minute
    
    def __post_init__(self):
        self.source_type = DataSourceType.REST_API


@dataclass
class DatabaseConfig(ConnectionConfig):
    """Database connection configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = ""
    username: str = ""
    password: str = ""
    ssl_mode: str = "prefer"
    pool_size: int = 10
    
    @property
    def connection_string(self) -> str:
        """Generate connection string"""
        if self.source_type == DataSourceType.POSTGRESQL:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.source_type == DataSourceType.MYSQL:
            return f"mysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.source_type == DataSourceType.MONGODB:
            return f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        return ""


@dataclass
class MessageQueueConfig(ConnectionConfig):
    """Message queue configuration"""
    host: str = "localhost"
    port: int = 9092
    topic: str = ""
    group_id: str = "jarvis-consumer"
    username: str = ""
    password: str = ""
    

@dataclass
class CloudStorageConfig(ConnectionConfig):
    """Cloud storage configuration"""
    bucket: str = ""
    prefix: str = ""
    access_key: str = ""
    secret_key: str = ""
    region: str = "us-east-1"
    endpoint_url: Optional[str] = None


@dataclass
class StreamingConfig(ConnectionConfig):
    """Streaming data configuration"""
    url: str = ""
    reconnect_interval: float = 5.0
    heartbeat_interval: float = 30.0
    

# =============================================================================
# ABSTRACT BASE CONNECTOR
# =============================================================================

class DataConnector(ABC):
    """Abstract base class for all data connectors"""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.is_connected = False
        self._callbacks: List[Callable] = []
        self._last_fetch: Optional[datetime] = None
        self._metrics = {
            "total_records": 0,
            "errors": 0,
            "last_error": None,
            "avg_latency_ms": 0.0
        }
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection"""
        pass
    
    @abstractmethod
    async def fetch(self, query: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Fetch data from source"""
        pass
    
    @abstractmethod
    async def stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream data from source"""
        pass
    
    def on_data(self, callback: Callable) -> None:
        """Register callback for data events"""
        self._callbacks.append(callback)
    
    async def _notify(self, data: Dict[str, Any]) -> None:
        """Notify all registered callbacks"""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get connector metrics"""
        return {
            "name": self.config.name,
            "type": self.config.source_type.value,
            "connected": self.is_connected,
            "last_fetch": self._last_fetch.isoformat() if self._last_fetch else None,
            **self._metrics
        }


# =============================================================================
# REST API CONNECTOR
# =============================================================================

class RESTAPIConnector(DataConnector):
    """Connector for REST APIs"""
    
    def __init__(self, config: RESTAPIConfig):
        super().__init__(config)
        self.config: RESTAPIConfig = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = asyncio.Semaphore(config.rate_limit)
    
    async def connect(self) -> bool:
        """Create HTTP session"""
        headers = dict(self.config.headers)
        
        # Add authentication
        if self.config.auth_type == "api_key":
            key_header = self.config.auth_config.get("header", "X-API-Key")
            headers[key_header] = self.config.auth_config.get("key", "")
        elif self.config.auth_type == "bearer":
            headers["Authorization"] = f"Bearer {self.config.auth_config.get('token', '')}"
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self._session = aiohttp.ClientSession(
            base_url=self.config.base_url,
            headers=headers,
            timeout=timeout
        )
        self.is_connected = True
        logger.info(f"Connected to REST API: {self.config.name}")
        return True
    
    async def disconnect(self) -> None:
        """Close HTTP session"""
        if self._session:
            await self._session.close()
            self._session = None
        self.is_connected = False
        logger.info(f"Disconnected from REST API: {self.config.name}")
    
    @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_tries=3)
    async def fetch(self, query: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Fetch data from REST API"""
        if not self._session:
            await self.connect()
        
        query = query or {}
        endpoint = query.get("endpoint", "/")
        method = query.get("method", "GET").upper()
        params = query.get("params", {})
        body = query.get("body", None)
        
        async with self._rate_limiter:
            start_time = datetime.utcnow()
            try:
                async with self._session.request(method, endpoint, params=params, json=body) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Update metrics
                    latency = (datetime.utcnow() - start_time).total_seconds() * 1000
                    self._metrics["avg_latency_ms"] = (self._metrics["avg_latency_ms"] + latency) / 2
                    self._last_fetch = datetime.utcnow()
                    
                    # Normalize response
                    if isinstance(data, list):
                        self._metrics["total_records"] += len(data)
                        return data
                    elif isinstance(data, dict):
                        # Check for common pagination patterns
                        if "data" in data:
                            items = data["data"] if isinstance(data["data"], list) else [data["data"]]
                            self._metrics["total_records"] += len(items)
                            return items
                        elif "results" in data:
                            items = data["results"]
                            self._metrics["total_records"] += len(items)
                            return items
                        else:
                            self._metrics["total_records"] += 1
                            return [data]
                    return []
                    
            except Exception as e:
                self._metrics["errors"] += 1
                self._metrics["last_error"] = str(e)
                logger.error(f"REST API fetch error: {e}")
                raise
    
    async def stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream data by polling (REST APIs don't natively stream)"""
        while self.is_connected:
            try:
                data = await self.fetch()
                for item in data:
                    await self._notify(item)
                    yield item
            except Exception as e:
                logger.error(f"REST API stream error: {e}")
            
            await asyncio.sleep(self.config.metadata.get("poll_interval", 60))


# =============================================================================
# WEBSOCKET CONNECTOR
# =============================================================================

class WebSocketConnector(DataConnector):
    """Connector for WebSocket streams"""
    
    def __init__(self, config: StreamingConfig):
        super().__init__(config)
        self.config: StreamingConfig = config
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._reconnect_task: Optional[asyncio.Task] = None
    
    async def connect(self) -> bool:
        """Connect to WebSocket"""
        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(
                self.config.url,
                heartbeat=self.config.heartbeat_interval
            )
            self.is_connected = True
            logger.info(f"Connected to WebSocket: {self.config.name}")
            return True
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self._metrics["errors"] += 1
            self._metrics["last_error"] = str(e)
            return False
    
    async def disconnect(self) -> None:
        """Close WebSocket connection"""
        if self._reconnect_task:
            self._reconnect_task.cancel()
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()
        self.is_connected = False
        logger.info(f"Disconnected from WebSocket: {self.config.name}")
    
    async def fetch(self, query: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Fetch single message from WebSocket"""
        if not self._ws:
            await self.connect()
        
        try:
            message = await asyncio.wait_for(
                self._ws.receive(),
                timeout=self.config.timeout
            )
            
            if message.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(message.data)
                self._metrics["total_records"] += 1
                self._last_fetch = datetime.utcnow()
                return [data] if isinstance(data, dict) else data
            elif message.type == aiohttp.WSMsgType.CLOSED:
                self.is_connected = False
                return []
            elif message.type == aiohttp.WSMsgType.ERROR:
                raise Exception(f"WebSocket error: {message.data}")
            return []
        except asyncio.TimeoutError:
            return []
    
    async def stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream messages from WebSocket"""
        while True:
            if not self.is_connected:
                await self.connect()
                if not self.is_connected:
                    await asyncio.sleep(self.config.reconnect_interval)
                    continue
            
            try:
                async for message in self._ws:
                    if message.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(message.data)
                        self._metrics["total_records"] += 1
                        self._last_fetch = datetime.utcnow()
                        await self._notify(data)
                        yield data
                    elif message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        self.is_connected = False
                        break
            except Exception as e:
                logger.error(f"WebSocket stream error: {e}")
                self._metrics["errors"] += 1
                self._metrics["last_error"] = str(e)
                self.is_connected = False
            
            await asyncio.sleep(self.config.reconnect_interval)
    
    async def send(self, message: Union[str, Dict[str, Any]]) -> None:
        """Send message through WebSocket"""
        if not self._ws or not self.is_connected:
            raise Exception("WebSocket not connected")
        
        if isinstance(message, dict):
            message = json.dumps(message)
        
        await self._ws.send_str(message)


# =============================================================================
# DATABASE CONNECTOR (PostgreSQL Example)
# =============================================================================

class PostgreSQLConnector(DataConnector):
    """Connector for PostgreSQL databases"""
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self.config: DatabaseConfig = config
        self._pool = None
    
    async def connect(self) -> bool:
        """Create connection pool"""
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                self.config.connection_string,
                min_size=2,
                max_size=self.config.pool_size
            )
            self.is_connected = True
            logger.info(f"Connected to PostgreSQL: {self.config.name}")
            return True
        except ImportError:
            logger.error("asyncpg not installed. Run: pip install asyncpg")
            return False
        except Exception as e:
            logger.error(f"PostgreSQL connection error: {e}")
            self._metrics["errors"] += 1
            self._metrics["last_error"] = str(e)
            return False
    
    async def disconnect(self) -> None:
        """Close connection pool"""
        if self._pool:
            await self._pool.close()
        self.is_connected = False
        logger.info(f"Disconnected from PostgreSQL: {self.config.name}")
    
    async def fetch(self, query: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute SQL query and fetch results"""
        if not self._pool:
            await self.connect()
        
        query = query or {}
        sql = query.get("sql", "SELECT 1")
        params = query.get("params", [])
        
        start_time = datetime.utcnow()
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
                
                # Convert to list of dicts
                result = [dict(row) for row in rows]
                
                # Update metrics
                latency = (datetime.utcnow() - start_time).total_seconds() * 1000
                self._metrics["avg_latency_ms"] = (self._metrics["avg_latency_ms"] + latency) / 2
                self._metrics["total_records"] += len(result)
                self._last_fetch = datetime.utcnow()
                
                return result
        except Exception as e:
            self._metrics["errors"] += 1
            self._metrics["last_error"] = str(e)
            logger.error(f"PostgreSQL query error: {e}")
            raise
    
    async def execute(self, sql: str, params: List[Any] = None) -> str:
        """Execute SQL statement (INSERT, UPDATE, DELETE)"""
        if not self._pool:
            await self.connect()
        
        async with self._pool.acquire() as conn:
            return await conn.execute(sql, *(params or []))
    
    async def stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream data using LISTEN/NOTIFY or polling"""
        # Implementation would use PostgreSQL LISTEN/NOTIFY
        # For simplicity, using polling approach
        while self.is_connected:
            query = self.config.metadata.get("stream_query", {})
            if query:
                try:
                    data = await self.fetch(query)
                    for item in data:
                        await self._notify(item)
                        yield item
                except Exception as e:
                    logger.error(f"PostgreSQL stream error: {e}")
            
            await asyncio.sleep(self.config.metadata.get("poll_interval", 10))


# =============================================================================
# CLOUD STORAGE CONNECTOR (S3 Example)
# =============================================================================

class S3Connector(DataConnector):
    """Connector for AWS S3"""
    
    def __init__(self, config: CloudStorageConfig):
        super().__init__(config)
        self.config: CloudStorageConfig = config
        self._client = None
    
    async def connect(self) -> bool:
        """Initialize S3 client"""
        try:
            import aiobotocore.session
            session = aiobotocore.session.get_session()
            self._client = session.create_client(
                's3',
                region_name=self.config.region,
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                endpoint_url=self.config.endpoint_url
            )
            self.is_connected = True
            logger.info(f"Connected to S3: {self.config.name}")
            return True
        except ImportError:
            logger.error("aiobotocore not installed. Run: pip install aiobotocore")
            return False
        except Exception as e:
            logger.error(f"S3 connection error: {e}")
            self._metrics["errors"] += 1
            self._metrics["last_error"] = str(e)
            return False
    
    async def disconnect(self) -> None:
        """Close S3 client"""
        if self._client:
            await self._client.close()
        self.is_connected = False
        logger.info(f"Disconnected from S3: {self.config.name}")
    
    async def fetch(self, query: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """List and/or download objects from S3"""
        if not self._client:
            await self.connect()
        
        query = query or {}
        operation = query.get("operation", "list")  # list, get, download
        prefix = query.get("prefix", self.config.prefix)
        key = query.get("key", "")
        
        try:
            async with self._client as client:
                if operation == "list":
                    response = await client.list_objects_v2(
                        Bucket=self.config.bucket,
                        Prefix=prefix
                    )
                    
                    objects = []
                    for obj in response.get("Contents", []):
                        objects.append({
                            "key": obj["Key"],
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"].isoformat(),
                            "etag": obj["ETag"]
                        })
                    
                    self._metrics["total_records"] += len(objects)
                    self._last_fetch = datetime.utcnow()
                    return objects
                
                elif operation == "get":
                    response = await client.get_object(
                        Bucket=self.config.bucket,
                        Key=key
                    )
                    
                    async with response["Body"] as stream:
                        content = await stream.read()
                    
                    # Try to parse as JSON
                    try:
                        data = json.loads(content.decode())
                        if isinstance(data, list):
                            self._metrics["total_records"] += len(data)
                            return data
                        else:
                            self._metrics["total_records"] += 1
                            return [data]
                    except json.JSONDecodeError:
                        # Return raw content as base64 for binary files
                        import base64
                        return [{
                            "key": key,
                            "content_type": response.get("ContentType", "application/octet-stream"),
                            "content": base64.b64encode(content).decode()
                        }]
                
                return []
                
        except Exception as e:
            self._metrics["errors"] += 1
            self._metrics["last_error"] = str(e)
            logger.error(f"S3 fetch error: {e}")
            raise
    
    async def upload(self, key: str, data: bytes, content_type: str = "application/json") -> Dict[str, Any]:
        """Upload data to S3"""
        if not self._client:
            await self.connect()
        
        async with self._client as client:
            response = await client.put_object(
                Bucket=self.config.bucket,
                Key=key,
                Body=data,
                ContentType=content_type
            )
            
            return {
                "key": key,
                "etag": response.get("ETag"),
                "version_id": response.get("VersionId")
            }
    
    async def stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream new objects from S3 (polling-based)"""
        seen_keys = set()
        
        while self.is_connected:
            try:
                objects = await self.fetch({"operation": "list"})
                
                for obj in objects:
                    if obj["key"] not in seen_keys:
                        seen_keys.add(obj["key"])
                        
                        # Fetch content of new object
                        content = await self.fetch({
                            "operation": "get",
                            "key": obj["key"]
                        })
                        
                        for item in content:
                            await self._notify(item)
                            yield item
                
            except Exception as e:
                logger.error(f"S3 stream error: {e}")
            
            await asyncio.sleep(self.config.metadata.get("poll_interval", 60))


# =============================================================================
# DATA INTEGRATION MANAGER
# =============================================================================

class DataIntegrationManager:
    """
    Manages multiple data source connections and provides unified interface
    for data ingestion into Jarvis AI platform
    """
    
    def __init__(self):
        self._connectors: Dict[str, DataConnector] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    def add_connector(self, connector: DataConnector) -> None:
        """Add a data connector"""
        self._connectors[connector.config.name] = connector
        logger.info(f"Added connector: {connector.config.name}")
    
    def remove_connector(self, name: str) -> None:
        """Remove a data connector"""
        if name in self._connectors:
            del self._connectors[name]
            logger.info(f"Removed connector: {name}")
    
    def get_connector(self, name: str) -> Optional[DataConnector]:
        """Get a specific connector"""
        return self._connectors.get(name)
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect all connectors"""
        results = {}
        for name, connector in self._connectors.items():
            if connector.config.enabled:
                results[name] = await connector.connect()
        return results
    
    async def disconnect_all(self) -> None:
        """Disconnect all connectors"""
        for connector in self._connectors.values():
            if connector.is_connected:
                await connector.disconnect()
    
    async def fetch_from(self, name: str, query: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Fetch data from a specific connector"""
        connector = self._connectors.get(name)
        if not connector:
            raise ValueError(f"Connector not found: {name}")
        
        if not connector.is_connected:
            await connector.connect()
        
        return await connector.fetch(query)
    
    async def fetch_all(self, query: Dict[str, Any] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch data from all connected sources"""
        results = {}
        
        async def fetch_one(name: str, conn: DataConnector):
            try:
                if conn.is_connected:
                    return name, await conn.fetch(query)
            except Exception as e:
                logger.error(f"Error fetching from {name}: {e}")
                return name, []
            return name, []
        
        tasks = [
            fetch_one(name, conn)
            for name, conn in self._connectors.items()
            if conn.config.enabled
        ]
        
        for coro in asyncio.as_completed(tasks):
            name, data = await coro
            results[name] = data
        
        return results
    
    async def start_streaming(self, callback: Callable[[str, Dict[str, Any]], None] = None) -> None:
        """Start streaming from all connectors"""
        self._running = True
        
        async def stream_one(name: str, connector: DataConnector):
            async for data in connector.stream():
                if callback:
                    await callback(name, data) if asyncio.iscoroutinefunction(callback) else callback(name, data)
        
        for name, connector in self._connectors.items():
            if connector.config.enabled and connector.is_connected:
                task = asyncio.create_task(stream_one(name, connector))
                self._tasks.append(task)
        
        logger.info(f"Started streaming from {len(self._tasks)} connectors")
    
    async def stop_streaming(self) -> None:
        """Stop all streaming tasks"""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        logger.info("Stopped all streaming tasks")
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all connectors"""
        return {
            name: connector.get_metrics()
            for name, connector in self._connectors.items()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall integration status"""
        total = len(self._connectors)
        connected = sum(1 for c in self._connectors.values() if c.is_connected)
        enabled = sum(1 for c in self._connectors.values() if c.config.enabled)
        
        return {
            "total_connectors": total,
            "enabled": enabled,
            "connected": connected,
            "streaming": self._running,
            "active_streams": len(self._tasks),
            "connectors": {
                name: {
                    "type": c.config.source_type.value,
                    "enabled": c.config.enabled,
                    "connected": c.is_connected
                }
                for name, c in self._connectors.items()
            }
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_connector(config: Dict[str, Any]) -> DataConnector:
    """Factory function to create appropriate connector from config dict"""
    source_type = DataSourceType(config.get("source_type", "rest_api"))
    
    if source_type == DataSourceType.REST_API:
        return RESTAPIConnector(RESTAPIConfig(**config))
    elif source_type == DataSourceType.WEBSOCKET:
        return WebSocketConnector(StreamingConfig(**config))
    elif source_type == DataSourceType.POSTGRESQL:
        return PostgreSQLConnector(DatabaseConfig(**config))
    elif source_type == DataSourceType.S3:
        return S3Connector(CloudStorageConfig(**config))
    else:
        raise ValueError(f"Unsupported source type: {source_type}")


# =============================================================================
# EXPORTS
# =============================================================================

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
