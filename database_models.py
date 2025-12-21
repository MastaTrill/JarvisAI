"""
JarvisAI Database Models
SQLAlchemy models for data persistence
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


class User(Base):
    """User model for authentication and authorization"""
    __tablename__ = 'users'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(String(20), default='user')  # creator, family, admin, user
    is_active = Column(Boolean, default=True)
    api_key = Column(String(100), unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    chat_histories = relationship("ChatHistory", back_populates="user", cascade="all, delete-orphan")
    model_runs = relationship("ModelRun", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(username='{self.username}', role='{self.role}')>"


class ChatHistory(Base):
    """Chat history for AI agent conversations"""
    __tablename__ = 'chat_histories'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    session_id = Column(String(100), index=True)
    message = Column(Text, nullable=False)
    response = Column(Text)
    is_user_message = Column(Boolean, default=True)
    quantum_enhanced = Column(Boolean, default=False)
    temporal_context = Column(JSON)  # Store temporal analysis data
    sentiment_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User", back_populates="chat_histories")
    
    def __repr__(self):
        return f"<ChatHistory(user_id='{self.user_id}', session='{self.session_id}')>"


class ModelRun(Base):
    """Model training and inference runs"""
    __tablename__ = 'model_runs'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    model_name = Column(String(100), nullable=False, index=True)
    model_type = Column(String(50))  # quantum, neural_network, cv, temporal
    model_version = Column(String(20))
    status = Column(String(20), default='running')  # running, completed, failed
    
    # Training metrics
    accuracy = Column(Float)
    loss = Column(Float)
    f1_score = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    
    # Configuration
    hyperparameters = Column(JSON)
    dataset_info = Column(JSON)
    
    # Performance
    training_time = Column(Float)  # seconds
    inference_time = Column(Float)  # seconds
    memory_usage = Column(Float)  # MB
    
    # Metadata
    mlflow_run_id = Column(String(100))
    model_path = Column(String(500))
    error_message = Column(Text)
    notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="model_runs")
    metrics = relationship("PerformanceMetric", back_populates="model_run", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ModelRun(model_name='{self.model_name}', status='{self.status}')>"


class PerformanceMetric(Base):
    """System and model performance metrics"""
    __tablename__ = 'performance_metrics'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_run_id = Column(String(36), ForeignKey('model_runs.id'))
    
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20))  # ops/sec, ms, MB, %
    
    # System metrics
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    disk_io = Column(Float)
    network_io = Column(Float)
    gpu_usage = Column(Float)
    
    # Application metrics
    request_count = Column(Integer)
    error_count = Column(Integer)
    avg_response_time = Column(Float)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    model_run = relationship("ModelRun", back_populates="metrics")
    
    def __repr__(self):
        return f"<PerformanceMetric(name='{self.metric_name}', value={self.metric_value})>"


class QuantumState(Base):
    """Quantum processor state snapshots"""
    __tablename__ = 'quantum_states'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    state_type = Column(String(50), nullable=False)  # superposition, entanglement, measurement
    num_qubits = Column(Integer)
    num_states = Column(Integer)
    
    # State data
    amplitudes = Column(JSON)  # Complex amplitude values
    probabilities = Column(JSON)  # Measurement probabilities
    entanglement_pairs = Column(JSON)
    
    # Metrics
    fidelity = Column(Float)
    coherence_time = Column(Float)
    operation_count = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<QuantumState(type='{self.state_type}', qubits={self.num_qubits})>"


class TemporalPattern(Base):
    """Temporal analysis patterns and predictions"""
    __tablename__ = 'temporal_patterns'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    pattern_name = Column(String(100), nullable=False, index=True)
    pattern_type = Column(String(50))  # trend, seasonal, cyclical, anomaly
    
    # Pattern data
    time_series_data = Column(JSON)
    detected_patterns = Column(JSON)
    anomalies = Column(JSON)
    
    # Predictions
    forecast_data = Column(JSON)
    confidence_interval = Column(JSON)
    accuracy = Column(Float)
    
    # Metadata
    timeframe_start = Column(DateTime)
    timeframe_end = Column(DateTime)
    data_points = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<TemporalPattern(name='{self.pattern_name}', type='{self.pattern_type}')>"


class AgentTask(Base):
    """AI agent autonomous tasks"""
    __tablename__ = 'agent_tasks'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_name = Column(String(200), nullable=False)
    task_type = Column(String(50))  # analysis, generation, automation, monitoring
    status = Column(String(20), default='pending')  # pending, running, completed, failed
    priority = Column(Integer, default=5)  # 1-10
    
    # Task details
    description = Column(Text)
    input_data = Column(JSON)
    output_data = Column(JSON)
    error_message = Column(Text)
    
    # Execution
    agent_id = Column(String(100))
    execution_time = Column(Float)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Scheduling
    scheduled_at = Column(DateTime)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<AgentTask(name='{self.task_name}', status='{self.status}')>"
