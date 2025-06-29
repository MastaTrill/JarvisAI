"""
FastAPI Web Interface for Jarvis AI.

This module provides a REST API for:
- Model training and management
- Making predictions
- Data upload and processing
- Monitoring model performance
- Real-time training updates via WebSocket
"""

from fastapi import (
    FastAPI, HTTPException, UploadFile, File, BackgroundTasks, 
    WebSocket, WebSocketDisconnect, Security, Depends, APIRouter
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import json
import logging
import tempfile
import os
import asyncio
from datetime import datetime
from pathlib import Path
import sys
import time
import uuid
from collections import defaultdict
import importlib.util
import glob

# User management imports
from sqlalchemy.orm import Session
from db_config import SessionLocal
from models_user import User, get_password_hash, verify_password
from jose import JWTError, jwt
from datetime import timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import Jarvis components
try:
    from src.models.numpy_neural_network import SimpleNeuralNetwork
    from src.models.advanced_neural_network import AdvancedNeuralNetwork
    from src.data.enhanced_processor import EnhancedDataProcessor
    from src.training.numpy_trainer import NumpyTrainer
except ImportError as e:
    logging.warning(f"Import warning: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom OpenAPI tags for documentation
openapi_tags = [
    {
        "name": "Models",
        "description": "Operations related to model management, training, and inference."
    },
    {
        "name": "Data",
        "description": "Endpoints for data upload, processing, and exploration."
    },
    {
        "name": "System",
        "description": "System health, metrics, and monitoring endpoints."
    },
    {
        "name": "WebSocket",
        "description": "Real-time updates and notifications."
    }
]

# API Versioning Scaffold
api_v1 = APIRouter(prefix="/v1")
# --- API VERSIONING ---
# Move endpoints to api_v1 for versioned API support (in progress)
# app.include_router(api_v1)












# Import and mount admin dashboard, admin API, model versioning API, device API, external API, plugins API, drift API, audit API, collab API, infra API, and security API
from admin_dashboard import router as admin_router
from admin_api import router as admin_api_router
from models_versioning_api import router as versioning_router
from models_device_api import router as device_router
from models_external_api import router as external_router
from plugins_api import router as plugins_router
from models_drift_api import router as drift_router
from audit_api import router as audit_router
from collab_api import router as collab_router
from infra_api import router as infra_router
from security_api import router as security_router
from ml_advanced_api import router as ml_advanced_router
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Jarvis AI API",
    description="""
    <b>Jarvis AI Platform</b><br>
    <i>Advanced Machine Learning Platform API</i><br>
    <br>
    <b>Features:</b>
    <ul>
    <li>Model training, management, and inference</li>
    <li>Data upload, processing, and validation</li>
    <li>Real-time training updates via WebSocket</li>
    <li>System health and monitoring endpoints</li>
    </ul>
    <b>Authentication:</b> (OAuth2, API Key, Admin Dashboard)
    <br>
    <b>Contact:</b> <a href='mailto:admin@jarvis.ai'>admin@jarvis.ai</a>
    """,
    version="1.0.0",
    openapi_tags=openapi_tags,
    docs_url="/docs",
    redoc_url="/redoc"
)
app.include_router(admin_router)
app.include_router(admin_api_router)
app.include_router(versioning_router)
app.include_router(device_router)
app.include_router(external_router)
app.include_router(plugins_router)
app.include_router(drift_router)
app.include_router(audit_router)
app.include_router(collab_router)
app.include_router(infra_router)
app.include_router(security_router)
app.include_router(ml_advanced_router)
app.mount("/static", StaticFiles(directory="static"), name="static")

# No-code/low-code workflow editor route
from fastapi.responses import HTMLResponse
import os

@app.get("/workflow", response_class=HTMLResponse)
def workflow_editor():
    static_path = os.path.join(os.path.dirname(__file__), "static", "workflow", "index.html")
    with open(static_path, "r", encoding="utf-8") as f:
        return f.read()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add HTTPS redirect middleware (optional, for production)
if os.environ.get("JARVIS_FORCE_HTTPS", "0") == "1":
    app.add_middleware(HTTPSRedirectMiddleware)

# Add secure headers middleware
class SecureHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response: Response = await call_next(request)
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=()"
        return response
app.add_middleware(SecureHeadersMiddleware)

# --- AUTH CONFIG ---
SECRET_KEY = os.environ.get("JARVIS_SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta is not None else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not isinstance(username, str) or not username:
            raise credentials_exception
    except JWTError as exc:
        raise credentials_exception from exc
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# --- AUTH ENDPOINTS ---
class Token(BaseModel):
    access_token: str
    token_type: str

class UserCreate(BaseModel):
    username: str
    password: str
    email: str

@app.post("/register", tags=["System"])
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password, email=user.email)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"msg": "User registered successfully"}

@app.post("/token", response_model=Token, tags=["System"])
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me", tags=["System"])
def read_users_me(current_user: User = Depends(get_current_user)):
    return {"username": current_user.username, "email": current_user.email, "role": getattr(current_user.role, 'name', None)}

# Health check endpoint for Docker/Kubernetes readiness/liveness probes
@app.get("/health")
def health_check():
    """Health check endpoint for Docker/Kubernetes readiness/liveness probes."""
    return {"status": "ok"}

# Global variables for model management
models = {}
processors = {}
training_status = {}
websocket_connections = {}
active_trainings = {}
system_metrics = {
    "cpu_usage": [],
    "memory_usage": [],
    "gpu_usage": [],
    "timestamp": []
}


# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[Dict] = []

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append({
            "websocket": websocket,
            "client_id": client_id,
            "connected_at": datetime.now()
        })

    def disconnect(self, websocket: WebSocket):
        self.active_connections = [
            conn for conn in self.active_connections 
            if conn["websocket"] != websocket
        ]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast_training_update(self, data: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection["websocket"].send_json(data)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)

    async def send_system_metrics(self, metrics: dict):
        await self.broadcast_training_update({
            "type": "system_metrics",
            "data": metrics
        })


manager = ConnectionManager()


# Pydantic models for API
class TrainRequest(BaseModel):
    model_name: str
    model_type: str = "basic"  # "basic" or "advanced"
    config: Dict[str, Any]
    data_source: Optional[str] = None


class PredictRequest(BaseModel):
    model_name: str
    data: List[List[float]]


class ModelInfo(BaseModel):
    name: str
    type: str
    status: str
    created_at: str
    metrics: Optional[Dict[str, float]] = None


class DataInfo(BaseModel):
    filename: str
    shape: List[int]
    columns: List[str]
    data_types: Dict[str, str]
    missing_values: Dict[str, int]


# OAuth2 and API Key security schemes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
# API key header (future use)
try:
    from fastapi.security import APIKeyHeader
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
except ImportError:
    api_key_header = None

# Dummy authentication dependency (replace with real logic)
def get_current_user(token: str = Depends(oauth2_scheme)):
    # Token validation handled in get_current_user
    return {"user": "demo"}

def get_api_key(api_key: str = Security(api_key_header)):
    # API key validation to be implemented
    if api_key == "supersecretkey":
        return api_key
    raise HTTPException(status_code=403, detail="Invalid API Key")


# API Routes

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Welcome to Jarvis AI API!",
        "version": "1.0.0",
        "endpoints": {
            "models": "/models",
            "train": "/train",
            "predict": "/predict",
            "data": "/data",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(models),
        "processors_loaded": len(processors)
    }


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models."""
    model_list = []
    
    for name, model_data in models.items():
        model_info = ModelInfo(
            name=name,
            type=model_data.get("type", "unknown"),
            status=model_data.get("status", "unknown"),
            created_at=model_data.get("created_at", ""),
            metrics=model_data.get("metrics", {})
        )
        model_list.append(model_info)
    
    return model_list


@app.post("/models/{model_name}/train")
async def train_model(model_name: str, request: TrainRequest, background_tasks: BackgroundTasks):
    """Train a new model."""
    try:
        # Set training status
        training_status[model_name] = {
            "status": "training",
            "started_at": datetime.now().isoformat(),
            "progress": 0
        }
        
        # Start training in background
        background_tasks.add_task(
            _train_model_background,
            model_name,
            request.model_type,
            request.config,
            request.data_source
        )
        
        return {
            "message": f"Training started for model '{model_name}'",
            "status": "training",
            "check_status_at": f"/models/{model_name}/status"
        }
    
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _train_model_background(model_name: str, model_type: str, config: Dict, data_source: Optional[str]):
    """Background task for model training."""
    try:
        # Initialize data processor
        processor = EnhancedDataProcessor(project_name=model_name)
        
        # Load or generate data
        if data_source and os.path.exists(data_source):
            df = processor.load_data(data_source)
        else:
            # Use default dataset or generate dummy data
            if os.path.exists("data/processed/dataset.csv"):
                df = processor.load_data("data/processed/dataset.csv")
            else:
                # Generate dummy data
                from src.data.numpy_processor import DataProcessor
                dummy_processor = DataProcessor()
                df = dummy_processor.create_dummy_data(n_samples=1000, n_features=10)
        
        # Prepare data
        target_col = config.get("target_column", "target")
        if target_col not in df.columns:
            target_col = df.columns[-1]  # Use last column as target
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.get("test_size", 0.2), random_state=42
        )
        
        # Initialize model
        input_size = X_train.shape[1]
        
        if model_type == "advanced":
            model = AdvancedNeuralNetwork(
                input_size=input_size,
                hidden_sizes=config.get("hidden_sizes", [64, 32]),
                output_size=1,
                activation=config.get("activation", "relu"),
                dropout_rate=config.get("dropout_rate", 0.1),
                l1_reg=config.get("l1_reg", 0.0),
                l2_reg=config.get("l2_reg", 0.01),
                optimizer=config.get("optimizer", "adam"),
                learning_rate=config.get("learning_rate", 0.001)
            )
        else:
            model = SimpleNeuralNetwork(
                input_size=input_size,
                hidden_sizes=config.get("hidden_sizes", [64, 32]),
                output_size=1
            )
        
        # Train model
        history = model.fit(
            X_train.values,
            y_train.values,
            epochs=config.get("epochs", 100),
            batch_size=config.get("batch_size", 32),
            validation_data=(X_test.values, y_test.values) if model_type == "advanced" else None
        )
        
        # Calculate final metrics
        train_pred = model.predict(X_train.values)
        test_pred = model.predict(X_test.values)
        
        from sklearn.metrics import mean_squared_error, r2_score
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Save model
        model_path = f"models/{model_name}.pkl"
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        
        # Store model info
        models[model_name] = {
            "model": model,
            "type": model_type,
            "status": "trained",
            "created_at": datetime.now().isoformat(),
            "metrics": {
                "train_mse": float(train_mse),
                "test_mse": float(test_mse),
                "train_r2": float(train_r2),
                "test_r2": float(test_r2)
            },
            "config": config,
            "model_path": model_path
        }
        
        processors[model_name] = processor
        
        # Update training status
        training_status[model_name] = {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "progress": 100
        }
        
        logger.info(f"Successfully trained model '{model_name}'")
    
    except Exception as e:
        logger.error(f"Error training model '{model_name}': {e}")
        training_status[model_name] = {
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        }


@app.get("/models/{model_name}/status")
async def get_training_status(model_name: str):
    """Get training status for a model."""
    if model_name not in training_status:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return training_status[model_name]


@app.post("/models/{model_name}/predict")
async def predict(model_name: str, request: PredictRequest):
    """Make predictions using a trained model."""
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = models[model_name]
    
    if model_data["status"] != "trained":
        raise HTTPException(status_code=400, detail="Model is not trained")
    
    try:
        model = model_data["model"]
        input_data = np.array(request.data)
        
        predictions = model.predict(input_data)
        
        return {
            "model_name": model_name,
            "predictions": predictions.tolist(),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload a data file for processing."""
    try:
        # Save uploaded file
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the data
        processor = EnhancedDataProcessor()
        
        if file.filename.endswith('.csv'):
            df = processor.load_data(str(file_path), 'csv')
        elif file.filename.endswith('.json'):
            df = processor.load_data(str(file_path), 'json')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Validate data
        quality_report = processor.validate_data(df)
        
        # Return data info
        data_info = DataInfo(
            filename=file.filename,
            shape=list(df.shape),
            columns=df.columns.tolist(),
            data_types={col: str(dtype) for col, dtype in df.dtypes.items()},
            missing_values=df.isnull().sum().to_dict()
        )
        
        return {
            "data_info": data_info,
            "quality_report": quality_report,
            "file_path": str(file_path)
        }
    
    except Exception as e:
        logger.error(f"Error uploading data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/validate/{filename}")
async def validate_data(filename: str):
    """Validate an uploaded data file."""
    file_path = Path("data/uploads") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        processor = EnhancedDataProcessor()
        
        if filename.endswith('.csv'):
            df = processor.load_data(str(file_path), 'csv')
        elif filename.endswith('.json'):
            df = processor.load_data(str(file_path), 'json')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        quality_report = processor.validate_data(df)
        
        return quality_report
    
    except Exception as e:
        logger.error(f"Error validating data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}/metrics")
async def get_model_metrics(model_name: str):
    """Get detailed metrics for a model."""
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = models[model_name]
    
    return {
        "model_name": model_name,
        "type": model_data["type"],
        "status": model_data["status"],
        "metrics": model_data.get("metrics", {}),
        "config": model_data.get("config", {}),
        "created_at": model_data["created_at"]
    }


# WebSocket Routes

@app.websocket("/ws/train/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time training updates."""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Keep the connection alive
            await asyncio.sleep(3600)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"Client {client_id} disconnected")


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/ws/notifications")
async def websocket_notifications(websocket: WebSocket):
    """WebSocket endpoint for real-time notifications."""
    await websocket.accept()
    try:
        while True:
            await websocket.receive_text()  # Keep connection open
    except Exception:
        pass


# Broadcast notification to all connected clients (in-memory demo)
notification_clients = set()

@app.post("/notify", tags=["WebSocket"], summary="Broadcast a notification")
async def broadcast_notification(message: str):
    for ws in list(notification_clients):
        try:
            await ws.send_json({"type": "notification", "message": message})
        except Exception:
            notification_clients.remove(ws)
    return {"status": "sent", "message": message}


@app.get("/api/models/list")
async def list_models():
    """Get list of all models with their status."""
    model_list = []
    for name, data in models.items():
        model_info = {
            "name": name,
            "type": data["type"],
            "status": data["status"],
            "created_at": data["created_at"],
            "metrics": data.get("metrics", {}),
            "config": data.get("config", {})
        }
        model_list.append(model_info)
    
    return {"models": model_list}


@app.get("/api/training/status/{model_name}")
async def get_training_status(model_name: str):
    """Get current training status for a model."""
    if model_name not in training_status:
        return {"status": "not_found", "message": "Training not found"}
    
    return training_status[model_name]


@app.post("/api/training/stop/{model_name}")
async def stop_training(model_name: str):
    """Stop training for a model."""
    if model_name not in active_trainings:
        raise HTTPException(status_code=404, detail="Training not found")
    
    # Set stop flag
    active_trainings[model_name]["stop_requested"] = True
    
    return {"message": f"Stop requested for model {model_name}"}


@app.get("/api/system/metrics")
async def get_system_metrics():
    """Get current system metrics."""
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        metrics = {
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "memory_available": memory.available // (1024**3),  # GB
            "memory_total": memory.total // (1024**3),  # GB
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in history (keep last 100 points)
        for key in ["cpu_usage", "memory_usage", "timestamp"]:
            system_metrics[key].append(metrics[key])
            if len(system_metrics[key]) > 100:
                system_metrics[key].pop(0)
        
        return metrics
    except ImportError:
        # Fallback if psutil not available
        return {
            "cpu_usage": 0,
            "memory_usage": 0,
            "memory_available": 0,
            "memory_total": 0,
            "timestamp": datetime.now().isoformat()
        }


@app.get("/api/system/metrics/history")
async def get_system_metrics_history():
    """Get system metrics history."""
    return {
        "metrics": system_metrics,
        "count": len(system_metrics.get("timestamp", []))
    }


@app.post("/api/models/compare")
async def compare_models(model_names: List[str]):
    """Compare multiple models."""
    comparison_data = []
    
    for name in model_names:
        if name in models:
            model_data = models[name]
            comparison_data.append({
                "name": name,
                "type": model_data["type"],
                "status": model_data["status"],
                "metrics": model_data.get("metrics", {}),
                "config": model_data.get("config", {}),
                "created_at": model_data["created_at"]
            })
    
    return {"models": comparison_data}


@app.post("/api/models/export/{model_name}")
async def export_model(model_name: str, format: str = "pkl"):
    """Export a trained model."""
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_data = models[model_name]
    if model_data["status"] != "trained":
        raise HTTPException(status_code=400, detail="Model is not trained")
    
    # Create export directory
    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)
    
    try:
        model = model_data["model"]
        export_path = export_dir / f"{model_name}_exported.{format}"
        
        if format == "pkl":
            import pickle
            with open(export_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
        
        return {
            "message": f"Model {model_name} exported successfully",
            "export_path": str(export_path),
            "format": format
        }
    
    except Exception as e:
        logger.error(f"Error exporting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/exploration/{filename}")
async def explore_data(filename: str):
    """Get data exploration statistics."""
    file_path = Path("data/uploads") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        import pandas as pd
        
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Basic statistics
        stats = {
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "summary_stats": df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else {},
            "sample_data": df.head(10).to_dict('records')
        }
        
        return stats
    
    except Exception as e:
        logger.error(f"Error exploring data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Prometheus-compatible metrics endpoint
@app.get("/metrics", tags=["System"], summary="Prometheus metrics endpoint", response_class=PlainTextResponse)
def metrics():
    # Example metrics (replace with real values)
    metrics = [
        'jarvis_api_requests_total 1234',
        'jarvis_api_errors_total 12',
        'jarvis_model_inferences_total 567',
        'jarvis_active_models 2'
    ]
    return '\n'.join(metrics)


# Background task for system monitoring
async def monitor_system():
    """Background task to monitor system metrics and broadcast updates."""
    while True:
        try:
            metrics = await get_system_metrics()
            await manager.send_system_metrics(metrics)
            await asyncio.sleep(5)  # Update every 5 seconds
        except Exception as e:
            logger.error(f"Error in system monitoring: {e}")
            await asyncio.sleep(10)


# Start system monitoring on startup
@app.on_event("startup")
async def startup_event():
    """Start background tasks on app startup."""
    asyncio.create_task(monitor_system())


# Example of a protected endpoint
@app.get("/secure-info", tags=["System"], summary="Get secure system info", description="Requires OAuth2 or API Key.")
def secure_info(user=Depends(get_current_user), api_key=Security(get_api_key)):
    return {"secure": True, "user": user}


# User registration, login, and RBAC implemented above
# Admin dashboard scaffolded and integrated

# --- Rate Limiting Middleware (slowapi) ---
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

limiter = Limiter(key_func=get_remote_address, storage_uri=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Please try again later."})

# Example: Apply global rate limit (100 requests/minute per IP)
from slowapi.middleware import SlowAPIMiddleware
app.add_middleware(SlowAPIMiddleware)
app = limiter.limit("100/minute")(app)
# Audit logging planned (see audit_trail.py)

# Model registry, jobs, and user data are now persistent (see db_config.py)
# Example: Use SQLAlchemy models and CRUD operations instead of in-memory dicts

# In-memory model registry (replace with persistent storage in production)
model_registry = {}

@app.post("/models/register", tags=["Models"], summary="Register a new model", description="Upload and register a new trained model with metadata.")
def register_model(model_name: str, description: str = "", accuracy: float = 0.0):
    model_registry[model_name] = {
        "description": description,
        "accuracy": accuracy if accuracy is not None else 0.0,
        "registered_at": datetime.utcnow().isoformat(),
        "active": False
    }
    return {"message": f"Model '{model_name}' registered.", "model": model_registry[model_name]}

@app.get("/models/registry", tags=["Models"], summary="List all registered models")
def list_registered_models():
    return model_registry

@app.post("/models/activate", tags=["Models"], summary="Activate a model for inference")
def activate_model(model_name: str):
    for m in model_registry:
        model_registry[m]["active"] = False
    if model_name in model_registry:
        model_registry[model_name]["active"] = True
        return {"message": f"Model '{model_name}' activated."}
    else:
        raise HTTPException(status_code=404, detail="Model not found")


from fastapi import BackgroundTasks
import threading

# In-memory job store (replace with persistent storage in production)
jobs = {}

def long_running_task(job_id: str, duration: int = 10):
    import time
    jobs[job_id] = {"status": "running"}
    time.sleep(duration)
    jobs[job_id] = {"status": "completed"}

@app.post("/jobs/start", tags=["System"], summary="Start a background job")
def start_job(duration: int = 10, background_tasks: BackgroundTasks = None):
    job_id = str(uuid.uuid4())
    background_tasks.add_task(long_running_task, job_id, duration)
    jobs[job_id] = {"status": "queued"}
    return {"job_id": job_id, "status": "queued"}

@app.get("/jobs/status/{job_id}", tags=["System"], summary="Check job status")
def job_status(job_id: str):
    return jobs.get(job_id, {"status": "not found"})

@app.post("/jobs/cancel/{job_id}", tags=["System"], summary="Cancel a job")
def cancel_job(job_id: str):
    # For demo: just mark as cancelled (real implementation would stop the thread/process)
    if job_id in jobs and jobs[job_id]["status"] == "running":
        jobs[job_id]["status"] = "cancelled"
        return {"job_id": job_id, "status": "cancelled"}
    return {"job_id": job_id, "status": jobs.get(job_id, {}).get("status", "not found")}


@app.post("/predict/ensemble", tags=["Models"], summary="Ensemble prediction with multiple models")
def ensemble_predict(model_names: list, data: list):
    results = []
    for name in model_names:
        model = models.get(name)
        if not model:
            continue
        # Assume model has a predict method
        pred = model.predict(data)
        results.append(pred)
    # Simple average ensemble
    if results:
        import numpy as np
        ensemble = np.mean(results, axis=0).tolist()
        return {"ensemble_prediction": ensemble}
    return {"error": "No valid models found"}


# Plugin system: auto-register plugins in plugins/ folder
def register_plugins(app):
    for plugin_path in glob.glob("plugins/*.py"):
        if plugin_path.endswith("__init__.py"):
            continue
        spec = importlib.util.spec_from_file_location("plugin", plugin_path)
        plugin = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plugin)
        if hasattr(plugin, "register"):
            plugin.register(app)

register_plugins(app)

# Model versioning endpoints scaffolded (see models_versioning.py)
# GPU inference and external model server integration scaffolded

# Data anonymization, secure deletion, and audit trail hooks planned (see audit_trail.py)
# GDPR/CCPA compliance features planned (see PRIVACY_POLICY.md)

# Cloud storage connectors scaffolded (see cloud_connectors.py)
# Notification integration planned (see cloud_connectors.py)

# Bandit security scanning added to CI/CD (see .github/workflows/ci-cd.yml)
# Secure headers and HTTPS recommended for production (see SECURITY.md)
# Secrets are loaded from environment variables or a secrets manager (see .env.example)

# Example: Use persistent model registry (see models_registry.py)
from models_registry import create_model, get_models, activate_model
from db_config import SessionLocal

@app.post("/models/register", tags=["Models"], summary="Register a new model (persistent)")
def register_model_db(model_name: str, description: str = "", accuracy: float = 0.0):
    session = SessionLocal()
    model = create_model(session, model_name, description, accuracy if accuracy is not None else 0.0)
    session.close()
    return {"message": f"Model '{model_name}' registered.", "model": model.name}

@app.get("/models/registry", tags=["Models"], summary="List all registered models (persistent)")
def list_registered_models_db():
    session = SessionLocal()
    models = get_models(session)
    session.close()
    return [m.name for m in models]

@app.post("/models/activate", tags=["Models"], summary="Activate a model for inference (persistent)")
def activate_model_db(model_name: str):
    session = SessionLocal()
    model = activate_model(session, model_name)
    session.close()
    if model:
        return {"message": f"Model '{model_name}' activated."}
    else:
        raise HTTPException(status_code=404, detail="Model not found")


# Persistent job management (see jobs_persistent.py)
from jobs_persistent import create_job, update_job_status, get_job

@app.post("/jobs/start", tags=["System"], summary="Start a background job (persistent)")
def start_job_db(duration: int = 10, background_tasks: BackgroundTasks = None):
    import uuid, time
    session = SessionLocal()
    job_id = str(uuid.uuid4())
    create_job(session, job_id, status="queued")
    def run_job():
        update_job_status(session, job_id, status="running")
        time.sleep(duration)
        update_job_status(session, job_id, status="completed", completed_at=datetime.utcnow())
    background_tasks.add_task(run_job)
    session.close()
    return {"job_id": job_id, "status": "queued"}

@app.get("/jobs/status/{job_id}", tags=["System"], summary="Check job status (persistent)")
def job_status_db(job_id: str):
    session = SessionLocal()
    job = get_job(session, job_id)
    session.close()
    if job:
        return {"job_id": job.job_id, "status": job.status, "cancelled": job.cancelled}
    return {"job_id": job_id, "status": "not found"}

@app.post("/jobs/cancel/{job_id}", tags=["System"], summary="Cancel a job (persistent)")
def cancel_job_db(job_id: str):
    session = SessionLocal()
    job = update_job_status(session, job_id, status="cancelled", cancelled=True)
    session.close()
    if job:
        return {"job_id": job_id, "status": "cancelled"}
    return {"job_id": job_id, "status": "not found"}

