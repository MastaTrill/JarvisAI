"""
Jarvis AI API main module.

Provides endpoints for model management, training, data upload, system monitoring.
"""

# Standard library imports
import sys
import os
import glob
import importlib.util
import logging
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Third-party imports
import numpy as np
import requests
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    BackgroundTasks,
    WebSocket,
    WebSocketDisconnect,
    Security,
    Depends,
    APIRouter,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm, APIKeyHeader
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# First-party imports
from cloud_connectors import upload_to_cloud, download_from_cloud
from db_config import SessionLocal
from database import get_db
from models_user import User, get_password_hash
from authentication import verify_password
from auth_helpers import get_current_user
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
from automation_api import router as automation_router
from src.interpretability.model_explainer import ModelInterpreter
from models_registry import create_model, get_models, activate_model
from jobs_persistent import create_job, update_job_status, get_job

try:
    from src.models.numpy_neural_network import SimpleNeuralNetwork
    from src.models.advanced_neural_network import AdvancedNeuralNetwork
    from src.data.enhanced_processor import EnhancedDataProcessor
except ImportError as e:
    logging.warning("Import warning: %s", e)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Custom OpenAPI tags for documentation
openapi_tags = [
    {
        "name": "Models",
        "description": "Operations related to model management, training, and inference.",
    },
    {
        "name": "Data",
        "description": "Endpoints for data upload, processing, and exploration.",
    },
    {
        "name": "System",
        "description": "System health, metrics, and monitoring endpoints.",
    },
    {"name": "WebSocket", "description": "Real-time updates and notifications."},
]

# API Versioning Scaffold
api_v1 = APIRouter(prefix="/v1")
# --- API VERSIONING ---
# Move endpoints to api_v1 for versioned API support (in progress)
# app.include_router(api_v1)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan: startup and shutdown logic."""
    # --- Startup ---
    from database_models import Base as DBBase
    from db_config import engine as db_engine, Base as ConfigBase
    from database import engine as app_engine

    DBBase.metadata.create_all(bind=db_engine)
    ConfigBase.metadata.create_all(bind=db_engine)
    DBBase.metadata.create_all(bind=app_engine)
    asyncio.create_task(monitor_system())
    yield
    # --- Shutdown (cleanup if needed) ---


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
    redoc_url="/redoc",
    lifespan=lifespan,
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
app.include_router(automation_router)
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static",
)

# --- Rate Limiting Middleware (slowapi) ---
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=os.getenv("REDIS_URL", "memory://"),
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(_request, _exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."},
    )


app.add_middleware(SlowAPIMiddleware)


# Serve dashboard at root URL
@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the dashboard HTML page."""
    static_path = os.path.join(
        os.path.dirname(__file__), "static", "dashboard", "index.html"
    )
    with open(static_path, "r", encoding="utf-8") as f:
        html = f.read()
    # Unique marker for debug
    html += "<!-- JARVIS_DEBUG_MARKER_2026 -->"
    return html


# --- XAI/Interpretability Endpoints ---


@app.get("/api/xai/global/{model_name}")
async def get_global_feature_importance(model_name: str):
    """Get global feature importance (SHAP-style) for a model."""
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    model_data = models[model_name]
    model = model_data["model"]
    processor = processors.get(model_name)
    if not processor:
        raise HTTPException(status_code=404, detail="No processor for model")
    # Use training data for explanations
    x_data = (
        processor.get_training_data()
        if hasattr(processor, "get_training_data")
        else None
    )
    if x_data is None:
        raise HTTPException(
            status_code=400, detail="No training data available for model"
        )
    feature_names = list(x_data.columns) if hasattr(x_data, "columns") else []
    interpreter = ModelInterpreter(model, feature_names)
    shap_result = interpreter.explain_with_shap(x_data.values)
    return {
        "feature_names": feature_names,
        "feature_importance": shap_result.get("feature_importance", []),
        "shap_values": shap_result.get("shap_values", []),
    }


@app.get("/api/xai/local/{model_name}")
async def get_local_explanation(model_name: str, instance_idx: int = 0):
    """Get local explanation (LIME-style) for a model and instance."""
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    model_data = models[model_name]
    model = model_data["model"]
    processor = processors.get(model_name)
    if not processor:
        raise HTTPException(status_code=404, detail="No processor for model")
    x_data = (
        processor.get_training_data()
        if hasattr(processor, "get_training_data")
        else None
    )
    if x_data is None:
        raise HTTPException(
            status_code=400, detail="No training data available for model"
        )
    feature_names = list(x_data.columns) if hasattr(x_data, "columns") else []
    interpreter = ModelInterpreter(model, feature_names)
    lime_result = interpreter.explain_with_lime(
        x_data.values, instance_idx=instance_idx
    )
    return lime_result


@app.get("/api/xai/counterfactuals/{model_name}")
async def get_counterfactuals(model_name: str, instance_idx: int = 0):
    """Get counterfactual suggestions for a model and instance."""
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    model_data = models[model_name]
    model = model_data["model"]
    processor = processors.get(model_name)
    if not processor:
        raise HTTPException(status_code=404, detail="No processor for model")
    x_data = (
        processor.get_training_data()
        if hasattr(processor, "get_training_data")
        else None
    )
    if x_data is None:
        raise HTTPException(
            status_code=400, detail="No training data available for model"
        )
    feature_names = list(x_data.columns) if hasattr(x_data, "columns") else []
    interpreter = ModelInterpreter(model, feature_names)
    shap_result = interpreter.explain_with_shap(x_data.values)
    shap_values = shap_result.get("shap_values", [])
    instance = x_data.values[instance_idx]
    counterfactuals = []
    for i, feature in enumerate(feature_names):
        shap_val = (
            shap_values[instance_idx][i] if len(shap_values) > instance_idx else 0
        )
        if shap_val != 0:
            needed_change = -2 * shap_val / abs(shap_val)
            new_value = instance[i] + needed_change
            counterfactuals.append(
                {
                    "feature": feature,
                    "current_value": float(instance[i]),
                    "suggested_value": float(new_value),
                    "change_required": float(needed_change),
                }
            )

    return {"counterfactuals": counterfactuals}


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "JARVIS_CORS_ORIGINS", "http://localhost:3000,http://localhost:8080"
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add HTTPS redirect middleware (optional, for production)
if os.environ.get("JARVIS_FORCE_HTTPS", "0") == "1":
    app.add_middleware(HTTPSRedirectMiddleware)


# Add secure headers middleware
class SecureHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add secure headers to responses."""

    async def dispatch(self, request, call_next):
        response: Response = await call_next(request)
        response.headers["Strict-Transport-Security"] = (
            "max-age=63072000; includeSubDomains; preload"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=()"
        return response


app.add_middleware(SecureHeadersMiddleware)


# --- AUTH CONFIG ---
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


# --- AUTH ENDPOINTS ---
class Token(BaseModel):
    """Token response model."""

    access_token: str
    token_type: str


class UserCreate(BaseModel):
    """User creation model."""

    username: str
    password: str
    email: str


@app.post("/register", tags=["System"])
@limiter.limit("5/minute")
def register(request: Request, user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    _ = request  # required by slowapi rate limiter
    try:
        db_user = db.query(User).filter(User.username == user.username).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Username already registered")
        hashed_password = get_password_hash(user.password)
        new_user = User(
            username=user.username, hashed_password=hashed_password, email=user.email
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        # Audit log
        try:
            from audit_trail import log_audit_event

            log_audit_event(user.username, "register", "user", f"email={user.email}")
        except (ImportError, SQLAlchemyError) as e:
            logger.warning("Audit logging failed: %s", e)
        return {"msg": "User registered successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Registration failed: %s", e)
        raise HTTPException(status_code=400, detail="Unable to register user") from e


@app.post("/token", response_model=Token, tags=["System"])
@limiter.limit("10/minute")
def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """Login and get access token."""
    _ = request  # required by slowapi rate limiter
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, str(user.hashed_password)):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    from authentication import create_access_token

    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/me", tags=["System"])
def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user info."""
    return {
        "username": current_user.username,
        "email": current_user.email,
        "role": getattr(current_user.role, "name", None),
    }


# Global variables for model management
models = {}
processors = {}
training_status = {}
websocket_connections = {}
active_trainings = {}
system_metrics = {"cpu_usage": [], "memory_usage": [], "gpu_usage": [], "timestamp": []}


# WebSocket Connection Manager
class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: List[Dict] = []

    async def connect(self, websocket: WebSocket, client_id: str):
        """Connect a new WebSocket client."""
        await websocket.accept()
        self.active_connections.append(
            {
                "websocket": websocket,
                "client_id": client_id,
                "connected_at": datetime.now(),
            }
        )

    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket client."""
        self.active_connections = [
            conn for conn in self.active_connections if conn["websocket"] != websocket
        ]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a personal message to a WebSocket client."""
        await websocket.send_text(message)

    async def broadcast_training_update(self, data: dict):
        """Broadcast training update to all clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection["websocket"].send_json(data)
            except (ConnectionError, OSError) as e:
                logger.warning("Failed to send training update to client: %s", e)
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)

    async def send_system_metrics(self, metric_data: dict):
        """Send system metrics to all clients."""
        await self.broadcast_training_update(
            {"type": "system_metrics", "data": metric_data}
        )


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


# Export PredictRequest for import in main_api.py
__all__ = ["PredictRequest"]


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


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key: str = Security(api_key_header)):
    """API key validation."""
    expected_key = os.environ.get("JARVIS_API_KEY", "")
    if not expected_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    if api_key == expected_key:
        return api_key
    raise HTTPException(status_code=403, detail="Invalid API Key")


def _safe_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal attacks."""
    # Take only the basename, stripping any directory components
    name = Path(filename).name
    if not name or name.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return name


# API Routes


@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Health check endpoint."""
    _ = request  # required by slowapi rate limiter
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(models),
        "processors_loaded": len(processors),
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
            metrics=model_data.get("metrics", {}),
        )
        model_list.append(model_info)

    return model_list


@app.post("/models/{model_name}/train")
@limiter.limit("5/minute")
async def train_model(
    request: Request,
    model_name: str,
    train_request: TrainRequest,
    background_tasks: BackgroundTasks,
    _current_user: User = Depends(get_current_user),
):
    """Train a model with the provided configuration."""
    _ = request  # required by slowapi rate limiter
    try:
        training_status[model_name] = {
            "status": "training",
            "started_at": datetime.now().isoformat(),
            "progress": 0,
        }

        # Start training in background
        background_tasks.add_task(
            _train_model_background,
            model_name,
            train_request.model_type,
            train_request.config,
            train_request.data_source,
        )

        return {
            "message": f"Training started for model '{model_name}'",
            "status": "training",
            "check_status_at": f"/models/{model_name}/status",
        }

    except Exception as e:
        logger.error("Error starting training: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _train_model_background(
    model_name: str, model_type: str, config: Dict, data_source: Optional[str]
):
    """Background task for model training."""
    try:
        processor = EnhancedDataProcessor(project_name=model_name)
        df = None
        if data_source and os.path.exists(data_source):
            df = processor.load_data(data_source)
        if df is None:
            raise HTTPException(status_code=400, detail="No data found for training")
        target_col = config.get("target_col")
        if not target_col or target_col not in df.columns:
            raise HTTPException(
                status_code=400, detail="Target column not specified or missing"
            )
        y = df[target_col]
        x = df.drop(columns=[target_col])
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=config.get("test_size", 0.2), random_state=42
        )
        input_size = x_train.shape[1]
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
                learning_rate=config.get("learning_rate", 0.001),
            )
        else:
            model = SimpleNeuralNetwork(
                input_size=input_size,
                hidden_sizes=config.get("hidden_sizes", [64, 32]),
                output_size=1,
            )
        model.fit(
            x_train.values,
            y_train.values,
            epochs=config.get("epochs", 100),
            batch_size=config.get("batch_size", 32),
        )
        train_pred = model.predict(x_train.values)
        test_pred = model.predict(x_test.values)
        from sklearn.metrics import mean_squared_error, r2_score

        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        model_path = f"models/{model_name}.pkl"
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        models[model_name] = {
            "model": model,
            "type": model_type,
            "status": "trained",
            "created_at": datetime.now().isoformat(),
            "metrics": {
                "train_mse": float(train_mse),
                "test_mse": float(test_mse),
                "train_r2": float(train_r2),
                "test_r2": float(test_r2),
            },
            "config": config,
            "model_path": model_path,
        }
        processors[model_name] = processor
        training_status[model_name] = {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "progress": 100,
        }
        logger.info("Successfully trained model '%s'", model_name)
    except (ValueError, RuntimeError, TypeError, OSError, ImportError) as e:
        logger.error("Error training model '%s': %s", model_name, e)
        training_status[model_name] = {
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat(),
        }


@app.get("/models/{model_name}/status")
async def get_training_status(model_name: str):
    """Get training status for a model."""
    if model_name not in training_status:
        raise HTTPException(status_code=404, detail="Model not found")

    return training_status[model_name]


@app.post("/models/{model_name}/predict")
@limiter.limit("30/minute")
async def predict(
    request: Request,
    model_name: str,
    predict_request: PredictRequest,
    _current_user: User = Depends(get_current_user),
):
    """Make predictions using a trained model."""
    _ = request  # required by slowapi rate limiter
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")

    model_data = models[model_name]

    if model_data["status"] != "trained":
        raise HTTPException(status_code=400, detail="Model is not trained")

    try:
        model = model_data["model"]
        input_data = np.array(predict_request.data)

        predictions = model.predict(input_data)

        return {
            "model_name": model_name,
            "predictions": predictions.tolist(),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error("Error making predictions: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/data/upload")
@limiter.limit("10/minute")
async def upload_data(
    request: Request,
    file: UploadFile = File(...),
    _current_user: User = Depends(get_current_user),
):
    """Upload a data file for processing."""
    _ = request  # required by slowapi rate limiter
    try:
        # Save uploaded file
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        safe_name = _safe_filename(file.filename)
        file_path = upload_dir / safe_name

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process the data
        processor = EnhancedDataProcessor()

        if file.filename.endswith(".csv"):
            df = processor.load_data(str(file_path), "csv")
        elif file.filename.endswith(".json"):
            df = processor.load_data(str(file_path), "json")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Validate data
        quality_report = processor.validate_data(df)

        # Return data info
        data_info = DataInfo(
            filename=file.filename or "",
            shape=list(df.shape),
            columns=df.columns.tolist(),
            data_types={str(col): str(dtype) for col, dtype in df.dtypes.items()},
            missing_values={
                str(k): int(v) for k, v in df.isnull().sum().to_dict().items()
            },
        )

        return {
            "data_info": data_info,
            "quality_report": quality_report,
            "file_path": str(file_path),
        }

    except Exception as e:
        logger.error("Error uploading data: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/data/validate/{filename}")
async def validate_data(filename: str, _current_user: User = Depends(get_current_user)):
    """Validate an uploaded data file."""
    safe_name = _safe_filename(filename)
    file_path = Path("data/uploads") / safe_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        processor = EnhancedDataProcessor()

        if filename.endswith(".csv"):
            df = processor.load_data(str(file_path), "csv")
        elif filename.endswith(".json"):
            df = processor.load_data(str(file_path), "json")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        quality_report = processor.validate_data(df)

        return quality_report

    except Exception as e:
        logger.error("Error validating data: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


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
        "created_at": model_data["created_at"],
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
        logger.info("Client %s disconnected", client_id)


@app.websocket("/ws/{client_id}")
async def websocket_general_endpoint(websocket: WebSocket, client_id: str):
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
    except (WebSocketDisconnect, ConnectionError, OSError) as e:
        logger.warning("WebSocket connection error: %s", e)


# Broadcast notification to all connected clients (in-memory demo)
notification_clients = set()


@app.post("/notify", tags=["WebSocket"], summary="Broadcast a notification")
async def broadcast_notification(message: str):
    for ws in list(notification_clients):
        try:
            await ws.send_json({"type": "notification", "message": message})
        except (ConnectionError, OSError) as e:
            logger.warning("Failed to send notification to client: %s", e)
            notification_clients.remove(ws)
    return {"status": "sent", "message": message}


@app.get("/api/models/list")
async def list_models_detailed():
    """Get list of all models with their status."""
    model_list = []
    for name, data in models.items():
        model_info = {
            "name": name,
            "type": data["type"],
            "status": data["status"],
            "created_at": data["created_at"],
            "metrics": data.get("metrics", {}),
            "config": data.get("config", {}),
        }
        model_list.append(model_info)

    return {"models": model_list}


@app.get("/api/training/status/{model_name}")
async def get_training_status_api(model_name: str):
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

        current_metrics = {
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "memory_available": memory.available // (1024**3),  # GB
            "memory_total": memory.total // (1024**3),  # GB
            "timestamp": datetime.now().isoformat(),
        }

        # Store in history (keep last 100 points)
        for key in ["cpu_usage", "memory_usage", "timestamp"]:
            system_metrics[key].append(current_metrics[key])
            if len(system_metrics[key]) > 100:
                system_metrics[key].pop(0)

        return current_metrics
    except ImportError:
        # Fallback if psutil not available
        return {
            "cpu_usage": 0,
            "memory_usage": 0,
            "memory_available": 0,
            "memory_total": 0,
            "timestamp": datetime.now().isoformat(),
        }


@app.get("/api/system/metrics/history")
async def get_system_metrics_history():
    """Get system metrics history."""
    return {
        "metrics": system_metrics,
        "count": len(system_metrics.get("timestamp", [])),
    }


@app.post("/api/models/compare")
async def compare_models(model_names: List[str]):
    """Compare multiple models."""
    comparison_data = []

    for name in model_names:
        if name in models:
            model_data = models[name]
            comparison_data.append(
                {
                    "name": name,
                    "type": model_data["type"],
                    "status": model_data["status"],
                    "metrics": model_data.get("metrics", {}),
                    "config": model_data.get("config", {}),
                    "created_at": model_data["created_at"],
                }
            )

    return {"models": comparison_data}


@app.post("/api/models/export/{model_name}")
async def export_model(model_name: str, export_format: str = "pkl"):
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
        export_path = export_dir / f"{model_name}_exported.{export_format}"

        if export_format == "pkl":
            import pickle

            with open(export_path, "wb") as f:
                pickle.dump(model, f)
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")

        return {
            "message": f"Model {model_name} exported successfully",
            "export_path": str(export_path),
            "format": export_format,
        }

    except Exception as e:
        logger.error("Error exporting model: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/data/exploration/{filename}")
async def explore_data(filename: str, _current_user: User = Depends(get_current_user)):
    """Get data exploration statistics."""
    safe_name = _safe_filename(filename)
    file_path = Path("data/uploads") / safe_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        import pandas as pd

        if filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif filename.endswith(".json"):
            df = pd.read_json(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Basic statistics
        stats = {
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "summary_stats": (
                df.describe().to_dict()
                if df.select_dtypes(include=[np.number]).shape[1] > 0
                else {}
            ),
            "sample_data": df.head(10).to_dict("records"),
        }

        return stats

    except Exception as e:
        logger.error("Error exploring data: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


# Prometheus-compatible metrics endpoint
@app.get(
    "/metrics",
    tags=["System"],
    summary="Prometheus metrics endpoint",
    response_class=PlainTextResponse,
)
def prometheus_metrics():
    # Example metrics (replace with real values)
    metric_lines = [
        "jarvis_api_requests_total 1234",
        "jarvis_api_errors_total 12",
        "jarvis_model_inferences_total 567",
        "jarvis_active_models 2",
    ]
    return "\n".join(metric_lines)


# Background task for system monitoring
async def monitor_system():
    """Background task to monitor system metrics and broadcast updates."""
    while True:
        try:
            sys_metrics = await get_system_metrics()
            await manager.send_system_metrics(sys_metrics)
            await asyncio.sleep(5)  # Update every 5 seconds
        except (OSError, RuntimeError) as e:
            logger.error("Error in system monitoring: %s", e)
            await asyncio.sleep(10)


# Start system monitoring on startup
# (Moved to lifespan handler above)


# Example of a protected endpoint
@app.get(
    "/secure-info",
    tags=["System"],
    summary="Get secure system info",
    description="Requires OAuth2 or API Key.",
)
def secure_info(user=Depends(get_current_user), _api_key=Security(get_api_key)):
    return {"secure": True, "user": user}


# Model registry, jobs, and user data are now persistent (see db_config.py)
# Use SQLAlchemy models and CRUD operations (see models_registry.py, jobs_persistent.py)


@app.post(
    "/predict/ensemble",
    tags=["Models"],
    summary="Ensemble prediction with multiple models",
)
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
        ensemble = np.mean(results, axis=0).tolist()
        return {"ensemble_prediction": ensemble}
    return {"error": "No valid models found"}


# Plugin system: auto-register plugins in plugins/ folder
def register_plugins(application):
    for plugin_path in glob.glob("plugins/*.py"):
        if plugin_path.endswith("__init__.py"):
            continue
        spec = importlib.util.spec_from_file_location("plugin", plugin_path)
        if spec is None or spec.loader is None:
            continue
        plugin = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plugin)
        if hasattr(plugin, "register"):
            plugin.register(application)


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

# Persistent model registry (see models_registry.py)


@app.post(
    "/models/register", tags=["Models"], summary="Register a new model (persistent)"
)
def register_model_db(
    model_name: str,
    description: str = "",
    accuracy: float = 0.0,
    _current_user: User = Depends(get_current_user),
):
    session = SessionLocal()
    model = create_model(
        session, model_name, description, accuracy if accuracy is not None else 0.0
    )
    session.close()
    # Audit log
    try:
        from audit_trail import log_audit_event

        log_audit_event(
            "system",
            "register_model",
            model_name,
            f"desc={description}, acc={accuracy}",
        )
    except (ImportError, SQLAlchemyError) as e:
        logger.warning("Audit logging failed: %s", e)
    return {"message": f"Model '{model_name}' registered.", "model": model.name}


@app.get(
    "/models/registry",
    tags=["Models"],
    summary="List all registered models (persistent)",
)
def list_registered_models_db(_current_user: User = Depends(get_current_user)):
    session = SessionLocal()
    db_models = get_models(session)
    session.close()
    return [m.name for m in db_models]


@app.post(
    "/models/activate",
    tags=["Models"],
    summary="Activate a model for inference (persistent)",
)
def activate_model_db(model_name: str, _current_user: User = Depends(get_current_user)):
    session = SessionLocal()
    model = activate_model(session, model_name)
    session.close()
    if model:
        return {"message": f"Model '{model_name}' activated."}
    else:
        raise HTTPException(status_code=404, detail="Model not found")


# Persistent job management (see jobs_persistent.py)


@app.post("/jobs/start", tags=["System"], summary="Start a background job (persistent)")
def start_job_db(
    duration: int = 10,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _current_user: User = Depends(get_current_user),
):
    session = SessionLocal()
    job_id = str(uuid.uuid4())
    create_job(session, job_id, status="queued")

    def run_job():
        update_job_status(session, job_id, status="running")
        time.sleep(duration)
        update_job_status(
            session, job_id, status="completed", completed_at=datetime.utcnow()
        )

    background_tasks.add_task(run_job)
    # Audit log
    try:
        from audit_trail import log_audit_event

        log_audit_event("system", "start_job", job_id, f"duration={duration}")
    except (ImportError, SQLAlchemyError) as e:
        logger.warning("Audit logging failed: %s", e)
    session.close()
    return {"job_id": job_id, "status": "queued"}


@app.get(
    "/jobs/status/{job_id}", tags=["System"], summary="Check job status (persistent)"
)
def job_status_db(job_id: str, _current_user: User = Depends(get_current_user)):
    session = SessionLocal()
    job = get_job(session, job_id)
    # Audit log
    try:
        from audit_trail import log_audit_event

        log_audit_event(
            "system",
            "job_status",
            job_id,
            f"status={getattr(job, 'status', 'not found')}",
        )
    except (ImportError, SQLAlchemyError) as e:
        logger.warning("Audit logging failed: %s", e)
    session.close()
    if job:
        return {"job_id": job.job_id, "status": job.status, "cancelled": job.cancelled}
    return {"job_id": job_id, "status": "not found"}


@app.post("/jobs/cancel/{job_id}", tags=["System"], summary="Cancel a job (persistent)")
def cancel_job_db(job_id: str, _current_user: User = Depends(get_current_user)):
    session = SessionLocal()
    job = update_job_status(session, job_id, status="cancelled", cancelled=True)
    # Audit log
    try:
        from audit_trail import log_audit_event

        log_audit_event("system", "cancel_job", job_id, "Job cancelled")
    except (ImportError, SQLAlchemyError) as e:
        logger.warning("Audit logging failed: %s", e)
    session.close()
    if job:
        return {"job_id": job_id, "status": "cancelled"}
    return {"job_id": job_id, "status": "not found"}


# --- Notification System ---
_ALLOWED_NOTIFICATION_HOSTS: set = set(
    h.strip()
    for h in os.environ.get("JARVIS_WEBHOOK_ALLOWLIST", "").split(",")
    if h.strip()
)


def send_notification(method: str, target: str, message: str):
    """Send notification via webhook or email."""
    try:
        if method == "webhook" and target:
            from urllib.parse import urlparse

            parsed = urlparse(target)
            if (
                _ALLOWED_NOTIFICATION_HOSTS
                and parsed.hostname not in _ALLOWED_NOTIFICATION_HOSTS
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Webhook target host not in allowlist",
                )
            requests.post(target, json={"message": message}, timeout=5)
        elif method == "email" and target:
            # Placeholder for email sending logic
            pass
        # Audit log
        try:
            from audit_trail import log_audit_event

            log_audit_event(
                "system", "send_notification", method, f"target={target}, msg={message}"
            )
        except (ImportError, SQLAlchemyError) as e:
            logger.warning("Audit logging failed: %s", e)
        return {
            "status": "sent",
            "method": method,
            "target": target,
            "message": message,
        }
    except Exception as e:
        logger.error("Notification error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/notifications/send", tags=["System"], summary="Send a notification")
async def send_notification_endpoint(
    method: str,
    target: str,
    message: str,
    _current_user: User = Depends(get_current_user),
):
    """Endpoint to send notifications."""
    return send_notification(method, target, message)


# --- Cloud Storage Integration ---
@app.post("/cloud/upload", tags=["Data"], summary="Upload file to cloud storage")
async def cloud_upload(
    filename: str,
    provider: str = "s3",
    bucket: Optional[str] = None,
    _current_user: User = Depends(get_current_user),
):
    """Upload a file to cloud storage."""
    try:
        safe_name = _safe_filename(filename)
        file_path = Path("data/uploads") / safe_name
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        result = upload_to_cloud(str(file_path), provider, bucket)
        # Audit log
        try:
            from audit_trail import log_audit_event

            log_audit_event(
                "system", "cloud_upload", provider, f"bucket={bucket}, file={filename}"
            )
        except (ImportError, SQLAlchemyError) as e:
            logger.warning("Audit logging failed: %s", e)
        return {
            "status": "uploaded",
            "provider": provider,
            "bucket": bucket,
            "filename": filename,
            "result": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Cloud upload error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/cloud/download", tags=["Data"], summary="Download file from cloud storage")
async def cloud_download(
    filename: str,
    provider: str = "s3",
    bucket: Optional[str] = None,
    _current_user: User = Depends(get_current_user),
):
    """Download a file from cloud storage."""
    try:
        result = download_from_cloud(filename, provider, bucket)
        # Audit log
        try:
            from audit_trail import log_audit_event

            log_audit_event(
                "system",
                "cloud_download",
                provider,
                f"bucket={bucket}, file={filename}",
            )
        except (ImportError, SQLAlchemyError) as e:
            logger.warning("Audit logging failed: %s", e)
        return {
            "status": "downloaded",
            "provider": provider,
            "bucket": bucket,
            "filename": filename,
            "result": result,
        }
    except Exception as e:
        logger.error("Cloud download error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


# --- GPU Inference & External Model Server Integration ---
@app.post(
    "/models/{model_name}/predict_gpu",
    tags=["Models"],
    summary="Predict using GPU (if available)",
)
async def predict_gpu(
    model_name: str,
    request: PredictRequest,
    _current_user: User = Depends(get_current_user),
):
    """Make predictions using GPU if available."""
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    model_data = models[model_name]
    if model_data["status"] != "trained":
        raise HTTPException(status_code=400, detail="Model is not trained")
    try:
        model = model_data["model"]
        input_data = np.array(request.data)
        # Example: Use GPU if available (pseudo-code, replace with actual GPU logic)
        if hasattr(model, "predict_gpu"):
            predictions = model.predict_gpu(input_data)
        else:
            predictions = model.predict(input_data)
        # Audit log
        try:
            from audit_trail import log_audit_event

            log_audit_event(
                "system", "predict_gpu", model_name, f"data_len={len(request.data)}"
            )
        except (ImportError, SQLAlchemyError) as e:
            logger.warning("Audit logging failed: %s", e)
        return {
            "model_name": model_name,
            "predictions": predictions.tolist(),
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in GPU prediction: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post(
    "/models/{model_name}/predict_external",
    tags=["Models"],
    summary="Predict using external model server",
)
async def predict_external(
    model_name: str,
    request: PredictRequest,
    _current_user: User = Depends(get_current_user),
):
    """Make predictions using an external model server."""
    # Example external server URL (should be configurable)
    external_url = f"http://external-model-server:8080/predict/{model_name}"
    try:
        response = requests.post(external_url, json={"data": request.data}, timeout=10)
        response.raise_for_status()
        result = response.json()
        # Audit log
        try:
            from audit_trail import log_audit_event

            log_audit_event(
                "system",
                "predict_external",
                model_name,
                f"data_len={len(request.data)}",
            )
        except (ImportError, SQLAlchemyError) as e:
            logger.warning("Audit logging failed: %s", e)
        return result
    except Exception as e:
        logger.error("Error in external prediction: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


# --- GDPR/CCPA Compliance Utilities ---
def anonymize_user_data(user_id: str, db: Session):
    """Anonymize user data for GDPR/CCPA compliance."""
    user = db.query(User).filter(User.username == user_id).first()
    if user:
        anon_id = uuid.uuid4().hex[:12]
        user.email = f"anon-{anon_id}@deleted.invalid"
        user.hashed_password = "REDACTED"
        user.full_name = None
        db.commit()
        return True
    return False


def secure_delete_user(user_id: str, db: Session):
    """Securely delete user data for GDPR/CCPA compliance."""
    user = db.query(User).filter(User.username == user_id).first()
    if user:
        db.delete(user)
        db.commit()
        return True
    return False


# --- GDPR/CCPA Compliance Endpoints ---
@app.post(
    "/gdpr/anonymize/{user_id}",
    tags=["System"],
    summary="Anonymize user data (GDPR/CCPA)",
)
def gdpr_anonymize(
    user_id: str,
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
):
    """Anonymize user data for GDPR/CCPA compliance."""
    try:
        success = anonymize_user_data(user_id, db)
    except (RuntimeError, OSError, ValueError) as e:
        logger.error("GDPR anonymize failed for %s: %s", user_id, e)
        return JSONResponse(status_code=404, content={"detail": "User not found"})
    # Audit log
    try:
        from audit_trail import log_audit_event

        log_audit_event(user_id, "anonymize", "user", "GDPR/CCPA anonymization")
    except (ImportError, SQLAlchemyError) as e:
        logger.warning("Audit logging failed: %s", e)
    if success:
        return {"message": f"User {user_id} anonymized."}
    return JSONResponse(status_code=404, content={"detail": "User not found"})


@app.delete(
    "/gdpr/delete/{user_id}",
    tags=["System"],
    summary="Securely delete user data (GDPR/CCPA)",
)
def gdpr_delete(
    user_id: str,
    db: Session = Depends(get_db),
    _current_user: User = Depends(get_current_user),
):
    """Securely delete user data for GDPR/CCPA compliance."""
    try:
        success = secure_delete_user(user_id, db)
    except (RuntimeError, OSError, ValueError) as e:
        logger.error("GDPR delete failed for %s: %s", user_id, e)
        return JSONResponse(status_code=404, content={"detail": "User not found"})
    # Audit log
    try:
        from audit_trail import log_audit_event

        log_audit_event(user_id, "secure_delete", "user", "GDPR/CCPA secure deletion")
    except (ImportError, SQLAlchemyError) as e:
        logger.warning("Audit logging failed: %s", e)
    if success:
        return {"message": f"User {user_id} deleted."}
    return JSONResponse(status_code=404, content={"detail": "User not found"})


# FastAPI/uvicorn entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)
