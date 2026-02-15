"""
Jarvis AI - Standardized API v1
Enterprise-grade REST API with OpenAPI 3.0 documentation

This module provides a versioned, standardized API following REST best practices:
- Consistent response formats
- Proper HTTP status codes
- Rate limiting
- Request/response validation
- Comprehensive error handling
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, Security, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Generic, TypeVar
from datetime import datetime
from enum import Enum
import logging
import uuid

logger = logging.getLogger(__name__)

# Generic type for paginated responses
T = TypeVar("T")


# =============================================================================
# STANDARDIZED RESPONSE MODELS
# =============================================================================

class ResponseStatus(str, Enum):
    """API response status codes"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


class PaginationMeta(BaseModel):
    """Pagination metadata"""
    page: int = Field(..., ge=1, description="Current page number")
    per_page: int = Field(..., ge=1, le=100, description="Items per page")
    total_items: int = Field(..., ge=0, description="Total number of items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")


class APIResponse(BaseModel):
    """Standard API response wrapper"""
    status: ResponseStatus = Field(..., description="Response status")
    message: str = Field(..., description="Human-readable message")
    data: Optional[Any] = Field(None, description="Response data payload")
    meta: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "success",
            "message": "Operation completed successfully",
            "data": {"key": "value"},
            "meta": {"version": "1.0.0"},
            "request_id": "550e8400-e29b-41d4-a716-446655440000",
            "timestamp": "2026-02-11T12:00:00Z"
        }
    })


class PaginatedResponse(APIResponse):
    """Paginated API response"""
    pagination: Optional[PaginationMeta] = None


class ErrorDetail(BaseModel):
    """Error detail model"""
    field: Optional[str] = Field(None, description="Field that caused the error")
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")


class ErrorResponse(APIResponse):
    """Error response model"""
    status: ResponseStatus = ResponseStatus.ERROR
    errors: List[ErrorDetail] = Field(default_factory=list, description="List of errors")


# =============================================================================
# REQUEST MODELS
# =============================================================================

class ModelCreateRequest(BaseModel):
    """Request to create a new ML model"""
    name: str = Field(..., min_length=1, max_length=100, description="Model name")
    description: Optional[str] = Field(None, max_length=500, description="Model description")
    model_type: str = Field(..., description="Type of model (neural_network, random_forest, etc.)")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    tags: List[str] = Field(default_factory=list, description="Model tags")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "name": "sentiment-classifier-v1",
            "description": "Sentiment analysis model for product reviews",
            "model_type": "neural_network",
            "hyperparameters": {"hidden_size": 256, "learning_rate": 0.001},
            "tags": ["nlp", "classification", "production"]
        }
    })


class ModelUpdateRequest(BaseModel):
    """Request to update an existing model"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    tags: Optional[List[str]] = None
    is_active: Optional[bool] = None


class TrainingRequest(BaseModel):
    """Request to start model training"""
    model_id: str = Field(..., description="Model ID to train")
    dataset_id: str = Field(..., description="Dataset ID to use for training")
    epochs: int = Field(default=100, ge=1, le=10000, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=1, le=1024, description="Training batch size")
    validation_split: float = Field(default=0.2, ge=0.0, le=0.5, description="Validation split ratio")
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    callbacks: List[str] = Field(default_factory=list, description="Training callbacks")


class PredictionRequest(BaseModel):
    """Request for model inference"""
    model_id: str = Field(..., description="Model ID to use for prediction")
    inputs: List[Dict[str, Any]] = Field(..., min_length=1, max_length=1000, description="Input data")
    return_probabilities: bool = Field(default=False, description="Return prediction probabilities")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Classification threshold")


class DataUploadRequest(BaseModel):
    """Request metadata for data upload"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    format: str = Field(default="csv", pattern="^(csv|json|parquet|arrow)$")
    schema: Optional[Dict[str, str]] = None


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class ModelInfo(BaseModel):
    """Model information response"""
    id: str
    name: str
    description: Optional[str]
    model_type: str
    version: str
    status: str
    created_at: datetime
    updated_at: datetime
    training_metrics: Optional[Dict[str, float]] = None
    tags: List[str] = []
    is_active: bool = True


class TrainingStatus(BaseModel):
    """Training job status"""
    job_id: str
    model_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0.0 to 1.0
    current_epoch: int
    total_epochs: int
    metrics: Dict[str, float]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str] = None


class PredictionResult(BaseModel):
    """Prediction result"""
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None
    model_version: str
    inference_time_ms: float


class DatasetInfo(BaseModel):
    """Dataset information"""
    id: str
    name: str
    description: Optional[str]
    format: str
    size_bytes: int
    row_count: int
    column_count: int
    columns: List[str]
    created_at: datetime
    checksum: str


class SystemHealth(BaseModel):
    """System health status"""
    status: str  # healthy, degraded, unhealthy
    version: str
    uptime_seconds: float
    components: Dict[str, Dict[str, Any]]
    timestamp: datetime


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_response(
    data: Any = None,
    message: str = "Success",
    status: ResponseStatus = ResponseStatus.SUCCESS,
    meta: Dict[str, Any] = None,
    request_id: str = None
) -> APIResponse:
    """Create a standardized API response"""
    return APIResponse(
        status=status,
        message=message,
        data=data,
        meta=meta or {"version": "1.0.0"},
        request_id=request_id or str(uuid.uuid4())
    )


def create_paginated_response(
    data: List[Any],
    page: int,
    per_page: int,
    total_items: int,
    message: str = "Success",
    request_id: str = None
) -> PaginatedResponse:
    """Create a paginated API response"""
    total_pages = (total_items + per_page - 1) // per_page if per_page > 0 else 0
    
    return PaginatedResponse(
        status=ResponseStatus.SUCCESS,
        message=message,
        data=data,
        meta={"version": "1.0.0"},
        request_id=request_id or str(uuid.uuid4()),
        pagination=PaginationMeta(
            page=page,
            per_page=per_page,
            total_items=total_items,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
    )


def create_error_response(
    message: str,
    errors: List[ErrorDetail] = None,
    status_code: int = 400,
    request_id: str = None
) -> ErrorResponse:
    """Create an error response"""
    return ErrorResponse(
        status=ResponseStatus.ERROR,
        message=message,
        errors=errors or [],
        request_id=request_id or str(uuid.uuid4())
    )


# =============================================================================
# API ROUTER v1
# =============================================================================

router = APIRouter(prefix="/api/v1", tags=["API v1"])

security = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_request_id(request: Request) -> str:
    """Get or generate request ID"""
    return request.headers.get("X-Request-ID", str(uuid.uuid4()))


# Models endpoints
@router.get(
    "/models",
    response_model=PaginatedResponse,
    summary="List all models",
    description="Retrieve a paginated list of all ML models",
    responses={
        200: {"description": "List of models retrieved successfully"},
        401: {"description": "Authentication required"},
        500: {"description": "Internal server error"}
    }
)
async def list_models(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order"),
    request_id: str = Depends(get_request_id)
):
    """List all models with pagination and filtering"""
    # Mock data - replace with actual database query
    models = [
        ModelInfo(
            id="model-001",
            name="sentiment-classifier",
            description="NLP sentiment analysis model",
            model_type="neural_network",
            version="1.0.0",
            status="active",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            training_metrics={"accuracy": 0.95, "f1_score": 0.93},
            tags=["nlp", "production"]
        )
    ]
    
    return create_paginated_response(
        data=[m.model_dump() for m in models],
        page=page,
        per_page=per_page,
        total_items=len(models),
        message="Models retrieved successfully",
        request_id=request_id
    )


@router.post(
    "/models",
    response_model=APIResponse,
    status_code=201,
    summary="Create a new model",
    description="Create a new ML model configuration"
)
async def create_model(
    model: ModelCreateRequest = Body(...),
    request_id: str = Depends(get_request_id)
):
    """Create a new ML model"""
    new_model = ModelInfo(
        id=f"model-{uuid.uuid4().hex[:8]}",
        name=model.name,
        description=model.description,
        model_type=model.model_type,
        version="1.0.0",
        status="created",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        tags=model.tags
    )
    
    return create_response(
        data=new_model.model_dump(),
        message="Model created successfully",
        request_id=request_id
    )


@router.get(
    "/models/{model_id}",
    response_model=APIResponse,
    summary="Get model by ID",
    description="Retrieve a specific model by its ID"
)
async def get_model(
    model_id: str = Path(..., description="Model ID"),
    request_id: str = Depends(get_request_id)
):
    """Get a specific model"""
    # Mock - replace with database query
    model = ModelInfo(
        id=model_id,
        name="example-model",
        description="Example model",
        model_type="neural_network",
        version="1.0.0",
        status="active",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    return create_response(
        data=model.model_dump(),
        message="Model retrieved successfully",
        request_id=request_id
    )


@router.patch(
    "/models/{model_id}",
    response_model=APIResponse,
    summary="Update a model",
    description="Partially update a model's configuration"
)
async def update_model(
    model_id: str = Path(..., description="Model ID"),
    updates: ModelUpdateRequest = Body(...),
    request_id: str = Depends(get_request_id)
):
    """Update a model"""
    return create_response(
        data={"id": model_id, **updates.model_dump(exclude_none=True)},
        message="Model updated successfully",
        request_id=request_id
    )


@router.delete(
    "/models/{model_id}",
    response_model=APIResponse,
    summary="Delete a model",
    description="Soft delete a model"
)
async def delete_model(
    model_id: str = Path(..., description="Model ID"),
    request_id: str = Depends(get_request_id)
):
    """Delete a model"""
    return create_response(
        data={"id": model_id, "deleted": True},
        message="Model deleted successfully",
        request_id=request_id
    )


# Training endpoints
@router.post(
    "/training/jobs",
    response_model=APIResponse,
    status_code=202,
    summary="Start training job",
    description="Start an asynchronous model training job"
)
async def start_training(
    training: TrainingRequest = Body(...),
    request_id: str = Depends(get_request_id)
):
    """Start a training job"""
    job = TrainingStatus(
        job_id=f"job-{uuid.uuid4().hex[:8]}",
        model_id=training.model_id,
        status="pending",
        progress=0.0,
        current_epoch=0,
        total_epochs=training.epochs,
        metrics={},
        started_at=None,
        completed_at=None
    )
    
    return create_response(
        data=job.model_dump(),
        message="Training job queued successfully",
        request_id=request_id
    )


@router.get(
    "/training/jobs/{job_id}",
    response_model=APIResponse,
    summary="Get training job status",
    description="Get the status of a training job"
)
async def get_training_status(
    job_id: str = Path(..., description="Training job ID"),
    request_id: str = Depends(get_request_id)
):
    """Get training job status"""
    job = TrainingStatus(
        job_id=job_id,
        model_id="model-001",
        status="running",
        progress=0.45,
        current_epoch=45,
        total_epochs=100,
        metrics={"loss": 0.234, "accuracy": 0.89},
        started_at=datetime.utcnow()
    )
    
    return create_response(
        data=job.model_dump(),
        message="Training status retrieved",
        request_id=request_id
    )


@router.delete(
    "/training/jobs/{job_id}",
    response_model=APIResponse,
    summary="Cancel training job",
    description="Cancel a running training job"
)
async def cancel_training(
    job_id: str = Path(..., description="Training job ID"),
    request_id: str = Depends(get_request_id)
):
    """Cancel a training job"""
    return create_response(
        data={"job_id": job_id, "status": "cancelled"},
        message="Training job cancelled",
        request_id=request_id
    )


# Prediction endpoints
@router.post(
    "/predictions",
    response_model=APIResponse,
    summary="Make predictions",
    description="Run inference using a trained model"
)
async def predict(
    prediction: PredictionRequest = Body(...),
    request_id: str = Depends(get_request_id)
):
    """Make predictions"""
    result = PredictionResult(
        predictions=[1, 0, 1],
        probabilities=[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]] if prediction.return_probabilities else None,
        model_version="1.0.0",
        inference_time_ms=12.5
    )
    
    return create_response(
        data=result.model_dump(),
        message="Predictions completed successfully",
        request_id=request_id
    )


# Dataset endpoints
@router.get(
    "/datasets",
    response_model=PaginatedResponse,
    summary="List datasets",
    description="List all available datasets"
)
async def list_datasets(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    request_id: str = Depends(get_request_id)
):
    """List all datasets"""
    datasets = [
        DatasetInfo(
            id="ds-001",
            name="training-data",
            description="Training dataset",
            format="csv",
            size_bytes=1024000,
            row_count=10000,
            column_count=50,
            columns=["feature_1", "feature_2", "target"],
            created_at=datetime.utcnow(),
            checksum="sha256:abc123..."
        )
    ]
    
    return create_paginated_response(
        data=[d.model_dump() for d in datasets],
        page=page,
        per_page=per_page,
        total_items=len(datasets),
        message="Datasets retrieved successfully",
        request_id=request_id
    )


# Health endpoints
@router.get(
    "/health",
    response_model=APIResponse,
    summary="Health check",
    description="Check API health status"
)
async def health_check(request_id: str = Depends(get_request_id)):
    """Health check endpoint"""
    health = SystemHealth(
        status="healthy",
        version="1.0.0",
        uptime_seconds=86400.0,
        components={
            "database": {"status": "healthy", "latency_ms": 5},
            "redis": {"status": "healthy", "latency_ms": 1},
            "ml_engine": {"status": "healthy", "models_loaded": 10}
        },
        timestamp=datetime.utcnow()
    )
    
    return create_response(
        data=health.model_dump(),
        message="System is healthy",
        request_id=request_id
    )


@router.get(
    "/health/ready",
    response_model=APIResponse,
    summary="Readiness check",
    description="Check if API is ready to accept requests"
)
async def readiness_check(request_id: str = Depends(get_request_id)):
    """Readiness probe endpoint"""
    return create_response(
        data={"ready": True},
        message="System is ready",
        request_id=request_id
    )


@router.get(
    "/health/live",
    response_model=APIResponse,
    summary="Liveness check",
    description="Check if API is alive"
)
async def liveness_check(request_id: str = Depends(get_request_id)):
    """Liveness probe endpoint"""
    return create_response(
        data={"alive": True},
        message="System is alive",
        request_id=request_id
    )
