"""
Enhanced Model Registry for Jarvis AI
- Extended SQLAlchemy ORM model with enterprise features
- Improved versioning, lineage tracking, and metadata management
- Model stage management (Staging, Production, Archived)
- Enhanced audit capabilities
"""

from src.infra.db_config import Base
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    JSON,
    ForeignKey,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import relationship, backref
from datetime import datetime, timezone
import json
from typing import Optional, Dict, Any, List


class ModelRegistry(Base):
    """Enhanced model registry with enterprise features"""
    __tablename__ = "model_registry"
    
    # Core identification
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    # Using semantic versioning: MAJOR.MINOR.PATCH
    version = Column(String, default="1.0.0", nullable=False)
    
    # Ensure unique combination of name and version
    __table_args__ = (
        UniqueConstraint('name', 'version', name='uq_model_name_version'),
        Index('idx_model_name', 'name'),
        Index('idx_model_active', 'active'),
        Index('idx_model_stage', 'stage'),
    )
    
    # Basic metadata
    description = Column(Text)
    # Enhanced versioning fields
    version_major = Column(Integer, default=0)
    version_minor = Column(Integer, default=0)
    version_patch = Column(Integer, default=0)
    version_label = Column(String, nullable=True)  # e.g., "beta", "rc1"
    
    # Model lifecycle and stage management
    stage = Column(String, default="development")  # development, staging, production, archived
    status = Column(String, default="training")    # training, validated, deployed, failed, deprecated
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    # Additional metrics can be stored in metrics_json
    
    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    deployed_at = Column(DateTime, nullable=True)
    archived_at = Column(DateTime, nullable=True)
    
    # Deployment and serving info
    active = Column(Boolean, default=False)  # Currently active/deployed version
    device = Column(String, default="cpu")   # cpu/gpu/tpu/other
    external_endpoint = Column(String, nullable=True)
    internal_model_path = Column(String, nullable=True)
    
    # Model lineage and provenance
    parent_id = Column(Integer, ForeignKey("model_registry.id"), nullable=True)
    parent = relationship("ModelRegistry", remote_side=[id], backref="children")
    # For tracking model families/lines
    model_family_id = Column(String, nullable=True, index=True)  # Groups related models
    
    # Enhanced audit and compliance
    created_by = Column(String, nullable=True)  # User who created/registerd the model
    updated_by = Column(String, nullable=True)  # User who last modified
    approved_by = Column(String, nullable=True)  # User who approved for production
    approval_date = Column(DateTime, nullable=True)
    audit_log = Column(Text, nullable=True)  # JSON string of audit events
    
    # Rich metadata and configuration
    hyperparameters = Column(JSON, nullable=True)  # Store training hyperparameters
    metrics_json = Column(JSON, nullable=True)     # Flexible storage for metrics
    tags = Column(JSON, nullable=True)             # Flexible tagging system
    model_schema = Column(JSON, nullable=True)     # Input/output schema
    
    # Model artifacts and dependencies
    requirements = Column(Text, nullable=True)     # Python package requirements
    model_size_bytes = Column(Integer, nullable=True)
    inference_latency_ms = Column(Float, nullable=True)
    
    # Risk and compliance
    risk_level = Column(String, default="low")     # low, medium, high, critical
    compliance_tags = Column(JSON, nullable=True)  # e.g., ["gdpr", "hipaa", "sox"]
    restrictions = Column(Text, nullable=True)     # Usage restrictions
    
    # Notes and documentation
    notes = Column(Text, nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "version_major": self.version_major,
            "version_minor": self.version_minor,
            "version_patch": self.version_patch,
            "version_label": self.version_label,
            "description": self.description,
            "stage": self.stage,
            "status": self.status,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "archived_at": self.archived_at.isoformat() if self.archived_at else None,
            "active": self.active,
            "device": self.device,
            "external_endpoint": self.external_endpoint,
            "internal_model_path": self.internal_model_path,
            "parent_id": self.parent_id,
            "model_family_id": self.model_family_id,
            "created_by": self.created_by,
            "updated_by": self.updated_by,
            "approved_by": self.approved_by,
            "approval_date": self.approval_date.isoformat() if self.approval_date else None,
            "risk_level": self.risk_level,
            "compliance_tags": self.compliance_tags or [],
            "restrictions": self.restrictions,
            "notes": self.notes,
            "hyperparameters": self.hyperparameters or {},
            "metrics": self.metric_json or {},
            "tags": self.tags or {},
            "model_schema": self.model_schema or {},
            "requirements": self.requirements,
            "model_size_bytes": self.model_size_bytes,
            "inference_latency_ms": self.inference_latency_ms,
        }
    
    def generate_model_card(self) -> str:
        """Generate a basic model card for documentation"""
        card = f"""
# Model Card: {self.name} v{self.version}

## Model Details
- **Name**: {self.name}
- **Version**: {self.version}
- **Stage**: {self.stage}
- **Status**: {self.status}
- **Created**: {self.created_at.isoformat() if self.created_at else 'Unknown'}
- **Last Updated**: {self.updated_at.isoformat() if self.updated_at else 'Unknown'}
- **Model Family**: {self.model_family_id or 'N/A'}

## Performance Metrics
"""
        if self.accuracy is not None:
            card += f"- **Accuracy**: {self.accuracy:.4f}\n"
        if self.precision is not None:
            card += f"- **Precision**: {self.precision:.4f}\n"
        if self.recall is not None:
            card += f"- **Recall**: {self.recall:.4f}\n"
        if self.f1_score is not None:
            card += f"- **F1 Score**: {self.f1_score:.4f}\n"
            
        card += f"""
## Technical Details
- **Device**: {self.device}
- **Framework**: {self.hyperparameters.get('framework', 'Unknown') if self.hyperparameters else 'Unknown'}
- **Model Size**: {self.model_size_bytes or 'Unknown'} bytes
- **Inference Latency**: {self.inference_latency_ms or 'Unknown'} ms

## Usage
"""
        if self.restrictions:
            card += f"### Restrictions\n{self.restrictions}\n\n"
        else:
            card += "### Restrictions\nNo specific restrictions documented.\n\n"
            
        if self.notes:
            card += f"### Notes\n{self.notes}\n\n"
            
        card += f"""## Lifecycle
- **Current Stage**: {self.stage}
- **Is Active**: {'Yes' if self.active else 'No'}
- **Deployment Date**: {self.deployed_at.isoformat() if self.deployed_at else 'Not deployed'}
- **Archive Date**: {self.archived_at.isoformat() if self.archived_at else 'Not archived'}

## Lineage
"""
        if self.parent_id:
            card += f"- **Parent Model ID**: {self.parent_id}\n"
        else:
            card += "- **Parent Model ID**: None (base model)\n"
            
        child_count = len(self.children) if hasattr(self, 'children') and self.children else 0
        card += f"- **Child Models (derived)**: {child_count}\n"
        
        return card.strip()


# Enhanced CRUD utilities with versioning support


def create_model(
    session, 
    name: str, 
    description: str = "", 
    version: str = "1.0.0",
    created_by: str = None,
    model_family_id: str = None,
    **kwargs
) -> ModelRegistry:
    """Create a new model registry entry with enhanced versioning"""
    
    # Parse semantic version if provided
    try:
        # Handle version string like "1.2.3" or "1.2.3-rc1"
        if '-' in version:
            version_core, version_label = version.split('-', 1)
        else:
            version_core, version_label = version, None
            
        parts = version_core.split('.')
        if len(parts) >= 3:
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            major, minor = int(parts[0]), int(parts[1])
            patch = 0
        else:
            major = int(parts[0]) if parts else 0
            minor = patch = 0
    except (ValueError, IndexError):
        # Default to 0.0.0 if parsing fails
        major = minor = patch = 0
        version_label = None
    
    model = ModelRegistry(
        name=name,
        description=description,
        version=version,
        version_major=max(0, major),
        version_minor=max(0, minor),
        version_patch=max(0, patch),
        version_label=version_label,
        created_by=created_by,
        model_family_id=model_family_id,
        **kwargs
    )
    session.add(model)
    session.commit()
    session.refresh(model)
    return model


def get_models(
    session, 
    name: str = None, 
    stage: str = None, 
    active_only: bool = False,
    limit: int = None
):
    """Get models with filtering options"""
    query = session.query(ModelRegistry)
    
    if name:
        query = query.filter(ModelRegistry.name == name)
    if stage:
        query = query.filter(ModelRegistry.stage == stage)
    if active_only:
        query = query.filter(ModelRegistry.active == True)
        
    query = query.order_by(ModelRegistry.created_at.desc())
    
    if limit:
        query = query.limit(limit)
        
    return query.all()


def get_model_by_name_and_version(session, name: str, version: str):
    """Get a specific model by name and version"""
    return session.query(ModelRegistry).filter(
        ModelRegistry.name == name,
        ModelRegistry.version == version
    ).first()


def get_latest_model(session, name: str, stage: str = None):
    """Get the latest version of a model"""
    query = session.query(ModelRegistry).filter(ModelRegistry.name == name)
    if stage:
        query = query.filter(ModelRegistry.stage == stage)
    return query.order_by(
        ModelRegistry.version_major.desc(),
        ModelRegistry.version_minor.desc(),
        ModelRegistry.version_patch.desc()
    ).first()


def promote_model(session, name: str, version: str, stage: str, promoted_by: str = None):
    """Promote a model to a specific stage (e.g., staging, production)"""
    model = get_model_by_name_and_version(session, name, version)
    if not model:
        return None
        
    # If promoting to production, deactivate other production models of same name
    if stage.lower() == "production":
        session.query(ModelRegistry).filter(
            ModelRegistry.name == name,
            ModelRegistry.stage == "production",
            ModelRegistry.id != model.id
        ).update({"stage": "staging", "active": False})
    
    model.stage = stage
    model.updated_by = promoted_by
    model.updated_at = datetime.now(timezone.utc)
    
    if stage == "production":
        model.active = True
        model.deployed_at = datetime.now(timezone.utc)
        if promoted_by:
            model.approved_by = promoted_by
            model.approval_date = datetime.now(timezone.utc)
    
    session.commit()
    session.refresh(model)
    return model


def model_exists(session, name: str, version: str) -> bool:
    """Check if a model with specific name and version exists"""
    return session.query(ModelRegistry).filter(
        ModelRegistry.name == name,
        ModelRegistry.version == version
    ).first() is not None


def get_model_lineage(session, model_id: int):
    """Get the lineage (parents and children) of a model"""
    model = session.query(ModelRegistry).get(model_id)
    if not model:
        return None
        
    # Get all ancestors
    ancestors = []
    current = model
    while current.parent:
        ancestors.append(current.parent)
        current = current.parent
    
    # Get all descendants (simplified - direct children only for now)
    descendants = model.children if hasattr(model, 'children') else []
    
    return {
        "model": model,
        "ancestors": list(reversed(ancestors)),  # Root first
        "descendants": descendants,
    }


def add_model_audit_entry(
    session, 
    model_id: int, 
    event: str, 
    user: str = None,
    details: Dict[str, Any] = None
):
    """Add an audit entry to a model's audit log"""
    model = session.query(ModelRegistry).get(model_id)
    if not model:
        return False
        
    audit_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "user": user,
        "details": details or {}
    }
    
    # Parse existing audit log or initialize
    try:
        audit_log = json.loads(model.audit_log) if model.audit_log else []
    except (json.JSONDecodeError, TypeError):
        audit_log = []
    
    audit_log.append(audit_entry)
    
    # Keep only last 100 entries to prevent unbounded growth
    if len(audit_log) > 100:
        audit_log = audit_log[-100:]
        
    model.audit_log = json.dumps(audit_log)
    model.updated_by = user
    model.updated_at = datetime.now(timezone.utc)
    
    session.commit()
    return True


def activate_model(session, name: str):
    """Activate a model (legacy function for backward compatibility)"""
    # Find the active model by name (highest version that's active)
    model = session.query(ModelRegistry).filter(
        ModelRegistry.name == name,
        ModelRegistry.active == True
    ).first()
    
    if model:
        # Deactivate all models with this name
        session.query(ModelRegistry).filter(
            ModelRegistry.name == name
        ).update({"active": False})
        
        # Reactivate the specified model
        model.active = True
        session.commit()
        return model
    return None