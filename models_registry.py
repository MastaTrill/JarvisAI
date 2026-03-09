"""
Persistent Model Registry for Jarvis AI
- SQLAlchemy ORM model for registered models
- CRUD utilities for database operations
"""

from db_config import Base
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from datetime import datetime, timezone


from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    version = Column(String, default="1.0.0")
    description = Column(String)
    accuracy = Column(Float)
    registered_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    active = Column(Boolean, default=False)
    device = Column(String, default="cpu")  # cpu/gpu/other
    external_endpoint = Column(String, nullable=True)
    drift_score = Column(Float, nullable=True)
    audit_log = Column(String, nullable=True)
    parent_id = Column(
        Integer, ForeignKey("model_registry.id"), nullable=True
    )  # for rollback
    parent = relationship("ModelRegistry", remote_side=[id])


# CRUD utilities


def create_model(session, name, description="", accuracy=None):
    model = ModelRegistry(name=name, description=description, accuracy=accuracy)
    session.add(model)
    session.commit()
    session.refresh(model)
    return model


def get_models(session):
    return session.query(ModelRegistry).all()


def activate_model(session, name):
    model = session.query(ModelRegistry).filter_by(name=name).first()
    if model:
        session.query(ModelRegistry).filter(ModelRegistry.active.is_(True)).update(
            {"active": False}
        )
        model.active = True
        session.commit()
        return model
    return None
