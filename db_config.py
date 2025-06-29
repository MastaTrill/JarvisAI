"""
Database configuration for Jarvis AI
- Uses SQLAlchemy for ORM
- Alembic for migrations
- Default: PostgreSQL (can be changed to SQLite or others)
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./jarvis.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Usage in models:
# from db_config import Base
# class Model(Base):
#     __tablename__ = "models"
#     ...
