"""
JarvisAI Database Connection and Session Management
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from pathlib import Path
import os
from typing import Generator

from database_models import Base

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./jarvis_ai.db")

# Create engine
if DATABASE_URL.startswith("sqlite"):
    # SQLite specific configuration
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False  # Set to True for SQL logging
    )
else:
    # PostgreSQL configuration
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        echo=False
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_database():
    """Initialize the database by creating all tables"""
    Base.metadata.create_all(bind=engine)
    print(f"✅ Database initialized successfully at {DATABASE_URL}")


def drop_database():
    """Drop all tables (use with caution!)"""
    Base.metadata.drop_all(bind=engine)
    print("⚠️ All database tables dropped")


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get database session
    
    Usage in FastAPI:
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session():
    """
    Context manager for database sessions
    
    Usage:
        with get_db_session() as db:
            user = db.query(User).filter_by(username="creator").first()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def create_default_users():
    """Create default users for the system"""
    from database_models import User
    import hashlib
    import secrets
    
    def hash_password(password: str) -> str:
        """Simple SHA-256 password hashing"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    default_users = [
        {
            "username": "creator",
            "email": "creator@jarvisai.com",
            "full_name": "System Creator",
            "role": "creator",
            "password": "Creator2024"
        },
        {
            "username": "admin",
            "email": "admin@jarvisai.com",
            "full_name": "System Administrator",
            "role": "admin",
            "password": "Admin2024"
        },
        {
            "username": "demo",
            "email": "demo@jarvisai.com",
            "full_name": "Demo User",
            "role": "user",
            "password": "Demo2024"
        }
    ]
    
    with get_db_session() as db:
        for user_data in default_users:
            # Check if user already exists
            existing_user = db.query(User).filter_by(username=user_data["username"]).first()
            if existing_user:
                print(f"ℹ️ User {user_data['username']} already exists")
                continue
            
            # Create new user
            user = User(
                username=user_data["username"],
                email=user_data["email"],
                full_name=user_data["full_name"],
                role=user_data["role"],
                hashed_password=hash_password(user_data["password"]),
                api_key=secrets.token_urlsafe(32),
                is_active=True
            )
            db.add(user)
            print(f"✅ Created user: {user_data['username']} (role: {user_data['role']})")
        
        db.commit()


# Database utilities
class DatabaseManager:
    """Database management utilities"""
    
    @staticmethod
    def backup_database(backup_path: str = None):
        """Backup SQLite database"""
        if not DATABASE_URL.startswith("sqlite"):
            raise NotImplementedError("Backup only supported for SQLite databases")
        
        import shutil
        from datetime import datetime
        
        db_file = DATABASE_URL.replace("sqlite:///", "")
        if backup_path is None:
            backup_path = f"backup_jarvis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        shutil.copy2(db_file, backup_path)
        print(f"✅ Database backed up to {backup_path}")
        return backup_path
    
    @staticmethod
    def restore_database(backup_path: str):
        """Restore SQLite database from backup"""
        if not DATABASE_URL.startswith("sqlite"):
            raise NotImplementedError("Restore only supported for SQLite databases")
        
        import shutil
        
        db_file = DATABASE_URL.replace("sqlite:///", "")
        shutil.copy2(backup_path, db_file)
        print(f"✅ Database restored from {backup_path}")
    
    @staticmethod
    def get_table_stats():
        """Get statistics about database tables"""
        from database_models import User, ChatHistory, ModelRun, PerformanceMetric, QuantumState, TemporalPattern, AgentTask
        
        models = [User, ChatHistory, ModelRun, PerformanceMetric, QuantumState, TemporalPattern, AgentTask]
        
        stats = {}
        with get_db_session() as db:
            for model in models:
                count = db.query(model).count()
                stats[model.__tablename__] = count
        
        return stats
    
    @staticmethod
    def cleanup_old_data(days: int = 30):
        """Clean up data older than specified days"""
        from datetime import datetime, timedelta
        from database_models import ChatHistory, PerformanceMetric, QuantumState, TemporalPattern
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        with get_db_session() as db:
            # Clean up old chat histories
            deleted_chats = db.query(ChatHistory).filter(ChatHistory.created_at < cutoff_date).delete()
            
            # Clean up old performance metrics
            deleted_metrics = db.query(PerformanceMetric).filter(PerformanceMetric.timestamp < cutoff_date).delete()
            
            # Clean up old quantum states
            deleted_quantum = db.query(QuantumState).filter(QuantumState.created_at < cutoff_date).delete()
            
            # Clean up old temporal patterns
            deleted_temporal = db.query(TemporalPattern).filter(TemporalPattern.created_at < cutoff_date).delete()
            
            db.commit()
            
            print(f"🧹 Cleaned up old data:")
            print(f"  - Chat histories: {deleted_chats}")
            print(f"  - Performance metrics: {deleted_metrics}")
            print(f"  - Quantum states: {deleted_quantum}")
            print(f"  - Temporal patterns: {deleted_temporal}")


if __name__ == "__main__":
    # Initialize database and create default users
    print("🚀 Initializing JarvisAI Database...")
    init_database()
    
    # Create default users (requires passlib)
    try:
        create_default_users()
    except ImportError:
        print("⚠️ Install passlib[bcrypt] to create default users: pip install passlib[bcrypt]")
    
    # Show table statistics
    print("\n📊 Database Statistics:")
    stats = DatabaseManager.get_table_stats()
    for table, count in stats.items():
        print(f"  - {table}: {count} records")
