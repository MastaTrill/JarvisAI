"""Tests for database_models.py — verify ORM models can be instantiated and persisted."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_models import (
    Base,
    User,
    ChatHistory,
    ModelRun,
    PerformanceMetric,
    QuantumState,
    TemporalPattern,
    AgentTask,
)


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestUserModel:
    def test_create_user(self, db_session):
        user = User(
            username="testuser",
            email="test@test.com",
            hashed_password="fakehash",
        )
        db_session.add(user)
        db_session.commit()
        assert user.id is not None
        assert str(user.role) == "user"
        assert user.is_active is True
        assert user.is_admin is False

    def test_user_repr(self, db_session):
        user = User(username="alice", email="a@b.com", hashed_password="h")
        db_session.add(user)
        db_session.commit()
        assert "alice" in repr(user)


class TestChatHistory:
    def test_create_chat(self, db_session):
        user = User(username="chatter", email="c@e.com", hashed_password="h")
        db_session.add(user)
        db_session.commit()
        chat = ChatHistory(
            user_id=user.id,
            session_id="sess1",
            message="Hello",
            response="Hi there",
        )
        db_session.add(chat)
        db_session.commit()
        assert chat.id is not None
        assert chat.is_user_message is True


class TestModelRun:
    def test_create_model_run(self, db_session):
        user = User(username="runner", email="r@e.com", hashed_password="h")
        db_session.add(user)
        db_session.commit()
        run = ModelRun(
            user_id=user.id,
            model_name="test_nn",
            model_type="neural_network",
            accuracy=0.95,
        )
        db_session.add(run)
        db_session.commit()
        assert str(run.status) == "running"
        assert "test_nn" in repr(run)


class TestPerformanceMetric:
    def test_create_metric(self, db_session):
        metric = PerformanceMetric(
            metric_name="accuracy",
            metric_value=0.95,
            metric_unit="%",
        )
        db_session.add(metric)
        db_session.commit()
        assert "accuracy" in repr(metric)


class TestQuantumState:
    def test_create_quantum_state(self, db_session):
        qs = QuantumState(
            state_type="superposition",
            num_qubits=4,
            fidelity=0.99,
        )
        db_session.add(qs)
        db_session.commit()
        assert qs.id is not None


class TestTemporalPattern:
    def test_create_pattern(self, db_session):
        tp = TemporalPattern(
            pattern_name="weekly_trend",
            pattern_type="seasonal",
            accuracy=0.88,
        )
        db_session.add(tp)
        db_session.commit()
        assert "weekly_trend" in repr(tp)


class TestAgentTask:
    def test_create_task(self, db_session):
        task = AgentTask(
            task_name="analyze_data",
            task_type="analysis",
            priority=3,
        )
        db_session.add(task)
        db_session.commit()
        assert str(task.status) == "pending"
        assert int(task.retry_count) == 0
